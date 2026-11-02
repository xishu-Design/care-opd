#!/usr/bin/env python3
"""Build a directly loadable Hugging Face release from a monolithic safetensors file.

The implementation is intentionally streaming: it rewrites complete tensors into
standard Transformers shards without importing torch or materializing the model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


EXPECTED_WEIGHT_SHA256 = (
    "87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4"
)
EXPECTED_MODEL_FINGERPRINT = (
    "4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd"
)
COPY_BUFFER_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TensorEntry:
    name: str
    spec: dict
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def parse_size(value: str) -> int:
    text = value.strip().upper()
    units = {
        "B": 1,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            if not number:
                break
            size = int(float(number) * units[suffix])
            if size <= 0:
                raise ValueError("max shard size must be positive")
            return size
    size = int(text)
    if size <= 0:
        raise ValueError("max shard size must be positive")
    return size


def read_safetensors_header(path: Path) -> tuple[bytes, dict, int, list[TensorEntry]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length_raw = handle.read(8)
        if len(length_raw) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_length = struct.unpack("<Q", length_raw)[0]
        if header_length <= 0 or 8 + header_length > file_size:
            raise ValueError(f"invalid safetensors header length: {header_length}")
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError("truncated safetensors header")

    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")

    tensors: list[TensorEntry] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"tensor {name!r} has an invalid spec")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"tensor {name!r} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError(f"tensor {name!r} has invalid offsets {offsets}")
        tensors.append(TensorEntry(name=name, spec=spec, start=start, end=end))

    if not tensors:
        raise ValueError("no tensors found")
    tensors.sort(key=lambda item: (item.start, item.end, item.name))
    cursor = 0
    for tensor in tensors:
        if tensor.start != cursor:
            raise ValueError(
                f"tensor payload is not contiguous at {tensor.name!r}: "
                f"expected {cursor}, found {tensor.start}"
            )
        cursor = tensor.end
    data_start = 8 + header_length
    if data_start + cursor != file_size:
        raise ValueError(
            "safetensors payload size mismatch: "
            f"header implies {data_start + cursor}, file has {file_size}"
        )
    return length_raw + header_raw, header, data_start, tensors


def group_tensors(tensors: list[TensorEntry], max_shard_bytes: int) -> list[list[TensorEntry]]:
    groups: list[list[TensorEntry]] = []
    current: list[TensorEntry] = []
    current_size = 0
    for tensor in tensors:
        if current and current_size + tensor.size > max_shard_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(tensor)
        current_size += tensor.size
    if current:
        groups.append(current)
    return groups


def encode_header(metadata: object, tensors: Iterable[TensorEntry]) -> tuple[bytes, dict[str, dict]]:
    shard_specs: dict[str, dict] = {}
    cursor = 0
    for tensor in tensors:
        spec = dict(tensor.spec)
        spec["data_offsets"] = [cursor, cursor + tensor.size]
        shard_specs[tensor.name] = spec
        cursor += tensor.size

    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    header.update(shard_specs)
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw, shard_specs


def copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    size: int,
    source_hash: hashlib._Hash,
    shard_hash: hashlib._Hash,
) -> None:
    remaining = size
    while remaining:
        block = source.read(min(COPY_BUFFER_BYTES, remaining))
        if not block:
            raise EOFError(f"source ended with {remaining} bytes still expected")
        destination.write(block)
        source_hash.update(block)
        shard_hash.update(block)
        remaining -= len(block)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_model_card() -> str:
    return """---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-generation
language:
- zh
- en
tags:
- qwen
- care-opd
- evidence-grounded-generation
- structured-generation
---

# CARE-OPD QZBC 4B

CARE-OPD QZBC 4B is a text-only Qwen3.5-4B checkpoint adapted for
evidence-bound structured response generation. The LoRA training adapters have
already been merged, so this repository contains a standalone Transformers
checkpoint.

## What the model does

The model accepts chat messages containing an Evidence Packet and produces a
structured `GeneratedResponse`. CARE-OPD training uses verifier-identified
semantic repair regions, teacher supervision on those regions, retention on
unaffected regions, and repeated recollection of failures from the updated
student. The verifier, teacher, retrieval system, and host Agent are training or
application-side components and are **not** embedded in this checkpoint.

## Intended use

- Research on evidence-grounded and structured response generation.
- Reproduction of the QZBC evaluation contract with the matching prompt,
  schema validator, source-boundary checks, and human review.
- Controlled study of model behavior under missing, conflicting, or unreliable
  evidence states.

## Out-of-scope and safety notice

This model is **not** a medical device and must not be used as an autonomous
diagnostic, prescribing, dosing, triage, or emergency-response system. It does
not replace qualified professionals. The checkpoint alone does not provide the
host verifier, retrieval policy, source validation, or clinical safeguards.
Outputs can be incorrect, unsupported, incomplete, or overly confident.

The strongest evaluations concern the QZBC contract and controlled evidence
perturbations. They do not establish clinical validity, patient benefit,
general out-of-domain robustness, or end-to-end Agent/tool-routing safety.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "YOUR_NAMESPACE/care-opd-qzbc-4b-final"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "请仅依据给定 Evidence Packet 作答。"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The exported configuration records Transformers 5.14.1. Use a Transformers
release that supports the `qwen3_5_text` architecture.

## Technical details

- Architecture: `Qwen3_5ForCausalLM`
- Parameters: approximately 4.206B
- Weight dtype: bfloat16
- Upstream model: `Qwen/Qwen3.5-4B`
- Adaptation: merged CARE-OPD LoRA checkpoint
- Serialization: sharded safetensors with a standard
  `model.safetensors.index.json`
- Original monolithic weight SHA-256:
  `87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4`
- Formal evaluation model fingerprint:
  `4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd`

The exact immutable revision of the upstream base model was not retained in the
export metadata and is therefore reported as unknown rather than guessed.

## Training data and privacy

The checkpoint was trained on QZBC-formatted data and controlled evidence-state
variants. Before making this repository public, the publisher must independently
confirm that all training data are synthetic or properly de-identified, contain
no PHI/PII or secrets, and are authorized for redistribution through model
weights. No raw training examples are included in this release directory.

## Evaluation summary

On the family-clean QZBC perturbed track, the formal CARE-OPD checkpoint reached
80.51% task success (2,355/2,925). Public benchmark results primarily support
competitive capability retention rather than universal superiority. These
figures are task-specific and should not be interpreted as clinical accuracy.

## Known limitations

- The largest QZBC gain is concentrated in the source-outage condition;
  evidence-conflict handling remains a major limitation.
- Host-side verification and source-boundary enforcement remain necessary.
- Evidence is strongest in-domain; broad OOD, multilingual, demographic, and
  clinical validation are incomplete.
- The tokenizer includes upstream multimodal special tokens, but this release is
  a text checkpoint and does not include a vision/audio encoder or processor.

## License and attribution

This derivative checkpoint is distributed under the included Apache License
2.0 text, subject to the publisher's confirmation of all upstream and training
data rights. It is based on `Qwen/Qwen3.5-4B` and has been modified through
CARE-OPD fine-tuning and adapter merging.
"""


def create_upload_guide(folder_name: str) -> str:
    return f"""# Upload this checkpoint to Hugging Face

This folder is already in a directly loadable Transformers layout. Do **not**
ZIP the shards and do not upload the GitHub 25 MB transport volumes.

## 1. Final local check

```powershell
python .\\verify_hf_package.py
```

## 2. Install the current Hub client and authenticate

```powershell
python -m pip install -U huggingface_hub hf_xet
hf auth login
hf auth whoami
```

Use a write or fine-grained token. Never paste the token into this folder,
README, scripts, or Git history.

## 3. Create a private repository first

Replace `YOUR_NAMESPACE` below:

```powershell
$RepoId = "YOUR_NAMESPACE/{folder_name}"
hf repos create $RepoId --repo-type model --private --exist-ok
```

## 4. Upload with resumable Xet transfer

Run from this folder:

```powershell
$env:HF_XET_HIGH_PERFORMANCE = "1"
hf upload $RepoId . . --repo-type model
```

If the transfer is interrupted, run the same `hf upload` command again; already
committed files and uploaded chunks are skipped. High-performance mode may use
most available bandwidth and CPU, so omit the environment variable when that is
undesirable.

## 5. Validate the private Hub copy before publishing

Use a clean environment and the repository ID:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "YOUR_NAMESPACE/{folder_name}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id, torch_dtype="auto", device_map="auto"
)
print(type(model).__name__, len(tokenizer))
```

Only switch the repository to public after the publishing checklist, license,
privacy, model-card, local-load, and remote-load checks all pass.
"""


def create_verifier() -> str:
    return r'''#!/usr/bin/env python3
"""Lightweight structural and checksum verification for this HF package."""

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(32 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


index = json.loads((ROOT / "model.safetensors.index.json").read_text(encoding="utf-8"))
weight_map = index["weight_map"]
assert weight_map, "empty weight_map"
shards = sorted(set(weight_map.values()))
seen = set()
total = 0
for shard_name in shards:
    path = ROOT / shard_name
    assert path.is_file(), f"missing shard: {shard_name}"
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    specs = {k: v for k, v in header.items() if k != "__metadata__"}
    assert set(specs).isdisjoint(seen), f"duplicate tensor in {shard_name}"
    seen.update(specs)
    end = max(v["data_offsets"][1] for v in specs.values())
    assert path.stat().st_size == 8 + header_len + end, f"bad size: {shard_name}"
    total += end
assert seen == set(weight_map), "index/header tensor set mismatch"
assert all(weight_map[name] in shards for name in seen), "index references unknown shard"
assert total == index["metadata"]["total_size"], "total_size mismatch"

checksums = {}
for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if line.strip():
        value, name = line.split("  ", 1)
        checksums[name] = value
for name, expected in checksums.items():
    actual = digest(ROOT / name)
    assert actual == expected, f"checksum mismatch: {name}"

print(f"HF_PACKAGE_VERIFY=PASS tensors={len(seen)} shards={len(shards)} bytes={total}")
'''


def create_upload_script() -> str:
    return r'''param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RepoId,
    [switch]$Public,
    [switch]$ExistingRepo,
    [switch]$DisableHighPerformance
)

$ErrorActionPreference = 'Stop'
$PackageRoot = $PSScriptRoot
$WorkspaceRoot = (Resolve-Path (Join-Path $PackageRoot '..\..')).Path
$LocalHf = Join-Path $WorkspaceRoot '.hf_upload_venv\Scripts\hf.exe'
$Hf = if (Test-Path -LiteralPath $LocalHf) { $LocalHf } else { (Get-Command hf -ErrorAction Stop).Source }

& python (Join-Path $PackageRoot 'verify_hf_package.py')
if ($LASTEXITCODE -ne 0) { throw "Local package verification failed." }
$PreviousErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Hf auth whoami
$ErrorActionPreference = $PreviousErrorAction
if ($LASTEXITCODE -ne 0) { throw "Run '$Hf auth login' and retry." }

if (-not $ExistingRepo) {
    $CreateArgs = @('repos', 'create', $RepoId, '--repo-type', 'model', '--exist-ok')
    if (-not $Public) { $CreateArgs += '--private' }
    $ErrorActionPreference = 'Continue'
    & $Hf @CreateArgs
    $ErrorActionPreference = $PreviousErrorAction
    if ($LASTEXITCODE -ne 0) { throw "Use a write token, or create the repo in the browser and retry with -ExistingRepo." }
}
if (-not $DisableHighPerformance) { $env:HF_XET_HIGH_PERFORMANCE = '1' }
$ErrorActionPreference = 'Continue'
& $Hf upload $RepoId $PackageRoot '.' --repo-type model
$ErrorActionPreference = $PreviousErrorAction
if ($LASTEXITCODE -ne 0) { throw "Re-run this script to resume the upload." }
Write-Host "HF_UPLOAD=COMPLETE https://huggingface.co/$RepoId"
'''


def build_release(source_dir: Path, output_dir: Path, public_assets: Path, max_shard: int) -> None:
    source_weight = source_dir / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")

    staging = output_dir.parent / f".{output_dir.name}.building_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)

    source_prefix, source_header, data_start, tensors = read_safetensors_header(source_weight)
    groups = group_tensors(tensors, max_shard)
    total_payload_bytes = tensors[-1].end
    metadata = source_header.get("__metadata__")
    source_hash = hashlib.sha256(source_prefix)
    weight_map: dict[str, str] = {}
    shard_records: list[dict] = []

    print(
        f"source={source_weight} tensors={len(tensors)} "
        f"payload={total_payload_bytes} shards={len(groups)}"
    )
    with source_weight.open("rb") as source:
        source.seek(data_start)
        for shard_index, group in enumerate(groups, start=1):
            shard_name = f"model-{shard_index:05d}-of-{len(groups):05d}.safetensors"
            shard_path = staging / shard_name
            header_bytes, shard_specs = encode_header(metadata, group)
            shard_hash = hashlib.sha256(header_bytes)
            payload_bytes = sum(tensor.size for tensor in group)
            with shard_path.open("wb") as destination:
                destination.write(header_bytes)
                for tensor in group:
                    copy_exact(source, destination, tensor.size, source_hash, shard_hash)
                    weight_map[tensor.name] = shard_name
            expected_size = len(header_bytes) + payload_bytes
            if shard_path.stat().st_size != expected_size:
                raise RuntimeError(f"written shard has wrong size: {shard_name}")
            shard_records.append(
                {
                    "name": shard_name,
                    "bytes": expected_size,
                    "payload_bytes": payload_bytes,
                    "tensor_count": len(shard_specs),
                    "sha256": shard_hash.hexdigest(),
                }
            )
            print(
                f"wrote {shard_name}: tensors={len(group)} "
                f"bytes={expected_size} sha256={shard_hash.hexdigest()}"
            )

    actual_source_sha = source_hash.hexdigest()
    if actual_source_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            "source weight SHA-256 mismatch after streaming: "
            f"expected {EXPECTED_WEIGHT_SHA256}, got {actual_source_sha}"
        )

    index = {
        "metadata": {"total_size": total_payload_bytes},
        "weight_map": weight_map,
    }
    write_json(staging / "model.safetensors.index.json", index)

    copy_names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    for name in copy_names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, staging / name)
    for name in ["LICENSE", "NOTICE", "PUBLISHING_CHECKLIST.md"]:
        source = public_assets / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, staging / name)

    (staging / "README.md").write_text(create_model_card(), encoding="utf-8")
    (staging / "UPLOAD_TO_HUGGINGFACE.md").write_text(
        create_upload_guide(output_dir.name), encoding="utf-8"
    )
    (staging / "verify_hf_package.py").write_text(create_verifier(), encoding="utf-8")
    (staging / "upload_to_hf.ps1").write_text(create_upload_script(), encoding="utf-8")

    old_public_manifest = json.loads(
        (public_assets / "model" / "export_manifest.public.json").read_text(encoding="utf-8")
    )
    public_manifest = {
        "schema_version": "care_opd.huggingface_export_manifest.v1",
        "artifact_type": "merged_transformers_checkpoint",
        "adapter_merged": True,
        "dtype": "bfloat16",
        "upstream_base_model": old_public_manifest["upstream_base_model"],
        "serialization": {
            "format": "transformers_sharded_safetensors",
            "index_file": "model.safetensors.index.json",
            "max_shard_bytes": max_shard,
            "total_tensor_bytes": total_payload_bytes,
            "tensor_count": len(tensors),
            "shards": shard_records,
        },
        "source_weight": {
            "name": "model.safetensors",
            "bytes": source_weight.stat().st_size,
            "sha256": actual_source_sha,
        },
        "config_sha256": old_public_manifest["config_sha256"],
        "evaluation_model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "qzbc_contract_fingerprint": old_public_manifest["qzbc_contract_fingerprint"],
        "privacy_note": (
            "Internal training paths and source-tree inventory are excluded from this release."
        ),
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    write_json(staging / "export_manifest.public.json", public_manifest)

    required_tensor_names = set(weight_map)
    seen: set[str] = set()
    for record in shard_records:
        _, header, _, shard_tensors = read_safetensors_header(staging / record["name"])
        names = {tensor.name for tensor in shard_tensors}
        if seen.intersection(names):
            raise RuntimeError(f"duplicate tensors detected in {record['name']}")
        seen.update(names)
        if header.get("__metadata__") != metadata:
            raise RuntimeError(f"metadata drift in {record['name']}")
    if seen != required_tensor_names:
        raise RuntimeError("index and shard tensor sets do not match")

    shard_hashes = {record["name"]: record["sha256"] for record in shard_records}
    checksum_names = sorted(
        path.name
        for path in staging.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_lines = []
    for name in checksum_names:
        value = shard_hashes.get(name) or sha256_file(staging / name)
        checksum_lines.append(f"{value}  {name}")
    (staging / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    os.replace(staging, output_dir)
    print(f"HF_PACKAGE_COMPLETE={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("final_model"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("huggingface_release/care-opd-qzbc-4b-final"),
    )
    parser.add_argument(
        "--public-assets-dir",
        type=Path,
        default=Path("github_release/care-opd-qzbc-4b-final"),
    )
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()
    build_release(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.public_assets_dir.resolve(),
        parse_size(args.max_shard_size),
    )


if __name__ == "__main__":
    main()
