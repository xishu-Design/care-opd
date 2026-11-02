#!/usr/bin/env python3
"""Create a regular-Git-friendly split package for the CARE-OPD checkpoint.

The source checkpoint is never modified.  The builder writes to a staging
directory, verifies every source file, creates fixed-size raw byte chunks, and
atomically renames the staging directory only after all manifests are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    "config.json": (1979, "5970c5b50e94644c199bcfc811e7aad93fd9f63574c9b8522996f13b8710791e"),
    "export_manifest.json": (2434, "8043270cd1f939362a254fc45fb26cc6393775c261c096daf1a6929b62085f8d"),
    "generation_config.json": (116, "4a4505ef94dbb317b8c508ff53777cb0dfed044027864a1c05822c07217251ef"),
    "model.safetensors": (8411558400, "87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4"),
    "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1124, "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"),
}

MODEL_FINGERPRINT = "4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd"

APACHE_LICENSE = """Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


def digest(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def build_readme(part_count: int, chunk_size: int) -> str:
    return f"""# CARE-OPD QZBC 4B merged checkpoint

This repository package contains a **text-only, merged Hugging Face checkpoint**
produced by CARE-OPD training.  The LoRA adapter has already been merged; no
adapter is needed at inference time.

The 8.41 GB `model.safetensors` file is represented as {part_count} raw byte
parts of at most {chunk_size:,} bytes ({chunk_size / 1024 / 1024:.0f} MiB) so
that each Git object remains below GitHub's regular-Git 100 MiB hard limit.
These files are transport chunks, **not Hugging Face safetensors shards**.

## Reassemble

Windows PowerShell:

```powershell
./scripts/assemble.ps1
```

Linux/macOS:

```bash
bash scripts/assemble.sh
```

Or directly:

```bash
python scripts/assemble.py
```

The assembler validates every part, writes a temporary output, validates the
complete file, and then atomically publishes `model/model.safetensors`.  It
refuses to overwrite an existing output.

Expected complete weight SHA-256:

```text
{EXPECTED['model.safetensors'][1]}
```

Formal evaluation model fingerprint (inventory fingerprint, not the weight
file hash):

```text
{MODEL_FINGERPRINT}
```

## Load after assembly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=False,
)
```

## Scope and limitations

- Upstream base model: `Qwen/Qwen3.5-4B`; the exact immutable upstream revision
  was not retained in the exported artifact and is therefore recorded as
  `unknown`, not guessed.
- The checkpoint is the response model used by the QZBC host.  It does not
  contain an Agent Graph, retriever, external tools, or the host verifier.
- Safety and JSON/source-boundary enforcement depend partly on the host system.
- This package is not a clinical diagnostic system and must not replace
  qualified professional review.
- Visual special tokens may exist in the tokenizer/template, but this package
  does not include a vision encoder or image preprocessor and is published as
  text-only.

## License and attribution

The upstream Qwen3.5-4B model is distributed under Apache License 2.0.  This
package contains a fine-tuned, adapter-merged derivative.  See `LICENSE` and
`NOTICE`.  The repository maintainer must confirm rights to the fine-tuning
data and any additional project code before public release.

## GitHub upload

Use command-line Git, not the browser uploader.  Follow
`UPLOAD_TO_GITHUB.md`: upload metadata first, then no more than 20 parts per
commit/push (about 960 MiB) so a single push stays comfortably below GitHub's
2 GiB push limit.  Avoid rewriting part history: old binary objects remain in
the Git history.
"""


ASSEMBLER = r'''#!/usr/bin/env python3
"""Verify and reconstruct model/model.safetensors from raw transport parts."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args=ap.parse_args()
    manifest=json.loads((ROOT/"package_manifest.json").read_text(encoding="utf-8"))
    parts=manifest["chunking"]["parts"]
    for item in parts:
        path=ROOT/item["path"]
        if not path.is_file(): raise SystemExit(f"missing part: {path}")
        if path.stat().st_size != item["bytes"]: raise SystemExit(f"bad length: {path}")
        actual=digest(path)
        if actual != item["sha256"]: raise SystemExit(f"bad sha256: {path}")
        print(f"verified {item['index']:04d}/{len(parts):04d}: {path.name}")
    if args.verify_only:
        print("PACKAGE_VERIFY=PASS")
        return
    out=ROOT/manifest["reassembly"]["output"]
    tmp=out.with_name(out.name+".tmp")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite existing output/temp: {out} / {tmp}")
    out.parent.mkdir(parents=True, exist_ok=True)
    h=hashlib.sha256(); total=0
    try:
        with tmp.open("xb") as dst:
            for item in parts:
                with (ROOT/item["path"]).open("rb") as src:
                    for block in iter(lambda:src.read(8*1024*1024), b""):
                        dst.write(block); h.update(block); total += len(block)
            dst.flush(); os.fsync(dst.fileno())
        artifact=manifest["artifact"]
        if total != artifact["bytes"]: raise RuntimeError(f"bad total bytes: {total}")
        if h.hexdigest() != artifact["sha256"]: raise RuntimeError("bad complete sha256")
        os.replace(tmp,out)
    except BaseException:
        if tmp.exists(): tmp.unlink()
        raise
    print(f"ASSEMBLY=PASS output={out} bytes={total} sha256={h.hexdigest()}")

if __name__=="__main__": main()
'''


def upload_guide(part_count: int) -> str:
    batches = (part_count + 19) // 20
    return f"""# Upload this package to GitHub with regular Git

GitHub regular Git enforces a 100 MiB maximum per object and a 2 GiB maximum
per push.  This package uses 48 MiB parts and must be pushed in {batches}
batches of at most 20 parts.  Do not run `git add .` before the parts are
uploaded, because that would place the whole 8.4 GB model into one push.

## 1. Create an empty GitHub repository

Do not initialize it with a README or license.  Copy its HTTPS or SSH URL.

## 2. Initialize and push metadata

From this package directory:

```bash
git init
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git add .gitattributes .gitignore README.md MODEL_CARD.md DEPLOYMENT.md \
  LICENSE NOTICE PUBLISHING_CHECKLIST.md UPLOAD_TO_GITHUB.md \
  package_manifest.json SHA256SUMS model scripts
git commit -m "Add CARE-OPD model metadata and reconstruction tools"
git push -u origin main
```

`model/model.safetensors` is ignored, so the command above adds only the small
model files.  The parts remain untracked until the next step.

## 3. Add and push no more than 20 parts at a time

PowerShell example:

```powershell
$parts = Get-ChildItem parts/model.safetensors/model.safetensors.part-* | Sort-Object Name
for ($start = 0; $start -lt $parts.Count; $start += 20) {{
    $end = [Math]::Min($start + 19, $parts.Count - 1)
    $batch = $parts[$start..$end].FullName
    git add -- $batch
    git commit -m "Add model parts $($start + 1)-$($end + 1) of $($parts.Count)"
    git push origin main
    if ($LASTEXITCODE -ne 0) {{ throw "Git push failed at part $($start + 1)" }}
}}
```

Bash example:

```bash
mapfile -t parts < <(find parts/model.safetensors -maxdepth 1 -type f \
  -name 'model.safetensors.part-*' | sort)
for ((start=0; start<${{#parts[@]}}; start+=20)); do
  batch=("${{parts[@]:start:20}}")
  end=$((start+${{#batch[@]}}))
  git add -- "${{batch[@]}}"
  git commit -m "Add model parts $((start+1))-${{end}} of ${{#parts[@]}}"
  git push origin main || exit 1
done
```

## 4. Clone and reconstruct

```bash
git clone YOUR_REPOSITORY_URL
cd REPOSITORY_NAME
python scripts/assemble.py
```

Keep this repository model-only and avoid replacing parts in later commits.
For routine model hosting, Git LFS, Hugging Face Hub, or object storage is more
appropriate than an 8+ GB regular Git history.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("final_model"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("github_release/care-opd-qzbc-4b-final"),
    )
    ap.add_argument("--chunk-mib", type=int, default=48)
    args = ap.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".building")
    chunk_size = args.chunk_mib * 1024 * 1024
    if not (1 <= args.chunk_mib < 50):
        raise SystemExit("--chunk-mib must be between 1 and 49 for this regular-Git package")
    if output.exists() or staging.exists():
        raise SystemExit(f"refusing existing output/staging: {output} / {staging}")

    print("Validating source checkpoint...")
    for name, (expected_bytes, expected_hash) in EXPECTED.items():
        path = source / name
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise SystemExit(f"source size mismatch: {path}")
        actual = digest(path)
        if actual != expected_hash:
            raise SystemExit(f"source sha256 mismatch: {path}")
        print(f"  PASS {name} {expected_bytes} {actual}")

    try:
        (staging / "model").mkdir(parents=True)
        parts_dir = staging / "parts" / "model.safetensors"
        parts_dir.mkdir(parents=True)
        (staging / "scripts").mkdir(parents=True)

        small_files = []
        for name in (
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            dst = staging / "model" / name
            shutil.copyfile(source / name, dst)
            small_files.append(
                {"path": dst.relative_to(staging).as_posix(), "bytes": dst.stat().st_size, "sha256": digest(dst)}
            )

        raw_export = json.loads((source / "export_manifest.json").read_text(encoding="utf-8"))
        public_export = {
            "schema_version": "care_opd.public_export_manifest.v1",
            "artifact_type": "merged_huggingface_checkpoint",
            "adapter_merged": True,
            "dtype": raw_export.get("dtype", "bfloat16"),
            "upstream_base_model": {"model_id": "Qwen/Qwen3.5-4B", "revision": "unknown_not_recorded"},
            "weight_file": {
                "path_after_reassembly": "model.safetensors",
                "bytes": EXPECTED["model.safetensors"][0],
                "sha256": EXPECTED["model.safetensors"][1],
            },
            "config_sha256": EXPECTED["config.json"][1],
            "evaluation_model_fingerprint": MODEL_FINGERPRINT,
            "qzbc_contract_fingerprint": raw_export.get("qzbc_contract", {}).get("contract_fingerprint"),
            "privacy_note": "Internal training paths and source-tree inventory were removed from the public manifest.",
        }
        public_export_path = staging / "model" / "export_manifest.public.json"
        write_text(public_export_path, json.dumps(public_export, ensure_ascii=False, indent=2))
        small_files.append(
            {
                "path": public_export_path.relative_to(staging).as_posix(),
                "bytes": public_export_path.stat().st_size,
                "sha256": digest(public_export_path),
            }
        )

        weight = source / "model.safetensors"
        part_count = (weight.stat().st_size + chunk_size - 1) // chunk_size
        parts = []
        print(f"Splitting {weight} into {part_count} parts of at most {chunk_size} bytes...")
        with weight.open("rb") as src:
            offset = 0
            for index in range(1, part_count + 1):
                name = f"model.safetensors.part-{index:04d}-of-{part_count:04d}"
                path = parts_dir / name
                h = hashlib.sha256()
                written = 0
                with path.open("xb") as dst:
                    remaining = chunk_size
                    while remaining:
                        block = src.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            break
                        dst.write(block)
                        h.update(block)
                        written += len(block)
                        remaining -= len(block)
                if written == 0:
                    raise RuntimeError("unexpected empty part")
                parts.append(
                    {
                        "index": index,
                        "path": path.relative_to(staging).as_posix(),
                        "offset_bytes": offset,
                        "bytes": written,
                        "sha256": h.hexdigest(),
                    }
                )
                offset += written
                print(f"  {index:04d}/{part_count:04d} {name} {written} {h.hexdigest()}")
        if offset != weight.stat().st_size:
            raise RuntimeError(f"split total mismatch: {offset}")

        manifest = {
            "schema_version": "care_opd.github_split_package.v1",
            "status": "COMPLETE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": {
                "path_after_reassembly": "model/model.safetensors",
                "bytes": EXPECTED["model.safetensors"][0],
                "sha256": EXPECTED["model.safetensors"][1],
                "format": "safetensors",
            },
            "evaluation_model_fingerprint": MODEL_FINGERPRINT,
            "chunking": {
                "algorithm": "raw-byte-concatenation",
                "chunk_size_bytes": chunk_size,
                "part_count": part_count,
                "ordering": "index-ascending; fixed-width filenames",
                "parts": parts,
            },
            "small_model_files": small_files,
            "reassembly": {
                "output": "model/model.safetensors",
                "script": "scripts/assemble.py",
                "overwrite_policy": "refuse",
            },
            "source_publication": {
                "upstream_base_model": "Qwen/Qwen3.5-4B",
                "upstream_revision": "unknown_not_recorded",
                "adapter_merged": True,
                "raw_internal_export_manifest_included": False,
            },
        }
        write_text(staging / "package_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        write_text(staging / "README.md", build_readme(part_count, chunk_size))
        write_text(
            staging / "MODEL_CARD.md",
            """# Model card — CARE-OPD QZBC 4B

## Model description

- Architecture: Qwen3.5 4B causal language model, text checkpoint.
- Upstream identifier: `Qwen/Qwen3.5-4B` (exact revision not recorded).
- Adaptation: CARE-OPD fine-tuning with merged LoRA adapter.
- Output role: structured QZBC `GeneratedResponse` generation from chat messages
  containing an Evidence Packet.

## Intended use

Research and engineering evaluation of evidence-bound structured response
generation.  The checkpoint should be integrated with the matching prompt,
schema validation, source-boundary checks, and human review appropriate to the
application.

## Out-of-scope use

Do not use as an autonomous medical diagnostic or treatment system, as a
replacement for qualified professionals, or as evidence that the complete
host Agent/tool-routing system has been validated.

## Known limitations

- Evaluation is strongest on the QZBC contract and controlled evidence-state
  perturbations; it does not establish general OOD or clinical validity.
- Host verifier and safety rules are not embedded in this checkpoint.
- The exported metadata did not retain an immutable upstream revision.
- This split package is transport packaging only; reconstruct the exact weight
  file before loading.

## Reproducibility identifiers

- Weight SHA-256: `87670f85af0db2abf466623ef5fcc2e4f1ac727aa07a2e58c2b15ff246a09fc4`
- Formal evaluation model fingerprint:
  `4cf77e31795163f32fa4e6715d42d06fec88ff98801b154934795b98281ecdcd`
""",
        )
        write_text(
            staging / "DEPLOYMENT.md",
            """# Deployment

1. Reconstruct `model/model.safetensors` with `python scripts/assemble.py`.
2. Install compatible versions of PyTorch, Transformers, Accelerate and
   Safetensors.
3. Load the local `model` directory with the standard Hugging Face APIs shown
   in `README.md`.
4. Integrate the checkpoint with the QZBC prompt/schema/verifier contract if
   reproducing the reported structured-response behavior.

The package contains no server, Agent Graph, retriever, tool router, API key or
external safety service.  `model/model.safetensors` is intentionally ignored by
Git after reconstruction.
""",
        )
        write_text(staging / "LICENSE", APACHE_LICENSE)
        write_text(
            staging / "NOTICE",
            """CARE-OPD QZBC 4B merged checkpoint

This package contains a fine-tuned and adapter-merged derivative of
Qwen/Qwen3.5-4B.  The upstream model is distributed under Apache License 2.0.

Modifications: CARE-OPD post-training and checkpoint merge.  Internal server
paths and private source-tree inventory were removed from the public export
metadata.

The exact immutable upstream revision was not retained in the exported
artifact and is intentionally reported as unknown rather than inferred.
""",
        )
        write_text(
            staging / "PUBLISHING_CHECKLIST.md",
            """# Required maintainer checks before making the repository public

- [ ] Confirm the repository owner agrees to publish the derivative checkpoint
      under the included Apache-2.0 terms.
- [ ] Confirm the exact upstream Qwen license/revision used; add the immutable
      revision when it can be proven from acquisition records.
- [ ] Confirm all 12k QZBC training seeds are synthetic or properly
      de-identified and contain no PHI/PII, secrets, private prompts, or data
      lacking redistribution rights.
- [ ] Confirm any QZBC names, schema, prompt and contract information included
      here are authorized for public release.
- [ ] Review the model card's medical-use limitation and host-verifier
      dependency.
- [ ] Run `python scripts/assemble.py --verify-only` before the first push.
- [ ] Push at most 20 parts per commit/push; never `git add .` before batching.
""",
        )
        write_text(staging / "UPLOAD_TO_GITHUB.md", upload_guide(part_count))
        write_text(staging / ".gitattributes", "*.part-* binary -diff -merge -delta\n")
        write_text(
            staging / ".gitignore",
            "/model/model.safetensors\n/model/model.safetensors.tmp*\n__pycache__/\n*.pyc\n",
        )
        write_text(staging / "scripts" / "assemble.py", ASSEMBLER)
        write_text(
            staging / "scripts" / "assemble.ps1",
            "$ErrorActionPreference = 'Stop'\n$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { 'python' }\n& $Python \"$PSScriptRoot/assemble.py\" @args\nexit $LASTEXITCODE\n",
        )
        write_text(
            staging / "scripts" / "assemble.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\nPYTHON_BIN=\"${PYTHON_BIN:-python3}\"\nexec \"$PYTHON_BIN\" \"$(dirname \"$0\")/assemble.py\" \"$@\"\n",
        )

        sums = []
        for path in sorted(p for p in staging.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
            sums.append(f"{digest(path)}  {path.relative_to(staging).as_posix()}")
        write_text(staging / "SHA256SUMS", "\n".join(sums))

        os.replace(staging, output)
        print(f"PACKAGE=COMPLETE path={output}")
        print(f"PARTS={part_count} CHUNK_BYTES={chunk_size}")
    except BaseException:
        print(f"PACKAGE=FAILED staging retained for inspection: {staging}")
        raise


if __name__ == "__main__":
    main()
