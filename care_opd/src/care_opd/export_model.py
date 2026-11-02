"""Export the trained policy as a standalone Hugging Face checkpoint for qzbc."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .models import ModelArguments, load_causal_lm, load_tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(path: Path) -> list[Path]:
    return sorted(
        [
            *path.glob("*.safetensors"),
            *path.glob("*.bin"),
        ]
    )


def export_standalone(
    base_model: str,
    output: Path,
    *,
    adapter: Optional[Path] = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    contract_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge PEFT into the base and save files loadable without CARE-OPD/PEFT."""

    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite {output}; choose a new versioned export directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    started_at = time.monotonic()

    def report(step: int, status: str) -> None:
        elapsed = int(time.monotonic() - started_at)
        print(
            "[progress] "
            f"stage=merge status={status} step={step}/5 "
            f"pct={step / 5 * 100:.2f}% elapsed_seconds={elapsed}",
            flush=True,
        )

    try:
        model_args = ModelArguments(
            student_model_name_or_path=base_model,
            tokenizer_name_or_path=base_model,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        report(1, "loading_tokenizer")
        tokenizer = load_tokenizer(model_args)
        report(2, "loading_base_model")
        model = load_causal_lm(
            base_model,
            model_args,
            trainable=False,
        )
        merged_adapter = False
        report(3, "merging_adapter" if adapter is not None else "adapter_not_requested")
        if adapter is not None:
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(adapter / "adapter_config.json")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("adapter export requires peft") from exc
            model = PeftModel.from_pretrained(model, str(adapter))
            model = model.merge_and_unload(safe_merge=True)
            merged_adapter = True

        report(4, "saving_weights")
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        config_path = temporary / "config.json"
        weights = _weight_files(temporary)
        if not config_path.is_file() or not weights:
            raise RuntimeError("standalone export is missing config.json or model weights")
        forbidden_adapter_files = [
            temporary / "adapter_config.json",
            temporary / "adapter_model.safetensors",
            temporary / "adapter_model.bin",
        ]
        leaked = [path.name for path in forbidden_adapter_files if path.exists()]
        if leaked:
            raise RuntimeError(f"export still depends on a PEFT adapter: {leaked}")

        qzbc_contract: Dict[str, Any] | None = None
        if contract_manifest is not None:
            payload = json.loads(contract_manifest.read_text(encoding="utf-8"))
            qzbc_contract = payload.get("qzbc_contract", payload)

        manifest = {
            "schema_version": "care-opd-qzbc-standalone-v1",
            "artifact_type": "merged_huggingface_checkpoint",
            "base_model": base_model,
            "adapter": str(adapter.resolve()) if adapter is not None else None,
            "adapter_merged": merged_adapter,
            "dtype": dtype,
            "config_sha256": _sha256(config_path),
            "weight_files": [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in weights
            ],
            "qzbc_contract": qzbc_contract,
            "runtime_dependencies": {
                "care_opd": False,
                "CareRecoveryEnv": False,
                "peft_adapter": False,
            },
            "qzbc_integration": {
                "openai_compatible": "serve this directory with vLLM/SGLang",
                "ollama": "convert this directory to GGUF, then run ollama create",
            },
        }
        (temporary / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "QZBC_DEPLOYMENT.md").write_text(
            """# qzbc deployment

This is a standalone merged model. CARE-OPD, PEFT, and CareRecoveryEnv are not
needed at inference time.

## OpenAI-compatible server (recommended for the HF checkpoint)

```bash
vllm serve . --served-model-name qzbc-care-opd
```

Set in qzbc `backend/.env`:

```dotenv
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_COMPATIBLE_API_KEY=local
OPENAI_COMPATIBLE_MODEL=qzbc-care-opd
```

## Ollama

Use llama.cpp's current `convert_hf_to_gguf.py` for this model architecture,
quantize the resulting GGUF if desired, then:

```bash
printf 'FROM /absolute/path/qzbc-care-opd.gguf\\n' > Modelfile
ollama create qzbc-care-opd:4b -f Modelfile
```

Set:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qzbc-care-opd:4b
```
""",
            encoding="utf-8",
        )
        report(5, "publishing")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    report(5, "complete")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge and export a standalone qzbc replacement base model"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--contract-manifest", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    manifest = export_standalone(
        args.base_model,
        args.output,
        adapter=args.adapter,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        contract_manifest=args.contract_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
