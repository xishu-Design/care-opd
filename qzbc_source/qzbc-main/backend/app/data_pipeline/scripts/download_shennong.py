from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pyarrow.parquet as pq

from app.core.config import get_settings


DATASET = "michaelwzhu/ShenNong_TCM_Dataset"
PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"
REPO_API = f"https://huggingface.co/api/datasets/{DATASET}"


def download_shennong_rows(*, output_path: Path | str | None = None, revision: str = "main", max_rows: int | None = None) -> dict:
    settings = get_settings()
    if not settings.allow_unverified_internal_sources:
        raise ValueError("ALLOW_UNVERIFIED_INTERNAL_SOURCES=true is required for internal source download")
    output = Path(output_path or Path(settings.internal_knowledge_dir) / "shennong" / f"{revision}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(REPO_API, timeout=60) as response:  # noqa: S310 - fixed public metadata endpoint
        repository = json.loads(response.read().decode("utf-8"))
    immutable_revision = str(repository.get("sha") or revision) if isinstance(repository, dict) else revision
    with urlopen(PARQUET_API, timeout=60) as response:  # noqa: S310 - fixed public metadata endpoint
        manifest = json.loads(response.read().decode("utf-8"))
    files = manifest.get("parquet_files", []) if isinstance(manifest, dict) else []
    if not files or not isinstance(files[0], dict) or not files[0].get("url"):
        raise ValueError("ShenNong parquet manifest is empty")
    parquet_path = output.with_suffix(".parquet")
    urlretrieve(str(files[0]["url"]), parquet_path)  # noqa: S310 - URL returned by fixed Hugging Face metadata endpoint
    total = 0
    table = pq.ParquetFile(parquet_path)
    with output.open("w", encoding="utf-8") as file:
        for batch in table.iter_batches(batch_size=1000):
            columns = batch.to_pydict()
            for index in range(batch.num_rows):
                if max_rows is not None and total >= max_rows:
                    break
                row = {key: values[index] for key, values in columns.items()}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
            if max_rows is not None and total >= max_rows:
                break
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"dataset": DATASET, "revision": immutable_revision, "path": str(output), "parquet_path": str(parquet_path), "rows": total, "sha256": digest, "distribution_scope": "internal_only"}


if __name__ == "__main__":
    print(json.dumps(download_shennong_rows(), ensure_ascii=False, indent=2))
