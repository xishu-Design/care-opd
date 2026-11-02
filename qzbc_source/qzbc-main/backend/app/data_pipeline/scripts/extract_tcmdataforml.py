from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data_pipeline.extractors.tcmdataforml_extractor import TCMDataForMLExtractor
from app.data_pipeline.validators.license_validator import DEFAULT_RAW_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract TCMDataForML raw files into candidate structures.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    args = parser.parse_args(argv)
    extractor = TCMDataForMLExtractor(
        Path(args.raw_dir),
        {"source_id": "SRC_TCM_DATA_FOR_ML", "license": "MIT"},
        {"source_id": "SRC_TCM_DATA_FOR_ML", "detected_license": "MIT", "can_process": True},
    )
    print(json.dumps(extractor.extract(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
