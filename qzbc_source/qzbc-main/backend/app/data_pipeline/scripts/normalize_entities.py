from __future__ import annotations

import argparse
import json

from app.data_pipeline.normalizers.tcm_entity_normalizer import normalize_entity_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a TCM entity name without adding medical knowledge.")
    parser.add_argument("name")
    args = parser.parse_args(argv)
    print(json.dumps({"original_name": args.name, "normalized_name": normalize_entity_name(args.name)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

