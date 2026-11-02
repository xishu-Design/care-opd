from __future__ import annotations

import argparse
import json

from app.data_pipeline.validators.license_validator import generate_license_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify licenses for downloaded GuardTCM data sources.")
    parser.add_argument("--source", help="Verify a single downloaded source_id.")
    args = parser.parse_args(argv)
    report = generate_license_report(source_id=args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
