#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.features import REQUIRED_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate schema compatibility for release CSV splits.")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="CSV path; pass this argument three times (synthetic/semi/real).",
    )
    parser.add_argument(
        "--output-json",
        default="dataset_release/schema_compatibility_report.json",
        help="Output report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [Path(p) for p in args.csv]
    report: dict[str, object] = {"required_columns": REQUIRED_COLUMNS, "splits": []}
    missing_any = False
    seen_columns: list[set[str]] = []

    for path in csv_paths:
        df = pd.read_csv(path)
        cols = set(df.columns)
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        split_report = {
            "path": str(path),
            "rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "missing_required_columns": missing,
        }
        report["splits"].append(split_report)
        seen_columns.append(cols)
        if missing:
            missing_any = True

    common = sorted(list(set.intersection(*seen_columns))) if seen_columns else []
    report["common_columns_all_splits"] = common
    report["all_splits_have_required_columns"] = not missing_any

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote schema report: {output}")

    if missing_any:
        raise SystemExit("Schema validation failed: one or more splits are missing required columns.")


if __name__ == "__main__":
    main()
