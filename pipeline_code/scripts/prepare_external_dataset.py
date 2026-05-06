#!/usr/bin/env python
"""Validate and normalize an external DriftQCap-compatible dataset CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.data import prepare_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an external measured-data CSV into DriftQCap schema.")
    parser.add_argument("--input-csv", required=True, help="Input CSV already containing DriftQCap-compatible circuit-level columns.")
    parser.add_argument("--output-csv", required=True, help="Output normalized CSV path.")
    parser.add_argument("--pass-threshold", type=float, default=0.080, help="Pass/fail threshold used if pass_label is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    df = pd.read_csv(input_path)
    normalized = prepare_dataframe(df, pass_threshold=args.pass_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(f"Rows: {len(normalized)}")
    print(f"Columns: {len(normalized.columns)}")


if __name__ == "__main__":
    main()
