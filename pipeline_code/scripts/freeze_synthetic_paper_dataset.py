#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap import build_profile_config, run_benchmark


EXPECTED_ROWS = 5152
EXPECTED_SOURCE = 1120
EXPECTED_TARGET = 4032


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the paper-profile synthetic dataset to CSV.")
    parser.add_argument(
        "--output-csv",
        default="dataset_release/synthetic/driftqcap_synthetic_paper_rf_v3.csv",
        help="Output path for the frozen synthetic CSV.",
    )
    parser.add_argument("--seed", type=int, default=20260308, help="Synthetic generation seed.")
    parser.add_argument(
        "--scratch-output-dir",
        default="outputs/_dataset_freeze_tmp",
        help="Scratch run output directory used to materialize the in-memory dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scratch_output_dir = Path(args.scratch_output_dir)

    config = build_profile_config("paper", random_seed=args.seed)
    artifacts = run_benchmark(config, dataset=None, output_dir=scratch_output_dir)
    df: pd.DataFrame = artifacts.dataset.copy()

    rows = len(df)
    source_rows = int((df["domain"] == "source").sum())
    target_rows = int((df["domain"] == "target").sum())
    if (rows, source_rows, target_rows) != (EXPECTED_ROWS, EXPECTED_SOURCE, EXPECTED_TARGET):
        raise SystemExit(
            "Synthetic freeze count mismatch: "
            f"rows={rows}, source={source_rows}, target={target_rows} "
            f"(expected {EXPECTED_ROWS}/{EXPECTED_SOURCE}/{EXPECTED_TARGET})"
        )

    df.to_csv(output_csv, index=False)
    print(f"Wrote frozen synthetic CSV: {output_csv}")
    print(f"Rows={rows}, source={source_rows}, target={target_rows}")


if __name__ == "__main__":
    main()
