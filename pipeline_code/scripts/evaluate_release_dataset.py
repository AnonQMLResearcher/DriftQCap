#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate any frozen release CSV with the paper profile.")
    parser.add_argument("--dataset-csv", required=True, help="Path to frozen dataset CSV.")
    parser.add_argument("--dataset-name", required=True, help="Short label for reporting.")
    parser.add_argument("--output-dir", required=True, help="Output run directory.")
    parser.add_argument("--seed", type=int, default=20260308, help="Top-level run seed.")
    parser.add_argument("--skip-bundle", action="store_true", help="Skip run-level zip bundle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/run_paper_ready.py",
        "--profile",
        "paper",
        "--seed",
        str(args.seed),
        "--dataset-csv",
        str(Path(args.dataset_csv)),
        "--dataset-name",
        args.dataset_name,
        "--output-dir",
        str(Path(args.output_dir)),
    ]
    if args.skip_bundle:
        cmd.append("--skip-bundle")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
