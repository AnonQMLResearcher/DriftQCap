#!/usr/bin/env python
"""Create a compact DriftQCap code/results bundle suitable for supplementary upload."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_CODE_GLOBS = [
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "src/driftqcap/*.py",
    "scripts/*.py",
    "tests/*.py",
]

DEFAULT_RUN_GLOBS = [
    "paper_asset_manifest.md",
    "run_report.md",
    "metadata/config.json",
    "metadata/environment.json",
    "tables/source_results.csv",
    "tables/adaptation_summary.csv",
    "tables/acquisition_summary.csv",
    "tables/adaptation_stats.csv",
    "tables/acquisition_stats.csv",
    "figures/*.png",
]


EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact DriftQCap submission bundle.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing pyproject.toml and src/.")
    parser.add_argument("--run-dir", default=None, help="Optional run directory to bundle summary artifacts from.")
    parser.add_argument("--output-zip", default="dist/driftqcap_submission_bundle.zip", help="Output ZIP path.")
    return parser.parse_args()


def _iter_paths(root: Path, globs: list[str]) -> list[Path]:
    items: list[Path] = []
    for pattern in globs:
        items.extend(root.glob(pattern))
    deduped = []
    seen = set()
    for item in items:
        if not item.is_file():
            continue
        if any(part in EXCLUDE_PARTS for part in item.parts):
            continue
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def create_submission_bundle(*, repo_root: str | Path = ".", run_dir: str | Path | None = None, output_zip: str | Path = "dist/driftqcap_submission_bundle.zip") -> Path:
    """Create a compact code/results ZIP suitable for supplementary upload."""
    repo_root = Path(repo_root).resolve()
    output_zip = Path(output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    resolved_run_dir = Path(run_dir).resolve() if run_dir is not None else None

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _iter_paths(repo_root, DEFAULT_CODE_GLOBS):
            zf.write(path, arcname=path.relative_to(repo_root))
        if resolved_run_dir is not None and resolved_run_dir.exists():
            for path in _iter_paths(resolved_run_dir, DEFAULT_RUN_GLOBS):
                zf.write(path, arcname=Path("run_artifacts") / path.relative_to(resolved_run_dir))
    return output_zip


def main() -> None:
    args = parse_args()
    output_zip = create_submission_bundle(repo_root=args.repo_root, run_dir=args.run_dir, output_zip=args.output_zip)
    print(f"Wrote {output_zip}")


if __name__ == "__main__":
    main()
