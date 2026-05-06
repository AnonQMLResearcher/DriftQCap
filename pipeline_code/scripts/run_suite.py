#!/usr/bin/env python
"""Run the DriftQCap benchmark suite from the command line."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap import build_profile_config, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DriftQCap synthetic benchmark suite.")
    parser.add_argument("--profile", default="paper", choices=["quick", "paper", "full"], help="Benchmark profile to run.")
    parser.add_argument("--seed", default=20260308, type=int, help="Top-level random seed.")
    parser.add_argument(
        "--base-estimator",
        default=None,
        choices=["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"],
        help="Override base estimator family for adaptation/source panels.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/driftqcap_paper",
        help="Directory where tables, figures, and metadata will be written.",
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional external CSV following the DriftQCap schema. If omitted, the synthetic benchmark is generated.",
    )
    parser.add_argument("--dataset-name", default=None, help="Optional label for an external dataset run.")
    parser.add_argument("--save-models", action="store_true", help="Persist fitted source models into the run artifacts directory.")
    parser.add_argument("--load-models", action="store_true", help="Reuse compatible fitted source models from the run artifacts directory if available.")
    parser.add_argument(
        "--qpa-mode",
        default=None,
        choices=["quick", "slow", "true"],
        help="Preset for qpa_nn only. quick: faster/lighter, slow: stronger/slower, true: heaviest optional run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_profile_config(args.profile, random_seed=args.seed)
    if args.base_estimator is not None:
        config = replace(config, model=replace(config.model, base_estimator=str(args.base_estimator)))
    if args.qpa_mode is not None:
        if config.model.base_estimator != "qpa_nn":
            raise ValueError("--qpa-mode is only valid when --base-estimator qpa_nn.")
        if args.qpa_mode == "quick":
            config = replace(
                config,
                model=replace(
                    config.model,
                    qpa_hidden_layers=(128, 64),
                    qpa_max_iter=250,
                    qpa_n_iter_no_change=25,
                    qpa_learning_rate_init=1e-3,
                    qpa_alpha=1e-4,
                    non_rf_members=4,
                ),
            )
        elif args.qpa_mode == "slow":
            config = replace(
                config,
                model=replace(
                    config.model,
                    qpa_hidden_layers=(256, 128, 64),
                    qpa_max_iter=1200,
                    qpa_n_iter_no_change=60,
                    qpa_learning_rate_init=7e-4,
                    qpa_alpha=5e-5,
                    non_rf_members=8,
                ),
            )
        else:
            config = replace(
                config,
                model=replace(
                    config.model,
                    qpa_hidden_layers=(384, 256, 128, 64),
                    qpa_max_iter=1800,
                    qpa_n_iter_no_change=100,
                    qpa_tol=1e-4,
                    qpa_learning_rate_init=5e-4,
                    qpa_alpha=3e-5,
                    non_rf_members=10,
                ),
            )
    config = config.__class__(
        profile_name=config.profile_name,
        synthetic=config.synthetic,
        model=config.model,
        adaptation=config.adaptation,
        calibration=config.calibration,
        evaluation=config.evaluation,
        reporting=config.reporting.__class__(
            save_raw_predictions=config.reporting.save_raw_predictions,
            save_submission_bundle=config.reporting.save_submission_bundle,
            save_model_artifacts=args.save_models,
            load_model_artifacts=args.load_models,
            artifacts_dirname=config.reporting.artifacts_dirname,
        ),
    )
    artifacts = run_benchmark(
        config,
        dataset=args.dataset_csv,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
    )
    print("Run complete.")
    print(f"Dataset rows: {len(artifacts.dataset)}")
    print(f"Source result rows: {len(artifacts.source_results)}")
    print(f"Adaptation result rows: {len(artifacts.adaptation_results)}")
    print(f"Acquisition result rows: {len(artifacts.acquisition_results)}")
    if not artifacts.external_validity_summary.empty:
        print(f"External-validity rows: {len(artifacts.external_validity_summary)}")
    if artifacts.model_artifact_paths:
        print(f"Model artifacts: {artifacts.model_artifact_paths['artifact_dir']}")
    if artifacts.figure_paths:
        print("Generated figures:")
        for key, value in sorted(artifacts.figure_paths.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
