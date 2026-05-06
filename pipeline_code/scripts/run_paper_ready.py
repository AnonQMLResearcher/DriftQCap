#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap import build_profile_config, run_benchmark
from driftqcap.paper_utils import build_readiness_report, export_standard_latex_tables
from make_submission_bundle import create_submission_bundle

DEFAULT_OUTPUTS = {
    "quick": "outputs/driftqcap_quick_ready",
    "paper": "outputs/driftqcap_paper_ready",
    "full": "outputs/driftqcap_full_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DriftQCap and export paper-facing artifacts in one command.")
    parser.add_argument("--profile", default="paper", choices=["quick", "paper", "full"], help="Benchmark profile to run.")
    parser.add_argument("--seed", default=20260308, type=int, help="Top-level random seed.")
    parser.add_argument(
        "--base-estimator",
        default=None,
        choices=["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"],
        help="Override base estimator family for adaptation/source panels.",
    )
    parser.add_argument("--output-dir", default=None, help="Run directory to create. Defaults to outputs/driftqcap_<profile>_ready.")
    parser.add_argument("--dataset-csv", default=None, help="Optional external CSV following the DriftQCap schema.")
    parser.add_argument("--dataset-name", default=None, help="Optional label for an external dataset run.")
    parser.add_argument("--pass-threshold", default=None, type=float, help="Override pass/fail threshold for this run.")
    parser.add_argument("--oracle-search-trials", default=None, type=int, help="Enable leaky true-oracle subset search with N random subset trials (diagnostic only).")
    parser.add_argument("--target-recalibration", action="store_true", help="Enable target-aware isotonic recalibration for adapted/acquisition runs.")
    parser.add_argument("--target-recalibration-mode", default=None, choices=["target_only", "source_plus_target"], help="Target-aware recalibration mode.")
    parser.add_argument("--conformal-recalibration-mode", default=None, choices=["source_only", "target_only", "source_plus_target"], help="Conformal interval recalibration mode.")
    parser.add_argument("--min-target-calibration-rows", default=None, type=int, help="Minimum rows required for target-aware calibration before fallback.")
    parser.add_argument("--prob-calibrator", default=None, choices=["isotonic", "platt"], help="Probability calibrator type.")
    parser.add_argument("--calibration-shrink-tau", default=None, type=float, help="Shrinkage tau for blending calibrated and raw probabilities.")
    parser.add_argument("--uncertainty-cap-quantile", default=None, type=float, help="Cap quantile for uncertainty-capped acquisition.")
    parser.add_argument("--density-weight", default=None, type=float, help="Representativeness weight for capped acquisition.")
    parser.add_argument("--enable-hybrid-shift-policy", action="store_true", help="Evaluate hybrid shift-aware acquisition (random for depth/family, diversity for noise/temporal by default).")
    parser.add_argument("--stratified-diversity-bins", default=None, type=int, help="Number of predicted-error strata for stratified diversity.")
    parser.add_argument("--disable-stratified-diversity-family", action="store_true", help="Disable family-level stratification within stratified diversity.")
    parser.add_argument("--std-temperature-quantile", default=None, type=float, help="Quantile used to fit adapter std temperature.")
    parser.add_argument("--std-temperature-min", default=None, type=float, help="Minimum adapter std temperature.")
    parser.add_argument("--std-temperature-max", default=None, type=float, help="Maximum adapter std temperature.")
    parser.add_argument("--std-floor", default=None, type=float, help="Minimum predicted std after adaptation.")
    parser.add_argument("--random-repeats", default=None, type=int, help="Override number of random repeats for adaptation/acquisition.")
    parser.add_argument(
        "--acquisition-budgets",
        default=None,
        help="Comma-separated acquisition budgets (e.g. 4,8,16,24). Keeps adaptation budgets unchanged.",
    )
    parser.add_argument("--save-models", action="store_true", help="Persist fitted source models into the run artifacts directory.")
    parser.add_argument("--load-models", action="store_true", help="Reuse compatible fitted source models from the run artifacts directory if available.")
    parser.add_argument("--skip-bundle", action="store_true", help="Skip building the supplementary ZIP bundle.")
    parser.add_argument(
        "--qpa-mode",
        default=None,
        choices=["quick", "slow", "true"],
        help="Preset for qpa_nn only. quick: faster/lighter, slow: stronger/slower, true: heaviest optional run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_OUTPUTS[args.profile])
    config = build_profile_config(args.profile, random_seed=args.seed)
    synthetic = config.synthetic
    if args.pass_threshold is not None:
        synthetic = replace(synthetic, pass_threshold=float(args.pass_threshold))
    adaptation = config.adaptation
    if args.oracle_search_trials is not None:
        adaptation = replace(adaptation, oracle_search_trials=max(int(args.oracle_search_trials), 0))
    if args.uncertainty_cap_quantile is not None:
        adaptation = replace(adaptation, uncertainty_cap_quantile=float(args.uncertainty_cap_quantile))
    if args.density_weight is not None:
        adaptation = replace(adaptation, density_weight=float(args.density_weight))
    if args.enable_hybrid_shift_policy:
        adaptation = replace(adaptation, enable_hybrid_shift_policy=True)
    if args.stratified_diversity_bins is not None:
        adaptation = replace(adaptation, stratified_diversity_bins=max(int(args.stratified_diversity_bins), 2))
    if args.disable_stratified_diversity_family:
        adaptation = replace(adaptation, stratified_diversity_family_stratify=False)
    if args.std_temperature_quantile is not None:
        adaptation = replace(adaptation, std_temperature_quantile=float(args.std_temperature_quantile))
    if args.std_temperature_min is not None:
        adaptation = replace(adaptation, std_temperature_min=float(args.std_temperature_min))
    if args.std_temperature_max is not None:
        adaptation = replace(adaptation, std_temperature_max=float(args.std_temperature_max))
    if args.std_floor is not None:
        adaptation = replace(adaptation, std_floor=max(float(args.std_floor), 1e-8))
    if args.random_repeats is not None:
        adaptation = replace(adaptation, random_repeats=max(int(args.random_repeats), 1))
    if args.acquisition_budgets is not None:
        parsed = tuple(sorted({int(part.strip()) for part in str(args.acquisition_budgets).split(",") if part.strip()}))
        parsed = tuple(value for value in parsed if value > 0)
        adaptation = replace(adaptation, acquisition_budgets=parsed)
    calibration = config.calibration
    if args.target_recalibration:
        calibration = replace(calibration, target_recalibration=True)
    if args.target_recalibration_mode is not None:
        calibration = replace(calibration, target_recalibration_mode=args.target_recalibration_mode)
    if args.conformal_recalibration_mode is not None:
        calibration = replace(calibration, conformal_recalibration_mode=args.conformal_recalibration_mode)
    if args.min_target_calibration_rows is not None:
        calibration = replace(calibration, min_target_calibration_rows=max(int(args.min_target_calibration_rows), 1))
    if args.prob_calibrator is not None:
        calibration = replace(calibration, prob_calibrator=args.prob_calibrator)
    if args.calibration_shrink_tau is not None:
        calibration = replace(calibration, calibration_shrink_tau=max(float(args.calibration_shrink_tau), 0.0))
    config = replace(
        config,
        model=replace(config.model, base_estimator=args.base_estimator) if args.base_estimator is not None else config.model,
        synthetic=synthetic,
        adaptation=adaptation,
        calibration=calibration,
        reporting=replace(
            config.reporting,
            save_model_artifacts=args.save_models,
            load_model_artifacts=args.load_models,
        ),
    )
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
    artifacts = run_benchmark(config, dataset=args.dataset_csv, output_dir=output_dir, dataset_name=args.dataset_name)
    latex_dir = output_dir / "latex_tables"
    export_standard_latex_tables(run_dir=output_dir, output_dir=latex_dir)
    readiness_path = output_dir / "readiness_report.md"
    readiness_path.write_text(build_readiness_report(output_dir), encoding="utf-8")
    if not args.skip_bundle:
        bundle_path = output_dir / "dist" / f"driftqcap_{args.profile}_submission_bundle.zip"
        create_submission_bundle(repo_root=Path("."), run_dir=output_dir, output_zip=bundle_path)
    print("Run complete.")
    print(f"Base estimator family: {config.model.base_estimator}")
    print(f"Dataset rows: {len(artifacts.dataset)}")
    print(f"Pass threshold: {config.synthetic.pass_threshold:.3f}")
    print(f"Oracle true-search trials: {config.adaptation.oracle_search_trials}")
    print(f"Hybrid shift policy enabled: {config.adaptation.enable_hybrid_shift_policy}")
    print(
        "Stratified diversity: "
        f"bins={config.adaptation.stratified_diversity_bins}, "
        f"family_stratify={config.adaptation.stratified_diversity_family_stratify}"
    )
    print(f"Target recalibration: {config.calibration.target_recalibration} ({config.calibration.target_recalibration_mode})")
    print(f"Probability calibrator: {config.calibration.prob_calibrator}, shrink tau: {config.calibration.calibration_shrink_tau:.3f}")
    print(
        "Adapter std temperature: "
        f"q={config.adaptation.std_temperature_quantile:.3f}, "
        f"range=[{config.adaptation.std_temperature_min:.3f}, {config.adaptation.std_temperature_max:.3f}], "
        f"floor={config.adaptation.std_floor:.6f}"
    )
    print(f"Figures: {len(artifacts.figure_paths)}")
    if artifacts.model_artifact_paths:
        print(f"Model artifacts directory: {artifacts.model_artifact_paths['artifact_dir']}")
    if not artifacts.external_validity_summary.empty:
        print(f"External-validity rows: {len(artifacts.external_validity_summary)}")
    print(f"LaTeX tables directory: {latex_dir}")
    print(f"Readiness report: {readiness_path}")
    if not args.skip_bundle:
        print(f"Submission bundle: {bundle_path}")


if __name__ == "__main__":
    main()
