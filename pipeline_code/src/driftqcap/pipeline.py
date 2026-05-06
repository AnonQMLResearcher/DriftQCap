"""End-to-end experiment pipeline for DriftQCap."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .active import (
    select_badge_proxy,
    select_density_ratio_uncertainty_diversity,
    select_diversity,
    select_two_stage_explore_exploit,
    select_entropy_threshold_diversity,
    select_oracle_residual,
    select_random,
    select_shift_aware_diversity,
    select_stratified_diversity,
    select_uncertainty,
    select_uncertainty_threshold_diversity,
    select_uncertainty_threshold_diversity_capped,
)
from .calibration import (
    CalibrationArtifacts,
    attach_intervals,
    blend_conformal_artifacts,
    calibrate_pass_probability,
    fit_pass_probability_calibrator,
    fit_pass_probability_platt,
    fit_locally_adaptive_conformal,
    fit_scaled_conformal,
)
from .config import BenchmarkRunConfig
from .data import make_source_splits, prepare_dataframe, split_target_episode
from .evaluation import build_prediction_frame, combine_metrics
from .logging_utils import configure_logging
from .models import (
    BaseCapabilityRegressor,
    ElasticResidualAdapter,
    EWCResidualAdapter,
    FewShotResidualAdapter,
    MeanShiftAdapter,
    attach_pass_probability,
    fit_pooled_retrain_model,
    fit_target_only_model,
    make_base_model,
)
from .plots import (
    plot_acquisition_curves,
    plot_adaptation_curves,
    plot_auc_ranking,
    plot_dataset_error_distribution,
    plot_interval_coverage_curves,
    plot_reliability_diagram_equal_mass,
    plot_reliability_diagram_from_predictions,
    plot_zero_shot_shift_mae,
    save_figure,
)
from .persistence import load_source_artifacts, save_source_artifacts
from .reporting import (
    select_overall_rows,
    rank_strategies,
    summarize_dataset,
    write_csv_tables,
    write_paper_asset_manifest,
    write_run_report,
)
from .stats import comparisons_to_frame, episode_level_auc, paired_strategy_comparison
from .synthetic import generate_synthetic_benchmark
from .utils import collect_environment_metadata, stable_int_hash, write_json


LOGGER = configure_logging()


@dataclass
class BenchmarkArtifacts:
    """Persisted results from a benchmark run."""

    dataset: pd.DataFrame
    source_results: pd.DataFrame
    adaptation_results: pd.DataFrame
    acquisition_results: pd.DataFrame
    raw_predictions: pd.DataFrame
    dataset_tables: dict[str, pd.DataFrame]
    adaptation_summary: pd.DataFrame
    acquisition_summary: pd.DataFrame
    adaptation_auc: pd.DataFrame
    acquisition_auc: pd.DataFrame
    adaptation_stats: pd.DataFrame
    acquisition_stats: pd.DataFrame
    acquisition_budget_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    acquisition_weight_selection_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    adapted_probability_diagnostic: pd.DataFrame = field(default_factory=pd.DataFrame)
    external_validity_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    external_validity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_artifact_paths: dict[str, str] = field(default_factory=dict)
    figure_paths: dict[str, str] = field(default_factory=dict)


_METRIC_COLUMNS = [
    "mae",
    "rmse",
    "r2",
    "spearman",
    "coverage_80",
    "coverage_90",
    "coverage_95",
    "width_80",
    "width_90",
    "width_95",
    "winkler_80",
    "winkler_90",
    "winkler_95",
    "coverage_80_stratum_0",
    "coverage_80_stratum_1",
    "coverage_80_stratum_2",
    "coverage_80_stratum_3",
    "width_80_stratum_0",
    "width_80_stratum_1",
    "width_80_stratum_2",
    "width_80_stratum_3",
    "std_temperature",
    "selected_alpha",
    "selected_ewc_lambda",
    "design_dim",
    "mean_shift",
    "ewc_lambda",
    "brier_raw",
    "brier_calibrated",
    "ece_raw",
    "ece_calibrated",
    "auroc_raw",
    "auroc_calibrated",
    "auprc_raw",
    "auprc_calibrated",
    "f1_raw",
    "f1_calibrated",
    "pass_rate",
]

_ADAPTED_STRATEGIES = {
    "era_adapter",
    "residual_adapter",
    "ewc_adapter",
    "mean_shift_adapter",
    "shuffled_label_adapter",
}


def _fit_conformal_from_df(
    model: BaseCapabilityRegressor,
    calibration_df: pd.DataFrame,
    *,
    config: BenchmarkRunConfig,
):
    bundle = model.predict_distribution(calibration_df)
    conformal_kwargs = dict(
        y_true=calibration_df["error_rate"].to_numpy(dtype=float),
        pred_mean=bundle.mean,
        pred_std=bundle.std,
        alphas=config.calibration.interval_alphas,
        min_std=config.calibration.min_std,
    )
    if config.calibration.conformal_mode == "locally_adaptive":
        return fit_locally_adaptive_conformal(
            **conformal_kwargs,
            n_bins=config.calibration.conformal_n_bins,
            min_bin_size=config.calibration.conformal_min_bin_size,
            fallback_to_global=config.calibration.conformal_fallback_to_global,
            stratify_by=config.calibration.conformal_strata,
        )
    return fit_scaled_conformal(**conformal_kwargs)


def _fit_prob_calibrator_from_df(
    model: BaseCapabilityRegressor,
    calibration_df: pd.DataFrame,
    *,
    config: BenchmarkRunConfig,
    calibrator_kind: str,
):
    bundle = model.predict_distribution(calibration_df)
    bundle = attach_pass_probability(bundle, threshold=config.synthetic.pass_threshold)
    labels = calibration_df["pass_label"].to_numpy(dtype=int)
    if bundle.pass_probability_raw is None or len(np.unique(labels)) <= 1:
        return None
    unique_prob = np.unique(np.round(bundle.pass_probability_raw, decimals=8))
    if unique_prob.size <= 1:
        return None
    if calibrator_kind == "platt":
        return fit_pass_probability_platt(raw_probability=bundle.pass_probability_raw, labels=labels)
    return fit_pass_probability_calibrator(raw_probability=bundle.pass_probability_raw, labels=labels)


def _load_or_generate_dataset(
    config: BenchmarkRunConfig,
    dataset: pd.DataFrame | str | Path | None,
) -> pd.DataFrame:
    if dataset is None:
        LOGGER.info("Generating synthetic DriftQCap benchmark dataset.")
        raw_df = generate_synthetic_benchmark(config.synthetic)
    elif isinstance(dataset, (str, Path)):
        LOGGER.info("Loading dataset from %s", dataset)
        raw_df = pd.read_csv(dataset)
    elif isinstance(dataset, pd.DataFrame):
        LOGGER.info("Using caller-supplied dataframe.")
        raw_df = dataset.copy()
    else:
        raise TypeError("dataset must be None, a path-like object, or a pandas DataFrame.")
    return prepare_dataframe(
        raw_df,
        pass_threshold=config.synthetic.pass_threshold,
        force_recompute_pass_label=True,
    )


def _fit_source_model(
    source_train_df: pd.DataFrame,
    *,
    config: BenchmarkRunConfig,
    feature_mode: Literal["full", "coarse"],
    random_seed: int,
) -> BaseCapabilityRegressor:
    model = make_base_model(
        model_config=config.model,
        feature_mode=feature_mode,
        random_seed=random_seed,
    )
    model.fit(source_train_df)
    return model


def _fit_calibration_artifacts(
    model: BaseCapabilityRegressor,
    source_calibration_df: pd.DataFrame,
    *,
    target_calibration_df: pd.DataFrame | None,
    config: BenchmarkRunConfig,
    strategy_name: str | None = None,
    probability_override: str | None = None,
) -> CalibrationArtifacts:
    is_adapted = strategy_name in _ADAPTED_STRATEGIES
    conformal_mode = "source_only"
    conformal_rows = int(len(source_calibration_df))
    if (
        is_adapted
        and target_calibration_df is not None
        and not target_calibration_df.empty
    ):
        source_conformal = _fit_conformal_from_df(model, source_calibration_df, config=config)
        target_conformal = _fit_conformal_from_df(model, target_calibration_df, config=config)
        target_rows = int(len(target_calibration_df))
        adapted_mode = config.calibration.adapted_conformal_mode
        if adapted_mode == "target_only":
            conformal = target_conformal
            conformal_mode = "target_only"
            conformal_rows = target_rows
        elif (
            adapted_mode == "source_plus_target_blend"
            and target_rows >= config.calibration.adapted_conformal_target_min_rows
        ):
            target_weight = target_rows / (target_rows + max(config.calibration.adapted_conformal_blend_tau, 1e-6))
            conformal = blend_conformal_artifacts(
                source=source_conformal,
                target=target_conformal,
                target_weight=target_weight,
            )
            conformal_mode = "source_plus_target_blend"
            conformal_rows = target_rows
        else:
            conformal = source_conformal
            conformal_mode = "source_only" if adapted_mode == "source_only" else "fallback_source"
        source_rows = int(len(source_calibration_df))
        conformal_rows = source_rows + target_rows if conformal_mode == "source_plus_target_blend" else conformal_rows
    else:
        conformal_df = source_calibration_df
        if (
            config.calibration.target_recalibration
            and target_calibration_df is not None
            and not target_calibration_df.empty
        ):
            if config.calibration.conformal_recalibration_mode == "target_only":
                conformal_df = target_calibration_df
                conformal_mode = "target_only"
            elif config.calibration.conformal_recalibration_mode == "source_plus_target":
                conformal_df = pd.concat([source_calibration_df, target_calibration_df], ignore_index=True)
                conformal_mode = "source_plus_target"
        if (
            config.calibration.target_recalibration
            and conformal_mode != "source_only"
            and len(conformal_df) < config.calibration.min_target_calibration_rows
        ):
            conformal_df = source_calibration_df
            conformal_mode = "fallback_source"
        conformal = _fit_conformal_from_df(model, conformal_df, config=config)
        conformal_rows = int(len(conformal_df))

    calibration_df = source_calibration_df
    mode = "source_only"
    if (
        config.calibration.target_recalibration
        and target_calibration_df is not None
        and not target_calibration_df.empty
    ):
        if config.calibration.target_recalibration_mode == "target_only":
            calibration_df = target_calibration_df
            mode = "target_only"
        else:
            calibration_df = pd.concat([source_calibration_df, target_calibration_df], ignore_index=True)
            mode = "source_plus_target"

    rows = int(len(calibration_df))
    labels = calibration_df["pass_label"].to_numpy(dtype=int)
    positives = int(labels.sum()) if rows else 0
    negatives = int(rows - positives) if rows else 0
    if config.calibration.target_recalibration and mode != "source_only":
        if (
            rows < config.calibration.min_target_calibration_rows
            or positives < config.calibration.min_target_positive
            or negatives < config.calibration.min_target_negative
        ):
            calibration_df = source_calibration_df
            mode = "fallback_source"
            rows = int(len(calibration_df))
            labels = calibration_df["pass_label"].to_numpy(dtype=int)
            positives = int(labels.sum()) if rows else 0
            negatives = int(rows - positives) if rows else 0

    calibrator = None
    calibrator_type = "none"
    calibration_weight = float(rows / (rows + max(config.calibration.calibration_shrink_tau, 1e-6))) if rows > 0 else 0.0
    if is_adapted and config.calibration.adapted_disable_prob_calibration and probability_override is None:
        mode = "adapted_disabled"
        calibration_weight = 0.0
        rows = 0
        positives = 0
        negatives = 0
    else:
        chosen_kind = probability_override or config.calibration.prob_calibrator
        try:
            calibrator = _fit_prob_calibrator_from_df(
                model,
                calibration_df,
                config=config,
                calibrator_kind=chosen_kind,
            )
            if calibrator is not None:
                calibrator_type = chosen_kind
            else:
                calibrator_type = "none"
        except ValueError:
            calibrator = None
            calibrator_type = "none"
        if probability_override == "none":
            calibrator = None
            calibrator_type = "none"
            calibration_weight = 0.0
            mode = "override_none"
    return CalibrationArtifacts(
        conformal=conformal,
        pass_probability_calibrator=calibrator,
        probability_calibration_mode=mode,
        probability_calibration_rows=rows,
        probability_calibration_pos=positives,
        probability_calibration_neg=negatives,
        probability_calibrator_type=calibrator_type,
        probability_calibration_weight=calibration_weight,
        conformal_calibration_mode=conformal_mode,
        conformal_calibration_rows=conformal_rows,
    )


def _predict_with_calibration(
    model: BaseCapabilityRegressor,
    eval_df: pd.DataFrame,
    *,
    calibration_artifacts: CalibrationArtifacts,
    config: BenchmarkRunConfig,
):
    bundle = model.predict_distribution(eval_df)
    bundle = attach_intervals(bundle, conformal=calibration_artifacts.conformal)
    bundle = attach_pass_probability(bundle, threshold=config.synthetic.pass_threshold)
    pass_probability_calibrated = calibrate_pass_probability(
        raw_probability=bundle.pass_probability_raw,
        calibrator=calibration_artifacts.pass_probability_calibrator,
    )
    w = float(np.clip(calibration_artifacts.probability_calibration_weight, 0.0, 1.0))
    bundle.pass_probability_calibrated = w * pass_probability_calibrated + (1.0 - w) * bundle.pass_probability_raw
    return bundle


def _evaluate_model_run(
    *,
    model: BaseCapabilityRegressor,
    source_calibration_df: pd.DataFrame,
    target_calibration_df: pd.DataFrame | None,
    eval_df: pd.DataFrame,
    config: BenchmarkRunConfig,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy_name = str(metadata.get("strategy", "")) if metadata else None
    calibration_artifacts = _fit_calibration_artifacts(
        model,
        source_calibration_df,
        target_calibration_df=target_calibration_df,
        config=config,
        strategy_name=strategy_name,
    )
    bundle = _predict_with_calibration(model, eval_df, calibration_artifacts=calibration_artifacts, config=config)
    run_metadata = dict(metadata)
    run_metadata.update(
        {
            "prob_calib_mode": calibration_artifacts.probability_calibration_mode,
            "prob_calib_rows": calibration_artifacts.probability_calibration_rows,
            "prob_calib_pos": calibration_artifacts.probability_calibration_pos,
            "prob_calib_neg": calibration_artifacts.probability_calibration_neg,
            "prob_calib_model": calibration_artifacts.probability_calibrator_type,
            "prob_calib_weight": calibration_artifacts.probability_calibration_weight,
            "conformal_calib_mode": calibration_artifacts.conformal_calibration_mode,
            "conformal_calib_rows": calibration_artifacts.conformal_calibration_rows,
        }
    )
    diagnostics = model.diagnostics()
    run_metadata.update(diagnostics)
    if "model_family" not in run_metadata:
        run_metadata["model_family"] = str(getattr(model, "model_family", diagnostics.get("model_family", "unknown")))
    row = combine_metrics(
        y_true=eval_df["error_rate"].to_numpy(dtype=float),
        bundle=bundle,
        threshold=config.synthetic.pass_threshold,
        n_bins=config.calibration.ece_bins,
        metadata=run_metadata,
    )
    prediction_df = build_prediction_frame(
        eval_df=eval_df,
        bundle=bundle,
        run_metadata=run_metadata,
        threshold=config.synthetic.pass_threshold,
    )
    return row, prediction_df


def _evaluate_probability_calibration_diagnostic(
    *,
    model: BaseCapabilityRegressor,
    source_calibration_df: pd.DataFrame,
    target_calibration_df: pd.DataFrame | None,
    eval_df: pd.DataFrame,
    config: BenchmarkRunConfig,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not config.calibration.adapted_prob_calibration_diagnostic:
        return []
    variants = ["none", "isotonic"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        calibration_artifacts = _fit_calibration_artifacts(
            model,
            source_calibration_df,
            target_calibration_df=target_calibration_df,
            config=config,
            strategy_name=str(metadata.get("strategy", "")),
            probability_override=variant,
        )
        bundle = _predict_with_calibration(model, eval_df, calibration_artifacts=calibration_artifacts, config=config)
        diagnostic = combine_metrics(
            y_true=eval_df["error_rate"].to_numpy(dtype=float),
            bundle=bundle,
            threshold=config.synthetic.pass_threshold,
            n_bins=config.calibration.ece_bins,
            metadata={
                "strategy": metadata.get("strategy"),
                "budget": metadata.get("budget"),
                "repeat_id": metadata.get("repeat_id"),
                "shift_type": metadata.get("shift_type"),
                "episode_id": metadata.get("episode_id"),
                "calibration_variant": variant,
            },
        )
        rows.append(diagnostic)
    return rows


def _summarise_results(df: pd.DataFrame, *, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=by)
    grouped = df.groupby(by, dropna=False)
    pieces = []
    for metric in _METRIC_COLUMNS:
        if metric not in df.columns:
            continue
        agg = grouped[metric].agg(["mean", "std"]).reset_index()
        agg = agg.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        pieces.append(agg)
    if not pieces:
        return grouped.size().reset_index(name="n_rows")
    summary = pieces[0]
    for piece in pieces[1:]:
        summary = summary.merge(piece, on=by, how="outer")
    counts = grouped.size().reset_index(name="n_runs")
    summary = summary.merge(counts, on=by, how="left")
    return summary.sort_values(by).reset_index(drop=True)


def _append_overall_rows(summary_df: pd.DataFrame, *, by: list[str]) -> pd.DataFrame:
    if summary_df.empty or "shift_type" not in by:
        return summary_df
    without_shift = [col for col in by if col != "shift_type"]
    overall = _summarise_results(
        summary_df.rename(columns={c: c.replace("_mean", "").replace("_std", "") for c in summary_df.columns}),
        by=without_shift,
    )
    # The generic route above is fragile for pre-aggregated inputs; handle by direct recomputation upstream.
    return summary_df


def _summarise_with_overall(df: pd.DataFrame, *, by: list[str]) -> pd.DataFrame:
    full = _summarise_results(df, by=by)
    if "shift_type" not in by:
        return full
    overall_by = [col for col in by if col != "shift_type"]
    overall = _summarise_results(df.assign(shift_type="overall"), by=overall_by + ["shift_type"])
    combined = pd.concat([full, overall], ignore_index=True)
    sort_cols = [col for col in by if col in combined.columns]
    return combined.sort_values(sort_cols).reset_index(drop=True)


def _average_over_repeats(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    numeric = [col for col in _METRIC_COLUMNS if col in df.columns]
    return (
        df.groupby(group_cols, dropna=False)[numeric]
        .mean()
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _build_auc_table(
    df: pd.DataFrame,
    *,
    strategy_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[strategy_col, "mae_auc", "ece_calibrated_auc"])
    mae_auc = episode_level_auc(df, group_cols=["episode_id", "shift_type"], strategy_col=strategy_col, metric_col="mae")
    ece_auc = episode_level_auc(df, group_cols=["episode_id", "shift_type"], strategy_col=strategy_col, metric_col="ece_calibrated")
    auc = mae_auc.merge(
        ece_auc,
        on=["episode_id", "shift_type", strategy_col],
        how="outer",
    )
    overall = auc.assign(shift_type="overall")
    return pd.concat([auc, overall], ignore_index=True)


def _build_comparison_table(
    df: pd.DataFrame,
    *,
    strategy_col: str,
    comparisons: list[tuple[str, str]],
    shift_types: list[str],
    evaluation_config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["mae", "ece_calibrated"]
    for shift_type in ["overall", *shift_types]:
        subset = df if shift_type == "overall" else df[df["shift_type"] == shift_type].copy()
        if subset.empty:
            continue
        for metric in metrics:
            comparison_objects = []
            for strategy_a, strategy_b in comparisons:
                comparison = paired_strategy_comparison(
                    subset,
                    strategy_col=strategy_col,
                    metric_col=metric,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    group_cols=["episode_id"],
                    bootstrap_resamples=evaluation_config.bootstrap_resamples,
                    random_state=20260308,
                )
                comparison_objects.append(comparison)
            frame = comparisons_to_frame(comparison_objects)
            if frame.empty:
                continue
            frame = frame.dropna(axis=1, how="all")
            if frame.empty:
                continue
            frame.insert(0, "metric", f"{metric}_auc")
            frame.insert(0, "shift_scope", shift_type)
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _build_budget_comparison_table(
    df: pd.DataFrame,
    *,
    strategy_col: str,
    comparisons: list[tuple[str, str]],
    shift_types: list[str],
    evaluation_config,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metrics = ["mae", "ece_calibrated"]
    if df.empty or "budget" not in df.columns:
        return pd.DataFrame()
    budgets = sorted({int(v) for v in df["budget"].dropna().tolist() if int(v) > 0})
    for shift_type in ["overall", *shift_types]:
        shift_df = df if shift_type == "overall" else df[df["shift_type"] == shift_type].copy()
        if shift_df.empty:
            continue
        for budget in budgets:
            budget_df = shift_df[shift_df["budget"] == budget].copy()
            if budget_df.empty:
                continue
            for metric in metrics:
                comparison_objects = []
                for strategy_a, strategy_b in comparisons:
                    comparison_objects.append(
                        paired_strategy_comparison(
                            budget_df,
                            strategy_col=strategy_col,
                            metric_col=metric,
                            strategy_a=strategy_a,
                            strategy_b=strategy_b,
                            group_cols=["episode_id"],
                            bootstrap_resamples=evaluation_config.bootstrap_resamples,
                            random_state=20260308 + budget,
                        )
                    )
                frame = comparisons_to_frame(comparison_objects)
                if frame.empty:
                    continue
                frame.insert(0, "budget", int(budget))
                frame.insert(0, "metric", f"{metric}_at_budget")
                frame.insert(0, "shift_scope", shift_type)
                rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _fit_adaptation_models(
    *,
    labeled_df: pd.DataFrame,
    source_train_df: pd.DataFrame,
    base_model: BaseCapabilityRegressor,
    config: BenchmarkRunConfig,
    random_seed: int,
) -> dict[str, BaseCapabilityRegressor]:
    models: dict[str, BaseCapabilityRegressor] = {}
    models["target_only"] = fit_target_only_model(
        labeled_df,
        model_config=config.model,
        random_seed=random_seed + 11,
        base_estimator=config.model.base_estimator,
    )
    models["pooled_retrain"] = fit_pooled_retrain_model(
        source_train_df,
        labeled_df,
        model_config=config.model,
        target_upweight=config.adaptation.target_upweight,
        random_seed=random_seed + 23,
        base_estimator=config.model.base_estimator,
    )
    models["residual_adapter"] = FewShotResidualAdapter(
        base_model,
        ridge_alpha=config.adaptation.ridge_alpha,
        residual_feature_mode=config.adaptation.residual_feature_mode,
        use_base_mean=config.adaptation.residual_use_base_mean,
        use_base_std=config.adaptation.residual_use_base_std,
        max_feature_dims=config.adaptation.residual_max_feature_dims,
        alpha_grid=config.adaptation.residual_alpha_grid,
        cv_folds=config.adaptation.residual_cv_folds,
        selection_metric=config.adaptation.residual_selection_metric,
        random_seed=random_seed + 31,
        std_temperature_quantile=config.adaptation.std_temperature_quantile,
        std_temperature_min=config.adaptation.std_temperature_min,
        std_temperature_max=config.adaptation.std_temperature_max,
        std_floor=config.adaptation.std_floor,
    ).fit(labeled_df)
    models["era_adapter"] = ElasticResidualAdapter(
        base_model,
        ridge_alpha=config.adaptation.era_ridge_alpha,
        ewc_lambda=config.adaptation.ewc_lambda,
        feature_mode=config.adaptation.era_feature_mode,
        use_base_mean=config.adaptation.era_use_base_mean,
        use_base_std=config.adaptation.era_use_base_std,
        max_feature_dims=config.adaptation.era_max_feature_dims,
        alpha_grid=config.adaptation.era_alpha_grid,
        ewc_lambda_grid=config.adaptation.era_ewc_lambda_grid,
        cv_folds=config.adaptation.era_cv_folds,
        selection_metric=config.adaptation.era_selection_metric,
        random_seed=random_seed + 32,
        std_temperature_quantile=config.adaptation.std_temperature_quantile,
        std_temperature_min=config.adaptation.std_temperature_min,
        std_temperature_max=config.adaptation.std_temperature_max,
        std_floor=config.adaptation.std_floor,
    ).fit(labeled_df)
    models["mean_shift_adapter"] = MeanShiftAdapter(
        base_model,
        random_seed=random_seed + 33,
        std_temperature_quantile=config.adaptation.std_temperature_quantile,
        std_temperature_min=config.adaptation.std_temperature_min,
        std_temperature_max=config.adaptation.std_temperature_max,
        std_floor=config.adaptation.std_floor,
    ).fit(labeled_df)
    models["ewc_adapter"] = EWCResidualAdapter(
        base_model,
        ridge_alpha=config.adaptation.ridge_alpha,
        ewc_lambda=config.adaptation.ewc_lambda,
        random_seed=random_seed + 35,
        std_temperature_quantile=config.adaptation.std_temperature_quantile,
        std_temperature_min=config.adaptation.std_temperature_min,
        std_temperature_max=config.adaptation.std_temperature_max,
        std_floor=config.adaptation.std_floor,
    ).fit(labeled_df)
    models["shuffled_label_adapter"] = FewShotResidualAdapter(
        base_model,
        ridge_alpha=config.adaptation.ridge_alpha,
        residual_feature_mode=config.adaptation.residual_feature_mode,
        use_base_mean=config.adaptation.residual_use_base_mean,
        use_base_std=config.adaptation.residual_use_base_std,
        max_feature_dims=config.adaptation.residual_max_feature_dims,
        alpha_grid=config.adaptation.residual_alpha_grid,
        cv_folds=config.adaptation.residual_cv_folds,
        selection_metric=config.adaptation.residual_selection_metric,
        random_seed=random_seed + 37,
        std_temperature_quantile=config.adaptation.std_temperature_quantile,
        std_temperature_min=config.adaptation.std_temperature_min,
        std_temperature_max=config.adaptation.std_temperature_max,
        std_floor=config.adaptation.std_floor,
    ).fit(labeled_df, shuffle_labels=True)
    return models


def _select_labeled_subset(
    *,
    acquisition_strategy: str,
    candidate_df: pd.DataFrame,
    base_model: BaseCapabilityRegressor,
    budget: int,
    config: BenchmarkRunConfig,
    random_seed: int,
    shift_type: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strategy_metadata: dict[str, Any] = {}

    def _normalize_weights(weights: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        total = float(sum(max(w, 0.0) for w in weights))
        if total <= 0.0:
            return (0.45, 0.20, 0.20, 0.15)
        return tuple(float(max(w, 0.0) / total) for w in weights)

    def _entropy_weights() -> tuple[float, float, float, float]:
        configured = _normalize_weights(
            (
                config.adaptation.acq_entropy_weight,
                config.adaptation.acq_std_weight,
                config.adaptation.acq_threshold_weight,
                config.adaptation.acq_diversity_weight,
            )
        )
        if not config.adaptation.acq_weight_grid_enabled or not config.adaptation.acq_weight_grid:
            strategy_metadata["acq_weight_source"] = "configured"
            return configured
        grid = [_normalize_weights(weights) for weights in config.adaptation.acq_weight_grid]
        chosen = min(
            grid,
            key=lambda w: (w[0] - configured[0]) ** 2 + (w[1] - configured[1]) ** 2 + (w[2] - configured[2]) ** 2 + (w[3] - configured[3]) ** 2,
        )
        strategy_metadata["acq_weight_source"] = "grid_closest"
        return chosen
    if acquisition_strategy == "hybrid_shift_aware":
        strategy_lookup = {
            "depth_shift": config.adaptation.hybrid_depth_strategy,
            "family_shift": config.adaptation.hybrid_family_strategy,
            "noise_shift": config.adaptation.hybrid_noise_strategy,
            "temporal_shift": config.adaptation.hybrid_temporal_strategy,
        }
        mapped = strategy_lookup.get(str(shift_type), "diversity")
        acquisition_strategy = mapped

    if acquisition_strategy == "random":
        selected_idx = select_random(candidate_df, budget=budget, random_state=random_seed)
    elif acquisition_strategy == "shift_aware_diversity":
        selected_idx = select_shift_aware_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            shift_weight=config.adaptation.shift_diversity_weight,
            coverage_weight=config.adaptation.shift_diversity_coverage_weight,
        )
    elif acquisition_strategy == "stratified_diversity":
        selected_idx = select_stratified_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            strata_bins=config.adaptation.stratified_diversity_bins,
            family_stratify=config.adaptation.stratified_diversity_family_stratify,
        )
    elif acquisition_strategy == "diversity":
        selected_idx = select_diversity(candidate_df, model=base_model, budget=budget, random_state=random_seed)
    elif acquisition_strategy == "uncertainty":
        selected_idx = select_uncertainty(
            candidate_df,
            model=base_model,
            budget=budget,
            quantile_min=0.0,
            quantile_max=1.0,
        )
    elif acquisition_strategy == "uncertainty_trimmed":
        selected_idx = select_uncertainty(
            candidate_df,
            model=base_model,
            budget=budget,
            quantile_min=config.adaptation.uncertainty_quantile_min,
            quantile_max=config.adaptation.uncertainty_quantile_max,
        )
    elif acquisition_strategy == "badge_proxy":
        selected_idx = select_badge_proxy(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            quantile_min=config.adaptation.acq_quantile_min,
            quantile_max=config.adaptation.acq_quantile_max,
        )
    elif acquisition_strategy == "density_ratio_uncertainty_diversity":
        selected_idx = select_density_ratio_uncertainty_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            ratio_weight=config.adaptation.density_ratio_weight,
            diversity_weight=config.adaptation.density_ratio_diversity_weight,
            quantile_min=config.adaptation.acq_quantile_min,
            quantile_max=config.adaptation.acq_quantile_max,
        )
        strategy_metadata.update(
            {
                "density_ratio_weight": config.adaptation.density_ratio_weight,
                "density_ratio_diversity_weight": config.adaptation.density_ratio_diversity_weight,
            }
        )
    elif acquisition_strategy == "stratified_predmean_diversity":
        selected_idx = select_stratified_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            strata_bins=config.adaptation.predmean_stratified_bins,
            family_stratify=False,
        )
        strategy_metadata.update({"predmean_stratified_bins": config.adaptation.predmean_stratified_bins})
    elif acquisition_strategy == "two_stage_explore_exploit":
        selected_idx = select_two_stage_explore_exploit(
            candidate_df,
            model=base_model,
            budget=budget,
            random_state=random_seed,
            explore_fraction=config.adaptation.two_stage_explore_fraction,
            quantile_min=config.adaptation.acq_quantile_min,
            quantile_max=config.adaptation.acq_quantile_max,
        )
        strategy_metadata.update({"two_stage_explore_fraction": config.adaptation.two_stage_explore_fraction})
    elif acquisition_strategy == "uncertainty_threshold_diversity":
        selected_idx = select_uncertainty_threshold_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            threshold=config.synthetic.pass_threshold,
            uncertainty_weight=config.adaptation.uncertainty_weight,
            threshold_weight=config.adaptation.threshold_weight,
            diversity_weight=config.adaptation.diversity_weight,
            threshold_beta=config.adaptation.threshold_beta,
            random_state=random_seed,
            quantile_min=0.0,
            quantile_max=1.0,
        )
    elif acquisition_strategy == "entropy_threshold_diversity":
        entropy_weight, std_weight, threshold_weight, diversity_weight = _entropy_weights()
        selected_idx = select_entropy_threshold_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            threshold=config.synthetic.pass_threshold,
            entropy_weight=entropy_weight,
            std_weight=std_weight,
            threshold_weight=threshold_weight,
            diversity_weight=diversity_weight,
            threshold_beta=config.adaptation.threshold_beta,
            random_state=random_seed,
            quantile_min=config.adaptation.acq_quantile_min,
            quantile_max=config.adaptation.acq_quantile_max,
        )
        strategy_metadata.update(
            {
                "acq_entropy_weight": entropy_weight,
                "acq_std_weight": std_weight,
                "acq_threshold_weight": threshold_weight,
                "acq_diversity_weight": diversity_weight,
                "acq_score_mode": config.adaptation.acq_score_mode,
            }
        )
    elif acquisition_strategy == "uncertainty_diversity_trimmed":
        selected_idx = select_uncertainty_threshold_diversity(
            candidate_df,
            model=base_model,
            budget=budget,
            threshold=config.synthetic.pass_threshold,
            uncertainty_weight=config.adaptation.uncertainty_weight,
            threshold_weight=config.adaptation.threshold_weight,
            diversity_weight=config.adaptation.diversity_weight,
            threshold_beta=config.adaptation.threshold_beta,
            random_state=random_seed,
            quantile_min=config.adaptation.uncertainty_quantile_min,
            quantile_max=config.adaptation.uncertainty_quantile_max,
        )
    elif acquisition_strategy == "uncertainty_threshold_diversity_capped":
        selected_idx = select_uncertainty_threshold_diversity_capped(
            candidate_df,
            model=base_model,
            budget=budget,
            threshold=config.synthetic.pass_threshold,
            uncertainty_weight=config.adaptation.uncertainty_weight,
            threshold_weight=config.adaptation.threshold_weight,
            diversity_weight=config.adaptation.diversity_weight,
            density_weight=config.adaptation.density_weight,
            threshold_beta=config.adaptation.threshold_beta,
            uncertainty_cap_quantile=config.adaptation.uncertainty_cap_quantile,
            random_state=random_seed,
            quantile_min=config.adaptation.uncertainty_quantile_min,
            quantile_max=config.adaptation.uncertainty_quantile_max,
        )
    elif acquisition_strategy == "oracle_residual":
        selected_idx = select_oracle_residual(
            candidate_df,
            model=base_model,
            budget=budget,
        )
    else:
        raise ValueError(f"Unknown acquisition strategy: {acquisition_strategy}")
    return candidate_df.iloc[selected_idx].reset_index(drop=True), strategy_metadata


def _select_oracle_true_search_subset(
    *,
    candidate_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    base_model: BaseCapabilityRegressor,
    source_calibration_df: pd.DataFrame,
    budget: int,
    config: BenchmarkRunConfig,
    random_seed: int,
    n_trials: int,
) -> pd.DataFrame:
    """Diagnostic leaky oracle: search random subsets and pick one with best eval MAE."""
    budget = min(int(budget), len(candidate_df))
    if budget <= 0:
        return candidate_df.iloc[[]].copy()
    n_trials = max(int(n_trials), 1)
    rng = np.random.default_rng(random_seed)
    best_idx: np.ndarray | None = None
    best_mae = float("inf")

    for trial_id in range(n_trials):
        trial_idx = np.sort(rng.choice(len(candidate_df), size=budget, replace=False)).astype(int)
        trial_df = candidate_df.iloc[trial_idx].reset_index(drop=True)
        trial_model = FewShotResidualAdapter(
            base_model,
            ridge_alpha=config.adaptation.ridge_alpha,
            residual_feature_mode=config.adaptation.residual_feature_mode,
            use_base_mean=config.adaptation.residual_use_base_mean,
            use_base_std=config.adaptation.residual_use_base_std,
            max_feature_dims=config.adaptation.residual_max_feature_dims,
            alpha_grid=config.adaptation.residual_alpha_grid,
            cv_folds=config.adaptation.residual_cv_folds,
            selection_metric=config.adaptation.residual_selection_metric,
            random_seed=random_seed + 10_000 + trial_id,
            std_temperature_quantile=config.adaptation.std_temperature_quantile,
            std_temperature_min=config.adaptation.std_temperature_min,
            std_temperature_max=config.adaptation.std_temperature_max,
            std_floor=config.adaptation.std_floor,
        ).fit(trial_df)
        row, _ = _evaluate_model_run(
            model=trial_model,
            source_calibration_df=source_calibration_df,
            target_calibration_df=trial_df,
            eval_df=eval_df,
            config=config,
            metadata={
                "panel": "oracle_search",
                "strategy": "residual_adapter",
                "acquisition_strategy": "oracle_true_search",
                "budget": budget,
                "repeat_id": trial_id,
                "shift_type": str(eval_df["shift_type"].iloc[0]),
                "episode_id": str(eval_df["episode_id"].iloc[0]),
            },
        )
        mae = float(row["mae"])
        if mae < best_mae:
            best_mae = mae
            best_idx = trial_idx

    if best_idx is None:
        raise RuntimeError("Oracle subset search failed to select a subset.")
    return candidate_df.iloc[best_idx].reset_index(drop=True)


def _generate_figures(
    *,
    output_dir: Path,
    dataset: pd.DataFrame,
    adaptation_summary: pd.DataFrame,
    acquisition_summary: pd.DataFrame,
    adaptation_auc: pd.DataFrame,
    acquisition_auc: pd.DataFrame,
    raw_predictions: pd.DataFrame,
) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, str] = {}

    path_map = [
        ("fig01_dataset_error_distribution", plot_dataset_error_distribution(dataset)),
        ("fig02_zero_shot_shift_mae", plot_zero_shot_shift_mae(adaptation_summary)),
        ("fig03_adaptation_overall_mae", plot_adaptation_curves(adaptation_summary, shift_type="overall", metric="mae")),
        ("fig04_acquisition_overall_mae", plot_acquisition_curves(acquisition_summary, shift_type="overall", metric="mae")),
    ]
    for key, fig in path_map:
        saved = save_figure(fig, figure_dir / key)
        figure_paths[key] = saved["png"]

    for shift_type in ["family_shift", "depth_shift", "noise_shift", "temporal_shift"]:
        if shift_type in set(adaptation_summary["shift_type"]):
            saved = save_figure(
                plot_adaptation_curves(adaptation_summary, shift_type=shift_type, metric="mae"),
                figure_dir / f"adaptation_{shift_type}_mae",
            )
            figure_paths[f"adaptation_{shift_type}_mae"] = saved["png"]
        if shift_type in set(acquisition_summary["shift_type"]):
            saved = save_figure(
                plot_acquisition_curves(acquisition_summary, shift_type=shift_type, metric="mae"),
                figure_dir / f"acquisition_{shift_type}_mae",
            )
            figure_paths[f"acquisition_{shift_type}_mae"] = saved["png"]

    if not raw_predictions.empty:
        source_pred = raw_predictions[(raw_predictions["panel"] == "adaptation") & (raw_predictions["strategy"] == "source_only")]
        if not source_pred.empty:
            saved = save_figure(
                plot_reliability_diagram_from_predictions(source_pred, title="Reliability under zero-shot transfer"),
                figure_dir / "fig05_reliability_source_only",
            )
            figure_paths["fig05_reliability_source_only"] = saved["png"]
            saved = save_figure(
                plot_reliability_diagram_equal_mass(
                    source_pred,
                    title="Reliability under zero-shot transfer (equal-mass bins)",
                ),
                figure_dir / "fig05b_reliability_source_only_equal_mass",
            )
            figure_paths["fig05b_reliability_source_only_equal_mass"] = saved["png"]

        max_budget = raw_predictions.loc[raw_predictions["panel"] == "adaptation", "budget"].max() if (raw_predictions["panel"] == "adaptation").any() else np.nan
        if not np.isnan(max_budget):
            adapter_pred = raw_predictions[
                (raw_predictions["panel"] == "adaptation")
                & (raw_predictions["strategy"] == "era_adapter")
                & (raw_predictions["budget"] == max_budget)
            ]
            if not adapter_pred.empty:
                saved = save_figure(
                    plot_reliability_diagram_from_predictions(
                        adapter_pred,
                        title=f"Reliability after elastic residual adaptation (budget={int(max_budget)})",
                    ),
                    figure_dir / "fig06_reliability_era_adapter",
                )
                figure_paths["fig06_reliability_era_adapter"] = saved["png"]
                saved = save_figure(
                    plot_reliability_diagram_equal_mass(
                        adapter_pred,
                        title=f"Reliability after elastic residual adaptation (budget={int(max_budget)}, equal-mass bins)",
                    ),
                    figure_dir / "fig06b_reliability_era_adapter_equal_mass",
                )
                figure_paths["fig06b_reliability_era_adapter_equal_mass"] = saved["png"]

    if not adaptation_summary.empty and "era_adapter" in set(adaptation_summary["strategy"]):
        saved = save_figure(
            plot_interval_coverage_curves(
                adaptation_summary,
                strategy_col="strategy",
                strategy_name="era_adapter",
                shift_type="overall",
                title="Empirical coverage of conformal intervals after elastic residual adaptation",
            ),
            figure_dir / "fig07_interval_coverage_era_adapter",
        )
        figure_paths["fig07_interval_coverage_era_adapter"] = saved["png"]

    adaptation_rank = rank_strategies(
        adaptation_auc[adaptation_auc["shift_type"] == "overall"],
        strategy_col="strategy",
        value_col="mae_auc",
    )
    acquisition_rank = rank_strategies(
        acquisition_auc[acquisition_auc["shift_type"] == "overall"],
        strategy_col="acquisition_strategy",
        value_col="mae_auc",
    )
    if not adaptation_rank.empty:
        saved = save_figure(
            plot_auc_ranking(adaptation_rank, strategy_col="strategy", value_col="mae_auc", title="Episode-level adaptation ranking by MAE AUC"),
            figure_dir / "fig08_adaptation_auc_ranking",
        )
        figure_paths["fig08_adaptation_auc_ranking"] = saved["png"]
    if not acquisition_rank.empty:
        saved = save_figure(
            plot_auc_ranking(
                acquisition_rank,
                strategy_col="acquisition_strategy",
                value_col="mae_auc",
                title="Episode-level acquisition ranking by MAE AUC",
            ),
            figure_dir / "fig09_acquisition_auc_ranking",
        )
        figure_paths["fig09_acquisition_auc_ranking"] = saved["png"]

    return figure_paths


def run_benchmark(
    config: BenchmarkRunConfig,
    *,
    dataset: pd.DataFrame | str | Path | None = None,
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
) -> BenchmarkArtifacts:
    """Run the full DriftQCap synthetic benchmark.

    If `dataset` is omitted, the synthetic benchmark defined by the profile is
    generated automatically. External CSV datasets are supported if they follow
    the schema described in the README.
    """
    dataset_df = _load_or_generate_dataset(config, dataset)
    source_splits = make_source_splits(dataset_df, random_state=config.synthetic.random_seed)

    artifact_paths: dict[str, str] = {}
    base_model = None
    coarse_model = None
    if output_dir is not None and config.reporting.load_model_artifacts:
        base_model, coarse_model, artifact_paths = load_source_artifacts(
            output_dir=output_dir,
            config=config,
            source_train_df=source_splits.train,
        )
        if base_model is not None and coarse_model is not None:
            LOGGER.info("Loaded persisted source models from %s", artifact_paths["artifact_dir"])

    if base_model is None or coarse_model is None:
        LOGGER.info("Fitting source models.")
        base_model = _fit_source_model(source_splits.train, config=config, feature_mode="full", random_seed=config.model.random_seed)
        coarse_model = _fit_source_model(source_splits.train, config=config, feature_mode="coarse", random_seed=config.model.random_seed + 101)
        if output_dir is not None and config.reporting.save_model_artifacts:
            artifact_paths = save_source_artifacts(
                output_dir=output_dir,
                config=config,
                source_train_df=source_splits.train,
                base_model=base_model,
                coarse_model=coarse_model,
            )
            LOGGER.info("Saved source models to %s", artifact_paths["artifact_dir"])

    acquisition_base_model: BaseCapabilityRegressor = base_model
    if config.model.base_estimator != "rf":
        LOGGER.info("Acquisition panel pinned to RF; fitting RF source model for acquisition strategies.")
        rf_model_config = replace(config.model, base_estimator="rf", random_seed=config.model.random_seed + 707)
        acquisition_base_model = make_base_model(
            model_config=rf_model_config,
            feature_mode="full",
            random_seed=rf_model_config.random_seed,
        ).fit(source_splits.train)

    source_results_rows: list[dict[str, Any]] = []
    raw_prediction_frames: list[pd.DataFrame] = []

    for strategy_name, model in [("source_only", base_model), ("coarse_source_only", coarse_model)]:
        metadata = {
            "panel": "source",
            "strategy": strategy_name,
            "budget": 0,
            "repeat_id": 0,
            "shift_type": "source",
            "episode_id": "source-test",
        }
        row, pred = _evaluate_model_run(
            model=model,
            source_calibration_df=source_splits.calibration,
            target_calibration_df=None,
            eval_df=source_splits.test,
            config=config,
            metadata=metadata,
        )
        source_results_rows.append(row)
        raw_prediction_frames.append(pred)

    target_df = dataset_df[dataset_df["domain"] == "target"].copy()
    adaptation_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    acquisition_weight_rows: list[dict[str, Any]] = []
    adapted_probability_rows: list[dict[str, Any]] = []

    positive_adaptation_budgets = sorted({int(budget) for budget in config.adaptation.budgets if int(budget) > 0})
    if config.adaptation.acquisition_budgets is None:
        positive_acquisition_budgets = list(positive_adaptation_budgets)
    else:
        positive_acquisition_budgets = sorted({int(budget) for budget in config.adaptation.acquisition_budgets if int(budget) > 0})
    positive_budgets = sorted(set(positive_adaptation_budgets).union(positive_acquisition_budgets))
    target_shift_types = sorted(target_df["shift_type"].dropna().unique().tolist())

    for episode_id, episode_df in target_df.groupby("episode_id", sort=True):
        episode_seed = stable_int_hash(config.synthetic.random_seed, episode_id)
        candidate_df, eval_df = split_target_episode(
            episode_df,
            candidate_fraction=config.adaptation.candidate_fraction,
            random_state=episode_seed,
        )
        shift_type = str(eval_df["shift_type"].iloc[0])
        LOGGER.info("Evaluating target episode %s (%s).", episode_id, shift_type)

        base_metadata = {
            "panel": "adaptation",
            "strategy": "source_only",
            "budget": 0,
            "repeat_id": 0,
            "shift_type": shift_type,
            "episode_id": episode_id,
        }
        row, pred = _evaluate_model_run(
            model=base_model,
            source_calibration_df=source_splits.calibration,
            target_calibration_df=None,
            eval_df=eval_df,
            config=config,
            metadata=base_metadata,
        )
        adaptation_rows.append(row)
        raw_prediction_frames.append(pred)

        coarse_metadata = {
            "panel": "adaptation",
            "strategy": "coarse_source_only",
            "budget": 0,
            "repeat_id": 0,
            "shift_type": shift_type,
            "episode_id": episode_id,
        }
        row, pred = _evaluate_model_run(
            model=coarse_model,
            source_calibration_df=source_splits.calibration,
            target_calibration_df=None,
            eval_df=eval_df,
            config=config,
            metadata=coarse_metadata,
        )
        adaptation_rows.append(row)
        raw_prediction_frames.append(pred)

        acquisition_metadata = {
            "panel": "acquisition",
            "strategy": "residual_adapter",
            "acquisition_strategy": "source_only",
            "budget": 0,
            "repeat_id": 0,
            "shift_type": shift_type,
            "episode_id": episode_id,
        }
        row, pred = _evaluate_model_run(
            model=acquisition_base_model,
            source_calibration_df=source_splits.calibration,
            target_calibration_df=None,
            eval_df=eval_df,
            config=config,
            metadata=acquisition_metadata,
        )
        acquisition_rows.append(row)
        raw_prediction_frames.append(pred)

        for budget in positive_budgets:
            for repeat_id in range(config.adaptation.random_repeats):
                repeat_seed = stable_int_hash(config.adaptation.random_seed, episode_id, budget, repeat_id)
                if budget in positive_adaptation_budgets:
                    labeled_random, _ = _select_labeled_subset(
                        acquisition_strategy="random",
                        candidate_df=candidate_df,
                        base_model=base_model,
                        budget=budget,
                        config=config,
                        random_seed=repeat_seed,
                        shift_type=shift_type,
                    )
                    adaptation_models = _fit_adaptation_models(
                        labeled_df=labeled_random,
                        source_train_df=source_splits.train,
                        base_model=base_model,
                        config=config,
                        random_seed=repeat_seed,
                    )
                    for strategy_name, fitted_model in adaptation_models.items():
                        metadata = {
                            "panel": "adaptation",
                            "strategy": strategy_name,
                            "budget": budget,
                            "repeat_id": repeat_id,
                            "shift_type": shift_type,
                            "episode_id": episode_id,
                        }
                        row, pred = _evaluate_model_run(
                            model=fitted_model,
                            source_calibration_df=source_splits.calibration,
                            target_calibration_df=labeled_random,
                            eval_df=eval_df,
                            config=config,
                            metadata=metadata,
                        )
                        adaptation_rows.append(row)
                        raw_prediction_frames.append(pred)
                        if strategy_name == "era_adapter":
                            adapted_probability_rows.extend(
                                _evaluate_probability_calibration_diagnostic(
                                    model=fitted_model,
                                    source_calibration_df=source_splits.calibration,
                                    target_calibration_df=labeled_random,
                                    eval_df=eval_df,
                                    config=config,
                                    metadata=metadata,
                                )
                            )

                if budget in positive_acquisition_budgets:
                    acquisition_strategies = [
                        "random",
                        "diversity",
                        "shift_aware_diversity",
                        "stratified_diversity",
                        "stratified_predmean_diversity",
                        "uncertainty",
                        "uncertainty_trimmed",
                        "badge_proxy",
                        "density_ratio_uncertainty_diversity",
                        "two_stage_explore_exploit",
                        "entropy_threshold_diversity",
                        "uncertainty_threshold_diversity",
                        "uncertainty_threshold_diversity_capped",
                        "uncertainty_diversity_trimmed",
                        "oracle_residual",
                    ]
                    if config.adaptation.enable_hybrid_shift_policy:
                        acquisition_strategies.insert(2, "hybrid_shift_aware")
                    if config.adaptation.oracle_search_trials > 0:
                        acquisition_strategies.append("oracle_true_search")
                    for acquisition_strategy in acquisition_strategies:
                        if acquisition_strategy == "oracle_true_search":
                            labeled_df = _select_oracle_true_search_subset(
                                candidate_df=candidate_df,
                                eval_df=eval_df,
                                base_model=acquisition_base_model,
                                source_calibration_df=source_splits.calibration,
                                budget=budget,
                                config=config,
                                random_seed=repeat_seed + 97,
                                n_trials=config.adaptation.oracle_search_trials,
                            )
                            selection_metadata = {}
                        else:
                            labeled_df, selection_metadata = _select_labeled_subset(
                                acquisition_strategy=acquisition_strategy,
                                candidate_df=candidate_df,
                                base_model=acquisition_base_model,
                                budget=budget,
                                config=config,
                                random_seed=repeat_seed + 97,
                                shift_type=shift_type,
                            )
                        acquisition_model = FewShotResidualAdapter(
                            acquisition_base_model,
                            ridge_alpha=config.adaptation.ridge_alpha,
                            residual_feature_mode=config.adaptation.residual_feature_mode,
                            use_base_mean=config.adaptation.residual_use_base_mean,
                            use_base_std=config.adaptation.residual_use_base_std,
                            max_feature_dims=config.adaptation.residual_max_feature_dims,
                            alpha_grid=config.adaptation.residual_alpha_grid,
                            cv_folds=config.adaptation.residual_cv_folds,
                            selection_metric=config.adaptation.residual_selection_metric,
                            random_seed=repeat_seed + 131,
                            std_temperature_quantile=config.adaptation.std_temperature_quantile,
                            std_temperature_min=config.adaptation.std_temperature_min,
                            std_temperature_max=config.adaptation.std_temperature_max,
                            std_floor=config.adaptation.std_floor,
                        ).fit(labeled_df)
                        metadata = {
                            "panel": "acquisition",
                            "strategy": "residual_adapter",
                            "acquisition_strategy": acquisition_strategy,
                            "budget": budget,
                            "repeat_id": repeat_id,
                            "shift_type": shift_type,
                            "episode_id": episode_id,
                        }
                        metadata.update(selection_metadata)
                        row, pred = _evaluate_model_run(
                            model=acquisition_model,
                            source_calibration_df=source_splits.calibration,
                            target_calibration_df=labeled_df,
                            eval_df=eval_df,
                            config=config,
                            metadata=metadata,
                        )
                        acquisition_rows.append(row)
                        raw_prediction_frames.append(pred)
                        if acquisition_strategy == "entropy_threshold_diversity":
                            acquisition_weight_rows.append(
                                {
                                    "episode_id": episode_id,
                                    "shift_type": shift_type,
                                    "budget": budget,
                                    "repeat_id": repeat_id,
                                    "acquisition_strategy": acquisition_strategy,
                                    "acq_score_mode": selection_metadata.get("acq_score_mode", config.adaptation.acq_score_mode),
                                    "acq_weight_source": selection_metadata.get("acq_weight_source", "configured"),
                                    "weight_entropy": selection_metadata.get("acq_entropy_weight", np.nan),
                                    "weight_std": selection_metadata.get("acq_std_weight", np.nan),
                                    "weight_threshold": selection_metadata.get("acq_threshold_weight", np.nan),
                                    "weight_diversity": selection_metadata.get("acq_diversity_weight", np.nan),
                                }
                            )

    source_results = pd.DataFrame(source_results_rows)
    adaptation_results = pd.DataFrame(adaptation_rows)
    acquisition_results = pd.DataFrame(acquisition_rows)
    acquisition_weight_selection_summary = pd.DataFrame(acquisition_weight_rows)
    adapted_probability_results = pd.DataFrame(adapted_probability_rows)
    adapted_probability_diagnostic = _summarise_with_overall(
        adapted_probability_results,
        by=["shift_type", "strategy", "budget", "calibration_variant"],
    ) if not adapted_probability_results.empty else pd.DataFrame()
    raw_predictions = pd.concat(raw_prediction_frames, ignore_index=True) if raw_prediction_frames else pd.DataFrame()

    dataset_tables = summarize_dataset(dataset_df)
    adaptation_summary = _summarise_with_overall(adaptation_results, by=["shift_type", "strategy", "budget"])
    acquisition_summary = _summarise_with_overall(acquisition_results, by=["shift_type", "acquisition_strategy", "budget"])

    adaptation_avg = _average_over_repeats(adaptation_results, group_cols=["episode_id", "shift_type", "strategy", "budget"])
    acquisition_avg = _average_over_repeats(acquisition_results, group_cols=["episode_id", "shift_type", "acquisition_strategy", "budget"])

    adaptation_auc = _build_auc_table(adaptation_avg, strategy_col="strategy")
    acquisition_auc = _build_auc_table(acquisition_avg, strategy_col="acquisition_strategy")

    adaptation_stats = _build_comparison_table(
        adaptation_avg,
        strategy_col="strategy",
        comparisons=[
            ("era_adapter", "source_only"),
            ("era_adapter", "pooled_retrain"),
            ("era_adapter", "target_only"),
            ("era_adapter", "mean_shift_adapter"),
            ("era_adapter", "residual_adapter"),
            ("era_adapter", "ewc_adapter"),
            ("era_adapter", "shuffled_label_adapter"),
        ],
        shift_types=target_shift_types,
        evaluation_config=config.evaluation,
    )
    acquisition_stats = _build_comparison_table(
        acquisition_avg,
        strategy_col="acquisition_strategy",
        comparisons=[
            ("diversity", "random"),
            ("badge_proxy", "random"),
            ("badge_proxy", "diversity"),
            ("density_ratio_uncertainty_diversity", "random"),
            ("density_ratio_uncertainty_diversity", "diversity"),
            ("two_stage_explore_exploit", "random"),
            ("two_stage_explore_exploit", "diversity"),
            ("stratified_predmean_diversity", "random"),
            ("stratified_predmean_diversity", "diversity"),
            ("shift_aware_diversity", "random"),
            ("shift_aware_diversity", "diversity"),
            ("shift_aware_diversity", "stratified_diversity"),
            ("stratified_diversity", "random"),
            ("stratified_diversity", "diversity"),
            ("diversity", "uncertainty_threshold_diversity"),
            ("entropy_threshold_diversity", "random"),
            ("entropy_threshold_diversity", "diversity"),
            ("entropy_threshold_diversity", "uncertainty_threshold_diversity"),
            ("uncertainty_threshold_diversity", "random"),
            ("uncertainty_threshold_diversity", "uncertainty"),
            ("uncertainty_threshold_diversity", "diversity"),
            ("hybrid_shift_aware", "random"),
            ("hybrid_shift_aware", "diversity"),
            ("uncertainty_threshold_diversity_capped", "random"),
            ("uncertainty_threshold_diversity_capped", "diversity"),
            ("uncertainty_threshold_diversity_capped", "uncertainty_threshold_diversity"),
            ("uncertainty_diversity_trimmed", "random"),
            ("uncertainty_diversity_trimmed", "diversity"),
            ("uncertainty_trimmed", "uncertainty"),
            ("oracle_residual", "diversity"),
            ("oracle_residual", "random"),
            ("oracle_residual", "uncertainty_threshold_diversity"),
            ("oracle_true_search", "diversity"),
            ("oracle_true_search", "random"),
            ("oracle_true_search", "oracle_residual"),
        ],
        shift_types=target_shift_types,
        evaluation_config=config.evaluation,
    )
    acquisition_budget_stats = _build_budget_comparison_table(
        acquisition_avg,
        strategy_col="acquisition_strategy",
        comparisons=[
            ("diversity", "random"),
            ("badge_proxy", "diversity"),
            ("density_ratio_uncertainty_diversity", "diversity"),
            ("stratified_predmean_diversity", "diversity"),
            ("two_stage_explore_exploit", "diversity"),
            ("entropy_threshold_diversity", "diversity"),
        ],
        shift_types=target_shift_types,
        evaluation_config=config.evaluation,
    )

    external_validity_results = pd.DataFrame()
    external_validity_summary = pd.DataFrame()
    if dataset is not None:
        resolved_name = dataset_name
        if resolved_name is None and isinstance(dataset, (str, Path)):
            resolved_name = Path(dataset).stem
        if resolved_name is None:
            resolved_name = "external_dataset"
        external_validity_results = adaptation_results.copy()
        if not external_validity_results.empty:
            external_validity_results.insert(0, "dataset_name", resolved_name)
        external_validity_summary = select_overall_rows(adaptation_summary).copy()
        if not external_validity_summary.empty:
            external_validity_summary.insert(0, "dataset_name", resolved_name)

    artifacts = BenchmarkArtifacts(
        dataset=dataset_df,
        source_results=source_results,
        adaptation_results=adaptation_results,
        acquisition_results=acquisition_results,
        raw_predictions=raw_predictions,
        dataset_tables=dataset_tables,
        adaptation_summary=adaptation_summary,
        acquisition_summary=acquisition_summary,
        adaptation_auc=adaptation_auc,
        acquisition_auc=acquisition_auc,
        adaptation_stats=adaptation_stats,
        acquisition_stats=acquisition_stats,
        acquisition_budget_stats=acquisition_budget_stats,
        acquisition_weight_selection_summary=acquisition_weight_selection_summary,
        adapted_probability_diagnostic=adapted_probability_diagnostic,
        external_validity_results=external_validity_results,
        external_validity_summary=external_validity_summary,
        model_artifact_paths=artifact_paths,
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing outputs to %s", output_path)
        tables = {
            **dataset_tables,
            "source_results": source_results,
            "adaptation_results": adaptation_results,
            "acquisition_results": acquisition_results,
            "adaptation_summary": adaptation_summary,
            "acquisition_summary": acquisition_summary,
            "adaptation_auc": adaptation_auc,
            "acquisition_auc": acquisition_auc,
            "adaptation_stats": adaptation_stats,
            "acquisition_stats": acquisition_stats,
            "acquisition_budget_stats": acquisition_budget_stats,
        }
        if not acquisition_weight_selection_summary.empty:
            tables["acquisition_weight_selection_summary"] = acquisition_weight_selection_summary
        if not adapted_probability_diagnostic.empty:
            tables["adapted_probability_calibration_diagnostic"] = adapted_probability_diagnostic
        if not external_validity_results.empty:
            tables["external_validity_results"] = external_validity_results
        if not external_validity_summary.empty:
            tables["external_validity_summary"] = external_validity_summary
        if config.reporting.save_raw_predictions:
            tables["raw_predictions"] = raw_predictions
        write_csv_tables(output_dir=output_path / "tables", tables=tables)
        write_json(config.to_dict(), output_path / "metadata" / "config.json")
        write_json(collect_environment_metadata(), output_path / "metadata" / "environment.json")
        figure_paths = _generate_figures(
            output_dir=output_path,
            dataset=dataset_df,
            adaptation_summary=adaptation_summary,
            acquisition_summary=acquisition_summary,
            adaptation_auc=adaptation_auc,
            acquisition_auc=acquisition_auc,
            raw_predictions=raw_predictions,
        )
        artifacts.figure_paths = figure_paths
        write_run_report(
            output_dir=output_path,
            config_dict=config.to_dict(),
            dataset=dataset_df,
            dataset_tables=dataset_tables,
            source_results=source_results,
            adaptation_summary=adaptation_summary,
            acquisition_summary=acquisition_summary,
            adaptation_auc=adaptation_auc,
            acquisition_auc=acquisition_auc,
            adaptation_stats=adaptation_stats,
            acquisition_stats=acquisition_stats,
            acquisition_weight_selection_summary=acquisition_weight_selection_summary,
            adapted_probability_diagnostic=adapted_probability_diagnostic,
            external_validity_summary=external_validity_summary,
            model_artifact_paths=artifact_paths,
            figure_paths=figure_paths,
        )
        write_paper_asset_manifest(output_dir=output_path, figure_paths=figure_paths)

    return artifacts
