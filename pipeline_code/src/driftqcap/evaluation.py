"""Evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

from .calibration import expected_calibration_error
from .models import PredictionBundle


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    spearman: float
    coverage_80: float | None
    coverage_90: float | None
    coverage_95: float | None
    width_80: float | None
    width_90: float | None
    width_95: float | None
    winkler_80: float | None
    winkler_90: float | None
    winkler_95: float | None
    coverage_80_stratum_0: float | None
    coverage_80_stratum_1: float | None
    coverage_80_stratum_2: float | None
    coverage_80_stratum_3: float | None
    width_80_stratum_0: float | None
    width_80_stratum_1: float | None
    width_80_stratum_2: float | None
    width_80_stratum_3: float | None


@dataclass(frozen=True)
class ClassificationMetrics:
    brier_raw: float | None
    brier_calibrated: float | None
    ece_raw: float | None
    ece_calibrated: float | None
    auroc_raw: float | None
    auroc_calibrated: float | None
    auprc_raw: float | None
    auprc_calibrated: float | None
    f1_raw: float | None
    f1_calibrated: float | None
    pass_rate: float



def _winkler_score(*, lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    penalty = np.zeros_like(y_true, dtype=float)
    penalty[below] = (2.0 / alpha) * (lower[below] - y_true[below])
    penalty[above] = (2.0 / alpha) * (y_true[above] - upper[above])
    return float(np.mean(width + penalty))


def _interval_metrics(bundle: PredictionBundle, level: str, y_true: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if level not in bundle.intervals:
        return None, None, None
    lower, upper = bundle.intervals[level]
    coverage = float(((y_true >= lower) & (y_true <= upper)).mean())
    width = float(np.mean(upper - lower))
    alpha = 1.0 - float(level) / 100.0
    winkler = _winkler_score(lower=lower, upper=upper, y_true=y_true, alpha=alpha)
    return coverage, width, winkler


def _conditional_interval_metrics(bundle: PredictionBundle, level: str, y_true: np.ndarray, n_strata: int = 4) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    if level not in bundle.intervals or y_true.size == 0:
        for idx in range(n_strata):
            results[f"coverage_{level}_stratum_{idx}"] = None
            results[f"width_{level}_stratum_{idx}"] = None
        return results
    order = np.argsort(bundle.mean)
    edges = np.linspace(0, y_true.size, n_strata + 1, dtype=int)
    lower, upper = bundle.intervals[level]
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right <= left:
            results[f"coverage_{level}_stratum_{idx}"] = None
            results[f"width_{level}_stratum_{idx}"] = None
            continue
        take = order[left:right]
        results[f"coverage_{level}_stratum_{idx}"] = float(((y_true[take] >= lower[take]) & (y_true[take] <= upper[take])).mean())
        results[f"width_{level}_stratum_{idx}"] = float(np.mean(upper[take] - lower[take]))
    return results



def evaluate_regression(*, y_true: np.ndarray, bundle: PredictionBundle) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    mae = mean_absolute_error(y_true, bundle.mean)
    rmse = np.sqrt(mean_squared_error(y_true, bundle.mean))
    r2 = r2_score(y_true, bundle.mean)
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(bundle.mean), 0.0):
        spear = 0.0
    else:
        spear = spearmanr(y_true, bundle.mean).statistic
    c80, w80, wk80 = _interval_metrics(bundle, "80", y_true)
    c90, w90, wk90 = _interval_metrics(bundle, "90", y_true)
    c95, w95, wk95 = _interval_metrics(bundle, "95", y_true)
    cond80 = _conditional_interval_metrics(bundle, "80", y_true, n_strata=4)
    return RegressionMetrics(
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        spearman=float(0.0 if spear is None or np.isnan(spear) else spear),
        coverage_80=c80,
        coverage_90=c90,
        coverage_95=c95,
        width_80=w80,
        width_90=w90,
        width_95=w95,
        winkler_80=wk80,
        winkler_90=wk90,
        winkler_95=wk95,
        coverage_80_stratum_0=cond80["coverage_80_stratum_0"],
        coverage_80_stratum_1=cond80["coverage_80_stratum_1"],
        coverage_80_stratum_2=cond80["coverage_80_stratum_2"],
        coverage_80_stratum_3=cond80["coverage_80_stratum_3"],
        width_80_stratum_0=cond80["width_80_stratum_0"],
        width_80_stratum_1=cond80["width_80_stratum_1"],
        width_80_stratum_2=cond80["width_80_stratum_2"],
        width_80_stratum_3=cond80["width_80_stratum_3"],
    )



def _binary_metrics(labels: np.ndarray, probability: np.ndarray | None, *, n_bins: int) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if probability is None:
        return None, None, None, None, None
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(labels)) == 1:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(labels, probability))
        auprc = float(average_precision_score(labels, probability))
    preds = (probability >= 0.5).astype(int)
    f1 = float(f1_score(labels, preds, zero_division=0))
    brier = float(brier_score_loss(labels, probability))
    ece = expected_calibration_error(probability=probability, labels=labels, n_bins=n_bins)
    return brier, ece, auroc, auprc, f1



def evaluate_classification(
    *,
    y_true: np.ndarray,
    pass_probability_raw: np.ndarray | None,
    pass_probability_calibrated: np.ndarray | None,
    threshold: float,
    n_bins: int = 10,
) -> ClassificationMetrics:
    labels = (np.asarray(y_true, dtype=float) <= threshold).astype(int)
    pass_rate = float(labels.mean())
    brier_raw, ece_raw, auroc_raw, auprc_raw, f1_raw = _binary_metrics(labels, pass_probability_raw, n_bins=n_bins)
    brier_cal, ece_cal, auroc_cal, auprc_cal, f1_cal = _binary_metrics(labels, pass_probability_calibrated, n_bins=n_bins)
    return ClassificationMetrics(
        brier_raw=brier_raw,
        brier_calibrated=brier_cal,
        ece_raw=ece_raw,
        ece_calibrated=ece_cal,
        auroc_raw=auroc_raw,
        auroc_calibrated=auroc_cal,
        auprc_raw=auprc_raw,
        auprc_calibrated=auprc_cal,
        f1_raw=f1_raw,
        f1_calibrated=f1_cal,
        pass_rate=pass_rate,
    )



def combine_metrics(
    *,
    y_true: np.ndarray,
    bundle: PredictionBundle,
    threshold: float,
    n_bins: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    reg = asdict(evaluate_regression(y_true=y_true, bundle=bundle))
    clf = asdict(
        evaluate_classification(
            y_true=y_true,
            pass_probability_raw=bundle.pass_probability_raw,
            pass_probability_calibrated=bundle.pass_probability_calibrated,
            threshold=threshold,
            n_bins=n_bins,
        )
    )
    return {**metadata, **reg, **clf}



def build_prediction_frame(
    *,
    eval_df: pd.DataFrame,
    bundle: PredictionBundle,
    run_metadata: dict[str, object],
    threshold: float,
) -> pd.DataFrame:
    out = eval_df[["circuit_id", "episode_id", "shift_type", "family", "domain", "error_rate", "pass_label"]].copy()
    out["prediction_mean"] = bundle.mean
    out["prediction_std"] = bundle.std
    out["pass_probability_raw"] = np.nan if bundle.pass_probability_raw is None else bundle.pass_probability_raw
    out["pass_probability_calibrated"] = np.nan if bundle.pass_probability_calibrated is None else bundle.pass_probability_calibrated
    out["threshold"] = threshold
    for level, (lower, upper) in bundle.intervals.items():
        out[f"lower_{level}"] = lower
        out[f"upper_{level}"] = upper
    for key, value in run_metadata.items():
        out[key] = value
    return out



def aggregate_results(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col not in by and pd.api.types.is_numeric_dtype(df[col])]
    grouped = df.groupby(by, dropna=False)[numeric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()
