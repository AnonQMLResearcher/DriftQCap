"""Calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from .models import PredictionBundle


@dataclass(frozen=True)
class ConformalArtifacts:
    """Conformal calibration artifacts based on scaled residuals."""

    quantiles: dict[str, float]
    min_std: float = 1e-4
    stratify_by: str = "global"
    bin_edges: np.ndarray | None = None
    bin_quantiles: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Combined calibration artifacts."""

    conformal: ConformalArtifacts
    pass_probability_calibrator: IsotonicRegression | LogisticRegression | None = None
    probability_calibration_mode: str = "source_only"
    probability_calibration_rows: int = 0
    probability_calibration_pos: int = 0
    probability_calibration_neg: int = 0
    probability_calibrator_type: str = "none"
    probability_calibration_weight: float = 0.0
    conformal_calibration_mode: str = "source_only"
    conformal_calibration_rows: int = 0


def blend_conformal_artifacts(
    *,
    source: ConformalArtifacts,
    target: ConformalArtifacts,
    target_weight: float,
) -> ConformalArtifacts:
    weight = float(np.clip(target_weight, 0.0, 1.0))
    quantiles = {
        level: float((1.0 - weight) * source.quantiles[level] + weight * target.quantiles[level])
        for level in source.quantiles
        if level in target.quantiles
    }
    if (
        source.bin_edges is None
        or target.bin_edges is None
        or source.bin_quantiles is None
        or target.bin_quantiles is None
        or source.stratify_by != target.stratify_by
        or source.bin_edges.shape != target.bin_edges.shape
        or not np.allclose(source.bin_edges, target.bin_edges)
    ):
        return ConformalArtifacts(
            quantiles=quantiles,
            min_std=min(source.min_std, target.min_std),
            stratify_by=source.stratify_by,
            bin_edges=source.bin_edges,
            bin_quantiles=source.bin_quantiles,
        )
    bin_quantiles: dict[str, np.ndarray] = {}
    for level, source_values in source.bin_quantiles.items():
        if level not in target.bin_quantiles:
            continue
        bin_quantiles[level] = (1.0 - weight) * np.asarray(source_values, dtype=float) + weight * np.asarray(
            target.bin_quantiles[level],
            dtype=float,
        )
    return ConformalArtifacts(
        quantiles=quantiles,
        min_std=min(source.min_std, target.min_std),
        stratify_by=source.stratify_by,
        bin_edges=source.bin_edges,
        bin_quantiles=bin_quantiles,
    )


def fit_scaled_conformal(
    *,
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    alphas: tuple[float, ...] = (0.20, 0.10, 0.05),
    min_std: float = 1e-4,
) -> ConformalArtifacts:
    y_true = np.asarray(y_true, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std, dtype=float), min_std)
    if y_true.size == 0:
        raise ValueError("Cannot fit conformal intervals on an empty calibration set.")
    scaled_residual = np.abs(y_true - pred_mean) / pred_std
    quantiles: dict[str, float] = {}
    n = scaled_residual.size
    for alpha in alphas:
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = float(np.clip(q_level, 0.0, 1.0))
        q = np.quantile(scaled_residual, q_level, method="higher")
        quantiles[f"{int(round((1 - alpha) * 100)):02d}"] = float(q)
    return ConformalArtifacts(quantiles=quantiles, min_std=min_std)


def fit_locally_adaptive_conformal(
    *,
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    alphas: tuple[float, ...] = (0.20, 0.10, 0.05),
    n_bins: int = 4,
    min_bin_size: int = 12,
    min_std: float = 1e-4,
    fallback_to_global: bool = True,
    stratify_by: str = "pred_mean",
) -> ConformalArtifacts:
    y_true = np.asarray(y_true, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std, dtype=float), min_std)
    if y_true.size == 0:
        raise ValueError("Cannot fit conformal intervals on an empty calibration set.")
    global_artifacts = fit_scaled_conformal(
        y_true=y_true,
        pred_mean=pred_mean,
        pred_std=pred_std,
        alphas=alphas,
        min_std=min_std,
    )
    n_bins = max(int(n_bins), 2)
    if y_true.size < max(min_bin_size * 2, n_bins):
        return global_artifacts
    if stratify_by == "pred_std":
        stratifier = pred_std
    else:
        stratifier = pred_mean
    q_edges = np.quantile(stratifier, np.linspace(0.0, 1.0, n_bins + 1))
    q_edges = np.unique(q_edges)
    if q_edges.size < 3:
        return global_artifacts
    bucket = np.digitize(stratifier, q_edges[1:-1], right=False)
    per_level: dict[str, np.ndarray] = {}
    for level, global_q in global_artifacts.quantiles.items():
        level_values = np.full(q_edges.size - 1, float(global_q), dtype=float)
        for bin_id in range(q_edges.size - 1):
            mask = bucket == bin_id
            if int(mask.sum()) < int(min_bin_size):
                if not fallback_to_global:
                    continue
                level_values[bin_id] = float(global_q)
                continue
            local = fit_scaled_conformal(
                y_true=y_true[mask],
                pred_mean=pred_mean[mask],
                pred_std=pred_std[mask],
                alphas=alphas,
                min_std=min_std,
            )
            level_values[bin_id] = float(local.quantiles[level])
        per_level[level] = level_values
    return ConformalArtifacts(
        quantiles=global_artifacts.quantiles,
        min_std=min_std,
        stratify_by=stratify_by,
        bin_edges=q_edges.astype(float),
        bin_quantiles=per_level,
    )


def attach_intervals(bundle: PredictionBundle, *, conformal: ConformalArtifacts) -> PredictionBundle:
    safe_std = np.maximum(bundle.std, conformal.min_std)
    if conformal.bin_edges is not None and conformal.bin_quantiles:
        if conformal.stratify_by == "pred_std":
            stratifier = safe_std
        else:
            stratifier = bundle.mean
        bucket = np.digitize(stratifier, conformal.bin_edges[1:-1], right=False)
    else:
        bucket = None
    for level, q in conformal.quantiles.items():
        if bucket is not None and level in conformal.bin_quantiles:
            q_values = np.asarray(conformal.bin_quantiles[level], dtype=float)
            q_per_row = q_values[np.clip(bucket, 0, len(q_values) - 1)]
        else:
            q_per_row = float(q)
        lower = np.clip(bundle.mean - q_per_row * safe_std, 0.0, 1.0)
        upper = np.clip(bundle.mean + q_per_row * safe_std, 0.0, 1.0)
        bundle.intervals[level] = (lower, upper)
    return bundle


def fit_pass_probability_calibrator(*, raw_probability: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(raw_probability), np.asarray(labels).astype(int))
    return calibrator


def fit_pass_probability_platt(*, raw_probability: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=20260308)
    x = np.asarray(raw_probability, dtype=float).reshape(-1, 1)
    y = np.asarray(labels).astype(int)
    model.fit(x, y)
    return model


def calibrate_pass_probability(
    *,
    raw_probability: np.ndarray,
    calibrator: IsotonicRegression | LogisticRegression | None,
) -> np.ndarray:
    raw_probability = np.asarray(raw_probability)
    if calibrator is None:
        return raw_probability
    if isinstance(calibrator, LogisticRegression):
        return np.asarray(calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(calibrator.predict(raw_probability))


def expected_calibration_error(*, probability: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not np.any(mask):
            continue
        bin_prob = probability[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.mean() * abs(bin_prob - bin_acc)
    return float(ece)


def reliability_curve(*, probability: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    accuracies = []
    counts = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not np.any(mask):
            continue
        centers.append(probability[mask].mean())
        accuracies.append(labels[mask].mean())
        counts.append(mask.sum())
    return np.asarray(centers), np.asarray(accuracies), np.asarray(counts)


def reliability_curve_equal_mass(
    *,
    probability: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve with equal-count bins for skewed label prevalence."""
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels).astype(int)
    if probability.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    order = np.argsort(probability)
    p_sorted = probability[order]
    y_sorted = labels[order]
    edges = np.linspace(0, probability.size, n_bins + 1, dtype=int)
    centers = []
    accuracies = []
    counts = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        p_bin = p_sorted[left:right]
        y_bin = y_sorted[left:right]
        centers.append(float(p_bin.mean()))
        accuracies.append(float(y_bin.mean()))
        counts.append(int(len(p_bin)))
    return np.asarray(centers), np.asarray(accuracies), np.asarray(counts)
