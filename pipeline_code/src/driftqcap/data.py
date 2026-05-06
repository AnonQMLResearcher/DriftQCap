"""Data validation and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import REQUIRED_COLUMNS, augment_features


@dataclass(frozen=True)
class SourceSplits:
    """Train/calibration/test splits for the source domain."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def ensure_pass_label(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Ensure that the pass_label column exists and matches the configured threshold."""
    out = df.copy()
    out["pass_label"] = (out["error_rate"].astype(float) <= float(threshold)).astype(int)
    return out


def validate_dataframe(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if df["error_rate"].isna().any():
        raise ValueError("Column 'error_rate' contains NaN values.")
    if (df["error_rate"] < 0).any():
        raise ValueError("Column 'error_rate' must be non-negative.")


def prepare_dataframe(
    df: pd.DataFrame,
    *,
    pass_threshold: float | None = None,
    force_recompute_pass_label: bool = False,
) -> pd.DataFrame:
    """Validate and augment a dataframe."""
    out = df.copy()
    if pass_threshold is not None and (force_recompute_pass_label or "pass_label" not in out.columns):
        out = ensure_pass_label(out, threshold=pass_threshold)
    validate_dataframe(out)
    return augment_features(out)


def make_source_splits(
    df: pd.DataFrame,
    *,
    random_state: int = 20260308,
    calibration_size: float = 0.15,
    test_size: float = 0.15,
) -> SourceSplits:
    validate_dataframe(df)
    source = df[df["domain"] == "source"].copy()
    if source.empty:
        raise ValueError("No source rows found.")

    def _safe_stratify_labels(labels: pd.Series) -> pd.Series | None:
        if labels.nunique() <= 1:
            return None
        counts = labels.value_counts(dropna=False)
        if counts.min() < 2:
            return None
        return labels

    stratify = _safe_stratify_labels(source["episode_id"])
    train_df, temp_df = train_test_split(
        source,
        test_size=calibration_size + test_size,
        random_state=random_state,
        stratify=stratify,
    )

    temp_stratify = _safe_stratify_labels(temp_df["episode_id"])
    calib_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + calibration_size),
        random_state=random_state,
        stratify=temp_stratify,
    )
    return SourceSplits(
        train=train_df.reset_index(drop=True),
        calibration=calib_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def split_target_episode(
    episode_df: pd.DataFrame,
    *,
    candidate_fraction: float = 0.45,
    random_state: int = 20260308,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episode_df["episode_id"].nunique() != 1:
        raise ValueError("split_target_episode expects a single target episode.")
    validate_dataframe(episode_df)
    n_rows = len(episode_df)
    if n_rows < 2:
        raise ValueError("Target episode must contain at least 2 rows for candidate/eval split.")

    # Convert fractional split to integer counts with hard non-empty guarantees.
    n_candidate = int(round(float(candidate_fraction) * n_rows))
    n_candidate = max(1, min(n_rows - 1, n_candidate))

    shuffled = episode_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    candidate_df = shuffled.iloc[:n_candidate].copy()
    eval_df = shuffled.iloc[n_candidate:].copy()
    return candidate_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def make_weight_vector(n_source: int, n_target: int, target_weight: float) -> np.ndarray:
    """Construct per-row weights for pooled retraining."""
    if n_target == 0:
        return np.ones(n_source, dtype=float)
    return np.concatenate([np.ones(n_source, dtype=float), np.full(n_target, float(target_weight), dtype=float)])
