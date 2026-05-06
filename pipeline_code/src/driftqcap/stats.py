"""Statistical summaries and paired comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedComparison:
    comparison: str
    n_pairs: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    p_value: float | None
    effect_size: float | None



def compute_curve_auc(
    df: pd.DataFrame,
    *,
    budget_col: str = "budget",
    metric_col: str = "mae",
) -> float:
    sub = df.sort_values(budget_col)
    x = sub[budget_col].to_numpy(dtype=float)
    y = sub[metric_col].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("Cannot compute AUC on an empty dataframe.")
    if len(x) == 1:
        return float(y[0])
    return float(np.trapezoid(y, x) / max(x.max() - x.min(), 1.0))



def episode_level_auc(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    strategy_col: str,
    metric_col: str,
    budget_col: str = "budget",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols + [strategy_col], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        data = {col: value for col, value in zip(group_cols + [strategy_col], keys)}
        data[f"{metric_col}_auc"] = compute_curve_auc(sub, budget_col=budget_col, metric_col=metric_col)
        rows.append(data)
    return pd.DataFrame(rows)



def bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape for paired bootstrapping.")
    rng = np.random.default_rng(random_state)
    n = len(a)
    if n == 0:
        return np.nan, np.nan
    deltas = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        deltas[idx] = float(np.mean(a[sample_idx] - b[sample_idx]))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)



def _rank_biserial_from_wilcoxon(x: np.ndarray, y: np.ndarray) -> float | None:
    diff = x - y
    diff = diff[diff != 0]
    if diff.size == 0:
        return None
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    return float((pos - neg) / (pos + neg))



def paired_strategy_comparison(
    df: pd.DataFrame,
    *,
    strategy_col: str,
    metric_col: str,
    strategy_a: str,
    strategy_b: str,
    group_cols: list[str],
    bootstrap_resamples: int,
    random_state: int,
) -> PairedComparison:
    auc_df = episode_level_auc(df, group_cols=group_cols, strategy_col=strategy_col, metric_col=metric_col)
    pivot = auc_df.pivot_table(index=group_cols, columns=strategy_col, values=f"{metric_col}_auc")
    if strategy_a not in pivot.columns or strategy_b not in pivot.columns:
        return PairedComparison(
            comparison=f"{strategy_a} vs {strategy_b}",
            n_pairs=0,
            mean_delta=np.nan,
            median_delta=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            p_value=None,
            effect_size=None,
        )
    paired = pivot[[strategy_a, strategy_b]].dropna()
    if paired.empty:
        return PairedComparison(
            comparison=f"{strategy_a} vs {strategy_b}",
            n_pairs=0,
            mean_delta=np.nan,
            median_delta=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            p_value=None,
            effect_size=None,
        )
    a = paired[strategy_a].to_numpy(dtype=float)
    b = paired[strategy_b].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_difference(a, b, n_resamples=bootstrap_resamples, random_state=random_state)
    diff = a - b
    if np.allclose(diff, 0.0):
        p_value = 1.0
        effect_size = 0.0
    else:
        try:
            p_value = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            p_value = None
        effect_size = _rank_biserial_from_wilcoxon(a, b)
    return PairedComparison(
        comparison=f"{strategy_a} vs {strategy_b}",
        n_pairs=len(a),
        mean_delta=float(np.mean(a - b)),
        median_delta=float(np.median(a - b)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        effect_size=effect_size,
    )



def holm_correction(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    out: list[float | None] = [None for _ in p_values]
    if not indexed:
        return out
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running = 0.0
    for rank, (original_idx, p) in enumerate(indexed, start=1):
        adjusted = min(1.0, (m - rank + 1) * p)
        running = max(running, adjusted)
        out[original_idx] = running
    return out



def comparisons_to_frame(comparisons: list[PairedComparison]) -> pd.DataFrame:
    df = pd.DataFrame([comparison.__dict__ for comparison in comparisons])
    if df.empty or "p_value" not in df.columns:
        return df
    df["p_holm"] = holm_correction(df["p_value"].tolist())
    return df
