"""Plotting utilities for DriftQCap."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import reliability_curve, reliability_curve_equal_mass


_DEFAULT_DPI = 220


def save_figure(fig: plt.Figure, base_path: str | Path, *, close: bool = True) -> dict[str, str]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = str(base_path.with_suffix(".png"))
    fig.savefig(png_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    return {"png": png_path}


def _new_figure(*, width: float = 7.0, height: float = 4.5) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def plot_dataset_error_distribution(dataset: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_figure(width=7.6, height=4.8)
    grouped = dataset.groupby("shift_type", dropna=False)["error_rate"]
    labels = []
    values = []
    for shift_type, series in grouped:
        labels.append(str(shift_type))
        values.append(series.to_numpy(dtype=float))
    ax.boxplot(values, tick_labels=labels, orientation="vertical", showfliers=False)
    ax.set_title("Synthetic benchmark difficulty by shift family")
    ax.set_xlabel("Shift family")
    ax.set_ylabel("Capability target (error rate)")
    ax.tick_params(axis="x", rotation=25)
    return fig


def plot_zero_shot_shift_mae(adaptation_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = _new_figure(width=7.2, height=4.4)
    sub = adaptation_summary[(adaptation_summary["budget"] == 0) & (adaptation_summary["strategy"] == "source_only")]
    sub = sub[sub["shift_type"] != "overall"].sort_values("mae_mean", ascending=False)
    ax.bar(sub["shift_type"].astype(str), sub["mae_mean"].astype(float))
    ax.set_title("Zero-shot target difficulty across shift families")
    ax.set_xlabel("Shift family")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=20)
    return fig


def _plot_budget_curves(
    summary: pd.DataFrame,
    *,
    strategy_col: str,
    metric_prefix: str,
    title: str,
    ylabel: str,
    shift_type: str = "overall",
    include_std: bool = True,
) -> plt.Figure:
    fig, ax = _new_figure(width=7.6, height=4.8)
    sub = summary[summary["shift_type"] == shift_type].copy()
    sub = sub.sort_values("budget")
    if sub.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel("Label budget")
        ax.set_ylabel(ylabel)
        return fig
    for strategy, strategy_df in sub.groupby(strategy_col, dropna=False):
        x = strategy_df["budget"].to_numpy(dtype=float)
        y = strategy_df[f"{metric_prefix}_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=str(strategy))
        if include_std and f"{metric_prefix}_std" in strategy_df.columns:
            std = strategy_df[f"{metric_prefix}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - std, y + std, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Target label budget")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_adaptation_curves(adaptation_summary: pd.DataFrame, *, shift_type: str = "overall", metric: str = "mae") -> plt.Figure:
    metric_label = {
        "mae": "MAE",
        "ece_calibrated": "Calibrated ECE",
        "coverage_90": "90% interval coverage",
    }.get(metric, metric)
    title = f"Adaptation sample-efficiency curves ({shift_type})"
    return _plot_budget_curves(
        adaptation_summary,
        strategy_col="strategy",
        metric_prefix=metric,
        title=title,
        ylabel=metric_label,
        shift_type=shift_type,
    )


def plot_acquisition_curves(acquisition_summary: pd.DataFrame, *, shift_type: str = "overall", metric: str = "mae") -> plt.Figure:
    metric_label = {
        "mae": "MAE",
        "ece_calibrated": "Calibrated ECE",
        "coverage_90": "90% interval coverage",
    }.get(metric, metric)
    title = f"Acquisition sample-efficiency curves ({shift_type})"
    return _plot_budget_curves(
        acquisition_summary,
        strategy_col="acquisition_strategy",
        metric_prefix=metric,
        title=title,
        ylabel=metric_label,
        shift_type=shift_type,
    )


def plot_reliability_diagram_from_predictions(
    predictions: pd.DataFrame,
    *,
    probability_col_raw: str = "pass_probability_raw",
    probability_col_calibrated: str = "pass_probability_calibrated",
    n_bins: int = 10,
    title: str,
) -> plt.Figure:
    fig, ax = _new_figure(width=6.0, height=5.2)
    labels = predictions["pass_label"].to_numpy(dtype=int)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    raw = predictions[probability_col_raw].to_numpy(dtype=float)
    centers_raw, acc_raw, counts_raw = reliability_curve(probability=raw, labels=labels, n_bins=n_bins)
    plotted_combined = False
    if probability_col_calibrated in predictions.columns and predictions[probability_col_calibrated].notna().any():
        cal = predictions[probability_col_calibrated].to_numpy(dtype=float)
        centers_cal, acc_cal, counts_cal = reliability_curve(probability=cal, labels=labels, n_bins=n_bins)
        same_shape = (centers_raw.shape == centers_cal.shape) and (acc_raw.shape == acc_cal.shape)
        if same_shape and np.allclose(centers_raw, centers_cal, atol=1e-10, rtol=1e-6) and np.allclose(
            acc_raw, acc_cal, atol=1e-10, rtol=1e-6
        ):
            ax.plot(centers_raw, acc_raw, marker="o", label=f"raw = calibrated (n={counts_raw.sum()})")
            plotted_combined = True
        else:
            ax.plot(centers_raw, acc_raw, marker="o", label=f"raw (n={counts_raw.sum()})")
            ax.plot(centers_cal, acc_cal, marker="s", label=f"calibrated (n={counts_cal.sum()})")
            plotted_combined = True
    if not plotted_combined:
        ax.plot(centers_raw, acc_raw, marker="o", label=f"raw (n={counts_raw.sum()})")
    ax.set_title(title)
    ax.set_xlabel("Predicted pass probability")
    ax.set_ylabel("Empirical pass rate")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_reliability_diagram_equal_mass(
    predictions: pd.DataFrame,
    *,
    probability_col_raw: str = "pass_probability_raw",
    probability_col_calibrated: str = "pass_probability_calibrated",
    n_bins: int = 10,
    title: str,
) -> plt.Figure:
    fig, ax = _new_figure(width=6.0, height=5.2)
    labels = predictions["pass_label"].to_numpy(dtype=int)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    raw = predictions[probability_col_raw].to_numpy(dtype=float)
    centers_raw, acc_raw, counts_raw = reliability_curve_equal_mass(probability=raw, labels=labels, n_bins=n_bins)
    plotted_combined = False
    if probability_col_calibrated in predictions.columns and predictions[probability_col_calibrated].notna().any():
        cal = predictions[probability_col_calibrated].to_numpy(dtype=float)
        centers_cal, acc_cal, counts_cal = reliability_curve_equal_mass(probability=cal, labels=labels, n_bins=n_bins)
        same_shape = (centers_raw.shape == centers_cal.shape) and (acc_raw.shape == acc_cal.shape)
        if same_shape and np.allclose(centers_raw, centers_cal, atol=1e-10, rtol=1e-6) and np.allclose(
            acc_raw, acc_cal, atol=1e-10, rtol=1e-6
        ):
            ax.plot(centers_raw, acc_raw, marker="o", label=f"raw = cal eq-mass (n={counts_raw.sum()})")
            plotted_combined = True
        else:
            ax.plot(centers_raw, acc_raw, marker="o", label=f"raw eq-mass (n={counts_raw.sum()})")
            ax.plot(centers_cal, acc_cal, marker="s", label=f"cal eq-mass (n={counts_cal.sum()})")
            plotted_combined = True
    if not plotted_combined:
        ax.plot(centers_raw, acc_raw, marker="o", label=f"raw eq-mass (n={counts_raw.sum()})")
    ax.set_title(title)
    ax.set_xlabel("Predicted pass probability")
    ax.set_ylabel("Empirical pass rate")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_interval_coverage_curves(
    summary: pd.DataFrame,
    *,
    strategy_col: str,
    strategy_name: str,
    shift_type: str = "overall",
    title: str | None = None,
) -> plt.Figure:
    fig, ax = _new_figure(width=7.0, height=4.6)
    sub = summary[(summary["shift_type"] == shift_type) & (summary[strategy_col] == strategy_name)].sort_values("budget")
    if sub.empty:
        ax.set_title((title or "Interval coverage") + " (no data)")
        ax.set_xlabel("Target label budget")
        ax.set_ylabel("Empirical coverage")
        return fig
    x = sub["budget"].to_numpy(dtype=float)
    for nominal, col in [(0.80, "coverage_80_mean"), (0.90, "coverage_90_mean"), (0.95, "coverage_95_mean")]:
        if col in sub.columns:
            ax.plot(x, sub[col].to_numpy(dtype=float), marker="o", label=f"empirical {int(nominal*100)}%")
            ax.axhline(nominal, linestyle="--", linewidth=1.0)
    ax.set_title(title or f"Interval coverage for {strategy_name} ({shift_type})")
    ax.set_xlabel("Target label budget")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_auc_ranking(
    auc_rank_df: pd.DataFrame,
    *,
    strategy_col: str,
    value_col: str,
    title: str,
) -> plt.Figure:
    fig, ax = _new_figure(width=7.4, height=4.8)
    if auc_rank_df.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Strategy")
        ax.set_ylabel(value_col)
        return fig
    x = auc_rank_df[strategy_col].astype(str)
    y = auc_rank_df[f"{value_col}_mean"].astype(float)
    ax.bar(x, y)
    if f"{value_col}_std" in auc_rank_df.columns:
        yerr = auc_rank_df[f"{value_col}_std"].fillna(0.0).astype(float)
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3)
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=25)
    return fig
