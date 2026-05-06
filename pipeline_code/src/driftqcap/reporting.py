"""Reporting helpers for saved benchmark artifacts."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_MAX_PREVIEW_ROWS = 12


def summarize_dataset(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create dataset summary tables used in the run report."""
    domain_summary = (
        dataset.groupby(["domain", "shift_type"], dropna=False)
        .agg(
            n_rows=("circuit_id", "count"),
            n_episodes=("episode_id", "nunique"),
            mean_error_rate=("error_rate", "mean"),
            mean_pass_rate=("pass_label", "mean"),
        )
        .reset_index()
        .sort_values(["domain", "shift_type"])
    )
    family_summary = (
        dataset.groupby(["domain", "family"], dropna=False)
        .agg(n_rows=("circuit_id", "count"), mean_error_rate=("error_rate", "mean"), mean_pass_rate=("pass_label", "mean"))
        .reset_index()
        .sort_values(["domain", "family"])
    )
    episode_summary = (
        dataset.groupby(["episode_id", "domain", "shift_type"], dropna=False)
        .agg(n_rows=("circuit_id", "count"), mean_error_rate=("error_rate", "mean"), mean_pass_rate=("pass_label", "mean"))
        .reset_index()
        .sort_values(["domain", "shift_type", "episode_id"])
    )
    return {
        "dataset_domain_summary": domain_summary,
        "dataset_family_summary": family_summary,
        "dataset_episode_summary": episode_summary,
    }


def _round_for_markdown(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, *, max_rows: int = _MAX_PREVIEW_ROWS) -> str:
    if df.empty:
        return "(empty)\n"
    preview = df.head(max_rows).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(_round_for_markdown(row[col]) for col in columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\nPreview truncated to {max_rows} rows out of {len(df)} total rows.\n")
    return "\n".join(lines) + "\n"


def rank_strategies(auc_df: pd.DataFrame, *, strategy_col: str, value_col: str) -> pd.DataFrame:
    if auc_df.empty or strategy_col not in auc_df.columns or value_col not in auc_df.columns:
        return pd.DataFrame(columns=[strategy_col, value_col])
    out = (
        auc_df.groupby(strategy_col, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    return out.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n_episodes"})


def select_overall_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "shift_type" not in summary_df.columns:
        return summary_df.copy()
    return summary_df[summary_df["shift_type"] == "overall"].copy()


def _safe_topline(rank_df: pd.DataFrame, *, strategy_col: str, value_col: str) -> str:
    if rank_df.empty:
        return "No ranking available."
    row = rank_df.iloc[0]
    return f"{row[strategy_col]} (mean {value_col} = {float(row[f'{value_col}_mean']):.4f} over {int(row['n_episodes'])} episodes)"


def write_run_report(
    *,
    output_dir: str | Path,
    config_dict: dict[str, object],
    dataset: pd.DataFrame,
    dataset_tables: dict[str, pd.DataFrame],
    source_results: pd.DataFrame,
    adaptation_summary: pd.DataFrame,
    acquisition_summary: pd.DataFrame,
    adaptation_auc: pd.DataFrame,
    acquisition_auc: pd.DataFrame,
    adaptation_stats: pd.DataFrame,
    acquisition_stats: pd.DataFrame,
    acquisition_weight_selection_summary: pd.DataFrame,
    adapted_probability_diagnostic: pd.DataFrame,
    external_validity_summary: pd.DataFrame,
    model_artifact_paths: dict[str, str],
    figure_paths: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adaptation_rank = rank_strategies(select_overall_rows(adaptation_auc), strategy_col="strategy", value_col="mae_auc")
    acquisition_rank = rank_strategies(select_overall_rows(acquisition_auc), strategy_col="acquisition_strategy", value_col="mae_auc")

    lines: list[str] = []
    lines.append("# DriftQCap run report\n")
    lines.append("## Topline\n")
    lines.append(f"- Rows in dataset: {len(dataset)}")
    lines.append(f"- Source rows: {int((dataset['domain'] == 'source').sum())}")
    lines.append(f"- Target rows: {int((dataset['domain'] == 'target').sum())}")
    lines.append(f"- Source episodes: {dataset.loc[dataset['domain'] == 'source', 'episode_id'].nunique()}")
    lines.append(f"- Target episodes: {dataset.loc[dataset['domain'] == 'target', 'episode_id'].nunique()}")
    lines.append(f"- Best adaptation strategy by MAE AUC: {_safe_topline(adaptation_rank, strategy_col='strategy', value_col='mae_auc')}")
    lines.append(f"- Best acquisition strategy by MAE AUC: {_safe_topline(acquisition_rank, strategy_col='acquisition_strategy', value_col='mae_auc')}\n")

    if model_artifact_paths:
        lines.append("## Persisted artifacts\n")
        for key, value in model_artifact_paths.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## Configuration preview\n")
    lines.append("```json")
    lines.append(json.dumps(config_dict, indent=2, sort_keys=True))
    lines.append("```\n")

    lines.append("## Dataset summary\n")
    for name, table in dataset_tables.items():
        lines.append(f"### {name}\n")
        lines.append(dataframe_to_markdown(table))

    lines.append("## Source-domain results\n")
    lines.append(dataframe_to_markdown(source_results))

    lines.append("## Adaptation summary (overall rows)\n")
    lines.append(dataframe_to_markdown(select_overall_rows(adaptation_summary)))

    lines.append("## Acquisition summary (overall rows)\n")
    lines.append(dataframe_to_markdown(select_overall_rows(acquisition_summary)))

    lines.append("## Adaptation episode-level AUC ranking\n")
    lines.append(dataframe_to_markdown(adaptation_rank))

    lines.append("## Acquisition episode-level AUC ranking\n")
    lines.append(dataframe_to_markdown(acquisition_rank))

    lines.append("## Adaptation paired comparisons\n")
    lines.append(dataframe_to_markdown(adaptation_stats))

    lines.append("## Acquisition paired comparisons\n")
    lines.append(dataframe_to_markdown(acquisition_stats))

    if not acquisition_weight_selection_summary.empty:
        lines.append("## Acquisition weight selection summary\n")
        lines.append(dataframe_to_markdown(acquisition_weight_selection_summary))

    if not adapted_probability_diagnostic.empty:
        lines.append("## Adapted probability calibration diagnostic\n")
        lines.append(dataframe_to_markdown(adapted_probability_diagnostic))

    if not external_validity_summary.empty:
        lines.append("## External validity summary\n")
        lines.append(dataframe_to_markdown(external_validity_summary))

    lines.append("## Generated figures\n")
    for key, value in figure_paths.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    (output_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_paper_asset_manifest(*, output_dir: str | Path, figure_paths: dict[str, str]) -> None:
    output_dir = Path(output_dir)
    lines = [
        "# Paper asset manifest\n",
        "Use these paths when assembling the draft. Suggested placements are intentionally conservative.\n",
        "| Asset key | Suggested use in paper | Path |",
        "| --- | --- | --- |",
    ]
    suggestions = {
        "fig01_dataset_error_distribution": "Background / benchmark difficulty figure",
        "fig02_zero_shot_shift_mae": "Benchmark difficulty and shift-family introduction",
        "fig03_adaptation_overall_mae": "Main adaptation figure",
        "fig04_acquisition_overall_mae": "Main active-selection figure",
        "fig05_reliability_source_only": "Calibration figure",
        "fig06_reliability_era_adapter": "Calibration figure",
        "fig07_interval_coverage_era_adapter": "Interval calibration appendix/main text",
        "fig08_adaptation_auc_ranking": "Ablation / summary figure",
        "fig09_acquisition_auc_ranking": "Ablation / summary figure",
    }
    for key, path in figure_paths.items():
        suggestion = suggestions.get(key, "Appendix / supplementary")
        lines.append(f"| {key} | {suggestion} | {path} |")
    (output_dir / "paper_asset_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_tables(*, output_dir: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
