"""Paper-facing utilities for exporting LaTeX tables and readiness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    evidence: str
    action: str


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pick(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in df.columns]
    if not selected:
        return pd.DataFrame()
    return df.loc[:, selected].copy()


def _rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    usable = {key: value for key, value in mapping.items() if key in df.columns}
    return df.rename(columns=usable)


def _sort_if_possible(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    usable = [column for column in columns if column in df.columns]
    if not usable:
        return df.reset_index(drop=True)
    return df.sort_values(usable).reset_index(drop=True)


def _format_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.4f}"


def dataframe_to_wrapped_latex(df: pd.DataFrame, *, caption: str, label: str, column_format: str | None = None) -> str:
    if df.empty:
        return "% Table omitted because the source dataframe was empty.\n"
    body = df.to_latex(index=False, escape=False, na_rep="", float_format=_format_float, column_format=column_format)
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        body.strip(),
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ])


def export_standard_latex_tables(run_dir: str | Path, output_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_dir / "tables"

    adaptation_summary = _load_csv(tables_dir / "adaptation_summary.csv")
    acquisition_summary = _load_csv(tables_dir / "acquisition_summary.csv")
    adaptation_auc = _load_csv(tables_dir / "adaptation_auc.csv")
    acquisition_auc = _load_csv(tables_dir / "acquisition_auc.csv")
    adaptation_stats = _load_csv(tables_dir / "adaptation_stats.csv")
    acquisition_stats = _load_csv(tables_dir / "acquisition_stats.csv")
    acquisition_budget_stats = _load_csv(tables_dir / "acquisition_budget_stats.csv")
    dataset_summary = _load_csv(tables_dir / "dataset_episode_summary.csv")
    external_validity_summary = _load_csv(tables_dir / "external_validity_summary.csv")

    specs: list[tuple[str, pd.DataFrame, str, str, str | None]] = []

    if not adaptation_summary.empty:
        overall = adaptation_summary[adaptation_summary["shift_type"] == "overall"].copy()
        overall = _sort_if_possible(overall, ["strategy", "budget"])
        overall = _pick(overall, ["strategy", "budget", "mae_mean", "mae_std", "rmse_mean", "ece_calibrated_mean", "coverage_90_mean", "width_90_mean", "auroc_calibrated_mean", "n_runs"])
        overall = _rename(overall, {"strategy": "Strategy", "budget": "Budget", "mae_mean": "MAE", "mae_std": "MAE SD", "rmse_mean": "RMSE", "ece_calibrated_mean": "Cal. ECE", "coverage_90_mean": "90\\% Cov.", "width_90_mean": "90\\% Width", "auroc_calibrated_mean": "Cal. AUROC", "n_runs": "Runs"})
        specs.append(("adaptation_overall_summary.tex", overall, "Overall adaptation summary across budgets. Lower MAE and lower calibrated ECE are better.", "tab:driftqcap_adaptation_overall", None))

    if not acquisition_summary.empty:
        overall = acquisition_summary[acquisition_summary["shift_type"] == "overall"].copy()
        overall = _sort_if_possible(overall, ["acquisition_strategy", "budget"])
        overall = _pick(overall, ["acquisition_strategy", "budget", "mae_mean", "mae_std", "ece_calibrated_mean", "auroc_calibrated_mean", "n_runs"])
        overall = _rename(overall, {"acquisition_strategy": "Acquisition", "budget": "Budget", "mae_mean": "MAE", "mae_std": "MAE SD", "ece_calibrated_mean": "Cal. ECE", "auroc_calibrated_mean": "Cal. AUROC", "n_runs": "Runs"})
        specs.append(("acquisition_overall_summary.tex", overall, "Overall acquisition summary across budgets. Lower MAE and lower calibrated ECE are better.", "tab:driftqcap_acquisition_overall", None))

    if not adaptation_auc.empty:
        overall = adaptation_auc[adaptation_auc["shift_type"] == "overall"].copy()
        ranking = overall.groupby("strategy", dropna=False)["mae_auc"].agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=True)
        ranking = _rename(ranking, {"strategy": "Strategy", "mean": "MAE AUC", "std": "MAE AUC SD", "count": "Episodes"})
        specs.append(("adaptation_auc_ranking.tex", ranking, "Episode-level ranking of adaptation strategies by area under the MAE-versus-budget curve.", "tab:driftqcap_adaptation_auc_ranking", None))

    if not acquisition_auc.empty:
        overall = acquisition_auc[acquisition_auc["shift_type"] == "overall"].copy()
        mae_ranking = overall.groupby("acquisition_strategy", dropna=False)["mae_auc"].agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=True)
        mae_ranking = _rename(mae_ranking, {"acquisition_strategy": "Acquisition", "mean": "MAE AUC", "std": "MAE AUC SD", "count": "Episodes"})
        specs.append(("acquisition_mae_gate_table.tex", mae_ranking, "Official acquisition gate ranking by MAE AUC (lower is better).", "tab:driftqcap_acquisition_mae_gate", None))
        specs.append(("acquisition_auc_ranking.tex", mae_ranking, "Episode-level ranking of acquisition strategies by area under the MAE-versus-budget curve.", "tab:driftqcap_acquisition_auc_ranking", None))
        ece_ranking = overall.groupby("acquisition_strategy", dropna=False)["ece_calibrated_auc"].agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=True)
        ece_ranking = _rename(ece_ranking, {"acquisition_strategy": "Acquisition", "mean": "Cal. ECE AUC", "std": "Cal. ECE AUC SD", "count": "Episodes"})
        specs.append(("acquisition_ece_diagnostic_table.tex", ece_ranking, "Diagnostic-only acquisition ranking by calibrated ECE AUC (lower is better). This table does not define the selection gate.", "tab:driftqcap_acquisition_ece_diagnostic", None))

    if not adaptation_stats.empty:
        overall = adaptation_stats[adaptation_stats["shift_scope"] == "overall"].copy()
        overall = _pick(overall, ["metric", "comparison", "mean_delta", "ci_low", "ci_high", "p_value", "p_holm", "effect_size", "n_pairs"])
        overall = _rename(overall, {"metric": "Metric", "comparison": "Comparison", "mean_delta": "Mean $\\Delta$", "ci_low": "CI Low", "ci_high": "CI High", "p_value": "$p$", "p_holm": "$p_{Holm}$", "effect_size": "Effect", "n_pairs": "Pairs"})
        specs.append(("adaptation_pairwise_tests.tex", overall, "Overall paired adaptation comparisons. Negative deltas favor the first strategy for error-like metrics.", "tab:driftqcap_adaptation_pairwise", None))

    if not acquisition_stats.empty:
        overall = acquisition_stats[acquisition_stats["shift_scope"] == "overall"].copy()
        overall = _pick(overall, ["metric", "comparison", "mean_delta", "ci_low", "ci_high", "p_value", "p_holm", "effect_size", "n_pairs"])
        overall = _rename(overall, {"metric": "Metric", "comparison": "Comparison", "mean_delta": "Mean $\\Delta$", "ci_low": "CI Low", "ci_high": "CI High", "p_value": "$p$", "p_holm": "$p_{Holm}$", "effect_size": "Effect", "n_pairs": "Pairs"})
        specs.append(("acquisition_pairwise_tests.tex", overall, "Overall paired acquisition comparisons. Negative deltas favor the first strategy for error-like metrics.", "tab:driftqcap_acquisition_pairwise", None))

    if not acquisition_budget_stats.empty:
        overall = acquisition_budget_stats[acquisition_budget_stats["shift_scope"] == "overall"].copy()
        overall = _sort_if_possible(overall, ["budget", "comparison", "metric"])
        overall = _pick(overall, ["budget", "metric", "comparison", "mean_delta", "ci_low", "ci_high", "p_value", "p_holm", "effect_size", "n_pairs"])
        overall = _rename(
            overall,
            {
                "budget": "Budget",
                "metric": "Metric",
                "comparison": "Comparison",
                "mean_delta": "Mean $\\Delta$",
                "ci_low": "CI Low",
                "ci_high": "CI High",
                "p_value": "$p$",
                "p_holm": "$p_{Holm}$",
                "effect_size": "Effect",
                "n_pairs": "Pairs",
            },
        )
        specs.append(
            (
                "acquisition_budget_pairwise_tests.tex",
                overall,
                "Per-budget paired acquisition comparisons. Negative deltas favor the first strategy for error-like metrics.",
                "tab:driftqcap_acquisition_budget_pairwise",
                None,
            )
        )

    if not dataset_summary.empty:
        dataset_summary = _pick(dataset_summary, ["episode_id", "domain", "shift_type", "n_rows", "mean_error_rate", "mean_pass_rate"])
        dataset_summary = _rename(dataset_summary, {"episode_id": "Episode", "domain": "Domain", "shift_type": "Shift", "n_rows": "Rows", "mean_error_rate": "Mean Error", "mean_pass_rate": "Mean Pass"})
        specs.append(("dataset_episode_summary.tex", dataset_summary, "Per-episode benchmark summary used for paper tables and appendix traceability.", "tab:driftqcap_dataset_episode_summary", None))

    if not external_validity_summary.empty:
        external_validity_summary = _pick(external_validity_summary, ["dataset_name", "strategy", "budget", "mae_mean", "ece_calibrated_mean", "auroc_calibrated_mean", "n_runs"])
        external_validity_summary = _rename(
            external_validity_summary,
            {
                "dataset_name": "Dataset",
                "strategy": "Strategy",
                "budget": "Budget",
                "mae_mean": "MAE",
                "ece_calibrated_mean": "Cal. ECE",
                "auroc_calibrated_mean": "Cal. AUROC",
                "n_runs": "Runs",
            },
        )
        specs.append(
            (
                "external_validity_summary.tex",
                external_validity_summary,
                "External-validity evaluation summary on measured or archived circuit data.",
                "tab:driftqcap_external_validity",
                None,
            )
        )

    exported: list[Path] = []
    manifest_lines = ["# LaTeX table export manifest", "", f"Run directory: {run_dir}", f"Output directory: {output_dir}", "", "Generated files:"]
    for filename, frame, caption, label, column_format in specs:
        text = dataframe_to_wrapped_latex(frame, caption=caption, label=label, column_format=column_format)
        path = output_dir / filename
        path.write_text(text, encoding="utf-8")
        exported.append(path)
        manifest_lines.append(f"- {filename}: {caption}")
    (output_dir / "README.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    exported.append(output_dir / "README.md")
    return exported


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_row(df: pd.DataFrame, comparison: str, metric: str, shift_scope: str = "overall") -> pd.Series | None:
    if df.empty:
        return None
    subset = df[(df["comparison"] == comparison) & (df["metric"] == metric) & (df["shift_scope"] == shift_scope)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _status_from_delta(*, mean_delta: float | None, ci_low: float | None, ci_high: float | None, lower_is_better: bool = True) -> str:
    if mean_delta is None:
        return "WARN"
    if lower_is_better:
        if ci_high is not None and ci_high < 0:
            return "PASS"
        if mean_delta < 0:
            return "WARN"
        return "FAIL"
    if ci_low is not None and ci_low > 0:
        return "PASS"
    if mean_delta > 0:
        return "WARN"
    return "FAIL"


def _readiness_gates(run_dir: Path) -> list[GateResult]:
    tables_dir = run_dir / "tables"
    adaptation_stats = _load_csv(tables_dir / "adaptation_stats.csv")
    acquisition_stats = _load_csv(tables_dir / "acquisition_stats.csv")
    adaptation_auc = _load_csv(tables_dir / "adaptation_auc.csv")
    adaptation_summary = _load_csv(tables_dir / "adaptation_summary.csv")
    dataset_episode_summary = _load_csv(tables_dir / "dataset_episode_summary.csv")
    config = _load_config(run_dir / "metadata" / "config.json")

    gates: list[GateResult] = []
    target_shift_types = sorted({str(item) for item in adaptation_summary.get("shift_type", pd.Series(dtype=object)).dropna().tolist() if str(item) != "overall"})
    gates.append(GateResult("Benchmark coverage", "PASS" if len(target_shift_types) >= 3 else "FAIL", f"Found target shift families: {', '.join(target_shift_types) if target_shift_types else 'none'}.", "Keep at least three mandatory shift families in every reported run."))

    total_target_episodes = int((dataset_episode_summary.get("domain", pd.Series(dtype=object)) == "target").sum()) if not dataset_episode_summary.empty else 0
    repeats = int(config.get("adaptation", {}).get("random_repeats", 0) or 0)
    power_status = "PASS" if total_target_episodes >= 12 and repeats >= 2 else ("WARN" if total_target_episodes >= 4 else "FAIL")
    gates.append(GateResult("Statistical depth", power_status, f"Target episodes: {total_target_episodes}; random repeats: {repeats}.", "Paper claims should be based on the paper or full profile, not the quick profile alone."))

    row = _comparison_row(adaptation_stats, "era_adapter vs source_only", "mae_auc")
    if row is None:
        gates.append(GateResult("ERA improvement over source-only", "FAIL", "No overall era_adapter vs source_only MAE AUC comparison was found.", "Run the full benchmark before making method claims."))
    else:
        status = _status_from_delta(mean_delta=row.get("mean_delta"), ci_low=row.get("ci_low"), ci_high=row.get("ci_high"))
        gates.append(GateResult("ERA improvement over source-only", status, f"Mean delta = {_format_float(row.get('mean_delta'))}, CI = [{_format_float(row.get('ci_low'))}, {_format_float(row.get('ci_high'))}], p = {_format_float(row.get('p_value'))}.", "Only call the method ready if the interval excludes zero or the larger run reproduces the same ordering cleanly."))

    row = _comparison_row(adaptation_stats, "era_adapter vs pooled_retrain", "mae_auc")
    if row is None:
        gates.append(GateResult("ERA improvement over pooled retraining", "WARN", "No overall era_adapter vs pooled_retrain MAE AUC comparison was found.", "Keep pooled retraining in every main table."))
    else:
        status = _status_from_delta(mean_delta=row.get("mean_delta"), ci_low=row.get("ci_low"), ci_high=row.get("ci_high"))
        gates.append(GateResult("ERA improvement over pooled retraining", status, f"Mean delta = {_format_float(row.get('mean_delta'))}, CI = [{_format_float(row.get('ci_low'))}, {_format_float(row.get('ci_high'))}], p = {_format_float(row.get('p_value'))}.", "If this stays only weakly better, phrase the method as a lightweight adapter rather than a dominant new architecture."))

    robustness_pairs = [
        _comparison_row(adaptation_stats, "era_adapter vs source_only", "mae_auc"),
        _comparison_row(adaptation_stats, "era_adapter vs pooled_retrain", "mae_auc"),
        _comparison_row(adaptation_stats, "era_adapter vs target_only", "mae_auc"),
    ]
    robustness_ok = all(
        row is not None and (row.get("ci_high") is not None and float(row.get("ci_high")) < 0.0)
        for row in robustness_pairs
    )
    robustness_detail = []
    for label, row in zip(["source_only", "pooled_retrain", "target_only"], robustness_pairs):
        if row is None:
            robustness_detail.append(f"{label}: missing")
        else:
            robustness_detail.append(f"{label}: delta={_format_float(row.get('mean_delta'))}, ci_hi={_format_float(row.get('ci_high'))}")
    gates.append(
        GateResult(
            "ERA main-method robustness",
            "PASS" if robustness_ok else "WARN",
            "; ".join(robustness_detail) + ".",
            "The main method should cleanly beat source-only, pooled retraining, and target-only before stronger ranking claims are made.",
        )
    )

    residual_row = _comparison_row(adaptation_stats, "era_adapter vs residual_adapter", "mae_auc")
    if residual_row is None:
        gates.append(GateResult("ERA vs residual", "WARN", "No overall era_adapter vs residual_adapter comparison was found.", "Keep the simpler residual adapter as an ablation baseline."))
    else:
        mean_delta = residual_row.get("mean_delta")
        ci_low = residual_row.get("ci_low")
        ci_high = residual_row.get("ci_high")
        if ci_high is not None and float(ci_high) < 0.0:
            status = "PASS"
        elif ci_low is not None and float(ci_low) > 0.0:
            status = "FAIL"
        else:
            status = "WARN"
        gates.append(
            GateResult(
                "ERA vs residual",
                status,
                f"Mean delta = {_format_float(mean_delta)}, CI = [{_format_float(ci_low)}, {_format_float(ci_high)}], p = {_format_float(residual_row.get('p_value'))}. Negative favors era_adapter.",
                "If the merged method does not improve on the simpler residual adapter, keep the simpler variant prominent in the ablations.",
            )
        )

    ewc_row = _comparison_row(adaptation_stats, "era_adapter vs ewc_adapter", "mae_auc")
    if ewc_row is None:
        gates.append(GateResult("ERA vs EWC", "WARN", "No overall era_adapter vs ewc_adapter comparison was found.", "Keep EWC in the benchmark and surface the comparison explicitly."))
    else:
        mean_delta = ewc_row.get("mean_delta")
        ci_low = ewc_row.get("ci_low")
        ci_high = ewc_row.get("ci_high")
        if ci_high is not None and float(ci_high) < 0.0:
            status = "PASS"
        elif ci_low is not None and float(ci_low) > 0.0:
            status = "FAIL"
        else:
            status = "WARN"
        gates.append(
            GateResult(
                "ERA vs EWC",
                status,
                f"Mean delta = {_format_float(mean_delta)}, CI = [{_format_float(ci_low)}, {_format_float(ci_high)}], p = {_format_float(ewc_row.get('p_value'))}. Negative favors era_adapter.",
                "If EWC still wins after fair tuning, keep that comparison explicit rather than hiding the tie.",
            )
        )

    row = _comparison_row(adaptation_stats, "era_adapter vs source_only", "ece_calibrated_auc")
    if row is None:
        gates.append(GateResult("Calibration improvement", "FAIL", "No overall calibrated-ECE comparison for era_adapter vs source_only was found.", "Do not claim calibrated decision support without this analysis."))
    else:
        status = _status_from_delta(mean_delta=row.get("mean_delta"), ci_low=row.get("ci_low"), ci_high=row.get("ci_high"))
        gates.append(GateResult("Calibration improvement", status, f"Calibrated ECE delta = {_format_float(row.get('mean_delta'))}, CI = [{_format_float(row.get('ci_low'))}, {_format_float(row.get('ci_high'))}], p = {_format_float(row.get('p_value'))}. Negative is better.", "If this gate fails, narrow the claim or add target-aware recalibration experiments before submission."))

    interval_row = adaptation_summary[
        (adaptation_summary.get("shift_type", pd.Series(dtype=object)) == "overall")
        & (adaptation_summary.get("strategy", pd.Series(dtype=object)) == "era_adapter")
        & (adaptation_summary.get("budget", pd.Series(dtype=float)) > 0)
    ].sort_values("budget")
    if interval_row.empty or "coverage_80_mean" not in interval_row.columns:
        gates.append(GateResult("Interval sharpness", "WARN", "No era_adapter interval summary rows were found.", "Add interval metrics before making calibrated-risk claims."))
    else:
        last = interval_row.iloc[-1]
        coverage = float(last.get("coverage_80_mean", float("nan")))
        width = float(last.get("width_80_mean", float("nan"))) if "width_80_mean" in last.index else float("nan")
        tol = float(config.get("evaluation", {}).get("coverage_tolerance", 0.03) or 0.03)
        if 0.80 - tol <= coverage <= 0.80 + tol:
            status = "PASS"
        elif coverage > 0.90:
            status = "FAIL"
        else:
            status = "WARN"
        gates.append(
            GateResult(
                "Interval sharpness",
                status,
                f"Era_adapter overall coverage_80 = {_format_float(coverage)}, width_80 = {_format_float(width)} at budget {int(last.get('budget', 0))}.",
                "Intervals should be close to nominal coverage without inflating width; near-100% coverage means they are too conservative.",
            )
        )

    claim_strategy = str(config.get("adaptation", {}).get("strategy_for_claim", "diversity"))
    acq_row = _comparison_row(acquisition_stats, f"{claim_strategy} vs random", "mae_auc")
    div_row = None if claim_strategy == "diversity" else _comparison_row(acquisition_stats, f"{claim_strategy} vs diversity", "mae_auc")
    if acq_row is None or (claim_strategy != "diversity" and div_row is None):
        gates.append(
            GateResult(
                "Acquisition advantage",
                "WARN",
                f"Required acquisition comparisons for strategy '{claim_strategy}' were not both present.",
                "Do not center the paper on active selection until the comparison matrix is complete.",
            )
        )
    else:
        better_than_random = _status_from_delta(mean_delta=acq_row.get("mean_delta"), ci_low=acq_row.get("ci_low"), ci_high=acq_row.get("ci_high"))
        not_worse_than_diversity = "PASS" if claim_strategy == "diversity" else ("PASS" if float(div_row.get("mean_delta", 0.0) or 0.0) <= 0 else "FAIL")
        status = "PASS" if better_than_random == "PASS" and not_worse_than_diversity == "PASS" else ("WARN" if better_than_random in {"PASS", "WARN"} else "FAIL")
        evidence_tail = (
            "by definition, not-worse-than-diversity holds."
            if claim_strategy == "diversity"
            else f"{claim_strategy} vs diversity mean delta = {_format_float(div_row.get('mean_delta'))}."
        )
        gates.append(
            GateResult(
                "Acquisition advantage",
                status,
                f"{claim_strategy} vs random mean delta = {_format_float(acq_row.get('mean_delta'))}; {evidence_tail}",
                "If this gate is not PASS, demote active selection to a secondary or appendix result.",
            )
        )

    residual_rank = None
    shuffled_rank = None
    if not adaptation_auc.empty:
        overall_auc = adaptation_auc[adaptation_auc["shift_type"] == "overall"].copy()
        if not overall_auc.empty:
            ranking = overall_auc.groupby("strategy", dropna=False)["mae_auc"].mean().sort_values(ascending=True)
            if "era_adapter" in ranking.index:
                residual_rank = int(ranking.index.get_loc("era_adapter")) + 1
            if "shuffled_label_adapter" in ranking.index:
                shuffled_rank = int(ranking.index.get_loc("shuffled_label_adapter")) + 1
    gates.append(GateResult("Negative-control sanity check", "PASS" if residual_rank is not None and shuffled_rank is not None and residual_rank < shuffled_rank else "WARN", f"ERA rank = {residual_rank if residual_rank is not None else 'n/a'}; shuffled-label rank = {shuffled_rank if shuffled_rank is not None else 'n/a'}.", "Keep the shuffled-label adapter in the supplement to rule out leakage and spurious gains."))

    external_path = tables_dir / "external_validity_summary.csv"
    gates.append(GateResult("External validity", "PASS" if external_path.exists() else "FAIL", "An external_validity_summary.csv file was found." if external_path.exists() else "No public-data or archived-hardware summary file was found in the run directory.", "At minimum, add one public-data or archived-hardware transfer evaluation before submission."))
    return gates


def build_readiness_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    gates = _readiness_gates(run_dir)
    pass_count = sum(g.status == "PASS" for g in gates)
    warn_count = sum(g.status == "WARN" for g in gates)
    fail_count = sum(g.status == "FAIL" for g in gates)
    calibration_gate = next((g for g in gates if g.name == "Calibration improvement"), None)
    external_gate = next((g for g in gates if g.name == "External validity"), None)
    mae_gate = next((g for g in gates if g.name == "MAE improvement over source-only"), None)
    acquisition_gate = next((g for g in gates if g.name == "Acquisition advantage"), None)

    if mae_gate and mae_gate.status == "PASS" and calibration_gate and calibration_gate.status == "PASS" and external_gate and external_gate.status == "PASS":
        track_recommendation = "Main track is scientifically defensible on the current evidence, provided the writing stays method-centric and the larger run reproduces the same ordering."
    elif mae_gate and mae_gate.status in {"PASS", "WARN"}:
        track_recommendation = "Lean benchmark-first and target the NeurIPS Evaluations & Datasets track unless stronger method-level calibration and external-validity evidence is added immediately."
    else:
        track_recommendation = "Do not lock a method-centric main-track story yet. The safer path is to strengthen the benchmark package first and treat the method as a secondary contribution."

    claim_lines: list[str] = []
    if calibration_gate and calibration_gate.status != "PASS":
        claim_lines.append("Do not currently describe the method as uncertainty-calibrated in the strong sense.")
    if acquisition_gate and acquisition_gate.status != "PASS":
        claim_lines.append("Do not currently make active selection a headline claim.")
    if external_gate and external_gate.status != "PASS":
        claim_lines.append("State explicitly that the current run is synthetic-only until the public-data check is completed.")
    if not claim_lines:
        claim_lines.append("The current run clears the major readiness gates.")

    lines = ["# DriftQCap readiness report", "", f"Run directory: `{run_dir}`", "", f"- PASS gates: {pass_count}", f"- WARN gates: {warn_count}", f"- FAIL gates: {fail_count}", "", "## Recommended track stance", "", track_recommendation, "", "## Claim-language recommendation", ""]
    lines.extend([f"- {item}" for item in claim_lines])
    lines.extend(["", "## Gate table", "", "| Gate | Status | Evidence | Action |", "| --- | --- | --- | --- |"])
    for gate in gates:
        lines.append(f"| {gate.name} | {gate.status} | {gate.evidence} | {gate.action} |")
    lines.extend(["", "## Interpretation", "", "A PASS means the current run already supports the relevant paper claim. A WARN means the ordering is promising but still too weak, too small, or too unstable to headline confidently. A FAIL means the corresponding claim should be narrowed, deferred, or repaired before the final paper is framed around it.", "", "For this project, the three most important gates are: (i) MAE AUC improvement over source-only, (ii) calibration improvement rather than calibration regression, and (iii) one public-data or archived-hardware external-validity check."])
    return "\n".join(lines) + "\n"
