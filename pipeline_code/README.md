# DriftQCap research suite

DriftQCap is a reproducible evaluation scaffold for **few-shot, calibrated quantum capability prediction under distribution shift**.

This package is designed for one-command generation of:

- synthetic drift benchmark data,
- source-only and few-shot adaptation baselines,
- acquisition-policy comparisons,
- calibration and interval metrics,
- paper-ready figures and CSV tables,
- a compact markdown run report and paper asset manifest.

## What is implemented

The package currently provides a **complete synthetic benchmark pipeline** that covers the core proposal requirements:

- circuit-family shift,
- depth shift,
- noise-regime shift,
- temporal drift episodes,
- source-only, target-only, pooled retraining, residual adaptation, and shuffled-label negative control,
- random, diversity, uncertainty, and uncertainty-threshold-diversity acquisition,
- post-hoc isotonic probability calibration,
- scaled conformal regression intervals,
- MAE / RMSE / rank correlation / coverage / Brier / ECE / AUROC / AUPRC metrics,
- episode-level AUC summaries and paired statistical comparisons.

## What is intentionally not claimed

This repository **does not** claim to be a faithful reproduction of the Sandia qpa-NN / QPANN architecture. Instead, it ships a strong, runnable, low-fragility benchmark stack for evaluating few-shot adaptation, calibration, and label-acquisition behavior under drift.

If you later add a graph or qpa-inspired neural backbone, the evaluation, calibration, acquisition, and reporting stack here can stay unchanged.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## One-command run

### Paper profile

```bash
python scripts/run_suite.py --profile paper --output-dir outputs/driftqcap_paper
```

### Paper-ready profile with saved source model artifacts

```bash
python scripts/run_paper_ready.py \
  --profile paper \
  --output-dir outputs/driftqcap_paper_ready \
  --save-models
```

### One-command paper-ready run

```bash
python scripts/run_paper_ready.py --profile paper --output-dir outputs/driftqcap_paper_ready
```

This wrapper runs the benchmark, exports LaTeX snippets, writes a readiness report, and builds a compact supplementary ZIP.

### Quick smoke test

```bash
python scripts/run_suite.py --profile quick --output-dir outputs/driftqcap_quick
```

### Larger rerun

```bash
python scripts/run_suite.py --profile full --output-dir outputs/driftqcap_full
```

## Expected outputs

The run directory contains:

- `tables/*.csv` — raw and aggregated metrics,
- `figures/*.png` and `figures/*.pdf` — paper-ready plots,
- `metadata/config.json` — exact experiment configuration,
- `metadata/environment.json` — environment snapshot,
- `run_report.md` — compact narrative summary,
- `paper_asset_manifest.md` — suggested paper placement for each figure.

### Where to read the main accuracy result

For the current benchmark, the main adaptation metric is `mae_auc`.

- `tables/adaptation_auc.csv` — episode-level MAE-vs-budget AUC values.
- `tables/adaptation_stats.csv` — paired comparisons such as `residual_adapter vs source_only`.
- `readiness_report.md` — high-level interpretation for paper positioning.

The standard paper command is:

```bash
python scripts/run_paper_ready.py --profile paper --output-dir outputs/driftqcap_paper_ready
```

## Creating a compact supplementary ZIP

```bash
python scripts/make_submission_bundle.py \
  --repo-root . \
  --run-dir outputs/driftqcap_paper \
  --output-zip dist/driftqcap_submission_bundle.zip
```

## External CSV support

You can supply your own dataset as long as it contains the required columns used by the benchmark.
The easiest path is to export a CSV with at least:

- `circuit_id`
- `episode_id`
- `domain`
- `shift_type`
- `family`
- `topology_kind`
- `qubit_count`
- `depth`
- `num_1q_gates`
- `num_2q_gates`
- `two_qubit_density`
- `avg_degree`
- `edge_density`
- `symmetry_score`
- `periodicity_score`
- `linearity_score`
- `idle_fraction`
- `layer_sparsity`
- `t1_us`
- `t2_us`
- `readout_error`
- `oneq_epg`
- `twoq_epg`
- `time_index`
- `error_rate`

`pass_label` will be derived automatically from the configured threshold if omitted.

Run with:

```bash
python scripts/run_suite.py --profile paper --dataset-csv /path/to/your_dataset.csv --output-dir outputs/my_run
```

To label the run and reuse fitted source artifacts:

```bash
python scripts/run_paper_ready.py \
  --profile paper \
  --dataset-csv /path/to/your_dataset.csv \
  --dataset-name archived_ibm_eval \
  --output-dir outputs/archived_ibm_eval \
  --save-models \
  --load-models
```

To validate and normalize an already circuit-level measured dataset:

```bash
python scripts/prepare_external_dataset.py \
  --input-csv /path/to/raw_or_prepared_measurements.csv \
  --output-csv data/external/archived_ibm_eval.csv
```

## Recommended paper use

For a main-track submission, use:

- `fig03_adaptation_overall_mae` as the main adaptation figure,
- `fig04_acquisition_overall_mae` as the active selection figure,
- `fig05_reliability_source_only` and `fig06_reliability_era_adapter` as calibration figures,
- `tables/adaptation_summary.csv` and `tables/adaptation_stats.csv` for the main tables.

## Testing

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Development note

The benchmark was written to be easy to extend. The most natural next addition is a graph/qpa-inspired neural predictor that plugs into the same evaluation interface as the current tree ensemble and residual adapter.

`run_paper_ready.py` already exports LaTeX tables and writes `readiness_report.md` automatically for each run.
