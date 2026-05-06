# DriftQCap Run Workflow (Code and Dataset Separate)

This is the exact workflow for users who download code and dataset separately.

## 1) Folder Layout

Use this structure (paths can differ; commands below use variables):

```text
<CODE_ROOT>/DriftQCap_submission/pipeline_code
<DATA_ROOT>/dataset_release/
  synthetic/driftqcap_synthetic_paper_rf_v3.csv
  semi_synth/semi_synth_from_snapshots_clean.csv
  real_hardware/real_data_multibackend_all.csv
```

## 2) Environment Setup

```bash
cd <CODE_ROOT>/DriftQCap_submission/pipeline_code
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For `qpa_nn` runs, also install optional graph deps:

```bash
pip install "torch>=2.2,<3" "torch-geometric>=2.5,<3"
```

## 3) Set Paths

```bash
export CODE_ROOT=<CODE_ROOT>/DriftQCap_submission/pipeline_code
export DATA_ROOT=<DATA_ROOT>/dataset_release
```

## 4) Evaluate Release Datasets

### Synthetic

```bash
cd "$CODE_ROOT"
python scripts/evaluate_release_dataset.py \
  --dataset-csv "$DATA_ROOT/synthetic/driftqcap_synthetic_paper_rf_v3.csv" \
  --dataset-name synthetic_paper \
  --output-dir outputs/release_synth
```

### Semi-synthetic

```bash
cd "$CODE_ROOT"
python scripts/evaluate_release_dataset.py \
  --dataset-csv "$DATA_ROOT/semi_synth/semi_synth_from_snapshots_clean.csv" \
  --dataset-name semi_synth_clean \
  --output-dir outputs/release_semi
```

### Real hardware

```bash
cd "$CODE_ROOT"
python scripts/evaluate_release_dataset.py \
  --dataset-csv "$DATA_ROOT/real_hardware/real_data_multibackend_all.csv" \
  --dataset-name real_hw \
  --output-dir outputs/release_real
```

## 5) Full Pipeline Run (Synthetic Generator)

No external dataset path needed:

```bash
cd "$CODE_ROOT"
python scripts/run_paper_ready.py \
  --profile paper \
  --output-dir outputs/driftqcap_paper_ready
```

## 6) QPA-NN Paper Run (Optional)

```bash
cd "$CODE_ROOT"
python scripts/run_paper_ready.py \
  --profile paper \
  --base-estimator qpa_nn \
  --qpa-mode slow \
  --random-repeats 5 \
  --output-dir outputs/qpa_nn_paper_slow_r5
```

## 7) Output Check

Each run directory should contain:

```text
tables/
figures/
metadata/config.json
run_report.md
readiness_report.md
```

