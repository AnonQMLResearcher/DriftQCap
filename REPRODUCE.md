# Reproduce DriftQCap Submission

For the full code+dataset-separated workflow, use:

- `RUN_WORKFLOW.md`

## Environment
```bash
pip install -r pipeline_code/requirements.txt
```

## Evaluate Synthetic Split
```bash
python pipeline_code/scripts/evaluate_release_dataset.py --dataset-csv dataset_release/synthetic/driftqcap_synthetic_paper_rf_v3.csv --dataset-name synthetic_paper --output-dir outputs/release_synth
```

## Evaluate Semi-Synthetic Split
```bash
python pipeline_code/scripts/evaluate_release_dataset.py --dataset-csv dataset_release/semi_synth/semi_synth_from_snapshots_clean.csv --dataset-name semi_synth_clean --output-dir outputs/release_semi
```

## Evaluate Real-Hardware Split
```bash
python pipeline_code/scripts/evaluate_release_dataset.py --dataset-csv dataset_release/real_hardware/real_data_multibackend_all.csv --dataset-name real_hw --output-dir outputs/release_real
```
