# DriftQCap Dataset Release

This folder contains the frozen benchmark CSV splits for synthetic, semi-synthetic, and real-hardware regimes.

Evaluate any split with:
```bash
python pipeline_code/scripts/evaluate_release_dataset.py --dataset-csv <path_to_csv> --dataset-name <name> --output-dir outputs/<run_name>
```
