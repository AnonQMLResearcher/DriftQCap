from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from driftqcap import build_profile_config, run_benchmark
from driftqcap.config import AdaptationConfig, BenchmarkRunConfig, ModelConfig, ReportingConfig, SyntheticBenchmarkConfig
from driftqcap.synthetic import generate_synthetic_benchmark


class DriftQCapExternalPersistenceTest(unittest.TestCase):
    def _small_config(self) -> BenchmarkRunConfig:
        base = build_profile_config("quick", random_seed=777)
        synthetic = SyntheticBenchmarkConfig(
            random_seed=777,
            pass_threshold=base.synthetic.pass_threshold,
            source_episodes=base.synthetic.source_episodes[:2],
            target_episodes=base.synthetic.target_episodes[:2],
        )
        model = ModelConfig(
            n_members=8,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap_fraction=0.85,
            n_jobs=1,
            random_seed=777,
        )
        adaptation = AdaptationConfig(
            budgets=(0, 4),
            random_repeats=1,
            candidate_fraction=0.45,
            ridge_alpha=1.0,
            target_upweight=5.0,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.010,
            random_seed=777,
        )
        reporting = ReportingConfig(
            save_raw_predictions=True,
            save_submission_bundle=False,
            save_model_artifacts=True,
            load_model_artifacts=False,
        )
        return BenchmarkRunConfig(
            profile_name="external-persistence-smoke",
            synthetic=synthetic,
            model=model,
            adaptation=adaptation,
            calibration=base.calibration,
            evaluation=replace(base.evaluation, bootstrap_resamples=20),
            reporting=reporting,
        )

    def test_external_summary_and_persistence(self) -> None:
        config = self._small_config()
        raw_df = generate_synthetic_benchmark(config.synthetic)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "external_eval.csv"
            pd.DataFrame(raw_df).to_csv(dataset_path, index=False)

            run_dir = tmp / "run"
            artifacts = run_benchmark(config, dataset=dataset_path, output_dir=run_dir, dataset_name="fixture_external")
            self.assertTrue((run_dir / "tables" / "external_validity_summary.csv").exists())
            self.assertFalse(artifacts.external_validity_summary.empty)
            self.assertTrue((run_dir / "artifacts" / "source_model.joblib").exists())

            reload_config = BenchmarkRunConfig(
                profile_name=config.profile_name,
                synthetic=config.synthetic,
                model=config.model,
                adaptation=config.adaptation,
                calibration=config.calibration,
                evaluation=config.evaluation,
                reporting=ReportingConfig(
                    save_raw_predictions=True,
                    save_submission_bundle=False,
                    save_model_artifacts=False,
                    load_model_artifacts=True,
                ),
            )
            rerun_dir = tmp / "run"
            loaded = run_benchmark(reload_config, dataset=dataset_path, output_dir=rerun_dir, dataset_name="fixture_external")
            self.assertFalse(loaded.external_validity_summary.empty)
            self.assertTrue((rerun_dir / "tables" / "external_validity_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
