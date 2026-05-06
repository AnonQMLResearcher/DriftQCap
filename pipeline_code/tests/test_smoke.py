from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap import build_profile_config, run_benchmark
from driftqcap.config import AdaptationConfig, BenchmarkRunConfig, ModelConfig, ReportingConfig, SyntheticBenchmarkConfig


class DriftQCapSmokeTest(unittest.TestCase):
    def test_small_benchmark_run(self) -> None:
        base = build_profile_config("quick", random_seed=123)
        synthetic = SyntheticBenchmarkConfig(
            random_seed=123,
            pass_threshold=base.synthetic.pass_threshold,
            source_episodes=base.synthetic.source_episodes[:2],
            target_episodes=base.synthetic.target_episodes[:3],
        )
        model = ModelConfig(
            n_members=12,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap_fraction=0.85,
            n_jobs=1,
            random_seed=123,
        )
        adaptation = AdaptationConfig(
            budgets=(0, 4, 8),
            random_repeats=1,
            candidate_fraction=0.45,
            ridge_alpha=1.0,
            enable_hybrid_shift_policy=True,
            target_upweight=5.0,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.010,
            random_seed=123,
        )
        reporting = ReportingConfig(save_raw_predictions=True, save_submission_bundle=False)
        config = BenchmarkRunConfig(
            profile_name="smoke",
            synthetic=synthetic,
            model=model,
            adaptation=adaptation,
            calibration=base.calibration,
            evaluation=replace(base.evaluation, bootstrap_resamples=50),
            reporting=reporting,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            artifacts = run_benchmark(config, output_dir=out_dir)
            self.assertGreater(len(artifacts.dataset), 0)
            self.assertGreater(len(artifacts.adaptation_results), 0)
            self.assertGreater(len(artifacts.acquisition_results), 0)
            self.assertIn("hybrid_shift_aware", set(artifacts.acquisition_results.get("acquisition_strategy", [])))
            self.assertIn("entropy_threshold_diversity", set(artifacts.acquisition_results.get("acquisition_strategy", [])))
            self.assertIn("ewc_adapter", set(artifacts.adaptation_results.get("strategy", [])))
            self.assertIn("era_adapter", set(artifacts.adaptation_results.get("strategy", [])))
            self.assertIn("mean_shift_adapter", set(artifacts.adaptation_results.get("strategy", [])))
            self.assertIn("prob_calib_mode", artifacts.raw_predictions.columns)
            self.assertIn("prob_calib_model", artifacts.raw_predictions.columns)
            self.assertIn("prob_calib_weight", artifacts.raw_predictions.columns)
            self.assertIn("selected_alpha", artifacts.raw_predictions.columns)
            self.assertIn("selected_ewc_lambda", artifacts.raw_predictions.columns)
            self.assertIn("std_temperature", artifacts.raw_predictions.columns)
            self.assertIn("winkler_80", artifacts.adaptation_results.columns)
            self.assertIn("coverage_80_stratum_0", artifacts.adaptation_results.columns)
            self.assertGreater(len(artifacts.adapted_probability_diagnostic), 0)
            self.assertGreater(len(artifacts.acquisition_weight_selection_summary), 0)
            self.assertTrue((out_dir / "tables" / "adapted_probability_calibration_diagnostic.csv").exists())
            self.assertTrue((out_dir / "tables" / "acquisition_weight_selection_summary.csv").exists())
            self.assertTrue((out_dir / "tables" / "adaptation_summary.csv").exists())
            self.assertTrue((out_dir / "figures" / "fig03_adaptation_overall_mae.png").exists())
            self.assertTrue((out_dir / "run_report.md").exists())


if __name__ == "__main__":
    unittest.main()
