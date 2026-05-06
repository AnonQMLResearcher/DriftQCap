from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap import build_profile_config, run_benchmark
from driftqcap.config import AdaptationConfig, BenchmarkRunConfig, ModelConfig, ReportingConfig, SyntheticBenchmarkConfig
from driftqcap.paper_utils import build_readiness_report, export_standard_latex_tables


class DriftQCapPaperUtilsTest(unittest.TestCase):
    def test_latex_export_and_readiness_report(self) -> None:
        base = build_profile_config("quick", random_seed=321)
        synthetic = SyntheticBenchmarkConfig(
            random_seed=321,
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
            random_seed=321,
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
            random_seed=321,
        )
        reporting = ReportingConfig(save_raw_predictions=True, save_submission_bundle=False)
        config = BenchmarkRunConfig(
            profile_name="paper-utils-smoke",
            synthetic=synthetic,
            model=model,
            adaptation=adaptation,
            calibration=base.calibration,
            evaluation=replace(base.evaluation, bootstrap_resamples=20),
            reporting=reporting,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_benchmark(config, output_dir=run_dir)
            latex_dir = run_dir / "latex_tables"
            exported = export_standard_latex_tables(run_dir=run_dir, output_dir=latex_dir)
            readiness = build_readiness_report(run_dir)

            self.assertTrue(any(path.name == "adaptation_overall_summary.tex" for path in exported))
            self.assertTrue((latex_dir / "README.md").exists())
            self.assertIn("DriftQCap readiness report", readiness)
            self.assertIn("Recommended track stance", readiness)


if __name__ == "__main__":
    unittest.main()
