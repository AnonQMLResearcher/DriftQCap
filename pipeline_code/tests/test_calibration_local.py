from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.calibration import attach_intervals, blend_conformal_artifacts, fit_locally_adaptive_conformal
from driftqcap.models import PredictionBundle


class LocalConformalTest(unittest.TestCase):
    def test_local_conformal_shrinks_easy_region(self) -> None:
        pred_mean = np.concatenate([np.linspace(0.1, 0.4, 20), np.linspace(0.6, 0.9, 20)])
        pred_std = np.full(40, 0.05, dtype=float)
        y_true = pred_mean.copy()
        y_true[:20] += 0.01
        y_true[20:] += 0.12
        artifacts = fit_locally_adaptive_conformal(
            y_true=y_true,
            pred_mean=pred_mean,
            pred_std=pred_std,
            alphas=(0.20,),
            n_bins=2,
            min_bin_size=10,
            stratify_by="pred_mean",
        )
        bundle = PredictionBundle(mean=pred_mean, std=pred_std, member_predictions=np.vstack([pred_mean, pred_mean]))
        bundle = attach_intervals(bundle, conformal=artifacts)
        lower, upper = bundle.intervals["80"]
        easy_width = float(np.mean(upper[:20] - lower[:20]))
        hard_width = float(np.mean(upper[20:] - lower[20:]))
        self.assertLess(easy_width, hard_width)

    def test_local_conformal_falls_back_to_global(self) -> None:
        pred_mean = np.linspace(0.1, 0.9, 8)
        pred_std = np.full(8, 0.05, dtype=float)
        y_true = pred_mean + 0.02
        artifacts = fit_locally_adaptive_conformal(
            y_true=y_true,
            pred_mean=pred_mean,
            pred_std=pred_std,
            alphas=(0.20,),
            n_bins=4,
            min_bin_size=10,
        )
        self.assertIsNone(artifacts.bin_edges)

    def test_blended_conformal_moves_toward_target(self) -> None:
        pred_mean = np.linspace(0.1, 0.9, 20)
        pred_std = np.full(20, 0.05, dtype=float)
        source = fit_locally_adaptive_conformal(
            y_true=pred_mean + 0.10,
            pred_mean=pred_mean,
            pred_std=pred_std,
            alphas=(0.20,),
            n_bins=2,
            min_bin_size=5,
        )
        target = fit_locally_adaptive_conformal(
            y_true=pred_mean + 0.02,
            pred_mean=pred_mean,
            pred_std=pred_std,
            alphas=(0.20,),
            n_bins=2,
            min_bin_size=5,
        )
        blended = blend_conformal_artifacts(source=source, target=target, target_weight=0.75)
        self.assertLess(blended.quantiles["80"], source.quantiles["80"])
        self.assertGreater(blended.quantiles["80"], target.quantiles["80"])


if __name__ == "__main__":
    unittest.main()
