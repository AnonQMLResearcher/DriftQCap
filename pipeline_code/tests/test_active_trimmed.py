from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.active import (
    select_badge_proxy,
    select_density_ratio_uncertainty_diversity,
    select_entropy_threshold_diversity,
    select_oracle_residual,
    select_shift_aware_diversity,
    select_two_stage_explore_exploit,
    select_uncertainty,
    select_uncertainty_threshold_diversity,
)
from driftqcap.models import PredictionBundle


class _DummyModel:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._bundle = PredictionBundle(
            mean=mean,
            std=std,
            member_predictions=np.vstack([mean, mean]),
        )
        self.source_feature_centroid_ = np.zeros(2, dtype=float)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        return self._bundle

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Use two simple numeric features for deterministic distances.
        return df[["f0", "f1"]].to_numpy(dtype=float)


class ActiveTrimmedSelectionTest(unittest.TestCase):
    def test_trimmed_uncertainty_selects_budget(self) -> None:
        n = 20
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.linspace(1.0, 0.0, n)})
        mean = np.linspace(0.0, 1.0, n)
        std = np.linspace(1.0, 0.1, n)
        model = _DummyModel(mean=mean, std=std)
        idx = select_uncertainty(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=5,
            quantile_min=0.20,
            quantile_max=0.80,
        )
        self.assertEqual(len(idx), 5)
        self.assertTrue(np.all(np.diff(idx) >= 0))

    def test_trimmed_utd_selects_budget(self) -> None:
        n = 24
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.cos(np.linspace(0, 1, n))})
        mean = np.linspace(0.0, 1.0, n)
        std = np.linspace(0.2, 1.0, n)
        model = _DummyModel(mean=mean, std=std)
        idx = select_uncertainty_threshold_diversity(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=6,
            threshold=0.14,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.01,
            random_state=123,
            quantile_min=0.15,
            quantile_max=0.85,
        )
        self.assertEqual(len(idx), 6)
        self.assertEqual(len(np.unique(idx)), 6)

    def test_entropy_threshold_diversity_selects_budget(self) -> None:
        n = 30
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.sin(np.linspace(0, 2, n))})
        mean = np.linspace(0.0, 1.0, n)
        std = np.linspace(0.1, 0.6, n)
        model = _DummyModel(mean=mean, std=std)
        idx = select_entropy_threshold_diversity(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=7,
            threshold=0.14,
            entropy_weight=0.45,
            std_weight=0.20,
            threshold_weight=0.20,
            diversity_weight=0.15,
            threshold_beta=0.01,
            random_state=9,
            quantile_min=0.05,
            quantile_max=0.95,
        )
        self.assertEqual(len(idx), 7)
        self.assertEqual(len(np.unique(idx)), 7)
        self.assertTrue(np.all(np.diff(idx) >= 0))

    def test_badge_proxy_selects_budget(self) -> None:
        n = 28
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.linspace(1.0, 0.0, n)})
        mean = np.linspace(0.1, 0.9, n)
        std = np.linspace(0.05, 0.50, n)
        model = _DummyModel(mean=mean, std=std)
        model.source_feature_reference_ = model.transform(candidate)[:10]
        idx = select_badge_proxy(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=6,
            random_state=3,
            quantile_min=0.05,
            quantile_max=0.95,
        )
        self.assertEqual(len(idx), 6)
        self.assertEqual(len(np.unique(idx)), 6)

    def test_density_ratio_uncertainty_diversity_selects_budget(self) -> None:
        n = 26
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.sin(np.linspace(0, 2, n))})
        mean = np.linspace(0.1, 0.9, n)
        std = np.linspace(0.05, 0.30, n)
        model = _DummyModel(mean=mean, std=std)
        source_ref = np.column_stack([np.linspace(-1.0, 0.0, 14), np.linspace(-0.5, 0.5, 14)])
        model.source_feature_reference_ = source_ref
        idx = select_density_ratio_uncertainty_diversity(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=5,
            random_state=11,
            ratio_weight=0.8,
            diversity_weight=0.2,
            quantile_min=0.05,
            quantile_max=0.95,
        )
        self.assertEqual(len(idx), 5)
        self.assertEqual(len(np.unique(idx)), 5)

    def test_two_stage_explore_exploit_selects_budget(self) -> None:
        n = 22
        candidate = pd.DataFrame({"f0": np.linspace(0.0, 1.0, n), "f1": np.cos(np.linspace(0, 1, n))})
        mean = np.linspace(0.1, 0.9, n)
        std = np.linspace(0.05, 0.40, n)
        model = _DummyModel(mean=mean, std=std)
        model.source_feature_reference_ = model.transform(candidate)[:8]
        idx = select_two_stage_explore_exploit(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=6,
            random_state=5,
            explore_fraction=0.5,
            quantile_min=0.05,
            quantile_max=0.95,
        )
        self.assertEqual(len(idx), 6)
        self.assertEqual(len(np.unique(idx)), 6)

    def test_oracle_residual_selects_largest_residuals(self) -> None:
        candidate = pd.DataFrame(
            {
                "f0": [0.0, 1.0, 2.0, 3.0, 4.0],
                "f1": [1.0, 1.0, 1.0, 1.0, 1.0],
                "error_rate": [0.10, 0.12, 0.40, 0.11, 0.35],
            }
        )
        # Predicted means make largest residuals at indices 2 and 4.
        mean = np.array([0.10, 0.11, 0.15, 0.10, 0.20], dtype=float)
        std = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        model = _DummyModel(mean=mean, std=std)
        idx = select_oracle_residual(candidate, model=model, budget=2)  # type: ignore[arg-type]
        self.assertSetEqual(set(idx.tolist()), {2, 4})

    def test_shift_aware_diversity_selects_unique_budget(self) -> None:
        candidate = pd.DataFrame(
            {
                "f0": [0.0, 0.1, 0.2, 3.0, 3.1, 3.2],
                "f1": [0.0, 1.0, 2.0, 0.1, 1.1, 2.1],
            }
        )
        mean = np.linspace(0.1, 0.6, len(candidate))
        std = np.full(len(candidate), 0.1)
        model = _DummyModel(mean=mean, std=std)
        idx = select_shift_aware_diversity(
            candidate,
            model=model,  # type: ignore[arg-type]
            budget=3,
            random_state=7,
            shift_weight=1.0,
            coverage_weight=0.8,
        )
        self.assertEqual(len(idx), 3)
        self.assertEqual(len(np.unique(idx)), 3)
        self.assertTrue(np.all(np.diff(idx) >= 0))


if __name__ == "__main__":
    unittest.main()
