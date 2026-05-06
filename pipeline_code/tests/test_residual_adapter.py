from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.models import ElasticResidualAdapter, FewShotResidualAdapter, PredictionBundle


class _DummyBaseModel:
    def __init__(self) -> None:
        self.source_feature_centroid_ = np.zeros(5, dtype=float)
        self.source_feature_rank_ = np.array([4, 3, 2, 1, 0], dtype=int)

    def _check_is_fitted(self) -> None:
        return None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df[[f"f{i}" for i in range(5)]].to_numpy(dtype=float)

    def predict_distribution(self, df: pd.DataFrame) -> PredictionBundle:
        mean = 0.2 + 0.05 * df["f0"].to_numpy(dtype=float)
        std = np.full(len(df), 0.1, dtype=float)
        members = np.vstack([mean - 0.01, mean + 0.01])
        return PredictionBundle(mean=mean, std=std, member_predictions=members)


class ResidualAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        n = 12
        self.df = pd.DataFrame(
            {
                "f0": np.linspace(0.0, 1.0, n),
                "f1": np.linspace(1.0, 0.0, n),
                "f2": np.sin(np.linspace(0.0, 1.0, n)),
                "f3": np.cos(np.linspace(0.0, 1.0, n)),
                "f4": np.linspace(-1.0, 1.0, n),
                "qubit_count": np.arange(4, 4 + n),
                "depth": np.arange(10, 10 + n),
                "two_qubit_density": np.linspace(0.1, 0.9, n),
                "avg_degree": np.linspace(1.0, 3.0, n),
                "readout_error": np.linspace(0.01, 0.03, n),
                "oneq_epg": np.linspace(0.001, 0.002, n),
                "twoq_epg": np.linspace(0.01, 0.02, n),
                "time_index": np.arange(n),
            }
        )
        base_mean = 0.2 + 0.05 * self.df["f0"].to_numpy(dtype=float)
        self.df["error_rate"] = np.clip(base_mean + 0.15 * self.df["f4"].to_numpy(dtype=float), 0.0, 1.0)
        self.base_model = _DummyBaseModel()

    def test_summary_mode_uses_compact_design(self) -> None:
        adapter = FewShotResidualAdapter(
            self.base_model,  # type: ignore[arg-type]
            residual_feature_mode="summary",
            alpha_grid=(0.5, 1.0),
            cv_folds=3,
        ).fit(self.df)
        transformed = adapter.transform(self.df)
        self.assertEqual(transformed.shape[1], 11)
        self.assertIn(adapter.selected_alpha_, {0.5, 1.0})

    def test_hybrid_mode_truncates_feature_block(self) -> None:
        adapter = FewShotResidualAdapter(
            self.base_model,  # type: ignore[arg-type]
            residual_feature_mode="hybrid",
            max_feature_dims=2,
            alpha_grid=(0.5, 1.0, 2.0),
            cv_folds=3,
        ).fit(self.df)
        transformed = adapter.transform(self.df)
        self.assertEqual(transformed.shape[1], 5)
        self.assertEqual(adapter.design_dim_, 5)
        self.assertIn(adapter.selected_alpha_, {0.5, 1.0, 2.0})

    def test_elastic_residual_adapter_selects_hyperparameters(self) -> None:
        adapter = ElasticResidualAdapter(
            self.base_model,  # type: ignore[arg-type]
            feature_mode="hybrid",
            max_feature_dims=2,
            alpha_grid=(0.5, 1.0),
            ewc_lambda_grid=(0.0, 1.0),
            cv_folds=3,
        ).fit(self.df)
        diagnostics = adapter.diagnostics()
        self.assertIn(diagnostics["selected_alpha"], {0.5, 1.0})
        self.assertIn(diagnostics["selected_ewc_lambda"], {0.0, 1.0})
        self.assertEqual(diagnostics["design_dim"], 5)


if __name__ == "__main__":
    unittest.main()
