from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.data import prepare_dataframe


class DataThresholdRecomputeTest(unittest.TestCase):
    def test_force_recompute_pass_label(self) -> None:
        row = {
            "circuit_id": "c0",
            "episode_id": "source-0",
            "domain": "source",
            "shift_type": "source",
            "family": "random_local",
            "topology_kind": "grid",
            "qubit_count": 8,
            "depth": 12,
            "num_1q_gates": 60,
            "num_2q_gates": 20,
            "two_qubit_density": 0.25,
            "avg_degree": 2.0,
            "edge_density": 0.2,
            "symmetry_score": 0.3,
            "periodicity_score": 0.2,
            "linearity_score": 0.5,
            "idle_fraction": 0.2,
            "layer_sparsity": 0.4,
            "t1_us": 120.0,
            "t2_us": 100.0,
            "readout_error": 0.02,
            "oneq_epg": 0.001,
            "twoq_epg": 0.01,
            "time_index": 0,
            "error_rate": 0.11,
            "pass_label": 0,
        }
        df = pd.DataFrame([row])

        keep = prepare_dataframe(df, pass_threshold=0.14, force_recompute_pass_label=False)
        self.assertEqual(int(keep.loc[0, "pass_label"]), 0)

        recomputed = prepare_dataframe(df, pass_threshold=0.14, force_recompute_pass_label=True)
        self.assertEqual(int(recomputed.loc[0, "pass_label"]), 1)


if __name__ == "__main__":
    unittest.main()
