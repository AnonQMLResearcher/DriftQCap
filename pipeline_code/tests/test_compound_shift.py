from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.config import build_profile_config
from driftqcap.synthetic import generate_synthetic_benchmark


class CompoundShiftProfileTest(unittest.TestCase):
    def test_paper_profile_includes_compound_shift_episodes(self) -> None:
        config = build_profile_config("paper", random_seed=123)
        target_shift_types = {spec.shift_type for spec in config.synthetic.target_episodes}
        self.assertIn("compound_shift", target_shift_types)
        compound_ids = [spec.episode_id for spec in config.synthetic.target_episodes if spec.shift_type == "compound_shift"]
        self.assertEqual(len(compound_ids), 4)

    def test_generated_dataset_contains_compound_shift_rows(self) -> None:
        config = build_profile_config("quick", random_seed=123)
        df = generate_synthetic_benchmark(config.synthetic)
        self.assertIn("compound_shift", set(df["shift_type"].astype(str)))
        compound_rows = df[df["shift_type"] == "compound_shift"]
        self.assertFalse(compound_rows.empty)
        self.assertTrue((compound_rows["domain"] == "target").all())


if __name__ == "__main__":
    unittest.main()
