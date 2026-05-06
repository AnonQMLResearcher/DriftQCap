from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftqcap.plots import plot_reliability_diagram_equal_mass, plot_reliability_diagram_from_predictions


class ReliabilityPlotTest(unittest.TestCase):
    def test_reliability_plot_collapses_identical_curves(self) -> None:
        n = 80
        p = np.linspace(0.05, 0.95, n)
        labels = (p > 0.5).astype(int)
        predictions = pd.DataFrame(
            {
                "pass_label": labels,
                "pass_probability_raw": p,
                "pass_probability_calibrated": p.copy(),
            }
        )
        fig = plot_reliability_diagram_from_predictions(predictions, title="test")
        try:
            ax = fig.axes[0]
            labels_out = [line.get_label() for line in ax.get_lines()]
            self.assertTrue(any("raw = calibrated" in label for label in labels_out))
            self.assertFalse(any(label.startswith("calibrated") for label in labels_out))
        finally:
            plt.close(fig)

    def test_reliability_plot_equal_mass_collapses_identical_curves(self) -> None:
        n = 80
        p = np.linspace(0.05, 0.95, n)
        labels = (p > 0.6).astype(int)
        predictions = pd.DataFrame(
            {
                "pass_label": labels,
                "pass_probability_raw": p,
                "pass_probability_calibrated": p.copy(),
            }
        )
        fig = plot_reliability_diagram_equal_mass(predictions, title="test")
        try:
            ax = fig.axes[0]
            labels_out = [line.get_label() for line in ax.get_lines()]
            self.assertTrue(any("raw = cal eq-mass" in label for label in labels_out))
            self.assertFalse(any(label.startswith("cal eq-mass") for label in labels_out))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
