"""Feature schema utilities."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


BASE_NUMERIC_FEATURES: Final[list[str]] = [
    "qubit_count",
    "depth",
    "num_1q_gates",
    "num_2q_gates",
    "two_qubit_density",
    "avg_degree",
    "edge_density",
    "symmetry_score",
    "periodicity_score",
    "linearity_score",
    "idle_fraction",
    "layer_sparsity",
    "t1_us",
    "t2_us",
    "readout_error",
    "oneq_epg",
    "twoq_epg",
    "time_index",
]

DERIVED_NUMERIC_FEATURES: Final[list[str]] = [
    "gate_count",
    "depth_per_qubit",
    "entangling_load",
    "stochastic_proxy",
    "decoherence_proxy",
    "coherent_proxy",
    "crosstalk_proxy",
    "context_proxy",
    "temporal_proxy",
    "mirror_like_proxy",
    "stability_proxy",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "family",
    "shift_type",
    "topology_kind",
]

FULL_NUMERIC_FEATURES: Final[list[str]] = [*BASE_NUMERIC_FEATURES, *DERIVED_NUMERIC_FEATURES]
COARSE_NUMERIC_FEATURES: Final[list[str]] = [
    "qubit_count",
    "depth",
    "num_2q_gates",
    "readout_error",
    "oneq_epg",
    "twoq_epg",
]

REQUIRED_COLUMNS: Final[list[str]] = [
    "circuit_id",
    "episode_id",
    "domain",
    "shift_type",
    "family",
    "topology_kind",
    *BASE_NUMERIC_FEATURES,
    "error_rate",
    "pass_label",
]

LATENT_COMPONENT_COLUMNS: Final[list[str]] = [
    "family_base_component",
    "size_component",
    "gate_component",
    "decoherence_component",
    "stochastic_component",
    "coherent_component",
    "crosstalk_component",
    "context_component",
    "temporal_component",
    "family_penalty_component",
    "observation_noise_component",
]



def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived, physics-aware features without touching the latent labels."""
    out = df.copy()
    out["gate_count"] = out["num_1q_gates"] + out["num_2q_gates"]
    out["depth_per_qubit"] = out["depth"] / np.maximum(out["qubit_count"], 1)
    out["entangling_load"] = out["two_qubit_density"] * out["depth"] * out["qubit_count"]
    out["stochastic_proxy"] = (
        0.50 * out["readout_error"]
        + 1.20 * out["oneq_epg"] * np.sqrt(np.maximum(out["num_1q_gates"], 1))
        + 1.80 * out["twoq_epg"] * np.sqrt(np.maximum(out["num_2q_gates"], 1))
    )
    out["decoherence_proxy"] = (
        0.020 * out["depth"] / np.maximum(out["t1_us"], 1.0)
        + 0.018 * out["depth"] / np.maximum(out["t2_us"], 1.0)
    )
    out["coherent_proxy"] = (
        (0.35 + out["symmetry_score"] + 0.55 * out["periodicity_score"])
        * (0.50 + out["two_qubit_density"])
        * (0.60 + (out["symmetry_score"] - out["periodicity_score"]) ** 2)
    )
    out["crosstalk_proxy"] = out["two_qubit_density"] * out["avg_degree"] * (out["qubit_count"] / 18.0)
    out["context_proxy"] = (
        (0.50 + out["linearity_score"])
        * (0.50 + out["edge_density"] * out["qubit_count"])
        * (0.50 + out["symmetry_score"])
    )
    out["temporal_proxy"] = np.maximum(out["depth"] - 18.0, 0.0) / 18.0 * (0.35 + out["symmetry_score"])
    out["mirror_like_proxy"] = out["symmetry_score"] * out["periodicity_score"]
    out["stability_proxy"] = 1.0 / np.maximum(out["readout_error"] + out["oneq_epg"] + out["twoq_epg"], 1e-5)
    return out
