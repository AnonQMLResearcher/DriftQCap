"""Synthetic benchmark generator for drift-aware quantum capability learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import EpisodeSpec, SyntheticBenchmarkConfig


FAMILY_LIBRARY: dict[str, dict[str, float]] = {
    "random_local": {
        "base": 0.006,
        "entangler_mean": 0.20,
        "symmetry_mean": 0.14,
        "periodicity_mean": 0.10,
        "linearity_mean": 0.72,
        "coherent_sensitivity": 0.30,
        "crosstalk_sensitivity": 0.40,
        "context_sensitivity": 0.25,
    },
    "random_entangler": {
        "base": 0.010,
        "entangler_mean": 0.55,
        "symmetry_mean": 0.18,
        "periodicity_mean": 0.16,
        "linearity_mean": 0.40,
        "coherent_sensitivity": 0.52,
        "crosstalk_sensitivity": 0.78,
        "context_sensitivity": 0.30,
    },
    "parity": {
        "base": 0.009,
        "entangler_mean": 0.44,
        "symmetry_mean": 0.66,
        "periodicity_mean": 0.60,
        "linearity_mean": 0.58,
        "coherent_sensitivity": 0.72,
        "crosstalk_sensitivity": 0.65,
        "context_sensitivity": 0.52,
    },
    "qaoa_like": {
        "base": 0.008,
        "entangler_mean": 0.38,
        "symmetry_mean": 0.46,
        "periodicity_mean": 0.30,
        "linearity_mean": 0.50,
        "coherent_sensitivity": 0.50,
        "crosstalk_sensitivity": 0.62,
        "context_sensitivity": 0.38,
    },
    "mirror": {
        "base": 0.009,
        "entangler_mean": 0.48,
        "symmetry_mean": 0.82,
        "periodicity_mean": 0.78,
        "linearity_mean": 0.60,
        "coherent_sensitivity": 0.90,
        "crosstalk_sensitivity": 0.55,
        "context_sensitivity": 0.62,
    },
    "ghz": {
        "base": 0.011,
        "entangler_mean": 0.35,
        "symmetry_mean": 0.92,
        "periodicity_mean": 0.18,
        "linearity_mean": 0.86,
        "coherent_sensitivity": 0.64,
        "crosstalk_sensitivity": 0.48,
        "context_sensitivity": 0.40,
    },
    "symmetry_fragment": {
        "base": 0.010,
        "entangler_mean": 0.33,
        "symmetry_mean": 0.88,
        "periodicity_mean": 0.70,
        "linearity_mean": 0.68,
        "coherent_sensitivity": 0.84,
        "crosstalk_sensitivity": 0.58,
        "context_sensitivity": 0.55,
    },
}

_TOPOLOGY_FACTOR = {
    "grid": 1.00,
    "heavy_hex": 0.94,
    "line": 1.08,
    "ring": 1.02,
}


@dataclass(frozen=True)
class DeviceState:
    """Known and hidden device state for one episode."""

    t1_us: float
    t2_us: float
    readout_error: float
    oneq_epg: float
    twoq_epg: float
    hidden_coherent_scale: float
    hidden_crosstalk_scale: float
    hidden_context_scale: float
    hidden_temporal_scale: float


class SyntheticCapabilityBenchmark:
    """Generate a synthetic benchmark with source and target drift episodes."""

    def __init__(self, config: SyntheticBenchmarkConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for spec in [*self.config.source_episodes, *self.config.target_episodes]:
            frames.append(self._generate_episode(spec))
        df = pd.concat(frames, ignore_index=True)
        df["pass_label"] = (df["error_rate"] <= self.config.pass_threshold).astype(int)
        return df

    def _generate_episode(self, spec: EpisodeSpec) -> pd.DataFrame:
        device_state = self._sample_device_state(spec)
        rows: list[dict[str, Any]] = []
        for local_idx in range(spec.circuits_per_episode):
            row = self._sample_circuit_row(spec=spec, device_state=device_state, local_idx=local_idx)
            rows.append(row)
        return pd.DataFrame(rows)

    def _sample_device_state(self, spec: EpisodeSpec) -> DeviceState:
        scale = spec.known_noise_scale
        smooth = 1.0 + 0.04 * np.sin(spec.smooth_drift_phase)
        t1_us = float(self.rng.uniform(90.0, 180.0) / (scale * smooth))
        t2_us = float(self.rng.uniform(70.0, 160.0) / (scale * smooth))
        readout_error = float(self.rng.uniform(0.010, 0.028) * scale * smooth)
        oneq_epg = float(self.rng.uniform(0.0002, 0.0012) * scale * smooth)
        twoq_epg = float(self.rng.uniform(0.0025, 0.0100) * scale * smooth)

        hidden_coherent = float(self.rng.normal(0.012 + spec.hidden_coherent_shift + 0.002 * np.sin(spec.smooth_drift_phase), 0.002))
        hidden_crosstalk = float(self.rng.normal(0.010 + spec.hidden_crosstalk_shift + 0.002 * np.cos(spec.smooth_drift_phase), 0.002))
        hidden_context = float(self.rng.normal(0.004 + spec.hidden_context_shift + 0.001 * np.sin(0.5 * spec.smooth_drift_phase), 0.0015))
        hidden_temporal = float(self.rng.normal(0.002 + spec.hidden_temporal_shift + 0.0008 * spec.time_index / 10.0, 0.001))
        return DeviceState(
            t1_us=t1_us,
            t2_us=t2_us,
            readout_error=readout_error,
            oneq_epg=oneq_epg,
            twoq_epg=twoq_epg,
            hidden_coherent_scale=max(hidden_coherent, 0.0005),
            hidden_crosstalk_scale=max(hidden_crosstalk, 0.0005),
            hidden_context_scale=max(hidden_context, 0.0002),
            hidden_temporal_scale=max(hidden_temporal, 0.0002),
        )

    def _sample_circuit_row(self, *, spec: EpisodeSpec, device_state: DeviceState, local_idx: int) -> dict[str, Any]:
        family = str(self.rng.choice(spec.families))
        params = FAMILY_LIBRARY[family]
        topology_factor = _TOPOLOGY_FACTOR.get(spec.topology_kind, 1.0)

        q_low, q_high = spec.qubit_range
        d_low, d_high = spec.depth_range
        qubit_count = int(self.rng.integers(q_low, q_high + 1))
        depth = int(self.rng.integers(d_low, d_high + 1))

        entangler_ratio = float(np.clip(self.rng.normal(params["entangler_mean"], 0.08), 0.05, 0.95))
        symmetry_score = float(np.clip(self.rng.normal(params["symmetry_mean"], 0.10), 0.0, 1.0))
        periodicity_score = float(np.clip(self.rng.normal(params["periodicity_mean"], 0.10), 0.0, 1.0))
        linearity_score = float(np.clip(self.rng.normal(params["linearity_mean"], 0.10), 0.0, 1.0))

        num_2q_gates = int(max(1, round(depth * qubit_count * entangler_ratio * self.rng.uniform(0.18, 0.32))))
        num_1q_gates = int(max(qubit_count, round(depth * qubit_count * (1.05 - 0.40 * entangler_ratio) * self.rng.uniform(0.85, 1.20))))
        total_gates = num_1q_gates + num_2q_gates
        two_qubit_density = float(num_2q_gates / total_gates)

        avg_degree = float(np.clip((1.20 + 2.10 * entangler_ratio + 0.30 * symmetry_score + self.rng.normal(0.0, 0.18)) * topology_factor, 1.0, 4.0))
        edge_density = float(np.clip(avg_degree / max(qubit_count - 1, 1), 0.02, 1.0))
        idle_fraction = float(np.clip(1.0 - total_gates / max(depth * qubit_count * 1.6, 1.0), 0.0, 1.0))
        layer_sparsity = float(np.clip(1.0 - entangler_ratio + 0.25 * idle_fraction, 0.0, 1.0))
        motif_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        family_hash = {
            "random_local": 0.2,
            "random_entangler": 0.8,
            "parity": 1.3,
            "qaoa_like": 1.9,
            "mirror": 2.4,
            "ghz": 2.9,
            "symmetry_fragment": 3.4,
        }[family]

        components = self._compute_error_components(
            family=family,
            params=params,
            qubit_count=qubit_count,
            depth=depth,
            num_1q_gates=num_1q_gates,
            num_2q_gates=num_2q_gates,
            two_qubit_density=two_qubit_density,
            avg_degree=avg_degree,
            edge_density=edge_density,
            symmetry_score=symmetry_score,
            periodicity_score=periodicity_score,
            linearity_score=linearity_score,
            idle_fraction=idle_fraction,
            layer_sparsity=layer_sparsity,
            motif_phase=motif_phase,
            family_hash=family_hash,
            time_index=spec.time_index,
            topology_factor=topology_factor,
            device_state=device_state,
        )
        error_rate = float(np.clip(sum(components.values()), 0.003, 0.30))

        row = {
            "circuit_id": f"{spec.episode_id}-{local_idx:05d}",
            "episode_id": spec.episode_id,
            "domain": spec.domain,
            "shift_type": spec.shift_type,
            "family": family,
            "topology_kind": spec.topology_kind,
            "qubit_count": qubit_count,
            "depth": depth,
            "num_1q_gates": num_1q_gates,
            "num_2q_gates": num_2q_gates,
            "two_qubit_density": two_qubit_density,
            "avg_degree": avg_degree,
            "edge_density": edge_density,
            "symmetry_score": symmetry_score,
            "periodicity_score": periodicity_score,
            "linearity_score": linearity_score,
            "idle_fraction": idle_fraction,
            "layer_sparsity": layer_sparsity,
            "t1_us": device_state.t1_us,
            "t2_us": device_state.t2_us,
            "readout_error": device_state.readout_error,
            "oneq_epg": device_state.oneq_epg,
            "twoq_epg": device_state.twoq_epg,
            "time_index": spec.time_index,
            "error_rate": error_rate,
            "_hidden_coherent_scale": device_state.hidden_coherent_scale,
            "_hidden_crosstalk_scale": device_state.hidden_crosstalk_scale,
            "_hidden_context_scale": device_state.hidden_context_scale,
            "_hidden_temporal_scale": device_state.hidden_temporal_scale,
        }
        row.update(components)
        return row

    @staticmethod
    def _compute_error_components(
        *,
        family: str,
        params: dict[str, float],
        qubit_count: int,
        depth: int,
        num_1q_gates: int,
        num_2q_gates: int,
        two_qubit_density: float,
        avg_degree: float,
        edge_density: float,
        symmetry_score: float,
        periodicity_score: float,
        linearity_score: float,
        idle_fraction: float,
        layer_sparsity: float,
        motif_phase: float,
        family_hash: float,
        time_index: int,
        topology_factor: float,
        device_state: DeviceState,
    ) -> dict[str, float]:
        family_base = params["base"]
        size_component = 0.00016 * qubit_count + 0.00022 * depth
        gate_component = 0.000007 * num_1q_gates + 0.000060 * num_2q_gates

        decoherence_component = (
            0.005 * depth / max(device_state.t1_us, 1.0)
            + 0.004 * depth / max(device_state.t2_us, 1.0)
        )
        stochastic_component = (
            0.20 * device_state.readout_error
            + 0.30 * device_state.oneq_epg * np.sqrt(max(num_1q_gates, 1))
            + 0.50 * device_state.twoq_epg * np.sqrt(max(num_2q_gates, 1))
        )

        coherent_pattern = 0.35 + symmetry_score + 0.55 * periodicity_score + 0.25 * abs(np.sin(motif_phase + family_hash))
        coherent_component = (
            device_state.hidden_coherent_scale
            * params["coherent_sensitivity"]
            * coherent_pattern
            * (0.50 + two_qubit_density)
            * (0.60 + np.sin(0.21 * depth + motif_phase + family_hash) ** 2)
            * 0.40  # Scale down from original to avoid saturation
        )

        crosstalk_component = (
            device_state.hidden_crosstalk_scale
            * params["crosstalk_sensitivity"]
            * two_qubit_density
            * avg_degree
            * (qubit_count / 18.0)
            * topology_factor
            * 0.50  # Scale down from original
        )

        context_component = (
            device_state.hidden_context_scale
            * params["context_sensitivity"]
            * (0.50 + abs(np.cos(motif_phase)))
            * (0.50 + linearity_score)
            * (0.60 + edge_density * qubit_count)
            * (0.80 + 0.40 * (1.0 - idle_fraction))
        )

        temporal_component = (
            device_state.hidden_temporal_scale
            * (0.35 + symmetry_score)
            * max(depth - 18, 0)
            / 18.0
            * (1.0 + 0.03 * time_index)
        )

        family_penalty_component = 0.0
        if family == "mirror":
            family_penalty_component = 0.004 * (0.40 + periodicity_score) * (0.60 + abs(np.sin(motif_phase)))
        elif family == "ghz":
            family_penalty_component = 0.003 * (0.50 + symmetry_score)
        elif family == "symmetry_fragment":
            family_penalty_component = 0.0025 * (0.50 + layer_sparsity) * (0.60 + periodicity_score)

        seed = int((qubit_count + 13 * depth + 1000 * motif_phase + 17 * time_index + 100 * family_hash) % (2**32 - 1))
        observation_noise_component = float(np.random.default_rng(seed).normal(0.0, 0.002))
        return {
            "family_base_component": float(family_base),
            "size_component": float(size_component),
            "gate_component": float(gate_component),
            "decoherence_component": float(decoherence_component),
            "stochastic_component": float(stochastic_component),
            "coherent_component": float(coherent_component),
            "crosstalk_component": float(crosstalk_component),
            "context_component": float(context_component),
            "temporal_component": float(temporal_component),
            "family_penalty_component": float(family_penalty_component),
            "observation_noise_component": float(observation_noise_component),
        }



def generate_synthetic_benchmark(config: SyntheticBenchmarkConfig) -> pd.DataFrame:
    return SyntheticCapabilityBenchmark(config).generate()
