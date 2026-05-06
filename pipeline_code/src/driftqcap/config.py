"""Configuration dataclasses and benchmark profiles."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(frozen=True)
class EpisodeSpec:
    """Specification for one synthetic source or target episode."""

    episode_id: str
    domain: Literal["source", "target"]
    shift_type: str
    families: tuple[str, ...]
    qubit_range: tuple[int, int]
    depth_range: tuple[int, int]
    circuits_per_episode: int
    known_noise_scale: float = 1.0
    hidden_coherent_shift: float = 0.0
    hidden_crosstalk_shift: float = 0.0
    hidden_context_shift: float = 0.0
    hidden_temporal_shift: float = 0.0
    time_index: int = 0
    smooth_drift_phase: float = 0.0
    topology_kind: str = "grid"


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Configuration for the synthetic drift benchmark."""

    random_seed: int = 20260308
    pass_threshold: float = 0.080
    source_episodes: tuple[EpisodeSpec, ...] = field(default_factory=tuple)
    target_episodes: tuple[EpisodeSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the tree ensemble regressor."""

    base_estimator: Literal["rf", "xgboost", "lightgbm", "mlp", "bayesian_ridge", "xgboost_monotonic", "tabnet", "qpa_nn"] = "rf"
    n_members: int = 96
    non_rf_members: int = 8
    max_depth: int = 14
    min_samples_leaf: int = 3
    max_features: float | str = "sqrt"
    bootstrap_fraction: float = 0.85
    xgb_learning_rate: float = 0.06
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_lambda: float = 1.0
    lgb_learning_rate: float = 0.06
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85
    lgb_reg_lambda: float = 1.0
    xgb_monotone_depth: int = 8
    xgb_monotone_constraints: tuple[tuple[str, int], ...] = (
        ("depth", 1),
        ("readout_error", 1),
        ("oneq_epg", 1),
        ("twoq_epg", 1),
        ("t1_us", -1),
        ("t2_us", -1),
    )
    bayesian_alpha_1: float = 1e-6
    bayesian_alpha_2: float = 1e-6
    bayesian_lambda_1: float = 1e-6
    bayesian_lambda_2: float = 1e-6
    # Conservative MLP defaults for reproducible benchmark baselines:
    # smaller network + stronger regularization + earlier stop when progress stalls.
    mlp_hidden_layers: tuple[int, ...] = (64, 32)
    mlp_alpha: float = 5e-4
    mlp_learning_rate_init: float = 5e-4
    mlp_max_iter: int = 1500
    mlp_tol: float = 5e-4
    mlp_n_iter_no_change: int = 20
    tabnet_n_d: int = 16
    tabnet_n_a: int = 16
    tabnet_n_steps: int = 4
    tabnet_gamma: float = 1.3
    tabnet_lambda_sparse: float = 1e-4
    tabnet_max_epochs: int = 200
    qpa_hidden_layers: tuple[int, ...] = (256, 128, 64)
    qpa_alpha: float = 5e-5
    qpa_learning_rate_init: float = 7e-4
    qpa_max_iter: int = 1200
    qpa_tol: float = 1e-4
    qpa_n_iter_no_change: int = 50
    n_jobs: int = 1
    random_seed: int = 20260308


@dataclass(frozen=True)
class AdaptationConfig:
    """Configuration for few-shot adaptation and active selection."""

    budgets: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    acquisition_budgets: tuple[int, ...] | None = None
    random_repeats: int = 10
    candidate_fraction: float = 0.45
    ridge_alpha: float = 1.0
    residual_feature_mode: Literal["full", "summary", "hybrid"] = "hybrid"
    residual_use_base_mean: bool = True
    residual_use_base_std: bool = True
    residual_max_feature_dims: int | None = 64
    residual_alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    residual_cv_folds: int = 3
    residual_selection_metric: Literal["mae"] = "mae"
    era_ridge_alpha: float = 1.0
    era_alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    era_cv_folds: int = 3
    era_feature_mode: Literal["full", "summary", "hybrid"] = "hybrid"
    era_use_base_mean: bool = True
    era_use_base_std: bool = True
    era_max_feature_dims: int | None = 64
    era_ewc_lambda_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    era_selection_metric: Literal["mae"] = "mae"
    std_temperature_quantile: float = 0.90
    std_temperature_min: float = 0.60
    std_temperature_max: float = 1.00
    std_floor: float = 1e-4
    target_upweight: float = 6.0
    uncertainty_weight: float = 1.0
    threshold_weight: float = 0.7
    diversity_weight: float = 0.35
    threshold_beta: float = 0.010
    uncertainty_quantile_min: float = 0.15
    uncertainty_quantile_max: float = 0.85
    acq_score_mode: Literal["legacy", "entropy_threshold_diversity"] = "entropy_threshold_diversity"
    acq_entropy_weight: float = 0.45
    acq_std_weight: float = 0.20
    acq_threshold_weight: float = 0.20
    acq_diversity_weight: float = 0.15
    acq_quantile_min: float = 0.05
    acq_quantile_max: float = 0.95
    acq_weight_grid_enabled: bool = True
    acq_weight_grid: tuple[tuple[float, float, float, float], ...] = (
        (0.45, 0.20, 0.20, 0.15),
        (0.40, 0.20, 0.25, 0.15),
        (0.35, 0.25, 0.20, 0.20),
        (0.50, 0.20, 0.15, 0.15),
        (0.30, 0.30, 0.20, 0.20),
    )
    uncertainty_cap_quantile: float = 0.90
    density_weight: float = 0.30
    density_ratio_weight: float = 0.75
    density_ratio_diversity_weight: float = 0.25
    two_stage_explore_fraction: float = 0.50
    predmean_stratified_bins: int = 10
    shift_diversity_weight: float = 0.75
    shift_diversity_coverage_weight: float = 0.50
    ewc_lambda: float = 2.0
    strategy_for_claim: Literal[
        "badge_proxy",
        "density_ratio_uncertainty_diversity",
        "two_stage_explore_exploit",
        "stratified_predmean_diversity",
        "entropy_threshold_diversity",
        "uncertainty_threshold_diversity",
        "uncertainty_threshold_diversity_capped",
        "diversity",
        "hybrid_shift_aware",
        "stratified_diversity",
        "shift_aware_diversity",
    ] = "diversity"
    enable_hybrid_shift_policy: bool = False
    hybrid_depth_strategy: Literal["random", "diversity"] = "random"
    hybrid_family_strategy: Literal["random", "diversity"] = "random"
    hybrid_noise_strategy: Literal["random", "diversity"] = "diversity"
    hybrid_temporal_strategy: Literal["random", "diversity"] = "diversity"
    stratified_diversity_bins: int = 4
    stratified_diversity_family_stratify: bool = True
    oracle_search_trials: int = 0
    random_seed: int = 20260308


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for calibration and interval estimation."""

    interval_alphas: tuple[float, ...] = (0.20, 0.10, 0.05)
    conformal_mode: Literal["global_scaled", "locally_adaptive"] = "locally_adaptive"
    conformal_strata: Literal["pred_mean", "pred_std", "pred_mean_x_std"] = "pred_mean"
    conformal_n_bins: int = 4
    conformal_min_bin_size: int = 12
    conformal_fallback_to_global: bool = True
    ece_bins: int = 10
    min_std: float = 1e-4
    target_recalibration: bool = False
    target_recalibration_mode: Literal["target_only", "source_plus_target"] = "source_plus_target"
    conformal_recalibration_mode: Literal["source_only", "target_only", "source_plus_target"] = "source_plus_target"
    adapted_conformal_mode: Literal["source_only", "target_only", "source_plus_target_blend"] = "target_only"
    adapted_conformal_target_min_rows: int = 12
    adapted_conformal_blend_tau: float = 24.0
    adapted_disable_prob_calibration: bool = True
    adapted_prob_calibration_diagnostic: bool = True
    adapted_prob_calibration_diagnostic_mode: Literal["none_vs_isotonic"] = "none_vs_isotonic"
    min_target_calibration_rows: int = 8
    min_target_positive: int = 1
    min_target_negative: int = 1
    prob_calibrator: Literal["isotonic", "platt"] = "platt"
    calibration_shrink_tau: float = 32.0
    ece_bins_min_count: int = 25


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for statistical summaries and reporting."""

    bootstrap_resamples: int = 800
    significance_alpha: float = 0.05
    interval_nominal_targets: tuple[float, ...] = (0.80, 0.90, 0.95)
    coverage_tolerance: float = 0.03


@dataclass(frozen=True)
class ReportingConfig:
    """Configuration for persisted artifacts."""

    save_raw_predictions: bool = True
    save_submission_bundle: bool = True
    save_model_artifacts: bool = False
    load_model_artifacts: bool = False
    artifacts_dirname: str = "artifacts"


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Top-level experiment configuration."""

    profile_name: str
    synthetic: SyntheticBenchmarkConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SOURCE_FAMILIES = ("random_local", "random_entangler", "parity", "qaoa_like")
_FAMILY_SHIFT_A = ("mirror", "ghz", "parity")
_FAMILY_SHIFT_B = ("mirror", "ghz", "qaoa_like")
_FAMILY_SHIFT_C = ("mirror", "symmetry_fragment", "ghz")


def _build_source_episodes(circuits_per_source_episode: int) -> tuple[EpisodeSpec, ...]:
    return (
        EpisodeSpec(
            episode_id="source-00",
            domain="source",
            shift_type="source",
            families=_SOURCE_FAMILIES,
            qubit_range=(6, 22),
            depth_range=(4, 26),
            circuits_per_episode=circuits_per_source_episode,
            known_noise_scale=1.00,
            hidden_coherent_shift=0.000,
            hidden_crosstalk_shift=0.000,
            hidden_context_shift=0.000,
            time_index=0,
            smooth_drift_phase=0.00,
            topology_kind="grid",
        ),
        EpisodeSpec(
            episode_id="source-01",
            domain="source",
            shift_type="source",
            families=_SOURCE_FAMILIES,
            qubit_range=(8, 24),
            depth_range=(5, 30),
            circuits_per_episode=circuits_per_source_episode,
            known_noise_scale=1.05,
            hidden_coherent_shift=0.003,
            hidden_crosstalk_shift=0.002,
            hidden_context_shift=-0.001,
            time_index=1,
            smooth_drift_phase=0.35,
            topology_kind="grid",
        ),
        EpisodeSpec(
            episode_id="source-02",
            domain="source",
            shift_type="source",
            families=_SOURCE_FAMILIES,
            qubit_range=(5, 20),
            depth_range=(3, 24),
            circuits_per_episode=circuits_per_source_episode,
            known_noise_scale=0.96,
            hidden_coherent_shift=-0.002,
            hidden_crosstalk_shift=0.001,
            hidden_context_shift=0.002,
            time_index=2,
            smooth_drift_phase=0.70,
            topology_kind="heavy_hex",
        ),
        EpisodeSpec(
            episode_id="source-03",
            domain="source",
            shift_type="source",
            families=_SOURCE_FAMILIES,
            qubit_range=(10, 26),
            depth_range=(6, 28),
            circuits_per_episode=circuits_per_source_episode,
            known_noise_scale=1.02,
            hidden_coherent_shift=0.001,
            hidden_crosstalk_shift=-0.001,
            hidden_context_shift=0.001,
            time_index=3,
            smooth_drift_phase=1.05,
            topology_kind="line",
        ),
    )



def _build_target_episodes(circuits_per_target_episode: int) -> tuple[EpisodeSpec, ...]:
    target: list[EpisodeSpec] = []

    # Family shift episodes.
    target.extend(
        [
            EpisodeSpec(
                episode_id="target-family-00",
                domain="target",
                shift_type="family_shift",
                families=_FAMILY_SHIFT_A,
                qubit_range=(8, 24),
                depth_range=(5, 28),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.02,
                hidden_coherent_shift=0.010,
                hidden_crosstalk_shift=0.004,
                hidden_context_shift=0.004,
                time_index=4,
                smooth_drift_phase=1.20,
                topology_kind="grid",
            ),
            EpisodeSpec(
                episode_id="target-family-01",
                domain="target",
                shift_type="family_shift",
                families=_FAMILY_SHIFT_B,
                qubit_range=(10, 26),
                depth_range=(6, 30),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.00,
                hidden_coherent_shift=0.008,
                hidden_crosstalk_shift=0.003,
                hidden_context_shift=0.003,
                time_index=5,
                smooth_drift_phase=1.35,
                topology_kind="heavy_hex",
            ),
            EpisodeSpec(
                episode_id="target-family-02",
                domain="target",
                shift_type="family_shift",
                families=_FAMILY_SHIFT_C,
                qubit_range=(7, 22),
                depth_range=(6, 24),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.01,
                hidden_coherent_shift=0.009,
                hidden_crosstalk_shift=0.005,
                hidden_context_shift=0.004,
                time_index=6,
                smooth_drift_phase=1.50,
                topology_kind="ring",
            ),
        ]
    )

    # Depth shift episodes.
    target.extend(
        [
            EpisodeSpec(
                episode_id="target-depth-00",
                domain="target",
                shift_type="depth_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(8, 24),
                depth_range=(28, 56),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.05,
                hidden_coherent_shift=0.004,
                hidden_crosstalk_shift=0.002,
                hidden_context_shift=0.001,
                time_index=7,
                smooth_drift_phase=1.65,
                topology_kind="grid",
            ),
            EpisodeSpec(
                episode_id="target-depth-01",
                domain="target",
                shift_type="depth_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(12, 28),
                depth_range=(24, 52),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=0.98,
                hidden_coherent_shift=0.006,
                hidden_crosstalk_shift=0.003,
                hidden_context_shift=0.002,
                time_index=8,
                smooth_drift_phase=1.80,
                topology_kind="line",
            ),
            EpisodeSpec(
                episode_id="target-depth-02",
                domain="target",
                shift_type="depth_shift",
                families=("random_entangler", "parity", "qaoa_like"),
                qubit_range=(10, 24),
                depth_range=(30, 60),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.01,
                hidden_coherent_shift=0.005,
                hidden_crosstalk_shift=0.003,
                hidden_context_shift=0.002,
                time_index=9,
                smooth_drift_phase=1.95,
                topology_kind="heavy_hex",
            ),
        ]
    )

    # Noise shift episodes.
    target.extend(
        [
            EpisodeSpec(
                episode_id="target-noise-00",
                domain="target",
                shift_type="noise_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(8, 24),
                depth_range=(5, 28),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.28,
                hidden_coherent_shift=0.011,
                hidden_crosstalk_shift=0.009,
                hidden_context_shift=0.005,
                hidden_temporal_shift=0.003,
                time_index=10,
                smooth_drift_phase=2.10,
                topology_kind="grid",
            ),
            EpisodeSpec(
                episode_id="target-noise-01",
                domain="target",
                shift_type="noise_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(10, 26),
                depth_range=(6, 30),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.22,
                hidden_coherent_shift=0.010,
                hidden_crosstalk_shift=0.008,
                hidden_context_shift=0.004,
                hidden_temporal_shift=0.003,
                time_index=11,
                smooth_drift_phase=2.25,
                topology_kind="ring",
            ),
            EpisodeSpec(
                episode_id="target-noise-02",
                domain="target",
                shift_type="noise_shift",
                families=("random_entangler", "parity", "qaoa_like"),
                qubit_range=(8, 22),
                depth_range=(6, 26),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.35,
                hidden_coherent_shift=0.012,
                hidden_crosstalk_shift=0.010,
                hidden_context_shift=0.006,
                hidden_temporal_shift=0.004,
                time_index=12,
                smooth_drift_phase=2.40,
                topology_kind="heavy_hex",
            ),
        ]
    )

    # Temporal adaptation episodes: same family mix, time-varying hidden drifts.
    for idx, (scale, coh, xalk, ctx, temp) in enumerate(
        [
            (1.00, 0.002, 0.001, 0.001, 0.001),
            (1.06, 0.004, 0.003, 0.002, 0.002),
            (1.14, 0.006, 0.005, 0.003, 0.003),
            (1.22, 0.009, 0.007, 0.004, 0.004),
        ]
    ):
        target.append(
            EpisodeSpec(
                episode_id=f"target-temporal-{idx:02d}",
                domain="target",
                shift_type="temporal_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(8, 24),
                depth_range=(5, 30),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=scale,
                hidden_coherent_shift=coh,
                hidden_crosstalk_shift=xalk,
                hidden_context_shift=ctx,
                hidden_temporal_shift=temp,
                time_index=20 + idx,
                smooth_drift_phase=2.55 + 0.25 * idx,
                topology_kind="grid" if idx % 2 == 0 else "line",
            )
        )

    # Compound shift episodes: simultaneous family, depth, and noise drift.
    target.extend(
        [
            EpisodeSpec(
                episode_id="target-compound-00",
                domain="target",
                shift_type="compound_shift",
                families=("mirror", "ghz", "parity"),
                qubit_range=(10, 26),
                depth_range=(26, 54),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.20,
                hidden_coherent_shift=0.011,
                hidden_crosstalk_shift=0.007,
                hidden_context_shift=0.005,
                hidden_temporal_shift=0.003,
                time_index=30,
                smooth_drift_phase=3.70,
                topology_kind="heavy_hex",
            ),
            EpisodeSpec(
                episode_id="target-compound-01",
                domain="target",
                shift_type="compound_shift",
                families=("mirror", "symmetry_fragment", "qaoa_like"),
                qubit_range=(9, 24),
                depth_range=(22, 48),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.16,
                hidden_coherent_shift=0.010,
                hidden_crosstalk_shift=0.008,
                hidden_context_shift=0.006,
                hidden_temporal_shift=0.004,
                time_index=31,
                smooth_drift_phase=3.95,
                topology_kind="ring",
            ),
            EpisodeSpec(
                episode_id="target-compound-02",
                domain="target",
                shift_type="compound_shift",
                families=("ghz", "symmetry_fragment", "random_entangler"),
                qubit_range=(12, 28),
                depth_range=(24, 58),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.24,
                hidden_coherent_shift=0.012,
                hidden_crosstalk_shift=0.009,
                hidden_context_shift=0.006,
                hidden_temporal_shift=0.004,
                time_index=32,
                smooth_drift_phase=4.20,
                topology_kind="line",
            ),
        ]
    )

    # Additional episodes to increase benchmark breadth.
    target.extend(
        [
            EpisodeSpec(
                episode_id="target-family-03",
                domain="target",
                shift_type="family_shift",
                families=("mirror", "ghz", "parity", "qaoa_like"),
                qubit_range=(9, 25),
                depth_range=(6, 30),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.03,
                hidden_coherent_shift=0.009,
                hidden_crosstalk_shift=0.004,
                hidden_context_shift=0.004,
                time_index=33,
                smooth_drift_phase=4.35,
                topology_kind="heavy_hex",
            ),
            EpisodeSpec(
                episode_id="target-depth-03",
                domain="target",
                shift_type="depth_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(11, 27),
                depth_range=(34, 64),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.00,
                hidden_coherent_shift=0.006,
                hidden_crosstalk_shift=0.003,
                hidden_context_shift=0.002,
                time_index=34,
                smooth_drift_phase=4.50,
                topology_kind="line",
            ),
            EpisodeSpec(
                episode_id="target-noise-03",
                domain="target",
                shift_type="noise_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(8, 24),
                depth_range=(6, 30),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.30,
                hidden_coherent_shift=0.011,
                hidden_crosstalk_shift=0.009,
                hidden_context_shift=0.006,
                hidden_temporal_shift=0.004,
                time_index=35,
                smooth_drift_phase=4.65,
                topology_kind="ring",
            ),
            EpisodeSpec(
                episode_id="target-temporal-04",
                domain="target",
                shift_type="temporal_shift",
                families=_SOURCE_FAMILIES,
                qubit_range=(9, 25),
                depth_range=(6, 32),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.28,
                hidden_coherent_shift=0.010,
                hidden_crosstalk_shift=0.008,
                hidden_context_shift=0.005,
                hidden_temporal_shift=0.005,
                time_index=36,
                smooth_drift_phase=4.80,
                topology_kind="grid",
            ),
            EpisodeSpec(
                episode_id="target-compound-03",
                domain="target",
                shift_type="compound_shift",
                families=("mirror", "symmetry_fragment", "parity", "qaoa_like"),
                qubit_range=(11, 28),
                depth_range=(28, 62),
                circuits_per_episode=circuits_per_target_episode,
                known_noise_scale=1.26,
                hidden_coherent_shift=0.013,
                hidden_crosstalk_shift=0.010,
                hidden_context_shift=0.007,
                hidden_temporal_shift=0.005,
                time_index=37,
                smooth_drift_phase=4.95,
                topology_kind="heavy_hex",
            ),
        ]
    )

    return tuple(target)



def build_profile_config(profile: str = "paper", *, random_seed: int = 20260308) -> BenchmarkRunConfig:
    """Build a standard experiment profile.

    Profiles:
        quick: small and fast smoke-test profile.
        paper: default paper-grade profile for the synthetic benchmark.
        full: larger profile for a final rerun.
    """
    normalized = profile.strip().lower()
    if normalized not in {"quick", "paper", "full"}:
        raise ValueError(f"Unknown profile: {profile}")

    if normalized == "quick":
        circuits_per_source_episode = 140
        circuits_per_target_episode = 96
        model = ModelConfig(n_members=24, max_depth=10, min_samples_leaf=2, bootstrap_fraction=0.85, n_jobs=1, random_seed=random_seed)
        adaptation = AdaptationConfig(
            budgets=(0, 4, 8, 16, 32),
            acquisition_budgets=(4, 8, 16),
            random_repeats=1,
            candidate_fraction=0.45,
            ridge_alpha=1.0,
            target_upweight=5.0,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.010,
            random_seed=random_seed,
        )
    elif normalized == "paper":
        circuits_per_source_episode = 280
        circuits_per_target_episode = 192
        model = ModelConfig(n_members=32, max_depth=12, min_samples_leaf=3, bootstrap_fraction=0.85, n_jobs=1, random_seed=random_seed)
        adaptation = AdaptationConfig(
            budgets=(0, 8, 16, 32, 64),
            acquisition_budgets=(4, 8, 16, 24),
            random_repeats=10,
            candidate_fraction=0.45,
            ridge_alpha=1.0,
            target_upweight=6.0,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.010,
            random_seed=random_seed,
        )
    else:
        circuits_per_source_episode = 420
        circuits_per_target_episode = 256
        model = ModelConfig(n_members=48, max_depth=14, min_samples_leaf=2, bootstrap_fraction=0.90, n_jobs=1, random_seed=random_seed)
        adaptation = AdaptationConfig(
            budgets=(0, 8, 16, 32, 64, 128),
            acquisition_budgets=(4, 8, 16, 24, 32),
            random_repeats=10,
            candidate_fraction=0.48,
            ridge_alpha=0.8,
            target_upweight=7.0,
            uncertainty_weight=1.0,
            threshold_weight=0.7,
            diversity_weight=0.35,
            threshold_beta=0.010,
            random_seed=random_seed,
        )

    source_episodes = _build_source_episodes(circuits_per_source_episode)
    target_episodes = _build_target_episodes(circuits_per_target_episode)
    if normalized == "quick":
        source_episodes = source_episodes[:2]
        target_episodes = (
            target_episodes[0],
            target_episodes[3],
            target_episodes[6],
            target_episodes[9],
            target_episodes[13],
        )
    synthetic = SyntheticBenchmarkConfig(
        random_seed=random_seed,
        pass_threshold=0.080,
        source_episodes=source_episodes,
        target_episodes=target_episodes,
    )
    calibration = CalibrationConfig(interval_alphas=(0.20, 0.10, 0.05), ece_bins=10, min_std=1e-4)
    evaluation = EvaluationConfig(bootstrap_resamples=800 if normalized != "quick" else 300, significance_alpha=0.05)
    reporting = ReportingConfig(save_raw_predictions=True, save_submission_bundle=True)
    return BenchmarkRunConfig(
        profile_name=normalized,
        synthetic=synthetic,
        model=model,
        adaptation=adaptation,
        calibration=calibration,
        evaluation=evaluation,
        reporting=reporting,
    )
