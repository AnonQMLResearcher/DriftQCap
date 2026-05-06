"""DriftQCap research code package.

This package provides a reproducible synthetic benchmark and experiment suite
for drift-aware few-shot quantum capability learning under distribution shift.
"""

from .config import (
    AdaptationConfig,
    BenchmarkRunConfig,
    CalibrationConfig,
    EvaluationConfig,
    ModelConfig,
    ReportingConfig,
    SyntheticBenchmarkConfig,
    build_profile_config,
)
from .paper_utils import build_readiness_report, export_standard_latex_tables
from .pipeline import BenchmarkArtifacts, run_benchmark

__all__ = [
    "AdaptationConfig",
    "BenchmarkArtifacts",
    "BenchmarkRunConfig",
    "CalibrationConfig",
    "EvaluationConfig",
    "ModelConfig",
    "ReportingConfig",
    "SyntheticBenchmarkConfig",
    "build_profile_config",
    "build_readiness_report",
    "export_standard_latex_tables",
    "run_benchmark",
]
