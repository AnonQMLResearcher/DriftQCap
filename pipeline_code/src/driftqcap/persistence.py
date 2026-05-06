"""Persistence helpers for reusable fitted source artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import BenchmarkRunConfig


def _normalize_df_for_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ordered_columns = sorted(out.columns.tolist())
    return out.loc[:, ordered_columns].sort_values(by=ordered_columns[: min(3, len(ordered_columns))]).reset_index(drop=True)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Build a stable fingerprint for a dataframe."""
    normalized = _normalize_df_for_fingerprint(df)
    hashed = pd.util.hash_pandas_object(normalized, index=True).to_numpy()
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return digest


def config_fingerprint(config: BenchmarkRunConfig) -> str:
    payload = {
        "profile_name": config.profile_name,
        "model": asdict(config.model),
        "calibration": asdict(config.calibration),
        "adaptation": asdict(config.adaptation),
        "pass_threshold": config.synthetic.pass_threshold,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_source_artifacts(
    *,
    output_dir: str | Path,
    config: BenchmarkRunConfig,
    source_train_df: pd.DataFrame,
    base_model: Any,
    coarse_model: Any,
) -> dict[str, str]:
    """Persist source-domain models and compatibility metadata."""
    output_dir = Path(output_dir)
    artifact_dir = output_dir / config.reporting.artifacts_dirname
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = artifact_dir / "source_model.joblib"
    coarse_model_path = artifact_dir / "coarse_source_model.joblib"
    manifest_path = artifact_dir / "model_manifest.json"

    joblib.dump(base_model, base_model_path)
    joblib.dump(coarse_model, coarse_model_path)

    manifest = {
        "config_fingerprint": config_fingerprint(config),
        "source_train_fingerprint": dataframe_fingerprint(source_train_df),
        "profile_name": config.profile_name,
        "artifact_version": 1,
        "base_model_path": str(base_model_path.name),
        "coarse_model_path": str(coarse_model_path.name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "base_model": str(base_model_path),
        "coarse_model": str(coarse_model_path),
        "manifest": str(manifest_path),
    }


def load_source_artifacts(
    *,
    output_dir: str | Path,
    config: BenchmarkRunConfig,
    source_train_df: pd.DataFrame,
) -> tuple[Any | None, Any | None, dict[str, str]]:
    """Load compatible source-domain models if available."""
    output_dir = Path(output_dir)
    artifact_dir = output_dir / config.reporting.artifacts_dirname
    manifest_path = artifact_dir / "model_manifest.json"
    base_model_path = artifact_dir / "source_model.joblib"
    coarse_model_path = artifact_dir / "coarse_source_model.joblib"

    paths = {
        "artifact_dir": str(artifact_dir),
        "base_model": str(base_model_path),
        "coarse_model": str(coarse_model_path),
        "manifest": str(manifest_path),
    }
    if not manifest_path.exists() or not base_model_path.exists() or not coarse_model_path.exists():
        return None, None, paths

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = config_fingerprint(config)
    expected_source = dataframe_fingerprint(source_train_df)
    if manifest.get("config_fingerprint") != expected_config:
        return None, None, paths
    if manifest.get("source_train_fingerprint") != expected_source:
        return None, None, paths

    base_model = joblib.load(base_model_path)
    coarse_model = joblib.load(coarse_model_path)
    return base_model, coarse_model, paths
