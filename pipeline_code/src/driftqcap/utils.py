"""Miscellaneous helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable



def stable_int_hash(*parts: object, modulus: int = 2**32 - 1) -> int:
    """Return a stable non-negative integer hash from arbitrary parts."""
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulus



def write_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")



def collect_environment_metadata(packages: Iterable[str] = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib")) -> dict[str, object]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }
