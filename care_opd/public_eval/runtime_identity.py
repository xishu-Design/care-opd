"""Small, CUDA-safe runtime identity used by formal public evaluators."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from typing import Any


PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "tokenizers",
    "peft",
)


def runtime_identity() -> dict[str, Any]:
    """Record inference-library versions without importing CUDA frameworks."""

    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
