"""Output for the detector registry (YAML)."""

from __future__ import annotations

from pathlib import Path


def write_detector_config(detectors: list, path: Path) -> None:
    """Write detector registry to YAML."""
    from litepath.detectors import detector_config_to_yaml
    detector_config_to_yaml(detectors, path)
