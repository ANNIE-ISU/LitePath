"""Choosing the Taichi backend (CPU or GPU) for every command.

The backend comes from ``--arch`` on the command line, else the
``LITEPATH_ARCH`` environment variable, else the default, ``cpu``:

``cpu``     the CPU (Windows, macOS, Linux; x86-64 and ARM)
``gpu``     the best GPU Taichi finds (CUDA, Vulkan or Metal), else the CPU
``metal``   Apple GPUs (macOS)
``cuda``    NVIDIA GPUs (Linux, Windows)
``vulkan``  Vulkan GPUs (Linux, Windows; macOS through MoltenVK)

If the requested GPU backend is not available, Taichi falls back to the
CPU; :func:`init` reports the backend actually in use.  The CPU is the
reference: results on a GPU agree statistically, not bit for bit (float
arithmetic and the order of parallel additions differ).
"""
from __future__ import annotations

import os

ARCH_CHOICES = ("cpu", "gpu", "metal", "cuda", "vulkan")
DEFAULT_ARCH = "cpu"
ENV_VAR = "LITEPATH_ARCH"


def requested_arch(arch: str | None = None) -> str:
    """The backend asked for: *arch*, else $LITEPATH_ARCH, else the default."""
    name = (arch or os.environ.get(ENV_VAR) or DEFAULT_ARCH).strip().lower()
    if name not in ARCH_CHOICES:
        raise ValueError(f"unknown backend {name!r}; choose one of {', '.join(ARCH_CHOICES)}")
    return name


def init(arch: str | None = None, quiet: bool = False) -> str:
    """``ti.init`` on the chosen backend; returns the backend in use (e.g. ``"metal"``)."""
    import taichi as ti

    name = requested_arch(arch)
    ti.init(arch=getattr(ti, name), default_fp=ti.f32)
    actual = active_arch()
    if not quiet:
        note = "" if actual == name or name == "gpu" else f" ({name} not available)"
        print(f"Taichi backend: {actual}{note}")
    return actual


def active_arch() -> str:
    """Name of the backend Taichi is running on (``x64``/``arm64`` are reported as ``cpu``)."""
    import taichi as ti

    arch = ti.lang.impl.current_cfg().arch
    name = getattr(arch, "name", str(arch)).split(".")[-1]
    return "cpu" if name in ("x64", "arm64") else name
