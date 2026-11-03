"""Backend selection (litepath/backend.py) and the CPU/GPU agreement check."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from litepath.backend import ARCH_CHOICES, ENV_VAR, requested_arch

REPO = Path(__file__).resolve().parents[1]
MUONS = "tests/reference/inputs/muons3.txt"


def test_requested_arch(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert requested_arch() == "cpu"
    monkeypatch.setenv(ENV_VAR, "Metal")
    assert requested_arch() == "metal"
    assert requested_arch("cuda") == "cuda"        # --arch wins over the environment
    with pytest.raises(ValueError, match="unknown backend"):
        requested_arch("tpu")
    assert set(ARCH_CHOICES) == {"cpu", "gpu", "metal", "cuda", "vulkan"}


def test_kernel_commands_take_arch():
    from litepath.cli import build_parser
    p = build_parser()
    for argv in (["batch"], ["batch-lappd"], ["fit", "x.h5"], ["viz-server"], ["viz-lappd"], ["geometry-viz"]):
        assert p.parse_args([*argv, "--arch", "gpu"]).arch == "gpu"


_RUN = """
import json, sys
from litepath import backend
arch = backend.init(sys.argv[1], quiet=True)
print(json.dumps({"arch": arch}))
"""


def _batch(tmp_path, arch):
    out = tmp_path / arch
    cmd = [sys.executable, "-m", "litepath", "batch", "--arch", arch, "--geometry", "annie-2026-data",
           "--muon-file", MUONS, "--photons-per-cm", "150", "--seed", "5", "-o", str(out)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    from litepath.io_h5 import load_table
    return load_table(out / "output.h5", "photon_hits")


@pytest.mark.gpu
def test_gpu_agrees_with_cpu(tmp_path):
    """Same muons and seed on the CPU and the GPU: the same light, to statistics.

    GPU float arithmetic differs in the last bits, so a few grazing photons
    change; totals and per-detector counts must agree closely.
    """
    probe = subprocess.run([sys.executable, "-c", _RUN, "gpu"], cwd=REPO, capture_output=True, text=True)
    arch = json.loads(probe.stdout.strip().splitlines()[-1])["arch"]
    if arch == "cpu":
        pytest.skip("no GPU backend available")
    cpu, gpu = _batch(tmp_path, "cpu"), _batch(tmp_path, "gpu")
    assert abs(len(gpu) - len(cpu)) <= max(5, 0.002 * len(cpu)), (len(cpu), len(gpu))
    key = ["detector_system", "detector_index"]
    a = cpu.groupby(key).size()
    b = gpu.groupby(key).size().reindex(a.index.union(gpu.groupby(key).size().index), fill_value=0)
    a = a.reindex(b.index, fill_value=0)
    # per detector: within a few photons, far below the Poisson spread
    assert (np.abs(a - b) <= 3 + 0.01 * a).all(), (a - b)[np.abs(a - b) > 3 + 0.01 * a]
    assert abs(np.median(cpu.arrival_time) - np.median(gpu.arrival_time)) < 0.01   # ns
