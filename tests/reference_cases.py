"""Reference runs that pin LitePath's end-to-end output.

Each case runs real CLI commands in a fresh process (so every run gets its
own ``ti.init``) and reduces the outputs to plain named arrays.
``tests/test_reference_outputs.py`` re-runs every case and compares against
the stored copies in ``tests/reference/``.

Rebuild the stored copies with ``python tools/make_reference_outputs.py``
only when a change to the output is intended, and say so in the commit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO / "tests" / "reference"

# Paths are relative to the repo root so they read the same in every run's
# stored config.
MUONS = "tests/reference/inputs/muons3.txt"
FULL = ["--pmt-csv", "PMTPositions_Scan.txt", "--surfboard", "3"]
FULL_MUONS = ["batch", *FULL, "--muon-file", MUONS, "--seed", "7", "-o", "{out}"]

# LAPPD digitisation sums currents with parallel atomic adds, so the order of
# float additions (and the last bits of the result) can vary between runs.
LOOSE_PREFIXES = ("h5/lappd_readouts/",)
LOOSE_RTOL = 1e-5


@dataclass(frozen=True)
class Case:
    name: str
    commands: tuple[tuple[str, ...], ...]  # litepath CLI argv; "{out}" = output dir
    slow: bool = True                       # builds the full detector
    needs: tuple[str, ...] = ()             # files that must exist (else skip)


CASES = [
    Case("full_muons", (
        (*FULL_MUONS, "--photons-per-cm", "30", "--pmt-response", "--lappd-response"),
    )),
    Case("full_burst", (
        ("batch", *FULL, "--light-burst", "--burst-n-phots", "20000", "--events", "2",
         "--pmt-response", "--seed", "7", "-o", "{out}"),
    )),
    Case("full_optics", (
        (*FULL_MUONS, "--photons-per-cm", "10", "--max-bounces", "2",
         "--water-absorption-mm", "30000", "--water-scattering-mm", "20000"),
    )),
    Case("full_scint", (
        (*FULL_MUONS, "--photons-per-cm", "20", "--scintillation",
         "--photons-per-cm-scint", "20"),
    )),
    Case("full_bare_lappd", (
        ("batch", "--pmt-csv", "PMTPositions_Scan.txt", "--manifest", "component_manifest.json",
         "--lappd-model", "default", "--muon-file", MUONS, "--photons-per-cm", "30",
         "--seed", "7", "-o", "{out}"),
    ), needs=("component_manifest.json",)),
    Case("fit", (
        (*FULL_MUONS, "--photons-per-cm", "30"),
        ("fit", "{out}/output.h5", *FULL, "--event", "0", "--grid-theta", "85 95 2",
         "--grid-phi", "130 140 2", "--seed", "3", "--save-grid", "{out}/fit_grid.npz"),
    )),
    Case("lappd_cherenkov", (
        ("batch-lappd", "--source", "cherenkov", "--sample-mode", "grid", "--grid-steps", "2", "2",
         "--photons-per-cm", "30", "--lappd-response", "--seed", "11", "-o", "{out}"),
    ), slow=False),
    Case("lappd_spot", (
        ("batch-lappd", "--source", "spot", "--n-photons", "3000", "--spread-deg", "2",
         "--events", "2", "--lappd-response", "--seed", "11", "-o", "{out}"),
    ), slow=False),
]


def missing_inputs(case: Case) -> list[str]:
    return [p for p in case.needs if not (REPO / p).exists()]


def run_case(case: Case, out_dir: Path) -> dict[str, np.ndarray]:
    """Run *case*'s commands into *out_dir* and return its outputs as arrays."""
    for cmd in case.commands:
        argv = [arg.format(out=out_dir) for arg in cmd]
        proc = subprocess.run([sys.executable, "-m", "litepath", *argv],
                              cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{case.name}: `litepath {' '.join(argv)}` failed\n"
                               f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return collect_outputs(out_dir)


def collect_outputs(out_dir: Path) -> dict[str, np.ndarray]:
    """Flatten output.h5 (one array per table column) and any .npz files."""
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(out_dir / "output.h5", "r") as f:
        for table, ds in f.items():
            data = ds[()]
            for col in data.dtype.names:
                values = data[col]
                if values.dtype == object:
                    values = np.array([v.decode() if isinstance(v, bytes) else str(v)
                                       for v in values])
                arrays[f"h5/{table}/{col}"] = values
        # 'created' is a timestamp; the output dir differs per run.
        attrs = {k: v for k, v in f.attrs.items() if k != "created"}
        config = json.loads(attrs.pop("config"))
        config.pop("output_dir", None)
        attrs["config"] = config
        arrays["h5/attrs"] = np.array(json.dumps(attrs, sort_keys=True, default=str))
    for npz in sorted(out_dir.glob("*.npz")):
        with np.load(npz) as data:
            for key in data.files:
                arrays[f"{npz.stem}/{key}"] = data[key]
    return arrays


def compare(reference: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> list[str]:
    """Return a description of every difference (empty list = identical)."""
    problems = []
    for key in sorted(reference.keys() - new.keys()):
        problems.append(f"{key}: missing from new output")
    for key in sorted(new.keys() - reference.keys()):
        problems.append(f"{key}: not in reference")
    for key in sorted(reference.keys() & new.keys()):
        ref, got = reference[key], new[key]
        if ref.shape != got.shape:
            problems.append(f"{key}: shape {ref.shape} -> {got.shape}")
        elif ref.dtype.kind == "f" and key.startswith(LOOSE_PREFIXES):
            atol = LOOSE_RTOL * float(np.nanmax(np.abs(ref))) if ref.size else 0.0
            if not np.allclose(ref, got, rtol=LOOSE_RTOL, atol=atol, equal_nan=True):
                problems.append(f"{key}: max |diff| {np.nanmax(np.abs(ref - got)):.3g}")
        elif ref.dtype.kind == "f":
            if not np.array_equal(ref, got, equal_nan=True):
                n = int(np.sum(~((ref == got) | (np.isnan(ref) & np.isnan(got)))))
                problems.append(f"{key}: {n} of {ref.size} values differ")
        elif not np.array_equal(ref, got):
            problems.append(f"{key}: values differ")
    return problems
