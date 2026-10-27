"""End-to-end timing sanity checks through the real ray-tracing pipeline.

These tests are slow (they build a detector Geometry and run the Taichi
ray-tracing kernel) and are therefore marked ``slow``.  Run them explicitly
with::

    pytest -m slow
"""

import numpy as np
import pytest

import taichi as ti

from litepath.tracer import (
    H_ARRIVAL, H_SOURCE,
    SOURCE_SCI,
    build_geometry,
    trace_muon_light,
)

pytestmark = pytest.mark.slow

# The GPU kernel must be initialised exactly like the CLI entry points
# (cli.py / viz_server.py), otherwise the taichi runtime errors out.
ti.init(arch=ti.cpu, default_fp=ti.f32)

# Two tau_fast values whose case we compare.  We compare two *scintillation*
# runs with identical origins/directions (same seed) but different decay
# constants; this cancels all geometry and propagation effects, isolating the
# decay delay exactly.
_TAU_A = 2.0
_TAU_B = 5.0
# Loose bounds: the mean arrival-time difference should be ~ (tau_b - tau_a),
# but detected-photon subsets add some variance, so allow generous slack.
_SHIFT_LO = 2.0
_SHIFT_HI = 4.0


@pytest.fixture(scope="module")
def geometry():
    """A fast detector geometry: tank + PMT positions, no GDML structure."""
    from pathlib import Path
    gdml = Path("PHASE2_INNER_STRUCTURE_closed.gdml")
    pmt_csv = Path("PMTPositions_Scan.txt")
    return build_geometry(
        gdml,
        pmt_csv_path=pmt_csv,
        no_gdml=True,        # skip structure mesh -> fast build
        no_pmt_holders=True, # skip PMT holder meshes -> fast
        no_lappd=True,       # skip LAPPD rectangles
        n_surfboards=0,
    )


def test_scintillation_decay_shifts_arrival_time(geometry):
    """Scintillation decay delay propagates into the final arrival time.

    We trace the same muon twice with pure scintillation: same origins and
    directions (same RNG seed, fast_fraction=1.0) but two different decay
    constants tau_fast.  Because origins/directions and hence the per-photon
    propagation time are identical in both runs, the only difference in the
    reported arrival_time must come from the decay delay, so::

        mean(arrival_time for tau_b) - mean(arrival_time for tau_a)
            ~ tau_b - tau_a

    This verifies, end-to-end through ``create_time + path/c_in_water`` in the
    ray tracer, that scintillation adds its decay on top of the muon arrival.
    """
    muon_pos = (0.0, 0.0, 600.0)   # mm, mid-tank
    muon_dir = (0.0, 0.0, -1.0)
    per_cm = 200  # enough detected hits to make the mean stable

    runs = {}
    for tau, seed in ((_TAU_A, 123), (_TAU_B, 123)):
        rng = np.random.default_rng(seed)
        full = trace_muon_light(
            muon_pos, muon_dir, photons_per_cm=0, geometry=geometry,
            rng=rng, scintillation_enabled=True,
            photons_per_cm_scint=per_cm,
            fast_fraction=1.0, tau_fast=tau,
        )
        hits = full[full[:, H_SOURCE] == SOURCE_SCI, H_ARRIVAL]
        hits = hits[hits > 0]  # keep only detected photons (arrival_time set)
        runs[tau] = hits
        assert len(hits) > 10, f"expected detected scintillation hits (tau={tau})"

    mean_a = float(np.mean(runs[_TAU_A]))
    mean_b = float(np.mean(runs[_TAU_B]))
    shift = mean_b - mean_a

    # The larger decay constant must push arrivals later.
    assert shift > 0.0
    # ... and the shift must be consistent with the delay difference.
    assert _SHIFT_LO < shift < _SHIFT_HI, (
        f"mean arrival shift {shift:.3f} ns not in "
        f"({_SHIFT_LO}, {_SHIFT_HI}); expected ~{_TAU_B - _TAU_A} ns "
        f"(decay-delay difference)"
    )
