"""Light is generated only on the part of a muon track inside the tank.

The track length is a simulation cutoff, not a physical range: surfboards
and LAPPD housings must not end it, and it must not extend past the tank
wall.  In the standalone LAPPD geometry light stops once the muon is behind
the housing's back face, or at the world boundary if it never gets there.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from litepath.geometry.build import build_geometry, build_single_lappd_geometry
from litepath.sources.track import (
    STANDALONE_MAX_TRACK_M, advance_muon, compute_track_length, light_segment,
)

R, Z_MIN, Z_MAX = 1524.0, 19.0, 3861.0
C_MUON = 0.999999 * 299792458.0


@pytest.fixture(scope="module")
def full_geometry():
    """Full detector with three surfboards and housed LAPPDs (no meshes)."""
    geo = build_geometry(
        Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
        pmt_csv_path=Path("PMTPositions_Scan.txt"),
        no_gdml=True, no_pmt_holders=True,
        lappd_model="annie", n_surfboards=3,
    )
    assert geo.surfboard_data.shape[0] == 3 and geo.lappd_housing_data.shape[0] == 3
    assert (geo.tank_radius, geo.tank_z_min, geo.tank_z_max) == (R, Z_MIN, Z_MAX)
    return geo


@pytest.mark.parametrize("pos, direction, expected_m", [
    ((0, 0, 300), (0, 0, -1), (300 - Z_MIN) / 1000),          # short, downward
    ((0, 0, 3500), (0, 0, 1), (Z_MAX - 3500) / 1000),          # short, upward
    ((0, 0, 1940), (1, 0, 0), R / 1000),                       # misses every surfboard
    ((0, 0, 1940), (-math.sqrt(0.5), math.sqrt(0.5), 0), R / 1000),  # beam: through a surfboard
])
def test_track_runs_to_tank_wall(full_geometry, pos, direction, expected_m):
    start, length = light_segment(pos, direction, full_geometry)
    assert start == 0.0
    assert length == pytest.approx(expected_m, abs=1e-9)
    assert compute_track_length(pos, direction, full_geometry) == length


def test_muon_starting_outside_tank_emits_from_entry(full_geometry):
    start, length = light_segment((-2000.0, 0.0, 1940.0), (1.0, 0.0, 0.0), full_geometry)
    assert start == pytest.approx((2000.0 - R) / 1000)
    assert length == pytest.approx(2 * R / 1000)


@pytest.mark.parametrize("pos, direction", [
    ((-2000.0, 0.0, 1940.0), (-1.0, 0.0, 0.0)),  # outside, heading away
    ((2000.0, 0.0, 1940.0), (0.0, 0.0, 1.0)),    # outside radially, vertical
    ((0.0, 0.0, 5000.0), (1.0, 0.0, 0.0)),       # above the tank, horizontal
])
def test_muon_that_never_enters_tank_has_no_light(full_geometry, pos, direction):
    assert light_segment(pos, direction, full_geometry) == (0.0, 0.0)


@pytest.fixture(scope="module")
def lappd_geometry():
    """Standalone module at the origin, front face +z, back face at z = -30 mm."""
    geo = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))
    hd = geo.lappd_housing_data[0]
    assert tuple(hd[9:12]) == (0, 0, 1) and hd[2] - hd[14] == -30.0
    return geo


@pytest.mark.parametrize("pos, direction, expected_m", [
    ((10.0, -20.0, 300.0), (0.0, 0.0, -1.0), 0.330),                  # through the module
    ((500.0, 0.0, 300.0), (0.0, 0.0, -1.0), 0.330),                   # beside it: same plane
    ((0.0, 0.0, 300.0), (math.sqrt(0.5), 0.0, -math.sqrt(0.5)), 0.330 * math.sqrt(2)),
    ((0.0, 0.0, 300.0), (0.0, 0.0, 1.0), STANDALONE_MAX_TRACK_M),     # heading away
    ((0.0, 0.0, 300.0), (1.0, 0.0, 0.0), STANDALONE_MAX_TRACK_M),     # parallel to the plane
    ((0.0, 0.0, 9000.0), (0.0, 0.0, -1.0), STANDALONE_MAX_TRACK_M),   # plane beyond the boundary
    ((0.0, 0.0, -100.0), (0.0, 0.0, -1.0), 0.0),                      # starts behind
    ((0.0, 0.0, -100.0), (0.0, 0.0, 1.0), 0.0),                       # starts behind, heading in
])
def test_standalone_lappd_stops_at_back_plane(lappd_geometry, pos, direction, expected_m):
    assert light_segment(pos, direction, lappd_geometry) == (0.0, pytest.approx(expected_m))


def test_standalone_world_boundary_is_settable(lappd_geometry):
    assert compute_track_length((0, 0, 300), (0, 0, 1), lappd_geometry, 1.5) == 1.5
    assert compute_track_length((0, 0, 300), (0, 0, -1), lappd_geometry, 0.1) == 0.1


def test_tank_ignores_world_boundary(full_geometry):
    assert light_segment((0, 0, 1940), (1, 0, 0), full_geometry, 0.1) == (0.0, pytest.approx(R / 1000))


def test_advance_muon_moves_start_and_time():
    x, y, z, t0 = advance_muon((-2000.0, 5.0, 1940.0, 1e-6), (1.0, 0.0, 0.0), 0.476)
    assert (x, y, z) == pytest.approx((-1524.0, 5.0, 1940.0))
    assert t0 == pytest.approx(1e-6 + 0.476 / C_MUON)


@pytest.mark.slow
def test_batch_generates_light_only_inside_tank(full_geometry, tmp_path):
    """A muon from outside gets light from the entry on; one that misses gets none."""
    import taichi as ti
    from litepath.batch import BatchConfig, run_batch
    from litepath.io_h5 import BatchReader

    ti.init(arch=ti.cpu, default_fp=ti.f32)
    muons = tmp_path / "muons.txt"
    muons.write_text("-2000 0 1940 0 1 0 0\n-2000 0 1940 0 -1 0 0\n")
    run_batch(full_geometry, BatchConfig(
        muon_file=muons, n_events=2, photons_per_cm=5, seed=3, output_dir=tmp_path,
    ))
    truth = BatchReader(tmp_path / "output.h5").load_table("muon_truth").set_index("event_id")

    assert truth.loc[0, "track_length_mm"] == pytest.approx(2 * R, rel=1e-6)
    assert truth.loc[0, "pos_x"] == -2000.0          # truth keeps the real start point
    assert truth.loc[0, "n_generated"] == (int(2 * R / 10) + 1) * 5
    assert truth.loc[1, "track_length_mm"] == 0.0
    assert truth.loc[1, "n_generated"] == 0
