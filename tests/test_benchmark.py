"""Tests for the WAGLE ROOT → litepath HDF5 conversion (``litepath.benchmark``).

The pure-python parts (frame transforms, angle conventions, PMT registry) are
tested here; the ROOT parsing itself is covered by a slow test that needs the
benchmarking data files.
"""

import math

import numpy as np
import pytest

from litepath.benchmark import (
    DET_ROTATION_DEG,
    HALF_WC_LENGTH,
    ROOT_TANK_AXIS_Y_MAX,
    ROOT_TANK_AXIS_Y_MIN,
    TANK_RADIUS_MM,
    TANK_Z_MAX_MM,
    TANK_Z_MIN_MM,
    PMTRegistry,
    _tank_chord_root,
    litepath_angles,
    load_pmt_registry,
    root_angles,
    root_to_world,
    root_to_world_direction,
    world_to_root,
    world_to_root_direction,
)


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------


def test_world_to_root_is_inverse_of_root_to_world():
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=1000.0, size=(200, 3))
    back = world_to_root(root_to_world(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_direction_transform_is_rotation_only():
    """Directions must not pick up the translation."""
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(100, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    out = root_to_world_direction(dirs)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)
    # Rotating a direction must equal rotating the same vector with an
    # arbitrary origin offset removed.
    assert np.allclose(world_to_root_direction(out), dirs, atol=1e-9)


def test_root_to_world_translation_is_tank_centre():
    """The ROOT origin (tank centre) maps to the structure-frame origin.

    The litepath world frame puts the structure-frame origin 1980 mm below
    the tank centre, i.e. at z = 1980 mm.
    """
    out = root_to_world(np.zeros(3))
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(0.0, abs=1e-6)
    # The WCSim tank centre sits 1980 mm above the structure-frame origin.
    assert out[2] == pytest.approx(1980.0, abs=1e-3)
    # The water tank spans 19..3861 mm, so its mid-plane is 40 mm lower than
    # the WCSim tank centre — a known 40 mm offset of the two conventions.
    assert out[2] - (TANK_Z_MIN_MM + TANK_Z_MAX_MM) / 2 == pytest.approx(40.0, abs=1e-3)


def test_transform_is_orthogonal():
    """The linear part of the transform is a pure rotation."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=3)
    b = rng.normal(size=3)
    ra = root_to_world_direction(a)
    rb = root_to_world_direction(b)
    assert float(ra @ rb) == pytest.approx(float(a @ b), rel=1e-9)


# ---------------------------------------------------------------------------
# Angle conventions
# ---------------------------------------------------------------------------


def test_litepath_angles_convention():
    """theta = angle from -z (0 = straight down); phi = atan2(y, x)."""
    theta, phi = litepath_angles((0.0, 0.0, -1.0))
    assert theta == pytest.approx(0.0, abs=1e-6)
    assert phi == pytest.approx(0.0, abs=1e-6)

    theta, phi = litepath_angles((0.0, 0.0, 1.0))
    assert theta == pytest.approx(180.0, abs=1e-6)

    theta, phi = litepath_angles((1.0, 0.0, 0.0))
    assert theta == pytest.approx(90.0, abs=1e-6)
    assert phi == pytest.approx(0.0, abs=1e-6)

    theta, phi = litepath_angles((0.0, 1.0, 0.0))
    assert phi == pytest.approx(90.0, abs=1e-6)


def test_root_angles_convention():
    """ROOT/WAGLE: theta = angle from +z (the beam axis)."""
    theta, phi = root_angles((0.0, 0.0, 1.0))
    assert theta == pytest.approx(0.0, abs=1e-6)
    theta, phi = root_angles((0.0, 0.0, -1.0))
    assert theta == pytest.approx(180.0, abs=1e-6)
    theta, phi = root_angles((1.0, 0.0, 0.0))
    assert theta == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PMT registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    from pathlib import Path
    return load_pmt_registry(Path("PMTPositions_Scan.txt"),
                              det_rotation_deg=DET_ROTATION_DEG)


def test_registry_has_132_pmts(registry: PMTRegistry):
    assert len(registry.tube_id) == 132
    assert len(registry.positions) == 132
    assert set(registry.index) == set(range(132))


def test_registry_matches_pmt_loader(registry: PMTRegistry):
    """The registry must agree with litepath.pmt_loader (detector indices)."""
    from pathlib import Path
    from litepath.pmt_loader import load_pmts

    pmt = load_pmts(Path("PMTPositions_Scan.txt"),
                    det_rotation_deg=DET_ROTATION_DEG)
    assert list(registry.tube_id) == list(pmt["detector_nums"])
    assert np.allclose(registry.positions,
                       np.asarray(pmt["centers"], dtype=float), atol=1e-3)


def test_registry_pmts_lie_inside_the_tank(registry: PMTRegistry):
    r = np.linalg.norm(registry.positions[:, :2], axis=1)
    assert np.all(r < TANK_RADIUS_MM)
    assert np.all(registry.positions[:, 2] > TANK_Z_MIN_MM - 200.0)
    assert np.all(registry.positions[:, 2] < TANK_Z_MAX_MM + 200.0)


def test_registry_lookup_roundtrip(registry: PMTRegistry):
    tube = int(registry.tube_id[7])
    assert registry.index_of(tube) == 7
    assert int(registry.index_array(np.array([tube]))[0]) == 7
    assert int(registry.index_array(np.array([-1]))[0]) == -1


def test_detectors_table(registry: PMTRegistry):
    table = registry.detectors_table()
    assert len(table) == 132
    assert set(table.dtype.names) >= {
        "system_code", "detector_index", "x", "y", "z", "label", "panel"}
    assert np.all(table["system_code"] == 0)


# ---------------------------------------------------------------------------
# ROOT parsing (needs the benchmarking data files)
# ---------------------------------------------------------------------------

DATA_ROOT = "benchmarking_data/MCTree_ThroughgoingMuons.root"
DATA_H5 = "results/benchmark/mc.h5"


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATA_ROOT).exists(),
    reason="WAGLE MC ROOT file not available",
)
def test_root_reader_reads_mc_tree():
    from litepath.benchmark import RootTreeReader

    reader = RootTreeReader(DATA_ROOT)
    assert reader.has_truth
    assert reader.num_entries > 0

    ev = next(reader.iter_events())
    assert ev.truth.kind == 0            # generator truth
    assert ev.truth.dir_root is not None
    # The world-frame vertex must sit inside the tank.
    pos = ev.truth.pos_world_mm()
    assert math.hypot(pos[0], pos[1]) < TANK_RADIUS_MM
    assert TANK_Z_MIN_MM < pos[2] < TANK_Z_MAX_MM


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATA_H5).exists(),
    reason="converted benchmark file not present",
)
def test_converted_file_has_expected_tables():
    from litepath.io_h5 import BatchReader

    br = BatchReader(DATA_H5)
    tables = br.summary()["tables"]
    for name in ("muon_truth", "pmt_responses", "detectors"):
        assert name in tables
    hits = br.load_table("pmt_hits")
    assert not hits.empty
    # A handful of WAGLE hits carry a slightly negative time (0.02 %).
    times = hits["arrival_time"].to_numpy()
    assert (times > 0).mean() > 0.99


# ---------------------------------------------------------------------------
# Cross-checks between the WAGLE data and the raytracer geometry
# ---------------------------------------------------------------------------

ROOT_AVAILABLE = __import__("pathlib").Path(DATA_ROOT).exists()


@pytest.fixture(scope="module")
def mc_hits():
    """The WAGLE MC hits, as (tube_id, position) pairs in the world frame."""
    from litepath.benchmark import RootTreeReader, root_to_world

    reader = RootTreeReader(DATA_ROOT)
    positions = {}
    tube_ids = set()
    for ev in reader.iter_events():
        world = root_to_world(np.asarray(ev.positions_root_m, float) * 1000.0)
        for k, tube in enumerate(ev.tube_ids):
            tube = int(tube)
            tube_ids.add(tube)
            positions.setdefault(tube, []).append(world[k])
    return positions, tube_ids


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="WAGLE MC ROOT file not available")
def test_hit_ids_are_a_subset_of_the_registry(mc_hits, registry: PMTRegistry):
    """Every WAGLE PMT id must resolve to a raytracer PMT (1:1 mapping)."""
    _positions, tube_ids = mc_hits
    scan_ids = set(int(t) for t in registry.tube_id)
    assert tube_ids <= scan_ids
    assert len(tube_ids) == 121
    for tube in tube_ids:
        assert registry.index_of(tube) >= 0


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="WAGLE MC ROOT file not available")
def test_wagle_stores_one_point_per_pmt(mc_hits):
    """WAGLE hit positions are the PMT centre, not per-photon positions.

    Raw WCSim keeps the PMT geometry in a separate tree and looks it up by
    Tube ID, so every hit of a given PMT carries the same point.
    """
    positions, _tube_ids = mc_hits
    for tube, pts in positions.items():
        pts = np.asarray(pts)
        assert np.abs(pts - pts.mean(axis=0)).max() < 1e-3, (
            f"PMT {tube} has more than one hit position")


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="WAGLE MC ROOT file not available")
def test_pmt_positions_agree_with_the_raytracer(mc_hits, registry: PMTRegistry):
    """The transformed WAGLE PMT point must coincide with the raytracer one.

    The residual is a constant offset *along the tube axis* (the WCSim
    PMT centre sits further out than the survey point); everything
    perpendicular to the axis must agree to < 0.3 mm.
    """
    positions, _tube_ids = mc_hits
    per_type: dict[str, list[float]] = {}
    for tube, pts in positions.items():
        index = registry.index_of(int(tube))
        p = np.asarray(pts).mean(axis=0)
        c = registry.positions[index]
        u = registry.directions[index]
        d = p - c
        axial = float(d @ u)
        perp = float(np.linalg.norm(d - axial * u))
        assert perp < 0.3, f"PMT {tube}: perpendicular residual {perp:.3f} mm"
        # The offset points outwards, i.e. against the inward normal.
        assert axial < 0.0
        per_type.setdefault(registry.types[index], []).append(abs(axial))
    # Constant per PMT type.
    for pmt_type, vals in per_type.items():
        assert np.std(vals) < 0.1, f"{pmt_type}: offset spread {np.std(vals):.3f}"
    assert abs(np.mean(per_type["LUX"]) - 10.42) < 0.3
    assert abs(np.mean(per_type["ETEL"]) - 23.72) < 0.3


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="WAGLE MC ROOT file not available")
def test_beam_azimuth_is_135_degrees():
    """In the world frame the WAGLE beam runs at phi = 135 deg.

    135 deg is a corner of the inner-structure octagon.  The litepath
    ``--muon-mode beam`` helper uses +Y (90 deg), which is 45 deg away.
    """
    from litepath.benchmark import RootTreeReader, root_to_world_direction

    reader = RootTreeReader(DATA_ROOT)
    phis = []
    for ev in reader.iter_events():
        d = ev.truth.dir_root
        if d is None:
            continue
        w = root_to_world_direction(np.asarray(d, float))
        if np.hypot(w[0], w[1]) < 1e-6:
            continue
        phis.append(math.degrees(math.atan2(w[1], w[0])) % 360.0)
    assert len(phis) > 100
    ang = np.radians(np.asarray(phis))
    mean_phi = math.degrees(math.atan2(float(np.sin(ang).mean()),
                                       float(np.cos(ang).mean()))) % 360.0
    assert mean_phi == pytest.approx(135.0, abs=2.0)


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="WAGLE MC ROOT file not available")
def test_world_frame_keeps_the_tank_vertical():
    """The y/z swap must not turn the beam vertical in the world frame."""
    from litepath.benchmark import RootTreeReader, root_to_world_direction

    reader = RootTreeReader(DATA_ROOT)
    dz = []
    for ev in reader.iter_events():
        d = ev.truth.dir_root
        if d is None:
            continue
        w = root_to_world_direction(np.asarray(d, float))
        dz.append(abs(w[2]))
    assert np.mean(dz) < 0.2, "tracks should be horizontal in the world frame"


# ---------------------------------------------------------------------------
# MRD track reconstruction (data sample): tank-chord vertex
# ---------------------------------------------------------------------------


def test_tank_chord_root_barrel_crossing():
    """A line through the tank axis crosses the barrel at ±R (root z here)."""
    entry, exit_ = _tank_chord_root(np.array([0.0, 0.0, 3500.0]),
                                    np.array([0.0, 0.0, 1.0]))
    assert entry[0] == pytest.approx(0.0, abs=1e-6)
    assert entry[1] == pytest.approx(0.0, abs=1e-6)
    assert entry[2] == pytest.approx(-TANK_RADIUS_MM, abs=1e-6)
    assert exit_[2] == pytest.approx(TANK_RADIUS_MM, abs=1e-6)
    chord = np.linalg.norm(exit_ - entry) / 1000.0
    assert chord == pytest.approx(2.0 * TANK_RADIUS_MM / 1000.0, rel=1e-9)


def test_tank_chord_root_miss_returns_none():
    # Offset by 3.5 m radially from the axis, travelling sideways.
    assert _tank_chord_root(np.array([0.0, 0.0, 3500.0]),
                            np.array([1.0, 0.0, 0.0])) is None


def test_root_tank_axis_matches_world_bounds():
    assert ROOT_TANK_AXIS_Y_MIN == pytest.approx(TANK_Z_MIN_MM - HALF_WC_LENGTH)
    assert ROOT_TANK_AXIS_Y_MAX == pytest.approx(TANK_Z_MAX_MM - HALF_WC_LENGTH)


def _mrd_chunk(start_m, stop_m):
    return {
        "MRDTrackStartX": np.array([start_m[0]]),
        "MRDTrackStartY": np.array([start_m[1]]),
        "MRDTrackStartZ": np.array([start_m[2]]),
        "MRDTrackStopX": np.array([stop_m[0]]),
        "MRDTrackStopY": np.array([stop_m[1]]),
        "MRDTrackStopZ": np.array([stop_m[2]]),
        "MRDTrackAngle": np.array([0.1]),
        "MRDTrackLength": np.array([100.0]),
    }


def test_mrd_track_vertex_lands_on_tank_boundary():
    from litepath.benchmark import TRUTH_KIND_MRD_RECO, RootTreeReader

    # MRD segment points back through the tank axis (start/stop in metres).
    track = RootTreeReader._mrd_track(
        None, _mrd_chunk((0.0, 0.0, 3.5), (0.0, 0.0, 4.5)), 0)
    assert track.kind == TRUTH_KIND_MRD_RECO
    assert np.all(np.isfinite(track.pos_root_mm))
    # Vertex is the upstream barrel crossing.
    assert np.hypot(track.pos_root_mm[0], track.pos_root_mm[2]) == \
        pytest.approx(TANK_RADIUS_MM, abs=1e-6)
    assert track.pos_root_mm[1] == pytest.approx(0.0, abs=1e-6)
    assert track.track_length_water_m == pytest.approx(
        2.0 * TANK_RADIUS_MM / 1000.0, rel=1e-9)


def test_mrd_track_miss_stores_nan_vertex():
    from litepath.benchmark import RootTreeReader

    track = RootTreeReader._mrd_track(
        None, _mrd_chunk((0.0, 0.0, 3.5), (1.0, 0.0, 3.5)), 0)
    assert track is not None
    assert np.all(np.isnan(track.pos_root_mm))
    assert np.isnan(track.track_length_water_m)
    # Direction and the raw MRD segment are still available.
    assert track.mrd_start_root_mm is not None


DATA_TREE = "benchmarking_data/ANNIETree_ThroughgoingMuons.root"
DATA_TREE_AVAILABLE = __import__("pathlib").Path(DATA_TREE).exists()


@pytest.mark.skipif(not DATA_TREE_AVAILABLE, reason="WAGLE data ROOT file not available")
def test_converted_data_vertices_are_physical():
    """Regression: reconstructed MRD vertices must not fly metres off-tank.

    Before the fix every data vertex was back-projected onto the ROOT ``z=0``
    plane, which produced ``world x == y`` vertices out to ~35 m.  Now finite
    vertices lie on the tank boundary; tracks that miss the tank are NaN.
    """
    from litepath.benchmark import RootTreeReader, root_to_world

    reader = RootTreeReader(DATA_TREE)
    n_finite = n_nan = 0
    for ev in reader.iter_events():
        tr = ev.truth
        if tr is None:
            continue
        if not np.all(np.isfinite(tr.pos_root_mm)):
            n_nan += 1
            continue
        n_finite += 1
        pos = root_to_world(tr.pos_root_mm)
        r = math.hypot(pos[0], pos[1])
        on_barrel = abs(r - TANK_RADIUS_MM) < 1e-3
        on_endcap = (abs(pos[2] - TANK_Z_MIN_MM) < 1e-3
                     or abs(pos[2] - TANK_Z_MAX_MM) < 1e-3) \
            and r <= TANK_RADIUS_MM + 1e-3
        assert on_barrel or on_endcap, (ev.event_id, pos, r)

    assert n_finite > 0
    assert n_finite + n_nan == 10112
