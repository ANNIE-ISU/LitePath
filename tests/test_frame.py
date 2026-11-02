"""The fixed LitePath frame (litepath/geometry/frame.py).

Checks the conversions, that every tool sees the same PMT positions, and the
orientation against the ToolAnalysis table and the 2019 PMT layout slides
(M. O'Flaherty, ANNIE collaboration meeting, slides 5 and 8).
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from litepath.geometry.frame import (
    frame_to_tank, structure_to_frame, table_to_frame, tank_to_frame,
    tank_to_frame_direction,
)
from litepath.geometry.versions import GeometryError, load_build, load_version, version_from_flags
from litepath.pmt_loader import load_pmts

TOOLANALYSIS = Path("geometry/base/FullTankPMTGeometry.csv")


def _without_meshes(name):
    """A version with the structure and PMT meshes switched off (fast to build)."""
    version = load_version(name)
    version.settings.update(structure_mesh=False, pmt_meshes=False)
    version.status, version.tagged = "unnamed", {}
    return version


def _azimuth(xy):
    return math.degrees(math.atan2(xy[1], xy[0])) % 360.0


@pytest.fixture(scope="module")
def placed():
    """ToolAnalysis records with their frame positions, by cable label and panel."""
    pmts = load_pmts(TOOLANALYSIS)
    return pmts["records"], np.asarray(pmts["centers"], dtype=float)


def test_tank_to_frame_is_a_proper_rotation():
    m = np.array([tank_to_frame_direction(e) for e in np.eye(3)]).T
    assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1.0)
    # beam (+z in WCSim) is +y, up (+y in WCSim) is +z
    assert np.allclose(tank_to_frame_direction([0, 0, 1]), [0, 1, 0])
    assert np.allclose(tank_to_frame_direction([0, 1, 0]), [0, 0, 1])
    pts = np.random.default_rng(0).normal(scale=1000, size=(50, 3))
    assert np.allclose(frame_to_tank(tank_to_frame(pts)), pts)


def test_structure_to_frame_is_a_22_5_degree_turn():
    out = structure_to_frame(np.array([[1000.0, 0.0, 5.0]], dtype=np.float32))
    assert out.dtype == np.float32
    assert _azimuth(out[0]) == pytest.approx(22.5, abs=1e-4) and out[0, 2] == 5.0


def test_pmts_are_the_table_mapped_into_the_frame(placed):
    records, centers = placed
    expected = table_to_frame(np.array([r.scan_position_cm for r in records]))
    np.testing.assert_allclose(centers, expected, atol=1e-3)
    # by hand for one PMT: x' = -x, y' = z - 1681 mm, z' = y + 144.5 + 1980 mm
    r = records[0]
    x, y, z = (10 * v for v in r.scan_position_cm)
    np.testing.assert_allclose(centers[0], [-x, z - 1681.0, y + 2124.5], atol=1e-3)


def test_every_tool_sees_the_same_pmts():
    from litepath.benchmark import load_pmt_registry
    version = _without_meshes("annie-2026-data")
    build = load_build(version, quiet=True).geometry          # batch, fit, viz
    registry = load_pmt_registry(TOOLANALYSIS)                # benchmark
    loader = load_pmts(TOOLANALYSIS, bottom_spin_deg=22.5)
    np.testing.assert_array_equal(registry.positions, build.pmt_centers)
    np.testing.assert_array_equal(loader["centers"], build.pmt_centers)
    np.testing.assert_array_equal(registry.directions, build.pmt_directions)


def test_barrel_panels_around_the_beam(placed):
    """Slide 5: the beam enters at the 4|5 corner and points at the 1|8 corner."""
    records, centers = placed
    az = {p: _azimuth(centers[[r.panel == p for r in records]][:, :2].mean(0))
          for p in range(1, 9)}
    # panels run clockwise seen from above, 45 deg apart, panel 1 at 67.5 deg
    for p in range(1, 9):
        assert az[p] == pytest.approx((67.5 - 45 * (p - 1)) % 360, abs=3.0)


def test_bottom_pmts_against_the_slides(placed):
    """Slides 5 and 8: N4 is furthest downstream, N5 furthest upstream, and
    the strips N4-E4-W2-N3 and E3-S5-S2-N1-S3-W5 run at 45 deg to the beam."""
    records, centers = placed
    bottom = {r.cable_label: centers[i] for i, r in enumerate(records) if r.panel == 0}
    ys = {k: v[1] for k, v in bottom.items()}
    assert max(ys, key=ys.get) == "N4" and min(ys, key=ys.get) == "N5"
    for strip in (["N4", "E4", "W2", "N3"], ["E3", "S5", "S2", "N1", "S3", "W5"]):
        pts = np.array([bottom[k][:2] for k in strip])
        steps = np.diff(pts, axis=0)
        for s in steps:
            assert _azimuth(s) == pytest.approx(225.0, abs=1.0)   # towards -x, upstream
        assert np.ptp(np.linalg.norm(steps, axis=1)) < 10.0   # ~280 mm apart


def test_lappds_and_surfboards_on_the_downstream_corners():
    version = _without_meshes("annie-2026-data")
    g = load_build(version, quiet=True).geometry
    for arr in (g.surfboard_data, g.lappd_housing_data):
        assert sorted(round(_azimuth(row[:2]), 3) for row in arr) == [45.0, 90.0, 135.0]


def test_orientation_settings_are_gone(tmp_path):
    doc = yaml.safe_load(Path("geometry/versions/annie-2026-data-tuning.yaml").read_text())
    doc["settings"]["det_rotation_deg"] = 22.5
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(GeometryError, match="orientation is fixed"):
        load_version(path)
    with pytest.raises(GeometryError, match="orientation is fixed"):
        version_from_flags({"det_rotation": 0.0})
    with pytest.raises(GeometryError, match="orientation is fixed"):
        version_from_flags({"bottom_rot": 0.0})
    # the one value the old flags ever had in use still works
    version_from_flags({"det_rotation": 22.5, "bottom_rot": 45.0})
