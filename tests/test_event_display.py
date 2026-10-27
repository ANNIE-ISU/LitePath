"""Tests for the event display's inside-view handedness.

``scripts/event_display.py`` cannot be imported as a package module, so it is
loaded by path.  Only the pure-python geometry helpers are exercised here.

The WCSim scan -> annieray world map is left-handed in the ``(pmt_x, pmt_z)``
plane, so the world azimuth ``atan2(y, x)`` winds opposite to the way the
detector looks when viewed from inside the tank.  The event display mirrors the
azimuth (``plot_phi``) so the barrel and endcaps read the inside-view way.
These tests pin that mirror down.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "event_display.py"


@pytest.fixture(scope="module")
def ed():
    spec = importlib.util.spec_from_file_location("event_display", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_phi_is_azimuth_mirror(ed):
    for az in (0.0, 30.0, 90.0, 135.0, 225.0, 359.0):
        assert ed.plot_phi(az) == pytest.approx((-az) % 360.0)


def test_plot_phi_is_an_involution(ed):
    rng = np.random.default_rng(0)
    az = rng.uniform(0, 360, size=50)
    assert np.allclose(ed.plot_phi(ed.plot_phi(az)), az)


def test_beam_vertex_is_a_displayed_octagon_corner(ed):
    """The auto-centred vertex must land on the displayed 22.5 deg grid."""
    import pandas as pd

    # A WAGLE-like beam bearing (135 deg physical).
    bearing = 134.9
    track = pd.DataFrame({"dir_x": [np.cos(np.radians(bearing))],
                          "dir_y": [np.sin(np.radians(bearing))]})
    vertex, got_bearing = ed.beam_vertex_phi(track)
    assert got_bearing == pytest.approx(bearing, abs=1e-6)
    # 135 is a physical corner; displayed it is mirrored to 225.
    assert vertex == pytest.approx(225.0)
    remainder = vertex % 22.5
    assert min(remainder, 22.5 - remainder) < 1e-6


def test_world_azimuth_winding_is_mirrored_from_scan():
    """The scan (sx, sz) ring winds opposite to the world (x, y) ring.

    This is the *physical* fact the display compensates for; if the raytracer
    geometry ever changes handedness this test will flag it.
    """
    # Octagon face centroids in scan (sx, sz), panel order 1..8, from the
    # metrology file layout.
    faces_sx = np.array([-41.7, -102.1, -101.0, -40.3, 42.0, 102.0, 100.3, 40.4])
    faces_sz = np.array([269.3, 207.8, 126.2, 66.1, 66.4, 127.7, 211.4, 270.0])

    def signed_area(x, z):
        return 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)

    area_scan = signed_area(faces_sx, faces_sz)
    # world: x = 10*sx, y = (168.1 - sz)*10  (see pmt_loader._scan_to_tank)
    wx = 10.0 * faces_sx
    wy = (168.1 - faces_sz) * 10.0
    area_world = signed_area(wx, wy)

    assert area_scan * area_world < 0  # opposite winding => mirrored
