"""PMT records from the ToolAnalysis table and the WCSim scan file."""

from pathlib import Path

import numpy as np
import pytest

from litepath.geometry.elements import STATUS_OFF, read_pmt_table
from litepath.pmt_loader import load_pmts

TOOLANALYSIS = Path("geometry/base/FullTankPMTGeometry.csv")
SCAN = Path("PMTPositions_Scan.txt")
SWAPPED = {453, 455}  # y positions swapped in the scan file, fixed in ToolAnalysis PR #387


@pytest.fixture(scope="module")
def tables():
    ta = {r.channel_key: r for r in read_pmt_table(TOOLANALYSIS)}
    scan = {r.channel_key: r for r in read_pmt_table(SCAN)}
    return ta, scan


def test_toolanalysis_ids(tables):
    ta, _ = tables
    assert sorted(ta) == list(range(332, 464))
    assert all(r.detector_key == r.channel_key for r in ta.values())
    assert all(r.cable_label for r in ta.values())
    assert sum(r.status == STATUS_OFF for r in ta.values()) == 15


def test_tables_agree_except_swapped_pmts(tables):
    ta, scan = tables
    assert sorted(ta) == sorted(scan)
    for key in ta:
        a, b = ta[key], scan[key]
        assert (a.detector_key, a.panel, a.pmt_type) == (b.detector_key, b.panel, b.pmt_type)
        diff = np.abs(np.subtract(a.scan_position_cm, b.scan_position_cm))
        if key in SWAPPED:
            assert diff[1] > 200.0 and diff[0] < 1e-4 and diff[2] < 1e-4
        else:
            assert diff.max() < 1e-4


def test_swap_is_exactly_453_455(tables):
    ta, scan = tables
    assert ta[453].scan_position_cm[1] == pytest.approx(scan[455].scan_position_cm[1], abs=1e-4)
    assert ta[455].scan_position_cm[1] == pytest.approx(scan[453].scan_position_cm[1], abs=1e-4)


def test_load_pmts_places_both_tables_alike():
    ta = load_pmts(TOOLANALYSIS)
    scan = load_pmts(SCAN)
    assert ta["detector_nums"] == scan["detector_nums"]
    assert ta["types"] == scan["types"] and ta["panels"] == scan["panels"]
    same = [i for i, k in enumerate(ta["detector_nums"]) if k not in SWAPPED]
    for key in ("centers", "directions", "instance_positions", "rotmats"):
        np.testing.assert_allclose(ta[key][same], scan[key][same], atol=1e-3)
