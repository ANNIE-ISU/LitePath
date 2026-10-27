"""Truth phi convention: azimuth stored and compared in [0, 360).

The fit grid scans phi over [0, 360] (``--grid-phi 0 360 37``), so truth
written as a raw ``arctan2`` result in [-180, 180] showed spurious 360-degree
residuals for muons with negative azimuth (GitHub issue #6).
"""

import h5py
import pytest

from litepath.batch import BatchAccumulator
from litepath.benchmark import litepath_angles
from litepath.fitting import load_observed_event
from litepath.io_h5 import load_table


def _write_truth(tmp_path, direc):
    acc = BatchAccumulator(output_dir=tmp_path)
    acc._record_muon(
        event_id=0, n_detected=0, pos=(0.0, 0.0, 0.0, 0.0), direc=direc,
        track_length=1.0, n_generated=1500,
    )
    acc.close()
    return acc.h5_path


def test_batch_truth_phi_in_0_360(tmp_path):
    h5 = _write_truth(tmp_path, (0.0, -1.0, 0.0))     # azimuth -90 deg
    phi = float(load_table(h5, "muon_truth")["phi_deg"].iloc[0])
    assert phi == pytest.approx(270.0)


def test_benchmark_litepath_phi_in_0_360():
    _, phi = litepath_angles((0.0, -1.0, 0.0))
    assert phi == pytest.approx(270.0)
    _, phi = litepath_angles((-1.0, -1e-9, 0.0))     # just below -180 boundary
    assert 0.0 <= phi < 360.0


def test_fit_wraps_old_negative_phi(tmp_path):
    """Files written before the fix store phi in [-180, 180]."""
    h5 = _write_truth(tmp_path, (0.0, -1.0, 0.0))
    with h5py.File(h5, "r+") as f:
        rows = f["muon_truth"][()]
        rows["phi_deg"] = -90.0
        f["muon_truth"][...] = rows

    observed = load_observed_event(h5, 0)
    assert observed.true_phi == pytest.approx(270.0)
