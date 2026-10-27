"""Tests for scintillation photon generation and Cherenkov/scintillation
source tagging in the expanded hit array.
"""

import numpy as np
import pytest

from litepath.scintillation import (
    sample_scintillation_delay,
    sample_scintillation_wavelengths,
    generate_scintillation_photons,
)
from litepath.cherenkov import generate_cherenkov_photons

# Speed of light in m/s (matches cherenkov.py / scintillation.py)
_C = 299792458.0
_BETA = 0.999999
_MUON_SPEED_M_S = _BETA * _C  # muon speed in m/s
from litepath.tracer import (
    H_SOURCE, H_WAVELEN, H_ARRIVAL,
    SOURCE_CKV, SOURCE_SCI,
    N_EXPANDED_COLS, N_HIT_COLS,
    trace_muon_light, trace_cherenkov,
)


def test_delay_statistics():
    """Scintillation delay should follow a mixture of exponentials."""
    rng = np.random.default_rng(0)
    n = 400_000
    delays = sample_scintillation_delay(
        n, rng, tau_fast=2.0, tau_slow=20.0, fast_fraction=0.95,
    )
    expected = 0.95 * 2.0 + 0.05 * 20.0  # 2.9 ns
    assert float(np.mean(delays)) == pytest.approx(expected, rel=0.02)


def test_delay_all_fast():
    """With fast_fraction=1 every delay should be exponential with tau_fast."""
    rng = np.random.default_rng(1)
    delays = sample_scintillation_delay(200_000, rng, fast_fraction=1.0)
    assert float(np.mean(delays)) == pytest.approx(2.0, rel=0.02)


def test_delay_finite_when_uniform_draw_is_zero():
    """A uniform draw of exactly 0.0 must not produce an infinite delay."""

    class ZeroRng:
        def random(self, n):
            return np.zeros(n)

    delays = sample_scintillation_delay(5, ZeroRng())
    assert np.all(np.isfinite(delays))


def test_generate_scintillation_photons():
    """Isotropic unit directions and positive, distributed emission times."""
    rng = np.random.default_rng(2)
    o, d, ct = generate_scintillation_photons(
        (0, 0, 0, 0.0), (0, 0, -1), photons_per_cm=50,
        track_length=1.0, rng=rng,
    )
    assert o.shape == (5050, 3)
    assert d.shape == (5050, 3)
    assert ct.shape == (5050,)
    assert ct.dtype == np.float32
    # Directions are unit vectors (isotropic)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6)
    # Emission times are later than muon arrival at the start (delay >= 0)
    assert np.all(ct >= 0)
    # Some delays exceed what pure Cherenkov instantaneous arrival would give
    assert np.any(ct > np.min(ct) + 5.0)


def test_wavelength_constant():
    """Default scintillation wavelengths are a single constant for now."""
    rng = np.random.default_rng(3)
    w = sample_scintillation_wavelengths(10, rng)
    assert np.all(w == w[0])


def test_trace_muon_light_source_tagging():
    """Combined trace yields correct source column and per-source wavelengths."""
    rng = np.random.default_rng(7)
    full = trace_muon_light(
        (0, 0, 2000), (0, 0, -1), photons_per_cm=5, geometry=None,
        rng=rng, scintillation_enabled=True, photons_per_cm_scint=5,
        wavelength_nm=350.0, wavelength_scint_nm=420.0,
    )
    assert full.shape[1] == N_EXPANDED_COLS
    counts = np.bincount(full[:, H_SOURCE].astype(int))
    assert counts[SOURCE_CKV] > 0
    assert counts[SOURCE_SCI] > 0
    # Correct per-source wavelengths
    assert np.all(full[full[:, H_SOURCE] == SOURCE_CKV, H_WAVELEN] == 350.0)
    assert np.all(full[full[:, H_SOURCE] == SOURCE_SCI, H_WAVELEN] == 420.0)


def test_cherenkov_timing_includes_muon_arrival():
    """Cherenkov emission time = t0 + muon transit time along the track.

    The earliest emitted photon (at the track start, distance -> 0) has
    creation time equal to the muon t0, and the full spread across the track
    equals the time for the muon to traverse track_length.
    """
    rng = np.random.default_rng(9)
    t0_s = 100e-9  # muon start time in SECONDS (generators store t0 in s)
    t0_ns = t0_s * 1e9  # 100 ns
    track_length = 1.0  # m
    o, d, ct = generate_cherenkov_photons(
        (0, 0, 0, t0_s), (0, 0, -1),
        photons_per_cm=20, track_length=track_length, rng=rng,
    )
    # Photon emitted nearest the track start -> muon arrival ~ t0. The min
    # sampled distance is ~track_length/N above 0, so allow ~5 ns slack.
    assert float(np.min(ct)) == pytest.approx(t0_ns, abs=5.0)
    # Time spread equals the muon's transit time down the full track.
    transit_ns = track_length / _MUON_SPEED_M_S * 1e9
    spread = float(np.max(ct) - np.min(ct))
    assert spread == pytest.approx(transit_ns, rel=0.01)


def test_scintillation_timing_includes_decay_on_muon_arrival():
    """Scintillation delay is added on top of muon arrival (>= 0, ~ tau_fast).

    With fast_fraction=1.0 every decay is an exponential with tau_fast, so the
    difference (create_time - muon_arrival) should be a non-negative
    exponential of mean ~ tau_fast, never preceding the muon.
    """
    rng = np.random.default_rng(10)
    t0_s = 50e-9  # muon start time in SECONDS (generators store t0 in s)
    t0_ns = t0_s * 1e9  # 50 ns
    track_length = 1.0  # m
    tau_fast = 2.0
    o, d, ct = generate_scintillation_photons(
        (0, 0, 0, t0_s), (0, 0, -1),
        photons_per_cm=100, track_length=track_length,
        rng=rng, fast_fraction=1.0, tau_fast=tau_fast,
    )
    # Origins are (dist*1000)*direc + start; direc = (0,0,-1) so z encodes -dist.
    dist_m = -o[:, 2] / 1000.0
    muon_arrival = dist_m / _MUON_SPEED_M_S * 1e9 + t0_ns
    decay = ct - muon_arrival
    # Scintillation never arrives before the muon.
    assert np.all(decay >= -1e-3)
    # Decay delay follows an exponential with mean ~ tau_fast.
    assert float(np.mean(decay)) == pytest.approx(tau_fast, rel=0.08)


def test_trace_cherenkov_backward_compat():
    """trace_cherenkov stays source-0 and returns the expanded width."""
    rng = np.random.default_rng(8)
    full = trace_cherenkov(
        (0, 0, 2000), (0, 0, -1), photons_per_cm=5, geometry=None, rng=rng,
    )
    assert full.shape[1] == N_EXPANDED_COLS
    assert np.all(full[:, H_SOURCE] == SOURCE_CKV)
