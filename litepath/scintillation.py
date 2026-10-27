"""Scintillation photon generation for wbLS (water-based liquid scintillator).

Unlike Cherenkov emission (a cone around the muon direction), scintillation is
emitted isotropically at each emission point along the track.  The emission
time is delayed relative to the muon arrival by the scintillation decay time,
modelled as a mixture of exponentials (Birks form) with a fast component
(tau_fast, dominant) and a slow component (tau_slow).

Default parameters follow the companion ANNIE wbLS characterization
(Caravaca et al., arXiv:2006.00173 and references therein):
    tau_fast  ~ 2 ns   (fast component, ~95% of light)
    tau_slow  ~ 20 ns  (slow component, ~5% of light)
    rise time ~ 0.25 ns (negligible compared to PMT/LAPPD timing)
"""

import numpy as np

# Speed of light in m/s
_C = 299792458.0

# Default muon speed as a fraction of the speed of light (matches cherenkov.py)
_BETA = 0.999999

# Track discretisation precision (cm steps along track, matches cherenkov.py)
_TRACK_PREC = 2

# Default scintillation timing parameters (wbLS, ns)
DEFAULT_TAU_FAST = 2.0
DEFAULT_TAU_SLOW = 20.0
DEFAULT_FAST_FRACTION = 0.95

# Default wavelength for scintillation emission when no colour spectrum is used
# (see sample_scintillation_wavelengths).
DEFAULT_SCINTILLATION_WAVELENGTH = 420.0


def sample_scintillation_wavelengths(
    n: int,
    rng: np.random.Generator,
    wavelength: float = DEFAULT_SCINTILLATION_WAVELENGTH,
) -> np.ndarray:
    """Sample scintillation photon wavelengths (nm).

    Currently returns a constant wavelength for every photon.  This is the
    future extension point for assigning a realistic wbLS emission spectrum
    (which peaks around 360-420 nm depending on fluor/dye content).  When a
    spectrum is added, sample per-photon wavelengths here and thread the
    returned array through to the H_WAVELEN hit column.
    """
    return np.full(n, wavelength, dtype=np.float32)


def sample_scintillation_delay(
    n: int,
    rng: np.random.Generator,
    tau_fast: float = DEFAULT_TAU_FAST,
    tau_slow: float = DEFAULT_TAU_SLOW,
    fast_fraction: float = DEFAULT_FAST_FRACTION,
) -> np.ndarray:
    """Sample scintillation emission delays (ns) from a mixture of exponentials.

    For each photon choose the fast component with probability fast_fraction
    (slow otherwise), then draw an exponential decay time with the chosen time
    constant:  delay = -tau * ln(U),  U ~ Uniform(0,1).
    """
    component = rng.random(n) < fast_fraction
    tau = np.where(component, tau_fast, tau_slow)
    return (-tau * np.log(rng.random(n))).astype(np.float32)


def generate_scintillation_photons(
    muon_pos,
    muon_direc,
    photons_per_cm: int = 100,
    rng: np.random.Generator | None = None,
    track_length: float = 4.0,
    tau_fast: float = DEFAULT_TAU_FAST,
    tau_slow: float = DEFAULT_TAU_SLOW,
    fast_fraction: float = DEFAULT_FAST_FRACTION,
    wavelength: float = DEFAULT_SCINTILLATION_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate isotropically-emitting scintillation photons along a muon track.

    Args:
        muon_pos: Muon start position (x, y, z[, t0]) in mm / ns.
        muon_direc: Muon direction (does not need to be a unit vector).
        photons_per_cm: Number of photons emitted per cm of track.
        rng: NumPy random generator.
        track_length: Track length inside the detector (m).
        tau_fast/tau_slow/fast_fraction: Scintillation timing model (ns, ns, -).
        wavelength: Photon wavelength (nm).

    Returns:
        (origins, directions, createTime) arrays of shape (N,3), (N,3), (N,),
        all float32.  These have the same layout as the Cherenkov generator and
        are consumed directly by trace_rays()/trace_with_optics().
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(muon_pos) != 4:
        muon_start = tuple([muon_pos[0], muon_pos[1], muon_pos[2], 0])
    else:
        muon_start = muon_pos

    n_steps = int(track_length * 10 ** _TRACK_PREC) + 1
    muon_speed = _BETA * _C  # m/s

    n_photons = n_steps * photons_per_cm

    # Emission points uniformly distributed along the track (like Cherenkov).
    muon_array = rng.uniform(0, track_length, size=n_photons)
    muon_array.sort()

    # Muon arrival time at each emission point (ns), then add decay delay.
    muon_arrival = (muon_array / muon_speed) * 10 ** 9 + muon_start[3] * 10 ** 9
    decay_delay = sample_scintillation_delay(
        n_photons, rng, tau_fast=tau_fast, tau_slow=tau_slow,
        fast_fraction=fast_fraction,
    )
    create_time = (muon_arrival + decay_delay).astype(np.float32)

    # Isotropic emission directions (uniform on the unit sphere).
    theta = np.arccos(rng.uniform(-1, 1, size=n_photons))
    phi = rng.uniform(0, 2 * np.pi, size=n_photons)

    phot_start_pos = muon_array[:, np.newaxis] * np.array(muon_direc)

    origins = np.empty((n_photons, 3), dtype=np.float32)
    directions = np.empty((n_photons, 3), dtype=np.float32)

    origins[:, 0] = 1000 * phot_start_pos[:, 0] + muon_start[0]
    origins[:, 1] = 1000 * phot_start_pos[:, 1] + muon_start[1]
    origins[:, 2] = 1000 * phot_start_pos[:, 2] + muon_start[2]

    directions[:, 0] = np.sin(theta) * np.cos(phi)
    directions[:, 1] = np.sin(theta) * np.sin(phi)
    directions[:, 2] = np.cos(theta)

    return origins, directions, create_time
