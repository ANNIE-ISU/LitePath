"""Generate muon light and trace it through the detector."""

import numpy as np

from litepath.engine.trace import trace_rays, trace_with_optics
from litepath.geometry.build import Geometry
from litepath.hits import (
    DET_SYS_NONE, HI, HT, HX, HZ, HDI, HDS, N_HIT_COLS,
    H_ARRIVAL, H_WAVELEN, H_BOUNCE, H_SOURCE, N_EXPANDED_COLS,
    SOURCE_CKV, SOURCE_SCI,
)
from litepath.optics import C_MM_NS, N_WATER_DEFAULT, OpticalMaterial, WaterAttenuation
from litepath.sources.track import compute_track_length


def trace_muon_light(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    photons_per_cm: int = 150,
    geometry: Geometry | None = None,
    rng: np.random.Generator | None = None,
    wavelength_nm: float = 350.0,
    n_water: float = N_WATER_DEFAULT,
    max_bounces: int = 0,
    optics_config: dict[int, OpticalMaterial] | None = None,
    water_config: WaterAttenuation | None = None,
    scintillation_enabled: bool = False,
    photons_per_cm_scint: int = 100,
    tau_fast: float = 2.0,
    tau_slow: float = 20.0,
    fast_fraction: float = 0.95,
    wavelength_scint_nm: float = 420.0,
) -> np.ndarray:
    """Trace Cherenkov and/or scintillation photons from a muon track.

    This is a general entry point that can emit Cherenkov light (a cone
    around the muon direction) and/or wbLS scintillation light (isotropic,
    delayed by the scintillation decay).  Both photon types are concatenated
    into a single ray-tracing call, then expanded to N_EXPANDED_COLS columns
    with the photon source type recorded in column H_SOURCE.

    Workflow:
      1. Compute track length from geometry (tank cylinder or LAPPD housing).
      2. Call generate_cherenkov_photons() and/or
         generate_scintillation_photons() with that track length.
      3. Run the GPU kernel via trace_rays() or trace_with_optics().
      4. Expand to (N, N_EXPANDED_COLS) by appending arrival_time, wavelength,
         bounce_count, and source type.

    Parameters
    ----------
    photons_per_cm : int
        Cherenkov photons generated per cm along the muon track.
    scintillation_enabled : bool
        If True, also emit scintillation photons.
    photons_per_cm_scint : int
        Scintillation photons generated per cm along the muon track.
    tau_fast, tau_slow, fast_fraction : float
        wbLS scintillation timing model (fast/slow decay times in ns and the
        fraction of light in the fast component).
    wavelength_scint_nm : float
        Wavelength (nm) assigned to scintillation photons when no colour
        spectrum is in use.
    geometry : Geometry | None
        If provided, the track length is automatically computed from the tank
        or LAPPD housing bounds.  When ``None``, a fixed 4.0 m track is used.
    water_config:
        Water absorption / Rayleigh scattering parameters, or ``None``
        to disable (default).

    Returns ``(N, N_EXPANDED_COLS)`` hit array with columns:
        0: hit_flag, 1: t (mm), 2-4: x,y,z, 5-7: nx,ny,nz,
        8: component_id, 9: detector_index, 10: detector_system,
        11: local_u, 12: local_v, 13: material_id,
        14: arrival_time (ns), 15: wavelength (nm), 16: bounce_count,
        17: photon_source (0=Cherenkov, 1=scintillation)
    """
    from litepath.cherenkov import generate_cherenkov_photons
    from litepath.lappd_model import compute_housing_track_length
    from litepath.optics import load_optics_config
    from litepath.scintillation import generate_scintillation_photons

    if rng is None:
        rng = np.random.default_rng()

    # ---- Compute track length from geometry ----
    track_length = compute_track_length(muon_pos, muon_dir, geometry)

    # ---- Generate chosen photon types and concatenate with source tagging ----
    origins_list: list[np.ndarray] = []
    directions_list: list[np.ndarray] = []
    ctimes_list: list[np.ndarray] = []
    sources_list: list[np.ndarray] = []

    ckv_origins, ckv_directions, ckv_ctimes = generate_cherenkov_photons(
        muon_pos, muon_dir, photons_per_cm, track_length=track_length, rng=rng,
    )
    origins_list.append(ckv_origins)
    directions_list.append(ckv_directions)
    ctimes_list.append(ckv_ctimes)
    sources_list.append(np.full(ckv_origins.shape[0], SOURCE_CKV, dtype=np.float32))

    if scintillation_enabled:
        sci_origins, sci_directions, sci_ctimes = generate_scintillation_photons(
            muon_pos, muon_dir, photons_per_cm_scint, track_length=track_length,
            rng=rng, tau_fast=tau_fast, tau_slow=tau_slow,
            fast_fraction=fast_fraction, wavelength=wavelength_scint_nm,
        )
        origins_list.append(sci_origins)
        directions_list.append(sci_directions)
        ctimes_list.append(sci_ctimes)
        sources_list.append(np.full(sci_origins.shape[0], SOURCE_SCI, dtype=np.float32))

    origins = np.vstack(origins_list)
    directions = np.vstack(directions_list)
    create_times = np.concatenate(ctimes_list)
    tag_sources = np.concatenate(sources_list)

    # Per-source wavelength for every generated photon.  Future colour-spectrum
    # sampling will produce per-photon wavelengths here instead of a constant.
    wav = np.where(tag_sources == SOURCE_SCI, wavelength_scint_nm, wavelength_nm)
    wav = wav.astype(np.float32)

    # ---- No geometry → just return the generated rays (no intersection testing) ----
    if geometry is None:
        n = origins.shape[0]
        full = np.zeros((n, N_EXPANDED_COLS), dtype=np.float32)
        full[:, HI] = 1.0
        full[:, HX:HZ + 1] = origins
        full[:, HDI] = -1.0
        full[:, HDS] = float(DET_SYS_NONE)
        full[:, H_WAVELEN] = wav
        full[:, H_ARRIVAL] = create_times
        full[:, H_SOURCE] = tag_sources
        return full

    use_water = water_config is not None and water_config.is_active
    if max_bounces > 0 or use_water:
        cfg = optics_config if optics_config is not None else load_optics_config(None)
        hits, bounce_counts, orig_indices = trace_with_optics(
            origins, directions, geometry, cfg,
            max_bounces=max_bounces, n_water=n_water, rng=rng,
            water_config=water_config,
        )
    else:
        hits = trace_rays(origins, directions, geometry)
        bounce_counts = np.zeros(hits.shape[0], dtype=np.int32)
        orig_indices = np.arange(hits.shape[0], dtype=np.int32)

    # ---- Expand from N_HIT_COLS to N_EXPANDED_COLS ----
    # Add arrival_time (14), wavelength (15), bounce_count (16), source (17).
    n = hits.shape[0]
    full = np.zeros((n, N_EXPANDED_COLS), dtype=np.float32)
    full[:, :N_HIT_COLS] = hits

    c_in_water = C_MM_NS / n_water
    hit_mask = hits[:, HI] > 0.5
    if hit_mask.any():
        full[hit_mask, H_ARRIVAL] = create_times[orig_indices[hit_mask]] + hits[hit_mask, HT] / c_in_water

    # Assign wavelength per source type using the original photon index.
    full[:, H_WAVELEN] = wav[orig_indices]

    full[:, H_BOUNCE] = bounce_counts
    full[:, H_SOURCE] = tag_sources[orig_indices]

    return full


def trace_cherenkov(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    photons_per_cm: int = 150,
    geometry: Geometry | None = None,
    rng: np.random.Generator | None = None,
    wavelength_nm: float = 350.0,
    n_water: float = N_WATER_DEFAULT,
    max_bounces: int = 0,
    optics_config: dict[int, OpticalMaterial] | None = None,
    water_config: WaterAttenuation | None = None,
) -> np.ndarray:
    """Trace Cherenkov photons from a muon track.

    Backward-compatible wrapper around :func:`trace_muon_light` with
    scintillation disabled.  All output columns are the same as previously
    (the new H_SOURCE column is set to 0 = Cherenkov for every photon).

    Returns ``(N, N_EXPANDED_COLS)`` hit array:
        0: hit_flag, 1: t (mm), 2-4: x,y,z, 5-7: nx,ny,nz,
        8: component_id, 9: detector_index, 10: detector_system,
        11: local_u, 12: local_v, 13: material_id,
        14: arrival_time (ns), 15: wavelength (nm), 16: bounce_count,
        17: photon_source (all 0 = Cherenkov)
    """
    return trace_muon_light(
        muon_pos, muon_dir, photons_per_cm, geometry, rng=rng,
        wavelength_nm=wavelength_nm, n_water=n_water, max_bounces=max_bounces,
        optics_config=optics_config, water_config=water_config,
        scintillation_enabled=False,
    )
