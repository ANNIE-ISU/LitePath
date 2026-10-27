"""Batch-mode event generation for a standalone LAPPD module.

This is the batch counterpart of the interactive ``viz-lappd`` viewer:
it builds the single-ANNIE-housing geometry, shoots light sources at it,
optionally runs the LAPPD digitizer, and writes the *same* standardized
HDF5 schema as the full-detector ``litepath batch`` command
(see ``docs/BATCH_OUTPUT.md``).

Both interactive source definitions are supported:

* ``cherenkov`` — a tracked muon; Cherenkov photons are generated along the
  muon trajectory (a chord of the housing box).
* ``spot`` — a directed diverging beam from a point (the viewer's spot gun),
  optionally aimed at an intercept point on the photocathode.

Usage::

    python -m litepath batch-lappd --source cherenkov --sample-mode grid \\
        --lappd-response --output-dir results_lappd

    python -m litepath batch-lappd --source spot --n-photons 5000 \\
        --spread-deg 2 --intercept 0 -45 3.5 --lappd-response
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from litepath.batch import BatchAccumulator, H5_OUTPUT_NAME, _load_muon_file, \
    _MUON_FILE_CACHE
from litepath.cherenkov import generate_cherenkov_photons, generate_spot_photons
from litepath.io_h5 import SOURCE_TRACK, SOURCE_SPOT
from litepath.tracer import (
    Geometry, build_single_lappd_geometry,
    trace_rays, compute_track_length,
    N_HIT_COLS, N_EXPANDED_COLS,
    HI, HT, HDS, H_ARRIVAL, H_WAVELEN, H_BOUNCE, H_SOURCE,
    DET_SYS_LAPPD_ANNIE, C_MM_NS, N_WATER_DEFAULT, SOURCE_CKV,
)


# Default upstream launch band when sampling muon vertices in front of the
# housing (housing-local frame, +Z = toward the photocathode).
_Z_MIN = 200.0
_Z_MAX = 800.0


@dataclass
class LAPPDBatchConfig:
    """Parameters for a standalone-LAPPD batch run."""

    n_events: int = 100
    sample_mode: str = "random"      # "random" | "grid" | "fixed"
    source: str = "cherenkov"        # "cherenkov" | "spot"

    # Source parameters
    photons_per_cm: int = 150
    wavelength_nm: float = 350.0
    n_photons_spot: int = 5000
    spread_deg: float = 2.0
    intercept: Optional[tuple[float, float, float]] = None
    muon_fixed: Optional[tuple] = None
    muon_file: Optional[Path] = None
    grid_steps: tuple[int, int] = (9, 5)   # (n_x, n_z) for "grid"

    # Response + optics
    lappd_response: bool = True
    max_bounces: int = 0
    water_absorption_mm: float = 0.0
    water_scattering_mm: float = 0.0

    # IO
    output_dir: Path = Path("results")
    record_events: bool = True
    seed: Optional[int] = None


def config_to_dict(config: LAPPDBatchConfig) -> dict:
    """JSON-friendly config dict for the output root-group attribute."""
    def _clean(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value
    return {k: _clean(v) for k, v in asdict(config).items()}


# ---------------------------------------------------------------------------
# Source sampling (housing-local frame)
# ---------------------------------------------------------------------------

def _housing_constants(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centre, half, pc_centre)`` for the single housing in *geometry*."""
    hd = geometry.lappd_housing_data[0]
    centre = hd[0:3]
    half = hd[12:15]
    pc = geometry.annie_lappd_data[0, 0:3]
    return centre, half, pc


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return (0.0, 0.0, -1.0)
    return tuple(float(x / norm) for x in v)


def _scatter_aim(aim: tuple[float, float, float], spread_deg: float,
                 rng: np.random.Generator) -> tuple[float, float, float]:
    """Perturb *aim* (unit) by a Rayleigh scatter of *spread_deg* degrees."""
    from litepath.cherenkov import generate_spot_photons
    _, dirs, _ = generate_spot_photons(
        (0.0, 0.0, 0.0, 0.0), aim, 1, spread_deg=spread_deg, rng=rng
    )
    return tuple(float(d) for d in dirs[0])


def sample_source_state(
    event_id: int,
    config: LAPPDBatchConfig,
    rng: np.random.Generator,
    geometry: Geometry,
) -> dict:
    """Sample one event's light-source state.

    Returns a dict with ``pos`` (x, y, z, t0) and ``dir`` (unit dx, dy, dz).
    ``pos`` is always a 4-tuple with t0 in seconds.
    """
    centre, half, pc = _housing_constants(geometry)

    if config.sample_mode == "fixed":
        if config.muon_file is not None:
            lines = _MUON_FILE_CACHE
            x, y, z, t0, dx, dy, dz = lines[event_id % len(lines)]
        elif config.muon_fixed is not None:
            x, y, z, t0, dx, dy, dz = config.muon_fixed
        else:
            raise ValueError("--sample-mode fixed requires --muon-fixed or --muon-file")
        return {
            "pos": (x, y, z, t0),
            "dir": _normalize((dx, dy, dz)),
            "pc_centre": tuple(float(v) for v in pc),
        }

    # random / grid: place the source on a plane upstream of the PC and aim it
    # at the PC centre (or the --intercept target).
    aim_target = config.intercept if config.intercept is not None else pc

    if config.sample_mode == "grid":
        nx, nz = config.grid_steps
        xs = np.linspace(-0.7 * half[0], 0.7 * half[0], nx)
        zs = np.linspace(_Z_MIN, _Z_MAX, nz)
        x = float(xs[event_id % nx])
        z = float(zs[(event_id // nx) % nz])
        y = float(pc[1])
    elif config.sample_mode == "random":
        x = float(rng.uniform(-0.7 * half[0], 0.7 * half[0]))
        y = float(rng.uniform(pc[1] - 0.6 * half[1], pc[1] + 0.6 * half[1]))
        z = float(rng.uniform(_Z_MIN, _Z_MAX))
    else:
        raise ValueError(f"Unknown sample_mode: {config.sample_mode}")

    aim = _normalize(tuple(float(aim_target[i]) - v for i, v in enumerate((x, y, z))))
    dir_use = _scatter_aim(aim, config.spread_deg, rng)

    return {
        "pos": (x, y, z, 0.0),
        "dir": dir_use,
        "pc_centre": tuple(float(v) for v in pc),
    }


# ---------------------------------------------------------------------------
# Hit expansion (18-column convention, mirroring run_batch)
# ---------------------------------------------------------------------------

def _expand_hits(
    hits: np.ndarray,
    create_times: np.ndarray,
    sources: np.ndarray,
    wavelength_nm: float,
) -> np.ndarray:
    """Pad the (N, 14) kernel array to (N, 18) with arrival time / wavelength."""
    n = hits.shape[0]
    full = np.zeros((n, N_EXPANDED_COLS), dtype=np.float32)
    full[:, :N_HIT_COLS] = hits
    full[:, H_WAVELEN] = wavelength_nm
    full[:, H_SOURCE] = sources
    hit_mask = hits[:, HI] > 0.5
    if hit_mask.any():
        c_in_water = C_MM_NS / N_WATER_DEFAULT
        full[hit_mask, H_ARRIVAL] = (
            create_times[hit_mask] + hits[hit_mask, HT] / c_in_water
        )
    return full


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lappd_batch(
    geometry: Geometry | None = None,
    config: LAPPDBatchConfig | None = None,
    optics_config: Optional[dict] = None,
) -> dict[str, Path]:
    """Run *config.n_events* events on a standalone LAPPD module.

    Parameters
    ----------
    geometry : Geometry or None
        Single-housing geometry.  Built at the origin with +Z normal when None.
    config : LAPPDBatchConfig
    optics_config : dict or None
        Per-material optical properties (only used with ``--max-bounces``).

    Returns
    -------
    dict[str, Path]
        Keys written to the output file (``photon_hits``, ``muon_truth``,
        ``lappd_readouts``).
    """
    if config is None:
        config = LAPPDBatchConfig()
    if geometry is None:
        geometry = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))

    rng = np.random.default_rng(config.seed)
    accumulator = BatchAccumulator(
        output_dir=config.output_dir,
        detector_mode="lappd-standalone",
        config=config_to_dict(config),
    )

    if config.muon_file is not None:
        _load_muon_file(config.muon_file)

    t_start = time.time()
    for event_id in range(config.n_events):
        state = sample_source_state(event_id, config, rng, geometry)

        if config.source == "spot":
            origins, dirs, ctimes = generate_spot_photons(
                state["pos"], state["dir"], config.n_photons_spot,
                spread_deg=config.spread_deg, rng=rng,
            )
            track_length = 0.0
            n_ckv = len(origins)
            n_sci = 0
        else:
            track_length = compute_track_length(state["pos"], state["dir"], geometry)
            origins, dirs, ctimes = generate_cherenkov_photons(
                state["pos"], state["dir"], config.photons_per_cm,
                track_length=track_length, rng=rng,
            )
            n_ckv = len(origins)
            n_sci = 0

        sources = np.full(len(origins), SOURCE_CKV, dtype=np.float32)
        hits = trace_rays(origins, dirs, geometry)
        full = _expand_hits(hits, ctimes, sources, config.wavelength_nm)

        lappd_responses = None
        if config.lappd_response:
            from litepath.lappd_response import process_hits
            lappd_responses = process_hits(
                full, rng_seed=rng.integers(0, 2**31),
            )

        if config.record_events:
            det_mask = np.abs(full[:, HDS] - DET_SYS_LAPPD_ANNIE) < 0.5
            n_det_ckv = int(det_mask.sum())
            n_det_sci = 0
            source_type = SOURCE_SPOT if config.source == "spot" else SOURCE_TRACK
            spread = config.spread_deg if config.source == "spot" else float("nan")
            muon_params = {
                "pos": state["pos"],
                "direc": state["dir"],
                "track_length": track_length,
                "n_generated": len(origins),
                "n_generated_ckv": n_ckv,
                "n_generated_sci": n_sci,
                "n_detected_ckv": n_det_ckv,
                "n_detected_sci": n_det_sci,
                "source_type": source_type,
                "spread_deg": spread,
                "intercept": config.intercept,
            }
            accumulator.append_event(
                event_id, full, None,
                lappd_responses=lappd_responses,
                muon_params=muon_params,
            )

        if (event_id + 1) % 10 == 0 or event_id + 1 == config.n_events:
            elapsed = time.time() - t_start
            print(f"  [{event_id + 1}/{config.n_events}] {elapsed:.1f}s elapsed")

    print(f"  Total: {time.time() - t_start:.1f}s for {config.n_events} events")

    if config.record_events:
        accumulator.close()
        paths: dict[str, Path] = {}
        h5_path = config.output_dir / H5_OUTPUT_NAME
        if accumulator._n_photon_rows > 0:
            paths["photon_hits"] = h5_path
            print(f"  Wrote {h5_path} ({accumulator._n_photon_rows} photon rows)")
        if accumulator._n_muon_rows > 0:
            paths["muon_truth"] = h5_path
            print(f"  Wrote {h5_path} ({accumulator._n_muon_rows} muon truth rows)")
        if accumulator._n_lappd_rows > 0:
            paths["lappd_readouts"] = h5_path
            print(f"  LAPPD readouts: {accumulator._n_lappd_rows} rows "
                  f"across {accumulator._n_lappd_events} events")
        if h5_path.exists() and len(geometry.detectors):
            from litepath.io_h5 import write_detectors, write_metadata
            write_metadata(h5_path, {"detector_mode": "lappd-standalone"})
            write_detectors(h5_path, geometry.detectors)
            paths["detectors"] = h5_path
        return paths

    return {}