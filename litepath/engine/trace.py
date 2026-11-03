"""Host-side entry points that run the trace kernel."""

import numpy as np

from litepath.engine.kernel import trace_kernel
from litepath.engine.scene import pack_scene
from litepath.geometry.build import Geometry
from litepath.hits import (
    DET_SYS_NONE, HI, HT, HX, HZ, HNX, HNZ, HDI, HDS, HMAT, N_HIT_COLS,
)
from litepath.optics import N_WATER_DEFAULT, OpticalMaterial, WaterAttenuation


def trace_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    geometry: Geometry,
) -> np.ndarray:
    n_rays = origins.shape[0]
    hits = np.zeros((n_rays, N_HIT_COLS), dtype=np.float32)
    # Default detector_index = -1, detector_system = -1
    hits[:, HDI] = -1.0
    hits[:, HDS] = DET_SYS_NONE * 1.0

    scene = pack_scene(geometry)
    trace_kernel(
        origins,
        directions,
        scene.scene_f,
        scene.scene_i,
        scene.layout,
        *scene.tank,
        hits,
    )
    return hits


def trace_with_optics(
    origins: np.ndarray,
    directions: np.ndarray,
    geometry: Geometry,
    optics_config: dict[int, OpticalMaterial],
    max_bounces: int = 3,
    n_water: float = N_WATER_DEFAULT,
    rng: np.random.Generator | None = None,
    water_config: WaterAttenuation | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace rays with multi-bounce optical surface physics.

    Calls ``trace_rays()`` repeatedly, each time processing surface
    interactions (Fresnel reflection/transmission, diffuse reflection,
    absorption) per the per-material ``optics_config``.  Detected hits
    carry the **total** optical path length in the ``HT`` column and
    the number of surface reflections in the returned ``bounce_counts``.

    Parameters
    ----------
    origins:
        ``(N, 3)`` float32 — ray start points.
    directions:
        ``(N, 3)`` float32 — ray unit directions.
    geometry:
        Geometry bundle.
    optics_config:
        Per-material :class:`OpticalMaterial` dict from
        :func:`load_optics_config`.
    max_bounces:
        Maximum number of surface reflections per photon (the first
        trace counts as bounce 0).
    n_water:
        Refractive index of water (used for Fresnel computation).
    rng:
        NumPy random generator.
    water_config:
        Water absorption / Rayleigh scattering parameters, or ``None``
        to disable water attenuation (default).

    Returns
    -------
    (hits, bounce_counts, orig_indices):
        ``hits`` is ``(M, N_HIT_COLS)`` float32 — detected hits only
        (a subset of the input rays), with ``HT`` = total accumulated
        path length.  ``bounce_counts`` is ``(M,)`` int32 — number of
        surface reflections for each hit.  ``orig_indices`` is ``(M,)``
        int32 — the original photon index for each detected hit (for
        lookups into per-photon data like creation time).  If no hits
        are detected, returns two empty arrays and an empty index array.
    """
    from litepath.optics import (
        ACTION_DETECT, ACTION_REFLECT, MODEL_MISSING, MaterialTable, evaluate_hits,
        rayleigh_scatter_dir_v,
    )

    if rng is None:
        rng = np.random.default_rng()

    origins = np.asarray(origins, dtype=np.float32).copy()
    directions = np.asarray(directions, dtype=np.float32).copy()
    n = len(origins)
    table = MaterialTable.from_config(optics_config)

    total_path = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=bool)
    detected_hits: list[np.ndarray] = []
    detected_bounces: list[np.ndarray] = []
    detected_indices: list[np.ndarray] = []

    use_water = water_config is not None and water_config.is_active
    mu_total = water_config.mu_total if use_water else 0.0

    # All live photons are handled together, one trace per pass.
    # n_surface counts passes with a surface reflection; water-scatter
    # retraces don't count toward max_bounces.
    n_surface = 0
    while n_surface <= max_bounces:
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break

        hits = trace_rays(origins[idx], directions[idx], geometry)
        mat = hits[:, HMAT].astype(np.int64)
        # missed everything, or a material without optics -> photon ends
        live = (hits[:, HI] != 0) & (table.lookup(mat) != MODEL_MISSING)
        next_alive = np.zeros(len(idx), dtype=bool)
        seg_len = hits[:, HT]

        # ---- Water attenuation (absorption / Rayleigh scatter) ----
        surface = live
        if use_water and mu_total > 0:
            w = np.flatnonzero(live)
            u = rng.random(w.size)
            interacts = u < 1.0 - np.exp(-seg_len[w] * mu_total)
            wi = w[interacts]
            s = (-np.log(1.0 - u[interacts]) / mu_total).astype(np.float32)
            absorbed = np.zeros(wi.size, dtype=bool)
            if water_config.mu_abs > 0:
                absorbed = rng.random(wi.size) < water_config.mu_abs / mu_total
            # Rayleigh scatter — not a surface bounce
            sc, s_sc = wi[~absorbed], s[~absorbed]
            gi = idx[sc]
            total_path[gi] += s_sc
            origins[gi] = origins[gi] + s_sc[:, None] * directions[gi]
            directions[gi] = rayleigh_scatter_dir_v(directions[gi], rng)
            next_alive[sc] = True
            surface = live.copy()
            surface[wi] = False

        # ---- No water interaction → process surface hit ----
        sj = np.flatnonzero(surface)
        gi = idx[sj]
        total_path[gi] += seg_len[sj]
        action, new_dir = evaluate_hits(
            table, mat[sj], directions[gi], hits[sj, HNX:HNZ + 1], n_water, rng)

        det = action == ACTION_DETECT
        if det.any():
            rows = hits[sj[det]].copy()
            rows[:, HT] = total_path[gi[det]]
            detected_hits.append(rows)
            detected_bounces.append(np.full(int(det.sum()), n_surface, dtype=np.int32))
            detected_indices.append(gi[det].astype(np.int32))

        ref = action == ACTION_REFLECT
        origins[gi[ref]] = hits[sj[ref], HX:HZ + 1]
        directions[gi[ref]] = new_dir[ref]
        next_alive[sj[ref]] = True
        # absorb: drop

        alive[idx] = next_alive

        if ref.any():
            n_surface += 1
        elif not use_water:
            break  # no surface interactions and no water → done

    if detected_hits:
        hits_out = np.concatenate(detected_hits, axis=0)
        bounces_out = np.concatenate(detected_bounces)
        indices_out = np.concatenate(detected_indices)
        # detections in photon order within each pass, as before
        return hits_out, bounces_out, indices_out
    return (np.zeros((0, N_HIT_COLS), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32))
