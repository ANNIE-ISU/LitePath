"""Host-side entry points that run the trace kernel."""

import numpy as np

from litepath.engine.kernel import trace_kernel
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

    trace_kernel(
        origins,
        directions,
        geometry.mesh_vertices,
        geometry.mesh_triangles.astype(np.int32),
        geometry.mesh_material_ids if geometry.mesh_material_ids is not None
        else np.zeros(0, dtype=np.int32),
        geometry.bvh_node_min,
        geometry.bvh_node_max,
        geometry.bvh_node_left,
        geometry.bvh_node_right,
        geometry.bvh_tri_start,
        geometry.bvh_tri_end,
        geometry.bvh_tri_ids,
        geometry.bvh_n_nodes,
        geometry.pmt_centers,
        geometry.pmt_radii,
        geometry.pmt_directions,
        geometry.pmt_body_tris,
        geometry.pmt_body_offsets,
        geometry.pmt_rotmats,
        geometry.pmt_mesh_types,
        geometry.pmt_hw_types,
        geometry.pmt_hw_tris,
        geometry.pmt_hw_mat_ids,
        geometry.pmt_instance_pos,
        geometry.pmt_bounding_radii,
        geometry.pmt_bvh_bounds,
        geometry.pmt_bvh_nodes,
        geometry.pmt_bvh_tri_ids,
        geometry.pmt_bvh_roots,
        geometry.lappd_data,
        geometry.lappd_strip_axes,
        geometry.tank_radius,
        geometry.tank_z_min,
        geometry.tank_z_max,
        geometry.lappd_housing_data,
        geometry.annie_lappd_data,
        geometry.surfboard_data,
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
    from litepath.optics import evaluate_hit, rayleigh_scatter_dir

    if rng is None:
        rng = np.random.default_rng()

    origins = np.asarray(origins, dtype=np.float32).copy()
    directions = np.asarray(directions, dtype=np.float32).copy()
    n = len(origins)

    total_path = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=bool)
    n_bounces = np.zeros(n, dtype=np.int32)
    detected_hits: list[np.ndarray] = []
    detected_bounces: list[np.int32] = []
    detected_indices: list[int] = []

    use_water = water_config is not None and water_config.is_active
    μ_total = water_config.mu_total if use_water else 0.0

    # n_surface_bounces tracks physical surface reflections only.
    # Water-scatter retraces don't count toward max_bounces.
    n_surface = 0
    while n_surface <= max_bounces:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break

        hits = trace_rays(origins[idx], directions[idx], geometry)

        next_alive = np.zeros(len(idx), dtype=bool)
        this_iter_had_surface = False

        for j in range(len(idx)):
            i = idx[j]
            if hits[j, HI] == 0:
                alive[i] = False
                continue

            mat_id = int(hits[j, HMAT])
            mat_opt = optics_config.get(mat_id)
            if mat_opt is None:
                alive[i] = False
                continue

            seg_len = hits[j, HT]

            # ---- Water attenuation (absorption / Rayleigh scatter) ----
            if use_water and μ_total > 0:
                u = rng.random()
                if u < 1.0 - np.exp(-seg_len * μ_total):
                    s = -np.log(1.0 - u) / μ_total
                    if water_config.mu_abs > 0 and rng.random() < water_config.mu_abs / μ_total:
                        # Absorbed
                        alive[i] = False
                        continue
                    # Rayleigh scatter — not a surface bounce
                    total_path[i] += s
                    origins[i] = origins[i] + s * directions[i]
                    directions[i] = rayleigh_scatter_dir(directions[i], rng)
                    next_alive[j] = True
                    continue

            # ---- No water interaction → process surface hit ----
            total_path[i] += seg_len

            incident_dir = directions[i]
            normal = hits[j, HNX:HNZ + 1]
            action, new_dir = evaluate_hit(mat_opt, incident_dir, normal, n_water, rng)

            if action == "detect":
                hits[j, HT] = total_path[i]
                detected_hits.append(hits[j:j + 1].copy())
                detected_bounces.append(np.int32(n_surface))
                detected_indices.append(i)
                alive[i] = False
            elif action == "reflect":
                origins[i] = hits[j, HX:HZ + 1]
                directions[i] = new_dir
                next_alive[j] = True
                this_iter_had_surface = True
            # absorb: drop

        alive[idx] = next_alive

        if this_iter_had_surface:
            n_surface += 1
        elif not use_water:
            break  # no surface interactions and no water → done

    if detected_hits:
        hits_out = np.concatenate(detected_hits, axis=0)
        bounces_out = np.array(detected_bounces, dtype=np.int32)
        indices_out = np.array(detected_indices, dtype=np.int32)
        return hits_out, bounces_out, indices_out
    return (np.zeros((0, N_HIT_COLS), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32))
