"""Muon track lengths through the tank and LAPPD housings."""

import math

import numpy as np

from litepath.geometry.build import Geometry


def _housing_from_array(arr: np.ndarray):
    """Convert a (16,) housing array back into a LAPPDHousing dataclass."""
    from litepath.lappd_model import LAPPDHousing
    cx, cy, cz = arr[0], arr[1], arr[2]
    ax = (arr[3], arr[4], arr[5])
    ay = (arr[6], arr[7], arr[8])
    az = (arr[9], arr[10], arr[11])
    hx, hy, hz = arr[12], arr[13], arr[14]
    return LAPPDHousing(
        centre=(cx, cy, cz),
        axes=(ax, ay, az),
        half=(float(hx), float(hy), float(hz)),
    )


def compute_tank_track_length(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    tank_radius: float,
    tank_z_min: float,
    tank_z_max: float,
) -> float:
    """Maximum muon track length (m) before exiting the tank cylinder.

    Returns 1.05 × the ray-to-exit distance in metres, or 4.0 m fallback.
    """
    Px, Py, Pz = muon_pos[0], muon_pos[1], muon_pos[2]
    Dx, Dy, Dz = muon_dir[0], muon_dir[1], muon_dir[2]
    R = tank_radius
    z_min = tank_z_min
    z_max = tank_z_max

    candidates = []

    # ---- Cylinder wall (radial exit) ----
    a = Dx * Dx + Dy * Dy
    if a > 1e-12:
        b = 2.0 * (Px * Dx + Py * Dy)
        c = Px * Px + Py * Py - R * R
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2.0 * a)
            if t1 > 0:
                z_at_wall = Pz + t1 * Dz
                if z_min <= z_at_wall <= z_max:
                    candidates.append(t1)

    # ---- Top cap (Z = z_max) ----
    if Dz > 0:
        t = (z_max - Pz) / Dz
        if t > 0:
            r2 = (Px + t * Dx) ** 2 + (Py + t * Dy) ** 2
            if r2 <= R * R:
                candidates.append(t)

    # ---- Bottom cap (Z = z_min) ----
    if Dz < 0:
        t = (z_min - Pz) / Dz
        if t > 0:
            r2 = (Px + t * Dx) ** 2 + (Py + t * Dy) ** 2
            if r2 <= R * R:
                candidates.append(t)

    if not candidates:
        return 4.0

    track_mm = min(candidates)
    return max(track_mm * 1.05 / 1000.0, 0.5)


def compute_track_length(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    geometry: Geometry | None,
) -> float:
    """Geometry-aware muon track length in metres.

    Delegates to ``compute_tank_track_length`` or
    ``compute_housing_track_length`` based on the geometry contents.
    Also tests against surfboard boxes and returns the shortest
    distance.  Falls back to 4.0 m when no geometry is available.
    """
    pos3 = muon_pos[:3] if len(muon_pos) > 3 else muon_pos
    dir3 = muon_dir[:3] if len(muon_dir) > 3 else muon_dir

    from litepath.lappd_model import compute_housing_track_length

    best = 1e30

    if geometry is not None and geometry.lappd_housing_data.shape[0] > 0:
        hd = geometry.lappd_housing_data[0]
        housing = _housing_from_array(hd)
        best = min(best, compute_housing_track_length(pos3, dir3, housing))

    if geometry is not None:
        for s in range(geometry.surfboard_data.shape[0]):
            row = geometry.surfboard_data[s]
            cx, cy, cz = row[0], row[1], row[2]
            ax = (row[3], row[4], row[5])
            ay = (row[6], row[7], row[8])
            az = (row[9], row[10], row[11])
            hx, hy, hz = row[12], row[13], row[14]

            ox, oy, oz = pos3
            dx, dy, dz = dir3
            t_min = -1e30
            t_max = 1e30
            for aax, aay, aaz, h in [(ax[0], ax[1], ax[2], hx),
                                       (ay[0], ay[1], ay[2], hy),
                                       (az[0], az[1], az[2], hz)]:
                denom = dx * aax + dy * aay + dz * aaz
                oc = (ox - cx) * aax + (oy - cy) * aay + (oz - cz) * aaz
                if abs(denom) > 1e-30:
                    t0 = (-h - oc) / denom
                    t1 = (h - oc) / denom
                else:
                    t0 = -1e30 if (-h - oc) < 0 else 1e30
                    t1 = 1e30 if (h - oc) > 0 else -1e30
                if t0 > t1:
                    t0, t1 = t1, t0
                if t0 > t_min:
                    t_min = t0
                if t1 < t_max:
                    t_max = t1
                if t_min > t_max:
                    break
            else:
                if t_max > 0 and t_min > -1e-6:
                    track_mm = max(t_max, 0.0)
                    best = min(best, track_mm * 1.05 / 1000.0)

    if geometry is not None and best > 1e29:
        best = compute_tank_track_length(
            pos3, dir3,
            geometry.tank_radius, geometry.tank_z_min, geometry.tank_z_max,
        )

    if best > 1e29:
        return 4.0
    return max(best, 0.5)
