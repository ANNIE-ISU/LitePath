"""Where along a muon track light is generated.

In the full detector, light is generated only on the part of the muon's
path inside the tank water.  This is a simulation cutoff, not a physical
range: the muon itself carries on past the tank wall (e.g. into the MRD),
and nothing here models a muon stopping.  Surfboards and LAPPD housings
block photons in the ray tracer but do not end the track.

The standalone LAPPD geometry (``batch-lappd``, ``viz-lappd``) has no tank;
there the track ends where the muon leaves the LAPPD housing box.
"""

import math

import numpy as np

from litepath.geometry.build import Geometry

# Muon speed used by the photon generators (cherenkov.py, scintillation.py).
_MUON_SPEED_M_S = 0.999999 * 299792458.0


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


def tank_light_segment(
    muon_pos: tuple[float, ...],
    muon_dir: tuple[float, float, float],
    tank_radius: float,
    tank_z_min: float,
    tank_z_max: float,
) -> tuple[float, float]:
    """Part of the muon's path inside the tank cylinder.

    Returns ``(start, length)`` in metres along the (unit) muon direction,
    measured from ``muon_pos``: ``start`` is 0 for a muon that starts inside
    the tank and the distance to the tank entry for one that starts outside;
    ``length`` is 0 if the muon never enters the tank.
    """
    px, py, pz = (float(v) for v in muon_pos[:3])
    dx, dy, dz = (float(v) for v in muon_dir[:3])
    lo, hi = 0.0, math.inf

    # Cylinder wall: |(px, py) + t (dx, dy)| <= R
    a = dx * dx + dy * dy
    c = px * px + py * py - tank_radius * tank_radius
    if a > 1e-12:
        b = px * dx + py * dy
        disc = b * b - a * c
        if disc < 0.0:
            return 0.0, 0.0
        root = math.sqrt(disc)
        lo = max(lo, (-b - root) / a)
        hi = min(hi, (-b + root) / a)
    elif c > 0.0:
        return 0.0, 0.0

    # End caps: z_min <= pz + t dz <= z_max
    if abs(dz) > 1e-12:
        t1 = (tank_z_min - pz) / dz
        t2 = (tank_z_max - pz) / dz
        lo = max(lo, min(t1, t2))
        hi = min(hi, max(t1, t2))
    elif not (tank_z_min <= pz <= tank_z_max):
        return 0.0, 0.0

    if hi <= lo:
        return 0.0, 0.0
    return lo / 1000.0, (hi - lo) / 1000.0


def light_segment(
    muon_pos: tuple[float, ...],
    muon_dir: tuple[float, float, float],
    geometry: Geometry | None,
) -> tuple[float, float]:
    """``(start, length)`` in metres of the stretch of track that emits light.

    Full detector: the part inside the tank (:func:`tank_light_segment`).
    Standalone LAPPD geometry (no tank): from the muon's start to where it
    leaves the LAPPD housing box.  No geometry: 4.0 m from the start.
    """
    if geometry is not None and geometry.tank_radius > 0.0:
        return tank_light_segment(muon_pos, muon_dir, geometry.tank_radius,
                                  geometry.tank_z_min, geometry.tank_z_max)

    if geometry is not None and geometry.lappd_housing_data.shape[0] > 0:
        from litepath.lappd_model import compute_housing_track_length
        housing = _housing_from_array(geometry.lappd_housing_data[0])
        return 0.0, compute_housing_track_length(muon_pos[:3], muon_dir[:3], housing)

    return 0.0, 4.0


def compute_track_length(
    muon_pos: tuple[float, ...],
    muon_dir: tuple[float, float, float],
    geometry: Geometry | None,
) -> float:
    """Length (m) of the track over which light is generated.

    See :func:`light_segment`.  In the full detector this is the length
    inside the tank, not a physical muon range.
    """
    return light_segment(muon_pos, muon_dir, geometry)[1]


def compute_tank_track_length(
    muon_pos: tuple[float, ...],
    muon_dir: tuple[float, float, float],
    tank_radius: float,
    tank_z_min: float,
    tank_z_max: float,
) -> float:
    """Length (m) of the muon's path inside the tank cylinder."""
    return tank_light_segment(muon_pos, muon_dir, tank_radius, tank_z_min, tank_z_max)[1]


def advance_muon(
    muon_pos: tuple[float, ...],
    muon_dir: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float, float]:
    """Muon state ``(x, y, z, t0)`` after travelling *distance* metres.

    Positions are in mm and ``t0`` in seconds, as the photon generators
    expect.  Used to start light generation at the tank entry.
    """
    t0 = float(muon_pos[3]) if len(muon_pos) > 3 else 0.0
    return (
        float(muon_pos[0]) + 1000.0 * distance * muon_dir[0],
        float(muon_pos[1]) + 1000.0 * distance * muon_dir[1],
        float(muon_pos[2]) + 1000.0 * distance * muon_dir[2],
        t0 + distance / _MUON_SPEED_M_S,
    )
