"""The LitePath detector frame: one definition, used by every tool.

LitePath frame (mm):

* ``z`` up, along the tank axis.  ``z = 0`` is 1980 mm below the tank
  centre (the origin of the GDML inner structure); the water runs from
  z = 19 to 3861 mm.
* ``+y`` along the beam (downstream), ``x = y × z`` (beam-right).
* The inner-structure octagon has its corners at 0°, 45°, …, 315° and its
  faces at 22.5° + k·45°.  The beam points at the 90° corner (between
  barrel panels 1 and 8, the central surfboard and LAPPD); it enters
  through the 270° corner (panels 4 | 5).  Barrel panels run clockwise
  seen from above (panel 1 at 67.5°, panel 2 at 22.5°, ...).  The bottom
  PMT grid and cross-bars run at 45° to the beam.

This orientation was checked against the ToolAnalysis PMT table and the
2019 PMT layout slides (M. O'Flaherty, ANNIE collaboration meeting).

Every input is brought into the frame by one of two conversions:

``tank_to_frame``
    WCSim tank coordinates (``x``, ``y`` up, ``z`` along the beam, origin
    at the tank centre): ``(x, y, z) → (−x, z, y + 1980)``.  This is what
    WCSim ROOT output uses.  PMT tables (ToolAnalysis CSV and the scan
    file, in cm) are offset from it; ``table_to_frame`` handles that.

``structure_to_frame``
    The GDML inner-structure frame (also used by the STEP manifest and
    the scan meshes): a 22.5° turn about ``z``.  WCSim places the
    structure with ``rotateZ(157.5°)``; Geant4 turns the frame, so the
    volume turns by −157.5° in the tank, and the tank → frame step adds
    180°: 180° − 157.5° = 22.5°.

Nothing else is rotated: the orientation is fixed, not a setting.  Things
LitePath places itself (surfboards, LAPPD housings) are placed directly in
this frame.
"""
from __future__ import annotations

import math

import numpy as np

#: z of the tank centre in the frame (half of WCLength = 3960 mm).
TANK_CENTRE_Z_MM = 1980.0

#: PMT tables are in cm with the tank centre at (0, -14.45, 168.1) cm.
TABLE_VERTICAL_OFFSET_CM = 14.45
TABLE_BEAM_CENTRE_CM = 168.1

#: Turn about z from the GDML structure frame to the LitePath frame.
STRUCTURE_TO_FRAME_DEG = 22.5

#: Azimuth of the beam direction (+y) and of the octagon faces.
BEAM_AZIMUTH_DEG = 90.0
FACE_OFFSET_DEG = 22.5


def tank_to_frame(points) -> np.ndarray:
    """WCSim tank coordinates (mm, (..., 3)) → LitePath frame (mm)."""
    p = np.asarray(points, dtype=np.float64)
    return np.stack([-p[..., 0], p[..., 2], p[..., 1] + TANK_CENTRE_Z_MM], axis=-1)


def frame_to_tank(points) -> np.ndarray:
    """Inverse of :func:`tank_to_frame`."""
    p = np.asarray(points, dtype=np.float64)
    return np.stack([-p[..., 0], p[..., 2] - TANK_CENTRE_Z_MM, p[..., 1]], axis=-1)


def tank_to_frame_direction(directions) -> np.ndarray:
    """WCSim tank direction vectors → LitePath frame."""
    d = np.asarray(directions, dtype=np.float64)
    return np.stack([-d[..., 0], d[..., 2], d[..., 1]], axis=-1)


def frame_to_tank_direction(directions) -> np.ndarray:
    """Inverse of :func:`tank_to_frame_direction`."""
    d = np.asarray(directions, dtype=np.float64)
    return np.stack([-d[..., 0], d[..., 2], d[..., 1]], axis=-1)


def table_to_frame(points_cm) -> np.ndarray:
    """PMT-table positions (cm; ToolAnalysis CSV or scan file) → frame (mm)."""
    p = np.asarray(points_cm, dtype=np.float64)
    tank = np.stack([p[..., 0], p[..., 1] + TABLE_VERTICAL_OFFSET_CM,
                     p[..., 2] - TABLE_BEAM_CENTRE_CM], axis=-1) * 10.0
    return tank_to_frame(tank)


def structure_to_frame(points) -> np.ndarray:
    """GDML structure-frame positions or directions (..., 3) → frame.

    Returns a new array with the input's float dtype.
    """
    p = np.asarray(points)
    dtype = p.dtype if p.dtype.kind == "f" else np.float64
    q = p.astype(np.float64)
    c = math.cos(math.radians(STRUCTURE_TO_FRAME_DEG))
    s = math.sin(math.radians(STRUCTURE_TO_FRAME_DEG))
    out = np.stack([q[..., 0] * c - q[..., 1] * s, q[..., 0] * s + q[..., 1] * c, q[..., 2]], axis=-1)
    return out.astype(dtype)


def frame_to_structure(points) -> np.ndarray:
    """Inverse of :func:`structure_to_frame`."""
    p = np.asarray(points)
    dtype = p.dtype if p.dtype.kind == "f" else np.float64
    q = p.astype(np.float64)
    c = math.cos(math.radians(STRUCTURE_TO_FRAME_DEG))
    s = math.sin(math.radians(STRUCTURE_TO_FRAME_DEG))
    out = np.stack([q[..., 0] * c + q[..., 1] * s, -q[..., 0] * s + q[..., 1] * c, q[..., 2]], axis=-1)
    return out.astype(dtype)


def barrel_face_normal(x: float, y: float) -> tuple[float, float, float]:
    """Inward unit normal of the octagon face nearest to azimuth atan2(y, x)."""
    az = math.atan2(y, x)
    step = math.pi / 4
    off = math.radians(FACE_OFFSET_DEG)
    face = round((az - off) / step) * step + off
    return (-math.cos(face), -math.sin(face), 0.0)
