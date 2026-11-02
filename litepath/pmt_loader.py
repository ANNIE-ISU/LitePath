"""Load PMT positions from a PMT table into the LitePath frame.

Positions go straight from the table into the frame with
``litepath.geometry.frame.table_to_frame`` (one fixed conversion, no
rotation settings; see that module for the frame definition).

Direction vectors are generated from the panel number / PMT type (matching
WCSim's per-face rotation matrices) rather than using the table's direction
columns, which are not used by WCSim:

- Bottom PMTs (LUX)   →  (0, 0, 1)   upward
- Top PMTs (ETEL)     →  (0, 0, -1)  downward
- Barrel PMTs         →  inward normal of their octagon face
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from litepath.geometry.frame import barrel_face_normal, table_to_frame

# ---------------------------------------------------------------------------
# PMT radii by type (mm)
# ---------------------------------------------------------------------------
PMT_RADII = {
    "Hamamatsu": 101.6,  # 8-inch  (R5912)
    "Watchboy": 127.0,   # 10-inch (R7081)
    "Watchman": 127.0,   # 10-inch (R7081HQE)
    "LUX": 127.0,        # 10-inch (R7081)
    "ETEL": 139.7,       # 11-inch (D784KFLB)
}

# ---------------------------------------------------------------------------
# pmt_type → type name mapping (WCSim v7 collection order)
# ---------------------------------------------------------------------------
# WCTankCollectionNames for v7:
#   [0] R7081       → 10" LUX (panel 0) or Watchboy (panel 1-8)
#   [1] D784KFLB    → 11" ETEL (panel 9)
#   [2] R5912HQE    →  8" Hamamatsu (barrel)
#   [3] R7081HQE    → 10" Watchman (barrel)
# The scan file's pmt_type column indexes this vector.
# Panel number disambiguates pmt_type=0 (LUX vs Watchboy).
PMT_TYPE_NAMES: dict[tuple[int, int], str] = {}
# panel_nr=0 → LUX
for pt in (0,):
    PMT_TYPE_NAMES[(pt, 0)] = "LUX"
# panel_nr=9 → ETEL
for pt in (1,):
    PMT_TYPE_NAMES[(pt, 9)] = "ETEL"
# panel_nr=1-8, pmt_type=2 → Hamamatsu
for pn in range(1, 9):
    PMT_TYPE_NAMES[(2, pn)] = "Hamamatsu"
# panel_nr=1-8, pmt_type=3 → Watchman
for pn in range(1, 9):
    PMT_TYPE_NAMES[(3, pn)] = "Watchman"
# panel_nr=1-8, pmt_type=0 → Watchboy
for pn in range(1, 9):
    PMT_TYPE_NAMES[(0, pn)] = "Watchboy"

# PMT type → mesh type index for GDML rendering
# 0: LUX bottom, 1: ETEL top, 2: 8" Hamamatsu, 3: 10" Watchboy/Watchman
PMT_MESH_TYPE: dict[str, int] = {
    "LUX": 0,
    "ETEL": 1,
    "Hamamatsu": 2,
    "Watchboy": 3,
    "Watchman": 3,
}

# PMT type → rest-pose forward axis (direction from body-center to bulb tip, in GDML mesh frame)
# LUX/10":  10" R7081, axis ≈ (0155, 0050, 9999) → +Z
# ETEL:     11" D784KFLB, axis ≈ (-0.0139, -0.0199, -0.9997) → -Z
# Hamamatsu: 8" R5912, axis ≈ (-0.9998, -0.0019, -0.0210) → +X  (bulb tip at +X)
# Watch:     10" R7081, mesh recentered at bulb tip (origin). PCA tube axis direction.
#            offset=0 (bulb tip at origin → instance placed at sphere center)
PMT_FORWARD: dict[str, tuple[float, float, float]] = {
    "LUX": (0.0, 0.0, 1.0),
    "ETEL": (0.0, 0.0, -1.0),
    "Hamamatsu": (1.0, 0.0, 0.0),
    "Watchboy": (0.0, 1.0, 0.0),
    "Watchman": (0.0, 1.0, 0.0),
}

# PMT type → forward offset (mm from centroid to bulb tip, along forward axis)
PMT_FORWARD_OFFSET: dict[str, float] = {
    "LUX": 172.7,    # centroid → bulb tip along +Z (triangle-soup centroid)
    "ETEL": 174.7,   # centroid → bulb tip along -Z (triangle-soup centroid)
    "Hamamatsu": 145.5,
    "Watchboy": 193.0,    # centroid → bulb tip along +Y (from STEP mesh measurement)
    "Watchman": 193.0,    # same
}


def _ideal_direction(pmt_type: str, x: float, y: float
                     ) -> tuple[float, float, float]:
    """Return the ideal inward-pointing unit direction in the frame."""
    if pmt_type == "LUX":
        return 0.0, 0.0, 1.0
    if pmt_type == "ETEL":
        return 0.0, 0.0, -1.0
    if x * x + y * y > 1e-12:
        return barrel_face_normal(x, y)
    return 0.0, 0.0, 1.0


def load_pmts(scan_path: Path, z_offset: float = 0.0,
              bottom_spin_deg: float = 0.0) -> dict:
    """Read a PMT table and return PMT data in the LitePath frame.

    Args:
        scan_path: PMT table: a WCSim scan file (``PMTPositions_Scan.txt``)
                   or a ToolAnalysis ``FullTankPMTGeometry.csv`` (see
                   :mod:`litepath.geometry.elements`).
        z_offset: Additional vertical shift (mm) applied after the
                  standard transform.  Use this to fine-tune alignment.
        bottom_spin_deg: Extra spin about the tube's own forward axis for
                         panel-0 (bottom) PMTs only, in degrees.  Rotates
                         the mesh orientation without changing the position.

    Returns dict with:
        centers: (N, 3) float32 — sphere centres in mm
        radii: (N,) float32 — per-PMT radius in mm
        types: list of str — PMT type name per PMT
        directions: (N, 3) float32 — inward-pointing face normal (unit)
        detector_nums: list of int — channel key (= WCSim TubeID)
        records: list of PMTRecord — the table rows (IDs, labels, status)
    """
    from litepath.geometry.elements import read_pmt_table

    records = read_pmt_table(scan_path)
    tube_ids = np.array([r.channel_key for r in records], dtype=int)
    panel_nrs = np.array([r.panel for r in records], dtype=int)
    scan_xyz = np.array([r.scan_position_cm for r in records], dtype=np.float64)  # cm

    n = len(records)
    centers = np.zeros((n, 3), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    directions = np.zeros((n, 3), dtype=np.float32)
    type_names = []
    det_nums = []

    frame_xyz = table_to_frame(scan_xyz)  # mm
    frame_xyz[:, 2] += z_offset

    for i in range(n):
        x, y, z = frame_xyz[i]
        ptype = records[i].pmt_type
        centers[i] = [x, y, z]
        radii[i] = PMT_RADII.get(ptype, 127.0)
        directions[i] = _ideal_direction(ptype, x, y)
        type_names.append(ptype)
        det_nums.append(int(tube_ids[i]))

    # After the loop, compute per-PMT mesh type index and rotation quaternion
    mesh_type_indices = np.array([PMT_MESH_TYPE.get(t, 0) for t in type_names], dtype=np.int32)
    quaternions = np.zeros((n, 4), dtype=np.float32)
    instance_positions = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        fwd = PMT_FORWARD.get(type_names[i], (0.0, 0.0, 1.0))
        tgt = directions[i]
        q = _forward_to_quat(np.array(fwd, dtype=np.float64), tgt)

        # Spin about the tube's forward axis for bottom panel
        if bottom_spin_deg != 0.0 and panel_nrs[i] == 0:
            half = math.radians(bottom_spin_deg) / 2.0
            s = math.sin(half)
            q_spin = np.array([tgt[0] * s, tgt[1] * s, tgt[2] * s, math.cos(half)], dtype=np.float64)
            # Multiply q_spin * q
            x1, y1, z1, w1 = q_spin
            x2, y2, z2, w2 = q
            q = np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2,
                          w1*y2 - x1*z2 + y1*w2 + z1*x2,
                          w1*z2 + x1*y2 - y1*x2 + z1*w2,
                          w1*w2 - x1*x2 - y1*y2 - z1*z2], dtype=np.float32)

        quaternions[i] = q
        # Shift position so bulb tip is at sphere surface (center + radius * direction)
        off = PMT_FORWARD_OFFSET.get(type_names[i], 0.0)
        r = radii[i]
        instance_positions[i] = [centers[i][j] + (r - off) * tgt[j] for j in range(3)]

    rotmats = np.zeros((n, 9), dtype=np.float32)
    for i in range(n):
        rotmats[i] = quat_to_rotmat(quaternions[i])

    return {
        "centers": centers,
        "radii": radii,
        "types": type_names,
        "directions": directions,
        "detector_nums": det_nums,
        "panels": panel_nrs.tolist(),
        "mesh_types": mesh_type_indices,
        "quaternions": quaternions,
        "instance_positions": instance_positions,
        "rotmats": rotmats,
        "records": records,
    }


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert (4,) quaternion (x,y,z,w) to flattened 3×3 R^T (row-major, 9 floats).

    The returned matrix R^T transforms global vectors to the mesh local frame
    (v_local = R^T · v_global).  The kernel stores these 9 floats per PMT for
    efficient local-frame ray transformation.
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return R.T.flatten().astype(np.float32)


def _forward_to_quat(forward: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return unit quaternion (qx,qy,qz,qw) rotating *forward* onto *target*."""
    fn = np.linalg.norm(forward)
    tn = np.linalg.norm(target)
    if fn < 1e-12 or tn < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    f = forward / fn
    t = target / tn
    dot = float(np.clip(f @ t, -1.0, 1.0))
    if abs(dot - 1.0) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if abs(dot + 1.0) < 1e-12:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(f @ perp) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(f, perp)
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
    axis = np.cross(f, t)
    axis = axis / np.linalg.norm(axis)
    half = math.acos(dot) / 2.0
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)
