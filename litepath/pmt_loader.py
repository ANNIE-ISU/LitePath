"""Load PMT positions from WCSim scan file, mapping to GDML coordinate frame.

Coordinate pipeline (scan → WCBarrel frame → structure rest frame):

  WCSim's ConstructANNIECylinderScan transforms scan columns (cm):
       pmt_x, pmt_y, pmt_z
    into WCBarrel (water volume) coordinates (mm):
       X_w  = pmt_x * 10
       Y_w  = (168.1 - pmt_z) * 10   (beam axis)
       Z_w  = (pmt_y + 14.45) * 10   (vertical)
    where 168.1 cm = beam centre, 14.45 cm = vertical offset.

  PMTs are placed directly into logicWCBarrel (water volume inside the
  steel tank). The inner structure GDML is placed in the same water
  volume with: rotateZ(157.5°), G4ThreeVector(0, 0, -0.5*WCLength).
  To get back to the structure rest frame (GDML frame):
       P_s = Rz(-157.5°) * P_w + (0, 0, 0.5*WCLength)

  Note: The viz server additionally applies Rx(-90°) at the frontend
  layer to both the structure mesh and PMT positions so the cylinder
  axis aligns with Y for display.

  3. Direction vectors are generated from the panel_nr / PMT type
      (matching WCSim's per-face rotation matrices) rather than using
      the scan file direction columns, which are not used by WCSim.
      - Bottom PMTs (LUX)   →  (0, 0, 1)   upward
      - Top PMTs (ETEL)     →  (0, 0, -1)  downward
      - Barrel PMTs         →  radially inward in the XY plane
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hall-frame constants (from scan-file metrology)
# ---------------------------------------------------------------------------
# Beam centre Z-position in scan coordinates (cm)
BEAM_CENTER_Z = 168.1  # cm
# Vertical offset between scan and hall frames (cm)
VERTICAL_OFFSET = 14.45  # cm

# ---------------------------------------------------------------------------
# Structure placement in water tank
# ---------------------------------------------------------------------------
# The GDML structure is placed in the water tank with:
#   rotateZ(157.5°) at G4ThreeVector(0, 0, -0.5*WCLength)
# WCLength = WCIDHeight = 3960 mm
WC_LENGTH = 3960.0  # mm
HALF_WC_LENGTH = WC_LENGTH / 2.0  # 1980 mm

# Rotation angle to undo the structure placement
STRUCTURE_ROTATION_DEG = 157.5
_rot_rad = math.radians(-STRUCTURE_ROTATION_DEG)
_cos_r = math.cos(_rot_rad)
_sin_r = math.sin(_rot_rad)

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


def _scan_to_tank(scan_x: float, scan_y: float, scan_z: float
                  ) -> tuple[float, float, float]:
    """Convert scan coordinates (cm) to hall frame (mm)."""
    x_h = scan_x * 10.0
    y_h = (BEAM_CENTER_Z - scan_z) * 10.0
    z_h = (scan_y + VERTICAL_OFFSET) * 10.0
    return x_h, y_h, z_h


def _tank_to_structure(x_w: float, y_w: float, z_w: float
                       ) -> tuple[float, float, float]:
    """Transform from WCBarrel (water tank) frame to structure rest frame.

    Undoes the WCSim structure placement: Rz(157.5°) + (0,0,-0.5*WCLength).
    PMT scan positions (already in WCBarrel frame) map directly to the
    GDML frame where the structure mesh lives.
    """
    x_s = x_w * _cos_r - y_w * _sin_r
    y_s = x_w * _sin_r + y_w * _cos_r
    z_s = z_w + HALF_WC_LENGTH
    return x_s, y_s, z_s


def rotate_z(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (N,3) array of positions/directions by angle_deg around Z (in-place)."""
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = x * c - y * s
    points[:, 1] = x * s + y * c
    return points


def _ideal_direction(pmt_type: str, x_s: float, y_s: float
                     ) -> tuple[float, float, float]:
    """Return the ideal inward-pointing unit direction in structure frame.

    LUX (bottom) → upward (+Z)
    ETEL (top)   → downward (-Z)
    Barrel types → octagon face inward normal in XY plane
    """
    if pmt_type == "LUX":
        return 0.0, 0.0, 1.0
    if pmt_type == "ETEL":
        return 0.0, 0.0, -1.0
    # Barrel — octagon face inward normal (closest of 8 faces)
    r = math.sqrt(x_s * x_s + y_s * y_s)
    if r > 1e-6:
        az = math.atan2(y_s, x_s)
        k = round(az / (math.pi / 4))
        face_angle = k * math.pi / 4
        return (-math.cos(face_angle), -math.sin(face_angle), 0.0)
    return 0.0, 0.0, 1.0


def load_pmts(scan_path: Path, z_offset: float = 0.0,
              bottom_rotation_deg: float = 0.0,
              bottom_spin_deg: float = 0.0,
              det_rotation_deg: float = 22.5) -> dict:
    """Parse WCSim scan file and return PMT data in structure rest frame.

    Pipeline: scan (cm) → hall frame (mm) → structure rest frame.
    The hall→structure step applies Rx(-90°) (cylinder-axis alignment)
    then Rz(-157.5°) + (0,0,+1980) (undo WCSim placement).

    Args:
        scan_path: Path to PMTPositions_Scan.txt.
        z_offset: Additional vertical shift (mm) applied after the
                  standard transform.  Use this to fine-tune alignment.
        bottom_rotation_deg: Extra rotation about Z for panel-0 (bottom)
                             PMTs only, in degrees.  Multiples of 22.5°
                             align with the 4-fold bottom structure.
        bottom_spin_deg: Extra spin about the tube's own forward axis for
                         panel-0 (bottom) PMTs only, in degrees.  Rotates
                         the mesh orientation without changing the position.

    Returns dict with:
        centers: (N, 3) float32 — sphere centres in mm
        radii: (N,) float32 — per-PMT radius in mm
        types: list of str — PMT type name per PMT
        directions: (N, 3) float32 — inward-pointing face normal (unit)
        detector_nums: list of int — PMTID from scan file
    """
    data = np.loadtxt(scan_path, delimiter="\t")
    # Columns: TubeID, panel_nr, scan_x, scan_y, scan_z, dirx, diry, dirz, pmt_type
    tube_ids = data[:, 0].astype(int)
    panel_nrs = data[:, 1].astype(int)
    scan_xyz = data[:, 2:5]   # (N, 3) in cm
    pmt_types_code = data[:, 8].astype(int)

    n = len(data)
    centers = np.zeros((n, 3), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    directions = np.zeros((n, 3), dtype=np.float32)
    type_names = []
    det_nums = []

    # Precompute bottom rotation (negate: visual alignment requires the
    # opposite sign from the standard CCW rotation matrix)
    _brot = -math.radians(bottom_rotation_deg)
    _bcos = math.cos(_brot)
    _bsin = math.sin(_brot)

    for i in range(n):
        sx, sy, sz = scan_xyz[i]
        pn = panel_nrs[i]
        pt = pmt_types_code[i]

        # Scan → WCBarrel (water tank) frame
        x_w, y_w, z_w = _scan_to_tank(sx, sy, sz)

        # WCBarrel → structure rest frame (undo WCSim placement)
        x_s, y_s, z_s = _tank_to_structure(x_w, y_w, z_w)

        # Vertical offset
        z_s += z_offset

        # Extra Z-rotation for bottom panel (panel 0)
        if bottom_rotation_deg != 0.0:
            xs = x_s
            x_s = xs * _bcos - y_s * _bsin
            y_s = xs * _bsin + y_s * _bcos

        # PMT type
        key = (pt, pn)
        ptype = PMT_TYPE_NAMES.get(key, "Unknown")
        radius = PMT_RADII.get(ptype, 127.0)

        # Idealized inward-pointing direction
        dx_s, dy_s, dz_s = _ideal_direction(ptype, x_s, y_s)

        centers[i] = [x_s, y_s, z_s]
        radii[i] = radius
        directions[i] = [dx_s, dy_s, dz_s]
        type_names.append(ptype)
        det_nums.append(int(tube_ids[i]))

    # Apply global detector rotation (Z-axis) to all PMT positions and directions
    if det_rotation_deg != 0.0:
        rotate_z(centers, det_rotation_deg)
        rotate_z(directions, det_rotation_deg)

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
