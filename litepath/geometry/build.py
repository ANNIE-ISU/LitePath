"""Detector geometry: the ``Geometry`` container and its builders."""

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from litepath import gdml_parser, step_parser
from litepath import pmt_loader
from litepath import pmt_mesh
from litepath.step_parser import LAPPD_HALF_SIZE


@dataclass
class Geometry:
    """Aggregate geometry passed to the Taichi kernel.

    Fields prefixed with mesh_/pmt_/lappd_ are GPU arrays consumed directly
    by trace_kernel().  The detectors list is the human-readable registry
    that maps detector_index → detector ID, type, position, etc.
    """
    # ---- GDML structure mesh ----
    mesh_vertices: np.ndarray   # (N, 3) float32 — triangle vertex XYZ (mm)
    mesh_triangles: np.ndarray  # (M, 3) int32   — vertex-index triplets

    # ---- PMT spheres ----
    pmt_centers: np.ndarray     # (P, 3) float32 — sphere centre in structure frame (mm)
    pmt_radii: np.ndarray       # (P,)   float32 — radius per PMT (mm)
    pmt_directions: np.ndarray  # (P, 3) float32 — inward-pointing unit normal

    # ---- Default LAPPD rectangles (bare photocathode) ----
    lappd_data: np.ndarray      # (L, 7) float32 — [cx,cy,cz, nx,ny,nz, half_size]
    lappd_strip_axes: np.ndarray  # (L, 3) float32 — strip-axis unit vector per LAPPD

    # ---- Tank cylinder bounds (used for _ray_tank_intersect) ----
    tank_radius: float  # mm
    tank_z_min: float   # mm
    tank_z_max: float   # mm

    # ---- ANNIE LAPPD housing model (when --lappd-model=annie) ----
    lappd_housing_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 16), dtype=np.float32)
    )  # (H, 16) float32 — see housing_to_arrays() for layout

    annie_lappd_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 7), dtype=np.float32)
    )  # (H, 7) float32 — photocathode rect [cx,cy,cz, nx,ny,nz, half]

    # ---- Detector registry (not used by the kernel) ----
    detectors: list = field(default_factory=list)  # list[DetectorInfo]

    # ---- Per-triangle material IDs for structure mesh ----
    mesh_material_ids: np.ndarray | None = None  # (M,) int32 — MaterialID per tri

    # ---- BVH acceleration for structure mesh ----
    bvh_node_min: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    bvh_node_max: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    bvh_node_left: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_node_right: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_start: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_end: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_tri_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    bvh_n_nodes: int = 0

    # ---- PMT body meshes (triangle soup per instance type, local frame) ----
    pmt_body_tris: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (T_global, 9) float32 — 3 vertices per row, local frame
    pmt_body_mat_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (T_global,) int32 — MaterialID per triangle
    pmt_body_offsets: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 0, 0], dtype=np.int32)
    )  # (5,) int32 — start index per mesh type 0-3 + sentinel

    # ---- PMT hardware (holder) meshes ----
    pmt_hw_tris: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (T_hw, 9) float32
    pmt_hw_mat_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (T_hw,) int32 — MaterialID per triangle (TEFLON=4, ACRYLIC=5)
    pmt_hw_offsets: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0], dtype=np.int32)
    )  # (3,) int32 — start index for types 4, 5 + sentinel

    # ---- Per-PMT instance data for mesh refinement ----
    pmt_rotmats: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=np.float32)
    )  # (P, 9) float32 — flattened R^T for local-frame transform
    pmt_mesh_types: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (P,) int32 — mesh type index per PMT (body)
    pmt_hw_types: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )  # (P,) int32 — HW type index per PMT (-1=none, 4=8", 5=10")
    pmt_instance_pos: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )  # (P, 3) float32 — instance position (mesh centroid)
    pmt_bounding_radii: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (P,) float32 — max(body, HW) bounding sphere radius

    # ---- Surfboard (obscurant PVC panel) oriented boxes ----
    surfboard_data: np.ndarray = field(
        default_factory=lambda: np.empty((0, 16), dtype=np.float32)
    )  # (S, 16) float32 — same layout as lappd_housing_data

    # ---- LAPPD correction state ----
    lappd_corrections_baked: np.ndarray | None = None  # (1,3) float32 — last-applied dx,dy,dz


def reload_lappd_corrections(geo: Geometry):
    """Read lappd_corrections.csv and apply delta to housing/kernel data."""
    # The file lives in the package root (litepath/), one level above this module.
    corr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lappd_corrections.csv")
    if not os.path.exists(corr_path):
        return
    n = geo.lappd_housing_data.shape[0]
    if n == 0:
        return

    new_corrs = np.zeros((n, 3), dtype=np.float32)
    with open(corr_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            if 0 <= idx < n:
                new_corrs[idx] = [float(row["dx"]), float(row["dy"]), float(row["dz"])]

    if geo.lappd_corrections_baked is None:
        geo.lappd_corrections_baked = new_corrs.copy()
        delta = new_corrs
    else:
        delta = new_corrs - geo.lappd_corrections_baked
        geo.lappd_corrections_baked = new_corrs.copy()

    for idx in range(n):
        if np.any(delta[idx] != 0.0):
            geo.lappd_housing_data[idx, 0:3] += delta[idx]
            geo.annie_lappd_data[idx, 0:3] += delta[idx]


def build_surfboards(n: int, tank_z_min: float, tank_z_max: float) -> np.ndarray:
    """Build oriented-box data for N surfboard obscurant panels.

    Surfboards are long rectangular PVC panels (2450×280×10 mm) mounted
    vertically on the octagonal columns at the forward-most vertices of
    the tank.  They absorb photons and stop muon tracks.

    Parameters
    ----------
    n : int
        Number of surfboards (0, 1, or 3).
    tank_z_min, tank_z_max : float
        Tank Z bounds (mm) — used to centre the surfboard vertically.

    Returns
    -------
    np.ndarray
        ``(n, 16)`` float32 — oriented-box layout matching
        ``lappd_housing_data``.

    Layout of the 16 columns::

        [cx, cy, cz, ax_x,ax_y,ax_z, ay_x,ay_y,ay_z, az_x,az_y,az_z, hx,hy,hz, pad]

    The local frame convention (same as the housing)::

        local_X : tangential to the tank (width direction, 280 mm total)
        local_Y : vertical, along +Z (length direction, 2450 mm total)
        local_Z : radially inward (thickness direction, 10 mm total)
    """
    if n not in (0, 1, 3):
        raise ValueError(f"n_surfboards must be 0, 1, or 3, got {n}")
    if n == 0:
        return np.empty((0, 16), dtype=np.float32)

    column_r = 1304.0
    z_centre = (tank_z_min + tank_z_max) / 2.0
    half_x = 140.0   # 280 mm / 2
    half_y = 1225.0  # 2450 mm / 2
    half_z = 5.0     # 10 mm / 2

    # User-requested offsets applied to all surfboards
    dz_vert = -114.0  # vertical (local_Y / +Z); net = +390 - 504
    dr_rad = 65.0     # +radial inward (local_Z toward tank centre); net = +50 + 15

    if n == 1:
        angles_deg = [90.0]
    else:
        angles_deg = [45.0, 90.0, 135.0]

    rows = []
    for a_deg in angles_deg:
        a = math.radians(a_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)

        # local_Z = radially inward
        az_x = -cos_a
        az_y = -sin_a
        az_z = 0.0

        # local_Y = vertical
        ay_x = 0.0
        ay_y = 0.0
        ay_z = 1.0

        # local_X = tangential = cross(local_Y, local_Z)
        ax_x = sin_a
        ax_y = -cos_a
        ax_z = 0.0

        cx = column_r * cos_a + dr_rad * az_x
        cy = column_r * sin_a + dr_rad * az_y
        cz = z_centre + dz_vert

        rows.append([
            cx, cy, cz,
            ax_x, ax_y, ax_z,
            ay_x, ay_y, ay_z,
            az_x, az_y, az_z,
            half_x, half_y, half_z,
            0.0,  # pad
        ])

    return np.array(rows, dtype=np.float32)


def build_surfboard_housings(surfboard_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build LAPPD housing arrays positioned in front of each surfboard.

    Each housing is offset from its surfboard along local_Z (radially inward)
    and shares the same local axes (aligned straight, not rotated).  The
    three housings are staggered vertically so the leftmost (45°) sits higher,
    the rightmost (135°) lower, and the centre (90°) at mid-Z, forming a
    diagonal in the visualisation frame.

    Parameters
    ----------
    surfboard_data : np.ndarray
        ``(S, 16)`` float32 — oriented-box surfboard data from build_surfboards().

    Returns
    -------
    housing_data : np.ndarray  ``(S, 16)`` float32 — lappd_housing_data layout.
    annie_data   : np.ndarray  ``(S, 7)`` float32  — annie_lappd_data layout.
    """
    from litepath.lappd_model import HOUSING_HALF, PC_LOCAL, PC_HALF

    n = surfboard_data.shape[0]
    if n == 0:
        return np.empty((0, 16), dtype=np.float32), np.empty((0, 7), dtype=np.float32)

    housing_data = np.empty((n, 16), dtype=np.float32)
    annie_data = np.empty((n, 7), dtype=np.float32)

    gap = 50.0  # mm between surfboard back face and housing front face
    hhx, hhy, hhz = HOUSING_HALF  # (165, 215, 30)

    # Vertical stagger: leftmost (45°) higher, rightmost (135°) lower, centre at mid-Z
    z_diag_offsets = [800.0, 0.0, -800.0] if n == 3 else [0.0]

    for i in range(n):
        row = surfboard_data[i]
        cx, cy, cz = row[0], row[1], row[2]
        ax = row[3:6]
        ay = row[6:9]
        az = row[9:12]
        shz = row[14]  # surfboard half_z (thickness)

        # Offset centre along local_Z (radially inward)
        offset = shz + gap + hhz
        hcx = cx + offset * az[0]
        hcy = cy + offset * az[1]
        hcz = cz + offset * az[2]

        # Vertical stagger along local_Y (global Z)
        hcz += z_diag_offsets[i]

        # Use surfboard axes directly (aligned straight, no rotation)
        hax = ax
        hay = ay
        haz = az

        # PC centre = housing centre + PC_LOCAL in housing local frame
        plx, ply, plz = PC_LOCAL  # (0, -45, 3.5)
        pcx = hcx + plx * hax[0] + ply * hay[0] + plz * haz[0]
        pcy = hcy + plx * hax[1] + ply * hay[1] + plz * haz[1]
        pcz = hcz + plx * hax[2] + ply * hay[2] + plz * haz[2]

        housing_data[i] = [
            hcx, hcy, hcz,
            hax[0], hax[1], hax[2],
            hay[0], hay[1], hay[2],
            haz[0], haz[1], haz[2],
            hhx, hhy, hhz,
            0.0,
        ]
        annie_data[i] = [
            pcx, pcy, pcz,
            haz[0], haz[1], haz[2],  # PC normal = local_Z (inward)
            PC_HALF[0],
        ]

    return housing_data, annie_data


def build_geometry(
    gdml_path: Path,
    step_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    pmt_csv_path: Optional[Path] = None,
    lappd_indices: Optional[list[int]] = None,
    no_lappd: bool = False,
    no_gdml: bool = False,
    no_pmt_holders: bool = False,
    z_offset: float = 0.0,
    lappd_model: str = "default",
    bottom_rotation_deg: float = 45.0,
    bottom_spin_deg: float = 0.0,
    det_rotation_deg: float = 22.5,
    n_surfboards: int = 0,
) -> Geometry:
    """Build a Geometry from a GDML file and optional PMT/STEP data sources.

    Stages:
      1. Parse the GDML structure mesh (skipped when no_gdml=True).
      2. Load PMT positions from CSV, or from STEP manifest as fallback.
      3. Load the STEP manifest (if available) for LAPPD and tank info.
      4. Build default LAPPD rectangles from manifest candidate positions.
      5. Optionally build the ANNIE LAPPD housing model.
      6. Build the detector registry (stable ID → DetectorInfo mapping).
      7. Load PMT body/hardware meshes (skipped when no_pmt_holders=True).
    """
    # ---- Stage 1: parse the GDML structure mesh ----
    if no_gdml:
        verts = np.zeros((0, 3), dtype=np.float32)
        tris = np.zeros((0, 3), dtype=np.int32)
        print("  Structure mesh: skipped (--no-gdml)")
    else:
        # Prefer a pre-built .npz cache (fast, no XML parsing).
        cache_path = gdml_path.with_suffix(".npz").with_name(
            gdml_path.stem + "_cache.npz"
        )
        if cache_path.exists():
            data = np.load(cache_path)
            verts, tris = data["vertices"], data["triangles"]
            print(f"  Structure mesh: {len(verts)} verts, {len(tris)} tris (cached)")
        else:
            verts, tris = gdml_parser.parse_gdml(gdml_path)
        if det_rotation_deg != 0.0 and verts.shape[0] > 0:
            pmt_loader.rotate_z(verts, det_rotation_deg)

    # ---- Stage 1b: build BVH for structure mesh ----
    from litepath.bvh import build_bvh
    bvh = build_bvh(verts, tris)

    # ---- Stage 2: load PMT positions ----
    # PMTs can come from: (a) Scan CSV file, (b) STEP manifest JSON, (c) STEP raw.
    # The CSV is preferred because it has per-PMT radii and type names.
    pmt_directions = np.zeros((0, 3), dtype=np.float32)
    if pmt_csv_path and pmt_csv_path.exists():
        pmt_data = pmt_loader.load_pmts(pmt_csv_path, z_offset=z_offset,
                                         bottom_rotation_deg=bottom_rotation_deg,
                                         bottom_spin_deg=bottom_spin_deg,
                                         det_rotation_deg=det_rotation_deg)
        pmt_centers = pmt_data["centers"]
        pmt_radii = pmt_data["radii"]
        pmt_directions = pmt_data["directions"]
    elif manifest_path and manifest_path.exists():
        manifest = step_parser.ComponentManifest.from_json(manifest_path)
        pmt_centers = np.array(manifest.pmt_centers, dtype=np.float32) if manifest.pmt_centers else np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.full(len(pmt_centers), manifest.pmt_radius, dtype=np.float32) if len(pmt_centers) > 0 else np.zeros(0, dtype=np.float32)
    elif step_path:
        manifest = step_parser.parse_step(step_path)
        pmt_centers = np.array(manifest.pmt_centers, dtype=np.float32) if manifest.pmt_centers else np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.full(len(pmt_centers), manifest.pmt_radius, dtype=np.float32) if len(pmt_centers) > 0 else np.zeros(0, dtype=np.float32)
    else:
        pmt_centers = np.zeros((0, 3), dtype=np.float32)
        pmt_radii = np.zeros(0, dtype=np.float32)

    # ---- Stage 3: load the STEP manifest (LAPPD candidates and tank bounds) ----
    manifest = None
    if manifest_path and manifest_path.exists():
        manifest = step_parser.ComponentManifest.from_json(manifest_path)
    elif step_path:
        manifest = step_parser.parse_step(step_path)

    # ---- Tank bounds ----
    # Used for the infinite-cylinder tank-wall intersection and for computing
    # the Z-centre when placing the ANNIE housing on the octagon.
    # The inner-structure manifest bbox does NOT reflect the tank dimensions,
    # so we always use hardcoded defaults (10 ft diameter, from detector specs).
    tank_radius = 1524.0
    tank_z_min = 19.0
    tank_z_max = 3861.0

    # Active LAPPD positions (shared by default rectangles and housing model)
    lappd_sources: list[tuple[float, float, float]] | None = None
    if not no_lappd and manifest:
        if lappd_indices is not None and manifest.lappd_candidates:
            lappd_sources = [tuple(manifest.lappd_candidates[i].center) for i in lappd_indices
                            if i < len(manifest.lappd_candidates)]
        if lappd_sources is None:
            lappd_sources = [tuple(c) for c in manifest.lappd_centers] if manifest.lappd_centers else None
        if lappd_sources is None and manifest.lappd_candidates:
            from litepath.step_parser import DEFAULT_LAPPD_INDICES
            lappd_sources = [tuple(manifest.lappd_candidates[i].center) for i in DEFAULT_LAPPD_INDICES
                            if i < len(manifest.lappd_candidates)]

    # ---- Surfboard obscurant panels (needed before Stage 5) ----
    surfboard_data = build_surfboards(n_surfboards, tank_z_min, tank_z_max)

    # ---- Stage 4: build default LAPPD rectangles ----
    # Each LAPPD is a square photocathode rectangle at a candidate position
    # from the STEP manifest.  The ANNIE model replaces one of these with
    # the full housing (Stage 5).
    lappd_data = np.zeros((0, 7), dtype=np.float32)
    lappd_strip_axes = np.zeros((0, 3), dtype=np.float32)
    housing_source: tuple[float, float, float] | None = None
    if lappd_sources:
        # If ANNIE model, place housing on an octagon vertex (22.5° from panel centre)
        if lappd_model == "annie" and len(lappd_sources) > 0:
            tank_cz = (tank_z_min + tank_z_max) / 2.0
            # Position housing so its back (+Z) face nearly touches
            # the cylindrical column at the octagon corner (r≈1304 at 22.5°).
            # The housing is 60 mm thick; centre is 30 mm inward from column.
            z_replaced = min(lappd_sources, key=lambda s: abs(s[2] - tank_cz))[2]
            from litepath.lappd_model import CORRECTION
            column_r = 1304.0
            half_thick = 30.0
            center_r = column_r - half_thick          # 1274 mm
            input_r  = center_r / CORRECTION           # pre-correction radius
            cos22 = np.cos(np.radians(22.5))
            sin22 = np.sin(np.radians(22.5))
            housing_source = (input_r * cos22, input_r * sin22, z_replaced)
            rect_sources = [s for s in lappd_sources if s[2] != z_replaced]
        else:
            rect_sources = lappd_sources

        lappd_list = []
        strip_list = []
        for cx, cy, cz in rect_sources:
            r = np.hypot(cx, cy)
            if r > 1:
                nlx = -cx / r
                nly = -cy / r
                nlz = 0.0
            else:
                nlx, nly, nlz = 0.0, 0.0, -1.0
            lappd_list.append([cx, cy, cz, nlx, nly, nlz, LAPPD_HALF_SIZE])
            # Strip axis = vertical (Z-axis) for barrel LAPPDs
            strip_list.append([0.0, 0.0, 1.0])
        lappd_data = np.array(lappd_list, dtype=np.float32) if lappd_list else np.zeros((0, 7), dtype=np.float32)
        lappd_strip_axes = np.array(strip_list, dtype=np.float32) if strip_list else np.zeros((0, 3), dtype=np.float32)
        if det_rotation_deg != 0.0 and lappd_data.shape[0] > 0:
            pmt_loader.rotate_z(lappd_data[:, :3], det_rotation_deg)
            # Also rotate the inward-normal direction (columns 3:6)
            pmt_loader.rotate_z(lappd_data[:, 3:6], det_rotation_deg)
            # Strip axis is Z-up (0,0,1) for barrel LAPPDs — stays unchanged by Z rotation

    # ---- Stage 5: ANNIE LAPPD housing model ----
    # When --lappd-model=annie, replaces the default rectangle closest to
    # the tank mid-plane with a 5-sided waterproof box (Kandemir design)
    # containing an off-centre photocathode.
    lappd_housing_data = np.empty((0, 16), dtype=np.float32)
    annie_lappd_data = np.empty((0, 7), dtype=np.float32)
    if lappd_model == "annie" and housing_source is not None:
        from litepath.lappd_model import build_housing, housing_to_arrays

        cx, cy, cz = housing_source
        r = max(np.hypot(cx, cy), 1.0)
        normal = (-cx / r, -cy / r, 0.0)

        housing = build_housing((cx, cy, cz), normal)
        lappd_housing_data, annie_lappd_data = housing_to_arrays(housing)
        if det_rotation_deg != 0.0:
            if lappd_housing_data.shape[0] > 0:
                # Columns: 0:3=centre, 3:6=ax, 6:9=ay, 9:12=az, 12:15=half (scalars), 15=pad
                pmt_loader.rotate_z(lappd_housing_data[:, 0:3], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 3:6], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 6:9], det_rotation_deg)
                pmt_loader.rotate_z(lappd_housing_data[:, 9:12], det_rotation_deg)
            if annie_lappd_data.shape[0] > 0:
                pmt_loader.rotate_z(annie_lappd_data[:, 0:3], det_rotation_deg)
                pmt_loader.rotate_z(annie_lappd_data[:, 3:6], det_rotation_deg)

    # ---- Stage 5b: surfboard-mounted LAPPD housings ----
    # Builds a housing in front of each surfboard (independent of Stage 5 housing source).
    if lappd_model == "annie" and n_surfboards > 0:
        sb_housing, sb_annie = build_surfboard_housings(surfboard_data)
        if sb_housing.shape[0] > 0:
            lappd_housing_data = np.concatenate([lappd_housing_data, sb_housing], axis=0)
            annie_lappd_data = np.concatenate([annie_lappd_data, sb_annie], axis=0)

    # ---- Stage 6: build detector registry ----
    # Creates a list of DetectorInfo objects with stable IDs (WCSim TubeIDs
    # 332–463 for PMTs, 1000+ for LAPPDs, 2000+ for ANNIE LAPPD) so that
    # hit records can be mapped back to hardware regardless of array indexing.
    from litepath.detectors import build_detector_registry

    pmt_types_list: list[str] = []
    pmt_det_nums: list[int] = []
    pmt_panels_list: list[int] = []
    if pmt_csv_path and pmt_csv_path.exists():
        pmt_types_list = pmt_data.get("types", [])
        pmt_det_nums = pmt_data.get("detector_nums", [])
        pmt_panels_list = pmt_data.get("panels", [])

    detectors = build_detector_registry(
        pmt_centers=pmt_centers,
        pmt_radii=pmt_radii,
        pmt_types=pmt_types_list,
        pmt_directions=pmt_directions,
        pmt_detector_nums=pmt_det_nums,
        pmt_panels=pmt_panels_list,
        lappd_rect_data=lappd_data if lappd_data.shape[0] > 0 else None,
        lappd_housing_data=lappd_housing_data if lappd_housing_data.shape[0] > 0 else None,
        annie_lappd_data=annie_lappd_data if annie_lappd_data.shape[0] > 0 else None,
    )

    # ---- Stage 7: PMT body meshes and instance data (for mesh-based ray tracing) ----
    pmt_body_tris = np.zeros((0, 9), dtype=np.float32)
    pmt_body_mat_ids = np.zeros(0, dtype=np.int32)
    pmt_body_offsets = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    pmt_hw_tris = np.zeros((0, 9), dtype=np.float32)
    pmt_hw_mat_ids = np.zeros(0, dtype=np.int32)
    pmt_hw_offsets = np.array([0, 0, 0], dtype=np.int32)
    pmt_rotmats_arr = np.zeros((0, 9), dtype=np.float32)
    pmt_mesh_types_arr = np.zeros(0, dtype=np.int32)
    pmt_hw_types_arr = np.zeros(0, dtype=np.int32)
    pmt_instance_pos_arr = np.zeros((0, 3), dtype=np.float32)
    pmt_bounding_radii_arr = np.zeros(0, dtype=np.float32)

    if not no_pmt_holders and pmt_csv_path and pmt_csv_path.exists() and len(pmt_centers) > 0:
        body_meshes = pmt_mesh.load_pmt_body_meshes()
        pmt_body_tris, pmt_body_mat_ids, pmt_body_offsets = \
            pmt_mesh.build_body_tris_arrays(body_meshes)

        hw_meshes = pmt_mesh.load_pmt_hw_meshes()
        pmt_hw_tris, pmt_hw_mat_ids, pmt_hw_offsets = \
            pmt_mesh.build_hw_tris_arrays(hw_meshes)

        _mesh_types = pmt_data.get("mesh_types", np.zeros(len(pmt_centers), dtype=np.int32))
        _rotmats = pmt_data.get("rotmats", np.zeros((len(pmt_centers), 9), dtype=np.float32))
        _inst_pos = pmt_data.get("instance_positions",
                                 np.zeros((len(pmt_centers), 3), dtype=np.float32))

        pmt_mesh_types_arr = np.asarray(_mesh_types, dtype=np.int32)
        pmt_rotmats_arr = np.asarray(_rotmats, dtype=np.float32)
        pmt_instance_pos_arr = np.asarray(_inst_pos, dtype=np.float32)

        # HW type per instance: Hamamatsu→4 (8" HW), Watchboy/Watchman→5 (10" HW)
        pmt_hw_types_arr = np.full(len(pmt_centers), -1, dtype=np.int32)
        for i in range(len(pmt_centers)):
            mt = int(pmt_mesh_types_arr[i])
            if mt == 2:
                pmt_hw_types_arr[i] = 4
            elif mt == 3:
                pmt_hw_types_arr[i] = 5

        # Bounding radius = max(body_br, hw_br) per instance
        pmt_bounding_radii_arr = np.zeros(len(pmt_centers), dtype=np.float32)
        for i in range(len(pmt_centers)):
            mt = int(pmt_mesh_types_arr[i])
            md = body_meshes.get(mt)
            br = md.bounding_radius if md is not None else 0.0
            hwmt = int(pmt_hw_types_arr[i])
            if hwmt >= 0:
                hw_md = hw_meshes.get(hwmt)
                if hw_md is not None:
                    br = max(br, hw_md.bounding_radius)
            pmt_bounding_radii_arr[i] = br

        n_loaded = sum(1 for v in body_meshes.values() if v is not None)
        print(f"  PMT body meshes: {n_loaded}/4 types loaded, "
              f"{pmt_body_tris.shape[0]} total tris, "
              f"{len(pmt_centers)} instances")
        if pmt_hw_tris.shape[0] > 0:
            print(f"  PMT HW meshes: {pmt_hw_tris.shape[0]} total tris")

    # Default: all structure mesh triangles are TEFLON (ID 4)
    n_tris = tris.shape[0]
    mesh_material_ids = np.full(n_tris, 4, dtype=np.int32) if n_tris > 0 else np.zeros(0, dtype=np.int32)

    return Geometry(
        mesh_vertices=verts,
        mesh_triangles=tris,
        mesh_material_ids=mesh_material_ids,
        pmt_centers=pmt_centers,
        pmt_radii=pmt_radii,
        pmt_directions=pmt_directions,
        lappd_data=lappd_data,
        lappd_strip_axes=lappd_strip_axes,
        tank_radius=tank_radius,
        tank_z_min=tank_z_min,
        tank_z_max=tank_z_max,
        lappd_housing_data=lappd_housing_data,
        annie_lappd_data=annie_lappd_data,
        surfboard_data=surfboard_data,
        detectors=detectors,
        pmt_body_tris=pmt_body_tris,
        pmt_body_mat_ids=pmt_body_mat_ids,
        pmt_body_offsets=pmt_body_offsets,
        pmt_hw_tris=pmt_hw_tris,
        pmt_hw_mat_ids=pmt_hw_mat_ids,
        pmt_hw_offsets=pmt_hw_offsets,
        pmt_rotmats=pmt_rotmats_arr,
        pmt_mesh_types=pmt_mesh_types_arr,
        pmt_hw_types=pmt_hw_types_arr,
        pmt_instance_pos=pmt_instance_pos_arr,
        pmt_bounding_radii=pmt_bounding_radii_arr,
        # BVH
        bvh_node_min=bvh.node_min,
        bvh_node_max=bvh.node_max,
        bvh_node_left=bvh.node_left,
        bvh_node_right=bvh.node_right,
        bvh_tri_start=bvh.tri_start,
        bvh_tri_end=bvh.tri_end,
        bvh_tri_ids=bvh.tri_ids,
        bvh_n_nodes=bvh.n_nodes,
    )


def build_single_lappd_geometry(
    cad_centre: tuple[float, float, float],
    cad_normal: tuple[float, float, float],
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Geometry:
    """Build a minimal Geometry containing a single ANNIE LAPPD housing.

    Useful for single-LAPPD visualisation, batch mode, or any context
    where only one housing is needed without the full detector geometry.
    """
    from litepath.detectors import build_detector_registry
    from litepath.lappd_model import build_housing, housing_to_arrays

    housing = build_housing(cad_centre, cad_normal, z_axis)
    hd, ad = housing_to_arrays(housing)
    detectors = build_detector_registry(
        pmt_centers=np.zeros((0, 3), dtype=np.float32),
        pmt_radii=np.zeros(0, dtype=np.float32),
        pmt_types=[],
        pmt_directions=np.zeros((0, 3), dtype=np.float32),
        pmt_detector_nums=[],
        lappd_rect_data=None,
        lappd_housing_data=hd,
        annie_lappd_data=ad,
    )

    return Geometry(
        mesh_vertices=np.zeros((0, 3), dtype=np.float32),
        mesh_triangles=np.zeros((0, 3), dtype=np.int32),
        pmt_centers=np.zeros((0, 3), dtype=np.float32),
        pmt_radii=np.zeros(0, dtype=np.float32),
        pmt_directions=np.zeros((0, 3), dtype=np.float32),
        lappd_data=np.zeros((0, 7), dtype=np.float32),
        lappd_strip_axes=np.zeros((0, 3), dtype=np.float32),
        tank_radius=0.0,
        tank_z_min=0.0,
        tank_z_max=0.0,
        lappd_housing_data=hd,
        annie_lappd_data=ad,
        detectors=detectors,
    )
