"""
Register scan PLY files to the structure GDML coordinate frame.

Uses the octagon geometry of the panels to determine the transform:
  scan → structure frame: X_gdml = s * R * (X_scan - C) + T

where:
  C = scan octagon center (XY)
  s = uniform scale factor (mm per scan unit)
  R = rotation aligning scan octagon to GDML octagon (~22.5°)
  T = translation to GDML origin (Z offset from layer surfaces)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

SCAN_DIR = Path(__file__).resolve().parent.parent / "scan files by part"

# Scan is in inches, GDML is in mm
INCH_TO_MM = 25.4


def compute_scan_octagon() -> dict:
    """Determine scan octagon center, panel angles, and scale from panel PLY files."""
    centers_xy = []
    for i in range(1, 9):
        mesh = trimesh.load(str(SCAN_DIR / f"Panel-{i}.ply"))
        c = mesh.vertices.mean(axis=0)
        centers_xy.append(c[:2])
    centers = np.array(centers_xy)

    center = centers.mean(axis=0)
    for _ in range(10):
        r = np.linalg.norm(centers - center, axis=1)
        dr = centers - center
        center = center + 0.5 * dr.T @ (r - r.mean()) / max(r.sum(), 1e-12)

    r = np.linalg.norm(centers - center, axis=1)
    apothem = float(r.mean())
    angles = np.degrees(np.arctan2(centers[:, 1] - center[1], centers[:, 0] - center[0]))
    return {"center_xy": center, "apothem": apothem, "angles": angles, "panel_centers": centers}


def compute_gdml_octagon() -> dict:
    """Determine GDML octagon geometry from the structure mesh."""
    from litepath.gdml_parser import parse_gdml

    gdml_path = Path(__file__).resolve().parent.parent / "PHASE2_INNER_STRUCTURE_closed.gdml"
    verts, _ = parse_gdml(gdml_path)

    bottom = verts[(verts[:, 2] >= 19.0) & (verts[:, 2] <= 30.0)]
    r = np.linalg.norm(bottom[:, :2], axis=1)
    apothem = float(r.min())
    circumradius = float(r.max())
    return {"apothem": apothem, "circumradius": circumradius}


def estimate_transform():
    # ---- Octagon center and apothem from panel vertices ----
    panel_files = sorted(SCAN_DIR.glob("Panel-[0-9].ply"))

    all_panel_pts = []
    panel_centers_xy = []

    for pf in panel_files:
        mesh = trimesh.load(str(pf))
        verts = np.asarray(mesh.vertices)
        all_panel_pts.append(verts)
        panel_centers_xy.append(verts[:, :2].mean(axis=0))

    all_panel_pts = np.concatenate(all_panel_pts, axis=0)
    panel_centers_xy = np.array(panel_centers_xy)

    scan_center = panel_centers_xy.mean(axis=0)
    radii = np.linalg.norm(panel_centers_xy - scan_center, axis=1)
    scan_apothem = float(radii.min())
    scan_angles = np.degrees(np.arctan2(
        panel_centers_xy[:, 1] - scan_center[1],
        panel_centers_xy[:, 0] - scan_center[0],
    ))

    # XY scale: scan is in inches, GDML in mm
    gdml_apothem_mm = 1304.3

    # ---- Precise rotation: match scan panel angles to GDML face angles ----
    gdml_face = np.arange(8) * 45.0
    scan_angles_sorted = np.sort([(a + 360) % 360 for a in np.where(scan_angles < 0, scan_angles + 360, scan_angles)])
    offsets = []
    used = set()
    for sa in scan_angles_sorted:
        best = None
        best_d = 360.0
        for j, ga in enumerate(gdml_face):
            if j in used:
                continue
            d = abs(sa - ga)
            d = min(d, 360 - d)
            if d < best_d:
                best_d = d
                best = j
        if best is not None:
            offsets.append(sa - gdml_face[best])
            used.add(best)
    # Combined: panel alignment (-22.91°) + bottom structure 45° CCW
    rot_deg = -np.mean(offsets) + 45.0
    rot_rad = math.radians(rot_deg)

    # ---- Z offset from matching horizontal cross-bracing ----
    # The GDML ring positions represent the TOP of the square tubing.
    # The scan captures the full tubing volume. z_off=160 aligned the
    # scan's tubing-bottom with the model's tubing-top. Subtract tubing
    # thickness (~2 inches = 50.8 mm) to align centers.
    z_offset = 160.0 - 50.8  # = 109.2 mm

    return {
        "scan_center_xy": scan_center.copy(),
        "s": INCH_TO_MM,
        "z_offset": z_offset,
        "z_offset_note": "aligned to ~1925mm GDML ring at scan Z≈70in",
        "rot_deg": rot_deg,
        "rot_rad": rot_rad,
        "scan_apothem": scan_apothem,
        "gdml_apothem": gdml_apothem_mm,
        "scan_panel_angles": scan_angles,
    }


def apply_transform(vertices: np.ndarray, params: dict) -> np.ndarray:
    """
    Transform vertices from scan coordinates to structure GDML frame.

    P = s * R * (P_scan - C)
    where:
      C = scan octagon center (XY only, Z unchanged in centering)
      R = rotation about Z by rot_rad
      s = uniform scale (using s_xy for XY, s_z for Z)
    """
    cx, cy = params["scan_center_xy"]
    c = math.cos(params["rot_rad"])
    sn = math.sin(params["rot_rad"])
    sc = params["s"]
    z_off = params["z_offset"]

    out = np.empty_like(vertices)
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    out[:, 0] = (dx * c - dy * sn) * sc
    out[:, 1] = (dx * sn + dy * c) * sc
    out[:, 2] = vertices[:, 2] * sc + z_off
    return out


def main():
    params = estimate_transform()
    print("=== Scan Registration Parameters (1 inch = 25.4 mm) ===\n")
    print(f"Scan octagon center:        ({params['scan_center_xy'][0]:.2f}, {params['scan_center_xy'][1]:.2f})")
    print(f"Scale:                      1 in = {params['s']} mm  (fixed)")
    print(f"Z offset:                   {params['z_offset']:.1f} mm  ({params['z_offset_note']})")
    print(f"Rotation:                   {params['rot_deg']:.2f}° (CW)")
    print()
    # Show key feature alignments
    s = params['s']; zo = params['z_offset']
    print(f"  Mid ring (scan 70 in):      {70*s+zo:.0f} mm  (GDML ~1925)")
    print(f"  Upper ring (scan 104 in):   {104*s+zo:.0f} mm  (GDML ~2807)")
    print(f"  SS bottom (scan -6.12 in):  {-6.12*s+zo:.0f} mm  (GDML 19)")
    print(f"  SS top (scan 144.98 in):    {144.98*s+zo:.0f} mm  (GDML 3861)")

    # Transform and save
    print("\n=== Transforming and Saving ===")
    output_dir = Path(__file__).resolve().parent.parent / "scan files by part" / "transformed"
    output_dir.mkdir(exist_ok=True)

    for fname in ["SuperStructure.ply", "BottomLayer.ply", "TopLayer.ply", "TankLid.ply"]:
        src = SCAN_DIR / fname
        if not src.exists():
            continue
        print(f"  Loading {fname}...")
        mesh = trimesh.load(str(src))
        new_verts = apply_transform(np.asarray(mesh.vertices), params)
        mesh.vertices = new_verts
        dst = output_dir / fname
        mesh.export(str(dst))
        print(f"    → {dst}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    ap_src = SCAN_DIR / "AllPMTs.ply"
    if ap_src.exists():
        print(f"  Loading AllPMTs.ply...")
        mesh = trimesh.load(str(ap_src))
        new_verts = apply_transform(np.asarray(mesh.vertices), params)
        mesh.vertices = new_verts
        dst = output_dir / "AllPMTs.ply"
        mesh.export(str(dst))
        print(f"    → {dst}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    # Also transform bottom/top PMTs
    for fname in ["BottomPMTs.ply", "TopPMTs.ply"]:
        src = SCAN_DIR / fname
        if not src.exists():
            continue
        print(f"  Loading {fname}...")
        mesh = trimesh.load(str(src))
        new_verts = apply_transform(np.asarray(mesh.vertices), params)
        mesh.vertices = new_verts
        dst = output_dir / fname
        mesh.export(str(dst))
        print(f"    → {dst}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    # Also transform individual panel PMTs
    for i in range(1, 9):
        src = SCAN_DIR / f"Panel-{i}-PMTs.ply"
        if not src.exists():
            continue
        mesh = trimesh.load(str(src))
        new_verts = apply_transform(np.asarray(mesh.vertices), params)
        mesh.vertices = new_verts
        dst = output_dir / f"Panel-{i}-PMTs.ply"
        mesh.export(str(dst))
        print(f"    → {dst}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    # Also transform individual panel structure meshes
    for i in range(1, 9):
        src = SCAN_DIR / f"Panel-{i}.ply"
        if not src.exists():
            continue
        mesh = trimesh.load(str(src))
        new_verts = apply_transform(np.asarray(mesh.vertices), params)
        mesh.vertices = new_verts
        dst = output_dir / f"Panel-{i}.ply"
        mesh.export(str(dst))
        print(f"    → {dst}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    print("\nDone.")


if __name__ == "__main__":
    main()
