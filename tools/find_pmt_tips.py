"""Find PMT bulb tips in the 3D scan mesh and display them interactively.

For each PMT in the metrological scan, locate the bulb tip (the furthest
point along the PMT's outward direction, where the tape X was placed).

Strategy: use the PMT positions as LitePath places them, taken back into
the GDML structure frame the scan meshes are registered in, as seeds; then
search the AllPMTs scan mesh for the tip vertex in each PMT's neighborhood.

PMT tip directions:
  - Bottom PMTs (LUX, panel 0):  tip points upward  → max Z
  - Top PMTs (ETEL, panel 9):    tip points downward → min Z
  - Barrel PMTs (panels 1–8):    tip points inward   → min XY distance from origin

Usage:
    python3 tools/find_pmt_tips.py
    python3 tools/find_pmt_tips.py --view        (launch 3D viewer)
    python3 tools/find_pmt_tips.py --no-view     (save only, no viewer)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import KDTree

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCAN_DIR = PROJECT_DIR / "scan files by part" / "transformed"
CSV_PATH = PROJECT_DIR / "PMTPositions_Scan.txt"            # panels, directions, type codes
SEED_TABLE = PROJECT_DIR / "geometry" / "base" / "FullTankPMTGeometry.csv"  # positions (453/455 fixed)

# PMT type info
PMT_TYPE_NAMES = {
    (0, 0): "LUX",
    (0, 1): "Watchboy", (0, 2): "Watchboy", (0, 3): "Watchboy",
    (0, 4): "Watchboy", (0, 5): "Watchboy", (0, 6): "Watchboy",
    (0, 7): "Watchboy", (0, 8): "Watchboy",
    (1, 9): "ETEL",
    (2, 1): "Hamamatsu", (2, 2): "Hamamatsu", (2, 3): "Hamamatsu",
    (2, 4): "Hamamatsu", (2, 5): "Hamamatsu", (2, 6): "Hamamatsu",
    (2, 7): "Hamamatsu", (2, 8): "Hamamatsu",
    (3, 2): "Watchman", (3, 4): "Watchman", (3, 6): "Watchman", (3, 8): "Watchman",
}


def transform_csv_to_gdml() -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Load CSV and transform to GDML structure frame.

    Returns (centers_mm, tube_ids, panel_nrs, dirs, type_codes).
    """
    from litepath.geometry.frame import frame_to_structure
    from litepath.pmt_loader import load_pmts

    data = np.loadtxt(CSV_PATH, delimiter="\t")
    tube_ids = data[:, 0].astype(int)
    panel_nrs = data[:, 1].astype(int)
    dirs = data[:, 5:8]
    type_codes = data[:, 8].astype(int)

    # PMT positions as LitePath builds them from the ToolAnalysis table, taken
    # back into the GDML structure frame the scan meshes are registered in
    pmts = load_pmts(SEED_TABLE)
    assert list(pmts["detector_nums"]) == list(tube_ids)
    centers = frame_to_structure(np.asarray(pmts["centers"], dtype=np.float64))
    return centers, tube_ids, panel_nrs, dirs, type_codes


def find_tips_from_seeds(
    mesh_verts: np.ndarray,
    seeds: np.ndarray,
    tube_ids: np.ndarray,
    panel_nrs: np.ndarray,
    dirs: np.ndarray,
    type_codes: np.ndarray,
    search_radius: float = 400,
) -> list[dict]:
    """For each CSV seed, find the bulb tip in the scan mesh.

    For each PMT:
      1. Query mesh vertices within search_radius of the CSV seed center
      2. Determine tip direction from panel+type
      3. Score vertices by projection onto tip direction,
         biased toward vertices close to the seed
      4. Pick the best vertex as the tip

    Returns list of tip dicts.
    """
    tree = KDTree(mesh_verts)
    tips = []

    for i in range(len(seeds)):
        seed = seeds[i]
        panel = int(panel_nrs[i])
        tcode = int(type_codes[i])
        tube_id = int(tube_ids[i])

        # Determine PMT type name
        ptype = PMT_TYPE_NAMES.get((tcode, panel), "Unknown")

        # Determine tip direction
        if panel == 0:  # LUX bottom: upward
            tip_dir = np.array([0.0, 0.0, 1.0])
        elif panel == 9:  # ETEL top: downward
            tip_dir = np.array([0.0, 0.0, -1.0])
        else:  # barrel: inward (toward origin in XY)
            xy = seed[:2]
            dist = np.linalg.norm(xy)
            if dist > 1:
                tip_dir = np.array([-xy[0] / dist, -xy[1] / dist, 0.0])
            else:
                tip_dir = np.array([0.0, 0.0, 0.0])

        # Search the mesh
        idx = tree.query_ball_point(seed, search_radius)
        if len(idx) == 0:
            tips.append({
                "tube_id": tube_id,
                "panel": panel,
                "type": ptype,
                "tip_x": float(seed[0]),
                "tip_y": float(seed[1]),
                "tip_z": float(seed[2]),
                "csv_x": float(seed[0]),
                "csv_y": float(seed[1]),
                "csv_z": float(seed[2]),
                "offset_mm": 0.0,
                "n_verts": 0,
                "reliability": 0.0,
                "found": False,
            })
            continue

        near = mesh_verts[idx]

        # Score: combination of tip-direction projection + proximity to seed
        # Score = α * proj - β * dist_to_seed
        # This favors vertices both in the tip direction AND close to the seed
        vecs = near - seed
        proj = vecs @ tip_dir
        dists = np.linalg.norm(vecs, axis=1)

        # Normalize: proj ~ 0-200mm, dists ~ 0-400mm
        alpha = 1.0
        beta = 1.5  # penalize distance more
        scores = alpha * (proj / 200) - beta * (dists / 400)

        best_idx = idx[scores.argmax()]
        best = mesh_verts[best_idx]
        offset = float(np.linalg.norm(best - seed))
        n = len(idx)

        # Reliability
        vert_score = min(1.0, n / 50)
        tip_extent = float(proj.max())  # how far tip extends in correct direction
        extent_score = min(1.0, tip_extent / 100)
        reliability = max(0.0, min(1.0, (vert_score + extent_score) / 2))

        tips.append({
            "tube_id": tube_id,
            "panel": panel,
            "type": ptype,
            "tip_x": float(best[0]),
            "tip_y": float(best[1]),
            "tip_z": float(best[2]),
            "csv_x": float(seed[0]),
            "csv_y": float(seed[1]),
            "csv_z": float(seed[2]),
            "offset_mm": offset,
            "n_verts": n,
            "reliability": reliability,
            "found": True,
        })

    return tips


def build_viewer_scene(scan_verts: np.ndarray, scan_faces: np.ndarray,
                       tips: list[dict]):
    """Build a trimesh scene showing the scan with tip markers."""
    scene = trimesh.Scene()

    # Scan mesh (translucent)
    scan_mesh = trimesh.Trimesh(vertices=scan_verts, faces=scan_faces)
    try:
        scan_mesh.visual.face_colors = [255, 136, 68, 100]
    except Exception:
        pass
    scene.add_geometry(scan_mesh, node_name="scan_mesh")

    # Tip markers
    found = [t for t in tips if t.get("found", False)]
    not_found = [t for t in tips if not t.get("found", False)]

    for tip_list, color, label in [
        (found, [0, 255, 0, 255], "tip_found"),
        (not_found, [255, 0, 0, 160], "tip_not_found"),
    ]:
        for i, t in enumerate(tip_list):
            pos = [t["tip_x"], t["tip_y"], t["tip_z"]]
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=8)
            sphere.apply_translation(pos)
            try:
                sphere.visual.vertex_colors = color
            except Exception:
                pass
            scene.add_geometry(sphere, node_name=f"{label}_{i}")

    # CSV seed positions (yellow, smaller)
    for i, t in enumerate(tips):
        pos = [t["csv_x"], t["csv_y"], t["csv_z"]]
        sphere = trimesh.creation.icosphere(subdivisions=0, radius=5)
        sphere.apply_translation(pos)
        try:
            sphere.visual.vertex_colors = [200, 200, 0, 160]
        except Exception:
            pass
        scene.add_geometry(sphere, node_name=f"csv_{i}")

    return scene


def main():
    parser = argparse.ArgumentParser(description="Find PMT bulb tips in 3D scan")
    parser.add_argument("--view", action="store_true", help="Show interactive viewer")
    parser.add_argument("--no-view", action="store_true", help="Skip viewer (save only)")
    parser.add_argument("--radius", type=float, default=400,
                        help="Search radius around CSV seed (mm, default=400)")
    args = parser.parse_args()

    print("=== Finding PMT bulb tips in 3D scan ===\n")

    # Load scan mesh
    print("Loading AllPMTs scan mesh...")
    mesh_path = SCAN_DIR / "AllPMTs.ply"
    npy_path = SCAN_DIR / "AllPMTs_verts.npy"   # written by preprocess_scan.py
    faces = None
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path))
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    elif npy_path.exists():
        verts = np.load(npy_path).astype(np.float64)
    else:
        print(f"ERROR: neither {mesh_path} nor {npy_path} found. Run register_scan.py first.")
        return
    print(f"  {len(verts)} vertices")

    # Load and transform CSV positions
    print("Loading CSV PMT positions...")
    seeds, tube_ids, panel_nrs, dirs, type_codes = transform_csv_to_gdml()
    print(f"  {len(seeds)} PMTs loaded")

    # Find tips
    print(f"\nFinding tips (search radius = {args.radius} mm)...")
    tips = find_tips_from_seeds(verts, seeds, tube_ids, panel_nrs, dirs,
                                 type_codes, search_radius=args.radius)

    n_found = sum(1 for t in tips if t.get("found", False))
    print(f"  Found tips for {n_found}/{len(tips)} PMTs")

    # Save results
    out_dir = SCAN_DIR
    out_dir.mkdir(exist_ok=True)

    dtype = [
        ("tube_id", "i4"), ("panel", "i4"),
        ("tip_x", "f4"), ("tip_y", "f4"), ("tip_z", "f4"),
        ("csv_x", "f4"), ("csv_y", "f4"), ("csv_z", "f4"),
        ("offset_mm", "f4"), ("n_verts", "i4"),
        ("reliability", "f4"), ("found", "?"),
    ]
    arr = np.array(
        [(t["tube_id"], t["panel"],
          t["tip_x"], t["tip_y"], t["tip_z"],
          t["csv_x"], t["csv_y"], t["csv_z"],
          t["offset_mm"], t["n_verts"],
          t["reliability"], t["found"]) for t in tips],
        dtype=dtype,
    )
    np.save(out_dir / "pmt_tip_positions.npy", arr)

    # CSV output
    import csv
    csv_out = out_dir / "pmt_tip_positions.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tube_id", "panel", "type",
            "tip_x", "tip_y", "tip_z",
            "csv_x", "csv_y", "csv_z",
            "offset_mm", "n_verts", "reliability", "found",
        ])
        for t in tips:
            writer.writerow([
                t["tube_id"], t["panel"], t["type"],
                f"{t['tip_x']:.2f}", f"{t['tip_y']:.2f}", f"{t['tip_z']:.2f}",
                f"{t['csv_x']:.2f}", f"{t['csv_y']:.2f}", f"{t['csv_z']:.2f}",
                f"{t['offset_mm']:.1f}", t["n_verts"],
                f"{t['reliability']:.3f}", int(t.get("found", False)),
            ])
    print(f"\nSaved: {csv_out}")
    print(f"Saved: {out_dir / 'pmt_tip_positions.npy'}")

    # Print summary
    print(f"\n{'ID':>6} {'Panel':>5} {'Type':>12} {'Tip X':>9} {'Tip Y':>9} {'Tip Z':>9} {'CSV Z':>8} {'Off':>6} {'Rel':>5}")
    print("-" * 80)
    for t in sorted(tips, key=lambda x: (x["panel"], x["tip_z"])):
        f = "✓" if t.get("found", False) else "✗"
        print(f"{t['tube_id']:>6} {t['panel']:>5} {t['type']:>12} "
              f"{t['tip_x']:>9.1f} {t['tip_y']:>9.1f} {t['tip_z']:>9.1f} "
              f"{t['csv_z']:>8.0f} {t['offset_mm']:>6.1f} {t['reliability']:.3f}")

    # Launch viewer
    should_view = args.view or not args.no_view
    if should_view and faces is None:
        print("\nThe viewer needs AllPMTs.ply (only the .npy vertices were found); skipping it.")
    elif should_view:
        try:
            print("\nLaunching 3D viewer...")
            scene = build_viewer_scene(verts, faces, tips)
            scene.show()
        except Exception as e:
            print(f"Viewer error: {e}")
            print("Try: pip install pyglet")
    else:
        print("\nDone. Use --view to launch the interactive viewer "
              "(green = tip found, green dot at tip; yellow = CSV seed; red = not found).")


if __name__ == "__main__":
    main()
