"""Spatial analysis of PMT components in the STEP file."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")

print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solid_objs = result.solids().objects
print(f"Total solids: {len(solid_objs)}")

# Extract all solid info
all_info = []
for i, solid in enumerate(solid_objs):
    try:
        vol = solid.Volume()
        bb = solid.BoundingBox()
        dx = bb.xmax - bb.xmin
        dy = bb.ymax - bb.ymin
        dz = bb.zmax - bb.zmin
        cx = (bb.xmin + bb.xmax) / 2
        cy = (bb.ymin + bb.ymax) / 2
        cz = (bb.zmin + bb.zmax) / 2
        all_info.append({
            "idx": i,
            "vol": vol, "solid": solid,
            "dx": dx, "dy": dy, "dz": dz,
            "cx": cx, "cy": cy, "cz": cz,
        })
    except Exception as e:
        print(f"Error on solid {i}: {e}")

print(f"Processed {len(all_info)} solids.\n")

# ============================================================
# DIAGNOSTIC: All solids with volume 3.5M-4.5M (glass range)
# ============================================================
print("=== DIAGNOSTIC: All solids with volume 3.5M-4.5M ===")
glass_candidates = [i for i in all_info if 3_500_000 <= i["vol"] <= 4_500_000]
print(f"Count: {len(glass_candidates)}")
for info in sorted(glass_candidates, key=lambda x: x["vol"]):
    sd = sorted([info["dx"], info["dy"], info["dz"]])
    print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
          f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
          f"sorted=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})  "
          f"cz={info['cz']:>8.1f}")

# Also diagnostic: base volumes
print("\n=== DIAGNOSTIC: All solids with volume 6.5M-7.5M (base range) ===")
base_candidates = [i for i in all_info if 6_500_000 <= i["vol"] <= 7_500_000]
print(f"Count: {len(base_candidates)}")
for info in sorted(base_candidates, key=lambda x: x["vol"]):
    sd = sorted([info["dx"], info["dy"], info["dz"]])
    print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
          f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
          f"sorted=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})  "
          f"cz={info['cz']:>8.1f}")
