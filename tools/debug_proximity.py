"""Debug 10" assembly proximity."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")

print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solid_objs = result.solids().objects

all_info = []
for i, solid in enumerate(solid_objs):
    try:
        vol = solid.Volume()
        bb = solid.BoundingBox()
        dx = bb.xmax - bb.xmin; dy = bb.ymax - bb.ymin; dz = bb.zmax - bb.zmin
        cx = (bb.xmin + bb.xmax) / 2; cy = (bb.ymin + bb.ymax) / 2; cz = (bb.zmin + bb.zmax) / 2
        all_info.append({"idx": i, "vol": vol,
            "dx": dx, "dy": dy, "dz": dz, "cx": cx, "cy": cy, "cz": cz})
    except: pass

# Classify
def classify(info):
    v = info["vol"]; sd = sorted([info["dx"], info["dy"], info["dz"]])
    if 3_600_000 <= v <= 4_000_000:
        if 195 <= sd[0] <= 210 and 250 <= sd[1] <= 270 and 250 <= sd[2] <= 270:
            if 250 <= sd[1] <= 256: return "glass_8inch"
            elif 257 <= sd[1] <= 265: return "glass_10inch"
        elif 195 <= sd[0] <= 210 and 195 <= sd[1] <= 210 and 280 <= sd[2] <= 300:
            return "pmt_slab"
    if 6_900_000 <= v <= 7_000_000:
        if 250 <= sd[0] <= 260 and 250 <= sd[1] <= 260 and 335 <= sd[2] <= 345:
            return "base_8inch"
        elif 250 <= sd[0] <= 260 and 305 <= sd[1] <= 345 and 330 <= sd[2] <= 345:
            return "base_10inch"
    if 40_000 <= v <= 43_000: return "perch"
    return None

classified = defaultdict(list)
for info in all_info:
    cat = classify(info)
    if cat: classified[cat].append(info)

# Pick all 10" glasses at Z=1278 and show positions + nearby parts
print("\n=== 10\" glasses at Z=1278 ===")
for g in classified["glass_10inch"]:
    if abs(g["cz"] - 1278) < 10:
        print(f"\nglass_10inch idx={g['idx']}: pos=({g['cx']:.1f},{g['cy']:.1f},{g['cz']:.1f}) dims={g['dx']:.1f}x{g['dy']:.1f}x{g['dz']:.1f}")
        gpos = np.array([g["cx"], g["cy"], g["cz"]])

        # Closest base_10inch
        best_b = min(classified["base_10inch"], key=lambda b: np.linalg.norm(np.array([b["cx"],b["cy"],b["cz"]])-gpos))
        bd = np.linalg.norm(np.array([best_b["cx"],best_b["cy"],best_b["cz"]])-gpos)
        print(f"  Closest base_10: idx={best_b['idx']} d={bd:.1f} pos=({best_b['cx']:.1f},{best_b['cy']:.1f},{best_b['cz']:.1f})")

        # Closest perch
        best_p = min(classified["perch"], key=lambda p: np.linalg.norm(np.array([p["cx"],p["cy"],p["cz"]])-gpos))
        pd = np.linalg.norm(np.array([best_p["cx"],best_p["cy"],best_p["cz"]])-gpos)
        print(f"  Closest perch:   idx={best_p['idx']} d={pd:.1f} pos=({best_p['cx']:.1f},{best_p['cy']:.1f},{best_p['cz']:.1f})")

# Also show what Z-levels 10" bases and perches are at that match 1278
print("\n=== base_10inch at Z=1253 (near glass Z=1278) ===")
for b in classified["base_10inch"]:
    if abs(b["cz"] - 1253) < 10:
        print(f"  idx={b['idx']}: pos=({b['cx']:.1f},{b['cy']:.1f},{b['cz']:.1f})")

print("\n=== Perches at Z levels near 1278 (1150-1350) ===")
for p in classified["perch"]:
    if 1150 < p["cz"] < 1350:
        print(f"  idx={p['idx']}: pos=({p['cx']:.1f},{p['cy']:.1f},{p['cz']:.1f})")

# Now same for 8" - pick a glass at Z=1278, find base at Z=1253
print("\n=== 8\" glasses at Z=1278 (should pair with base at Z=1253) ===")
for g in classified["glass_8inch"]:
    if abs(g["cz"] - 1278) < 10:
        print(f"\nglass_8inch idx={g['idx']}: pos=({g['cx']:.1f},{g['cy']:.1f},{g['cz']:.1f})")
        gpos = np.array([g["cx"], g["cy"], g["cz"]])
        best_b = min(classified["base_8inch"], key=lambda b: np.linalg.norm(np.array([b["cx"],b["cy"],b["cz"]])-gpos))
        bd = np.linalg.norm(np.array([best_b["cx"],best_b["cy"],best_b["cz"]])-gpos)
        print(f"  Closest base_8:  idx={best_b['idx']} d={bd:.1f} pos=({best_b['cx']:.1f},{best_b['cy']:.1f},{best_b['cz']:.1f})")

# And the 8" base at Z=1253
print("\n=== base_8inch at Z=1253 ===")
for b in classified["base_8inch"]:
    if abs(b["cz"] - 1253) < 10:
        print(f"  idx={b['idx']}: pos=({b['cx']:.1f},{b['cy']:.1f},{b['cz']:.1f})")

# Check: are there perch-like objects at the glass Z-levels?
print("\n=== Are perches actually associated with 10\" glasses? ===")
# Look at perches in the Z range 1200-1800
perches_mid = [p for p in classified["perch"] if 1150 < p["cz"] < 1800]
print(f"Perches in 1150-1800 Z range: {len(perches_mid)}")
# Their Z distribution
zcount = defaultdict(int)
for p in perches_mid: zcount[round(p["cz"])] += 1
for z in sorted(zcount): print(f"  Z={z}: {zcount[z]}")

print("\nDone.")
