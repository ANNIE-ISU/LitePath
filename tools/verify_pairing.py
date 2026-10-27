"""Verify pairing: pmt_slab with base_8, glass_bulb with base_10."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path
import math

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects

all_info = []
for i, s in enumerate(solids):
    try:
        vol = s.Volume()
        bb = s.BoundingBox()
        dx=bb.xmax-bb.xmin; dy=bb.ymax-bb.ymin; dz=bb.zmax-bb.zmin
        cx=(bb.xmin+bb.xmax)/2; cy=(bb.ymin+bb.ymax)/2; cz=(bb.zmin+bb.zmax)/2
        all_info.append({"idx":i,"vol":vol,"dx":dx,"dy":dy,"dz":dz,"cx":cx,"cy":cy,"cz":cz})
    except: pass

def classify(info):
    v=info["vol"]; sd=sorted([info["dx"],info["dy"],info["dz"]])
    if 3_600_000<=v<=4_000_000:
        if 195<=sd[0]<=210 and 250<=sd[1]<=270 and 250<=sd[2]<=270:
            return "glass_bulb"  # 252×252 or 260×260 face
        elif 195<=sd[0]<=210 and 195<=sd[1]<=210 and 280<=sd[2]<=300:
            return "pmt_slab"    # 202×202×290 slab
    if 6_900_000<=v<=7_000_000:
        if 250<=sd[0]<=260 and 250<=sd[1]<=260 and 335<=sd[2]<=345: return "base_8inch"
        elif 250<=sd[0]<=260 and 305<=sd[1]<=345 and 330<=sd[2]<=345: return "base_10inch"
    if 40_000<=v<=43_000: return "perch_or_plate"
    return None

def angle(x,y): return (math.degrees(math.atan2(y,x)) % 360)

classified = defaultdict(list)
for info in all_info:
    cat = classify(info)
    if cat: classified[cat].append(info)

# === Test pairing: pmt_slab with base_8inch ===
print("=== Pairing hypothesis ===")
print("8\" system: pmt_slab(202×202×290) + base_8inch(254×254×340)")
print("10\" system: glass_bulb(252×252 + 260×260, nested) + base_10inch(254×311×338)")
print()

for label, glass_list, base_list, base_name in [
    ("8\" PMT: pmt_slab + base_8", classified["pmt_slab"], classified["base_8inch"], "base_8inch"),
    ("10\" PMT: glass_bulb + base_10", classified["glass_bulb"], classified["base_10inch"], "base_10inch"),
]:
    print(f"\n=== {label} ===")
    pairs_found = 0
    for g in glass_list[:5]:
        ga = angle(g["cx"], g["cy"])
        # Find base with closest angle
        best_b = min(base_list, key=lambda b: abs(angle(b["cx"],b["cy"])-ga))
        ba = angle(best_b["cx"], best_b["cy"])
        d = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2 + (g["cz"]-best_b["cz"])**2)
        d_xy = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2)
        dz = abs(g["cz"]-best_b["cz"])
        print(f"  glass idx={g['idx']:>5}: pos=({g['cx']:>7.1f},{g['cy']:>7.1f},{g['cz']:>7.1f}) dims={g['dx']:.1f}x{g['dy']:.1f}x{g['dz']:.1f}  angle={ga:.1f}°")
        print(f"  {base_name} idx={best_b['idx']:>5}: pos=({best_b['cx']:>7.1f},{best_b['cy']:>7.1f},{best_b['cz']:>7.1f}) dims={best_b['dx']:.1f}x{best_b['dy']:.1f}x{best_b['dz']:.1f}  angle={ba:.1f}°")
        print(f"  → d_xy={d_xy:.1f}mm  ΔZ={dz:.0f}mm  d_3d={d:.1f}mm")
        # Δ angle
        da = min(abs(ga-ba), 360-abs(ga-ba))
        print(f"  → Δangle={da:.1f}°")
        if d < 200 and da < 5:
            pairs_found += 1
        print()
    print(f"  Well-paired (d<200mm, Δangle<5°): {pairs_found}/{min(5,len(glass_list))}")

# === For the 8" system: also show all pmt_slab positions and their matched bases ===
print("\n\n=== Full 8\" system pairing: all pmt_slabs at Z=1278 with base_8 at Z=1253 ===")
for g in classified["pmt_slab"]:
    if abs(g["cz"]-1278) > 10: continue
    ga = angle(g["cx"], g["cy"])
    best_b = min(classified["base_8inch"], key=lambda b: abs(angle(b["cx"],b["cy"])-ga))
    d = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2 + (g["cz"]-best_b["cz"])**2)
    ba = angle(best_b["cx"], best_b["cy"])
    da = min(abs(ga-ba), 360-abs(ga-ba))
    good = "✓" if d<150 and da<5 else "✗"
    print(f"  {good} slab idx={g['idx']:>5}: ({g['cx']:>7.1f},{g['cy']:>7.1f},{g['cz']:>7.1f}) vol={g['vol']:.0f}  "
          f"→ base_8 idx={best_b['idx']:>5}: ({best_b['cx']:>7.1f},{best_b['cy']:>7.1f},{best_b['cz']:>7.1f})  "
          f"d={d:>5.1f}mm  Δa={da:.1f}°")

# === For the 10" system: all glass_bulbs with base_10 ===
print("\n\n=== Full 10\" system pairing: all glass_bulbs at Z=1278 with base_10 at Z=1253 ===")
for g in classified["glass_bulb"]:
    if abs(g["cz"]-1278) > 10: continue
    ga = angle(g["cx"], g["cy"])
    best_b = min(classified["base_10inch"], key=lambda b: abs(angle(b["cx"],b["cy"])-ga))
    d = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2 + (g["cz"]-best_b["cz"])**2)
    ba = angle(best_b["cx"], best_b["cy"])
    da = min(abs(ga-ba), 360-abs(ga-ba))
    good = "✓" if d<150 and da<5 else "✗"
    print(f"  {good} bulb idx={g['idx']:>5}: ({g['cx']:>7.1f},{g['cy']:>7.1f},{g['cz']:>7.1f}) vol={g['vol']:.0f}  "
          f"→ base_10 idx={best_b['idx']:>5}: ({best_b['cx']:>7.1f},{best_b['cy']:>7.1f},{best_b['cz']:>7.1f})  "
          f"d={d:>5.1f}mm  Δa={da:.1f}°")

print("\nDone.")
