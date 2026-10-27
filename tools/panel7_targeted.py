"""Targeted analysis for Panel 7 at the glass Z-level (Z≈1278) and all holder components."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
GLASS_8 = np.array([-1234.1, 177.9, 1278.3])
GLASS_10 = np.array([-1222.7, 178.4, 1278.4])

print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects
print(f"Total solids: {len(solids)}")

all_info = []
for i, s in enumerate(solids):
    try:
        vol = s.Volume()
        bb = s.BoundingBox()
        dx = bb.xmax - bb.xmin; dy = bb.ymax - bb.ymin; dz = bb.zmax - bb.zmin
        cx = (bb.xmin + bb.xmax)/2; cy = (bb.ymin + bb.ymax)/2; cz = (bb.zmin + bb.zmax)/2
        all_info.append({"idx": i, "vol": vol, "dx": dx, "dy": dy, "dz": dz,
                         "cx": cx, "cy": cy, "cz": cz})
    except: pass
print(f"Processed {len(all_info)} solids.")

# ================================================================
# ANALYSIS 1: What's at Z≈1278 (glass level) on Panel 7?
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 1: Z≈1276-1282 on Panel 7 (X=[-1260,-1190])")
print("="*100)
z1278 = [s for s in all_info if 1270 <= s["cz"] <= 1290
         and -1260 <= s["cx"] <= -1190 and s["vol"] > 100]
print(f"Solids at Z≈1278 on Panel 7: {len(z1278)}")
print(f"{'Idx':>5} {'Vol':>9} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D8':>7} {'D10':>7}")
print("-"*85)
for s in sorted(z1278, key=lambda x: np.linalg.norm(np.array([x["cx"],x["cy"],x["cz"]])-GLASS_8)):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"{s['idx']:>5} {s['vol']:>9.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d8:>7.1f} {d10:>7.1f}")

# ================================================================
# ANALYSIS 2: Look at the full X-range at Z≈1276-1282
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 2: ALL solids at Z≈1276-1282, vol>1000")
print("="*100)
all_z1278 = [s for s in all_info if 1270 <= s["cz"] <= 1290 and s["vol"] > 1000]
print(f"Total: {len(all_z1278)}")
print(f"{'Idx':>5} {'Vol':>9} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D8':>7} {'D10':>7}")
print("-"*85)
for s in sorted(all_z1278, key=lambda x: x["cx"]):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"{s['idx']:>5} {s['vol']:>9.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d8:>7.1f} {d10:>7.1f}")

# ================================================================
# ANALYSIS 3: Check ALL holder_narrow dims at Z=1253 specifically
# to see if any are on the positive Y side of panel 7
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 3: All holder_narrow (73,755) at Z=1253 band")
print("="*100)
hn1253 = [s for s in all_info if 73600 <= s["vol"] <= 73900 and 1240 <= s["cz"] <= 1265]
for s in sorted(hn1253, key=lambda x: (x["cx"], x["cy"])):
    print(f"  idx={s['idx']:>5} dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})")

# ================================================================
# ANALYSIS 4: Check ALL bracket_70506 at Z=1184 (near base of 10")
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 4: All bracket_70506 (70,506) at Z=1184 band")
print("="*100)
b1184 = [s for s in all_info if 70400 <= s["vol"] <= 70600 and 1170 <= s["cz"] <= 1200]
for s in sorted(b1184, key=lambda x: (x["cx"], x["cy"])):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d10={d10:.1f}")

# ================================================================
# ANALYSIS 5: Perch and perch_plate positions at Z=1331/1174/1346/1159
# to understand the 10" mounting geometry
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 5: Perches (41,872) on Panel 7 (X=[-1260,-1190])")
print("="*100)
perches_p7 = [s for s in all_info if 41700 <= s["vol"] <= 42000 and -1260 <= s["cx"] <= -1190]
for s in sorted(perches_p7, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>6.0f} dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d10={d10:.1f}")

print("\n" + "="*100)
print("ANALYSIS 5b: Perch_plates (42,671) on Panel 7")
print("="*100)
pp7 = [s for s in all_info if 42500 <= s["vol"] <= 42800 and -1260 <= s["cx"] <= -1190]
for s in sorted(pp7, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>6.0f} dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d10={d10:.1f}")

# ================================================================
# ANALYSIS 6: What is the relationship between mounting plate types?
# Show the two distinct shapes for mounting_plate (429,337)
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 6: Mounting plate (429,337) - shape variants")
print("="*100)
mp = [s for s in all_info if 429000 <= s["vol"] <= 430000]
shapes = defaultdict(list)
for s in mp:
    key = f"{s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}"
    shapes[key].append(s)
print(f"Total mounting plates: {len(mp)}")
for key, items in sorted(shapes.items()):
    z_levels = sorted(set(round(i["cz"]) for i in items))
    print(f"  {key}: {len(items)} instances, Z-levels: {z_levels}")

# Show all on Panel 7 or near it
print("\nMounting plates on/near Panel 7 (X < -1000):")
for s in mp:
    if s["cx"] < -1000:
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        print(f"  idx={s['idx']:>5} {s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f} "
              f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f}")

# ================================================================
# ANALYSIS 7: Check solids with vol in [10k, 70k] near the 8" and 10" glasses
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 7: Unidentified components near 8\" or 10\" glasses")
print("solids with vol 10k-70k within 400mm of either glass")
print("="*100)
candidates = []
for s in all_info:
    if s["vol"] < 10000 or s["vol"] > 70000:
        continue
    # Skip known buckets
    if (41700 <= s["vol"] <= 42000 or 42500 <= s["vol"] <= 42800 or
        70400 <= s["vol"] <= 70600 or 73600 <= s["vol"] <= 73900 or
        82400 <= s["vol"] <= 82700):
        continue
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    if d8 < 400 or d10 < 400:
        candidates.append((s, d8, d10))

candidates.sort(key=lambda x: min(x[1], x[2]))
print(f"{'Idx':>5} {'Vol':>9} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D8':>7} {'D10':>7}")
print("-"*85)
for s, d8, d10 in candidates:
    print(f"{s['idx']:>5} {s['vol']:>9.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d8:>7.1f} {d10:>7.1f}")

# ================================================================
# ANALYSIS 8: Check for ANY other ring/bracket-like component near 8"
# Search the entire file for anything plate-like near the 8" glass
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 8: All non-tiny solids near 8\" glass (d<200mm)")
print("="*100)
near8 = [s for s in all_info if s["vol"] > 1000 and
         np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8) < 200]
for s in sorted(near8, key=lambda x: np.linalg.norm(np.array([x["cx"],x["cy"],x["cz"]])-GLASS_8)):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>9.0f} dims={s['dx']:>7.1f}×{s['dy']:>7.1f}×{s['dz']:>7.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f}")

# ================================================================
# ANALYSIS 9: Full cross-section at Z≈1278 for ALL panels
# to understand the 8" vs 10" mounting plate distribution
# ================================================================
print("\n" + "="*100)
print("ANALYSIS 9: Which panels have which mounting plate orientations?")
print("="*100)

# Thin plate type: dims ≈ 6.4 × 343 × 235 (narrow bar along panel)
# Square plate type: dims ≈ 247 × 247 × 235 (corner mounted)

mp_thin = [s for s in mp if abs(s["dx"] - 6.4) < 1 or abs(s["dy"] - 6.4) < 1]
mp_square = [s for s in mp if abs(s["dx"] - 247) < 1 or abs(s["dy"] - 247) < 1]
print(f"Thin (bar) mounting plates: {len(mp_thin)}")
print(f"Square mounting plates: {len(mp_square)}")

print("\nThin plates at Z=1276:")
for s in sorted(mp_thin, key=lambda x: (x["cz"], x["cy"])):
    if abs(s["cz"] - 1276) < 5:
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        print(f"  idx={s['idx']:>5} {s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f} "
              f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f}")

print("\nSquare plates at Z=1276:")
for s in sorted(mp_square, key=lambda x: (x["cz"], x["cy"])):
    if abs(s["cz"] - 1276) < 5:
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        print(f"  idx={s['idx']:>5} {s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f} "
              f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f}")

print("\nDone.")
