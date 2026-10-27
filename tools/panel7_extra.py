"""Extra checks for ring-like and bracket-like components near Panel 7 PMTs."""
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

# ================================================================
# EXTRA 1: Look at ALL base instances to find 10" base on Panel 7
# ================================================================
print("="*100)
print("EXTRA 1: Finding the 10\" PMT base on Panel 7")
print("="*100)

# All bases (vol 6.9M)
bases = [s for s in all_info if 6_900_000 <= s["vol"] <= 7_000_000]
print(f"Total bases: {len(bases)}")
for s in sorted(bases, key=lambda x: (x["cz"], x["cx"])):
    sd = sorted([s["dx"], s["dy"], s["dz"]])
    print(f"  idx={s['idx']:>5} sdims=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f}) "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})")

# Is the base at (-1122.8, -152.2, 1252.5) the 8" or 10" base?
print("\nBases on/near Panel 7 (X < -1000):")
for s in bases:
    if s["cx"] < -1000:
        sd = sorted([s["dx"], s["dy"], s["dz"]])
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
        print(f"  idx={s['idx']:>5} sdims=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f}) "
              f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f} d10={d10:.1f}")

# ================================================================
# EXTRA 2: Check if holder_narrow types might represent the
# "two pieces near the front of the bulb"
# Check all holder_narrow with the NARROW dims (25.4×67.8×165.1)
# at Z=821 (a different band) to see the full pattern
# ================================================================
print("\n"+"="*100)
print("EXTRA 2: Holder_narrow SIDE-mounted type (25.4×67.8×165.1) pattern")
print("="*100)

# Count the 4 positions at each Z-level on panel 7
hn_side = [s for s in all_info if 73600 <= s["vol"] <= 73900 
           and abs(s["dx"] - 25.4) < 1]  # narrow type (not square)

# Group by Z-level and count
z_hn = defaultdict(lambda: defaultdict(list))
for s in hn_side:
    zr = round(s["cz"])
    z_hn[zr][s["cx"]].append(s)

for z in sorted(z_hn.keys()):
    cx_counts = {cx: len(items) for cx, items in z_hn[z].items()}
    print(f"  Z={z}: {dict(sorted(cx_counts.items()))}")

# Show the actual positions at each Z for Panel 7
print("\nPanel 7 (X=-1220.6) side-mounted holder_narrow positions:")
for z in sorted(z_hn.keys()):
    items = z_hn[z].get(-1220.6, [])
    if items:
        ys = sorted([s["cy"] for s in items])
        print(f"  Z={z}: Y={ys}")

# ================================================================
# EXTRA 3: What about vol≈390,598 (structural_bar_381)?
# Could any of them be acting as a 2nd ring?
# ================================================================
print("\n"+"="*100)
print("EXTRA 3: Structural bars near the PMTs")
print("="*100)

sbars_near = [s for s in all_info if 390_000 <= s["vol"] <= 391_000
              and abs(s["cx"] - (-1220)) < 150
              and 1200 <= s["cz"] <= 1350]
for s in sorted(sbars_near, key=lambda x: x["cz"]):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d8={d8:.1f} d10={d10:.1f}")

# ================================================================
# EXTRA 4: Look for ANY thin plate-like solid near the 8" glass
# that might be a 2nd ring. Search by shape rather than volume.
# A ring-like component would be thin (one dim << others) and 
# roughly square/round with a hole.
# ================================================================
print("\n"+"="*100)
print("EXTRA 4: Thin plate-like solids near 8\" glass (d<200mm)")
print("Looking for ANYTHING with one dimension < 20mm")
print("="*100)

thins_near = [s for s in all_info
              if np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8) < 200
              and s["vol"] > 500
              and (s["dx"] < 20 or s["dy"] < 20 or s["dz"] < 20)]
for s in sorted(thins_near, key=lambda x: np.linalg.norm(np.array([x["cx"],x["cy"],x["cz"]])-GLASS_8)):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} dims={s['dx']:>7.1f}×{s['dy']:>7.1f}×{s['dz']:>7.1f} "
          f"pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f})  d8={d8:.1f}")

# ================================================================
# EXTRA 5: Show the complete 10" mounting point arrangement
# by finding ALL components on Panel 7 from Z=1150 to Z=1350
# that are likely PMT mounting hardware
# ================================================================
print("\n"+"="*100)
print("EXTRA 5: Complete 10\" PMT mounting assembly on Panel 7")
print("All components Z=[1150,1350] on Panel 7, vol>10000")
print("="*100)

p7_mount = [s for s in all_info
            if -1260 <= s["cx"] <= -1190
            and 1150 <= s["cz"] <= 1350
            and s["vol"] > 10000]
p7_mount.sort(key=lambda x: (x["cz"], abs(x["cy"])))

def classify_vol(v):
    if 41700 <= v <= 42000: return "perch"
    if 42500 <= v <= 42800: return "perch_plate"
    if 70400 <= v <= 70600: return "bracket_70506"
    if 73600 <= v <= 73900: return "holder_narrow"
    if 429000 <= v <= 430000: return "mount_plate"
    if 82400 <= v <= 82700: return "bracket_plate"
    if 390000 <= v <= 391000: return "struct_bar_381"
    return f"vol={v:.0f}"

print(f"{'Idx':>5} {'Type':>15} {'Vol':>8} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D10':>7}")
print("-"*95)
for s in p7_mount:
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    t = classify_vol(s["vol"])
    print(f"{s['idx']:>5} {t:>15} {s['vol']:>8.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d10:>7.1f}")

# ================================================================
# EXTRA 6: Check the perch_plate (42,671) distribution on Panel 7
# ALL Z-levels
# ================================================================
print("\n"+"="*100)
print("EXTRA 6: ALL perch_plate on Panel 7 (all Z-levels)")
print("="*100)
pp7_all = [s for s in all_info if 42500 <= s["vol"] <= 42800 and -1260 <= s["cx"] <= -1190]
for s in sorted(pp7_all, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")

print("\nDone.")
