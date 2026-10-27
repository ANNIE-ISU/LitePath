"""Final comprehensive analysis of Panel 7 mounting components.

Searches for:
1. The "positive Y mirror" of holder_narrow for 10" PMT
2. "Two pieces near the front of the bulb" for 10" PMT
3. The "one more ring" for 8" PMT
"""
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

# ================================================================
# FINDING 1: The 10" PMT mounting system
# The 10" glass is at (-1222.7, 178.4, 1278.4)
# Components believed to be part of 10" system:
#   - bracket_70506 (70,506): found at (-1220.6, -63.3, 1184.2)
#   - perch_plate (42,671): found at (-1095.1, -59.0, 1159.3) and (-1095.1, -59.0, 1345.7)
#   - perch (41,872): found at (-1201.5, -73.6, 1173.9) and (-1201.5, -73.6, 1331.1)
#   - holder_narrow (73,755): found at (-1220.6, -116.3, 1252.5)
#
# These are all on the NEGATIVE Y side. Where is the POSITIVE Y side?
# ================================================================
print("\n" + "="*100)
print("FINDING 1: 10\" PMT system - looking for positive Y side mirror")
print("="*100)

# Check: Is there a bracket_70506 on the positive Y side of panel 7?
print("\n1a. bracket_70506 at Z=1184 on Panel 7:")
b1184_p7 = [s for s in all_info if 70400 <= s["vol"] <= 70600
            and -1260 <= s["cx"] <= -1190 and 1170 <= s["cz"] <= 1200]
for s in b1184_p7:
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")
if not b1184_p7:
    print("  NONE found on panel 7")

# Check ALL bracket_70506 on Panel 7 regardless of Z
print("\n1b. ALL bracket_70506 on Panel 7 (X in [-1260,-1190]):")
all_b70_p7 = [s for s in all_info if 70400 <= s["vol"] <= 70600 and -1260 <= s["cx"] <= -1190]
for s in sorted(all_b70_p7, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")

# Check perch on positive Y side of panel 7
print("\n1c. Perches on Panel 7 at Z=1174 and 1331:")
perches_p7_mid = [s for s in all_info if 41700 <= s["vol"] <= 42000
                  and -1260 <= s["cx"] <= -1190 and 1150 <= s["cz"] <= 1350]
for s in sorted(perches_p7_mid, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")

# Check perch_plate on positive Y side of panel 7
print("\n1d. Perch plates on Panel 7 at Z=1159 and 1346:")
pp7_mid = [s for s in all_info if 42500 <= s["vol"] <= 42800
           and -1260 <= s["cx"] <= -1190 and 1140 <= s["cz"] <= 1360]
for s in sorted(pp7_mid, key=lambda x: x["cz"]):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")

# Check holder_narrow on positive Y side of panel 7
print("\n1e. Holder_narrow at Z=1253 on Panel 7:")
hn_p7 = [s for s in all_info if 73600 <= s["vol"] <= 73900
         and -1260 <= s["cx"] <= -1190 and 1240 <= s["cz"] <= 1265]
for s in hn_p7:
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")

# BRING IT TOGETHER: nearest components to 10" glass on Panel 7
print("\n" + "="*100)
print("FINDING 1 SUMMARY: ALL 10\"-system components on Panel 7")
print("Ordered by distance from 10\" glass:")
print("="*100)

ten_system = [s for s in all_info
              if -1260 <= s["cx"] <= -1190
              and s["vol"] > 1000
              and np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10) < 500
              and (41700 <= s["vol"] <= 42000 or   # perch
                   42500 <= s["vol"] <= 42800 or   # perch_plate
                   70400 <= s["vol"] <= 70600 or   # bracket_70506
                   73600 <= s["vol"] <= 73900 or   # holder_narrow
                   429000 <= s["vol"] <= 430000 or # mounting_plate
                   82400 <= s["vol"] <= 82700)]    # bracket_plate

ten_system.sort(key=lambda x: np.linalg.norm(np.array([x["cx"],x["cy"],x["cz"]])-GLASS_10))
print(f"{'Idx':>5} {'Vol':>8} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D10':>7} {'Type'}")
print("-"*85)
for s in ten_system:
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    if 41700 <= s["vol"] <= 42000: t = "perch"
    elif 42500 <= s["vol"] <= 42800: t = "perch_plate"
    elif 70400 <= s["vol"] <= 70600: t = "bracket_70506"
    elif 73600 <= s["vol"] <= 73900: t = "holder_narrow"
    elif 429000 <= s["vol"] <= 430000: t = "mount_plate"
    elif 82400 <= s["vol"] <= 82700: t = "bracket_plate"
    else: t = "?"
    print(f"{s['idx']:>5} {s['vol']:>8.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d10:>7.1f}  {t}")

# ================================================================
# FINDING 2: Look for the "other side" of holder_narrow globally
# The holder_narrow at (-1220.6, -116.3, 1252.5) on Panel 7
# Its mirror would be at (x≈same, y≈+138, z≈same) relative to PMT
# ================================================================
print("\n" + "="*100)
print("FINDING 2: Looking for mirror of holder_narrow on Panel 7")
print("We have: (-1220.6, -116.3, 1252.5) — negative Y side")
print("Expected mirror at positive Y: roughly X≈-1220, Y≈120-250, Z≈1253")
print("Checking ALL solids in that region with vol > 5000")
print("="*100)

mirror_region = [s for s in all_info
                 if -1260 <= s["cx"] <= -1190
                 and 100 <= s["cy"] <= 300
                 and 1240 <= s["cz"] <= 1265
                 and s["vol"] > 5000]
if mirror_region:
    for s in mirror_region:
        d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
        print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
              f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")
else:
    print("  NOTHING found in mirror region! The positive Y side is empty at this Z-level.")

# Also check what's at Z=1278 but shifted in Y
print("\n2b. Check positive Y side at Z≈1278 (glass level):")
mirror_z1278 = [s for s in all_info
                if -1260 <= s["cx"] <= -1190
                and 100 <= s["cy"] <= 300
                and 1270 <= s["cz"] <= 1290
                and s["vol"] > 1000]
if mirror_z1278:
    for s in mirror_z1278:
        d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
        print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
              f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d10={d10:.1f}")
else:
    print("  NOTHING found.")

# Check ALL holder_narrow across the file to understand the full pattern
print("\n2c. ALL holder_narrow positions sorted by X, Y:")
hn_all = [s for s in all_info if 73600 <= s["vol"] <= 73900]
for s in sorted(hn_all, key=lambda x: (round(x["cz"]), x["cx"], x["cy"])):
    print(f"  cz={s['cz']:>7.1f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}")

# ================================================================
# FINDING 3: What are the "two pieces near the front of the bulb" for 10"?
# These would be at Z≈1278 (glass level), near the 10" glass
# ================================================================
print("\n" + "="*100)
print("FINDING 3: \"Two pieces near the front of the 10\" bulb\"")
print("Looking for components near the 10\" glass face (Z≈1278, X≈-1223)")
print("with bracket-like volumes (10k-200k) that we haven't classified yet")
print("="*100)

# ALL solids within 350mm of 10" glass, vol 1000-400k
near10 = [s for s in all_info 
          if np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10) < 350
          and 5000 <= s["vol"] <= 400000]
print(f"{'Idx':>5} {'Vol':>8} {'DX':>7} {'DY':>7} {'DZ':>7} {'CX':>9} {'CY':>9} {'CZ':>9} {'D10':>7}")
print("-"*75)
for s in sorted(near10, key=lambda x: np.linalg.norm(np.array([x["cx"],x["cy"],x["cz"]])-GLASS_10)):
    d10 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_10)
    print(f"{s['idx']:>5} {s['vol']:>8.0f} {s['dx']:>7.1f} {s['dy']:>7.1f} {s['dz']:>7.1f} "
          f"{s['cx']:>9.1f} {s['cy']:>9.1f} {s['cz']:>9.1f} {d10:>7.1f}")

# ================================================================
# FINDING 4: The "one more ring" for 8" PMT
# The existing ring (bracket_plate) is at (-1178.1, 178.4, 1277.9)
# Is there another ring-like component nearby?
# ================================================================
print("\n" + "="*100)
print("FINDING 4: \"One more ring\" for 8\" PMT")
print("Existing ring idx=6124 at (-1178.1, 178.4, 1277.9), vol=82,584")
print("Looking for another bracket_plate near the 8\" glass")
print("="*100)

# All bracket_plate near panel 7
bplate_near = [s for s in all_info if 82400 <= s["vol"] <= 82700
               and abs(s["cx"] - GLASS_8[0]) < 200
               and 1200 <= s["cz"] <= 1350]
for s in bplate_near:
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d8={d8:.1f}")
if not bplate_near:
    print("  NONE within 200mm of 8\" X and Z=1200-1350")

# Look further - all bracket_plate in the same X range as panel 7, all Z
print("\n4b. ALL bracket_plate with X≈-1178 (Panel 7 X-range):")
all_bp_p7 = [s for s in all_info if 82400 <= s["vol"] <= 82700 and -1250 <= s["cx"] <= -1150]
for s in sorted(all_bp_p7, key=lambda x: x["cz"]):
    d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
    print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d8={d8:.1f}")

# What about at Z=1709 (next Z band up)?
print("\n4c. Bracket_plate at Z≈1709 near Panel 7:")
for s in all_info:
    if 82400 <= s["vol"] <= 82700 and -1250 <= s["cx"] <= -1150 and 1690 <= s["cz"] <= 1720:
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
              f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d8={d8:.1f}")

# Check: could there be TWO bracket_plate on the same panel at Z=1278?
# One at (-1178.1, 178.4, 1277.9) and another...
print("\n4d. ALL bracket_plate at Z≈1278 (any panel):")
for s in all_info:
    if 82400 <= s["vol"] <= 82700 and 1270 <= s["cz"] <= 1290:
        d8 = np.linalg.norm(np.array([s["cx"],s["cy"],s["cz"]])-GLASS_8)
        print(f"  idx={s['idx']:>5} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
              f"dims={s['dx']:.1f}×{s['dy']:.1f}×{s['dz']:.1f}  d8={d8:.1f}")

print("\nDone.")
