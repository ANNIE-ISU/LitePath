"""Comprehensive analysis of Panel 7 barrel PMT mounting components.

Usage:
    python3 tools/panel7_comprehensive.py
"""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
GLASS_8_CENTER = np.array([-1234.1, 177.9, 1278.3])
BARREL_Z_BANDS = [821, 1253, 2116, 2979]

print("=" * 120)
print("LOADING STEP FILE...")
print("=" * 120)
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects
print(f"Total solids: {len(solids)}")

print("\nExtracting solid info...")
all_info = []
for i, solid in enumerate(solids):
    if i % 1000 == 0 and i > 0:
        print(f"  Processing solid {i}/{len(solids)}...")
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
            "idx": i, "vol": vol,
            "dx": dx, "dy": dy, "dz": dz,
            "cx": cx, "cy": cy, "cz": cz,
        })
    except Exception as e:
        print(f"  Error solid {i}: {e}")

print(f"\nProcessed {len(all_info)} solids (vol>0).")

# ================================================================
# STEP 1: Full panel scan — all solids at Panel 7
# ================================================================
print("\n" + "=" * 120)
print("STEP 1: FULL PANEL 7 SCAN")
print("X in [-1260, -1190], Z in [700, 3500], vol > 100")
print("=" * 120)

panel_solids = [s for s in all_info
                if s["vol"] > 100
                and -1260 <= s["cx"] <= -1190
                and 700 <= s["cz"] <= 3500]

print(f"\nTotal Panel 7 solids (vol>100): {len(panel_solids)}")

# Group by Z-band
print(f"\n{'Z Band':>12} {'Count':>8}")
print("-" * 25)
for z_band in BARREL_Z_BANDS:
    count = sum(1 for s in panel_solids if abs(s["cz"] - z_band) < 20)
    print(f"{z_band:>8}-{z_band+1:<4} {count:>8}")

# List all solids at each Z-band with details
for z_band in BARREL_Z_BANDS:
    band_solids = [s for s in panel_solids if abs(s["cz"] - z_band) < 20]
    if not band_solids:
        continue
    print(f"\n{'─' * 120}")
    print(f"Z ≈ {z_band} ({len(band_solids)} solids)")
    print(f"{'Idx':>6} {'Volume':>12} {'DX':>8} {'DY':>8} {'DZ':>8} "
          f"{'CX':>10} {'CY':>10} {'CZ':>10} {'Dist8\"':>8}")
    print("-" * 120)
    for s in sorted(band_solids, key=lambda x: abs(x["cz"] - z_band)):
        d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
        print(f"{s['idx']:>6} {s['vol']:>12.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
              f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f} {d:>8.1f}")

# ================================================================
# STEP 2: Search for specific component types across entire STEP
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 2: SEARCH FOR SPECIFIC VOLUME BUCKETS (ENTIRE STEP FILE)")
print("=" * 120)

volume_buckets = {
    "glass_8inch (3,738,632)": (3_738_000, 3_739_500),
    "glass_10inch (3,913,227)": (3_912_000, 3_914_500),
    "mounting_plate (429,337)": (429_000, 430_000),
    "bracket_plate (82,584)": (82_400, 82_700),
    "bracket_70506 (70,506)": (70_400, 70_600),
    "holder_narrow (73,755)": (73_600, 73_900),
    "perch (41,872)": (41_700, 42_000),
    "perch_plate (42,671)": (42_500, 42_800),
    "structural_bar_381 (390,598)": (390_000, 391_200),
    "structural_bar_long (624,956)": (624_000, 626_000),
    "base_8inch (6,942,848)": (6_900_000, 7_000_000),
}

for label, (vmin, vmax) in volume_buckets.items():
    matches = [s for s in all_info if vmin <= s["vol"] <= vmax]
    if not matches:
        continue
    
    # Group by Z-level
    z_groups = defaultdict(list)
    for s in matches:
        zr = round(s["cz"])
        z_groups[zr].append(s)
    
    print(f"\n{'─' * 120}")
    print(f"{label}: {len(matches)} instances total")
    print(f"Volume range: {min(s['vol'] for s in matches):.0f} - {max(s['vol'] for s in matches):.0f}")
    
    # Show distribution by Z-level
    print(f"{'Z-level':>8} {'Count':>6} {'Sample positions (cx, cy, cz)':>60}")
    for z in sorted(z_groups.keys()):
        grp = z_groups[z]
        # Show up to 3 sample positions
        samples = grp[:3]
        pos_strs = [f"({s['cx']:.1f},{s['cy']:.1f},{s['cz']:.1f})" for s in samples]
        extra = f"... +{len(grp)-3} more" if len(grp) > 3 else ""
        print(f"{z:>8} {len(grp):>6} {', '.join(pos_strs)}{extra}")

    # Show full detail for small sets (<= 10 instances)
    if len(matches) <= 30:
        print(f"  Full list:")
        print(f"  {'Idx':>6} {'Vol':>10} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10}")
        for s in sorted(matches, key=lambda x: (x["cz"], x["cx"])):
            print(f"  {s['idx']:>6} {s['vol']:>10.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
                  f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f}")

# ================================================================
# STEP 3: Find unclassified components in vol range [10k, 70k]
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 3: UNCLASSIFIED COMPONENTS (vol 10k-70k) not in known buckets")
print("=" * 120)

known_vol_ranges = [
    (41_700, 42_000),    # perch
    (42_500, 42_800),    # perch_plate
    (70_400, 70_600),    # bracket_70506
    (73_600, 73_900),    # holder_narrow
    (82_400, 82_700),    # bracket_plate
]

def is_known_vol(vol):
    for vmin, vmax in known_vol_ranges:
        if vmin <= vol <= vmax:
            return True
    return False

unclassified_10k_70k = [s for s in all_info
                        if 10_000 <= s["vol"] <= 70_000
                        and not is_known_vol(s["vol"])]

# Group by rounded volume
vol_groups = defaultdict(list)
for s in unclassified_10k_70k:
    bucket = round(s["vol"] / 500) * 500
    vol_groups[bucket].append(s)

print(f"\nTotal unclassified solids in [10k, 70k]: {len(unclassified_10k_70k)}")
print(f"\n{'Count':>6} {'VolBuck':>8} {'VolRange':>16} {'Z-levels':>30}")
print("-" * 65)
sorted_buckets = sorted(vol_groups.items(), key=lambda x: len(x[1]), reverse=True)
for bucket, items in sorted_buckets:
    z_levels = sorted(set(round(s["cz"]) for s in items))
    z_str = ",".join(str(z) for z in z_levels)[:28]
    vols = [s["vol"] for s in items]
    print(f"{len(items):>6} {bucket:>8} {min(vols):>8.0f}-{max(vols):>8.0f}  [{z_str}]")

# Show details for items at barrel Z-levels
print(f"\n\n--- Unclassified in [10k,70k] at BARREL Z-levels ---")
for z_band in BARREL_Z_BANDS:
    at_z = [s for s in unclassified_10k_70k if abs(s["cz"] - z_band) < 20]
    if not at_z:
        continue
    print(f"\nZ ≈ {z_band} ({len(at_z)} solids):")
    print(f"{'Idx':>6} {'Vol':>10} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10} {'Dist8\"':>8}")
    print("-" * 85)
    for s in sorted(at_z, key=lambda x: x["vol"]):
        d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
        print(f"{s['idx']:>6} {s['vol']:>10.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
              f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f} {d:>8.1f}")

# ================================================================
# STEP 4: Find holder_narrow (73,755) - the 10" holder side -
# look for its mirror on positive Y side of panel 7
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 4: 10\" HOLDER NARROW (73,755) - Looking for positive Y mirror")
print("=" * 120)

holder_narrow = [s for s in all_info if 73_600 <= s["vol"] <= 73_900]
print(f"Total holder_narrow instances: {len(holder_narrow)}")

# The one we already found is at (-1220.6, -116.3, 1252.5)
# Look for one with similar X, positive Y at same Z
for s in holder_narrow:
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}")

# ================================================================
# STEP 5: Check specifically at Z=1253 band for bracket_70506
# and perch_plate types near panel 7
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 5: ALL COMPONENTS at Z≈1253 within X=[-1260,-1190]")
print("=" * 120)

z1253_solids = [s for s in panel_solids if abs(s["cz"] - 1253) < 20]
print(f"Total at Z≈1253: {len(z1253_solids)}")
print(f"{'Idx':>6} {'Vol':>10} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10} {'Dist8\"':>8}")
print("-" * 85)
for s in sorted(z1253_solids, key=lambda x: x["vol"]):
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"{s['idx']:>6} {s['vol']:>10.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
          f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f} {d:>8.1f}")

# ================================================================
# STEP 6: Also check if there's a 2nd ring for 8" (bracket_plate type)
# near the 8" PMT by looking at all 82,584 components
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 6: BRACKET_PLATE (82,584) - Looking for 2nd ring for 8\"")
print("=" * 120)

bracket_plate = [s for s in all_info if 82_400 <= s["vol"] <= 82_700]
print(f"Total bracket_plate instances: {len(bracket_plate)}")
for s in bracket_plate:
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  dist8={d:.1f}")

# ================================================================
# STEP 7: Mounting plate (429,337) - check all instances
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 7: MOUNTING_PLATE (429,337) - All instances")
print("=" * 120)

mounting = [s for s in all_info if 429_000 <= s["vol"] <= 430_000]
print(f"Total mounting_plate instances: {len(mounting)}")
for s in mounting:
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  dist8={d:.1f}")

# ================================================================
# STEP 8: Find bracket_70506 (70,506) instances at Z≈1253
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 8: BRACKET_70506 (70,506) - All instances at Z≈1253")
print("=" * 120)

bracket70 = [s for s in all_info if 70_400 <= s["vol"] <= 70_600]
print(f"Total bracket_70506 instances: {len(bracket70)}")
for s in bracket70:
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  dist8={d:.1f}")

# ================================================================
# STEP 9: Perch (41,872) & Perch Plate (42,671) at Z≈1253
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 9: PERCH & PERCH_PLATE at Z≈1253")
print("=" * 120)

perches = [s for s in all_info if 41_700 <= s["vol"] <= 42_000 and abs(s["cz"] - 1253) < 20]
perch_plates = [s for s in all_info if 42_500 <= s["vol"] <= 42_800 and abs(s["cz"] - 1253) < 20]

print(f"Perches at Z≈1253: {len(perches)}")
for s in perches:
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  dist8={d:.1f}")

print(f"\nPerch plates at Z≈1253: {len(perch_plates)}")
for s in perch_plates:
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"  idx={s['idx']:>5} vol={s['vol']:>8.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  dist8={d:.1f}")

# ================================================================
# STEP 10: What's in [110k, 120k] range - look for anything not LAPPD
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 10: SOLIDS in [100k, 200k] vol range at barrel Z-levels")
print("=" * 120)

mid_solids = [s for s in all_info if 100_000 <= s["vol"] <= 200_000]
# Check if any at barrel Z-levels
at_barrel_z = [s for s in mid_solids if any(abs(s["cz"] - z) < 20 for z in BARREL_Z_BANDS)]
print(f"Total in [100k,200k]: {len(mid_solids)}")
print(f"At barrel Z-levels: {len(at_barrel_z)}")
for s in at_barrel_z:
    z_band = min(BARREL_Z_BANDS, key=lambda z: abs(s["cz"] - z))
    print(f"  idx={s['idx']:>5} vol={s['vol']:>10.0f} pos=({s['cx']:>8.1f},{s['cy']:>8.1f},{s['cz']:>8.1f}) "
          f"dims={s['dx']:>6.1f}×{s['dy']:>6.1f}×{s['dz']:>6.1f}  near_Z≈{z_band}")

# ================================================================
# STEP 11: Look for any other bracket-like components that could
# be the "2 pieces near the front of the bulb" for 10"
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 11: Searching for 10\" front-of-bulb mounting components")
print("Looking near the 10\" glass at (-1222.7, 178.4, 1278.4)")
print("=" * 120)

GLASS_10_CENTER = np.array([-1222.7, 178.4, 1278.4])

# Look at ALL solids within 350mm of 10" glass with vol > 1000
near_10 = [s for s in all_info if s["vol"] > 1000
           and np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_10_CENTER) < 350]
print(f"All solids within 350mm of 10\" glass: {len(near_10)}")
print(f"{'Idx':>6} {'Vol':>10} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10} {'Dist10\"':>8}")
print("-" * 85)
for s in sorted(near_10, key=lambda x: np.linalg.norm(np.array([x["cx"], x["cy"], x["cz"]]) - GLASS_10_CENTER)):
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_10_CENTER)
    print(f"{s['idx']:>6} {s['vol']:>10.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
          f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f} {d:>8.1f}")

# ================================================================
# STEP 12: Look for 2nd ring for 8" - check ALL solids near 8" glass
# with bracket-like dimensions
# ================================================================
print("\n\n" + "=" * 120)
print("STEP 12: Searching for 8\" 2nd ring component")
print("Looking near the 8\" glass at (-1234.1, 177.9, 1278.3)")
print("=" * 120)

# All solids within 350mm of 8" glass, vol > 1000
near_8 = [s for s in all_info if s["vol"] > 1000
          and np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER) < 350]
print(f"All solids within 350mm of 8\" glass: {len(near_8)}")
print(f"{'Idx':>6} {'Vol':>10} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10} {'Dist8\"':>8}")
print("-" * 85)
for s in sorted(near_8, key=lambda x: np.linalg.norm(np.array([x["cx"], x["cy"], x["cz"]]) - GLASS_8_CENTER)):
    d = np.linalg.norm(np.array([s["cx"], s["cy"], s["cz"]]) - GLASS_8_CENTER)
    print(f"{s['idx']:>6} {s['vol']:>10.0f} {s['dx']:>8.1f} {s['dy']:>8.1f} {s['dz']:>8.1f} "
          f"{s['cx']:>10.1f} {s['cy']:>10.1f} {s['cz']:>10.1f} {d:>8.1f}")

print("\nDone.")
