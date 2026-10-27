"""Analyze STEP file solids to find PMT holder components."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")

print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solids_wp = result.solids()
solids = solids_wp.objects
print(f"Total solids found: {len(solids)}")

# Classify each solid by volume and bounding box
classified_counts = {}
unclassified = []

# Known categories (from user's classification)
categories = {
    "glass_bulb_3.7M": (3_600_000, 3_800_000),
    "glass_bulb_3.9M": (3_800_000, 4_000_000),
    "barrel_base_6.94M": (6_800_000, 7_100_000),
    "perch_41.9K": (40_000, 43_000),
    "bottom_housing_10.6M": (10_500_000, 10_800_000),
    "top_housing_11.7M": (11_500_000, 12_000_000),
}

# LAPPD volumes (multiple sizes, unknown exact range)
lapd_volumes = []

all_info = []

for i, solid in enumerate(solids):
    if i % 500 == 0:
        print(f"  Processing solid {i}/{len(solids)}...")
    try:
        vol = solid.Volume()
        bb = solid.BoundingBox()
        xmin, ymin, zmin = bb.xmin, bb.ymin, bb.zmin
        xmax, ymax, zmax = bb.xmax, bb.ymax, bb.zmax
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2
        all_info.append({
            "idx": i,
            "vol": vol,
            "dx": dx, "dy": dy, "dz": dz,
            "cx": cx, "cy": cy, "cz": cz,
        })
    except Exception as e:
        print(f"  Error on solid {i}: {e}")

print(f"\nProcessed {len(all_info)} solids successfully.\n")

# === STEP 1: Classify known categories ===
classified_indices = set()
classified_detail = defaultdict(list)

for info in all_info:
    vol = info["vol"]
    assigned = False
    for cat, (vmin, vmax) in categories.items():
        if vmin <= vol <= vmax:
            classified_detail[cat].append(info)
            classified_indices.add(info["idx"])
            assigned = True
            break
    if not assigned:
        unclassified.append(info)

print("=== CLASSIFIED SOLIDS ===")
for cat, items in sorted(classified_detail.items()):
    vols = [it["vol"] for it in items]
    print(f"  {cat}: {len(items)} solids, volume range {min(vols):.0f} - {max(vols):.0f}")

# Sum known classified
known_classified = sum(len(v) for v in classified_detail.values())
print(f"\nKnown categories total: {known_classified} solids")
print(f"Unclassified: {len(unclassified)} solids")
print(f"Total accounted: {known_classified + len(unclassified)}")

# === STEP 2: Group unclassified by rounded volume ===
print("\n\n=== UNCLASSIFIED: Group by volume (nearest 100) ===")
vol_buckets = defaultdict(list)
for info in unclassified:
    bucket = round(info["vol"] / 100) * 100
    vol_buckets[bucket].append(info)

# Sort by count descending
sorted_buckets = sorted(vol_buckets.items(), key=lambda x: len(x[1]), reverse=True)

print(f"{'Count':>6} {'VolumeBucket':>12} {'VolRange':>18} {'DX range':>16} {'DY range':>16} {'DZ range':>16} {'Z levels':>30}")
print("=" * 120)
for bucket, items in sorted_buckets:
    if len(items) < 2:
        continue  # skip singletons for now
    vols = [it["vol"] for it in items]
    dxs = [it["dx"] for it in items]
    dys = [it["dy"] for it in items]
    dzs = [it["dz"] for it in items]
    z_levels = set(round(it["cz"]) for it in items)
    z_str = ",".join(str(z) for z in sorted(z_levels))[:28]
    print(f"{len(items):>6} {bucket:>12} {min(vols):>8.0f}-{max(vols):>8.0f} "
          f"{min(dxs):>5.1f}-{max(dxs):>5.1f} {min(dys):>5.1f}-{max(dys):>5.1f} {min(dzs):>5.1f}-{max(dzs):>5.1f} "
          f"  [{z_str}]")

# === STEP 3: Look for barrel PMT holder candidates ===
print("\n\n=== CANDIDATE HOLDER ANALYSIS ===")
print("Looking for solids with counts matching barrel PMT distribution...")

barrel_z_levels = {820, 1252, 2115, 2979}

# Check multiples of 28, 56, 112, 184, 224
multiples_of_interest = {28, 56, 84, 92, 112, 184, 224}

candidates = []
for bucket, items in sorted_buckets:
    count = len(items)
    if count in multiples_of_interest or (count > 1 and any(count % m == 0 for m in [28, 56])):
        vols = [it["vol"] for it in items]
        dxs = [it["dx"] for it in items]
        dys = [it["dy"] for it in items]
        dzs = [it["dz"] for it in items]
        z_levels = set(round(it["cz"]) for it in items)
        # Check if Z-levels match barrel PMT Z-levels
        matching_z = z_levels & barrel_z_levels
        candidates.append({
            "bucket": bucket,
            "count": count,
            "vol_range": (min(vols), max(vols)),
            "dx_range": (min(dxs), max(dxs)),
            "dy_range": (min(dys), max(dys)),
            "dz_range": (min(dzs), max(dzs)),
            "z_levels": sorted(z_levels),
            "matching_barrel_z": sorted(matching_z),
        })

candidates.sort(key=lambda x: x["count"], reverse=True)
print(f"{'Count':>6} {'VolBucket':>12} {'VolRange':>18} {'DX':>14} {'DY':>14} {'DZ':>14} {'Z-levels':>30} {'MatchBarrelZ':>16}")
print("=" * 130)
for c in candidates:
    z_str = ",".join(str(z) for z in c["z_levels"])[:28]
    mz_str = ",".join(str(z) for z in c["matching_barrel_z"])
    print(f"{c['count']:>6} {c['bucket']:>12} {c['vol_range'][0]:>8.0f}-{c['vol_range'][1]:>8.0f} "
          f"{c['dx_range'][0]:>5.1f}-{c['dx_range'][1]:>5.1f} {c['dy_range'][0]:>5.1f}-{c['dy_range'][1]:>5.1f} {c['dz_range'][0]:>5.1f}-{c['dz_range'][1]:>5.1f} "
          f"  [{z_str}]  [{mz_str}]")

# === STEP 4: Specifically look for 50-200mm brackets at barrel Z-levels ===
print("\n\n=== HOLDER BRACKET SEARCH (50-200mm bounding box, barrel Z-levels) ===")
bracket_candidates = []
for info in unclassified:
    # Size filter: all dimensions between 50 and 200mm
    if not (50 <= info["dx"] <= 200 and 50 <= info["dy"] <= 200 and 50 <= info["dz"] <= 200):
        continue
    # Z-level filter: near barrel PMT Z-levels
    z_round = round(info["cz"])
    if z_round not in barrel_z_levels:
        continue
    bracket_candidates.append(info)

# Group bracket candidates by volume
bracket_buckets = defaultdict(list)
for bc in bracket_candidates:
    bucket = round(bc["vol"] / 100) * 100
    bracket_buckets[bucket].append(bc)

print(f"Found {len(bracket_candidates)} bracket candidates matching size and Z criteria")
print(f"\n{'Count':>6} {'VolBucket':>12} {'VolRange':>18} {'DX':>14} {'DY':>14} {'DZ':>14} {'Z-levels':>30}")
print("=" * 100)
for bucket in sorted(bracket_buckets.keys(), reverse=True):
    items = bracket_buckets[bucket]
    count = len(items)
    if count < 2:
        continue
    vols = [it["vol"] for it in items]
    dxs = [it["dx"] for it in items]
    dys = [it["dy"] for it in items]
    dzs = [it["dz"] for it in items]
    z_levels = set(round(it["cz"]) for it in items)
    z_str = ",".join(str(z) for z in sorted(z_levels))[:28]
    print(f"{count:>6} {bucket:>12} {min(vols):>8.0f}-{max(vols):>8.0f} "
          f"{min(dxs):>5.1f}-{max(dxs):>5.1f} {min(dys):>5.1f}-{max(dys):>5.1f} {min(dzs):>5.1f}-{max(dzs):>5.1f} "
          f"  [{z_str}]")

# === STEP 5: Volumes 10K-500K with counts 56, 92, 112, 184+ ===
print("\n\n=== HIGH-INSTANCE CANDIDATES (vol 10K-500K, count 56/92/112/184+) ===")
high_candidates = []
for bucket, items in sorted_buckets:
    count = len(items)
    if count < 56:
        continue
    vols = [it["vol"] for it in items]
    if not (10_000 <= min(vols) <= 500_000 and 10_000 <= max(vols) <= 500_000):
        continue
    dxs = [it["dx"] for it in items]
    dys = [it["dy"] for it in items]
    dzs = [it["dz"] for it in items]
    z_levels = set(round(it["cz"]) for it in items)
    high_candidates.append({
        "bucket": bucket,
        "count": count,
        "vol_range": (min(vols), max(vols)),
        "dx_range": (min(dxs), max(dxs)),
        "dy_range": (min(dys), max(dys)),
        "dz_range": (min(dzs), max(dzs)),
        "z_levels": sorted(z_levels),
    })

if high_candidates:
    print(f"{'Count':>6} {'VolBucket':>12} {'VolRange':>18} {'DX':>14} {'DY':>14} {'DZ':>14} {'Z-levels':>30}")
    print("=" * 100)
    for c in high_candidates:
        z_str = ",".join(str(z) for z in c["z_levels"])[:28]
        print(f"{c['count']:>6} {c['bucket']:>12} {c['vol_range'][0]:>8.0f}-{c['vol_range'][1]:>8.0f} "
              f"{c['dx_range'][0]:>5.1f}-{c['dx_range'][1]:>5.1f} {c['dy_range'][0]:>5.1f}-{c['dy_range'][1]:>5.1f} {c['dz_range'][0]:>5.1f}-{c['dz_range'][1]:>5.1f} "
              f"  [{z_str}]")

# === STEP 6: Check for 56-count groups specifically (barrel PMT holders likely) ===
print("\n\n=== 56-COUNT GROUPS (strong holder candidates) ===")
for c in candidates:
    if c["count"] == 56:
        print(f"\nVolume bucket {c['bucket']}:")
        print(f"  Count: {c['count']}")
        print(f"  Volume range: {c['vol_range'][0]:.0f} - {c['vol_range'][1]:.0f}")
        print(f"  DX range: {c['dx_range'][0]:.1f} - {c['dx_range'][1]:.1f}")
        print(f"  DY range: {c['dy_range'][0]:.1f} - {c['dy_range'][1]:.1f}")
        print(f"  DZ range: {c['dz_range'][0]:.1f} - {c['dz_range'][1]:.1f}")
        print(f"  Z levels: {c['z_levels']}")
        print(f"  Matching barrel Z: {c['matching_barrel_z']}")

# === STEP 7: Dumper for deeper inspection ===
print("\n\n=== ALL UNCLASSIFIED SOLIDS (detailed) ===")
print(f"{'Idx':>6} {'Volume':>12} {'DX':>8} {'DY':>8} {'DZ':>8} {'CX':>10} {'CY':>10} {'CZ':>10}")
print("=" * 75)
for info in unclassified:
    print(f"{info['idx']:>6} {info['vol']:>12.0f} {info['dx']:>8.1f} {info['dy']:>8.1f} {info['dz']:>8.1f} "
          f"{info['cx']:>10.1f} {info['cy']:>10.1f} {info['cz']:>10.1f}")

print("\nDone.")
