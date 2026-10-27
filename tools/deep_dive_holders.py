"""Deep-dive analysis into candidate PMT holder solids."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")

print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects
print(f"Total solids: {len(solids)}")

# Extract all solid info
all_info = []
for i, solid in enumerate(solids):
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
            "vol": vol,
            "dx": dx, "dy": dy, "dz": dz,
            "cx": cx, "cy": cy, "cz": cz,
        })
    except:
        pass

# ============================================================
# Focus on the 73,800 (73755) bucket - top holder candidate
# ============================================================
print("\n" + "="*80)
print("DEEP DIVE: Volume bucket ~73,800 (73755)")
print("="*80)

bucket_73800 = [info for info in all_info if 73000 <= info["vol"] <= 75000]
print(f"Count: {len(bucket_73800)}")
print(f"Volume range: {min(i['vol'] for i in bucket_73800):.0f} - {max(i['vol'] for i in bucket_73800):.0f}")
print(f"DX range: {min(i['dx'] for i in bucket_73800):.1f} - {max(i['dx'] for i in bucket_73800):.1f}")
print(f"DY range: {min(i['dy'] for i in bucket_73800):.1f} - {max(i['dy'] for i in bucket_73800):.1f}")
print(f"DZ range: {min(i['dz'] for i in bucket_73800):.1f} - {max(i['dz'] for i in bucket_73800):.1f}")

# Count per Z-level (rounded to nearest 1)
z_counts = defaultdict(int)
for info in bucket_73800:
    z_counts[round(info["cz"])] += 1

print("\nDistribution by Z-level:")
for z in sorted(z_counts.keys()):
    print(f"  Z={z:6.0f}: {z_counts[z]} instances")

# Unique shapes (by dx,dy,dz rounded to 0.1)
shape_counts = defaultdict(list)
for info in bucket_73800:
    key = (round(info["dx"], 1), round(info["dy"], 1), round(info["dz"], 1))
    shape_counts[key].append(info)

print(f"\nUnique shape variants in 73,800 bucket: {len(shape_counts)}")
for key, items in sorted(shape_counts.items()):
    dx, dy, dz = key
    z_levels = set(round(i["cz"]) for i in items)
    print(f"  {dx:6.1f} × {dy:6.1f} × {dz:6.1f} : {len(items):>4} instances at Z-levels {sorted(z_levels)}")

# ============================================================
# Also check 112,300 and 112,100 buckets
# ============================================================
for label, vmin, vmax in [("112,300", 112000, 112500), ("112,100", 111800, 112200)]:
    print(f"\n" + "="*80)
    print(f"DEEP DIVE: Volume bucket ~{label}")
    print("="*80)
    bucket = [info for info in all_info if vmin <= info["vol"] <= vmax]
    print(f"Count: {len(bucket)}")
    if not bucket:
        continue
    print(f"Volume range: {min(i['vol'] for i in bucket):.0f} - {max(i['vol'] for i in bucket):.0f}")
    print(f"DX range: {min(i['dx'] for i in bucket):.1f} - {max(i['dx'] for i in bucket):.1f}")
    print(f"DY range: {min(i['dy'] for i in bucket):.1f} - {max(i['dy'] for i in bucket):.1f}")
    print(f"DZ range: {min(i['dz'] for i in bucket):.1f} - {max(i['dz'] for i in bucket):.1f}")
    
    z_counts = defaultdict(int)
    for info in bucket:
        z_counts[round(info["cz"])] += 1
    print("\nDistribution by Z-level:")
    for z in sorted(z_counts.keys()):
        print(f"  Z={z:6.0f}: {z_counts[z]} instances")
    
    shape_counts = defaultdict(list)
    for info in bucket:
        key = (round(info["dx"], 1), round(info["dy"], 1), round(info["dz"], 1))
        shape_counts[key].append(info)
    print(f"\nUnique shape variants: {len(shape_counts)}")
    for key, items in sorted(shape_counts.items()):
        dx, dy, dz = key
        z_levels = set(round(i["cz"]) for i in items)
        print(f"  {dx:6.1f} × {dy:6.1f} × {dz:6.1f} : {len(items):>4} instances at Z-levels {sorted(z_levels)}")
        # Show first few CX/CY positions
        positions = [(i["cx"], i["cy"]) for i in items[:4]]
        print(f"    Sample positions (cx,cy): {positions}")

# ============================================================
# Check the 56-count barrel bases: where are they?
# ============================================================
print("\n" + "="*80)
print("BARREL PMT BASES (6.94M volume) - Z distribution")
print("="*80)
bucket_694M = [info for info in all_info if 6_900_000 <= info["vol"] <= 7_000_000]
print(f"Count: {len(bucket_694M)}")
z_counts = defaultdict(int)
for info in bucket_694M:
    z_counts[round(info["cz"])] += 1
for z in sorted(z_counts.keys()):
    print(f"  Z={z:6.0f}: {z_counts[z]} instances")

# Perch locations
print("\n" + "="*80)
print("PERCHES (41,872-42,671) - Z distribution")
print("="*80)
bucket_perch = [info for info in all_info if 41000 <= info["vol"] <= 43000]
print(f"Count: {len(bucket_perch)}")
z_counts = defaultdict(int)
for info in bucket_perch:
    z_counts[round(info["cz"])] += 1
for z in sorted(z_counts.keys()):
    print(f"  Z={z:6.0f}: {z_counts[z]} instances")

# ============================================================
# Check 70,500 bucket (another holder candidate)
# ============================================================
print("\n" + "="*80)
print("DEEP DIVE: Volume bucket ~70,500 (70506)")
print("="*80)
bucket_70500 = [info for info in all_info if 70000 <= info["vol"] <= 71000]
print(f"Count: {len(bucket_70500)}")
if bucket_70500:
    print(f"Volume range: {min(i['vol'] for i in bucket_70500):.0f} - {max(i['vol'] for i in bucket_70500):.0f}")
    print(f"DX range: {min(i['dx'] for i in bucket_70500):.1f} - {max(i['dx'] for i in bucket_70500):.1f}")
    print(f"DY range: {min(i['dy'] for i in bucket_70500):.1f} - {max(i['dy'] for i in bucket_70500):.1f}")
    print(f"DZ range: {min(i['dz'] for i in bucket_70500):.1f} - {max(i['dz'] for i in bucket_70500):.1f}")
    z_counts = defaultdict(int)
    for info in bucket_70500:
        z_counts[round(info["cz"])] += 1
    print("\nDistribution by Z-level:")
    for z in sorted(z_counts.keys()):
        print(f"  Z={z:6.0f}: {z_counts[z]} instances")
    
    shape_counts = defaultdict(list)
    for info in bucket_70500:
        key = (round(info["dx"], 1), round(info["dy"], 1), round(info["dz"], 1))
        shape_counts[key].append(info)
    print(f"\nUnique shape variants: {len(shape_counts)}")
    for key, items in sorted(shape_counts.items()):
        dx, dy, dz = key
        z_levels = set(round(i["cz"]) for i in items)
        print(f"  {dx:6.1f} × {dy:6.1f} × {dz:6.1f} : {len(items):>4} instances at Z-levels {sorted(z_levels)}")
