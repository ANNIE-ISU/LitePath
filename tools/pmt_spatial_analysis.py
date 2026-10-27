"""Final PMT spatial analysis — v3 with correct perch Z-matching."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
print("Loading STEP file...")
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects
print(f"Total solids: {len(solids)}")

all_info = []
for i, s in enumerate(solids):
    try:
        vol = s.Volume()
        bb = s.BoundingBox()
        dx = bb.xmax-bb.xmin; dy = bb.ymax-bb.ymin; dz = bb.zmax-bb.zmin
        cx = (bb.xmin+bb.xmax)/2; cy = (bb.ymin+bb.ymax)/2; cz = (bb.zmin+bb.zmax)/2
        all_info.append({"idx":i,"vol":vol,"dx":dx,"dy":dy,"dz":dz,"cx":cx,"cy":cy,"cz":cz})
    except: pass

def classify(info):
    v=info["vol"]; sd=sorted([info["dx"],info["dy"],info["dz"]])
    if 3_600_000<=v<=4_000_000:
        if 195<=sd[0]<=210 and 250<=sd[1]<=270 and 250<=sd[2]<=270:
            if 250<=sd[1]<=256: return "glass_8inch"
            elif 257<=sd[1]<=265: return "glass_10inch"
        elif 195<=sd[0]<=210 and 195<=sd[1]<=210 and 280<=sd[2]<=300:
            return "pmt_slab"
    if 6_900_000<=v<=7_000_000:
        if 250<=sd[0]<=260 and 250<=sd[1]<=260 and 335<=sd[2]<=345: return "base_8inch"
        elif 250<=sd[0]<=260 and 305<=sd[1]<=345 and 330<=sd[2]<=345: return "base_10inch"
    if 40_000<=v<=41_900: return "perch"       # only true perches
    if 42_000<=v<=43_000: return "perch_plate" # flat plate variant
    if 10_500_000<=v<=10_800_000: return "lux_bottom"
    if 11_500_000<=v<=12_000_000: return "etel_top"
    return None

classified = defaultdict(list)
for info in all_info:
    cat = classify(info)
    if cat: classified[cat].append(info)

print(f"Classification: glass_8={len(classified['glass_8inch'])}, glass_10={len(classified['glass_10inch'])}")
print(f"  base_8={len(classified['base_8inch'])}, base_10={len(classified['base_10inch'])}")
print(f"  perch={len(classified['perch'])}, perch_plate={len(classified['perch_plate'])}")
print(f"  pmt_slab={len(classified['pmt_slab'])}, unclassified={len(all_info)-sum(len(v) for v in classified.values())}")

def angle(x,y): return np.degrees(np.arctan2(y,x)) % 360

# =============================================================
# TASK 1: 8" holder candidates
# =============================================================
print("\n"+"="*80)
print("TASK 1: 8\" holder candidates from unclassified at barrel Z-levels")
print("="*80)

barrel_z = {821, 1253, 2116, 2979}
uncl = [i for i in all_info if classify(i) is None]

for zlvl in sorted(barrel_z):
    at_z = [i for i in uncl if abs(i["cz"]-zlvl)<10]
    buckets = defaultdict(list)
    for i in at_z:
        buckets[round(i["vol"]/100)*100].append(i)
    
    print(f"\nZ≈{zlvl}: {len(at_z)} unclassified solids")
    for bucket, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        count = len(items)
        vols = [i["vol"] for i in items]
        dxs = [i["dx"] for i in items]
        print(f"  bucket {bucket:>8}: count={count:>4}  vol {min(vols):.0f}-{max(vols):.0f}  dx {min(dxs):.1f}-{max(dxs):.1f}")
        # Show unique shapes
        shapes = defaultdict(int)
        for i in items:
            shapes[(round(i["dx"],1),round(i["dy"],1),round(i["dz"],1))] += 1
        for key,n in sorted(shapes.items(), key=lambda x: -x[1])[:4]:
            print(f"    {key[0]:6.1f}x{key[1]:6.1f}x{key[2]:6.1f} : {n}")

# =============================================================
# TASK 2: One 8" PMT assembly
# =============================================================
print("\n"+"="*80)
print("TASK 2: One 8\" PMT assembly (matching by azimuth angle)")
print("="*80)

g8 = classified["glass_8inch"]
b8 = classified["base_8inch"]

# Find a glass with matching-angle base near same Z
for g in g8:
    ga = angle(g["cx"], g["cy"])
    # Find base at barrel Z closest to this glass Z
    gz = g["cz"]
    # Which barrel Z is closest?
    best_b = None; best_bd = float("inf")
    for b in b8:
        bd = abs(b["cz"]-gz)
        if bd < best_bd and abs(angle(b["cx"],b["cy"])-ga) < 30:
            best_bd = bd
            best_b = b
    
    if best_b and best_bd < 500:
        d = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2 + (g["cz"]-best_b["cz"])**2)
        b_angle = angle(best_b["cx"], best_b["cy"])
        print(f"\nglass_8 idx={g['idx']}: pos=({g['cx']:>7.1f},{g['cy']:>7.1f},{g['cz']:>7.1f})  dims={g['dx']:.1f}x{g['dy']:.1f}x{g['dz']:.1f}  angle={ga:.1f}°")
        print(f"base_8  idx={best_b['idx']}: pos=({best_b['cx']:>7.1f},{best_b['cy']:>7.1f},{best_b['cz']:>7.1f})  dims={best_b['dx']:.1f}x{best_b['dy']:.1f}x{best_b['dz']:.1f}  angle={b_angle:.1f}°")
        print(f"  distance={d:.1f}mm  ΔZ={best_bd:.0f}mm")
        rel = (best_b["cx"]-g["cx"], best_b["cy"]-g["cy"], best_b["cz"]-g["cz"])
        print(f"  relative(base-glass): ({rel[0]:.1f}, {rel[1]:.1f}, {rel[2]:.1f})")
        
        # Nearby solids (within 200mm of glass)
        gpos = np.array([g["cx"],g["cy"],g["cz"]])
        print(f"  Solids within 200mm:")
        nearby = []
        for info in all_info:
            if info["idx"]==g["idx"]: continue
            pp = np.array([info["cx"],info["cy"],info["cz"]])
            dd = np.linalg.norm(pp-gpos)
            if dd<200: nearby.append((dd,info))
        nearby.sort()
        for dd,info in nearby[:20]:
            cat = classify(info) or "uncl"
            print(f"    d={dd:>6.1f}  {cat:20s} vol={info['vol']:>8.0f}  dims={info['dx']:>6.1f}x{info['dy']:>6.1f}x{info['dz']:>6.1f}")
        if len(nearby)>20: print(f"    ... and {len(nearby)-20} more")
        break

# =============================================================
# TASK 3: One 10" PMT assembly
# =============================================================
print("\n"+"="*80)
print("TASK 3: One 10\" PMT assembly (match by azimuth, near-Z perch)")
print("="*80)

g10 = classified["glass_10inch"]
b10 = classified["base_10inch"]
perches = classified["perch"]  # only 41,872 vol true perches

holder_all = [i for i in all_info if 73_000<=i["vol"]<=75_000]
h_narrow = [h for h in holder_all if round(h["dx"],1)==25.4 and round(h["dy"],1)==67.8]
h_narrow_rot = [h for h in holder_all if round(h["dx"],1)==67.8 and round(h["dy"],1)==25.4]
h_square = [h for h in holder_all if round(h["dx"],1)==65.9 and round(h["dy"],1)==65.9]

for g in g10:
    ga = angle(g["cx"], g["cy"])
    gpos = np.array([g["cx"],g["cy"],g["cz"]])
    
    # Find matching-angle base at closest barrel Z
    best_b = None; best_bd = float("inf")
    for b in b10:
        if abs(angle(b["cx"],b["cy"])-ga) < 30:
            if abs(b["cz"]-g["cz"]) < best_bd:
                best_bd = abs(b["cz"]-g["cz"])
                best_b = b
    
    # Find matching-angle perch closest in Z
    best_p = None; best_pd = float("inf")
    for p in perches:
        if abs(angle(p["cx"],p["cy"])-ga) < 30:
            pd = np.linalg.norm(np.array([p["cx"],p["cy"],p["cz"]])-gpos)
            if pd < best_pd:
                best_pd = pd
                best_p = p
    
    if best_b and best_bd < 500 and best_p and best_pd < 300:
        d_b = np.sqrt((g["cx"]-best_b["cx"])**2 + (g["cy"]-best_b["cy"])**2 + (g["cz"]-best_b["cz"])**2)
        b_angle = angle(best_b["cx"], best_b["cy"])
        print(f"\nglass_10 idx={g['idx']}: pos=({g['cx']:>7.1f},{g['cy']:>7.1f},{g['cz']:>7.1f})  dims={g['dx']:.1f}x{g['dy']:.1f}x{g['dz']:.1f}  angle={ga:.1f}°")
        print(f"base_10  idx={best_b['idx']}: pos=({best_b['cx']:>7.1f},{best_b['cy']:>7.1f},{best_b['cz']:>7.1f})  dims={best_b['dx']:.1f}x{best_b['dy']:.1f}x{best_b['dz']:.1f}  angle={b_angle:.1f}°  d={d_b:.1f}mm  ΔZ={best_bd:.0f}mm")
        
        rel_b = (best_b["cx"]-g["cx"], best_b["cy"]-g["cy"], best_b["cz"]-g["cz"])
        print(f"  relative(base-glass): ({rel_b[0]:.1f}, {rel_b[1]:.1f}, {rel_b[2]:.1f})")
        
        print(f"perch    idx={best_p['idx']}: pos=({best_p['cx']:>7.1f},{best_p['cy']:>7.1f},{best_p['cz']:>7.1f})  dims={best_p['dx']:.1f}x{best_p['dy']:.1f}x{best_p['dz']:.1f}  d_gp={best_pd:.1f}mm")
        rel_p = (best_p["cx"]-g["cx"], best_p["cy"]-g["cy"], best_p["cz"]-g["cz"])
        print(f"  relative(perch-glass): ({rel_p[0]:.1f}, {rel_p[1]:.1f}, {rel_p[2]:.1f})")
        
        # Holders near this glass (within 300mm)
        print(f"\n  Holders within 300mm:")
        for hlist,hname in [(h_narrow,"narrow"),(h_narrow_rot,"narrow_rot"),(h_square,"square")]:
            for h in hlist:
                dh = np.linalg.norm(np.array([h["cx"],h["cy"],h["cz"]])-gpos)
                if dh<300:
                    hrel = (h["cx"]-g["cx"], h["cy"]-g["cy"], h["cz"]-g["cz"])
                    print(f"    {hname:16s} d={dh:>6.1f}  pos=({h['cx']:>7.1f},{h['cy']:>7.1f},{h['cz']:>7.1f})  dims={h['dx']:.1f}x{h['dy']:.1f}x{h['dz']:.1f}  rel=({hrel[0]:>7.1f},{hrel[1]:>7.1f},{hrel[2]:>7.1f})")
        
        print(f"\n  Unclassified within 300mm (grouped by volume):")
        nearby_u = []
        for info in all_info:
            if classify(info) is not None: continue
            pp = np.array([info["cx"],info["cy"],info["cz"]])
            dd = np.linalg.norm(pp-gpos)
            if dd<300: nearby_u.append((dd,info))
        nearby_u.sort()
        ub = defaultdict(list)
        for dd,info in nearby_u:
            ub[round(info["vol"]/100)*100].append((dd,info))
        for bucket in sorted(ub, key=lambda b: -len(ub[b]))[:8]:
            items = ub[bucket]
            print(f"    bucket {bucket:>8}: {len(items):>3} items")
            for dd,info in items[:2]:
                cat = classify(info) or "uncl"
                print(f"      d={dd:>6.1f}  {cat:20s} vol={info['vol']:>8.0f}  dims={info['dx']:>6.1f}x{info['dy']:>6.1f}x{info['dz']:>6.1f}")
        break

# =============================================================
# TASK 4: Missing 10" parts
# =============================================================
print("\n"+"="*80)
print("TASK 4: Missing 10\" components — all unclassified near any 10\" assembly")
print("="*80)

# Collect all unclassified within 300mm of any 10" glass
near10 = set()
for g in g10:
    gpos = np.array([g["cx"],g["cy"],g["cz"]])
    for info in all_info:
        if classify(info) is not None: continue
        pp = np.array([info["cx"],info["cy"],info["cz"]])
        dd = np.linalg.norm(pp-gpos)
        if dd<300: near10.add(info["idx"])

print(f"Unique unclassified solids within 300mm of any 10\" glass: {len(near10)}")
# Group by volume
vb = defaultdict(list)
for idx in near10:
    info = next(i for i in all_info if i["idx"]==idx)
    vb[round(info["vol"]/100)*100].append(info)

print(f"\n{'Count':>6} {'Bucket':>10} {'VolRange':>16} {'DX range':>16} {'Z-levels':>30}")
print("="*80)
for bucket, items in sorted(vb.items(), key=lambda x: -len(x[1])):
    if len(items) < 5: continue
    vols = [i["vol"] for i in items]
    dxs = [i["dx"] for i in items]
    zlevs = set(round(i["cz"]) for i in items)
    print(f"{len(items):>6} {bucket:>10} {min(vols):.0f}-{max(vols):.0f}  {min(dxs):.1f}-{max(dxs):.1f}  {sorted(zlevs)}")

print("\nDone.")
