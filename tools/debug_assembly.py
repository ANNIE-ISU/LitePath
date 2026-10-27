"""Investigate: are PMT glasses at barrel Z-levels?"""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
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

# 1. Are there any solids with volume 3.5M-4.0M at barrel Z-levels?
barrel_z = {820, 1252, 2115, 2979}
print("=== Solids vol 3.5-4.0M at barrel Z-levels ===")
count = 0
for info in all_info:
    if 3_500_000 <= info["vol"] <= 4_000_000:
        zr = round(info["cz"])
        if zr in barrel_z:
            sd = sorted([info["dx"], info["dy"], info["dz"]])
            print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
                  f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
                  f"sorted=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})  "
                  f"pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})")
            count += 1
print(f"Total: {count}")

# 2. What's at Z=820 with vol > 100,000?
print("\n=== All solids at Z≈820 with vol > 1000 ===")
for info in all_info:
    if abs(info["cz"] - 820) < 10 and info["vol"] > 1000:
        sd = sorted([info["dx"], info["dy"], info["dz"]])
        print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
              f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
              f"pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})  "
              f"sorted=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})")

# 3. What's at Z=1253 with vol > 1000?
print("\n=== All solids at Z≈1253 with vol > 1000 ===")
for info in all_info:
    if abs(info["cz"] - 1253) < 10 and info["vol"] > 1000:
        sd = sorted([info["dx"], info["dy"], info["dz"]])
        print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
              f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
              f"pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})  "
              f"sorted=({sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f})")

# 4. Check: what is the "pmt_slab" at Z=1278? Near what?
print("\n=== pmt_slab (202x202x290) at Z≈1278 ===")
for info in all_info:
    if 3_600_000 <= info["vol"] <= 4_000_000:
        sd = sorted([info["dx"], info["dy"], info["dz"]])
        if abs(sd[0]-202)<2 and abs(sd[1]-202)<2 and abs(sd[2]-290)<5:
            if abs(info["cz"]-1278) < 10:
                print(f"  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
                      f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
                      f"pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})")

# 5. Find ANY solids (any vol) at Z=1278 near the glass positions
print("\n=== All solids at Z≈1278 near glass pos (-983,-731) ===")
for info in all_info:
    if abs(info["cz"]-1278) < 10:
        d = np.sqrt((info["cx"]+983.5)**2 + (info["cy"]+731.9)**2)
        if d < 300:
            sd = sorted([info["dx"], info["dy"], info["dz"]])
            print(f"  d={d:>6.1f}  idx={info['idx']:>5} vol={info['vol']:>10.0f}  "
                  f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}  "
                  f"pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})")

# 6. Perch positions relative to glass
print("\n=== Perches near 10\" glass at (-983,-731,1278) ===")
for info in all_info:
    if 40_000 <= info["vol"] <= 43_000:
        d = np.sqrt((info["cx"]+983.5)**2 + (info["cy"]+731.9)**2 + (info["cz"]-1278.4)**2)
        if d < 300:
            print(f"  d={d:>6.1f}  idx={info['idx']:>5} pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})  "
                  f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}")

# 7. base_10 positions relative to that glass
print("\n=== base_10inch near glass (-983,-731,1278) ===")
for info in all_info:
    sd = sorted([info["dx"], info["dy"], info["dz"]])
    if 6_900_000 <= info["vol"] <= 7_000_000 and 250 <= sd[0] <= 260 and 305 <= sd[1] <= 345:
        d = np.sqrt((info["cx"]+983.5)**2 + (info["cy"]+731.9)**2 + (info["cz"]-1278.4)**2)
        if d < 400:
            print(f"  d={d:>6.1f}  idx={info['idx']:>5} pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})  "
                  f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}")

# 8. What about the 73,755 holders near that glass?
print("\n=== 73,755 holders near glass (-983,-731,1278) ===")
for info in all_info:
    if 73_000 <= info["vol"] <= 75_000:
        d = np.sqrt((info["cx"]+983.5)**2 + (info["cy"]+731.9)**2 + (info["cz"]-1278.4)**2)
        if d < 400:
            print(f"  d={d:>6.1f}  idx={info['idx']:>5} pos=({info['cx']:>7.1f},{info['cy']:>7.1f},{info['cz']:>7.1f})  "
                  f"dims={info['dx']:>6.1f}×{info['dy']:>6.1f}×{info['dz']:>6.1f}")

print("\nDone.")
