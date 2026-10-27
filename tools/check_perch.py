"""Quick check: perch classification."""
import cadquery as cq
import numpy as np
from collections import defaultdict
from pathlib import Path

STEP_PATH = str(Path(__file__).resolve().parents[1] / "F10091903_-.step")
result = cq.importers.importStep(STEP_PATH)
solids = result.solids().objects

# Check perches near Z=1331
for s in solids:
    try:
        vol = s.Volume()
        bb = s.BoundingBox()
        cx=(bb.xmin+bb.xmax)/2; cy=(bb.ymin+bb.ymax)/2; cz=(bb.zmin+bb.zmax)/2
        dx=bb.xmax-bb.xmin; dy=bb.ymax-bb.ymin; dz=bb.zmax-bb.zmin
    except: continue
    
    if 40_000 <= vol <= 43_000 and 1300 < cz < 1400:
        print(f"vol={vol:.0f} pos=({cx:.1f},{cy:.1f},{cz:.1f}) dims={dx:.1f}x{dy:.1f}x{dz:.1f}")
