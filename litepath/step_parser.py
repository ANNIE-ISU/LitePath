"""Parse STEP CAD file and extract component manifest for ANNIE detector.

Strategy:
  - Identify PMT perches (volume ~41872, 224 instances) on octagonal panels
  - Pair adjacent perches (Δφ≈6°) at same Z-level — each pair is one PMT
  - Place PMT sphere center INWARD from perch-pair midpoint by PMT radius
  - Identify LAPPD modules as rectangular boxes near Unistrut rail positions
  - Extract tank and inner structure panel positions
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PMT_PERCH_VOLUMES = {41872.2}
PMT_RADIUS_MM = 127.0  # 10-inch PMT
PERCH_PAIR_AZ_TOL = 0.15  # radians (~8.6°) — adjacent perches at same Z


@dataclass
class BBox:
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2,
        )

    @property
    def dims(self) -> Tuple[float, float, float]:
        return (
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            self.zmax - self.zmin,
        )


@dataclass
class Component:
    label: str
    bbox: BBox
    volume: float
    center: Tuple[float, float, float]


LAPPD_HALF_SIZE = 101.0  # mm, half of 202mm sensitive area
DEFAULT_LAPPD_INDICES = [5, 25, 34]  # 3 default positions (diverse Z, diverse azimuth)


@dataclass
class ComponentManifest:
    pmt_centers: List[Tuple[float, float, float]] = field(default_factory=list)
    pmt_radius: float = PMT_RADIUS_MM
    lappd_candidates: List[Component] = field(default_factory=list)
    lappd_centers: List[Tuple[float, float, float]] = field(default_factory=list)
    inner_panels: List[Component] = field(default_factory=list)
    tank_bbox: Optional[BBox] = None
    all_solids: List[Component] = field(default_factory=list)
    perch_centers: List[Tuple[float, float, float]] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        def convert(obj):
            if isinstance(obj, Component):
                return asdict(obj)
            if isinstance(obj, BBox):
                return asdict(obj)
            return obj
        data = {k: convert(v) if not isinstance(v, list) else [convert(x) for x in v]
                for k, v in asdict(self).items()}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ComponentManifest":
        data = json.loads(path.read_text())
        mani = cls()
        for key, items in data.items():
            if key in ("pmt_centers", "perch_centers", "lappd_centers"):
                setattr(mani, key, [tuple(c) for c in items])
            elif key == "pmt_radius":
                setattr(mani, key, items)
            elif key == "tank_bbox":
                setattr(mani, key, BBox(**items) if items else None)
            elif key in ("all_solids", "lappd_candidates", "inner_panels"):
                comps = [Component(label=c["label"], bbox=BBox(**c["bbox"]), volume=c["volume"], center=tuple(c["center"])) for c in items]
                setattr(mani, key, comps)
        return mani

    def summary(self) -> str:
        lines = []
        lines.append(f"  PMT perches: {len(self.perch_centers)}")
        lines.append(f"  PMT spheres: {len(self.pmt_centers)} (R={self.pmt_radius:.0f} mm)")
        lines.append(f"  LAPPD candidates: {len(self.lappd_candidates)}")
        lines.append(f"  Inner panels: {len(self.inner_panels)}")
        lines.append(f"  Tank: {self.tank_bbox is not None}")
        lines.append(f"  Total solids: {len(self.all_solids)}")
        return "\n".join(lines)


def bb_from_solid(solid) -> BBox:
    b = solid.BoundingBox()
    return BBox(b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax)


def parse_step(step_path: Path) -> ComponentManifest:
    import cadquery as cq

    assembly = cq.importers.importStep(str(step_path))
    compound = assembly.val()
    solids = compound.Solids()
    manifest = ComponentManifest()

    perch_centers = []
    panel_solids = []
    lappd_solids = []
    all_comps = []

    pmt_housings = []

    for solid in solids:
        bbox = bb_from_solid(solid)
        vol = solid.Volume()
        cx, cy, cz = bbox.center
        dx, dy, dz = bbox.dims
        sorted_d = sorted([dx, dy, dz])

        comp = Component(label="", bbox=bbox, volume=vol, center=(cx, cy, cz))
        all_comps.append(comp)

        rvol = round(vol, 1)

        if rvol in PMT_PERCH_VOLUMES:
            perch_centers.append((cx, cy, cz))
        elif _is_pmt_housing(vol, sorted_d):
            pmt_housings.append((cx, cy, cz))
        elif _is_panel(sorted_d):
            panel_solids.append(comp)
        elif _is_lappd_module(sorted_d, rvol, cx, cy, cz):
            lappd_solids.append(comp)

    manifest.all_solids = all_comps
    manifest.perch_centers = perch_centers
    manifest.pmt_centers = pmt_housings

    manifest.lappd_candidates = lappd_solids
    if lappd_solids:
        manifest.lappd_centers = [
            lappd_solids[i].center for i in DEFAULT_LAPPD_INDICES
            if i < len(lappd_solids)
        ]
    manifest.inner_panels = panel_solids

    _extract_tank(manifest, all_comps)
    _deduplicate_panels(manifest)

    return manifest


def _is_pmt_housing(vol: float, sorted_d: list) -> bool:
    """Detect PMT housing by volume and dimensions.

    10-inch: 340×254×254mm, vol~6.9M (30 instances)
    Bottom:  253×253×374mm, vol~10.6M (2 instances)
    """
    if 6e6 < vol < 8e6:
        s0, s1, s2 = sorted_d
        if 200 < s0 < 300 and abs(s0 - s1) < 15 and 300 < s2 < 400:
            return True
    if 10e6 < vol < 12e6:
        s0, s1, s2 = sorted_d
        if 240 < s0 < 270 and abs(s0 - s1) < 15 and 360 < s2 < 390:
            return True
    return False


def _is_panel(sorted_d) -> bool:
    s0, s1, s2 = sorted_d
    return s0 < 50 and s2 > 200


def _is_lappd_module(sorted_d, rvol, cx, cy, cz) -> bool:
    # Volume ~112050 identifies the 202×202×279mm LAPPD module boxes
    return abs(rvol - 112050) < 2000


def _extract_tank(manifest: ComponentManifest, all_comps: List[Component]) -> None:
    xs = [c.center[0] for c in all_comps]
    ys = [c.center[1] for c in all_comps]
    zs = [c.center[2] for c in all_comps]

    def robust_bounds(vals, frac=0.005):
        sv = sorted(vals)
        n = len(sv)
        lo = sv[int(n * frac)]
        hi = sv[int(n * (1 - frac))]
        return lo, hi

    xlo, xhi = robust_bounds(xs)
    ylo, yhi = robust_bounds(ys)
    zlo, zhi = robust_bounds(zs)

    manifest.tank_bbox = BBox(xlo, ylo, zlo, xhi, yhi, zhi)


def _deduplicate_panels(manifest: ComponentManifest) -> None:
    panels = manifest.inner_panels
    if not panels:
        return
    kept = []
    for p in panels:
        is_dup = False
        for k in kept:
            d = math.sqrt(
                (p.center[0] - k.center[0]) ** 2
                + (p.center[1] - k.center[1]) ** 2
                + (p.center[2] - k.center[2]) ** 2
            )
            if d < 100:
                is_dup = True
                break
        if not is_dup:
            kept.append(p)
    manifest.inner_panels = kept
