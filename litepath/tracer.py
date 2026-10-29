"""Compatibility module: the ray tracer now lives in separate layers.

Everything this module used to define is re-exported from its new home so
existing scripts keep working.  New code should import from:

- ``litepath.hits``             hit-array columns and detector/source codes
- ``litepath.optics``           ``C_MM_NS``, ``N_WATER_DEFAULT``
- ``litepath.geometry.build``   ``Geometry`` and the geometry builders
- ``litepath.engine.kernel``    Taichi intersection functions, ``trace_kernel``
- ``litepath.engine.trace``     ``trace_rays``, ``trace_with_optics``
- ``litepath.sources.track``    muon track lengths
- ``litepath.simulate``         ``trace_muon_light``, ``trace_cherenkov``
"""

from litepath.hits import (  # noqa: F401
    CID_NO_HIT, CID_INNER_STRUCTURE, CID_PMT, CID_LAPPD, CID_TANK_WALL,
    DET_SYS_NONE, DET_SYS_PMT, DET_SYS_LAPPD_DEFAULT, DET_SYS_LAPPD_ANNIE,
    HI, HT, HX, HY, HZ, HNX, HNY, HNZ, HCID, HDI, HDS, HLU, HLV, HMAT,
    N_HIT_COLS, H_ARRIVAL, H_WAVELEN, H_BOUNCE, H_SOURCE, N_EXPANDED_COLS,
    SOURCE_CKV, SOURCE_SCI,
)
from litepath.optics import C_MM_NS, N_WATER_DEFAULT  # noqa: F401
from litepath.geometry.build import (  # noqa: F401
    Geometry, reload_lappd_corrections, build_surfboards, build_surfboard_housings,
    build_geometry, build_single_lappd_geometry,
)
from litepath.engine.kernel import (  # noqa: F401
    _ray_bbox_intersect, _ray_triangle_intersect, _ray_sphere_intersect,
    _ray_rectangle_intersect, _ray_tank_intersect, _ray_box_intersect,
    _box_hit_normal, _pmt_local_coords, _lappd_local_coords, trace_kernel,
)
from litepath.engine.trace import trace_rays, trace_with_optics  # noqa: F401
from litepath.sources.track import (  # noqa: F401
    _housing_from_array, compute_tank_track_length, compute_track_length,
)
from litepath.simulate import trace_muon_light, trace_cherenkov  # noqa: F401
