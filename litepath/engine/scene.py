"""The scene the trace kernel reads, packed into a few buffers.

GPU backends limit how many buffers one kernel can take (Metal: 31), and
the detector kept growing one array at a time.  So the kernel gets the
whole scene as three arrays, however many element types there are:

``scene_f``  float32, every float section back to back (flattened rows)
``scene_i``  int32, every integer section back to back
``layout``   int32, per section its start in its buffer and its row count

A section is one of the Geometry arrays listed below, with a fixed row
width.  The kernel reads them through the accessors in
:mod:`litepath.engine.kernel` (``_f`` / ``_i`` with a section's slot), so
the kernel code still reads by name.

To give the kernel a new kind of data, add a section here (and an accessor
use in the kernel); never add a kernel argument.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (Geometry attribute, row width).  The order fixes each section's slot.
F_SECTIONS: tuple[tuple[str, int], ...] = (
    ("mesh_vertices", 3),
    ("bvh_node_min", 3),
    ("bvh_node_max", 3),
    ("pmt_centers", 3),
    ("pmt_radii", 1),
    ("pmt_directions", 3),
    ("pmt_body_tris", 9),
    ("pmt_rotmats", 9),
    ("pmt_hw_tris", 9),
    ("pmt_instance_pos", 3),
    ("pmt_bounding_radii", 1),
    ("pmt_bvh_bounds", 6),
    ("lappd_data", 7),
    ("lappd_strip_axes", 3),
    ("lappd_housing_data", 16),
    ("annie_lappd_data", 7),
    ("surfboard_data", 16),
)
I_SECTIONS: tuple[tuple[str, int], ...] = (
    ("mesh_triangles", 3),
    ("mesh_material_ids", 1),
    ("bvh_node_left", 1),
    ("bvh_node_right", 1),
    ("bvh_tri_start", 1),
    ("bvh_tri_end", 1),
    ("bvh_tri_ids", 1),
    ("pmt_body_offsets", 1),
    ("pmt_mesh_types", 1),
    ("pmt_hw_types", 1),
    ("pmt_hw_mat_ids", 1),
    ("pmt_bvh_nodes", 4),
    ("pmt_bvh_tri_ids", 1),
    ("pmt_bvh_roots", 1),
)

# Slot numbers (F sections first, then I sections); the kernel imports these.
SLOT = {name: k for k, (name, _w) in enumerate(F_SECTIONS + I_SECTIONS)}
WIDTH = {name: w for name, w in F_SECTIONS + I_SECTIONS}
N_SLOTS = len(SLOT)

# Scalars, after the sections in ``layout``: tank and BVH settings.
LAYOUT_SCALARS = ("bvh_n_nodes",)


@dataclass
class PackedScene:
    """The three kernel buffers and the float scalars."""

    scene_f: np.ndarray      # (Nf,) float32
    scene_i: np.ndarray      # (Ni,) int32
    layout: np.ndarray       # (2 * N_SLOTS + len(LAYOUT_SCALARS),) int32: starts, then row counts, then scalars
    tank: tuple[float, float, float]   # radius, z_min, z_max (mm)

    def section(self, name: str) -> np.ndarray:
        """A section back as a (rows, width) view, for inspection and tests."""
        k, w = SLOT[name], WIDTH[name]
        start, rows = int(self.layout[k]), int(self.layout[N_SLOTS + k])
        buf = self.scene_f if k < len(F_SECTIONS) else self.scene_i
        return buf[start:start + rows * w].reshape(rows, w)


def _rows(array: np.ndarray | None, width: int, dtype) -> np.ndarray:
    """*array* as contiguous (rows, width) of *dtype*; refuses a lossy cast."""
    if array is None:
        return np.zeros((0, width), dtype=dtype)
    a = np.asarray(array)
    if a.size == 0:
        return np.zeros((0, width), dtype=dtype)
    a = a.reshape(-1, width) if width > 1 or a.ndim == 1 else a
    if a.shape[1] != width:
        raise ValueError(f"scene section has rows of {a.shape[1]}, expected {width}")
    out = np.ascontiguousarray(a, dtype=dtype)
    if not np.array_equal(out, a):
        raise ValueError(f"scene section changes value when cast to {np.dtype(dtype).name}")
    return out


def pack_scene(geometry) -> PackedScene:
    """Pack a Geometry's kernel data (see the module docstring)."""
    starts = np.zeros(N_SLOTS, dtype=np.int64)
    counts = np.zeros(N_SLOTS, dtype=np.int64)
    parts_f, parts_i = [], []
    offset = 0
    for name, width in F_SECTIONS:
        rows = _rows(getattr(geometry, name), width, np.float32)
        starts[SLOT[name]], counts[SLOT[name]] = offset, rows.shape[0]
        parts_f.append(rows.ravel())
        offset += rows.size
    offset = 0
    for name, width in I_SECTIONS:
        rows = _rows(getattr(geometry, name), width, np.int32)
        starts[SLOT[name]], counts[SLOT[name]] = offset, rows.shape[0]
        parts_i.append(rows.ravel())
        offset += rows.size
    scene_f = np.concatenate(parts_f) if parts_f else np.zeros(0, np.float32)
    scene_i = np.concatenate(parts_i) if parts_i else np.zeros(0, np.int32)
    # Buffers are never empty (some backends reject zero-size buffers)
    if scene_f.size == 0:
        scene_f = np.zeros(1, np.float32)
    if scene_i.size == 0:
        scene_i = np.zeros(1, np.int32)
    if max(scene_f.size, scene_i.size) >= 2**31:
        raise ValueError("scene too large for 32-bit indexing")
    scalars = [int(geometry.bvh_n_nodes)]
    layout = np.concatenate([starts, counts, scalars]).astype(np.int32)
    return PackedScene(
        scene_f=scene_f, scene_i=scene_i, layout=layout,
        tank=(float(geometry.tank_radius), float(geometry.tank_z_min), float(geometry.tank_z_max)),
    )
