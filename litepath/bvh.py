"""BVH acceleration structure for triangle meshes.

Builds a median-split BVH (flat arrays) suitable for Taichi kernel traversal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BVHData:
    """Flat-array BVH ready for the Taichi kernel.

    Nodes are stored in post-order (children before parent).  The root
    is always at index ``n_nodes - 1``.
    """
    node_min: np.ndarray   # (N, 3) float32 — bbox lower corner
    node_max: np.ndarray   # (N, 3) float32 — bbox upper corner
    node_left: np.ndarray  # (N,)  int32 — left child, -1 for leaf
    node_right: np.ndarray # (N,)  int32 — right child, -1 for leaf
    tri_start: np.ndarray  # (N,)  int32 — leaf: start index in tri_ids
    tri_end: np.ndarray    # (N,)  int32 — leaf: exclusive end in tri_ids
    tri_ids: np.ndarray    # (M,)  int32 — reordered triangle indices

    @property
    def n_nodes(self) -> int:
        return self.node_min.shape[0]

    @property
    def root(self) -> int:
        return self.n_nodes - 1


def build_bvh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_leaf: int = 8,
) -> BVHData:
    """Build a median-split BVH for the given triangle mesh.

    Parameters
    ----------
    vertices:
        (N, 3) float32 — vertex positions in mm.
    triangles:
        (M, 3) int32 — vertex-index triplets.
    max_leaf:
        Maximum number of triangles per leaf node.

    Returns
    -------
    BVHData with flat arrays arranged in post-order (root at end).
    """
    n = triangles.shape[0]
    if n == 0:
        return BVHData(
            node_min=np.empty((0, 3), dtype=np.float32),
            node_max=np.empty((0, 3), dtype=np.float32),
            node_left=np.empty(0, dtype=np.int32),
            node_right=np.empty(0, dtype=np.int32),
            tri_start=np.empty(0, dtype=np.int32),
            tri_end=np.empty(0, dtype=np.int32),
            tri_ids=np.empty(0, dtype=np.int32),
        )

    # Precompute per-triangle centroids and bounding boxes
    tri_verts = vertices[triangles]
    centroids = tri_verts.mean(axis=1).astype(np.float64)
    tri_min = tri_verts.min(axis=1)
    tri_max = tri_verts.max(axis=1)

    nodes: list[dict] = []
    tri_ids_buf: list[int] = []

    indices = np.arange(n, dtype=np.int32)
    _build_recursive(indices, centroids, tri_min, tri_max,
                     nodes, tri_ids_buf, max_leaf)

    n_nodes = len(nodes)
    node_min = np.empty((n_nodes, 3), dtype=np.float32)
    node_max = np.empty((n_nodes, 3), dtype=np.float32)
    node_left = np.empty(n_nodes, dtype=np.int32)
    node_right = np.empty(n_nodes, dtype=np.int32)
    tri_start = np.empty(n_nodes, dtype=np.int32)
    tri_end = np.empty(n_nodes, dtype=np.int32)

    for i, nd in enumerate(nodes):
        node_min[i] = nd["min"]
        node_max[i] = nd["max"]
        node_left[i] = nd["left"]
        node_right[i] = nd["right"]
        tri_start[i] = nd["tri_start"]
        tri_end[i] = nd["tri_end"]

    return BVHData(
        node_min=node_min,
        node_max=node_max,
        node_left=node_left,
        node_right=node_right,
        tri_start=tri_start,
        tri_end=tri_end,
        tri_ids=np.array(tri_ids_buf, dtype=np.int32),
    )


def _build_recursive(
    indices: np.ndarray,
    centroids: np.ndarray,
    tri_min: np.ndarray,
    tri_max: np.ndarray,
    nodes: list,
    tri_ids_buf: list,
    max_leaf: int,
) -> int:
    """Recursive median-split builder.  Returns this node's index.

    Nodes are appended to ``nodes`` in post-order (children first).
    """
    n = len(indices)

    # Combined bounding box of all triangles in this subset
    bmin = tri_min[indices].min(axis=0)
    bmax = tri_max[indices].max(axis=0)

    if n <= max_leaf:
        start = len(tri_ids_buf)
        tri_ids_buf.extend(indices.tolist())
        end = len(tri_ids_buf)
        idx = len(nodes)
        nodes.append({
            "min": bmin.copy(),
            "max": bmax.copy(),
            "left": -1,
            "right": -1,
            "tri_start": start,
            "tri_end": end,
        })
        return idx

    # Split on the longest axis at the median centroid
    sizes = bmax - bmin
    axis = int(np.argmax(sizes))

    order = np.argsort(centroids[indices, axis])
    sorted_indices = indices[order]
    mid = n // 2

    left_idx = _build_recursive(
        sorted_indices[:mid], centroids, tri_min, tri_max,
        nodes, tri_ids_buf, max_leaf,
    )
    right_idx = _build_recursive(
        sorted_indices[mid:], centroids, tri_min, tri_max,
        nodes, tri_ids_buf, max_leaf,
    )

    idx = len(nodes)
    nodes.append({
        "min": bmin.copy(),
        "max": bmax.copy(),
        "left": left_idx,
        "right": right_idx,
        "tri_start": -1,
        "tri_end": -1,
    })
    return idx


@dataclass
class PackedBVH:
    """Several per-mesh BVHs packed into shared flat arrays.

    Used for instanced meshes (PMT bodies and holders): each mesh type gets
    its own BVH in the mesh's local frame, shared by every instance of that
    type.  Slot *k* is one mesh type; ``roots[k]`` is its root node, or -1
    when the slot has no triangles.  Leaf triangle ids index the source
    triangle array the slot was built from.
    """
    bounds: np.ndarray   # (N, 6) float32 — [min x,y,z, max x,y,z] per node
    nodes: np.ndarray    # (N, 4) int32 — [left, right, tri_start, tri_end]; left = -1 for a leaf
    tri_ids: np.ndarray  # (M,)  int32 — source triangle index per leaf entry
    roots: np.ndarray    # (K,)  int32 — root node per slot, -1 = empty


def build_packed_bvh(
    slots: list[tuple[np.ndarray, np.ndarray]],
    max_leaf: int = 8,
    pad: float = 1e-3,
) -> PackedBVH:
    """Build one BVH per slot and pack them into shared arrays.

    Parameters
    ----------
    slots:
        One ``(tris, ids)`` pair per slot: ``tris`` is an (T, 9) float32
        triangle soup (three vertices per row) and ``ids`` the indices of
        the triangles in ``tris`` that this slot's BVH should contain.
    max_leaf:
        Maximum number of triangles per leaf node.
    pad:
        Margin (mm) added to every node box so float rounding can never
        reject a triangle lying on a box face.
    """
    bounds, nodes, tri_ids = [], [], []
    roots = np.full(len(slots), -1, dtype=np.int32)
    n_nodes = n_ids = 0

    for k, (tris, ids) in enumerate(slots):
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 0:
            continue
        verts = tris[ids].reshape(-1, 3).astype(np.float32)
        local = build_bvh(verts, np.arange(verts.shape[0], dtype=np.int32).reshape(-1, 3),
                          max_leaf=max_leaf)
        leaf = local.node_left == -1
        node = np.stack([local.node_left, local.node_right,
                         local.tri_start, local.tri_end], axis=1).astype(np.int32)
        node[~leaf, 0:2] += n_nodes
        node[leaf, 2:4] += n_ids
        bounds.append(np.hstack([local.node_min - pad, local.node_max + pad]).astype(np.float32))
        nodes.append(node)
        tri_ids.append(ids[local.tri_ids])
        roots[k] = n_nodes + local.root
        n_nodes += local.n_nodes
        n_ids += local.tri_ids.shape[0]

    return PackedBVH(
        bounds=np.vstack(bounds) if bounds else np.zeros((0, 6), dtype=np.float32),
        nodes=np.vstack(nodes) if nodes else np.zeros((0, 4), dtype=np.int32),
        tri_ids=np.concatenate(tri_ids).astype(np.int32) if tri_ids else np.zeros(0, dtype=np.int32),
        roots=roots,
    )
