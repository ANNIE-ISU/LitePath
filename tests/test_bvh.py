"""Structure of the packed per-mesh-type BVH used for PMT meshes."""

import numpy as np

from litepath.bvh import build_packed_bvh


def _walk(pb, root):
    """Yield (node, leaf triangle ids) for every node under *root*."""
    stack = [root]
    while stack:
        node = stack.pop()
        left, right, start, end = pb.nodes[node]
        if left == -1:
            yield node, pb.tri_ids[start:end]
        else:
            yield node, None
            stack += [left, right]


def test_packed_bvh_covers_each_slot_exactly():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 9)).astype(np.float32)
    b = rng.normal(size=(300, 9)).astype(np.float32)
    slots = [(a, np.arange(0, 200)), (a, np.array([], dtype=int)),
             (a, np.arange(200, 500)), (b, np.arange(300))]

    pb = build_packed_bvh(slots, max_leaf=8)

    assert pb.roots[1] == -1  # empty slot
    for k, (tris, ids) in enumerate(slots):
        if len(ids) == 0:
            continue
        seen = []
        for node, leaf_ids in _walk(pb, pb.roots[k]):
            lo, hi = pb.bounds[node, :3], pb.bounds[node, 3:]
            if leaf_ids is None:
                for child in pb.nodes[node, :2]:
                    assert (pb.bounds[child, :3] >= lo).all()
                    assert (pb.bounds[child, 3:] <= hi).all()
            else:
                assert len(leaf_ids) <= 8
                verts = tris[leaf_ids].reshape(-1, 3)
                assert (verts >= lo).all() and (verts <= hi).all()
                seen += leaf_ids.tolist()
        assert sorted(seen) == sorted(ids.tolist())


def test_packed_bvh_all_empty():
    pb = build_packed_bvh([(np.zeros((0, 9), dtype=np.float32), np.array([], dtype=int))] * 3)
    assert (pb.roots == -1).all()
    assert pb.bounds.shape == (0, 6) and pb.nodes.shape == (0, 4) and pb.tri_ids.shape == (0,)
