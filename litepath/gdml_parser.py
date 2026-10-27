"""Parse InnerStructure.gdml triangle mesh."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
from lxml import etree

# Matches stl_v42, v42, or PHASE2_INNER_STRUCTURE.stl_v42
_VNAME_PATTERN = re.compile(r"(?:stl_)?v(\d+)")


def _extract_index(name: str) -> int:
    m = _VNAME_PATTERN.search(name)
    if m is None:
        msg = f"Cannot extract vertex index from '{name}'"
        raise ValueError(msg)
    return int(m.group(1))


def parse_gdml(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse GDML file, return (vertices, triangles).

    vertices: shape (N, 3) float32 array of XYZ positions in mm
    triangles: shape (M, 3) int32 array of vertex indices forming each triangle
    """
    tree = etree.parse(str(path))
    root = tree.getroot()

    positions = root.findall(".//position")
    triangles_elem = root.findall(".//triangular")

    n_verts = len(positions)
    vertices = np.empty((n_verts, 3), dtype=np.float32)

    for pos in positions:
        idx = _extract_index(pos.get("name"))
        vertices[idx, 0] = float(pos.get("x"))
        vertices[idx, 1] = float(pos.get("y"))
        vertices[idx, 2] = float(pos.get("z"))

    n_tris = len(triangles_elem)
    triangles = np.empty((n_tris, 3), dtype=np.int32)

    for i, tri in enumerate(triangles_elem):
        for j, attr in enumerate(("vertex1", "vertex2", "vertex3")):
            v = tri.get(attr)
            triangles[i, j] = _extract_index(v)

    return vertices, triangles
