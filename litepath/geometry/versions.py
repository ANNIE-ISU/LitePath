"""Named geometry versions and the one detector build every tool loads.

A geometry version is a YAML file in ``geometry/versions/`` naming the input
files (PMT table, structure GDML, optional STEP manifest), the build
settings and the overlays applied on top.  :func:`load_build` turns a
version into the :class:`~litepath.geometry.build.Geometry` the kernel uses,
together with the version and its fingerprint.

The fingerprint is a SHA-256 over the build settings, the resolved PMT
offsets and the content hashes of every input file (including the PMT mesh
files).  Two builds with the same fingerprint used the same detector,
whatever name or flags produced it.

Statuses:

- ``tagged``: frozen.  Tagging records the fingerprint and per-file hashes;
  a build refuses to run if anything has changed since.
- ``draft``: may change; builds record their fingerprint and warn.
- ``unnamed``: built on the fly from the older command-line flags.

Paths in version and overlay files are relative to the repository root.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_DIR = REPO_ROOT / "geometry"
VERSIONS_DIR = GEOMETRY_DIR / "versions"

DEFAULT_GEOMETRY = "annie-2022-scan"

STATUS_TAGGED = "tagged"
STATUS_DRAFT = "draft"
STATUS_UNNAMED = "unnamed"

# Build settings and their defaults (those of build_geometry()).
DEFAULT_SETTINGS = {
    "det_rotation_deg": 22.5,
    "bottom_rotation_deg": 45.0,
    "bottom_spin_deg": 0.0,
    "z_offset_mm": 0.0,
    "lappd_model": "annie",
    "surfboards": 0,
    "lappd_indices": None,
    "structure_mesh": True,
    "pmt_meshes": True,
    "lappds": True,
}

INPUT_ROLES = ("pmt_table", "structure", "manifest", "step")


class GeometryError(Exception):
    """A geometry version cannot be loaded or has changed since tagging."""


@dataclass
class GeometryVersion:
    """A geometry version as read from its file (paths resolved)."""

    name: str
    status: str
    description: str = ""
    inputs: dict[str, Optional[Path]] = field(default_factory=dict)
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    overlays: list[Path] = field(default_factory=list)
    tagged: dict = field(default_factory=dict)   # date, fingerprint, files
    path: Optional[Path] = None                  # the version file


@dataclass
class Build:
    """A built detector and where it came from."""

    geometry: "Geometry"  # noqa: F821  (litepath.geometry.build.Geometry)
    version: GeometryVersion
    fingerprint: str
    pmt_offsets: dict[int, tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Reading and listing versions
# ---------------------------------------------------------------------------

def _repo_path(value) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    """Path as written in version files: relative to the repo root if inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def version_path(name_or_path: str | Path) -> Path:
    """Resolve a version name (``annie-2026-data``) or file path."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        if not p.exists():
            raise GeometryError(f"Geometry version file not found: {p}")
        return p
    candidate = VERSIONS_DIR / f"{name_or_path}.yaml"
    if not candidate.exists():
        known = ", ".join(list_versions()) or "none"
        raise GeometryError(f"Unknown geometry version '{name_or_path}' (known: {known})")
    return candidate


def list_versions() -> list[str]:
    return sorted(p.stem for p in VERSIONS_DIR.glob("*.yaml"))


def load_version(name_or_path: str | Path) -> GeometryVersion:
    """Read a geometry version file."""
    path = version_path(name_or_path)
    with open(path) as f:
        doc = yaml.safe_load(f) or {}

    unknown = set(doc.get("settings", {})) - set(DEFAULT_SETTINGS)
    if unknown:
        raise GeometryError(f"{path}: unknown settings {sorted(unknown)}")
    status = doc.get("status", STATUS_DRAFT)
    if status not in (STATUS_TAGGED, STATUS_DRAFT):
        raise GeometryError(f"{path}: status must be '{STATUS_TAGGED}' or '{STATUS_DRAFT}'")

    inputs = doc.get("inputs", {})
    return GeometryVersion(
        name=doc.get("name", path.stem),
        status=status,
        description=" ".join(str(doc.get("description", "")).split()),
        inputs={role: _repo_path(inputs.get(role)) for role in INPUT_ROLES},
        settings={**DEFAULT_SETTINGS, **doc.get("settings", {})},
        overlays=[_repo_path(o) for o in doc.get("overlays", [])],
        tagged=doc.get("tagged", {}) or {},
        path=path,
    )


def version_from_flags(values: dict) -> GeometryVersion:
    """Unnamed version from the older geometry flags (``--pmt-csv`` etc.).

    *values* holds the parsed flag values by argparse ``dest``; missing or
    ``None`` entries take build_geometry()'s defaults.
    """
    v = {k: val for k, val in values.items() if val is not None}
    indices = v.get("lappd_indices")
    if isinstance(indices, str):
        indices = [int(x) for x in indices.split(",")]
    no_gdml = bool(v.get("no_gdml", False))
    return GeometryVersion(
        name="unnamed",
        status=STATUS_UNNAMED,
        description="built from command-line geometry flags",
        inputs={
            "pmt_table": Path(v["pmt_csv"]) if v.get("pmt_csv") else None,
            "structure": None if no_gdml else Path(v.get("gdml", "PHASE2_INNER_STRUCTURE_closed.gdml")),
            "manifest": Path(v["manifest"]) if v.get("manifest") else None,
            "step": Path(v["step"]) if v.get("step") else None,
        },
        settings={
            **DEFAULT_SETTINGS,
            "det_rotation_deg": float(v.get("det_rotation", DEFAULT_SETTINGS["det_rotation_deg"])),
            "bottom_rotation_deg": float(v.get("bottom_rot", DEFAULT_SETTINGS["bottom_rotation_deg"])),
            "bottom_spin_deg": float(v.get("bottom_spin", DEFAULT_SETTINGS["bottom_spin_deg"])),
            "z_offset_mm": float(v.get("z_offset", 0.0)),
            "lappd_model": v.get("lappd_model", DEFAULT_SETTINGS["lappd_model"]),
            "surfboards": int(v.get("surfboard", 0)),
            "lappd_indices": indices,
            "structure_mesh": not no_gdml,
            "pmt_meshes": not v.get("no_pmt_holders", False),
            "lappds": not v.get("no_lappd", False),
        },
    )


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

def load_pmt_offsets(overlays: list[Path]) -> dict[int, tuple[float, float, float]]:
    """Sum the ``pmt-offsets`` overlays into one offset (mm) per channel key."""
    total: dict[int, list[float]] = {}
    for path in overlays:
        if not path.exists():
            raise GeometryError(f"Overlay not found: {path}")
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        if doc.get("kind") != "pmt-offsets":
            raise GeometryError(f"{path}: unsupported overlay kind {doc.get('kind')!r}")
        for key, offset in (doc.get("offsets") or {}).items():
            if len(offset) != 3:
                raise GeometryError(f"{path}: offset for {key} needs 3 values")
            acc = total.setdefault(int(key), [0.0, 0.0, 0.0])
            for i in range(3):
                acc[i] += float(offset[i])
    return {k: tuple(v) for k, v in sorted(total.items()) if any(v)}


# ---------------------------------------------------------------------------
# Fingerprint and tagging
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(version: GeometryVersion) -> dict[str, Path]:
    """Every file the build reads, keyed as recorded in version files."""
    from litepath.pmt_mesh import MESH_DIR

    files: dict[str, Path] = {}
    s = version.settings
    for role in INPUT_ROLES:
        p = version.inputs.get(role)
        if p is None or (role == "structure" and not s["structure_mesh"]):
            continue
        files[_rel(p)] = p
        if role == "structure":
            cache = p.with_name(p.stem + "_cache.npz")
            if cache.exists():
                files[_rel(cache)] = cache
    if s["pmt_meshes"] and version.inputs.get("pmt_table") is not None:
        for p in sorted(MESH_DIR.glob("*")):
            if p.is_file():
                files[_rel(p)] = p
    for p in version.overlays:
        files[_rel(p)] = p
    return files


def file_hashes(version: GeometryVersion) -> dict[str, str]:
    hashes = {}
    for key, p in input_files(version).items():
        if not p.exists():
            raise GeometryError(f"Geometry input not found: {p}")
        hashes[key] = _sha256(p)
    return hashes


def fingerprint(version: GeometryVersion, hashes: Optional[dict[str, str]] = None) -> str:
    """SHA-256 of what the build depends on: settings, offsets, file contents.

    Names and paths are left out, so the same detector built from different
    places gets the same fingerprint.
    """
    if hashes is None:
        hashes = file_hashes(version)
    offsets = load_pmt_offsets(version.overlays)
    content = {
        "settings": version.settings,
        "pmt_offsets": {str(k): list(v) for k, v in offsets.items()},
        "inputs": {role: hashes[_rel(version.inputs[role])]
                   for role in INPUT_ROLES
                   if version.inputs.get(role) is not None and _rel(version.inputs[role]) in hashes},
        "files": sorted(hashes.values()),
    }
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def verify_tagged(version: GeometryVersion, hashes: dict[str, str], fp: str) -> None:
    """Raise if a tagged version no longer matches what was tagged."""
    if fp == version.tagged.get("fingerprint"):
        return
    recorded = version.tagged.get("files", {})
    changed = sorted(k for k in set(recorded) | set(hashes) if recorded.get(k) != hashes.get(k))
    detail = f"changed files: {', '.join(changed)}" if changed else "settings or overlays changed"
    raise GeometryError(
        f"Tagged geometry '{version.name}' has changed since it was tagged ({detail}). "
        "Make a new version instead of editing a tagged one."
    )


def tag_version(name_or_path: str | Path, date: Optional[str] = None) -> str:
    """Tag a version: record its file hashes and fingerprint, mark it tagged."""
    version = load_version(name_or_path)
    if version.status == STATUS_TAGGED:
        raise GeometryError(f"'{version.name}' is already tagged")
    hashes = file_hashes(version)
    fp = fingerprint(version, hashes)

    with open(version.path) as f:
        doc = yaml.safe_load(f)
    doc["status"] = STATUS_TAGGED
    doc["tagged"] = {
        "date": date or datetime.date.today().isoformat(),
        "fingerprint": fp,
        "files": hashes,
    }
    with open(version.path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=88)
    return fp


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def load_build(version: GeometryVersion | str | Path, quiet: bool = False) -> Build:
    """Build the detector for *version* (a name, a file, or a GeometryVersion)."""
    from litepath.geometry.build import build_geometry

    if not isinstance(version, GeometryVersion):
        version = load_version(version)
    version = copy.deepcopy(version)

    hashes = file_hashes(version)
    fp = fingerprint(version, hashes)
    if version.status == STATUS_TAGGED:
        verify_tagged(version, hashes, fp)
    elif version.status == STATUS_DRAFT and not quiet:
        print(f"  Warning: geometry '{version.name}' is a draft; results record "
              f"fingerprint {fp[:12]} but the version may still change")

    s = version.settings
    offsets = load_pmt_offsets(version.overlays)
    structure = version.inputs.get("structure") or Path("PHASE2_INNER_STRUCTURE_closed.gdml")
    if not quiet:
        print(f"Geometry: {version.name} ({version.status}, fingerprint {fp[:12]})")
    geometry = build_geometry(
        structure,
        step_path=version.inputs.get("step"),
        manifest_path=version.inputs.get("manifest"),
        pmt_csv_path=version.inputs.get("pmt_table"),
        lappd_indices=s["lappd_indices"],
        no_lappd=not s["lappds"],
        no_gdml=not s["structure_mesh"],
        no_pmt_holders=not s["pmt_meshes"],
        z_offset=s["z_offset_mm"],
        lappd_model=s["lappd_model"],
        bottom_rotation_deg=s["bottom_rotation_deg"],
        bottom_spin_deg=s["bottom_spin_deg"],
        det_rotation_deg=s["det_rotation_deg"],
        n_surfboards=s["surfboards"],
        pmt_offsets=offsets,
    )
    return Build(geometry=geometry, version=version, fingerprint=fp, pmt_offsets=offsets)


def rotate_offsets_xy(offsets: dict[int, tuple[float, float, float]], angle_deg: float
                      ) -> dict[int, tuple[float, float, float]]:
    """Rotate (dx, dy) of each offset by *angle_deg* about Z (dz unchanged)."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return {k: (dx * c - dy * s, dx * s + dy * c, dz) for k, (dx, dy, dz) in offsets.items()}
