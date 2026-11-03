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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_DIR = REPO_ROOT / "geometry"
VERSIONS_DIR = GEOMETRY_DIR / "versions"

DEFAULT_GEOMETRY = "annie-2026-data"

STATUS_TAGGED = "tagged"
STATUS_DRAFT = "draft"
STATUS_UNNAMED = "unnamed"

# Build settings and their defaults (those of build_geometry()).
DEFAULT_SETTINGS = {
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

# Orientation settings that existed before the frame was fixed
# (litepath.geometry.frame), with the only value they ever had in use.
# The old flags (--det-rotation, --bottom-rot) still accept that value.
REMOVED_SETTINGS = {"det_rotation_deg": 22.5, "bottom_rotation_deg": 45.0}
REMOVED_FLAGS = {"det_rotation": "det_rotation_deg", "bottom_rot": "bottom_rotation_deg"}


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
    lappd_offsets: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    surfboard_offsets: dict[int, tuple[float, float, float]] = field(default_factory=dict)


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

    removed = set(doc.get("settings", {})) & set(REMOVED_SETTINGS)
    if removed:
        raise GeometryError(
            f"{path}: settings {sorted(removed)} were removed: the detector orientation is "
            "fixed (see litepath/geometry/frame.py). Delete them from the file.")
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
    for flag, setting in REMOVED_FLAGS.items():
        if flag in v and float(v[flag]) != REMOVED_SETTINGS[setting]:
            raise GeometryError(
                f"--{flag.replace('_', '-')} {v[flag]} is not supported: the detector orientation "
                f"is fixed (see litepath/geometry/frame.py). Drop the flag.")
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

def _format_offset(value: float) -> str:
    return "0" if abs(value) < 5e-7 else f"{value:.6f}".rstrip("0").rstrip(".")


# An ``element-offsets`` overlay holds one total offset (mm, LitePath frame)
# per element, relative to the element's base position in the version:
#   pmts:       channel key -> [dx, dy, dz]
#   lappds:     LAPPD detector key (2000+ housed) -> [dx, dy, dz]
#   surfboards: surfboard index (0, 1, 2 at 45, 90, 135 deg) -> [dx, dy, dz]
#   history:    one entry per save (date, which elements changed)
# The older ``pmt-offsets`` kind (PMTs only, under ``offsets``) is still read.
OFFSET_SECTIONS = ("pmts", "lappds", "surfboards")
_SECTION_COMMENTS = {
    "pmts": "channel key: [dx, dy, dz]",
    "lappds": "LAPPD detector key: [dx, dy, dz]",
    "surfboards": "surfboard index (0, 1, 2 at 45, 90, 135 deg): [dx, dy, dz]",
}
Offsets = dict[int, tuple[float, float, float]]


def _read_overlay(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise GeometryError(f"Overlay not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _overlay_sections(doc: dict, path: Path) -> dict[str, dict]:
    kind = doc.get("kind")
    if kind == "pmt-offsets":
        return {"pmts": doc.get("offsets") or {}}
    if kind == "element-offsets":
        return {s: doc.get(s) or {} for s in OFFSET_SECTIONS}
    raise GeometryError(f"{path}: unsupported overlay kind {kind!r}")


def load_offsets(overlays: list[Path]) -> dict[str, Offsets]:
    """Sum the offset overlays into one offset (mm) per element, by section."""
    total: dict[str, dict[int, list[float]]] = {s: {} for s in OFFSET_SECTIONS}
    for path in overlays:
        for section, entries in _overlay_sections(_read_overlay(path), path).items():
            for key, offset in entries.items():
                if len(offset) != 3:
                    raise GeometryError(f"{path}: offset for {section} {key} needs 3 values")
                acc = total[section].setdefault(int(key), [0.0, 0.0, 0.0])
                for i in range(3):
                    acc[i] += float(offset[i])
    return {s: {k: tuple(v) for k, v in sorted(d.items()) if any(v)} for s, d in total.items()}


def load_pmt_offsets(overlays: list[Path]) -> Offsets:
    """Sum the overlays' PMT offsets into one offset (mm) per channel key."""
    return load_offsets(overlays)["pmts"]


def read_offsets(path: Path) -> dict[str, Offsets]:
    """Offsets of one overlay file, by section."""
    return load_offsets([Path(path)])


def read_pmt_offsets(path: Path) -> Offsets:
    """PMT offsets of one overlay file."""
    return read_offsets(path)["pmts"]


def _offset_lines(name: str, offsets: Offsets) -> list[str]:
    rows = [f"  {key}: [{', '.join(_format_offset(v) for v in offsets[key])}]"
            for key in sorted(offsets) if any(abs(v) >= 5e-7 for v in offsets[key])]
    return [f"{name}:   # {_SECTION_COMMENTS.get(name, 'key: [dx, dy, dz]')}", *rows] if rows \
        else [f"{name}: {{}}"]


def write_offsets(path: Path, offsets: dict[str, Offsets], header: Optional[dict] = None,
                  history_entry: Optional[dict] = None) -> None:
    """Write an ``element-offsets`` overlay (one line per element; zeros dropped).

    *header* defaults to the existing file's descriptive fields; the history
    is kept and *history_entry*, if given, is appended to it.
    """
    old = _read_overlay(path) if Path(path).exists() else {}
    if header is None:
        header = {k: v for k, v in old.items()
                  if k not in ("kind", "offsets", "history", *OFFSET_SECTIONS)}
    history = list(old.get("history") or [])
    if history_entry:
        history.append(history_entry)
    lines = [yaml.safe_dump({"kind": "element-offsets", **header}, sort_keys=False, width=88).rstrip()]
    for section in OFFSET_SECTIONS:
        lines += _offset_lines(section, offsets.get(section, {}))
    if history:
        lines.append(yaml.safe_dump({"history": history}, sort_keys=False, width=88,
                                    default_flow_style=None).rstrip())
    Path(path).write_text("\n".join(lines) + "\n")


def write_pmt_offsets(path: Path, offsets: Offsets, header: Optional[dict] = None) -> None:
    """Replace the PMT offsets of an overlay, keeping its other sections.

    A new file or an older ``pmt-offsets`` file is written in that format;
    an ``element-offsets`` file keeps its LAPPD and surfboard offsets and
    history.  *header* defaults to the existing file's fields.  Zero offsets
    are dropped.
    """
    if Path(path).exists() and _read_overlay(path).get("kind") == "element-offsets":
        write_offsets(path, {**read_offsets(path), "pmts": offsets}, header)
        return
    if header is None:
        with open(path) as f:
            header = {k: v for k, v in (yaml.safe_load(f) or {}).items() if k != "offsets"}
    header = {"kind": "pmt-offsets", **header}
    lines = [yaml.safe_dump(header, sort_keys=False, width=88).rstrip(),
             *_offset_lines("offsets", offsets)]
    lines[1] = lines[1].replace("# key:", "# channel key:")
    Path(path).write_text("\n".join(lines) + "\n")


def editable_overlay(version: GeometryVersion) -> Optional[Path]:
    """The overlay that adjustments are saved to, or None if not editable.

    Only draft versions are editable; the last offsets overlay receives the
    changes.
    """
    if version.status != STATUS_DRAFT:
        return None
    for path in reversed(version.overlays):
        if _read_overlay(path).get("kind") in ("pmt-offsets", "element-offsets"):
            return path
    return None


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
    offsets = load_offsets(version.overlays)
    content = {
        "settings": version.settings,
        "pmt_offsets": {str(k): list(v) for k, v in offsets["pmts"].items()},
        "inputs": {role: hashes[_rel(version.inputs[role])]
                   for role in INPUT_ROLES
                   if version.inputs.get(role) is not None and _rel(version.inputs[role]) in hashes},
        "files": sorted(hashes.values()),
    }
    # Only when present, so versions without them keep their fingerprints
    for section in ("lappds", "surfboards"):
        if offsets[section]:
            content[f"{section[:-1]}_offsets"] = {str(k): list(v) for k, v in offsets[section].items()}
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
    offsets = load_offsets(version.overlays)
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
        bottom_spin_deg=s["bottom_spin_deg"],
        n_surfboards=s["surfboards"],
        pmt_offsets=offsets["pmts"],
        lappd_offsets=offsets["lappds"],
        surfboard_offsets=offsets["surfboards"],
    )
    return Build(geometry=geometry, version=version, fingerprint=fp, pmt_offsets=offsets["pmts"],
                 lappd_offsets=offsets["lappds"], surfboard_offsets=offsets["surfboards"])


# ---------------------------------------------------------------------------
# Provenance for output files
# ---------------------------------------------------------------------------

STATUS_BUILT_IN = "built-in"
STANDALONE_VERSION = "lappd-standalone"


def git_commit() -> str:
    """LitePath's git commit, with ``-dirty`` if tracked files changed; ``unknown`` without git."""
    import subprocess

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def build_provenance(build: Build) -> dict:
    """Root attributes recording which detector a run simulated (schema v2)."""
    return version_provenance(build.version, build.fingerprint)


def version_provenance(version: GeometryVersion, fp: Optional[str] = None) -> dict:
    """Root attributes for *version* (fingerprint computed unless given)."""
    table = version.inputs.get("pmt_table")
    return {
        "geometry_version": version.name,
        "geometry_status": version.status,
        "geometry_hash": fp or fingerprint(version),
        "geometry_pmt_table": _rel(table) if table else "",
        "geometry_pmt_table_sha256": _sha256(table) if table and table.exists() else "",
        "litepath_commit": git_commit(),
    }


def standalone_fingerprint(geometry) -> str:
    """Fingerprint of the built-in standalone LAPPD module.

    Hashes the built housing and photocathode (positions rounded to 1 µm),
    so any change to the module model changes it.
    """
    import numpy as np
    blob = {
        "name": STANDALONE_VERSION,
        "housing": np.round(np.asarray(geometry.lappd_housing_data, dtype=np.float64), 3).tolist(),
        "lappd": np.round(np.asarray(geometry.annie_lappd_data, dtype=np.float64), 3).tolist(),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def standalone_provenance(geometry) -> dict:
    """Root attributes for ``batch-lappd`` output (the built-in module)."""
    return {
        "geometry_version": STANDALONE_VERSION,
        "geometry_status": STATUS_BUILT_IN,
        "geometry_hash": standalone_fingerprint(geometry),
        "geometry_pmt_table": "",
        "geometry_pmt_table_sha256": "",
        "litepath_commit": git_commit(),
    }
