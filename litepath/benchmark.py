"""Benchmarking framework: WAGLE ROOT trees → litepath batch HDF5 schema.

This module provides

* a reader for the WAGLE ``Event`` trees shipped in ``benchmarking_data/``
  (``ANNIETree_ThroughgoingMuons.root`` = data,
  ``MCTree_ThroughgoingMuons.root`` = Monte Carlo with truth tracks),
* the coordinate-frame transform between the WAGLE frame and the litepath
  world frame,
* a writer that re-exports the parsed events in the *same* HDF5 schema
  that ``litepath batch`` produces, so data, MC and raytracer output can
  be analysed with a single code path.

Frames
------
``root`` (WAGLE)
    ``x`` horizontal transverse, ``y`` vertical (up), ``z`` along the beam
    (downstream).  Origin at the centre of the water tank.  Positions are
    in metres, times in ns, charges in PE (``hitQ`` in pC).

``world`` (litepath)
    The GDML *structure* frame: ``z`` vertical (up), ``x``/``y`` in the
    horizontal plane, the tank centred at ``z = WCLength / 2``.  This is
    the frame the ray tracer works in; the PMT positions are derived from
    ``PMTPositions_Scan.txt`` (see :mod:`litepath.pmt_loader`).

The transform is::

    world = Rz(det_rotation) · ( Rz(-157.5°) · Rx(+90°) · root + (0, 0, 1980) )

It is *not* fitted to the WAGLE data: the constants come from the litepath
geometry chain (scan frame → WCSim hall frame → structure frame, see
:mod:`litepath.pmt_loader`), and the WAGLE data is used to *check* it.
Two independent checks pass:

* the hall ↔ ROOT relation is a pure *y/z swap* (``Rx(-90°)``) — a Procrustes
  fit of the 121 hit PMT point clouds returns that swap, and the free
  rotation gives the same residual as the constrained one, so there is no
  hidden azimuthal rotation in the chain;
* the transformed WAGLE PMT positions agree with the raytracer PMT centres
  to < 0.01 mm radially and < 0.09° in azimuth.  The residual is a constant
  *axial* offset per PMT type (-10.43 mm for the LUX/Watchboy/Watchman 8",
  -10.55 mm for the Hamamatsu 8", -23.70 mm for the ETEL 10"), i.e. WCSim
  places the PMT centre that much further *outward* than the survey point.

Note that in the WAGLE trees ``hitX/hitY/hitZ`` is *not* a per-photon
position: raw WCSim stores the PMT geometry separately and the hit position
is looked up by Tube ID, so every hit of a given PMT carries the same
point.  The ``pmt_hits`` table therefore holds PMT centres, not photocathode
landing points.

In the litepath world frame the WAGLE beam runs at ``phi = 135°`` (an
octagon corner of the inner structure); the ``--muon-mode beam`` helper in
:mod:`litepath.cli` uses ``+Y`` (90°), which is 45° away from it.

Units used in the ROOT trees (verified against the geometry)
------------------------------------------------------------
=========================  ==========================================
``hitT``                   ns, relative to the event reference time
``hitPE``                  photoelectrons
``hitQ``                   pC (Monte Carlo tree only)
``hitX/Y/Z``               m, ROOT frame
``trueVtxX/Y/Z``           mm, ROOT frame
``trueAngle``              rad, angle between the track and ``+z``
``truePhi``                rad, azimuth in the ``x``/``y`` plane
``trueTrackLengthInWater`` m
``trueTrackLengthInMRD``   cm
``MRDTrackLength``         cm  (the start/stop coordinates are in m)
``MRDTrackAngle``          rad
=========================  ==========================================
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from litepath.io_h5 import (
    DETECTOR_DTYPE,
    MUON_TRUTH_DTYPE,
    PMT_RESPONSE_DTYPE,
    SOURCE_TRACK,
)

__all__ = [
    "PMTRegistry",
    "TruthTrack",
    "BenchmarkEvent",
    "RootTreeReader",
    "load_pmt_registry",
    "root_to_world",
    "world_to_root",
    "root_to_world_direction",
    "world_to_root_direction",
    "write_benchmark_h5",
    "ROOT_BRANCHES",
    "MC_BRANCHES",
    "PMT_HIT_DTYPE",
    "BENCH_TRUTH_DTYPE",
    "TRUTH_KIND_TRUE",
    "TRUTH_KIND_MRD_RECO",
]


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------

#: Rotation that undoes the WCSim placement of the inner structure.
STRUCTURE_ROTATION_DEG = -157.5

#: Global Z-rotation applied to the detector registry (litepath default).
DET_ROTATION_DEG = 22.5

#: Half the water-tank length: the structure frame origin sits this far
#: below the tank centre (WCLength = 3960 mm).
HALF_WC_LENGTH = 1980.0

#: Water-tank geometry (mm) — identical to ``litepath.geometry.build.build_geometry``.
TANK_RADIUS_MM = 1524.0
TANK_Z_MIN_MM = 19.0
TANK_Z_MAX_MM = 3861.0

#: The tank is a vertical cylinder, so in the ROOT frame its axis is ``y``
#: (``root_to_world`` maps ``world_z = root_y + HALF_WC_LENGTH``) and its
#: cross-section is the ``(x, z)`` plane (``world_r² = root_x² + root_z²``).
#: These are the axial bounds of the water volume expressed as ROOT ``y``.
ROOT_TANK_AXIS_Y_MIN = TANK_Z_MIN_MM - HALF_WC_LENGTH
ROOT_TANK_AXIS_Y_MAX = TANK_Z_MAX_MM - HALF_WC_LENGTH


def _tank_chord_root(start_mm: np.ndarray, direction: np.ndarray
                     ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Intersect a line with the water tank in the ROOT frame.

    Parameters
    ----------
    start_mm : (3,) array
        Any point on the line, in ROOT-frame millimetres.
    direction : (3,) array
        Line direction (need not be unit length).

    Returns
    -------
    (entry, exit) or ``None``
        The two points where the line enters and leaves the tank cylinder
        (axis ``y``, radius :data:`TANK_RADIUS_MM`, axial bounds
        ``ROOT_TANK_AXIS_Y_MIN…MAX``).  ``None`` when the line misses the tank.
    """
    p = np.asarray(start_mm, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm == 0 or not np.isfinite(norm):
        return None
    d = d / norm

    lo, hi = -np.inf, np.inf

    # Axial extent: y in [y_min, y_max].
    if abs(d[1]) > 1e-12:
        t0 = (ROOT_TANK_AXIS_Y_MIN - p[1]) / d[1]
        t1 = (ROOT_TANK_AXIS_Y_MAX - p[1]) / d[1]
        lo = max(lo, min(t0, t1))
        hi = min(hi, max(t0, t1))
    elif not (ROOT_TANK_AXIS_Y_MIN <= p[1] <= ROOT_TANK_AXIS_Y_MAX):
        return None

    # Radial extent: (x + t·dx)² + (z + t·dz)² ≤ R².
    a = d[0] ** 2 + d[2] ** 2
    b = 2.0 * (p[0] * d[0] + p[2] * d[2])
    c = p[0] ** 2 + p[2] ** 2 - TANK_RADIUS_MM ** 2
    if a < 1e-18:
        if c > 0:
            return None
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        lo = max(lo, (-b - sq) / (2.0 * a))
        hi = min(hi, (-b + sq) / (2.0 * a))

    if lo > hi or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return p + lo * d, p + hi * d


def _rot_z(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an (..., 3) array about the z axis."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    x = points[..., 0]
    y = points[..., 1]
    return np.stack([x * c - y * s, x * s + y * c, points[..., 2]], axis=-1)


def _rot_x(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an (..., 3) array about the x axis."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    y = points[..., 1]
    z = points[..., 2]
    return np.stack([points[..., 0], y * c - z * s, y * s + z * c], axis=-1)


def root_to_world(points: np.ndarray,
                  det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
    """Transform ROOT-frame positions (mm) into the litepath world frame (mm).

    Parameters
    ----------
    points : array_like
        (..., 3) positions in the ROOT frame, in millimetres.
    det_rotation_deg : float
        Global Z-rotation used by the raytracer (``--det-rotation``).

    Returns
    -------
    numpy.ndarray
        (..., 3) positions in the litepath world frame, in millimetres.
    """
    p = np.asarray(points, dtype=np.float64)
    p = _rot_x(p, 90.0)
    p = _rot_z(p, STRUCTURE_ROTATION_DEG)
    p = p + np.array([0.0, 0.0, HALF_WC_LENGTH])
    p = _rot_z(p, det_rotation_deg)
    return p


def world_to_root(points: np.ndarray,
                  det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
    """Inverse of :func:`root_to_world` (positions only)."""
    p = np.asarray(points, dtype=np.float64)
    p = _rot_z(p, -det_rotation_deg)
    p = p - np.array([0.0, 0.0, HALF_WC_LENGTH])
    p = _rot_z(p, -STRUCTURE_ROTATION_DEG)
    p = _rot_x(p, -90.0)
    return p


def root_to_world_direction(directions: np.ndarray,
                            det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
    """Transform ROOT-frame direction vectors into the litepath world frame."""
    d = np.asarray(directions, dtype=np.float64)
    d = _rot_x(d, 90.0)
    d = _rot_z(d, STRUCTURE_ROTATION_DEG)
    d = _rot_z(d, det_rotation_deg)
    return d


def world_to_root_direction(directions: np.ndarray,
                            det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
    """Inverse of :func:`root_to_world_direction`."""
    d = np.asarray(directions, dtype=np.float64)
    d = _rot_z(d, -det_rotation_deg)
    d = _rot_z(d, -STRUCTURE_ROTATION_DEG)
    d = _rot_x(d, -90.0)
    return d


def litepath_angles(direction: np.ndarray) -> tuple[float, float]:
    """Return ``(theta_deg, phi_deg)`` in the litepath convention.

    ``theta`` is the polar angle measured from *downward* (-z):
    0° = straight down, 90° = horizontal, 180° = straight up.
    ``phi`` is the azimuth ``atan2(dy, dx)`` in the horizontal plane,
    wrapped to [0, 360) to match the fit grid.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    theta = math.degrees(math.acos(float(np.clip(-d[2], -1.0, 1.0))))
    phi = math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360.0
    return theta, phi


def root_angles(direction: np.ndarray) -> tuple[float, float]:
    """Return ``(theta_deg, phi_deg)`` in the WAGLE convention.

    ``theta`` is measured from ``+z`` (the beam axis) and ``phi`` is the
    azimuth ``atan2(dy, dx)``.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    theta = math.degrees(math.acos(float(np.clip(d[2], -1.0, 1.0))))
    phi = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return theta, phi


# ---------------------------------------------------------------------------
# Extension tables (beyond the standard litepath batch schema)
# ---------------------------------------------------------------------------

#: Raw digitised PMT hits (one row per ROOT hit).
PMT_HIT_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("pmt_index", "i4"),      # litepath detector index (0..131)
    ("detector_id", "i4"),    # WCSim TubeID (332..463)
    ("arrival_time", "f4"),   # ns
    ("charge_pe", "f4"),      # photoelectrons
    ("charge_pc", "f4"),      # pC (NaN when the tree has no hitQ)
    ("x", "f4"),              # mm, world frame (WAGLE hit position = PMT centre)
    ("y", "f4"),
    ("z", "f4"),
])

#: Per-event track truth / reconstruction, in both frames.
BENCH_TRUTH_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("track_kind", "i4"),       # 0 = true track, 1 = MRD reconstruction
    ("pos_x", "f4"),            # world frame (mm)
    ("pos_y", "f4"),
    ("pos_z", "f4"),
    ("dir_x", "f4"),
    ("dir_y", "f4"),
    ("dir_z", "f4"),
    ("theta_deg", "f4"),        # litepath convention (0 = down)
    ("phi_deg", "f4"),
    ("root_pos_x", "f4"),       # ROOT frame (mm)
    ("root_pos_y", "f4"),
    ("root_pos_z", "f4"),
    ("root_dir_x", "f4"),
    ("root_dir_y", "f4"),
    ("root_dir_z", "f4"),
    ("root_theta_deg", "f4"),   # WAGLE convention (0 = +z, the beam axis)
    ("root_phi_deg", "f4"),
    ("energy_mev", "f4"),       # true muon energy (NaN for reco tracks)
    ("pdg", "i4"),
    ("track_length_water_mm", "f4"),
    ("track_length_mrd_mm", "f4"),
    ("mrd_start_x", "f4"),      # ROOT frame (mm)
    ("mrd_start_y", "f4"),
    ("mrd_start_z", "f4"),
    ("mrd_stop_x", "f4"),
    ("mrd_stop_y", "f4"),
    ("mrd_stop_z", "f4"),
    ("mrd_angle_rad", "f4"),
    ("mrd_length_mm", "f4"),
    ("n_hits_total", "i4"),     # digitised PMT hits in the event
    ("n_pmt_hit", "i4"),        # PMTs with at least one hit
    ("pe_total", "f4"),         # summed photoelectrons in the event
])

#: Track provenance of a row in ``BENCH_TRUTH_DTYPE``.
TRUTH_KIND_TRUE = 0        # generator truth (Monte Carlo tree)
TRUTH_KIND_MRD_RECO = 1    # MRD reconstruction (data tree, and MC as cross-check)


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------


@dataclass
class PMTRegistry:
    """PMT lookup shared by the ROOT trees and the litepath geometry.

    ``tube_id`` are the WCSim TubeIDs (332-463) used by the ROOT
    ``hitDetID`` branch; ``index`` is the position of the PMT in the
    litepath geometry arrays (i.e. the ``detector_index`` of the standard
    ``pmt_responses`` / ``photon_hits`` tables).
    """

    tube_id: np.ndarray        # (N,) int
    index: np.ndarray          # (N,) int  (== arange(N))
    positions: np.ndarray      # (N, 3) float, world frame (mm)
    directions: np.ndarray     # (N, 3) float, inward unit normals
    panels: np.ndarray         # (N,) int
    types: list[str]
    radii: np.ndarray          # (N,) float (mm)
    det_rotation_deg: float = DET_ROTATION_DEG

    def __post_init__(self) -> None:
        self._by_tube = {int(t): i for i, t in enumerate(self.tube_id)}

    def __len__(self) -> int:
        return len(self.tube_id)

    def index_of(self, tube_id: int) -> int:
        """Return the litepath detector index for a WCSim TubeID."""
        try:
            return self._by_tube[int(tube_id)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown PMT TubeID {tube_id}") from exc

    def index_array(self, tube_ids: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`index_of` (unknown IDs map to -1)."""
        lut = np.full(int(self.tube_id.max()) + 1, -1, dtype=np.int32)
        lut[self.tube_id.astype(np.int64)] = self.index.astype(np.int32)
        ids = np.asarray(tube_ids, dtype=np.int64)
        out = np.full(ids.shape, -1, dtype=np.int32)
        ok = (ids >= 0) & (ids < len(lut))
        out[ok] = lut[ids[ok]]
        return out

    def detectors_table(self) -> np.ndarray:
        """Return the standard ``detectors`` table rows for these PMTs."""
        arr = np.zeros(len(self.tube_id), dtype=DETECTOR_DTYPE)
        arr["system_code"] = 0        # DET_SYS_PMT
        arr["detector_index"] = self.index.astype(np.int32)
        arr["x"] = self.positions[:, 0]
        arr["y"] = self.positions[:, 1]
        arr["z"] = self.positions[:, 2]
        arr["label"] = [f"PMT_{int(t)}" for t in self.tube_id]
        arr["panel"] = self.panels.astype(np.int32)
        return arr


def load_pmt_registry(scan_path: Path | str,
                      det_rotation_deg: float = DET_ROTATION_DEG) -> PMTRegistry:
    """Build the :class:`PMTRegistry` from the WCSim scan file.

    The scan file order defines the litepath detector indices, so the
    registry is consistent with ``litepath.pmt_loader.load_pmts``.
    """
    from litepath.pmt_loader import load_pmts

    pmt = load_pmts(Path(scan_path), det_rotation_deg=det_rotation_deg)
    n = len(pmt["detector_nums"])
    return PMTRegistry(
        tube_id=np.asarray(pmt["detector_nums"], dtype=np.int32),
        index=np.arange(n, dtype=np.int32),
        positions=np.asarray(pmt["centers"], dtype=np.float64),
        directions=np.asarray(pmt["directions"], dtype=np.float64),
        panels=np.asarray(pmt["panels"], dtype=np.int32),
        types=list(pmt["types"]),
        radii=np.asarray(pmt["radii"], dtype=np.float32),
        det_rotation_deg=det_rotation_deg,
    )


# ---------------------------------------------------------------------------
# Event containers
# ---------------------------------------------------------------------------


@dataclass
class TruthTrack:
    """A muon track: either generator truth or an MRD reconstruction."""

    pos_root_mm: np.ndarray        # (3,) vertex in the ROOT frame (mm)
    dir_root: np.ndarray           # (3,) unit direction in the ROOT frame
    kind: int = TRUTH_KIND_TRUE

    energy_mev: float = float("nan")
    pdg: int = 0
    track_length_water_m: float = float("nan")
    track_length_mrd_cm: float = float("nan")

    mrd_start_root_mm: Optional[np.ndarray] = None   # (3,)
    mrd_stop_root_mm: Optional[np.ndarray] = None    # (3,)
    mrd_angle_rad: float = float("nan")
    mrd_length_cm: float = float("nan")

    def pos_world_mm(self, det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
        """Vertex in the litepath world frame (mm)."""
        return root_to_world(self.pos_root_mm, det_rotation_deg)

    def dir_world(self, det_rotation_deg: float = DET_ROTATION_DEG) -> np.ndarray:
        """Unit direction in the litepath world frame."""
        return root_to_world_direction(self.dir_root, det_rotation_deg)

    def world_angles(self, det_rotation_deg: float = DET_ROTATION_DEG
                     ) -> tuple[float, float]:
        """``(theta_deg, phi_deg)`` in the litepath convention."""
        return litepath_angles(self.dir_world(det_rotation_deg))

    def root_angles(self) -> tuple[float, float]:
        """``(theta_deg, phi_deg)`` in the WAGLE convention."""
        return root_angles(self.dir_root)


@dataclass
class BenchmarkEvent:
    """One parsed WAGLE event."""

    event_id: int
    run_number: int = 0
    event_number: int = 0

    # Digitised PMT hits
    tube_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    pe: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    charge_pc: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    positions_root_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Cluster summary (may be empty)
    cluster_time: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    cluster_pe: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    cluster_charge: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    cluster_max_pe: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    cluster_n_hits: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    cluster_charge_balance: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))

    truth: Optional[TruthTrack] = None
    event_time_tank: int = 0
    event_time_mrd: int = 0
    nhits: int = 0
    n_clusters: int = 0

    @property
    def has_truth(self) -> bool:
        return self.truth is not None


# ---------------------------------------------------------------------------
# ROOT tree reader
# ---------------------------------------------------------------------------

#: Branches present in both trees.
ROOT_BRANCHES = [
    "runNumber", "eventNumber", "nhits", "numberOfClusters",
    "eventTimeTank", "eventTimeMRD",
    "hitType", "hitDetID", "hitChankey", "hitT", "hitPE", "hitX", "hitY", "hitZ",
    "clusterTime", "clusterCharge", "clusterPE", "clusterMaxPE", "clusterHits",
    "clusterChargeBalance",
    "MRDTrackStartX", "MRDTrackStartY", "MRDTrackStartZ",
    "MRDTrackStopX", "MRDTrackStopY", "MRDTrackStopZ",
    "MRDTrackAngle", "MRDTrackLength", "MRDPenetrationDepth",
]

#: Extra branches of the Monte Carlo tree (generator truth).
MC_BRANCHES = [
    "trueVtxX", "trueVtxY", "trueVtxZ", "trueVtxTime",
    "trueDirX", "trueDirY", "trueDirZ", "trueAngle", "truePhi",
    "trueMuonEnergy", "truePrimaryPdg",
    "trueTrackLengthInWater", "trueTrackLengthInMRD",
    "hitQ",
]


def _require_uproot():
    """Import uproot with a helpful error message."""
    try:
        import uproot  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading ROOT files requires uproot.  Install it with "
            "`pip install uproot` (or `pip install -e .[benchmark]`)."
        ) from exc
    return uproot


def _flat_values(event_array) -> np.ndarray:
    """Flatten one event's branch value into a 1-D numpy array.

    Handles both regular (one value per event) and jagged (one list per
    event) branches.
    """
    import awkward as ak

    if isinstance(event_array, (np.ndarray, np.generic)):
        return np.atleast_1d(np.asarray(event_array)).reshape(-1)
    arr = ak.Array(event_array)
    if arr.ndim <= 1:
        return np.asarray(ak.to_numpy(arr)).reshape(-1)
    return np.asarray(ak.to_numpy(ak.flatten(arr))).reshape(-1)


def _np(event_array) -> np.ndarray:
    """Convert a per-event awkward array to a flat float64 array."""
    values = _flat_values(event_array)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    return values.astype(np.float64)


def _np_int(event_array) -> np.ndarray:
    values = _flat_values(event_array)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    return values.astype(np.int64)


def _scalar(value) -> float:
    """Convert an awkward scalar (possibly a length-1 list) to float."""
    values = _flat_values(value)
    if values.size == 0:
        return float("nan")
    return float(values.reshape(-1)[0])


class RootTreeReader:
    """Reader for a WAGLE ``Event`` tree.

    Parameters
    ----------
    path : Path | str
        ROOT file to read.
    tree : str
        Name of the tree (``"Event"`` for both benchmark files).
    max_events : int
        Optional cap on the number of events to yield.

    Examples
    --------
    >>> reader = RootTreeReader("benchmarking_data/MCTree_ThroughgoingMuons.root")
    >>> reader.has_truth
    True
    >>> ev = next(reader.iter_events())
    >>> ev.truth.dir_root
    """

    def __init__(self, path: Path | str, tree: str = "Event",
                 max_events: Optional[int] = None):
        self.path = Path(path)
        self.tree_name = tree
        self.max_events = max_events

        uproot = _require_uproot()
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._file = uproot.open(self.path)
        # ROOT cycle suffixes ("Event;1") are stripped for the lookup.
        names = {k.split(";")[0]: k for k in self._file.keys()}
        if tree not in names:
            raise KeyError(
                f"{self.path} has no tree '{tree}' (found {list(self._file.keys())})"
            )
        self._tree = self._file[names[tree]]
        self._available = set(self._tree.keys())
        self._branches = [b for b in ROOT_BRANCHES if b in self._available]
        self._has_truth = "trueVtxX" in self._available
        if self._has_truth:
            self._branches += [b for b in MC_BRANCHES if b in self._available]

    # -- metadata -----------------------------------------------------------

    @property
    def has_truth(self) -> bool:
        """True when the tree carries generator truth (Monte Carlo tree)."""
        return self._has_truth

    @property
    def num_entries(self) -> int:
        return self._tree.num_entries

    def branches(self) -> list[str]:
        """Branches that will be read."""
        return list(self._branches)

    def summary(self) -> dict:
        """File-level metadata and simple branch statistics."""
        return {
            "path": str(self.path),
            "tree": self.tree_name,
            "n_entries": self.num_entries,
            "has_truth": self.has_truth,
            "branches": self.branches(),
        }

    # -- iteration ----------------------------------------------------------

    def iter_events(self, step_size: str = "100 MB") -> Iterator[BenchmarkEvent]:
        """Yield :class:`BenchmarkEvent` objects one at a time.

        Parameters
        ----------
        step_size : str
            uproot chunk size (e.g. ``"100 MB"`` or ``"5000"`` entries).
        """
        yielded = 0
        for chunk in self._tree.iterate(expressions=self._branches,
                                        step_size=step_size):
            for i in range(len(chunk["runNumber"])):
                if self.max_events is not None and yielded >= self.max_events:
                    return
                yield self._make_event(chunk, i, event_id=yielded)
                yielded += 1

    def _make_event(self, chunk: dict, i: int, event_id: int = 0) -> BenchmarkEvent:
        hit_type = _np_int(chunk["hitType"][i])
        keep = (hit_type == 0) if hit_type.size else np.zeros(0, dtype=bool)
        # hitType == 0 marks PMT hits; anything else is not a PMT.

        tube_ids = _np_int(chunk["hitDetID"][i])[keep].astype(np.int32)
        times = _np(chunk["hitT"][i])[keep]
        pe = _np(chunk["hitPE"][i])[keep]
        if "hitQ" in chunk.fields:
            charge = _np(chunk["hitQ"][i])[keep]
        else:
            charge = np.full(times.shape, np.nan)
        pos = np.stack([
            _np(chunk["hitX"][i])[keep],
            _np(chunk["hitY"][i])[keep],
            _np(chunk["hitZ"][i])[keep],
        ], axis=1) if times.size else np.zeros((0, 3))

        ev = BenchmarkEvent(
            event_id=event_id,
            run_number=int(chunk["runNumber"][i]),
            event_number=int(chunk["eventNumber"][i]),
            tube_ids=tube_ids,
            times=times,
            pe=pe,
            charge_pc=charge,
            positions_root_m=pos,
            cluster_time=_np(chunk["clusterTime"][i]),
            cluster_pe=_np(chunk["clusterPE"][i]),
            cluster_charge=_np(chunk["clusterCharge"][i]),
            cluster_max_pe=_np(chunk["clusterMaxPE"][i]),
            cluster_n_hits=_np_int(chunk["clusterHits"][i]).astype(np.int32),
            cluster_charge_balance=_np(chunk["clusterChargeBalance"][i]),
            event_time_tank=int(chunk["eventTimeTank"][i]),
            event_time_mrd=int(chunk["eventTimeMRD"][i]),
            nhits=int(chunk["nhits"][i]),
            n_clusters=int(chunk["numberOfClusters"][i]),
        )
        ev.truth = self._make_truth(chunk, i)
        return ev

    # -- truth --------------------------------------------------------------

    def _mrd_track(self, chunk: dict, i: int) -> Optional[TruthTrack]:
        """Reconstruct a track from the MRD hit positions (if available).

        The MRD gives a track segment downstream of the tank.  For the data
        sample there is no generator truth, so the muon is described by that
        straight line: we intersect it with the water tank and use the
        upstream crossing as the in-tank vertex, with the chord length as the
        track length.  This avoids the old back-projection onto the
        ``z = 0`` plane, which extrapolated the (often short, poorly aimed)
        MRD direction ~3.4 m and produced vertices many metres outside the
        tank.

        If the MRD line does not cross the tank, the vertex is stored as NaN
        (the direction and MRD segment are still available) so that consumers
        can skip the event instead of drawing a phantom track.
        """
        start = np.array([
            _scalar(chunk["MRDTrackStartX"][i]),
            _scalar(chunk["MRDTrackStartY"][i]),
            _scalar(chunk["MRDTrackStartZ"][i]),
        ])
        stop = np.array([
            _scalar(chunk["MRDTrackStopX"][i]),
            _scalar(chunk["MRDTrackStopY"][i]),
            _scalar(chunk["MRDTrackStopZ"][i]),
        ])
        direction = stop - start
        norm = np.linalg.norm(direction)
        if norm <= 0 or not np.isfinite(norm):
            return None
        direction = direction / norm

        # MRDTrackStart/Stop are in metres; the tank geometry is in mm.
        start_mm = start * 1000.0
        chord = _tank_chord_root(start_mm, direction)
        if chord is None:
            vertex_mm = np.full(3, np.nan)
            track_length_m = float("nan")
        else:
            entry_mm, exit_mm = chord
            vertex_mm = entry_mm
            track_length_m = float(np.linalg.norm(exit_mm - entry_mm)) / 1000.0

        return TruthTrack(
            pos_root_mm=vertex_mm,
            dir_root=direction,
            kind=TRUTH_KIND_MRD_RECO,
            track_length_water_m=track_length_m,
            mrd_start_root_mm=start_mm,
            mrd_stop_root_mm=stop * 1000.0,
            mrd_angle_rad=_scalar(chunk["MRDTrackAngle"][i]),
            mrd_length_cm=_scalar(chunk["MRDTrackLength"][i]),
        )

    def _make_truth(self, chunk: dict, i: int) -> Optional[TruthTrack]:
        """Build the truth track: generator truth if present, else the MRD reco."""
        mrd = self._mrd_track(chunk, i)
        if not self._has_truth:
            return mrd

        pos = np.array([
            _scalar(chunk["trueVtxX"][i]),
            _scalar(chunk["trueVtxY"][i]),
            _scalar(chunk["trueVtxZ"][i]),
        ])
        direction = np.array([
            _scalar(chunk["trueDirX"][i]),
            _scalar(chunk["trueDirY"][i]),
            _scalar(chunk["trueDirZ"][i]),
        ])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        track = TruthTrack(
            pos_root_mm=pos,
            dir_root=direction,
            kind=TRUTH_KIND_TRUE,
            energy_mev=_scalar(chunk["trueMuonEnergy"][i]),
            pdg=int(_scalar(chunk["truePrimaryPdg"][i])),
            track_length_water_m=_scalar(chunk["trueTrackLengthInWater"][i]),
            track_length_mrd_cm=_scalar(chunk["trueTrackLengthInMRD"][i]),
        )
        if mrd is not None:
            track.mrd_start_root_mm = mrd.mrd_start_root_mm
            track.mrd_stop_root_mm = mrd.mrd_stop_root_mm
            track.mrd_angle_rad = mrd.mrd_angle_rad
            track.mrd_length_cm = mrd.mrd_length_cm
        return track


# ---------------------------------------------------------------------------
# Conversion to the litepath batch schema
# ---------------------------------------------------------------------------


def _aggregate_pmt_hits(times: np.ndarray, pe: np.ndarray,
                        pmt_index: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate raw hits into one row per PMT.

    Mirrors the raytracer's PMT response model: the charge is the sum of
    the photoelectrons and the time is the earliest hit.

    Returns
    -------
    (indices, charges, times, n_hits)
    """
    if pmt_index.size == 0:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_i

    order = np.argsort(pmt_index, kind="stable")
    idx_sorted = pmt_index[order]
    t_sorted = times[order]
    pe_sorted = pe[order]

    uniq, starts = np.unique(idx_sorted, return_index=True)
    ends = np.append(starts[1:], idx_sorted.size)

    charges = np.add.reduceat(pe_sorted, starts)
    first = t_sorted[starts]
    n_hits = (ends - starts).astype(np.int32)
    return uniq.astype(np.int32), charges.astype(np.float32), \
        first.astype(np.float32), n_hits


def _truth_row(event: BenchmarkEvent, registry: PMTRegistry,
               n_pmt_hit: int, n_hits_total: int, pe_total: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``muon_truth`` and ``benchmark_truth`` rows for one event."""
    mu = np.zeros(1, dtype=MUON_TRUTH_DTYPE)
    bt = np.zeros(1, dtype=BENCH_TRUTH_DTYPE)
    mu["event_id"] = event.event_id
    bt["event_id"] = event.event_id
    bt["n_hits_total"] = n_hits_total
    bt["n_pmt_hit"] = n_pmt_hit
    bt["pe_total"] = pe_total

    truth = event.truth
    if truth is None:
        mu["n_detected"] = n_pmt_hit
        bt["track_kind"] = TRUTH_KIND_MRD_RECO
        return mu, bt

    det_rot = registry.det_rotation_deg
    pos_world = truth.pos_world_mm(det_rot)
    dir_world = truth.dir_world(det_rot)
    theta, phi = litepath_angles(dir_world)
    rtheta, rphi = root_angles(truth.dir_root)

    mu["event_id"] = event.event_id
    mu["pos_x"], mu["pos_y"], mu["pos_z"] = pos_world
    mu["t0"] = 0.0
    mu["dir_x"], mu["dir_y"], mu["dir_z"] = dir_world
    mu["theta_deg"] = theta
    mu["phi_deg"] = phi
    mu["track_length_mm"] = truth.track_length_water_m * 1000.0 \
        if np.isfinite(truth.track_length_water_m) else np.nan
    mu["n_detected"] = n_pmt_hit
    mu["source_type"] = SOURCE_TRACK
    mu["spread_deg"] = np.nan
    mu["intercept_x"] = mu["intercept_y"] = mu["intercept_z"] = np.nan

    bt["event_id"] = event.event_id
    bt["track_kind"] = truth.kind
    bt["pos_x"], bt["pos_y"], bt["pos_z"] = pos_world
    bt["dir_x"], bt["dir_y"], bt["dir_z"] = dir_world
    bt["theta_deg"], bt["phi_deg"] = theta, phi
    bt["root_pos_x"], bt["root_pos_y"], bt["root_pos_z"] = truth.pos_root_mm
    bt["root_dir_x"], bt["root_dir_y"], bt["root_dir_z"] = truth.dir_root
    bt["root_theta_deg"], bt["root_phi_deg"] = rtheta, rphi
    bt["energy_mev"] = truth.energy_mev
    bt["pdg"] = truth.pdg
    bt["track_length_water_mm"] = truth.track_length_water_m * 1000.0 \
        if np.isfinite(truth.track_length_water_m) else np.nan
    bt["track_length_mrd_mm"] = truth.track_length_mrd_cm * 10.0 \
        if np.isfinite(truth.track_length_mrd_cm) else np.nan
    if truth.mrd_start_root_mm is not None:
        bt["mrd_start_x"], bt["mrd_start_y"], bt["mrd_start_z"] = truth.mrd_start_root_mm
        bt["mrd_stop_x"], bt["mrd_stop_y"], bt["mrd_stop_z"] = truth.mrd_stop_root_mm
        bt["mrd_angle_rad"] = truth.mrd_angle_rad
        bt["mrd_length_mm"] = truth.mrd_length_cm * 10.0
    return mu, bt


def write_benchmark_h5(
    path: Path | str,
    events: Iterable[BenchmarkEvent],
    registry: PMTRegistry,
    *,
    kind: str = "data",
    source_file: str = "",
    det_rotation_deg: float = DET_ROTATION_DEG,
    write_raw_hits: bool = True,
    chunk_events: int = 500,
) -> Path:
    """Write parsed WAGLE events in the litepath batch HDF5 schema.

    The file contains the standard tables (``muon_truth``, ``pmt_responses``,
    ``detectors``) with exactly the dtypes used by ``litepath batch``, plus
    two extension tables: ``pmt_hits`` (the raw digitised hits) and
    ``benchmark_truth`` (the track truth/reconstruction in both frames).

    Parameters
    ----------
    path : Path | str
        Output HDF5 path.
    events : iterable of BenchmarkEvent
        Events to write.
    registry : PMTRegistry
        PMT lookup used to map WCSim TubeIDs to litepath detector indices.
    kind : str
        ``"data"`` or ``"mc"`` — recorded in the root attributes.
    source_file : str
        Path of the originating ROOT file (recorded in the root attributes).
    det_rotation_deg : float
        Global Z-rotation the transform was built with.
    write_raw_hits : bool
        Write the per-hit ``pmt_hits`` table.
    chunk_events : int
        Number of events buffered before flushing to disk.

    Returns
    -------
    Path
        The path that was written.
    """
    import h5py

    from litepath.io_h5 import FORMAT_NAME, SCHEMA_VERSION

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pmt_hits_buf: list[np.ndarray] = []
    pmt_resp_buf: list[np.ndarray] = []
    mu_buf: list[np.ndarray] = []
    bt_buf: list[np.ndarray] = []
    n_events = 0

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["format"] = FORMAT_NAME
        f.attrs["detector_mode"] = f"benchmark-{kind}"
        f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["source_file"] = source_file
        f.attrs["det_rotation_deg"] = det_rotation_deg
        f.attrs["config"] = json.dumps({
            "kind": kind,
            "source_file": source_file,
            "det_rotation_deg": det_rotation_deg,
            "frame": "world (litepath structure frame, mm)",
        })
        # Tank geometry — matches litepath.geometry.build.build_geometry so that the
        # standard event display / analysis scripts can use these files.
        f.attrs["tank_radius_mm"] = TANK_RADIUS_MM
        f.attrs["tank_z_min_mm"] = TANK_Z_MIN_MM
        f.attrs["tank_z_max_mm"] = TANK_Z_MAX_MM

        det_table = registry.detectors_table()
        f.create_dataset("detectors", data=det_table, compression="gzip")

        def _flush() -> None:
            if pmt_resp_buf:
                _append(f, "pmt_responses", np.concatenate(pmt_resp_buf))
                pmt_resp_buf.clear()
            if mu_buf:
                _append(f, "muon_truth", np.concatenate(mu_buf))
                mu_buf.clear()
            if bt_buf:
                _append(f, "benchmark_truth", np.concatenate(bt_buf))
                bt_buf.clear()
            if pmt_hits_buf:
                _append(f, "pmt_hits", np.concatenate(pmt_hits_buf))
                pmt_hits_buf.clear()

        for ev in events:
            pmt_index = registry.index_array(ev.tube_ids)
            good = pmt_index >= 0
            pmt_index = pmt_index[good]
            times = ev.times[good]
            pe_raw = ev.pe[good]
            # WAGLE occasionally reports non-finite PE (digitiser overflow);
            # treat those hits as carrying no charge.
            pe = np.where(np.isfinite(pe_raw), pe_raw, 0.0)
            charge = ev.charge_pc[good] if ev.charge_pc.size == ev.times.size \
                else np.full(times.shape, np.nan)

            indices, charges, first_times, n_hits = _aggregate_pmt_hits(
                times, pe, pmt_index,
            )

            resp = np.zeros(len(indices), dtype=PMT_RESPONSE_DTYPE)
            resp["event_id"] = ev.event_id
            resp["pmt_index"] = indices
            resp["charge"] = charges
            resp["time"] = first_times
            resp["n_hits"] = n_hits
            pmt_resp_buf.append(resp)

            mu, bt = _truth_row(ev, registry, len(indices), int(times.size),
                                 float(pe.sum()))
            mu_buf.append(mu)
            bt_buf.append(bt)

            if write_raw_hits and times.size:
                pos_world = root_to_world(ev.positions_root_m[good] * 1000.0,
                                          det_rotation_deg)
                raw = np.zeros(times.size, dtype=PMT_HIT_DTYPE)
                raw["event_id"] = ev.event_id
                raw["pmt_index"] = pmt_index
                raw["detector_id"] = ev.tube_ids[good]
                raw["arrival_time"] = times
                raw["charge_pe"] = pe
                raw["charge_pc"] = charge
                raw["x"] = pos_world[:, 0]
                raw["y"] = pos_world[:, 1]
                raw["z"] = pos_world[:, 2]
                pmt_hits_buf.append(raw)

            n_events += 1
            if n_events % chunk_events == 0:
                _flush()

        _flush()

    return path


def _append(f, name: str, data: np.ndarray) -> None:
    """Append rows to a resizable dataset, creating it on first use."""

    if name in f:
        dset = f[name]
        n = dset.shape[0]
        dset.resize((n + len(data),))
        dset[n:] = data
    else:
        f.create_dataset(name, shape=(len(data),), maxshape=(None,),
                         dtype=data.dtype, data=data, compression="gzip")
