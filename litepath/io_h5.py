"""HDF5 I/O helpers for batch output, single-shot hits, and readback."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numpy structured dtypes  (field names → DataFrame column names)
# ---------------------------------------------------------------------------

# Batch photon hits (8 columns)
PHOTON_HIT_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("detector_system", "i4"),
    ("detector_index", "i4"),
    ("local_u", "f4"),
    ("local_v", "f4"),
    ("arrival_time", "f4"),
    ("wavelength", "f4"),
    ("photon_source", "i4"),
])

# Per-event light-source truth (17 + 5 + 7 columns).
# The table records the "true" parameters of whatever produced the light
# in each event.  ``source_type`` discriminates between the definitions
# (see SOURCE_TRACK / SOURCE_LIGHT_BURST / SOURCE_SPOT below); columns that
# do not apply to a given source type are NaN.
MUON_TRUTH_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("pos_x", "f4"),
    ("pos_y", "f4"),
    ("pos_z", "f4"),
    ("t0", "f4"),
    ("dir_x", "f4"),
    ("dir_y", "f4"),
    ("dir_z", "f4"),
    ("theta_deg", "f4"),
    ("phi_deg", "f4"),
    ("track_length_mm", "f4"),
    ("n_generated", "i4"),
    ("n_detected", "i4"),
    ("n_generated_ckv", "i4"),
    ("n_generated_sci", "i4"),
    ("n_detected_ckv", "i4"),
    ("n_detected_sci", "i4"),
    # Light-source discriminator and spot-beam extras
    ("source_type", "i4"),          # 0 = tracked Cherenkov muon, 1 = light burst, 2 = spot beam
    ("spread_deg", "f4"),           # spot-beam dispersion (°); NaN for track/burst
    ("intercept_x", "f4"),          # PC target the beam was aimed at (mm); NaN when unused
    ("intercept_y", "f4"),
    ("intercept_z", "f4"),
    # v2: track_length_mm under a clearer name (same value; track_length_mm
    # stays for now), and where the light-emitting stretch starts and ends:
    # the tank entry (or the start, for a muon starting inside) and the tank
    # exit (mm).  NaN when no light is generated in a tank (missed tank,
    # bursts, spot beams, standalone LAPPD).
    ("light_track_length_mm", "f4"),
    ("tank_entry_x", "f4"),
    ("tank_entry_y", "f4"),
    ("tank_entry_z", "f4"),
    ("tank_exit_x", "f4"),
    ("tank_exit_y", "f4"),
    ("tank_exit_z", "f4"),
])

# Light-source types for the per-event truth table (muon_truth.source_type)
SOURCE_TRACK = 0          # tracked particle (muon) -> Cherenkov cone along a track
SOURCE_LIGHT_BURST = 1    # isotropic light burst at a point
SOURCE_SPOT = 2           # directed spot beam (origin + dir + spread), optional PC intercept

# PMT responses (5 columns)
PMT_RESPONSE_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("pmt_index", "i4"),
    ("charge", "f4"),
    ("time", "f4"),
    ("n_hits", "i4"),
])

# Full LAPPD readout traces (one row per (event, LAPPD)).
# side0/side1 are the 28-strip x 256-time-bin voltage-time matrices (V).
LAPPD_READOUT_DTYPE = np.dtype([
    ("event_id", "i8"),
    ("lappd_index", "i4"),
    ("side0", "f4", (28, 256)),
    ("side1", "f4", (28, 256)),
    ("n_photons", "i4"),
    ("n_passed_qe", "i4"),
])

# Detector registry (7 + 5 columns)
DETECTOR_DTYPE = np.dtype([
    ("system_code", "i4"),
    ("detector_index", "i4"),
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("label", h5py.string_dtype()),
    ("panel", "i4"),
    # v2: real-detector identifiers.  PMTs: ToolAnalysis channel key and
    # detector key (equal for tank PMTs, = WCSim TubeID), cable label, type,
    # ON/OFF.  LAPPDs: channel_key -1, detector_key = LitePath ID (1000+/2000+).
    ("channel_key", "i4"),
    ("detector_key", "i4"),
    ("cable_label", h5py.string_dtype()),
    ("pmt_type", h5py.string_dtype()),
    ("status", h5py.string_dtype()),
])

# Geometry elements (v2): the simulated detector, one row per element, so a
# file can be analysed without LitePath's geometry files.
GEOMETRY_ELEMENT_DTYPE = np.dtype([
    ("element_id", "i4"),        # detector_key (PMT channel key, LAPPD 1000+/2000+)
    ("system_code", "i4"),
    ("detector_index", "i4"),
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("dir_x", "f8"),             # inward-facing unit normal
    ("dir_y", "f8"),
    ("dir_z", "f8"),
    ("size_mm", "f8"),           # PMT sphere radius or LAPPD half-side
    ("type", h5py.string_dtype()),
    ("status", h5py.string_dtype()),
])

# Full hit table for single-shot mode (18 columns)
FULL_HIT_DTYPE = np.dtype([
    ("hit_flag", "i4"),
    ("t", "f4"),
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("nx", "f4"),
    ("ny", "f4"),
    ("nz", "f4"),
    ("component_id", "i4"),
    ("detector_index", "i4"),
    ("detector_system", "i4"),
    ("local_u", "f4"),
    ("local_v", "f4"),
    ("arrival_time", "f4"),
    ("wavelength", "f4"),
    ("bounce_count", "i4"),
    ("photon_source", "i4"),
    ("photon_id", "i8"),
])

DTYPES = {
    "photon_hits": PHOTON_HIT_DTYPE,
    "muon_truth": MUON_TRUTH_DTYPE,
    "pmt_responses": PMT_RESPONSE_DTYPE,
    "lappd_readouts": LAPPD_READOUT_DTYPE,
    "detectors": DETECTOR_DTYPE,
    "geometry_elements": GEOMETRY_ELEMENT_DTYPE,
    "hits": FULL_HIT_DTYPE,
}

# Schema version stamped into every output.h5 root group.
# v2 (2026-11): geometry provenance attributes, geometry_elements table,
# detector identifiers, light_track_length_mm and tank entry/exit points.
# See "Changes in v2" in docs/BATCH_OUTPUT.md.
SCHEMA_VERSION = 2
FORMAT_NAME = "litepath-batch"


# ---------------------------------------------------------------------------
# Append helper (resizable dataset)
# ---------------------------------------------------------------------------

def _ensure_dataset(f: h5py.File, name: str, dtype: np.dtype):
    """Return existing dataset *name* or create a resizable one."""
    if name in f:
        return f[name]
    return f.create_dataset(name, shape=(0,), maxshape=(None,),
                            dtype=dtype, compression="gzip")


def append_table(f: h5py.File, name: str, data: np.ndarray) -> None:
    """Append *data* (a numpy structured array) to a resizable dataset."""
    dset = _ensure_dataset(f, name, data.dtype)
    n = dset.shape[0]
    dset.resize((n + len(data),))
    dset[n:] = data


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _records_to_dataframe(records: np.ndarray) -> pd.DataFrame:
    """Build a DataFrame from a structured array.

    Structured dtypes may contain sub-array fields (e.g. the (28, 256)
    LAPPD readout matrices).  pandas cannot vectorise those, so each row's
    sub-array is stored as an object (2-D ndarray) column entry.
    """
    cols: dict[str, object] = {}
    for name in records.dtype.names:
        val = records[name]
        if val.dtype == object:  # HDF5 strings (label, cable_label, ...) → str
            val = np.array([v.decode() if isinstance(v, bytes) else v for v in val], dtype=object)
        cols[name] = list(val) if val.ndim > 1 else val
    return pd.DataFrame(cols)


def _add_v2_columns(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Fill in the v2 columns of a table read from an older (v1) file."""
    if table == "muon_truth" and "track_length_mm" in df and "light_track_length_mm" not in df:
        df["light_track_length_mm"] = df["track_length_mm"]
        for col in ("tank_entry_x", "tank_entry_y", "tank_entry_z",
                    "tank_exit_x", "tank_exit_y", "tank_exit_z"):
            df[col] = np.float32(np.nan)
    elif table == "detectors" and "label" in df and "channel_key" not in df:
        def _key(label: str) -> int:
            return int(label[4:]) if label.startswith("PMT_") and label[4:].isdigit() else -1
        labels = df["label"].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
        df["channel_key"] = labels.map(_key).astype(np.int32)
        df["detector_key"] = df["channel_key"]
        for col in ("cable_label", "pmt_type", "status"):
            df[col] = ""
    return df


def load_table(path: Path | str, table: str) -> pd.DataFrame:
    """Load an HDF5 table as a pandas DataFrame.

    Returns an empty DataFrame if the dataset does not exist.  Tables from
    v1 files get the v2 columns filled in (see docs/BATCH_OUTPUT.md).
    """
    with h5py.File(path, "r") as f:
        if table not in f:
            dtype = DTYPES.get(table)
            if dtype is None:
                return pd.DataFrame()
            return pd.DataFrame({n: [] for n in dtype.names})
        return _add_v2_columns(table, _records_to_dataframe(f[table][()]))


def load_all(path: Path | str) -> dict[str, pd.DataFrame]:
    """Load all standard batch-output tables."""
    return {key: load_table(path, key) for key in DTYPES}


def load_event_lappd_readout(path: str | Path, event_id: int
                              ) -> dict[int, np.ndarray] | None:
    """Read LAPPD readout matrices for *event_id*.

    Returns ``{detector_index: (28, 256, 2) readout}``, or ``None``
    when *event_id* has no LAPPD readout data.

    Supports the standardized flat ``lappd_readouts`` table (schema v1+)
    and falls back to the older per-event group layout (``event_N/...``).
    """
    with h5py.File(path, "r") as f:
        if "lappd_readouts" in f:
            table = f["lappd_readouts"][:]
            recs = table[table["event_id"] == event_id]
            if len(recs) == 0:
                return None
            return {
                int(r["lappd_index"]): np.stack([r["side0"], r["side1"]], axis=-1)
                for r in recs
            }
        grp = f.get(f"event_{event_id}")
        if grp is None or "lappd_readouts" not in grp:
            return None
        data = grp["lappd_readouts"][:]   # (n_lapds, 28, 256, 2)
        indices = grp["lappd_readouts"].attrs.get("detector_indices")
        if indices is None:
            return {0: data[0]} if data.shape[0] == 1 else {}
        return {int(indices[i]): data[i] for i in range(data.shape[0])}


def read_attrs(path: Path | str) -> dict:
    """Read root-group attributes (e.g. tank metadata)."""
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


# ---------------------------------------------------------------------------
# Parsing framework: BatchReader / EventData
# ---------------------------------------------------------------------------

def _nan_line_source(row) -> dict:
    """Extract source-type extras from a muon_truth row as plain Python types."""
    nan_cols = ("intercept_x", "intercept_y", "intercept_z")
    if np.all([np.isnan(row[c]) for c in nan_cols]):
        intercept = None
    else:
        intercept = tuple(float(row[c]) for c in nan_cols)
    return {
        "event_id": int(row["event_id"]),
        "source_type": int(row["source_type"]),
        "spread_deg": float(row["spread_deg"]),
        "intercept": intercept,
    }


@dataclass
class EventData:
    """One event's full recorded data, bundled for analysis."""

    event_id: int
    truth: dict                            # muon_truth row (all fields, plain floats)
    hits: pd.DataFrame                     # photon_hits for this event
    pmt_responses: pd.DataFrame            # pmt_responses for this event (may be empty)
    lappd_readouts: dict[int, np.ndarray]  # {lappd_index: (28, 256, 2) readout}

    @property
    def source_type(self) -> int:
        return int(self.truth["source_type"])

    def is_track(self) -> bool:
        """True when the light came from a tracked particle (Cherenkov muon)."""
        return self.source_type == SOURCE_TRACK

    def is_beam(self) -> bool:
        """True when the light came from a directed spot beam."""
        return self.source_type == SOURCE_SPOT

    def n_generated(self) -> int:
        return int(self.truth["n_generated"])

    def n_detected(self) -> int:
        return int(self.truth["n_detected"])


class BatchReader:
    """Read the standardized ``output.h5`` batch schema.

    Both the full-detector ``litepath batch`` and the standalone-LAPPD
    ``litepath batch-lappd`` commands write this schema, so a single reader
    works for either mode.  Reads v1 and v2 files; v1 tables get the v2
    columns filled in.

    Examples
    --------
    >>> reader = BatchReader("results/output.h5")
    >>> reader.summary()
    >>> for ev in reader.iter_events():
    ...     print(ev.event_id, ev.truth["theta_deg"], ev.n_detected())
    >>> ev = reader.load_event(3)
    >>> ev.lappd_readouts  # {lappd_index: (28, 256, 2)}
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._attrs: dict | None = None
        self._event_ids: list[int] | None = None

    # -- file-level ---------------------------------------------------------

    def summary(self) -> dict:
        """Return file-level metadata: schema version, mode, tables, events."""
        with h5py.File(self.path, "r") as f:
            attrs = dict(f.attrs)
            tables = [k for k in DTYPES if k in f]
            return {
                "schema_version": attrs.get("schema_version"),
                "format": attrs.get("format"),
                "detector_mode": attrs.get("detector_mode"),
                "geometry_version": attrs.get("geometry_version"),
                "geometry_hash": attrs.get("geometry_hash"),
                "config": attrs.get("config"),
                "tables": tables,
                "n_events": len(self.event_ids()),
            }

    def attrs(self) -> dict:
        if self._attrs is None:
            self._attrs = read_attrs(self.path)
        return dict(self._attrs)

    def event_ids(self) -> list[int]:
        """Sorted event IDs present in the truth table."""
        if self._event_ids is None:
            truth = load_table(self.path, "muon_truth")
            if truth.empty:
                self._event_ids = []
            else:
                self._event_ids = sorted(truth["event_id"].unique().tolist())
        return list(self._event_ids)

    def load_table(self, name: str) -> pd.DataFrame:
        """Load one standardized table (photon_hits, muon_truth, ...)."""
        return load_table(self.path, name)

    # -- per-event ----------------------------------------------------------

    def load_event(self, event_id: int) -> EventData:
        """Bundle everything recorded for *event_id* into an EventData."""
        truth = load_table(self.path, "muon_truth")
        row = truth[truth["event_id"] == event_id]
        if row.empty:
            raise KeyError(f"event_id {event_id} not found in muon_truth")

        t = row.iloc[0]
        meta = {
            key: (float(val) if key != "event_id" and key != "source_type" else int(val))
            for key, val in t.items()
            if key not in ("spread_deg", "intercept_x", "intercept_y", "intercept_z")
        }
        meta.update(_nan_line_source(t))

        hits = load_table(self.path, "photon_hits")
        pmts = load_table(self.path, "pmt_responses")
        lappd = load_event_lappd_readout(self.path, event_id) or {}

        return EventData(
            event_id=int(event_id),
            truth=meta,
            hits=hits[hits["event_id"] == event_id],
            pmt_responses=pmts[pmts["event_id"] == event_id] if not pmts.empty
            else pd.DataFrame(),
            lappd_readouts=lappd,
        )

    def iter_events(self) -> "BatchReader._EventIterator":
        return self._EventIterator(self)

    class _EventIterator:
        def __init__(self, reader: "BatchReader"):
            self._reader = reader
            self._ids = reader.event_ids()
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self) -> EventData:
            if self._i >= len(self._ids):
                raise StopIteration
            ev = self._reader.load_event(self._ids[self._i])
            self._i += 1
            return ev

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Write helpers for batch mode
# ---------------------------------------------------------------------------

def _system_code(system: str) -> int:
    from litepath.hits import DET_SYS_PMT, DET_SYS_LAPPD_DEFAULT, DET_SYS_LAPPD_ANNIE
    return {
        "pmt": DET_SYS_PMT,
        "lappd_default": DET_SYS_LAPPD_DEFAULT,
        "lappd_annie": DET_SYS_LAPPD_ANNIE,
    }.get(system, -1)


def _pmt_records(pmt_info: dict | None) -> dict:
    """PMT table records by channel key (empty when there is no table)."""
    if not pmt_info or "records" not in pmt_info:
        return {}
    return {r.channel_key: r for r in pmt_info["records"]}


def write_detectors(path: Path, detectors: list, pmt_info: dict | None = None) -> None:
    """Write the detector registry (``detectors``) and ``geometry_elements``.

    *pmt_info* is ``Geometry.pmt_info``; its PMT table records supply the
    cable labels and ON/OFF status.
    """
    records = _pmt_records(pmt_info)
    rows, elements = [], []
    for d in detectors:
        sys_code = _system_code(d.system)
        panel = getattr(d, "panel", -1)
        if d.system == "pmt":
            rec = records.get(d.id)
            channel_key, kind = d.id, d.pmt_type
            cable = rec.cable_label if rec else ""
            status = rec.status if rec else "ON"
            size = d.radius
        else:
            channel_key, kind, cable, status = -1, "LAPPD", "", "ON"
            size = d.half_size
        rows.append((sys_code, d.index, d.position[0], d.position[1], d.position[2],
                     str(d.label), panel, channel_key, d.id, cable, kind, status))
        elements.append((d.id, sys_code, d.index, *d.position, *d.direction,
                         size, kind, status))
    arr = np.array(rows, dtype=DETECTOR_DTYPE)
    elem = np.array(elements, dtype=GEOMETRY_ELEMENT_DTYPE)
    with h5py.File(path, "a") as f:
        append_table(f, "detectors", arr)
        append_table(f, "geometry_elements", elem)
    print(f"  Wrote detectors and geometry_elements tables ({len(rows)} rows) to {path}")


def write_metadata(path: Path, meta: dict) -> None:
    """Write tank metadata as HDF5 root-group attributes."""
    with h5py.File(path, "a") as f:
        for key, val in meta.items():
            f.attrs[key] = val
