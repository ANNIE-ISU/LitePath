"""PMT records read from the detector's PMT tables.

A :class:`PMTRecord` is one tank PMT as a table describes it, before any
placement: its IDs, type, panel, scan-frame position and status.  Two
tables are supported and give the same records apart from what each one
knows:

- the ToolAnalysis ``FullTankPMTGeometry.csv`` used for real data, which
  also carries cable labels and ON/OFF status;
- the WCSim-style ``PMTPositions_Scan.txt`` (tab-separated, no labels, all
  PMTs ON).

For tank PMTs the ToolAnalysis detector key and channel key both equal the
WCSim TubeID (332–463), so ``channel_key`` is the element ID everywhere.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATUS_ON = "ON"
STATUS_OFF = "OFF"


@dataclass(frozen=True)
class PMTRecord:
    """One tank PMT as given by a PMT table."""

    channel_key: int
    detector_key: int
    panel: int
    pmt_type: str                               # e.g. "LUX", "ETEL", "Hamamatsu"
    scan_position_cm: tuple[float, float, float]  # scan frame, cm
    cable_label: str = ""
    status: str = STATUS_ON


def read_pmt_table(path: Path) -> list[PMTRecord]:
    """Read a ToolAnalysis CSV (``.csv``) or a WCSim scan file (anything else)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_toolanalysis_pmts(path)
    return read_scan_pmts(path)


def read_toolanalysis_pmts(path: Path) -> list[PMTRecord]:
    """Read ToolAnalysis ``FullTankPMTGeometry.csv``.

    The header row starts with ``detector_num``; data rows sit between the
    ``DATA_START`` and ``DATA_END`` markers.  Positions are in metres.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f)]

    header = next((row for row in rows if row and row[0] == "detector_num"), None)
    if header is None:
        raise ValueError(f"{path}: no header row starting with 'detector_num'")
    try:
        start = next(i for i, row in enumerate(rows) if row and row[0] == "DATA_START")
    except StopIteration:
        raise ValueError(f"{path}: no DATA_START marker") from None

    records = []
    for row in rows[start + 1:]:
        if not row or not row[0]:
            continue
        if row[0] == "DATA_END":
            break
        r = dict(zip(header, row))
        records.append(PMTRecord(
            channel_key=int(r["channel_num"]),
            detector_key=int(r["detector_num"]),
            panel=int(r["panel_number"]),
            pmt_type=r["PMT_type"],
            scan_position_cm=tuple(100.0 * float(r[k]) for k in ("x_pos", "y_pos", "z_pos")),
            cable_label=r["cable_label"],
            status=r["detector_status"],
        ))
    if not records:
        raise ValueError(f"{path}: no PMT rows found")
    return records


def read_scan_pmts(path: Path) -> list[PMTRecord]:
    """Read a WCSim scan file.

    Tab-separated columns: TubeID, panel, x, y, z (cm), dir x/y/z (unused),
    WCSim PMT-collection index.  The type name comes from the collection
    index and panel (see ``pmt_loader.PMT_TYPE_NAMES``).
    """
    from litepath.pmt_loader import PMT_TYPE_NAMES

    data = np.loadtxt(path, delimiter="\t", ndmin=2)
    records = []
    for row in data:
        tube_id, panel, type_code = int(row[0]), int(row[1]), int(row[8])
        records.append(PMTRecord(
            channel_key=tube_id,
            detector_key=tube_id,
            panel=panel,
            pmt_type=PMT_TYPE_NAMES.get((type_code, panel), "Unknown"),
            scan_position_cm=(float(row[2]), float(row[3]), float(row[4])),
        ))
    return records
