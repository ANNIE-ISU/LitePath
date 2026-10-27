"""Showcase: load batch output (full-detector or batch-lappd) with BatchReader.

Usage:
    python scripts/load_batch.py results/output.h5
    python scripts/load_batch.py /tmp/batch_out/       # reads <dir>/output.h5

Both `litepath batch` and `litepath batch-lappd` write the same HDF5 format,
so this script works on either.
"""

import argparse
from pathlib import Path

from litepath.io_h5 import BatchReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect batch output")
    parser.add_argument("output", help="HDF5 file or directory containing output.h5")
    args = parser.parse_args()

    path = Path(args.output)
    if path.is_dir():
        path = path / "output.h5"
    if not path.exists():
        print(f"Not found: {path}")
        return

    reader = BatchReader(path)
    summary = reader.summary()
    print(f"format={summary['format']} schema_version={summary['schema_version']}")
    print(f"detector_mode={summary['detector_mode']} n_events={summary['n_events']}")
    print(f"tables: {sorted(summary['tables'])}")
    print(f"config: {summary['config']}\n")

    # ── Light-source truth ──────────────────────────────────────────
    muons = reader.load_table("muon_truth")
    print(f"── muon_truth: {len(muons)} events ──")
    cols = ["event_id", "source_type", "pos_x", "pos_y", "pos_z",
            "dir_x", "dir_y", "dir_z", "track_length_mm",
            "n_generated", "n_detected", "spread_deg",
            "intercept_x", "intercept_y", "intercept_z"]
    print(muons[[c for c in cols if c in muons]].to_string(index=False))

    # Per-event detection efficiency
    muons["efficiency"] = muons["n_detected"] / muons["n_generated"].clip(1)
    print(f"\nefficiency: {muons['efficiency'].min():.4f} – "
          f"{muons['efficiency'].max():.4f}")

    # Source mix
    import numpy as np
    src_names = {0: "track", 1: "light_burst", 2: "spot"}
    if "source_type" in muons:
        counts = muons["source_type"].value_counts()
        for st, n in counts.items():
            print(f"source_type={st} ({src_names.get(st, '?')}): {n} events")

    # ── Photon hits ─────────────────────────────────────────────────
    hits = reader.load_table("photon_hits")
    print(f"\n── photon_hits: {len(hits)} rows, "
          f"{hits.event_id.nunique()} events ──")

    per_event = hits.groupby("event_id").agg(
        n_photons=("detector_index", "count"),
        n_pmt=("detector_system", lambda x: (x == 0).sum()),
        n_lappd=("detector_system", lambda x: (x != 0).sum()),
    )
    print("Per-event hit counts (first 5):")
    print(per_event.head())

    # ── LAPPD readouts (if enabled) ─────────────────────────────────
    readouts = reader.load_table("lappd_readouts")
    if len(readouts):
        print(f"\n── lappd_readouts: {len(readouts)} rows across "
              f"{readouts.event_id.nunique()} events ──")
        print(readouts[["event_id", "lappd_index", "n_photons",
                        "n_passed_qe"]].head())
        # Full per-LAPPD traces for a single event
        ev = reader.load_event(int(readouts["event_id"].iloc[0]))
        for idx, arr in ev.lappd_readouts.items():
            print(f"event {ev.event_id} LAPPD {idx}: readout shape "
                  f"{arr.shape} (side0 sum={arr[..., 0].sum():.4f}, "
                  f"side1 sum={arr[..., 1].sum():.4f})")

    # ── PMT responses (full-detector batches) ───────────────────────
    pmts = reader.load_table("pmt_responses")
    if len(pmts):
        print(f"\n── pmt_responses: {len(pmts)} rows ──")
        per_charge = pmts.groupby("event_id")["charge"].sum()
        print(f"total charge per event: {per_charge.head().to_dict()}")

    # ── Per-event shortcut via EventData ────────────────────────────
    ev = reader.load_event(0)
    print(f"\n── Event 0 ──")
    print(f"  is_beam={ev.is_beam()} n_generated={ev.n_generated()} "
          f"n_detected={ev.n_detected()}")
    print(f"  hits={len(ev.hits)}, lappd_readouts="
          f"{list(ev.lappd_readouts)}")


if __name__ == "__main__":
    main()