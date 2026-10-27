#!/usr/bin/env python3
"""Extract a WAGLE MC muon track as an litepath muon topology file.

The result can be fed straight to ``litepath batch --muon-file`` so that the
raytracer reproduces the light pattern of a specific WAGLE Monte-Carlo event:

    python scripts/mc_muon_file.py results/benchmark/mc.h5 --event 0 \
        --out /tmp/ev0.txt
    python -m litepath batch --muon-file /tmp/ev0.txt --events 1 \
        --pmt-csv PMTPositions_Scan.txt --surfboard 3
    python scripts/event_display.py results/benchmark/mc.h5 \
        --compare results/raytrace/output.h5 --compare-event 0 --charge

Input can be either a converted benchmark file (``results/benchmark/mc.h5``)
or the original WAGLE ROOT file (``benchmarking_data/MCTree_*.root``).

The muon file format is ``x y z t0 dx dy dz`` with positions in mm, ``t0`` in
seconds and the direction a unit vector, in the litepath world frame.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _from_h5(path: Path, event_id: int):
    from litepath.io_h5 import load_table

    truth = load_table(path, "muon_truth")
    rows = truth[truth["event_id"] == event_id]
    if rows.empty:
        raise SystemExit(f"Error: event {event_id} not found in {path}")
    r = rows.iloc[0]
    pos = np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])])
    d = np.array([float(r["dir_x"]), float(r["dir_y"]), float(r["dir_z"])])
    return pos, d


def _from_root(path: Path, event_id: int, tree: str):
    from litepath.benchmark import RootTreeReader

    reader = RootTreeReader(path, tree=tree)
    for i, ev in enumerate(reader.iter_events()):
        if i == event_id:
            return ev.truth.pos_world_mm(), ev.truth.dir_world()
    raise SystemExit(f"Error: event {event_id} not found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Write a WAGLE MC muon track as an litepath muon file")
    parser.add_argument("input", type=Path,
                        help="Benchmark HDF5 (mc.h5) or WAGLE ROOT file")
    parser.add_argument("--event", type=int, default=0, help="Event id")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--tree", type=str, default="Event",
                        help="TTree name for ROOT input (default: Event)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Error: {args.input} not found")

    if args.input.suffix == ".root":
        pos, d = _from_root(args.input, args.event, args.tree)
    else:
        pos, d = _from_h5(args.input, args.event)

    line = " ".join(f"{v:.6f}" for v in (*pos, 0.0, *d))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(line + "\n")
        print(f"Wrote {args.out}: {line}")
    else:
        print(line)


if __name__ == "__main__":
    main()
