"""LAPPD hit maps and shadowing-ratio maps from batch output.

Plots, for each ANNIE LAPPD, the number of hits for every muon topology
(start position + direction) of a grid scan, laid out as in Fig. 7.7 of
Yue Feng's thesis (the shadowing study): one chunk of dots per start
position (x, z), and inside a chunk one dot per direction, placed at its
slopes (dx/dy, dz/dy).

Usage:
    python scripts/LAPPD_shadowing_map.py RUN
    python scripts/LAPPD_shadowing_map.py RUN --ratio-to OTHER_RUN

    RUN, OTHER_RUN: batch output files (output.h5) or folders holding them.

Hits mode (default): hits per event on each LAPPD, log colour scale.

Ratio mode (--ratio-to): the shadowing ratio of Yue's eq. 7.1,

    s = (mean hits per event in RUN) / (mean hits per event in OTHER_RUN)

e.g. full geometry over a run with --no-gdml.  Blue (0) to white (1) to
red (2); ratios above 2 show as 2.  Topologies with no hits in OTHER_RUN
have no ratio and are left blank.  Run both with the same muon file and
the same --seed so they see the same photons (see docs/README.md).

Topologies are matched by position and direction; events that repeat a
topology (--events larger than the muon file) are averaged.

Muon files with a single direction per position are drawn as a colour
mesh over (x, z) instead.

Example muon file: muon_files/MuonStartsAndDirecs_Yue_13x13pos_9x9dir.txt
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from matplotlib.collections import PatchCollection

LAPPD_SYSTEMS = (1, 2)   # detector_system codes of LAPPDs (default, ANNIE housing)


def h5_file(path: str) -> Path:
    p = Path(path)
    return p if p.suffix == ".h5" else p / "output.h5"


def hits_per_topology(h5_path: Path) -> pd.DataFrame:
    """Mean LAPPD hits per event for each muon topology.

    Index: pos_x, pos_z (mm) and dir_x, dir_z "unnormalized" to dy = 1
    (the slopes dx/dy, dz/dy).  Columns: n_pmt<detector_index>.
    """
    with h5py.File(h5_path, "r") as f:
        # only the columns needed: the hit table can have ~10^8 rows
        hits = f["photon_hits"].fields(["event_id", "detector_system", "detector_index"])[:]
        truth = f["muon_truth"].fields(["event_id", "pos_x", "pos_z",
                                        "dir_x", "dir_y", "dir_z"])[:]
    if len(truth) == 0:
        raise SystemExit(f"{h5_path}: no muon_truth rows")

    # Creating LAPPD hit data: hits per event and LAPPD
    lappd = np.isin(hits["detector_system"], LAPPD_SYSTEMS)
    counts = (pd.DataFrame({"event_id": hits["event_id"][lappd],
                            "detector_index": hits["detector_index"][lappd]})
              .groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits"))
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits")
    pivoted.columns = [f"n_pmt{c}" for c in pivoted.columns]

    muons = pd.DataFrame(truth)
    table = muons[["event_id", "pos_x", "pos_z", "dir_x", "dir_y", "dir_z"]] \
        .merge(pivoted, on="event_id", how="left").fillna(0)   # no LAPPD hits -> 0

    # Unnormalizing the direction columns (dy = 1); dir_y has to be last
    table["dir_x"] = (table["dir_x"] / table["dir_y"]).round(3)
    table["dir_z"] = (table["dir_z"] / table["dir_y"]).round(3)
    table["dir_y"] = 1.0
    table["pos_x"] = table["pos_x"].round(1)
    table["pos_z"] = table["pos_z"].round(1)

    # One row per topology (averages events that repeat a topology)
    return (table.drop(columns=["event_id", "dir_y"])
            .groupby(["pos_x", "pos_z", "dir_x", "dir_z"]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="LAPPD hit maps and shadowing-ratio maps")
    parser.add_argument("run", help="batch output (file or folder)")
    parser.add_argument("--ratio-to", metavar="OTHER_RUN",
                        help="plot hits in RUN divided by hits in OTHER_RUN")
    parser.add_argument("--table", metavar="FILE",
                        help="also write the per-topology table as text")
    parser.add_argument("--save", metavar="PREFIX",
                        help="also save each map as PREFIX_lappd<N>.png")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────
    h5_path = h5_file(args.run)
    data = hits_per_topology(h5_path)
    ratio_mode = args.ratio_to is not None
    if ratio_mode:
        h5_path2 = h5_file(args.ratio_to)
        data2 = hits_per_topology(h5_path2)
        n_both = len(data.index.intersection(data2.index))
        if n_both < max(len(data), len(data2)):
            print(f"Warning: {len(data)} topologies in {h5_path}, {len(data2)} in "
                  f"{h5_path2}, {n_both} in both; only those are plotted")

    columns = sorted(set(data.columns) | (set(data2.columns) if ratio_mode else set()),
                     key=lambda c: int(c[len("n_pmt"):]))
    if not columns:
        raise SystemExit("No LAPPD hits found.")
    data = data.reindex(columns=columns, fill_value=0)   # an LAPPD with no hits
    if ratio_mode:
        data2 = data2.reindex(columns=columns, fill_value=0)
        data, data2 = data.align(data2, join="inner", axis=0)

    if args.table:
        out = data.join(data2, rsuffix="_other") if ratio_mode else data
        out.reset_index().to_csv(args.table, sep=" ", index=False)
        print(f"Wrote {args.table}")

    table = data.reset_index()
    xpos = table["pos_x"].to_numpy()     # mm
    zpos = table["pos_z"].to_numpy()     # mm
    xDirec = table["dir_x"].to_numpy()   # dx/dy
    zDirec = table["dir_z"].to_numpy()   # dz/dy

    uniqueX, uniqueZ = np.unique(xpos), np.unique(zpos)
    numX, numZ = len(uniqueX), len(uniqueZ)
    uniqueDirecCount = len(np.unique(np.column_stack((xDirec, zDirec)), axis=0))

    def values(col):
        """Hits per event, or the ratio (NaN where OTHER_RUN has no hits)."""
        n = data[col].to_numpy(dtype=float)
        if not ratio_mode:
            return n
        d = data2[col].to_numpy(dtype=float)
        return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)

    def norm_for(v):
        if ratio_mode:
            return colors.Normalize(vmin=0, vmax=2)
        return colors.LogNorm(vmin=min(1, np.nanmax(v)), vmax=max(1, np.nanmax(v)))

    cmap = "coolwarm" if ratio_mode else "viridis"
    label = "hit ratio" if ratio_mode else "hits"
    title_tail = (f"\n{h5_path.parent.name} / {h5_path2.parent.name}" if ratio_mode
                  else f"\n{h5_path.parent.name}")

    figures = []
    if uniqueDirecCount > 1:
        # Several directions per position: a chunk of circles per position
        direc_step_size = np.diff(np.unique(np.concatenate([xDirec, zDirec]))).min()
        direc_steps = round((np.ptp(np.concatenate([xDirec, zDirec]))) / direc_step_size) + 1
        if direc_steps * direc_steps != uniqueDirecCount:
            print(f"Note: {uniqueDirecCount} directions per position, not a full "
                  f"{direc_steps}x{direc_steps} slope grid")
        spacing = min(np.diff(uniqueX).min() if numX > 1 else np.inf,
                      np.diff(uniqueZ).min() if numZ > 1 else np.inf)
        if not np.isfinite(spacing):     # a single start position
            spacing = 200.0
        diameter = np.trunc(spacing / direc_steps * 0.85)   # mm; the constant is a size multiplier
        print(f"The spacing between unique positions is: {spacing}")
        print(f"The step size between each direction vector is: {direc_step_size}")
        print(f"The radius of each circle is: {diameter / 2.0}")

        centres = np.column_stack((xpos + diameter * (xDirec / direc_step_size),
                                   zpos + diameter * (zDirec / direc_step_size)))

        for col in columns:
            v = values(col)
            drawn = ~np.isnan(v) & (v > 0 if not ratio_mode else True)
            circles = [patches.Circle(c, diameter / 2.0) for c in centres[drawn]]
            fig, ax = plt.subplots(figsize=(8, 8))
            circ_collec = PatchCollection(circles, cmap=cmap, norm=norm_for(v))
            circ_collec.set_array(np.minimum(v[drawn], 2.0) if ratio_mode else v[drawn])
            ax.add_collection(circ_collec)
            ax.set_xticks(uniqueX)
            ax.set_yticks(uniqueZ)
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_aspect("equal")
            ax.autoscale_view()
            fig.colorbar(circ_collec, fraction=0.046, pad=0.04, ax=ax, label=label)
            figures.append((col, fig, ax, v))
    else:
        # NOTE: one direction per position: colour mesh over (x, z)
        for col in columns:
            v = values(col)
            grid = (pd.DataFrame({"x": xpos, "z": zpos, "v": v})
                    .pivot(index="z", columns="x", values="v").to_numpy())
            fig, ax = plt.subplots(figsize=(8, 8))
            mesh = ax.pcolormesh(uniqueX, uniqueZ, np.ma.masked_invalid(
                np.minimum(grid, 2.0) if ratio_mode else np.where(grid > 0, grid, np.nan)),
                cmap=cmap, shading="auto", edgecolors="r", linewidths=0.5,
                norm=norm_for(v))
            ax.set_xticks(uniqueX)
            ax.set_yticks(uniqueZ)
            ax.tick_params(axis="x", labelrotation=90)
            fig.colorbar(mesh, ax=ax, label=label)
            figures.append((col, fig, ax, v))

    for col, fig, ax, v in figures:
        lappd = col[len("n_pmt"):]
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("z position (mm)")   # LitePath frame: z up, beam along +y
        ax.set_title(f"LAPPD {lappd} {'Hit Ratio' if ratio_mode else 'Hits'}{title_tail}")
        fig.tight_layout()
        if ratio_mode:
            ok = ~np.isnan(v)
            print(f"LAPPD {lappd}: {ok.sum()} topologies with a ratio, {(~ok).sum()} blank; "
                  f"median ratio {np.median(v[ok]) if ok.any() else float('nan'):.3f}")
        if args.save:
            out = f"{args.save}_lappd{lappd}.png"
            fig.savefig(out, dpi=150)
            print(f"  saved {out}")

    plt.show()


if __name__ == "__main__":
    main()
