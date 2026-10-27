#!/usr/bin/env python3
"""Event display for ANNIE batch output.

Reads ``output.h5`` produced by ``annieray batch``.  Shows a three-panel
layout:

  Top view    – top endcap PMTs (panel 9) at (x, y) in tank cross-section
  Barrel      – unrolled cylinder (φ vs Z), barrel PMTs (panels 1-8) + LAPPDs
  Bottom view – bottom endcap PMTs (panel 0) at (x, y) in tank cross-section

The barrel φ axis is centred on the octagon corner the beam looks into
(auto-detected from the track table; 135° for the WAGLE files, whose beam runs
at φ = 135°).  ``--center-phi`` overrides that choice.  The azimuth is mirrored
about the vertical plane so the barrel and endcaps read left-right as seen from
*inside* the tank (the WCSim scan→world map is left-handed; see
:func:`plot_phi`); the endcap views are also rotated so the beam points up.
``--mark-pmt ID ...`` rings/labels specific PMTs (annieray detector index or
WAGLE Tube ID) for orientation cross-checks.

It also reads the WAGLE benchmark files written by ``annieray benchmark``
(``results/benchmark/{data,mc}.h5``): those have no ``photon_hits`` table,
so hit counts and first-hit times are taken from the ``pmt_hits`` table
instead, and the track's entry/exit points are marked on the tank boundary.

Controls: ◀/▶ buttons or left/right arrow keys to navigate events.

Usage:
    python scripts/event_display.py results/
    python scripts/event_display.py results/output.h5
    python scripts/event_display.py results/benchmark/mc.h5 --charge \
        --mark-pmt 428 406
    python scripts/event_display.py results/benchmark/mc.h5 \
        --compare results/raytrace/output.h5 --compare-event 0 --charge
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

from annieray.io_h5 import load_table, read_attrs


# ---------------------------------------------------------------------------
# Handedness
# ---------------------------------------------------------------------------
# The WCSim scan -> annieray world map is left-handed in the (pmt_x, pmt_z)
# plane (``pmt_loader._scan_to_tank`` uses the WCSim-faithful
# ``Y_w = (168.1 - pmt_z) * 10``), so the world azimuth ``atan2(y, x)`` winds
# opposite to the way the detector looks when viewed from inside the tank.
# The event display mirrors the azimuth about the vertical plane so the barrel
# and endcaps read left-right the way you see them standing inside the tank.
# This is a *display-only* change: the raytracer geometry is untouched.


def plot_phi(az_deg):
    """World azimuth (deg) -> displayed azimuth, viewed from inside (mirrored)."""
    return (-np.asarray(az_deg, dtype=float)) % 360.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(h5_path, first_hit_time=False, charge=False, time=False):
    """Load hit/charge/time data from an annieray HDF5 file.

    Works for both the raytracer output (``photon_hits``) and the WAGLE
    benchmark files (``pmt_hits``).  Counts and first-hit times are returned
    as ``{(event_id, system_code, detector_index): value}`` dicts.
    """
    meta = read_attrs(h5_path)
    det = load_table(h5_path, "detectors")

    hits = load_table(h5_path, "photon_hits")
    if hits.empty:
        # Benchmark files: the raw digitised PMT hits live in pmt_hits.
        hits = load_table(h5_path, "pmt_hits")

    counts: dict = {}
    first_times: dict = {}
    if not hits.empty:
        if "detector_system" in hits.columns:
            # Raytracer output: (event_id, detector_system, detector_index).
            grp = hits.groupby(["event_id", "detector_system", "detector_index"])
            counts = {k: int(v) for k, v in grp.size().items()}
            if first_hit_time:
                first_times = {
                    k: float(v) for k, v in grp["arrival_time"].min().items()
                }
        else:
            # Benchmark output: (event_id, pmt_index); PMTs are system 0.
            grp = hits.groupby(["event_id", "pmt_index"])
            counts = {(int(e), 0, int(i)): int(v) for (e, i), v in grp.size().items()}
            if first_hit_time:
                first_times = {
                    (int(e), 0, int(i)): float(v)
                    for (e, i), v in grp["arrival_time"].min().items()
                }

    pmt_charge = pmt_time = None
    if charge or time:
        pmt_df = load_table(h5_path, "pmt_responses")
        if not pmt_df.empty:
            pmt_charge = pmt_df.set_index(["event_id", "pmt_index"])["charge"]
            pmt_time = pmt_df.set_index(["event_id", "pmt_index"])["time"]

    # Muon track (true for the raytracer, true/reconstructed for WAGLE).
    track = load_table(h5_path, "muon_truth")

    return counts, first_times, det, meta, pmt_charge, pmt_time, track


def load_track(event_id, track_df):
    """Return ``(vertex_xyz, direction_xyz)`` for an event, or ``None``."""
    if track_df is None or track_df.empty:
        return None
    rows = track_df[track_df["event_id"] == event_id]
    if rows.empty:
        return None
    r = rows.iloc[0]
    pos = np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])])
    d = np.array([float(r["dir_x"]), float(r["dir_y"]), float(r["dir_z"])])
    return pos, d


def mean_track_bearing_deg(track_df):
    """Mean azimuth (deg) of the track directions, or ``None`` if unavailable.

    Directions are summed as unit transverse vectors, so a beam that runs at a
    single azimuth (e.g. the WAGLE beam at 135 deg) averages cleanly even with
    per-event jitter.
    """
    if track_df is None or track_df.empty:
        return None
    dx = track_df["dir_x"].to_numpy(dtype=float)
    dy = track_df["dir_y"].to_numpy(dtype=float)
    if len(dx) == 0:
        return None
    sx, sy = float(np.sum(dx)), float(np.sum(dy))
    if np.hypot(sx, sy) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(sy, sx)) % 360.0)


def beam_vertex_phi(track_df, fallback=90.0, step=22.5, offset=0.0):
    """Displayed azimuth of the octagon corner the beam looks into.

    The ANNIE Phase II barrel is an octagon whose corners (where two faces
    meet) sit at ``offset + k*step``.  Measured from the PMT positions the
    corner azimuths fall on a 22.5 deg grid (16 corners), and the muon beam
    runs along one of them (135 deg for WAGLE).  This returns the corner
    nearest the mean track bearing so the unrolled barrel can be centred on the
    beam-facing region.  Falls back to *fallback* when no track information is
    available.

    The returned value is in the *displayed* (inside-view) convention, i.e. it
    has the same mirror applied as the barrel azimuth (see :func:`plot_phi`).
    The second element of the tuple is the physical mean track bearing (deg).
    """
    bearing = mean_track_bearing_deg(track_df)
    if bearing is None:
        return fallback, None
    k = round((bearing - offset) / step)
    vertex = float((offset + k * step) % 360.0)
    return float(plot_phi(vertex)), bearing


# ---------------------------------------------------------------------------
# Detector classification
# ---------------------------------------------------------------------------

# Panel number semantics (from PMT scan file):
#   0  → bottom endcap (LUX, upward-facing)
#   1–8 → barrel octagon faces (inward-facing)
#   9  → top endcap (ETEL, downward-facing)


def classify_detectors(det_df):
    """Split detectors into top/barrel-PMT/barrel-LAPPD/bottom groups."""
    top = []
    barrel_pmt = []
    barrel_lappd = []
    bottom = []

    for _, row in det_df.iterrows():
        sys_code = int(row["system_code"])
        idx = int(row["detector_index"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        label = str(row["label"])
        panel = int(row["panel"]) if "panel" in row and pd.notna(row["panel"]) else -1

        is_pmt = sys_code == 0

        if is_pmt and panel == 9:
            top.append((sys_code, idx, x, y, label))
        elif is_pmt and panel == 0:
            bottom.append((sys_code, idx, x, y, label))
        elif is_pmt:
            phi = float(plot_phi(np.degrees(np.arctan2(y, x))))
            barrel_pmt.append((sys_code, idx, phi, z, label))
        else:
            phi = float(plot_phi(np.degrees(np.arctan2(y, x))))
            barrel_lappd.append((sys_code, idx, phi, z, label))

    return top, barrel_pmt, barrel_lappd, bottom


def keys_and_coords(det_list):
    """Return ``(keys, xs, ys)`` from a detector list."""
    if not det_list:
        return np.empty((0, 2), dtype=np.int32), np.array([]), np.array([])
    keys = np.array([(e[0], e[1]) for e in det_list], dtype=np.int32)
    xs = np.array([e[2] for e in det_list])
    ys = np.array([e[3] for e in det_list])
    return keys, xs, ys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Track drawing
# ---------------------------------------------------------------------------

#: Markers used for the two points where the track crosses the tank boundary.
ENTRY_MARKER = dict(marker="o", markerfacecolor="white", markeredgecolor="crimson",
                    markeredgewidth=2.0, markersize=10, linestyle="none")
EXIT_MARKER = dict(marker="s", markerfacecolor="white", markeredgecolor="navy",
                   markeredgewidth=2.0, markersize=9, linestyle="none")


def track_endpoints(pos, direction, tank_r, z_min, z_max):
    """Return the ``(entry, exit)`` points where the track crosses the tank.

    The tank is a cylinder of radius *tank_r* spanning ``z_min…z_max``; the
    track is the straight line ``p(t) = pos + t·direction``.  The two
    parameters are solved analytically (cylinder side and endcaps), so the
    result is exact rather than sampled.

    Returns ``None`` when the line misses the tank.
    """
    pos = np.asarray(pos, dtype=float)
    d = np.asarray(direction, dtype=float)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(d)):
        return None
    nrm = np.linalg.norm(d)
    if nrm == 0:
        return None
    d = d / nrm

    lo, hi = -np.inf, np.inf

    # --- end caps: z_min <= p_z + t·d_z <= z_max ---
    if abs(d[2]) > 1e-12:
        t0 = (z_min - pos[2]) / d[2]
        t1 = (z_max - pos[2]) / d[2]
        lo = max(lo, min(t0, t1))
        hi = min(hi, max(t0, t1))
    elif not (z_min <= pos[2] <= z_max):
        return None

    # --- cylinder side: (p_x + t·d_x)^2 + (p_y + t·d_y)^2 <= tank_r^2 ---
    a = d[0] ** 2 + d[1] ** 2
    b = 2.0 * (pos[0] * d[0] + pos[1] * d[1])
    c = pos[0] ** 2 + pos[1] ** 2 - tank_r ** 2
    if a < 1e-18:
        if c > 0:
            return None            # parallel to the axis and outside
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None            # misses the cylinder entirely
        sq = np.sqrt(disc)
        lo = max(lo, (-b - sq) / (2.0 * a))
        hi = min(hi, (-b + sq) / (2.0 * a))

    if lo > hi:
        return None

    entry = pos + lo * d
    exit_ = pos + hi * d
    return entry, exit_


def rotate_xy(x, y, rot_deg):
    """Rotate ``(x, y)`` about the origin by *rot_deg* (endcap re-orientation)."""
    th = np.radians(rot_deg)
    c, s = np.cos(th), np.sin(th)
    return c * np.asarray(x) - s * np.asarray(y), s * np.asarray(x) + c * np.asarray(y)


def draw_endpoints_xy(ax, entry, exit_, rot_deg=0.0):
    """Mark the entry/exit points in an (x, y) cross-section view.

    The point is mirrored (``x -> -x``) to match the inside-view handedness and
    then rotated by *rot_deg* into the displayed endcap frame, matching the PMT
    scatter.
    """
    for pt, style in ((entry, ENTRY_MARKER), (exit_, EXIT_MARKER)):
        if pt is None:
            continue
        px, py = rotate_xy(-pt[0], pt[1], rot_deg)
        ax.plot([px], [py], zorder=6, **style)


def draw_endpoints_barrel(ax, entry, exit_, center_phi):
    """Mark the entry/exit points in the unrolled (phi, z) view.

    The physical azimuth is mirrored (see :func:`plot_phi`) and *center_phi* is
    given in the displayed convention, matching the barrel PMTs.
    """
    for pt, style in ((entry, ENTRY_MARKER), (exit_, EXIT_MARKER)):
        if pt is None:
            continue
        if np.hypot(pt[0], pt[1]) < 1.0:
            continue              # on the axis: phi is undefined
        phi = float(plot_phi(np.degrees(np.arctan2(pt[1], pt[0]))))
        phi = ((phi - center_phi + 180) % 360) - 180
        ax.plot([phi], [pt[2]], zorder=6, **style)


def endpoint_legend_handles():
    """Proxy artists for the entry/exit legend."""
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], marker=ENTRY_MARKER["marker"], linestyle="none",
               markerfacecolor=ENTRY_MARKER["markerfacecolor"],
               markeredgecolor=ENTRY_MARKER["markeredgecolor"],
               markeredgewidth=ENTRY_MARKER["markeredgewidth"],
               markersize=ENTRY_MARKER["markersize"], label="track entry"),
        Line2D([], [], marker=EXIT_MARKER["marker"], linestyle="none",
               markerfacecolor=EXIT_MARKER["markerfacecolor"],
               markeredgecolor=EXIT_MARKER["markeredgecolor"],
               markeredgewidth=EXIT_MARKER["markeredgewidth"],
               markersize=EXIT_MARKER["markersize"], label="track exit"),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class Display:
    """One loaded HDF5 file rendered as top/barrel/bottom panels."""

    def __init__(self, h5_path, args, label):
        self.path = Path(h5_path)
        self.label = label
        (self.counts, self.first_times, self.det_df, self.meta,
         self.pmt_charge, self.pmt_time, self.track) = load_data(
            self.path,
            first_hit_time=args.first_hit_time,
            charge=args.charge,
            time=args.time,
        )
        if (args.charge or args.time) and self.pmt_charge is None:
            print("Warning: no pmt_responses table in HDF5.  "
                  "Falling back to raw hit counts.")
            args.charge = args.time = False

        self.events = sorted({k[0] for k in self.counts})
        self.top, self.barrel_pmt, self.barrel_lappd, self.bottom = \
            classify_detectors(self.det_df)
        self.top_keys, self.top_xs, self.top_ys = keys_and_coords(self.top)
        self.bp_keys, self.bp_xs, self.bp_ys = keys_and_coords(self.barrel_pmt)
        self.bl_keys, self.bl_xs, self.bl_ys = keys_and_coords(self.barrel_lappd)
        self.bot_keys, self.bot_xs, self.bot_ys = keys_and_coords(self.bottom)

        # Label lookup: annieray detector_index and WAGLE Tube ID -> label.
        # The PMT label is stored as e.g. b"PMT_410"; the trailing integer is
        # the WAGLE Tube ID, which is what the user knows from the hardware.
        self.labels = {}
        for _, row in self.det_df.iterrows():
            di = int(row["detector_index"])
            lab = row["label"]
            lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else str(lab)
            self.labels[di] = lab

        self.ax_top = self.ax_barrel = self.ax_bottom = None
        self.scatters = []
        self.beam_bearing = mean_track_bearing_deg(self.track)
        self.track_points_xy = {}   # recomputed each event, for markers

    # -- values ----------------------------------------------------------
    def raw_value(self, event_id, sc, di, args):
        if args.charge and sc == 0 and self.pmt_charge is not None:
            try:
                return float(self.pmt_charge.loc[(event_id, di)])
            except KeyError:
                return 0.0
        if args.time and sc == 0 and self.pmt_time is not None:
            try:
                return float(self.pmt_time.loc[(event_id, di)])
            except KeyError:
                return 0.0
        if args.first_hit_time:
            return float(self.first_times.get((event_id, sc, di), 0.0))
        return float(self.counts.get((event_id, sc, di), 0.0))

    def color_value(self, event_id, sc, di, args):
        v = self.raw_value(event_id, sc, di, args)
        if args.time:
            return v
        if args.linear:
            return v
        return np.log10(v + 1.0)

    # -- drawing ---------------------------------------------------------
    def build_axes(self, fig, gs, row0, tank_r, tank_z_min, tank_z_max,
                   center_phi, cmap, norm, args):
        self.ax_top = fig.add_subplot(gs[row0])
        self.ax_barrel = fig.add_subplot(gs[row0 + 1])
        self.ax_bottom = fig.add_subplot(gs[row0 + 2])

        # Endcaps are mirrored (x -> -x) and rotated so the beam points up (+y)
        # in the plot, matching the barrel's inside-view handedness.  A physical
        # beam bearing b maps to displayed azimuth (180 - b) after the mirror,
        # so rotate by (b - 90) to bring it to +y (90 deg).
        if args.no_rotate_endcaps or self.beam_bearing is None:
            rot = 0.0
        else:
            rot = self.beam_bearing - 90.0
        self.endcap_rot = rot
        beam_txt = "" if self.beam_bearing is None else \
            f"  (beam at {self.beam_bearing:.0f}° \u2191)"

        # --- top view ---
        self.ax_top.add_patch(Circle((0, 0), tank_r, fill=False,
                                      color="gray", linestyle="--", linewidth=0.7))
        self.ax_top.set_aspect("equal")
        self.ax_top.set_xlim(-tank_r * 1.15, tank_r * 1.15)
        self.ax_top.set_ylim(-tank_r * 1.15, tank_r * 1.15)
        self.ax_top.set_title("Top View (panel 9 – ETEL endcap)" + beam_txt)
        self.ax_top.set_xlabel("x (mm, inside view)")
        self.ax_top.set_ylabel("y (mm)")
        if not args.no_track:
            self.ax_top.legend(handles=endpoint_legend_handles(), loc="upper right",
                               fontsize=8, framealpha=0.8)
        top_xs, top_ys = rotate_xy(-self.top_xs, self.top_ys, rot)
        s_top = self.ax_top.scatter(
            top_xs, top_ys, marker="o", s=40, c=np.zeros(len(self.top)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3,
        ) if len(self.top) else None

        # --- barrel ---
        all_z = np.concatenate([self.bp_ys, self.bl_ys])
        if len(all_z) > 0:
            z_lo, z_hi = all_z.min(), all_z.max()
            z_pad = max(100.0, (z_hi - z_lo) * 0.08)
            z_lo, z_hi = z_lo - z_pad, z_hi + z_pad
        else:
            z_lo, z_hi = tank_z_min, tank_z_max
        self.ax_barrel.set_xlim(-190.0, 190.0)
        self.ax_barrel.set_ylim(z_lo, z_hi)
        self.ax_barrel.set_title(
            f"Barrel (unrolled, inside view, φ centred on "
            f"{getattr(self, 'physical_center_phi', center_phi):.0f}°)")
        self.ax_barrel.set_xlabel("φ (degrees, inside view)")
        self.ax_barrel.set_ylabel("z (mm)")
        self.ax_barrel.axhline(tank_z_min, color="gray", linestyle=":", linewidth=0.5)
        self.ax_barrel.axhline(tank_z_max, color="gray", linestyle=":", linewidth=0.5)
        tick_vals = np.arange(-180, 181, 60)
        self.ax_barrel.set_xticks(tick_vals)
        self.ax_barrel.set_xticklabels(
            [f"{((v + center_phi + 180) % 360 - 180):.0f}°" for v in tick_vals])

        # Beam direction: the muon travels along the beam bearing; the axis
        # pointing back at the source is bearing+180.  Both are mirrored with
        # the barrel, so mark the (displayed) beam azimuth.
        if self.beam_bearing is not None:
            beam_disp = float(plot_phi(self.beam_bearing))
            beam_x = ((beam_disp - center_phi + 180) % 360) - 180
            self.ax_barrel.axvline(beam_x, color="crimson", linestyle="--",
                                   linewidth=1.0, alpha=0.8)
            self.ax_barrel.text(beam_x - 3.0, z_hi, "beam \u2190 source",
                                rotation=90, va="top", ha="right",
                                fontsize=8, color="crimson")
        # Label the octagon corner held at plot centre (the centred vertex).
        self.ax_barrel.axvline(0.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        self.ax_barrel.text(3.0, z_hi, f"vertex {center_phi:.0f}°",
                            va="top", ha="left", fontsize=8, color="dimgray")

        bp_xs = ((self.bp_xs - center_phi + 180) % 360) - 180
        bl_xs = ((self.bl_xs - center_phi + 180) % 360) - 180
        s_bp = self.ax_barrel.scatter(
            bp_xs, self.bp_ys, marker="o", s=60, c=np.zeros(len(self.barrel_pmt)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3, label="PMT",
        ) if len(self.barrel_pmt) else None
        s_bl = self.ax_barrel.scatter(
            bl_xs, self.bl_ys, marker="s", s=40, c=np.zeros(len(self.barrel_lappd)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3, label="LAPPD",
        ) if len(self.barrel_lappd) else None
        if s_bp or s_bl:
            self.ax_barrel.legend(loc="upper right", fontsize=9, markerscale=0.9)

        # Mark specific PMTs by annieray detector index or WAGLE Tube ID, so the
        # user can cross-check the orientation against the hardware.
        self.mark_keys = []
        for token in args.mark_pmt:
            di = self._resolve_pmt(token)
            if di is None:
                print(f"Warning: --mark-pmt {token!r} not found in detectors")
                continue
            self.mark_keys.append(di)
        for di in self.mark_keys:
            sel = [e for e in self.barrel_pmt if e[1] == di]
            if not sel:
                continue
            _, _, bx, by = sel[0][:4]
            mx = ((bx - center_phi + 180) % 360) - 180
            self.ax_barrel.scatter([mx], [by], marker="o", s=170, facecolors="none",
                                   edgecolors="chartreuse", linewidth=1.6, zorder=7)
            self.ax_barrel.annotate(self.labels.get(di, str(di)), (mx, by),
                                    textcoords="offset points", xytext=(7, 4),
                                    fontsize=7, color="darkgreen", zorder=8)
        if self.mark_keys:
            self.ax_barrel.legend(loc="upper right", fontsize=9, markerscale=0.9)

        # --- bottom view ---
        self.ax_bottom.add_patch(Circle((0, 0), tank_r, fill=False,
                                         color="gray", linestyle="--", linewidth=0.7))
        self.ax_bottom.set_aspect("equal")
        self.ax_bottom.set_xlim(-tank_r * 1.15, tank_r * 1.15)
        self.ax_bottom.set_ylim(-tank_r * 1.15, tank_r * 1.15)
        self.ax_bottom.set_title("Bottom View (panel 0 – LUX endcap)" + beam_txt)
        self.ax_bottom.set_xlabel("x (mm, inside view)")
        self.ax_bottom.set_ylabel("y (mm)")
        bot_xs, bot_ys = rotate_xy(-self.bot_xs, self.bot_ys, rot)
        s_bot = self.ax_bottom.scatter(
            bot_xs, bot_ys, marker="o", s=40, c=np.zeros(len(self.bottom)),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=0.3,
        ) if len(self.bottom) else None

        self.scatters = [(s_top, self.top_keys), (s_bp, self.bp_keys),
                         (s_bl, self.bl_keys), (s_bot, self.bot_keys)]

    def _resolve_pmt(self, token):
        """Resolve a --mark-pmt token to an annieray detector index.

        Accepts an annieray detector index (e.g. ``96``), a WAGLE Tube ID (e.g.
        ``428`` or ``PMT_428``), or a full label.
        """
        s = str(token)
        if s.startswith("PMT_"):
            s = s[4:]
        try:
            num = int(s)
        except ValueError:
            for di, lab in self.labels.items():
                if lab == str(token):
                    return di
            return None
        if num in self.labels:
            return num
        for di, lab in self.labels.items():
            if lab == f"PMT_{num}":
                return di
        return None

    def update(self, event_id, args, tank_r, tank_z_min, tank_z_max, center_phi):
        all_colors = []
        for scatter, keys in self.scatters:
            if scatter is None or len(keys) == 0:
                continue
            colors = np.array([self.color_value(event_id, sc, di, args)
                               for sc, di in keys])
            if args.hide_zero:
                colors[colors == 0] = np.nan
            scatter.set_array(colors)
            all_colors.extend(colors[~np.isnan(colors)]
                              if args.hide_zero else colors)

        # Track overlay: mark where the track enters and leaves the tank.
        tr = None if args.no_track else load_track(event_id, self.track)
        self.track_status = ""
        for ax in (self.ax_top, self.ax_barrel, self.ax_bottom):
            if ax is not None:
                for ln in list(ax.get_lines()):
                    ln.remove()
        if tr is not None:
            pos, d = tr
            ends = track_endpoints(pos, d, tank_r, tank_z_min, tank_z_max)
            if ends is None:
                self.track_status = "  [track misses the tank]"
            else:
                entry, exit_ = ends
                draw_endpoints_xy(self.ax_top, entry, exit_, self.endcap_rot)
                draw_endpoints_xy(self.ax_bottom, entry, exit_, self.endcap_rot)
                draw_endpoints_barrel(self.ax_barrel, entry, exit_, center_phi)
        return all_colors


def main():
    parser = argparse.ArgumentParser(description="ANNIE event display")
    parser.add_argument("input", type=Path, default=Path("results"),
                        nargs="?",
                        help="Batch output dir or output.h5 path (default: results/)")
    parser.add_argument("--linear", action="store_true",
                        help="Linear color scale (default: log\u2081\u2080)")
    parser.add_argument("--hide-zero", action="store_true",
                        help="Show PMTs with zero charge/hits as white")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--first-hit-time", action="store_true",
                      help="Color by first-hit arrival time (ns)")
    mode.add_argument("--charge", action="store_true",
                      help="Color PMTs by integrated charge (PE) from digital model")
    mode.add_argument("--time", action="store_true",
                      help="Color PMTs by response time (ns) from digital model")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Second HDF5 file to show below (e.g. a raytrace)")
    parser.add_argument("--compare-event", type=int, default=0,
                        help="Event id to show in the --compare file")
    parser.add_argument("--no-track", action="store_true",
                        help="Do not draw the muon track")
    parser.add_argument("--no-rotate-endcaps", action="store_true",
                        help="Keep the endcap (x, y) views in the tank frame. "
                             "By default they are rotated so the beam points up.")
    parser.add_argument("--mark-pmt", type=str, nargs="+", default=[],
                        metavar="ID",
                        help="Ring and label the given PMTs on the barrel to "
                             "cross-check the orientation.  Accepts an annieray "
                             "detector index or a WAGLE Tube ID (e.g. 428 or "
                             "PMT_428).")
    parser.add_argument("--center-phi", type=float, default=None,
                        help="Azimuth (deg) the unrolled barrel is centred on. "
                             "Default: auto-detect the octagon corner the beam "
                             "looks into (135 deg for WAGLE, whose beam runs at "
                             "phi=135 in the annieray world frame).")
    args = parser.parse_args()

    if args.input.suffix == ".h5":
        h5_path = args.input
    else:
        h5_path = args.input / "output.h5"
    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return
    print(f"HDF5: {h5_path}")

    primary = Display(h5_path, args, h5_path.stem)
    if not primary.events:
        print("No events with hits found.")
        return
    print(f"{len(primary.det_df)} detectors, {len(primary.events)} events")

    cmp_display = None
    if args.compare:
        if not args.compare.exists():
            print(f"Error: {args.compare} not found")
            return
        print(f"Compare: {args.compare}")
        cmp_display = Display(args.compare, args, args.compare.stem)

    meta = primary.meta
    tank_r = float(meta.get("tank_radius_mm", 1524.0))
    tank_z_min = float(meta.get("tank_z_min_mm", 19.0))
    tank_z_max = float(meta.get("tank_z_max_mm", 3861.0))

    if args.center_phi is not None:
        # The flag is given in the physical convention; the barrel is displayed
        # mirrored, so negate it for plotting.  Remember the physical value for
        # the axis label so the flag stays interpretable.
        physical_center_phi = args.center_phi
        center_phi = float(plot_phi(args.center_phi))
    else:
        center_phi, bearing = beam_vertex_phi(primary.track, fallback=90.0)
        physical_center_phi = float(plot_phi(center_phi))
        if bearing is not None:
            print(f"Auto-centred barrel on beam vertex {physical_center_phi:.0f}° "
                  f"(mean track bearing {bearing:.1f}°)")
        else:
            print(f"No track info: barrel centred on default "
                  f"{physical_center_phi:.0f}°")

    primary.physical_center_phi = physical_center_phi
    if cmp_display is not None:
        cmp_display.physical_center_phi = physical_center_phi

    n_rows = 6 if cmp_display is not None else 3
    height = 19.0 if cmp_display is not None else 10.0
    fig = plt.figure(figsize=(12, height))
    gs = fig.add_gridspec(
        n_rows, 1,
        height_ratios=[1, 1.5, 1] * (n_rows // 3),
        hspace=0.45,
        left=0.08, right=0.88, top=0.96, bottom=0.06,
    )

    cmap = plt.cm.plasma.copy()
    if args.hide_zero:
        cmap.set_bad("white")
    norm = Normalize(vmin=0, vmax=None)

    primary.build_axes(fig, gs, 0, tank_r, tank_z_min, tank_z_max,
                       center_phi, cmap, norm, args)
    if cmp_display is not None:
        cmp_display.build_axes(fig, gs, 3, tank_r, tank_z_min, tank_z_max,
                               center_phi, cmap, norm, args)

    def colorbar_label() -> str:
        if args.charge:
            return "charge (PE)" if args.linear else "log\u2081\u2080(charge + 1) (PE)"
        if args.time:
            return "response time (ns)"
        if args.first_hit_time:
            return "first hit time (ns)"
        return "log\u2081\u2080(n_hits + 1)" if not args.linear else "n_hits"

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax,
                 label=colorbar_label())

    current = [0]
    title_text = fig.text(0.01, 0.995, "", ha="left", va="top",
                           fontsize=13, fontweight="bold")

    def update(event_id):
        all_colors = primary.update(event_id, args, tank_r, tank_z_min,
                                    tank_z_max, center_phi)
        if cmp_display is not None:
            all_colors += cmp_display.update(args.compare_event, args, tank_r,
                                             tank_z_min, tank_z_max, center_phi)
        norm.vmax = max(all_colors) if all_colors else 1
        title_text.set_text(
            f"{primary.label}: event {event_id} "
            f"({current[0] + 1}/{len(primary.events)})"
            f"{primary.track_status}"
            + (f"   |   {cmp_display.label}: event {args.compare_event}"
               f"{cmp_display.track_status}" if cmp_display is not None else ""))
        fig.canvas.draw_idle()

    def next_event(_):
        current[0] = (current[0] + 1) % len(primary.events)
        update(primary.events[current[0]])

    def prev_event(_):
        current[0] = (current[0] - 1) % len(primary.events)
        update(primary.events[current[0]])

    def on_key(event):
        if event.key == "right":
            next_event(event)
        elif event.key == "left":
            prev_event(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    btn_prev = Button(fig.add_axes([0.35, 0.015, 0.10, 0.02]), "\u25c0 Prev")
    btn_next = Button(fig.add_axes([0.55, 0.015, 0.10, 0.02]), "Next \u25b6")
    btn_prev.on_clicked(prev_event)
    btn_next.on_clicked(next_event)

    update(primary.events[0])
    print("Controls: \u25c0/\u25b6 buttons or left/right arrow keys")
    plt.show()


if __name__ == "__main__":
    main()
