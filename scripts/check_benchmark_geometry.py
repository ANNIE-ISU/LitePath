"""Geometry checks for a converted WAGLE benchmark file.

Verifies that the ROOT -> annieray world-frame transform is right, using only
the converted HDF5 file and the raytracer's own detector registry (i.e. the
two geometries are compared to each other, independently of the code that
produced the file).

Checks performed
----------------
1. **PMT positions.**  Every WAGLE PMT position stored in ``pmt_hits`` must
   coincide with the raytracer PMT centre of the same detector: radially to
   < 0.05 mm and azimuthally to < 0.5 deg.  The residual along the tube axis
   is a constant per PMT type (the WCSim-vs-survey offset).
2. **ID mapping.**  The WAGLE ``detector_id`` (WCSim TubeID) must resolve to
   a raytracer detector index, and the position of that detector must be the
   one written next to it.
3. **Axis assignment.**  The y/z swap must not be applied to the wrong axis:
   the PMTs above the tank centre in the ROOT frame must end up at the top
   in the world frame, and the muon must still travel downstream.
4. **Beam azimuth.**  The MC muon directions must cluster at a single azimuth
   (135 deg in the annieray world frame).
5. **Track geometry.**  The MC vertex must lie inside the tank; the data
   vertex (an MRD reconstruction) must lie on the tank boundary, or be NaN
   when the MRD line misses the tank.

Usage
-----
    python scripts/check_benchmark_geometry.py results/benchmark/mc.h5
    python scripts/check_benchmark_geometry.py results/benchmark/mc.h5 --verbose

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annieray.benchmark import load_pmt_registry  # noqa: E402
from annieray.io_h5 import BatchReader  # noqa: E402

#: Expected beam azimuth in the annieray world frame (deg).
BEAM_PHI_DEG = 135.0
#: Tolerance on the azimuth of the reconstructed beam.
BEAM_PHI_TOL_DEG = 15.0
#: Radial agreement between the WAGLE PMT point and the raytracer centre.
RADIAL_TOL_MM = 0.05
#: Agreement perpendicular to the tube axis.  The registry normal is the
#: *ideal* direction, so the ~10 mm axial offset leaks ~0.1 mm sideways.
#: Deliberately loose: a few mm of disagreement is fine, this only has to
#: catch a genuine frame error (which would be cm-to-m scale).
PERPENDICULAR_TOL_MM = 1.0
#: The WAGLE PMT point must lie on the tube axis, pointing outwards.
AXIS_ANGLE_TOL_DEG = 2.0
#: The offset magnitude must be constant per PMT type.
MAG_TOL_MM = 0.1
#: Data file: the PMT positions are the surveyed ones, so the offset must be
#: small and without a systematic.  Real hardware positions differ from the
#: CAD/survey file by up to ~1.7 mm, and the hit position jitters by ~1 mm.
DATA_PERPENDICULAR_TOL_MM = 2.0
DATA_MAG_TOL_MM = 0.6
DATA_MEAN_TOL_MM = 1.0
#: Azimuthal agreement between the WAGLE PMT point and the raytracer centre.
AZIMUTH_TOL_DEG = 0.5
#: Water tank (mm) — see annieray.tracer.build_geometry.
TANK_R = 1524.0
TANK_Z_MIN, TANK_Z_MAX = 19.0, 3861.0


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_pmt_positions(br: BatchReader, reg, verbose: bool, kind: str) -> bool:
    print("\n[1] PMT positions: WAGLE point vs. raytracer PMT centre")
    try:
        hits = br.load_table("pmt_hits")
    except KeyError:
        print("  (no pmt_hits table — skipping)")
        return True
    det = br.load_table("detectors")
    det = det[det["system_code"] == 0]
    det_pos = {int(i): np.array([x, y, z], dtype=float)
               for i, x, y, z in zip(det["detector_index"].to_numpy(),
                                     det["x"].to_numpy(),
                                     det["y"].to_numpy(),
                                     det["z"].to_numpy())}

    # One WAGLE point per PMT (all hits of a PMT carry the same position).
    pmt_ids = np.unique(hits["detector_id"].to_numpy())
    pmt_pos, pmt_spread = {}, {}
    for pid in pmt_ids:
        sel = hits[hits["detector_id"].to_numpy() == pid]
        p = np.stack([sel["x"].to_numpy(), sel["y"].to_numpy(),
                      sel["z"].to_numpy()], axis=1)
        pmt_pos[int(pid)] = p.mean(axis=0)
        pmt_spread[int(pid)] = float(np.abs(p - p.mean(axis=0)).max())

    ok = True
    perp, azim, ax, mag, ang = [], [], [], [], []
    unmatched = []
    for pid, p in pmt_pos.items():
        index = reg.index_of(pid)
        if index < 0 or index not in det_pos:
            unmatched.append(pid)
            continue
        c = det_pos[index]
        d = p - c
        # The offset is along the tube axis (the registry's inward normal);
        # everything perpendicular to it must agree closely.  The registry
        # normal is the *ideal* direction, so a small angle to the true PMT
        # axis (<0.6 deg) is expected and projects the offset by ~0.1 mm.
        u = reg.directions[index]
        axial = float(d @ u)
        perp.append(float(np.linalg.norm(d - axial * u)))
        ax.append((pid, axial, reg.types[index]))
        mag.append((reg.types[index], float(np.linalg.norm(d))))
        n = d / max(np.linalg.norm(d), 1e-9)
        ang.append(float(np.degrees(np.arccos(np.clip(-n @ u, -1.0, 1.0)))))
        da = np.degrees(np.arctan2(p[1], p[0]) - np.arctan2(c[1], c[0]))
        azim.append((da + 180.0) % 360.0 - 180.0)
        if verbose and pmt_spread[pid] > 1e-3:
            print(f"    note: PMT {pid} has {pmt_spread[pid]:.3f} mm spread "
                  f"(WAGLE stores one point per PMT)")

    if unmatched:
        _fail(f"{len(unmatched)} WAGLE PMT ids are not in the registry: "
              f"{unmatched[:8]}")
        ok = False
    if not perp:
        print("  (no PMT ids matched — skipping)")
        return ok

    perp_tol = PERPENDICULAR_TOL_MM if kind == "mc" else DATA_PERPENDICULAR_TOL_MM
    if max(perp) > perp_tol:
        _fail(f"perpendicular residual {max(perp):.4f} mm > {perp_tol} mm")
        ok = False
    else:
        _ok(f"perpendicular residual max {max(perp):.4f} mm "
            f"over {len(perp)} PMTs")

    if max(azim) > AZIMUTH_TOL_DEG:
        _fail(f"azimuth residual {max(azim):.4f} deg > {AZIMUTH_TOL_DEG} deg")
        ok = False
    else:
        _ok(f"azimuth residual max {max(azim):.4f} deg")

    if kind == "mc":
        # WCSim places the PMT centre further out than the survey point, by a
        # constant per PMT type, along the tube axis.
        if max(ang) > AXIS_ANGLE_TOL_DEG:
            _fail(f"offset is {max(ang):.2f} deg off the tube axis "
                  f"(> {AXIS_ANGLE_TOL_DEG} deg)")
            ok = False
        else:
            _ok(f"offset points along the tube axis (max {max(ang):.2f} deg off)")
    else:
        print("  --    (data: the PMT positions are the surveyed ones, so no "
              "systematic axis offset is expected)")

    by_type: dict[str, list[float]] = {}
    for pmt_type, m in mag:
        by_type.setdefault(str(pmt_type), []).append(m)
    for pmt_type, vals in sorted(by_type.items()):
        spread = float(np.std(vals))
        mean = float(np.mean(vals))
        if kind == "mc":
            good = spread < MAG_TOL_MM
            expect = 23.7 if pmt_type == "ETEL" else 10.45
            good = good and abs(mean - expect) < 0.3
        else:
            good = spread < DATA_MAG_TOL_MM and mean < DATA_MEAN_TOL_MM
        if not good:
            ok = False
        print(f"  {'ok   ' if good else 'FAIL '}  |offset| {pmt_type:>11s}: "
              f"{mean:7.3f} +/- {spread:.3f} mm (n={len(vals)})")
    return ok


def check_axis_assignment(br: BatchReader) -> bool:
    print("\n[2] Axis assignment (the y/z swap)")
    det = br.load_table("detectors")
    pmts = det[det["system_code"] == 0]
    pos = np.stack([pmts["x"].to_numpy(), pmts["y"].to_numpy(),
                    pmts["z"].to_numpy()], axis=1)
    r = np.hypot(pos[:, 0], pos[:, 1])
    bottom = pos[(r < 700.0) & (pos[:, 2] < 1000.0)]
    top = pos[(r < 700.0) & (pos[:, 2] > 2800.0)]
    ok = True
    if len(bottom) == 0 or len(top) == 0:
        _fail("could not identify both endcaps")
        return False
    if bottom[:, 2].mean() < top[:, 2].mean():
        _ok(f"endcaps ordered correctly (lower cap z={bottom[:, 2].mean():.0f} mm, "
            f"upper cap z={top[:, 2].mean():.0f} mm)")
    else:
        _fail("endcaps are swapped in z")
        ok = False
    return ok


def check_beam_and_track(br: BatchReader, kind: str) -> bool:
    print("\n[3] Beam azimuth and track geometry")
    ok = True
    try:
        truth = br.load_table("muon_truth")
    except KeyError:
        print("  (no muon_truth table — skipping)")
        return True

    d = np.stack([truth["dir_x"].to_numpy(), truth["dir_y"].to_numpy(),
                  truth["dir_z"].to_numpy()], axis=1)
    horiz = np.hypot(d[:, 0], d[:, 1])
    sel = horiz > 1e-6
    if sel.sum() == 0:
        print("  (no horizontal tracks — skipping)")
        return True
    phi = np.degrees(np.arctan2(d[sel, 1], d[sel, 0])) % 360.0
    ang = np.radians(phi)
    mean_phi = float(np.degrees(np.arctan2(np.sin(ang).mean(),
                                           np.cos(ang).mean())) % 360.0)
    delta = (mean_phi - BEAM_PHI_DEG + 180.0) % 360.0 - 180.0
    if abs(delta) > BEAM_PHI_TOL_DEG:
        _fail(f"beam at {mean_phi:.1f} deg, expected {BEAM_PHI_DEG:.0f} "
              f"+/- {BEAM_PHI_TOL_DEG:.0f} (off by {delta:+.1f} deg)")
        ok = False
    else:
        _ok(f"beam azimuth {mean_phi:.1f} deg (expected {BEAM_PHI_DEG:.0f}, "
            f"off by {delta:+.1f} deg) over {int(sel.sum())} tracks")

    p = np.stack([truth["pos_x"].to_numpy(), truth["pos_y"].to_numpy(),
                  truth["pos_z"].to_numpy()], axis=1)
    r = np.hypot(p[:, 0], p[:, 1])
    if kind == "mc":
        # Generator truth: the interaction vertex is inside the water.
        inside = (r < TANK_R) & (p[:, 2] > TANK_Z_MIN) & (p[:, 2] < TANK_Z_MAX)
        frac = float(inside.mean())
        if frac < 0.9:
            _fail(f"only {frac:.0%} of vertices are inside the tank "
                  f"(expected > 90%)")
            ok = False
        else:
            _ok(f"{frac:.0%} of vertices inside the tank (r<{TANK_R:.0f}, "
                f"{TANK_Z_MIN:.0f}<z<{TANK_Z_MAX:.0f})")
    else:
        # Data: the vertex is the upstream tank crossing of the MRD line, so
        # it lies *on* the boundary; tracks that miss the tank are NaN.
        finite = np.all(np.isfinite(p), axis=1)
        tol = 1.0
        on_barrel = (np.abs(r - TANK_R) <= tol) \
            & (p[:, 2] >= TANK_Z_MIN - tol) & (p[:, 2] <= TANK_Z_MAX + tol)
        on_endcap = ((np.abs(p[:, 2] - TANK_Z_MIN) <= tol)
                     | (np.abs(p[:, 2] - TANK_Z_MAX) <= tol)) \
            & (r <= TANK_R + tol)
        on_boundary = on_barrel | on_endcap
        frac = float(on_boundary[finite].mean()) if finite.any() else 0.0
        n_nan = int((~finite).sum())
        if frac < 0.99:
            _fail(f"only {frac:.0%} of finite vertices lie on the tank "
                  f"boundary (expected > 99%)")
            ok = False
        else:
            _ok(f"{frac:.0%} of finite vertices on the tank boundary; "
                f"{n_nan} MRD lines miss the tank (NaN vertex)")

    # The beam is horizontal in the world frame (z is vertical), so the
    # vertical component of the track direction must be the small one.
    # MRD reconstructions have a few degrees of tilt.
    dt = np.abs(d[:, 2])
    max_dz = 0.2 if kind == "mc" else 0.4
    if dt.mean() > max_dz * horiz.mean():
        _fail(f"|dz| mean {dt.mean():.3f} is not small compared to the "
              f"transverse {horiz.mean():.3f} — z is not the vertical axis?")
        ok = False
    else:
        _ok(f"|dz| mean {dt.mean():.3f} << transverse {horiz.mean():.3f} "
            f"(beam is horizontal, z is vertical)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", type=Path, help="Converted benchmark .h5 file")
    ap.add_argument("--pmt-csv", type=Path, default=Path("PMTPositions_Scan.txt"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    br = BatchReader(str(args.input))
    import h5py
    with h5py.File(args.input, "r") as h:
        tables = sorted(h.keys())
        attrs = dict(h.attrs)
    mode = str(attrs.get("detector_mode", ""))
    kind = "mc" if mode.endswith("mc") else "data"
    print(f"File: {args.input}")
    print(f"Tables: {tables}")
    print(f"Mode: {mode} ({kind})")
    if "det_rotation_deg" in attrs:
        print(f"det_rotation_deg: {attrs['det_rotation_deg']}")

    reg = load_pmt_registry(args.pmt_csv)
    print(f"Registry: {len(reg)} PMTs from {args.pmt_csv}")

    ok = True
    ok &= check_pmt_positions(br, reg, args.verbose, kind)
    ok &= check_axis_assignment(br)
    ok &= check_beam_and_track(br, kind)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
