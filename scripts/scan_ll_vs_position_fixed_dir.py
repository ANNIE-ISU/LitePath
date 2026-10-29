"""Scan log-likelihood vs muon vertex position at fixed direction.

At each (x, y) or (x, z) position, evaluates the charge-only Poisson
log-likelihood using a single on-the-fly raytracing call (fixed direction).

Plots ΔLL = LL − LL_best as a 2-D heatmap.

Usage:
    python scripts/scan_ll_vs_position_fixed_dir.py test_fit/output.h5 \\
        --grid-x "-500 500 11" --grid-z "1500 2500 11" --show

    python scripts/scan_ll_vs_position_fixed_dir.py test_fit/output.h5 \\
        --grid-x "-500 500 11" --grid-y "0 1000 11" --show
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from litepath.hits import DET_SYS_PMT, HDI, HDS, H_ARRIVAL
from litepath.geometry.build import build_geometry
from litepath.simulate import trace_cherenkov
from litepath.fitting import load_observed_event
from litepath.likelihood import poisson_charge_ll


def _parse_range(s: str) -> tuple[float, float, int]:
    parts = s.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 'start stop steps', got '{s}'")
    return float(parts[0]), float(parts[1]), int(parts[2])


def _dir_from_angles(theta_deg: float, phi_deg: float
                     ) -> tuple[float, float, float]:
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    st = math.sin(th)
    return (st * math.cos(ph), st * math.sin(ph), -math.cos(th))


def _count_hits_per_pmt(hits: np.ndarray) -> dict[int, int]:
    pmt_mask = np.abs(hits[:, HDS] - DET_SYS_PMT) < 0.5
    pmt_hits = hits[pmt_mask]
    if len(pmt_hits) == 0:
        return {}
    indices = pmt_hits[:, HDI].astype(np.int32)
    unique = np.unique(indices)
    counts: dict[int, int] = {}
    for d_idx in unique:
        counts[int(d_idx)] = int(np.sum(indices == d_idx))
    return counts


def main() -> None:
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    parser = argparse.ArgumentParser(
        description="Scan log-likelihood vs muon position at fixed direction"
    )

    # Input
    parser.add_argument("h5", type=Path,
                        help="Path to batch output HDF5 file")
    parser.add_argument("--event", type=int, default=0,
                        help="Event ID to fit")

    # Geometry
    parser.add_argument("--gdml", type=Path,
                        default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                        help="Path to GDML geometry mesh")
    parser.add_argument("--pmt-csv", type=Path, default=None,
                        help="Path to PMT scan file or CSV")
    parser.add_argument("--step", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--no-lappd", action="store_true")
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--lappd-model", choices=["default", "annie"],
                        default="annie")
    parser.add_argument("--det-rotation", type=float, default=22.5)
    parser.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3])

    # Spatial grid — provide exactly two of: x, y, z
    parser.add_argument("--grid-x", type=str, default="0 2000 11",
                        help="X range: 'start stop steps' (mm)")
    parser.add_argument("--grid-y", type=str, default=None,
                        help="Y range: 'start stop steps' (mm).  "
                             "If omitted, use a fixed value (--fix-y).")
    parser.add_argument("--grid-z", type=str, default=None,
                        help="Z range: 'start stop steps' (mm).  "
                             "If omitted, use a fixed value (--fix-z).")
    parser.add_argument("--fix-y", type=float, default=None,
                        help="Fixed Y coordinate (default: true y, fallback 0)")
    parser.add_argument("--fix-z", type=float, default=None,
                        help="Fixed Z coordinate (default: true z, fallback 1940)")

    # Fixed direction
    parser.add_argument("--fix-theta", type=float, default=None,
                        help="Fixed θ (deg, default: from muon_truth)")
    parser.add_argument("--fix-phi", type=float, default=None,
                        help="Fixed φ (deg, default: from muon_truth)")

    # Photons per cm
    parser.add_argument("--photons-per-cm", type=int, default=None,
                        help="Photons per cm (default: auto-detect)")

    # RNG
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--clip", type=float, default=None,
                        help="Colormap range: [best_LL − clip, best_LL].  "
                             "If not set, auto-scale to the 5th–95th percentiles.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save results to NPZ file")
    parser.add_argument("--show", action="store_true",
                        help="Show heatmap of ΔLL vs position")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine scan mode and fixed coordinate
    # ------------------------------------------------------------------
    have_x = args.grid_x is not None
    have_y = args.grid_y is not None
    have_z = args.grid_z is not None

    provided = sum([have_x, have_y, have_z])
    if provided != 2:
        parser.error("Provide exactly two of --grid-x, --grid-y, --grid-z")
    if have_x and have_y and have_z:
        parser.error("Cannot scan three axes; provide exactly two")

    # The missing axis is held fixed
    if have_x and have_y:
        scan_mode = "xy"
        missing_axis = "z"
        x_range = _parse_range(args.grid_x)
        y_range = _parse_range(args.grid_y)
        x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
        y_vals = np.linspace(y_range[0], y_range[1], y_range[2])
    else:
        scan_mode = "xz"
        missing_axis = "y"
        x_range = _parse_range(args.grid_x)
        z_range = _parse_range(args.grid_z)
        x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
        z_vals = np.linspace(z_range[0], z_range[1], z_range[2])
        y_vals = z_vals  # reuse for the second axis in plotting

    n_x = len(x_vals)
    n_yz = len(y_vals)
    n_total = n_x * n_yz

    print(f"Scan mode: {scan_mode} — {n_x}×{n_yz} = {n_total} positions")

    # ------------------------------------------------------------------
    # Build geometry
    # ------------------------------------------------------------------
    pmt_csv = args.pmt_csv
    if pmt_csv is None:
        candidate = Path("PMTPositions_Scan.txt")
        if candidate.exists():
            pmt_csv = candidate

    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(
        args.gdml,
        step_path=args.step,
        manifest_path=args.manifest,
        pmt_csv_path=pmt_csv,
        no_lappd=args.no_lappd,
        z_offset=args.z_offset,
        lappd_model=args.lappd_model,
        det_rotation_deg=args.det_rotation,
        n_surfboards=args.surfboard,
    )

    # ------------------------------------------------------------------
    # Load observed data
    # ------------------------------------------------------------------
    print(f"Loading observed data for event {args.event}...")
    observed = load_observed_event(args.h5, args.event)

    print(f"  True direction: θ={observed.true_theta:.1f}°, "
          f"φ={observed.true_phi:.1f}°")
    if observed.true_pos:
        print(f"  True vertex: "
              f"({observed.true_pos[0]:.0f}, {observed.true_pos[1]:.0f}, "
              f"{observed.true_pos[2]:.0f}) mm")
    print(f"  {len(observed.hit_pmt_indices)} / "
          f"{len(observed.all_pmt_indices)} PMTs hit")

    # ------------------------------------------------------------------
    # Fixed direction
    # ------------------------------------------------------------------
    fix_theta = args.fix_theta if args.fix_theta is not None else observed.true_theta
    fix_phi = args.fix_phi if args.fix_phi is not None else observed.true_phi
    if fix_theta is None:
        print("Error: no known direction.  Provide --fix-theta/--fix-phi.")
        return
    muon_dir = _dir_from_angles(fix_theta, fix_phi)
    print(f"  Fixed direction: θ={fix_theta:.1f}°, φ={fix_phi:.1f}°")

    # ------------------------------------------------------------------
    # Fixed missing coordinate
    # ------------------------------------------------------------------
    true_pos = observed.true_pos or (0, 0, 1940)
    if missing_axis == "y":
        fix_val = args.fix_y if args.fix_y is not None else true_pos[1]
        print(f"  Fixed y = {fix_val:.0f} mm")
    else:
        fix_val = args.fix_z if args.fix_z is not None else true_pos[2]
        print(f"  Fixed z = {fix_val:.0f} mm")

    # ------------------------------------------------------------------
    # Photons per cm
    # ------------------------------------------------------------------
    ppcm = args.photons_per_cm or observed.true_photons_per_cm or 150
    if args.photons_per_cm is None:
        print(f"  Photons per cm: auto-detected as {ppcm}")
    else:
        print(f"  Photons per cm: {ppcm} (user-specified)")

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------
    ll = np.full((n_x, n_yz), -np.inf)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"\nScanning {n_total} positions...")

    for ix, x in enumerate(x_vals):
        for iy, yz in enumerate(y_vals):
            n_done = ix * n_yz + iy
            elapsed = time.time() - t_start
            if n_done > 0 and n_done % 10 == 0:
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{n_total}] "
                      f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining")

            if scan_mode == "xy":
                vertex = (x, yz, fix_val)
            else:
                vertex = (x, fix_val, yz)

            hits = trace_cherenkov(
                (*vertex, 0.0),  # t0 = 0 (time offsets don't affect charge LL)
                muon_dir,
                photons_per_cm=ppcm,
                geometry=geom,
                rng=rng,
            )
            expected_counts = _count_hits_per_pmt(hits)
            ll[ix, iy] = poisson_charge_ll(
                observed.pmt_counts,
                expected_counts,
                observed.all_pmt_indices,
            )

    elapsed = time.time() - t_start
    print(f"  Done — {n_total} positions in {elapsed:.0f}s "
          f"({n_total / elapsed:.2f} pos/s)")

    # Best position
    best_flat = int(np.argmax(ll))
    best_ix = best_flat // n_yz
    best_iy = best_flat % n_yz
    best_ll = ll[best_ix, best_iy]

    # ΔLL
    dll = ll - best_ll

    if scan_mode == "xy":
        best_x = x_vals[best_ix]
        best_y = y_vals[best_iy]
        best_z = fix_val
        true_x = true_pos[0]
        true_y = true_pos[1]
        true_z = true_pos[2]
    else:
        best_x = x_vals[best_ix]
        best_y = fix_val
        best_z = y_vals[best_iy]
        true_x = true_pos[0]
        true_y = true_pos[1]
        true_z = true_pos[2]

    print(f"\nBest position: x={best_x:.0f}, y={best_y:.0f}, z={best_z:.0f} mm, "
          f"LL = {best_ll:.1f}")
    print(f"  True position: x={true_x:.0f}, y={true_y:.0f}, z={true_z:.0f} mm")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.save:
        np.savez(args.save,
                 x_vals=x_vals, yz_vals=y_vals,
                 scan_mode=scan_mode,
                 fix_val=fix_val,
                 ll=ll,
                 dll=dll,
                 best_x=best_x, best_y=best_y, best_z=best_z,
                 true_x=true_x, true_y=true_y, true_z=true_z,
                 fix_theta=fix_theta, fix_phi=fix_phi,
                 fix_y=best_y if scan_mode == "xz" else None,
                 fix_z=best_z if scan_mode == "xy" else None,
                 )
        print(f"  Saved to {args.save}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if args.show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        XX, YY = np.meshgrid(x_vals, y_vals, indexing="ij")

        fig, ax = plt.subplots(figsize=(7, 6))

        im = ax.pcolormesh(XX, YY, ll, cmap="inferno_r",
                           shading="auto")
        if args.clip is not None:
            im.set_clim(best_ll - args.clip, best_ll)
        else:
            vmin, vmax = np.percentile(ll, [5, 95])
            if vmax - vmin < 1:
                vmin, vmax = ll.min(), ll.max()
            im.set_clim(vmin, vmax)

        # Markers
        if scan_mode == "xy":
            ax.plot(true_x, true_y, marker="D", color="cyan",
                    markersize=10, mew=1.5, mec="white", zorder=5,
                    label="True")
            ax.plot(best_x, best_y, marker="*", color="lime",
                    markersize=14, mew=1.5, mec="white", zorder=5,
                    label="Best fit")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        else:
            ax.plot(true_x, true_z, marker="D", color="cyan",
                    markersize=10, mew=1.5, mec="white", zorder=5,
                    label="True")
            ax.plot(best_x, best_z, marker="*", color="lime",
                    markersize=14, mew=1.5, mec="white", zorder=5,
                    label="Best fit")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")

        ax.set_title(f"Log-likelihood  (best: {best_ll:.0f}), "
                     f"θ={fix_theta:.0f}° φ={fix_phi:.0f}°")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, label="Log-likelihood")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
