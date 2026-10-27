"""Scan best-fit log-likelihood vs muon vertex position.

At each (x, z) position on a grid, runs a direction grid scan and records
the best-fit log-likelihood (and best θ, φ).  Produces a 2D heatmap.

Usage:
    python scripts/scan_ll_vs_position.py test_fit/output.h5 --event 0 --show

    python scripts/scan_ll_vs_position.py test_fit/output.h5 --event 0 \\
        --grid-x "0 2000 5" --grid-z "0 2000 5" \\
        --grid-theta "0 180 9" --grid-phi "0 360 9" --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from litepath.tracer import build_geometry
from litepath.fitting import load_observed_event, grid_scan_direction


def _parse_range(s: str) -> tuple[float, float, int]:
    parts = s.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 'start stop steps', got '{s}'")
    return float(parts[0]), float(parts[1]), int(parts[2])


def main() -> None:
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    parser = argparse.ArgumentParser(
        description="Scan best-fit log-likelihood vs muon vertex position"
    )

    # Input
    parser.add_argument("h5", type=Path,
                        help="Path to batch output HDF5 file")
    parser.add_argument("--event", type=int, default=0,
                        help="Event ID to fit")

    # Geometry (mirrors fit subcommand in cli.py)
    parser.add_argument("--gdml", type=Path,
                        default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                        help="Path to GDML geometry mesh")
    parser.add_argument("--pmt-csv", type=Path, default=None,
                        help="Path to PMT scan file or CSV")
    parser.add_argument("--step", type=Path, default=None,
                        help="Path to STEP CAD file")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to cached component manifest JSON")
    parser.add_argument("--no-lappd", action="store_true",
                        help="Skip LAPPD rectangles")
    parser.add_argument("--z-offset", type=float, default=0.0,
                        help="Vertical offset (mm)")
    parser.add_argument("--lappd-model",
                        choices=["default", "annie"], default="annie",
                        help="LAPPD geometry model")
    parser.add_argument("--det-rotation", type=float, default=22.5,
                        help="Global Z-rotation (deg)")
    parser.add_argument("--surfboard", type=int, default=0,
                        choices=[0, 1, 3],
                        help="Number of obscurant PVC surfboards")

    # Spatial grid
    parser.add_argument("--grid-x", type=str, default="0 2000 5",
                        help="X range: 'start stop steps' (mm)")
    parser.add_argument("--grid-z", type=str, default="0 2000 5",
                        help="Z range: 'start stop steps' (mm)")
    parser.add_argument("--fix-y", type=float, default=0,
                        help="Fixed Y coordinate for all scan positions (mm)")

    # Direction grid (coarse by default for speed)
    parser.add_argument("--grid-theta", type=str, default="0 180 9",
                        help="Theta range: 'start stop steps' in deg")
    parser.add_argument("--grid-phi", type=str, default="0 360 9",
                        help="Phi range: 'start stop steps' in deg")

    # Fit options
    parser.add_argument("--photons-per-cm", type=int, default=None,
                        help="Photons per cm (default: auto-detect)")
    parser.add_argument("--fix-t0", type=float, default=None,
                        help="Fixed t0 in ns (default: from muon_truth)")
    parser.add_argument("--use-time", action="store_true",
                        help="Include time residual likelihood")
    parser.add_argument("--time-sigma", type=float, default=None,
                        help="Global timing sigma (ns)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scale factor for time likelihood")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-evaluation progress in direction scans")

    # Output
    parser.add_argument("--save", type=Path, default=None,
                        help="Save results to NPZ file")
    parser.add_argument("--show", action="store_true",
                        help="Show heatmap of best LL vs position")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve PMT CSV
    # ------------------------------------------------------------------
    pmt_csv = args.pmt_csv
    if pmt_csv is None:
        candidate = Path("PMTPositions_Scan.txt")
        if candidate.exists():
            pmt_csv = candidate

    # ------------------------------------------------------------------
    # Parse grids
    # ------------------------------------------------------------------
    x_range = _parse_range(args.grid_x)
    z_range = _parse_range(args.grid_z)
    theta_range = _parse_range(args.grid_theta)
    phi_range = _parse_range(args.grid_phi)

    x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
    z_vals = np.linspace(z_range[0], z_range[1], z_range[2])
    n_x = len(x_vals)
    n_z = len(z_vals)

    print(f"Spatial grid: {n_x}×{n_z} = {n_x * n_z} positions")
    print(f"  x ∈ [{x_range[0]:.0f}, {x_range[1]:.0f}] × {x_range[2]}")
    print(f"  z ∈ [{z_range[0]:.0f}, {z_range[1]:.0f}] × {z_range[2]}")
    print(f"  y fixed at {args.fix_y} mm")
    print(f"Direction grid: {theta_range[2]}×{phi_range[2]} = "
          f"{theta_range[2] * phi_range[2]} evals per position")
    total_evals = n_x * n_z * theta_range[2] * phi_range[2]
    print(f"  Total raytracing calls: {total_evals}")

    # ------------------------------------------------------------------
    # Build geometry
    # ------------------------------------------------------------------
    print(f"\nLoading geometry from {args.gdml}...")
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
    # Load observed event
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

    ppcm = args.photons_per_cm or observed.true_photons_per_cm or 150
    if args.photons_per_cm is None:
        print(f"  Photons per cm: auto-detected as {ppcm} from muon_truth")
    else:
        print(f"  Photons per cm: {ppcm} (user-specified)")

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------
    best_ll = np.full((n_x, n_z), -np.inf)
    best_theta = np.full((n_x, n_z), np.nan)
    best_phi = np.full((n_x, n_z), np.nan)
    true_theta = observed.true_theta
    true_phi = observed.true_phi

    t_start = time.time()
    n_total = n_x * n_z
    print(f"\nScanning {n_total} positions...")

    for ix, x in enumerate(x_vals):
        for iz, z in enumerate(z_vals):
            n_done = ix * n_z + iz
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = (n_total - n_done) / rate if rate > 0 else 0
            if n_done > 0:
                print(
                    f"  [{n_done}/{n_total}] "
                    f"({x:.0f}, {args.fix_y:.0f}, {z:.0f}) mm — "
                    f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining"
                )
            else:
                print(f"  [{n_done}/{n_total}] "
                      f"({x:.0f}, {args.fix_y:.0f}, {z:.0f}) mm")

            vertex = (x, args.fix_y, z)
            result = grid_scan_direction(
                observed,
                geom,
                theta_range=theta_range,
                phi_range=phi_range,
                fix_vertex=vertex,
                fix_t0=args.fix_t0,
                photons_per_cm=args.photons_per_cm,
                use_time=args.use_time,
                alpha=args.alpha,
                time_sigma=args.time_sigma,
                rng_seed=args.seed + n_done,
                verbose=args.verbose,
            )

            best_ll[ix, iz] = result.best_score
            best_theta[ix, iz] = result.best_theta
            best_phi[ix, iz] = result.best_phi

            # Incremental save
            if args.save:
                np.savez(args.save,
                         x_vals=x_vals, z_vals=z_vals,
                         fix_y=args.fix_y,
                         best_ll=best_ll,
                         best_theta=best_theta,
                         best_phi=best_phi,
                         true_theta=true_theta,
                         true_phi=true_phi,
                         )

    elapsed = time.time() - t_start
    print(f"\nDone — {n_total} positions in {elapsed:.0f}s "
          f"({n_total / elapsed:.2f} pos/s)")

    # Best overall
    best_flat = int(np.argmax(best_ll))
    best_ix = best_flat // n_z
    best_iz = best_flat % n_z
    print(f"Overall best LL = {best_ll[best_ix, best_iz]:.1f} at "
          f"x={x_vals[best_ix]:.0f} mm, z={z_vals[best_iz]:.0f} mm, "
          f"θ={best_theta[best_ix, best_iz]:.1f}°, "
          f"φ={best_phi[best_ix, best_iz]:.1f}°")
    if true_theta is not None:
        print(f"  True vertex: ({observed.true_pos[0]:.0f}, "
              f"{observed.true_pos[1]:.0f}, "
              f"{observed.true_pos[2]:.0f}) mm")
        print(f"  True direction: θ={true_theta:.1f}°, φ={true_phi:.1f}°")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if args.show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        XX, ZZ = np.meshgrid(x_vals, z_vals, indexing="ij")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                 layout="constrained")

        # --- Best LL ---
        ax = axes[0]
        ll_plot = ax.pcolormesh(XX, ZZ, best_ll, shading="auto",
                                cmap="viridis")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title("Best log-likelihood")
        ax.set_aspect("equal")
        fig.colorbar(ll_plot, ax=ax)

        # --- Best theta ---
        ax = axes[1]
        th_plot = ax.pcolormesh(XX, ZZ, best_theta, shading="auto",
                                cmap="twilight")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title("Best θ (deg)")
        ax.set_aspect("equal")
        fig.colorbar(th_plot, ax=ax)

        # --- Best phi ---
        ax = axes[2]
        ph_plot = ax.pcolormesh(XX, ZZ, best_phi, shading="auto",
                                cmap="twilight")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title("Best φ (deg)")
        ax.set_aspect("equal")
        fig.colorbar(ph_plot, ax=ax)

        fig.suptitle(
            f"Direction scan results (event {args.event}, "
            f"{theta_range[2]}×{phi_range[2]} grid)",
            fontsize=13,
        )

        plt.show()


if __name__ == "__main__":
    main()
