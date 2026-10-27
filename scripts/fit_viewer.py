"""Polar / rectangular contour plot of 2-D likelihood surface.

Usage:
    python scripts/fit_viewer.py <grid.npz>              # auto-detect
    python scripts/fit_viewer.py <grid.npz> --polar       # force polar
    python scripts/fit_viewer.py <grid.npz> --rect        # force rectangular
    python -m litepath fit ... --show                     # auto-detect
    python -m litepath fit ... --show --polar             # force polar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_polar(result, ax=None, show_colorbar=True, clip=20):
    """Polar contour plot of log-likelihood vs (θ, φ)."""
    if ax is None:
        ax = plt.subplot(111, projection="polar")

    theta_deg = result.theta_grid
    phi_deg = result.phi_grid

    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    TH, PH = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    delta = result.scores - result.scores.max()
    delta_clipped = np.clip(delta, -clip, 0)
    levels = np.linspace(-clip, 0, 21)
    cf = ax.contourf(PH, TH, delta_clipped, levels=levels, cmap="inferno_r")

    if result.true_theta is not None:
        ax.plot(
            np.radians(result.true_phi % 360),
            np.radians(result.true_theta),
            marker="*", color="cyan", markersize=14,
            mew=1.5, mec="white", zorder=5,
            label="True",
        )

    ax.plot(
        np.radians(result.best_phi % 360),
        np.radians(result.best_theta),
        marker="o", color="lime", markersize=10,
        mew=1.5, mec="white", zorder=5,
        label="Best fit",
    )

    # Zoom axes to data extent
    t_min, t_max = float(theta_deg.min()), float(theta_deg.max())
    p_min_orig = float(phi_deg.min())
    p_max_orig = float(phi_deg.max())

    ax.set_ylim(np.radians(t_min), np.radians(t_max))

    t_ticks = np.linspace(t_min, t_max, 5)
    ax.set_yticks(np.radians(t_ticks))
    ax.set_yticklabels([f"{t:.1f}°" for t in t_ticks])

    if p_max_orig - p_min_orig < 360:
        ax.set_thetamin(p_min_orig % 360)
        ax.set_thetamax(p_max_orig % 360)

    ax.set_title(f"Δ log-likelihood (clipped at -{clip})", pad=20)
    ax.legend(loc="upper right", fontsize=10)

    if show_colorbar:
        plt.colorbar(cf, ax=ax, label="Δ log-likelihood", pad=0.12)

    plt.tight_layout()
    return ax


def plot_rectangular(result, ax=None, show_colorbar=True, clip=20):
    """Rectilinear 2-D heatmap of Δ log-likelihood vs (θ, φ).

    Preferred over ``plot_polar`` when the grid covers a narrow angular
    window, since there is no polar distortion.
    """
    if ax is None:
        ax = plt.subplot(111)

    theta_deg = result.theta_grid
    phi_deg = result.phi_grid
    P, T = np.meshgrid(phi_deg, theta_deg, indexing="xy")

    delta = result.scores - result.scores.max()
    delta_clipped = np.clip(delta, -clip, 0)

    im = ax.pcolormesh(P, T, delta_clipped, cmap="inferno_r",
                       shading="auto")
    im.set_clim(-clip, 0)

    if result.true_theta is not None:
        ax.plot(result.true_phi, result.true_theta,
                marker="*", color="cyan", markersize=14,
                mew=1.5, mec="white", zorder=5,
                label="True")

    ax.plot(result.best_phi, result.best_theta,
            marker="o", color="lime", markersize=10,
            mew=1.5, mec="white", zorder=5,
            label="Best fit")

    ax.set_xlabel("φ (deg)")
    ax.set_ylabel("θ (deg)")
    ax.set_title(f"Δ log-likelihood (clipped at -{clip})")
    ax.legend(loc="upper right", fontsize=10)

    # Tick every few degrees; fall back to ~5 labels
    n_ticks = 5
    for axis, vals in [(ax.xaxis, phi_deg), (ax.yaxis, theta_deg)]:
        lo, hi = float(vals.min()), float(vals.max())
        # Snap to integers if range > 2 * n_ticks
        step = max(1, int((hi - lo) / n_ticks))
        ticks = np.arange(np.ceil(lo / step) * step,
                          np.floor(hi / step) * step + step / 2, step)
        if len(ticks) < 2:
            ticks = np.linspace(lo, hi, n_ticks)
        axis.set_ticks(ticks)

    ax.set_aspect("auto")

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Δ log-likelihood")

    plt.tight_layout()
    return ax


def is_full_sky(result) -> bool:
    """True if the grid covers essentially the full sphere."""
    t_span = float(result.theta_grid.max() - result.theta_grid.min())
    p_span = float(result.phi_grid.max() - result.phi_grid.min())
    return t_span > 170 and p_span > 350


def main():
    p = argparse.ArgumentParser(description="Plot grid-scan likelihood surface")
    p.add_argument("grid_file", type=Path, help="NPZ file from --save-grid")
    p.add_argument("--save", type=Path, default=None, help="Save figure to file")
    p.add_argument("--clip", type=float, default=20,
                   help="ΔLL clipping window (default: 20)")
    p.add_argument("--polar", action="store_true",
                   help="Force polar projection even for zoomed-in grids")
    p.add_argument("--rect", action="store_true",
                   help="Force rectangular plot even for full-sky grids")
    args = p.parse_args()

    data = np.load(args.grid_file)

    class FakeResult:
        theta_grid = data["theta_grid"]
        phi_grid = data["phi_grid"]
        scores = data["scores"]
        best_theta = float(data["best_theta"])
        best_phi = float(data["best_phi"])
        true_theta = float(data["true_theta"]) if "true_theta" in data else None
        true_phi = float(data["true_phi"]) if "true_phi" in data else None

    use_polar = args.polar or (is_full_sky(FakeResult) and not args.rect)

    if use_polar:
        plot_polar(FakeResult(), clip=args.clip)
    else:
        plot_rectangular(FakeResult(), clip=args.clip)

    if args.save:
        plt.savefig(args.save, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
