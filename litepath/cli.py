"""CLI for litepath ray tracer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from litepath.geometry.build import build_geometry, Geometry
from litepath.engine.trace import trace_rays
from litepath.simulate import trace_cherenkov
from litepath.optics import WaterAttenuation, load_optics_config
from litepath.output import write_hits, write_detector_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="litepath",
        description="GPU-accelerated ray tracer for ANNIE detector",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run ray tracing")
    run.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    run.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    run.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    run.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file or CSV (default: PMTPositions_Scan.txt near GDML)")
    run.add_argument("--photons-per-cm", type=int, default=150, help="Photons per cm along the 4 m muon track (total ≈ 401 × this value)")
    run.add_argument("--output", "-o", type=Path, default=Path("hits.parquet"), help="Output Parquet path")
    run.add_argument("--seed", type=int, default=None, help="Random seed")
    run.add_argument("--mode", choices=["uniform", "cherenkov"], default="uniform",
                     help="Photon generation mode")
    run.add_argument("--lappd-indices", type=str, default=None,
                     help="Comma-separated LAPPD candidate indices (default: use 3 from manifest)")
    run.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    run.add_argument("--no-gdml", action="store_true",
                     help="Skip inner structure GDML mesh (tank + PMTs only)")
    run.add_argument("--no-pmt-holders", action="store_true",
                     help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")
    run.add_argument("--z-offset", type=float, default=0.0,
                     help="Vertical offset (mm) — use when PMT CSV and GDML use different Z origins")
    run.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")
    run.add_argument("--detector-config", type=Path, default=None,
                     help="Path to detector registry YAML (auto-built if absent)")
    run.add_argument("--wavelength", type=float, default=350.0,
                     help="Photon wavelength in nm (default: 350)")
    run.add_argument("--det-rotation", type=float, default=22.5,
                     help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")
    run.add_argument("--max-bounces", type=int, default=0,
                     help="Number of surface reflections per photon (0 = off, default: 0)")
    run.add_argument("--optics-config", type=Path, default=None,
                     help="YAML file with per-material optical properties (default: built-in)")
    run.add_argument("--water-absorption-mm", type=float, default=0.0,
                     help="Water absorption length in mm (0 = off, default: 0)")
    run.add_argument("--water-scattering-mm", type=float, default=0.0,
                     help="Rayleigh scattering length in mm (0 = off, default: 0)")

    cherenkov = sub.add_parser("extract-manifest", help="Extract component manifest from STEP")
    cherenkov.add_argument("--step", required=True, type=Path, help="Path to STEP CAD file")
    cherenkov.add_argument("--output", "-o", type=Path, default=Path("component_manifest.json"), help="Output JSON path")

    viz = sub.add_parser("viz-server", help="Start interactive Cherenkov visualization server")
    viz.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    viz.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file for component positions")
    viz.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    viz.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    viz.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    viz.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    viz.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    viz.add_argument("--no-gdml", action="store_true",
                     help="Skip inner structure GDML mesh (tank + PMTs only)")
    viz.add_argument("--no-pmt-holders", action="store_true",
                     help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")
    viz.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    viz.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                     help="LAPPD geometry model (default: bare rectangle; annie: housed LAPPD)")
    viz.add_argument("--bottom-rot", type=float, default=45.0,
                     help="Extra Z-rotation (deg) for all PMTs, aligning scan mesh with structure (default: 45)")
    viz.add_argument("--bottom-spin", type=float, default=22.5,
                     help="Per-PMT spin (deg) about forward axis for bottom PMTs (default: 22.5)")
    viz.add_argument("--det-rotation", type=float, default=22.5,
                     help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")
    viz.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                     help="Number of obscurant PVC surfboards (0, 1, or 3)")

    detcfg = sub.add_parser("build-detector-config",
                            help="Build detector registry YAML from geometry")
    detcfg.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                        help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    detcfg.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    detcfg.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    detcfg.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file")
    detcfg.add_argument("--output", "-o", type=Path, default=Path("detectors.yaml"),
                        help="Output YAML path (default: detectors.yaml)")
    detcfg.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    detcfg.add_argument("--no-gdml", action="store_true",
                        help="Skip inner structure GDML mesh (tank + PMTs only)")
    detcfg.add_argument("--no-pmt-holders", action="store_true",
                        help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")
    detcfg.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    detcfg.add_argument("--lappd-model", choices=["default", "annie"], default="default",
                        help="LAPPD geometry model")
    detcfg.add_argument("--det-rotation", type=float, default=22.5,
                        help="Global Z-rotation (deg) so +Y aligns with octagon corner (default: 22.5)")

    lappd = sub.add_parser("viz-lappd", help="Start standalone LAPPD module viewer")
    lappd.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    lappd.add_argument("--port", type=int, default=8081, help="Port to bind (default: 8081)")

    batch = sub.add_parser("batch", help="Batch-mode event generation")
    batch.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                       help="Path to GDML geometry mesh")
    batch.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    batch.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    batch.add_argument("--pmt-csv", type=Path, default=None, help="Path to PMT scan file or CSV")
    batch.add_argument("--events", type=int, default=-1,
                        help="Number of events (default: 100, or auto-count from --muon-file)")
    batch.add_argument("--muon-fixed", type=str, default=None,
                       help="Fixed muon topology: 'x y z t0 dx dy dz' (7 floats)")
    batch.add_argument("--muon-file", type=Path, default=None,
                       help="File with one topology per line: 'x y z t0 dx dy dz'")
    batch.add_argument("--muon-mode", type=str, default="isotropic",
                       choices=["downward", "isotropic", "beam"],
                       help="Muon direction sampling (default: isotropic). "
                            "'downward' = mostly -Z (atmospheric), "
                            "'isotropic' = uniform on sphere (default), "
                            "'beam' = forward along +Y (beam direction)")
    batch.add_argument("--photons-per-cm", type=int, default=150,
                       help="Photons per cm along the muon track")
    batch.add_argument("--batch-size", type=int, default=50,
                       help="Events per GPU launch (default: 50, higher = faster)")
    batch.add_argument("--output-dir", "-o", type=Path, default=Path("results"),
                       help="Output directory (writes output.h5)")
    batch.add_argument("--no-record", action="store_true",
                       help="Skip writing per-event output files")
    batch.add_argument("--pmt-response", action="store_true",
                       help="Enable PMT digital model")
    batch.add_argument("--full-wf", action="store_true",
                       help="Use full waveform path for PMT response")
    batch.add_argument("--lappd-response", action="store_true",
                       help="Enable LAPPD digital readout pipeline")
    batch.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    batch.add_argument("--no-gdml", action="store_true",
                       help="Skip inner structure GDML mesh (tank + PMTs only)")
    batch.add_argument("--no-pmt-holders", action="store_true",
                       help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")
    batch.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                       help="Number of obscurant PVC surfboards (0, 1, or 3)")
    batch.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    batch.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                       help="LAPPD geometry model")
    batch.add_argument("--det-rotation", type=float, default=22.5,
                       help="Global Z-rotation (deg) so +Y aligns with octagon corner")
    batch.add_argument("--lappd-indices", type=str, default=None,
                       help="Comma-separated LAPPD candidate indices")
    batch.add_argument("--wavelength", type=float, default=350.0,
                       help="Photon wavelength in nm (default: 350)")
    batch.add_argument("--scintillation", action="store_true",
                       help="Enable scintillation photon generation (wbLS)")
    batch.add_argument("--photons-per-cm-scint", type=int, default=100,
                       help="Scintillation photons per cm of track (default: 100)")
    batch.add_argument("--wavelength-scint", type=float, default=420.0,
                       help="Scintillation photon wavelength in nm (default: 420)")
    batch.add_argument("--tau-fast", type=float, default=2.0,
                       help="Fast scintillation decay time in ns (default: 2.0)")
    batch.add_argument("--tau-slow", type=float, default=20.0,
                       help="Slow scintillation decay time in ns (default: 20.0)")
    batch.add_argument("--fast-fraction", type=float, default=0.95,
                       help="Fraction of scintillation light in the fast component (default: 0.95)")
    batch.add_argument("--max-bounces", type=int, default=0,
                        help="Number of surface reflections per photon (0 = off)")
    batch.add_argument("--optics-config", type=Path, default=None,
                        help="YAML file with per-material optical properties")
    batch.add_argument("--water-absorption-mm", type=float, default=0.0,
                        help="Water absorption length in mm (0 = off, default: 0)")
    batch.add_argument("--water-scattering-mm", type=float, default=0.0,
                        help="Rayleigh scattering length in mm (0 = off, default: 0)")
    batch.add_argument("--seed", type=int, default=None, help="Random seed")
    batch.add_argument("--light-burst", action="store_true",
                       help="Isotropic light burst (instead of muon Cherenkov)")
    batch.add_argument("--burst-n-phots", type=int, default=1000,
                       help="Number of isotropic photons per burst")
    batch.add_argument("--burst-position", type=str, default=None,
                       help="Burst centre: 'x y z' in mm (default: tank centre) ")

    # ---- benchmark subcommand (WAGLE ROOT -> litepath HDF5) ----
    bench = sub.add_parser(
        "benchmark",
        help="Convert WAGLE ROOT trees (data / MC) into the litepath "
             "batch HDF5 schema for benchmarking",
    )
    bench.add_argument("--data", type=Path, default=None,
                       help="ANNIE data ROOT file (ANNIETree_*.root)")
    bench.add_argument("--mc", type=Path, default=None,
                       help="Monte Carlo ROOT file (MCTree_*.root)")
    bench.add_argument("--tree", type=str, default="Event",
                       help="TTree name inside the ROOT files (default: Event)")
    bench.add_argument("--pmt-csv", type=Path, default=Path("PMTPositions_Scan.txt"),
                       help="PMT scan file used to build the detector registry")
    bench.add_argument("--det-rotation", type=float, default=22.5,
                       help="Global Z-rotation (deg) matching the raytracer run")
    bench.add_argument("--output-dir", "-o", type=Path, default=Path("results/benchmark"),
                       help="Directory for the converted HDF5 files")
    bench.add_argument("--max-events", type=int, default=0,
                       help="Convert at most N events per file (0 = all)")
    bench.add_argument("--no-raw-hits", action="store_true",
                       help="Skip the per-hit 'pmt_hits' extension table")

    # ---- standalone LAPPD batch subcommand ----
    lappd_batch = sub.add_parser(
        "batch-lappd",
        help="Batch-mode events on a standalone LAPPD module (same output "
             "format as 'batch')",
    )
    lappd_batch.add_argument("--events", type=int, default=-1,
                             help="Number of events (default: 100, or grid "
                                  "nx*nz / line count of --muon-file)")
    lappd_batch.add_argument("--source", choices=["cherenkov", "spot"], default="cherenkov",
                             help="Light source definition (default: cherenkov). "
                                  "'cherenkov' = tracked muon cone, "
                                  "'spot' = directed diverging beam")
    lappd_batch.add_argument("--sample-mode", choices=["random", "grid", "fixed"],
                             default="random",
                             help="Where to place the source (default: random). "
                                  "random/grid aim at the photocathode; "
                                  "fixed uses --muon-fixed/--muon-file")
    lappd_batch.add_argument("--grid-steps", nargs=2, type=int, default=[9, 5],
                             metavar=("NX", "NZ"),
                             help="Grid size for --sample-mode grid: NX transverse "
                                  "by NZ distance steps (default: 9 5)")
    lappd_batch.add_argument("--muon-fixed", type=str, default=None,
                             help="Fixed source: 'x y z t0 dx dy dz' (7 floats)")
    lappd_batch.add_argument("--muon-file", type=Path, default=None,
                             help="File with one topology per line: 'x y z t0 dx dy dz'")
    lappd_batch.add_argument("--photons-per-cm", type=int, default=150,
                             help="Cherenkov photons per cm along the muon track")
    lappd_batch.add_argument("--n-photons", type=int, default=5000,
                             help="Photons in the --source spot beam")
    lappd_batch.add_argument("--spread-deg", type=float, default=2.0,
                             help="Angular spread (deg) of the launched source")
    lappd_batch.add_argument("--intercept", nargs=3, type=float, default=None,
                             metavar=("X", "Y", "Z"),
                             help="Aim target (photocathode intercept point): "
                                  "'x y z'. Overrides aiming at the PC centre "
                                  "and is recorded in the truth table")
    lappd_batch.add_argument("--wavelength", type=float, default=350.0,
                             help="Photon wavelength in nm (default: 350)")
    lappd_batch.add_argument("--lappd-response", action="store_true",
                             help="Enable LAPPD digital readout pipeline")
    lappd_batch.add_argument("--max-bounces", type=int, default=0,
                             help="Number of surface reflections per photon (0 = off)")
    lappd_batch.add_argument("--water-absorption-mm", type=float, default=0.0,
                             help="Water absorption length in mm (0 = off)")
    lappd_batch.add_argument("--water-scattering-mm", type=float, default=0.0,
                             help="Rayleigh scattering length in mm (0 = off)")
    lappd_batch.add_argument("--output-dir", "-o", type=Path, default=Path("results"),
                             help="Output directory (writes output.h5)")
    lappd_batch.add_argument("--no-record", action="store_true",
                             help="Skip writing per-event output files")
    lappd_batch.add_argument("--seed", type=int, default=None, help="Random seed")
    lappd_batch.add_argument("--show-config", action="store_true",
                             help="Print resolved batch configuration and geometry "
                                  "then exit")

    # ---- fit subcommand ----
    fit = sub.add_parser("fit", help="Grid-scan direction fit for a single event")
    fit.add_argument("h5", type=Path, help="Path to batch output HDF5 file")
    fit.add_argument("--event", type=int, default=0, help="Event ID to fit")
    fit.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh")
    fit.add_argument("--pmt-csv", type=Path, default=None,
                     help="Path to PMT scan file or CSV (default: PMTPositions_Scan.txt)")
    fit.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file")
    fit.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    fit.add_argument("--no-lappd", action="store_true", help="Skip LAPPD rectangles")
    fit.add_argument("--no-gdml", action="store_true",
                     help="Skip inner structure GDML mesh (tank + PMTs only)")
    fit.add_argument("--no-pmt-holders", action="store_true",
                     help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")
    fit.add_argument("--z-offset", type=float, default=0.0, help="Vertical offset (mm)")
    fit.add_argument("--lappd-model", choices=["default", "annie"], default="annie",
                     help="LAPPD geometry model")
    fit.add_argument("--det-rotation", type=float, default=22.5,
                     help="Global Z-rotation (deg) so +Y aligns with octagon corner")
    fit.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                     help="Number of obscurant PVC surfboards (0, 1, or 3)")
    fit.add_argument("--grid-theta", type=str, default="0 180 19",
                     help="Theta range: 'start stop steps' in deg (default: 0 180 19)")
    fit.add_argument("--grid-phi", type=str, default="0 360 37",
                     help="Phi range: 'start stop steps' in deg (default: 0 360 37)")
    fit.add_argument("--theta-window", type=float, default=None,
                     help="Theta half-window (deg) around true direction. "
                          "Overrides --grid-theta range.  Use with --grid-steps.")
    fit.add_argument("--phi-window", type=float, default=None,
                     help="Phi half-window (deg).  Defaults to --theta-window value.")
    fit.add_argument("--grid-steps", type=int, default=41,
                     help="Number of steps per axis in windowed mode (default: 41)")
    fit.add_argument("--photons-per-cm", type=int, default=None,
                     help="Photons per cm for each hypothesis evaluation (default: auto-detect from muon_truth)")
    fit.add_argument("--fix-vertex", type=str, default=None,
                     help="Fixed vertex 'x y z' (default: from muon_truth)")
    fit.add_argument("--fix-t0", type=float, default=None,
                     help="Fixed t0 in ns (default: from muon_truth)")
    fit.add_argument("--use-time", action="store_true",
                     help="Include time residual likelihood")
    fit.add_argument("--time-sigma", type=float, default=None,
                     help="Global timing sigma (ns). Default: per-PMT-type from geometry.")
    fit.add_argument("--alpha", type=float, default=1.0,
                     help="Scale factor for time likelihood (default: 1.0)")
    fit.add_argument("--seed", type=int, default=42, help="RNG seed for raytracing")
    fit.add_argument("--save-grid", type=Path, default=None,
                     help="Save likelihood grid to NPZ file")
    fit.add_argument("--show", action="store_true",
                     help="Show plot of likelihood surface")
    fit.add_argument("--polar", action="store_true",
                     help="Force polar projection (default: rectangular for zoomed-in, polar for full-sky)")

    return p


def _generate_uniform(geometry: Geometry, n: int, rng: np.random.Generator) -> tuple:
    """Generate random photons uniformly inside the tank volume.

    Positions are rejection-sampled inside a cylinder 90 % of the tank
    radius.  Directions are isotropic (uniform on the sphere).

    This is useful for flat efficiency scans — each photon has equal
    probability of hitting any detector regardless of emission model.

    To add a new emission model, create a function with the same
    signature (geometry, n, rng) → (origins, directions) and wire
    it into run_command() as a new --mode branch.

    Returns (origins, directions) arrays, each (N, 3) float32.
    """
    r_tank = geometry.tank_radius * 0.9
    z_min = geometry.tank_z_min + 100.0
    z_max = geometry.tank_z_max - 100.0

    origins = np.zeros((n, 3), dtype=np.float32)
    directions = np.zeros((n, 3), dtype=np.float32)

    # Rejection sampling for uniform positions in cylinder
    batch = int(n * 1.3) + 1
    accepted = 0
    while accepted < n:
        xs = rng.uniform(-r_tank, r_tank, batch)
        ys = rng.uniform(-r_tank, r_tank, batch)
        zs = rng.uniform(z_min, z_max, batch)
        rs2 = xs * xs + ys * ys
        mask = rs2 <= r_tank * r_tank
        n_mask = mask.sum()
        to_take = min(n_mask, n - accepted)
        if to_take == 0:
            continue
        origins[accepted:accepted + to_take, 0] = xs[mask][:to_take]
        origins[accepted:accepted + to_take, 1] = ys[mask][:to_take]
        origins[accepted:accepted + to_take, 2] = zs[mask][:to_take]
        accepted += to_take

    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    directions[:, 0] = np.cos(theta) * np.cos(phi)
    directions[:, 1] = np.sin(theta) * np.cos(phi)
    directions[:, 2] = np.sin(phi)

    return origins, directions

#Call generate_cherenkov for each muon, make a seperate function for multiple muon generation
def _generate_cherenkov(
    geometry: Geometry,
    photons_per_cm: int,
    rng: np.random.Generator,
    muon_pos: tuple = (0.0, 0.0, 2000.0), #Tank coordinate system (cartesian based on the cylinder) where z is height
    muon_dir: tuple = (0.0, 0.0, -1.0),
    cherenkov_angle: float = 0.73,
) -> tuple:
    """Generate Cherenkov cone from a muon track (delegates to cherenkov module).

    This is a thin wrapper that calls generate_cherenkov_photons() in
    cherenkov.py.  The student modifying the emission model should work
    in cherenkov.py directly; this function just passes through.

    Returns (origins, directions) arrays, each (N, 3) float32.
    """
    from litepath.cherenkov import generate_cherenkov_photons

    origins, directions, create_times = generate_cherenkov_photons(
        muon_pos, muon_dir, photons_per_cm,
        cherenkov_angle=cherenkov_angle, rng=rng,
    )
    return origins, directions


def run_command(args: argparse.Namespace) -> None:
    # Validate args
    if args.pmt_csv and not args.pmt_csv.exists():
        print(f"Error: PMT CSV not found: {args.pmt_csv}")
        return
    if not args.pmt_csv and not args.step and not args.manifest:
        print("Warning: no PMT data source provided (--pmt-csv, --step, or --manifest)")

    # Handle lappd-indices override
    lappd_indices = None
    if args.lappd_indices:
        import json
        lappd_indices = [int(x) for x in args.lappd_indices.split(",")]

    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, lappd_indices=lappd_indices,
                          no_lappd=args.no_lappd, no_gdml=args.no_gdml, no_pmt_holders=args.no_pmt_holders,
                          z_offset=args.z_offset, lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation, n_surfboards=args.surfboard)

    print(f"  Mesh: {geom.mesh_vertices.shape[0]} vertices, {geom.mesh_triangles.shape[0]} triangles")
    print(f"  PMTs: {geom.pmt_centers.shape[0]} (radii: {set(f'{r:.1f}' for r in geom.pmt_radii)})")
    print(f"  LAPPDs: {geom.lappd_data.shape[0]}")
    if geom.lappd_housing_data.shape[0] > 0:
        print(f"  ANNIE LAPPD housing: 1 at centre {geom.lappd_housing_data[0, :3]}")
    if geom.detectors:
        print(f"  Detector registry: {len(geom.detectors)} entries")
    print(f"  Tank: R={geom.tank_radius:.0f} mm, Z=[{geom.tank_z_min:.0f}, {geom.tank_z_max:.0f}]")

    # Write detector config if requested
    if args.detector_config:
        print(f"\nWriting detector registry to {args.detector_config}...")
        write_detector_config(geom.detectors, args.detector_config)

    rng = np.random.default_rng(args.seed)

    water_cfg = WaterAttenuation(
        absorption_length_mm=args.water_absorption_mm,
        scattering_length_mm=args.water_scattering_mm,
    )
    needs_optics = args.max_bounces > 0 or water_cfg.is_active
    if needs_optics and args.mode == "cherenkov":
        what = []
        if args.max_bounces > 0:
            what.append(f"max {args.max_bounces} reflections")
        if water_cfg.is_active:
            parts = []
            if water_cfg.absorption_length_mm > 0:
                parts.append(f"absorption={water_cfg.absorption_length_mm:.0f} mm")
            if water_cfg.scattering_length_mm > 0:
                parts.append(f"scattering={water_cfg.scattering_length_mm:.0f} mm")
            what.append("water " + ", ".join(parts))
        print(f"\nMulti-bounce optics enabled: {'; '.join(what)} per photon")

    total_photons = args.photons_per_cm * 401
    print(f"\nGenerating {args.photons_per_cm} photons/cm × 401 steps = {total_photons} total ({args.mode} mode)...")
    t0 = time.time()
    if args.mode == "uniform":
        origins, directions = _generate_uniform(geom, total_photons, rng)
        hits = trace_rays(origins, directions, geom)
    else:
        optics_cfg = load_optics_config(args.optics_config) if needs_optics else None
        hits = trace_cherenkov(
            (0.0, 0.0, 2000.0), (0.0, 0.0, -1.0),
            args.photons_per_cm, geom, rng=rng,
            wavelength_nm=args.wavelength,
            max_bounces=args.max_bounces,
            optics_config=optics_cfg,
            water_config=water_cfg if water_cfg.is_active else None,
        )
    t_gen = time.time() - t0
    print(f"  Generated/traced in {t_gen:.2f}s")

    n_hit = int(hits[:, 0].sum())
    print(f"\nResults: {n_hit}/{total_photons} hit ({n_hit / total_photons * 100:.1f}%)")

    from litepath.hits import CID_PMT, CID_LAPPD, CID_INNER_STRUCTURE, CID_TANK_WALL
    cid_names = {CID_INNER_STRUCTURE: "structure", CID_PMT: "PMT", CID_LAPPD: "LAPPD", CID_TANK_WALL: "tank"}
    for cid in range(1, 5):
        count = int((hits[:, 8] == cid).sum())
        if count > 0:
            print(f"  {cid_names[cid]}: {count}")

    n_det = int((hits[:, 9] >= 0).sum())
    if n_det > 0:
        print(f"  Detector hits: {n_det}")

    print(f"\nWriting {args.output}...")
    write_hits(hits, args.output)
    print("Done.")


def build_detector_config_command(args: argparse.Namespace) -> None:
    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, no_lappd=args.no_lappd,
                          no_gdml=args.no_gdml, no_pmt_holders=args.no_pmt_holders,
                          z_offset=args.z_offset, lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation)

    print(f"  Built {len(geom.detectors)} detectors:")
    for d in geom.detectors:
        print(f"    {d.id:5d}  {d.system:16s}  {d.label}")

    print(f"\nWriting {args.output}...")
    write_detector_config(geom.detectors, args.output)
    print("Done.")


def extract_manifest_command(args: argparse.Namespace) -> None:
    from litepath.step_parser import parse_step

    print(f"Parsing {args.step}...")
    t0 = time.time()
    manifest = parse_step(args.step)
    manifest.to_json(args.output)
    t_elapsed = time.time() - t0
    print(f"Extracted {len(manifest.pmt_centers)} PMT candidates, "
          f"{len(manifest.lappd_candidates)} LAPPD candidates")
    print(f"Saved to {args.output} in {t_elapsed:.1f}s")


def _count_muon_lines(path: Path) -> int:
    """Count valid topology lines in a muon file (skips blanks and comments)."""
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                n += 1
    return n


def batch_command(args: argparse.Namespace) -> None:
    from litepath.batch import BatchConfig, run_batch

    # Auto-detect event count from muon file
    if args.events == -1:
        if args.muon_file and args.muon_file.exists():
            args.events = _count_muon_lines(args.muon_file)
            print(f"Auto-detected {args.events} events from {args.muon_file}")
        else:
            args.events = 100

    # Parse muon-fixed
    muon_fixed = None
    if args.muon_fixed:
        parts = args.muon_fixed.split()
        if len(parts) != 7:
            print("Error: --muon-fixed requires 7 floats: x y z t0 dx dy dz")
            return
        muon_fixed = tuple(float(p) for p in parts)

    # Parse burst position
    burst_position = None
    if args.burst_position:
        parts = args.burst_position.split()
        if len(parts) != 3:
            print("Error: --burst-position requires 3 floats: x y z")
            return
        burst_position = tuple(float(p) for p in parts)

    config = BatchConfig(
        n_events=args.events,
        muon_fixed=muon_fixed,
        muon_file=args.muon_file,
        muon_mode=args.muon_mode,
        photons_per_cm=args.photons_per_cm,
        wavelength_nm=args.wavelength,
        max_bounces=args.max_bounces,
        batch_size=args.batch_size,
        scintillation_enabled=args.scintillation,
        photons_per_cm_scint=args.photons_per_cm_scint,
        wavelength_scint_nm=args.wavelength_scint,
        tau_fast=args.tau_fast,
        tau_slow=args.tau_slow,
        fast_fraction=args.fast_fraction,
        pmt_response=args.pmt_response,
        pmt_full_wf=args.full_wf,
        lappd_response=args.lappd_response,
        output_dir=args.output_dir,
        record_events=not args.no_record,
        seed=args.seed,
        light_burst=args.light_burst,
        burst_n_photons=args.burst_n_phots,
        burst_position=burst_position,
        water_absorption_mm=args.water_absorption_mm,
        water_scattering_mm=args.water_scattering_mm,
    )

    # Validate
    if args.muon_file and not args.muon_file.exists():
        print(f"Error: muon file not found: {args.muon_file}")
        return
    if args.pmt_csv and not args.pmt_csv.exists():
        print(f"Error: PMT CSV not found: {args.pmt_csv}")
        return

    # Build geometry (reuse the same geometry for all events)
    lappd_indices = None
    if args.lappd_indices:
        import json
        lappd_indices = [int(x) for x in args.lappd_indices.split(",")]

    print(f"Loading geometry from {args.gdml}...")
    geom = build_geometry(args.gdml, step_path=args.step, manifest_path=args.manifest,
                          pmt_csv_path=args.pmt_csv, lappd_indices=lappd_indices,
                          no_lappd=args.no_lappd, no_gdml=args.no_gdml, no_pmt_holders=args.no_pmt_holders,
                          z_offset=args.z_offset,
                          lappd_model=args.lappd_model,
                          det_rotation_deg=args.det_rotation, n_surfboards=args.surfboard)

    print(f"  Mesh: {geom.mesh_vertices.shape[0]} verts, {geom.mesh_triangles.shape[0]} tris")
    print(f"  PMTs: {geom.pmt_centers.shape[0]}")
    print(f"  LAPPDs: {geom.lappd_data.shape[0]}")
    print(f"  Tank: R={geom.tank_radius:.0f} mm, Z=[{geom.tank_z_min:.0f}, {geom.tank_z_max:.0f}]")
    if geom.surfboard_data.shape[0] > 0:
        print(f"  Surfboards: {geom.surfboard_data.shape[0]} PVC panels")

    # Save companion files for event display / analysis
    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from litepath.batch import H5_OUTPUT_NAME
    h5_path = output_dir / H5_OUTPUT_NAME

    if args.light_burst:
        cx, cy, cz = burst_position or (0, 0, 1940)
        print(f"  Light burst: {config.burst_n_photons} isotropic photons at ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm")
    elif args.muon_file:
        print(f"  Muon topology: from file ({args.muon_file})")
    elif muon_fixed:
        print(f"  Muon topology: fixed {muon_fixed}")
    else:
        mode_names = {"downward": "downward (atmospheric)",
                      "isotropic": "isotropic (totally random, default)",
                      "beam": "forward along +Y (beam-like)"}
        print(f"  Muon topology: random position, {mode_names.get(config.muon_mode, config.muon_mode)}")

    print(f"\nGenerating {config.n_events} events ({config.photons_per_cm} ph/cm)...")
    if config.pmt_response:
        mode = "waveform" if config.pmt_full_wf else "fast"
        print(f"  PMT response: enabled ({mode} path)")
    if config.lappd_response:
        print(f"  LAPPD response: enabled (6-stage Taichi pipeline)")

    water_cfg = WaterAttenuation(
        absorption_length_mm=args.water_absorption_mm,
        scattering_length_mm=args.water_scattering_mm,
    )
    needs_optics = args.max_bounces > 0 or water_cfg.is_active
    optics_cfg = None
    if needs_optics:
        optics_cfg = load_optics_config(args.optics_config)
        what = []
        if args.max_bounces > 0:
            what.append(f"max {args.max_bounces} reflections")
        if water_cfg.is_active:
            parts = []
            if water_cfg.absorption_length_mm > 0:
                parts.append(f"absorption={water_cfg.absorption_length_mm:.0f} mm")
            if water_cfg.scattering_length_mm > 0:
                parts.append(f"scattering={water_cfg.scattering_length_mm:.0f} mm")
            what.append("water " + ", ".join(parts))
        print(f"  Multi-bounce optics: {'; '.join(what)} per photon")

    paths = run_batch(geom, config, optics_config=optics_cfg)

    # Write metadata and detectors into the HDF5 file (append mode)
    if h5_path.exists():
        from litepath.io_h5 import write_metadata, write_detectors
        meta = {
            "tank_radius_mm": geom.tank_radius,
            "tank_z_min_mm": geom.tank_z_min,
            "tank_z_max_mm": geom.tank_z_max,
        }
        write_metadata(h5_path, meta)
        print(f"  Wrote metadata to {h5_path}")
        write_detectors(h5_path, geom.detectors)

    print("Done.")


# ---------------------------------------------------------------------------
# Benchmark conversion command
# ---------------------------------------------------------------------------

def benchmark_command(args: argparse.Namespace) -> None:
    """Convert WAGLE ROOT trees into the litepath batch HDF5 schema."""
    from litepath.benchmark import (
        RootTreeReader,
        load_pmt_registry,
        write_benchmark_h5,
    )

    sources: list[tuple[Path, str]] = []
    if args.data:
        sources.append((args.data, "data"))
    if args.mc:
        sources.append((args.mc, "mc"))
    if not sources:
        print("Error: pass --data and/or --mc")
        return

    print("Building PMT registry from", args.pmt_csv)
    registry = load_pmt_registry(args.pmt_csv, det_rotation_deg=args.det_rotation)
    print(f"  {len(registry)} PMTs (TubeIDs {registry.tube_id.min()}-"
          f"{registry.tube_id.max()})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_events = args.max_events if args.max_events > 0 else None

    for src, kind in sources:
        reader = RootTreeReader(src, tree=args.tree, max_events=max_events)
        n = reader.num_entries if max_events is None else min(max_events,
                                                              reader.num_entries)
        print(f"\nConverting {src} ({kind}): {n} events, "
              f"truth={'yes' if reader.has_truth else 'no (MRD reconstruction)'}")
        out_path = args.output_dir / f"{kind}.h5"
        write_benchmark_h5(
            out_path,
            reader.iter_events(),
            registry,
            kind=kind,
            source_file=str(src),
            det_rotation_deg=args.det_rotation,
            write_raw_hits=not args.no_raw_hits,
        )
        print(f"  Wrote {out_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Standalone LAPPD batch command
# ---------------------------------------------------------------------------

def lappd_batch_command(args: argparse.Namespace) -> None:
    from litepath.lappd_batch import LAPPDBatchConfig, run_lappd_batch
    from litepath.geometry.build import build_single_lappd_geometry

    # --events default: grid product, muon-file line count, or 100
    n_events = args.events
    if n_events == -1:
        if args.sample_mode == "grid":
            n_events = args.grid_steps[0] * args.grid_steps[1]
        elif args.muon_file and args.muon_file.exists():
            n_events = _count_muon_lines(args.muon_file)
        else:
            n_events = 100

    # Parse composite arguments
    muon_fixed = None
    if args.muon_fixed:
        parts = args.muon_fixed.split()
        if len(parts) != 7:
            print("Error: --muon-fixed requires 7 floats: x y z t0 dx dy dz")
            return
        muon_fixed = tuple(float(p) for p in parts)

    intercept = None
    if args.intercept:
        intercept = tuple(float(v) for v in args.intercept)

    nx, nz = args.grid_steps

    config = LAPPDBatchConfig(
        n_events=n_events,
        sample_mode=args.sample_mode,
        source=args.source,
        photons_per_cm=args.photons_per_cm,
        wavelength_nm=args.wavelength,
        n_photons_spot=args.n_photons,
        spread_deg=args.spread_deg,
        intercept=intercept,
        muon_fixed=muon_fixed,
        muon_file=args.muon_file,
        grid_steps=(nx, nz),
        lappd_response=args.lappd_response,
        max_bounces=args.max_bounces,
        water_absorption_mm=args.water_absorption_mm,
        water_scattering_mm=args.water_scattering_mm,
        output_dir=args.output_dir,
        record_events=not args.no_record,
        seed=args.seed,
    )

    if args.muon_file and not args.muon_file.exists():
        print(f"Error: muon file not found: {args.muon_file}")
        return

    geometry = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))

    hd = geometry.lappd_housing_data[0]
    pc = geometry.annie_lappd_data[0]
    print("Standalone LAPPD module (single ANNIE housing)")
    print(f"  Housing centre: ({hd[0]:.0f}, {hd[1]:.0f}, {hd[2]:.0f}) mm, "
          f"half-extents: ({hd[12]:.0f}, {hd[13]:.0f}, {hd[14]:.0f}) mm")
    print(f"  Photocathode centre: ({pc[0]:.0f}, {pc[1]:.0f}, {pc[2]:.1f}) mm, "
          f"half: {pc[6]:.1f} mm")
    print(f"  Source: {config.source}, sample-mode: {config.sample_mode}, "
          f"events: {config.n_events}")

    if args.show_config:
        print(f"\n{'=' * 20} config {'=' * 20}")
        from litepath.lappd_batch import config_to_dict
        for k, v in config_to_dict(config).items():
            print(f"  {k}: {v}")
        print("=" * 48)
        return

    optics_cfg = None
    from litepath.optics import load_optics_config, WaterAttenuation
    water_cfg = WaterAttenuation(
        absorption_length_mm=args.water_absorption_mm,
        scattering_length_mm=args.water_scattering_mm,
    )
    if args.max_bounces > 0 or water_cfg.is_active:
        optics_cfg = load_optics_config(None)

    paths = run_lappd_batch(geometry, config, optics_config=optics_cfg)

    # run_lappd_batch writes the 'detectors' table into the file itself.
    h5_path = args.output_dir / "output.h5"
    if h5_path.exists():
        print(f"  Output: {h5_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# Fit command
# ---------------------------------------------------------------------------


def _parse_range(s: str) -> tuple[float, float, int]:
    """Parse 'start stop steps' into a 3-tuple."""
    parts = s.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 'start stop steps', got '{s}'")
    return float(parts[0]), float(parts[1]), int(parts[2])


def fit_command(args: argparse.Namespace) -> None:
    from litepath.fitting import load_observed_event, grid_scan_direction

    # Resolve PMT CSV (default: PMTPositions_Scan.txt in current directory)
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
        no_gdml=args.no_gdml,
        no_pmt_holders=args.no_pmt_holders,
        z_offset=args.z_offset,
        lappd_model=args.lappd_model,
        det_rotation_deg=args.det_rotation,
        n_surfboards=args.surfboard,
    )

    # Parse grid ranges
    print(f"Loading observed data for event {args.event}...")
    observed = load_observed_event(args.h5, args.event)

    theta_range = _parse_range(args.grid_theta)
    phi_range = _parse_range(args.grid_phi)

    # Windowed mode: centre on true/known direction
    if args.theta_window is not None:
        phi_window = args.phi_window if args.phi_window is not None else args.theta_window
        centre_theta = observed.true_theta
        centre_phi = observed.true_phi
        if centre_theta is None:
            print("Error: --theta-window requires a known true direction "
                  "(muon_truth in HDF5)")
            return
        steps = args.grid_steps
        t_start = centre_theta - args.theta_window
        t_end = centre_theta + args.theta_window
        p_start = centre_phi - phi_window
        p_end = centre_phi + phi_window
        theta_range = (t_start, t_end, steps)
        phi_range = (p_start, p_end, steps)

    print(f"  Grid: θ∈[{theta_range[0]:.1f},{theta_range[1]:.1f}]×{theta_range[2]} "
          f"φ∈[{phi_range[0]:.1f},{phi_range[1]:.1f}]×{phi_range[2]}")
    
    # Parse fix-vertex
    fix_vertex = None
    if args.fix_vertex:
        parts = args.fix_vertex.split()
        if len(parts) != 3:
            print("Error: --fix-vertex requires 3 floats: x y z")
            return
        fix_vertex = tuple(float(p) for p in parts)

    print(f"  True direction: θ={observed.true_theta:.1f}°, φ={observed.true_phi:.1f}°")
    if observed.true_pos:
        print(f"  True vertex: ({observed.true_pos[0]:.0f}, {observed.true_pos[1]:.0f}, "
              f"{observed.true_pos[2]:.0f}) mm")
    print(f"  {len(observed.hit_pmt_indices)} / {len(observed.all_pmt_indices)} PMTs hit")

    ppcm = args.photons_per_cm or observed.true_photons_per_cm or 150
    if args.photons_per_cm is None:
        print(f"  Photons per cm: auto-detected as {ppcm} from muon_truth")
    else:
        print(f"  Photons per cm: {ppcm} (user-specified)")

    time_sigma: float | None = args.time_sigma

    result = grid_scan_direction(
        observed,
        geom,
        theta_range=theta_range,
        phi_range=phi_range,
        fix_vertex=fix_vertex,
        fix_t0=args.fix_t0,
        photons_per_cm=args.photons_per_cm,
        use_time=args.use_time,
        alpha=args.alpha,
        time_sigma=time_sigma,
        rng_seed=args.seed,
    )

    print(f"\nBest fit: θ={result.best_theta:.1f}°, φ={result.best_phi:.1f}°")
    if result.true_theta is not None:
        d_theta = result.best_theta - result.true_theta
        # Wrap to [-180, 180) so 359° vs 1° reads as 2°, not 358°.
        d_phi = (result.best_phi - result.true_phi + 180.0) % 360.0 - 180.0
        print(f"  True:    θ={result.true_theta:.1f}°, φ={result.true_phi:.1f}°")
        print(f"  Δθ={d_theta:+.1f}°, Δφ={d_phi:+.1f}°")
    print(f"  LL = {result.best_score:.1f}")
    print(f"  {result.timing['n_evaluations']} evaluations in {result.timing['elapsed_s']:.1f}s")

    if args.save_grid:
        np.savez(
            args.save_grid,
            theta_grid=result.theta_grid,
            phi_grid=result.phi_grid,
            scores=result.scores,
            best_theta=result.best_theta,
            best_phi=result.best_phi,
            true_theta=result.true_theta,
            true_phi=result.true_phi,
        )
        print(f"  Saved grid to {args.save_grid}")

    if args.show:
        try:
            from scripts.fit_viewer import plot_polar, plot_rectangular, is_full_sky
            import matplotlib.pyplot as plt
            use_polar = args.polar or is_full_sky(result)
            if use_polar:
                plot_polar(result)
            else:
                plot_rectangular(result)
            plt.show()
        except ImportError:
            print("  --show requires scripts/fit_viewer.py (not found)")


def main(argv: list[str] | None = None) -> None:
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f32)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_command(args)
    elif args.command == "extract-manifest":
        extract_manifest_command(args)
    elif args.command == "build-detector-config":
        build_detector_config_command(args)
    elif args.command == "viz-server":
        from litepath.viz_server import run_server

        run_server(args)
    elif args.command == "viz-lappd":
        from litepath.viz_lappd_server import run_server as run_lappd_server

        run_lappd_server(host=args.host, port=args.port)
    elif args.command == "batch":
        batch_command(args)
    elif args.command == "batch-lappd":
        lappd_batch_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    elif args.command == "fit":
        fit_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
