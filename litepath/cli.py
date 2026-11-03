"""CLI for litepath ray tracer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from litepath.optics import WaterAttenuation, load_optics_config
from litepath.output import write_detector_config

# Older per-command geometry flags.  They still work, but build an unnamed
# geometry version and print a deprecation warning; --geometry replaces them.
_GEOMETRY_FLAGS = (
    "gdml", "step", "manifest", "pmt_csv", "z_offset", "lappd_model", "det_rotation",
    "surfboard", "lappd_indices", "bottom_rot", "bottom_spin",
)
# Switches that drop parts of whichever version is used (for speed or
# debugging).  Not deprecated; the result is an unnamed derived version.
_GEOMETRY_SWITCHES = {
    "no_gdml": "structure_mesh", "no_pmt_holders": "pmt_meshes", "no_lappd": "lappds",
}


def add_geometry_option(parser: argparse.ArgumentParser) -> None:
    """Add --geometry and make the parser's older geometry flags overrides.

    Call after all of the parser's arguments are defined.  The older flags
    and the switches get a ``None`` default so :func:`resolve_geometry` can
    tell whether the user set them; the flags' original defaults are kept in
    ``_geometry_defaults``.
    """
    from litepath.geometry.versions import DEFAULT_GEOMETRY
    parser.add_argument("--geometry", type=str, default=None, metavar="NAME",
                        help="Geometry version name or YAML file (default: "
                             f"{DEFAULT_GEOMETRY}); see 'litepath geometry list'. "
                             "Replaces the individual geometry flags below")
    dests = {a.dest for a in parser._actions}
    defaults = {a.dest: a.default for a in parser._actions if a.dest in _GEOMETRY_FLAGS}
    switches = [k for k in _GEOMETRY_SWITCHES if k in dests]
    parser.set_defaults(_geometry_defaults=defaults,
                        **{k: None for k in [*defaults, *switches]})


def resolve_geometry(args: argparse.Namespace):
    """The GeometryVersion selected by --geometry or by the older flags."""
    from litepath.geometry.versions import (
        DEFAULT_GEOMETRY, STATUS_DRAFT, STATUS_UNNAMED, GeometryError, fingerprint, list_versions,
        load_version, version_from_flags,
    )
    defaults = args._geometry_defaults
    given = {k: getattr(args, k) for k in defaults if getattr(args, k) is not None}
    switches = [k for k in _GEOMETRY_SWITCHES if getattr(args, k, None)]
    if given and args.geometry:
        flags = ", ".join("--" + k.replace("_", "-") for k in given)
        raise GeometryError(f"--geometry cannot be combined with {flags}")
    if given:
        flags = ", ".join("--" + k.replace("_", "-") for k in given)
        version = version_from_flags({**defaults, **given, **{k: True for k in switches}})
        fp = fingerprint(version)
        same = [n for n in list_versions() if fingerprint(load_version(n)) == fp]
        hint = f"--geometry {same[0]} builds the same detector" if same \
            else "use --geometry NAME (see 'litepath geometry list')"
        print(f"Warning: {flags} {'is' if len(given) == 1 else 'are'} deprecated and will "
              f"be removed; {hint}")
        return version

    version = load_version(args.geometry or DEFAULT_GEOMETRY)
    if switches:
        for k in switches:
            version.settings[_GEOMETRY_SWITCHES[k]] = False
        version.name += "".join("+" + k.replace("_", "-") for k in switches)
        if version.status != STATUS_DRAFT:  # a draft stays editable
            version.status = STATUS_UNNAMED
        version.tagged = {}
    return version


def load_geometry_build(args: argparse.Namespace):
    """Resolve and build the geometry for a command; None (after a message) on error."""
    from litepath.geometry.versions import GeometryError, load_build
    try:
        return load_build(resolve_geometry(args))
    except GeometryError as exc:
        print(f"Error: {exc}")
        return None


def geometry_command(args: argparse.Namespace) -> None:
    from litepath.geometry.versions import (
        GeometryError, file_hashes, fingerprint, list_versions, load_version, tag_version,
    )
    try:
        if args.geometry_action == "list":
            for name in list_versions():
                v = load_version(name)
                desc = v.description if len(v.description) <= 70 else v.description[:67] + "..."
                print(f"{name:24s} {v.status:7s} {fingerprint(v)[:12]}  {desc}")
        elif args.geometry_action == "show":
            v = load_version(args.name)
            hashes = file_hashes(v)
            print(f"{v.name} ({v.status})\n  {v.description}\n  file: {v.path}")
            print(f"  fingerprint: {fingerprint(v, hashes)}")
            for role, path in v.inputs.items():
                if path is not None:
                    print(f"  {role}: {path}")
            for key, val in v.settings.items():
                print(f"  {key}: {val}")
            for o in v.overlays:
                print(f"  overlay: {o}")
            print("  files:")
            for key, h in hashes.items():
                print(f"    {h[:12]}  {key}")
        elif args.geometry_action == "tag":
            fp = tag_version(args.name)
            print(f"Tagged {args.name}: fingerprint {fp}")
    except GeometryError as exc:
        print(f"Error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="litepath",
        description="GPU-accelerated ray tracer for ANNIE detector",
    )
    sub = p.add_subparsers(dest="command", required=True)

    geo = sub.add_parser("geometry", help="List, show or tag geometry versions")
    geo_sub = geo.add_subparsers(dest="geometry_action", required=True)
    geo_sub.add_parser("list", help="List geometry versions")
    geo_show = geo_sub.add_parser("show", help="Show a version's inputs, settings and hashes")
    geo_show.add_argument("name", help="Version name or YAML file")
    geo_tag = geo_sub.add_parser("tag", help="Freeze a draft version (records its hashes)")
    geo_tag.add_argument("name", help="Version name or YAML file")

    cherenkov = sub.add_parser("extract-manifest", help="Extract component manifest from STEP")
    cherenkov.add_argument("--step", required=True, type=Path, help="Path to STEP CAD file")
    cherenkov.add_argument("--output", "-o", type=Path, default=Path("component_manifest.json"), help="Output JSON path")

    viz = sub.add_parser("viz-server", help="Start interactive Cherenkov visualization server")
    viz.add_argument("--gdml", type=Path, default=Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
                     help="Path to GDML geometry mesh (default: PHASE2_INNER_STRUCTURE_closed.gdml)")
    viz.add_argument("--step", type=Path, default=None, help="Path to STEP CAD file for component positions")
    viz.add_argument("--manifest", type=Path, default=None, help="Path to cached component manifest JSON")
    viz.add_argument("--pmt-csv", type=Path, default=Path("PMTPositions_Scan.txt"),
                     help="Path to PMT scan file")
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
                     help="Deprecated and ignored: the detector orientation is fixed (only 45 is accepted)")
    viz.add_argument("--bottom-spin", type=float, default=22.5,
                     help="Per-PMT spin (deg) about forward axis for bottom PMTs (default: 22.5)")
    viz.add_argument("--det-rotation", type=float, default=22.5,
                     help="Deprecated and ignored: the detector orientation is fixed (only 22.5 is accepted)")
    viz.add_argument("--surfboard", type=int, default=0, choices=[0, 1, 3],
                     help="Number of obscurant PVC surfboards (0, 1, or 3)")

    gviz = sub.add_parser("geometry-viz",
                          help="Start the geometry workbench: view, compare and adjust the detector geometry")
    gviz.add_argument("--host", type=str, default="localhost", help="Host to bind (default: localhost)")
    gviz.add_argument("--port", type=int, default=8082, help="Port to bind (default: 8082)")
    gviz.add_argument("--no-lappd", action="store_true", help="Skip LAPPDs")
    gviz.add_argument("--no-gdml", action="store_true",
                      help="Skip inner structure GDML mesh (tank + PMTs only)")
    gviz.add_argument("--no-pmt-holders", action="store_true",
                      help="Skip PMT body and hardware holder meshes (PMT positions still loaded)")

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
                        help="Deprecated and ignored: the detector orientation is fixed (only 22.5 is accepted)")

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
    batch.add_argument("--mask-dead", action="store_true",
                       help="Drop hits on PMTs marked OFF in the geometry's PMT table "
                            "(ToolAnalysis status)")
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
                       help="Deprecated and ignored: the detector orientation is fixed (only 22.5 is accepted)")
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
                       help="Deprecated and ignored: the detector orientation is fixed (only 22.5 is accepted)")
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
    lappd_batch.add_argument("--max-track-length", type=float, default=4.0,
                             help="World boundary (m): light stops this far from the "
                                  "muon start if it never passes behind the module "
                                  "(default: 4.0)")
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
                             help="Not implemented for batch-lappd yet; must be 0")
    lappd_batch.add_argument("--water-absorption-mm", type=float, default=0.0,
                             help="Not implemented for batch-lappd yet; must be 0")
    lappd_batch.add_argument("--water-scattering-mm", type=float, default=0.0,
                             help="Not implemented for batch-lappd yet; must be 0")
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
    fit.add_argument("--pmt-csv", type=Path, default=Path("PMTPositions_Scan.txt"),
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
                     help="Deprecated and ignored: the detector orientation is fixed (only 22.5 is accepted)")
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

    for geometry_parser in (detcfg, batch, fit, bench, viz, gviz):
        add_geometry_option(geometry_parser)

    return p


def build_detector_config_command(args: argparse.Namespace) -> None:
    build = load_geometry_build(args)
    if build is None:
        return
    geom = build.geometry

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
    build = load_geometry_build(args)
    if build is None:
        return
    geom = build.geometry
    if args.mask_dead:
        records = (geom.pmt_info or {}).get("records", [])
        config.masked_pmt_indices = tuple(i for i, r in enumerate(records) if r.status == "OFF")
        keys = [records[i].channel_key for i in config.masked_pmt_indices]
        print(f"  Masking {len(keys)} PMTs marked OFF: {keys}")

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
        from litepath.geometry.versions import build_provenance
        from litepath.io_h5 import write_metadata, write_detectors
        meta = {
            "tank_radius_mm": geom.tank_radius,
            "tank_z_min_mm": geom.tank_z_min,
            "tank_z_max_mm": geom.tank_z_max,
            **build_provenance(build),
        }
        write_metadata(h5_path, meta)
        print(f"  Wrote metadata to {h5_path}")
        write_detectors(h5_path, geom.detectors, geom.pmt_info)

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

    from litepath.geometry.versions import GeometryError
    try:
        version = resolve_geometry(args)
    except GeometryError as exc:
        print(f"Error: {exc}")
        return
    pmt_csv = version.inputs["pmt_table"]
    print(f"Building PMT registry from {pmt_csv} (geometry: {version.name})")
    registry = load_pmt_registry(pmt_csv)
    from litepath.geometry.versions import version_provenance
    provenance = version_provenance(version)
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
            provenance=provenance,
            write_raw_hits=not args.no_raw_hits,
        )
        print(f"  Wrote {out_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Standalone LAPPD batch command
# ---------------------------------------------------------------------------

def lappd_batch_command(args: argparse.Namespace) -> None:
    from litepath.lappd_batch import LAPPDBatchConfig, check_supported_optics, run_lappd_batch
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
        max_track_length_m=args.max_track_length,
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

    try:
        check_supported_optics(config)
    except ValueError as exc:
        print(f"Error: {exc}")
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

    paths = run_lappd_batch(geometry, config)

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

    build = load_geometry_build(args)
    if build is None:
        return
    geom = build.geometry

    from litepath.io_h5 import read_attrs
    file_geometry = read_attrs(args.h5).get("geometry_hash")
    if file_geometry and file_geometry != build.fingerprint:
        name = read_attrs(args.h5).get("geometry_version", "?")
        print(f"  Warning: {args.h5} was simulated with geometry '{name}' "
              f"({file_geometry[:12]}), but the fit uses '{build.version.name}' "
              f"({build.fingerprint[:12]}); pass the same --geometry")

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

    if args.command == "geometry":
        geometry_command(args)
    elif args.command == "extract-manifest":
        extract_manifest_command(args)
    elif args.command == "build-detector-config":
        build_detector_config_command(args)
    elif args.command == "viz-server":
        from litepath.viz_server import run_server

        run_server(args)
    elif args.command == "geometry-viz":
        from litepath.geometry_viz import run_server as run_geometry_viz

        run_geometry_viz(args)
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
