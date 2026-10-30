# LitePath

GPU-accelerated ray tracer for the ANNIE Phase II water Cherenkov detector at Fermilab. Replaces Geant4's stepwise photon integration with analytic ray-geometry intersections on the GPU using Taichi.

## Physics Context

ANNIE (Accelerator Neutrino Neutron Interaction Experiment) measures final-state neutron multiplicity from CC neutrino interactions on oxygen. Phase II is a 26-ton Gd-doped water Cherenkov detector with 132 PMTs, 5 LAPPDs, Front Muon Veto, and Muon Range Detector. This tool simulates Cherenkov photon propagation through the detector geometry to predict PMT/LAPPD hit patterns.

Key physics:
- Cherenkov photons generated from muon tracks (cos θ = 1/nβ)
- Multi-bounce optics: Fresnel reflection/transmission, diffuse reflection
- Water attenuation: absorption + Rayleigh scattering
- PMT digital model: SPE charge + transit time spread
- LAPPD readout: strip-aligned local coordinates

## Project Structure

```
litepath/
├── litepath/          # Main package (28 Python files)
│   ├── cli.py         # CLI entry point (batch, fit, viz-server, etc.)
│   ├── geometry/build.py  # Geometry dataclass, build_geometry(), surfboards
│   ├── engine/kernel.py   # Taichi intersection functions, trace_kernel
│   ├── engine/trace.py    # trace_rays(), trace_with_optics()
│   ├── sources/track.py   # Light-generating part of a muon track (inside the tank)
│   ├── simulate.py    # trace_muon_light() / trace_cherenkov()
│   ├── hits.py        # Hit-array columns and detector/source codes
│   ├── tracer.py      # Compatibility re-exports of the above (old imports)
│   ├── batch.py       # Batch-mode event loop, HDF5 output
│   ├── cherenkov.py   # Cherenkov cone / isotropic photon generator
│   ├── fitting.py     # Direction grid scan, observed event loader
│   ├── likelihood.py  # Poisson charge + Gaussian time residual LL
│   ├── gdml_parser.py # GDML XML → vertex/triangle arrays
│   ├── pmt_loader.py  # PMT CSV parser, coordinate transforms
│   ├── pmt_mesh.py    # PMT body/hardware mesh loading
│   ├── step_parser.py # STEP CAD → component manifest
│   ├── detectors.py   # DetectorInfo dataclass, YAML I/O
│   ├── optics.py      # Optical material database, Fresnel evaluation
│   ├── pmt_response.py  # PMT digital model (fast + full waveform)
│   ├── lappd_response.py # LAPPD digitization (Taichi-accelerated)
│   ├── lappd_model.py  # LAPPD housing geometry builder
│   ├── benchmark.py    # WAGLE ROOT (data/MC) → litepath HDF5 conversion
│   ├── viz_server.py  # Interactive Three.js 3D viewer
│   └── io_h5.py       # HDF5 table I/O
├── scripts/           # Analysis/plotting utilities (13 files)
│   ├── event_display.py  # Event display (batch + WAGLE benchmark files)
│   ├── mc_muon_file.py   # Export a MC track as a muon topology file
│   └── check_benchmark_geometry.py  # Verify the WAGLE→world transform
├── tools/             # Developer/debugging utilities (23 files)
├── docs/              # Documentation, thesis PDFs (BENCHMARK.md = WAGLE workflow)
├── pmt_meshes/        # PMT body/hardware meshes (GDML + numpy cache)
├── extern/LApyPD/     # Git submodule (LAPPD physics model)
├── results/           # Batch simulation output (HDF5)
└── PHASE2_INNER_STRUCTURE_closed.gdml  # Primary detector geometry
```

## Build & Run

```bash
# Install (editable mode)
pip install -e .

# Run tests
pytest

# Batch simulation
python -m litepath batch \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100 --photons-per-cm 150

# Batch simulation without inner structure (PMTs + LAPPDs only)
python -m litepath batch --no-gdml \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100

# Batch simulation without PMT holder meshes (faster, PMT positions still loaded)
python -m litepath batch --no-pmt-holders \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100

# Batch simulation with Cherenkov + wbLS scintillation
python -m litepath batch \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100 --photons-per-cm 150 \
    --scintillation --photons-per-cm-scint 100

# Scintillation-only batch (Cherenkov off)
python -m litepath batch \
    --pmt-csv PMTPositions_Scan.txt \
    --events 100 --photons-per-cm 0 \
    --scintillation --photons-per-cm-scint 100

# Batch mode on a standalone LAPPD module (same output format as 'batch')
python -m litepath batch-lappd \
    --source cherenkov --sample-mode grid \
    --grid-steps 9 5 --photons-per-cm 150 --lappd-response

# Standalone LAPPD batch with a spot beam onto the photocathode
python -m litepath batch-lappd \
    --source spot --n-photons 5000 --spread-deg 2 \
    --intercept 0 -45 3.5 --lappd-response

# Direction fit
python -m litepath fit output.h5 --event 0 --show

# 3D viewer (scintillation toggle + parameters in the Trace panel)
python -m litepath viz-server --pmt-csv PMTPositions_Scan.txt --surfboard 3 --port 8080

# Convert WAGLE ROOT (ANNIE data / MC) into the litepath HDF5 schema
python -m litepath benchmark \
    --data benchmarking_data/ANNIETree_ThroughgoingMuons.root \
    --mc   benchmarking_data/MCTree_ThroughgoingMuons.root \
    --output-dir results/benchmark

# Check the converted file against the raytracer PMT registry
python scripts/check_benchmark_geometry.py results/benchmark/mc.h5

# Show a WAGLE MC event (use --center-phi 135: the WAGLE beam is at phi=135)
python scripts/event_display.py results/benchmark/mc.h5 --charge --center-phi 135
```

### CLI Flags for Geometry Control

| Flag | Default | Description |
|------|---------|-------------|
| `--no-gdml` | false | Skip inner structure GDML mesh (tank + PMTs only) |
| `--no-pmt-holders` | false | Skip PMT body and hardware holder meshes (PMT positions still loaded) |
| `--no-lappd` | false | Skip LAPPD rectangles |
| `--surfboard` | 0 | Number of obscurant PVC surfboards (0, 1, or 3) — use 3 for Phase II |
| `--scintillation` | false | Enable wbLS scintillation photon generation (adds to Cherenkov) |
| `--photons-per-cm-scint` | 100 | Scintillation photons per cm of track |
| `--tau-fast` | 2.0 | Fast scintillation decay time (ns) |
| `--tau-slow` | 20.0 | Slow scintillation decay time (ns) |
| `--fast-fraction` | 0.95 | Fraction of scintillation light in the fast component |
| `--wavelength-scint` | 420.0 | Scintillation photon wavelength (nm) |

### `batch-lappd` Flags (standalone LAPPD module)

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `cherenkov` | Light source: `cherenkov` (tracked muon cone) or `spot` (directed beam) |
| `--sample-mode` | `random` | Where to place the source: `random`, `grid`, or `fixed` |
| `--events` | `-1` | Number of events (default: grid `NX*NZ`, muon-file lines, or 100) |
| `--grid-steps NX NZ` | `9 5` | Grid size for `--sample-mode grid` |
| `--muon-fixed` | — | Fixed source: `'x y z t0 dx dy dz'` (7 floats) |
| `--muon-file` | — | File with one `x y z t0 dx dy dz` topology per line |
| `--photons-per-cm` | 150 | Cherenkov photons per cm along the muon track |
| `--n-photons` | 5000 | Photons in the `spot` beam |
| `--spread-deg` | 2.0 | Angular spread (deg) of the launched source |
| `--intercept X Y Z` | — | Aim target (PC intercept); overrides aiming at PC centre, recorded in truth |
| `--wavelength` | 350.0 | Photon wavelength (nm) |
| `--lappd-response` | false | Enable LAPPD digital readout pipeline (writes `lappd_readouts`) |
| `--max-bounces` | 0 | Number of surface reflections per photon (0 = off) |
| `--water-absorption-mm` | 0.0 | Water absorption length (mm, 0 = off) |
| `--water-scattering-mm` | 0.0 | Rayleigh scattering length (mm, 0 = off) |
| `--output-dir`/`-o` | `results` | Output directory (writes `output.h5`) |
| `--no-record` | false | Skip writing per-event output files |
| `--seed` | — | Random seed |

Both `batch` and `batch-lappd` write the same HDF5 schema (v1):
`photon_hits`, `muon_truth`, `lappd_readouts` (flat, per event+LAPPD),
`pmt_responses` (`batch` only), `detectors`, plus `format`/`schema_version`/
`detector_mode`/`config` root attrs.  See `docs/BATCH_OUTPUT.md` and use
`BatchReader` in `litepath/io_h5.py` to load files.

## Key Data Files

| File | Description |
|------|-------------|
| `PHASE2_INNER_STRUCTURE_closed.gdml` | Primary GDML detector geometry mesh |
| `PMTPositions_Scan.txt` | 132 PMT positions (TSV, 9 columns) |
| `component_manifest.json` | Cached STEP parser output (PMT/LAPPD positions) |
| `corrections.csv` | LAPPD correction data |
| `MuonStartsAndDirecs.txt` | Muon start positions/directions |
| `benchmarking_data/ANNIETree_ThroughgoingMuons.root` | WAGLE ANNIE data (10112 events, MRD reconstruction only) |
| `benchmarking_data/MCTree_ThroughgoingMuons.root` | WAGLE Monte Carlo (1432 events, generator truth) |

## Conventions

- **Python 3.11+** with type hints
- **Taichi** for GPU kernels (`@ti.kernel`, `@ti.func`)
- **NumPy** for CPU-side array operations
- **HDF5** (h5py) for batch output
- **GDML** (lxml) for detector geometry parsing
- **CadQuery** for STEP CAD file parsing
- CLI via `python -m litepath <command>`
- No heavy frameworks — keep dependencies minimal

## Muon Direction Conventions

- **θ (theta)**: polar angle from vertical. 0° = downward (−z), 90° = horizontal, 180° = upward (+z)
- **φ (phi)**: azimuthal in XY plane. 0° = +x, 90° = +y

## WAGLE / Benchmark Conventions

- **ROOT (WAGLE) frame**: x horizontal transverse, **y vertical (up)**, **z along the beam**
  (downstream), origin at the tank centre.  The WCSim *hall* frame instead has y
  along the beam and z vertical, so the two differ by a pure y/z swap.
- **World frame** = litepath structure frame: z vertical, tank axis along z.
  `world = Rz(det_rotation)·(Rz(-157.5°)·Rx(+90°)·root + (0,0,1980))` — see
  `litepath/benchmark.py` and `docs/BENCHMARK.md`.
- The **WAGLE beam runs at φ = 135°** in the world frame (an octagon corner).
  `litepath batch --muon-mode beam` points along **+Y (90°)**, which is 45°
  away — a CLI labelling convention, not a frame error.
- WAGLE `hitX/Y/Z` is the **PMT centre** (looked up by Tube ID), not a
  per-photon position; WCSim places it 10.4 mm (8") / 23.7 mm (ETEL 10%)
  further outward than the survey position.
- The raytracer machinery is **not** modified by the benchmarking work; the
  conversion is a read-only consumer of the litepath geometry.

## Current State

The core ray tracer is complete and working. All major features implemented:
- GDML and STEP geometry parsing
- Taichi GPU ray tracing with BVH acceleration
- Cherenkov photon generation (vectorized)
- wbLS scintillation photon generation (isotropic, delayed emission) with
  per-source tagging (`photon_source`: 0=Cherenkov, 1=scintillation) through
  the hit array, HDF5 output, and 3D viewer
- Batch-mode simulation with HDF5 output
- PMT and LAPPD digital response models
- Multi-bounce Fresnel optics
- Direction fitting framework with grid scanning
- Interactive 3D visualization server

Recent work focuses on validation, analysis scripts, and fitting accuracy.

## Testing

```bash
pytest                         # Run all fast tests (slow ones deselected)
pytest -m slow                 # Run slow end-to-end tests (geometry + GPU kernel)
pytest tests/test_reference_outputs.py -m "slow or not slow"  # End-to-end output vs stored reference
pytest tests/test_timing_e2e.py -m slow  # End-to-end arrival-time sanity test
pytest tests/test_batch_event_split.py  # Per-event hit attribution in batch mode
```

Timing tests assert that final `arrival_time = create_time + path/c_in_water`,
where `create_time` already includes the muon arrival at the emission point
(Cherenkov) or muon arrival + scintillation decay delay (scintillation).

Note: the generators store the muon start time `t0` in **seconds** (they
internally convert via `*10**9` to ns), so pass `t0` as seconds, not ns.

**Batched hits and per-event attribution.**  `run_batch` traces a whole
batch of events in one GPU launch and then slices the detected hits back
into events.  The hit array is *not* one row per ray when the optics path
runs (`--max-bounces > 0` or water attenuation), so the per-event
boundaries must be derived from `orig_indices`, not from the photon
counts — see `tests/test_batch_event_split.py`, which asserts that the
per-event result does not depend on the batch size.

## Git Submodules

```bash
git submodule update --init --recursive
```

## Tips for New Sessions

- The project is already mature — ask what the user wants to work on rather than starting from scratch
- GDML parsing is in `gdml_parser.py`, PMT loading in `pmt_loader.py`
- The Taichi kernel is in `engine/kernel.py` (`trace_kernel`); `tracer.py` only re-exports
- Batch output goes to `results/` as HDF5 files
- Use `@` to reference files when discussing specific code
- PDFs in `docs/` can be read with `pymupdf` (already installed)
