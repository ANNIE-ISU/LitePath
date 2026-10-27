# WAGLE Benchmark Data → litepath HDF5

`litepath benchmark` converts the two WAGLE ROOT files in `benchmarking_data/`
into the litepath batch HDF5 schema, so that data, Monte Carlo and raytracer
output can be analysed with the same tools.

| File | Events | Truth |
|------|--------|-------|
| `benchmarking_data/ANNIETree_ThroughgoingMuons.root` | 10112 | none (MRD reconstruction only) |
| `benchmarking_data/MCTree_ThroughgoingMuons.root` | 1432 | generator truth (`trueVtx*`, `trueDir*`, …) |

## Usage

```bash
# Convert both files (≈20 s total)
python -m litepath benchmark \
    --data benchmarking_data/ANNIETree_ThroughgoingMuons.root \
    --mc   benchmarking_data/MCTree_ThroughgoingMuons.root \
    --output-dir results/benchmark

# One file, first 100 events only
python -m litepath benchmark --mc benchmarking_data/MCTree_ThroughgoingMuons.root \
    --output-dir results/benchmark --max-events 100
```

Outputs `results/benchmark/data.h5` and `results/benchmark/mc.h5`.

| Flag | Default | Description |
|------|---------|-------------|
| `--data` / `--mc` | — | ROOT files to convert (at least one required) |
| `--tree` | `Event` | TTree name inside the ROOT files |
| `--pmt-csv` | `PMTPositions_Scan.txt` | PMT scan file used to build the detector registry |
| `--det-rotation` | `22.5` | Global Z-rotation (deg), must match the raytracer runs |
| `--output-dir` / `-o` | `results/benchmark` | Output directory |
| `--max-events` | `0` (all) | Convert at most N events per file |
| `--no-raw-hits` | false | Skip the per-hit `pmt_hits` table |

Requires `uproot` and `awkward` (`pip install -e '.[benchmark]'`).

## Coordinate frames

WAGLE (ROOT) frame: **x** horizontal transverse, **y** vertical (up),
**z** along the beam (downstream), origin at the tank centre.

The WCSim *hall* frame that the PMT survey is expressed in has **x**
horizontal, **y along the beam** and **z vertical** — so the WAGLE tree and
the survey differ by a **pure y/z swap** (`Rx(-90°)`), with no rotation
about the beam axis in between.  This was confirmed with a Procrustes fit of
the 121 hit PMT point clouds: the fitted rotation *is* that swap, and a free
rotation gives the same residual (12.7 mm) as the constrained swap
(12.5 mm), so nothing is hidden in the chain.

litepath world frame: **z** vertical, tank axis along z, tank spans
z = 19…3861 mm, radius 1524 mm.

The transform comes from the litepath geometry chain (scan frame → hall
frame → structure frame, see `litepath/pmt_loader.py`) and is **not fitted to
the WAGLE data**:

```
world = Rz(det_rotation) · ( Rz(-157.5°) · Rx(+90°) · root + (0, 0, 1980) )
```

with `det_rotation = 22.5°` by default — the same value the raytracer uses
(`litepath pmt_loader.load_pmts`). `litepath.benchmark` exposes
`root_to_world`, `world_to_root` and the direction-only variants.

Note: the WCSim tank centre maps to z = 1980 mm while the water tank
mid-plane is at z = 1940 mm — a known 40 mm offset between the two
conventions.

### What the WAGLE data is checked against

| Check | Result |
|---|---|
| PMT radial position (transformed WAGLE vs. raytracer centre) | < 0.01 mm |
| PMT azimuth | < 0.09° |
| hall ↔ ROOT rotation | pure y/z swap (Procrustes fit) |
| GDML inner-structure panels vs. PMT rows | panels at 22.5° + k·45°, PMT rows straddle them at ±8.5° → `det_rotation = 22.5°` is the correct GDML↔PMT alignment |
| Beam azimuth, MC truth muon directions (400 events) | 135.2° (σ = 10.5°) |
| Beam azimuth, MRD track start points | 134.6° |
| Track exit point azimuth | 135.1° |

### The beam is at φ = 135°, not at +Y

In the litepath world frame the WAGLE beam runs at **φ = 135°**, which is a
corner of the inner-structure octagon (the panels are centred on
22.5° + k·45°, so the corners are at 45° + k·45°).

`litepath batch --muon-mode beam` points the muon along **+Y** (φ = 90°),
which is the *neighbouring* octagon corner — 45° away from the WAGLE beam.
That is a labelling convention of the CLI, not a frame error: the GDML, the
PMT registry and the WAGLE data all agree with each other.  When comparing
WAGLE events with raytracer output, remember that the beam is at 135°, and the
event display centres the barrel on the beam-facing octagon corner
automatically:

```bash
python scripts/event_display.py results/benchmark/mc.h5 --charge
```

### Angle conventions

| | theta | phi |
|---|---|---|
| litepath | `acos(-dz)` — 0° = straight down | `atan2(dy, dx)` |
| WAGLE | `acos(dirZ)` — from the beam axis (+z) | `atan2(dy, dx)` |

Both are written out (`theta_deg`/`phi_deg` and `root_theta_deg`/`root_phi_deg`)
so no conversion is needed downstream.

## Geometry provenance

The two geometries come from *disjoint* data, and only the frame transform
joins them:

| Piece | Source |
|---|---|
| PMT positions, directions, radii | the **ID**: `hitDetID` (WCSim TubeID 332–463) → `PMTPositions_Scan.txt` → the litepath registry |
| Track geometry (MC) | the truth branches `trueVtx*` / `trueDir*` — independent of any PMT data |
| Track geometry (data) | the MRD reconstruction (no generator truth available) |
| WAGLE `hitX/Y/Z` | the PMT centre as WCSim stores it — used only as a cross-check |

The 121 PMT IDs that appear in the MC hits are a strict subset of the 132
registry IDs (no foreign IDs), so the mapping is unambiguous and 1:1.  The
11 registry PMTs that are never hit in the MC are 6 LUX on the bottom
endcap, 1 ETEL on top and 4 Watchboy on the barrel.

Because the WAGLE `hitX/Y/Z` columns are looked up per Tube ID (raw WCSim
keeps the PMT geometry in a separate tree precisely so that it does not have
to be stored per hit), **every hit of a given PMT carries the same
position**: 0 of the 121 hit PMTs have more than one distinct value.  The
`pmt_hits` table therefore holds PMT centres, not per-photon photocathode
landing points — there is no photon-level position information in these
files to compare against the raytracer's `photon_hits`.

## Output schema

The standard tables use exactly the dtypes of `litepath batch`, so the files
can be fed to `BatchReader`, `litepath fit` and the analysis scripts:

| Table | Contents |
|-------|----------|
| `muon_truth` | Vertex, direction, angles, track length (true for MC, MRD reconstruction for data) |
| `pmt_responses` | Per-PMT charge (PE), first-hit time (ns) and hit count |
| `detectors` | The 132 PMTs (same rows as the raytracer writes) |

Two extension tables carry the information the standard schema has no place for:

| Table | Contents |
|-------|----------|
| `pmt_hits` | The raw digitised hits: per hit arrival time, PE, pC and hit position |
| `benchmark_truth` | Track truth/reconstruction in both frames, energy, track lengths, MRD info, hit/PE totals |

Root attributes: `schema_version`, `format`, `detector_mode`
(`benchmark-data` / `benchmark-mc`), `source_file`, `det_rotation_deg`,
`config`, plus the tank geometry (`tank_radius_mm`, `tank_z_min_mm`,
`tank_z_max_mm`) so the standard event display works on these files.

There is deliberately **no** `photon_hits` table: the ROOT files contain
digitised PMT hits, not photon-level truth.

### Units

| Quantity | Unit |
|----------|------|
| Positions, track lengths (water) | mm (m in the ROOT tree → converted) |
| `hitT` | ns |
| `hitPE` | PE |
| `hitQ` | pC (MC only) |
| `MRDTrackLength` | cm |
| `MRDTrackAngle` | rad |

## Event display

`scripts/event_display.py` reads the standard schema and now also handles the
benchmark files (hit counts and first-hit times come from `pmt_hits` when
there is no `photon_hits` table).

The muon track is not drawn as a line — instead the two points where it
crosses the tank boundary are marked: a **circle** for the entry point and a
**square** for the exit point.  The entry/exit parameters are solved
analytically (cylinder side and end caps), so they are exact.  A track that
misses the tank is flagged in the figure title.  Use `--no-track` to switch
the markers off.

```bash
# Monte-Carlo event, coloured by charge
python scripts/event_display.py results/benchmark/mc.h5 --charge

# Data events, coloured by first-hit time
python scripts/event_display.py results/benchmark/data.h5 --first-hit-time
```

Panels: top endcap (panel 9), unrolled barrel (panels 1–8), bottom endcap
(panel 0). Navigate with ◀/▶ or the arrow keys.

**Orientation.**  The barrel PMTs form 16 azimuthal columns and the octagon
corners between them fall on a 22.5° grid (22.5°·k); the WAGLE beam runs at
φ = 135°, straight into one of those corners.  By default the display now
derives the beam bearing from the `muon_truth` table, snaps it to the nearest
octagon corner, and centres the unrolled barrel on it — so for the WAGLE files
the beam-facing corner lands at plot centre and the barrel shows the region
that looks back at the beam.  A dashed vertical line marks the beam axis and
the centred corner is labelled.  `--center-phi` overrides the automatic choice
and is given in the *physical* convention (the title shows that value).

**Inside-view handedness.**  The WCSim scan → litepath world map is
left-handed in the `(pmt_x, pmt_z)` plane (`pmt_loader._scan_to_tank` uses the
WCSim-faithful `Y_w = (168.1 − pmt_z)·10`), so the world azimuth `atan2(y, x)`
winds opposite to the way the detector looks when viewed from *inside* the
tank.  The event display mirrors the azimuth about the vertical plane
(`plot_phi`) for both the barrel and the endcaps, so the plots read left-right
the way you see them standing inside the tank.  This is a **display-only**
change: the raytracer geometry (`pmt_loader`/`benchmark`/`tracer`) and the
visualiser (`viz_server`) are untouched, and the mirror is pinned by
`tests/test_event_display.py`.

To cross-check the orientation against the hardware, `--mark-pmt ID [ID ...]`
rings and labels specific PMTs on the barrel.  Each ID may be an litepath
detector index or a WAGLE Tube ID (e.g. `428` or `PMT_428`):

```bash
python scripts/event_display.py results/benchmark/mc.h5 --charge \
    --mark-pmt 428 406
```

The barrel is centred on the beam-facing corner automatically; add
`--center-phi 135` only if you want to pin it explicitly.

## Verifying the conversion

`scripts/check_benchmark_geometry.py` cross-checks a converted file against
the raytracer's own PMT registry — the two geometries are compared to each
other, independently of the code that produced the file:

```bash
python scripts/check_benchmark_geometry.py results/benchmark/mc.h5
```

It checks that

1. every WAGLE PMT id resolves to a raytracer PMT, and the two positions
   agree to < 0.3 mm perpendicular to the tube axis and < 0.5° in azimuth;
2. the offset between them is a constant *along the tube axis*, per PMT type
   (10.42 mm LUX/Watchboy/Watchman, 10.55 mm Hamamatsu, 23.72 mm ETEL — WCSim
   places the PMT centre that much further out than the survey point);
3. the endcaps stay on top and bottom, and the beam stays horizontal, i.e.
   the y/z swap went onto the right axis;
4. the beam azimuth is 135° and the vertices are inside the tank.

The script adapts to the file kind: for the *data* file the offsets are
expected to be small (the surveyed positions), and the track checks are
loosened because the vertices are MRD reconstructions rather than truth.

The same invariants are covered by `tests/test_benchmark.py`.

## Raytracing the same track

To run the litepath raytracer on a WAGLE MC event, export its track as a
muon topology file and feed it to `litepath batch`:

```bash
python scripts/mc_muon_file.py results/benchmark/mc.h5 --event 0 --out /tmp/ev0.txt
python -m litepath batch --muon-file /tmp/ev0.txt --events 1 \
    --pmt-csv PMTPositions_Scan.txt --surfboard 3 --pmt-response \
    --max-bounces 3 \
    --output-dir results/raytrace_ev0
```

`scripts/mc_muon_file.py` also accepts the original ROOT file directly
(`benchmarking_data/MCTree_ThroughgoingMuons.root`).

> **Note.**  Reflections (`--max-bounces`) are needed to reproduce the WAGLE
> hit pattern: without them only ~25 of the 132 PMTs see any light, because
> the photons that miss a PMT are lost instead of bouncing off the tank.
> `batch` had a bug that mis-attributed the detected hits to the wrong
> events when reflections were on (it sliced the hit array with the
> *generated photon* counts); this was fixed in `litepath/batch.py` and
> covered by `tests/test_batch_event_split.py`.

Show both side by side:

```bash
python scripts/event_display.py results/benchmark/mc.h5 \
    --compare results/raytrace_ev0/output.h5 --compare-event 0 --charge
```

The upper three panels are the WAGLE MC event, the lower three the raytracer
event; both share one colour scale.

### What the comparison shows

Over 20 WAGLE MC events (same tracks, `--max-bounces 3`, 150 photons/cm):

| | WAGLE MC | litepath |
|---|---|---|
| PMTs with a hit, per event | 120 (114–121) | 131 (128–132) |
| PMTs hit in both | 119 per event (99.5 % of the WAGLE set) | |
| WAGLE-hit PMTs missed by the raytracer | 0.6 per event | |
| Extra PMTs in the raytracer | | 12 per event |

i.e. essentially every PMT the WAGLE simulation lights up is also lit in the
raytracer (13 of the 20 events reproduce the WAGLE set exactly), and the
azimuth of the hit pattern agrees (median 192° vs 180°, both on the far side
of the tank from the φ = 135° beam).  The extra raytracer PMTs are the ones
closest to the digitisation threshold, which WCSim's stronger water
absorption keeps empty.  This is the end-to-end confirmation that the
ROOT → litepath transform is right.

## Notes and known differences

* **Timing reference.** In the MC tree, `hitT` is relative to the true vertex
  (t = 0). In the data tree, `hitT` is relative to the tank trigger, so the
  absolute times carry an event-dependent offset (~1.7 µs) and the raw values
  are kept as they are.
* **Data truth.** The data tree has no generator truth. The MRD reconstructed
  track is used instead. Its start/stop points define a straight line; that
  line is intersected with the water-tank cylinder and the **upstream
  crossing** is stored as the vertex, with the chord length as
  `track_length_mm`. If the line misses the tank (the MRD direction is
  reconstructed from short tracks and is sometimes inconsistent, ~33 % of
  data events), the vertex is stored as **NaN** so consumers skip the event
  instead of drawing a phantom track. The `benchmark_truth` table flags the
  source with `track_kind` (0 = generator truth, 1 = MRD reconstruction) and
  always keeps the raw MRD `mrd_start_*`/`mrd_stop_*` points.

  > History: the first version back-projected onto the ROOT `z = 0` plane.
  > Because that plane maps to `world x = y` under `root_to_world`, and the
  > MRD line was extrapolated ~3.4 m, vertices reached tens of metres from
  > the tank. This is fixed by the cylinder intersection above; see
  > `test_converted_data_vertices_are_physical`.
* **LAPPDs.** The data tree has LAPPD branches but only for a small subset of
  events, and the MC tree has none, so LAPPD readout is not converted.
* **Non-finite charge.** `hitPE` entries that are `inf`/`nan` are stored as 0.
* **PMT centre offset.** WCSim places the PMT sphere centre 10.4 mm (8") /
  23.7 mm (ETEL 10%) further *outward* than the survey position that the
  raytracer uses.  The raytracer was left untouched, so this offset is a
  known systematic of at most ~10 % of the PMT radius when comparing hit
  patterns.  The *data* file does not have it — its PMT positions agree with
  the survey to 0.3 mm (median) / 1.7 mm (worst PMT), with a per-hit jitter
  of ~1 mm — which is an independent confirmation of the frame transform.
* **No photon-level information.** The ROOT files hold digitised PMT hits
  only: there is no per-photon truth, and the hit positions are the PMT
  centres (see above).  The `pmt_hits` table therefore cannot be compared
  row-by-row against the raytracer's `photon_hits`.
* **Nothing in the raytracer was changed.** The conversion is a read-only
  consumer of the litepath geometry (`litepath.pmt_loader`); `litepath
  benchmark` is a new subcommand and `scripts/` gained two new tools.
