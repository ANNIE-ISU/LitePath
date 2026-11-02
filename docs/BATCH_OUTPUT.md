# Batch Output Format

Both `litepath batch` (full detector) and `litepath batch-lappd`
(standalone LAPPD module) write a single HDF5 file — `output.h5` in the
output directory — using a **shared schema** so analysis code can load any
batch output the same way.

Schema version is stamped in the file; see `litepath/io_h5.py` for the
authoritative dtype definitions (`SCHEMA_VERSION`, `*_DTYPE`).  The current
version is **2**; see [Changes in v2](#changes-in-v2).  Version 1 files are
still readable.

## Files

| File | Always written? | Description |
|------|:-:|-------------|
| `output.h5` | yes | All batch data: tables + root metadata |

There are no sidecar files any more.  (Older versions of the code wrote
Parquet files — that layout is obsolete.)

## Root attributes

Set on the HDF5 root group (`/`) by the batch pipeline:

| Attribute | Value |
|-----------|-------|
| `format` | `litepath-batch` (files written before the LitePath rename say `annieray-batch`) |
| `schema_version` | `2` (`1` for files written before 2026-11) |
| `detector_mode` | `full` \| `lappd-standalone` |
| `created` | ISO timestamp of the run |
| `config` | JSON string with the full resolved run configuration |
| `geometry_version` | Geometry version simulated, e.g. `annie-2026-data`; `lappd-standalone` for `batch-lappd` |
| `geometry_status` | `tagged`, `draft` or `unnamed` (`built-in` for `lappd-standalone`) |
| `geometry_hash` | The version's SHA-256 fingerprint (the 12-character form is printed by every run) |
| `geometry_pmt_table` | PMT table used (repo-relative path); empty for `batch-lappd` |
| `geometry_pmt_table_sha256` | SHA-256 of that file; `geometry/base/SOURCES.md` says where each table comes from |
| `litepath_commit` | LitePath git commit, with `-dirty` if tracked files had changes; `unknown` outside git |
| `tank_radius_mm`, `tank_z_min_mm`, `tank_z_max_mm` | Tank cylinder (`batch` only) |

`config` includes `masked_pmt_indices`: the PMTs (detector indices) whose
hits were dropped with `--mask-dead`, empty otherwise.

For `batch-lappd` the config also carries the light source and sampling
parameters (`sample_mode`, `source`, `grid_steps`, `intercept`,
`spread_deg`, `photons_per_cm`, `n_photons_spot`, …).

## Tables (HDF5 datasets)

All tables are flat, resizable datasets of numpy structured arrays
(compressed with gzip).  Load them with `load_table()` or
`BatchReader` (see [Reading](#reading)).

| Table | Rows | Written by |
|-------|------|-----------|
| `photon_hits` | One per detected photon | `batch`, `batch-lappd` |
| `muon_truth` | One per event (light-source truth) | `batch`, `batch-lappd` |
| `lappd_readouts` | One per (event, LAPPD) with `--lappd-response` | `batch`, `batch-lappd` |
| `pmt_responses` | One per hit PMT with `--pmt-response` | `batch` only |
| `detectors` | Detector registry | both (via `write_detectors`) |
| `geometry_elements` | One per PMT / LAPPD: the simulated detector (v2) | both (via `write_detectors`) |
| `hits` | Full single-shot trace | legacy: written only by the removed `run` command; still readable |

### `photon_hits`

One row per photon that reaches a sensitive detector (PMT or LAPPD).
Photons that miss all detectors or hit only structural elements are
**not** recorded.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number (0-based) |
| `detector_system` | int32 | 0 = PMT, 1 = LAPPD (default), 2 = LAPPD (ANNIE housing) |
| `detector_index` | int32 | **Global** detector index within the system |
| `local_u` | float32 | Along-strip position (LAPPD, mm) / unused for PMT |
| `local_v` | float32 | Across-strip position (LAPPD, mm) / unused for PMT |
| `arrival_time` | float32 | Photon arrival time at the photocathode (ns) |
| `wavelength` | float32 | Photon wavelength (nm) |
| `photon_source` | int32 | 0 = Cherenkov, 1 = scintillation |

With `batch --mask-dead`, photons on PMTs marked OFF in the geometry's PMT
table are dropped (and not counted in `n_detected`), as if those PMTs were
switched off.

### `muon_truth`

One row per event.  Despite the name, this is the truth table for the
**light source** of the event — a tracked muon, an isotropic light burst,
or a directed spot beam.  `source_type` discriminates between them;
columns that do not apply to a given source type are NaN.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `source_type` | int32 | 0 = track, 1 = light burst, 2 = spot beam |
| `pos_x/y/z` | float32 | Source point (mm): start of track / burst centre / beam origin |
| `t0` | float32 | Emission start time (ns) |
| `dir_x/y/z` | float32 | Unit direction: muon direction / unused / beam aim direction |
| `theta_deg` | float32 | Polar angle from vertical (°); 0 = downward (−z) |
| `phi_deg` | float32 | Azimuthal angle in XY (°), in [0, 360); 0 = +x |
| `track_length_mm` | float32 | Same value as `light_track_length_mm` (kept for older scripts; will be removed in a later release, with notice). For `benchmark` files: WCSim's own track length in water |
| `light_track_length_mm` | float32 | (v2) Length of track over which light was generated (mm). Full detector: the part inside the tank, from the start point (or the tank entry, if the muon starts outside) to the tank exit. This is a simulation cutoff, not a physical range: the muon is not assumed to stop there. `batch-lappd`: from the start point until the muon passes behind the plane of the housing's back face, or `--max-track-length` (default 4 m) if it never does; 0 if it starts behind that plane. 0 for point sources and for muons that never enter the tank |
| `n_generated` | int32 | Total photons generated |
| `n_detected` | int32 | Photons detected by PMTs / LAPPDs |
| `n_generated_ckv` | int32 | Cherenkov component of `n_generated` |
| `n_generated_sci` | int32 | Scintillation component of `n_generated` |
| `n_detected_ckv` | int32 | Cherenkov component of `n_detected` |
| `n_detected_sci` | int32 | Scintillation component of `n_detected` |
| `spread_deg` | float32 | Spot-beam angular spread (°); NaN for track/burst |
| `intercept_x/y/z` | float32 | PC target the beam was aimed at (mm); NaN when unused |
| `tank_entry_x/y/z` | float32 | (v2) Where the light-emitting stretch starts (mm): the tank entry, or the start point for a muon starting inside the tank. NaN when no light is generated in a tank (muon misses the tank, bursts, spot beams, `batch-lappd`) |
| `tank_exit_x/y/z` | float32 | (v2) Where it ends: the tank exit (mm). NaN as above. `light_track_length_mm` is the distance from entry to exit |

Source-type semantics:

| `source_type` | Config (`batch-lappd --source`) | Used columns |
|---------------|---------------------------------|--------------|
| 0 (track) | `cherenkov` | `pos_*`, `dir_*`, `light_track_length_mm`, `tank_entry_*`/`tank_exit_*` (full detector), `t0` |
| 1 (burst) | `batch --light-burst` | `pos_*`, `t0` (isotropic) |
| 2 (spot) | `spot` | `pos_*`, `dir_*`, `spread_deg`, `intercept_*`, `t0` |

Direction conventions match the main detector: `theta_deg` is the polar
angle from vertical with 0° = downward (−z); `phi_deg` is the azimuth in
the XY plane with 0° = +x, in [0°, 360°).  Files written before
[#6](https://github.com/ANNIE-ISU/LitePath/issues/6) store `phi_deg` in
[−180°, 180°]; `litepath fit` wraps it on read.

### `lappd_readouts`

Full digital readout traces, one row per (event, LAPPD).  Only written when
the run enables the LAPPD response pipeline (`--lappd-response`).

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `lappd_index` | int32 | Detector index within the LAPPD system |
| `side0` | float32 (28, 256) | Strip × time-bin waveform matrix (V) |
| `side1` | float32 (28, 256) | Second-side waveform matrix (V) |
| `n_photons` | int32 | Photon hits fed into the readout |
| `n_passed_qe` | int32 | Photons that passed the quantum-efficiency step |

Time bin width is 0.1 ns (256 bins → 25.6 ns window; see
`litepath/lappd_response.py`).

**Note on backward compatibility:** schema v1 introduced this flat table.
Files written by older code stored the same data as `event_N/lappd_readouts`
groups.  `load_event_lappd_readout()` (and `BatchReader.load_event`) read the
flat table first and fall back to the old group layout, so old files keep
working.

### `pmt_responses`

Only written when `batch --pmt-response` is passed.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int64 | Event number |
| `pmt_index` | int32 | PMT index (0 … n_pmts−1) |
| `charge` | float32 | Integrated charge (pC) |
| `time` | float32 | Hit time (ns) |
| `n_hits` | int32 | Photon hits folded into this response |

### `detectors`

Detector registry (one row per PMT / LAPPD).

| Column | Type | Description |
|--------|------|-------------|
| `system_code` | int32 | 0 = PMT, 1 = lappd_default, 2 = lappd_annie |
| `detector_index` | int32 | Global index (matches `photon_hits`) |
| `x, y, z` | float64 | Detector position (mm) |
| `label` | str | e.g. `PMT_332`, `LAPPD_ANNIE_0` |
| `panel` | int32 | PMT panel number (0=bottom, 9=top, 1–8=barrel); −1 for LAPPDs |
| `channel_key` | int32 | (v2) ToolAnalysis channel key (= WCSim TubeID, 332–463); −1 for LAPPDs |
| `detector_key` | int32 | (v2) ToolAnalysis detector key (equal to `channel_key` for tank PMTs); LAPPDs: LitePath ID (1000+ default, 2000+ housed) |
| `cable_label` | str | (v2) PMT cable label, e.g. `E2`; empty when the PMT table has none (the scan file) |
| `pmt_type` | str | (v2) `LUX`, `ETEL`, `Hamamatsu`, `Watchboy`, `Watchman`; `LAPPD` for LAPPDs |
| `status` | str | (v2) `ON` / `OFF` from the PMT table (ToolAnalysis); always `ON` for the scan file and LAPPDs |

`detector_index` is LitePath's internal index; join on it to turn
`photon_hits` / `pmt_responses` rows into real channels:

```python
det = br.load_table("detectors")
hits = br.load_table("photon_hits").merge(
    det[["system_code", "detector_index", "channel_key"]],
    left_on=["detector_system", "detector_index"],
    right_on=["system_code", "detector_index"])
```

For `batch-lappd` this table contains the single standalone
`lappd_annie` entry.

### `geometry_elements`

(v2) The simulated detector, one row per PMT / LAPPD, so a file can be
analysed without LitePath's geometry files.  Same rows and order as
`detectors`.

| Column | Type | Description |
|--------|------|-------------|
| `element_id` | int32 | `detector_key` (PMT channel key; LAPPD 1000+/2000+) |
| `system_code` | int32 | As in `detectors` |
| `detector_index` | int32 | As in `detectors` |
| `x, y, z` | float64 | Position (mm): PMT sphere centre, LAPPD photocathode centre |
| `dir_x/y/z` | float64 | Inward-facing unit normal |
| `size_mm` | float64 | PMT sphere radius or LAPPD half-side (mm) |
| `type` | str | As `detectors.pmt_type` |
| `status` | str | As `detectors.status` |

Positions are in the LitePath frame (`litepath/geometry/frame.py`: z up,
+y along the beam).

## Reading

```python
from litepath.io_h5 import BatchReader, load_table

# Whole-file viewer
br = BatchReader("results/output.h5")
print(br.summary())            # schema version, mode, modes, table row counts
print(br.event_ids())

muons = br.load_table("muon_truth")        # pandas DataFrame
hits  = br.load_table("photon_hits")
readouts = br.load_table("lappd_readouts") # side0/side1 are 2-D ndarray objects

# Per-event convenience access
ev = br.load_event(0)
ev.truth            # dict of truth columns
ev.hits             # DataFrame of this event's photon hits
ev.lappd_readouts   # {lappd_index: (28, 256, 2) array}
ev.is_beam()        # source_type == SOURCE_SPOT
ev.n_generated() / ev.n_detected()

# Iterate all events (streams one event at a time)
for ev in br.iter_events():
    ...
```

`load_table(path, "muon_truth")` returns an empty DataFrame for a missing
table, so probing is optional.  String columns come back as Python `str`.  `EventData` and `BatchReader` live in
`litepath/io_h5.py`.

## Typical usage

```python
from litepath.io_h5 import BatchReader

br = BatchReader("results/output.h5")
muons = br.load_table("muon_truth")

# Per-event detection efficiency
eff = muons["n_detected"] / muons["n_generated"].clip(1)

# Filter spot-beam runs / events aimed at a specific PC point
beams = muons[muons["source_type"] == 2]
targeted = beams[beams["intercept_x"].notna()]
```

## Which tables exist for which invocation

| Command / flag | `photon_hits` | `muon_truth` | `lappd_readouts` | `pmt_responses` | `detectors` |
|---|---|---|---|---|---|
| `batch` | yes | yes | `--lappd-response` | `--pmt-response` | yes (+ `geometry_elements`) |
| `batch` (no LAPPDs in geometry) | yes | yes | — | `--pmt-response` | yes |
| `batch-lappd` | yes | yes | `--lappd-response` | — | yes |
| `batch-lappd --source spot` | yes | yes (source_type=2) | `--lappd-response` | — | yes |
| `batch-lappd --source cherenkov` | yes | yes (source_type=0) | `--lappd-response` | — | yes |
| `benchmark` (WAGLE ROOT) | — | yes | — | yes | yes |

The `benchmark` command writes the same standard tables plus two extension
tables (`pmt_hits` with the raw digitised hits, `benchmark_truth` with the
track in both frames) and no `photon_hits` — the ROOT trees carry no
photon-level truth.  See `docs/BENCHMARK.md`.

## System codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `pmt` | 10-inch PMT (LUX / ETEL) |
| 1 | `lappd_default` | Rectangular LAPPD (original model) |
| 2 | `lappd_annie` | LAPPD in waterproof Kandemir housing |

## Changes in v2

Version 2 (2026-11) adds information and removes none: every v1 table and
column is still there with the same meaning.  Scripts that read columns by
name keep working.

| Change | Notes |
|---|---|
| `schema_version` = 2 | Scripts that require exactly 1 must accept 2 |
| New root attributes `geometry_version`, `geometry_status`, `geometry_hash`, `geometry_pmt_table`, `geometry_pmt_table_sha256`, `litepath_commit` | Which detector was simulated, and with which code |
| New table `geometry_elements` | Scripts that loop over all tables see one more |
| `detectors`: new `channel_key`, `detector_key`, `cable_label`, `pmt_type`, `status` | Appended after the v1 columns |
| `muon_truth`: new `light_track_length_mm`, `tank_entry_x/y/z`, `tank_exit_x/y/z` | Appended after the v1 columns. `track_length_mm` is still written with the same value; prefer `light_track_length_mm` |
| `batch --mask-dead` | Optional; off unless asked for |
| `load_table` / `BatchReader` return string columns as `str` | v1 code got `bytes` for `label` |
| `benchmark` files: `frame` attribute replaces `det_rotation_deg` | The fixed LitePath frame; see `docs/BENCHMARK.md` |

**Reading v1 files.** `load_table` and `BatchReader` fill in the v2 columns
when a v1 file lacks them: `light_track_length_mm` = `track_length_mm`;
`tank_entry_*`/`tank_exit_*` = NaN; `channel_key`/`detector_key` from the PMT
label (`PMT_<key>`), −1 for LAPPDs; `cable_label`, `pmt_type`, `status` empty.
`geometry_elements` reads as an empty table.  `tests/reference/v1_output.h5`
keeps this tested.  For a script that must also run on raw v1 files (via
h5py), choose the column at run time:

```python
col = "light_track_length_mm" if "light_track_length_mm" in truth else "track_length_mm"
```
