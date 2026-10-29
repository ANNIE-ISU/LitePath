# Batch Output Format

Both `litepath batch` (full detector) and `litepath batch-lappd`
(standalone LAPPD module) write a single HDF5 file — `output.h5` in the
output directory — using a **shared schema** so analysis code can load any
batch output the same way.

Schema version is stamped in the file; see `litepath/io_h5.py` for the
authoritative dtype definitions (`SCHEMA_VERSION`, `*_DTYPE`).

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
| `schema_version` | `1` |
| `detector_mode` | `full` \| `lappd-standalone` |
| `created` | ISO timestamp of the run |
| `config` | JSON string with the full resolved run configuration |

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
| `track_length_mm` | float32 | Chord through the detector (mm); 0 for point sources |
| `n_generated` | int32 | Total photons generated |
| `n_detected` | int32 | Photons detected by PMTs / LAPPDs |
| `n_generated_ckv` | int32 | Cherenkov component of `n_generated` |
| `n_generated_sci` | int32 | Scintillation component of `n_generated` |
| `n_detected_ckv` | int32 | Cherenkov component of `n_detected` |
| `n_detected_sci` | int32 | Scintillation component of `n_detected` |
| `spread_deg` | float32 | Spot-beam angular spread (°); NaN for track/burst |
| `intercept_x/y/z` | float32 | PC target the beam was aimed at (mm); NaN when unused |

Source-type semantics:

| `source_type` | Config (`batch-lappd --source`) | Used columns |
|---------------|---------------------------------|--------------|
| 0 (track) | `cherenkov` | `pos_*`, `dir_*`, `track_length_mm`, `t0` |
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

For `batch-lappd` this table contains the single standalone
`lappd_annie` entry.

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
table, so probing is optional.  `EventData` and `BatchReader` live in
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
| `batch` | yes | yes | `--lappd-response` | `--pmt-response` | yes |
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