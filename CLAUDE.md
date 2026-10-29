# LitePath — Claude Context

This file is loaded automatically by Claude Code. Read it at the start of every
session before touching any code.

## Project Identity

- **Name:** `LitePath` (package and CLI: `litepath`)
- **Repo:** `ANNIE-ISU/LitePath` (transferred from `wetstein/ANNIEraytracing`
  on 2026-10-27; GitHub redirects the old URL)
- **Local folder:** still `ANNIEraytracing` on the original Mac (optional rename)
- **Rename notes:** clean break — no `annieray` shim. HDF5 `format` attr is
  now `litepath-batch`; older files say `annieray-batch` (reader does not check
  it). After pulling the rename, reinstall: `pip uninstall annieray && pip install -e .`

## What This Is

GPU-accelerated analytic Cherenkov ray tracer for the ANNIE Phase II water
Cherenkov detector at Fermilab. Uses Taichi for GPU kernels. Reads GDML detector
geometry, simulates muon Cherenkov + scintillation photons, writes HDF5 output,
fits muon direction via likelihood grid scan.

The codebase is mature. Do not rewrite things unprompted. Ask what the user wants
changed before starting.

## Pending Work (prioritised)

1. **Fix bugs** — tracked as GitHub issues (see `BUGS.md` for the mapping).
   Priorities revised by the maintainer on 2026-10-27 (the severity headings in
   `BUGS.md` are the original review's and are out of date):
   - **Fix now (`critical`):** BUG-3 (#4), BUG-5 (#6), BUG-7 (#8) — DONE
     (2026-10-27, commits bfb616b, 75ef905, 52f0c1d; issues closed)
   - **Minor, in v0.2:** BUG-6 (#7) — current work uses the housed ANNIE LAPPD
     (system 2); BUG-1 (#2) — `run --mode uniform` is unused and may be removed;
     BUG-8, BUG-9, BUG-10
   - **Backlog (no milestone):** BUG-4 (#5) — tracks are metre-scale for now;
     BUG-2 (#3) — reclassified as an enhancement (future wavelength feature)
   Run the test suite after each fix; put "Fixes #N" in the commit message.
2. **Rename to LitePath** — DONE (2026-10-26/27): package, CLI, imports, HDF5
   format name, docs, GitHub repo. Only the local folder name remains (optional).
3. **GitHub project infrastructure** — DONE for bugs (2026-10-27): labels
   (critical/medium/minor, physics, io, cli, fitting), milestone
   `v0.2-bugfixes`, issues #2–#11 = BUG-1–BUG-10 (offset by one: #1 is a
   Discussion). Reference issues in fix commits ("Fixes #N"). Still to do:
   issues for pending features; `v1.0-validated` milestone once defined.
4. **Org transfer** — DONE (2026-10-27): now `ANNIE-ISU/LitePath`.
5. **Restore Metal GPU backend** — issue #12 (opened 2026-10-27). All
   `ti.init` calls are hard-coded to `ti.cpu` because `trace_kernel` takes 32
   ndarray args (> Metal's 31-buffer limit). Soon a priority; not started.

## Key Facts for Code Changes

- Python 3.11+; Taichi ≥ 1.7; GPU kernels use `@ti.kernel` / `@ti.func`
- Hit array column indices are in `hits.py` (HI, HT, HX ... H_SOURCE). `tracer.py`
  is now only a compatibility module; code lives in `geometry/`, `engine/`,
  `sources/` and `simulate.py` (see `docs/README.md` code map)
- HDF5 schema version is `SCHEMA_VERSION = 1` in `io_h5.py`
- phi convention: truth `phi_deg` stored in `[0, 360)` to match the fit grid
  (BUG-5, #6); files written before the fix hold `[−180, 180]` and
  `load_observed_event()` wraps them on read
- `DET_SYS_PMT=0`, `DET_SYS_LAPPD_DEFAULT=1`, `DET_SYS_LAPPD_ANNIE=2`
- Speed of light: `C_MM_NS = 299.792458`, default water index `N_WATER_DEFAULT = 1.34`
- Tests: `pytest` skips GPU tests by default (`-m 'not slow'`); run
  `pytest -m slow` on a machine with a GPU to include them
- Reference outputs: `tests/test_reference_outputs.py` re-runs fixed-seed CLI
  cases (`tests/reference_cases.py`) and compares with `tests/reference/*.npz`.
  Any refactor must keep them passing (`pytest -m "slow or not slow"`).
  Regenerate with `python tools/make_reference_outputs.py` only for an
  intended output change, and say so in the commit

## Running the Project

```bash
pip install -e .
pytest                            # fast tests (no GPU)
pytest -m slow                    # full suite (needs GPU)
litepath batch --help             # main simulation CLI
litepath fit results/output.h5    # direction fitter
```

## Development History

- Originally developed with OpenCode
- First Claude Code session: 2026-10-26 (bug review, rename + GitHub setup pending)
- Previous session summary: see project memory in claude.ai "LitePath" project
