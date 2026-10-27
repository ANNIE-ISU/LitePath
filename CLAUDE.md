# LitePath — Claude Context

This file is loaded automatically by Claude Code. Read it at the start of every
session before touching any code.

## Project Identity

- **Current name on disk:** `ANNIEraytracing` (package: `annieray`)
- **Target name:** `LitePath` (package: `litepath`)
- **Repo:** `wetstein/ANNIEraytracing` → target org: `ANNIE-ISU/LitePath`
- **Rename status:** NOT STARTED — do not rename files unless explicitly asked

## What This Is

GPU-accelerated analytic Cherenkov ray tracer for the ANNIE Phase II water
Cherenkov detector at Fermilab. Uses Taichi for GPU kernels. Reads GDML detector
geometry, simulates muon Cherenkov + scintillation photons, writes HDF5 output,
fits muon direction via likelihood grid scan.

The codebase is mature. Do not rewrite things unprompted. Ask what the user wants
changed before starting.

## Pending Work (prioritised)

1. **Fix bugs** — see `BUGS.md` for the full list. Fix critical bugs (BUG-1 to
   BUG-4) first, run the test suite after each fix.
2. **Rename to LitePath** — package rename (`annieray` → `litepath`), directory
   rename, `pyproject.toml`, all imports, CLI entry point, HDF5 format name,
   docs. Update `AGENTS.md` and this file after renaming.
3. **GitHub project infrastructure** — create Issues for each bug and pending
   feature, add labels (bug/critical/medium/minor, physics, io, cli, fitting),
   create milestones (v0.2-bugfixes, v0.3-rename, v1.0-validated).
4. **Org transfer** — transfer `wetstein/ANNIEraytracing` → `ANNIE-ISU/LitePath`
   after the rename is complete and pushed.

## Key Facts for Code Changes

- Python 3.11+; Taichi ≥ 1.7; GPU kernels use `@ti.kernel` / `@ti.func`
- Hit array column indices are in `tracer.py` (HI, HT, HX ... H_SOURCE)
- HDF5 schema version is `SCHEMA_VERSION = 1` in `io_h5.py`
- phi convention: truth stored as `arctan2` result → `[−180, 180]`; fitting
  grid uses `[0, 360]` — BUG-5 not yet fixed
- `DET_SYS_PMT=0`, `DET_SYS_LAPPD_DEFAULT=1`, `DET_SYS_LAPPD_ANNIE=2`
- Speed of light: `C_MM_NS = 299.792458`, default water index `N_WATER_DEFAULT = 1.34`
- Tests: `pytest` skips GPU tests by default (`-m 'not slow'`); run
  `pytest -m slow` on a machine with a GPU to include them

## Running the Project

```bash
pip install -e .
pytest                            # fast tests (no GPU)
pytest -m slow                    # full suite (needs GPU)
annieray batch --help             # main simulation CLI
annieray fit results/output.h5    # direction fitter
```

## Development History

- Originally developed with OpenCode
- First Claude Code session: 2026-10-26 (bug review, rename + GitHub setup pending)
- Previous session summary: see project memory in claude.ai "LitePath" project
