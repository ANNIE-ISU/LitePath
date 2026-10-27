# Known Bugs

Identified during code review (2026-10-26). Ordered by severity.

## 🔴 Critical — Fix before training / fitting

### BUG-1 · `cli.py` — Non-isotropic direction sampling in `--mode uniform`
**Location:** `_generate_uniform()`, direction generation block  
**Problem:** `phi = rng.uniform(-np.pi, np.pi)` then `z = sin(phi)` gives density
∝ 1/√(1−z²) — photon directions are biased toward the poles, not uniform on the
sphere. Breaks "flat efficiency scan" use case.  
**Fix:**
```python
xyz = rng.standard_normal((n, 3)).astype(np.float32)
directions = (xyz / np.linalg.norm(xyz, axis=1, keepdims=True)).astype(np.float32)
```

### BUG-2 · `cherenkov.py` — `wavelength` parameter dead; wrong default
**Location:** `generate_cherenkov_photons(..., wavelength=defaultWavelength, ...)`  
**Problem:** `defaultWavelength = 1234` (mid-infrared). Parameter is accepted by
the function signature but never referenced in the body. Callers passing
`--wavelength 350` have no effect here.  
**Fix:** Either wire `wavelength` into the emission (e.g. refractive index
dispersion, QE curve) or remove the parameter and fix the default to 350 nm.

### BUG-3 · `scintillation.py` — `log(0)` risk on large batches
**Location:** `generate_scintillation_photons()`, delay sampling  
**Problem:** `np.log(rng.random(n))` — `rng.random()` can return exactly `0.0`,
giving `−inf` delay. Rare per photon but will occur over large training runs.  
**Fix:** `np.log(np.clip(rng.random(n), 1e-38, None))`

### BUG-4 · `fitting.py` — Short tracks misidentified as old-format values
**Location:** `load_observed_event()`, track-length normalization  
**Problem:**
```python
track_cm = row["track_length_mm"] / 10.0
if track_cm < 10:   # fires for any track < 100 mm — valid physics!
    track_cm = row["track_length_mm"] * 1000.0 / 10.0   # ×100 error
```
Any genuine track under 10 cm gets its length multiplied by 100×, making
`true_photons_per_cm` wildly wrong and the fitter uncalibrated.  
**Fix:** Use `f.attrs["schema_version"]` from the HDF5 file to discriminate
old/new format instead of a value-range heuristic.

---

## 🟠 Medium — Data integrity

### BUG-5 · `batch.py` — phi convention mismatch (truth vs fit grid)
**Location:** `run_batch()`, muon_truth phi storage  
**Problem:** `phi_deg = np.degrees(np.arctan2(dy, dx))` stores in `[−180, 180]`.
The fitting grid default scans `[0, 360]`. Angular residuals will show spurious
360° wraps for muons with negative phi.  
**Fix:** `phi_deg = phi_deg % 360.0` before writing to HDF5.

### BUG-6 · `batch.py` — Default LAPPD hits silently excluded from `photon_hits`
**Location:** `_append_photon_hits()`  
**Problem:** `DET_SYS_LAPPD_DEFAULT = 1` hits are filtered out without warning.
Running `--lappd-model default` produces hits that never appear in HDF5 output.  
**Fix:** Include them, or add an explicit warning and document the exclusion.

### BUG-7 · `batch.py` — `return` in `finally` swallows write errors
**Location:** `run_batch()`, try/finally block  
**Problem:** `return paths` inside the `finally` block suppresses any exception
raised by `accumulator.close()`. HDF5 write failures are silently discarded and
corrupt paths returned as if successful.  
**Fix:** Move `accumulator.close()` into `finally` without a `return` there.

---

## 🟡 Minor

### BUG-8 · Off-by-one step count in photon generators
**Files:** `cherenkov.py`, `scintillation.py`  
`n_steps = int(track_length * 10**trackPrec) + 1` produces one extra emission
point. Docstrings say "photons_per_cm along the track" but actual count is
`(track_length_cm + ε) × photons_per_cm`. Consistent with "401 steps for 4 m"
in `cli.py` docs, but the off-by-one is undocumented.

### BUG-9 · `io_h5.py` — `write_full_hits` docstring says "(N, 17)"
Expanded hit array has 18 columns (`N_EXPANDED_COLS = 18`, indices 0–17).

### BUG-10 · `io_h5.py` — `BatchReader.load_event` is O(N) per call
Loads the entire `photon_hits` table then filters to one event. Fine for
inspection; will be slow iterating over large training files. Consider an
HDF5 event-index dataset or at least a warning.

---

## Status

Each bug is tracked as a GitHub issue in the [`v0.2-bugfixes` milestone](https://github.com/ANNIE-ISU/LitePath/milestone/1). Open/closed status lives there, not here.

| Bug | Issue |
|-----|-------|
| BUG-1 | [#2](https://github.com/ANNIE-ISU/LitePath/issues/2) |
| BUG-2 | [#3](https://github.com/ANNIE-ISU/LitePath/issues/3) |
| BUG-3 | [#4](https://github.com/ANNIE-ISU/LitePath/issues/4) |
| BUG-4 | [#5](https://github.com/ANNIE-ISU/LitePath/issues/5) |
| BUG-5 | [#6](https://github.com/ANNIE-ISU/LitePath/issues/6) |
| BUG-6 | [#7](https://github.com/ANNIE-ISU/LitePath/issues/7) |
| BUG-7 | [#8](https://github.com/ANNIE-ISU/LitePath/issues/8) |
| BUG-8 | [#9](https://github.com/ANNIE-ISU/LitePath/issues/9) |
| BUG-9 | [#10](https://github.com/ANNIE-ISU/LitePath/issues/10) |
| BUG-10 | [#11](https://github.com/ANNIE-ISU/LitePath/issues/11) |
