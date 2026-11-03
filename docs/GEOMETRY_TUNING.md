# Geometry tuning, snapshots and tagging

How to adjust PMT, LAPPD and surfboard positions in the geometry workbench,
save the work, and freeze a good state as a named, tagged geometry version
that batch runs and analyses can cite.

## The pieces

LitePath builds its detector from a **geometry version**: a small YAML
recipe in `geometry/versions/`. `--geometry NAME` reads
`geometry/versions/NAME.yaml`.

```
geometry/
  versions/    recipes: annie-2022-scan, annie-2026-data, annie-2026-data-tuning, annie-2026-data-v0.1, …
  overlays/    offset files a recipe applies: metrology-draft.yaml (the working file), annie-2026-data-v0.1.yaml, …
  base/        upstream tables, copied in unchanged: FullTankPMTGeometry.csv (ToolAnalysis)
```

A recipe names its **inputs** (PMT table, structure GDML), its **settings**
(bottom-PMT spin, LAPPD model, number of surfboards, …) and its
**overlays**: files of position offsets applied on top.

| Part | Where it lives | Changes? |
|---|---|---|
| Inner-structure mesh | `PHASE2_INNER_STRUCTURE_closed.gdml` (+ `_cache.npz`) | Rarely; a new mesh is a new file |
| PMT body and holder meshes | `pmt_meshes/` | Rarely; same rule |
| PMT positions (base grid) | `geometry/base/FullTankPMTGeometry.csv` | When ToolAnalysis publishes a new table; same rule |
| LAPPD housing model, surfboard base positions | Code (`litepath/lappd_model.py`, `build_surfboards`) | With the code; recorded through `litepath_commit` in outputs, not hashed |
| Your adjustments | An overlay (`element-offsets`) | Every time you save in the workbench |

### Offsets

An overlay holds one **total offset** per element, in mm in the LitePath
frame (`litepath/geometry/frame.py`), relative to the element's **base
position** (where the table and the code put it):

```yaml
kind: element-offsets
pmts:         # channel key: [dx, dy, dz]
  382: [0, 0, -25.4]
lappds:       # LAPPD detector key (2000, 2001, 2002 = housings at 45, 90, 135 deg): [dx, dy, dz]
  2001: [0, 0, 50]
surfboards:   # surfboard index (0, 1, 2 at 45, 90, 135 deg): [dx, dy, dz]
  1: [0, -5, 0]
history:      # one entry per save
- {date: '2026-11-03T10:12:00', lappds: [2001]}
```

Because offsets are totals from the base, work accumulates across sessions:
each session starts from what is saved. A housing's base position comes
from its surfboard's base position, so moving a surfboard does not move
its LAPPD.

### Draft and tagged versions

- A **draft** (`status: draft`) can change. `annie-2026-data-tuning` is the
  working draft; its overlay `geometry/overlays/metrology-draft.yaml` is the
  **working file** the workbench saves to.
- A **tagged** version (`status: tagged`) is frozen. Tagging records the
  SHA-256 **hash** (a fingerprint of a file's contents) of every file the
  build reads, plus a **fingerprint** of the whole detector, in a `tagged:`
  block at the end of the recipe. Every later build recomputes the hashes and
  refuses to run if any file changed, naming it.
- Every run, draft or tagged, prints the fingerprint and writes it into its
  output file (`geometry_hash`), so results stay traceable even on a draft.

Hashes detect changes; they do not keep copies. Git keeps the files.

## Tuning in the workbench

1. Start the workbench on the working draft and open
   <http://localhost:8082>:

   ```bash
   litepath geometry-viz --geometry annie-2026-data-tuning
   ```

   The header says which version is loaded and whether changes can be saved
   (only for a draft). `viz-server` (port 8080) no longer has these tools;
   both can run at once.

2. Optional references:
   - **Show Scan Overlay** (with the mesh selector) draws the 3D laser scan
     of the structure and PMTs. The meshes are
     `scan files by part/transformed/*_verts.npy` / `*_tris.npy` (gitignored,
     69 MB); copy them from the machine that has them if the overlay is
     empty.
     The selector's panel names are the scan's own numbering, which differs
     from ToolAnalysis: scan panels 1–8 are ToolAnalysis panels
     7, 8, 1, 2, 3, 6, 4, 5 (so "Panel 1" shows ToolAnalysis panel 7).
   - **Show Scan Tips** marks the PMT bulb tips found in the scan
     (`pmt_tip_positions.csv`, made by `tools/find_pmt_tips.py`).

3. Click a PMT, LAPPD housing or surfboard. Its popup shows sliders for the
   element's **total offset from its base** (PMTs: axial / tangential /
   vertical, or dX / dY / dZ for the caps; LAPPDs and surfboards: vertical /
   radial / tangential), and its saved offset. Moving a slider only
   previews.

4. Finish with one of:

   | Button | Effect |
   |---|---|
   | **Keep** | Add this position to the session's changes |
   | **Cancel** (or ✕, or clicking elsewhere) | Undo the preview since the popup opened |
   | **Revert to saved** | Back to the position saved in the working file |
   | **Base** | Sliders to zero: the base position, no offset |

5. The bar at the top counts the kept changes and lists them. **Save to
   draft** writes them all to the working file (with a history entry);
   **Discard all** puts them back. Closing the page with unsaved changes
   asks first.

6. To see the effect in a simulation, run batch on the draft:

   ```bash
   litepath batch --geometry annie-2026-data-tuning --muon-file muons.txt -o results/check
   ```

   The draft's saved PMT, LAPPD and surfboard offsets are part of the build,
   so batch simulates what the workbench shows. Runs on a draft print a
   warning; cite a snapshot in results.

## Freezing a good state: snapshots

A **snapshot** freezes the working draft's saved state as a new, named,
tagged version. It copies the offsets into a new overlay that nothing edits
afterwards, writes a recipe pointing at it, and tags that recipe. The draft
is untouched and you keep working in it.

1. Save or discard the workbench's changes (the Snapshot panel is disabled
   while there are unsaved changes).
2. Commit any changed input files (the snapshot refuses inputs with
   uncommitted git changes, so it always matches files in git; the working
   file itself is exempt, since its contents are copied).
3. Create the snapshot, from the workbench's **Snapshot (freeze & tag)**
   panel or the command line:

   ```bash
   litepath geometry snapshot annie-2026-data-tuning --as annie-2026-data-v1.0 \
       --description "PMT and LAPPD alignment against the 2022 scan, first pass"
   ```

   This writes and tags:

   ```
   geometry/versions/annie-2026-data-v1.0.yaml    recipe (status: tagged, with hashes)
   geometry/overlays/annie-2026-data-v1.0.yaml    frozen copy of the offsets and their history
   ```

4. Check it:

   ```bash
   litepath geometry show annie-2026-data-v1.0
   litepath geometry-viz --geometry annie-2026-data-v1.0     # view only
   ```

5. Commit and push both files:

   ```bash
   git add geometry/versions/annie-2026-data-v1.0.yaml geometry/overlays/annie-2026-data-v1.0.yaml
   git commit -m "Geometry snapshot annie-2026-data-v1.0"
   git push
   ```

6. Use it: `--geometry annie-2026-data-v1.0` in batch, fit and the viewers,
   and cite the name (and fingerprint) in results. Keep tuning in
   `annie-2026-data-tuning`; the next snapshot is `v1.1`.

### Naming convention

LitePath does not interpret version names; the convention is for people:

- `annie-2026-data-vX.Y` for snapshots: bump **Y** for tuning snapshots,
  **X** for a new base (a new ToolAnalysis table, a new structure mesh).
- The draft keeps its fixed name and is never cited in results.
- Names use letters, digits, `.`, `_` and `-`; a taken name is refused.

`annie-2026-data-v0.1` is the archive of the corrections developed before
the 2026-11 reorganisation: the 62 PMT offsets of `corrections.csv`
(2026-06-22), nothing else. PMTs 453 and 455 were tuned while their
positions were swapped.

## Tagging by hand

`litepath geometry tag NAME` tags a draft recipe in place: it computes the
hashes of every input file (PMT table, GDML and its cache, manifest or STEP
if used, every file in `pmt_meshes/`, every overlay), writes the `tagged:`
block and sets `status: tagged`. Use it for a recipe that is finished and
whose overlays will not be edited again (a new base version, say). Do not
tag the working draft: the next save would change its overlay and the
version would stop building. That is what snapshots are for.

## Rules that keep tagged versions building

- Never edit a file a tagged version uses. New contents go in a new file
  under a new name, and a new recipe points at it. Example: a new
  ToolAnalysis table goes in as `geometry/base/FullTankPMTGeometry_2027-03.csv`,
  with a new base version (and then new snapshots) using it.
- Never edit a tagged recipe (its settings and overlay list are covered too).
- The test suite checks every tagged version in `geometry/versions/`
  against its recorded hashes (`tests/test_geometry_versions.py`), so an
  in-place edit fails the tests before it is pushed.

## Commands

| Command | What it does |
|---|---|
| `litepath geometry list` | Every version: name, status, fingerprint, description |
| `litepath geometry show NAME` | Inputs, settings, overlays, file hashes |
| `litepath geometry snapshot DRAFT --as NAME [--description TEXT]` | Freeze a draft's state as a new tagged version |
| `litepath geometry tag NAME` | Tag a recipe in place |
| `litepath geometry-viz --geometry NAME` | Workbench (port 8082); saves only on a draft |

## Troubleshooting

| Message | Meaning |
|---|---|
| "view only (not a draft)" in the workbench | The version is tagged; start it on a draft to save |
| "Tagged geometry '…' has changed since it was tagged (changed files: …)" | A file a tagged version uses was edited; restore it from git, and put new contents in a new file |
| Snapshot "already exists" | Choose a new name |
| Snapshot "inputs have uncommitted changes" | Commit those files first |
| Scan overlay shows nothing | The scan meshes (`*_verts.npy`) are not in this checkout; copy them in |
