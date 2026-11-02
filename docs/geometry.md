# Geometry Pipeline

How the inner structure mesh and PMT positions are loaded, transformed, and assembled for raytracing and visualization.

## The LitePath frame

Everything is built in one fixed frame, defined in `litepath/geometry/frame.py`
(the module docstring is the reference):

- `z` up along the tank axis, `z = 0` is 1980 mm below the tank centre (water from z = 19 to 3861 mm)
- `+y` along the beam (downstream), `x = y × z`
- octagon corners at 0°, 45°, …; faces at 22.5° + k·45°. The beam points at the 90° corner (barrel panels 1 | 8, central surfboard and LAPPD) and enters through the 270° corner (panels 4 | 5); panels run clockwise seen from above; the bottom PMT grid runs at 45° to the beam

Checked against the ToolAnalysis table and the 2019 PMT layout slides (`tests/test_frame.py`).
Inputs come in through exactly two conversions, and nothing is rotated afterwards:

| Input | Conversion |
|---|---|
| PMT tables (ToolAnalysis CSV, scan file; cm) | `table_to_frame` |
| WCSim ROOT output (tank-centred, mm) | `tank_to_frame`: `(x, y, z) → (−x, z, y + 1980)` |
| GDML structure mesh, STEP manifest, scan meshes | `structure_to_frame`: 22.5° turn about z |

The 22.5° follows from WCSim: it places the structure with `rotateZ(157.5°)`,
and Geant4 turns the frame, so the volume turns by −157.5° in the tank; the
tank → frame step adds 180°. (Before 2026-11 the code undid the placement with
the opposite sign and made up the difference with a 45° turn of all PMTs plus a
22.5° "detector rotation". The result was the same; the `det_rotation_deg` and
`bottom_rotation_deg` settings and the `--det-rotation`/`--bottom-rot` flags are gone.)

---

## 1. Inner Structure (GDML Mesh)

**File:** `PHASE2_INNER_STRUCTURE_closed.gdml`

| # | What | Where |
|---|------|-------|
| 1 | `build_geometry()` calls `gdml_parser.parse_gdml(gdml_path)` | `geometry/build.py` |
| 2 | `parse_gdml` uses `lxml` to find all `<position>` elements → vertex XYZ (float32 N×3) | `gdml_parser.py:39-43` |
| 3 | Finds all `<triangular>` elements → triangle index triplets (int32 M×3) | `gdml_parser.py:48-51` |
| 4 | Returns `(vertices, triangles)` | `gdml_parser.py:53` |

The mesh is in the GDML structure frame; `build_geometry()` turns it into the LitePath frame with `structure_to_frame`.

---

## 2. PMT Scan File

**File:** `PMTPositions_Scan.txt` — 9-column TSV:

```
TubeID  panel_nr  scan_x(cm)  scan_y(cm)  scan_z(cm)  dirx  diry  dirz  pmt_type
```

### Stage A: Raw load

| # | What | Where |
|---|------|-------|
| 5 | `load_pmts()` reads the TSV with `np.loadtxt` | `pmt_loader.py:200-205` |

### Stage B: Table → LitePath frame

`load_pmts()` converts every row with `frame.table_to_frame`:

```
x  =  −scan_x · 10
y  =  (scan_z − 168.1) · 10      (beam axis, +y downstream)
z  =  (scan_y + 14.45) · 10 + 1980
```

The constants are `TABLE_BEAM_CENTRE_CM = 168.1` and `TABLE_VERTICAL_OFFSET_CM = 14.45` in `frame.py`. Both table formats use the same coordinates.

### Stage C: Z-offset (optional)

| # | What | Where |
|---|------|-------|
| 12 | Add `z_offset` to z | `pmt_loader.py` |

---

## 3. Direction Vectors

**The scan-file direction columns (`dirx, diry, dirz`) are NOT used.** Directions are synthesized from the PMT type and panel number, matching WCSim's behaviour.

| # | What | Where |
|---|------|-------|
| 13 | `_ideal_direction(ptype, x_s, y_s)` called per PMT | `pmt_loader.py:246` |

| PMT type | Panel | Direction | Lines |
|----------|-------|-----------|-------|
| LUX (bottom) | 0 | `(0, 0, +1)` — upward | `163-164` |
| ETEL (top) | 9 | `(0, 0, -1)` — downward | `165-166` |
| Hamamatsu / Watchboy / Watchman (barrel) | 1–8 | Inward normal of the nearest octagon face (faces at `22.5° + k·45°`), `frame.barrel_face_normal` | |

---

## 4. Instance Transforms (for GDML PMT mesh rendering)

After the main loop, each PMT gets a quaternion and an offset instance position.

| # | What | Where |
|---|------|-------|
| 14 | Rest-pose forward axis from `PMT_FORWARD` dict | `pmt_loader.py:114-120` |
| 15 | Target direction = the `_ideal_direction` result | `pmt_loader.py:264` |
| 16 | Watchboy/Watchman additionally rotated -45° about Z (clockwise one panel) | `pmt_loader.py:266-270` |
| 17 | `_forward_to_quat()` computes quaternion from forward → target | `pmt_loader.py:271, 290-312` |
| 18 | Instance position = centre − (forward_offset × target_dir) — shifts bulb tip to sphere centre | `pmt_loader.py:273-275` |

Forward offsets (`pmt_loader.py:123-129`):

| Type | Offset (mm) |
|------|-------------|
| LUX | 200.8 |
| ETEL | 208.3 |
| Hamamatsu | 145.5 |
| Watchboy / Watchman | −127.0 |

---

## 5. Assembly in `build_geometry()`

| # | What | Where |
|---|------|-------|
| 19 | Parse GDML → `(vertices, triangles)` | `geometry/build.py` |
| 20 | Load PMT CSV → `load_pmts()` → centres, radii, directions | `geometry/build.py` |
| 21 | Load STEP manifest → LAPPD positions and tank bounds | `geometry/build.py` |
| 22 | Build default LAPPD rectangles or ANNIE housing model | `geometry/build.py` |
| 23 | Return `Geometry` dataclass | `geometry/build.py` |

`Geometry` dataclass fields (`geometry/build.py`):

- `mesh_vertices` (float32 N×3) — GDML triangle vertex positions
- `mesh_triangles` (int32 M×3) — triangle index list
- `pmt_centers` (float32 P×3) — sphere centres in the LitePath frame
- `pmt_radii` (float32 P) — per-PMT radius
- `pmt_directions` (float32 P×3) — inward-pointing normals
- `lappd_data` (float32 L×7) — LAPPD rectangle `[cx,cy,cz, nx,ny,nz, half_size]`
- `tank_radius`, `tank_z_min`, `tank_z_max` — cylinder bounds
- `lappd_housing_data`, `annie_lappd_data` — ANNIE housing model (optional)
- `detectors` — list of `DetectorInfo` for hit-to-hardware mapping

---

## 6. Viz Server Frontend Layer

The interactive viewer applies `Rx(-90°)` to both mesh and PMTs when sending data to the browser, rotating the **Z-up** GDML frame into **Y-up** (Three.js convention).

| # | What | Where |
|---|------|-------|
| 24 | `_send_mesh_verts()` applies `Rx(-90°)` to each vertex | `viz_server.py:~40-50` |
| 25 | `_send_pmts()` applies `Rx(-90°)` to centres and directions | `viz_server.py:~80-100` |

The `Rx(-90°)` lives only in the viz-server send methods. The raytracer kernel processes everything in the Z-up LitePath frame.

---

## Reference: Complete Call Chain

```
geometry/build.py:build_geometry()
  ├── gdml_parser.py:parse_gdml()          →  (vertices, triangles)
  ├── frame.structure_to_frame()            →  mesh in the LitePath frame
  └── pmt_loader.py:load_pmts()
        ├── frame.table_to_frame()           →  LitePath frame (mm)
        ├── z_offset
        ├── _ideal_direction()               →  direction per PMT
        └── _forward_to_quat()               →  instance quaternion + position

viz_server.py:run_server()
  └── VizHandler._send_mesh_verts()          →  Rx(-90°) mesh
  └── VizHandler._send_pmts()                →  Rx(-90°) PMTs
```

---

## 7. Scan Overlay Registration

The 3D scan PLY files (SuperStructure, BottomLayer, TopLayer, AllPMTs, etc.) are in inches in a free-form scan coordinate system. They are transformed to the GDML structure frame (turned into the LitePath frame with `structure_to_frame` when loaded) via a fixed sequence of scale, rotation, and translation, all implemented in `tools/register_scan.py`.

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Scale | 25.4 mm/in | GDML height (3842 mm) ÷ scan height (151.1 in) |
| Octagon centre (scan XY) | (−43.67, −8.62) in | Fitted from scan octagon panel corners |
| Rotation | 22.09° CW | Panel-angle matching (−22.91°) + bottom structure correction (+45° CCW) |
| Z offset | 109.2 mm | Horizontal cross-brace ring alignment, corrected for square-tubing thickness (−2 in) |

### Transform Pipeline

For each point `(x, y, z)` in scan inches:

```
# 1. Scale inches → mm
x *= 25.4;  y *= 25.4;  z *= 25.4

# 2. Octagon centre in mm
cx = -43.67 * 25.4
cy = -8.62 * 25.4

# 3. Rotate XY about centre (22.09° CW)
θ = math.radians(22.09)
x -= cx;  y -= cy
xʹ =  x·cos(θ) + y·sin(θ)
yʹ = -x·sin(θ) + y·cos(θ)
x = xʹ + cx;  y = yʹ + cy

# 4. Z offset
z += 109.2
```

### Verification

- **Octagon faces**: Scan panels align with GDML octagon faces after XY rotation.
- **Height**: SuperStructure Z range matches GDML Z [19, 3861] mm within ~2%.
- **Horizontal cross-bracing**: Scan ring at ~70 in (after transform ≈ 1887 mm) is 38 mm below the strongest GDML ring (1925 mm), consistent with the scan capturing the tubing bottom and GDML representing the tubing top.
- **Bottom structure**: Corrected by +45° CCW rotation to align vertical columns with GDML column positions.

### Implementation

- `tools/register_scan.py` — computes parameters, transforms all PLY files, saves to `scan files by part/transformed/`.
- `tools/preprocess_scan.py` — converts transformed PLY to `.npy` vertex/triangle arrays for the viz server.
- `tools/find_pmt_tips.py` — finds PMT bulb tips in the AllPMTs scan mesh using the PMT positions (in the GDML structure frame) as seeds, outputs `pmt_tip_positions.npy`/`.csv`. The shipped `pmt_tip_positions.csv` predates the fixed frame: its seeds were placed without the 45° turn, so they sit 45° from the PMTs; rerun the tool to regenerate it.
- Viz server serves these arrays via `/api/scan_mesh/{name}/verts` and `/api/scan_mesh/{name}/tris`.
