# Geometry Pipeline

How the inner structure mesh and PMT positions are loaded, transformed, and assembled for raytracing and visualization.

---

## 1. Inner Structure (GDML Mesh)

**File:** `PHASE2_INNER_STRUCTURE_closed.gdml`

| # | What | Where |
|---|------|-------|
| 1 | `build_geometry()` calls `gdml_parser.parse_gdml(gdml_path)` | `geometry/build.py` |
| 2 | `parse_gdml` uses `lxml` to find all `<position>` elements → vertex XYZ (float32 N×3) | `gdml_parser.py:39-43` |
| 3 | Finds all `<triangular>` elements → triangle index triplets (int32 M×3) | `gdml_parser.py:48-51` |
| 4 | Returns `(vertices, triangles)` | `gdml_parser.py:53` |

The mesh is **already in structure rest frame** — no transform is applied.

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

### Stage B: Scan → WCBarrel (water tank frame)

| # | What | Where |
|---|------|-------|
| 6 | `_scan_to_tank(sx, sy, sz)` converts cm → mm and re-orders axes | `pmt_loader.py:132-138` |
| 7 | Called per PMT | `pmt_loader.py:226` |

Transform:

```
x_w  =  scan_x * 10             (horizontal)
y_w  =  (168.1 - scan_z) * 10   (beam axis, flipped)
z_w  =  (scan_y + 14.45) * 10   (vertical)
```

Constants at `pmt_loader.py:41-43`:

- `BEAM_CENTER_Z = 168.1` cm
- `VERTICAL_OFFSET = 14.45` cm

### Stage C: WCBarrel → Structure rest frame

| # | What | Where |
|---|------|-------|
| 8 | `_tank_to_structure(x_w, y_w, z_w)` undoes WCSim's Geant4 placement | `pmt_loader.py:141-152` |
| 9 | Called per PMT | `pmt_loader.py:229` |

WCSim placement in `WCBarrel`: `Rz(+157.5°) + (0, 0, -1980 mm)`. We undo it:

```
x_s = x_w · cos(-157.5°)  -  y_w · sin(-157.5°)
y_s = x_w · sin(-157.5°)  +  y_w · cos(-157.5°)
z_s = z_w + 1980
```

Precomputed rotation at `pmt_loader.py:54-58`:

```python
STRUCTURE_ROTATION_DEG = 157.5
_rot_rad = math.radians(-STRUCTURE_ROTATION_DEG)
_cos_r = math.cos(_rot_rad)
_sin_r = math.sin(_rot_rad)
```

Translation: `HALF_WC_LENGTH = 1980` mm (`pmt_loader.py:52`).

**After this step, PMT positions are in the same coordinate frame as the GDML mesh.**

### Stage D: Bottom-PMT Z-rotation (optional)

| # | What | Where |
|---|------|-------|
| 10 | Apply extra `Rz(θ)` only to panel-0 (LUX) PMTs | `pmt_loader.py:234-238` |
| 11 | Angle negated: `_brot = -radians(bottom_rotation_deg)` | `pmt_loader.py:214-218` |

Default is `0.0`; pass `--bottom-rot 45` for correct bottom alignment. The scan grid axes naturally sit at the octagon vertices after Stage C; `--bottom-rot 45` rotates them onto the face centres, aligning the grid with the structure.

### Stage E: Z-offset (optional)

| # | What | Where |
|---|------|-------|
| 12 | Add `z_offset` to Z_s | `pmt_loader.py:232` |

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
| Hamamatsu / Watchboy / Watchman (barrel) | 1–8 | Snaps XY angle to nearest octagon face (`k × 45°`), returns `(-cos(k·45°), -sin(k·45°), 0)` — radially inward | `168-173` |

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
- `pmt_centers` (float32 P×3) — sphere centres in structure frame
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

**The `_tank_to_structure` function (`pmt_loader.py:141-152`) does NOT contain `Rx(-90°)`** — that rotation was removed earlier and lives only in the viz-server send methods. The raytracer kernel processes everything in the Z-up GDML frame.

---

## Reference: Complete Call Chain

```
geometry/build.py:build_geometry()
  ├── gdml_parser.py:parse_gdml()          →  (vertices, triangles)
  └── pmt_loader.py:load_pmts()
        ├── _scan_to_tank()                  →  WCBarrel frame (mm)
        ├── _tank_to_structure()             →  structure rest frame
        ├── bottom Z-rotation (panel 0 only)
        ├── z_offset
        ├── _ideal_direction()               →  direction per PMT
        └── _forward_to_quat()               →  instance quaternion + position

viz_server.py:run_server()
  └── VizHandler._send_mesh_verts()          →  Rx(-90°) mesh
  └── VizHandler._send_pmts()                →  Rx(-90°) PMTs
```

---

## 7. Scan Overlay Registration

The 3D scan PLY files (SuperStructure, BottomLayer, TopLayer, AllPMTs, etc.) are in inches in a free-form scan coordinate system. They are transformed to the structure rest frame (GDML frame) via a fixed sequence of scale, rotation, and translation, all implemented in `tools/register_scan.py`.

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
- `tools/find_pmt_tips.py` — finds PMT bulb tips in the AllPMTs scan mesh using CSV positions as seeds, outputs `pmt_tip_positions.npy`/`.csv`.
- Viz server serves these arrays via `/api/scan_mesh/{name}/verts` and `/api/scan_mesh/{name}/tris`.
