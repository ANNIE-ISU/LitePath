"""Geometry workbench: view, compare and adjust the detector geometry.

The geometry tools of the viz server (scan overlays and tip markers, PMT
position adjustment, LAPPD and surfboard positions) in a server of their own,
without the muon and photon-trace parts.  Step 1 of splitting the viz server:
the tools are copied as they are; once this tool is approved they are removed
from ``litepath viz-server``.

PMT adjustments are saved to the geometry version's pmt-offsets overlay, and
only for a draft version (e.g. ``annie-2026-data-tuning``).

    litepath geometry-viz --geometry annie-2026-data-tuning     # http://localhost:8082
"""
from __future__ import annotations

import datetime

import numpy as np

from litepath import viz_server
from litepath.detectors import LAPPD_ID_OFFSET_ANNIE
from litepath.geometry.versions import (
    REPO_ROOT, OFFSET_SECTIONS, GeometryError, load_version, read_offsets, snapshot_version,
    write_offsets,
)
from litepath.viz_server import VizHandler

# Routes of the viz server that belong to the muon / photon-trace side.
_OPERATIONAL_ROUTES = ("/api/trace", "/api/lappd/readout")
# The viz server's save-immediately routes; the workbench saves through
# /api/offsets/save instead.
_DIRECT_SAVE_ROUTES = ("/api/correction/save", "/api/surfboard/adjust", "/api/lappd/adjust",
                       "/api/lappd_correction")


class GeometryVizHandler(VizHandler):
    """The viz server's handler with the geometry page and without photon tracing.

    Adjustments are previewed in the page and saved together, as totals
    relative to each element's base position, to the draft's overlay.
    """

    _saved: dict | None = None   # saved totals by section (from the build, then the saves)

    @classmethod
    def _saved_offsets(cls) -> dict:
        if cls._saved is None:
            build = VizHandler._build
            cls._saved = {
                "pmts": dict(build.pmt_offsets),
                "lappds": dict(build.lappd_offsets),
                "surfboards": dict(build.surfboard_offsets),
            }
        return cls._saved

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in _OPERATIONAL_ROUTES:
            self._send_json({"error": "not available in the geometry workbench"}, 404)
        elif path == "/api/geometry":
            self._send_json({
                "name": VizHandler._geometry_name,
                "status": VizHandler._geometry_status,
                "editable": VizHandler._overlay_path is not None,
            })
        elif path == "/api/offsets":
            overlay = VizHandler._overlay_path
            self._send_json({
                "editable": overlay is not None,
                "overlay": str(overlay) if overlay else None,
                "saved": {s: {str(k): list(v) for k, v in d.items()}
                          for s, d in self._saved_offsets().items()},
            })
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path in _OPERATIONAL_ROUTES or path in _DIRECT_SAVE_ROUTES:
            self._send_json({"error": "not available in the geometry workbench"}, 404)
        elif path in ("/api/offsets/save", "/api/snapshot"):
            import json
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            (self._save_offsets if path == "/api/offsets/save" else self._snapshot)(data)
        else:
            super().do_POST()

    def _save_offsets(self, data: dict) -> None:
        """Save new totals ``{section: {key: [dx, dy, dz]}}`` to the draft overlay."""
        overlay = VizHandler._overlay_path
        if overlay is None:
            self._send_json({"error": f"geometry '{VizHandler._geometry_name}' is not a draft; "
                                      "start the workbench with a draft version to save"}, 400)
            return
        geometry = viz_server.geometry
        pmt_keys = list(VizHandler._pmt_data["detector_nums"])
        n_lappd = geometry.lappd_housing_data.shape[0]
        n_board = geometry.surfboard_data.shape[0]
        valid = {
            "pmts": lambda k: k in pmt_keys,
            "lappds": lambda k: 0 <= k - LAPPD_ID_OFFSET_ANNIE < n_lappd,
            "surfboards": lambda k: 0 <= k < n_board,
        }
        try:
            changes = {s: {int(k): tuple(float(x) for x in v) for k, v in (data.get(s) or {}).items()}
                       for s in OFFSET_SECTIONS}
        except (TypeError, ValueError):
            self._send_json({"error": "invalid offsets"}, 400)
            return
        for section, entries in changes.items():
            bad = [k for k in entries if not valid[section](k) or len(entries[k]) != 3]
            if bad:
                self._send_json({"error": f"unknown {section}: {bad}"}, 400)
                return
        if not any(changes.values()):
            self._send_json({"error": "nothing to save"}, 400)
            return

        # The overlay gets the change in the total (other overlays may contribute too)
        saved = self._saved_offsets()
        in_file = read_offsets(overlay)
        for section, entries in changes.items():
            for key, total in entries.items():
                old = saved[section].get(key, (0.0, 0.0, 0.0))
                cur = in_file[section].get(key, (0.0, 0.0, 0.0))
                in_file[section][key] = tuple(c + n - o for c, n, o in zip(cur, total, old))
        entry = {"date": datetime.datetime.now().isoformat(timespec="seconds")}
        entry.update({s: sorted(e) for s, e in changes.items() if e})
        write_offsets(overlay, in_file, history_entry=entry)

        # Same change in the served geometry, so a page reload shows it
        for section, entries in changes.items():
            for key, total in entries.items():
                old = np.asarray(saved[section].get(key, (0.0, 0.0, 0.0)))
                delta = (np.asarray(total) - old).astype(np.float32)
                if section == "pmts":
                    i = pmt_keys.index(key)
                    VizHandler._pmt_data["instance_positions"][i] += delta
                    geometry.pmt_centers[i] += delta
                    if i < geometry.pmt_instance_pos.shape[0]:  # empty with --no-pmt-holders
                        geometry.pmt_instance_pos[i] += delta
                    VizHandler._corrections[key] = total
                elif section == "lappds":
                    i = key - LAPPD_ID_OFFSET_ANNIE
                    geometry.lappd_housing_data[i, 0:3] += delta
                    geometry.annie_lappd_data[i, 0:3] += delta
                else:
                    geometry.surfboard_data[key, 0:3] += delta
                saved[section][key] = total
        n = sum(len(e) for e in changes.values())
        print(f"  Saved {n} adjustment(s) to {overlay}")
        self._send_json({"ok": True, "saved": n, "overlay": str(overlay)})

    def _snapshot(self, data: dict) -> None:
        """Freeze the draft's saved state as a new tagged version (geometry snapshot)."""
        if VizHandler._overlay_path is None:
            self._send_json({"error": "only a draft version can be snapshot"}, 400)
            return
        name = str(data.get("name", "")).strip()
        try:
            recipe, fp = snapshot_version(VizHandler._build.version.path, name,
                                          str(data.get("description", "")).strip() or None)
        except GeometryError as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        files = [str(p.relative_to(REPO_ROOT)) for p in (recipe, load_version(recipe).overlays[0])]
        print(f"  Snapshot {name}: tagged, fingerprint {fp[:12]} ({', '.join(files)})")
        self._send_json({"ok": True, "name": name, "fingerprint": fp, "files": files})

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())


def run_server(args):
    viz_server.run_server(args, handler_cls=GeometryVizHandler, title="Geometry workbench")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANNIE Geometry Workbench</title>
<style>
  body { margin:0; overflow:hidden; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  #controls {
    position:absolute; top:10px; left:10px; z-index:100;
    background:rgba(245,247,250,0.92); color:#333; padding:14px 16px;
    border-radius:8px; width:270px; font-size:13px; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 2px 12px rgba(0,0,0,0.08);
  }
  #controls h3 { margin:0 0 10px; color:#111; font-size:15px; }
  #controls label { display:block; margin:6px 0; }
  #controls .row { display:flex; gap:6px; align-items:center; }
  #controls .row input[type=number] { width:60px; background:#fff; border:1px solid rgba(0,0,0,0.2); color:#111; padding:3px 6px; border-radius:4px; }
  #controls input[type=range] { width:100%; margin:2px 0; accent-color:#4488cc; }
  #controls input[type=checkbox] { accent-color:#4488cc; }
  #controls button {
    width:100%; margin-top:8px; padding:6px;
    background:#4488cc; color:#fff; border:none; border-radius:4px;
    font-size:13px; cursor:pointer;
  }
  #controls button:hover { background:#5599dd; }
  #controls button:disabled { opacity:0.5; cursor:default; }
  #result { margin-top:8px; font-size:12px; line-height:1.5; color:#333; }
  #result b { color:#111; }
  #status {
    position:absolute; top:10px; right:10px; z-index:100;
    background:rgba(245,247,250,0.92); color:#333; padding:8px 14px;
    border-radius:6px; font-size:13px; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12);
  }
  #info {
    position:absolute; bottom:10px; left:10px; z-index:100;
    color:rgba(0,0,0,0.5); font-size:12px;
    background:rgba(255,255,255,0.7); padding:4px 10px; border-radius:4px;
  }
  .hit-summary { display:grid; grid-template-columns:1fr 1fr; gap:2px 12px; }
  .hit-summary .label { color:#666; }
  .hit-summary .value { text-align:right; color:#000; }
  #pmt-adjust {
    display:none; position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:340px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #pmt-adjust .adj-row { display:flex; align-items:center; gap:8px; margin:4px 0; justify-content:center; }
  #pmt-adjust .adj-row label { min-width:80px; text-align:right; }
  #pmt-adjust .adj-row input[type=range] { width:140px; margin:0; accent-color:#4488cc; }
  #pmt-adjust .adj-row .val { min-width:50px; text-align:left; font-family:monospace; }
  #pmt-adjust button {
    margin:6px 4px 0; padding:5px 16px; border:none; border-radius:4px; font-size:12px; cursor:pointer;
  }
  #pmt-adjust #adj-save { background:#4488cc; color:#fff; }
  #pmt-adjust #adj-save:hover { background:#5599dd; }
  #pmt-adjust #adj-reset { background:#888; color:#fff; }
  #pmt-adjust #adj-reset:hover { background:#999; }
  #pmt-adjust #adj-cancel { background:#ccc; color:#333; }
  #pmt-adjust #adj-cancel:hover { background:#ddd; }
  #housing-popup {
    display:none; position:fixed; bottom:80px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:400px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #surfboard-popup {
    display:none; position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(240,242,245,0.95); color:#333; padding:12px 20px;
    border-radius:8px; font-size:13px; z-index:200; min-width:400px;
    text-align:center; backdrop-filter:blur(4px);
    border:1px solid rgba(0,0,0,0.12); box-shadow:0 4px 20px rgba(0,0,0,0.15);
  }
  #surfboard-popup .sb-title { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
  #surfboard-popup .sb-title h3 { margin:0; font-size:14px; }
  #surfboard-popup .sb-title button { background:none; border:none; font-size:16px; cursor:pointer; color:#666; padding:2px 6px; border-radius:4px; }
  #surfboard-popup .sb-title button:hover { background:rgba(0,0,0,0.1); color:#000; }
  #surfboard-popup .sb-row { display:flex; align-items:center; gap:6px; margin:4px 0; justify-content:center; }
  #surfboard-popup .sb-row label { min-width:80px; text-align:right; font-size:12px; }
  #surfboard-popup .sb-row input[type=number] { width:60px; text-align:center; font-family:monospace; font-size:12px; }
  #surfboard-popup .sb-row input[type=range] { width:180px; margin:0; accent-color:#4488cc; }
  #surfboard-popup button { margin:6px 4px 0; padding:5px 16px; border:none; border-radius:4px; font-size:12px; cursor:pointer; }
  #surfboard-popup #surfReset { background:#888; color:#fff; }
  #session-bar {
    position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:150;
    background:rgba(245,247,250,0.95); color:#333; padding:8px 14px; border-radius:6px;
    font-size:13px; border:1px solid rgba(0,0,0,0.12); box-shadow:0 2px 12px rgba(0,0,0,0.08);
    text-align:center; max-width:560px;
  }
  #session-bar.dirty { background:rgba(255,236,200,0.97); border-color:#d9a441; }
  #session-bar button { margin-left:8px; padding:4px 12px; border:none; border-radius:4px; font-size:12px;
    cursor:pointer; background:#4488cc; color:#fff; }
  #session-bar button#discardAll { background:#888; }
  #session-bar button:disabled { opacity:0.45; cursor:default; }
  #sessionDetail { font-size:11px; color:#666; margin-top:3px; }
  .keep-row { margin-top:6px; }
  .keep-row button { margin:4px 3px 0 !important; }
  .btn-keep { background:#3a9a5b !important; color:#fff; }
  .btn-revert { background:#b0763a !important; color:#fff; }
  .btn-cancel { background:#ccc !important; color:#333 !important; }
  .adj-state { font-size:11px; color:#996600; margin-top:4px; min-height:14px; }
  #surfboard-popup #surfReset:hover { background:#999; }
</style>
</head>
<body>
<div id="controls">
  <h3>Geometry Workbench</h3>
  <div id="geometryName" style="font-size:12px;color:#555;margin-bottom:6px;"></div>
  <details id="snapshotBox" style="font-size:12px;margin-bottom:6px;">
    <summary style="cursor:pointer;font-weight:600;">Snapshot (freeze &amp; tag)</summary>
    <div style="color:#666;margin:4px 0;">Copies the saved offsets into a new, tagged version.
      Save or discard your changes first.</div>
    <label>Name <input type="text" id="snapName" placeholder="annie-2026-data-v1.0" style="width:100%;"></label>
    <label>Description <input type="text" id="snapDesc" placeholder="what this snapshot is" style="width:100%;"></label>
    <button id="snapBtn" disabled>Create snapshot</button>
    <div id="snapResult" style="margin-top:4px;line-height:1.4;"></div>
  </details>
  <div style="font-size:12px;font-weight:600;margin-top:4px;">Light source</div>
  <label>Position
    <div class="row">
      X: <input type="number" id="mx" value="0" step="50">
      Y: <input type="number" id="my" value="0" step="50">
      Z: <input type="number" id="mz" value="2000" step="50">
    </div>
  </label>
  <label>θ (polar) <input type="range" id="theta" min="0" max="3.1416" step="any" value="1.5708"></label>
  <label>φ (azimuth) <input type="range" id="phi" min="0" max="6.2832" step="any" value="1.5708"></label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label>Ambient <span id="ambientVal">0.50</span>
    <input type="range" id="ambient" min="0" max="2.0" step="0.01" value="0.50"></label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <button id="focusLAPPD" disabled>Focus on LAPPD</button>
  <label style="margin-top:6px;"><input type="checkbox" id="lappdGrey"> Grey LAPPD</label>
  <div id="surfboard-adjust" style="display:none;"></div>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.1);">
  <label style="margin-top:6px;"><input type="checkbox" id="structGrey"> Grey Structure</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="structOpacity" min="0" max="1.0" step="0.01" value="1.0" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="pmtGrey"> Grey PMTs</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="pmtOpacity" min="0" max="1.0" step="0.01" value="0.85" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="showHW" checked> Show Holders</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showRefSpheres"> Show Reference Spheres</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">Opacity <input type="range" id="refOpacity" min="0" max="0.5" step="0.01" value="0.12" style="width:100px;"></label>
  <label style="margin-top:2px;"><input type="checkbox" id="showTubeIDs"> Show Tube IDs</label>
  <label style="margin-top:2px;"><input type="checkbox" id="showScan"> Show Scan Overlay</label>
  <label style="margin-top:0;font-size:12px;padding-left:24px;">
    Mesh <select id="scanMeshSelect" style="font-size:11px;">
      <option value="SuperStructure">SuperStructure</option>
      <option value="AllPMTs">All PMTs</option>
      <option value="BottomLayer">Bottom Layer</option>
      <option value="TopLayer">Top Layer</option>
      <option value="Panel-1">Panel 1</option>
      <option value="Panel-2">Panel 2</option>
      <option value="Panel-3">Panel 3</option>
      <option value="Panel-4">Panel 4</option>
      <option value="Panel-5">Panel 5</option>
      <option value="Panel-6">Panel 6</option>
      <option value="Panel-7">Panel 7</option>
      <option value="Panel-8">Panel 8</option>
      <option value="Panel-1-PMTs">Panel 1 PMTs</option>
      <option value="Panel-2-PMTs">Panel 2 PMTs</option>
      <option value="Panel-3-PMTs">Panel 3 PMTs</option>
      <option value="Panel-4-PMTs">Panel 4 PMTs</option>
      <option value="Panel-5-PMTs">Panel 5 PMTs</option>
      <option value="Panel-6-PMTs">Panel 6 PMTs</option>
      <option value="Panel-7-PMTs">Panel 7 PMTs</option>
      <option value="Panel-8-PMTs">Panel 8 PMTs</option>
      <option value="TopPMTs">Top PMTs</option>
      <option value="BottomPMTs">Bottom PMTs</option>
      <option value="TankLid">Tank Lid</option>
    </select>
  </label>
  <label style="margin-top:2px;"><input type="checkbox" id="showScanTips"> Show Scan Tips</label>
  <hr style="margin:8px 0;border:none;border-top:1px solid rgba(0,0,0,0.15);">
  <h3 style="margin-top:6px;">View</h3>
  <label>Azimuth <span id="viewAzVal">270</span>
    <input type="range" id="viewAz" min="0" max="360" step="1" value="270"></label>
  <label>Polar <span id="viewElVal">90</span>
    <input type="range" id="viewEl" min="0" max="180" step="1" value="90"></label>
  <label>Distance <span id="viewDistVal">4416</span>
    <input type="range" id="viewDist" min="100" max="10000" step="50" value="4416"></label>
  <div style="margin-top:4px;font-size:11px;color:#666;">Click a PMT, surfboard or LAPPD housing to adjust it</div>
</div>
<div id="status">Loading geometry…</div>
<div id="session-bar">
  <span id="sessionText">No unsaved changes</span>
  <button id="saveAll" disabled>Save to draft</button>
  <button id="discardAll" disabled>Discard all</button>
  <div id="sessionDetail"></div>
</div>
<div id="info">Drag to orbit · Scroll to zoom · Right-drag to pan</div>
<div id="pmt-adjust">
  <div style="margin-bottom:6px;"><b>PMT <span id="adj-tube-id"></span></b> <span id="adj-type" style="color:#888;font-size:11px;"></span></div>
  <div id="adj-sliders">
    <div class="adj-row">
      <label id="adj-label-0">Axial</label>
      <input type="range" id="adj-0" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-0">0.0</span> mm
    </div>
    <div class="adj-row">
      <label id="adj-label-1">Tangential</label>
      <input type="range" id="adj-1" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-1">0.0</span> mm
    </div>
    <div class="adj-row">
      <label id="adj-label-2">Vertical</label>
      <input type="range" id="adj-2" min="-100" max="100" step="0.1" value="0">
      <span class="val" id="adj-val-2">0.0</span> mm
    </div>
  </div>
  <div class="adj-state" id="adj-state"></div>
  <div style="margin-top:4px;">
    <button id="adj-save" class="btn-keep" title="Keep this position for the session (Save to draft makes it permanent)">Keep</button>
    <button id="adj-cancel" class="btn-cancel" title="Undo the slider changes since you opened this PMT">Cancel</button>
    <button id="adj-revert" class="btn-revert" title="Back to the position saved in the draft">Revert to saved</button>
    <button id="adj-reset" title="Sliders to zero: the base position, no offset">Base</button>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ---- State ----
let muonGroup = null;
let coneVisual = null;
let spotLight = null;
let spotTarget = null;
let housingData = null;
let housingMeshes = [];
let tracePoints = null;   // group of dot meshes for traced photon hits
let ckvGroup = null;      // subgroup of Cherenkov hit dots inside tracePoints
let sciGroup = null;      // subgroup of scintillation hit dots inside tracePoints
let rayLines = null;      // group of line meshes for sampled ray paths
let refSpheres = null;    // group of translucent reference spheres
let scanOverlay = null;   // group for scan mesh overlay
let pmtGroup = null;      // group of PMT body meshes
let pmtHWGroup = null;    // group of PMT holder meshes
let pmtData = null;       // full PMT data from /api/pmts
let selectedIdx = -1;     // index of selected PMT, -1 = none
let selectedSurfboard = -1; // index of selected surfboard, -1 = none
let selectedHousing = -1; // index of selected LAPPD housing, -1 = none
let selectedMesh = null;  // the mesh object currently highlighted

// ---- DOM refs ----
const statusEl = document.getElementById('status');
const thetaSlider = document.getElementById('theta');
const phiSlider = document.getElementById('phi');

// ---- Scene ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x7088a0);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 20000);
camera.up.set(0, 0, 1);
// Start upstream, looking downstream along the beam (+y)
camera.position.set(0, -4416, 2000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NoToneMapping;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2000);
controls.update();

// Environment map for glass/reflections
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(renderer), 0.04).texture;
pmremGenerator.dispose();

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);

// Spotlight from muon vertex (Cherenkov cone light source)
spotTarget = new THREE.Object3D();
scene.add(spotTarget);

spotLight = new THREE.SpotLight(0xffffff, 1.0);
spotLight.angle = 0.73;
spotLight.penumbra = 0.08;
spotLight.decay = 0;
spotLight.distance = 0;
spotLight.castShadow = true;
spotLight.shadow.mapSize.width = 2048;
spotLight.shadow.mapSize.height = 2048;
spotLight.shadow.camera.near = 10;
spotLight.shadow.camera.far = 5000;
spotLight.shadow.bias = 0;
spotLight.shadow.normalBias = 0;
scene.add(spotLight);

// ---- Labeled axes + origin ----
function makeTextSprite(text, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'Bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, 64, 32);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(200, 100, 1);
    return sprite;
}

function addAxesAndOrigin() {
    const axisLen = 600;
    const group = new THREE.Group();

    // Axis arrows
    const axisDefs = [
        { dir: [1,0,0], color: 0xff0000, label: 'X' },
        { dir: [0,1,0], color: 0x00ff00, label: 'Y' },
        { dir: [0,0,1], color: 0x0066ff, label: 'Z' },
    ];
    for (const a of axisDefs) {
        const d = new THREE.Vector3(a.dir[0], a.dir[1], a.dir[2]);
        const arrow = new THREE.ArrowHelper(d, new THREE.Vector3(0,0,0),
            axisLen, a.color, 80, 40);
        group.add(arrow);

        // Label at tip
        const tip = d.clone().multiplyScalar(axisLen * 1.25);
        const sp = makeTextSprite(a.label, '#' + a.color.toString(16).padStart(6, '0'));
        sp.position.copy(tip);
        group.add(sp);
    }

    // Negative axis stubs (dashed or thin)
    for (const a of axisDefs) {
        const d = new THREE.Vector3(-a.dir[0], -a.dir[1], -a.dir[2]);
        const mat = new THREE.LineBasicMaterial({
            color: a.color, transparent: true, opacity: 0.25,
        });
        const pts = [new THREE.Vector3(0,0,0), d.clone().multiplyScalar(axisLen * 0.4)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        group.add(new THREE.Line(geo, mat));
    }

    // Origin sphere
    const originGeo = new THREE.SphereGeometry(20, 16, 12);
    const originMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const originMesh = new THREE.Mesh(originGeo, originMat);
    originMesh.position.set(0, 0, 0);
    group.add(originMesh);

    // "O" label
    const oLabel = makeTextSprite('O', '#000000');
    oLabel.position.set(0, -70, 0);
    group.add(oLabel);

    scene.add(group);
}

addAxesAndOrigin();

// ---- Material colours ----
const SURFACE_COLOR = new THREE.Color(0xffffff);
const GREY_COLOR = new THREE.Color(0x999999);

// ---- Load mesh ----
async function loadMesh() {
    const [vertsResp, trisResp] = await Promise.all([
        fetch('/api/mesh/verts'),
        fetch('/api/mesh/tris'),
    ]);
    const vertsBuf = await vertsResp.arrayBuffer();
    const trisBuf = await trisResp.arrayBuffer();

    const positions = new Float32Array(vertsBuf);
    const indices = new Int32Array(trisBuf);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    return geo;
}

// ---- Load PMTs ----
async function loadPMTs() {
    const resp = await fetch('/api/pmts');
    return await resp.json();
}

// ---- PMT type colours ----
const PMT_COLORS = {
    'LUX': 0xffaa00,
    'ETEL': 0x00ccff,
    'Hamamatsu': 0x44aaff,
    'Watchboy': 0xff8844,
    'Watchman': 0xff8844,
};

// ---- Light update ----
function updateMuonAndLight(pos, dir) {
    const d = new THREE.Vector3(dir.x, dir.y, dir.z).normalize();
    const p = new THREE.Vector3(pos.x, pos.y, pos.z);

    // Spotlight from the light-source position, along its direction
    spotLight.position.copy(p);
    const targetPos = p.clone().add(d.clone().multiplyScalar(3000));
    spotTarget.position.copy(targetPos);
    spotLight.target = spotTarget;
}

// ---- Ambient slider ----
const ambientSlider = document.getElementById('ambient');
const ambientLabel = document.getElementById('ambientVal');
ambientSlider.addEventListener('input', () => {
    ambient.intensity = parseFloat(ambientSlider.value);
    ambientLabel.textContent = ambientSlider.value;
});

// ---- Update from controls ----
function updateScene() {
    const mx = parseFloat(document.getElementById('mx').value);
    const my = parseFloat(document.getElementById('my').value);
    const mz = parseFloat(document.getElementById('mz').value);
    const theta = parseFloat(thetaSlider.value);
    const phi = parseFloat(phiSlider.value);

    const dx = Math.sin(theta) * Math.cos(phi);
    const dy = Math.sin(theta) * Math.sin(phi);
    const dz = Math.cos(theta);

    updateMuonAndLight({ x: mx, y: my, z: mz }, { x: dx, y: dy, z: dz });
}

// ---- Events ----
thetaSlider.addEventListener('input', updateScene);
phiSlider.addEventListener('input', updateScene);
document.getElementById('mx').addEventListener('change', updateScene);
document.getElementById('my').addEventListener('change', updateScene);
document.getElementById('mz').addEventListener('change', updateScene);

// ---- Geometry name ----
fetch('/api/geometry').then(r => r.json()).then(g => {
    document.getElementById('geometryName').textContent =
        `${g.name} (${g.status})` + (g.editable ? ' · adjustments can be saved to the draft'
                                                : ' · view only (not a draft): changes cannot be saved');
}).catch(() => {});

// ---- Adjustments: preview, keep, save ----
// Every element's offset is a total relative to its base position in the
// geometry version.  Slider moves are previews; Keep adds the element's new
// total to this session's changes; Save to draft writes them all.
const SECTIONS = ['pmts', 'lappds', 'surfboards'];
const SECTION_NAMES = {pmts: 'PMT', lappds: 'LAPPD', surfboards: 'Surfboard'};
let savedOffsets = {pmts: {}, lappds: {}, surfboards: {}};
const pending = {pmts: {}, lappds: {}, surfboards: {}};
const pendingRevert = {pmts: {}, lappds: {}, surfboards: {}};  // key -> fn putting it back to saved
let offsetsEditable = false;
let overlayName = '';

async function loadOffsets() {
    const o = await (await fetch('/api/offsets')).json();
    for (const s of SECTIONS) savedOffsets[s] = o.saved[s] || {};
    offsetsEditable = o.editable;
    overlayName = o.overlay ? o.overlay.split('/').pop() : '';
    updateSessionBar();
}
function savedOf(section, key) { return savedOffsets[section][String(key)] || [0, 0, 0]; }
function totalOf(section, key) { return pending[section][String(key)] || savedOf(section, key); }
function nearlyEqual(a, b) { return a.every((v, i) => Math.abs(v - b[i]) < 0.05); }
function keepChange(section, key, total, revertFn) {
    const k = String(key);
    if (nearlyEqual(total, savedOf(section, k))) {
        delete pending[section][k]; delete pendingRevert[section][k];
    } else {
        pending[section][k] = total.map(v => Math.round(v * 1000) / 1000);
        pendingRevert[section][k] = revertFn;
    }
    updateSessionBar();
}
function dropChange(section, key) {
    delete pending[section][String(key)]; delete pendingRevert[section][String(key)];
    updateSessionBar();
}
function nPending() { return SECTIONS.reduce((n, s) => n + Object.keys(pending[s]).length, 0); }
function stateText(section, key) {
    const k = String(key);
    const fmt = v => '(' + v.map(x => x.toFixed(1)).join(', ') + ') mm';
    const saved = 'saved offset ' + fmt(savedOf(section, k));
    return pending[section][k] ? `kept, not saved yet: ${fmt(pending[section][k])} · ${saved}` : saved;
}
function updateSessionBar() {
    const n = nPending();
    const bar = document.getElementById('session-bar');
    bar.classList.toggle('dirty', n > 0);
    document.getElementById('sessionText').textContent = n === 0
        ? (offsetsEditable ? `No unsaved changes · saving to ${overlayName}` : 'View only: changes cannot be saved')
        : `${n} unsaved change${n > 1 ? 's' : ''}`;
    document.getElementById('saveAll').disabled = n === 0 || !offsetsEditable;
    document.getElementById('snapBtn').disabled = n > 0 || !offsetsEditable;
    document.getElementById('snapBtn').title = !offsetsEditable ? 'Only a draft version can be snapshot'
        : n > 0 ? 'Save to draft or Discard all first' : 'Freeze the saved offsets as a new tagged version';
    document.getElementById('discardAll').disabled = n === 0;
    const parts = [];
    for (const s of SECTIONS) for (const k of Object.keys(pending[s])) parts.push(`${SECTION_NAMES[s]} ${k}`);
    document.getElementById('sessionDetail').textContent =
        parts.join(', ') + (n > 0 && !offsetsEditable ? ' · not a draft: cannot be saved' : '');
}
document.getElementById('saveAll').addEventListener('click', async () => {
    const resp = await fetch('/api/offsets/save', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(pending),
    });
    const r = await resp.json();
    if (!resp.ok) { document.getElementById('sessionDetail').textContent = 'Save failed: ' + r.error; return; }
    for (const s of SECTIONS) {
        for (const k of Object.keys(pending[s])) savedOffsets[s][k] = pending[s][k];
        pending[s] = {}; pendingRevert[s] = {};
    }
    updateSessionBar();
    document.getElementById('sessionText').textContent = `Saved ${r.saved} change${r.saved > 1 ? 's' : ''} to ${overlayName}`;
});
document.getElementById('discardAll').addEventListener('click', () => {
    for (const s of SECTIONS) for (const k of Object.keys(pendingRevert[s])) pendingRevert[s][k]();
    for (const s of SECTIONS) { pending[s] = {}; pendingRevert[s] = {}; }
    updateSessionBar();
});
document.getElementById('snapBtn').addEventListener('click', async () => {
    const name = document.getElementById('snapName').value.trim();
    const out = document.getElementById('snapResult');
    if (!name) { out.textContent = 'Give the snapshot a name, e.g. annie-2026-data-v1.0'; return; }
    out.textContent = 'Creating…';
    const resp = await fetch('/api/snapshot', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, description: document.getElementById('snapDesc').value}),
    });
    const r = await resp.json();
    if (!resp.ok) { out.textContent = 'Not created: ' + r.error; return; }
    out.innerHTML = `Tagged <b>${r.name}</b> (fingerprint ${r.fingerprint.slice(0, 12)}).<br>`
        + `Commit and push: <code>git add ${r.files.map(f => `"${f}"`).join(' ')}</code><br>`
        + `Use it with <code>--geometry ${r.name}</code>.`;
});
window.addEventListener('beforeunload', (e) => {
    if (nPending() > 0) { e.preventDefault(); e.returnValue = ''; }
});

// ---- Init ----
async function init() {
    try {
        await loadOffsets();
        statusEl.textContent = 'Loading structure mesh…';
        const meshGeo = await loadMesh();

        // Flatten indexed geometry to non-indexed (indexed path has rendering issues)
        const pos = meshGeo.getAttribute('position');
        const idx = meshGeo.getIndex();
        const triCount = idx.count / 3;
        const flatPos = new Float32Array(triCount * 9);
        for (let i = 0; i < triCount; i++) {
            for (let j = 0; j < 3; j++) {
                const vi = idx.getX(i * 3 + j);
                flatPos[i * 9 + j * 3 + 0] = pos.getX(vi);
                flatPos[i * 9 + j * 3 + 1] = pos.getY(vi);
                flatPos[i * 9 + j * 3 + 2] = pos.getZ(vi);
            }
        }
        const flatGeo = new THREE.BufferGeometry();
        flatGeo.setAttribute('position', new THREE.BufferAttribute(flatPos, 3));
        flatGeo.computeVertexNormals();

        const meshMat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            roughness: 0.5,
            metalness: 0.05,
            transparent: true,
            opacity: 1.0,
            side: THREE.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: 1.0,
            polygonOffsetUnits: 4.0,
        });
        const mesh = new THREE.Mesh(flatGeo, meshMat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        statusEl.textContent = 'Loading PMTs…';
        const pmtData = await loadPMTs();

        // Load PMT mesh geometries for needed types
        const meshGeos = {};       // glass outer shell (full mesh)
        const meshPcGeos = {};     // PC inner layer (brown, 1 mm inside glass)
        const meshPvcGeos = {};    // PVC housing/wings (dark grey)
        const hwMeshGeos = {};
        const neededTypes = new Set(pmtData.mesh_types || []);
        const neededHwTypes = new Set((pmtData.hw_mesh_types || []).filter(v => v >= 0));
        for (const mt of neededTypes) {
            try {
                const meshResp = await fetch(`/api/pmt_mesh/${mt}`);
                if (meshResp.ok) {
                    const buf = await meshResp.arrayBuffer();
                    const flat = new Float32Array(buf);
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                    geo.computeVertexNormals();
                    meshGeos[mt] = geo;
                }
                // PC sub-mesh (translucent brown inner layer)
                const pcResp = await fetch(`/api/pmt_mesh_pc/${mt}`);
                if (pcResp.ok) {
                    const buf = await pcResp.arrayBuffer();
                    if (buf.byteLength > 0) {
                        const flat = new Float32Array(buf);
                        const geo = new THREE.BufferGeometry();
                        geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                        geo.computeVertexNormals();
                        meshPcGeos[mt] = geo;
                    }
                }
                // PVC sub-mesh (dark grey housing / wings)
                const pvcResp = await fetch(`/api/pmt_mesh_pvc/${mt}`);
                if (pvcResp.ok) {
                    const buf = await pvcResp.arrayBuffer();
                    if (buf.byteLength > 0) {
                        const flat = new Float32Array(buf);
                        const geo = new THREE.BufferGeometry();
                        geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                        geo.computeVertexNormals();
                        meshPvcGeos[mt] = geo;
                    }
                }
            } catch (e) {
                console.warn(`No mesh for type ${mt}`);
            }
        }
        for (const mt of neededHwTypes) {
            try {
                const [meshResp, colResp] = await Promise.all([
                    fetch(`/api/pmt_mesh/${mt}`),
                    fetch(`/api/pmt_mesh_colors/${mt}`),
                ]);
                if (meshResp.ok) {
                    const buf = await meshResp.arrayBuffer();
                    const flat = new Float32Array(buf);
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
                    if (colResp.ok) {
                        const colBuf = await colResp.arrayBuffer();
                        const colors = new Uint8Array(colBuf);
                        geo.setAttribute('color', new THREE.BufferAttribute(colors, 4, true));
                    }
                    geo.computeVertexNormals();
                    hwMeshGeos[mt] = geo;
                }
            } catch (e) {
                console.warn(`No HW mesh for type ${mt}`);
            }
        }

        // Render PMTs — dual-layer: glass outer shell + PC inner coating
        pmtGroup = new THREE.Group();
        pmtHWGroup = new THREE.Group();

        // Materials (cloned per-PMT so grey/opacity toggles work individually)
        function makeGlassMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.97, 0.97, 0.99),  // faint cool white
                roughness: 0.0,
                metalness: 0.0,
                transparent: true,
                opacity: 1.0,
                side: THREE.DoubleSide,
                envMap: scene.environment,
                envMapIntensity: 1.8,
                depthWrite: false,
                transmission: 0.92,
                thickness: 0.5,
                ior: 1.5,
            });
        }
        function makePcMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.55, 0.20, 0.10),  // warm brown
                roughness: 0.4,
                metalness: 0.0,
                transparent: true,
                opacity: 0.35,   // thin-film PC — subtle tint through glass
                side: THREE.DoubleSide,
                depthWrite: true,
                polygonOffset: true,
                polygonOffsetFactor: 1.0,
                polygonOffsetUnits: 4.0,
            });
        }
        function makePvcMat() {
            return new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(0.10, 0.10, 0.12),  // dark grey
                roughness: 0.8,
                metalness: 0.0,
                side: THREE.DoubleSide,
                depthWrite: true,
                polygonOffset: true,
                polygonOffsetFactor: 1.0,
                polygonOffsetUnits: 4.0,
            });
        }

        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const r = pmtData.radii[i];
            const d = pmtData.directions[i];
            const type = pmtData.types[i];
            const color = PMT_COLORS[type] || 0x999999;

            const mt = pmtData.mesh_types ? pmtData.mesh_types[i] : -1;
            const geo = meshGeos[mt];

            if (geo) {
                const ip = pmtData.instance_positions[i];
                const q = pmtData.quaternions[i];

                // Glass outer shell (transparent, reflective)
                const glassMat = makeGlassMat();
                const glassMesh = new THREE.Mesh(geo, glassMat);
                glassMesh.position.set(ip[0], ip[1], ip[2]);
                glassMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                glassMesh.castShadow = true;
                glassMesh.receiveShadow = true;
                glassMesh.renderOrder = 1;
                glassMesh.userData.pmtIdx = i;
                glassMesh.userData.tubeId = pmtData.detector_nums[i];
                pmtGroup.add(glassMesh);

                // PC inner layer (brown, offset 1 mm inside glass)
                const pcGeo = meshPcGeos[mt];
                if (pcGeo) {
                    const pcMatMesh = makePcMat();
                    const pcMesh = new THREE.Mesh(pcGeo, pcMatMesh);
                    pcMesh.position.set(ip[0], ip[1], ip[2]);
                    pcMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    pcMesh.castShadow = true;
                    pcMesh.receiveShadow = true;
                    pcMesh.renderOrder = 0;
                    pcMesh.userData.pmtIdx = i;
                    pcMesh.userData.tubeId = pmtData.detector_nums[i];
                    pcMesh.userData.isPC = true;
                    pmtGroup.add(pcMesh);
                }

                // PVC housing / wings (dark grey, only LUX/ETEL have these)
                const pvcGeo = meshPvcGeos[mt];
                if (pvcGeo) {
                    const pvcMat = makePvcMat();
                    const pvcMesh = new THREE.Mesh(pvcGeo, pvcMat);
                    pvcMesh.position.set(ip[0], ip[1], ip[2]);
                    pvcMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    pvcMesh.castShadow = true;
                    pvcMesh.receiveShadow = true;
                    pvcMesh.renderOrder = 2;
                    pvcMesh.userData.pmtIdx = i;
                    pvcMesh.userData.tubeId = pmtData.detector_nums[i];
                    pvcMesh.userData.isPVC = true;
                    pmtGroup.add(pvcMesh);
                }

                // Hardware mesh at same position/orientation
                const hwmt = pmtData.hw_mesh_types ? pmtData.hw_mesh_types[i] : -1;
                const hwGeo = hwMeshGeos[hwmt];
                if (hwGeo) {
                    const hwHasColors = hwGeo.getAttribute('color') != null;
                    const hwMat = new THREE.MeshPhysicalMaterial({
                        color: hwHasColors ? 0xffffff : 0x887766,
                        vertexColors: hwHasColors,
                        roughness: 0.5,
                        metalness: 0.3,
                        transparent: true,
                        opacity: 0.7,
                        side: THREE.DoubleSide,
                        polygonOffset: true,
                        polygonOffsetFactor: 1.0,
                        polygonOffsetUnits: 4.0,
                    });
                    const hwMesh = new THREE.Mesh(hwGeo, hwMat);
                    hwMesh.position.set(ip[0], ip[1], ip[2]);
                    hwMesh.quaternion.set(q[0], q[1], q[2], q[3]);
                    hwMesh.castShadow = true;
                    hwMesh.receiveShadow = true;
                    hwMesh.userData.pmtIdx = i;
                    hwMesh.userData.tubeId = pmtData.detector_nums[i];
                    pmtHWGroup.add(hwMesh);
                }
            } else {
                const sphereGeo = new THREE.SphereGeometry(r * 0.35, 16, 12);
                const sphereMat = new THREE.MeshStandardMaterial({
                    color: color,
                    roughness: 0.4,
                    metalness: 0.05,
                    transparent: true,
                    opacity: 0.85,
                });
                const sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.set(c[0], c[1], c[2]);
                sphere.castShadow = true;
                sphere.receiveShadow = true;
                sphere.userData.pmtIdx = i;
                sphere.userData.tubeId = pmtData.detector_nums[i];
                pmtGroup.add(sphere);
            }

            const dir = new THREE.Vector3(d[0], d[1], d[2]).normalize();
            const origin = new THREE.Vector3(c[0], c[1], c[2]);
            const arrow = new THREE.ArrowHelper(dir, origin, 50, color, 18, 10);
            pmtGroup.add(arrow);

            // Red dot at bulb tip (centers[i]) for alignment debugging
            const dotGeo = new THREE.SphereGeometry(6, 8, 6);
            const dotMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.position.set(c[0], c[1], c[2]);
            pmtGroup.add(dot);
        }
        scene.add(pmtGroup);
        pmtHWGroup.visible = document.getElementById('showHW').checked;  // on by default
        scene.add(pmtHWGroup);

        // ---- Tube ID labels ----
        const labelGroup = new THREE.Group();
        labelGroup.visible = false;
        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const tid = pmtData.detector_nums[i];
            const canvas = document.createElement('canvas');
            canvas.width = 192;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgba(0,0,0,0.55)';
            ctx.beginPath();
            ctx.roundRect(0, 0, 192, 64, 8);
            ctx.fill();
            ctx.font = 'bold 28px monospace';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(tid), 96, 34);
            const tex = new THREE.CanvasTexture(canvas);
            tex.needsUpdate = true;
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, sizeAttenuation: true });
            const sprite = new THREE.Sprite(mat);
            sprite.position.set(c[0], c[1], c[2] + 200);
            sprite.scale.set(400, 140, 1);
            sprite.userData.pmtIdx = i;
            labelGroup.add(sprite);
        }
        scene.add(labelGroup);

        // Reference spheres at each PMT center+radius (translucent)
        refSpheres = new THREE.Group();
        for (let i = 0; i < pmtData.centers.length; i++) {
            const c = pmtData.centers[i];
            const r = pmtData.radii[i];
            const type = pmtData.types[i];
            const color = PMT_COLORS[type] || 0x999999;
            const sphereGeo = new THREE.SphereGeometry(r, 32, 24);
            const sphereMat = new THREE.MeshPhysicalMaterial({
                color: color,
                transparent: true,
                opacity: 0.12,
                roughness: 0.0,
                metalness: 0.0,
                depthWrite: false,
                side: THREE.DoubleSide,
            });
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.set(c[0], c[1], c[2]);
            refSpheres.add(sphere);
        }
        refSpheres.visible = false;
        scene.add(refSpheres);

        // Scan overlay (loaded on demand)
        scanOverlay = new THREE.Group();
        scanOverlay.visible = false;
        scene.add(scanOverlay);

        async function loadScanMesh(name) {
            const [vertsResp, trisResp] = await Promise.all([
                fetch(`/api/scan_mesh/${name}/verts`),
                fetch(`/api/scan_mesh/${name}/tris`),
            ]);
            if (!vertsResp.ok || !trisResp.ok) return null;
            const vertsBuf = await vertsResp.arrayBuffer();
            const trisBuf = await trisResp.arrayBuffer();
            const positions = new Float32Array(vertsBuf);
            const indices = new Int32Array(trisBuf);
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setIndex(new THREE.BufferAttribute(indices, 1));
            // Flatten indexed to non-indexed (indexed path has rendering issues)
            const pos = geo.getAttribute('position');
            const idx = geo.getIndex();
            const triCount = idx.count / 3;
            const flatPos = new Float32Array(triCount * 9);
            for (let i = 0; i < triCount; i++) {
                for (let j = 0; j < 3; j++) {
                    const vi = idx.getX(i * 3 + j);
                    flatPos[i * 9 + j * 3 + 0] = pos.getX(vi);
                    flatPos[i * 9 + j * 3 + 1] = pos.getY(vi);
                    flatPos[i * 9 + j * 3 + 2] = pos.getZ(vi);
                }
            }
            const flatGeo = new THREE.BufferGeometry();
            flatGeo.setAttribute('position', new THREE.BufferAttribute(flatPos, 3));
            flatGeo.computeVertexNormals();
            return flatGeo;
        }

        statusEl.textContent = `Loaded ${pmtData.centers.length} PMTs.`;

        // Load ANNIE LAPPD housings (if present)
        const housingResp = await fetch('/api/housing');
        housingData = await housingResp.json();
        housingMeshes.length = 0;
        let housingCorrOrigins = [];  // original centers per housing (before global correction)
        if (Array.isArray(housingData.housing) && housingData.housing.length > 0) {
            const boxMat = new THREE.MeshStandardMaterial({
                color: 0x446688,
                transparent: true,
                opacity: 0.8,
                roughness: 0.6,
                metalness: 0.0,
                side: THREE.DoubleSide,
            });
            const pcMat = new THREE.MeshStandardMaterial({
                color: 0x88bbdd,
                roughness: 0.3,
                metalness: 0.1,
                side: THREE.DoubleSide,
                polygonOffset: true,
                polygonOffsetFactor: 1.0,
                polygonOffsetUnits: 4.0,
            });
            let hidx = 0;
            for (const h of housingData.housing) {
                const origCenter = new THREE.Vector3(h.center[0], h.center[1], h.center[2]);
                const origPC = new THREE.Vector3(h.pc_center[0], h.pc_center[1], h.pc_center[2]);

                const boxGeo = new THREE.BoxGeometry(h.half[0]*2, h.half[1]*2, h.half[2]*2);
                const boxMesh = new THREE.Mesh(boxGeo, boxMat);
                boxMesh.position.copy(origCenter);
                const m4 = new THREE.Matrix4();
                m4.set(
                    h.axis_x[0], h.axis_y[0], h.axis_z[0], 0,
                    h.axis_x[1], h.axis_y[1], h.axis_z[1], 0,
                    h.axis_x[2], h.axis_y[2], h.axis_z[2], 0,
                    0, 0, 0, 1,
                );
                boxMesh.quaternion.setFromRotationMatrix(m4);
                boxMesh.castShadow = true;
                boxMesh.receiveShadow = false;  // opaque here: self-shadowing speckles it
                boxMesh.userData = {
                    isHousing: true,
                    hIdx: hidx,
                    origCenter: origCenter.clone(),
                    origPC: origPC.clone(),
                    axisX: new THREE.Vector3(h.axis_x[0], h.axis_x[1], h.axis_x[2]),
                    axisY: new THREE.Vector3(h.axis_y[0], h.axis_y[1], h.axis_y[2]),
                    axisZ: new THREE.Vector3(h.axis_z[0], h.axis_z[1], h.axis_z[2]),
                };
                const hSaved = new THREE.Vector3(...savedOf('lappds', 2000 + hidx));  // served positions include saved offsets
                boxMesh.userData.base = origCenter.clone().sub(hSaved);
                boxMesh.userData.basePC = origPC.clone().sub(hSaved);
                scene.add(boxMesh);

                const pcGeo = new THREE.PlaneGeometry(h.pc_half[0]*2, h.pc_half[0]*2);
                const pcMesh = new THREE.Mesh(pcGeo, pcMat);
                pcMesh.position.copy(origPC);
                pcMesh.quaternion.copy(boxMesh.quaternion);
                pcMesh.castShadow = true;
                pcMesh.receiveShadow = false;
                scene.add(pcMesh);

                housingMeshes.push({box: boxMesh, pc: pcMesh});
                housingCorrOrigins.push({center: origCenter.clone(), pc: origPC.clone()});
                hidx++;
            }

            // Grey LAPPD checkbox
            const greyCheck = document.getElementById('lappdGrey');
            greyCheck.addEventListener('change', () => {
                const isGrey = greyCheck.checked;
                for (const hm of housingMeshes) {
                    if (isGrey) {
                        hm.box.material.color.setHex(0x999999);
                        hm.box.material.transparent = false;
                        hm.box.material.opacity = 1.0;
                        hm.box.material.roughness = 0.5;
                        hm.box.material.metalness = 0.05;
                        hm.pc.material.color.setHex(0x999999);
                        hm.pc.material.roughness = 0.5;
                        hm.pc.material.metalness = 0.05;
                    } else {
                        hm.box.material.color.setHex(0x446688);
                        hm.box.material.transparent = true;
                        hm.box.material.opacity = 0.8;
                        hm.box.material.roughness = 0.6;
                        hm.box.material.metalness = 0.0;
                        hm.pc.material.color.setHex(0x88bbdd);
                        hm.pc.material.roughness = 0.3;
                        hm.pc.material.metalness = 0.1;
                    }
                }
            });

            document.getElementById('focusLAPPD').disabled = false;
        }

        // ---- Surfboard obscurant panels ----
        const surfResp = await fetch('/api/surfboards');
        const surfData = await surfResp.json();
        const surfMeshes = [];
        if (Array.isArray(surfData.surfboards)) {
            const surfMat = new THREE.MeshStandardMaterial({
                color: 0x333344,
                roughness: 0.8,
                metalness: 0.0,
                side: THREE.DoubleSide,
                polygonOffset: true,
                polygonOffsetFactor: 1.0,
                polygonOffsetUnits: 4.0,
            });
            let sbIdx = 0;
            for (const sb of surfData.surfboards) {
                const geo = new THREE.BoxGeometry(sb.half[0]*2, sb.half[1]*2, sb.half[2]*2);
                const mesh = new THREE.Mesh(geo, surfMat);
                mesh.position.set(sb.center[0], sb.center[1], sb.center[2]);
                const m4 = new THREE.Matrix4();
                m4.set(
                    sb.axis_x[0], sb.axis_y[0], sb.axis_z[0], 0,
                    sb.axis_x[1], sb.axis_y[1], sb.axis_z[1], 0,
                    sb.axis_x[2], sb.axis_y[2], sb.axis_z[2], 0,
                    0, 0, 0, 1,
                );
                mesh.quaternion.setFromRotationMatrix(m4);
                mesh.castShadow = true;
                mesh.receiveShadow = false;  // a 10 mm board self-shadows into speckle
                mesh.userData = {
                    isSurfboard: true,
                    sbIdx: sbIdx,
                    origCenter: new THREE.Vector3(sb.center[0], sb.center[1], sb.center[2]),
                    axisX: new THREE.Vector3(sb.axis_x[0], sb.axis_x[1], sb.axis_x[2]),
                    axisY: new THREE.Vector3(sb.axis_y[0], sb.axis_y[1], sb.axis_y[2]),
                    axisZ: new THREE.Vector3(sb.axis_z[0], sb.axis_z[1], sb.axis_z[2]),
                };
                const sbSaved = savedOf('surfboards', sbIdx);   // served positions include saved offsets
                mesh.userData.base = mesh.userData.origCenter.clone().sub(new THREE.Vector3(...sbSaved));
                scene.add(mesh);
                surfMeshes.push(mesh);
                sbIdx++;
            }
        }

        // ---- Grey toggles for structure and PMTs ----
        const structGreyCheck = document.getElementById('structGrey');
        const pmtGreyCheck = document.getElementById('pmtGrey');

        structGreyCheck.addEventListener('change', () => {
            meshMat.color.setHex(structGreyCheck.checked ? 0x999999 : 0xffffff);
        });

        const structOpacitySlider = document.getElementById('structOpacity');
        structOpacitySlider.addEventListener('input', () => {
            meshMat.opacity = parseFloat(structOpacitySlider.value);
        });

        pmtGreyCheck.addEventListener('change', () => {
            const isGrey = pmtGreyCheck.checked;
            pmtGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    const idx = child.userData.pmtIdx;
                    if (idx === undefined) return;
                    if (isGrey) {
                        child.material.color.setHex(0x999999);
                        if (!child.userData.isPC && !child.userData.isPVC) {
                            child.material.transmission = 0.0;
                        }
                    } else {
                        if (child.userData.isPC) {
                            child.material.color.setRGB(0.55, 0.20, 0.10);
                        } else if (child.userData.isPVC) {
                            child.material.color.setRGB(0.10, 0.10, 0.12);
                        } else {
                            child.material.color.setRGB(0.97, 0.97, 0.99);
                            child.material.transmission = 0.92;
                        }
                    }
                }
            });
        });

        const showHWCheck = document.getElementById('showHW');
        showHWCheck.addEventListener('change', () => {
            pmtHWGroup.visible = showHWCheck.checked;
        });

        const showRefCheck = document.getElementById('showRefSpheres');
        showRefCheck.addEventListener('change', () => {
            refSpheres.visible = showRefCheck.checked;
        });

        const refOpacitySlider = document.getElementById('refOpacity');
        refOpacitySlider.addEventListener('input', () => {
            const op = parseFloat(refOpacitySlider.value);
            refSpheres.children.forEach(child => {
                if (child.isMesh && child.material) {
                    child.material.opacity = op;
                }
            });
        });

        const showTubeIDsCheck = document.getElementById('showTubeIDs');
        showTubeIDsCheck.addEventListener('change', () => {
            labelGroup.visible = showTubeIDsCheck.checked;
        });

        const showScanCheck = document.getElementById('showScan');
        const scanMeshSelect = document.getElementById('scanMeshSelect');
        let scanLoadedName = '';

        async function loadScanOverlay(name) {
            statusEl.textContent = `Loading ${name}…`;
            const geo = await loadScanMesh(name);
            if (!geo) {
                statusEl.textContent = `Scan mesh '${name}' not available.`;
                return null;
            }
            const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute('position'));
            console.log(`Scan mesh '${name}' bounds:`, box.min.toArray(), box.max.toArray());
            const mat = new THREE.MeshPhysicalMaterial({
                color: 0x44aadd,
                roughness: 0.5,
                metalness: 0.0,
                transparent: true,
                opacity: 0.65,
                side: THREE.DoubleSide,
                depthWrite: true,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.castShadow = false;
            mesh.receiveShadow = false;
            scanOverlay.add(mesh);
            scanLoadedName = name;
            statusEl.textContent = `Scan overlay '${name}' loaded.`;
            return mesh;
        }

        showScanCheck.addEventListener('change', async () => {
            if (showScanCheck.checked) {
                if (scanOverlay.children.length === 0) {
                    await loadScanOverlay(scanMeshSelect.value);
                }
                scanOverlay.visible = true;
            } else {
                scanOverlay.visible = false;
            }
        });

        scanMeshSelect.addEventListener('change', async () => {
            const name = scanMeshSelect.value;
            // Clear current mesh
            while (scanOverlay.children.length > 0) {
                const child = scanOverlay.children[0];
                child.geometry.dispose();
                child.material.dispose();
                scanOverlay.remove(child);
            }
            scanLoadedName = '';
            if (showScanCheck.checked) {
                await loadScanOverlay(name);
                scanOverlay.visible = true;
            }
        });

        // ---- PMT Scan Tips ----
        let pmtTips = null;
        const showScanTipsCheck = document.getElementById('showScanTips');
        showScanTipsCheck.addEventListener('change', () => {
            if (pmtTips) pmtTips.visible = showScanTipsCheck.checked;
        });

        async function loadPmtTips() {
            const resp = await fetch('/api/pmt_tips');
            const data = await resp.json();
            if (!data.tips || data.tips.length === 0) return;

            const tipGroup = new THREE.Group();
            const highMat = new THREE.MeshBasicMaterial({ color: 0x44dd44 });
            const medMat = new THREE.MeshBasicMaterial({ color: 0xdddd44 });
            const lowMat = new THREE.MeshBasicMaterial({ color: 0xdd4444 });

            for (const t of data.tips) {
                if (!t.found) continue;
                const r = t.reliability;
                const mat = r >= 0.7 ? highMat : r >= 0.3 ? medMat : lowMat;
                const geo = new THREE.SphereGeometry(8, 8, 6);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(t.tip_x, t.tip_y, t.tip_z);
                tipGroup.add(mesh);

                // CSV seed position (smaller yellow dot)
                const csvGeo = new THREE.SphereGeometry(4, 6, 4);
                const csvMat = new THREE.MeshBasicMaterial({ color: 0xdddd00 });
                const csvMesh = new THREE.Mesh(csvGeo, csvMat);
                csvMesh.position.set(t.csv_x, t.csv_y, t.csv_z);
                tipGroup.add(csvMesh);
            }

            tipGroup.visible = showScanTipsCheck.checked;
            scene.add(tipGroup);
            pmtTips = tipGroup;
            statusEl.textContent += ` ${data.tips.length} tips loaded.`;
        }
        loadPmtTips();

        const pmtOpacitySlider = document.getElementById('pmtOpacity');
        pmtOpacitySlider.addEventListener('input', () => {
            const op = parseFloat(pmtOpacitySlider.value);
            pmtGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    if (child.userData.isPC) {
                        child.material.opacity = 0.35;
                    } else if (child.userData.isPVC) {
                        child.material.opacity = 1.0;
                    } else {
                        child.material.opacity = op;
                    }
                }
            });
            pmtHWGroup.children.forEach(child => {
                if (child.isMesh && child.material) {
                    child.material.opacity = op;
                }
            });
        });

        // ---- View controls (azimuth, elevation, distance) ----
        const viewAzSlider = document.getElementById('viewAz');
        const viewElSlider = document.getElementById('viewEl');
        const viewDistSlider = document.getElementById('viewDist');
        const viewAzVal = document.getElementById('viewAzVal');
        const viewElVal = document.getElementById('viewElVal');
        const viewDistVal = document.getElementById('viewDistVal');

        function sphericalToCartesian(azDeg, polDeg, dist) {
            const az = azDeg * Math.PI / 180;
            const pol = polDeg * Math.PI / 180;
            const x = dist * Math.sin(pol) * Math.cos(az);
            const y = dist * Math.sin(pol) * Math.sin(az);
            const z = dist * Math.cos(pol);
            return new THREE.Vector3(x, y, z);
        }

        function cartesianToSpherical(pos) {
            const dist = pos.length();
            const pol = Math.acos(Math.min(1, Math.max(-1, pos.z / dist)));
            const az = Math.atan2(pos.y, pos.x);
            return { az: (az * 180 / Math.PI + 360) % 360, pol: pol * 180 / Math.PI, dist };
        }

        function updateView() {
            const az = parseFloat(viewAzSlider.value);
            const pol = parseFloat(viewElSlider.value);
            const dist = parseFloat(viewDistSlider.value);
            const t = controls.target;
            const rel = sphericalToCartesian(az, pol, dist);
            camera.position.set(t.x + rel.x, t.y + rel.y, t.z + rel.z);
            controls.update();
        }

        viewAzSlider.addEventListener('input', updateView);
        viewElSlider.addEventListener('input', updateView);
        viewDistSlider.addEventListener('input', updateView);

        // Sync sliders when user drags with mouse
        controls.addEventListener('change', () => {
            const rel = new THREE.Vector3().copy(camera.position).sub(controls.target);
            const s = cartesianToSpherical(rel);
            viewAzSlider.value = s.az;
            viewElSlider.value = s.pol;
            viewDistSlider.value = Math.round(s.dist / 50) * 50;
            viewAzVal.textContent = Math.round(s.az);
            viewElVal.textContent = Math.round(s.pol);
            viewDistVal.textContent = Math.round(s.dist);
        });

        // Initial scene (muon at default position)
        updateScene();

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // ---- PMT click selection ----
        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const adjPanel = document.getElementById('pmt-adjust');
        let meshStartPos = null;
        let selectedStartSliders = null;
        let adjBasis = null;
        let pmtBase = null;  // base position of the selected PMT (no offset)

        function movePMT(idx, pos) {
            for (const g of [pmtGroup, pmtHWGroup]) {
                g.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx === idx) ch.position.copy(pos); });
            }
        }

        function sliderToCartesian(vals, basis) {
            if (basis.mode === 'global') {
                return new THREE.Vector3(vals[0], vals[1], vals[2]);
            }
            const a = vals[0], t = vals[1], v = vals[2];
            return new THREE.Vector3(
                a * basis.eAxial.x + t * basis.eTang.x + v * basis.eVert.x,
                a * basis.eAxial.y + t * basis.eTang.y + v * basis.eVert.y,
                a * basis.eAxial.z + t * basis.eTang.z + v * basis.eVert.z
            );
        }

        function updatePMTPreview() {
            if (selectedIdx < 0 || !adjBasis || !meshStartPos || !selectedStartSliders) return;
            const vals = [
                parseFloat(document.getElementById('adj-0').value),
                parseFloat(document.getElementById('adj-1').value),
                parseFloat(document.getElementById('adj-2').value),
            ];
            const currentCorr = sliderToCartesian(vals, adjBasis);
            const startCorr = sliderToCartesian(selectedStartSliders, adjBasis);
            const delta = currentCorr.clone().sub(startCorr);
            const newPos = meshStartPos.clone().add(delta);
            pmtGroup.children.forEach(ch => {
                if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                    ch.position.copy(newPos);
                }
            });
            pmtHWGroup.children.forEach(ch => {
                if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                    ch.position.copy(newPos);
                }
            });
        }

        function cancelAdjustment() {
            if (selectedMesh && meshStartPos) {
                pmtGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(meshStartPos);
                    }
                });
                pmtHWGroup.children.forEach(ch => {
                    if (ch.userData && ch.userData.pmtIdx === selectedIdx) {
                        ch.position.copy(meshStartPos);
                    }
                });
            }
            deselectPMT();
        }

        function deselectPMT() {
            // Leaving a PMT without Keep undoes its preview
            if (selectedIdx >= 0 && meshStartPos) movePMT(selectedIdx, meshStartPos);
            if (selectedMesh) {
                selectedMesh.material.emissive.setHex(0x000000);
                selectedMesh.material.emissiveIntensity = 0;
            }
            selectedIdx = -1;
            selectedMesh = null;
            adjPanel.style.display = 'none';
            meshStartPos = null;
            selectedStartSliders = null;
            adjBasis = null;
        }

        function selectPMT(idx) {
            if (selectedMesh) {
                selectedMesh.material.emissive.setHex(0x000000);
                selectedMesh.material.emissiveIntensity = 0;
            }
            selectedIdx = idx;
            selectedMesh = null;
            const tubeId = pmtData.detector_nums[idx];
            const type = pmtData.types[idx];
            const dir = pmtData.directions[idx];
            const corr = totalOf('pmts', tubeId);
            document.getElementById('adj-state').textContent = stateText('pmts', tubeId);

            document.getElementById('adj-tube-id').textContent = tubeId;
            document.getElementById('adj-type').textContent = type;
            adjPanel.style.display = 'block';

            // Compute basis and initial slider values
            let v0, v1, v2, labels;
            const eVert = new THREE.Vector3(0, 0, 1);
            const eAxial = new THREE.Vector3(dir[0], dir[1], dir[2]);
            const eTang = new THREE.Vector3().crossVectors(eVert, eAxial);
            const tLen = eTang.length();
            let mode;

            if (tLen < 1e-6) {  // bottom/top — global XYZ
                mode = 'global';
                labels = ['dX', 'dY', 'dZ'];
                v0 = corr[0]; v1 = corr[1]; v2 = corr[2];
            } else {
                mode = 'local';
                eTang.divideScalar(tLen);
                labels = ['Axial', 'Tangential', 'Vertical'];
                v0 = corr[0]*eAxial.x + corr[1]*eAxial.y + corr[2]*eAxial.z;
                v1 = corr[0]*eTang.x  + corr[1]*eTang.y  + corr[2]*eTang.z;
                v2 = corr[2];  // dot with (0,0,1)
            }

            for (let k = 0; k < 3; k++) {
                document.getElementById('adj-label-' + k).textContent = labels[k];
                const sl = document.getElementById('adj-' + k);
                const vals = [v0, v1, v2];
                sl.value = vals[k];
                document.getElementById('adj-val-' + k).textContent = parseFloat(vals[k]).toFixed(1);
            }

            // Highlight the clicked mesh — find a mesh belonging to this PMT
            const allMeshes = [];
            pmtGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx === idx) allMeshes.push(ch); });
            pmtHWGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx === idx) allMeshes.push(ch); });
            if (allMeshes.length > 0) {
                selectedMesh = allMeshes[0];
                selectedMesh.material.emissive.setHex(0x4444ff);
                selectedMesh.material.emissiveIntensity = 0.5;
            }

            // Record start state for preview
            meshStartPos = selectedMesh ? selectedMesh.position.clone() : new THREE.Vector3(0,0,0);
            pmtBase = meshStartPos.clone().sub(new THREE.Vector3(corr[0], corr[1], corr[2]));
            selectedStartSliders = [v0, v1, v2];
            adjBasis = { mode, eAxial: eAxial.clone(), eTang: eTang.clone(), eVert: eVert.clone() };
        }

        renderer.domElement.addEventListener('click', (event) => {
            const rect = renderer.domElement.getBoundingClientRect();
            pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(pointer, camera);

            const targets = [];
            pmtGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx !== undefined) targets.push(ch); });
            pmtHWGroup.children.forEach(ch => { if (ch.userData && ch.userData.pmtIdx !== undefined && ch.visible) targets.push(ch); });

            const pmtHits = raycaster.intersectObjects(targets);
            if (pmtHits.length > 0) {
                deselectPMT();
                deselectSurfboard();
                selectPMT(pmtHits[0].object.userData.pmtIdx);
                return;
            }

            // Check housing clicks
            const housingTargets = housingMeshes.map(hm => hm.box);
            const housingHits = raycaster.intersectObjects(housingTargets);
            if (housingHits.length > 0) {
                deselectPMT();
                deselectSurfboard();
                selectHousing(housingHits[0].object.userData.hIdx);
                return;
            }

            const surfHits = raycaster.intersectObjects(surfMeshes);
            if (surfHits.length > 0) {
                deselectPMT();
                deselectHousing();
                selectSurfboard(surfHits[0].object.userData.sbIdx);
            } else if (selectedSurfboard >= 0) {
                deselectSurfboard();
            } else if (selectedHousing >= 0) {
                deselectHousing();
            } else if (selectedIdx >= 0) {
                deselectPMT();
            }
        });

        // Slider live update
        for (let k = 0; k < 3; k++) {
            document.getElementById('adj-' + k).addEventListener('input', (e) => {
                const idx = parseInt(e.target.id.split('-')[1]);
                document.getElementById('adj-val-' + idx).textContent = parseFloat(e.target.value).toFixed(1);
                updatePMTPreview();
            });
        }

        document.getElementById('adj-save').addEventListener('click', () => {
            if (selectedIdx < 0 || !adjBasis) return;
            const idx = selectedIdx;
            const tubeId = pmtData.detector_nums[idx];
            const vals = [0, 1, 2].map(k => parseFloat(document.getElementById('adj-' + k).value));
            const total = sliderToCartesian(vals, adjBasis);
            const base = pmtBase.clone();
            keepChange('pmts', tubeId, [total.x, total.y, total.z], () => {
                const s = savedOf('pmts', tubeId);
                movePMT(idx, base.clone().add(new THREE.Vector3(s[0], s[1], s[2])));
            });
            meshStartPos = base.clone().add(total);   // kept: deselecting no longer undoes it
            deselectPMT();
        });

        document.getElementById('adj-revert').addEventListener('click', () => {
            if (selectedIdx < 0) return;
            const tubeId = pmtData.detector_nums[selectedIdx];
            const s = savedOf('pmts', tubeId);
            meshStartPos = pmtBase.clone().add(new THREE.Vector3(s[0], s[1], s[2]));
            dropChange('pmts', tubeId);
            deselectPMT();
        });

        document.getElementById('adj-reset').addEventListener('click', () => {
            for (let k = 0; k < 3; k++) {
                document.getElementById('adj-' + k).value = '0';
                document.getElementById('adj-val-' + k).textContent = '0.0';
            }
            updatePMTPreview();
        });

        document.getElementById('adj-cancel').addEventListener('click', cancelAdjustment);

        // ---- Local frames for surfboards and housings: [vertical, radial, tangential] ----
        function localToCart(v, ud) {
            return ud.axisY.clone().multiplyScalar(v[0])
                .add(ud.axisZ.clone().multiplyScalar(v[1]))
                .add(ud.axisX.clone().multiplyScalar(v[2]));
        }
        function cartToLocal(c, ud) {
            const t = new THREE.Vector3(c[0], c[1], c[2]);
            return [t.dot(ud.axisY), t.dot(ud.axisZ), t.dot(ud.axisX)];
        }
        function setSliders(prefix, local) {
            ['Vert', 'Rad', 'Tang'].forEach((axis, k) => {
                const v = Math.round(local[k] * 10) / 10;
                document.getElementById(prefix + axis).value = v;
                document.getElementById(prefix + axis + 'Num').value = v;
            });
        }
        function readSliders(prefix) {
            return ['Vert', 'Rad', 'Tang'].map(axis => parseFloat(document.getElementById(prefix + axis).value) || 0);
        }

        // ---- Surfboard interactive position popup ----
        const surfPopup = document.getElementById('surfboard-popup');
        let surfStart = null;   // position when the surfboard was opened (Cancel goes back here)

        function updateSurfboardPosition() {
            const idx = selectedSurfboard;
            if (idx < 0 || idx >= surfMeshes.length) return;
            const mesh = surfMeshes[idx];
            mesh.position.copy(mesh.userData.base.clone().add(localToCart(readSliders('surf'), mesh.userData)));
        }

        function selectSurfboard(idx) {
            selectedSurfboard = idx;
            const mesh = surfMeshes[idx];
            surfStart = mesh.position.clone();
            document.getElementById('surfPopupIdx').textContent = idx + 1;
            setSliders('surf', cartToLocal(totalOf('surfboards', idx), mesh.userData));
            document.getElementById('surfState').textContent = stateText('surfboards', idx);
            surfPopup.style.display = 'block';
        }

        function deselectSurfboard() {
            if (selectedSurfboard < 0) return;
            if (surfStart) surfMeshes[selectedSurfboard].position.copy(surfStart);  // not kept: undo
            selectedSurfboard = -1;
            surfStart = null;
            surfPopup.style.display = 'none';
        }

        // Bidirectional sync: slider → number, number → slider
        const surfAxes = [
            { slider: 'surfVert', num: 'surfVertNum' },
            { slider: 'surfRad',  num: 'surfRadNum' },
            { slider: 'surfTang', num: 'surfTangNum' },
        ];
        for (const { slider, num } of surfAxes) {
            document.getElementById(slider).addEventListener('input', () => {
                document.getElementById(num).value = document.getElementById(slider).value;
                updateSurfboardPosition();
            });
            document.getElementById(num).addEventListener('input', () => {
                document.getElementById(slider).value = document.getElementById(num).value;
                updateSurfboardPosition();
            });
        }

        document.getElementById('surfReset').addEventListener('click', () => {
            if (selectedSurfboard < 0) return;
            setSliders('surf', [0, 0, 0]);
            updateSurfboardPosition();
        });
        document.getElementById('surfKeep').addEventListener('click', () => {
            const idx = selectedSurfboard;
            if (idx < 0) return;
            const mesh = surfMeshes[idx];
            const total = localToCart(readSliders('surf'), mesh.userData);
            keepChange('surfboards', idx, [total.x, total.y, total.z], () => {
                const s = savedOf('surfboards', idx);
                mesh.position.copy(mesh.userData.base.clone().add(new THREE.Vector3(s[0], s[1], s[2])));
            });
            surfStart = mesh.position.clone();
            deselectSurfboard();
        });
        document.getElementById('surfCancel').addEventListener('click', deselectSurfboard);
        document.getElementById('surfRevert').addEventListener('click', () => {
            const idx = selectedSurfboard;
            if (idx < 0) return;
            const mesh = surfMeshes[idx];
            const s = savedOf('surfboards', idx);
            surfStart = mesh.userData.base.clone().add(new THREE.Vector3(s[0], s[1], s[2]));
            dropChange('surfboards', idx);
            deselectSurfboard();
        });

        document.getElementById('surfPopupClose').addEventListener('click', deselectSurfboard);  // = Cancel

        // ---- LAPPD housing interactive position popup ----
        const housingPopup = document.getElementById('housing-popup');
        const LAPPD_KEY0 = 2000;   // housed-LAPPD detector keys are 2000 + index
        let housingStart = null;   // [box, pc] positions when opened (Cancel goes back here)

        function placeHousing(hm, total) {
            const ud = hm.box.userData;
            hm.box.position.copy(ud.base.clone().add(total));
            hm.pc.position.copy(ud.basePC.clone().add(total));
        }

        function updateHousingPosition() {
            const idx = selectedHousing;
            if (idx < 0 || idx >= housingMeshes.length) return;
            const hm = housingMeshes[idx];
            placeHousing(hm, localToCart(readSliders('h'), hm.box.userData));
        }

        function selectHousing(idx) {
            selectedHousing = idx;
            const hm = housingMeshes[idx];
            housingStart = [hm.box.position.clone(), hm.pc.position.clone()];
            document.getElementById('hPopupIdx').textContent = `${idx + 1} (detector key ${LAPPD_KEY0 + idx})`;
            setSliders('h', cartToLocal(totalOf('lappds', LAPPD_KEY0 + idx), hm.box.userData));
            document.getElementById('hState').textContent = stateText('lappds', LAPPD_KEY0 + idx);
            housingPopup.style.display = 'block';
        }

        function deselectHousing() {
            if (selectedHousing < 0) return;
            if (housingStart) {  // not kept: undo
                const hm = housingMeshes[selectedHousing];
                hm.box.position.copy(housingStart[0]);
                hm.pc.position.copy(housingStart[1]);
            }
            selectedHousing = -1;
            housingStart = null;
            housingPopup.style.display = 'none';
        }

        const housingAxes = [
            { slider: 'hVert', num: 'hVertNum' },
            { slider: 'hRad',  num: 'hRadNum' },
            { slider: 'hTang', num: 'hTangNum' },
        ];
        for (const { slider, num } of housingAxes) {
            document.getElementById(slider).addEventListener('input', () => {
                document.getElementById(num).value = document.getElementById(slider).value;
                updateHousingPosition();
            });
            document.getElementById(num).addEventListener('input', () => {
                document.getElementById(slider).value = document.getElementById(num).value;
                updateHousingPosition();
            });
        }

        document.getElementById('hReset').addEventListener('click', () => {
            if (selectedHousing < 0) return;
            setSliders('h', [0, 0, 0]);
            updateHousingPosition();
        });
        document.getElementById('hKeep').addEventListener('click', () => {
            const idx = selectedHousing;
            if (idx < 0) return;
            const hm = housingMeshes[idx];
            const total = localToCart(readSliders('h'), hm.box.userData);
            keepChange('lappds', LAPPD_KEY0 + idx, [total.x, total.y, total.z], () => {
                const s = savedOf('lappds', LAPPD_KEY0 + idx);
                placeHousing(hm, new THREE.Vector3(s[0], s[1], s[2]));
            });
            housingStart = [hm.box.position.clone(), hm.pc.position.clone()];
            deselectHousing();
        });
        document.getElementById('hCancel').addEventListener('click', deselectHousing);
        document.getElementById('hRevert').addEventListener('click', () => {
            const idx = selectedHousing;
            if (idx < 0) return;
            const hm = housingMeshes[idx];
            const s = savedOf('lappds', LAPPD_KEY0 + idx);
            placeHousing(hm, new THREE.Vector3(s[0], s[1], s[2]));
            housingStart = [hm.box.position.clone(), hm.pc.position.clone()];
            dropChange('lappds', LAPPD_KEY0 + idx);
            deselectHousing();
        });

        document.getElementById('hPopupClose').addEventListener('click', deselectHousing);  // = Cancel

        // For the browser console and the tests: open an element's popup by index
        window.workbench = { selectPMT, selectHousing, selectSurfboard, pending, savedOffsets: () => savedOffsets };

    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        console.error(e);
    }
}

function focusLAPPD() {
    if (!housingMeshes || housingMeshes.length === 0) return;
    const hm = housingMeshes[0];
    const target = hm.box.position.clone();
    const ud = hm.box.userData;
    const normal = ud.axisZ.clone();  // inward-facing normal
    const eye = target.clone().add(normal.clone().multiplyScalar(800));
    camera.position.copy(eye);
    controls.target.copy(target);
    controls.update();
}

document.getElementById('focusLAPPD').addEventListener('click', focusLAPPD);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

init();
</script>

<!-- Surfboard position popup -->
<div id="surfboard-popup">
  <div class="sb-title">
    <h3>Surfboard <span id="surfPopupIdx">0</span></h3>
    <button id="surfPopupClose">✕</button>
  </div>
  <div class="sb-row">
    <label>Vertical</label>
    <input type="number" id="surfVertNum" step="1" value="0">
    <input type="range" id="surfVert" min="-1500" max="1500" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Radial</label>
    <input type="number" id="surfRadNum" step="1" value="0">
    <input type="range" id="surfRad" min="-800" max="800" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Tangential</label>
    <input type="number" id="surfTangNum" step="1" value="0">
    <input type="range" id="surfTang" min="-800" max="800" step="1" value="0">
  </div>
  <div class="adj-state" id="surfState"></div>
  <div class="keep-row">
    <button id="surfKeep" class="btn-keep" title="Keep this position for the session (Save to draft makes it permanent)">Keep</button>
    <button id="surfCancel" class="btn-cancel" title="Undo the slider changes since you opened this Surfboard">Cancel</button>
    <button id="surfRevert" class="btn-revert" title="Back to the position saved in the draft">Revert to saved</button>
    <button id="surfReset" title="Sliders to zero: the base position, no offset">Base</button>
  </div>
</div>

<!-- LAPPD Housing position popup -->
<div id="housing-popup">
  <div class="sb-title">
    <h3>LAPPD Housing <span id="hPopupIdx">0</span></h3>
    <button id="hPopupClose">✕</button>
  </div>
  <div class="sb-row">
    <label>Vertical</label>
    <input type="number" id="hVertNum" step="1" value="0">
    <input type="range" id="hVert" min="-1500" max="1500" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Radial</label>
    <input type="number" id="hRadNum" step="1" value="0">
    <input type="range" id="hRad" min="-800" max="800" step="1" value="0">
  </div>
  <div class="sb-row">
    <label>Tangential</label>
    <input type="number" id="hTangNum" step="1" value="0">
    <input type="range" id="hTang" min="-800" max="800" step="1" value="0">
  </div>
  <div class="adj-state" id="hState"></div>
  <div class="keep-row">
    <button id="hKeep" class="btn-keep" title="Keep this position for the session (Save to draft makes it permanent)">Keep</button>
    <button id="hCancel" class="btn-cancel" title="Undo the slider changes since you opened this LAPPD">Cancel</button>
    <button id="hRevert" class="btn-revert" title="Back to the position saved in the draft">Revert to saved</button>
    <button id="hReset" title="Sliders to zero: the base position, no offset">Base</button>
  </div>
</div>

</body>
</html>
"""
