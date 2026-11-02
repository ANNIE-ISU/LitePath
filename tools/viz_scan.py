"""Interactive 3D viewer for AllPMTs scan mesh with PMT tip positions.

Usage:
    python3 tools/viz_scan.py
    # Opens http://localhost:8081
"""

from __future__ import annotations

import csv
import json
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

from litepath.geometry.frame import STRUCTURE_TO_FRAME_DEG

# The scan meshes and tip positions are in the GDML structure frame; this
# turn brings them into the LitePath frame (litepath/geometry/frame.py).
DET_ROTATION = STRUCTURE_TO_FRAME_DEG


def rotate_z(points: np.ndarray, angle_deg: float) -> None:
    """Rotate (N,3) array in-place by angle_deg around Z."""
    if angle_deg == 0.0 or points.shape[0] == 0:
        return
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = x * c - y * s
    points[:, 1] = x * s + y * c

SCAN_DIR = Path("scan files by part") / "transformed"
TIPS_PATH = SCAN_DIR / "pmt_tip_positions.csv"
PLACED_PATH = Path(__file__).resolve().parent.parent / "placed_tips.csv"
CORRECTIONS_PATH = Path(__file__).resolve().parent.parent / "corrections.csv"

# ---- Load all available scan meshes into byte caches ----
MESH_CACHE: dict[str, tuple[bytes, bytes, int]] = {}

for _f in sorted(SCAN_DIR.glob("*_verts.npy")):
    _name = _f.stem.replace("_verts", "")
    _tris = SCAN_DIR / f"{_name}_tris.npy"
    if _tris.exists():
        _v = np.load(_f).astype(np.float32)
        if DET_ROTATION != 0.0:
            rotate_z(_v, DET_ROTATION)
        _t = np.load(_tris).astype(np.int32)
        _n = len(_v)
        MESH_CACHE[_name] = (_v.tobytes(), _t.tobytes(), _n)

MESH_NAMES = sorted(MESH_CACHE.keys())
print(f"  Meshes loaded: {len(MESH_CACHE)}")
for _mn in MESH_NAMES:
    _nv = MESH_CACHE[_mn][2]
    print(f"    {_mn}: {_nv} verts")

def _rotate_tip_pt(x: float, y: float) -> tuple[float, float]:
    """Rotate an XY point by the detector rotation."""
    if DET_ROTATION == 0.0:
        return x, y
    theta = math.radians(DET_ROTATION)
    c = math.cos(theta)
    s = math.sin(theta)
    return x * c - y * s, x * s + y * c


def _load_tips():
    """Load, correct, and rotate PMT tip positions."""
    tips = []
    if TIPS_PATH.exists():
        with open(TIPS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                tips.append({
                    "tip_x": float(row["tip_x"]),
                    "tip_y": float(row["tip_y"]),
                    "tip_z": float(row["tip_z"]),
                    "csv_x": float(row["csv_x"]),
                    "csv_y": float(row["csv_y"]),
                    "csv_z": float(row["csv_z"]),
                    "tube_id": int(row["tube_id"]),
                    "panel": int(row["panel"]),
                    "type": row["type"],
                    "reliability": float(row["reliability"]),
                    "offset_mm": float(row.get("offset_mm", 0)),
                    "found": row["found"].strip() in ("1", "true", "True"),
                })
        print(f"  Tips: {len(tips)} loaded")
    else:
        print("  Tip positions NOT FOUND — run find_pmt_tips.py first")
    return tips


def _load_placed():
    """Load placed tips."""
    placed = []
    if PLACED_PATH.exists():
        with open(PLACED_PATH, newline="") as f:
            for row in csv.DictReader(f):
                px, py = _rotate_tip_pt(float(row["x"]), float(row["y"]))
                placed.append({
                    "id": int(row["id"]),
                    "x": px,
                    "y": py,
                    "z": float(row["z"]),
                })
        print(f"  Placed tips: {len(placed)} loaded")
    else:
        print("  Placed tips file NOT FOUND — no placed_tips.csv at project root")
    return placed


def _load_corrections() -> dict[int, tuple[float, float, float]]:
    """Load corrections, rotating them into the detector frame."""
    corr = {}
    if CORRECTIONS_PATH.exists():
        with open(CORRECTIONS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                tid = int(row["tube_id"])
                dx = float(row.get("dx", 0))
                dy = float(row.get("dy", 0))
                dz = float(row.get("dz", 0))
                if DET_ROTATION != 0.0:
                    dx, dy = _rotate_tip_pt(dx, dy)
                corr[tid] = (dx, dy, dz)
        print(f"  Corrections: {len(corr)} loaded")
    return corr


# ---- Load data (order: tips → corrections → rotate → apply) ----
TIPS = _load_tips()
PLACED_TIPS = _load_placed()
CORRECTIONS = _load_corrections()

# Rotate all tip positions into the detector frame
if DET_ROTATION != 0.0:
    for tip in TIPS:
        tip["tip_x"], tip["tip_y"] = _rotate_tip_pt(tip["tip_x"], tip["tip_y"])
        tip["csv_x"], tip["csv_y"] = _rotate_tip_pt(tip["csv_x"], tip["csv_y"])

# Apply rotated corrections to rotated simulated (csv) positions
for tip in TIPS:
    c = CORRECTIONS.get(tip["tube_id"])
    if c:
        tip["csv_x"] += c[0]
        tip["csv_y"] += c[1]
        tip["csv_z"] += c[2]
if CORRECTIONS:
    n_applied = sum(1 for t in TIPS if t['tube_id'] in CORRECTIONS)
    print(f"  Corrections applied to {n_applied} csv positions")

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANNIE Scan Mesh Viewer</title>
<style>
  body { margin:0; overflow:hidden; background:#1a1a2e; color:#eee; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  #status {
    position:absolute; bottom:10px; left:10px; z-index:100;
    background:rgba(0,0,0,0.6); padding:6px 14px; border-radius:4px; font-size:13px;
  }
  #controls {
    position:absolute; top:10px; left:10px; z-index:100;
    background:rgba(20,22,28,0.92); padding:10px 12px; border-radius:8px;
    font-size:12px; width:230px; max-height:calc(100vh - 40px); overflow-y:auto;
  }
  #controls h4 { margin:6px 0 3px; color:#aaa; font-size:11px; text-transform:uppercase; }
  #controls label { display:block; margin:2px 0; cursor:pointer; font-size:11px; }
  #controls label:hover { color:#fff; }
  #controls input[type=checkbox] { accent-color:#4488cc; margin-right:3px; }
  #controls button {
    width:100%; margin:4px 0; padding:4px; font-size:11px;
    background:#4488cc; color:#fff; border:none; border-radius:3px; cursor:pointer;
  }
  #controls button:hover { background:#5599dd; }
  #controls button.active { background:#cc6644; }
  #controls button.active:hover { background:#dd7755; }
  #err {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); z-index:200;
    background:rgba(0,0,0,0.85); color:#ff6666; padding:16px 24px; border-radius:8px; font-size:14px;
    display:none; max-width:500px;
  }
  .sep { border:none; border-top:1px solid rgba(255,255,255,0.08); margin:5px 0; }
  #placeList {
    font-size:10px; color:#aaa; max-height:100px; overflow-y:auto;
    margin:3px 0; padding:2px 4px; background:rgba(0,0,0,0.3); border-radius:3px;
  }
  #placeList .entry { padding:1px 0; }
</style>
</head>
<body>
<div id="controls">
  <h4>Structure</h4>
  <label><input type="checkbox" id="chk_SuperStructure"> Frame</label>
  <label><input type="checkbox" id="chk_BottomLayer"> Bottom</label>
  <label><input type="checkbox" id="chk_TopLayer"> Top</label>
  <hr class="sep">
  <h4>Panels</h4>
  <label style="font-size:11px;color:#888;"><input type="checkbox" id="chk_allStructPanels"> Show all panels</label>
  <div id="structPanelToggles">
    <label><input type="checkbox" id="chk_Panel-1"> Panel-1</label>
    <label><input type="checkbox" id="chk_Panel-2"> Panel-2</label>
    <label><input type="checkbox" id="chk_Panel-3"> Panel-3</label>
    <label><input type="checkbox" id="chk_Panel-4"> Panel-4</label>
    <label><input type="checkbox" id="chk_Panel-5"> Panel-5</label>
    <label><input type="checkbox" id="chk_Panel-6"> Panel-6</label>
    <label><input type="checkbox" id="chk_Panel-7"> Panel-7</label>
    <label><input type="checkbox" id="chk_Panel-8"> Panel-8</label>
  </div>
  <hr class="sep">
  <h4>Panel Overlays</h4>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px 6px;">
    <label><input type="checkbox" id="chk_overlay_1"> Panel 1</label>
    <label><input type="checkbox" id="chk_overlay_2"> Panel 2</label>
    <label><input type="checkbox" id="chk_overlay_3"> Panel 3</label>
    <label><input type="checkbox" id="chk_overlay_4"> Panel 4</label>
    <label><input type="checkbox" id="chk_overlay_5"> Panel 5</label>
    <label><input type="checkbox" id="chk_overlay_6"> Panel 6</label>
    <label><input type="checkbox" id="chk_overlay_7"> Panel 7</label>
    <label><input type="checkbox" id="chk_overlay_8"> Panel 8</label>
  </div>
  <hr class="sep">
  <h4>PMT Groups</h4>
  <label><input type="checkbox" id="chk_AllPMTs" checked> AllPMTs</label>
  <label><input type="checkbox" id="chk_BottomPMTs"> BottomPMTs</label>
  <label><input type="checkbox" id="chk_TopPMTs"> TopPMTs</label>
  <div style="margin-top:2px;font-size:11px;color:#888;">
    <input type="checkbox" id="chk_allPanelPMTs"> Show all panel PMTs
  </div>
  <div id="pmtPanelToggles">
    <label><input type="checkbox" id="chk_Panel-1-PMTs"> Panel-1 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-2-PMTs"> Panel-2 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-3-PMTs"> Panel-3 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-4-PMTs"> Panel-4 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-5-PMTs"> Panel-5 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-6-PMTs"> Panel-6 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-7-PMTs"> Panel-7 PMTs</label>
    <label><input type="checkbox" id="chk_Panel-8-PMTs"> Panel-8 PMTs</label>
  </div>
  <hr class="sep">
  <h4>Overlays</h4>
  <label><input type="checkbox" id="chk_tips" checked> Tips (red)</label>
  <label><input type="checkbox" id="chk_csv" checked> CSV seeds (yellow)</label>
  <label><input type="checkbox" id="chk_placed" checked> Placed tips (green)</label>
  <hr class="sep">
  <h4>Place Tips</h4>
  <button id="placeBtn">Place Mode: OFF</button>
  <div id="placeList"></div>
  <button id="exportBtn">Download CSV</button>
</div>
<div id="status">Loading…</div>
<div id="err"></div>

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

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');

function fail(msg) {
    errEl.style.display = 'block';
    errEl.textContent = 'ERROR: ' + msg;
    statusEl.textContent = 'Failed';
    console.error(msg);
}

// ---- Setup scene ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 20000);
camera.up.set(0, 0, 1);
camera.position.set(3000, -2000, 3000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2000);
controls.update();
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 1.0);
dl.position.set(2000, 3000, 4000);
scene.add(dl);
const grid = new THREE.GridHelper(4000, 20, 0x446688, 0x334466);
grid.position.z = 0;
scene.add(grid);

// Axes
[[1,0,0],[0,1,0],[0,0,1]].forEach((d, i) => {
    scene.add(new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(0,0,0), 800,
        [0xff4444, 0x44ff44, 0x4488ff][i], 30, 15));
});

// ---- Mesh loading ----
async function loadMeshBinary(name) {
    const [vr, tr] = await Promise.all([
        fetch('/api/mesh/' + name + '/verts').then(r => { if (!r.ok) throw Error('verts ' + r.status); return r.arrayBuffer(); }),
        fetch('/api/mesh/' + name + '/tris').then(r => { if (!r.ok) throw Error('tris ' + r.status); return r.arrayBuffer(); }),
    ]);
    const positions = new Float32Array(vr);
    const indices = new Int32Array(tr);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    const pos = geo.getAttribute('position');
    const idx = geo.getIndex();
    const triCount = idx.count / 3;
    const flatPos = new Float32Array(triCount * 9);
    for (let i = 0; i < triCount; i++) {
        for (let j = 0; j < 3; j++) {
            const vi = idx.getX(i * 3 + j);
            flatPos[i * 9 + j * 3] = pos.getX(vi);
            flatPos[i * 9 + j * 3 + 1] = pos.getY(vi);
            flatPos[i * 9 + j * 3 + 2] = pos.getZ(vi);
        }
    }
    const flat = new THREE.BufferGeometry();
    flat.setAttribute('position', new THREE.BufferAttribute(flatPos, 3));
    flat.computeVertexNormals();
    return flat;
}

function getProps(name) {
    if (name === 'AllPMTs') return { color: 0x44aadd, opacity: 0.5, depthWrite: true };
    if (name === 'SuperStructure') return { color: 0x667799, opacity: 0.3, depthWrite: false };
    if (name === 'BottomLayer') return { color: 0x779966, opacity: 0.35, depthWrite: false };
    if (name === 'TopLayer') return { color: 0x9977aa, opacity: 0.35, depthWrite: false };
    if (name === 'BottomPMTs') return { color: 0xcc8844, opacity: 0.35, depthWrite: false };
    if (name === 'TopPMTs') return { color: 0xcc44cc, opacity: 0.35, depthWrite: false };
    if (name.startsWith('Panel-') && name.endsWith('-PMTs')) return { color: 0x88cc44, opacity: 0.35, depthWrite: false };
    return { color: 0x888888, opacity: 0.35, depthWrite: false };
}

// ---- Wire mesh checkboxes ----
const meshGroups = {};
const meshLoaded = {};

function loadGroup(name) {
    if (meshLoaded[name]) return;
    meshLoaded[name] = true;
    const group = meshGroups[name];
    if (!group) return;
    loadMeshBinary(name).then(geo => {
        if (!geo) return;
        const props = getProps(name);
        const mat = new THREE.MeshPhysicalMaterial({
            color: props.color, roughness: 0.4, metalness: 0.05,
            transparent: true, opacity: props.opacity,
            side: THREE.DoubleSide, depthWrite: props.depthWrite,
        });
        group.add(new THREE.Mesh(geo, mat));
    }).catch(e => fail(name + ': ' + e.message));
}

function wireCheckbox(name, defaultVisible) {
    const cb = document.getElementById('chk_' + name);
    if (!cb) return;
    cb.checked = defaultVisible;
    const group = new THREE.Group();
    group.visible = defaultVisible;
    scene.add(group);
    meshGroups[name] = group;
    cb.addEventListener('change', () => {
        if (cb.checked) {
            loadGroup(name);
            group.visible = true;
        } else {
            group.visible = false;
        }
    });
    if (defaultVisible) loadGroup(name);
}

// ---- Tips ----
const tipGroup = new THREE.Group();
tipGroup.visible = true;
scene.add(tipGroup);
const csvGroup = new THREE.Group();
csvGroup.visible = true;
scene.add(csvGroup);

document.getElementById('chk_tips').addEventListener('change', () => {
    tipGroup.visible = document.getElementById('chk_tips').checked;
});
document.getElementById('chk_csv').addEventListener('change', () => {
    csvGroup.visible = document.getElementById('chk_csv').checked;
});
const placedGroup = new THREE.Group();
placedGroup.visible = true;
scene.add(placedGroup);
document.getElementById('chk_placed').addEventListener('change', () => {
    placedGroup.visible = document.getElementById('chk_placed').checked;
});

// ---- Place-tip mode ----
let placeMode = false;
const placedTips = [];
const placeBtn = document.getElementById('placeBtn');
const placeList = document.getElementById('placeList');
const placeGroup = new THREE.Group();
scene.add(placeGroup);

placeBtn.addEventListener('click', () => {
    placeMode = !placeMode;
    placeBtn.textContent = 'Place Mode: ' + (placeMode ? 'ON' : 'OFF');
    placeBtn.className = placeMode ? 'active' : '';
    controls.enableRotate = !placeMode;
});

function addPlacedTip(x, y, z) {
    const id = placedTips.length + 1;
    placedTips.push({ id, x, y, z });
    const geo = new THREE.SphereGeometry(8, 8, 6);
    const mat = new THREE.MeshBasicMaterial({ color: 0x44ff88 });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    placeGroup.add(m);
    const entry = document.createElement('div');
    entry.className = 'entry';
    entry.textContent = '#' + id + ': (' + x.toFixed(1) + ', ' + y.toFixed(1) + ', ' + z.toFixed(1) + ')';
    placeList.appendChild(entry);
    placeList.scrollTop = placeList.scrollHeight;
}

// ---- Raycaster: identify tips OR place new tips ----
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener('click', (event) => {
    if (event.button !== 0) return;
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);

    if (placeMode) {
        // Intersect all loaded meshes
        const meshes = [];
        scene.traverse(child => { if (child.isMesh && child.geometry) meshes.push(child); });
        const hits = raycaster.intersectObjects(meshes);
        if (hits.length > 0) {
            // Snap to nearest vertex of the intersected triangle
            const hit = hits[0];
            const posAttr = hit.object.geometry.getAttribute('position');
            const fi = hit.faceIndex;
            const i0 = fi * 3, i1 = fi * 3 + 1, i2 = fi * 3 + 2;
            const p = hit.point;
            const verts = [
                new THREE.Vector3(posAttr.getX(i0), posAttr.getY(i0), posAttr.getZ(i0)),
                new THREE.Vector3(posAttr.getX(i1), posAttr.getY(i1), posAttr.getZ(i1)),
                new THREE.Vector3(posAttr.getX(i2), posAttr.getY(i2), posAttr.getZ(i2)),
            ];
            const snap = verts.reduce((a, b) => a.distanceTo(p) < b.distanceTo(p) ? a : b);
            addPlacedTip(snap.x, snap.y, snap.z);
        }
    } else {
        // Identify existing tip
        const hits = raycaster.intersectObjects(tipGroup.children);
        if (hits.length > 0) {
            const t = hits[0].object.userData;
            if (t.tip_x !== undefined) {
                statusEl.textContent = 'PMT #' + t.tube_id + ' Panel ' + t.panel + ' ' + t.type
                    + ' | Tip: (' + t.tip_x.toFixed(1) + ', ' + t.tip_y.toFixed(1) + ', ' + t.tip_z.toFixed(1) + ')';
            }
        }
    }
});

// ---- Download CSV ----
document.getElementById('exportBtn').addEventListener('click', () => {
    if (placedTips.length === 0) return;
    let csv = 'id,x,y,z\n';
    for (const t of placedTips) {
        csv += t.id + ',' + t.x.toFixed(3) + ',' + t.y.toFixed(3) + ',' + t.z.toFixed(3) + '\n';
    }
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'placed_tips.csv';
    a.click();
    URL.revokeObjectURL(url);
});

// ---- Init ----
async function init() {
    try {
        const meshResp = await fetch('/api/meshes');
        const meshData = await meshResp.json();
        const names = meshData.meshes || [];

        for (const n of names) {
            wireCheckbox(n, n === 'AllPMTs');
        }

        // Master toggles
        document.getElementById('chk_allStructPanels').addEventListener('change', () => {
            const show = document.getElementById('chk_allStructPanels').checked;
            for (let i = 1; i <= 8; i++) {
                const cb = document.getElementById('chk_Panel-' + i);
                if (cb) { cb.checked = show; cb.dispatchEvent(new Event('change')); }
            }
        });
        document.getElementById('chk_allPanelPMTs').addEventListener('change', () => {
            const show = document.getElementById('chk_allPanelPMTs').checked;
            for (let i = 1; i <= 8; i++) {
                const cb = document.getElementById('chk_Panel-' + i + '-PMTs');
                if (cb) { cb.checked = show; cb.dispatchEvent(new Event('change')); }
            }
        });

        // Panel overlays — toggle both structural + PMT mesh for each panel
        for (let i = 1; i <= 8; i++) {
            const overlayCb = document.getElementById('chk_overlay_' + i);
            if (!overlayCb) continue;
            overlayCb.addEventListener('change', () => {
                const show = overlayCb.checked;
                for (const suffix of [i.toString(), i + '-PMTs']) {
                    const cb = document.getElementById('chk_Panel-' + suffix);
                    if (cb) { cb.checked = show; cb.dispatchEvent(new Event('change')); }
                }
            });
        }

        // Load tips
        const tipResp = await fetch('/api/tips');
        const tipData = await tipResp.json();
        const tips = tipData.tips || [];
        const tipGeo = new THREE.SphereGeometry(10, 10, 8);
        const tipMat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
        const csvGeo = new THREE.SphereGeometry(5, 8, 6);
        const csvMat = new THREE.MeshBasicMaterial({ color: 0xdddd00 });
        let found = 0;
        for (const t of tips) {
            if (!t.found) continue;
            found++;
            const m = new THREE.Mesh(tipGeo, tipMat);
            m.position.set(t.tip_x, t.tip_y, t.tip_z);
            m.userData = t;
            tipGroup.add(m);
            const cm = new THREE.Mesh(csvGeo, csvMat);
            cm.position.set(t.csv_x, t.csv_y, t.csv_z);
            csvGroup.add(cm);
        }
        statusEl.textContent = found + ' tips. Click red to identify, toggle "Place Mode" to add.';

        // Load previously placed tips
        const placedResp = await fetch('/api/placed_tips');
        const placedData = await placedResp.json();
        const placedTipsData = placedData.placed_tips || [];
        const placedGeo = new THREE.SphereGeometry(8, 8, 6);
        const placedMat = new THREE.MeshBasicMaterial({ color: 0x00dd00 });
        for (const p of placedTipsData) {
            const m = new THREE.Mesh(placedGeo, placedMat);
            m.position.set(p.x, p.y, p.z);
            placedGroup.add(m);
        }
        if (placedTipsData.length) {
            statusEl.textContent += ' (' + placedTipsData.length + ' placed)';
        }
    } catch (e) {
        fail('Init: ' + e.message);
    }
}

init();

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data, content_type="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_GET(self):
        if self.path == "/":
            self._send_html()
        elif self.path == "/api/meshes":
            self._send_json({"meshes": MESH_NAMES})
        elif self.path.startswith("/api/mesh/"):
            parts = self.path.split("/")
            if len(parts) != 5:
                self.send_error(400)
                return
            _, _, _, mesh_name, data_type = parts
            entry = MESH_CACHE.get(mesh_name)
            if entry is None:
                self._send_json({"error": f"mesh '{mesh_name}' not found"}, 404)
                return
            verts_bytes, tris_bytes, _ = entry
            if data_type == "verts":
                self._send_binary(verts_bytes)
            elif data_type == "tris":
                self._send_binary(tris_bytes)
            else:
                self.send_error(400)
        elif self.path == "/api/tips":
            self._send_json({"tips": TIPS})
        elif self.path == "/api/placed_tips":
            self._send_json({"placed_tips": PLACED_TIPS})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


def main():
    HOST = "localhost"
    PORT = 8081
    server = HTTPServer((HOST, PORT), Handler)
    print(f"\nScan viz server at http://{HOST}:{PORT}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
