# Linux Installation

This project uses [Taichi](https://taichi-lang.org/) for ray tracing.  Taichi
auto-detects CUDA (NVIDIA GPU), Vulkan, or Metal backends, and falls back to
CPU if none are found — no code changes needed either way.  All launch code
defaults to `ti.cpu`; with a GPU you can switch to `ti.cuda` for roughly
5-10× faster batch simulations (see step 6).

The interactive 3D viewer needs a browser for WebGL rendering but no GPU on
the server for compute.

## Prerequisites

- **Python ≥ 3.11** — check with `python3 --version`.  If older, install
  via your package manager (e.g. `apt install python3.11 python3.11-venv`
  on Debian/Ubuntu) or from [python.org](https://python.org).
- **git** — `apt install git` / `dnf install git`
- **C++ toolchain** (for Taichi's JIT compiler):
  - Debian/Ubuntu: `apt install build-essential cmake`
  - Fedora: `dnf install gcc-c++ cmake`
  - Arch: `pacman -S base-devel cmake`
- **NVIDIA GPU owners only — CUDA toolkit:**
  - Debian/Ubuntu: `sudo apt install nvidia-cuda-toolkit`
  - Fedora: `sudo dnf install cuda-toolkit`
  - Arch: `sudo pacman -S cuda`
  - Or download from <https://developer.nvidia.com/cuda-downloads>
  - Verify with: `nvcc --version`

## 1. Clone and initialise submodules

```bash
git clone https://github.com/ANNIE-ISU/LitePath.git
cd LitePath
git submodule update --init
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip: `pip install --upgrade pip`

## 3. Install system packages (required by lxml)

Debian / Ubuntu:

```bash
sudo apt install libxml2-dev libxslt1-dev
```

Fedora:

```bash
sudo dnf install libxml2-devel libxslt-devel
```

These are only needed if you don't already have them — lxml's pip wheel may
already bundle them on your platform.

## 4. Install Python dependencies

Taichi works on both GPU and CPU from the same pip package.  No separate
CUDA-wheel is needed — the `taichi` pip install is universal.

**x86_64 (Intel/AMD) — simple:**

```bash
pip install -e .
```

This installs everything from `pyproject.toml`, including cadquery, which
distributes a manylinux wheel on x86_64.

**aarch64 (ARM, e.g. Raspberry Pi 4/5, Ampere) — cadquery needs conda:**

CadQuery does not publish a pip wheel for Linux ARM.  Use conda:

```bash
# Install miniforge (no NVIDIA GPU needed → no CUDA)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# (follow the prompts, then restart your shell)

# Create environment with cadquery from conda-forge
conda create -n annie python=3.12
conda activate annie
conda install -c conda-forge cadquery h5py lxml pyyaml numpy
pip install taichi
```

Then install the package itself (still inside the conda env):

```bash
cd LitePath
pip install -e . --no-deps
```

(`--no-deps` avoids re-resolving dependencies already satisfied by conda.)

**Alternative for ARM: skip cadquery entirely.** The geometry versions
(`--geometry`) read a PMT table and never need cadquery.  Just `pip install -e .` will fail on
cadquery, so install the other deps by hand:

```bash
pip install taichi numpy h5py lxml pyyaml
pip install -e . --no-deps
```

## 5. Verify the installation

```bash
python -c "import litepath; print('OK')"
# Optional — which GPU backend LitePath would use:
python -c "from litepath import backend; backend.init('gpu')"
```

You should see `OK` with no errors.  The backend check prints
`Taichi backend: cuda` (NVIDIA GPU), `vulkan`, or `cpu` if no GPU backend
works.  If cadquery is missing,
the package will still import — cadquery is only used when `--step` or
`--manifest` are passed on the command line.

## 6. Run a test simulation

On CPU (any machine):

```bash
python -m litepath batch --geometry annie-2026-data --events 10 --photons-per-cm 50
```

Expected output:

```
Wrote results/output.h5 (NNNN photon rows)
Wrote results/output.h5 (10 muon truth rows)
Done.
```

**With a GPU:** add `--arch gpu` (best available GPU: CUDA, then Vulkan),
or `--arch cuda` / `--arch vulkan` to choose; `export LITEPATH_ARCH=gpu`
makes it the default for every command and script:

```bash
python -m litepath batch --arch gpu --geometry annie-2026-data --events 500 --photons-per-cm 150
```

Every run prints the backend in use (`Taichi backend: cuda`); if the GPU
backend is unavailable, Taichi falls back to the CPU and says so. GPU
results agree with the CPU statistically, not bit for bit. To check a GPU
against the CPU on this machine: `pytest -m gpu`.

### Standard full detector (three surfboards with LAPPDs)

```bash
# Get the PMT positions file (ask the collaboration)
# Then:
python -m litepath batch \
    --geometry annie-2026-data \
    --events 100 --photons-per-cm 150
```

## 7. Interactive visualiser

The 3D viewer runs as a local web server — it serves a Three.js page to
your browser.  The Taichi ray tracing runs on the server (CPU or GPU); the
browser only does WebGL rendering of the scene.  No special server GPU is
needed for the viewer.

```bash
python -m litepath viz-server \
    --geometry annie-2026-data \
    --port 8080
```

Open `http://localhost:8080` in any browser on the same machine (or another
machine on the same network).  Chrome, Firefox, and Edge all support
WebGL for the Three.js rendering.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `taichi.lang.exception.TaichiCompilationError: ...` | Taichi JIT needs C++ tools | `apt install build-essential cmake` |
| `ModuleNotFoundError: No module named 'cadquery'` | ARM Linux + no conda | Install via conda-forge (section 4) or use a `--geometry` version (doesn't need cadquery) |
| `Cannot open self /usr/bin/python3` | System Python too old | Install Python 3.11+ from deadsnakes PPA or python.org |
| `import taichi` crashes or CUDA unavailable | CUDA not installed or incompatible | Use `--arch cpu` (the default) or `--arch vulkan`; Taichi falls back to the CPU if it can't start the requested backend |
| `Taichi backend: cpu (cuda not available)` | No usable GPU backend | Normal on CPU-only machines (about 5× slower than a GPU). For an NVIDIA card, install the CUDA driver |
 | `h5py failed to open file` | File corrupt from killed job | Delete and re-run with smaller `--events` or longer timeout |
