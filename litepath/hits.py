"""Layout of the hit array returned by the ray-tracing engine.

Column indices, component IDs, detector-system codes and photon-source
codes shared by the engine, the response models and the I/O layer.
"""

# ---- Component IDs (written to hits[:, 8] = HCID) ----
CID_NO_HIT = 0           # photon escaped or was absorbed
CID_INNER_STRUCTURE = 1  # hit the GDPM structure mesh (steel, acrylic, etc.)
CID_PMT = 2              # hit a PMT photocathode sphere
CID_LAPPD = 3            # hit a LAPPD photocathode rectangle
CID_TANK_WALL = 4        # hit the invisible outer tank cylinder

# ---- Detector system codes (written to hits[:, 10] = HDS) ----
DET_SYS_NONE = -1          # not a detector hit (structure, tank, or none)
DET_SYS_PMT = 0            # hit recorded against a PMT detector
DET_SYS_LAPPD_DEFAULT = 1  # hit recorded against a default bare-rectangle LAPPD
DET_SYS_LAPPD_ANNIE = 2    # hit recorded against the ANNIE housed LAPPD

# ---- Column indices for the 13-column hit array produced by the kernel ----
# These names are used throughout the code to avoid magic-number indexing.
HI = 0   # hit_flag      (1 = hit, 0 = miss)
HT = 1   # t             (path length from origin to hit, mm)
HX = 2   # hit x         (detector-frame X coordinate, mm)
HY = 3   # hit y         (detector-frame Y coordinate, mm)
HZ = 4   # hit z         (detector-frame Z coordinate, mm)
HNX = 5  # hit normal x  (outward-facing normal at hit point)
HNY = 6  # hit normal y
HNZ = 7  # hit normal z
HCID = 8 # component_id  (which geometry element was hit: 1-4 above)
HDI = 9  # detector_index(which detector in the registry; -1 = not a detector)
HDS = 10 # detector_system(which detector type: 0 = PMT, 1 = LAPPD default, 2 = LAPPD annie)
HLU = 11 # local_u      (PMT: polar angle θ from direction in rad; LAPPD: position along strips in mm)
HLV = 12 # local_v      (PMT: azimuthal angle φ in rad; LAPPD: position across strips in mm)
HMAT = 13 # material_id  (MaterialID enum value, e.g. GLASS=1, PHOTOCATHODE=2, ...)

N_HIT_COLS = 14

# Column indices in the expanded (post-kernel) hit array.
# The kernel produces N_HIT_COLS columns (0..N_HIT_COLS-1).
# trace_cherenkov/trace_muon_light append the columns below.
H_ARRIVAL = N_HIT_COLS      # 14 — arrival_time (ns)
H_WAVELEN = N_HIT_COLS + 1  # 15 — wavelength (nm)
H_BOUNCE  = N_HIT_COLS + 2  # 16 — number of surface reflections
H_SOURCE  = N_HIT_COLS + 3  # 17 — photon source type (0=Cherenkov, 1=scintillation)
N_EXPANDED_COLS = N_HIT_COLS + 4  # 18

# ---- Photon source types (written to hits[:, 17] = H_SOURCE) ----
SOURCE_CKV = 0  # Cherenkov photon
SOURCE_SCI = 1  # Scintillation photon
