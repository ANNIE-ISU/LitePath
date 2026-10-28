"""Build the dark-mode README logo from the light one by palette unmixing.

The logo has four flat colours (black lettering, white drop fill, blue drop,
yellow beam).  Each pixel is treated as a blend of those colours: the simplest
blend that reproduces it is found (two colours; three only where two cannot,
e.g. where the beam crosses a letter edge), and the dark-mode colours are
re-blended with the same weights.  This keeps anti-aliased edges smooth.

Dark-mode mapping: black -> light grey, white -> transparent, blue and yellow
unchanged.

Usage:
    python tools/make_dark_logo.py
"""
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
SRC_PATH = IMAGES / "LitePathLogo.png"
OUT_PATH = IMAGES / "LitePathLogo-dark.png"

# black lettering, white drop fill, blue drop, yellow beam
PALETTE = np.array([(0, 0, 0), (255, 255, 255), (72, 159, 247), (252, 241, 113)], float)
DARK_RGB = np.array([(230, 237, 243), (0, 0, 0), (72, 159, 247), (252, 241, 113)], float)
DARK_ALPHA = np.array([1.0, 0.0, 1.0, 1.0])  # white becomes transparent

# Extra fit error charged to three-colour blends so the simpler two-colour
# blend wins whenever it explains the pixel about as well.
TRIPLE_PENALTY = 6.0

src = np.asarray(Image.open(SRC_PATH).convert("RGBA")).astype(float)
rgb = src[..., :3].reshape(-1, 3)
alpha_src = src[..., 3].reshape(-1) / 255.0
n = len(rgb)

best_err = np.full(n, np.inf)
best_w = np.zeros((n, 4))


def consider(idx, w_sub, penalty):
    """Keep the blend (palette indices *idx*, weights *w_sub*) where it fits best."""
    err = np.linalg.norm(rgb - w_sub @ PALETTE[list(idx)], axis=1) + penalty
    take = err < best_err
    w = np.zeros((n, 4))
    w[:, list(idx)] = w_sub
    best_err[take] = err[take]
    best_w[take] = w[take]


# Two-colour blends: project onto the segment between the two colours.
for i, j in itertools.combinations(range(4), 2):
    d = PALETTE[i] - PALETTE[j]
    t = np.clip(((rgb - PALETTE[j]) @ d) / (d @ d), 0, 1)
    consider((i, j), np.stack([t, 1 - t], 1), 0.0)

# Three-colour blends: least squares with a heavily weighted sum-to-1 row.
for idx in itertools.combinations(range(4), 3):
    M = np.vstack([PALETTE[list(idx)].T, 100 * np.ones(3)])
    rhs = np.vstack([rgb.T, 100 * np.ones(n)])
    w = np.linalg.lstsq(M, rhs, rcond=None)[0].T
    w = np.clip(w, 0, None)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    consider(idx, w, TRIPLE_PENALTY)

out_alpha = best_w @ DARK_ALPHA
premul = (best_w * DARK_ALPHA) @ DARK_RGB
colour = np.where(out_alpha[:, None] > 1e-6,
                  premul / np.maximum(out_alpha, 1e-6)[:, None], 0)
out = np.dstack([colour.reshape(src.shape[:2] + (3,)),
                 (out_alpha * alpha_src * 255).reshape(src.shape[:2])[..., None]])
Image.fromarray(np.clip(out.round(), 0, 255).astype(np.uint8)).save(OUT_PATH)
print(f"Wrote {OUT_PATH}")
