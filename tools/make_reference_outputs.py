"""Regenerate the stored reference outputs in tests/reference/.

Run this only when a change to LitePath's output is intended; the reference
test exists to catch unintended changes.

Usage:
    python tools/make_reference_outputs.py            # all cases
    python tools/make_reference_outputs.py fit lappd_spot
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.reference_cases import CASES, REFERENCE_DIR, missing_inputs, run_case  # noqa: E402

wanted = set(sys.argv[1:])
unknown = wanted - {c.name for c in CASES}
if unknown:
    sys.exit(f"Unknown case(s): {', '.join(sorted(unknown))}")

for case in CASES:
    if wanted and case.name not in wanted:
        continue
    missing = missing_inputs(case)
    if missing:
        print(f"{case.name}: skipped (missing {', '.join(missing)})")
        continue
    with tempfile.TemporaryDirectory() as tmp:
        arrays = run_case(case, Path(tmp))
    out = REFERENCE_DIR / f"{case.name}.npz"
    np.savez_compressed(out, **arrays)
    print(f"{case.name}: wrote {out.relative_to(REFERENCE_DIR.parents[1])} "
          f"({out.stat().st_size / 1024:.0f} KiB, {len(arrays)} arrays)")
