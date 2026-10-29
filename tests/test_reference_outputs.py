"""End-to-end output must match the stored reference runs.

Guards refactors that should not change physics: every case in
``tests/reference_cases.py`` is re-run through the CLI and compared with
``tests/reference/<case>.npz``.  Full-detector cases are marked ``slow``.
"""

import numpy as np
import pytest

from tests.reference_cases import CASES, REFERENCE_DIR, compare, missing_inputs, run_case


@pytest.mark.parametrize("case", [
    pytest.param(c, id=c.name, marks=[pytest.mark.slow] if c.slow else [])
    for c in CASES
])
def test_matches_reference(case, tmp_path):
    missing = missing_inputs(case)
    if missing:
        pytest.skip(f"input not present: {', '.join(missing)}")
    ref_path = REFERENCE_DIR / f"{case.name}.npz"
    assert ref_path.exists(), (
        f"{ref_path} missing; run `python tools/make_reference_outputs.py {case.name}`")

    new = run_case(case, tmp_path)
    with np.load(ref_path) as ref:
        problems = compare(dict(ref), new)
    assert not problems, f"{case.name} differs from reference:\n" + "\n".join(problems)
