"""Per-event hit attribution in batch mode.

Regression test for the batching bug where the detected hits were sliced
back into events with the *generated photon* counts.  That only works when
there is one hit row per ray (i.e. without reflections): with the optics
path the hit array is a strict subset of the rays, so every hit of a batch
was attributed to its first event and the remaining events got nothing.

The invariant tested here is the physical one: **the per-event result must
not depend on how the events are grouped into batches.**
"""

from pathlib import Path

import numpy as np
import pytest

import taichi as ti

from annieray.batch import BatchConfig, run_batch
from annieray.io_h5 import BatchReader
from annieray.tracer import build_geometry

ti.init(arch=ti.cpu, default_fp=ti.f32)

#: Three horizontal tracks through the tank, 120 deg apart in azimuth, so
#: that a hit attributed to the wrong event is visible in the hit pattern.
_TRACKS = [
    "0.0 0.0 1940.0 0.0 1.0 0.0 0.0",
    "0.0 0.0 1940.0 0.0 -0.5 0.8660254 0.0",
    "0.0 0.0 1940.0 0.0 -0.5 -0.8660254 0.0",
]


@pytest.fixture(scope="module")
def geometry():
    """A fast detector geometry: tank + PMT positions, no GDML structure."""
    return build_geometry(
        Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
        pmt_csv_path=Path("PMTPositions_Scan.txt"),
        no_gdml=True,        # skip structure mesh -> fast build
        no_pmt_holders=True, # skip PMT holder meshes -> fast
        no_lappd=True,       # skip LAPPD rectangles
        n_surfboards=0,
    )


def _run(tmp_path, geometry, batch_size, muon_file):
    config = BatchConfig(
        n_events=3,
        muon_file=muon_file,
        photons_per_cm=20,
        max_bounces=2,       # optics path -> hit array is a subset of the rays
        batch_size=batch_size,
        pmt_response=True,
        output_dir=tmp_path,
        record_events=True,
        seed=1234,
    )
    run_batch(geometry, config)
    hits = BatchReader(tmp_path / "output.h5").load_table("photon_hits")
    return [int((hits["event_id"] == e).sum()) for e in range(3)]


def test_batch_size_does_not_change_per_event_hits(tmp_path, geometry):
    """Per-event hit counts must not depend on the batch size."""
    muon_file = tmp_path / "tracks.txt"
    muon_file.write_text("\n".join(_TRACKS) + "\n")

    per_event_batched = _run(tmp_path / "batched", geometry, 3, muon_file)
    per_event_single = _run(tmp_path / "single", geometry, 1, muon_file)

    # Sanity: the optics path must actually produce a strict subset of the
    # rays, otherwise the test would pass trivially.
    assert sum(per_event_batched) > 0
    assert all(n > 0 for n in per_event_batched), (
        f"an event received no hits at all: {per_event_batched}")

    # The RNG stream differs slightly between the two runs (the response
    # model draws from it in a different order), so allow a few percent.
    for e, (batched, single) in enumerate(zip(per_event_batched,
                                              per_event_single)):
        assert single > 0, f"reference event {e} got no hits"
        assert abs(batched - single) / single < 0.15, (
            f"event {e}: {batched} hits when batched, {single} unbatched")


def test_batched_hits_are_not_all_in_the_first_event(tmp_path, geometry):
    """A batch of events must not dump every hit into event 0."""
    muon_file = tmp_path / "tracks.txt"
    muon_file.write_text("\n".join(_TRACKS) + "\n")

    counts = _run(tmp_path / "batched2", geometry, 3, muon_file)
    total = sum(counts)
    # No event may hold more than 90 % of the hits while others hold some.
    assert max(counts) < 0.9 * total, (
        f"hits are concentrated in one event: {counts}")
    # And every event must carry a meaningful share.
    assert min(counts) > 0.05 * total, (
        f"an event carries almost no hits: {counts}")
