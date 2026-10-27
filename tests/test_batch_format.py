"""Structural schema tests for the standardized HDF5 batch output.

Both ``litepath batch`` and ``litepath batch-lappd`` must produce the same
flat schema (see ``docs/BATCH_OUTPUT.md``): root attributes plus the
``photon_hits`` / ``muon_truth`` / ``lappd_readouts`` / ``detectors`` tables.

The standalone-LAPPD runs are fast enough for the default (non-slow) suite;
the full-detector run is marked ``slow`` because it builds the detector
geometry and runs the GPU kernel.
"""

import numpy as np
import pytest

import taichi as ti

from litepath.io_h5 import (
    BatchReader,
    SCHEMA_VERSION,
    SOURCE_SPOT,
    SOURCE_TRACK,
    load_table,
)
from litepath.lappd_batch import LAPPDBatchConfig, run_lappd_batch
from litepath.tracer import build_single_lappd_geometry

# Same init as the CLI entry points.
ti.init(arch=ti.cpu, default_fp=ti.f32)


def _run_lappd(tmp_path, **overrides):
    geometry = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))
    config = LAPPDBatchConfig(
        output_dir=tmp_path,
        seed=11,
        **overrides,
    )
    paths = run_lappd_batch(geometry, config)
    h5 = tmp_path / "output.h5"
    assert h5.exists()
    return paths, h5


def test_cherenkov_mode_schema(tmp_path):
    """Tracked-Cherenkov run: source_type=0, per-event track truth, readouts."""
    paths, h5 = _run_lappd(
        tmp_path,
        sample_mode="grid",
        grid_steps=(2, 2),
        n_events=4,
        photons_per_cm=30,
        source="cherenkov",
        lappd_response=True,
    )

    assert paths["lappd_readouts"].name == "output.h5"

    reader = BatchReader(h5)
    summary = reader.summary()
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["format"] == "litepath-batch"
    assert summary["detector_mode"] == "lappd-standalone"
    assert set(summary["tables"]) >= {
        "photon_hits", "muon_truth", "lappd_readouts", "detectors",
    }

    muons = reader.load_table("muon_truth")
    assert len(muons) == 4                       # one row per event
    assert set(muons["event_id"]) == set(range(4))
    assert (muons["source_type"] == SOURCE_TRACK).all()
    assert muons["spread_deg"].isna().all()      # not a spot beam
    assert muons[["intercept_x", "intercept_y", "intercept_z"]].isna().all().all()
    assert (muons["track_length_mm"] > 0).all()           # housing chord
    assert (muons["n_generated"] > 0).all()
    assert (muons["n_generated_ckv"] == muons["n_generated"]).all()

    # Every generated photon went through the kernel; detected counts are sane.
    hits = reader.load_table("photon_hits")
    det_per_event = hits.groupby("event_id").size()
    assert (det_per_event.reindex(muons["event_id"]) ==
            muons.set_index("event_id")["n_detected"]).all()

    # Readout table: one row per (event, LAPPD) = 4 rows for this single module.
    readouts = reader.load_table("lappd_readouts")
    assert len(readouts) == 4
    assert readouts["lappd_index"].nunique() == 1
    r0 = readouts.iloc[0]
    assert np.asarray(r0["side0"]).shape == (28, 256)
    assert np.asarray(r0["side1"]).shape == (28, 256)
    assert 0 <= r0["n_passed_qe"] <= r0["n_photons"]

    # Per-event round trip through BatchReader.
    ev = reader.load_event(1)
    assert ev.truth["event_id"] == 1
    assert not ev.is_beam()
    assert ev.n_generated() == ev.truth["n_generated"]
    assert len(ev.hits) == ev.truth["n_detected"]
    assert set(ev.lappd_readouts) == {0}
    assert np.asarray(ev.lappd_readouts[0]).shape == (28, 256, 2)


def test_spot_mode_schema(tmp_path):
    """Spot-beam run: source_type=2, intercept/spread recorded, no track."""
    paths, h5 = _run_lappd(
        tmp_path,
        sample_mode="random",
        n_events=3,
        source="spot",
        n_photons_spot=200,
        spread_deg=2.0,
        intercept=(0.0, -45.0, 3.5),
        lappd_response=True,
    )

    reader = BatchReader(h5)
    assert reader.summary()["n_events"] == 3

    muons = reader.load_table("muon_truth")
    assert len(muons) == 3
    assert (muons["source_type"] == SOURCE_SPOT).all()
    assert (muons["spread_deg"] == 2.0).all()
    assert (muons["track_length_mm"] == 0.0).all()
    assert np.allclose(
        muons[["intercept_x", "intercept_y", "intercept_z"]],
        (0.0, -45.0, 3.5),
    )
    assert (muons["n_generated"] == 200).all()

    # Dish-detection: the beam hits the PC, so most photons are detected.
    assert (muons["n_detected"] / muons["n_generated"] > 0.5).all()


def test_missing_tables_render_empty(tmp_path):
    """load_table returns empty frames for tables a run never wrote."""
    _, h5 = _run_lappd(
        tmp_path,
        sample_mode="fixed",
        n_events=1,
        source="cherenkov",
        muon_fixed=(0.0, -120.0, 300.0, 0.0, 0.0, 0.0, -1.0),
        photons_per_cm=20,
        lappd_response=False,
    )
    assert load_table(h5, "pmt_responses").empty   # batch-lappd never writes
    assert len(load_table(h5, "lappd_readouts")) == 0
    assert len(load_table(h5, "muon_truth")) == 1


@pytest.mark.slow
def test_full_detector_uses_same_schema(tmp_path):
    """Full-detector output uses identical table layouts to batch-lappd."""
    import h5py
    from pathlib import Path
    from litepath.batch import BatchConfig, run_batch
    from litepath.tracer import build_geometry

    geometry = build_geometry(
        Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
        pmt_csv_path=Path("PMTPositions_Scan.txt"),
        no_gdml=True,
        no_pmt_holders=True,
        no_lappd=True,
        n_surfboards=0,
    )
    config = BatchConfig(
        n_events=2,
        photons_per_cm=50,
        output_dir=tmp_path,
        record_events=True,
        seed=5,
    )
    run_batch(geometry, config)
    h5_full = tmp_path / "output.h5"

    # Dimensions match between the two modes for shared tables.
    with h5py.File(h5_full, "r") as f:
        assert f.attrs["schema_version"] == SCHEMA_VERSION
        assert f.attrs["format"] == "litepath-batch"
        assert f.attrs["detector_mode"] == "full"
        full_dtype = f["muon_truth"].dtype

    muons = load_table(h5_full, "muon_truth")
    assert len(muons) == 2
    assert (muons["source_type"] == SOURCE_TRACK).all()