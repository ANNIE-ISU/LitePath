"""Output format v2 (docs/BATCH_OUTPUT.md, "Changes in v2").

New provenance attributes, the geometry_elements table, detector identifiers,
light_track_length_mm and tank entry/exit points, --mask-dead, and reading
v1 files.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import taichi as ti

from litepath.geometry.versions import (
    STANDALONE_VERSION, load_build, load_version, standalone_fingerprint,
)
from litepath.hits import N_EXPANDED_COLS
from litepath.io_h5 import BatchReader, load_table, write_detectors

ti.init(arch=ti.cpu, default_fp=ti.f32)

V1_FILE = Path("tests/reference/v1_output.h5")


def _light_build(name="annie-2026-data", **settings):
    """A version without the meshes (fast to build)."""
    version = load_version(name)
    version.settings.update(structure_mesh=False, pmt_meshes=False, **settings)
    version.status, version.tagged = "unnamed", {}
    return load_build(version, quiet=True)


def test_v1_file_still_loads():
    reader = BatchReader(V1_FILE)
    assert reader.summary()["schema_version"] == 1
    truth = reader.load_table("muon_truth")
    assert (truth["light_track_length_mm"] == truth["track_length_mm"]).all()
    assert truth["tank_entry_x"].isna().all() and truth["tank_exit_z"].isna().all()
    det = reader.load_table("detectors")
    pmts = det[det["system_code"] == 0]
    assert sorted(pmts["channel_key"]) == list(range(332, 464))
    assert (det.loc[det["system_code"] != 0, "channel_key"] == -1).all()
    assert reader.load_table("geometry_elements").empty
    ev = reader.load_event(0)
    assert ev.truth["light_track_length_mm"] == ev.truth["track_length_mm"]


def test_detector_identifiers_and_elements(tmp_path):
    g = _light_build().geometry
    path = tmp_path / "out.h5"
    write_detectors(path, g.detectors, g.pmt_info)
    det = load_table(path, "detectors")
    pmts = det[det["system_code"] == 0].set_index("channel_key")
    assert len(pmts) == 132 and (pmts["detector_key"] == pmts.index).all()
    assert pmts.loc[332, "cable_label"] == "E2" and pmts.loc[332, "pmt_type"] == "LUX"
    assert (pmts["status"] == "OFF").sum() == 15 and pmts.loc[333, "status"] == "OFF"
    lappds = det[det["system_code"] != 0]
    assert len(lappds) == 3 and (lappds["pmt_type"] == "LAPPD").all()

    elem = load_table(path, "geometry_elements")
    assert list(elem["element_id"]) == list(det["detector_key"])
    np.testing.assert_allclose(elem[["x", "y", "z"]].to_numpy()[:132], g.pmt_centers, atol=1e-3)
    np.testing.assert_allclose(elem[["dir_x", "dir_y", "dir_z"]].to_numpy()[:132],
                               g.pmt_directions, atol=1e-6)
    assert (elem["size_mm"][:132] > 100).all()


def test_tank_entry_and_exit(tmp_path):
    from litepath.batch import BatchAccumulator
    acc = BatchAccumulator(output_dir=tmp_path)
    no_hits = np.zeros((0, N_EXPANDED_COLS), dtype=np.float32)
    base = dict(pos=(0, -2000, 1940, 0), direc=(0, 1, 0), n_generated=0)
    acc.append_event(0, no_hits, muon_params={
        **base, "track_length": 3.048,
        "tank_entry": (0, -1524, 1940, 0), "tank_exit": (0, 1524, 1940, 0)})
    acc.append_event(1, no_hits, muon_params={**base, "track_length": 0.0})
    acc.close()
    truth = load_table(tmp_path / "output.h5", "muon_truth")
    assert truth.loc[0, "light_track_length_mm"] == pytest.approx(3048.0)
    assert truth.loc[0, "track_length_mm"] == pytest.approx(3048.0)
    assert (truth.loc[0, "tank_entry_y"], truth.loc[0, "tank_exit_y"]) == (-1524.0, 1524.0)
    assert np.isnan(truth.loc[1, ["tank_entry_x", "tank_exit_z"]].astype(float)).all()


def test_standalone_lappd_provenance(tmp_path):
    from litepath.geometry.build import build_single_lappd_geometry
    from litepath.lappd_batch import LAPPDBatchConfig, run_lappd_batch
    geometry = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))
    run_lappd_batch(geometry, LAPPDBatchConfig(output_dir=tmp_path, n_events=1, seed=1,
                                               photons_per_cm=5))
    with h5py.File(tmp_path / "output.h5") as f:
        assert f.attrs["schema_version"] == 2
        assert f.attrs["geometry_version"] == STANDALONE_VERSION
        assert f.attrs["geometry_status"] == "built-in"
        assert f.attrs["geometry_hash"] == standalone_fingerprint(geometry)
        assert len(f["geometry_elements"]) == len(geometry.detectors)
    # the fingerprint follows the module
    moved = build_single_lappd_geometry((0, 0, 10), (0, 0, 1))
    assert standalone_fingerprint(moved) != standalone_fingerprint(geometry)


@pytest.mark.slow
def test_mask_dead_drops_only_the_off_pmts(tmp_path):
    from litepath.batch import BatchConfig, run_batch
    g = _light_build(lappds=False).geometry
    off = tuple(i for i, r in enumerate(g.pmt_info["records"]) if r.status == "OFF")
    assert len(off) == 15

    def run(out, masked):
        cfg = BatchConfig(n_events=1, light_burst=True, burst_n_photons=40000,
                          output_dir=out, seed=3, masked_pmt_indices=masked)
        run_batch(g, cfg)
        return load_table(out / "output.h5", "photon_hits")

    plain = run(tmp_path / "plain", ())
    masked = run(tmp_path / "masked", off)
    assert plain["detector_index"].isin(off).any()
    assert not masked["detector_index"].isin(off).any()
    keep = plain[~plain["detector_index"].isin(off)].reset_index(drop=True)
    assert keep.equals(masked.reset_index(drop=True))
