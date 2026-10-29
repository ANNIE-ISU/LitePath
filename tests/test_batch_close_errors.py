"""HDF5 close failures must propagate out of the batch runners.

Both runners used to build their return value in a ``finally`` block that
ended in ``return paths``, which swallowed any exception raised by
``BatchAccumulator.close()`` (GitHub issue #8).
"""

from pathlib import Path

import pytest

import taichi as ti

from litepath.batch import BatchAccumulator, BatchConfig, run_batch
from litepath.lappd_batch import LAPPDBatchConfig, run_lappd_batch
from litepath.geometry.build import build_geometry, build_single_lappd_geometry

ti.init(arch=ti.cpu, default_fp=ti.f32)


@pytest.fixture
def failing_close(monkeypatch):
    real_close = BatchAccumulator.close

    def close(self):
        real_close(self)
        raise OSError("simulated HDF5 write failure")

    monkeypatch.setattr(BatchAccumulator, "close", close)


def test_run_lappd_batch_propagates_close_error(tmp_path, failing_close):
    geometry = build_single_lappd_geometry((0, 0, 0), (0, 0, 1))
    config = LAPPDBatchConfig(
        output_dir=tmp_path, seed=11, sample_mode="grid", grid_steps=(1, 1),
        n_events=1, photons_per_cm=10, source="cherenkov",
    )
    with pytest.raises(OSError, match="simulated HDF5 write failure"):
        run_lappd_batch(geometry, config)


def test_run_batch_propagates_close_error(tmp_path, failing_close):
    geometry = build_geometry(
        Path("PHASE2_INNER_STRUCTURE_closed.gdml"),
        pmt_csv_path=Path("PMTPositions_Scan.txt"),
        no_gdml=True, no_pmt_holders=True, no_lappd=True, n_surfboards=0,
    )
    muon_file = tmp_path / "tracks.txt"
    muon_file.write_text("0.0 0.0 1940.0 0.0 1.0 0.0 0.0\n")
    config = BatchConfig(
        n_events=1, muon_file=muon_file, photons_per_cm=10,
        output_dir=tmp_path, record_events=True, seed=1,
    )
    with pytest.raises(OSError, match="simulated HDF5 write failure"):
        run_batch(geometry, config)
