"""Geometry versions: loading, fingerprints, tagging, overlays and the CLI."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from litepath.geometry.versions import (
    STATUS_TAGGED, GeometryError, file_hashes, fingerprint, list_versions, load_build,
    load_pmt_offsets, load_version, tag_version, verify_tagged, version_from_flags,
)

OLD_FULL_FLAGS = {"pmt_csv": "PMTPositions_Scan.txt", "surfboard": 3}


def test_shipped_versions():
    assert {"annie-2022-scan", "annie-2026-data", "annie-2026-data-tuning"} <= set(list_versions())
    for name in ("annie-2022-scan", "annie-2026-data"):
        v = load_version(name)
        assert v.status == STATUS_TAGGED
        hashes = file_hashes(v)
        verify_tagged(v, hashes, fingerprint(v, hashes))  # unchanged since tagging
    assert load_version("annie-2026-data-tuning").status == "draft"


def test_versions_differ_where_intended():
    old, new = load_version("annie-2022-scan"), load_version("annie-2026-data")
    assert old.settings["bottom_spin_deg"] == 0.0 and new.settings["bottom_spin_deg"] == 22.5
    assert old.inputs["pmt_table"].name == "PMTPositions_Scan.txt"
    assert new.inputs["pmt_table"].name == "FullTankPMTGeometry.csv"
    assert fingerprint(old) != fingerprint(new)


def test_old_flags_give_the_same_fingerprint_as_annie_2022_scan():
    flags = version_from_flags({"gdml": "PHASE2_INNER_STRUCTURE_closed.gdml", **OLD_FULL_FLAGS})
    assert fingerprint(flags) == fingerprint(load_version("annie-2022-scan"))
    # A single changed setting changes the fingerprint
    other = version_from_flags({**OLD_FULL_FLAGS, "bottom_spin": 22.5})
    assert fingerprint(other) != fingerprint(flags)


def test_tagged_version_refuses_changes():
    v = load_version("annie-2026-data")
    hashes = file_hashes(v)
    edited = copy.deepcopy(v)
    edited.settings["surfboards"] = 1
    with pytest.raises(GeometryError, match="settings or overlays changed"):
        verify_tagged(edited, hashes, fingerprint(edited, hashes))
    swapped = dict(hashes, **{"PHASE2_INNER_STRUCTURE_closed.gdml": "0" * 64})
    with pytest.raises(GeometryError, match="PHASE2_INNER_STRUCTURE_closed.gdml"):
        verify_tagged(v, swapped, fingerprint(v, swapped))


def test_tag_writes_hashes(tmp_path):
    doc = yaml.safe_load(Path("geometry/versions/annie-2026-data-tuning.yaml").read_text())
    path = tmp_path / "trial.yaml"
    path.write_text(yaml.safe_dump(doc))
    fp = tag_version(path, date="2026-11-01")
    tagged = load_version(path)
    assert tagged.status == STATUS_TAGGED and tagged.tagged["fingerprint"] == fp
    assert "geometry/overlays/metrology-draft.yaml" in tagged.tagged["files"]
    with pytest.raises(GeometryError, match="already tagged"):
        tag_version(path)


def test_metrology_draft_overlay():
    offsets = load_pmt_offsets([Path("geometry/overlays/metrology-draft.yaml")])
    assert len(offsets) == 62
    assert offsets[382] == pytest.approx((0.0, 0.0, -25.4))


def test_overlays_add_up(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("kind: pmt-offsets\noffsets:\n  400: [1, 2, 3]\n  401: [1, 0, 0]\n")
    b.write_text("kind: pmt-offsets\noffsets:\n  400: [1, 0, -3]\n  401: [-1, 0, 0]\n")
    assert load_pmt_offsets([a, b]) == {400: (2.0, 2.0, 0.0)}
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: pmt-tilts\n")
    with pytest.raises(GeometryError, match="unsupported overlay kind"):
        load_pmt_offsets([bad])


def test_build_applies_offsets():
    tuning = load_version("annie-2026-data-tuning")
    tuning.settings.update(structure_mesh=False, pmt_meshes=False)
    base = copy.deepcopy(tuning)
    base.overlays = []
    with_offsets, without = load_build(tuning, quiet=True), load_build(base, quiet=True)
    keys = [d.id for d in with_offsets.geometry.detectors if d.system == "pmt"]
    shift = with_offsets.geometry.pmt_centers - without.geometry.pmt_centers
    expected = np.array([with_offsets.pmt_offsets.get(k, (0, 0, 0)) for k in keys])
    np.testing.assert_allclose(shift, expected, atol=1e-3)
    assert with_offsets.fingerprint != without.fingerprint


def test_cli_rejects_geometry_with_old_flags(capsys):
    from litepath.cli import build_parser, resolve_geometry
    args = build_parser().parse_args(
        ["batch", "--geometry", "annie-2026-data", "--surfboard", "1"])
    with pytest.raises(GeometryError, match="--surfboard"):
        resolve_geometry(args)


def test_cli_old_flags_warn(capsys):
    from litepath.cli import build_parser, resolve_geometry
    args = build_parser().parse_args(["fit", "x.h5", "--surfboard", "3"])
    version = resolve_geometry(args)
    assert "deprecated" in capsys.readouterr().out
    assert version.status == "unnamed"
    assert version.inputs["pmt_table"] is None or version.inputs["pmt_table"].name == "PMTPositions_Scan.txt"


def test_cli_default_geometry():
    from litepath.cli import build_parser, resolve_geometry
    assert resolve_geometry(build_parser().parse_args(["batch"])).name == "annie-2026-data"


def test_cli_switches_derive_from_a_version():
    from litepath.cli import build_parser, resolve_geometry
    args = build_parser().parse_args(
        ["batch", "--geometry", "annie-2026-data", "--no-pmt-holders", "--no-gdml"])
    version = resolve_geometry(args)
    assert version.name == "annie-2026-data+no-gdml+no-pmt-holders"
    assert version.status == "unnamed" and not version.tagged
    assert not version.settings["pmt_meshes"] and not version.settings["structure_mesh"]
    assert version.settings["bottom_spin_deg"] == 22.5
    assert fingerprint(version) != fingerprint(load_version("annie-2026-data"))


def test_cli_switches_keep_a_draft_editable():
    from litepath.cli import build_parser, resolve_geometry
    from litepath.geometry.versions import editable_overlay
    version = resolve_geometry(build_parser().parse_args(
        ["viz-server", "--geometry", "annie-2026-data-tuning", "--no-gdml"]))
    assert version.status == "draft"
    assert editable_overlay(version).name == "metrology-draft.yaml"
    tagged = resolve_geometry(build_parser().parse_args(
        ["viz-server", "--geometry", "annie-2026-data", "--no-gdml"]))
    assert editable_overlay(tagged) is None


def test_overlay_write_round_trip(tmp_path):
    from litepath.geometry.versions import read_pmt_offsets, write_pmt_offsets
    path = tmp_path / "o.yaml"
    write_pmt_offsets(path, {401: (1.5, 0.0, -2.0), 402: (0.0, 0.0, 0.0)},
                      header={"description": "test"})
    assert read_pmt_offsets(path) == {401: (1.5, 0.0, -2.0)}
    write_pmt_offsets(path, {401: (0.0, 0.0, 0.0)})
    assert read_pmt_offsets(path) == {}
    assert "description: test" in path.read_text()


def test_element_offsets_round_trip(tmp_path):
    from litepath.geometry.versions import read_offsets, write_offsets, write_pmt_offsets
    path = tmp_path / "e.yaml"
    write_offsets(path, {"pmts": {401: (1.0, 0.0, 0.0)}, "lappds": {2001: (0.0, 0.0, 50.0)},
                         "surfboards": {1: (0.0, -5.0, 0.0)}},
                  header={"description": "test"}, history_entry={"date": "2026-11-02", "lappds": [2001]})
    assert read_offsets(path) == {"pmts": {401: (1.0, 0.0, 0.0)}, "lappds": {2001: (0.0, 0.0, 50.0)},
                                  "surfboards": {1: (0.0, -5.0, 0.0)}}
    # the viz server's PMT save keeps the other sections and the history
    write_pmt_offsets(path, {402: (0.0, 2.0, 0.0)})
    doc = yaml.safe_load(path.read_text())
    assert doc["kind"] == "element-offsets" and doc["description"] == "test"
    assert read_offsets(path)["pmts"] == {402: (0.0, 2.0, 0.0)}
    assert read_offsets(path)["lappds"] == {2001: (0.0, 0.0, 50.0)}
    assert doc["history"] == [{"date": "2026-11-02", "lappds": [2001]}]


def _draft_with(tmp_path, lappds=None, surfboards=None):
    from litepath.geometry.versions import read_offsets, write_offsets
    overlay = tmp_path / "o.yaml"
    write_offsets(overlay, {**read_offsets(Path("geometry/overlays/metrology-draft.yaml")),
                            "lappds": lappds or {}, "surfboards": surfboards or {}}, header={})
    v = load_version("annie-2026-data-tuning")
    v.overlays = [overlay]
    v.settings.update(structure_mesh=False, pmt_meshes=False)
    return v


def test_build_applies_lappd_and_surfboard_offsets(tmp_path):
    base = load_build(_draft_with(tmp_path), quiet=True)
    moved = load_build(_draft_with(tmp_path, {2001: (0.0, 0.0, 50.0)}, {0: (10.0, 0.0, 0.0)}), quiet=True)
    g0, g1 = base.geometry, moved.geometry
    np.testing.assert_allclose(g1.lappd_housing_data[:, :3] - g0.lappd_housing_data[:, :3],
                               [[0, 0, 0], [0, 0, 50], [0, 0, 0]], atol=1e-3)
    np.testing.assert_allclose(g1.annie_lappd_data[1, :3] - g0.annie_lappd_data[1, :3], [0, 0, 50], atol=1e-3)
    np.testing.assert_allclose(g1.surfboard_data[0, :3] - g0.surfboard_data[0, :3], [10, 0, 0], atol=1e-3)
    pos = {d.id: d.position for d in g1.detectors}
    pos0 = {d.id: d.position for d in g0.detectors}
    np.testing.assert_allclose(np.subtract(pos[2001], pos0[2001]), [0, 0, 50], atol=1e-3)
    assert moved.fingerprint != base.fingerprint
    # the build records the version's fingerprint
    assert base.fingerprint == fingerprint(_draft_with(tmp_path))
    with pytest.raises(ValueError, match="LAPPD offset for 2005"):
        load_build(_draft_with(tmp_path, {2005: (0.0, 0.0, 1.0)}), quiet=True)
