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
    other = version_from_flags({**OLD_FULL_FLAGS, "det_rotation": 0.0})
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
    assert resolve_geometry(build_parser().parse_args(["batch"])).name == "annie-2022-scan"


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
