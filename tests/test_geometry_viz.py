"""The geometry workbench page (litepath geometry-viz)."""

import re

from litepath.geometry_viz import _OPERATIONAL_ROUTES, HTML_PAGE


def test_every_element_the_script_uses_exists():
    ids_html = set(re.findall(r'id="([\w-]+)"', HTML_PAGE))
    ids_js = set(re.findall(r"getElementById\('([\w-]+)'\)", HTML_PAGE))
    assert ids_js <= ids_html, ids_js - ids_html


def test_geometry_tools_kept_and_tracing_left_out():
    for kept in ("adj-save", "surfboard-popup", "housing-popup", "showScan", "showScanTips",
                 "surfKeep", "hKeep", "saveAll", "discardAll", "/api/offsets/save"):
        assert kept in HTML_PAGE, kept
    # no tracing, and nothing saved except through Save to draft
    for gone in ("traceBtn", "showCone", "lappd-readout-popup", "/api/trace",
                 "/api/correction/save", "/api/surfboard/adjust", "/api/lappd/adjust", "saveLappdCorr"):
        assert gone not in HTML_PAGE, gone
    assert set(_OPERATIONAL_ROUTES) == {"/api/trace", "/api/lappd/readout"}


def test_viz_server_page_has_no_geometry_tools():
    from litepath.viz_server import HTML_PAGE as VIZ_PAGE
    ids_html = set(re.findall(r'id="([\w-]+)"', VIZ_PAGE))
    ids_js = set(re.findall(r"getElementById\('([\w-]+)'\)", VIZ_PAGE))
    assert ids_js <= ids_html, ids_js - ids_html
    for kept in ("traceBtn", "showCone", "lappd-readout-popup", "showHW", "focusLAPPD"):
        assert kept in VIZ_PAGE, kept
    for gone in ("pmt-adjust", "surfboard-popup", "housing-popup", "showScan", "saveLappdCorr",
                 "/api/correction/save", "/api/scan_mesh", "/api/pmt_tips"):
        assert gone not in VIZ_PAGE, gone
