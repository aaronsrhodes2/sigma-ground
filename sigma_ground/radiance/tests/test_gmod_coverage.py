"""The GMod coverage ledger must account for every Mentat tool and simulation verb.

A ledger that cannot fail is decoration, so this also feeds the validator deliberately broken
inputs and requires it to complain, and checks the scenarios cite bundles that really exist.
"""
from __future__ import annotations

import copy
import fnmatch
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
COV = REPO / "stage" / "gmod" / "coverage"
DATA = REPO / "sigma_ground" / "radiance" / "web" / "data"

sys.path.insert(0, str(COV))
_spec = importlib.util.spec_from_file_location("check_coverage", COV / "check_coverage.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
L = C.L


def test_ledger_is_complete_against_the_frozen_toolbelt():
    assert C.validate() == []


def test_snapshot_size_matches_what_the_live_server_reported():
    snap = C.load_snapshot()
    assert len(snap["tools"]) == 221 and len(snap["verbs"]) == 68
    assert len(L.TOOLS) == 221 and len(L.VERBS) == 68


def test_validator_catches_a_new_unclassified_tool():
    snap = copy.deepcopy(C.load_snapshot())
    snap["tools"].append({"name": "brand_new_physics_tool", "domain": "Physics", "summary": "x"})
    assert any("brand_new_physics_tool" in p and "no ledger entry" in p for p in C.validate(snap))


def test_validator_catches_a_removed_tool_and_a_new_verb():
    snap = copy.deepcopy(C.load_snapshot())
    removed = snap["tools"].pop(0)["name"]
    snap["verbs"].append({"name": "warp_drive", "summary": "x", "slots": []})
    problems = C.validate(snap)
    assert any(removed in p and "not in the Mentat snapshot" in p for p in problems)
    assert any("warp_drive" in p for p in problems)


@pytest.mark.parametrize("breakage,expect", [
    (lambda e: e.update(cls="OUT", kind=None, scenarios=[]), "OUT needs a kind"),
    (lambda e: e.update(cls="ENGINE", scenarios=[]), "needs at least one scenario"),
    (lambda e: e.update(cls="ENGINE", scenarios=["S99"]), "unknown scenario"),
    (lambda e: e.update(cls="NONSENSE"), "unknown class"),
    (lambda e: e.update(cls="ENGINE", kind="ASTRO", scenarios=["S01"]), "only OUT entries carry a kind"),
])
def test_validator_catches_inconsistent_entries(breakage, expect):
    saved = copy.deepcopy(L.TOOLS["free_fall_time"])
    try:
        breakage(L.TOOLS["free_fall_time"])
        assert any(expect in p for p in C.validate()), expect
    finally:
        L.TOOLS["free_fall_time"].clear()
        L.TOOLS["free_fall_time"].update(saved)
    assert C.validate() == []


def test_out_of_scale_entries_name_no_scenario_and_scenarios_all_cover_something():
    for table in (L.TOOLS, L.VERBS):
        for n, e in table.items():
            if e["cls"] == "OUT":
                assert e["kind"] in L.OUT_KINDS and not e["scenarios"], n
    assert all(any(s in e["scenarios"] for t in (L.TOOLS, L.VERBS) for e in t.values()) for s in L.SCENARIOS)


def test_every_bundle_a_scenario_cites_exists():
    names = [p.stem for p in DATA.glob("*.json")]
    for sid, sc in L.SCENARIOS.items():
        for pat in sc["bundles"]:
            assert any(fnmatch.fnmatch(n, pat) for n in names), f"{sid} cites missing bundle {pat!r}"


def test_headline_numbers_are_stable_and_honest():
    s = C.summary()
    # Only ENGINE entries are cross-validated by an independent physics engine.
    engine = s["tools_by_class"]["ENGINE"] + s["verbs_by_class"]["ENGINE"]
    total = s["tools"] + s["verbs"]
    assert 0 < engine < total * 0.2, "ENGINE coverage should stay a small, clearly-labelled minority"
    # the greedy plan reaches full in-scope coverage
    assert s["greedy_order"][-1][3] == 100.0
    # scenarios that are not buildable are excluded from the effort total
    assert L.SCENARIOS["S17"]["effort_days"] is None
