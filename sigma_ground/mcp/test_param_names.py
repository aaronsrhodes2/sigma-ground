"""Tests for mcp/param_names.py and its use in the contract: one vocabulary for parameter names, the old names still accepted (2026-10-15)."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp import contract as C
from sigma_ground.mcp import param_names as PN

REPO = Path(__file__).resolve().parents[2]


# ── the map ────────────────────────────────────────────────────────────────────────────────────────

def test_a_rename_applies_when_it_is_unambiguous():
    assert PN.rename_map("anything", ["material_key", "strain"]) == {"material_key": "material"}
    assert PN.rename_map("t", ["hot_temperature_k", "cold_temperature_k"]) == {"hot_temperature_k": "t_hot_k", "cold_temperature_k": "t_cold_k"}
    assert PN.rename_map("t", ["material_1", "material_2", "pressure_pa"]) == {"material_1": "material_a", "material_2": "material_b"}


def test_a_rename_is_skipped_when_it_would_collide():
    assert PN.rename_map("t", ["material", "material_key"]) == {}                          # the new name is already a parameter
    assert PN.rename_map("t", ["material_key", "metal_key"]) == {}                         # two spellings of one idea in one tool: two roles, keep both
    assert PN.rename_map("wetting_analysis", ["solid_key", "liquid_key"]) == {}            # never renamed: distinct roles


def test_tool_specific_renames_only_apply_to_their_tool():
    assert PN.rename_map("semiconductor_junction_analysis", ["sc_key", "area_m2"]) == {"sc_key": "semiconductor"}
    assert PN.rename_map("some_other_tool", ["sc_key"]) == {}
    assert PN.rename_map("natural_convection_analysis", ["gas_key", "gas_key_2"]) == {"gas_key": "material_a", "gas_key_2": "material_b"}


def test_canonical_args_translates_values_untouched_and_the_canonical_wins():
    params = ["material_key", "strain"]
    assert PN.canonical_args("t", {"material_key": "steel", "strain": 0.1}, params) == {"material": "steel", "strain": 0.1}
    assert PN.canonical_args("t", {"material": "steel", "strain": 0.1}, params) == {"material": "steel", "strain": 0.1}
    assert PN.canonical_args("t", {"material_key": "copper", "material": "steel"}, params) == {"material": "steel"}
    assert PN.canonical_args("t", {"unknown": 1}, params) == {"unknown": 1}


def test_accepted_old_names_are_worked_out_from_the_schema_alone():
    assert {"material_key", "metal_key"} <= PN.accepted_old_names("t", ["material", "strain"])
    assert PN.accepted_old_names("t", ["strain"]) == set()
    assert "material_key" not in PN.accepted_old_names("t", ["material", "material_key"])


def test_every_target_is_a_name_the_tables_agree_on():
    for old, new in {**PN.RENAMES, **{k: v for d in PN.TOOL_RENAMES.values() for k, v in d.items()}}.items():
        assert old != new and new not in PN.RENAMES, f"{old} -> {new}: a canonical name must not itself be renamed"
        assert re.fullmatch(r"[a-z][a-z0-9_]*", new), f"{new} is not a plain lower-case name"


# ── through the server ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mentat_contract_census"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.build_server()


def test_the_schema_shows_canonical_names_only(server):
    mod, srv = server
    assert C.ALIASES, "no tool was renamed"
    for tool, renamed in C.ALIASES.items():
        props = set(srv._tool_manager.get_tool(tool).parameters["properties"])
        assert set(renamed.values()) <= props, tool
        assert not (set(renamed) & props), f"{tool} still shows an old name"


def test_old_names_canonical_names_and_a_mix_give_the_same_answer(server):
    mod, srv = server
    call = lambda args: mod.call(srv, "thermal_contact_analysis", args)["result"]
    old = call({"material_1": "copper", "material_2": "aluminum"})
    new = call({"material_a": "copper", "material_b": "aluminum"})
    mixed = call({"material_1": "copper", "material_b": "aluminum"})
    assert old["value"] == new["value"] == mixed["value"] and old["value"]["contact_conductance_W_m2K"] > 0
    assert list(new["inputs"])[:2] == ["material_a", "material_b"] == list(old["inputs"])[:2]        # the echo always uses the canonical names


def test_when_both_spellings_are_given_the_canonical_one_wins(server):
    mod, srv = server
    r = mod.call(srv, "thermal_contact_analysis", {"material_1": "copper", "material_a": "aluminum", "material_b": "copper"})["result"]
    assert r["inputs"]["material_a"] == "aluminum"


def test_a_unit_param_that_was_renamed_still_takes_a_quantity_and_converts(server):
    mod, srv = server
    r = mod.call(srv, "carnot_efficiency", {"t_hot_k": "600 K", "t_cold_k": "27 degC"})["result"]
    assert r["value"] == pytest.approx(1 - 300.15 / 600, rel=1e-9)                # 27 degC is 300.15 K: an offset unit, converted through a renamed parameter
    thermo = srv._tool_manager.get_tool("thermoelectric_generator_analysis").parameters["properties"]["t_hot_k"]
    assert "K" in thermo.get("description", "")                                   # documented under its canonical name


def test_descriptions_no_longer_mention_the_old_names(server):
    mod, srv = server
    for tool, renamed in C.ALIASES.items():
        desc = srv._tool_manager.get_tool(tool).description
        for old in renamed:
            assert not re.search(rf"\b{re.escape(old)}\b", desc), f"{tool}'s description still says {old}"


def test_the_vocabulary_shrank(server):
    mod, srv = server
    names = {p for t in srv._tool_manager.list_tools() for p in t.parameters.get("properties", {})}
    # 291 since 2026-10-26: body_position adds ONE genuinely new concept, `when` (the time asked about); its other names (body_name, central_body) are existing ones.
    # 293 since 2026-10-30: the celestial tools add TWO genuinely new concepts, `step_s` (an integration step) and `other_body` (the second body of a synodic period or a transfer); their other names are existing ones.
    assert len(names) <= 293
