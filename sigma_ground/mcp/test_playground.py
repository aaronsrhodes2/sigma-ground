"""Tests for the stateful simulation-playground tools (conversation mode).

Drives the tool layer directly (no ollama needed). Verifies that a loaded scene
holds MUTABLE state that persists across calls: temperature expands bonds,
pressure compresses them, a magnetic field flips the net particle spin, two
applies compound, and reset restores the pristine matter.
"""
from __future__ import annotations

from sigma_ground.mcp.tools import playground as P


def _val(d):
    return d["value"]


def test_load_reports_matter_and_knobs():
    P.playground_clear("t_load")
    v = _val(P.playground_load("water_molecule", "t_load"))
    assert v["formula"] == "H2O"
    assert round(v["protons"]) == 10 and round(v["neutrons"]) == 8 and round(v["electrons"]) == 10
    assert set(v["tunable_knobs"]) >= {"temperature_k", "pressure_pa", "magnetic_field_t"}
    P.playground_clear("t_load")


def test_temperature_expands_bonds_and_persists():
    P.playground_clear("t_temp")
    P.playground_load("water_molecule", "t_temp")
    pristine = _val(P.playground_inspect("t_temp", "matter"))["operable_fields"]
    assert pristine["mol0.bond0.length_A"] == 0.96  # O-H reference

    diff = _val(P.playground_apply("t_temp", {"temperature_k": 1500.0}))["changed"]
    assert "mol0.bond0.length_A" in diff
    assert diff["mol0.bond0.length_A"]["after"] > diff["mol0.bond0.length_A"]["before"]

    # mutation persists into a later, independent inspect call
    after = _val(P.playground_inspect("t_temp", "matter"))["operable_fields"]
    assert after["mol0.bond0.length_A"] > 0.96
    P.playground_clear("t_temp")


def test_pressure_compresses_bonds():
    P.playground_clear("t_press")
    P.playground_load("water_molecule", "t_press")
    diff = _val(P.playground_apply("t_press", {"pressure_pa": 5.0e9}))["changed"]
    assert "mol0.bond0.length_A" in diff
    assert diff["mol0.bond0.length_A"]["after"] < diff["mol0.bond0.length_A"]["before"]
    P.playground_clear("t_press")


def test_magnetic_field_flips_net_spin():
    P.playground_clear("t_mag")
    P.playground_load("water_molecule", "t_mag")
    before = _val(P.playground_inspect("t_mag", "matter"))["operable_fields"]["net_spin_projection"]
    diff = _val(P.playground_apply("t_mag", {"magnetic_field_t": 10.0}))["changed"]
    assert "net_spin_projection" in diff
    after = diff["net_spin_projection"]["after"]
    assert before != 0 and after == -before  # all spins flipped
    P.playground_clear("t_mag")


def test_applies_compound_then_reset_restores():
    P.playground_clear("t_seq")
    P.playground_load("water_molecule", "t_seq")
    pristine = _val(P.playground_inspect("t_seq", "matter"))["operable_fields"]

    P.playground_apply("t_seq", {"temperature_k": 1500.0})
    P.playground_apply("t_seq", {"pressure_pa": 3.0e9})
    hist = _val(P.playground_inspect("t_seq", "matter"))["applied_history"]
    assert len(hist) == 2  # both mutations recorded (state carried across calls)

    restored = _val(P.playground_reset("t_seq"))
    assert round(restored["electrons"]) == 10
    again = _val(P.playground_inspect("t_seq", "matter"))["operable_fields"]
    assert again == pristine  # pristine matter recovered, history cleared
    assert _val(P.playground_inspect("t_seq", "matter"))["applied_history"] == []
    P.playground_clear("t_seq")


def test_simulate_bridge_runs_against_scene():
    P.playground_clear("t_sim")
    P.playground_load("bronze_cube", "t_sim")
    out = P.playground_simulate("t_sim", "how fast does a 5 cm copper ball hit the ground from 10 km")
    assert isinstance(out, dict)
    assert out["inputs"]["scene_material"] == "bronze_cube"
    # keyword router resolves this offline; value is a speed (m/s) or an honest decline
    assert out["value"] is None or out["value"] > 0
    P.playground_clear("t_sim")


def test_render_returns_ascii():
    P.playground_clear("t_rend")
    P.playground_load("bronze_cube", "t_rend")
    art = _val(P.playground_render("t_rend", "ascii"))
    assert isinstance(art, str) and art.strip()
    assert len(art.splitlines()) >= 8
    P.playground_clear("t_rend")


def test_missing_handle_declines_gracefully():
    P.playground_clear("nope")
    d = P.playground_inspect("nope")
    assert d["value"] is None
    assert "no scene loaded" in (d["notes"] or "")


# ── make: guess when you can, ASK when the variable is decisive ──────────────

def test_make_named_object_fills_standard_size():
    P.playground_clear("mk")
    d = P.playground_make("make a brick", handle="mk")
    assert d["provenance_tag"] == "DERIVED"
    v = d["value"]
    assert v["shape"] == "box" and v["material"] == "fired_clay"
    assert 2.0 < v["mass_kg"] < 4.0          # a standard brick ~2.7 kg
    assert "brick" in d["notes"].lower()
    P.playground_clear("mk")


def test_make_fissile_ball_asks_for_size():
    P.playground_clear("mk")
    d = P.playground_make("a ball of plutonium", handle="mk")
    assert d["provenance_tag"] == "NEEDS-INPUT"   # must ask, not guess
    assert d["value"] is None
    assert d["inputs"]["needed_variable"] == "size"
    assert "critical" in d["inputs"]["reason"].lower()
    assert "mk" not in P._SCENES                   # nothing was made
    P.playground_clear("mk")


def test_make_fissile_with_size_reports_criticality():
    P.playground_clear("mk")
    big = P.playground_make("ball of plutonium", 0.07, handle="mk")["value"]
    assert big["mass_kg"] > big["criticality"]["bare_critical_mass_kg"]
    assert "SUPERCRITICAL" in big["criticality"]["verdict"]
    small = P.playground_make("ball of plutonium", 0.01, handle="mk")["value"]
    assert "subcritical" in small["criticality"]["verdict"]
    P.playground_clear("mk")


def test_make_nondecisive_material_uses_default_with_note():
    P.playground_clear("mk")
    d = P.playground_make("a ball of copper", handle="mk")
    assert d["provenance_tag"] == "DERIVED"        # copper isn't size-decisive
    assert d["value"]["mass_kg"] > 0
    assert "assumed" in d["notes"].lower()         # states the guessed size
    P.playground_clear("mk")


def test_make_unknown_material_asks():
    P.playground_clear("mk")
    d = P.playground_make("a ball of unobtainium", handle="mk")
    assert d["provenance_tag"] == "NEEDS-INPUT"
    assert d["inputs"]["needed_variable"] == "material"
    P.playground_clear("mk")


def test_apply_to_bulk_object_declines():
    P.playground_clear("mk")
    P.playground_make("a lead cube", handle="mk")
    d = P.playground_apply("mk", {"temperature_k": 500.0})
    assert d["value"] is None and d["provenance_tag"] == "SPECULATIVE-PENDING"
    assert "bulk object" in d["notes"]
    P.playground_clear("mk")


def test_request_clarification_tool():
    d = P.request_clarification("size", "How big?", "a fissile sphere is size-decisive")
    assert d["provenance_tag"] == "NEEDS-INPUT"
    assert d["value"] is None
    assert d["inputs"]["needed_variable"] == "size"
