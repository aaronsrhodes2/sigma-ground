"""The railgun first slice: verb, self-checks, recorder (offline half).

A shot is RLC discharge into a moving load (back-EMF), F = 1/2 L' I^2 on the
armature, Joule heating of the armature with library resistivity(T).  These
tests pin: the verb and the recorded bundle agree; the energy ledger closes;
every refusal path really refuses (a mutated law must FAIL the self-checks);
sanity limits come from library numbers; the bundle is deterministic.
"""
import json
import math

import pytest

from sigma_ground import materia
from sigma_ground.field.interface import electronics, mobius
from sigma_ground.field.interface import phase_transition as pt
from sigma_ground.materia import railgun as RG
from sigma_ground.materia.scenarios import rail_shot
from sigma_ground.materia.spec import SimulationSpec, SpecStep, run_spec
from sigma_ground.materia.translator import _auto_bind
from sigma_ground.materia.manifest import VERB_MANIFEST
from sigma_ground.mcp.tools.circuits import rlc_resonant_frequency
from sigma_ground.mcp.tools.simulation import run_simulation
from sigma_ground.radiance import record_rail_shot

# Illustrative INPUTS (design choices and an owner-measured bank), not library data.
HOBBY = dict(cap_F=0.02, volts=450.0, rail_len_m=0.5, rail_radius_m=0.005,
             rail_spacing_m=0.025, armature_material="aluminum",
             armature_mass_kg=1.0e-3, rail_material="copper",
             circuit_R_ohm=0.005, circuit_L0_H=3.0e-6)
LARGE = dict(cap_F=1.0, volts=10_000.0, rail_len_m=10.0, rail_radius_m=0.05,
             rail_spacing_m=0.25, armature_material="aluminum",
             armature_mass_kg=3.0, rail_material="copper",
             circuit_R_ohm=0.002, circuit_L0_H=2.0e-6)
MELTER = dict(HOBBY, armature_mass_kg=1.0e-4)       # thin armature: melts at ~7 cm


# ── verb == bundle ───────────────────────────────────────────────────────
def test_run_simulation_and_recorded_bundle_agree():
    r = run_simulation("rail_shot", dict(HOBBY, T0_K=300.0))   # every input stated: nothing is filled in
    assert r.value is not None and "FAILED" not in r.notes
    out = r.inputs["chain"][0]["outputs"]
    b = record_rail_shot(**HOBBY)
    tr = b["trajectory"]
    last, first = tr["frames"][-1], tr["frames"][0]
    assert r.value == out["muzzle_speed_m_s"]
    assert last["readouts"]["v_m_s"] == pytest.approx(out["muzzle_speed_m_s"], rel=1e-9)
    assert last["bodies"][0]["pos"][0] == pytest.approx(HOBBY["rail_len_m"], rel=1e-9)
    assert max(abs(f["readouts"]["I_A"]) for f in tr["frames"]) == pytest.approx(
        out["peak_current_A"], rel=0.02)           # frames sample the pulse
    assert last["bodies"][0]["temperature_k"] - HOBBY.get("T0_K", 300.0) == \
        pytest.approx(out["delta_T_K"], rel=1e-9)
    assert last["t_sim"] == pytest.approx(out["shot_time_s"], rel=1e-9)
    assert first["bodies"][0]["pos"][0] == 0.0 and first["readouts"]["I_A"] == 0.0
    assert first["readouts"]["V_cap_V"] == HOBBY["volts"]


def test_bundle_contract_and_rate():
    b = record_rail_shot(**HOBBY)
    tr = b["trajectory"]
    ts = [f["t_sim"] for f in tr["frames"]]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert set(tr["frames"][0]["readouts"]) == set(tr["readout_spec"])
    for spec in tr["readout_spec"].values():
        assert spec["unit"] and spec["source"]
    assert tr["suggested_rate"] == pytest.approx(tr["t_end_s"] / 15.0)
    assert tr["suggested_rate"] < 1e-2          # a millisecond shot, slowed down
    assert tr["events"][0]["kind"] == "discharge_start"
    assert tr["events"][-1]["kind"] == "muzzle_exit"
    kinds = {e["kind"] for f in tr["frames"] for e in f.get("events", [])}
    assert {"discharge_start", "peak_current", "muzzle_exit"} <= kinds
    for f in tr["frames"]:
        body = f["bodies"][0]
        assert len(body["pos"]) == 3 and len(body["quat"]) == 4
        assert len(body["rgb"]) == 3 and 0.0 <= body["emit"] <= 1.0
    assert b["scene"]["csg_leaves"][0]["shape"]["type"] == "Box"   # export_gmod-friendly


def test_deterministic_bundle_is_bit_identical():
    a = json.dumps(record_rail_shot(**HOBBY), sort_keys=True)
    b = json.dumps(record_rail_shot(**HOBBY), sort_keys=True)
    assert a == b


# ── the self-checks and the ledger ──────────────────────────────────────
@pytest.mark.parametrize("p", [HOBBY, LARGE, MELTER])
def test_energy_ledger_closes_and_all_checks_pass(p):
    sim = RG.simulate_rail_shot(**p)
    val = RG.check_rail_shot(sim)
    assert val["passed"], val["failed_checks"]
    s = sim["scalars"]
    assert s["ledger_rel"] < 1e-9                      # far inside the stated 1e-3
    E0 = electronics.energy_stored(p["cap_F"], p["volts"])
    assert s["E_cap0_J"] == E0
    tot = (s["E_cap_residual_J"] + s["E_mag_end_J"] + s["KE_end_J"]
           + s["joule_circuit_J"] + s["joule_rails_J"] + s["joule_armature_J"])
    assert tot == pytest.approx(E0, rel=1e-9)
    assert s["speed_end_m_s"] < math.sqrt(2 * E0 / p["armature_mass_kg"])
    assert 0.0 < s["efficiency"] < 1.0


def test_library_numbers_flow_through():
    sim = RG.simulate_rail_shot(**HOBBY)
    mo = sim["model"]
    assert mo["L_prime_H_m"] == mobius.parallel_pair_inductance_per_m(
        HOBBY["rail_radius_m"], HOBBY["rail_spacing_m"])
    assert mo["T_melt_K"] == pt.PHASE_DATA["aluminum"]["T_melt_K"]
    assert mo["rho_armature_T0_ohm_m"] == pytest.approx(
        electronics.resistivity("aluminum", 300.0), rel=1e-12)


def test_owner_inductance_override_is_flagged():
    r = rail_shot(**HOBBY, L_per_m=0.45e-6)
    assert r.validation["passed"]
    assert any("owner" in s.formula for s in r.steps if "inductance" in s.label)


def test_mutated_force_law_fails_the_selfchecks(monkeypatch):
    monkeypatch.setattr(RG, "_force_law", lambda Lp, I: Lp * I * I)   # lost the 1/2
    r = rail_shot(**HOBBY)
    assert not r.validation["passed"]
    assert "energy_ledger" in r.validation["failed_checks"]
    assert "peak_force_vs_library_lorentz" in r.validation["failed_checks"]
    assert "muzzle_speed_m_s" not in r.outputs        # bad speed withheld
    assert r.summary.startswith("[SELF-CHECK FAILED")


def test_mutated_back_emf_fails_the_ledger(monkeypatch):
    monkeypatch.setattr(RG, "_back_emf_voltage", lambda Lp, v, I: 0.0)
    r = rail_shot(**HOBBY)
    assert "energy_ledger" in r.validation["failed_checks"]
    assert "muzzle_speed_m_s" not in r.outputs


def test_recorder_refuses_a_mutated_law(monkeypatch):
    monkeypatch.setattr(RG, "_force_law", lambda Lp, I: Lp * I * I)
    with pytest.raises(ValueError, match="self-checks"):
        record_rail_shot(**HOBBY)


def test_recorder_refuses_disagreement_with_the_verbs_scalar():
    wrong = dict(rail_shot(**dict(HOBBY, volts=300.0)).outputs)     # a different shot
    with pytest.raises(ValueError, match="scalar cross-check"):
        record_rail_shot(expected=wrong, **HOBBY)
    with pytest.raises(ValueError, match="lacks"):
        record_rail_shot(expected={"peak_current_A": 1.0}, **HOBBY)


def test_recorder_refuses_a_law_changed_after_the_verb_ran(monkeypatch):
    good = dict(rail_shot(**HOBBY).outputs)
    # a different (but self-consistent) physics: wrong L' gradient inside the recorder
    real = RG.simulate_rail_shot

    def other(**kw):
        kw["L_per_m"] = 0.9e-6
        return real(**kw)
    monkeypatch.setattr(RG, "simulate_rail_shot", other)
    with pytest.raises(ValueError, match="scalar cross-check"):
        record_rail_shot(expected=good, **HOBBY)


def test_shot_that_never_clears_the_rails_is_refused():
    # time budget ends while the armature is still on the rails: no muzzle exit
    sim = RG.simulate_rail_shot(**HOBBY, t_max_s=2e-4)
    assert sim["scalars"]["reason"] == "t_max" and not sim["scalars"]["exited_rails"]
    val = RG.check_rail_shot(sim)
    assert not val["passed"] and val["failed_checks"] == ["shot_terminated"]


def test_heavy_armature_coasts_out_exactly_when_the_circuit_is_spent():
    sim = RG.simulate_rail_shot(**dict(HOBBY, armature_mass_kg=50.0))
    s = sim["scalars"]
    assert s["reason"] == "end_of_rails" and s["coast_s"] > 0.0
    assert RG.check_rail_shot(sim)["passed"]
    assert s["x_end_m"] == HOBBY["rail_len_m"]


# ── needs-data and bad inputs ────────────────────────────────────────────
def test_missing_owner_data_is_needs_data_not_a_default():
    p = dict(HOBBY)
    del p["circuit_R_ohm"], p["circuit_L0_H"]
    with pytest.raises(RG.NeedsData, match="circuit_R_ohm") as e:
        rail_shot(**p)
    assert "circuit_L0_H" in str(e.value)
    r = run_simulation("rail_shot", dict(p, T0_K=300.0))   # the MCP boundary declines, no crash
    assert r.value is None and "needs data" in r.notes


def test_material_without_library_data_needs_data():
    with pytest.raises(RG.NeedsData, match="supported"):
        rail_shot(**dict(HOBBY, armature_material="unobtainium"))
    with pytest.raises(RG.NeedsData):
        rail_shot(**dict(HOBBY, armature_material="glass"))       # insulator


def test_unphysical_geometry_and_values_rejected():
    with pytest.raises(ValueError, match="overlap"):
        rail_shot(**dict(HOBBY, rail_spacing_m=0.01))
    with pytest.raises(ValueError):
        rail_shot(**dict(HOBBY, armature_mass_kg=-1.0))
    with pytest.raises(ValueError):
        rail_shot(**dict(HOBBY, circuit_R_ohm=-0.1))


# ── sanity limits built from library numbers ─────────────────────────────
def test_no_force_limit_is_the_library_lc_discharge():
    """With L' ~ 0 the shot is a plain RLC discharge: current peaks a quarter
    period after switch-on (library resonance) at sqrt(2 E / L0)."""
    p = dict(HOBBY, armature_mass_kg=50.0, circuit_R_ohm=1e-6)
    f0 = rlc_resonant_frequency(p["circuit_L0_H"], p["cap_F"]).value
    quarter = 1.0 / (4.0 * f0)
    sim = RG.simulate_rail_shot(**p, L_per_m=1e-15, t_max_s=2.0 * quarter)
    i_pk = math.sqrt(2.0 * electronics.energy_stored(p["cap_F"], p["volts"])
                     / p["circuit_L0_H"])
    k = int(sim["I"].argmax())
    assert sim["I"][k] == pytest.approx(i_pk, rel=1e-3)
    assert sim["t"][k] == pytest.approx(quarter, rel=1e-2)


def test_more_energy_means_more_speed_and_heat():
    lo = rail_shot(**dict(HOBBY, volts=300.0)).outputs
    hi = rail_shot(**HOBBY).outputs
    assert hi["muzzle_speed_m_s"] > lo["muzzle_speed_m_s"]
    assert hi["delta_T_K"] > lo["delta_T_K"]
    assert hi["peak_force_N"] > lo["peak_force_N"]


def test_resistivity_feedback_heats_more_than_cold_resistance_would():
    """R(T) rises with T (library Bloch-Gruneisen): the armature Joule heat of the
    melt case exceeds what a frozen cold-resistance armature would dissipate
    for the same current history."""
    sim = RG.simulate_rail_shot(**MELTER)
    mo = sim["model"]
    cold_R = mo["rho_armature_T0_ohm_m"] * mo["gap_m"] / mo["A_armature_m2"]
    import numpy as np
    t, I = sim["t"], sim["I"]
    cold_W = cold_R * float(np.sum(0.5 * (I[1:] ** 2 + I[:-1] ** 2) * np.diff(t)))
    assert sim["scalars"]["joule_armature_J"] > 1.1 * cold_W


def test_hobby_regression_pin():
    """Regression pin (not a truth claim): guards accidental physics edits."""
    s = RG.simulate_rail_shot(**HOBBY)["scalars"]
    assert s["speed_end_m_s"] == pytest.approx(128.949, rel=1e-4)
    assert s["peak_current_A"] == pytest.approx(27593.08, rel=1e-4)


# ── melt end ─────────────────────────────────────────────────────────────
def test_thin_armature_melts_before_the_muzzle_and_glows():
    r = rail_shot(**MELTER)
    assert r.validation["passed"]
    o = r.outputs
    assert o["armature_melted"] and not o["exited_rails"]
    assert o["peak_T_K"] == pytest.approx(pt.PHASE_DATA["aluminum"]["T_melt_K"], rel=1e-6)
    assert "muzzle_speed_m_s" not in o and o["speed_at_failure_m_s"] > 0
    b = record_rail_shot(**MELTER)
    last = b["trajectory"]["frames"][-1]["bodies"][0]
    assert last["rgb"] != [0.0, 0.0, 0.0] and last["emit"] > 0.0
    kinds = [e["kind"] for e in b["trajectory"]["events"]]
    assert "armature_melt" in kinds and "visible_glow" in kinds
    assert "muzzle_exit" not in kinds
    assert last["pos"][0] < MELTER["rail_len_m"]


# ── chaining (bind_from) and impact ──────────────────────────────────────
def test_registered_with_verb_manifest_and_outputs_match():
    assert "rail_shot" in materia.SCENARIOS and "rail_shot" in VERB_MANIFEST
    out = rail_shot(**HOBBY).outputs
    for name in VERB_MANIFEST["rail_shot"]["outputs"]:
        assert name in out, name
    assert VERB_MANIFEST["rail_shot"]["triggers"]


def test_chain_into_supersonic_projectile_via_bind_from():
    spec = SimulationSpec("q", [SpecStep("rail_shot", dict(LARGE)),
                                SpecStep("supersonic_projectile",
                                         {"diameter_m": 0.1})])
    _auto_bind(spec.steps)
    assert spec.steps[1].bindings == {"launch_mach": (0, "muzzle_mach"),
                                      "mass_kg": (0, "projectile_mass_kg")}
    shot, flight = run_spec(spec)
    assert flight.inputs["launch_mach"] == shot.outputs["muzzle_mach"]
    assert flight.inputs["mass_kg"] == LARGE["armature_mass_kg"]
    assert flight.outputs["distance_to_subsonic_m"] > 0.0


def test_chain_breaks_loudly_when_there_is_no_muzzle_exit():
    spec = SimulationSpec("q", [SpecStep("rail_shot", dict(MELTER)),
                                SpecStep("supersonic_projectile", {})])
    _auto_bind(spec.steps)
    with pytest.raises(KeyError, match="muzzle_mach"):
        run_spec(spec)


def test_muzzle_state_into_kinetic_impact():
    out = rail_shot(**LARGE).outputs
    plate = RG.rail_shot_impact(out, "steel_mild", 0.05, radius_m=0.05)
    assert plate["penetrated"] and plate["velocity_out_ms"] < out["muzzle_speed_m_s"]
    thin = RG.rail_shot_impact(rail_shot(**HOBBY).outputs, "steel_mild", 0.05,
                               radius_m=0.005)
    assert not thin["penetrated"]
    with pytest.raises(KeyError):
        RG.rail_shot_impact(rail_shot(**MELTER).outputs, "steel_mild", 0.05, 0.005)
