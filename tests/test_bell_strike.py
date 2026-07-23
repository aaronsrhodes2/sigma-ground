"""record_bell_strike gates: real contact detection via the engine's own
pairwise solver (not scripted), ring frequency from the cited formula, and
a valid, correctly-sampled audio artifact."""
import math
import os
import wave

import pytest

from sigma_ground.radiance.trajectory import record_bell_strike
from sigma_ground.field.interface.acoustics import ring_frequency


def test_stone_actually_strikes_the_bell():
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    v = bundle["trajectory"]["validation"]
    assert v["struck"] is True
    assert v["struck_at_s"] is not None
    assert v["v_impact_m_s"] > 0
    assert v["impact_energy_j"] > 0


def test_ring_frequency_matches_the_cited_formula_directly():
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    expected = ring_frequency("iron", 0.15)
    assert bundle["trajectory"]["validation"]["ring_frequency_hz"] == pytest.approx(expected)
    assert bundle["acoustics"]["ring_frequency_hz"] == pytest.approx(expected)


def test_audio_artifact_is_a_valid_wav_above_the_nyquist_safe_margin():
    """Regression gate for the aliasing bug found live: a small/stiff bell's
    ring_frequency can exceed 22050 Hz's default Nyquist -- the WAV's own
    sample rate must always comfortably exceed 2x the actual tone."""
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    path = bundle["audio_path"]
    assert os.path.exists(path)
    f_ring = bundle["acoustics"]["ring_frequency_hz"]
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        assert w.getnchannels() == 1
        assert w.getnframes() > 0
    assert sr > 2.2 * f_ring


def test_bigger_bell_rings_lower():
    small = record_bell_strike("iron", bell_diameter_m=0.10)
    big = record_bell_strike("iron", bell_diameter_m=0.40)
    assert big["acoustics"]["ring_frequency_hz"] < small["acoustics"]["ring_frequency_hz"]


def test_bell_hangs_still_before_the_strike_then_swings_after():
    """The bell now hangs from a REAL modeled suspension chain (rigid
    links + RevoluteJoint hinges, not a bare world-weld at its own CoM) --
    it should stay essentially fixed BEFORE the stone strikes (a chain
    hanging plumb has zero net torque at every hinge), then visibly swing
    AFTER impact, which the old rigid weld could never do at all (it was
    welded directly to its own center of mass -- literally rigid, zero
    DOF). A Gauss-Seidel joint-chain solver does not hold a 4-deep hinge
    chain's anchors to the bit-for-bit stillness a single WeldJoint gave
    for free (tests/test_joints.py's own double-pendulum gate names this
    "the GS chain-coupling limit" for even a 2-joint chain) -- the
    pre-strike tolerance here is loosened accordingly, but still tight
    relative to the bell's own 150 mm scale."""
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    frames = bundle["trajectory"]["frames"]
    struck_at = bundle["trajectory"]["validation"]["struck_at_s"]
    assert struck_at is not None
    p0 = frames[0]["bodies"][0]["pos"]
    pre = [f for f in frames if f["t_sim"] < struck_at]
    assert pre
    for f in pre:
        d = math.dist(p0, f["bodies"][0]["pos"])
        assert d < 2e-3, f"bell moved {d} m before being struck"
    post = [f for f in frames if f["t_sim"] > struck_at + 0.1]
    assert post, "no frames recorded after the strike settled"
    moved = max(math.dist(p0, f["bodies"][0]["pos"]) for f in post)
    assert moved > 1e-4, "bell never swung after being struck"


def test_propagation_report_present_and_consistent():
    bundle = record_bell_strike("iron", bell_diameter_m=0.15, observer_distance_m=100.0)
    from sigma_ground.field.interface.atmosphere import speed_of_sound
    v = speed_of_sound(288.15)
    assert bundle["acoustics"]["arrival_time_s"] == pytest.approx(100.0 / v, rel=1e-6)


def test_provenance_cites_the_solver_not_a_scripted_impact():
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    src = bundle["scene"]["source"]
    assert "pairwise contact solver" in src
    assert "SIMPLIFIED_MODEL" in src
    assert bundle["scene"]["choices"]
    # the bell now hangs from a REAL modeled chain (rigid links + hinges,
    # dynamics/joints.py) -- no support Plug is needed any more (Phase B:
    # real structure retires the hand-wave disclosure it used to require;
    # the top link's world-anchor weld is a standard rigid-body
    # idealization, not an unmodeled force/holder)
    assert bundle["scene"]["plugs"] == []
    assert "suspension chain" in src


def test_dissipated_energy_matches_the_real_impact_partition_model():
    """The WAV's loudness input must be the REAL dissipated energy from
    field.interface.impact's Johnson-Thornton energy partition (the same
    model record_fall's own bounce restitution already uses), not raw
    kinetic energy -- part of the impact KE stays as the stone's rebound
    motion and never becomes sound at all."""
    from sigma_ground.field.interface.impact import impact_energy_partition
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    v = bundle["trajectory"]["validation"]
    assert v["dissipated_energy_j"] > 0
    assert v["dissipated_energy_j"] <= v["impact_energy_j"]     # e<=1 always
    # re-derive independently from the reported v_impact/E_total to confirm
    # the SAME formula was actually used, not a different invented curve
    stone_mass_kg = 2.0 * v["impact_energy_j"] / v["v_impact_m_s"] ** 2
    expected = impact_energy_partition("iron", velocity=v["v_impact_m_s"],
                                       mass_kg=stone_mass_kg)
    assert v["dissipated_energy_j"] == pytest.approx(
        expected["E_dissipated_J"], rel=1e-6)


def test_impact_click_frequency_is_real_and_distinct_from_ring_frequency():
    """The Hertzian impact "click" (field.interface.impact.
    impact_sound_frequency) is a real, cited, different number from the
    bell's sustained ring_frequency -- both should be reported, not
    conflated."""
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    v = bundle["trajectory"]["validation"]
    assert v["impact_click_frequency_hz"] > 0
    assert v["contact_duration_s"] > 0
    assert v["impact_click_frequency_hz"] != pytest.approx(
        v["ring_frequency_hz"], rel=0.01)
    assert bundle["acoustics"]["impact_click_frequency_hz"] == pytest.approx(
        v["impact_click_frequency_hz"])


def test_every_defaulted_scenario_parameter_is_a_recorded_choice():
    """The Captain's own prompt names no bell size/material, no stone mass,
    no observer distance -- every one of those defaults must be an
    exposed Choice, not a silent assumption (Choice doctrine, 2026-07-16:
    'Mark ANY variable in simulations you have to come up with')."""
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    choices = bundle["scene"]["choices"]
    scenario = next(c for c in choices
                    if "bell_material" in c["value"]
                    and "stone_mass_kg" in c["value"])
    for key in ("bell_material", "bell_diameter_m", "stone_mass_kg",
               "stone_material", "observer_distance_m"):
        assert key in scenario["value"]
    assert scenario["adjustable"]
