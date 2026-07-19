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


def test_held_bell_stays_essentially_fixed_while_stone_falls():
    bundle = record_bell_strike("iron", bell_diameter_m=0.15)
    frames = bundle["trajectory"]["frames"]
    p0 = frames[0]["bodies"][0]["pos"]
    for f in frames:
        p = f["bodies"][0]["pos"]
        d = math.dist(p0, p)
        assert d < 1e-6, f"welded bell moved {d} m"


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
    assert bundle["scene"]["plugs"]


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
