"""Tests for audio_events.py + the recording-only event source (E4).

The load-bearing invariant: sim.events is PRESENTATION — physics never
reads it. §1 pins the recording itself (a scripted drop lands with
vimp ≈ sqrt(2·g·h)); §2 pins hash-blindness (state floats identical with
events present vs cleared); §3 pins the server-side synthesis params
against INDEPENDENT acoustics.py calls; §4 the honest thud fallback; §5
annotate's purity/floor/guard; §6 the bounded buffer.

House style of test_host.py: plain pytest functions, the shared offline
stack built once per module, exact-float assertions (headless — no wall
clock anywhere).
"""
import math
import types

import pytest

from sigma_ground.engine import audio_events
from sigma_ground.engine.bench.determinism import _state_stacksim
from sigma_ground.engine.host import _full_state_floats
from sigma_ground.field.interface import acoustics
from sigma_ground.radiance.push_off import G, StackSim, make_lookups

ITEMS = ["1700s table", "plate", "Doughnut"]


@pytest.fixture(scope="module")
def built():
    from sigma_ground.deckard.scene import build_stack
    return build_stack(ITEMS, allow_llm=False)


def _fake_construct(bbox, mass, com, material="granite"):
    layer = types.SimpleNamespace(name="body", material=material,
                                  mass_kg=mass)
    return types.SimpleNamespace(bbox=bbox, mass_kg=mass, com_m=list(com),
                                 layers=[layer])


def _drop_sim(h=1.0):
    """A minimal synthetic stack: a slab base and one body whose CoM sits
    OUTSIDE the base rectangle, so the first step drops it into free fall
    from height h (its bottom at floor_z + h)."""
    base = _fake_construct(((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.1)),
                           10.0, (0.0, 0.0, 0.05))
    body = _fake_construct(((1.9, 2.1), (-0.1, 0.1), (h, h + 0.2)),
                           0.5, (2.0, 0.0, h + 0.1))
    mu, e = make_lookups(1)
    return StackSim([("slab", base), ("ball", body)],
                    mu_lookup=mu, e_lookup=e)


# ═══════════════════════════════════════════════════════════════════════════
# § 1. the recording — a scripted drop lands with vimp ≈ sqrt(2gh)
# ═══════════════════════════════════════════════════════════════════════════

def test_scripted_drop_records_impact_with_ballistic_speed():
    h = 1.0
    sim = _drop_sim(h)
    dt = 1.0 / 960.0
    for _ in range(2000):
        sim.step(dt)
        if sim.events:
            break
    assert sim.events, "the drop never recorded an event"
    ev = sim.events[0]
    v_expect = math.sqrt(2.0 * G * h)
    assert ev["body"] == 1 and ev["surface"] == "floor"
    # one substep of over-travel tolerance (the integrator lands within dt)
    assert abs(ev["impact_speed_m_s"] - v_expect) < v_expect * 0.02
    assert ev["e"] == 0.30                  # DEFAULT_E "plate" (body 1 IS
    #                                         make_lookups' pushed_index)
    assert ev["t"] > 0.0
    # e*vimp ≈ 1.33 ≥ 0.25 → a visible bounce, recorded as one, with the
    # outgoing speed exactly e*vimp (same floats, same op)
    assert ev["kind"] == "bounce"
    assert ev["v_out_m_s"] == ev["e"] * ev["impact_speed_m_s"]
    # run to rest: the chain ends in a settled "impact" (v_out == 0)
    for _ in range(6000):
        sim.step(dt)
        if sim.settled():
            break
    assert sim.settled()
    last = sim.events[-1]
    assert last["kind"] == "impact" and last["v_out_m_s"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# § 2. hash-blindness — state floats identical with events present/cleared
# ═══════════════════════════════════════════════════════════════════════════

def test_state_floats_blind_to_events():
    sim = _drop_sim()
    dt = 1.0 / 960.0
    for _ in range(2000):
        sim.step(dt)
    assert sim.events                       # events ARE present
    with_events = list(_state_stacksim(sim)), _full_state_floats(sim, 2000)
    sim.events.clear()
    without = list(_state_stacksim(sim)), _full_state_floats(sim, 2000)
    assert with_events == without           # exact — never approx


# ═══════════════════════════════════════════════════════════════════════════
# § 3. build_profiles — frequencies match INDEPENDENT acoustics calls
# ═══════════════════════════════════════════════════════════════════════════

def test_build_profiles_real_stack_frequencies(built):
    profiles = audio_events.build_profiles(built)
    assert set(profiles) == set(ITEMS)
    plate = profiles["plate"]
    assert plate["mapped_key"] == "ceramic_alumina"
    assert plate["synth"] == "ring"
    assert plate["frequency_hz"] == acoustics.ring_frequency(
        "ceramic_alumina", plate["size_m"])          # EXACT — same call
    assert plate["q"] == audio_events.Q_BY_KEY["ceramic_alumina"]
    table = profiles["1700s table"]
    assert table["mapped_key"] == "wood_oak"         # identity arm
    assert table["frequency_hz"] == acoustics.ring_frequency(
        "wood_oak", table["size_m"])
    # masses are the Constructs' own Facts
    for name, c in built:
        assert profiles[name]["mass_kg"] == float(c.mass_kg)


def test_unmapped_material_is_honest_thud(built):
    profiles = audio_events.build_profiles(built)
    dough = profiles["Doughnut"]
    assert dough["synth"] == "thud"
    assert dough["frequency_hz"] is None             # NO pitch claim
    assert dough["q"] is None
    assert dough["mapped_key"] is None
    assert "no pitch claim" in dough["provenance"]


# ═══════════════════════════════════════════════════════════════════════════
# § 4-5. annotate — purity, energy Deduction, floor, non-finite guard
# ═══════════════════════════════════════════════════════════════════════════

_RAW = {"kind": "impact", "body": 1, "surface": "floor",
        "impact_speed_m_s": 2.0, "e": 0.3, "mu": 0.4, "v_out_m_s": 0.0,
        "x": 0.5, "z": 0.0, "t": 1.25}
_NAMES = ["slab", "ball"]
_MASSES = {"slab": 10.0, "ball": 0.5}


def _profiles():
    return {"ball": {"material": "granite", "mapped_key": None,
                     "size_m": 0.2, "mass_kg": 0.5, "synth": "thud",
                     "frequency_hz": None, "q": None, "provenance": "test"},
            "slab": {"material": "granite", "mapped_key": None,
                     "size_m": 1.0, "mass_kg": 10.0, "synth": "thud",
                     "frequency_hz": None, "q": None, "provenance": "test"}}


def test_annotate_pure_and_energy_exact():
    a = audio_events.annotate(dict(_RAW), _NAMES, _MASSES, _profiles(), 42)
    b = audio_events.annotate(dict(_RAW), _NAMES, _MASSES, _profiles(), 42)
    assert a == b                                    # purity
    assert a["energy_j"] == 0.5 * 0.5 * 2.0 ** 2     # the Deduction, exact
    assert a["acoustic_energy_j"] == \
        audio_events.ETA_ACOUSTIC * a["energy_j"]
    assert a["body"] == "ball" and a["surface"] == "floor"
    assert a["materials"]["surface"] == \
        audio_events.FLOOR_PROFILE["material"]
    assert a["position"] == [0.5, 0.0, 0.0]
    assert a["step"] == 42 and a["t"] == 1.25
    assert a["restitution_e"] == 0.3
    import json
    json.dumps(a, allow_nan=False)                   # strict-JSON clean


def test_annotate_floor_and_nonfinite_guard():
    slow = dict(_RAW, impact_speed_m_s=0.04)         # below the floor
    assert audio_events.annotate(slow, _NAMES, _MASSES, _profiles(), 0) \
        is None
    at_floor = dict(_RAW, impact_speed_m_s=audio_events.MIN_IMPACT_SPEED)
    assert audio_events.annotate(at_floor, _NAMES, _MASSES,
                                 _profiles(), 0) is not None
    for bad in (float("nan"), float("inf")):
        for key in ("impact_speed_m_s", "x", "z", "t"):
            raw = dict(_RAW)
            raw[key] = bad
            assert audio_events.annotate(raw, _NAMES, _MASSES,
                                         _profiles(), 0) is None, (key, bad)
    # malformed shapes are drops, never raises
    assert audio_events.annotate({}, _NAMES, _MASSES, _profiles(), 0) is None
    assert audio_events.annotate(dict(_RAW, body=99), _NAMES, _MASSES,
                                 _profiles(), 0) is None


def test_annotate_no_profiles_falls_back_to_thud():
    a = audio_events.annotate(dict(_RAW), _NAMES, _MASSES, None, 0)
    assert a is not None
    assert a["synth"]["kind"] == "thud"
    assert a["synth"]["frequency_hz"] is None        # NO pitch claim


# ═══════════════════════════════════════════════════════════════════════════
# § 6. the bounded buffer — disclosed drop-oldest
# ═══════════════════════════════════════════════════════════════════════════

def test_event_buffer_caps_at_256_drop_oldest():
    sim = _drop_sim()
    for k in range(257):
        sim.events.append({"n": k})
    assert len(sim.events) == 256
    assert sim.events[0] == {"n": 1}                 # oldest dropped
    assert sim.events[-1] == {"n": 256}
