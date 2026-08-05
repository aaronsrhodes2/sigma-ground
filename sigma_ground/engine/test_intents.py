"""Tests for intents.py — E3's gate, headless, no browser.

THE GATE (the approved plan's wording): "every game verb reduces to a
valid AppliedForce, proven by a scripted command-injection test with no
browser involved." Concretely: load the REAL shipped
data/input_bindings.json, and for EVERY world-channel verb reachable from
it, reduce with representative params → AppliedForce.from_dict parses it
AND a real StackSimRegion (the proven offline build_stack stack) either
ACCEPTS it via region.apply or refuses it honestly (RegionRefusal with a
message) — no crashes, no silent drops — and at least
impulse/force/hold/tilt must be ACCEPTED against real bodies.

Plus the adversary surface: engine verbs structurally cannot produce
forces; unknown verbs are refused naming the closed set; param fuzz
raises IntentRefusal and ONLY IntentRefusal; the shipped bindings file
validates against the registry; the committed intent_verbs.json mirror is
fresh; and an intent-driven run's command log replays hash-identical
(reusing E2 save/load) — input-driven forces are replay-exact.

House style of test_host.py: plain pytest functions, the shared offline
stack built once per module, exact-float assertions wherever the wall
clock is not involved (it never is here — everything is headless).
"""
import dataclasses
import json
import os
import re

import pytest

from sigma_ground.dynamics.mechanisms.applied import AppliedForce
from sigma_ground.engine import intents
from sigma_ground.engine.ecs import World
from sigma_ground.engine.host import WorldHost
from sigma_ground.engine.intents import (ENGINE_VERBS, PRESS_TICK_S,
                                         WORLD_VERBS, EngineVerb,
                                         IntentRefusal, Param, reduce,
                                         verb_names, verbs_json)
from sigma_ground.engine.region import RegionRefusal, StackSimRegion
from sigma_ground.radiance.push_off import make_lookups
from sigma_ground.radiance.web import serve

ITEMS = ["1700s table", "plate", "Doughnut"]
_WEB_DATA = os.path.join(os.path.dirname(os.path.abspath(serve.__file__)),
                         "data")
BINDINGS_PATH = os.path.join(_WEB_DATA, "input_bindings.json")
MIRROR_PATH = os.path.join(_WEB_DATA, "intent_verbs.json")


@pytest.fixture(scope="module")
def built():
    from sigma_ground.deckard.scene import build_stack
    return build_stack(ITEMS, allow_llm=False)


def _region(built):
    mu, e = make_lookups(1)
    return StackSimRegion(World(), built, mu_lookup=mu, e_lookup=e)


def _mass_table(sim):
    return {sim.names[i]: float(sim.mass[i]) for i in range(sim.n)}


def _bindings():
    with open(BINDINGS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# § 1. THE GATE — every bound world verb lands on a real region
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_every_bound_world_verb_lands_on_a_real_region(built):
    """The REAL shipped binding file drives the REAL production reduction
    onto the REAL offline stack. Derive entries use scale*1.0 — full
    analog deflection — which doubles as a shipped-scale range tripwire
    (the disclosed Python test-twin of input.js's analog composition)."""
    region = _region(built)
    sim = region.sim
    masses = _mass_table(sim)
    accepted, world_entries = set(), 0
    for entry in _bindings()["bindings"]:
        verb = entry["verb"]
        if verb not in WORLD_VERBS:
            continue
        world_entries += 1
        spec = WORLD_VERBS[verb]
        params = dict(entry.get("params") or {})
        for pname, d in (entry.get("derive") or {}).items():
            params[pname] = d["scale"] * 1.0        # full deflection
        target = "plate" if spec.needs_target else None
        red = reduce(verb, params, target=target, masses=masses)
        # the force dict must survive the wire (a real JSON round trip)
        force = json.loads(json.dumps(red["force"]))
        af = AppliedForce.from_dict(force)
        eid = (region.entity_ids[sim.names.index(red["target"])]
               if red["target"] is not None else None)
        try:
            region.apply(eid, af)
            accepted.add(af.kind)
        except RegionRefusal as e:                  # honest refusal is legal
            assert str(e), "a refusal must carry a message"
        # ANY other exception = the gate fails (no crashes, no silent drops)
    assert world_entries >= 10, "the shipped table lost world coverage"
    assert {"impulse", "force", "hold", "tilt"} <= accepted, (
        "every live kind must be ACCEPTED against real bodies from the "
        f"shipped bindings, not just refused — accepted: {sorted(accepted)}")


# ═══════════════════════════════════════════════════════════════════════════
# § 2. structural split — engine verbs can never produce forces
# ═══════════════════════════════════════════════════════════════════════════

def test_engine_verbs_structurally_cannot_produce_forces():
    assert not (set(WORLD_VERBS) & set(ENGINE_VERBS))
    # the force path is ABSENT from the type, not merely untested
    assert "reduce" not in {f.name for f in dataclasses.fields(EngineVerb)}
    ops = {"pace", "pause", "resume", "cam_orbit", "cam_zoom", "cam_pan",
           "cam_move", "cam_mode", "follow"}
    for name in ENGINE_VERBS:
        r = reduce(name)                            # defaults fill
        assert r["channel"] == "engine"
        assert "force" not in r
        assert "target" not in r
        assert r["op"] in ops


# ═══════════════════════════════════════════════════════════════════════════
# § 3. refusal vocabulary — closed set, honest bounds
# ═══════════════════════════════════════════════════════════════════════════

def test_unknown_verb_refused_naming_closed_set():
    for bad in ("yeet", None, 3):
        with pytest.raises(IntentRefusal) as ei:
            reduce(bad, {})
        for name in verb_names():                   # the WHOLE closed set
            assert name in str(ei.value)


def test_param_fuzz_raises_only_intent_refusal(built):
    """Totality: every invalid input dies as IntentRefusal — never any
    other exception, never a silent acceptance."""
    masses = {"plate": 0.5}
    junk = ["fast", 3, [],
            {"speed_m_s": "fast"}, {"speed_m_s": True},
            {"speed_m_s": float("nan")}, {"speed_m_s": float("inf")},
            {"speed_m_s": 1e308}, {"speed_m_s": 0.0},
            {"speed_m_s": 25.0}, {"extra": 1}, {"direction": "up"}]
    for p in junk:
        with pytest.raises(IntentRefusal):
            reduce("shove", p, target="plate", masses=masses)
    # messages name the param and the bound
    with pytest.raises(IntentRefusal, match=r"speed_m_s.*20"):
        reduce("shove", {"speed_m_s": 25.0}, target="plate", masses=masses)
    with pytest.raises(IntentRefusal, match=r"speed_m_s.*number"):
        reduce("shove", {"speed_m_s": True}, target="plate", masses=masses)
    with pytest.raises(IntentRefusal, match=r"force_n.*nonzero"):
        reduce("press", {"force_n": 0.0}, target="plate")
    # defaults fill when params are omitted entirely
    red = reduce("shove", None, target="plate", masses=masses)
    assert red["force"]["variables"]["push_speed_m_s"] == 2.2
    # the required-param mechanism (no SHIPPED verb requires one — defaults
    # everywhere is deliberate so bare chords work; the validator still
    # enforces default=None => required)
    with pytest.raises(IntentRefusal, match=r"requires parameter 'x'"):
        intents._validated("synthetic", (Param("x"),), {})


def test_json_integers_too_large_for_a_double_are_refused():
    """ADVERSARIAL REGRESSION. JSON integers have no width limit, so
    ``{"speed_m_s": 999…}`` (400 digits) reaches reduce() as an
    arbitrary-precision Python int — and float() on one raises
    OverflowError, an ArithmeticError that is NOT an IntentRefusal. It used
    to escape reduce()'s totality contract, sail past cockpit's single
    IntentRefusal arm and KILL the /world connection instead of answering
    with a refusal envelope. Every numeric param of every verb, both signs,
    and via a real json.loads of the wire text."""
    masses = {"plate": 0.5}
    wire = json.loads('{"big": ' + "9" * 400 + ', "negbig": -'
                      + "9" * 400 + "}")
    assert all(isinstance(v, int) for v in wire.values())
    for name, spec in list(WORLD_VERBS.items()) + list(ENGINE_VERBS.items()):
        for p in spec.params:
            if p.choices is not None:
                continue
            for huge in wire.values():
                with pytest.raises(IntentRefusal, match="not a finite number"):
                    reduce(name, {p.name: huge}, masses=masses,
                           target=("plate" if name in WORLD_VERBS
                                   and spec.needs_target else None))


def test_refusal_messages_never_echo_unbounded_hostile_values():
    """A refusal quotes the offending value; the value is hostile input, so
    the quote must be BOUNDED — otherwise a 1 MB param (or a deeply nested
    one, whose repr recurses) becomes a 1 MB reply or a RecursionError."""
    deep = {"leaf": 1}
    for _ in range(2000):
        deep = {"n": [deep]}
    for bad in ("x" * 500_000, deep, list(range(10_000)), 10 ** 5000):
        with pytest.raises(IntentRefusal) as ei:
            reduce("press", {"force_n": bad}, target="plate")
        assert len(str(ei.value)) < 400, len(str(ei.value))
    with pytest.raises(IntentRefusal) as ei:                 # unknown verb
        reduce("y" * 500_000, {})
    assert len(str(ei.value)) < 600, len(str(ei.value))
    with pytest.raises(IntentRefusal) as ei:                 # unknown params
        reduce("press", {f"k{i}": 1 for i in range(5000)}, target="plate")
    assert len(str(ei.value)) < 400, len(str(ei.value))


def test_reduce_is_total_over_the_public_api_too():
    """reduce() is a PUBLIC API, not only a socket handler: shapes JSON
    cannot produce (non-str param keys, a broken mass table) must still be
    IntentRefusal, never TypeError/ValueError/OverflowError."""
    with pytest.raises(IntentRefusal):
        reduce("shove", {1: 2, "zz": 3}, target="plate", masses={"plate": 0.5})
    for table in ({"plate": "heavy"}, {"plate": None}, {"plate": 10 ** 400}):
        with pytest.raises(IntentRefusal):
            reduce("shove", None, target="plate", masses=table)


def test_target_rules(built):
    region = _region(built)
    masses = _mass_table(region.sim)
    with pytest.raises(IntentRefusal, match="needs a target body"):
        reduce("shove", None, target=None, masses=masses)
    with pytest.raises(IntentRefusal) as ei:
        reduce("shove", None, target="not-a-body", masses=masses)
    for name in sorted(masses):                     # names the mass table
        assert name in str(ei.value)
    with pytest.raises(IntentRefusal, match="scene-scoped"):
        reduce("tilt", None, target="plate")
    for name in ENGINE_VERBS:
        with pytest.raises(IntentRefusal, match="cannot touch physics"):
            reduce(name, None, target="plate")


# ═══════════════════════════════════════════════════════════════════════════
# § 4. exact-float reductions — the wire conventions, bit for bit
# ═══════════════════════════════════════════════════════════════════════════

def test_exact_float_reductions(built):
    region = _region(built)
    masses = _mass_table(region.sim)
    m = masses["plate"]

    red = reduce("shove", {"speed_m_s": 2.2, "direction": "+x"},
                 target="plate", masses=masses)
    f = red["force"]
    assert f["kind"] == "impulse"
    assert f["vector"] == [m * 2.2, 0.0, 0.0]       # EXACT — never rounded
    assert f["variables"]["push_speed_m_s"] == 2.2  # the dv authority
    assert "duration" not in f                      # impulse: duration 0
    red2 = reduce("shove", {"speed_m_s": 2.2, "direction": "-x"},
                  target="plate", masses=masses)
    assert red2["force"]["vector"][0] == -(m * 2.2)  # sign flip is exact
    assert red2["force"]["variables"]["push_speed_m_s"] == -2.2

    red = reduce("press", {"force_n": 5.0}, target="plate")
    f = red["force"]
    assert f["kind"] == "force"
    assert f["duration"] == PRESS_TICK_S == 1.0 / 60.0
    assert f["vector"][0] == 5.0

    for verb, on in (("grab", True), ("release", False)):
        f = reduce(verb, None, target="plate")["force"]
        assert f["kind"] == "hold"
        assert f["variables"] == {"on": on}
        assert "point" not in f                     # region refuses point

    red = reduce("tilt", {"angle_deg": 6.0, "duration_s": 2.0})
    assert red["target"] is None                    # envelope: scene-scoped
    f = red["force"]
    assert f["kind"] == "tilt" and f["target"] == "world"
    assert f["variables"] == {"angle_deg": 6.0}     # NO axis key ever
    assert f["duration"] == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# § 5. the shipped data files — bindings valid, mirror fresh
# ═══════════════════════════════════════════════════════════════════════════

_CHORD_RE = re.compile(
    r"^(key:((Shift|Ctrl|Alt)\+)*[A-Za-z0-9]+"
    r"|mouse:(wheel|move)"
    r"|pad:(button|axis):\d+)$")


def test_bindings_file_validates_against_registry():
    data = _bindings()
    assert data["format"] == "sg-input-bindings" and data["version"] == 1
    seen = set()
    for entry in data["bindings"]:
        # channel is DERIVED from the registry — stating it in the JSON is
        # the contradiction hazard the design forbids
        assert "channel" not in entry, entry
        chord, verb = entry["chord"], entry["verb"]
        assert _CHORD_RE.match(chord), chord
        spec = WORLD_VERBS.get(verb) or ENGINE_VERBS.get(verb)
        assert spec is not None, f"unknown verb {verb!r} in bindings"
        declared = {p.name for p in spec.params}
        fixed = dict(entry.get("params") or {})
        assert set(fixed) <= declared, (verb, fixed)
        derive = entry.get("derive") or {}
        assert set(derive) <= declared, (verb, derive)
        if derive:      # derive only on analog chords
            assert (chord.startswith("pad:axis:")
                    or chord in ("mouse:wheel", "mouse:move")), chord
        key = (chord, verb)
        assert key not in seen, f"duplicate {key}"
        seen.add(key)
        # fixed params merged with defaults pass the production validation
        if verb in WORLD_VERBS:
            t = "plate" if spec.needs_target else None
            reduce(verb, fixed, target=t, masses={"plate": 0.5})
        else:
            reduce(verb, fixed)


def test_intent_verbs_mirror_is_fresh():
    with open(MIRROR_PATH, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(json.dumps(verbs_json())), (
        "data/intent_verbs.json is stale — regenerate it: "
        "python -m sigma_ground.engine.intents")


# ═══════════════════════════════════════════════════════════════════════════
# § 6. replay — an intent-driven run's log replays hash-identical
# ═══════════════════════════════════════════════════════════════════════════

def test_intent_driven_run_replays_hash_identical(built):
    """The schedule is built ONLY from reduce() outputs. The press
    re-issues at 16-step spacing — exactly PRESS_TICK_S at 960 Hz, the
    held-key cadence as the command log records it. Save at 800 lands
    mid-tilt-ramp (480→2400) — the nasty case E2's hash covers."""
    host = WorldHost.from_built(built, items=ITEMS)
    masses = _mass_table(host.region.sim)
    sched = (
        (0,   "shove",   {"speed_m_s": 2.2},                    "plate"),
        (240, "press",   {"force_n": 5.0},                      "plate"),
        (256, "press",   {"force_n": 5.0},                      "plate"),
        (272, "press",   {"force_n": 5.0},                      "plate"),
        (480, "tilt",    {"angle_deg": 6.0, "duration_s": 2.0}, None),
        (520, "grab",    None,                                  "Doughnut"),
        (560, "release", None,                                  "Doughnut"),
    )
    for k in range(800):
        for (at, verb, params, target) in sched:
            if at == k:
                red = reduce(verb, params, target=target, masses=masses)
                res = host.apply(red["target"], red["force"])
                assert res["ok"], res
        host.advance(1)
    saved = host.save()
    assert len(saved["command_log"]) == len(sched)  # every intent logged
    h2 = WorldHost.load(json.loads(json.dumps(saved)), prebuilt=built)
    assert h2.state_hash() == host.state_hash()     # string ==
    # the E2 continuation idiom: input-driven forces are replay-exact
    host.advance(500)
    h2.advance(500)
    assert h2.state_hash() == host.state_hash()


# ═══════════════════════════════════════════════════════════════════════════
# § 7. E4 — the pace-scaled press window (the E3 disclosed deferral, fixed)
# ═══════════════════════════════════════════════════════════════════════════

def test_press_duration_scales_by_pace_exactly():
    for pace in (0.01, 0.5, 1.0, 10.0):
        f = reduce("press", {"force_n": 5.0}, target="plate",
                   pace=pace)["force"]
        assert f["duration"] == PRESS_TICK_S * pace     # EXACT floats
    # pace=None (headless/no-host callers) → the unscaled tick
    f = reduce("press", {"force_n": 5.0}, target="plate")["force"]
    assert f["duration"] == PRESS_TICK_S


def test_tick_scaled_is_press_only_and_not_exported():
    assert WORLD_VERBS["press"].tick_scaled is True
    for name, spec in WORLD_VERBS.items():
        if name != "press":
            assert spec.tick_scaled is False, name
    # NOT a client parameter: verbs_json (the mirror) never mentions it
    assert "tick_scaled" not in json.dumps(verbs_json())


def test_shove_and_tilt_ignore_pace():
    masses = {"plate": 0.5}
    a = reduce("shove", {"speed_m_s": 2.2}, target="plate", masses=masses)
    b = reduce("shove", {"speed_m_s": 2.2}, target="plate", masses=masses,
               pace=0.25)
    assert a["force"] == b["force"]                     # no duration at all
    ta = reduce("tilt", {"angle_deg": 6.0, "duration_s": 2.0})
    tb = reduce("tilt", {"angle_deg": 6.0, "duration_s": 2.0}, pace=0.25)
    assert ta["force"] == tb["force"]
    assert ta["force"]["duration"] == 2.0               # USER sim-time ramp


def test_hostile_pace_values_refuse_and_only_refuse():
    """Totality extends to the new kwarg: every hostile pace dies as
    IntentRefusal — never TypeError/OverflowError — for world AND engine
    verbs alike (validation runs before verb dispatch)."""
    hostile = [True, False, "fast", float("nan"), float("inf"),
               -float("inf"), 0.0, 0.0099, 10.01, -1.0, 10 ** 400,
               [1.0], {"pace": 1.0}]
    for bad in hostile:
        with pytest.raises(IntentRefusal):
            reduce("press", {"force_n": 5.0}, target="plate", pace=bad)
        with pytest.raises(IntentRefusal):
            reduce("pause", pace=bad)
    # the bounds are the host's own PACE_MIN/PACE_MAX — edges accepted
    from sigma_ground.engine.host import PACE_MAX, PACE_MIN
    for edge in (PACE_MIN, PACE_MAX):
        f = reduce("press", {"force_n": 5.0}, target="plate",
                   pace=edge)["force"]
        assert f["duration"] == PRESS_TICK_S * edge


def test_pace_scaled_press_replays_hash_identical(built):
    """The command log records the ACTUAL scaled duration — replay stays
    exact by construction (the log is the state)."""
    host = WorldHost.from_built(built, items=ITEMS)
    masses = _mass_table(host.region.sim)
    for k in range(300):
        if k in (0, 16, 32):
            red = reduce("press", {"force_n": 5.0}, target="plate",
                         masses=masses, pace=0.5)
            assert red["force"]["duration"] == PRESS_TICK_S * 0.5
            assert host.apply(red["target"], red["force"])["ok"]
        host.advance(1)
    saved = host.save()
    for (_k, _t, force) in saved["command_log"]:
        assert force["duration"] == PRESS_TICK_S * 0.5  # logged as applied
    h2 = WorldHost.load(json.loads(json.dumps(saved)), prebuilt=built)
    assert h2.state_hash() == host.state_hash()
