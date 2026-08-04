"""Tests for host.py — the WorldHost (engine phase E2), headless-first.

The two E2 gates live here in scripted form:
  GATE 1 — two observers of one host receive IDENTICAL snapshot sequences
           (the socket-level twin lives in mcp/test_cockpit.py).
  GATE 2 — save() mid-run WITH a force window still open and a tilt
           mid-ramp (the nasty case), load() back, replay → state_hash()
           EXACTLY equal (string ==); then 500 more identical steps on
           both → hashes STILL equal.

Plain pytest functions, house style of test_region_stacksim.py. The stack
is the proven offline scenario: build_stack(["1700s table", "plate",
"Doughnut"], allow_llm=False) — deterministic, no network, no LLM. Built
ONCE (module fixture) and shared — safe: StackSim copies com into its own
lists at construction and never mutates Constructs; the GATE 2 load
deliberately pays for a REAL rebuild (that IS the gate).

The paced-loop tests make only LOOSE liveness assertions (the wall clock
is involved); every EXACT assertion lives on log-replay hash equality,
which is wall-clock-independent by design.
"""
import json
import time

import pytest

from sigma_ground.dynamics.mechanisms.applied import AppliedForce
from sigma_ground.engine.host import HostBusy, WorldHost

DT = 1.0 / 960.0
ITEMS = ["1700s table", "plate", "Doughnut"]


@pytest.fixture(scope="module")
def built():
    from sigma_ground.deckard.scene import build_stack
    return build_stack(ITEMS, allow_llm=False)


def _schedule(built):
    """The scripted command schedule — irregular indices, exact floats.

    At step 600 (where GATE 2 saves): the step-560 force window is LIVE
    (ends ~step 944), the tilt is MID-RAMP (480→960), the hold is ON."""
    m_plate = dict(built)["plate"].mass_kg
    return (
        (0,   "plate",    {"kind": "impulse", "target": "plate",
                           "vector": [m_plate * 2.2, 0.0, 0.0],
                           "variables": {"push_speed_m_s": 2.2}}),
        (240, "Doughnut", {"kind": "force", "target": "Doughnut",
                           "vector": [0.8, 0.0, 0.0], "duration": 0.25}),
        (480, None,       {"kind": "tilt", "target": "world",
                           "variables": {"angle_deg": 6.0},
                           "duration": 0.5}),
        (520, "Doughnut", {"kind": "hold", "target": "Doughnut",
                           "variables": {"on": True}}),
        (560, "plate",    {"kind": "force", "target": "plate",
                           "vector": [1.2, 0.0, 0.0], "duration": 0.4}),
    )


HOLD_ON = {"kind": "hold", "target": "Doughnut", "variables": {"on": True}}
HOLD_OFF = {"kind": "hold", "target": "Doughnut", "variables": {"on": False}}


def _walk(host, upto, schedule):
    """Apply due entries at their exact indices, apply-then-step (E1's
    convention), from wherever the host currently stands up to `upto`."""
    for k in range(host.step_count, upto):
        for (at, target, force) in schedule:
            if at == k:
                res = host.apply(target, force)
                assert res["ok"], res
        host.advance(1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1. GATE 2 — save mid-run, load, replay: hash EXACTLY equal
# ═══════════════════════════════════════════════════════════════════════════

def test_gate2_save_load_replay_hash_equal(built):
    h1 = WorldHost.from_built(built, items=ITEMS)
    _walk(h1, 600, _schedule(built))
    # the trailing-boundary case: applied at 600, not yet stepped
    res = h1.apply("Doughnut", HOLD_OFF)
    assert res["ok"] and res["step_index"] == 600
    hash0 = h1.state_hash()
    saved = h1.save()

    assert saved == h1.save()                       # double-save idempotence
    assert json.loads(json.dumps(saved)) == saved   # strict-JSON clean
    assert saved["state_hash"] == hash0
    assert saved["step_count"] == 600
    assert saved["format"] == "sg-world-save" and saved["version"] == 1

    # THE GATE: real build_stack + a real serialization round trip
    h2 = WorldHost.load(json.loads(json.dumps(saved)))
    assert h2.state_hash() == hash0                 # string ==
    assert h2.save() == saved                       # load-then-save equality

    # continue BOTH hosts 500 more identical steps — the save captured
    # everything, including the in-flight force window and tilt ramp
    futures = ((700, "Doughnut", HOLD_ON), (900, "Doughnut", HOLD_OFF))
    _walk(h1, 1100, futures)
    _walk(h2, 1100, futures)
    assert h1.state_hash() == h2.state_hash()


def test_save_at_step_zero_round_trips(built):
    h = WorldHost.from_built(built, items=ITEMS)
    saved = h.save()
    assert saved["command_log"] == [] and saved["step_count"] == 0
    h2 = WorldHost.load(saved, prebuilt=built)
    assert h2.state_hash() == h.state_hash()


def test_load_refuses_unknown_format_and_version(built):
    h = WorldHost.from_built(built, items=ITEMS)
    saved = h.save()
    with pytest.raises(ValueError, match="sg-world-save"):
        WorldHost.load({"format": "not-a-save"})
    bad = dict(saved)
    bad["version"] = 99
    with pytest.raises(ValueError, match="version"):
        WorldHost.load(bad, prebuilt=built)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. command log hygiene — refusals never enter, accepted in order
# ═══════════════════════════════════════════════════════════════════════════

def test_refused_commands_never_enter_the_log(built):
    h = WorldHost.from_built(built, items=ITEMS)
    n0 = len(h.command_log)
    r = h.apply("plate", {"kind": "source", "target": "plate",
                          "power_w": 50.0, "duration": 1.0})
    assert not r["ok"] and "thermal" in r["refusal"]
    r = h.apply("plate", {"kind": "drive", "target": "plate",
                          "variables": {"torque_nm": 1.0}})
    assert not r["ok"] and "joint" in r["refusal"]
    r = h.apply("spoon", {"kind": "impulse", "target": "spoon",
                          "vector": [1.0, 0.0, 0.0],
                          "variables": {"push_speed_m_s": 1.0}})
    assert not r["ok"] and "spoon" in r["refusal"]
    r = h.apply("plate", {"kind": "bogus", "target": "plate"})
    assert not r["ok"] and "CLOSED" in r["refusal"]
    r = h.apply("1700s table", {"kind": "impulse", "target": "1700s table",
                                "vector": [1.0, 0.0, 0.0],
                                "variables": {"push_speed_m_s": 1.0}})
    assert not r["ok"] and "base" in r["refusal"]
    assert len(h.command_log) == n0


def test_command_log_records_application_order_and_round_trips(built):
    h = WorldHost.from_built(built, items=ITEMS)
    _walk(h, 600, _schedule(built))
    log = h.command_log
    assert [e[0] for e in log] == [0, 240, 480, 520, 560]
    assert [e[1] for e in log] == ["plate", "Doughnut", None, "Doughnut",
                                   "plate"]
    for (_k, _target, force) in log:
        # the canonical form survives a JSON trip without drift
        assert AppliedForce.from_dict(force) == \
            AppliedForce.from_dict(json.loads(json.dumps(force)))


# ═══════════════════════════════════════════════════════════════════════════
# § 3. GATE 1 (headless form) — two observers, identical sequences
# ═══════════════════════════════════════════════════════════════════════════

def test_gate1_two_observers_identical_snapshot_sequences(built):
    h = WorldHost.from_built(built, items=ITEMS)
    qa, qb = h.subscribe(), h.subscribe()           # BEFORE any step
    sched = _schedule(built)
    for chunk in range(0, 608, 32):
        for k in range(chunk, chunk + 32):
            for (at, target, force) in sched:
                if at == k:
                    assert h.apply(target, force)["ok"]
            h.advance(1)
        h.emit_snapshot()

    def _drain_all(q):
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    fa, fb = _drain_all(qa), _drain_all(qb)
    assert fa and len(fa) == len(fb) == 608 // 32
    for a, b in zip(fa, fb):
        assert a == b                               # identical sequences
    # the pushed body actually moved across the frames (display pos, +x)
    assert fa[-1]["bodies"][1]["pos"][0] > fa[0]["bodies"][1]["pos"][0]
    assert [f["step"] for f in fa] == sorted(f["step"] for f in fa)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. the paced loop — liveness, pause, and replay-invariance
# ═══════════════════════════════════════════════════════════════════════════

def test_paced_loop_pause_resume_and_replay(built):
    h = WorldHost.from_built(built, items=ITEMS)
    h.start(pace=10.0)
    try:
        ticket = h.submit("plate", {"kind": "impulse", "target": "plate",
                                    "vector": [1.0, 0.0, 0.0],
                                    "variables": {"push_speed_m_s": 2.2}})
        r = ticket.wait(2.0)
        assert r["ok"], r

        h.pause()
        # poll until quiescent (the in-flight burst is bounded)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            t_probe = h.region.sim.t
            time.sleep(0.05)
            if h.region.sim.t == t_probe:
                break
        else:
            raise AssertionError("host never went quiescent after pause()")
        t0 = h.region.sim.t
        time.sleep(0.1)
        assert h.region.sim.t == t0     # wall time consumed, sim frozen

        # commands still drain at the frozen boundary while paused
        cur = h.step_count
        r2 = h.submit("Doughnut", HOLD_ON).wait(2.0)
        assert r2["ok"] and r2["step_index"] == cur

        h.resume()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and h.region.sim.t <= t0:
            time.sleep(0.02)
        assert h.region.sim.t > t0      # resumed
    finally:
        h.stop()

    # pause/resume/pace scattering is INVISIBLE to log-replay: the save
    # replays headlessly (uninterrupted) to the exact same hash
    saved = h.save()
    assert WorldHost.load(saved, prebuilt=built).state_hash() == \
        h.state_hash()


def test_headless_api_refused_while_loop_runs(built):
    h = WorldHost.from_built(built, items=ITEMS)
    h.start()
    try:
        with pytest.raises(HostBusy):
            h.advance(1)
        with pytest.raises(HostBusy):
            h.apply("plate", {"kind": "hold", "target": "plate",
                              "variables": {"on": True}})
        with pytest.raises(HostBusy):
            h.emit_snapshot()
        h.pause()                       # paused or not: still HostBusy
        with pytest.raises(HostBusy):
            h.advance(1)
    finally:
        h.stop()


def test_hash_includes_progress(built):
    h = WorldHost.from_built(built, items=ITEMS)
    h0 = h.state_hash()
    h.advance(100)
    assert h.state_hash() != h0


def test_pace_clamps_to_the_documented_range(built):
    h = WorldHost.from_built(built, items=ITEMS)
    assert h.set_pace(1000.0) == 10.0
    assert h.set_pace(0.0) == 0.01
    assert h.set_pace(2.5) == 2.5
