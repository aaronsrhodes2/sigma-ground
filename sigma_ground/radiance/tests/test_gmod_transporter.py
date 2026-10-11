"""The transporter effect for found objects (and for Skippy's arrival): offline.

What is pinned here: the PURE coherence function (progress, stalled seconds, time) -> beam parameters (it must condense with progress,
destabilise on a stall, collapse on failure: a flat or constant implementation fails these); the progress the server publishes is built
from REAL pipeline events only (the bridge's /progress, the chunks sent), is monotonic, and reaches a late joiner; a stalled or failed
pipeline ends in a visible failure, the placeholder's removal and a sentence from Skippy; the bridge's /progress is guarded and does not
change /mesh; and the client's drawing (against fakes of the render library) really gets more coherent with progress, destabilises on
silence, collapses on failure and reveals the mesh bottom to top. What these tests CANNOT show is how it looks: that was read off
screenshots in the real game (stage/gmod/holodeck/README.md, "The transporter effect").
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage" / "gmod" / "holodeck"))

import skippy_bridge  # noqa: E402
from test_gmod_found import ALICE, BOB, Http  # noqa: E402
from test_gmod_found_mesh import (  # noqa: E402,F401
    MeshGame, TALL, _answer_with, big_payload, box_blob, client_entity, client_with_payload, draws, feed, fetch, ix, make_msgs, mesh_client, summon,
)
from test_gmod_glue import World  # noqa: E402


@pytest.fixture()
def game(ix):
    return MeshGame()


@pytest.fixture(autouse=True)
def fresh_progress(monkeypatch):
    monkeypatch.setattr(skippy_bridge, "MESH_PROGRESS", {})


# ============================================================================================================ the pure function
def model():
    w = World(cvars={}, load=(), server=False)
    return w, w.rt.eval("function() return dofile(MOCK.ROOT .. '/mentat/fx/transporter_model.lua') end")()


PROGRESS = [0.0, 0.25, 0.5, 0.75, 1.0]


def coh(M, p, stalled=0.0, t=0.0, stall_s=None):
    return M.coherence(p, stalled, t, None if stall_s is None else {"stall_s": stall_s})


def strictly(values, increasing):
    pairs = list(zip(values, values[1:]))
    return all((b > a) if increasing else (b < a) for a, b in pairs)


def test_coherence_condenses_monotonically_as_progress_rises():
    w, M = model()
    rows = [coh(M, p) for p in PROGRESS]
    for name, up in [("noise", False), ("spread", False), ("flicker", False), ("jitter_hz", False),
                     ("density", True), ("alpha", True), ("outline", True), ("fill", True), ("scan_alpha", True), ("scan_hz", True)]:
        vals = [r[name] for r in rows]
        assert strictly(vals, up), f"{name} must {'rise' if up else 'fall'} with progress: {vals}"
    assert rows[0].density < 0.25 and rows[-1].density > 0.95, "sparse at 0, full at 1"
    assert rows[0].noise > 6 * rows[-1].noise, "wide and noisy at 0, tight at 1"
    assert rows[0].outline < 0.05 and rows[-1].outline > 0.95, "the outline locks in"
    assert rows[0].scan_alpha == 0 and rows[2].scan_alpha > 0.3, "the scan band appears as it condenses"


def test_the_colour_is_the_projects_green_when_healthy():
    w, M = model()
    c = coh(M, 0.6).colour
    assert (c[1], c[2], c[3]) == (0, 255, 136)


def test_a_stall_destabilises_the_beam_and_drifts_it_to_amber():
    w, M = model()
    for p in (0.2, 0.5, 0.9):
        base = coh(M, p, 0)
        quiet = [coh(M, p, s) for s in (0, 1, 2, 3, 4.5, 6, 9)]
        assert quiet[0].noise == base.noise and quiet[1].stall == 0 and quiet[2].stall == 0, "no false alarm in the first third of the window"
        for name, up in [("noise", True), ("flicker", True), ("jitter_hz", True), ("stall", True),
                         ("density", False), ("alpha", False), ("outline", False), ("scan_hz", False)]:
            vals = [r[name] for r in quiet]
            assert all((b >= a - 1e-12) if up else (b <= a + 1e-12) for a, b in zip(vals, vals[1:])), f"{name} monotone in the stall at p={p}: {vals}"
            assert (vals[-1] > vals[0]) if up else (vals[-1] < vals[0]), f"{name} must move at p={p}"
        red = [r.colour[1] for r in quiet]
        green = [r.colour[2] for r in quiet]
        assert red[-1] > 200 and green[-1] < 190, "amber (255, 170, 0) when fully destabilised"
        assert strictly(red, True) or red[-1] > red[0]


def test_the_stall_window_is_the_callers_to_set():
    w, M = model()
    assert coh(M, 0.5, 3, stall_s=30).stall == 0
    assert coh(M, 0.5, 3, stall_s=3).stall > 0.9, "the same silence is fatal in a short window"


def test_a_stalled_beam_is_still_less_coherent_than_it_was_but_keeps_its_progress():
    w, M = model()
    # progress is not touched by the stall: the beam does not forget how far it had got
    assert coh(M, 0.6, 9).progress == pytest.approx(0.6) == coh(M, 0.6, 0).progress
    assert coh(M, 0.8, 9).outline > coh(M, 0.2, 9).outline, "even a stalled beam at 80 % is more formed than one at 20 %"


def test_failure_collapses_from_the_top_down_and_turns_red():
    w, M = model()
    rows = [M.failing(s, 0.6) for s in (0, 0.2, 0.5, 0.8, 1.1, 1.4, 1.5)]
    heights = [r.height for r in rows]
    assert heights[0] == 1 and heights[-1] == 0 and strictly(heights[1:-1], False) and heights[1] < 1, f"height must fall to 0: {heights}"
    scatter = [r.scatter for r in rows]
    assert strictly(scatter, True), "the particles scatter outward more and more"
    assert all(r.flicker >= 0.55 for r in rows), "wild flicker throughout"
    assert rows[0].colour[2] > 100 and rows[-1].colour[2] < 80, "amber to red (green channel falls)"
    assert rows[-1].colour[1] == pytest.approx(255, abs=1) and rows[-1].colour[3] == pytest.approx(68, abs=3)
    assert rows[-1].alpha == 0 and rows[-1].done and not rows[0].done and not rows[3].done
    assert rows[0].alpha > 0.5


def test_arrival_reveals_the_mesh_floor_up_and_fades_the_beam():
    w, M = model()
    rows = [M.arriving(s) for s in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.7, 0.8)]
    rev = [r.reveal for r in rows]
    assert rev[0] == 0 and rev[-1] == 1 and strictly(rev[:7], True) and rev[6] == 1 or rev[6] > 0.99
    beam = [r.beam_alpha for r in rows]
    assert beam[0] == 1 and beam[-1] == 0 and all(b <= a + 1e-12 for a, b in zip(beam, beam[1:]))
    assert rows[-1].done and not rows[5].done
    assert rows[1].noise > rows[4].noise, "the shimmer collapses onto the object"


def test_progress_is_monotone_across_the_stages_and_inside_them():
    w, M = model()
    names = [M.STAGES[i] for i in range(1, 7)]
    assert names == ["resolving", "fetching", "measuring", "packing", "sending", "decoding"]
    last = -1
    for n in names:
        a, b = M.progress(n, 0), M.progress(n, 1)
        assert a < b and a >= last - 1e-12, f"{n}: {a} .. {b} after {last}"
        assert M.progress(n, 0.5) == pytest.approx((a + b) / 2)
        last = b
    assert M.progress("done", 0) == 1 and M.progress("nonsense", 0.5) is None
    assert M.progress("fetching", -3) == M.progress("fetching", 0) and M.progress("fetching", 9) == M.progress("fetching", 1)


def test_the_shown_progress_eases_toward_the_real_one_and_never_goes_back():
    w, M = model()
    shown, seen = 0.0, []
    for target in [0.0, 0.3, 0.3, 0.6, 0.55, 0.9, 1.0]:                  # (the real value never drops, but if it were told to, the beam would not)
        for _ in range(3):
            shown = M.ease(shown, target, 0.05)
            seen.append(shown)
    assert all(b >= a for a, b in zip(seen, seen[1:])), "never goes back"
    assert seen[3] < 0.3, "it eases rather than jumping"
    for _ in range(200):
        shown = M.ease(shown, 1.0, 0.05)
    assert shown == pytest.approx(1.0)
    assert M.ease(0.7, 0.2, 0.05) == 0.7


def test_stalled_for_is_never_negative():
    w, M = model()
    assert M.stalled_for(10, 4) == 6 and M.stalled_for(3, 4) == 0 and M.stalled_for(10, None) == 0 and M.stalled_for(10, 0) == 0


def test_the_avatar_beam_condenses_on_arrival_and_dissolves_on_leaving_and_is_gone_at_the_end():
    w, M = model()
    arr = [M.beam("arrive", s) for s in (0, 0.25, 0.5, 0.75, 0.95, 1.0)]
    assert strictly([r.density for r in arr[:4]], True), "arriving condenses"
    assert arr[-1].done and arr[-1].alpha == 0 and not arr[0].done
    lv = [M.beam("leave", s) for s in (0, 0.2, 0.4, 0.6, 0.79)]
    assert strictly([r.density for r in lv], False), "leaving dissolves"
    assert M.beam("leave", 0.8).done
    assert max(r.band for r in arr) > 0.8, "a scan band sweeps through the middle of it"


# ============================================================================================================ the server: events -> NetworkVars
SERVER_FX = r"""
function T.fx()
  local o = {}
  for _, e in ipairs(MOCK.ents) do
    if e.class == "holodeck_found" then
      o[#o + 1] = { valid = e.valid, raw = e:GetFxStage(), stage = e:FxStageName() or "none", progress = e:GetFxProgress(), beat = e:GetFxBeat(), pending = e:FxPending() }
    end
  end
  return MOCK.json_encode(o)
end
function T.fx_say() return Holodeck.Mesh.FP.last_say or "" end
function T.fx_set(i, stage, frac, hb)
  local n = 0
  for _, e in ipairs(MOCK.ents) do
    if e.class == "holodeck_found" and e.valid then n = n + 1; if n == i then return e:FxSet(stage, frac, hb) end end
  end
end
"""


class FxGame(MeshGame):
    def __init__(self, **cv):
        super().__init__(**cv)
        self.w.rt.execute(SERVER_FX)
        self.progress_answered = set()

    def fx(self, valid_only=True):
        rows = json.loads(self.w.call("T.fx"))
        return [r for r in rows if r["valid"]] if valid_only else rows

    def progress_posts(self):
        return self.posts("/progress")

    def answer_progress(self, **obj):
        """Answer every unanswered /progress with `obj` (+ ok/key)."""
        n = 0
        for i in self.progress_posts():
            if i in self.progress_answered:
                continue
            self.progress_answered.add(i)
            body = {"ok": True, "key": self.body(i)["key"], **obj}
            self.w.rt.eval("function(i, code, body) MOCK.respond(i, code, body) end")(i, 200, json.dumps(body))
            n += 1
        return n

    def say_fx(self):
        return self.w.call("T.fx_say")


@pytest.fixture()
def fx(ix):
    return FxGame()


def test_a_new_found_object_with_a_mesh_enters_fetching_with_the_resolve_already_done(fx):
    p = summon(fx)
    (e,) = fx.fx()
    assert e["stage"] == "fetching" and e["pending"]
    assert e["progress"] == pytest.approx(0.08, abs=1e-6), "resolving (0 .. 0.08) is behind it: the /resolve is what produced the object"
    assert e["beat"] > 0


def test_a_card_without_a_mesh_never_waits_for_anything(ix):
    g = FxGame()
    p = g.player(ALICE, admin=True)
    g.summon(p, "add a fuel widget")
    (e,) = g.fx()
    assert e["stage"] == "none" and not e["pending"], "a plain box that was never waiting: no beam"


def test_fetch_progress_comes_from_the_bridges_byte_counts_and_only_ever_rises(fx):
    summon(fx)
    fx.w.advance(1.0)
    assert len(fx.progress_posts()) == 1, "one poll a second while it is pending"
    assert fx.body(fx.progress_posts()[0]) == {"key": skippy_bridge.mesh_key(TALL)}, "just the key; no player data goes with a poll"
    seen = []
    for got in (0, 200, 600, 400, 900):                      # (400 after 600 would be a regression the game must not show)
        fx.answer_progress(stage="fetching", bytes=got, total=1000, age_s=1)
        seen.append(fx.fx()[0]["progress"])
        fx.w.advance(1.0)
    assert seen[0] == pytest.approx(0.08) and seen[1] == pytest.approx(0.08 + 0.42 * 0.2)
    assert seen[2] == pytest.approx(0.08 + 0.42 * 0.6)
    assert seen[3] == seen[2], "a smaller count later never lowers it"
    assert seen[4] == pytest.approx(0.08 + 0.42 * 0.9)
    assert fx.fx()[0]["stage"] == "fetching"


def test_an_unknown_total_still_gives_a_rising_fraction_from_the_bytes(fx):
    summon(fx)
    seen = []
    for got in (100_000, 800_000, 3_000_000):
        fx.w.advance(1.0)
        fx.answer_progress(stage="fetching", bytes=got, total=0, age_s=1)
        seen.append(fx.fx()[0]["progress"])
    assert seen[0] < seen[1] < seen[2] < 0.5


def test_every_event_stamps_the_beat_and_a_poll_with_nothing_new_does_not(fx):
    summon(fx)
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=500, total=1000, age_s=1)
    b1 = fx.fx()[0]["beat"]
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=500, total=1000, age_s=2)           # the same bytes: no news
    assert fx.fx()[0]["beat"] == b1
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=700, total=1000, age_s=3)
    assert fx.fx()[0]["beat"] > b1


def test_packing_is_a_heartbeat_while_the_worker_lives_and_not_after_the_grace(fx):
    summon(fx)
    fx.w.advance(1.0)
    fx.answer_progress(stage="packing", bytes=1000, total=1000, age_s=1)
    e = fx.fx()[0]
    assert e["stage"] == "packing" and e["progress"] == pytest.approx(0.58)
    b = e["beat"]
    fx.w.advance(1.0)
    fx.answer_progress(stage="packing", bytes=1000, total=1000, age_s=5)
    assert fx.fx()[0]["beat"] > b, "still in packing: alive"
    b = fx.fx()[0]["beat"]
    fx.w.advance(1.0)
    fx.answer_progress(stage="packing", bytes=1000, total=1000, age_s=40)            # a worker that has been 'packing' for 40 s is not news
    assert fx.fx()[0]["beat"] == b


def test_queued_is_not_progress_and_does_not_refresh_the_beat(fx):
    summon(fx)
    b = fx.fx()[0]["beat"]
    fx.w.advance(1.0)
    fx.answer_progress(stage="queued", bytes=0, total=0, age_s=3)
    e = fx.fx()[0]
    assert e["beat"] == b and e["progress"] == pytest.approx(0.08), "waiting in line is honest silence: the beam may destabilise"


def test_garbage_from_the_progress_poll_changes_nothing(fx):
    summon(fx)
    before = fx.fx()[0]
    for body in ('{"ok": false}', "not json", '{"ok": true, "stage": "who knows"}', '{"ok": true, "stage": "fetching", "bytes": "x", "total": -5}'):
        fx.w.advance(1.0)
        for i in fx.progress_posts():
            if i not in fx.progress_answered:
                fx.progress_answered.add(i)
                fx.w.rt.eval("function(i, code, body) MOCK.respond(i, code, body) end")(i, 200, body)
    assert fx.fx()[0]["progress"] == before["progress"] and fx.fx()[0]["stage"] == "fetching"


def test_the_polls_are_not_counted_as_work_in_flight_and_stop_when_the_answer_comes(fx):
    summon(fx)
    for k in range(3):
        fx.w.advance(1.0)
        assert fx.w.rt.eval("Holodeck.thinking") == 1, "only the /mesh call itself counts, however many polls are out"
        fx.answer_progress(stage="fetching", bytes=100 * (k + 1), total=1000, age_s=k)
    n = len(fx.progress_posts())
    assert n >= 3
    fx.serve_mesh()
    fx.w.advance(5.0)
    assert len(fx.progress_posts()) == n, "no polls once the shape has arrived"


def test_the_whole_good_path_sending_follows_the_chunks_then_decoding_then_done(fx):
    raw, blob = big_payload()
    summon(fx)
    fx.register(blob, raw)
    obj = {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "bytes": len(blob), "crc32": __import__("zlib").crc32(blob),
           "b64": __import__("base64").b64encode(blob).decode()}
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=900, total=1000, age_s=1)
    fetching_end = fx.fx()[0]["progress"]
    fx.serve_mesh(lambda b: (200, obj))
    e = fx.fx()[0]
    assert e["stage"] == "sending" and e["progress"] >= fetching_end and e["progress"] == pytest.approx(0.74)
    seen = [e["progress"]]
    for _ in range(12):
        fx.w.advance(0.1)
        seen.append(fx.fx()[0]["progress"])
        if fx.fx()[0]["stage"] != "sending":
            break
    sending = [x for x in seen if x < 0.9]
    assert strictly(sending[:4], True) or len(set(sending)) >= 3, f"each chunk moves it: {seen}"
    assert all(b >= a for a, b in zip(seen, seen[1:])), f"monotone: {seen}"
    assert fx.fx()[0]["stage"] == "decoding", "the last chunk is out: waiting for the clients to decode"
    fx.w.advance(1.5)
    assert fx.fx()[0]["stage"] == "done" and fx.fx()[0]["progress"] == 1 and not fx.fx()[0]["pending"]


def test_a_second_object_of_a_model_already_on_its_way_shows_the_models_state_not_zero(fx):
    p = summon(fx)
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=800, total=1000, age_s=1)
    fx.summon(p, "add a garden gnome")
    rows = fx.fx()
    assert len(rows) == 2
    assert rows[1]["progress"] == pytest.approx(rows[0]["progress"]), "joined at the truth"
    assert len(fx.mesh_posts()) == 1, "one download for both"


def test_an_object_of_a_model_the_server_already_has_materialises_at_once_and_settles(fx):
    summon(fx)
    fx.serve_mesh()
    fx.w.advance(3)
    assert fx.fx()[0]["stage"] == "done"
    p = fx.player(ALICE)
    fx.w.rt.execute("Holodeck.Mesh.FP.DONE_AFTER = 50")
    fx.ask(p, "add a garden gnome")
    fx.serve()
    fx.w.advance(0.3)
    rows = fx.fx()
    assert len(rows) == 2
    assert rows[1]["stage"] == "decoding" and rows[1]["pending"], "the clients that hold the shape draw it with the materialise"
    fx.w.advance(51)
    assert fx.fx()[1]["stage"] == "done"


def test_progress_never_goes_back_and_a_failed_object_stays_failed(fx):
    summon(fx)
    assert fx.w.call("T.fx_set", 1, "packing", 0.5, False) is True
    assert fx.w.call("T.fx_set", 1, "fetching", 1.0, False) is False, "a stage never goes back"
    assert fx.w.call("T.fx_set", 1, "packing", 0.2, False) is False, "nor does the progress inside one"
    assert fx.w.call("T.fx_set", 1, "failed", 0, False) is True
    assert fx.w.call("T.fx_set", 1, "done", 1, False) is False, "failed is final"
    assert fx.fx()[0]["stage"] == "failed"


# ------------------------------------------------------------------------------------------------ giving up
def test_a_bridge_error_collapses_the_beam_removes_the_placeholder_and_skippy_says_why(fx):
    summon(fx)
    fx.serve_mesh(lambda b: (502, {"ok": False, "error": "boom"}))
    (e,) = fx.fx()
    assert e["stage"] == "failed", "immediately, not after the timeout"
    t = fx.w.rt.eval("CurTime()")
    assert abs(e["beat"] - t) < 1e-6, "the failure is stamped now: every client plays the collapse from this moment"
    say = fx.say_fx()
    assert say.startswith("I could not bring in the garden gnome:") and "bridge could not fetch it" in say and "took the placeholder down" in say
    assert fx.w.call("T.say") == say, "it is what Skippy says in chat"
    fx.w.advance(1.0)
    assert len(fx.fx()) == 1, "the collapse is still playing"
    fx.w.advance(1.0)
    assert fx.fx() == [], "then it is gone"
    assert fx.count() == 0, "and the registry no longer counts it"


@pytest.mark.parametrize("answer,expect", [
    (lambda b: (200, {"ok": True, "key": "0" * 16, "format": "svm1-lzma", "b64": "AAAA"}), "not a model"),
    (lambda b: (200, {"ok": True, "key": skippy_bridge.mesh_key(TALL), "format": "svm1-lzma", "b64": "A" * (400 * 1024)}), "too big"),
    (lambda b: (500, {"ok": False}), "bridge could not fetch it"),
])
def test_each_kind_of_failure_gets_its_own_reason_in_the_sentence(fx, answer, expect):
    summon(fx)
    fx.serve_mesh(answer)
    assert fx.fx()[0]["stage"] == "failed" and expect in fx.say_fx()


def test_a_busy_bridge_is_waited_for_not_given_up_on_and_the_beam_shows_the_wait(fx):
    """One download at a time: a second model asks, is told 503, and queues. Silence (no progress) is what the player sees: the beam destabilises."""
    summon(fx)
    busy = lambda b: (503, {"ok": False, "error": "busy; try again shortly"})                 # noqa: E731
    fx.serve_mesh(busy)
    assert fx.fx()[0]["stage"] == "fetching", "not a failure"
    for _ in range(3):
        fx.w.advance(2.1)
        fx.serve_mesh(busy)
    assert len(fx.mesh_posts()) == 4, "it asked again every couple of seconds"
    assert fx.fx()[0]["stage"] == "fetching" and fx.w.call("T.mesh_state") == "pending"
    fx.w.advance(2.1)
    fx.serve_mesh()                                                         # its turn came
    fx.w.advance(3)
    assert fx.fx()[0]["stage"] == "done" and fx.visuals() == [{"visual": "mesh", "mesh": 1}]


def test_a_bridge_that_stays_busy_ends_in_the_timeout_failure(fx):
    fx.w.rt.eval("function() GetConVar('holodeck_found_timeout'):SetValue(20) end")()
    summon(fx)
    busy = lambda b: (503, {"ok": False})                                                       # noqa: E731
    for _ in range(14):
        fx.serve_mesh(busy)
        fx.w.advance(2.1)
    assert fx.fx(valid_only=False)[0]["stage"] == "failed" and "slot was busy" in fx.say_fx(), "after the window, the last answer is the reason"
    n = len(fx.mesh_posts())
    fx.w.advance(10)
    assert len(fx.mesh_posts()) == n, "it stopped asking"


def test_a_damaged_download_is_said_to_be_damaged(fx):
    raw, blob = big_payload(200)
    summon(fx)
    fx.register(blob, raw)
    fx.serve_mesh(_answer_with(blob, crc32=__import__("zlib").crc32(blob) ^ 1))
    assert fx.fx()[0]["stage"] == "failed" and "damaged" in fx.say_fx()


def test_silence_past_the_timeout_gives_up_collapses_removes_and_says_so(fx):
    summon(fx)                                                         # (12 s pass while the objects materialise)
    fx.w.advance(55)
    (e,) = fx.fx()
    assert e["stage"] == "fetching", "75 s is the default: a slow transfer is not given up on early"
    fx.w.advance(12)
    rows = fx.fx(valid_only=False)
    assert rows[0]["stage"] == "failed"
    assert "took too long" in fx.say_fx() or "too long" in fx.say_fx()
    fx.w.advance(2.5)
    assert fx.fx() == []
    assert fx.w.call("T.mesh_state") == "failed"


def test_the_timeout_is_a_convar(fx):
    fx.w.rt.eval("function() GetConVar('holodeck_found_timeout'):SetValue(20) end")()
    summon(fx)
    fx.w.advance(4)
    assert fx.fx()[0]["stage"] == "fetching"
    fx.w.advance(6)
    assert fx.fx(valid_only=False)[0]["stage"] == "failed"


def test_the_stall_window_is_a_replicated_convar_the_clients_read(fx):
    cv = fx.w.rt.eval("function() return GetConVar('holodeck_found_stall') end")()
    assert cv.GetFloat(cv) == 6 and cv.flags & 16, "FCVAR_REPLICATED: the clients read the server's value"


def test_three_objects_of_one_model_fail_together_with_one_sentence(fx):
    summon(fx, "add three garden gnomes")
    assert len(fx.fx()) == 3
    fx.serve_mesh(lambda b: (502, {"ok": False}))
    assert [r["stage"] for r in fx.fx()] == ["failed"] * 3
    assert fx.say_fx().startswith("I could not bring in 3 of the garden gnome")
    fx.w.advance(2.2)
    assert fx.fx() == []


def test_a_late_answer_after_the_timeout_does_not_resurrect_anything(fx):
    summon(fx)
    fx.w.advance(70)
    fx.w.advance(2.5)
    assert fx.fx() == []
    fx.serve_mesh()
    fx.w.advance(3)
    assert fx.fx() == [] and fx.w.call("T.mesh_state") == "ready", "the model is kept for the next one; nothing comes back"


def test_the_owner_leaving_is_not_a_crash(fx):
    p = summon(fx)
    fx.w.rt.eval("function(p) p.valid = false end")(p)
    fx.serve_mesh(lambda b: (502, {"ok": False}))
    fx.w.advance(3)
    assert fx.fx() == []


# ------------------------------------------------------------------------------------------------ late joiners
def test_a_late_joiner_reads_the_same_three_networked_values_and_so_sees_the_same_state(fx):
    summon(fx)
    fx.w.advance(1.0)
    fx.answer_progress(stage="fetching", bytes=700, total=1000, age_s=1)
    fx.w.advance(2.0)                                                   # two quiet seconds since the last event
    srv = fx.fx()[0]
    # the values a joining client is sent are the entity's NetworkVars; here they are copied to a client entity (and its clock set to the server's)
    w = fx_client()
    e = pending_entity(w, "sending", 0)                                 # (any state: it is overwritten below)
    w.advance(fx.w.rt.eval("CurTime()") - w.rt.eval("CurTime()"))
    for setter, val in (("SetFxStage", srv["raw"]), ("SetFxProgress", srv["progress"]), ("SetFxBeat", srv["beat"])):
        getattr(e, setter)(e, val)
    assert e.FxStageName(e) == "fetching" and e.FxPending(e)
    f = frame(w, e, 0.0)
    assert e.fx.st.shown == pytest.approx(srv["progress"]), "it starts at the truth"
    assert f["n"] > 0
    text, col = w.rt.eval("function(e) return e:StatusLine() end")(e)
    assert text.startswith("fetching 3") and (col.r, col.g, col.b) == (0, 204, 255), text         # 8 + 42 * 0.7 = 37 %: healthy after 2 s of quiet


# ============================================================================================================ the bridge
def test_the_progress_of_a_mesh_request_is_reported_while_it_downloads_and_packs(ix, monkeypatch, tmp_path):
    from test_gmod_found_mesh import Fetch, box_routes
    key = skippy_bridge.mesh_key(TALL)
    assert skippy_bridge.mesh_progress(key)["stage"] == "idle"
    seen = []
    routes = box_routes("garden_gnome")
    inner = Fetch(routes)

    def spying(url, *, max_bytes):
        seen.append((url.rsplit("/", 1)[-1], dict(skippy_bridge.mesh_progress(key))))
        return inner(url, max_bytes=max_bytes)

    monkeypatch.setattr(skippy_bridge, "FOUND_FETCHER", spying)
    monkeypatch.setattr(skippy_bridge, "MESH_CACHE_DIR", tmp_path / "a2")
    monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
    monkeypatch.setattr(skippy_bridge, "MESH_PROGRESS", {})
    skippy_bridge.mesh_payload(TALL, key)
    listing, gltf, binfile = (s[1] for s in seen)
    assert listing["stage"] == "fetching" and listing["bytes"] == 0 and listing["total"] == 0
    assert gltf["bytes"] > 0 and gltf["total"] > gltf["bytes"], "the listing told it how much the geometry needs: gltf + bin, not the texture"
    assert binfile["bytes"] > gltf["bytes"] and binfile["total"] == gltf["total"]
    gl = len(routes["https://dl.example/garden_gnome_1k.gltf"])
    bn = len(routes["https://dl.example/garden_gnome.bin"])
    ls = len(routes["https://api.polyhaven.com/files/garden_gnome"])
    assert gltf["total"] == ls + gl + bn, "the 5-byte texture is not part of what it will fetch"
    after = skippy_bridge.mesh_progress(key)
    assert after["stage"] == "done" and after["bytes"] == ls + gl + bn


def test_a_failure_in_the_bridge_is_reported_as_failed_with_the_reason(ix, monkeypatch, tmp_path):
    def boom(url, *, max_bytes):
        raise OSError("connection reset")
    monkeypatch.setattr(skippy_bridge, "FOUND_FETCHER", boom)
    monkeypatch.setattr(skippy_bridge, "MESH_CACHE_DIR", tmp_path / "a3")
    monkeypatch.setattr(skippy_bridge, "MESH_PACKED", {})
    monkeypatch.setattr(skippy_bridge, "MESH_PROGRESS", {})
    key = skippy_bridge.mesh_key(TALL)
    with pytest.raises(Exception):
        skippy_bridge.mesh_payload(TALL, key)
    p = skippy_bridge.mesh_progress(key)
    assert p["stage"] == "failed" and "connection reset" in p["error"]


def test_progress_is_token_guarded_validated_budgeted_and_leaves_mesh_alone(ix, fetch, monkeypatch):
    monkeypatch.setattr(skippy_bridge, "MESH_PROGRESS", {})
    h = Http(monkeypatch)
    try:
        key = skippy_bridge.mesh_key(TALL)
        assert h.post("/progress", {"key": key}, token="wrong")[0] == 401
        code, got = h.post("/progress", {"key": key})
        assert code == 200 and got["ok"] and got["stage"] == "idle"
        assert h.post("/progress", {"key": "../../x"})[0] == 400 and h.post("/progress", {})[0] == 400
        code, got = h.post("/mesh", {"card": TALL, "key": key, "who": "alice"})
        assert code == 200 and set(got) == {"ok", "key", "card", "format", "bytes", "crc32", "tris", "verts", "bbox_m", "b64"}, "/mesh answers exactly as before"
        assert h.post("/progress", {"key": key})[1]["stage"] == "done"
        codes = [h.post("/progress", {"key": key})[0] for _ in range(skippy_bridge.PER_CALLER_PER_MINUTE["progress"] + 400)]
        assert 429 in codes, "a budget, like every other path"
    finally:
        h.close()


def test_the_bounded_fetch_reports_every_chunk_as_it_arrives(monkeypatch):
    class Resp:
        def __init__(self, n):
            self.n, self.headers = n, {}
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def read(self, k):
            k = min(k, self.n)
            self.n -= k
            return b"x" * k

    class Opener:
        def open(self, req, timeout=None): return Resp(700_000)

    monkeypatch.setattr(skippy_bridge.urllib.request, "build_opener", lambda *a: Opener())
    got = []
    f = skippy_bridge.bounded_fetch(skippy_bridge.time.monotonic() + 30, on_bytes=got.append)
    f("https://dl.polyhaven.org/x.bin")
    assert sum(got) == 700_000 and len(got) >= 3, "told per 256 KB chunk, not at the end"


# ============================================================================================================ the client drawing
FX_FAKES = r"""
FXR = { sprites = {}, quads = {}, clips = {}, clipping = false, wires = {}, boxes = {} }
render.DrawSprite = function(pos, w, h, col) FXR.sprites[#FXR.sprites + 1] = { x = pos.x, y = pos.y, z = pos.z, w = w, r = col.r, g = col.g, b = col.b, a = col.a } end
render.DrawQuad = function(a, b, c, d, col) FXR.quads[#FXR.quads + 1] = { z = a.z, a = col.a, r = col.r, g = col.g, b = col.b } end
render.EnableClipping = function(on) FXR.clipping = on end
render.PushCustomClipPlane = function(n, d) FXR.clips[#FXR.clips + 1] = { n = { x = n.x, y = n.y, z = n.z }, d = d } end
render.PopCustomClipPlane = function() end
render.DrawWireframeBox = function(pos, ang, mn, mx, col, z) FXR.wires[#FXR.wires + 1] = { col = { r = col.r, g = col.g, b = col.b, a = col.a }, pos = { x = pos.x, y = pos.y, z = pos.z }, top = mx.z } end
local plain_box = render.DrawBox
render.DrawBox = function(pos, ang, mn, mx, col, z) FXR.boxes[#FXR.boxes + 1] = { col = { r = col.r, g = col.g, b = col.b, a = col.a }, top = mx.z }; plain_box(pos, ang, mn, mx, col, z) end
function T.reset() FXR.sprites, FXR.quads, FXR.clips, FXR.wires, FXR.boxes = {}, {}, {}, {}, {}; RENDER.boxes, RENDER.wires = {}, {}; RENDER2.draws = 0 end
function T.frame(e, dt)
  MOCK.advance(dt)
  T.reset()
  e:Draw()
  local n, sx, sy, sz, inside, maxz, minz, r, g, b, a = #FXR.sprites, 0, 0, 0, 0, -1e9, 1e9, 0, 0, 0, 0
  local mn, mx = e:GetBoxMins(), e:GetBoxMaxs()
  local cx, cy = e.pos.x, e.pos.y
  local ca, sa = math.cos(math.rad(-e.ang.y)), math.sin(math.rad(-e.ang.y))
  for _, s in ipairs(FXR.sprites) do
    local wx, wy = s.x - cx, s.y - cy
    local dx, dy = wx * ca - wy * sa, wx * sa + wy * ca                 -- into the entity's frame
    sx = sx + math.sqrt(dx * dx + dy * dy); sz = sz + s.z
    if s.z > maxz then maxz = s.z end
    if s.z < minz then minz = s.z end
    r, g, b, a = r + s.r, g + s.g, b + s.b, a + s.a
    if math.abs(dx) <= (mx.x - mn.x) / 2 + 2 and math.abs(dy) <= (mx.y - mn.y) / 2 + 2 and s.z >= mn.z - 2 and s.z <= mx.z + 2 then inside = inside + 1 end
  end
  local d = math.max(n, 1)
  return MOCK.json_encode({ n = n, radius = sx / d, inside = inside / d, maxz = n > 0 and maxz or 0, minz = n > 0 and minz or 0, r = r / d, g = g / d, b = b / d, a = a / d,
    quads = #FXR.quads, clips = #FXR.clips, mesh = RENDER2.draws, wires = #FXR.wires, wire_a = FXR.wires[1] and FXR.wires[1].col.a or 0,
    boxes = #FXR.boxes, clipd = FXR.clips[1] and FXR.clips[1].d or nil, clipnz = FXR.clips[1] and FXR.clips[1].n.z or nil,
    quad_z = FXR.quads[1] and FXR.quads[1].z or nil })
end
"""

BOX_H_HU = 62.99


def fx_client():
    w = mesh_client()
    w.rt.execute(FX_FAKES)
    w.rt.execute("CreateConVar('holodeck_found_stall', '6', 16)")      # (the server's replicated convar)
    w.advance(50.0)                                                    # (a real server has been up a while: a beat of 0 means "none")
    return w


def pending_entity(w, stage="fetching", frac=0.5, x=100.0):
    e = client_entity(w)
    w.rt.eval("function(e) e:SetVisual('box') end")(e)               # (not yet a mesh: the box's beam)
    w.rt.eval("function(e, s, f) e:FxSet(s, f) end")(e, stage, frac)
    return e


def frame(w, e, dt=0.0):
    return json.loads(w.call("T.frame", e, dt))


def at_progress(stage, frac, n=3):
    w = fx_client()
    e = pending_entity(w, stage, frac)
    f = frame(w, e, 0.0)
    return w, e, f


def test_a_pending_object_draws_a_beam_not_the_plain_box():
    w, e, f = at_progress("fetching", 0.5)
    assert f["n"] > 20 and f["boxes"] <= 1 and f["wires"] == 1
    assert all(b.col.a != 60 for b in [w.rt.eval("FXR.boxes[1]")] if b), "the translucent placeholder fill (alpha 60) is not what is drawn"


def test_the_beam_gets_visibly_more_coherent_as_the_pipeline_progresses():
    rows = []
    for stage, frac in [("fetching", 0.0), ("fetching", 1.0), ("packing", 1.0), ("sending", 1.0), ("decoding", 1.0)]:
        w, e, f = at_progress(stage, frac)
        rows.append(f)
    # more sparkles, held closer to the object (inside its box), a sharper outline, the silhouette filling in
    assert strictly([r["n"] for r in rows], True), [r["n"] for r in rows]
    inside = [r["inside"] for r in rows]
    assert inside[-1] > 0.9 and inside[0] < 0.55 and inside[-1] > inside[0] + 0.4, f"inside the footprint box: {inside}"
    assert rows[-1]["wire_a"] > 3 * rows[0]["wire_a"] and rows[-1]["wire_a"] > 150
    assert all(r["n"] <= 220 for r in rows), "bounded"
    assert rows[0]["radius"] > rows[-1]["radius"] * 0.9 or rows[0]["inside"] < rows[-1]["inside"]


def test_the_shown_progress_eases_but_a_late_joiner_starts_at_the_truth():
    w = fx_client()
    e = pending_entity(w, "packing", 1.0)
    f0 = frame(w, e, 0.0)
    assert e.fx.st.shown == pytest.approx(0.74), "no ramp from zero for someone who arrives mid-way"
    w.rt.eval("function(e) e:FxSet('sending', 1.0) end")(e)
    f1 = frame(w, e, 0.05)
    assert 0.74 < e.fx.st.shown < 0.92, "then it eases toward the next real value"


def test_silence_makes_the_beam_flicker_dim_and_drift_toward_amber():
    w, e, healthy = at_progress("fetching", 0.7)
    w.advance(8.0)                                                      # no events for 8 s > the 6 s window
    f = frame(w, e, 0.0)
    assert f["r"] > healthy["r"] + 60 and f["g"] < healthy["g"] - 20, f"amber: {healthy['r']:.0f},{healthy['g']:.0f} -> {f['r']:.0f},{f['g']:.0f}"
    assert f["n"] < healthy["n"] and f["a"] < healthy["a"], "fewer, dimmer sparkles"
    assert f["wire_a"] < healthy["wire_a"], "the outline loses its crispness"
    # news restores it (a stage can add)
    w.rt.eval("function(e) e:FxSet('packing', 0.5, false) end")(e)
    g = frame(w, e, 0.0)
    assert g["g"] > f["g"] + 20 and g["n"] > f["n"]


def test_the_status_line_says_what_is_going_on_and_goes_amber_when_quiet():
    w, e, _ = at_progress("fetching", 0.5)
    text, col = w.rt.eval("function(e) return e:StatusLine() end")(e)
    assert text.startswith("fetching") and "%" in text and (col.r, col.g, col.b) == (0, 204, 255)
    w.advance(5.0)
    text, col = w.rt.eval("function(e) return e:StatusLine() end")(e)
    assert "no news" in text and (col.r, col.g, col.b) == (255, 170, 0)
    w.rt.eval("function(e) e:FxSet('failed', 0) end")(e)
    text, col = w.rt.eval("function(e) return e:StatusLine() end")(e)
    assert "failed" in text and (col.r, col.g, col.b) == (255, 68, 68)


def test_failure_collapses_flickers_reddens_scatters_and_ends():
    w, e, before = at_progress("packing", 0.5)
    w.rt.eval("function(e) e:FxSet('failed', 0) end")(e)
    top = 63 + 0.0
    frames = [frame(w, e, dt) for dt in (0.0, 0.3, 0.3, 0.3, 0.3)]               # 0, .3, .6, .9, 1.2 s after
    mx = [f["maxz"] for f in frames]
    assert mx[0] > mx[-1] + 10 and all(b <= a + 4 for a, b in zip(mx, mx[1:])), f"the top comes down: {mx}"
    assert frames[-1]["r"] > 200 and frames[-1]["g"] < 150, "red"
    assert frames[0]["g"] > frames[-1]["g"]
    assert frames[-1]["radius"] > frames[0]["radius"], "scattering outward"
    end = frame(w, e, 0.6)                                                        # 1.8 s: over
    assert end["n"] == 0 and end["wires"] == 0 and end["boxes"] == 0, "nothing left to draw"


def test_the_materialise_reveals_the_mesh_floor_up_behind_a_band_then_the_beam_fades():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    w.rt.execute(FX_FAKES)
    e = client_entity(w)
    w.rt.eval("function(e) e:SetVisual('box') end")(e)
    w.rt.eval("function(e) e:FxSet('sending', 0.5) end")(e)
    f = frame(w, e, 0.0)
    assert f["n"] > 0 and f["mesh"] == 0, "pending: a beam, no mesh yet"
    w.rt.eval("function(e) e:SetVisual('mesh', { id = 1 }) end")(e)
    feed(w, make_msgs(blob))
    clip = []
    for dt in (0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1):
        f = frame(w, e, dt)
        assert f["mesh"] >= 1, "the mesh is drawn from the first frame of the arrival"
        if f["clipd"] is not None:
            clip.append(f["clipd"])
    assert f["clipnz"] == pytest.approx(-1, abs=1e-6), "the plane keeps what is below the band"
    assert strictly(clip[:4], False) or clip[0] != clip[3], f"the plane moves up (distance falls as the normal points down): {clip}"
    assert f["quads"] > 0, "the band is drawn"
    end = frame(w, e, 0.4)                                                        # 1.0 s after: over
    assert end["clips"] == 0 and end["n"] == 0 and end["quads"] == 0 and end["mesh"] == 1, "plain mesh, no beam"


def test_an_object_that_arrived_long_ago_is_not_materialised_again_for_a_late_joiner():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    w.rt.execute(FX_FAKES)
    e = client_entity(w)
    w.rt.eval("function(e) e:FxSet('done', 1) end")(e)
    feed(w, make_msgs(blob))
    f = frame(w, e, 0.0)
    assert f["mesh"] == 1 and f["clips"] == 0 and f["n"] == 0, "straight to the mesh"


def test_a_late_joiner_who_gets_the_mesh_mid_transfer_does_see_the_materialise():
    raw, blob = box_blob()
    w = client_with_payload(raw, blob)
    w.rt.execute(FX_FAKES)
    e = client_entity(w)
    w.rt.eval("function(e) e:FxSet('sending', 0.5) end")(e)
    feed(w, make_msgs(blob))
    f = frame(w, e, 0.0)
    assert f["mesh"] == 1 and f["clips"] == 1


def test_a_plain_box_that_never_waited_is_exactly_as_before():
    w = fx_client()
    e = client_entity(w)
    w.rt.eval("function(e) e:SetVisual('box') end")(e)
    f = frame(w, e, 0.0)
    assert f["n"] == 0 and f["boxes"] == 1 and f["wires"] == 1


def test_the_effect_can_be_switched_off_and_the_old_box_comes_back():
    w, e, _ = at_progress("fetching", 0.5)
    w.rt.eval("function() GetConVar('holodeck_fx'):SetValue(0) end")()
    f = frame(w, e, 0.0)
    assert f["n"] == 0 and f["boxes"] == 1


def test_far_away_nothing_is_drawn_and_distance_thins_the_sparkles():
    counts = []
    for dist in (100, 2000, 4500):
        w = fx_client()
        e = pending_entity(w, "packing", 1.0)
        w.rt.eval("function(d) MOCK.me.pos = Vector(d, 0, 0) end")(dist + 100.0)       # (the entity stands at x = 100)
        counts.append(frame(w, e, 0.0)["n"])
    assert counts[0] > counts[1] > 0 and counts[2] == 0, f"near, mid, beyond FAR: {counts}"


def test_the_particle_budget_is_bounded_whatever_the_progress():
    for stage, frac in [("fetching", 0), ("sending", 1), ("decoding", 1)]:
        w, e, f = at_progress(stage, frac)
        assert f["n"] <= 220


# ============================================================================================================ Skippy's beams
def test_the_avatars_hooks_become_one_small_beam_message_each():
    w = World(cvars={}, load=())
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/fx/beam_net.lua")')
    w.rt.execute("""
      local bot = MOCK.newplayer('bot', { pos = Vector(1, 2, 3) })
      hook.Run("SkippyAvatarLeave", bot, Vector(10, 20, 30))
      hook.Run("SkippyAvatarArrive", bot, Vector(10, 20, 30), Vector(40, 50, 60))
      hook.Run("SkippyAvatarArrive", bot, nil, Vector(7, 8, 9))
    """)
    nets = w.rt.eval("MOCK.nets")
    msgs = [nets[i] for i in range(1, len(nets) + 1)]
    assert [m.name for m in msgs] == ["mentat_fx_beam"] * 3
    assert [m.writes[1] for m in msgs] == [False, True, True], "leave, arrive, first arrival"
    assert (msgs[0].vector.x, msgs[1].vector.x, msgs[2].vector.x) == (10, 40, 7), "leave at where he was; arrive where he is now"
    assert all(m.kind == "broadcast" for m in msgs)


def beam_client():
    w = fx_client()
    w.rt.execute("MOCK.me = MOCK.newplayer('1', { pos = Vector(0, 0, 0) })")
    return w


def test_his_body_waits_for_the_arrival_beam_and_not_for_the_leaving_one():
    w = beam_client()
    FX = w.rt.eval("Mentat.FX")
    assert FX.avatar_hidden() is False
    w.rt.execute("Mentat.FX.add_beam('leave', Vector(50, 0, 0))")
    assert FX.avatar_hidden() is False, "leaving does not hide a body that is already gone"
    w.rt.execute("Mentat.FX.add_beam('arrive', Vector(0, 0, 0))")
    assert FX.avatar_hidden() is True
    w.advance(0.5)
    assert FX.avatar_hidden() is True
    w.advance(0.55)
    assert FX.avatar_hidden() is False, "about a second"


def test_the_beams_draw_sparkles_at_the_right_place_and_remove_themselves():
    w = beam_client()
    w.rt.execute("Mentat.FX.add_beam('arrive', Vector(300, 0, 0))")
    w.rt.execute("MOCK.hooks.PostDrawTranslucentRenderables['Mentat.FXBeams'](nil, false)")
    r = w.rt.eval("FXR")
    assert len(r.sprites) > 10
    xs = [r.sprites[i].x for i in range(1, len(r.sprites) + 1)]
    assert 270 < sum(xs) / len(xs) < 330, "around the arrival point"
    w.advance(1.1)
    w.rt.execute("FXR.sprites = {}; MOCK.hooks.PostDrawTranslucentRenderables['Mentat.FXBeams'](nil, false)")
    assert len(w.rt.eval("FXR").sprites) == 0 and len(w.rt.eval("Mentat.FX.beams")) == 0


def test_the_arrival_beam_follows_him_while_it_plays():
    w = beam_client()
    w.rt.execute("BOT = MOCK.newplayer('bot', { pos = Vector(100, 0, 0) }); Mentat.FX.add_beam('arrive', Vector(100, 0, 0), BOT)")
    w.advance(0.5)
    w.rt.execute("BOT.pos = Vector(100, 400, 0); FXR.sprites = {}; MOCK.hooks.PostDrawTranslucentRenderables['Mentat.FXBeams'](nil, false)")
    r = w.rt.eval("FXR")
    ys = [r.sprites[i].y for i in range(1, len(r.sprites) + 1)]
    assert 360 < sum(ys) / len(ys) < 440


def test_with_the_effect_off_the_body_is_never_held_back():
    w = beam_client()
    w.rt.execute("Mentat.FX.add_beam('arrive', Vector(0, 0, 0)); GetConVar('holodeck_fx'):SetValue(0)")
    assert w.rt.eval("Mentat.FX.avatar_hidden()") is False


def test_the_avatar_client_consults_the_beam_before_it_draws_him():
    src = (ROOT / "stage" / "gmod" / "addon" / "lua" / "mentat" / "avatar_client.lua").read_text(encoding="utf-8")
    assert 'include("mentat/fx/transporter.lua")' in src and "if FX.avatar_hidden() then return end" in src
    i = src.index("local function draw()")
    assert src.index("FX.avatar_hidden()", i) < src.index("local real =", i), "before anything is drawn"
