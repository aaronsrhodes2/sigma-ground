"""Offline tests for the GMod engine experiments (stage/gmod/experiments).

Nothing here starts the game or calls the live Mentat. Mentat's answers come from the committed cache
(`mentat_cache.json`), the engine's answers from six recorded runs (`results/S0N.json.gz`, taken on the dedicated
server). The tests check that

  * every experiment spec builds from the cache, deterministically, with provenance for each prediction, and that the
    committed generated Lua specs are the ones the builder produces today;
  * the independent grader passes each recorded run, and fails when a run is stale, unfinished, mis-placed, or when the
    engine's physics is changed (a fall with 5 % less gravity);
  * the grader's numeric helpers do what they say;
  * every Lua file of the addon at least parses (LuaJIT via `lupa`, skipped when it is not installed).
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[3] / "stage" / "gmod" / "experiments"
LUA_ROOT = EXP.parent / "addon" / "lua"
SIDS = ["S01", "S02", "S03", "S04", "S05", "S06"]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import fit  # noqa: E402
import grade  # noqa: E402
import luaout  # noqa: E402
import specs  # noqa: E402


def _load_build():
    """build.py by path: the bare name `build` is also the PyPA packaging module, so do not import it by name."""
    spec = importlib.util.spec_from_file_location("gmod_exp_build", EXP / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("MENTAT_OFFLINE", "1")          # a cache miss is an error, never a live call
    monkeypatch.delenv("MENTAT_REFRESH", raising=False)


def _result(sid):
    with gzip.open(EXP / "results" / f"{sid}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ specs
@pytest.mark.parametrize("sid", SIDS)
def test_spec_builds_offline_deterministically_with_provenance(sid):
    a, b = specs.build(sid), specs.build(sid)
    assert a["sha"] == b["sha"]
    assert a["id"] == sid and a["props"] and a["timeline"]["run_s"] > 0
    assert a["provenance"], "every spec must carry the Mentat calls behind its predictions"
    for item in a["provenance"]:
        assert item["tool"] and "args" in item, item         # procedure tools carry no top-level source; the call itself is the record
    ids = [p["id"] for p in a["props"]]
    assert len(ids) == len(set(ids)), "prop ids must be unique"


@pytest.mark.parametrize("sid", SIDS)
def test_committed_lua_spec_is_what_the_builder_produces(sid):
    committed = (LUA_ROOT / "mentat" / "exp_specs" / f"{sid.lower()}.lua").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == _load_build().render(sid), (
        f"{sid.lower()}.lua is stale: run python stage/gmod/experiments/build.py {sid}")


@pytest.mark.parametrize("sid", SIDS)
def test_a_recorded_run_belongs_to_the_current_spec(sid):
    assert _result(sid)["spec_sha"] == specs.build(sid)["sha"], f"re-record {sid}: the spec changed since the stored run"


def test_at_most_one_custom_surface_per_scene():
    """The engine honours only the first custom surface a session adds (measured); a spec asking for two cannot work."""
    for sid in SIDS:
        names = {json.dumps(p["surface"], sort_keys=True) for p in specs.build(sid)["props"] if isinstance(p.get("surface"), dict)}
        assert len(names) <= 1, f"{sid} asks for {len(names)} custom surfaces"


# ------------------------------------------------------------------------------------------------ grading recorded runs
@pytest.mark.parametrize("sid", SIDS)
def test_the_grader_passes_the_recorded_run(sid):
    _spec, rep = grade.grade(sid, _result(sid))
    assert not rep.fails, rep.fails
    assert any(name == "samples recorded" for name, _v, _n in rep.rows)


def test_a_stale_result_is_rejected():
    res = _result("S03")
    res["spec_sha"] = "0000000000000000"
    _spec, rep = grade.grade("S03", res)
    assert any("recorded against spec" in f for f in rep.fails)


def test_a_result_from_another_run_is_rejected():
    _spec, rep = grade.grade("S03", _result("S03"), run_id="not-this-run")
    assert any("STALE RESULT" in f for f in rep.fails)


def test_an_interrupted_run_is_rejected():
    res = _result("S03")
    res["finished"] = False
    _spec, rep = grade.grade("S03", res)
    assert any("not marked finished" in f for f in rep.fails)


def test_a_run_that_reported_an_error_is_rejected():
    res = _result("S03")
    res["error"] = "ents.Create('mentat_prop') failed"
    _spec, rep = grade.grade("S03", res)
    assert any("reported an error" in f for f in rep.fails)


def test_a_misplaced_body_is_caught():
    res = _result("S03")
    res["samples"][0]["p"][0][0] += 1e-3
    _spec, rep = grade.grade("S03", res)
    assert "bodies were not placed where the spec says" in rep.fails


def test_a_wrong_engine_mass_is_caught():
    res = _result("S03")
    i = next(k for k, p in enumerate(res["props"]) if not p["static"])
    res["props"][i]["mass_kg"] *= 1.01
    _spec, rep = grade.grade("S03", res)
    assert any("engine mass" in f for f in rep.fails)


def test_a_surface_the_engine_did_not_apply_is_caught():
    res = _result("S03")
    res["props"][0]["engine_surface"]["friction"] = 0.8       # the engine fell back to a default surface
    _spec, rep = grade.grade("S03", res)
    assert any("did not apply a requested surface" in f for f in rep.fails)


def test_a_tilted_body_is_caught():
    res = _result("S03")
    res["samples"][1]["p"][1][9:12] = [0.2, 0.0, 0.9798]       # the up vector, 11.5 degrees off
    _spec, rep = grade.grade("S03", res)
    assert "bodies were not oriented as the spec says" in rep.fails


def _stub_mentat(tool, args):
    """Textbook answers for the tools S01 uses, so a PERTURBED run can be graded offline. Test-only: the real grader
    always asks Mentat; this exists to prove the grader notices when the engine's physics is wrong."""
    g = args.get("g_m_s2", 9.80665)
    if tool == "free_fall_time":
        return {"value": math.sqrt(2 * args["height_m"] / g)}
    if tool == "free_fall_velocity":
        return {"value": math.sqrt(2 * g * args["height_m"])}
    if tool == "kinetic_energy":
        return {"value": 0.5 * args["mass_kg"] * args["velocity_m_s"] ** 2}
    if tool == "gravitational_potential_energy":
        return {"value": args["mass_kg"] * g * args["height_m"]}
    if tool == "momentum":
        return {"value": args["mass_kg"] * args["velocity_m_s"]}
    raise AssertionError(f"stub has no {tool}")


def test_the_grader_notices_when_the_engine_falls_with_the_wrong_gravity(monkeypatch):
    monkeypatch.setattr(grade, "call", _stub_mentat)
    _spec, rep = grade.grade("S01", _result("S01"))
    assert not rep.fails, "the stub must agree with the recorded run, or this test proves nothing"
    res = _result("S01")
    for row in res["samples"]:
        if row["k"] >= 1:
            for p in row["p"]:
                p[2] = 5.0 + (p[2] - 5.0) * 0.95            # 5 % less gravity
                p[5] *= 0.95
    _spec, rep = grade.grade("S01", res)
    assert any("acceleration" in f for f in rep.fails), rep.fails


def test_the_grader_notices_a_leftover_damping_transient(monkeypatch):
    """The engine fault that cost S01 5 %: a body slowed by 0.1/s damping. Apply it to one sphere and require a failure."""
    monkeypatch.setattr(grade, "call", _stub_mentat)
    res = _result("S01")
    tick, g = res["tick"], 9.80665
    z, v = 5.0, 0.0
    for row in res["samples"]:
        if row["k"] >= 1:
            row["p"][3][2], row["p"][3][5] = z, v
            z += v * tick
            v += (-g - 0.1 * v) * tick                       # semi-implicit fall with linear damping
    _spec, rep = grade.grade("S01", res)
    assert rep.fails, "a 0.1/s damped fall must not pass"


# ------------------------------------------------------------------------------------------------ numeric helpers
def test_quad_fit_recovers_a_parabola_and_its_roots():
    ts = [i * 0.015 for i in range(60)]
    ys = [5.0 + 0.3 * t - 4.9 * t * t for t in ts]
    c0, c1, c2, resid = fit.quad_fit(ts, ys)
    assert (c0, c1, c2) == pytest.approx((5.0, 0.3, -4.9), rel=1e-9) and resid < 1e-12
    roots = fit.poly_roots_in((c0, c1, c2), 4.0, -2.0, 2.0)
    assert [5.0 + 0.3 * r - 4.9 * r * r for r in roots] == pytest.approx([4.0, 4.0])
    assert fit.quad_through([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_plateau_ignores_single_tick_glitches():
    tick = 0.015
    vs = [0.0] * 40
    for j in range(1, 40):
        vs[j] = vs[j - 1] + 2.0 * tick
    vs[20] += 0.2                                               # a one-tick spike and its recovery
    for j in range(21, 40):
        vs[j] -= 0.2
    a, ok, n = grade._plateau(vs, list(range(1, 41)), tick, 5, 38)
    assert a == pytest.approx(2.0, abs=1e-6) and ok > 0.9 and n == 33


def test_apex_is_the_continuous_extremum():
    """The three-point vertex of a parabola sampled on a regular grid is its exact extremum, wherever it falls between samples."""
    top = [-((i - 3.3) ** 2) for i in range(8)]
    assert grade._apex(top, 3, +1) == pytest.approx(0.0, abs=1e-12)
    bottom = [(i - 2.8) ** 2 + 1.5 for i in range(8)]
    assert grade._apex(bottom, 3, -1) == pytest.approx(1.5, abs=1e-12)


def test_oscillation_helpers_find_the_period():
    tick, T = 0.015, 2.0
    ts = [i * tick for i in range(600)]
    ys = [math.sin(2 * math.pi * t / T + 0.3) for t in ts]
    up = grade._crossings_up(ts, ys)
    assert (up[-1] - up[0]) / (len(up) - 1) == pytest.approx(T, rel=1e-3)
    tops = grade._extrema(ts, ys, +1)
    bottoms = grade._extrema(ts, ys, -1)
    assert all(abs(v - 1.0) < 1e-3 for _, v in tops) and all(abs(v + 1.0) < 1e-3 for _, v in bottoms)
    assert (tops[-1][0] - tops[0][0]) / (len(tops) - 1) == pytest.approx(T, rel=1e-3)


def test_line_fit():
    c0, c1 = grade._line([0, 1, 2, 3], [1, 3, 5, 7])
    assert (c0, c1) == pytest.approx((1.0, 2.0))


def test_source_angle_vectors_known_values():
    fwd, up = grade._angle_vectors(0, 0, 0)
    assert fwd == pytest.approx((1, 0, 0)) and up == pytest.approx((0, 0, 1))
    fwd, up = grade._angle_vectors(90, 0, 0)                    # pitched straight down
    assert fwd == pytest.approx((0, 0, -1), abs=1e-12) and up == pytest.approx((1, 0, 0), abs=1e-12)
    fwd, up = grade._angle_vectors(0, 90, 0)                    # yawed to the left (+y)
    assert fwd == pytest.approx((0, 1, 0), abs=1e-12)
    fwd, up = grade._angle_vectors(0, 0, 90)                    # rolled: the body's up now points to -y
    assert up == pytest.approx((0, -1, 0), abs=1e-12)


def test_lua_literals_round_trip_through_the_writer():
    text = luaout.lua({"a": 1, "b": [1.5, 2.5], "c": "x\"y\n", "d": [{"e": True}]})
    assert "a = 1" in text and "{1.5, 2.5}" in text and "e = true" in text
    assert chr(92) + chr(34) in text and chr(92) + "n" in text          # the quote and the newline are escaped
    with pytest.raises(ValueError):
        luaout.lua({"bad": float("nan")})


# ------------------------------------------------------------------------------------------------ the Lua addon
def _runtime():
    lupa = pytest.importorskip("lupa", reason="lupa (LuaJIT binding) not installed; Lua checks skipped")
    try:
        from lupa import luajit21 as mod
    except ImportError:
        from lupa import lua51 as mod
    return mod.LuaRuntime(unpack_returned_tuples=True)


def test_every_lua_file_of_the_addon_parses():
    rt = _runtime()
    check = rt.eval("function(s) local f, e = loadstring(s); return f ~= nil, e end")
    bad = []
    for path in sorted(LUA_ROOT.rglob("*.lua")):
        ok, err = check(path.read_text(encoding="utf-8"))
        if not ok:
            bad.append(f"{path.relative_to(LUA_ROOT)}: {err}")
    assert not bad, bad


@pytest.mark.parametrize("sid", SIDS)
def test_generated_lua_spec_evaluates_to_the_python_spec(sid):
    rt = _runtime()
    spec = specs.build(sid)
    tbl = rt.eval("function(s) return loadstring(s)() end")((LUA_ROOT / "mentat" / "exp_specs" / f"{sid.lower()}.lua").read_text(encoding="utf-8"))
    assert tbl["sha"] == spec["sha"] and tbl["id"] == sid
    assert len(tbl["props"]) == len(spec["props"])
    first = tbl["props"][1]
    assert first["id"] == spec["props"][0]["id"]
    assert float(tbl["gravity_m_s2"]) == pytest.approx(spec["gravity_m_s2"])
