"""Router dataset generator: the label must stay true to the question.

Every check here guards one way a training example could teach the router
something false: a rendered "550 nm" whose label isn't 5.5e-7, a key the tool
doesn't know, an intent that names a different key, a rewrite that drops a
number, a benchmark question leaking into training, a seed's rewrites split
across train/val. The last test builds a small dataset against the real MCP
server and re-runs every gold call independently.
"""
import asyncio
import json
import os
import random
import re
import sys

import pytest

from sigma_ground.mcp.benchmark import router_dataset as D

# ── quantities ───────────────────────────────────────────────────────────


def _factor(unit: D.Unit, disp: str) -> float:
    return dict(unit.renderings)[disp]


@pytest.mark.parametrize("param,tool", [
    ("wavelength_m", "photon_energy_from_wavelength"), ("frequency_hz", "em_wave_wavelength"),
    ("mass_kg", "kinetic_energy"), ("mass_kg", "schwarzschild_radius"),
    ("velocity_m_s", "kinetic_energy"), ("capacitance_f", "rc_time_constant"),
    ("pressure_pa", "ideal_gas"), ("concentration_mol_l", "weak_acid_ph"),
    ("acid_concentration_M", "titration"), ("barrier_width_nm", "quantum_tunneling"),
])
def test_rendered_text_and_label_agree(param, tool):
    unit = D._unit_for(param, tool)
    for seed in range(200):
        a = D.sample_numeric(param, "number", tool, random.Random(seed))
        num, disp = a.text.split(" ", 1)
        if disp == "°C":
            assert a.value == pytest.approx(float(num) + 273.15, rel=1e-12)
        else:
            assert a.value == pytest.approx(float(num) * _factor(unit, disp), rel=1e-11)
        assert unit.lo / 1.01 <= a.value <= unit.hi * 1.01 or disp == "°C"


def test_suffixes_are_case_sensitive():
    assert D._unit_for("acid_concentration_M", "t").renderings[0][0] == "M"   # molar
    assert D._unit_for("length_m", "t").renderings[0][0] == "m"               # metres
    assert D._unit_for("energy_eV", "t").renderings == D._unit_for("energy_ev", "t").renderings


def test_quantum_number_j_is_not_joules():
    # read as energy, total_j once sampled J = 3e8 and the tool tried to list
    # 6e8 m_j values -- the server was OOM-killed at 13.7 GB
    for seed in range(100):
        a = D.sample_numeric("total_j", "number", "atomic_angular_momentum", random.Random(seed))
        assert a.value in D.DISCRETE["total_j"] and (2 * a.value) == int(2 * a.value)
    assert D._unit_for("energy_j", "energy_to_mass").renderings[0][0] == "J"


def test_labels_carry_no_float_noise():
    unit = D._u(1, 1e9, ("MHz", 1e6))
    for seed in range(100):
        _, label, shown, _ = D.render_quantity(1.66e7, unit, random.Random(seed))
        assert repr(label) == repr(float(f"{label:.12g}"))


def test_ordered_pairs_stay_ordered():
    t = D.ToolSpec("carnot_efficiency", "", {
        "properties": {"t_hot_k": {"type": "number"}, "t_cold_k": {"type": "number"}},
        "required": ["t_hot_k", "t_cold_k"]})
    for seed in range(200):
        args, _ = D.sample_args(t, D.Vocab(), random.Random(seed))
        got = {a.name: a.value for a in args}
        assert got["t_hot_k"] >= got["t_cold_k"]


# ── string keys ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("notes,expected", [
    ("'x' not in lookup. Available: ['earth', 'jupiter', 'mars']", ["earth", "jupiter", "mars"]),
    ("Needs a metal in ['aluminum', 'copper']", ["aluminum", "copper"]),
    ("'x' not in lookup table. Some available: air, air_stp, aluminum, ...", ["air", "air_stp", "aluminum"]),
    ("Element 'x' not found", []),
])
def test_parse_vocab(notes, expected):
    assert D.parse_vocab(notes) == expected


def test_intents_that_name_a_key_are_dropped():
    kws = ["voltage of a copper and zinc battery", "Daniell cell voltage", "cell potential"]
    assert D.drop_contradicting(kws, ["copper", "silver"]) == kws[1:]
    # one-letter element symbols match case-sensitively, as whole words
    assert D.drop_contradicting(["ionization energy", "K shell"], ["K"]) == ["ionization energy"]
    assert D.drop_contradicting(["K shell"], ["K"]) == ["K shell"]     # never empty


def test_intent_phrases_skip_formulas_years_and_fragments():
    entry = {"keywords_formal": ["Schwarzschild radius", "r_s = 2GM/c^2", "Karl Schwarzschild 1916",
                                 "6.242e18"],
             "keywords_colloquial": ["crush it to a black hole", "how small to be a black hole"]}
    got = D.intent_phrases(D.ToolSpec("schwarzschild_radius", "", {}), entry)
    assert got == ["Schwarzschild radius", "how small to be a black hole"]


def test_rendering_reads_cleanly():
    copper = D.Arg("material", "copper", "copper", "material", key_text="copper")
    q = D.render_question("density", [copper], 0)
    assert q == "What is the density for copper?"
    expr = D.Arg("expression", "x**2", "x**2", "expression", key_text="x**2")
    var = D.Arg("variable", "x", "with respect to x", "variable", key_text="x")
    assert D.render_question("integral", [expr, var], 0) == "What is the integral for x**2 with respect to x?"
    assert "for for" not in D.render_question("density", [copper], 1)
    assert D.render_question("how long before it lands", [], 0) == "How long before it lands?"
    mercury = D.Arg("body_name", "mercury", "mercury", "body name", key_text="mercury")
    assert D.render_question("how strong is gravity on", [mercury], 0) == \
        "How strong is gravity on mercury?"
    assert D.render_question("how strong is gravity on", [], 0) == "How strong is gravity?"


def test_has_content_tells_answers_from_empty_shells():
    assert D.has_content({"solar_system_bodies": ["earth", "mars"]})          # a catalogue
    assert D.has_content([{"label": "Range", "value": 40.8}])                 # procedure steps
    assert D.has_content("x**3/3")
    assert not D.has_content({"uniaxial_stress_Pa": None, "shear_stress_Pa": None})
    assert not D.has_content({"open_scenes": [], "scenes": {}})
    assert not D.has_content(float("nan"))


# ── second opinion + guard ───────────────────────────────────────────────

def test_classifier_conflict_is_detected():
    q = "If the Sun collapsed into a black hole, what would its event horizon radius be?"
    assert D.classifier_verdict(q, {"tool": "schwarzschild_radius", "args": {}}) == "agrees"
    assert D.classifier_verdict(q, {"tool": "density", "args": {}}) == "conflict"
    assert D.classifier_verdict("zxqv plorb", {"decline": "x"}) == "silent"


def test_guard_keeps_the_benchmark_out():
    guard = D.BenchmarkGuard.from_repo()
    corpus = json.loads((D.HERE / "questions.json").read_text(encoding="utf-8"))
    q0 = corpus[0]
    # a light rewording of a benchmark question
    assert guard.check(q0["question"].replace("seconds", "secs"))
    # same tool, same answer, different words
    assert guard.check("totally different words", q0["primary_tool_expected"],
                       q0["expected_value"])
    # routing-corpus negatives are protected too
    assert guard.check("the stock market crashed today")
    assert guard.check("What is the density for osmium?", "density", 22590.0) is None
    # an answer that barely depends on the input doesn't make two questions
    # the same instance (doppler ~1.000 at any ordinary speed; inf matches nothing)
    assert guard.check("Doppler factor at 11.6 m/s?", "doppler_shift_factor", 1.00000004) is None
    assert guard.check("Redshift for mass of 2.4e24 kg and radius of 1.19e7 m?",
                       "gravitational_redshift", 1.5e-10) is None


# ── rewrites + splits ────────────────────────────────────────────────────

def test_rewrite_checks():
    args = [D.Arg("mass_kg", 0.0206, "20.6 g", "mass", "g"),
            D.Arg("material", "copper", "copper", "material", key_text="copper")]
    q = "What is the kinetic energy for mass of 20.6 g, copper?"
    ok = "How much kinetic energy does a 20.6 g copper pellet carry?"
    assert D.check_rewrite(q, ok, args) is None
    assert D.check_rewrite(q, ok.replace("20.6", "21"), args) == "numbers changed"
    assert D.check_rewrite(q, ok + " It moves at 3 m/s.", args) == "numbers changed"
    assert D.check_rewrite(q, ok.replace("20.6 g", "20.6 kg"), args) == "unit 'g' changed"
    assert D.check_rewrite(q, ok.replace("20.6 g", "20.6 grams"), args) == "unit 'g' changed"
    assert D.check_rewrite(q, ok.replace("a 20.6 g", "a 20.6-g"), args) is None
    assert D.check_rewrite(q, ok.replace("copper", "brass"), args) == "'copper' dropped"
    assert D.check_rewrite(q, q, args) == "unchanged"


def test_numbers_in_reads_scientific_forms():
    assert D.numbers_in("1.5e-7 m, 3 × 10^8 m/s and 1,200 kg in m³") == [1.5e-7, 3e8, 1200.0]


def test_a_seed_and_its_rewrites_share_a_split():
    for g in range(500):
        group = f"tool:{g}:0"
        assert len({D.split_of(group, 10) for _ in range(3)}) == 1
    vals = sum(D.split_of(f"tool:{g}:0", 10) == "val" for g in range(5000))
    assert 350 < vals < 650


# ── end to end, against the real server ──────────────────────────────────

def test_build_against_the_real_server(tmp_path):
    pytest.importorskip("mcp.server.fastmcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    def teacher(q, k):
        if k % 3 == 0:
            return f'"Quick one: {q}"'                       # fine, quotes stripped
        if k % 3 == 1:
            return re.sub(r"\d", "7", q)                     # corrupts the numbers
        return q                                            # no change

    async def run():
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "sigma_ground.mcp.server"],
                                       env=dict(os.environ))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = [t for t in await D.visible_tools(session) if t.name in
                         {"kinetic_energy", "density", "convert_units", "integrate_expr"}]
                examples, report = await D.build(
                    session, tools, D.manifest_by_tool(), per_tool=3, seed=1,
                    clarify_rate=1.0, guard=D.BenchmarkGuard.from_repo(),
                    teacher=teacher, variants=3, log=lambda _: None)
                # independent re-verification of every gold call
                for e in examples:
                    if e.plan.get("tool") not in (None, "request_clarification"):
                        assert await D.execute(session, e.plan["tool"], e.plan["args"]), e.question
                return tools, examples, report

    tools, examples, report = asyncio.run(run())
    assert {t.name for t in tools} == set(report.per_tool)
    by_source = {e.source for e in examples}
    assert {"template", "clarify", "decline", "rewrite"} <= by_source
    for e in examples:
        if e.source == "rewrite":
            assert e.question.startswith("Quick one: ")
            assert sorted(D.numbers_in(e.question)) == sorted(D.numbers_in(e.meta["rewrite_of"]))
    assert report.drops["rewrite: numbers changed"] > 0
    assert report.drops["rewrite: unchanged"] > 0

    manifest = D.write_dataset(examples, report, tmp_path, tools, 10, {"test": True})
    assert manifest["tool_surface_hash"] == D.surface_hash(tools)
    rows = [json.loads(line) for f in ("train.jsonl", "val.jsonl")
            for line in (tmp_path / f).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(examples)
    for row in rows:
        sys_msg, user, assistant = row["messages"]
        assert sys_msg["content"] == D.ROUTER_SYSTEM_PROMPT
        plan = json.loads(assistant["content"])
        assert ("tool" in plan) != ("decline" in plan)
    splits = {}
    for f in ("train", "val"):
        for line in (tmp_path / f"{f}.jsonl").read_text(encoding="utf-8").splitlines():
            splits.setdefault(json.loads(line)["meta"]["group"], set()).add(f)
    assert all(len(s) == 1 for s in splits.values())


def test_ollama_teacher_speaks_the_chat_api(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = {}

    class Ollama(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"message": {"content": '"How long would a 10 m drop take?"\nnote'}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    try:
        rewrite = D.ollama_teacher(f"http://127.0.0.1:{srv.server_port}", "qwen2.5:32b")
        got = D.clean_rewrite(rewrite("Height of 10 m: how long before it lands?", 7))
    finally:
        srv.shutdown()
    assert got == "How long would a 10 m drop take?"
    assert seen["model"] == "qwen2.5:32b" and seen["options"]["seed"] == 7
    assert seen["messages"][0]["content"] == D.TEACHER_PROMPT
