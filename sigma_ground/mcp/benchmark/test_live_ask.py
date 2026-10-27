"""live_ask: a deployed Mentat's /ask replies become harness records, a recreation during the run is seen, scoring and history work.

The service is a fake HTTP server in this process; no network, no model.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sigma_ground.mcp.benchmark import live_ask as L
from sigma_ground.mcp.benchmark import score as SC


class FakeMentat:
    """Answers by question text; `since` can be changed mid-run; `fail_next` makes the next N /ask replies 502."""

    def __init__(self):
        self.answers: dict[str, dict] = {}
        self.since = "2026-10-27T08:00:00Z"
        self.ref = "abc123"
        self.fail_next = 0
        self.flip_since_after: int | None = None               # after this many asks the container "restarts": since changes
        self.asked: list[str] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"ready": True, "since": outer.since})
                elif self.path == "/version":
                    self._send(200, {"sigma_ground_ref": outer.ref})
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                q = json.loads(self.rfile.read(n))["question"]
                outer.asked.append(q)
                if outer.flip_since_after is not None and len(outer.asked) > outer.flip_since_after:
                    outer.since = "2026-10-27T08:05:00Z"
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self._send(502, {"error": "bad gateway"})
                    return
                self._send(200, {"ok": True, "result": outer.answers.get(q, {"value": None, "units": "", "inputs": {"route": "clarify"}})})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    f = FakeMentat()
    yield f
    f.close()


def test_a_number_a_string_and_a_record_map_as_the_audit_says(fake):
    fake.answers["q1"] = {"value": 2953.25, "units": "m", "notes": "", "inputs": {"route": "router", "chain": [{"tool": "schwarzschild_radius", "args": {"mass_kg": 1.989e30}}]}}
    fake.answers["q2"] = {"value": "x**2*cos(x) + 2*x*sin(x)", "units": "", "inputs": {"route": "router"}}
    fake.answers["q3"] = {"value": {"mass_kg": 6.4e23, "radius_m": 3.4e6}, "units": "", "inputs": {"route": "router"}}
    fake.answers["q4"] = {"value": None, "units": "", "notes": "no", "clarification": {"sentence": "Which body?"},
                          "inputs": {"route": "clarify", "router_miss": {"reason": "clarify: body"}}}
    state = L.service_state(fake.url)
    assert state == {"since": "2026-10-27T08:00:00Z", "ready": True, "ref": "abc123"}
    r1 = L.record("q1", L.ask(fake.url, "q1")[0], 0.5, None, state)
    assert r1["extracted_value"] == 2953.25 and r1["extracted_units"] == "m" and r1["tool_calls"][0]["name"] == "schwarzschild_radius"
    assert r1["system"] == "live_mentat" and r1["sigma_ground_ref"] == "abc123" and r1["route"] == "router"
    r2 = L.record("q2", L.ask(fake.url, "q2")[0], 0.5, None, state)
    assert r2["extracted_value"] == "x**2*cos(x) + 2*x*sin(x)"
    r3 = L.record("q3", L.ask(fake.url, "q3")[0], 0.5, None, state)
    assert r3["extracted_value"] is None                       # a whole record is not the asked quantity
    fake.answers["q5"] = {"value": ["-sqrt(2)", "sqrt(2)"], "units": "", "inputs": {"route": "router"}}
    r5 = L.record("q5", L.ask(fake.url, "q5")[0], 0.5, None, state)
    assert r5["extracted_value"] == ["-sqrt(2)", "sqrt(2)"]    # roots stay a list: the maths oracle compares them as a set
    r4 = L.record("q4", L.ask(fake.url, "q4")[0], 0.5, None, state)
    assert r4["extracted_value"] is None and "Which body?" in r4["answer_text"] and r4["router_miss"] == {"reason": "clarify: body"}


def test_score_py_reads_the_records_unchanged(fake):
    truth = {"expected_value": 2953.25, "expected_units": "m", "tolerance_rel": 0.01, "question_type": "numerical"}
    q = {"id": "x", "domain": "gr", "question_type": "numerical"}
    good = L.record("x", {"value": 2960.0, "units": "m"}, 0.1, None, {})
    assert SC._score_one(good, q, truth)[0] is True
    none = L.record("x", {"value": {"a": 1}, "units": ""}, 0.1, None, {})
    assert SC._score_one(none, q, truth)[0] is False


def test_a_502_during_the_run_is_re_asked_and_a_changed_since_marks_the_run_cut(fake):
    items = [{"id": f"q{i}", "question": f"q{i}"} for i in range(4)]
    for it in items:
        fake.answers[it["question"]] = {"value": 1.0, "units": "s", "inputs": {"route": "router"}}
    fake.fail_next = 2                                           # the first two asks hit a recreation
    recs, before, after = L.run_corpus(fake.url, items, workers=1, timeout=5)
    assert all(r["extracted_value"] == 1.0 for r in recs) and sum(bool(r.get("re_asked")) for r in recs) == 2
    assert before["since"] == after["since"]
    # the container restarts partway through: every row still answers, and the run is reported as cut (before != after)
    fake.flip_since_after = len(fake.asked) + 1
    recs2, b2, a2 = L.run_corpus(fake.url, items, workers=1, timeout=5)
    assert all(r["extracted_value"] == 1.0 for r in recs2)
    assert b2["since"] == "2026-10-27T08:00:00Z" and a2["since"] == "2026-10-27T08:05:00Z"


def test_transport_failure_is_an_error_record_not_an_answer(fake):
    fake.fail_next = 1
    result, err = L.ask(fake.url, "anything", timeout=5)
    assert result == {} and err == "HTTP 502"
    rec = L.record("q", result, 0.1, err, {})
    assert rec["extracted_value"] is None and rec["answer_text"].startswith("<ERROR transport") and L.hit_outage(rec)


def test_unreachable_is_its_own_error():
    with pytest.raises(L.Unreachable):
        L.service_state("http://127.0.0.1:9")


def test_math_scoring_uses_same_answer(fake):
    items = [{"id": "m1", "question": "differentiate x^2", "expected": {"tool": "differentiate_expr", "result": "2*x"}, "kind": "wild"},
             {"id": "m2", "question": "integrate 1", "expected": {"tool": "integrate_expr", "result": "x"}, "kind": "wild"},
             {"id": "m3", "question": "solve 4x^2 - 8 = 0", "expected": {"tool": "solve_equation", "result": ["sqrt(2)", "-sqrt(2)"]}, "kind": "wild"}]
    rows = [L.record("m1", {"value": "2*x", "units": "", "inputs": {"route": "router"}}, 0.1, None, {}),
            L.record("m2", {"value": "x + 1", "units": "", "inputs": {"route": "router"}}, 0.1, None, {}),
            L.record("m3", {"value": ["-sqrt(2)", "sqrt(2)"], "units": "", "inputs": {"route": "router"}}, 0.1, None, {})]
    scored = L.score_math(rows, items)
    assert scored["summary"]["answer"] == "2/3" and scored["rows"][0]["answer_ok"] and not scored["rows"][1]["answer_ok"] and scored["rows"][2]["answer_ok"]


def test_run_all_writes_the_files_and_history(fake, tmp_path, monkeypatch):
    phys = [{"id": "p1", "question": "p1"}]
    adv = [{"id": "a1", "question": "a1"}]
    math = [{"id": "m1", "question": "m1", "expected": {"tool": "differentiate_expr", "result": "2*x"}, "kind": "wild"}]
    monkeypatch.setattr(L, "load_corpora", lambda: {"physics": phys, "adversarial": adv, "math": math})
    fake.answers.update({"p1": {"value": 3.0, "units": "s", "inputs": {"route": "router"}}, "a1": {"value": None, "units": "", "notes": "no, false"},
                         "m1": {"value": "2*x", "units": "", "inputs": {"route": "router"}}})
    meta = L.run_all(fake.url, tmp_path)
    assert (tmp_path / "live_mentat_run.json").exists() and (tmp_path / "adversarial_live_mentat_run.json").exists()
    assert json.loads((tmp_path / "live_mentat_math.json").read_text())["summary"]["answer"] == "1/1"
    assert meta["ref"] == "abc123" and meta["cut"] is False and meta["suites"]["physics"]["answered"] == 1
    assert L.previous_history(tmp_path) is None
    L.append_history(tmp_path, meta, {"physics": 1, "adversarial": 0, "math": 1})
    assert L.previous_history(tmp_path)["physics"] == 1
    assert L.previous_history(tmp_path, before_at=meta["at"]) is None


def test_a_cut_suite_is_asked_once_more(fake, tmp_path, monkeypatch):
    phys = [{"id": "p1", "question": "p1"}, {"id": "p2", "question": "p2"}]
    monkeypatch.setattr(L, "load_corpora", lambda: {"physics": phys, "adversarial": [], "math": []})
    fake.answers.update({"p1": {"value": 1.0, "units": "s"}, "p2": {"value": 2.0, "units": "s"}})
    fake.flip_since_after = 1                                    # the container restarts during the first pass only
    meta = L.run_all(fake.url, tmp_path)
    assert meta["retried_suites"] == ["physics"] and meta["cut"] is False and meta["suites"]["physics"]["answered"] == 2
    assert len(fake.asked) == 4
    meta2 = L.run_all(fake.url, tmp_path, retry_cut=False)
    assert meta2["retried_suites"] == []
