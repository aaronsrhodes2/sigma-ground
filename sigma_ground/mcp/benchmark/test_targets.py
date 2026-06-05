"""Target tests — PROJECT_GOAL.md success metrics.

Targets the benchmark now meets are NORMAL tests (they must keep passing).
Targets not yet reached stay `xfail(strict=True)`: a silent XFAIL today, and
the day the benchmark clears the bar it XPASSes -> strict turns that into a
FAILURE — the alert to promote it to a normal test here, as we just did for
the six achieved targets.

Still aspirational (xfail): all-14-domains >= 80%, and conversation mode.

Run:
    pytest sigma_ground/mcp/benchmark/test_targets.py -v

Tests read the latest scored output from sigma_ground/mcp/benchmark/results/.
If a results file is missing, the test is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_HERE = Path(__file__).parent
_RESULTS = _HERE / "results"


def _load_scored(system: str) -> dict | None:
    p = _RESULTS / f"{system}_scored.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_run(system: str) -> list[dict] | None:
    p = _RESULTS / f"{system}_run.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ============================================================
# PRIMARY: Match Wolfram on the subset of questions WA gets right.
# (Comparing overall % is misleading -- WA fails to PARSE 74% of our
# conversational-English questions, so its overall is artificially low.
# The fair test: of the questions Wolfram answered correctly, what
# fraction does sigma-ground also answer correctly? Target >=98%.)
# ============================================================
def test_qa_matches_wolfram_on_subset_wa_got_right() -> None:
    sg = _load_scored("sigma_ground")
    wf = _load_scored("wolfram")
    if sg is None or wf is None:
        pytest.skip("scored data missing; run daily_job first")
    wf_correct_ids = {r["id"] for r in wf["rows"] if r.get("correct")}
    if not wf_correct_ids:
        pytest.skip("Wolfram got 0 questions correct")
    sg_by_id = {r["id"]: r for r in sg["rows"]}
    overlap = sum(1 for qid in wf_correct_ids
                    if sg_by_id.get(qid, {}).get("correct"))
    rate = overlap / len(wf_correct_ids)
    assert rate >= 0.98, (
        f"sigma-ground matched Wolfram on {overlap}/{len(wf_correct_ids)} "
        f"({rate:.0%}) of questions WA got right; target >=98%"
    )


# ============================================================
# Q&A overall >= Wolfram's overall (the easier lower-bound check)
# ============================================================
def test_qa_overall_at_least_wolfram() -> None:
    sg = _load_scored("sigma_ground")
    wf = _load_scored("wolfram")
    if sg is None or wf is None:
        pytest.skip("scored data missing")
    sg_pct = sg["summary"]["overall_accuracy_pct"]
    wf_pct = wf["summary"]["overall_accuracy_pct"]
    assert sg_pct >= wf_pct, (
        f"sigma-ground {sg_pct:.1f}% < Wolfram {wf_pct:.1f}% overall"
    )


# ============================================================
# Stretch: Q&A overall >= 80%
# ============================================================
def test_qa_overall_above_80_percent() -> None:
    sg = _load_scored("sigma_ground")
    if sg is None:
        pytest.skip("scored data missing")
    sg_pct = sg["summary"]["overall_accuracy_pct"]
    assert sg_pct >= 80.0, (
        f"sigma-ground overall {sg_pct:.1f}% < 80% stretch target"
    )


# ============================================================
# All 14 corpus domains >= 80%
# ============================================================
@pytest.mark.xfail(strict=True,
                    reason="Target: all 14 corpus domains >= 80%.")
def test_all_domains_above_80_percent() -> None:
    sg = _load_scored("sigma_ground")
    if sg is None:
        pytest.skip("scored data missing")
    by_domain = sg["summary"].get("by_domain", {}) or {}
    if not by_domain:
        pytest.skip("no by_domain data")
    failing = {d: s["pct"] for d, s in by_domain.items() if s["pct"] < 80.0}
    assert not failing, (
        f"{len(failing)} domains below 80%: "
        + ", ".join(f"{d}={p:.0f}%" for d, p in sorted(failing.items()))
    )


# ============================================================
# Q&A median latency <= 30 seconds
# ============================================================
def test_qa_median_latency_under_30s() -> None:
    rows = _load_run("sigma_ground")
    if rows is None:
        pytest.skip("no run data")
    times = sorted(r["elapsed_s"] for r in rows
                     if isinstance(r.get("elapsed_s"), (int, float)))
    if not times:
        pytest.skip("no elapsed_s data")
    median = times[len(times) // 2]
    assert median <= 30.0, f"median latency {median:.1f}s > 30s target"


# ============================================================
# Library-gap rate <= 5% of total questions
# ============================================================
def test_library_gap_rate_under_5_percent() -> None:
    rows = _load_run("sigma_ground")
    if rows is None:
        pytest.skip("no run data")
    if not rows:
        pytest.skip("empty run data")
    fitted = sum(
        1 for r in rows
        if isinstance(r.get("extracted_value"), str)
        and "Fitted due to incompetence" in (r.get("extracted_value") or "")
    )
    rate = fitted / len(rows)
    assert rate <= 0.05, (
        f"library-gap rate {rate:.1%} > 5% target "
        f"({fitted}/{len(rows)} 'Fitted due to incompetence')"
    )


# ============================================================
# Adversarial: refusal-expected questions get correct refusal
# (Looser target: 50% of refusal_expected at least)
# ============================================================
def test_adversarial_refusal_rate() -> None:
    sg = _load_scored("adversarial_sigma_ground")
    if sg is None:
        pytest.skip("adversarial scored data missing")
    rows = sg.get("rows", [])
    refusal_domains = {"adversarial_false_premise", "adversarial_nonsense"}
    refusal = [r for r in rows if r.get("domain") in refusal_domains]
    if not refusal:
        pytest.skip("no refusal_expected data in adversarial run")
    correct = sum(1 for r in refusal if r.get("correct"))
    rate = correct / len(refusal)
    assert rate >= 0.5, (
        f"refusal-expected accuracy {rate:.0%} < 50% "
        f"({correct}/{len(refusal)} refused correctly)"
    )


# ============================================================
# Conversation mode (placeholder until implemented)
# ============================================================
@pytest.mark.xfail(strict=True,
                    reason="Conversation mode not implemented yet (roadmap item).")
def test_conversation_mode_implemented() -> None:
    # When the conversation runner exists, this test reads
    # results/conversation_run.json and asserts at least one multi-turn
    # session completed with persisted state.
    p = _RESULTS / "conversation_run.json"
    assert p.exists(), "conversation mode not built yet"
