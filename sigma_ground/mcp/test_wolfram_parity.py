"""Regression guard: Mentat's covered physics must stay matched to Wolfram Alpha.

Drives the frozen stratified parity sample (sigma_ground/mcp/benchmark/
wolfram_parity_cases.py) -- 34 cases across 14 domains, each computed by a
Mentat library/tool call and compared against the value Wolfram Alpha returned
(captured via the Wolfram MCP on 2026-06-05) within a per-case relative
tolerance.

This is DETERMINISTIC and OFFLINE: it re-runs Mentat's computation and compares
to the frozen oracle snapshot, so it catches drift in Mentat without needing
Wolfram (the Wolfram MCP is absent in CI). See misc/WOLFRAM_PARITY.md.

NOTE (truth-first): passing this proves Mentat still matches Wolfram on these
34 representative quantities -- NOT that all ~1000 covered functions are
value-correct. The sample is biased toward cleanly-numeric quantities.
"""
from __future__ import annotations

import pytest

from sigma_ground.mcp.benchmark.wolfram_parity_cases import CASES, evaluate


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_matches_wolfram(case):
    mentat = float(case["compute"]())
    oracle = case["oracle"]
    rel = abs(mentat - oracle) / abs(oracle) if oracle else abs(mentat)
    assert rel <= case["tol"], (
        f"{case['id']}: Mentat={mentat:.6g} vs Wolfram={oracle:.6g} "
        f"rel_err={rel:.3%} > tol={case['tol']:.1%}  [{case['oracle_kind']}] {case['wq']}"
    )


def test_overall_parity_rate():
    rows = evaluate()
    failures = [r["id"] for r in rows if not r["passed"]]
    ok = len(rows) - len(failures)
    assert not failures, f"{ok}/{len(rows)} within tolerance; failures: {failures}"


def test_sample_breadth():
    """Guard the sample stays broad (don't let it silently shrink)."""
    domains = {c["domain"] for c in CASES}
    assert len(CASES) >= 34, f"sample shrank to {len(CASES)} cases"
    assert len(domains) >= 12, f"only {len(domains)} domains: {sorted(domains)}"
