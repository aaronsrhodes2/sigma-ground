"""The infallibility gate: an answer is grounded, or it is refused — never
confidently wrong. Plus the refusal ledger (the Phase-2 backlog)."""
import math
from types import SimpleNamespace

from sigma_ground.materia import groundedness as g
from sigma_ground.materia.scenarios import black_hole


# ── gate 1: no grounded value ──
def test_gate_value_refuses_none_and_nonfinite():
    assert not g.check_value(None).grounded
    assert g.check_value(None).reason == g.NO_VALUE
    assert not g.check_value(float("nan")).grounded
    assert not g.check_value(float("inf")).grounded
    assert not g.check_value("[Fitted due to incompetence ...]").grounded
    assert not g.check_value(r"\[ 1 + z = \frac").grounded   # half-rendered latex


def test_gate_value_accepts_real_values():
    assert g.check_value(2.95e4).grounded
    assert g.check_value("mond").grounded          # a real categorical answer
    assert g.check_value({"a": 1, "b": 2}).grounded


# ── gate 2: cross-check ──
def test_gate_cross_refuses_disagreement():
    assert g.check_cross(100.0, 100.1).grounded            # agree (0.1%)
    bad = g.check_cross(13.7e9, 4.58)                       # Hubble time: way off
    assert not bad.grounded and bad.reason == g.CROSS_CHECK_FAILED
    assert g.check_cross(5.0, None).grounded                # no cross-check → pass


# ── gate 3: provenance ──
def test_gate_provenance():
    assert not g.check_provenance(False).grounded
    assert g.check_provenance(True).grounded
    assert g.check_provenance(None).grounded                # unknown ≠ untrusted


# ── combined gate ──
def test_gate_combines_cheapest_first():
    assert g.gate(938.27).grounded
    assert g.gate(None).reason == g.NO_VALUE
    assert g.gate(13.7e9, cross=4.58).reason == g.CROSS_CHECK_FAILED
    assert g.gate(5.0, trusted=False).reason == g.UNTRUSTED_PROVENANCE


# ── Materia results gate ──
def test_gate_results_passes_validated_verb():
    assert g.gate_results([black_hole()]).grounded         # self-validates → ok


def test_gate_results_refuses_failed_selfcheck():
    bad = SimpleNamespace(name="x", validation={"passed": False, "note": "drift"},
                          outputs={"v": 1.0}, steps=[])
    v = g.gate_results([bad])
    assert not v.grounded and v.reason == g.CROSS_CHECK_FAILED


# ── ledger ──
def test_ledger_records_and_summarises():
    led = g.Ledger()
    led.record("q1", g.Verdict(False, g.NO_VALUE, "none"))
    led.record("q2", g.Verdict(False, g.NO_VALUE, "nan"))
    led.record("q3", g.Verdict(False, g.CROSS_CHECK_FAILED, "off"))
    assert len(led) == 3
    assert led.summary() == {g.NO_VALUE: 2, g.CROSS_CHECK_FAILED: 1}
