"""cascade_answer: the harness's deterministic cascade as one function (lifted 2026-10-27), pinned on questions the stored runs answered
through it, and off when every cascade.* switch is off. No model, no embedding server: the semantic stage is skipped (its switch off)."""
from __future__ import annotations

import pytest

from sigma_ground.mcp.benchmark import run_sigma_ground as H
from sigma_ground.mcp.benchmark import tweaks as T


@pytest.fixture(autouse=True)
def _deterministic_only():
    T.configure(None, off=["cascade.semantic"])
    yield
    T.reset()


def test_a_refusal_question_is_refused_before_any_model():
    rec = H.cascade_answer("Is the Earth flat?")
    assert rec is not None and rec["turns"] == 0 and rec["tool_calls"] == []
    assert rec["extracted_value"] is not None and "flat" in rec["answer_text"].lower() or "sphere" in rec["answer_text"].lower()


def test_nonsense_is_declined_not_computed():
    rec = H.cascade_answer("What is the kinetic energy of love?")
    assert rec is not None and rec["turns"] == 0
    assert not isinstance(rec["extracted_value"], (int, float))


def test_a_conversion_is_answered_by_the_conversion_stage():
    rec = H.cascade_answer("Convert 1 light year to meters.")                  # math_006 in the corpus; the stored run answered it here
    assert rec is not None and isinstance(rec["extracted_value"], (int, float))
    assert abs(rec["extracted_value"] / 9.4607e15 - 1) < 1e-4


def test_an_unmatched_question_falls_through():
    assert H.cascade_answer("zxq plorvian quux") is None


def test_every_stage_off_means_no_answer():
    T.configure(None, off=["cascade.*"])
    assert H.cascade_answer("Is the Earth flat?") is None
    assert H.cascade_answer("Convert 1 light year to meters.") is None
