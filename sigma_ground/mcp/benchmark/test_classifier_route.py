"""classifier_route must not hide a broken environment (2026-10-16).

It wraps each deterministic classifier in try/except so one odd question cannot stop a build. It used to swallow a missing package too: without `periodictable`
the astro classifier raised on a few questions, went "silent", and the dataset built on that machine differed from the one built elsewhere (the Linux rehearsal of the
M5 run found it: rebuild-check DIFFERENT). A missing package now raises; any other error in a classifier is still skipped.
"""
from __future__ import annotations

import pytest

from sigma_ground.mcp.benchmark import astro_classifier
from sigma_ground.mcp.benchmark import router_dataset as D


def test_a_missing_package_in_a_classifier_is_an_error_not_a_silent_skip(monkeypatch):
    def broken(question):
        raise ModuleNotFoundError("No module named 'periodictable'")
    monkeypatch.setattr(astro_classifier, "classify_for_astro", broken)
    # a question no earlier classifier claims, so the cascade reaches the broken one
    with pytest.raises(ModuleNotFoundError, match="periodictable"):
        D.classifier_route("zzzz qqqq nothing any classifier would recognise 12345")


def test_any_other_error_in_a_classifier_is_still_skipped(monkeypatch):
    def odd(question):
        raise ValueError("a question this classifier cannot parse")
    monkeypatch.setattr(astro_classifier, "classify_for_astro", odd)
    assert D.classifier_route("zzzz qqqq nothing any classifier would recognise 12345") == (None, None)


def test_the_cascade_is_unchanged_when_nothing_is_broken():
    assert D.classifier_route("What is the Schwarzschild radius of a 3 solar mass black hole?")[0] == "tool"
    assert D.classifier_route("zzzz qqqq nothing any classifier would recognise 12345") == (None, None)
