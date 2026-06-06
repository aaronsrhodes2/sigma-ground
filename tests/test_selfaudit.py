"""Deckard self-audits a generated object: how much is grounded vs guessed, and
is it physically sane. Three verdicts — verified / estimated / suspect. Offline
(stubbed researcher + local data).
"""
import json

from sigma_ground.deckard import compile, audit
from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard.selfaudit import render


def _audit(name, payload, res=48):
    spec = research_spec(name, ask=lambda n: payload, model="stub")
    return audit(spec, compile(spec, resolution=res))


def test_grounded_object_is_verified():
    a = _audit("tennis ball", json.dumps({"kind": "composite", "parts": [
        {"name": "ball", "shape": "sphere", "dims": {"radius_m": 0.05}, "material": "steel"}]}))
    assert a["verdict"] == "verified"                 # radius grounded (ITF), steel grounded
    assert a["groundedness"] == 1.0 and a["dimensions_grounded"] == "1/1"
    assert not a["warnings"]


def test_ungrounded_but_sane_object_is_estimated():
    a = _audit("widget", json.dumps({"kind": "composite", "parts": [
        {"name": "blob", "shape": "sphere", "dims": {"radius_m": 0.05}, "material": "steel"}]}))
    assert a["verdict"] == "estimated"                # proportions guessed, but plausible + consistent
    assert a["dimensions_grounded"] == "0/1"
    assert not a["warnings"]


def test_implausibly_heavy_object_is_suspect():
    a = _audit("megablock", json.dumps({"kind": "composite", "parts": [
        {"name": "blob", "shape": "box", "dims": {"x_m": 1.0, "y_m": 1.0, "z_m": 1.0},
         "material": "steel"}]}), res=32)
    assert a["verdict"] == "suspect"                  # ~7850 kg of steel
    assert any("heavy" in w for w in a["warnings"])
    assert "SUSPECT" in render(a)
