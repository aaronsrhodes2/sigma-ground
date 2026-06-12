"""ShapeNetSem aspect-ratio correction — offline, monkeypatched Sem dims.

The pass is SHAPE-only (geometric-mean normalized; absolute scale stays with
_scale_to_typical_size), rank-matches axes, clamps factors, keeps everything
``estimated``, and skips when parts are rotated or anything is cited.
"""
import json

from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard import researcher as R


def _sem(monkeypatch, dims):
    monkeypatch.setattr(
        R._sources.shapenetsem, "dims_of",
        lambda n: ((dims, 99, "ShapeNetSem test", "test-lic") if "zz" in n else None))


def _spec(payload, name="zz widget"):
    return research_spec(name, ask=lambda n: json.dumps(payload), model="stub")


def test_squashed_model_guess_is_stretched_to_real_proportions(monkeypatch):
    # the model guesses a CUBE-ish body; the census says the real thing is
    # twice as tall as it is wide (e.g. a bottle-like 1:1:2 aspect)
    _sem(monkeypatch, [0.1, 0.1, 0.2])
    spec = _spec({"kind": "composite", "parts": [
        {"name": "body", "shape": "box",
         "dims": {"x_m": 0.3, "y_m": 0.3, "z_m": 0.3}, "material": "plastic"}]})
    d = spec.parts[0].dims
    aspect = d["z_m"].value / d["x_m"].value
    assert 1.7 < aspect < 2.3                              # ~the census 2:1
    # shape-only: total volume is roughly preserved (geometric-mean normalized)
    vol = d["x_m"].value * d["y_m"].value * d["z_m"].value
    assert abs(vol - 0.027) / 0.027 < 0.15
    assert all(f.estimated for f in d.values())            # still estimated
    assert any("aspect ratio" in s.get("name", "") for s in spec.sources)


def test_already_right_proportions_are_left_alone(monkeypatch):
    _sem(monkeypatch, [0.1, 0.1, 0.12])
    spec = _spec({"kind": "composite", "parts": [
        {"name": "body", "shape": "box",
         "dims": {"x_m": 0.2, "y_m": 0.2, "z_m": 0.25}, "material": "plastic"}]})
    assert spec.parts[0].dims["z_m"].value == 0.25         # within tolerance band


def test_rotated_parts_skip_the_correction(monkeypatch):
    _sem(monkeypatch, [0.1, 0.1, 0.4])
    spec = _spec({"kind": "composite", "parts": [
        {"name": "body", "shape": "box",
         "dims": {"x_m": 0.3, "y_m": 0.3, "z_m": 0.3}, "material": "plastic",
         "euler_deg": [0, 90, 0]}]})
    assert spec.parts[0].dims["z_m"].value == 0.3          # untouched


def test_extreme_census_factors_are_clamped(monkeypatch):
    _sem(monkeypatch, [0.01, 0.01, 1.0])                   # absurd 100:1 aspect
    spec = _spec({"kind": "composite", "parts": [
        {"name": "body", "shape": "box",
         "dims": {"x_m": 0.3, "y_m": 0.3, "z_m": 0.3}, "material": "plastic"}]})
    d = spec.parts[0].dims
    assert d["z_m"].value / d["x_m"].value <= 4.001        # 2.0 / 0.5 clamp bound
