"""SkippyCraft's shooting range (stage/skippycraft/shooting_range.py): the game's result, carved into a voxel target."""
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))
import shooting_range as R  # noqa: E402


def _centre_column(w):
    return w.label[int(round(w.ci)), :, int(round(w.ck))]


def test_front_layer_faces_the_shooter():
    w = R.RangeWorld("ak47", "nij_iv", armed=False)
    _, armors = R.presets()
    first = armors["nij_iv"]["layers"][0]["material"]
    assert w.materials[w.label[int(w.ci), w.jhi, int(w.ck)]] == first


def test_a_breach_holes_every_layer_and_leaves_the_rest():
    w = R.RangeWorld("ak47", "medieval_plate")
    before = (w.label > 0).sum()
    w.step(0.02)
    assert w.verdict() == "BREACHED"
    assert not _centre_column(w).any()                                     # a clear line through the target
    rb = R.hole_radius_m("ak47") / R.PITCH
    depth = sum(max(1, round(L["thickness_m"] / R.PITCH)) for L in R.presets()[1]["medieval_plate"]["layers"])
    assert before - (w.label > 0).sum() <= math.pi * (rb + 1) ** 2 * depth   # only the hole is gone
    assert w.label[int(w.ci) + 10, w.jhi, int(w.ck)] > 0


def test_a_stopped_round_leaves_a_crater_and_the_back_intact():
    w = R.RangeWorld("pistol_9mm", "nij_iv")
    w.step(0.02)
    assert w.verdict().startswith("stopped in layer 1")
    col = _centre_column(w)
    crater = math.ceil(w.layers[0]["damage_depth_m"] / R.PITCH - 1e-9)
    assert not col[w.jhi - crater + 1: w.jhi + 1].any() and col[: w.jhi - crater + 1].any()


def test_the_crater_glows_at_the_games_temperature_then_cools():
    w = R.RangeWorld("m2_browning", "nij_iv")
    w.step(0.02)
    hottest = max(L["temperature_k"] for L in w.layers if not L.get("gap"))
    assert w.T.max() == pytest.approx(hottest)
    for _ in range(50):
        w.step(0.02)
    assert w.T.max() < 0.5 * hottest


def test_only_standard_physics_weapons():
    weapons, _ = R.presets()
    assert {weapons[k]["weapon"]["type"] for k in R.WEAPONS} <= {"kinetic", "laser", "explosive"}
