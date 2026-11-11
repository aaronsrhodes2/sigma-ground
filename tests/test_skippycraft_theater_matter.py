"""The theater in matter mode: cells become skippycraft:matter blocks carrying sigma-ground's record (stage/skippycraft/theater.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))

import matter  # noqa: E402
import theater  # noqa: E402


class FakeBlocks:
    def __init__(self):
        self.frames = []

    def send_matter(self, records, runs):
        self.frames.append((list(records), list(runs)))
        return (sum(r[3] for r in runs), 0)

    def send(self, palette, runs):                      # the carrier path must not be used
        raise AssertionError("matter mode sent carrier blocks")


@pytest.fixture(scope="module")
def slosh():
    field, world = theater.build("slosh")
    looks, _ = theater.shared_looks()
    return field, world, looks


def _theater(slosh, blocks):
    _, world, looks = slosh
    return theater.Theater(world, looks, (0, -60, 100), lambda cmds: None, blocks, None, matter=True), world


def test_first_draw_sends_every_material_as_its_sigma_ground_record(slosh):
    fb = FakeBlocks()
    th, world = _theater(slosh, fb)
    th.draw()
    records, runs = fb.frames[0]
    mats = {json.loads(r)["material"] for r in records if r}
    assert {"glass", "water", "aluminum"} <= mats and "gold" in mats and "wood_oak" in mats
    assert all(json.loads(r)["schema"] == matter.SCHEMA for r in records if r)
    water = next(json.loads(r) for r in records if r and json.loads(r)["material"] == "water")
    assert water["n"] and water["absorption_per_m"] and water["t"] is None       # clear; no clock: it is not moving
    assert water["pitch_m"] == pytest.approx(world.pitch) and water["mass_kg"] == pytest.approx(1000 * world.pitch ** 3, rel=0.05)
    assert sum(r[3] for r in runs) > 0


def test_a_falling_body_gets_a_live_record_with_its_velocity_and_time(slosh):
    fb = FakeBlocks()
    th, world = _theater(slosh, fb)
    th.draw()
    for _ in range(12):
        world.step(0.04)
    fb.frames.clear()
    th.draw()
    records, runs = fb.frames[-1]
    live = [json.loads(r) for r in records if r and json.loads(r)["t"] is not None]
    assert live and all(any(abs(c) > 0 for c in rec["v"]) for rec in live)       # moving, so each has a velocity and a clock
    assert any(abs(rec["v"][1]) > 0 for rec in live)                             # something falls: vertical, in Minecraft's axes
    assert all(rec["t"] == pytest.approx(world.t) for rec in live)
    assert {rec["material"] for rec in live} <= {"gold", "wood_oak"}


def test_a_resting_world_sends_nothing(slosh):
    fb = FakeBlocks()
    th, world = _theater(slosh, fb)
    th.draw()
    fb.frames.clear()
    th.draw()
    assert fb.frames == []


def test_a_hot_world_still_needs_the_glow_carriers():
    field, world = theater.build("playground")
    looks, _ = theater.shared_looks()
    with pytest.raises(ValueError, match="isothermal"):
        theater.Theater(world, looks, (0, -60, 100), lambda c: None, FakeBlocks(), None, matter=True)
