"""Tests for stage/skippycraft/theater.py -- a live theater, against a fake Minecraft."""
import re
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))

import theater  # noqa: E402


class FakeMinecraft:
    """Applies setblock/bossbar commands to a dict, the way the server would."""

    def __init__(self):
        self.blocks, self.bar = {}, None

    def send(self, cmds):
        for c in cmds:
            if c.startswith("setblock"):
                x, y, z, b = c.split()[1:]
                self.blocks[(int(x), int(y), int(z))] = b
            elif " name " in c:
                self.bar = c.split(" name ", 1)[1]

    def shown(self):
        return {k: v for k, v in self.blocks.items() if v != "air"}


def _expected(th):
    return {th.mc(i, j, k): th.block[th.world.label[i, j, k]] for i, j, k in np.argwhere(th.world.label > 0)}


@pytest.fixture(scope="module")
def looks():
    return theater.shared_looks()[0]


def test_minecraft_shows_exactly_the_world_after_every_tick(looks):
    _, world = theater.build("sink")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (4, -60, 20), mc.send)
    th.open()
    assert mc.shown() == _expected(th)
    for _ in range(8):
        world.step(0.05)
        th.draw()
        assert mc.shown() == _expected(th)


def test_only_changed_blocks_are_sent(looks):
    _, world = theater.build("sink")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (0, 0, 0), mc.send)
    first = th.draw()
    assert first == int(world.label[:, :, th.k0:].size)       # the first draw sets every cell from the ground up
    assert all(y >= 0 for (_, y, _) in mc.blocks)             # and never touches Minecraft's ground below it
    assert th.draw() == 0                                      # nothing moved, nothing sent


def test_a_rerun_clears_what_the_last_run_left():
    looks = theater.shared_looks()[0]
    mc = FakeMinecraft()
    _, w1 = theater.build("sink")
    t1 = theater.Theater(w1, looks, (0, 0, 0), mc.send)
    t1.open()
    t1.run(12, 0.05, rate_hz=1e6)
    _, w2 = theater.build("sink")
    t2 = theater.Theater(w2, looks, (0, 0, 0), mc.send)
    t2.open()
    assert mc.shown() == _expected(t2)


def test_gold_sinks_and_lands_on_the_floor_of_the_mercury(looks):
    _, world = theater.build("sink")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (0, 0, 0), mc.send)
    th.open()
    th.run(12, 0.05, rate_hz=1e6)
    gold = [p for p, b in mc.shown().items() if b == looks[("gold", 0.05)]]
    assert len(gold) == 1 and gold[0][1] == 1                  # MineBench y 1: the bottom layer of the mercury


def test_readout_is_the_worlds_own_ledger(looks):
    _, world = theater.build("melt")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (0, 0, 0), mc.send)
    th.open()
    th.run(3, 0.05, rate_hz=1e6)
    assert f"E {world.energy_J():.6g} J" in mc.bar and f"radiated {world.ledger['radiated_J']:.4g} J" in mc.bar


def test_every_scene_material_has_its_own_carrier(looks):
    assert len(set(looks.values())) == len(looks)
    assert looks[("lead", 0.01)] != looks[("granite", 0.05)]
