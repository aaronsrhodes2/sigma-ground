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
            elif c.startswith("fill"):
                x0, y0, z0, x1, y1, z1, b = c.split()[1:]
                for x in range(int(x0), int(x1) + 1):
                    for y in range(int(y0), int(y1) + 1):
                        for z in range(int(z0), int(z1) + 1):
                            self.blocks[(x, y, z)] = b
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
    t1.run(theater.SCENES["sink"][4], 0.05, rate_hz=1e6)
    _, w2 = theater.build("sink")
    t2 = theater.Theater(w2, looks, (0, 0, 0), mc.send)
    t2.open()
    assert mc.shown() == _expected(t2)


def test_gold_sinks_and_lands_on_the_floor_of_the_mercury(looks):
    _, world = theater.build("sink")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (0, 0, 0), mc.send)
    th.open()
    th.run(theater.SCENES["sink"][4], 0.05, rate_hz=1e6)
    n = theater.RESOLUTION
    gold = [p for p, b in mc.shown().items() if b == looks[("gold", 0.05 / n)]]
    assert len(gold) == n ** 3                                  # one written block = n³ theater blocks
    assert {y for _, y, _ in gold} == set(range(n, 2 * n))     # written y 1: the bottom layer of the mercury


def test_readout_is_the_worlds_own_ledger(looks):
    _, world = theater.build("melt")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, (0, 0, 0), mc.send)
    th.open()
    th.run(3, 0.05, rate_hz=1e6)
    assert f"E {world.energy_J():.6g} J" in mc.bar and f"radiated {world.ledger['radiated_J']:.4g} J" in mc.bar


def test_every_scene_material_has_its_own_carrier(looks):
    assert len(set(looks.values())) == len(looks)
    n = theater.RESOLUTION
    assert looks[("lead", 0.01 / n)] != looks[("granite", 0.05 / n)]


def test_panel_puts_a_button_on_a_command_block_that_records_the_request():
    cmds = theater.panel_commands("sink")
    x, y, z = theater.STAGES["sink"]["run"]
    assert f'setblock {x} {y} {z} minecraft:command_block{{Command:"scoreboard players set #request skippycraft 1"}}' in cmds
    assert f"setblock {x} {y + 1} {z} minecraft:stone_button[face=floor]" in cmds
    assert any(f"#request skippycraft {theater.STOP}" in c for c in cmds)
    assert any(c.startswith("summon minecraft:text_display") and '"Run sink"' in c for c in cmds)


def test_read_request_takes_the_value_and_clears_it():
    class FakeRcon:
        def __init__(self):
            self.value, self.sent = 2, []

        def command(self, c):
            self.sent.append(c)
            if c.startswith("scoreboard players get"):
                return f"#request has {self.value} [skippycraft]"
            if c.startswith("scoreboard players set"):
                self.value = int(c.split()[-1])
            return ""
    r = FakeRcon()
    assert theater.read_request(r) == 2 and r.value == 0
    assert theater.read_request(r) == 0
