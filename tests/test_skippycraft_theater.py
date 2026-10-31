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


def test_desk_has_run_stop_and_go_buttons_on_command_blocks_with_labels():
    cmds = theater.desk_commands("sink")
    x, y, z = theater.STAGES["sink"]["desk"]
    assert f'setblock {x - 2} {y} {z} minecraft:command_block{{Command:"scoreboard players set #request skippycraft 1"}}' in cmds
    assert f"setblock {x - 2} {y} {z - 1} minecraft:stone_button[face=wall,facing=north]" in cmds
    assert any(f"#request skippycraft {theater.STOP}" in c for c in cmds)
    sx, sy, sz, yaw, pitch = theater.stand("melt")
    assert any(f"tp @p {sx} {sy} {sz} {yaw} {pitch}" in c for c in cmds)       # Go to melt lands on the melt desk
    assert sum(c.startswith("summon minecraft:text_display") for c in cmds) == 3
    assert any(c.startswith("fill") and "sea_lantern" in c for c in cmds)          # the platform he stands on


def test_he_stands_on_the_platform_behind_the_buttons():
    for name in theater.STAGES:
        x, y, z = theater.STAGES[name]["desk"]
        sx, sy, sz, _, _ = theater.stand(name)
        assert sy == y and z - 4 <= sz <= z - 1 and abs(sx - (x + 0.5)) < 1


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


def test_no_desk_is_inside_any_theaters_grid():
    # a theater's first draw sets every cell of its grid from the ground up: a desk inside one would be erased by a run
    looks = theater.shared_looks()[0]
    for name, st in theater.STAGES.items():
        _, world = theater.build(name)
        th = theater.Theater(world, looks, st["origin"], lambda c: None)
        nx, ny, nz = world.label.shape
        xs = sorted(th.mc(i, 0, 0)[0] for i in (0, nx - 1))
        ys = sorted(th.mc(0, 0, k)[1] for k in (th.k0, nz - 1))
        zs = sorted(th.mc(0, j, 0)[2] for j in (0, ny - 1))
        for other in theater.STAGES.values():
            x, y, z = other["desk"]
            for bx, by, bz in [(x + dx, y + dy, z + dz) for dx in range(-2, 3) for dy in (-1, 0) for dz in range(-4, 1)]:
                inside = xs[0] <= bx <= xs[1] and ys[0] <= by <= ys[1] and zs[0] <= bz <= zs[1]
                assert not inside, f"{name}'s grid covers a desk block at {(bx, by, bz)}"
