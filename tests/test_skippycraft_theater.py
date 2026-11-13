"""Tests for stage/skippycraft/theater.py -- a live theater, against a fake Minecraft.

Matter-mode scenes (sink, melt, playground, strength, impact, slosh) are tested
in test_skippycraft_theater_matter.py. This file tests non-matter scenes
(slit screen, shooting range) and infrastructure (desk commands, RCON, etc.).
"""
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


# ── non-matter scenes: slit screen (carrier blocks) ──


def test_slit_screen_shows_the_photons_in_the_lasers_colour(looks):
    _, world = theater.build("slit")
    screen = {**looks, **theater.slit_looks()[0]}
    mc = FakeMinecraft()
    th = theater.Theater(world, screen, theater.STAGES["slit"]["origin"], mc.send)
    th.open()
    assert set(mc.shown().values()) == {"black_terracotta"} and len(mc.shown()) == 128 * 32
    for _ in range(30):
        world.step(0.1)
        th.draw()
    assert mc.shown() == _expected(th)
    lit = {b for b in mc.shown().values() if b.endswith("_wool")}
    assert lit and lit <= {theater.slit_looks()[0][(f"light 650 {lv}", 0.001)] for lv in range(1, 6)}
    assert "photons" in mc.bar


def test_every_scene_material_has_its_own_carrier(looks):
    """Non-matter scenes: each material in shared_looks gets a unique carrier."""
    assert len(set(looks.values())) == len(looks)
    # Range scene materials at pitch 0.002
    assert looks[("steel_mild", 0.002)] != looks[("aluminum", 0.002)]


# ── range scene ──


def test_the_range_target_and_its_glowing_crater_reach_minecraft(looks):
    import shooting_range
    world = shooting_range.RangeWorld("m2_browning", "nij_iv")
    mc = FakeMinecraft()
    th = theater.Theater(world, looks, theater.STAGES["range"]["origin"], mc.send)
    th.open()
    world.step(0.02)
    th.draw()
    shown = th.display()
    assert {th.mc(i, j, k): th.block[shown[i, j, k]] for i, j, k in np.argwhere(world.label > 0)} == mc.shown()


# ── infrastructure tests (no matter-mode dependency) ──


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
    # Only test non-matter scenes here (matter scenes use matter mode, not carrier blocks)
    looks = {**theater.shared_looks()[0], **theater.slit_looks()[0], **theater.cat_looks(),
             **theater.lights_looks(), **theater.atoms_looks()}
    non_matter = {n: s for n, s in theater.STAGES.items() if not s.get("matter")}
    for name, name_st in non_matter.items():
        _, world = theater.build(name)
        if world is None:
            continue
        th = theater.Theater(world, looks, name_st["origin"], lambda c: None)
        nx, ny, nz = world.label.shape
        xs = sorted(th.mc(i, 0, 0)[0] for i in (0, nx - 1))
        ys = sorted(th.mc(0, 0, k)[1] for k in (th.k0, nz - 1))
        zs = sorted(th.mc(0, j, 0)[2] for j in (0, ny - 1))
        for other in theater.STAGES.values():
            x, y, z = other["desk"]
            for bx, by, bz in [(x + dx, y + dy, z + dz) for dx in range(-2, 3) for dy in (-1, 0) for dz in range(-4, 1)]:
                inside = xs[0] <= bx <= xs[1] and ys[0] <= by <= ys[1] and zs[0] <= bz <= zs[1]
                assert not inside, f"{name}'s grid covers a desk block at {(bx, by, bz)}"


def test_a_broken_desk_is_rebuilt():
    class World:
        def __init__(self):
            self.ok = set()
            for name in theater.STAGES:
                self.ok.update(theater.desk_checks(name))
            self.rebuilt = 0

        def command(self, c):
            if c.startswith("execute if block"):
                return "Test passed" if c in self.ok else "Test failed"
            if "stone_button" in c and c.startswith("setblock"):
                self.rebuilt += 1
            return ""

        def commands(self, cmds):
            return [self.command(c) for c in cmds]
    w = World()
    assert theater.repair_desks(w) == []                       # all there: nothing touched
    w.ok.discard(theater.desk_checks("melt")[3])               # he breaks the melt desk's Run button
    assert theater.repair_desks(w) == ["melt"] and w.rebuilt == 3
