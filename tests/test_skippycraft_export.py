"""Tests for stage/skippycraft/export_mc.py -- a voxel theater as a Minecraft data pack."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))

import export_mc  # noqa: E402
from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402

TANK = ('box(0,0,0,7,0,7,"granite"); box(1,1,1,6,5,6,"glass"); box(2,1,2,5,5,5,"air");'
        'box(2,1,2,5,2,5,"mercury"); block(3,3,3,"iron")')


def _replay(commands):
    """Run setblock/fill commands into a dict {(x, y, z): block}."""
    world = {}
    for c in commands:
        nums = [int(v) for v in re.findall(r"~(-?\d+)", c)]
        block = c.split()[-1]
        if c.startswith("setblock"):
            world[tuple(nums)] = block
        else:
            (x0, y0, z0), (x1, y1, z1) = nums[:3], nums[3:]
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    for z in range(z0, z1 + 1):
                        world[(x, y, z)] = block
    return {k: v for k, v in world.items() if v != "air"}


def test_commands_rebuild_the_build_as_written():
    field, _ = field_from_voxel_exec(TANK, block_m=0.05)
    place, clear, rep = export_mc.field_to_commands(field)
    world = _replay(place)
    assert len(world) == rep["blocks"] == int((field.label_grid > 0).sum())
    # MineBench coordinates come back: the iron block was written at (3,3,3), the build's corner at (0,0,0)
    assert world[(3, 3, 3)] == "iron_block" and world[(0, 0, 0)] == "granite"
    assert world[(2, 1, 2)] == "light_gray_stained_glass" and (2, 3, 2) not in world   # mercury, then air above it
    assert len(place) < rep["blocks"]                                                     # runs merged into fills
    assert _replay(clear) == {}


def test_unchosen_looks_are_reported_not_guessed():
    field, _ = field_from_voxel_exec('block(0,0,0,"iron"); block(1,0,0,"depleted_uranium")')
    _, _, rep = export_mc.field_to_commands(field)
    assert rep["no_look_chosen"] == ["depleted_uranium"]
    assert rep["looks"]["depleted_uranium"] == export_mc.FALLBACK


def test_looks_avoid_blocks_minecraft_would_change():
    for block in export_mc.LOOKS.values():
        assert block not in {"water", "lava", "sand", "gravel", "ice", "copper_block"}


def test_writes_a_loadable_pack(tmp_path):
    field, _ = field_from_voxel_exec('box(0,0,0,2,0,2,"gold")')
    rep = export_mc.write_datapack(field, "pad", tmp_path)
    root = tmp_path / "skippycraft"
    assert json.loads((root / "pack.mcmeta").read_text())["pack"]["pack_format"] >= 48
    assert (root / "data/skippycraft/function/pad.mcfunction").read_text().count("gold_block") == 3
    assert json.loads((root / "data/minecraft/tags/function/load.json").read_text())["values"] == ["skippycraft:theater_rules"]
    assert rep["size_xyz"] == [3, 1, 3]


def test_theater_rules_use_minecraft_26_names():
    # 26.x rejects the old camelCase names, and one bad line makes it refuse the whole function
    for cmd in export_mc.LOAD_COMMANDS:
        if cmd.startswith("gamerule "):
            name = cmd.split()[1]
            assert name == name.lower() and not any(c.isupper() for c in name), name
