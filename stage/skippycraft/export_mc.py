"""
SkippyCraft Phase 0: a voxel theater as a Minecraft data pack (misc/SKIPPYCRAFT_PLAN.md).

A ``VoxelField`` (``deckard.voxel_exec``) becomes one function, ``skippycraft:<name>``, that places the theater's blocks
relative to where it runs (``~`` coordinates), so the Captain stands where he wants it and types
``/function skippycraft:<name>``. ``skippycraft:clear_<name>`` puts air back over the same box.

Minecraft only SHOWS the theater. Each material gets a block for its LOOK (``LOOKS``); its real identity, temperature and phase
stay in sigma-ground. The table avoids blocks Minecraft would change on its own: no water or lava (they flow), no sand or
gravel (they fall), no plain ice (it melts in light), waxed copper (plain copper oxidises). A material with no chosen look is
drawn as ``FALLBACK`` and listed in the report, never silently guessed.

Axes: sigma-ground is z-up, Minecraft is y-up. Grid cell (i, j, k) → Minecraft (x, y, z) = (i, k, ny-1-j) from the box's low
corner: the inverse of voxel_exec's MineBench mapping, so a build comes back the way it was written.

    python stage/skippycraft/export_mc.py --code 'box(0,0,0,7,0,7,"granite"); ...' --name tank --out <world>/datapacks
"""
from __future__ import annotations

import json
from pathlib import Path

# material → a Minecraft Java 1.21 block that looks like it and that Minecraft leaves alone
LOOKS = {
    "iron": "iron_block", "steel_mild": "iron_block", "gold": "gold_block", "copper": "waxed_copper_block",
    "tungsten": "netherite_block", "lead": "light_gray_concrete", "silver": "white_concrete", "aluminum": "smooth_stone",
    "platinum": "white_concrete", "nickel": "light_gray_concrete", "zinc": "light_gray_concrete", "titanium": "gray_concrete",
    "mercury": "light_gray_stained_glass", "water": "blue_stained_glass", "water_ice": "packed_ice",
    "glass": "glass", "granite": "granite", "concrete": "gray_concrete", "wood_oak": "oak_planks",
    "rubber": "black_concrete", "silicon": "gray_concrete", "ceramic_alumina": "white_terracotta",
}
FALLBACK = "white_concrete"

# a theater world runs no physics of its own: these switch off what Minecraft would otherwise change
LOAD_COMMANDS = [
    "gamerule randomTickSpeed 0", "gamerule doFireTick false", "gamerule doDaylightCycle false",
    "gamerule doWeatherCycle false", "gamerule doMobSpawning false", "gamerule mobGriefing false",
    "time set noon", "weather clear",
]


def field_to_commands(field):
    """``(place, clear, report)``: setblock/fill commands for the field, x-runs merged into one ``fill``."""
    import numpy as np

    label = np.asarray(field.label_grid)
    occ = np.argwhere(label > 0)
    lo, hi = occ.min(0), occ.max(0)
    ny = label.shape[1]
    looks, unchosen = {}, []
    for k, m in enumerate(field.materials):
        if k:
            looks[k] = LOOKS.get(m, FALLBACK)
            if m not in LOOKS:
                unchosen.append(m)

    def mc(i, j, k):                                   # grid → Minecraft offset from the box's low corner
        return int(i - lo[0]), int(k - lo[2]), int((ny - 1 - j) - (ny - 1 - hi[1]))

    place = []
    for j in range(lo[1], hi[1] + 1):
        for k in range(lo[2], hi[2] + 1):
            row = label[:, j, k]
            i = lo[0]
            while i <= hi[0]:
                m = row[i]
                if not m:
                    i += 1
                    continue
                end = i
                while end + 1 <= hi[0] and row[end + 1] == m:
                    end += 1
                (x0, y, z), x1 = mc(i, j, k), mc(end, j, k)[0]
                block = looks[int(m)]
                place.append(f"setblock ~{x0} ~{y} ~{z} {block}" if x0 == x1 else
                             f"fill ~{x0} ~{y} ~{z} ~{x1} ~{y} ~{z} {block}")
                i = end + 1
    sx, sy, sz = (int(hi[0] - lo[0]), int(hi[2] - lo[2]), int(hi[1] - lo[1]))
    clear = [f"fill ~0 ~0 ~0 ~{sx} ~{sy} ~{sz} air"]
    report = {"blocks": int(len(occ)), "commands": len(place), "size_xyz": [sx + 1, sy + 1, sz + 1],
              "block_m": field.voxel_size,
              "looks": {field.materials[k]: b for k, b in looks.items()}, "no_look_chosen": unchosen}
    return place, clear, report


def write_datapack(field, name, datapacks_dir):
    """Write (or refresh) the ``skippycraft`` data pack with ``<name>`` and ``clear_<name>`` in it. Returns the report."""
    place, clear, report = field_to_commands(field)
    root = Path(datapacks_dir) / "skippycraft"
    fn = root / "data" / "skippycraft" / "function"
    fn.mkdir(parents=True, exist_ok=True)
    (root / "pack.mcmeta").write_text(json.dumps(
        # 1.21 reads pack_format/supported_formats; 1.21.9+ reads min_format/max_format: both, so any 1.21+ server loads it
        {"pack": {"pack_format": 48, "supported_formats": [48, 999], "min_format": 48, "max_format": 999,
                  "description": "SkippyCraft theaters (sigma-ground)"}}, indent=2), encoding="utf-8")
    header = [f"# {name}: {report['blocks']} blocks of {report['block_m']:g} m, from sigma-ground (SkippyCraft)"]
    (fn / f"{name}.mcfunction").write_text("\n".join(header + place) + "\n", encoding="utf-8")
    (fn / f"clear_{name}.mcfunction").write_text("\n".join(clear) + "\n", encoding="utf-8")
    (fn / "theater_rules.mcfunction").write_text("\n".join(LOAD_COMMANDS) + "\n", encoding="utf-8")
    tags = root / "data" / "minecraft" / "tags" / "function"
    tags.mkdir(parents=True, exist_ok=True)
    (tags / "load.json").write_text(json.dumps({"values": ["skippycraft:theater_rules"]}), encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    from sigma_ground.deckard.voxel_exec import field_from_voxel_exec

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--code", required=True, help="MineBench voxel.exec code whose blocks are materials")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="the world's datapacks folder")
    ap.add_argument("--block-m", type=float, default=1.0)
    ap.add_argument("--resolution", type=float, default=1)
    a = ap.parse_args(argv)
    field, _ = field_from_voxel_exec(a.code, block_m=a.block_m, resolution=a.resolution)
    print(json.dumps(write_datapack(field, a.name, a.out), indent=2))


if __name__ == "__main__":
    main()
