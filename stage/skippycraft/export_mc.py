"""
SkippyCraft Phase 0: a voxel theater as a Minecraft data pack (misc/SKIPPYCRAFT_PLAN.md).

A ``VoxelField`` (``deckard.voxel_exec``) becomes one function, ``skippycraft:<name>``, that places the theater's blocks
relative to where it runs (``~`` coordinates), so the Captain stands where he wants it and types
``/function skippycraft:<name>``. ``skippycraft:clear_<name>`` puts air back over the same box.

Minecraft only SHOWS the theater; each material's real identity, temperature and phase stay in sigma-ground. Its LOOK:

  - derived (the default, the Captain's rule): ``field.interface.block.appearance`` gives each material its colour, its
    opacity through ONE block of this theater's size, its reflectance and smoothness. ``write_resourcepack`` repaints one
    "carrier" block per material with that: a concrete colour for an opaque material, a stained-glass colour for a see-through
    one (stained glass draws with the texture's alpha), plus a LabPBR specular map (smoothness, F0, metal) that shader packs
    read for real reflections; vanilla Minecraft draws no reflections at all. A material sigma-ground has no colour for is
    drawn neutral grey and named in the report.
  - vanilla (``LOOKS``): a hand-picked stock block per material, no resource pack.

Both avoid blocks Minecraft would change on its own (no water, lava, sand, gravel, plain ice, unwaxed copper).

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

_DYES = ["white", "light_gray", "gray", "black", "brown", "red", "orange", "yellow", "lime", "green", "cyan", "light_blue",
         "blue", "purple", "magenta", "pink"]
CARRIERS_OPAQUE = [f"{d}_concrete" for d in _DYES] + [f"{d}_terracotta" for d in _DYES if d != "black"]  # black: the slit screen
CARRIERS_CLEAR = [f"{d}_stained_glass" for d in _DYES]
CLEAR_BELOW = 0.99           # opacity under this draws through stained glass (translucent)
NO_COLOUR = (0.5, 0.5, 0.5)  # neutral grey for a material with no colour model: shown, and reported as a gap
RESOURCE_FORMAT = 97         # Minecraft 26.3's resource pack format (its version.json, pack_version.resource_major)


def derive_looks(field, temperature_k=293.15):
    """``(looks, appearances)``: material → carrier block, and material → ``appearance`` for this theater's block size."""
    from sigma_ground.field.interface.block import appearance

    looks, appearances, opaque, clear = {}, {}, iter(CARRIERS_OPAQUE), iter(CARRIERS_CLEAR)
    for m in field.materials[1:]:
        a = appearances[m] = appearance(m, field.voxel_size, temperature_k)
        try:
            looks[m] = next(clear if a["opacity"] < CLEAR_BELOW else opaque)
        except StopIteration:
            raise ValueError("more materials than carrier blocks (16 opaque, 16 see-through)") from None
    return looks, appearances


def write_resourcepack(looks, appearances, packs_dir, name="skippycraft"):
    """Repaint each carrier with its material's derived look. Returns ``{material: {carrier, rgba, ...}}``."""
    from PIL import Image

    root = Path(packs_dir) / name
    tex = root / "assets" / "minecraft" / "textures" / "block"
    tex.mkdir(parents=True, exist_ok=True)
    (root / "pack.mcmeta").write_text(json.dumps(
        {"pack": {"pack_format": RESOURCE_FORMAT, "min_format": RESOURCE_FORMAT, "max_format": RESOURCE_FORMAT,
                  "description": "SkippyCraft: looks derived by sigma-ground"}}, indent=2), encoding="utf-8")
    legend = {}
    for m, carrier in looks.items():
        a = appearances[m]
        rgb = a["color_rgb"] or NO_COLOUR
        alpha = 1.0 if a["opacity"] >= CLEAR_BELOW else a["opacity"]
        rgba = tuple(round(255 * max(0.0, min(1.0, v))) for v in (*rgb, alpha))
        Image.new("RGBA", (16, 16), rgba).save(tex / f"{carrier}.png")
        # LabPBR specular: R perceptual smoothness, G F0 (0-229 linear) or 255 = metal (albedo is F0), B 0, A 255 = no emission
        f0 = a["reflectance_f0"]
        g = 255 if a["metal"] else (min(229, round(255 * f0)) if isinstance(f0, float) else 10)
        r = round(255 * a["smoothness"]) if a["smoothness"] is not None else 0
        Image.new("RGBA", (16, 16), (r, g, 0, 255)).save(tex / f"{carrier}_s.png")
        key = f"{m[0]}@{m[1]:g}m" if isinstance(m, tuple) else m   # (material, block size) when theaters share a pack
        legend[key] = {"carrier": carrier, "rgba": rgba, "opacity": round(a["opacity"], 4), "metal": a["metal"],
                     "reflectance_f0": f0, "smoothness": a["smoothness"], "gaps": a["gaps"], "sources": a["sources"]}
    (root / "skippycraft_looks.json").write_text(json.dumps(legend, indent=2, default=str), encoding="utf-8")
    return legend

# ── glow: a block hot enough to shine (field.interface.block.glow) ───────────────────────────────────────────────────────
# A resource pack can repaint a block but not make it give off light, so a glowing cell is drawn with a block that already
# does, repainted in its glow colour. Light-emitting blocks that never change by themselves (random ticks are off; copper
# bulbs keep lit=true without redstone), dimmest first: (block state, light level, the textures that paint it).
GLOW_CARRIERS = [
    ("magma_block", 3, ["magma"]),
    ("waxed_oxidized_copper_bulb[lit=true]", 4, ["oxidized_copper_bulb_lit"]),
    ("waxed_weathered_copper_bulb[lit=true]", 8, ["weathered_copper_bulb_lit"]),
    ("crying_obsidian", 10, ["crying_obsidian"]),
    ("waxed_exposed_copper_bulb[lit=true]", 12, ["exposed_copper_bulb_lit"]),
    ("waxed_copper_bulb[lit=true]", 15, ["copper_bulb_lit"]),
    ("glowstone", 15, ["glowstone"]),
    ("shroomlight", 15, ["shroomlight"]),
    ("ochre_froglight", 15, ["ochre_froglight_top", "ochre_froglight_side"]),
    ("verdant_froglight", 15, ["verdant_froglight_top", "verdant_froglight_side"]),
    ("pearlescent_froglight", 15, ["pearlescent_froglight_top", "pearlescent_froglight_side"]),
    ("jack_o_lantern", 15, ["jack_o_lantern", "pumpkin_side", "pumpkin_top"]),
]
# a cell is drawn in the band of the highest edge at or below its temperature (the first is the Draper point)
GLOW_BANDS_K = (700.0, 900.0, 1100.0, 1300.0, 1500.0, 1800.0, 2200.0, 2800.0)
# display exposure (a screen, not physics): real glow brightens ~10^12-fold from the Draper point to 3000 K; a texture has
# 256 levels, so brightness is drawn on a log scale from GLOW_FLOOR at the Draper point to 1 at GLOW_FULL_K
GLOW_FLOOR, GLOW_FULL_K = 0.3, 3000.0


def _srgb(v):
    """Linear light → the sRGB encoding textures are stored in."""
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def glow_display(material, band_k):
    """``(rgba, brightness, glow)`` for drawing ``material`` at ``band_k``: its glow hue (sRGB-encoded) times a log-scale
    brightness from its own radiance."""
    import math
    from sigma_ground.field.interface.block import glow

    g = glow(material, band_k)
    lo = glow(material, GLOW_BANDS_K[0])["radiance_W_m2_sr_m"][0]
    hi = glow(material, GLOW_FULL_K)["radiance_W_m2_sr_m"][0]
    L = g["radiance_W_m2_sr_m"][0]
    v = GLOW_FLOOR + (1 - GLOW_FLOOR) * max(0.0, min(1.0, math.log(L / lo) / math.log(hi / lo))) if L > 0 else 0.0
    rgba = tuple(round(255 * max(0.0, min(1.0, _srgb(c) * v))) for c in g["rgb"]) + (255,)
    return rgba, v, g


def write_glow_textures(glows, packs_dir, name="skippycraft"):
    """Repaint each glow carrier: ``glows`` maps a key to ``(carrier index, rgba, brightness)``. LabPBR's emission (the
    specular map's alpha) carries the brightness for shader packs. Returns a legend."""
    from PIL import Image

    tex = Path(packs_dir) / name / "assets" / "minecraft" / "textures" / "block"
    tex.mkdir(parents=True, exist_ok=True)
    legend = {}
    for key, (ci, rgba, v) in glows.items():
        state, light, textures = GLOW_CARRIERS[ci]
        for t in textures:
            Image.new("RGBA", (16, 16), rgba).save(tex / f"{t}.png")
            Image.new("RGBA", (16, 16), (40, 10, 0, max(1, min(254, round(254 * v))))).save(tex / f"{t}_s.png")
            meta = tex / f"{t}.png.mcmeta"                            # magma's animation would read a missing strip
            meta.write_text(json.dumps({"animation": {"frametime": 1000}}), encoding="utf-8")
        legend[str(key)] = {"carrier": state, "light": light, "rgba": rgba, "brightness": round(v, 3)}
    (Path(packs_dir) / name / "skippycraft_glow.json").write_text(json.dumps(legend, indent=2), encoding="utf-8")
    return legend


def write_flat_textures(paint, packs_dir, name="skippycraft"):
    """Repaint blocks in one flat colour each: ``paint`` maps a block (its texture name) to rgba."""
    from PIL import Image
    tex = Path(packs_dir) / name / "assets" / "minecraft" / "textures" / "block"
    tex.mkdir(parents=True, exist_ok=True)
    for block, rgba in paint.items():
        Image.new("RGBA", (16, 16), rgba).save(tex / f"{block}.png")


# a theater world runs no physics of its own: these switch off what Minecraft would otherwise change. Minecraft 26.x names
# (snake_case; checked against Paper 26.3 on 2026-10-30: the old camelCase names are rejected, and one bad line makes Minecraft
# refuse the whole function)
LOAD_COMMANDS = [
    "gamerule random_tick_speed 0", "gamerule fire_spread_radius_around_player 0", "gamerule advance_time false",
    "gamerule advance_weather false", "gamerule spawn_mobs false", "gamerule mob_griefing false",
    "time set noon", "weather clear",
]


def field_to_commands(field, looks=None):
    """``(place, clear, report)``: setblock/fill commands for the field, x-runs merged into one ``fill``.
    ``looks``: material → block (``derive_looks``); None = the hand-picked ``LOOKS``."""
    import numpy as np

    label = np.asarray(field.label_grid)
    occ = np.argwhere(label > 0)
    lo, hi = occ.min(0), occ.max(0)
    ny = label.shape[1]
    blocks, unchosen = {}, []
    for k, m in enumerate(field.materials):
        if k:
            table = LOOKS if looks is None else looks
            blocks[k] = table.get(m, FALLBACK)
            if m not in table:
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
                block = blocks[int(m)]
                place.append(f"setblock ~{x0} ~{y} ~{z} {block}" if x0 == x1 else
                             f"fill ~{x0} ~{y} ~{z} ~{x1} ~{y} ~{z} {block}")
                i = end + 1
    sx, sy, sz = (int(hi[0] - lo[0]), int(hi[2] - lo[2]), int(hi[1] - lo[1]))
    clear = [f"fill ~0 ~0 ~0 ~{sx} ~{sy} ~{sz} air"]
    report = {"blocks": int(len(occ)), "commands": len(place), "size_xyz": [sx + 1, sy + 1, sz + 1],
              "block_m": field.voxel_size,
              "looks": {field.materials[k]: b for k, b in blocks.items()}, "no_look_chosen": unchosen}
    return place, clear, report


def write_datapack(field, name, datapacks_dir, looks=None):
    """Write (or refresh) the ``skippycraft`` data pack with ``<name>`` and ``clear_<name>`` in it. Returns the report."""
    place, clear, report = field_to_commands(field, looks)
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
    ap.add_argument("--looks", choices=("derived", "vanilla"), default="derived")
    ap.add_argument("--resourcepacks", help="the client's resourcepacks folder (derived looks); default: this PC's Minecraft")
    a = ap.parse_args(argv)
    field, _ = field_from_voxel_exec(a.code, block_m=a.block_m, resolution=a.resolution)
    looks = None
    if a.looks == "derived":
        import os
        looks, appearances = derive_looks(field)
        packs = a.resourcepacks or os.path.join(os.environ["APPDATA"], ".minecraft", "resourcepacks")
        legend = write_resourcepack(looks, appearances, packs)
        print(json.dumps({m: {k: v for k, v in d.items() if k != "sources"} for m, d in legend.items()}, indent=2,
                         default=str))
    print(json.dumps(write_datapack(field, a.name, a.out, looks), indent=2))


if __name__ == "__main__":
    main()
