"""
MineBench builds as matter: voxel.exec code → a Deckard ``VoxelField``.

MineBench (github.com/Ammaar-Alam/minebench) has a model write JavaScript that
places blocks with ``block`` / ``box`` / ``line`` in a 256³ grid. Local models
build well that way: every block is countable and checkable, where a guessed
continuous shape is not. This module keeps that exact authoring language but
makes each block a MATERIAL: the model writes ``box(0,0,0,9,0,9,"iron")`` and
the cell becomes sigma-ground iron, with its density (and everything else the
interface cascade knows about iron) behind it.

    code (JS) ──node, sandboxed──► cells + names ──resolve each name──► label grid
                                                                        │
                                                       deckard._finalize (the same
                                                       SDF + mass/CoM/inertia +
                                                       interface scan as voxelize)

Honesty rules:
  - A name sigma-ground cannot ground (``material_profile`` says not identified,
    or has no sourced density) is REFUSED with every such name and its block
    count, never given a default density.
  - Air is not a material here: an unset cell is air (label 0, as in every
    VoxelField), and writing "air" carves a cell back to empty.
  - Names that resolve to the same canonical material share one label
    ("ice" and "water_ice" are one material).

Axes: MineBench is y-up (Minecraft); sigma-ground is z-up. A block at MineBench
(x, y, z) lands at sigma-ground (x, −z, y), right-handed, so a build's height
stays its height. The ground of the build (its lowest y) sits at world z =
``y_min · block_m``.

The model's code runs in ``node --permission``: no file, network or
child-process access, and a timeout.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_RUNTIME = Path(__file__).with_name("voxel_exec.js")

# Minecraft block names that ARE a material, one to one. Anything vaguer
# (stone, dirt, wool, glowstone) is left out on purpose: it must be named as a
# real material, not mapped to a guess.
MINECRAFT_ALIASES = {
    "iron_block": "iron",
    "gold_block": "gold",
    "oak_planks": "wood_oak",
    "oak_log": "wood_oak",
}


class UngroundedMaterialError(ValueError):
    """One or more block materials have no sourced properties in sigma-ground."""

    def __init__(self, unknown: dict):
        self.unknown = dict(unknown)            # {name as written: block count}
        listed = ", ".join(f"{n!r} ({c} blocks)" for n, c in sorted(unknown.items()))
        super().__init__(f"sigma-ground cannot ground these materials: {listed}. "
                         "Name a real material, or add it to sigma-ground with a source.")


def run_voxel_exec(code: str, *, grid_size: int = 256, seed: int = 1,
                   timeout_s: float = 60.0) -> dict:
    """Run voxel.exec code; return the raw ``{names, cells, placed, outside, ms}``."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("voxel_exec needs Node.js (`node`) on PATH")
    proc = subprocess.run(
        [node, "--permission", "-e", _RUNTIME.read_text(encoding="utf-8")],
        input=json.dumps({"code": code, "gridSize": int(grid_size), "seed": int(seed)}),
        capture_output=True, text=True, encoding="utf-8", timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"voxel.exec code failed: {proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout)


def _resolve(names, counts, aliases):
    """name as written → (canonical, MaterialFact density), or raise for the ungrounded."""
    from ..field.interface.resolve import material_profile

    resolved, unknown = {}, {}
    for name in names:
        prof = material_profile(aliases.get(name, name))
        dens = prof.get("density")
        if not prof.get("identified") or dens is None or dens.estimated:
            unknown[name] = counts[name]
        else:
            resolved[name] = (prof["canonical"], dens)
    if unknown:
        raise UngroundedMaterialError(unknown)
    return resolved


def field_from_voxel_exec(code: str, *, block_m: float | None = None, extent_m: float | None = None,
                          grid_size: int = 256, seed: int = 1, aliases: dict | None = None,
                          margin: int = 2, timeout_s: float = 60.0):
    """Build a ``VoxelField`` from MineBench voxel.exec code whose blocks are materials.

    Scale: the theater is a real piece of space. ``extent_m`` is its edge in
    metres (a room, a planet, a star system), and a block is then
    ``extent_m / grid_size`` across, holding that block's volume of its
    material. Or give ``block_m`` directly. Neither: 1 m blocks (Minecraft's).
    ``aliases``: extra name → material mappings, on top of ``MINECRAFT_ALIASES``.

    Returns ``(field, report)``. ``report`` says, per canonical material, the
    names it was written as, its block count, and the density used with its
    source and confidence; plus the run's placed / outside / ms counts.
    """
    import numpy as np
    from .voxelize import _finalize

    if block_m is not None and extent_m is not None:
        raise ValueError("give block_m or extent_m, not both")
    if extent_m is not None:
        block_m = float(extent_m) / grid_size
    elif block_m is None:
        block_m = 1.0
    raw = run_voxel_exec(code, grid_size=grid_size, seed=seed, timeout_s=timeout_s)
    cells = np.asarray(raw["cells"], dtype=np.int64).reshape(-1, 4)
    if not len(cells):
        raise ValueError("voxel.exec code placed no blocks inside the grid")

    names = raw["names"]
    present = np.bincount(cells[:, 3], minlength=len(names))
    used = [i for i in range(len(names)) if present[i]]   # "air" may have carved a name away
    counts = {names[i]: int(present[i]) for i in used}
    resolved = _resolve([names[i] for i in used], counts, {**MINECRAFT_ALIASES, **(aliases or {})})

    materials, label_of_canon, label_of_name = ["air"], {}, np.zeros(len(names), dtype=np.int32)
    report_mats = {}
    for i in used:
        canon, dens = resolved[names[i]]
        if canon not in label_of_canon:
            label_of_canon[canon] = len(materials)
            materials.append(canon)
            report_mats[canon] = {"written_as": [], "blocks": 0, "density_kg_m3": float(dens.value),
                                  "source": dens.source, "confidence": dens.confidence}
        label_of_name[i] = label_of_canon[canon]
        report_mats[canon]["written_as"].append(names[i])
        report_mats[canon]["blocks"] += counts[names[i]]

    # MineBench (x, y, z), y up  →  sigma-ground (x, −z, y), z up
    x, y, z = cells[:, 0], cells[:, 1], cells[:, 2]
    gi = x - x.min() + margin
    gj = z.max() - z + margin
    gk = y - y.min() + margin
    label = np.zeros((gi.max() + 1 + margin, gj.max() + 1 + margin, gk.max() + 1 + margin), dtype=np.int32)
    label[gi, gj, gk] = label_of_name[cells[:, 3]]

    pitch = float(block_m)
    lo = (np.array([x.min(), -z.max(), y.min()], dtype=float) - margin) * pitch
    density = {canon: report_mats[canon]["density_kg_m3"] for canon in report_mats}
    field = _finalize(label, materials, pitch, lo, density.__getitem__, 1.0,
                      extra_note=f"from MineBench voxel.exec, {len(cells)} blocks of {block_m} m")
    report = {"block_m": pitch, "materials": report_mats, "blocks": int(len(cells)), "placed": raw["placed"],
              "outside": raw["outside"], "ms": raw["ms"]}
    return field, report


__all__ = ["MINECRAFT_ALIASES", "UngroundedMaterialError", "run_voxel_exec", "field_from_voxel_exec"]
