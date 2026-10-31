"""
SkippyCraft's shooting range: the Weapon vs. Armor game (games/weapon_vs_armor) as a real target of voxels.

The target is the armour preset built as layers of blocks, each 2 mm (``PITCH``), front layer toward the shooter, 64 x 64
blocks across (12.8 cm square); a layer thinner than a block is one block. A shot runs the game's own per-layer physics
(``physics.kinetic`` Poncelet / Birkhoff, ``physics.laser``, ``physics.explosive`` Munroe jet): which layers it goes
through, how deep it gets into the one that stops it, and the temperature it leaves. That result is carved into the
target (a hole of the weapon's radius through every layer it defeats, a crater in the one that stops it) and heats the
crater's walls to the game's temperature; from then on the target is a ``materia.voxel_world.VoxelWorld``, so the hot
channel glows (``block.glow``), conducts its heat away, melts or boils off by sigma-ground's own data.

Only the standard-physics weapons: the game's plasma and sigma-field weapons import ``SIGMA_CONV``, which sigma-ground moved
to ssbm-theory (speculative), so they can't load. Known model choices the game makes (calibrated factors such as its
shaped-charge capacity and fibre/ceramic multipliers) are the game's, shown as they are.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

GAMES = Path(__file__).resolve().parents[2] / "games"
if str(GAMES) not in sys.path:
    sys.path.insert(0, str(GAMES))

PITCH = 0.002
FACE = 64
WEAPONS = ["ak47", "pistol_9mm", "crossbow_bolt", "m2_browning", "armor_piercing_30mm", "rpg7", "sabot_apfsds",
           "hel_100kw", "railgun_hypersonic"]
ARMORS = ["nij_iv", "nij_iiia", "medieval_plate", "roman_lorica", "wwii_helmet", "reactive_armor", "tank_composite",
          "spacecraft_hull"]
JET_RADIUS_M = 0.01           # a shaped charge's hole: the game gives no jet radius (exhibit choice, ~1/4 of an RPG-7's cone)


def presets():
    from weapon_vs_armor.presets import ARMOR_PRESETS, WEAPON_PRESETS
    return WEAPON_PRESETS, ARMOR_PRESETS


def materials():
    """Every material any range armour is made of."""
    _, armors = presets()
    return sorted({L["material"] for a in ARMORS for L in armors[a]["layers"] if not L.get("gap")})


def target_code(armor):
    """MineBench code for the armour, front layer at z = 0, and each layer's ``(z_first, z_last)`` (None for a gap)."""
    _, armors = presets()
    code, spans, z = [], [], 0
    for L in armors[armor]["layers"]:
        n = max(1, round(L["thickness_m"] / PITCH))
        if L.get("gap"):
            spans.append(None)
        else:
            code.append(f'box(0,0,{z},{FACE - 1},{FACE - 1},{z + n - 1},"{L["material"]}")')
            spans.append((z, z + n - 1))
        z += n
    return "; ".join(code), spans


def shoot(weapon, armor, environment="atmosphere_earth", distance_m=10.0):
    """The game's per-layer loop (``physics.solver.solve``) over the standard-physics weapons only. Returns one dict per
    layer the shot reached: ``material, penetrated, damage_depth_m, temperature_k, notes`` (gaps: ``{"gap": True}``)."""
    from weapon_vs_armor.physics import explosive, kinetic, laser
    from weapon_vs_armor.physics.environment import attenuate

    weapons, armors = presets()
    modules = {"kinetic": kinetic, "laser": laser, "explosive": explosive}
    w = attenuate(dict(weapons[weapon]["weapon"]), environment, distance_m)
    out = []
    for L in armors[armor]["layers"]:
        if L.get("gap"):
            w = attenuate(w, environment if environment == "atmosphere_earth" else "vacuum", L["thickness_m"])
            out.append({"gap": True})
            continue
        r = modules[w["type"]].interact(w, L["material"], L["thickness_m"])
        out.append({"material": L["material"], "penetrated": bool(r.get("penetrated")),
                    "damage_depth_m": float(r.get("damage_depth_m", 0.0)), "temperature_k": float(r.get("temperature_k", 300.0)),
                    "notes": r.get("notes", "")})
        if not r.get("penetrated") or r.get("weapon_out") is None:
            break
        w = r["weapon_out"]
    return out


def hole_radius_m(weapon):
    weapons, _ = presets()
    w = weapons[weapon]["weapon"]
    return w.get("radius_m") or w.get("spot_radius_m") or JET_RADIUS_M


class RangeWorld:
    """The target as a running world. ``armed``: the shot hits on the first step; otherwise it just stands there."""

    def __init__(self, weapon=WEAPONS[0], armor=ARMORS[0], armed=True):
        import numpy as np
        from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
        from sigma_ground.materia.voxel_world import VoxelWorld

        weapons, armors = presets()
        self.weapon, self.armor, self.armed = weapon, armor, armed
        self.preview = not armed
        code, self.spans = target_code(armor)
        field, _ = field_from_voxel_exec(code, block_m=PITCH)
        self.world = VoxelWorld(field, env="STP", gravity=False)
        self.materials, self.pitch = self.world.materials, self.world.pitch
        occ = np.argwhere(self.world.label > 0)
        self.jhi = int(occ[:, 1].max())                          # MineBench z = 0 (the front) is the grid's highest j
        self.ci = (occ[:, 0].min() + occ[:, 0].max()) / 2
        self.ck = (occ[:, 2].min() + occ[:, 2].max()) / 2
        self.layers, self.t, self.hit = [], 0.0, False
        self.names = (weapons[weapon]["name"], armors[armor]["name"])

    @property
    def label(self):
        return self.world.label

    @property
    def T(self):
        return self.world.T

    def impact(self):
        """Carve the shot into the target and heat the crater's walls to the game's temperatures."""
        import numpy as np
        self.layers = shoot(self.weapon, self.armor)
        rb = max(0.5, hole_radius_m(self.weapon) / self.pitch)
        nx, _, nz = self.world.label.shape
        ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        r = np.sqrt((ii - self.ci) ** 2 + (kk - self.ck) ** 2)
        hole, wall = r <= rb, (r > rb) & (r <= rb + 1.5)
        for span, L in zip(self.spans, self.layers):
            if span is None or L.get("gap"):
                continue
            z0, z1 = span
            depth = (z1 - z0 + 1) if L["penetrated"] else math.ceil(L["damage_depth_m"] / self.pitch - 1e-9)
            for z in range(z0, z1 + 1):
                j = self.jhi - z
                if z < z0 + depth:
                    self.world.label[:, j, :][hole] = 0
                    self.world.T[:, j, :][wall] = np.maximum(self.world.T[:, j, :][wall], L["temperature_k"])
                elif z == z0 + depth:                             # the crater's floor
                    self.world.T[:, j, :][hole] = np.maximum(self.world.T[:, j, :][hole], L["temperature_k"])
        self.world.T[self.world.label == 0] = self.world.ambient_T
        self.hit = True

    def step(self, dt):
        """The first step of an armed run is the impact itself (drawn at its hottest); then heat flows."""
        if self.armed and not self.hit:
            self.impact()
            return
        if self.hit:
            self.world.step(dt)
            self.t = self.world.t

    def verdict(self):
        if not self.layers:
            return "ready"
        through = sum(1 for L in self.layers if not L.get("gap") and L["penetrated"])
        total = sum(1 for s in self.spans if s is not None)
        return "BREACHED" if through >= total else f"stopped in layer {through + 1} of {total}"

    def readout(self):
        hot = float(self.world.T.max())
        return (f"{self.names[0]} -> {self.names[1]} | {self.verdict()} | hottest {hot:.4g} K | t {self.t:.2f} s | "
                f"vented {self.world.ledger['vented_kg'] * 1e3:.3g} g")
