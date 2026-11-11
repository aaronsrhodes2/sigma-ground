"""The matter record: one block of matter, as any renderer needs it, from sigma-ground's own numbers.

A renderer (the SkippyCraft Minecraft mod, a Unity or Blender scene, anything) draws a block of matter from this record and computes no
physics of its own. It is three layers, and time enters only where the physics needs it:

  invariant   material, mass_kg, pitch_m                      (conserved or chosen: no clock)
  state       temperature_k, melt, phase, and, if it moves, t (sim seconds) with v      (a snapshot "as of t")
  look        rgba, light, emission, glow_rgb, f0, metal, smoothness, n, absorption_per_m, tables   (derived from the state)

Position is the block's own place in the world, so it is not in the record; ``t`` is None for a block that is not moving. Every value is
from ``block`` / ``appearance`` / ``glow`` (and through them ``optics`` and ``thermal_emission``); what they cannot give is a named gap,
never filled in. For a CLEAR material the record carries sigma-ground's optics as ``tables`` (Fresnel reflectance and the refraction angle
at the interface, air into the material and back out, against the cosine of incidence; and the transmittance of 0..DEPTH_CELLS cells of
it), so the renderer only interpolates and places geometry. The one display choice is the glow's brightness, a flagged tonemap
(``thermal_emission.DISPLAY_GLOW_P0_W_M2``); the sRGB encoding of colours is ``optics.srgb_encode``.
"""
from __future__ import annotations

import json

import numpy as np

SCHEMA = 3
DEPTH_CELLS = 256         # the transmittance table reaches this many cells of the material (a deeper body reads the last entry)
LIGHT_LEVELS = 15         # a block's light level is an integer 0..15 (Minecraft's): the glow's display level, quantised


def matter_record(material, mass_kg, pitch_m, temperature_k=293.15, *, t=None, v=(0.0, 0.0, 0.0), melt=0.0,
                  light_levels=LIGHT_LEVELS):
    """The record of ``mass_kg`` of ``material`` filling a cube ``pitch_m`` across, at ``temperature_k``. ``mass_kg`` None: the cube is
    full, so its mass is the material's density (at that temperature) times the volume.

    ``t``: the sim time (s) this state is as of, or None for a block with no clock. ``v``: its velocity (m/s); non-zero needs a ``t``.
    ``melt``: the fraction melted (0..1), from the world's own ledger."""
    from . import optics
    from .block import appearance, block, glow

    T = float(temperature_k)
    if any(float(c) != 0.0 for c in v) and t is None:
        raise ValueError("a moving block needs a time: velocity is a rate of change")
    b = block(material, T, mass_kg=None if mass_kg is None else float(mass_kg), volume_m3=float(pitch_m) ** 3)
    if mass_kg is None:
        mass_kg = b["amount"].get("mass_kg", 0.0)
        if not b["amount"]:
            b["gaps"].append("no density: mass unknown, recorded as 0")
    a = appearance(material, float(pitch_m), T)
    gaps = list(dict.fromkeys(b["gaps"] + a["gaps"]))
    rgba = [round(255 * optics.srgb_encode(c)) for c in (a["color_rgb"] or (0.49, 0.49, 0.49))]
    if a["color_rgb"] is None:
        gaps.append("no colour model: drawn flat grey")
    rgba.append(round(255 * a["opacity"]))

    light, emission, glow_rgb = 0, 0.0, [0, 0, 0]
    g = glow(material, T)
    if g["visible"]:
        emission = round(g["level"], 4)
        light = round(light_levels * g["level"])
        glow_rgb = [round(255 * optics.srgb_encode(c)) for c in g["rgb"]]
        gaps += g["gaps"] + ["glow brightness is a display tonemap of the emitted power (thermal_emission.DISPLAY_GLOW_P0_W_M2)"]

    tables = None
    if a["n_refractive"] is not None:
        n = a["n_refractive"]

        def rounded(table):                      # 6 decimals: a table is data, and 17 digits would make every record 20 kB
            return {k: [round(x, 6) for x in vals] for k, vals in table.items()}
        tables = {"in": rounded(optics.interface_table(1.0, n)), "out": rounded(optics.interface_table(n, 1.0)),
                  "transmittance": {"step_m": float(pitch_m),
                                    "rgb": [[round(x, 6) for x in ch] for ch in
                                            optics.transmittance_table(a["absorption_per_m"], float(pitch_m), DEPTH_CELLS + 1)]}}
    prop = b["properties"]
    return {
        "schema": SCHEMA,
        "material": material, "canonical": b["canonical"], "mass_kg": float(mass_kg), "pitch_m": float(pitch_m),
        "volume_m3": b["amount"].get("volume_m3"), "fill_fraction": b["amount"].get("fill_fraction"),
        "density": prop.get("density", {}).get("value"),
        "temperature_k": T, "phase": b["phase"], "melt": float(melt),
        "t": None if t is None else float(t), "v": [float(c) for c in v],
        "rgba": rgba, "opacity_physical": a["opacity"], "n": a["n_refractive"],
        "absorption_per_m": None if a["absorption_per_m"] is None else list(a["absorption_per_m"]),
        "f0": None if a["reflectance_f0"] is None else [float(c) for c in np.atleast_1d(a["reflectance_f0"])],
        "metal": bool(a["metal"]), "smoothness": a["smoothness"],
        "light": int(light), "emission": emission, "glow_rgb": glow_rgb,
        "tables": tables,
        "gaps": gaps,
    }


def to_wire(record) -> str:
    """The record as compact JSON text (a renderer ignores keys it does not know, so the schema can grow)."""
    return json.dumps(record, separators=(",", ":"))


__all__ = ["matter_record", "to_wire", "SCHEMA", "DEPTH_CELLS", "LIGHT_LEVELS"]
