"""The matter record: one block of matter as the SkippyCraft mod carries it (SKIPPYCRAFT_PLAN 10b, after P6).

A record is three layers, and time enters only where the physics needs it:

  invariant   material, mass_kg, pitch_m                      (conserved or chosen: no clock)
  state       temperature_k, melt, phase, and, if it moves, t (sim seconds) with x, v      (a snapshot "as of t")
  look        rgba, light, emission, f0, metal, smoothness     (derived from the state by sigma-ground's optics)

Position is the block's own place in the world, so it is not in the record; ``t`` is None for a block that is not moving
and has no history (a static record needs no clock). Every value is from ``sigma_ground.field.interface.block``
(``block``, ``appearance``, ``glow``); what it cannot give is a named gap, never filled in. The look is drawn at the
display exposure ``export_mc.legible`` names, and says so (``drawn_at_m``); the physical opacity stays beside it.

``to_wire`` is what the mod receives: a JSON string, frame type 'SKCM' (see ``theater.MatterWriter``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = 1
MAX_LIGHT = 15


def record(material, mass_kg, pitch_m, temperature_k=293.15, *, t=None, v=(0.0, 0.0, 0.0), melt=0.0):
    """The record of ``mass_kg`` of ``material`` filling a cube ``pitch_m`` across, at ``temperature_k``.

    ``t``: the sim time (s) this state is as of, or None for a block with no clock. ``v``: its velocity (m/s); non-zero
    needs a ``t``. ``melt``: the fraction melted (0..1), from the world's own ledger."""
    import export_mc
    from sigma_ground.field.interface.block import appearance, block

    T = float(temperature_k)
    if any(float(c) != 0.0 for c in v) and t is None:
        raise ValueError("a moving block needs a time: velocity is a rate of change")
    b = block(material, T, mass_kg=float(mass_kg), volume_m3=float(pitch_m) ** 3)
    a = appearance(material, float(pitch_m), T)
    shown, drawn_at = export_mc.legible(a)
    gaps = list(dict.fromkeys(b["gaps"] + shown["gaps"]))
    rgba = [round(255 * export_mc._srgb(c)) for c in (shown["color_rgb"] or (0.49, 0.49, 0.49))]
    if shown["color_rgb"] is None:
        gaps.append("no colour model: drawn flat grey")
    rgba.append(round(255 * shown["opacity"]))

    light, emission, glow_rgb = 0, 0.0, [0, 0, 0]
    from sigma_ground.field.interface import thermal_emission as te
    if te.is_visibly_glowing(T):
        g_rgba, v_glow, g = export_mc.glow_display(material, max(T, export_mc.GLOW_BANDS_K[0]))
        light, emission, glow_rgb = round(MAX_LIGHT * v_glow), round(v_glow, 4), list(g_rgba[:3])
        gaps += g["gaps"]

    prop = b["properties"]
    return {
        "schema": SCHEMA,
        "material": material, "canonical": b["canonical"], "mass_kg": float(mass_kg), "pitch_m": float(pitch_m),
        "volume_m3": b["amount"].get("volume_m3"), "fill_fraction": b["amount"].get("fill_fraction"),
        "density": prop.get("density", {}).get("value"),
        "temperature_k": T, "phase": b["phase"], "melt": float(melt),
        "t": None if t is None else float(t), "v": [float(c) for c in v],
        "rgba": rgba, "opacity_physical": shown["opacity"] if drawn_at is None else a["opacity"], "drawn_at_m": drawn_at,
        "f0": None if shown["reflectance_f0"] is None else [float(c) for c in np.atleast_1d(shown["reflectance_f0"])],
        "metal": bool(shown["metal"]), "smoothness": shown["smoothness"],
        "light": int(light), "emission": emission, "glow_rgb": glow_rgb,
        "gaps": gaps,
    }


def to_wire(rec):
    """The record as the mod's JSON text (compact; unknown keys are ignored by the mod, so the schema can grow)."""
    return json.dumps(rec, separators=(",", ":"))
