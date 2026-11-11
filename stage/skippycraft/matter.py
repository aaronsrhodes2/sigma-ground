"""The matter record: one block of matter as the SkippyCraft mod carries it (SKIPPYCRAFT_PLAN 10b, after P6).

A record is three layers, and time enters only where the physics needs it:

  invariant   material, mass_kg, pitch_m                      (conserved or chosen: no clock)
  state       temperature_k, melt, phase, and, if it moves, t (sim seconds) with x, v      (a snapshot "as of t")
  look        rgba, light, emission, f0, metal, smoothness, n, absorption_per_m, tables   (derived from the state by sigma-ground's optics)

Position is the block's own place in the world, so it is not in the record; ``t`` is None for a block that is not moving
and has no history (a static record needs no clock). Every value is from ``sigma_ground.field.interface.block``
(``block``, ``appearance``, ``glow``); what it cannot give is a named gap, never filled in. Nothing here is tuned for
the screen: ``rgba`` is the physical look at the block's own size, and for a clear material the record also carries ``n``
(index at 550 nm) and ``absorption_per_m`` (R, G, B), so the mod draws the Fresnel reflectance at the real angle and the
Beer-Lambert transmittance through the real path length of the body, the way a ray meets it. The mod carries NO physics: every
physical number it draws with is a table in ``tables`` that sigma-ground's ``optics`` computed here (Fresnel reflectance and the
refraction angle at the interface, air into the material and back out, and the transmittance of 0..256 cells of it); the mod only
interpolates them and places geometry. (``export_mc.legible`` is the
carrier-block path's stand-in for that; it goes when the theater writes matter blocks.)

``to_wire`` is what the mod receives: a JSON string, frame type 'SKCM' (see ``theater.MatterWriter``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = 2
DEPTH_CELLS = 256         # the transmittance table reaches this many cells of the material (a deeper body reads the last entry)
MAX_LIGHT = 15


def record(material, mass_kg, pitch_m, temperature_k=293.15, *, t=None, v=(0.0, 0.0, 0.0), melt=0.0):
    """The record of ``mass_kg`` of ``material`` filling a cube ``pitch_m`` across, at ``temperature_k``. ``mass_kg`` None:
    the cube is full, so its mass is the material's density (at that temperature) times the volume.

    ``t``: the sim time (s) this state is as of, or None for a block with no clock. ``v``: its velocity (m/s); non-zero
    needs a ``t``. ``melt``: the fraction melted (0..1), from the world's own ledger."""
    import export_mc
    from sigma_ground.field.interface.block import appearance, block

    T = float(temperature_k)
    if any(float(c) != 0.0 for c in v) and t is None:
        raise ValueError("a moving block needs a time: velocity is a rate of change")
    b = block(material, T, mass_kg=None if mass_kg is None else float(mass_kg), volume_m3=float(pitch_m) ** 3)
    if mass_kg is None:
        mass_kg = b["amount"].get("mass_kg", 0.0)
        if not b["amount"]:
            b["gaps"].append("no density: mass unknown, recorded as 0")
    a = appearance(material, float(pitch_m), T)
    shown = a
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

    tables = None
    if a["n_refractive"] is not None:
        from sigma_ground.field.interface import optics
        n = a["n_refractive"]
        def rounded(table):                      # 6 decimals: a table is data, and 17 digits would make every record 20 kB
            return {k: [round(v, 6) for v in vals] for k, vals in table.items()}
        tables = {"in": rounded(optics.interface_table(1.0, n)), "out": rounded(optics.interface_table(n, 1.0)),
                  "transmittance": {"step_m": float(pitch_m),
                                    "rgb": [[round(v, 6) for v in ch] for ch in
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
        "f0": None if shown["reflectance_f0"] is None else [float(c) for c in np.atleast_1d(shown["reflectance_f0"])],
        "metal": bool(shown["metal"]), "smoothness": shown["smoothness"],
        "light": int(light), "emission": emission, "glow_rgb": glow_rgb,
        "tables": tables,
        "gaps": gaps,
    }


def to_wire(rec):
    """The record as the mod's JSON text (compact; unknown keys are ignored by the mod, so the schema can grow)."""
    return json.dumps(rec, separators=(",", ":"))
