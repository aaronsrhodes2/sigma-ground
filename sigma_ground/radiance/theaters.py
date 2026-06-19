"""Radiance theaters — the staging presets a Deckard object or a Materia sim
drops into.

A *theater* is the environment AROUND the content: its boundaries (walls / floor
/ none), its lights, its camera, and the PHYSICS it implies (gravity, the medium,
temperature). Radiance owns the render half (geometry, lights, camera); the
physics half is stamped as a ``physics_env`` block for the **Materia** lane to
honour when it runs a dynamic sim — Radiance never simulates, it only stages.

Four canonical contexts (doctrine: physics to screen, nothing in between — every
light here is real (the sun, or a blackbody radiator) or honestly artificial (a
studio rig); empty space stays black):

  room        bounded indoor — a lit box, gravity, air. Convection / drag / bounced light.
  plain       open outdoor — one ground plane, a sun, gravity, air. The drop stage.
  void        free-space STUDIO — no boundaries, 8 lights ringing the content,
              vacuum, zero-g. Inspect an object from every side.
  deep_space  free space, NO artificial light — matter lights ITSELF (incandescence,
              a star), vacuum, no floor. The SSBM cosmological stage.

Lights are physically grounded: a directional 'sun' (``scene.lights[0]``) or a hot
glowing ``Sphere`` (``temperature_k`` > 700 → the renderer's NEE area-light path).
A studio light is a 5500 K blackbody sphere — a real radiator, artificially placed.

    from sigma_ground.radiance.ask_deckard import request_shape_scene
    from sigma_ground.radiance.theaters import stage
    scene = stage(request_shape_scene("coffee cup"), "room")
"""
from __future__ import annotations

import copy
import math

from .scene_export import _bake_material, _light_color, _perp_frame


# ── building blocks ──────────────────────────────────────────────────────
def _r(v):
    return [round(x, 5) for x in v]


def _metrics(content):
    """Content centre and bounding radius, in the z-up frame Deckard builds in."""
    (x0, x1), (y0, y1), (z0, z1) = content["bbox"]
    c = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
    r = 0.5 * math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) or 0.1
    return (x0, x1, y0, y1, z0, z1), c, r


def _light_material():
    """A pure emitter: black cold surface, perfect (ε=1) radiator. At its leaf's
    ``temperature_k`` it incandesces (Planck × ε), and the renderer treats the
    hot Sphere as an NEE area light. The cold albedo is black so it only emits —
    it never reflects the scene back."""
    return {"color_rgb": [0.0, 0.0, 0.0], "emergent": True, "metal": False,
            "emissivity_rgb": [1.0, 1.0, 1.0],
            "mechanism": "artificial light (blackbody radiator)"}


def _light_leaf(center, radius, temp_k):
    return {"op": "add", "material": "studio_light",
            "shape": {"type": "Sphere", "center": _r(center), "radius": round(radius, 5)},
            "temperature_k": temp_k}


def _box_leaf(material, center, ext):
    """A Box leaf; ``ext`` is the FULL (x, y, z) extents (the viewer halves them)."""
    return {"op": "add", "material": material,
            "shape": {"type": "Box", "center": _r(center),
                      "x": round(ext[0], 5), "y": round(ext[1], 5), "z": round(ext[2], 5)}}


def _no_sun():
    """An explicit zero light so the PT does NOT fall back to its default sun
    (viewer.js: lights[0] || default). Used by the lightless / studio theaters."""
    return [{"dir": [0.0, 0.0, -1.0], "color": [0.0, 0.0, 0.0], "intensity": 0.0,
             "temperature_k": 0.0}]


def _begin(content):
    s = copy.deepcopy(content)
    s.setdefault("csg_leaves", [])
    s.setdefault("materials", {})
    return s


# ── theaters ─────────────────────────────────────────────────────────────
def plain(content, *, sun_temp_k=5778, gravity=True):
    """Open outdoor: a ground plane, a sun, sky, gravity, air at STP. The drop stage."""
    s = _begin(content)
    (x0, x1, y0, y1, z0, z1), c, r = _metrics(s)
    up = [0.0, 0.0, 1.0]
    g = max(24 * r, 6.0)                                   # ground reaches past the view
    s["csg_leaves"].append(_box_leaf("ground", [c[0], c[1], z0 - 0.06], [g, g, 0.1]))
    s["materials"]["ground"] = _bake_material("concrete", 2400.0)
    U, S1, S2 = _perp_frame(up)

    def comb(a, b, cc):
        v = [a * S1[i] + b * S2[i] + cc * U[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [round(x / n, 4) for x in v]

    s["lights"] = [{"dir": comb(0.40, 0.30, -0.86),
                    "color": _light_color(sun_temp_k, [1.0, 0.96, 0.90]),
                    "intensity": 1.7, "temperature_k": sun_temp_k}]
    s["ambient"] = {"sky": [0.30, 0.42, 0.62], "ground": [0.17, 0.15, 0.13], "up": up}
    s["pt_env"] = 1.6
    s["camera"] = {"target": [c[0], c[1], z0 + (z1 - z0) * 0.45],
                   "orbit_radius": max(2.6 * r, 0.3), "fov_deg": 42.0,
                   "up": up, "az0": 0.6, "el0": 0.22}
    s["bbox"] = [[c[0] - 6 * r, c[0] + 6 * r], [c[1] - 6 * r, c[1] + 6 * r], [z0 - 0.2, z1 + 4 * r]]
    s["physics_env"] = {"gravity_m_s2": 9.81 if gravity else 0.0, "gravity_dir": [0, 0, -1],
                        "medium": "air", "pressure_pa": 101325.0, "temperature_k": 273.15}
    s["theater"] = "plain"
    return s


def room(content, *, light_temp_k=5000, gravity=True, temperature_k=293.15):
    """Bounded indoor: a lit box (concrete floor, plaster walls), four corner lights
    near the ceiling, gravity, air. Walls bounce the light to fill the room."""
    s = _begin(content)
    (x0, x1, y0, y1, z0, z1), c, r = _metrics(s)
    up = [0.0, 0.0, 1.0]
    half = max(2.1 * r, 0.42)                              # room half-width (x, y) — kept tight so the
    H = max(3.2 * r, 0.65)                                 # corner lights stay close (3-bounce GI is shallow)
    top = z0 + H
    t = max(0.04, 0.06 * r)                                # wall thickness
    s["materials"]["floor"] = _bake_material("concrete", 2400.0)
    s["materials"]["wall"] = _bake_material("gypsum", 2300.0)        # plaster — white, bounces light
    s["csg_leaves"].append(_box_leaf("floor", [c[0], c[1], z0 - t / 2], [2 * half, 2 * half, t]))
    s["csg_leaves"].append(_box_leaf("wall", [c[0], c[1], top + t / 2], [2 * half, 2 * half, t]))   # ceiling
    for sx in (-1, 1):
        s["csg_leaves"].append(_box_leaf("wall", [c[0] + sx * half, c[1], z0 + H / 2], [t, 2 * half, H]))
    for sy in (-1, 1):
        s["csg_leaves"].append(_box_leaf("wall", [c[0], c[1] + sy * half, z0 + H / 2], [2 * half, t, H]))
    s["materials"]["studio_light"] = _light_material()
    lr = max(0.6 * r, 0.06)                                # big softbox-like emitters → strong DIRECT light
    for sx in (-1, 1):
        for sy in (-1, 1):
            s["csg_leaves"].append(_light_leaf(
                [c[0] + sx * half * 0.78, c[1] + sy * half * 0.78, top - H * 0.14], lr, light_temp_k))
    s["lights"] = _no_sun()                                # the corner lights do the work
    s["ambient"] = {"sky": [0.04, 0.04, 0.05], "ground": [0.03, 0.03, 0.03], "up": up}
    s["pt_env"] = 0.0                                      # enclosed — the sky never reaches in
    s["camera"] = {"target": [c[0], c[1], z0 + (z1 - z0) * 0.5],
                   "orbit_radius": max(1.7 * r, 0.24), "fov_deg": 52.0,
                   "up": up, "az0": 0.7, "el0": 0.16}
    s["bbox"] = [[c[0] - half - t, c[0] + half + t], [c[1] - half - t, c[1] + half + t],
                 [z0 - t, top + t]]
    s["physics_env"] = {"gravity_m_s2": 9.81 if gravity else 0.0, "gravity_dir": [0, 0, -1],
                        "medium": "air", "pressure_pa": 101325.0, "temperature_k": temperature_k}
    s["theater"] = "room"
    return s


def void(content, *, light_temp_k=5500):
    """Free-space studio: no boundaries, eight blackbody lights at the corners of a
    cube around the content (even, all-sides fill), vacuum, zero-g. For inspection."""
    s = _begin(content)
    _bb, c, r = _metrics(s)
    up = [0.0, 0.0, 1.0]
    s["materials"]["studio_light"] = _light_material()
    L = 1.5 * r                                            # half-spacing of the light cube (beyond content)
    lr = max(0.18 * r, 0.025)                              # small, point-like — a ring of lights, not walls of white
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                s["csg_leaves"].append(_light_leaf(
                    [c[0] + sx * L, c[1] + sy * L, c[2] + sz * L], lr, light_temp_k))
    s["lights"] = _no_sun()
    s["ambient"] = {"sky": [0.015, 0.015, 0.02], "ground": [0.01, 0.01, 0.012], "up": up}
    s["pt_env"] = 0.0                                      # studio dark — only the 8 lights illuminate
    s["camera"] = {"target": list(c), "orbit_radius": max(4.0 * r, 0.34), "fov_deg": 38.0,
                   "up": up, "az0": 0.6, "el0": 0.22}      # pulled back so no light sits on the lens
    s["bbox"] = [[c[0] - 2.4 * L, c[0] + 2.4 * L], [c[1] - 2.4 * L, c[1] + 2.4 * L], [c[2] - 2.4 * L, c[2] + 2.4 * L]]
    s["physics_env"] = {"gravity_m_s2": 0.0, "medium": "vacuum", "pressure_pa": 0.0,
                        "temperature_k": 293.15}
    s["theater"] = "void"
    return s


def deep_space(content):
    """Free space, NO artificial light: the matter lights itself (incandescence — a
    star is its own light), vacuum, no floor. Cold matter is invisible here, by
    design — the SSBM cosmological stage where only tangle-connected matter renders."""
    s = _begin(content)
    _bb, c, r = _metrics(s)
    up = [0.0, 0.0, 1.0]
    s["lights"] = _no_sun()
    s["ambient"] = {"sky": [0.0, 0.0, 0.0], "ground": [0.0, 0.0, 0.0], "up": up}
    s["pt_env"] = 0.0                                      # the void; matter is its own light
    s["camera"] = {"target": list(c), "orbit_radius": max(3.2 * r, 0.3), "fov_deg": 40.0,
                   "up": up, "az0": 0.6, "el0": 0.18}
    s["bbox"] = [[c[0] - 4 * r, c[0] + 4 * r], [c[1] - 4 * r, c[1] + 4 * r], [c[2] - 4 * r, c[2] + 4 * r]]
    s["physics_env"] = {"gravity_m_s2": 0.0, "medium": "vacuum", "pressure_pa": 0.0,
                        "temperature_k": 2.725}            # the CMB floor
    s["theater"] = "deep_space"
    return s


THEATERS = {"plain": plain, "room": room, "void": void, "deep_space": deep_space}


def stage(content: dict, theater: str = "plain", **opts) -> dict:
    """Drop ``content`` (a Deckard/Radiance SceneSpec) into a named theater and
    return the staged scene. Raises KeyError on an unknown theater name."""
    if content is None:
        return None
    if theater not in THEATERS:
        raise KeyError(f"unknown theater {theater!r}; choose from {sorted(THEATERS)}")
    return THEATERS[theater](content, **opts)
