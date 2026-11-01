"""
Controls from inside the game (SKIPPYCRAFT_PLAN 10b, P6; section 8): what the Captain says to Skippy in chat ("Skippy,
heat that to 1500 K", "drop a gold block there", "shoot", "push it north at 2 m/s", "pause", "step", "slower", "what is
that?") or types as ``/theater ...``, turned into actions on the running world. The plugin queues each line with where
he stands and the block he is looking at; the service polls it every tick (``theater.Blocks.poll``).

``parse(text)`` is pure: words -> ``{"kind", ...}`` (or None when it is not a command). ``Session`` holds the clock
(paused, speed, steps owed, the round for "shoot") and ``apply`` does the action on the world through the theater's
coordinate mapping, answering in words. Every number it puts into the world goes through the world's own methods
(``PlaygroundWorld.place / set_temperature / push / impact / remove``), which keep the books.

Directions: Minecraft's x is the grid's i, its y the grid's k (up), its z runs AGAINST the grid's j (``Theater.mc``), so
north (-z) is +j, east (+x) is +i, up is +k.
"""
from __future__ import annotations

import math
import re

import numpy as np

# what he can ask for by name -> the material the world and the pack know (block names)
MATERIAL_WORDS = {
    "gold": "gold", "lead": "lead", "granite": "granite", "tungsten": "tungsten", "steel": "steel_mild",
    "iron": "iron", "copper": "copper", "aluminium": "aluminum", "aluminum": "aluminum", "oak": "wood_oak",
    "wood": "wood_oak", "water": "water", "glass": "glass",
}
PLACEABLE = tuple(sorted(set(MATERIAL_WORDS.values())))
PLACE_PITCH = 0.005                      # the pack carries every placeable material at this block size (the playground's)
DEFAULT_SIZE_M = 0.02
DEFAULT_HEAT_RADIUS_M = 0.02
DEFAULT_PUSH_MS = 1.0
ROUNDS = ("ak47", "m2_browning", "pistol_9mm", "crossbow_bolt", "armor_piercing_30mm", "sabot_apfsds",
          "railgun_hypersonic", "hel_100kw")
DIRECTIONS = {"north": (0, 1, 0), "south": (0, -1, 0), "east": (1, 0, 0), "west": (-1, 0, 0), "up": (0, 0, 1),
              "down": (0, 0, -1)}

_NUM = r"(-?\d+(?:\.\d+)?)"


def _temperature(text):
    m = re.search(_NUM + r"\s*(k\b|kelvin|°c|c\b|celsius|°f|f\b|fahrenheit)?", text)
    if not m:
        return None
    v, u = float(m.group(1)), (m.group(2) or "k").lower()
    if u.startswith("°c") or u.startswith("c"):
        return v + 273.15
    if u.startswith("°f") or u.startswith("f"):
        return (v - 32) * 5 / 9 + 273.15
    return v


def _length(text, default):
    m = re.search(_NUM + r"\s*(mm|cm|m)\b", text)
    if not m:
        return default
    return float(m.group(1)) * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[m.group(2)]


def _speed(text, default):
    m = re.search(_NUM + r"\s*m/s", text)
    return float(m.group(1)) if m else default


def _material(text):
    for word, name in MATERIAL_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return name
    return None


def _direction(text):
    for word, d in DIRECTIONS.items():
        if re.search(rf"\b{word}\b", text):
            return d
    if re.search(r"\b(away|forward|that way)\b", text):
        return "look"
    if re.search(r"\b(to(wards?)? me|back)\b", text):
        return "back"
    return None


def parse(text: str):
    """Words -> an action, or None. Kinds: pause, play, step, speed, heat, place, push, shoot, remove, what, run, stop,
    reset, help."""
    t = text.strip().lower().rstrip("?.!")
    if not t:
        return None
    if re.fullmatch(r"(pause|freeze|hold( it)?|stop time)", t):
        return {"kind": "pause"}
    if re.fullmatch(r"(play|resume|go on|continue|unpause|run time)", t):
        return {"kind": "play"}
    if m := re.fullmatch(r"step(?: (\d+))?(?: ticks?)?", t):
        return {"kind": "step", "n": int(m.group(1) or 1)}
    if re.fullmatch(r"(slower|slow(er)? down|slow motion|slo-?mo)", t):
        return {"kind": "speed", "factor": 0.5}
    if re.fullmatch(r"(faster|speed up)", t):
        return {"kind": "speed", "factor": 2.0}
    if re.fullmatch(r"(normal speed|real time|full speed|1x)", t):
        return {"kind": "speed", "set": 1.0}
    if m := re.fullmatch(r"(?:speed|x)\s*" + _NUM + r"x?", t) or re.fullmatch(_NUM + r"\s*x", t):
        return {"kind": "speed", "set": float(m.group(1))}
    if re.fullmatch(r"(stop|halt|end( it)?)", t):
        return {"kind": "stop"}
    if re.fullmatch(r"(reset|start over|again|restart)", t):
        return {"kind": "reset"}
    if m := re.fullmatch(r"(?:run|start|open|go to)(?: the)? ([a-z]+)(?: theater| scene)?", t):
        return {"kind": "run", "scene": m.group(1)}
    if re.search(r"\b(what is|what's|whats|tell me about|describe)\b", t):
        return {"kind": "what"}
    if re.fullmatch(r"(help|what can (i|you) do|commands)", t):
        return {"kind": "help"}
    if re.search(r"\b(heat|warm|cool|chill|freeze)\b", t) and _temperature(re.sub(r"\b\d+\s*(mm|cm|m)\b", "", t)) is not None:
        return {"kind": "heat", "temperature_k": _temperature(re.sub(r"\b\d+\s*(mm|cm|m)\b", "", t)),
                "radius_m": _length(t, DEFAULT_HEAT_RADIUS_M)}
    if re.search(r"\b(shoot|fire)\b", t):
        weapon = next((r for r in ROUNDS if r.replace("_", " ") in t or r in t), None)
        return {"kind": "shoot", "weapon": weapon}
    if re.search(r"\b(push|shove|kick|nudge|hit)\b", t):
        return {"kind": "push", "direction": _direction(t), "speed_ms": _speed(t, DEFAULT_PUSH_MS)}
    if re.search(r"\b(remove|delete|take away|clear)\b", t):
        return {"kind": "remove", "radius_m": _length(t, DEFAULT_HEAT_RADIUS_M)}
    if re.search(r"\b(drop|place|put|add|make|build|spawn|give me)\b", t):
        mat = _material(t)
        if mat is None:
            return {"kind": "place", "material": None}
        T = _temperature(re.sub(r"\b\d+(?:\.\d+)?\s*(mm|cm|m)\b", "", t)) if re.search(r"\b(at|hot|cold)\b", t) else None
        return {"kind": "place", "material": mat, "size_m": _length(t, DEFAULT_SIZE_M),
                "drop": bool(re.search(r"\b(drop|above|over)\b", t)), "temperature_k": T}
    return None


HELP = ("Say: pause, play, step [n], slower, faster, speed 0.25; heat that to 1500 K [radius 3 cm]; drop a gold block "
        "[2 cm] [at 800 K]; push it north at 2 m/s; shoot [ak47]; remove that; what is that; run playground; stop; reset. "
        "Materials: " + ", ".join(sorted(MATERIAL_WORDS)) + ".")


class Session:
    """The clock and the aim, across ticks: ``paused``, ``speed`` (simulated seconds per real second, times the scene's
    own rate), ``steps`` owed while paused, ``weapon`` for "shoot"."""

    def __init__(self):
        self.paused, self.speed, self.steps, self.weapon = False, 1.0, 0, ROUNDS[0]
        self.request = None                  # a scene to run or STOP, for the service loop

    def tick_allowed(self):
        """Whether the loop should advance the world this tick (and uses up a step)."""
        if not self.paused:
            return True
        if self.steps > 0:
            self.steps -= 1
            return True
        return False

    def apply(self, msg, world, theater):
        """One message from the plugin: ``{"player", "text", "eye": (x, y, z), "look": (dx, dy, dz), "hit", "block":
        (x, y, z)}``. Returns what to tell him."""
        a = parse(msg["text"])
        if a is None:
            return "I don't know that one. " + HELP
        k = a["kind"]
        if k == "help":
            return HELP
        if k == "pause":
            self.paused = True
            return "Paused. Say step, or play."
        if k == "play":
            self.paused, self.steps = False, 0
            return "Running."
        if k == "step":
            self.paused = True
            self.steps += a["n"]
            return f"Stepping {a['n']}."
        if k == "speed":
            self.speed = a["set"] if "set" in a else max(0.03125, min(8.0, self.speed * a["factor"]))
            return f"Speed {self.speed:g}x."
        if k == "stop":
            self.request = "stop"
            return "Stopping."
        if k == "reset":
            self.request = "reset"
            return "Starting this theater over."
        if k == "run":
            self.request = a["scene"]
            return f"Running {a['scene']}."
        if world is None or theater is None:
            return "No theater is running. Say run playground (or sink, melt, slosh, slit, range, strength, impact)."
        if not hasattr(world, "place_above"):
            return "This theater is an exhibit; the playground, strength and impact theaters take these."
        # world actions need a target
        grid = self._target(msg, theater, world)
        if k == "what":
            if grid is None:
                return "Look at a block of the theater."
            return world.card(grid)
        if k == "heat":
            if grid is None:
                return "Look at a block of the theater to heat."
            return world.set_temperature_around(grid, a["radius_m"], a["temperature_k"])
        if k == "push":
            if grid is None:
                return "Look at the body to push."
            d = a["direction"]
            if d is None or d == "look":
                v = self._look_grid(msg)
            elif d == "back":
                v = -self._look_grid(msg)
            else:
                v = np.array(d, float)
            return world.push_at(grid, v * a["speed_ms"])
        if k == "remove":
            if grid is None:
                return "Look at what to remove."
            return world.remove_around(grid, a["radius_m"])
        if k == "place":
            if a["material"] is None:
                return "Which material? " + ", ".join(sorted(MATERIAL_WORDS)) + "."
            if grid is None:
                return "Look at a block of the theater: it goes on top of it."
            return world.place_above(grid, a["material"], a["size_m"], a.get("temperature_k"), body=True)
        if k == "shoot":
            if a["weapon"]:
                self.weapon = a["weapon"]
            if not hasattr(world, "impact"):
                return "This theater cannot be shot at."
            import shooting_range
            origin = self._eye_grid(msg, theater, world)
            if origin is None:
                return "Stand where the theater can see you (inside its box of air) to shoot."
            rec = world.impact(origin, self._look_grid(msg), shooting_range.layer_solver(self.weapon),
                               shooting_range.hole_radius_m(self.weapon))
            if not rec:
                return f"{self.weapon}: missed everything."
            return f"{self.weapon}:" + world.shot_line().split("shot:", 1)[1]
        return "I don't know that one."

    # ── where he looks, in the grid ───────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _target(msg, theater, world):
        if not msg.get("hit"):
            return None
        cell = theater.cell(*msg["block"])
        if cell is None or not all(0 <= c < n for c, n in zip(cell, world.label.shape)):
            return None
        return tuple(int(c) for c in cell)

    @staticmethod
    def _look_grid(msg):
        dx, dy, dz = msg["look"]
        v = np.array([dx, dz * -1.0, dy])            # MC (x, y, z) -> grid (i, j, k): j runs against z, k is y
        return v / max(np.linalg.norm(v), 1e-12)

    @staticmethod
    def _eye_grid(msg, theater, world):
        """His eye in grid metres: a shot starts there (its path then crosses the theater's cells)."""
        x, y, z = msg["eye"]
        cell = theater.cell_float(x, y, z)
        p = np.asarray(cell) * world.pitch
        if (p < 0).any() or (p >= np.array(world.label.shape) * world.pitch).any():
            # outside the grid: start the shot where his line of sight enters the grid's box
            d = Session._look_grid(msg)
            lo, hi = np.zeros(3), np.array(world.label.shape) * world.pitch
            t_in = max((((lo if d[i] > 0 else hi)[i] - p[i]) / d[i]) if d[i] else -math.inf for i in range(3))
            q = p + (t_in + 1e-6) * d
            if (q < 0).any() or (q >= hi).any():
                return None
            return q
        return p
