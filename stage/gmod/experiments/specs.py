"""Experiment specs: the physical setup of each ENGINE scenario plus Mentat's predictions for it.

Every prediction is a Mentat tool call (through mentat_io, cached with provenance); this file only chooses the setup
and does bookkeeping arithmetic (a sphere's volume, a list of cases). A spec is engine-neutral data: the Lua
framework builds from it, the grader judges against it, and its `sha` ties a result file to the spec it ran.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mentat_io import call, provenance  # noqa: E402

VERSION = 1


def _sha(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k != "sha"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


class Recorder:
    """Collects the Mentat calls behind a spec so the spec carries its own provenance."""

    def __init__(self):
        self.items = []

    def __call__(self, tool, args):
        r = call(tool, args)
        self.items.append(provenance(tool, args, r))
        return r


def _finish(spec: dict, rec: Recorder) -> dict:
    spec["version"] = VERSION
    spec["provenance"] = rec.items
    spec["sha"] = _sha(spec)
    return spec


# ---------------------------------------------------------------------------------------------- S01
def spec_s01() -> dict:
    """Drop tower: four metal spheres of different mass fall together from rest; timing gates 1, 2 and 4 m below."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]       # Mentat's own standard gravity
    radius, z0 = 0.04, 5.0
    mats = [("copper", [0.85, 0.45, 0.25]), ("aluminum", [0.75, 0.78, 0.82]), ("lead", [0.35, 0.37, 0.42]),
            ("iron", [0.55, 0.55, 0.58])]
    vol = 4.0 / 3.0 * math.pi * radius ** 3                                  # geometry bookkeeping
    props, masses = [], {}
    for i, (m, col) in enumerate(mats):
        rho = rec("density", {"material": m})["value"]
        masses[m] = rho * vol
        props.append({"id": m, "label": f"{m} sphere", "shape": "sphere", "dims": {"radius": radius},
                      "mass_kg": masses[m], "material": m, "color": col,
                      "pos_m": [1.5, -0.45 + 0.3 * i, z0], "v0_m_s": [0.0, 0.0, 0.0]})
    gates = []
    for d in (1.0, 2.0, 4.0):
        t = rec("free_fall_time", {"height_m": d, "g_m_s2": g})
        v = rec("free_fall_velocity", {"height_m": d, "g_m_s2": g})
        per = {}
        for m, mass in masses.items():
            per[m] = {
                "kinetic_energy_J": rec("kinetic_energy", {"mass_kg": mass, "velocity_m_s": v["value"]})["value"],
                "momentum_kg_m_s": rec("momentum", {"mass_kg": mass, "velocity_m_s": v["value"]})["value"],
                "potential_energy_J": rec("gravitational_potential_energy",
                                          {"mass_kg": mass, "height_m": d, "g_m_s2": g})["value"],
            }
        gates.append({"drop_m": d, "z_m": z0 - d, "time_s": t["value"], "speed_m_s": v["value"], "per_prop": per})
    tnt = rec("joules_to_TNT", {"energy_joules": gates[-1]["per_prop"]["lead"]["kinetic_energy_J"], "unit": "ton"})
    live = [{"kind": "gate_time", "prop": p["id"], "z_m": z0 - 2.0, "label": f"{p['id']} falls 2 m",
             "predicted": gates[1]["time_s"], "units": "s"} for p in props]
    spec = {
        "id": "S01", "name": "Drop tower", "gravity_m_s2": g,
        "needs": {"forward_m": 2.0, "left_m": 1.6, "up_m": 5.6}, "focus_m": [1.5, 0.0, 3.0],
        "timeline": {"settle_s": 1.0, "run_s": 1.0},
        "props": props, "predictions": {"gates": gates, "lead_ke_in_TNT_tons_at_4m": tnt["value"]},
        "live": live,
    }
    return _finish(spec, rec)


# ---------------------------------------------------------------------------------------------- S02
def _num(x) -> float:
    """A Mentat symbolic tool returns numbers as strings (sympy); turn one into a float."""
    return float(str(x).replace("*", ""))


def spec_s02() -> dict:
    """Cannon range: three steel balls launched at 30, 45 and 60 degrees from 1 m above level ground, no air."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]
    speed, z0, x0, radius = 8.0, 1.0, 0.5, 0.03
    rho = rec("density", {"material": "iron"})["value"]
    mass = rho * 4.0 / 3.0 * math.pi * radius ** 3
    props, preds, live = [], {}, []
    for i, ang in enumerate((30.0, 45.0, 60.0)):
        a = math.radians(ang)
        vx, vz = speed * math.cos(a), speed * math.sin(a)                   # setup: the launch velocity components
        pid = f"ball_{int(ang)}"
        props.append({"id": pid, "label": f"steel ball at {int(ang)} deg", "shape": "sphere", "dims": {"radius": radius},
                      "mass_kg": mass, "material": "iron", "color": [0.55, 0.6, 0.7],
                      "pos_m": [x0, -0.8 + 0.8 * i, z0], "v0_m_s": [vx, 0.0, vz]})
        pa = rec("projectile_analysis", {"speed_m_s": speed, "angle_deg": ang})["value"]
        arg = {"initial_speed_m_s": speed, "launch_angle_deg": ang, "g_m_s2": g}
        rng, hmax, tof = (rec(t, arg)["value"] for t in ("projectile_range", "projectile_max_height", "projectile_flight_time"))
        proc = rec("procedure_projectile_trajectory", {"initial_speed_m_s": speed, "launch_angle_deg": ang})
        steps = {s["label"]: s["value"] for s in proc["steps"]}
        d_expr = rec("differentiate_expr", {"expression": f"{vz!r}*t - {g}*t**2/2", "variable": "t"})["value"]
        t_apex = _num(rec("solve_equation", {"equation": f"{vz!r} - {g}*t", "variable": "t"})["value"][0])
        roots = sorted(_num(r) for r in rec("solve_equation", {"equation": f"{vz!r}*t - {g}*t**2/2", "variable": "t"})["value"])
        preds[pid] = {
            "angle_deg": ang, "vx_m_s": vx, "vz_m_s": vz,
            "range_m": rng, "max_height_m": hmax, "flight_time_s": tof,
            "analysis": {"range_m": pa["range_m"], "max_height_m": pa["max_height_m"], "flight_time_s": pa["time_of_flight_s"]},
            "procedure": {"range_m": steps["Range"], "max_height_m": steps["Max height"], "flight_time_s": steps["Time of flight"]},
            "symbolic": {"vertical_velocity_expression": d_expr, "apex_time_s": t_apex, "return_time_s": roots[-1]},
        }
        live += [
            {"kind": "apex", "prop": pid, "label": f"{int(ang)} deg apex height", "predicted": hmax, "units": "m"},
            {"kind": "return_time", "prop": pid, "label": f"{int(ang)} deg flight time", "predicted": tof, "units": "s"},
            {"kind": "range_at_return", "prop": pid, "label": f"{int(ang)} deg range", "predicted": rng, "units": "m"},
        ]
    spec = {
        "id": "S02", "name": "Cannon range", "gravity_m_s2": g, "launch_speed_m_s": speed, "launch_height_m": z0,
        "needs": {"forward_m": 8.0, "left_m": 2.4, "up_m": 3.2}, "focus_m": [3.5, 0.0, 1.8],
        "timeline": {"settle_s": 1.0, "run_s": 1.5}, "props": props, "predictions": preds, "live": live,
    }
    return _finish(spec, rec)


SPECS = {"S01": spec_s01, "S02": spec_s02}


def build(sid: str) -> dict:
    return SPECS[sid]()
