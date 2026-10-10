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


# ---------------------------------------------------------------------------------------------- S03
# What the engine applies for the stock surface the blocks use. MEASURED once (util.GetSurfaceData on the server);
# the grader re-checks it from every result file, so a changed engine data set fails loudly instead of silently.
STOCK_SURFACE = {"solidmetal": {"friction": 0.8, "elasticity": 0.1}}


def _frame_for_slope(theta_deg: float):
    """Unit vectors in the local frame for a plane that descends toward +x at theta: down-slope d and normal n."""
    t = math.radians(theta_deg)
    return (math.cos(t), 0.0, -math.sin(t)), (math.sin(t), 0.0, math.cos(t))


def spec_s03() -> dict:
    """Ramp and brake. Iron blocks on an iron-like plate and ramps, all handed the friction Mentat computes for
    iron on iron. Four kinds of lane, all in one scene: braking on the flat at three speeds, sliding down two steep
    ramps, sliding up two ramps (the block stops, then slides back), and a row of ramps around the critical angle."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]
    fa = rec("friction_analysis", {"material_key_1": "iron", "material_key_2": "iron", "normal_force_n": 10.0})["value"]
    mu_nom = fa["friction_force_N"] / 10.0                        # Mentat's friction coefficient for the pair
    rho = rec("density", {"material": "iron"})["value"]
    # The sliding body is a thin ring (a puck). MEASURED: a box on a plane drifts sideways and yaws (the engine
    # resolves its four corner contacts unevenly), which scatters the realized friction by +-5 %; a thin ring slides
    # dead straight with zero yaw and realizes exactly the product of the two surface frictions.
    r_out, r_in, bz = 0.2, 0.15, 0.05
    mass = rho * math.pi * (r_out ** 2 - r_in ** 2) * bz
    f_block = STOCK_SURFACE["solidmetal"]["friction"]
    f_plate = mu_nom / f_block                                    # the engine's friction law is a product of the two surfaces
    plate_surface = {"friction": f_plate, "elasticity": 0.0, "density": 7850}   # the ONE custom surface of this scene
    T, L, gap, spacing = 0.1, 2.4, 0.0105, 0.5                    # ramp thickness and length, drop gap, lane pitch (m)
    lanes = ([("brake", 0.0, v) for v in (1.5, 2.5, 3.5)] + [("down", a, 0.0) for a in (52.0, 60.0)] +
             [("up", a, v) for a, v in ((47.0, 3.0), (50.0, 3.0))] + [("crit", a, 0.0) for a in (38.0, 40.0, 42.0, 44.0, 46.0, 48.0)])
    y0 = -(len(lanes) - 1) * spacing / 2
    props, plan, preds = [], [], {}
    brake_lanes = [i for i, ln in enumerate(lanes) if ln[0] == "brake"]
    yb = [y0 + spacing * i for i in brake_lanes]
    props.append({"id": "plate", "shape": "box", "dims": {"size": [5.2, spacing * len(brake_lanes) + 0.1, T]}, "static": True,
                  "mass_kg": 400.0, "surface": plate_surface, "color": [0.4, 0.42, 0.48],
                  "pos_m": [2.7, sum(yb) / len(yb), T / 2]})
    cx = 2.4
    for i, (kind, theta, v0) in enumerate(lanes):
        y = y0 + spacing * i
        d, n = _frame_for_slope(theta)
        lane = {"lane": i, "kind": kind, "angle_deg": theta, "v0_m_s": v0, "y_m": y}
        if kind == "brake":
            c = [0.4, y, T + bz / 2 + gap]
            vel = [v0, 0.0, 0.0]
        else:
            zc = 0.35 + (L / 2) * math.sin(math.radians(theta))
            props.append({"id": f"ramp{i}", "shape": "box", "dims": {"size": [L, 0.46, T]}, "static": True, "mass_kg": 400.0,
                          "surface": plate_surface, "color": [0.4, 0.42, 0.48], "pos_m": [cx, y, zc], "ang_deg": [theta, 0.0, 0.0]})
            s0 = 0.7 if kind == "up" else -0.85                   # start position along the slope from the ramp centre (+ is down-slope)
            lift = T / 2 + bz / 2 + gap
            c = [cx + n[0] * lift + s0 * d[0], y, zc + n[2] * lift + s0 * d[2]]
            vel = [-v0 * d[0], 0.0, -v0 * d[2]]                   # launched up the slope
            lane["start_slope_m"] = s0
        props.append({"id": f"block{i}", "shape": "ring", "dims": {"outer_radius": r_out, "inner_radius": r_in, "height": bz, "seg": 32},
                      "mass_kg": mass,
                      "surface": "solidmetal", "color": [0.78, 0.5, 0.25], "pos_m": c, "ang_deg": [theta, 0.0, 0.0], "v0_m_s": vel})
        # what Mentat predicts for the friction it was asked to supply (the grader asks again at the friction the engine realized)
        if kind == "brake":
            lane["nominal"] = {"stopping_distance_m": rec("friction_stopping_distance",
                               {"mass_kg": mass, "initial_velocity_m_s": v0, "friction_coefficient": mu_nom, "g_m_s2": g})["value"]}
        else:
            ia = rec("incline_analysis", {"angle_deg": theta, "friction_coefficient": mu_nom, "height_m": 1.0, "speed_m_s": v0 or 1.0})["value"]
            lane["nominal"] = {"critical_angle_deg": ia["critical_sliding_angle_deg"], "slide_up_m": ia["sliding_distance_up_m"],
                               "speed_at_bottom_1m_m_s": ia["speed_at_bottom_m_s"]}
        plan.append(lane)
    spec = {
        "id": "S03", "name": "Ramp and brake", "gravity_m_s2": g,
        "needs": {"forward_m": 5.4, "left_m": spacing * len(lanes) + 0.6, "up_m": 3.2}, "focus_m": [2.4, 0.0, 1.2],
        "timeline": {"settle_s": 1.0, "run_s": 1.6}, "props": props,
        "predictions": {"mu_nominal": mu_nom, "friction_analysis": fa, "plate_friction": f_plate, "block_friction": f_block,
                        "block_mass_kg": mass, "block_ring_m": [r_out, r_in, bz], "lanes": plan},
        "live": [],
    }
    return _finish(spec, rec)


# ---------------------------------------------------------------------------------------------- S04
def spec_s04() -> dict:
    """Collision track. Pairs of ring pucks slide toward each other on ice and collide head-on. Lanes differ in surface
    (bounciness), mass ratio, and approach speed. Only the engine's stock surfaces are used, so no restitution is
    handed to it; the grader measures the restitution the engine realized and asks Mentat about the same collision."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]
    r_out, r_in, hz, gap, spacing = 0.2, 0.15, 0.05, 0.0105, 0.55
    plate_len, x_a, x_b = 6.0, 1.0, 2.4
    # (label, puck surface, m1 kg, v1 m/s, m2 kg, v2 m/s)
    cases = [("popcan, equal masses, one at rest", "popcan", 5.0, 2.0, 5.0, 0.0),
             ("popcan, equal masses, head-on", "popcan", 5.0, 1.0, 5.0, -1.0),
             ("popcan, heavy hits light", "popcan", 10.0, 2.0, 2.5, 0.0),
             ("popcan, light hits heavy", "popcan", 2.5, 2.0, 10.0, 0.0),
             ("default, equal masses, head-on", "default", 5.0, 2.0, 5.0, -2.0),
             ("default, heavy hits light, light moving", "default", 10.0, 2.0, 2.5, -1.0),
             ("tile, equal masses, one at rest", "tile", 5.0, 2.0, 5.0, 0.0),
             ("popcan, slow", "popcan", 5.0, 0.5, 5.0, -0.5),
             ("popcan, fast", "popcan", 5.0, 2.0, 5.0, -2.0)]
    y0 = -(len(cases) - 1) * spacing / 2
    props = [{"id": "plate", "shape": "box", "dims": {"size": [plate_len, spacing * len(cases) + 0.2, 0.1]}, "static": True, "mass_kg": 600.0,
              "surface": "gmod_ice", "color": [0.55, 0.62, 0.7], "pos_m": [plate_len / 2 - 0.4, 0.0, 0.05]}]
    plan = []
    for i, (label, surf, m1, v1, m2, v2) in enumerate(cases):
        y = y0 + spacing * i
        for tag, x, m, v in (("a", x_a, m1, v1), ("b", x_b, m2, v2)):
            props.append({"id": f"{tag}{i}", "shape": "ring", "dims": {"outer_radius": r_out, "inner_radius": r_in, "height": hz, "seg": 32},
                          "mass_kg": m, "surface": surf, "color": [0.85, 0.55, 0.25] if tag == "a" else [0.3, 0.6, 0.85],
                          "pos_m": [x, y, 0.1 + hz / 2 + gap], "v0_m_s": [v, 0.0, 0.0]})
        elastic = rec("collision_analysis", {"mass1_kg": m1, "velocity1_m_s": v1, "mass2_kg": m2, "velocity2_m_s": v2, "restitution": 1.0})["value"]
        hz_cor = rec("hertzian_impact_analysis", {"velocity_m_s": abs(v1 - v2)})["value"]["coefficient_of_restitution"]
        plan.append({"lane": i, "label": label, "surface": surf, "m1": m1, "v1": v1, "m2": m2, "v2": v2,
                     "elastic_v1": elastic["elastic_v1_m_s"], "elastic_v2": elastic["elastic_v2_m_s"],
                     "hertz_cor_iron": hz_cor})
    spec = {
        "id": "S04", "name": "Collision track", "gravity_m_s2": g,
        "needs": {"forward_m": plate_len + 0.2, "left_m": spacing * len(cases) + 0.6, "up_m": 1.0}, "focus_m": [2.0, 0.0, 0.3],
        "timeline": {"settle_s": 1.0, "run_s": 2.0}, "props": props,
        "predictions": {"puck_ring_m": [r_out, r_in, hz], "lanes": plan}, "live": [],
    }
    return _finish(spec, rec)


# ---------------------------------------------------------------------------------------------- S05
def _sim_steps(rec, verb: str, params: dict) -> dict:
    """The labelled quantities of a Mentat simulation verb, as floats where they are numbers."""
    r = rec("run_simulation", {"verb": verb, "params": params})
    out = {}
    for st in r["inputs"]["chain"][0]["steps"]:
        try:
            out[st["label"]] = float(st["value"])
        except (TypeError, ValueError):
            out[st["label"]] = st["value"]
    return out


def spec_s05() -> dict:
    """Rolling race and inertia. Three bodies of equal mass and radius roll down the same ramp: a solid sphere, a
    solid disc and a hollow sphere (a sphere body whose inertia is set to the hollow value). Mentat supplies every
    moment of inertia and every rolling acceleration. Four more bodies keep the engine's own inertia so it can be
    compared with Mentat's formulas. The gear train and hand tool of the plan are not built here."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]
    m, r, h_disc, theta = 2.0, 0.15, 0.1, 20.0
    steps = _sim_steps(rec, "rolling_object", {"mass_kg": m, "radius_m": r, "incline_angle_deg": theta, "height_m": 1.0})
    I_sphere, I_hollow, I_disk = steps["moment of inertia sphere"], steps["moment of inertia hollow sphere"], steps["moment of inertia disk"]
    rd = rec("rotational_dynamics", {"mass_kg": m, "radius_m": r, "angle_deg": theta})["value"]
    assert abs(rd["shape_moment_of_inertia_kg_m2"] - I_sphere) < 1e-9          # Mentat's two routes agree on the sphere
    I_tr = float(rec("simplify_expr", {"expression": f"{m}*(3*{r}**2+{h_disc}**2)/12"})["value"])   # disc about a diameter

    def a_roll(I):                                                              # rolling acceleration: a_roll = g sin(th) / (1 + I / (m r^2))
        eq = f"a*(1+{I}/({m}*{r}**2)) - {g}*sin({theta}*pi/180)"                # (the formula Mentat prints for rotational_dynamics)
        return float(rec("solve_equation", {"equation": eq, "variable": "a"})["value"][0])

    bodies = [  # id, label, geometry, inertia (kg m^2 about local x, y, z), roll angle that puts the axis across the slope
        ("sphere", "solid sphere", ("sphere", {"radius": r}), [I_sphere] * 3, 0.0, rd["shape_rolling_acceleration_m_s2"]),
        ("disc", "solid disc", ("cylinder", {"radius": r, "height": h_disc, "seg": 96}), [I_tr, I_tr, I_disk], 90.0, a_roll(I_disk)),
        ("hollow", "hollow sphere", ("sphere", {"radius": r}), [I_hollow] * 3, 0.0, a_roll(I_hollow)),
    ]
    T, L, gap, spacing, cx = 0.1, 3.2, 0.0105, 0.6, 2.0
    th = math.radians(theta)
    d, n = (math.cos(th), 0.0, -math.sin(th)), (math.sin(th), 0.0, math.cos(th))
    ramp_surface = {"friction": 2.0, "elasticity": 0.0, "density": 7850}       # the ONE custom surface: grippy, so the bodies roll
    props, race = [], []
    zc = 0.35 + (L / 2) * math.sin(th)
    for i, (bid, label, (shape, dims), inertia, roll, a_pred) in enumerate(bodies):
        y = (i - 1) * spacing
        props.append({"id": f"ramp{i}", "shape": "box", "dims": {"size": [L, 0.46, T]}, "static": True, "mass_kg": 400.0,
                      "surface": ramp_surface, "color": [0.4, 0.42, 0.48], "pos_m": [cx, y, zc], "ang_deg": [theta, 0.0, 0.0]})
        s0, lift = -1.1, T / 2 + r + gap
        c = [cx + n[0] * lift + s0 * d[0], y, zc + n[2] * lift + s0 * d[2]]
        props.append({"id": bid, "shape": shape, "dims": dims, "mass_kg": m, "surface": "default", "color": [0.85, 0.55, 0.25],
                      "pos_m": c, "ang_deg": [theta, 0.0, roll], "inertia_si": inertia, "damping_angular": 0.0})
        race.append({"body": bid, "label": label, "inertia_si": inertia, "smooth": shape == "sphere",
                     "axis_inertia": inertia[2] if roll else inertia[0],
                     "k": (inertia[2] if roll else inertia[0]) / (m * r * r), "a_mentat": a_pred})
    # bodies that keep the engine's own inertia, for the audit
    audit_defs = [("a_sphere", "sphere r 0.15 m", "sphere", {"radius": r}, [I_sphere] * 3),
                  ("a_cyl", "cylinder r 0.15 m, h 0.1 m", "cylinder", {"radius": r, "height": h_disc, "seg": 32}, [I_tr, I_tr, I_disk]),
                  ("a_box", "box 0.3 x 0.2 x 0.1 m", "box", {"size": [0.3, 0.2, 0.1]}, None),
                  ("a_ring", "ring R 0.2, r 0.15, h 0.05 m", "ring", {"outer_radius": 0.2, "inner_radius": 0.15, "height": 0.05, "seg": 32}, None)]
    audit = []
    for j, (aid, label, shape, dims, I) in enumerate(audit_defs):
        if aid == "a_box":
            a_, b_, c_ = 0.3, 0.2, 0.1
            I = [float(rec("simplify_expr", {"expression": f"{m}*({p}**2+{q}**2)/12"})["value"]) for p, q in ((b_, c_), (a_, c_), (a_, b_))]
        if aid == "a_ring":
            R_, ri, hh = 0.2, 0.15, 0.05
            ax = float(rec("simplify_expr", {"expression": f"{m}*({R_}**2+{ri}**2)/2"})["value"])
            tr = float(rec("simplify_expr", {"expression": f"{m}*(3*({R_}**2+{ri}**2)+{hh}**2)/12"})["value"])
            I = [tr, tr, ax]
        props.append({"id": aid, "shape": shape, "dims": dims, "mass_kg": m, "surface": "default", "color": [0.55, 0.62, 0.7],
                      "pos_m": [1.0, -2.9 + 0.5 * j, 0.8]})
        audit.append({"body": aid, "label": label, "analytic_si": I})
    spec = {
        "id": "S05", "name": "Rolling race and inertia", "gravity_m_s2": g,
        "needs": {"forward_m": 4.2, "left_m": 7.4, "up_m": 2.6}, "focus_m": [2.0, 0.0, 1.0],
        "timeline": {"settle_s": 1.0, "run_s": 1.6}, "props": props,
        "predictions": {"mass_kg": m, "radius_m": r, "angle_deg": theta, "race": race, "audit": audit,
                        "sphere_inertia_mentat_two_routes": [I_sphere, rd["shape_moment_of_inertia_kg_m2"]]},
        "live": [],
    }
    return _finish(spec, rec)


# ---------------------------------------------------------------------------------------------- S06
def spec_s06() -> dict:
    """Pendulum and spring. Four pendulums of the same length swing from different starting angles on rigid ropes; three
    masses (1, 2, 4 kg) hang from the same spring and are released from its natural length; a fourth, 2 kg mass hangs
    from a damped spring. Mentat's ODE solver, series expansion and limits supply the predictions; the spring's
    stiffness and damping are measured from the engine itself, because their units are the engine's own."""
    rec = Recorder()
    g = rec("free_fall_time", {"height_m": 1.0})["inputs"]["g_m_s2"]
    L, bob_r, bob_m, z_p, x_p = 1.0, 0.05, 1.0, 3.4, 1.5
    amps = (5.0, 20.0, 40.0, 60.0)
    spacing = 0.55
    k_engine, c_engine = 40.0, 2.0                          # handed to constraint.Elastic as given; the grader measures what they became
    masses = (1.0, 2.0, 4.0)
    z_a, L0, x_s = 4.0, 1.0, 1.5
    ncol = len(amps) + len(masses) + 1
    y0 = -(ncol - 1) * spacing / 2
    props, constraints, pend, spr = [], [], [], []
    for i, th in enumerate(amps):
        y = y0 + spacing * i
        t = math.radians(th)
        bob = [x_p + L * math.sin(t), y, z_p - L * math.cos(t)]
        props.append({"id": f"pivot{i}", "shape": "box", "dims": {"size": [0.08, 0.08, 0.08]}, "static": True, "mass_kg": 50.0,
                      "color": [0.5, 0.5, 0.55], "pos_m": [x_p, y, z_p]})
        props.append({"id": f"bob{i}", "shape": "sphere", "dims": {"radius": bob_r}, "mass_kg": bob_m, "color": [0.85, 0.55, 0.25],
                      "pos_m": bob, "damping_linear": 0.0})
        constraints.append({"kind": "rope", "a": f"pivot{i}", "b": f"bob{i}", "a_point_m": [x_p, y, z_p], "b_point_m": bob})
        pend.append({"lane": i, "start_deg": th, "y_m": y})
    T0 = float(rec("simplify_expr", {"expression": f"2*3.141592653589793*sqrt({L}/{g})"})["value"])     # small-angle period, 2 pi sqrt(L/g)
    ode_free = rec("solve_ode", {"equation": "diff(y,x,2) + w**2*y"})["value"]                        # the general solution: sin and cos of w x
    series = rec("series_expansion", {"expression": "2/pi*elliptic_k(sin(x/2)**2)", "variable": "x", "order": 12})["value"]
    lim = rec("compute_limit", {"expression": "sin(x)/x", "variable": "x", "point": "0"})["value"]
    sin_series = rec("series_expansion", {"expression": "sin(x)", "variable": "x", "order": 6})["value"]
    for p_ in pend:
        th = math.radians(p_["start_deg"])
        ratio = float(rec("simplify_expr", {"expression": series.replace("x", f"({th!r})")})["value"])    # T / T0 from Mentat's series for the exact period
        p_["period_ratio"], p_["period_s"] = ratio, T0 * ratio
    for j, m in enumerate(masses + (2.0,)):
        i = len(amps) + j
        y = y0 + spacing * i
        damped = j == len(masses)
        props.append({"id": f"anchor{j}", "shape": "box", "dims": {"size": [0.08, 0.08, 0.08]}, "static": True, "mass_kg": 50.0,
                      "color": [0.5, 0.5, 0.55], "pos_m": [x_s, y, z_a]})
        props.append({"id": f"mass{j}", "shape": "sphere", "dims": {"radius": 0.07}, "mass_kg": m, "color": [0.3, 0.6, 0.85],
                      "pos_m": [x_s, y, z_a - L0], "damping_linear": 0.0})
        constraints.append({"kind": "elastic", "a": f"anchor{j}", "b": f"mass{j}", "a_point_m": [x_s, y, z_a], "b_point_m": [x_s, y, z_a - L0],
                            "constant": k_engine, "damping": c_engine if damped else 0.0})
        spr.append({"lane": j, "mass_kg": m, "damped": damped, "y_m": y, "constant_engine": k_engine, "damping_engine": c_engine if damped else 0.0})
    spec = {
        "id": "S06", "name": "Pendulum and spring", "gravity_m_s2": g,
        "needs": {"forward_m": 3.2, "left_m": spacing * ncol + 0.8, "up_m": 4.2}, "focus_m": [1.5, 0.0, 2.6],
        "timeline": {"settle_s": 1.0, "run_s": 7.0}, "props": props, "constraints": constraints,
        "predictions": {"length_m": L, "pivot_m": [x_p, z_p], "natural_length_m": L0, "anchor_z_m": z_a, "small_angle_period_s": T0,
                        "ode_undamped_general_solution": ode_free, "period_ratio_series": series, "limit_sin_x_over_x": lim,
                        "sin_series": sin_series, "pendulums": pend, "springs": spr},
        "live": [],
    }
    return _finish(spec, rec)


SPECS = {"S01": spec_s01, "S02": spec_s02, "S03": spec_s03, "S04": spec_s04, "S05": spec_s05, "S06": spec_s06}


def build(sid: str) -> dict:
    return SPECS[sid]()
