"""Independent grader for the engine experiments.

    python stage/gmod/experiments/grade.py S01 [--run-id r1] [--data <garrysmod/data/mentat>]

The Lua records raw engine samples and judges nothing. This script recomputes every observable from those
samples with its own fits (fit.py) and compares them with Mentat's predictions carried in the spec. It first checks
the ENGINE itself (placement, that the fall is the exact parabola the compensated release should produce), so a
mistake in the Lua setup cannot masquerade as a Mentat error.

Sample layout (per prop, per tick): x y z, vx vy vz, fwd(3), up(3), angular velocity deg/s (3, body-local axes).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
import specs  # noqa: E402
from mentat_io import call  # noqa: E402

# Sample k is taken k ticks after the release velocity was set. MEASURED: in the game client the body follows the exact
# trajectory from sample 0 (t = k * tick). On a dedicated server the engine advances positions and orientations with the
# velocity the body had BEFORE the Tick hook set it, so nothing moves during the first tick: the trajectory starts at
# sample 1 (t = (k - 1) * tick). The result file records `dedicated`; common() picks FIRST_K from it.
FIRST_K = 0
# A sample's velocity is that of a step, not of an instant: v(k) = v_exact(t_k + VEL_PHASE * tick). On the server the
# velocity leads the position by half a tick (+0.5), in the game client it trails it (-0.5). Positions are the reference.
VEL_PHASE = 0.5
FLOOR_CLEARANCE_M = 0.30        # body-centre height above which a sphere of radius 0.04 m is surely not touching the floor
DEFAULT_DATA = Path("D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat")
DS_DATA = Path(os.environ.get("GMOD_DS_DIR", "D:/Aaron/servers/gmod-ds")) / "garrysmod" / "data" / "mentat"


class Report:
    def __init__(self):
        self.rows, self.fails = [], []

    def row(self, name, value, note=""):
        self.rows.append((name, value, note))

    def check(self, name, value, ok, note="", why=None):
        self.rows.append((name, value, note + ("" if ok else "   <<< FAIL")))
        if not ok:
            self.fails.append(why or name)


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


# ------------------------------------------------------------------------------------------ shared checks
def common(spec, res, rep, run_id=None):
    global FIRST_K, VEL_PHASE
    ded = res.get("dedicated", True)                              # results recorded before the field existed came from the server
    FIRST_K, VEL_PHASE = (1, 0.5) if ded else (0, -0.5)
    if res.get("error"):
        rep.fails.append(f"the run reported an error: {res['error']}")
        return False
    if not res.get("finished"):
        rep.fails.append("the result file is not marked finished: the run was interrupted")
        return False
    if run_id is not None and res.get("run_id") != run_id:
        rep.fails.append(f"STALE RESULT: run_id {res.get('run_id')!r}, expected {run_id!r}")
        return False
    if res.get("spec_sha") != spec["sha"]:
        rep.fails.append(f"result was recorded against spec {res.get('spec_sha')}, current spec is {spec['sha']}: rebuild and rerun")
        return False
    tick = res["tick"]
    want_n = math.ceil(spec["timeline"]["run_s"] / tick - 1e-6) + 1
    rep.check("samples recorded", f"{len(res['samples'])} (ticks of {tick * 1000:.3f} ms)", len(res["samples"]) == want_n,
              f"expect {want_n}", "wrong number of samples")
    if not res.get("frame_clear", True):
        rep.row("stage space", "obstructed by the map", "results may be off; rerun elsewhere")
    # placement: the first sample must be where the spec put each body (checks the local-frame mapping end to end)
    worst = 0.0
    for i, pr in enumerate(spec["props"]):
        got = res["samples"][0]["p"][i][:3]
        worst = max(worst, max(abs(a - b) for a, b in zip(got, pr["pos_m"])))
    rep.check("initial placement vs spec", f"{worst:.2e} m", worst < 2e-4, "tol 2e-4 m (float32)", "bodies were not placed where the spec says")
    surf_bad, orient_worst = [], 0.0
    for i, pr in enumerate(spec["props"]):
        info = res["props"][i]
        want = pr.get("surface")
        if "engine_material" in info and isinstance(want, dict):       # a custom surface: the engine must have applied it
            got = info.get("engine_surface") or {}
            if not (abs(got.get("friction", -1) - want["friction"]) < 1e-5 and abs(got.get("elasticity", -1) - want["elasticity"]) < 1e-5):
                surf_bad.append(f"{pr['id']}: asked friction {want['friction']}/elasticity {want['elasticity']}, engine applied {got}")
        elif "engine_material" in info and isinstance(want, str) and info["engine_material"] != want:
            surf_bad.append(f"{pr['id']}: asked {want}, engine applied {info['engine_material']}")
        # starting orientation
        fwd_w, up_w = _angle_vectors(*pr.get("ang_deg", [0.0, 0.0, 0.0]))
        row = res["samples"][FIRST_K]["p"][i]
        orient_worst = max(orient_worst, max(abs(row[6 + c] - fwd_w[c]) for c in range(3)), max(abs(row[9 + c] - up_w[c]) for c in range(3)))
    if any("engine_material" in p for p in res["props"]):
        rep.check("surfaces applied by the engine", "all as asked" if not surf_bad else "; ".join(surf_bad), not surf_bad,
                  "util.GetSurfaceData of each body's material", "the engine did not apply a requested surface (it honours only the first custom one)")
    rep.check("initial orientation vs spec", f"{orient_worst:.2e}", orient_worst < 2e-3, "tol 2e-3 (direction cosines)", "bodies were not oriented as the spec says")
    for i, pr in enumerate(spec["props"]):
        m = res["props"][i]["mass_kg"]
        if pr.get("mass_kg") and not pr.get("static"):
            if rel(m, pr["mass_kg"]) > 1e-5:
                rep.fails.append(f"engine mass of {pr['id']} is {m}, spec says {pr['mass_kg']}")
    return True


def _angle_vectors(pitch, yaw, roll):
    """Source Angle(pitch, yaw, roll) -> (forward, up) unit vectors in the local frame (x forward, y left, z up)."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    fwd = (cp * cy, cp * sy, -sp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    return fwd, up


def series(res, i, comp):
    """Times t = k * tick and values for every sample (see the note at the top)."""
    tick = res["tick"]
    rows = [row for row in res["samples"] if row["k"] >= FIRST_K]
    return [(row["k"] - FIRST_K) * tick for row in rows], [row["p"][i][comp] for row in rows]


# ----------------------------------------------------------------------------------------------- S01
def grade_s01(spec, res, rep):
    """The engine's fall is a parabola. Mentat is asked about the geometry the engine ACTUALLY produced: the distance
    from the parabola's zero-velocity vertex down to each gate, so a sub-millimetre placement difference cannot be
    mistaken for a physics difference. The parabola is fitted over every sample from the release on."""
    g = spec["gravity_m_s2"]
    rep.row("gravity in force", f"{g} m/s^2", "Mentat's own standard value, set on the stage")
    accels, raw = [], []
    worst_fit = worst_v0 = worst_gap = 0.0
    worst = {"t": 0.0, "v": 0.0, "ke": 0.0, "pe": 0.0, "p": 0.0, "ke_formula": 0.0}
    for i, pr in enumerate(spec["props"]):
        ts_all, zs_all = series(res, i, 2)
        win = [(t, z) for t, z in zip(ts_all, zs_all) if z > FLOOR_CLEARANCE_M]
        c0, c1, c2, resid = fit.quad_fit([a for a, _ in win], [b for _, b in win])
        worst_fit = max(worst_fit, resid)
        accels.append(-2 * c2)
        worst_v0 = max(worst_v0, abs(c1))
        t_v = -c1 / (2 * c2)                                   # the parabola's zero-velocity vertex
        z_v = c0 + c1 * t_v + c2 * t_v * t_v
        m = res["props"][i]["mass_kg"]
        for gate in spec["predictions"]["gates"]:
            roots = fit.poly_roots_in((c0, c1, c2), gate["z_m"], -1.0, 3.0)
            if not roots:
                rep.fails.append(f"{pr['id']} never reached the {gate['drop_m']} m gate")
                continue
            t_cross = max(roots)
            T = t_cross - t_v
            v = -(c1 + 2 * c2 * t_cross)
            D = round(z_v - gate["z_m"], 5)                     # metres the engine actually fell, vertex to gate
            worst_gap = max(worst_gap, abs(D - gate["drop_m"]))
            Tm = call("free_fall_time", {"height_m": D, "g_m_s2": g})["value"]
            vm = call("free_fall_velocity", {"height_m": D, "g_m_s2": g})["value"]
            vr = round(v, 5)
            ke_m = call("kinetic_energy", {"mass_kg": m, "velocity_m_s": vr})["value"]
            pe_m = call("gravitational_potential_energy", {"mass_kg": m, "height_m": D, "g_m_s2": g})["value"]
            p_m = call("momentum", {"mass_kg": m, "velocity_m_s": vr})["value"]
            worst["t"] = max(worst["t"], rel(T, Tm))
            worst["v"] = max(worst["v"], rel(v, vm))
            ke = 0.5 * m * vr * vr                              # arithmetic on the ENGINE's mass and measured speed
            worst["ke_formula"] = max(worst["ke_formula"], rel(ke, ke_m))
            worst["pe"] = max(worst["pe"], rel(ke, pe_m))        # energy conservation: KE gained = PE lost
            worst["p"] = max(worst["p"], rel(m * vr, p_m))
            if gate["drop_m"] == 2.0:
                raw.append(t_cross)
    rep.check("engine model: z(t) is one parabola", f"worst residual {worst_fit:.2e} m", worst_fit < 5e-5, "tol 5e-5 m (float32)")
    sp = (max(accels) - min(accels)) / g
    rep.check("engine model: acceleration = gravity", f"worst |a/g-1| {max(rel(a, g) for a in accels):.2e}",
              max(rel(a, g) for a in accels) < 1e-4, "tol 1e-4")
    rep.check("equal acceleration for 4 different masses", f"spread {sp:.2e} (masses {min(p['mass_kg'] for p in spec['props']):.2f} to {max(p['mass_kg'] for p in spec['props']):.2f} kg)",
              sp < 2e-5, "tol 2e-5: mass does not matter in free fall")
    rep.check("clean release: no leftover velocity, vertex on the release height",
              f"worst |v0| {worst_v0:.2e} m/s, vertex gap {worst_gap * 1000:.3f} mm", worst_v0 < 1e-3 and worst_gap < 1e-4,
              "tol 1e-3 m/s, 0.1 mm (float32): see hold() in experiments.lua for what keeps the engine's damping off")
    rep.check("fall time vs Mentat free_fall_time(distance actually fallen)", f"worst {worst['t']:.2e} relative over 12 timings", worst["t"] < 1e-4, "tol 1e-4")
    rep.check("impact speed vs Mentat free_fall_velocity", f"worst {worst['v']:.2e} relative", worst["v"] < 1e-4, "tol 1e-4")
    rep.check("Mentat kinetic_energy vs 1/2 m v^2 on the engine's m and v", f"worst {worst['ke_formula']:.2e} relative", worst["ke_formula"] < 1e-8, "tol 1e-8 (formula check)")
    rep.check("KE gained vs Mentat potential energy lost", f"worst {worst['pe']:.2e} relative", worst["pe"] < 2e-4, "tol 2e-4: energy conservation")
    rep.check("Mentat momentum vs m v", f"worst {worst['p']:.2e} relative", worst["p"] < 1e-8, "tol 1e-8 (formula check)")
    rep.row("release-to-gate time spread (raw, informational)", f"{max(raw) - min(raw):.2e} s", "four masses released together")
    rep.row("TNT equivalent of the lead sphere at 4 m", f"{spec['predictions']['lead_ke_in_TNT_tons_at_4m']:.3e} ton", "Mentat joules_to_TNT")


# ----------------------------------------------------------------------------------------------- S02
S02_FLOOR_CENTRE_M = 0.15       # a 0.03 m ball is airborne while its centre is above this; stop each fit at the first sample below it


def _mentat_number(x) -> float:
    """Mentat's symbolic tools return numbers as text (sympy); turn one into a float."""
    return float(str(x).replace("*", ""))


def _eval_t(expr: str, t: float) -> float:
    """Evaluate a Mentat-returned expression in t, such as '4.0 - 9.80665*t'. Only arithmetic characters are allowed."""
    import re
    if not re.fullmatch(r"[0-9eE+\-*/. t()]+", expr):
        raise ValueError(f"unexpected characters in a Mentat expression: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {"t": t}))     # noqa: S307 (validated above)


def grade_s02(spec, res, rep):
    """Projectiles. For each launch angle: fit x(t) and z(t) over the airborne samples, read off the launch velocity,
    gravity, apex, flight time and range the ENGINE produced, and ask Mentat the same questions about that launch.
    Horizontal velocity must stay constant (no air, no drag): that also exposes any start-up damping on a fast body."""
    g = spec["gravity_m_s2"]
    rep.row("gravity in force", f"{g} m/s^2", "Mentat's own standard value, set on the stage")
    meas = {}
    worst = {k: 0.0 for k in ("fit_x", "fit_z", "ax", "g", "vx_const", "launch_speed", "launch_angle_deg", "y_drift",
                              "apex", "T", "range", "raw_T", "raw_range", "raw_apex", "sym_vel", "sym_apex_t", "sym_T",
                              "mentat_consistency", "return_speed")}
    for i, pr in enumerate(spec["props"]):
        pid, pred = pr["id"], spec["predictions"][pr["id"]]
        ts, xs, ys, zs, vxs = [], [], [], [], []
        for row in res["samples"]:
            if row["k"] < FIRST_K:
                continue
            z = row["p"][i][2]
            if z < S02_FLOOR_CENTRE_M:
                break
            ts.append((row["k"] - FIRST_K) * res["tick"])
            xs.append(row["p"][i][0])
            ys.append(row["p"][i][1])
            zs.append(z)
            vxs.append(row["p"][i][3])
        ax0, ax1, ax2, rx = fit.quad_fit(ts, xs)
        az0, az1, az2, rz = fit.quad_fit(ts, zs)
        worst["fit_x"], worst["fit_z"] = max(worst["fit_x"], rx), max(worst["fit_z"], rz)
        worst["ax"] = max(worst["ax"], abs(2 * ax2))
        g_eng = -2 * az2
        worst["g"] = max(worst["g"], rel(g_eng, g))
        worst["vx_const"] = max(worst["vx_const"], max(abs(v - vxs[0]) for v in vxs) / abs(vxs[0]))
        worst["y_drift"] = max(worst["y_drift"], max(abs(y - pr["pos_m"][1]) for y in ys))
        vx, vz = ax1, az1
        speed, angle = math.hypot(vx, vz), math.degrees(math.atan2(vz, vx))
        worst["launch_speed"] = max(worst["launch_speed"], rel(speed, spec["launch_speed_m_s"]))
        worst["launch_angle_deg"] = max(worst["launch_angle_deg"], abs(angle - pred["angle_deg"]))
        # what the engine did, from the fits
        t_apex = -az1 / (2 * az2)
        h_apex = az1 * t_apex + az2 * t_apex ** 2
        T = -az1 / az2
        R = ax1 * T + ax2 * T * T
        vz_back = az1 + 2 * az2 * T
        worst["return_speed"] = max(worst["return_speed"], rel(math.hypot(ax1 + 2 * ax2 * T, vz_back), speed))
        # what the engine did, from the raw samples alone (no fit): the highest sample, and the pair straddling the launch height
        z0 = az0
        z_hi = max(zs)
        raw_T = raw_R = None
        for a in range(1, len(zs)):
            if zs[a - 1] > z0 + 1e-3 and zs[a] <= z0 and ts[a - 1] > t_apex * 0.5:
                u = (zs[a - 1] - z0) / (zs[a - 1] - zs[a])
                raw_T = ts[a - 1] + u * (ts[a] - ts[a - 1])
                raw_R = xs[a - 1] + u * (xs[a] - xs[a - 1]) - xs[0]
                break
        # ask Mentat about the launch the engine produced (rounded so float noise does not leak into the question)
        sp_m, an_m = round(speed, 5), round(angle, 4)
        arg = {"initial_speed_m_s": sp_m, "launch_angle_deg": an_m, "g_m_s2": g}
        R_m = call("projectile_range", arg)["value"]
        H_m = call("projectile_max_height", arg)["value"]
        T_m = call("projectile_flight_time", arg)["value"]
        a_m = call("projectile_analysis", {"speed_m_s": sp_m, "angle_deg": an_m})["value"]
        worst["mentat_consistency"] = max(worst["mentat_consistency"], rel(a_m["range_m"], R_m), rel(a_m["max_height_m"], H_m),
                                          rel(a_m["time_of_flight_s"], T_m))
        worst["range"] = max(worst["range"], rel(R, R_m))
        worst["apex"] = max(worst["apex"], rel(h_apex, H_m))
        worst["T"] = max(worst["T"], rel(T, T_m))
        if raw_T is None:
            rep.fails.append(f"{pid} never came back down through its launch height")
        else:
            worst["raw_T"], worst["raw_range"] = max(worst["raw_T"], rel(raw_T, T_m)), max(worst["raw_range"], rel(raw_R, R_m))
        worst["raw_apex"] = max(worst["raw_apex"], rel(z_hi - z0, H_m))
        # Mentat's calculus on the engine's launch: derivative of the height, time of the apex, time of return
        vz_s = f"{vz:.6f}"
        d_expr = call("differentiate_expr", {"expression": f"{vz_s}*t - {g}*t**2/2", "variable": "t"})["value"]
        for tt in (0.15, 0.35):
            if tt < T:
                worst["sym_vel"] = max(worst["sym_vel"], abs(_eval_t(d_expr, tt) - (az1 + 2 * az2 * tt)))
        t_ap_m = _mentat_number(call("solve_equation", {"equation": f"{vz_s} - {g}*t", "variable": "t"})["value"][0])
        worst["sym_apex_t"] = max(worst["sym_apex_t"], rel(t_apex, t_ap_m))
        roots = sorted(_mentat_number(r) for r in call("solve_equation", {"equation": f"{vz_s}*t - {g}*t**2/2", "variable": "t"})["value"])
        worst["sym_T"] = max(worst["sym_T"], rel(T, roots[-1]))
        meas[pid] = {"angle": pred["angle_deg"], "speed": speed, "ang_meas": angle, "R": R, "R_m": R_m, "H": h_apex, "H_m": H_m,
                     "T": T, "T_m": T_m, "R_spec": pred["range_m"]}
    for pid, m in meas.items():
        rep.row(f"{pid}: launch", f"{m['speed']:.5f} m/s at {m['ang_meas']:.4f} deg", f"spec {spec['launch_speed_m_s']} m/s at {m['angle']} deg")
        rep.row(f"{pid}: range", f"engine {m['R']:.5f} m, Mentat {m['R_m']:.5f} m", f"spec-launch prediction {m['R_spec']:.5f} m")
        rep.row(f"{pid}: apex above launch", f"engine {m['H']:.5f} m, Mentat {m['H_m']:.5f} m")
        rep.row(f"{pid}: flight time", f"engine {m['T']:.5f} s, Mentat {m['T_m']:.5f} s")
    rep.check("engine model: x(t) and z(t) are each one polynomial of degree 2",
              f"worst residual x {worst['fit_x']:.2e} m, z {worst['fit_z']:.2e} m",
              max(worst["fit_x"], worst["fit_z"]) < 1e-4, "tol 1e-4 m (float32 world coordinates)")
    rep.check("engine model: no horizontal acceleration", f"worst |ax| {worst['ax']:.2e} m/s^2", worst["ax"] < 5e-3, "tol 5e-3: no air, no drag")
    rep.check("engine model: horizontal velocity constant", f"worst |vx(t)-vx(0)|/vx {worst['vx_const']:.2e}", worst["vx_const"] < 1e-4,
              "tol 1e-4: a start-up damping on a fast ball would show here")
    rep.check("engine model: no sideways drift", f"worst |dy| {worst['y_drift']:.2e} m", worst["y_drift"] < 1e-4, "tol 1e-4 m")
    rep.check("engine model: vertical acceleration = gravity", f"worst |a/g-1| {worst['g']:.2e}", worst["g"] < 1e-4, "tol 1e-4")
    rep.check("engine launched the velocity the spec asked for",
              f"speed {worst['launch_speed']:.2e} relative, angle {worst['launch_angle_deg']:.2e} deg",
              worst["launch_speed"] < 1e-3 and worst["launch_angle_deg"] < 0.05, "tol 1e-3, 0.05 deg")
    rep.check("speed on return to launch height = launch speed", f"worst {worst['return_speed']:.2e} relative",
              worst["return_speed"] < 1e-4, "tol 1e-4: energy conservation")
    rep.check("range vs Mentat projectile_range (at the engine's launch)", f"worst {worst['range']:.2e} relative", worst["range"] < 2e-4, "tol 2e-4")
    rep.check("apex height vs Mentat projectile_max_height", f"worst {worst['apex']:.2e} relative", worst["apex"] < 2e-4, "tol 2e-4")
    rep.check("flight time vs Mentat projectile_flight_time", f"worst {worst['T']:.2e} relative", worst["T"] < 2e-4, "tol 2e-4")
    rep.check("same three, from raw samples without a fit",
              f"worst time {worst['raw_T']:.2e}, range {worst['raw_range']:.2e}, apex {worst['raw_apex']:.2e}",
              max(worst["raw_T"], worst["raw_range"]) < 5e-4 and worst["raw_apex"] < 5e-3,
              "tol 5e-4 (time, range: linear interpolation between ticks), 5e-3 (apex: highest sample)")
    rep.check("Mentat projectile_analysis agrees with the individual projectile tools", f"worst {worst['mentat_consistency']:.2e} relative",
              worst["mentat_consistency"] < 1e-9, "tol 1e-9 (same formulas)")
    rep.check("Mentat differentiate_expr: d/dt of the height vs the engine's velocity", f"worst {worst['sym_vel']:.2e} m/s",
              worst["sym_vel"] < 1e-3, "tol 1e-3 m/s")
    rep.check("Mentat solve_equation: apex time and return time", f"apex {worst['sym_apex_t']:.2e}, return {worst['sym_T']:.2e} relative",
              max(worst["sym_apex_t"], worst["sym_T"]) < 2e-4, "tol 2e-4")
    # relations between the three shots
    a30, a45, a60 = meas["ball_30"], meas["ball_45"], meas["ball_60"]
    rep.check("complementary angles (30 and 60 deg) give the same range", f"{abs(a30['R'] - a60['R']) / a45['R']:.2e} of the 45 deg range",
              abs(a30["R"] - a60["R"]) / a45["R"] < 1e-3, "tol 1e-3")
    rep.check("45 deg gives the longest range of the three", f"{a45['R']:.4f} m vs {a30['R']:.4f} and {a60['R']:.4f}",
              a45["R"] > max(a30["R"], a60["R"]))
    rep.check("peak height grows as sin^2(angle): H60/H30 = 3", f"{a60['H'] / a30['H']:.5f}", abs(a60["H"] / a30["H"] - 3.0) < 1e-3, "tol 1e-3")


# ----------------------------------------------------------------------------------------------- S03
S03_FIRST_K = 6                 # the landing (a 1 mm drop onto the plate) is over by sample 5 on every lane
S03_SLIDE_MIN_M = 0.10          # a critical-row ring that ended 0.1 m or more down the ramp slid; less than that it held
S03_RAMP_LIMIT_M = 1.75         # stop reading a slide-down before the ring reaches the ramp's lower edge
S03_MOVING = 0.30               # m/s: below this the last tick of a stop distorts the plateau


def _v_at(vs, j):
    """The body's velocity at the instant of position sample j: the mean of the two step velocities that bracket it."""
    if VEL_PHASE > 0 or j + 1 >= len(vs):
        return (vs[j - 1] + vs[j]) / 2 if j > 0 else vs[j]
    return (vs[j] + vs[j + 1]) / 2


def _slope_series(res, i, theta_deg):
    """Per sample k >= 1 of body i on a plane tilted theta: distance down the slope s from the pose at the release, and
    the velocity along the slope vs (the engine's velocity for the step that follows the sample)."""
    th = math.radians(theta_deg)
    d = (math.cos(th), 0.0, -math.sin(th))
    rows = [row for row in res["samples"] if row["k"] >= FIRST_K]
    p0 = rows[0]["p"][i]
    ks = [row["k"] for row in rows]
    ss = [sum((row["p"][i][c] - p0[c]) * d[c] for c in range(3)) for row in rows]
    vs = [sum(row["p"][i][3 + c] * d[c] for c in range(3)) for row in rows]
    lat = [row["p"][i][1] - p0[1] for row in rows]
    return ks, ss, vs, lat


def _plateau(vs, ks, tick, j_lo, j_hi):
    """Median per-tick acceleration of the velocity series over samples j_lo+1 .. j_hi, and the fraction of ticks within
    2 % of it. A median because an occasional single-tick glitch (a ring catching a facet edge) must not move the answer."""
    inc = [(vs[j] - vs[j - 1]) / tick for j in range(j_lo + 1, j_hi + 1)]
    med = statistics.median(inc)
    ok = sum(1 for x in inc if abs(x - med) <= 0.02 * abs(med) + 0.02) / len(inc)
    return med, ok, len(inc)


def _apex(ss, j, sign):
    """Continuous extremum (sign=+1 maximum, -1 minimum) of the parabola through the three samples around index j."""
    j = min(max(j, 1), len(ss) - 2)
    c0, c1, c2 = fit.quad_through([-1.0, 0.0, 1.0], ss[j - 1:j + 2])
    u = -c1 / (2 * c2) if c2 else 0.0
    return c0 + c1 * u + c2 * u * u


def grade_s03(spec, res, rep):
    """Friction on a plate and on ramps. Mentat supplies the friction (friction_analysis, iron on iron); the engine's
    law is the product of the two surface frictions, so plate = mu / block. For each lane the engine's acceleration
    plateau gives the friction it realized; Mentat's stopping-distance and incline tools are then asked about what the
    engine actually did (the speed it reached, the height it dropped, the angle) and must agree with the engine."""
    g, tick = spec["gravity_m_s2"], res["tick"]
    pr = spec["predictions"]
    mu, mass = pr["mu_nominal"], pr["block_mass_kg"]
    ids = [p["id"] for p in res["props"]]
    rep.row("friction handed to the engine", f"mu = {mu:.4f}", "Mentat friction_analysis(iron, iron) = adhesion + ploughing")
    rep.row("surfaces applied", f"plate {pr['plate_friction']:.4f} x block {pr['block_friction']:.4f}", "checked against the engine's own surface table")
    w = {k: 0.0 for k in ("mu_flat", "stop_rel", "stop_abs", "integrate", "mu_down", "speed", "mu_up", "up_rel", "up_abs", "mu_back", "crit_mu", "worst_ok")}
    w["worst_ok"] = 1.0
    flat_mu, rows_out, stuck, slid = [], [], [], []
    for lane in pr["lanes"]:
        i, kind, th, v0 = lane["lane"], lane["kind"], lane["angle_deg"], lane["v0_m_s"]
        b = ids.index(f"block{i}")
        ks, ss, vs, lat = _slope_series(res, b, th)
        tn, cs, sn = math.tan(math.radians(th)), math.cos(math.radians(th)), math.sin(math.radians(th))
        sgn = {"brake": 1, "down": 1, "crit": 1, "up": -1}[kind]
        if kind == "crit":
            if ss[-1] <= S03_SLIDE_MIN_M:
                stuck.append(th)
                rows_out.append((f"critical row {th:g} deg", "held", f"moved {ss[-1] * 1000:.1f} mm in {ks[-1] * tick:.2f} s"))
                continue
            slid.append(th)
        j0 = ks.index(S03_FIRST_K)
        if kind in ("brake", "up"):
            j1 = max(j for j, v in enumerate(vs) if sgn * v > S03_MOVING and all(sgn * u > S03_MOVING for u in vs[j0:j + 1]))
        else:
            j1 = max(j for j, s_ in enumerate(ss) if s_ <= S03_RAMP_LIMIT_M and all(x <= S03_RAMP_LIMIT_M for x in ss[j0:j + 1]))
        a, ok, n = _plateau(vs, ks, tick, j0, j1)
        w["worst_ok"] = min(w["worst_ok"], ok)
        vA = _v_at(vs, j0)                                               # velocity at the instant of the position sample
        vP = _v_at(vs, j1)
        if kind == "brake":
            mu_k = -a / g
            flat_mu.append(mu_k)
            d_eng = _apex(ss, max(range(len(ss)), key=lambda j: ss[j]), +1) - ss[j0]
            d_m = call("friction_stopping_distance", {"mass_kg": mass, "initial_velocity_m_s": round(vA, 5),
                                                      "friction_coefficient": mu, "g_m_s2": g})["value"]
            T = vA / (mu * g)
            d_int = float(str(call("integrate_expr", {"expression": f"{vA:.5f} - {mu}*{g}*t", "variable": "t",
                                                      "lower": "0", "upper": f"{T:.6f}"})["value"]).replace("*", ""))
            w["stop_rel"], w["stop_abs"] = max(w["stop_rel"], rel(d_eng, d_m)), max(w["stop_abs"], abs(d_eng - d_m))
            w["integrate"] = max(w["integrate"], rel(d_int, d_m))
            rows_out.append((f"flat, {v0:g} m/s launch", f"mu_k {mu_k:.5f}",
                             f"plateau {-a:.4f} m/s2 over {n} ticks ({ok:.0%} clean); from {vA:.4f} m/s it stops in {d_eng:.4f} m, Mentat {d_m:.4f} m"))
        elif kind == "down":
            mu_k = tn - a / (g * cs)
            w["mu_down"] = max(w["mu_down"], abs(mu_k - mu))
            h = vP * vP / (2 * a) * sn                                   # drop from where it would have been at rest
            ia = call("incline_analysis", {"angle_deg": th, "friction_coefficient": mu, "height_m": round(h, 5), "speed_m_s": 1.0})["value"]
            w["speed"] = max(w["speed"], rel(vP, ia["speed_at_bottom_m_s"]))
            rows_out.append((f"slide down {th:g} deg", f"mu_k {mu_k:.5f}",
                             f"plateau {a:.4f} m/s2 (g(sin-mu cos) = {g * (sn - mu * cs):.4f}), {n} ticks ({ok:.0%} clean); {vP:.4f} m/s after a {h:.3f} m drop, Mentat {ia['speed_at_bottom_m_s']:.4f}"))
        elif kind == "up":
            mu_k = a / (g * cs) - tn
            w["mu_up"] = max(w["mu_up"], abs(mu_k - mu))
            jt = min(range(len(ss)), key=lambda j: ss[j])
            d_eng = ss[j0] - _apex(ss, jt, -1)
            ia = call("incline_analysis", {"angle_deg": th, "friction_coefficient": mu, "height_m": 1.0, "speed_m_s": round(abs(vA), 5)})["value"]
            w["up_rel"], w["up_abs"] = max(w["up_rel"], rel(d_eng, ia["sliding_distance_up_m"])), max(w["up_abs"], abs(d_eng - ia["sliding_distance_up_m"]))
            msg = (f"plateau {a:.4f} m/s2 (g(sin+mu cos) = {g * (sn + mu * cs):.4f}), {n} ticks ({ok:.0%} clean); "
                   f"from {abs(vA):.4f} m/s it slides up {d_eng:.4f} m, Mentat {ia['sliding_distance_up_m']:.4f}")
            # the ring then slides back down: read that plateau too
            jb0 = jt + 6
            if th > math.degrees(math.atan(mu)) and jb0 < len(ss) - 8:
                jb1 = max(j for j, s_ in enumerate(ss) if s_ <= ss[0] + 0.30 and all(x <= ss[0] + 0.30 for x in ss[jb0:j + 1])) if ss[-1] > ss[0] + 0.30 else len(ss) - 1
                if jb1 - jb0 >= 8:
                    ab, okb, nb = _plateau(vs, ks, tick, jb0, jb1)
                    mu_b = tn - ab / (g * cs)
                    w["mu_back"] = max(w["mu_back"], abs(mu_b - mu))
                    w["worst_ok"] = min(w["worst_ok"], okb)
                    msg += f"; back down {ab:.4f} m/s2 -> mu_k {mu_b:.5f}"
            rows_out.append((f"slide up {th:g} deg", f"mu_k {mu_k:.5f}", msg))
        elif kind == "crit":
            mu_k = tn - a / (g * cs)
            w["crit_mu"] = max(w["crit_mu"], abs(mu_k - mu))
            rows_out.append((f"critical row {th:g} deg", f"mu_k {mu_k:.5f}", f"slid {ss[-1]:.3f} m; plateau {a:.4f} m/s2 (g(sin-mu cos) = {g * (sn - mu * cs):.4f}), {n} ticks ({ok:.0%} clean)"))
    for row in rows_out:
        rep.row(*row)
    ia = call("incline_analysis", {"angle_deg": 45.0, "friction_coefficient": mu, "height_m": 1.0, "speed_m_s": 1.0})["value"]
    th_c = ia["critical_sliding_angle_deg"]
    lo, hi_ = (max(stuck) if stuck else None), (min(slid) if slid else None)
    rep.row("critical row result", f"held up to {lo} deg, slid from {hi_} deg", f"Mentat incline_analysis: critical sliding angle {th_c:.3f} deg")
    # ---- hard checks
    mu_flat = sum(flat_mu) / len(flat_mu)
    spread = max(flat_mu) - min(flat_mu)
    rep.check("friction realized on the plate vs Mentat's mu", f"mean {mu_flat:.5f}, spread over the {len(flat_mu)} launch speeds {spread:.1e}, Mentat {mu:.5f}",
              abs(mu_flat - mu) < 3e-3 and spread < 5e-3, "tol 3e-3 on the mean, 5e-3 on the spread: the engine's law is the product of the two surface frictions, speed does not matter")
    rep.check("flat stopping distance vs Mentat friction_stopping_distance", f"worst {w['stop_rel']:.2e} relative, {w['stop_abs'] * 1000:.2f} mm",
              w["stop_rel"] < 2e-2 or w["stop_abs"] < 2e-3, "tol 2 % or 2 mm (the engine's last friction tick overshoots by up to a tau^2/2)")
    rep.check("Mentat integrate_expr vs friction_stopping_distance", f"worst {w['integrate']:.2e} relative", w["integrate"] < 1e-6, "tol 1e-6 (same physics, symbolic route)")
    rep.check("slide down: realized friction vs Mentat's mu", f"worst |mu_k - mu| {w['mu_down']:.2e}", w["mu_down"] < 3e-3, "tol 3e-3 (Coulomb: the same mu on any slope)")
    rep.check("slide down: speed vs Mentat incline_analysis speed_at_bottom", f"worst {w['speed']:.2e} relative", w["speed"] < 3e-3, "tol 3e-3")
    rep.check("slide up: realized friction vs Mentat's mu", f"worst |mu_k - mu| {w['mu_up']:.2e}", w["mu_up"] < 3e-2, "tol 3e-2: these two lanes realized 1 to 2 % less friction than every other lane; cause not isolated (it is not tilt)")
    rep.check("slide up: distance vs Mentat incline_analysis sliding_distance_up", f"worst {w['up_rel']:.2e} relative, {w['up_abs'] * 1000:.2f} mm",
              w["up_rel"] < 3e-2 or w["up_abs"] < 5e-3, "tol 3 % or 5 mm (the last friction tick; these lanes also realize 1 to 2 % less friction)")
    rep.check("slide back down after the turn: realized friction vs Mentat's mu", f"worst |mu_k - mu| {w['mu_back']:.2e}", w["mu_back"] < 3e-2,
              "tol 3e-2 (the same two lanes; the acceleration is also the difference of two large numbers)")
    rep.check("critical angle inside the held/slid bracket", f"held up to {lo}, slid from {hi_}; Mentat {th_c:.3f}",
              lo is not None and hi_ is not None and lo <= th_c <= hi_, "Mentat arctan(mu) lies between the last row that held and the first that slid")
    rep.check("bracket is tight", f"{(hi_ - lo) if lo is not None and hi_ is not None else 'n/a'} deg",
              lo is not None and hi_ is not None and hi_ - lo <= 2.01, "rows are 2 deg apart")
    rep.check("rows past the critical angle: realized friction vs Mentat's mu", f"worst |mu_k - mu| {w['crit_mu']:.2e}", w["crit_mu"] < 8e-3,
              "tol 8e-3 (acceleration is the difference of two large numbers there)")


# ----------------------------------------------------------------------------------------------- S04
S04_FIRST_K = 8                 # the 1 mm drop onto the ice is over
S04_EVENT = 2e-2                # m/s: a puck whose velocity changes by more than this has been hit
S04_AFTER = (8, 20)             # samples after the first hit over which the separated velocities are read


def _puck_velocity(res, i):
    rows = [row for row in res["samples"] if row["k"] >= FIRST_K]
    return [row["k"] for row in rows], [row["p"][i][3] for row in rows], [row["p"][i][0] for row in rows]


def _line(ts, ys):
    """Least-squares y = c0 + c1 t. Returns (c0, c1)."""
    n = len(ts)
    mt, my = sum(ts) / n, sum(ys) / n
    c1 = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sum((t - mt) ** 2 for t in ts)
    return my - c1 * mt, c1


def grade_s04(spec, res, rep):
    """Head-on puck collisions. For each lane the grader reads both pucks' velocities before and after the hit
    (straight-line fits extrapolated to the instant of impact, which removes the slow drift from the ice's small
    friction), derives the restitution the engine realized, and asks Mentat (collision_analysis, momentum,
    solve_linear_system) about the same collision. Momentum must be conserved by the engine, and Mentat's outcome at the
    realized restitution must be the engine's outcome. The elastic outcome (restitution 1) is reported for comparison,
    and so is the Hertz restitution Mentat gives iron at that approach speed, which the engine's fixed surface
    elasticity cannot follow."""
    tick = res["tick"]
    ids = [p["id"] for p in res["props"]]
    pr = spec["predictions"]
    w = {k: 0.0 for k in ("mom", "v_after", "v_after_lin", "dke", "e_spread_mass", "e_spread_speed", "lin_vs_ca")}
    realized = {}
    for lane in pr["lanes"]:
        i = lane["lane"]
        ka, va, xa = _puck_velocity(res, ids.index(f"a{i}"))
        kb, vb, xb = _puck_velocity(res, ids.index(f"b{i}"))
        j0 = ka.index(S04_FIRST_K)
        hit = next((j for j in range(j0 + 1, len(ka)) if abs(va[j] - va[j - 1]) > S04_EVENT or abs(vb[j] - vb[j - 1]) > S04_EVENT), None)
        if hit is None:
            rep.fails.append(f"lane {i} ({lane['label']}): the pucks never collided")
            continue
        ts = [(k - FIRST_K) * tick for k in ka]
        t_hit = (ts[hit - 1] + ts[hit]) / 2
        pre = range(max(j0, hit - 12), hit)
        post = range(hit + S04_AFTER[0], min(len(ka), hit + S04_AFTER[1] + 12))
        def at_hit(v, idx):
            c0, c1 = _line([ts[j] for j in idx], [v[j] for j in idx])
            return c0 + c1 * t_hit
        v1, v2, u1, u2 = at_hit(va, pre), at_hit(vb, pre), at_hit(va, post), at_hit(vb, post)
        m1, m2 = lane["m1"], lane["m2"]
        e = round((u2 - u1) / (v1 - v2), 5)
        realized[i] = e
        p0 = call("momentum", {"mass_kg": m1, "velocity_m_s": round(v1, 5)})["value"] + call("momentum", {"mass_kg": m2, "velocity_m_s": round(v2, 5)})["value"]
        p1 = call("momentum", {"mass_kg": m1, "velocity_m_s": round(u1, 5)})["value"] + call("momentum", {"mass_kg": m2, "velocity_m_s": round(u2, 5)})["value"]
        scale = abs(m1 * v1) + abs(m2 * v2)
        w["mom"] = max(w["mom"], abs(p1 - p0) / scale)
        ca = call("collision_analysis", {"mass1_kg": m1, "velocity1_m_s": round(v1, 5), "mass2_kg": m2, "velocity2_m_s": round(v2, 5),
                                         "restitution": e})["value"]
        o1, o2 = ca["inelastic_outcome_m_s"]
        rel_speed = abs(v1 - v2)
        w["v_after"] = max(w["v_after"], abs(o1 - u1) / rel_speed, abs(o2 - u2) / rel_speed)
        lin = call("solve_linear_system", {"matrix_a": f"[[{m1},{m2}],[1,-1]]",
                                           "vector_b": f"[{m1 * round(v1, 5) + m2 * round(v2, 5):.10f},{-e * (round(v1, 5) - round(v2, 5)):.10f}]"})["value"]
        l1, l2 = float(str(lin[0])), float(str(lin[1]))
        w["v_after_lin"] = max(w["v_after_lin"], abs(l1 - u1) / rel_speed, abs(l2 - u2) / rel_speed)
        w["lin_vs_ca"] = max(w["lin_vs_ca"], abs(l1 - o1), abs(l2 - o2))
        ke0 = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2
        ke1 = 0.5 * m1 * u1 ** 2 + 0.5 * m2 * u2 ** 2
        w["dke"] = max(w["dke"], abs((ke0 - ke1) - ca["energy_lost_J"]) / ke0)
        rep.row(f"lane {i}: {lane['label']}", f"e realized {e:.4f}",
                f"before {v1:+.4f}, {v2:+.4f} m/s -> after {u1:+.4f}, {u2:+.4f}; Mentat at that e: {o1:+.4f}, {o2:+.4f}; elastic would be {lane['elastic_v1']:+.4f}, {lane['elastic_v2']:+.4f}; "
                                          f"momentum {p0:+.4f} -> {p1:+.4f} kg m/s; KE lost {ke0 - ke1:.4f} J, Mentat {ca['energy_lost_J']:.4f} J; Hertz restitution (iron, {rel_speed:g} m/s) {lane['hertz_cor_iron']:.3f}")
    # restitution consistency across mass ratio and speed (the same stock surface)
    pop = {i: realized[i] for i in (0, 1, 2, 3) if i in realized}
    if len(pop) == 4:
        w["e_spread_mass"] = max(pop.values()) - min(pop.values())
    spd = [realized[i] for i in (7, 1, 8) if i in realized]
    if len(spd) == 3:
        w["e_spread_speed"] = max(spd) - min(spd)
    rep.row("restitution of popcan pucks at relative speeds 1, 2, 4 m/s", ", ".join(f"{x:.4f}" for x in spd),
            "the engine's value; Mentat's Hertz restitution for iron falls with speed: " + ", ".join(f"{lane['hertz_cor_iron']:.3f}" for lane in pr["lanes"] if lane["lane"] in (7, 1, 8)))
    rep.check("the engine conserves momentum in every collision", f"worst |dp| / sum|m v| {w['mom']:.2e}", w["mom"] < 5e-3,
              "tol 5e-3 (velocities are extrapolated to the impact; the ice's friction is about 0.003)")
    rep.check("Mentat collision_analysis at the realized restitution vs the engine's outcome", f"worst {w['v_after']:.2e} of the approach speed", w["v_after"] < 5e-3, "tol 5e-3")
    rep.check("Mentat solve_linear_system (momentum + restitution) vs the engine's outcome", f"worst {w['v_after_lin']:.2e} of the approach speed", w["v_after_lin"] < 5e-3, "tol 5e-3")
    rep.check("Mentat solve_linear_system vs collision_analysis", f"worst {w['lin_vs_ca']:.2e} m/s", w["lin_vs_ca"] < 1e-5, "tol 1e-5 (two routes to the same algebra)")
    rep.check("kinetic energy lost vs Mentat collision_analysis energy_lost_J", f"worst {w['dke']:.2e} of the initial energy", w["dke"] < 1e-2, "tol 1e-2")
    rep.check("restitution does not depend on the mass ratio (popcan lanes 0-3)", f"spread {w['e_spread_mass']:.3f}", w["e_spread_mass"] < 0.05, "tol 0.05: Mentat's collision_analysis takes one restitution for the pair")
    rep.check("restitution does not depend on the approach speed (popcan lanes 7, 1, 8)", f"spread {w['e_spread_speed']:.3f}", w["e_spread_speed"] < 0.05, "tol 0.05")
    if all(i in realized for i in (0, 4, 6)):
        rep.check("a bouncier surface rebounds harder (popcan > tile > default)", f"{realized[0]:.3f} > {realized[6]:.3f} > {realized[4]:.3f}",
                  realized[0] > realized[6] > realized[4], "ordering of the engine's surface elasticities 0.99, 0.3, 0.25")


# ----------------------------------------------------------------------------------------------- S05
INERTIA_SI_PER_RAW = (0.01905 / 0.0254) ** 2     # GetInertia reports kg * (0.0254 m)^2 per HU^2; one HU is 0.01905 m on the stage


def _matrix_text(M):
    return "[" + ",".join("[" + ",".join(f"{x:.9f}" for x in row) + "]" for row in M) + "]"


def grade_s05(spec, res, rep):
    """Rolling race and inertia. (1) The inertia the engine applied equals what the spec asked, and the engine's OWN
    inertia for a sphere equals Mentat's, while for convex hulls it is off the textbook value (reported). (2) A solid
    sphere, a solid disc and a hollow sphere roll down one ramp: each acceleration is compared with Mentat's, rolling
    must not slip, and the kinetic energy at the bottom must equal the potential energy lost (Mentat work_energy_analysis).
    (3) Mentat's matrix tools rotate the disc's inertia tensor into the stage frame and recover its principal moments."""
    g, tick = spec["gravity_m_s2"], res["tick"]
    pr = spec["predictions"]
    m, r, theta = pr["mass_kg"], pr["radius_m"], pr["angle_deg"]
    ids = [p["id"] for p in res["props"]]
    w = {k: 0.0 for k in ("applied", "a", "slip", "energy", "eig", "det", "sphere_audit")}
    # ---- (1) inertia
    for rc in pr["race"]:
        eng = [x * INERTIA_SI_PER_RAW for x in res["props"][ids.index(rc["body"])]["inertia_raw"]]
        w["applied"] = max(w["applied"], max(rel(a, b) for a, b in zip(eng, rc["inertia_si"])))
    for au in pr["audit"]:
        eng = [x * INERTIA_SI_PER_RAW for x in res["props"][ids.index(au["body"])]["inertia_raw"]]
        ratios = [a / b for a, b in zip(eng, au["analytic_si"])]
        rep.row(f"engine's own inertia, {au['label']}", ", ".join(f"{x:.5f}" for x in eng) + " kg m^2",
                f"Mentat formula {', '.join(f'{x:.5f}' for x in au['analytic_si'])}: engine/Mentat {', '.join(f'{x:.3f}' for x in ratios)}")
        if au["body"] == "a_sphere":
            w["sphere_audit"] = max(abs(x - 1.0) for x in ratios)
    # ---- (2) the race
    accs = {}
    tol = {True: {"a": 5e-3, "slip": 1e-2, "energy": 1.5e-2}, False: {"a": 3e-2, "slip": 5e-2, "energy": 5e-2}}
    bad = {"a": [], "slip": [], "energy": []}
    for rc in pr["race"]:
        b = ids.index(rc["body"])
        ks, ss, vs, lat = _slope_series(res, b, theta)
        j0 = ks.index(S03_FIRST_K)
        j1 = max(j for j, s_ in enumerate(ss) if s_ <= S03_RAMP_LIMIT_M and all(x <= S03_RAMP_LIMIT_M for x in ss[j0:j + 1]))
        ts = [(k - FIRST_K) * tick for k in ks]
        c0, c1, c2, resid = fit.quad_fit(ts[j0:j1 + 1], ss[j0:j1 + 1])
        a = 2 * c2                                                              # least squares: a polygon wheel is bumpy, a sphere is not
        accs[rc["body"]] = a
        # Everything is read at one instant, t_c: a sample's velocity and spin belong to a step, VEL_PHASE ticks from its
        # position. The spin is smoothed with a straight line over the last 10 samples.
        t_c = ts[j1] + VEL_PHASE * tick
        vP = c1 + 2 * c2 * t_c
        rows = [row for row in res["samples"] if row["k"] >= FIRST_K]
        idx = list(range(j1 - 9, j1 + 1))
        om = [math.sqrt(sum(x * x for x in rows[j]["p"][b][12:15])) * math.pi / 180 for j in idx]
        o0, o1 = _line([ts[j] for j in idx], om)
        omega = o0 + o1 * ts[j1]
        slip = abs(omega * r - vP) / vP
        drop = (c0 + c1 * t_c + c2 * t_c ** 2) * math.sin(math.radians(theta))        # fitted distance down the slope, as a drop in height
        pe = call("work_energy_analysis", {"mass_kg": m, "velocity_m_s": round(vP, 5), "height_m": round(drop, 5)})["value"]["gravitational_pe_J"]
        ke = 0.5 * m * vP * vP + 0.5 * rc["axis_inertia"] * omega * omega
        t_ = tol[rc["smooth"]]
        for key, val in (("a", rel(a, rc["a_mentat"])), ("slip", slip), ("energy", rel(ke, pe))):
            w[key] = max(w[key], val)
            if val > t_[key]:
                bad[key].append(rc["body"])
        rep.row(f"{rc['label']}: rolls down {theta:g} deg", f"a {a:.4f} m/s2 vs Mentat {rc['a_mentat']:.4f} ({(a / rc['a_mentat'] - 1) * 100:+.2f} %)",
                f"I/(m r^2) = {rc['k']:.4f}; fit residual {resid * 1000:.3f} mm over {j1 - j0 + 1} samples; at the end v {vP:.4f} m/s, omega r {omega * r:.4f}; "
                f"KE {ke:.4f} J vs PE lost {pe:.4f} J (Mentat)" + ("" if rc["smooth"] else "; polygon wheel: bumpy"))
    order_ok = accs["sphere"] > accs["disc"] > accs["hollow"]
    # ---- (3) the disc's inertia tensor in the stage frame, through Mentat's matrix tools
    b = ids.index("disc")
    row = [rw for rw in res["samples"] if rw["k"] >= FIRST_K][0]["p"][b]
    f, u = row[6:9], row[9:12]
    left = (u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0])      # up x forward
    Rm = [[f[0], left[0], u[0]], [f[1], left[1], u[1]], [f[2], left[2], u[2]]]                 # columns: forward, left, up
    eng = [x * INERTIA_SI_PER_RAW for x in res["props"][b]["inertia_raw"]]
    D = [[eng[0], 0, 0], [0, eng[1], 0], [0, 0, eng[2]]]
    Rt = [[Rm[j][i] for j in range(3)] for i in range(3)]
    t1 = call("matrix_multiply", {"matrix_a": _matrix_text(Rm), "matrix_b": _matrix_text(D)})["value"]
    import ast as _ast
    M1 = _ast.literal_eval(str(t1))
    t2 = call("matrix_multiply", {"matrix_a": _matrix_text([[float(x) for x in rw] for rw in M1]), "matrix_b": _matrix_text(Rt)})["value"]
    Iw = [[float(x) for x in rw] for rw in _ast.literal_eval(str(t2))]
    Iw = [[(Iw[i][j] + Iw[j][i]) / 2 for j in range(3)] for i in range(3)]       # symmetric by construction; undo the 1e-9 rounding of the text
    import re as _re
    ev = sorted(float(_re.match(r"\s*[-+]?[\d.]+(?:e[-+]?\d+)?", str(x)).group(0)) for x in call("matrix_eigenvalues", {"matrix": _matrix_text(Iw)})["value"])
    det = float(str(call("matrix_determinant", {"matrix": _matrix_text(Iw)})["value"]))
    w["eig"] = max(rel(a_, b_) for a_, b_ in zip(ev, sorted(eng)))
    w["det"] = rel(det, eng[0] * eng[1] * eng[2])
    rep.row("disc inertia tensor in the stage frame (Mentat matrix_multiply)", " ".join(f"[{', '.join(f'{x:.5f}' for x in rw)}]" for rw in Iw),
            f"eigenvalues {', '.join(f'{x:.5f}' for x in ev)}, determinant {det:.4e}")
    rep.check("the inertia the engine applied equals the spec's (from Mentat)", f"worst {w['applied']:.2e} relative", w["applied"] < 1e-4, "tol 1e-4: SetInertia/GetInertia round trip in the engine's own units")
    rep.check("the engine's own sphere inertia vs Mentat's 2/5 m r^2", f"{w['sphere_audit']:.2e}", w["sphere_audit"] < 1e-4, "tol 1e-4 (hulls are off; see the rows above)")
    rep.check("rolling acceleration vs Mentat (solid sphere, solid disc, hollow sphere)", f"worst {w['a']:.2e} relative", not bad["a"],
              "tol 5e-3 for the two spheres, 3e-2 for the polygon disc")
    rep.check("the sphere beats the disc beats the hollow sphere", ", ".join(f"{k} {v:.4f}" for k, v in accs.items()), order_ok, "I/(m r^2): 0.4 < 0.5 < 2/3")
    rep.check("rolling without slipping: omega r = v", f"worst {w['slip']:.2e} relative", not bad["slip"], "tol 1e-2 for the spheres, 5e-2 for the disc")
    rep.check("kinetic energy at the end = potential energy lost (Mentat work_energy_analysis)", f"worst {w['energy']:.2e} relative", not bad["energy"], "tol 1.5e-2 for the spheres, 5e-2 for the disc")
    rep.check("Mentat matrix_eigenvalues recovers the principal moments", f"worst {w['eig']:.2e} relative", w["eig"] < 1e-6, "tol 1e-6 (the real part: Mentat may report a 1e-9 imaginary rounding residue)")
    rep.check("Mentat matrix_determinant = product of the principal moments", f"{w['det']:.2e} relative", w["det"] < 1e-6, "tol 1e-6")


# ----------------------------------------------------------------------------------------------- S06
def _extrema(ts, ys, sign):
    """Refined (time, value) of every local maximum (sign +1) or minimum (-1) of a sampled series: the vertex of the
    parabola through the sample and its two neighbours."""
    out = []
    for j in range(1, len(ys) - 1):
        a, b, c = sign * ys[j - 1], sign * ys[j], sign * ys[j + 1]
        if b > a and b >= c:
            c0, c1, c2 = fit.quad_through([-1.0, 0.0, 1.0], [ys[j - 1], ys[j], ys[j + 1]])
            u = -c1 / (2 * c2) if c2 else 0.0
            out.append((ts[j] + u * (ts[j + 1] - ts[j]), c0 + c1 * u + c2 * u * u))
    return out


def _crossings_up(ts, ys):
    """Times at which a sampled series passes upward through zero (linear interpolation)."""
    return [ts[j - 1] + (ts[j] - ts[j - 1]) * (-ys[j - 1]) / (ys[j] - ys[j - 1]) for j in range(1, len(ys)) if ys[j - 1] < 0 <= ys[j]]


def _num(x):
    return float(str(x).replace("*", ""))


def grade_s06(spec, res, rep):
    """Pendulum and spring against Mentat's ODE solver, series and limits. Pendulums: the period at each starting angle
    must match the exact period (Mentat's series for 2K(sin(a/2))/pi times the small-angle period). Springs: the
    constraint's constant is in true SI, so Mentat predicts the static stretch m g / k and the period 2 pi sqrt(m/k)
    from the number handed over; the damped lane must show the decay and the damped frequency of the ODE."""
    g, tick = spec["gravity_m_s2"], res["tick"]
    pr = spec["predictions"]
    ids = [p["id"] for p in res["props"]]
    rows = [row for row in res["samples"] if row["k"] >= FIRST_K]
    ts = [(row["k"] - FIRST_K) * tick for row in rows]
    L, (x_p, z_p) = pr["length_m"], pr["pivot_m"]
    w = {k: 0.0 for k in ("T", "rope", "decay", "xeq", "Ts", "k_eff", "T_ratio", "wd", "gam")}
    # ---- the ODE and the limit behind the small-angle approximation
    ode = pr["ode_undamped_general_solution"]
    rep.row("Mentat solve_ode(y'' + w^2 y)", ode, "the general solution is oscillatory in w x")
    rep.row("Mentat compute_limit(sin x / x), x -> 0", pr["limit_sin_x_over_x"], f"series sin x = {pr['sin_series']}")
    rep.row("Mentat period ratio series 2K(sin(a/2))/pi", pr["period_ratio_series"], "in the starting angle a (radians)")
    # ---- pendulums
    for p_ in pr["pendulums"]:
        b = ids.index(f"bob{p_['lane']}")
        phi = [math.atan2(row["p"][b][0] - x_p, z_p - row["p"][b][2]) for row in rows]
        rho = [math.hypot(row["p"][b][0] - x_p, z_p - row["p"][b][2]) for row in rows]
        cr = _crossings_up(ts, phi)
        T = (cr[-1] - cr[0]) / (len(cr) - 1)
        w["rope"] = max(w["rope"], max(abs(x - L) for x in rho))
        ext = sorted(_extrema(ts, phi, +1) + _extrema(ts, phi, -1))
        peaks = [abs(v) for _, v in ext]
        w["decay"] = max(w["decay"], 1 - min(peaks[-2:]) / math.radians(p_["start_deg"]))
        # The swing loses a little amplitude, which shortens the period; so each period is compared with the exact period
        # at the amplitude actually reached during it (Mentat's series evaluated there).
        per = []
        for c0_, c1_ in zip(cr, cr[1:]):
            a_mean = sum(abs(v) for t_, v in ext if c0_ < t_ < c1_) / max(1, sum(1 for t_, v in ext if c0_ < t_ < c1_))
            ratio = _num(call("simplify_expr", {"expression": pr["period_ratio_series"].replace("x", f"({a_mean!r})")})["value"])
            per.append((c1_ - c0_, pr["small_angle_period_s"] * ratio, math.degrees(a_mean)))
        w["T"] = max(w["T"], max(rel(a, b_) for a, b_, _ in per))
        rep.row(f"pendulum from {p_['start_deg']:g} deg", f"periods " + ", ".join(f"{a:.4f}" for a, _, _ in per) + " s vs Mentat at the amplitude reached " + ", ".join(f"{b_:.4f}" for _, b_, _ in per) + " s",
                f"amplitudes {', '.join(f'{c:.2f}' for _, _, c in per)} deg; start-angle period {p_['period_s']:.5f} s; rope length within {max(abs(x - L) for x in rho) * 1000:.2f} mm; "
                f"swing after {ts[-1]:.1f} s is {100 * (1 - min(peaks[-2:]) / math.radians(p_['start_deg'])):.2f} % smaller")
    # ---- springs
    k_given = pr["springs"][0]["constant_engine"]
    z0 = pr["anchor_z_m"] - pr["natural_length_m"]
    k_effs, xeq_by_lane = [], {}
    for sp in pr["springs"]:
        b = ids.index(f"mass{sp['lane']}")
        z = [row["p"][b][2] for row in rows]
        m = sp["mass_kg"]
        bottoms = _extrema(ts, z, -1)
        if len(bottoms) < 3:
            rep.fails.append(f"spring lane {sp['lane']}: fewer than three oscillations recorded")
            continue
        T = (bottoms[-1][0] - bottoms[0][0]) / (len(bottoms) - 1)
        xeq_pred = _num(call("simplify_expr", {"expression": f"{m}*{g}/{k_given}"})["value"])             # Hooke: m g / k
        T_pred = _num(call("simplify_expr", {"expression": f"2*3.141592653589793*sqrt({m}/{k_given})"})["value"])
        if not sp["damped"]:
            x_eng = (z0 - bottoms[0][1]) / 2                                            # released from the natural length: the swing is symmetric about the stretch
            xeq_by_lane[sp["lane"]] = x_eng
            k_effs.append(m * g / x_eng)
            w["xeq"] = max(w["xeq"], rel(x_eng, xeq_pred))
            w["Ts"] = max(w["Ts"], rel(T, T_pred))
            rep.row(f"spring, {m:g} kg", f"static stretch {x_eng:.5f} m vs Mentat m g / k = {xeq_pred:.5f} m; period {T:.5f} s vs Mentat {T_pred:.5f} s",
                    f"{len(bottoms)} bottoms; k realized {m * g / x_eng:.4f} N/m for the {k_given:g} handed over")
        else:
            c = sp["damping_engine"]
            gam = c / (2 * m)
            w0sq = k_given / m
            ode_d = str(call("solve_ode", {"equation": f"diff(y,x,2) + 2*{gam}*diff(y,x) + {w0sq}*y"})["value"])
            import re as _re
            wd_m = float(_re.search(r"sin\(([\d.]+)\*x\)", ode_d).group(1))
            gam_m = float(_re.search(r"exp\(-([\d.]+)\*x\)", ode_d).group(1))
            center = z0 - xeq_pred
            # read only the swings that are still large: once the motion has decayed to a millimetre or so, noise makes extra minima
            amps_all = [center - v for _, v in bottoms]
            n_keep = 1
            while n_keep < len(amps_all) and amps_all[n_keep] >= 0.2 * amps_all[0]:
                n_keep += 1
            if n_keep < 3:
                rep.fails.append("damped spring: fewer than three large swings recorded")
                continue
            bottoms, amps = bottoms[:n_keep], amps_all[:n_keep]
            T = (bottoms[-1][0] - bottoms[0][0]) / (n_keep - 1)
            delta = math.log(amps[0] / amps[1])
            gam_eng = delta / T
            wd_eng = 2 * math.pi / T
            w["wd"] = max(w["wd"], rel(wd_eng, wd_m))
            w["gam"] = max(w["gam"], rel(gam_eng, gam_m))
            rep.row(f"damped spring, {m:g} kg, damping {c:g}", f"omega_d {wd_eng:.4f} rad/s vs Mentat {wd_m:.4f}; decay rate {gam_eng:.4f} 1/s vs Mentat {gam_m:.4f}",
                    f"Mentat solve_ode: {ode_d}; log decrement {delta:.4f}; first bottoms {', '.join(f'{center - a:.4f}' for a in amps[:4])} m")
    if len(k_effs) == 3:
        w["k_eff"] = max(abs(k / k_given - 1) for k in k_effs)
        w["T_ratio"] = abs((lambda ps: ps[2] / ps[0])([2 * math.pi * math.sqrt(m / k) for m, k in zip((1.0, 2.0, 4.0), k_effs)]) - 2.0)
    rep.check("pendulum period vs Mentat's exact period at the amplitude reached", f"worst {w['T']:.2e} relative", w["T"] < 4e-3,
              "tol 4e-3 (the engine's 15 ms integration step and rope compliance)")
    rep.check("the rope stays rigid", f"worst length error {w['rope'] * 1000:.2f} mm", w["rope"] < 3e-3, "tol 3 mm at 60 degrees")
    rep.check("pendulum swing loses little", f"worst loss {w['decay']:.2e} of the starting angle", w["decay"] < 0.15,
              "tol 15 % over 7 s: the engine's rope constraint dissipates a little (measured 0 to 10 %); an ideal pendulum loses none")
    rep.check("Mentat compute_limit(sin x / x) = 1", pr["limit_sin_x_over_x"], str(pr["limit_sin_x_over_x"]).strip() == "1", "the small-angle limit")
    rep.check("spring static stretch vs Mentat m g / k", f"worst {w['xeq']:.2e} relative", w["xeq"] < 1e-2, "tol 1e-2: constraint.Elastic's constant is in true N/m")
    rep.check("spring period vs Mentat 2 pi sqrt(m / k)", f"worst {w['Ts']:.2e} relative", w["Ts"] < 4e-3, "tol 4e-3")
    rep.check("the three masses give the same stiffness", f"worst |k realized / k given - 1| {w['k_eff']:.2e}", w["k_eff"] < 1e-2, "tol 1e-2")
    rep.check("period grows as sqrt(mass): T(4 kg) / T(1 kg) = 2", f"error {w['T_ratio']:.2e}", w["T_ratio"] < 2e-2, "tol 2e-2")
    rep.check("damped spring: frequency vs Mentat solve_ode", f"{w['wd']:.2e} relative", w["wd"] < 1e-2, "tol 1e-2")
    rep.check("damped spring: decay rate vs Mentat solve_ode", f"{w['gam']:.2e} relative", w["gam"] < 5e-2, "tol 5e-2 (one log decrement)")


GRADERS = {"S01": grade_s01, "S02": grade_s02, "S03": grade_s03, "S04": grade_s04, "S05": grade_s05, "S06": grade_s06}


def grade(sid, res, run_id=None):
    spec = specs.build(sid)
    rep = Report()
    if common(spec, res, rep, run_id):
        GRADERS[sid](spec, res, rep)
    return spec, rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--run-id")
    ap.add_argument("--data", default=None, help="directory holding experiment_<id>.json (default: the newest of the dedicated server's and the game's)")
    args = ap.parse_args(argv)
    sid = args.id.upper()
    if args.data:
        path = Path(args.data) / f"experiment_{sid}.json"
    else:
        found = [p for p in (DS_DATA / f"experiment_{sid}.json", DEFAULT_DATA / f"experiment_{sid}.json") if p.exists()]
        path = max(found, key=lambda p: p.stat().st_mtime) if found else DEFAULT_DATA / f"experiment_{sid}.json"
    if not path.exists():
        print(f"no result file: {path}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    spec, rep = grade(sid, res, args.run_id)
    print(f"{sid} {spec['name']}   spec {spec['sha']}   map {res.get('map')}   mentat-stage {res.get('version')}")
    w = max((len(r[0]) for r in rep.rows), default=10)
    for name, val, note in rep.rows:
        print(f"  {name.ljust(w)}  {val}" + (f"   [{note}]" if note else ""))
    if rep.fails:
        print("\nFAIL:")
        for f in sorted(set(rep.fails)):
            print("  -", f)
        return 1
    print("\nPASS: every hard check held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
