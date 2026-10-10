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
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
import specs  # noqa: E402
from mentat_io import call  # noqa: E402

# The engine's first physics step after the release velocity is set moves nothing (measured), so samples k >= 1 lie on
# the exact trajectory and the release instant is sample 1: time t' = (k - 1) * tick. Sample 0 is the pre-release pose.
FLOOR_CLEARANCE_M = 0.30        # body-centre height above which a sphere of radius 0.04 m is surely not touching the floor
DEFAULT_DATA = Path("D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat")


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
    want_n = int(round(spec["timeline"]["run_s"] / tick)) + 1
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
    for i, pr in enumerate(spec["props"]):
        m = res["props"][i]["mass_kg"]
        if pr.get("mass_kg") and not pr.get("static"):
            if rel(m, pr["mass_kg"]) > 1e-5:
                rep.fails.append(f"engine mass of {pr['id']} is {m}, spec says {pr['mass_kg']}")
    return True


def series(res, i, comp):
    """Times t' = (k-1)*tick and values for samples k >= 1 (see the note at the top)."""
    tick = res["tick"]
    rows = [row for row in res["samples"] if row["k"] >= 1]
    return [(row["k"] - 1) * tick for row in rows], [row["p"][i][comp] for row in rows]


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
            if row["k"] < 1:
                continue
            z = row["p"][i][2]
            if z < S02_FLOOR_CENTRE_M:
                break
            ts.append((row["k"] - 1) * res["tick"])
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


GRADERS = {"S01": grade_s01, "S02": grade_s02}


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
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args(argv)
    sid = args.id.upper()
    path = Path(args.data) / f"experiment_{sid}.json"
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
