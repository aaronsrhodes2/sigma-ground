"""U13 -- How much of an SPH sigma-source would be artificial viscosity?

Virial share of the SPH artificial viscosity vs the isothermal pressure in the
collapsing P2b-type cloud:   ratio = sum m r.a_visc / sum m r.a_press.

Run A  = the design as written (100 steps at cfl_timestep, steps 0/50/100,
         alpha in {0.25, 0.5, 1, 2}; h-scan at alpha = 1).
Run B  = the corrected protocol (declared before running, see PREREG below):
         dt = t_ff/600 (the X2/U2 step) to 1.0 t_ff, every 30 steps, same alphas,
         plus a dt = t_ff/1200 control at alpha = 1, h-scans at steps 50 and 300.

Nothing in the repo is modified.  Every physical leaf comes from Mentat
(tools/mentat_client.mentat) or a repo engine import, with its source kept.
Run from the repo root:  python -X utf8 <this file>
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

REPO = r"D:/Aaron/development/sigma-ground"
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U13"
sys.path.insert(0, REPO + "/tools")
sys.path.insert(0, REPO)
from mentat_client import mentat  # noqa: E402

from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum  # noqa: E402
from sigma_ground.dynamics.continuum import (  # noqa: E402
    continuum_step, cfl_timestep, _sph_density_sum, _pressure_isothermal,
    _sph_pressure_force, _sph_viscous_force)
from sigma_ground.dynamics.fluid.kernel import grad_W  # noqa: E402
from sigma_ground.field.interface.fluid import eyring_viscosity  # noqa: E402

# ---------------------------------------------------------------- Mentat leaves
SRC = {}


def leaf(key, tool, args):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


N, M, R = 64, 1.0e12, 5000.0                      # P2b geometry (Choice, from the design)
SEED = 7
G = leaf("G", "lookup_constant", {"name": "G"})
C_LIGHT = leaf("c", "lookup_constant", {"name": "c"})
V_REF = leaf("v_ref_sqrt_GM_over_R", "circular_orbit_velocity", {"central_mass_kg": M, "radius_m": R})
RHO_FE = leaf("rho_iron", "density", {"material": "iron"})
T_MELT_FE = leaf("T_melt_iron", "melting_point", {"material": "iron"})
_tff = leaf("t_ff", "simplify_expr",
            {"expression": f"N(sqrt(3*pi/(32*{G!r}*(3*{M!r}/(4*pi*{R!r}**3)))))"})
T_FF = float(_tff)
# physical-viscosity comparison row: the repo's own liquid-iron viscosity (engine = Mentat),
# flagged by KNOWN_GAPS A5 as 10^3-10^6 x too high -> an UPPER bound on the physical term
ETA_FE = eyring_viscosity("iron", T=T_MELT_FE)
SRC["eta_iron_eyring_at_Tmelt"] = {"tool": "sigma_ground.field.interface.fluid.eyring_viscosity",
                                   "args": {"material_key": "iron", "T": T_MELT_FE}, "value": ETA_FE,
                                   "source": "repo engine import; PHYSICS_GAP A5 (10^3-10^6 x too high)"}
# viscous_flow_analysis was named by the design as the source of iron's eta: it TAKES eta as input.
try:
    vfa, vfa_src = mentat("viscous_flow_analysis", {"velocity_m_s": 0.01, "radius_m": 1.0})
    SRC["viscous_flow_analysis_probe"] = {"inputs_echo": "requires viscosity_pa_s as an INPUT (default 1e-3, water)",
                                          "source": vfa_src}
except Exception as e:  # pragma: no cover
    SRC["viscous_flow_analysis_probe"] = {"error": str(e)}

CS = 0.40 * V_REF                                   # isothermal c_s (Choice, from the design)
D_SPACING = (4.0 / 3.0 * math.pi * R ** 3 / N) ** (1.0 / 3.0)
H0 = 1.5 * D_SPACING                                # h = 1.5 (V/N)^(1/3)
EPS = 0.02 * R
ALPHAS = [0.25, 0.5, 1.0, 2.0]
HMULTS = [1.0, 1.5, 2.0]

CHOICES = {
    "N": N, "M_kg": M, "R_m": R, "seed": SEED, "omega_rad_s": 0.0, "jitter_frac": 0.0,
    "material_key": "iron", "particle_density_kg_m3": RHO_FE,
    "c_s_m_s": CS, "c_s_over_v_ref": 0.40, "h_m": H0, "h_over_spacing": 1.5,
    "two_h_over_R": 2 * H0 / R, "eps_m": EPS, "eps_over_R": 0.02,
    "c_override": "c_s (mandatory; passed explicitly to _sph_viscous_force)",
    "three_P_over_rho_c2": 3 * CS ** 2 / C_LIGHT ** 2,
}

PREREG = {
    "design_pass_fail_verbatim": ("Slope in alpha = 1.0 +- 0.15; in h = 1.0 +- 0.3. Scope rule: ratio at alpha = 1, "
                                  "step 50 > 0.1 => no sigma-source or Tolman-source from an SPH run until a "
                                  "physical-viscosity path exists; < 0.01 => allowed with the viscous term booked "
                                  "as a disclosed Choice. Predicted ~0.1 alpha."),
    "gray_band_rule_added_before_run": ("0.01 <= ratio <= 0.1: the design gives no branch; reported GRAY, "
                                        "neither branch fires (recommendation only: the stricter one)."),
    "agent_prediction_run_A": ("cfl_timestep uses the parcels' MATERIAL sound speed (iron 5942 m/s), not c_s = 0.046 "
                               "m/s, so dt_cfl = 0.153 s and 100 steps cover ~15 s = 3e-4 t_ff: the cloud has barely "
                               "moved. Continuum estimate ratio ~ alpha*k*h/(2 c_s) with homologous k = (GM/R^3) t "
                               "=> ratio(alpha=1, step 50) ~ 1e-4 (ALLOW branch, but an artefact of the time step); "
                               "step 0 ratio = 0 exactly (cold start, v = 0); alpha-slope = 1.000 by construction "
                               "(Pi is linear in alpha at an alpha-independent state); h-slope ~ 1."),
    "agent_prediction_run_B": ("dt = t_ff/600: ratio(alpha=1) ~ 0.84 (t/t_ff) early -> ~0.07 at step 50 (GRAY), "
                               "crosses 0.1 near 0.12 t_ff, reaches O(1) late in the collapse; alpha-slope falls "
                               "below 0.85 late (viscosity slows the infall); h-slope ~ 1 +- 0.3 early."),
    "corrected_criterion": ("Scope rule applied (i) at step 50 of Run B and (ii) to max ratio over 0-1 t_ff of Run B; "
                            "alpha-slope and h-slope bands unchanged, evaluated on Run B states."),
}


def jdump(obj, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------- measurement
def make_scene():
    bodies = seed_rotating_cloud(N, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                 particle_density_kg_m3=RHO_FE, jitter_frac=0.0, seed=SEED)
    return to_continuum(bodies, material_key="iron", G=G, smoothing_h=H0, gravity_softening_m=EPS,
                        eos="isothermal", sound_speed_m_s=CS)


def evaluate(ps, h, alpha):
    """Virial sums at the CURRENT state. Density, neighbours and the isothermal pressure are
    recomputed here (the design sketch read p.pressure, which is 0.0 before the first step)."""
    _sph_density_sum(ps, h)
    for p in ps:
        _pressure_isothermal(p, CS)
    ap = _sph_pressure_force(ps, h)
    av = _sph_viscous_force(ps, h, alpha_visc=alpha, c_override=CS)
    Vp = sum(p.mass * (p.position.x * ap[i].x + p.position.y * ap[i].y + p.position.z * ap[i].z)
             for i, p in enumerate(ps))
    Vv = sum(p.mass * (p.position.x * av[i].x + p.position.y * av[i].y + p.position.z * av[i].z)
             for i, p in enumerate(ps))
    m = np.array([p.mass for p in ps])
    x = np.array([[p.position.x, p.position.y, p.position.z] for p in ps])
    v = np.array([[p.velocity.x, p.velocity.y, p.velocity.z] for p in ps])
    rho = np.array([p.sph_density for p in ps])
    xc = (m[:, None] * x).sum(0) / m.sum()
    vc = (m[:, None] * v).sum(0) / m.sum()
    dx, dv = x - xc, v - vc
    r2 = (dx ** 2).sum(1)
    div_fit = 3.0 * float((m * (dx * dv).sum(1)).sum() / (m * r2).sum())   # homologous fit, v = (div/3) r
    # SPH divergence estimator, mass-weighted mean
    div_i = []
    n_pairs = n_appr = 0
    for i, p in enumerate(ps):
        s = 0.0
        for j in p.neighbors:
            q = ps[j]
            gx, gy, gz = grad_W(p.position.x - q.position.x, p.position.y - q.position.y,
                                p.position.z - q.position.z, h)
            s += q.mass * ((q.velocity.x - p.velocity.x) * gx + (q.velocity.y - p.velocity.y) * gy
                           + (q.velocity.z - p.velocity.z) * gz)
            vr = ((p.velocity.x - q.velocity.x) * (p.position.x - q.position.x)
                  + (p.velocity.y - q.velocity.y) * (p.position.y - q.position.y)
                  + (p.velocity.z - q.velocity.z) * (p.position.z - q.position.z))
            n_pairs += 1
            n_appr += vr < 0
        div_i.append(s / p.sph_density)
    div_sph = float((m * np.array(div_i)).sum() / m.sum())
    vol = float((m / rho).sum())
    P_mean_V = CS ** 2 * float(m.sum()) / vol           # volume-weighted <P> = c_s^2 M / V_sph
    rho_bar = float(m.sum()) / vol
    ratio = Vv / Vp if Vp != 0 else None
    out = dict(ratio=ratio, V_press_J=Vp, V_visc_J=Vv, div_fit_s=div_fit, div_sph_s=div_sph,
               rms_radius_over_R=float(math.sqrt((m * r2).sum() / m.sum())) / R,
               v_rms_over_cs=float(math.sqrt((m * (dv ** 2).sum(1)).sum() / m.sum())) / CS,
               mean_neighbours=n_pairs / len(ps), frac_pairs_approaching=(n_appr / n_pairs) if n_pairs else None,
               P_mean_V_Pa=P_mean_V, rho_bar_kg_m3=rho_bar)
    if ratio is not None and div_fit < 0:
        zeta_eff = ratio * P_mean_V / abs(div_fit)                      # Pa s, ratio = zeta|div|/<P>
        out.update(zeta_eff_Pa_s=zeta_eff,
                   C_kin=zeta_eff / (rho_bar * alpha * CS * h),         # vs [LIT] 1/6 (Price 2012)
                   ratio_phys_eyring=ETA_FE * abs(div_fit) / P_mean_V,  # zeta_phys ~ eta proxy, upper bound
                   zeta_eff_over_eta_eyring=zeta_eff / ETA_FE)
    return out


def hscan(ps_live, alpha=1.0):
    rows = []
    for hm in HMULTS:
        ps = copy.deepcopy(ps_live)
        rows.append(dict(h_mult=hm, h_m=hm * D_SPACING, **evaluate(ps, hm * D_SPACING, alpha)))
    return rows


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = (ys > 0) & np.isfinite(ys)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


def run(alpha, dt_mode, n_steps, every, hscan_steps=()):
    sc = make_scene()
    rows, hs = [], {}
    t0 = time.time()
    for st in range(n_steps + 1):
        if st % every == 0 or st in hscan_steps:
            ps = sc.continuum_parcels()
            if st % every == 0:
                rows.append(dict(step=st, t_s=sc.time, t_over_tff=sc.time / T_FF, **evaluate(ps, H0, alpha)))
            if st in hscan_steps:
                hs[str(st)] = hscan(ps, alpha)
        if st == n_steps:
            break
        dt = cfl_timestep(sc) if dt_mode == "cfl" else dt_mode
        continuum_step(sc, dt, alpha_visc=alpha)
    led = sc.energy_ledger()
    return dict(alpha=alpha, dt_mode=("cfl_timestep" if dt_mode == "cfl" else dt_mode), rows=rows, hscan=hs,
                energy_ledger_end=led, wall_s=time.time() - t0)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {"id": "U13", "mentat_sources": SRC, "choices": CHOICES, "prereg": PREREG, "t_ff_s": T_FF}
    jdump(res, "u13_results.json")          # pre-registration on disk before any run
    sc0 = make_scene()
    res["dt_cfl_initial_s"] = cfl_timestep(sc0)
    res["material_sound_speed_iron_m_s"] = sc0.continuum_parcels()[0].sound_speed

    # ---- Run A: the design as written
    runA = [run(a, "cfl", 100, 50, hscan_steps=(50, 100) if a == 1.0 else ()) for a in ALPHAS]
    res["runA"] = runA
    print("Run A done", flush=True)
    # ---- Run B: corrected protocol
    dtB = T_FF / 600.0
    runB = [run(a, dtB, 600, 30, hscan_steps=(50, 300) if a == 1.0 else ()) for a in ALPHAS]
    runB_ctrl = run(1.0, T_FF / 1200.0, 1200, 60, hscan_steps=())
    res["runB"] = runB
    res["runB_dt_control_alpha1"] = runB_ctrl
    print("Run B done", flush=True)

    # ---- analysis
    def ratio_at(runs, a, step):
        r = [x for x in runs if x["alpha"] == a][0]
        return [row for row in r["rows"] if row["step"] == step][0]["ratio"]

    A = {}
    for st in (0, 50, 100):
        rs = [ratio_at(runA, a, st) for a in ALPHAS]
        A[f"step{st}"] = dict(ratios=dict(zip(map(str, ALPHAS), rs)), alpha_slope=slope(ALPHAS, rs))
    for st in ("50", "100"):
        hsr = runA[2]["hscan"][st]
        A[f"h_slope_step{st}"] = slope([x["h_mult"] for x in hsr], [x["ratio"] for x in hsr])
    rA = ratio_at(runA, 1.0, 50)
    A["t_step50_over_tff"] = [row for row in runA[2]["rows"] if row["step"] == 50][0]["t_over_tff"]
    A["scope_rule"] = "FORBID" if rA > 0.1 else ("ALLOW" if rA < 0.01 else "GRAY")
    A["alpha_slope_pass"] = abs(A["step50"]["alpha_slope"] - 1.0) <= 0.15
    A["h_slope_pass"] = abs(A["h_slope_step50"] - 1.0) <= 0.3
    A["prediction_0p1alpha_ratio_measured_over_predicted"] = rA / 0.1

    B = {"steps": [row["step"] for row in runB[0]["rows"]],
         "t_over_tff": [row["t_over_tff"] for row in runB[0]["rows"]]}
    B["ratio_by_alpha"] = {str(r["alpha"]): [row["ratio"] for row in r["rows"]] for r in runB}
    B["alpha_slope_vs_time"] = [slope(ALPHAS, [r["rows"][k]["ratio"] for r in runB])
                                for k in range(len(runB[0]["rows"]))]
    hs50, hs300 = runB[2]["hscan"]["50"], runB[2]["hscan"]["300"]
    B["h_slope_step50"] = slope([x["h_mult"] for x in hs50], [x["ratio"] for x in hs50])
    B["h_slope_step300"] = slope([x["h_mult"] for x in hs300], [x["ratio"] for x in hs300])
    r1 = runB[2]["rows"]
    # Run B step 50 is not on the every-30 grid -> read it from the h-scan at h_mult = 1.5
    rB50 = [x for x in hs50 if x["h_mult"] == 1.5][0]["ratio"]
    B["ratio_alpha1_step50"] = rB50
    B["scope_rule_step50"] = "FORBID" if rB50 > 0.1 else ("ALLOW" if rB50 < 0.01 else "GRAY")
    rmax = max(row["ratio"] for row in r1 if row["ratio"] is not None)
    B["max_ratio_alpha1_0_to_1tff"] = rmax
    B["scope_rule_whole_collapse"] = "FORBID" if rmax > 0.1 else ("ALLOW" if rmax < 0.01 else "GRAY")

    def first_cross(rows, thr):
        for row in rows:
            if row["ratio"] is not None and row["ratio"] > thr:
                return row["t_over_tff"]
        return None
    B["t_cross_0p01_over_tff"] = first_cross(r1, 0.01)
    B["t_cross_0p1_over_tff"] = first_cross(r1, 0.1)
    ctrl = {round(row["t_over_tff"], 6): row["ratio"] for row in runB_ctrl["rows"]}
    B["dt_control_max_rel_diff"] = max(abs(row["ratio"] / ctrl[round(row["t_over_tff"], 6)] - 1.0)
                                       for row in r1 if row["ratio"] and round(row["t_over_tff"], 6) in ctrl
                                       and ctrl[round(row["t_over_tff"], 6)])
    B["dt_control_rel_diff_by_time"] = [(row["t_over_tff"], row["ratio"] / ctrl[round(row["t_over_tff"], 6)] - 1.0)
                                        for row in r1 if row["ratio"] and round(row["t_over_tff"], 6) in ctrl
                                        and ctrl[round(row["t_over_tff"], 6)]]
    res["analysis"] = {"runA": A, "runB": B}
    jdump(res, "u13_results.json")
    print(json.dumps(res["analysis"], indent=1, default=str))
