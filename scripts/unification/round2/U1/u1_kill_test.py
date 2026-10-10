"""U1 — sigma-bridge kill-test, completed: PSR J1738+0333 on nbody gw_damping + the five weak-field gate rows
-> GO/NO-GO record per row and the three routes (no route chosen).

Design: misc/unification/design_synthesis.json, experiments_ranked[id == "U1"] (read in full before writing this).
PRE-REGISTERED (copied from the design, not moved after the run):
  Engine:  Pbdot_engine within 2 % of Peters (by construction: _apply_gw_damping IS the Peters formula);
           > 5 % off would be an nbody bug.
  Physics: coded sigma passes only if ALL of
             |gamma-1| < 4.6e-5, |eta_N| < 2.2e-4, eta(Ti,Pt) < 5.4e-15, |beta_LPI| < 9e-7, (dalpha)^2 < 3.6e-6
           PREDICTED: NO-GO on all five, excesses 4.0 / 3.3 / 11.1 / 5.5 / >= 3 orders (X1 text: ">= 2 orders").
           X1 phrasing of the dipole gate: GO only if the dipole at the repo's dalpha is within +3.7/-3.6 fs/s of zero.

PLATINUM 21: every constant and every leaf number is a Mentat call (tools/mentat_client.py) or a repo engine
(sigma_ground.*), with its source logged in the JSON.  [LIT] inputs are tagged and were verified by web search.
Runs from the repo root (cwd) so `import sigma_ground` works.  Writes only into this script's folder.
"""
import json, math, os, sys, time
import numpy as np

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/tools")
sys.path.insert(0, REPO)
from mentat_client import mentat_raw  # noqa: E402

from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles  # noqa: E402
from sigma_ground.inventory.core.sigma import XI, nucleon_qcd_fraction, proton_mass_kg, three_measures_atom  # noqa: E402

LOG = []          # every Mentat / engine leaf: label, value, source, call
CALLS = [0]


def mentat_call(tool, args, label):
    r = mentat_raw(tool, args)
    CALLS[0] += 1
    if not r.get("ok"):
        raise RuntimeError(f"mentat {tool} {args}: {r.get('error')}")
    res = r["result"]
    LOG.append(dict(label=label, tool=tool, args=args, value=res.get("value"), source=res.get("source")))
    return res.get("value"), res.get("source")


def const(name, label=None):
    v, _ = mentat_call("lookup_constant", {"name": name}, label or name)
    return float(v)


def calc(expr, label):
    """Mentat does the arithmetic: sympy simplify of a fully numeric expression, evaluated to 17 digits."""
    v, _ = mentat_call("simplify_expr", {"expression": f"N(({expr}), 17)"}, label)
    return float(str(v))


def engine_leaf(label, value, source):
    LOG.append(dict(label=label, tool="repo engine", args=None, value=value, source=source))
    return value


t_start = time.time()
# ════════════════════════════════════════════════════════════════════════
# 1. Leaves
# ════════════════════════════════════════════════════════════════════════
G = const("G"); C = const("c"); MSUN = const("M_sun"); HBAR = const("hbar"); AU = const("au")
DAY, _ = mentat_call("convert_units", {"value": 1, "from_units": "day", "to_units": "s"}, "1 day [s]")
DAY = float(DAY)
sun, _ = mentat_call("solar_system_body", {"body_name": "sun"}, "sun record")
earth, _ = mentat_call("solar_system_body", {"body_name": "earth"}, "earth record")
R_SUN, M_SUN_BODY = sun["radius_m"], sun["mass_kg"]
R_E, M_E = earth["radius_m"], earth["mass_kg"]

# [LIT] PSR J1738+0333 — Freire et al. 2012, MNRAS 423, 3328 (arXiv:1205.1450), verified 2026-10-10 by web fetch:
#   Pb = 0.3547907398724(13) d, m_p = 1.46(+0.06/-0.05), m_c = 0.181(+0.008/-0.007) Msun, q = 8.1(2),
#   Pbdot_obs = -17.0(3.1) fs/s, Pbdot_int = -25.9 +/- 3.2 fs/s, Pbdot_GR = -27.7(+1.5/-1.9) fs/s,
#   Pbdot_xs = +2.0(+3.7/-3.6) fs/s, Eq. 21: Pbdot_D = -2 pi n_b (G M_c/c^3) q/(q+1) (1+e^2/2)/(1-e^2)^(5/2) (a_p-a_c)^2
LIT = dict(
    J1738=dict(Pb_d=0.3547907398724, m_p_msun=1.46, m_c_msun=0.181, q=8.1, Pbdot_obs_fs=-17.0, Pbdot_int_fs=-25.9,
               Pbdot_int_err_fs=3.2, Pbdot_GR_fs=-27.7, Pbdot_GR_err_fs=(+1.5, -1.9), Pbdot_xs_fs=2.0,
               Pbdot_xs_err_fs=(+3.7, -3.6), ref="Freire et al. 2012, MNRAS 423, 3328, arXiv:1205.1450"),
    cassini=dict(gamma_minus_1=2.1e-5, err=2.3e-5, b_min_rsun=1.6,
                 ref="Bertotti, Iess & Tortora 2003, Nature 425, 374 (doi:10.1038/nature01997); b_min = 1.6 R_sun"),
    llr=dict(eta_N=-0.2e-4, err=1.1e-4, ref="Hofmann & Mueller 2018, CQG 35, 035015 (doi:10.1088/1361-6382/aa8f7a)"),
    microscope=dict(eta=-1.5e-15, stat=2.3e-15, syst=1.5e-15, altitude_m=710e3,
                    ref="Touboul et al. 2022, PRL 129, 121102 (arXiv:2209.15487); 710 km sun-synchronous orbit"),
    clocks=dict(k_mu=7e-8, err=45e-8, ref="Lange et al. 2021, PRL 126, 011102 (arXiv:2010.06620): (c^2/mu) dmu/dPhi = 7(45)e-8"),
    earth_e=dict(e=0.0167, ref="NASA Planetary Fact Sheet (Earth), orbit eccentricity 0.0167 — not in the Mentat body record"),
    alloys=dict(TA6V=[(22, 48, 0.90), (13, 27, 0.06), (23, 51, 0.04)], PtRh10=[(78, 195, 0.90), (45, 103, 0.10)],
                ref="MICROSCOPE test-mass alloys TA6V and PtRh10 by mass, as in scripts/unification/sigma_exclusion.py"),
)
J = LIT["J1738"]

# ════════════════════════════════════════════════════════════════════════
# 2. Engine: J1738+0333 on NBodySystem with gw_damping off/on (separately built body lists)
# ════════════════════════════════════════════════════════════════════════
Pb = calc(f"{J['Pb_d']}*{DAY}", "Pb [s]")
m1 = calc(f"{J['m_p_msun']}*{MSUN}", "m_p [kg]")
m2 = calc(f"{J['m_c_msun']}*{MSUN}", "m_c [kg]")
Mt = calc(f"{m1}+{m2}", "M [kg]")
a_rel = calc(f"({G}*{Mt}*{Pb}**2/(4*pi**2))**(1/3)", "a (Kepler) [m]")
v_rel = calc(f"sqrt({G}*{Mt}/{a_rel})", "v_rel circular [m/s]")
E_kep = calc(f"-{G}*{m1}*{m2}/(2*{a_rel})", "E_orbit = -G m1 m2/(2a) [J]")
pbdot_peters = calc(f"-(192*pi/5)*(2*pi*{G}/({C}**3*{Pb}))**(5/3)*{m1}*{m2}*{Mt}**(-1/3)",
                    "Pbdot Peters (circular quadrupole) [s/s]")
dlnP_dE, _ = mentat_call("differentiate_expr", {"expression": "log(2*pi*sqrt((-G*m1*m2/(2*En))**3/(G*M)))",
                                                 "variable": "En"}, "d ln Pb / dE  (Kepler)")
# Mentat diagnostic found while building this: differentiate_expr with variable "E" returns 0 (E parsed as Euler's number)
dE2, _ = mentat_call("differentiate_expr", {"expression": "E**2", "variable": "E"}, "DIAGNOSTIC d(E**2)/dE")
rs_p, _ = mentat_call("schwarzschild_radius", {"mass_kg": m1}, "r_s pulsar [m]")
rs_c, _ = mentat_call("schwarzschild_radius", {"mass_kg": m2}, "r_s companion [m]")
pn_param = calc(f"{G}*{Mt}/({a_rel}*{C}**2)", "GM/(a c^2)  (PN parameter)")


import sigma_ground.field.interface.nbody as _nb  # noqa: E402


class NBodyPetersR5(NBodySystem):
    """COUNTERFACTUAL, scratch only (this folder; the engine is NOT patched): the engine's own _apply_gw_damping,
    copied line for line, with the single change r**4 -> r**5 (Peters 1964: L_GW = (32/5) G^4 m1^2 m2^2 M/(c^5 r^5)).
    Used only to show that the measured engine excess is exactly that exponent."""

    def _apply_gw_damping(self, vel_new, dt):
        b0, b1 = self.bodies[0], self.bodies[1]
        r_vec = b1.position_m - b0.position_m
        r = float(np.linalg.norm(r_vec))
        if r == 0.0:
            return vel_new
        mu_red = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg)
        M_tot = b0.mass_kg + b1.mass_kg
        dE_dt = -(32.0 / 5.0) * _nb._G ** 4 * b0.mass_kg ** 2 * b1.mass_kg ** 2 * M_tot / (_nb._c ** 5 * r ** 5)
        v_rel = vel_new[1] - vel_new[0]
        v_mag = float(np.linalg.norm(v_rel))
        if v_mag == 0.0:
            return vel_new
        ke_rel = 0.5 * mu_red * v_mag ** 2
        if ke_rel == 0.0:
            return vel_new
        damp = max(0.0, 1.0 + dE_dt * dt / ke_rel)
        v_cm = (b0.mass_kg * vel_new[0] + b1.mass_kg * vel_new[1]) / M_tot
        vel_new[0] = v_cm - math.sqrt(damp) * b1.mass_kg / M_tot * v_rel
        vel_new[1] = v_cm + math.sqrt(damp) * b0.mass_kg / M_tot * v_rel
        return vel_new


def bodies(a, v):
    # Barycentric circular orbit; radii/k2 as in the design sketch (irrelevant: tidal_force off, k2 = 0)
    return [CelestialBody(m1, np.array([-m2 / Mt * a, 0.0, 0.0]), np.array([0.0, -m2 / Mt * v, 0.0]), 1.2e4, 0.0),
            CelestialBody(m2, np.array([m1 / Mt * a, 0.0, 0.0]), np.array([0.0, m1 / Mt * v, 0.0]), 1.0e7, 0.0)]


def run(gw, steps_per_orbit, n_orb, a, v, P, cls):
    bl = bodies(a, v)                       # a fresh, separately constructed list for every run
    snapshot = [b.position_m.copy() for b in bl]
    s = cls(bl, toggles=PhysicsToggles(gw_damping=gw))
    dt = P / steps_per_orbit
    n = steps_per_orbit * n_orb
    t = np.empty(n + 1); E = np.empty(n + 1); ph = np.empty(n + 1)

    def phase():
        r = s.bodies[1].position_m - s.bodies[0].position_m
        return math.atan2(r[1], r[0])
    t[0] = 0.0; E[0] = s.total_energy(); ph[0] = phase()
    for k in range(1, n + 1):
        s.step(dt)
        t[k] = s.time; E[k] = s.total_energy(); ph[k] = phase()
    alias_free = all(np.array_equal(b.position_m, p) for b, p in zip(bl, snapshot))
    return dict(t=t, E=E, ph=np.unwrap(ph), alias_free=alias_free, dt=dt)


def measure(steps_per_orbit, n_orb=20, a=None, v=None, P=None, cls=NBodySystem):
    a = a_rel if a is None else a; v = v_rel if v is None else v; P = Pb if P is None else P
    off = run(False, steps_per_orbit, n_orb, a, v, P, cls)
    on = run(True, steps_per_orbit, n_orb, a, v, P, cls)
    T = n_orb * P
    E0 = off["E"][0]
    # (i) design estimator: endpoint energy difference
    dE_off = off["E"][-1] - off["E"][0]; dE_on = on["E"][-1] - on["E"][0]
    Edot_end = (dE_on - dE_off) / T
    # (ii) least-squares slope of E_on - E_off vs t (same physics, less round-off)
    dEt = on["E"] - off["E"]
    Edot_fit = np.polyfit(on["t"], dEt, 1)[0]
    # Kepler: dlnPb/dE = -3/(2E)  (Mentat differentiate_expr) -> Pbdot = Pb * (-3/(2E)) * Edot
    pb_end = P * (-1.5 / E0) * Edot_end
    pb_fit = P * (-1.5 / E0) * Edot_fit
    # (iii) timing (independent of the energy bookkeeping): orbital phase lag dphi = -pi Pbdot t^2 / Pb^2
    dph = on["ph"] - off["ph"]
    tt = on["t"]
    A = np.vstack([tt ** 2, tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, dph, rcond=None)
    pb_tim = -coef[0] * P ** 2 / math.pi
    resid = dph - A @ coef
    # integrator energy wobble (gw off) relative to |E| — what the subtraction removes
    wobble = float(np.max(np.abs(off["E"] - E0)) / abs(E0))
    return dict(cls=cls.__name__, a_m=a, Pb_s=P, steps_per_orbit=steps_per_orbit, n_orbits=n_orb, dt_s=off["dt"], E0_J=E0,
                dE_off_J=dE_off, dE_on_J=dE_on, Edot_endpoint_W=Edot_end, Edot_fit_W=Edot_fit,
                Pbdot_energy_endpoint=pb_end, Pbdot_energy_fit=pb_fit, Pbdot_timing=pb_tim,
                timing_fit_rms_rad=float(np.sqrt(np.mean(resid ** 2))), timing_lag_end_rad=float(dph[-1]),
                energy_wobble_rel_gw_off=wobble, alias_free=bool(off["alias_free"] and on["alias_free"]),
                series=dict(t=tt[::50].tolist(), dE=dEt[::50].tolist(), dphi=dph[::50].tolist()))


eng = {}
for spo in (2000, 1000, 4000):   # 2000 = pre-registered; 1000 / 4000 = dt-doubling / dt-halving controls
    eng[str(spo)] = measure(spo)
main = eng["2000"]
rel = {k: calc(f"({main[k]})/({pbdot_peters})-1", f"rel dev {k} vs Peters") for k in
       ("Pbdot_energy_endpoint", "Pbdot_energy_fit", "Pbdot_timing")}
rel_lit = calc(f"({main['Pbdot_energy_endpoint']})/({J['Pbdot_GR_fs']}e-15)-1", "rel dev engine vs Freire GR -27.7 fs/s")
peters_vs_lit = calc(f"({pbdot_peters})/({J['Pbdot_GR_fs']}e-15)-1", "rel dev Mentat Peters vs Freire GR -27.7")
engine_pass = abs(rel["Pbdot_energy_endpoint"]) < 0.02
engine_bug = abs(rel["Pbdot_energy_endpoint"]) > 0.05

# defect diagnosis (run because the pre-registered "> 5 % = nbody bug" branch fired):
# (1) ratio engine/Peters divided by the separation in metres -> 1 if the power carries r^-4 instead of r^-5
ratio_over_a = calc(f"({main['Pbdot_energy_endpoint']})/({pbdot_peters})/{a_rel}", "engine/Peters divided by a [1/m]")
# (2) separation scaling: same masses at a/10 -> the ratio must drop 10x if r^-4 vs r^-5
a10 = calc(f"{a_rel}/10", "a/10 [m]")
v10 = calc(f"sqrt({G}*{Mt}/{a10})", "v_rel at a/10")
P10 = calc(f"2*pi*sqrt({a10}**3/({G}*{Mt}))", "Pb at a/10 [s]")
peters10 = calc(f"-(192*pi/5)*(2*pi*{G}/({C}**3*{P10}))**(5/3)*{m1}*{m2}*{Mt}**(-1/3)", "Pbdot Peters at a/10")
eng["a10_2000"] = measure(2000, 20, a10, v10, P10)
ratio10_over_a = calc(f"({eng['a10_2000']['Pbdot_energy_endpoint']})/({peters10})/{a10}", "engine/Peters at a/10 divided by a/10")
# (3) counterfactual r^5 copy (scratch subclass): pre-registered 20 orbits, plus 100 orbits for precision
eng["R5_2000"] = measure(2000, 20, cls=NBodyPetersR5)
eng["R5_2000_100orb"] = measure(2000, 100, cls=NBodyPetersR5)
rel_r5 = {k: {m: calc(f"({eng[k][m]})/({pbdot_peters})-1", f"counterfactual r^5 {k} {m} vs Peters")
              for m in ("Pbdot_energy_endpoint", "Pbdot_energy_fit", "Pbdot_timing")} for k in ("R5_2000", "R5_2000_100orb")}
# (4) the same exponent in the tests: Earth-Sun GW power with r^-4 (as coded / as test_physics_stability states) vs r^-5
P_es_r4 = calc(f"(32/5)*{G}**4*{M_SUN_BODY}**2*{M_E}**2*({M_SUN_BODY}+{M_E})/({C}**5*{AU}**4)", "Earth-Sun L_GW with r^-4 [W]")
P_es_r5 = calc(f"(32/5)*{G}**4*{M_SUN_BODY}**2*{M_E}**2*({M_SUN_BODY}+{M_E})/({C}**5*{AU}**5)", "Earth-Sun L_GW Peters r^-5 [W]")

# ════════════════════════════════════════════════════════════════════════
# 3. The sigma-bridge couplings from the repo (inventory/core/sigma.py is the only live copy of XI)
# ════════════════════════════════════════════════════════════════════════
xi = engine_leaf("XI", XI, "sigma_ground.inventory.core.sigma.XI")
f_p = engine_leaf("f_p = qcd_fraction(proton)", nucleon_qcd_fraction()["proton"]["qcd_fraction"],
                  "sigma_ground.inventory.core.sigma.nucleon_qcd_fraction")
d = 1e-6
f_p_num = engine_leaf("f_p numeric dln m_p/dsigma", (math.log(proton_mass_kg(d)) - math.log(proton_mass_kg(-d))) / (2 * d),
                      "central difference of sigma_ground.inventory.core.sigma.proton_mass_kg")
iso = json.load(open(os.path.join(REPO, "sigma_ground", "inventory", "data", "isotopes.json"), encoding="utf-8"))


def f_atom(Z, A):
    rec = next(r for r in iso if r["Z"] == Z and r["A"] == A)
    v, _ = mentat_call("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": rec["atomic_mass_u"]},
                       f"BE Z={Z} A={A}")
    be = v["binding_energy_MeV"]
    m = lambda s: three_measures_atom(Z, A - Z, be, s)["atom_stable_mass_kg"]
    f = (math.log(m(d)) - math.log(m(-d))) / (2 * d)
    engine_leaf(f"f_A Z={Z} A={A}", f, "central difference of sigma_ground.inventory.core.sigma.three_measures_atom")
    return f


F = {(Z, A): f_atom(Z, A) for Z, A, _ in LIT["alloys"]["TA6V"] + LIT["alloys"]["PtRh10"]}
f_Ti, f_Pt = F[(22, 48)], F[(78, 195)]
f_TA6V = sum(w * F[(Z, A)] for Z, A, w in LIT["alloys"]["TA6V"])
f_PtRh = sum(w * F[(Z, A)] for Z, A, w in LIT["alloys"]["PtRh10"])

# ════════════════════════════════════════════════════════════════════════
# 4. The five gate rows (massless sigma, xi = XI) — Mentat arithmetic
# ════════════════════════════════════════════════════════════════════════
a2 = calc(f"2*{xi}*{f_p}", "alpha_eff^2 = 2 xi f_p")
alpha = calc(f"sqrt({a2})", "alpha_DEF = sqrt(alpha_eff^2)")
gam_attr = calc(f"-2*{a2}/(1+{a2})", "gamma-1 (DEF attractive sign)")
gam_rep = calc(f"2*{a2}/(1-{a2})", "gamma-1 (repo repulsive sign)")
omega_bd = calc(f"(1/{a2}-3)/2", "omega_BD = (1/alpha^2-3)/2")
eta_N = calc(f"2*{a2}/(1+{a2})", "eta_N = 4beta-gamma-3 = 2 alpha^2/(1+alpha^2) (BD, beta=1)")
eta_TiPt_iso = calc(f"2*{xi}*({f_Ti}-{f_Pt})", "eta(Ti-48,Pt-195) = 2 xi (f_Ti - f_Pt)  [design: pure isotopes]")
eta_TiPt_alloy = calc(f"2*{xi}*({f_TA6V}-{f_PtRh})", "eta(TA6V,PtRh10) = 2 xi (f_TA6V - f_PtRh10)  [S9a: alloys]")
beta_lpi = calc(f"2*{xi}*{f_p}", "beta_LPI = |(c^2/mu) dmu/dPhi| = 2 xi f_p (m_e sigma-invariant)")
dU_ann = calc(f"{G}*{M_SUN_BODY}*{LIT['earth_e']['e']}/({earth['semimajor_axis_au']}*{AU}*{C}**2)",
              "annual Delta U/c^2 amplitude at Earth (e from [LIT])")
dlnmu_ann = calc(f"{beta_lpi}*{dU_ann}", "predicted annual Delta ln mu amplitude")

# dipole (Freire+2012 Eq. 21, e -> 0): Pbdot_D = -coef * dalpha^2
q = calc(f"{m1}/{m2}", "q = m_p/m_c")
coef = calc(f"2*pi*(2*pi/{Pb})*({G}*{m2}/{C}**3)*({q}/({q}+1))", "|Pbdot_D|/(dalpha)^2 [s/s]")
DALPHA = (1.9e-3, 0.056, 0.11)    # pre-registered rows
dalpha_repo = {b: calc(f"{alpha}*{b}", f"dalpha = alpha_DEF*{b} (crude NS binding {int(b*100)} %)") for b in (0.1, 0.2)}
dip = {da: -calc(f"{coef}*{da}**2", f"Pbdot_D at dalpha={da}") for da in DALPHA}

# thresholds (pre-registered) and excesses (Mentat log10)
xs_lo = J["Pbdot_xs_err_fs"][1]
BOUND = dict(cassini=4.6e-5, llr=2.2e-4, microscope=5.4e-15, clocks=9e-7, dipole_dalpha2=3.6e-6)


def lg(x, label):
    return calc(f"log({x}, 10)", label)


rows = []
def row(name, pred, bound, test, lit, extra=None):
    ex = lg(f"{abs(pred)}/{bound}", f"log10 excess {name}")
    r = dict(row=name, test=test, prediction=pred, bound_2sigma=bound, log10_excess=ex,
             verdict="GO" if abs(pred) < bound else "NO-GO", lit=lit)
    if extra:
        r.update(extra)
    rows.append(r)
    return r


row("|gamma-1| (Cassini)", gam_attr, BOUND["cassini"], "Shapiro delay, Cassini 2002", LIT["cassini"]["ref"],
    dict(alt_prediction_repulsive_sign=gam_rep, alt_log10_excess=lg(f"{gam_rep}/{BOUND['cassini']}", "log10 excess gamma repulsive"),
         predicted_log10_excess=4.0))
row("|eta_N| (LLR Nordtvedt)", eta_N, BOUND["llr"], "lunar laser ranging", LIT["llr"]["ref"], dict(predicted_log10_excess=3.3))
row("eta(Ti,Pt) (MICROSCOPE)", eta_TiPt_iso, BOUND["microscope"], "WEP in orbit", LIT["microscope"]["ref"],
    dict(alt_prediction_alloys=eta_TiPt_alloy, alt_log10_excess=lg(f"{eta_TiPt_alloy}/{BOUND['microscope']}", "log10 excess eta alloys"),
         predicted_log10_excess=11.1))
row("|beta_LPI| (Yb+/Cs clocks)", beta_lpi, BOUND["clocks"], "mu = m_p/m_e vs solar potential", LIT["clocks"]["ref"],
    dict(predicted_log10_excess=5.5, annual_dlnmu_amplitude=dlnmu_ann))
dip_rows = []
for da in DALPHA:
    da2 = calc(f"{da}**2", f"dalpha^2 at {da}")
    ex = lg(f"{da2}/{BOUND['dipole_dalpha2']}", f"log10 excess dipole dalpha={da}")
    dip_rows.append(dict(dalpha=da, dalpha2=da2, Pbdot_D_fs=dip[da] * 1e15,
                         times_observed_decay=abs(dip[da]) / abs(J["Pbdot_int_fs"] * 1e-15),
                         log10_excess_vs_bound=ex,
                         verdict_design_dalpha2=("GO" if da2 < BOUND["dipole_dalpha2"] else "NO-GO"),
                         verdict_x1_sketch_abs_lt_3p7fs=("GO" if abs(dip[da]) < 3.7e-15 else "NO-GO"),
                         verdict_x1_text_within_minus3p6fs=("GO" if dip[da] > xs_lo * 1e-15 else "NO-GO"),
                         verdict_corrected_2sigma=("GO" if dip[da] > (J["Pbdot_xs_fs"] + 2 * xs_lo) * 1e-15 else "NO-GO"),
                         verdict_corrected_1sigma=("GO" if dip[da] > (J["Pbdot_xs_fs"] + xs_lo) * 1e-15 else "NO-GO")))
# the pre-registered physics row is the repo's dalpha (0.056-0.11); 1.9e-3 is the threshold row itself
repo_dip = [r for r in dip_rows if r["dalpha"] >= 0.05]
rows.append(dict(row="(dalpha)^2 (J1738+0333 dipole)", test="NS-WD dipole radiation", prediction=[r["dalpha2"] for r in repo_dip],
                 bound_2sigma=BOUND["dipole_dalpha2"], log10_excess=[r["log10_excess_vs_bound"] for r in repo_dip],
                 verdict="NO-GO" if all(r["verdict_design_dalpha2"] == "NO-GO" for r in repo_dip) else "GO",
                 lit=J["ref"], predicted_log10_excess=">= 3 (U1 pass_fail) / >= 2 (X1)"))
all_nogo = all(r["verdict"] == "NO-GO" for r in rows)
overall = "NO-GO" if any(r["verdict"] == "NO-GO" for r in rows) else "GO"

# corrected dipole thresholds (design used the +3.7 error on |dipole|; the dipole is negative, so the -3.6 side and the
# +2.0 central value of Pbdot_xs are what bound it)
da2_1s = calc(f"({-(J['Pbdot_xs_fs'] + xs_lo)}e-15)/{coef}", "(dalpha)^2 bound, 1 sigma, from Pbdot_xs = 2.0 - 3.6")
da2_2s = calc(f"({-(J['Pbdot_xs_fs'] + 2 * xs_lo)}e-15)/{coef}", "(dalpha)^2 bound, 2 sigma, from Pbdot_xs = 2.0 - 7.2")

# minimal pulsar compactness that still gives NO-GO, with the trace charge M_sigma/M_T = 1 - (3/(5-n)) r_s/R (n = 1)
# (program sec. 2.2, [DERIVED]); WD charge ~ 1.
rsR_min = calc(f"sqrt({BOUND['dipole_dalpha2']})/{alpha}/0.75", "r_s/R minimum for NO-GO (trace charge, n=1)")
R_max_ns = calc(f"{rs_p}/{rsR_min}", "largest pulsar radius still giving NO-GO [m]")

# ════════════════════════════════════════════════════════════════════════
# 5. Routes (recorded, not chosen)
# ════════════════════════════════════════════════════════════════════════
# (a) long-range, xi small: xi_max per row
def xi_from_eta2(B, label):  # 2a/(1+a) = B -> a = B/(2-B); xi = a/(2 f_p)
    return calc(f"({B}/(2-{B}))/(2*{f_p})", label)


route_a = dict(
    cassini=xi_from_eta2(BOUND["cassini"], "xi_max Cassini (long range)"),
    llr=xi_from_eta2(BOUND["llr"], "xi_max LLR"),
    microscope=calc(f"{BOUND['microscope']}/(2*({f_Ti}-{f_Pt}))", "xi_max MICROSCOPE (isotopes)"),
    clocks=calc(f"{BOUND['clocks']}/(2*{f_p})", "xi_max clocks"),
    dipole_b10=calc(f"{BOUND['dipole_dalpha2']}/(2*{f_p}*0.1**2)", "xi_max dipole (binding 10 %)"),
    dipole_b20=calc(f"{BOUND['dipole_dalpha2']}/(2*{f_p}*0.2**2)", "xi_max dipole (binding 20 %)"),
)
route_a["binding"] = min(route_a, key=route_a.get)
route_a["xi_max_all_rows"] = route_a[route_a["binding"]]
route_a["xi_max_if_universal_coupling"] = min(route_a["cassini"], route_a["llr"])  # universal f: MICROSCOPE and clock rows -> 0


# (b) massive sigma at the Captain's xi: Yukawa range lambda_max per row (sphere form factor, as sigma_exclusion.py)
def S(R, r, lam):
    x = R / lam
    if x < 1e-3:
        return math.exp(-r / lam) * (1 + x * x / 10)
    ex = -(r - R) / lam
    if ex < -700:
        return 0.0
    return math.exp(ex) * 3 * ((x - 1) + math.exp(-2 * x) * (x + 1)) / (2 * x ** 3)


def lam_max(supp, need):
    """largest lambda with supp(lambda) <= need (supp increases with lambda); bisection in log lambda."""
    lo, hi = 1e-20, 1e30
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if supp(mid) > need:
            hi = mid
        else:
            lo = mid
    return lo


R_ORB = R_E + LIT["microscope"]["altitude_m"]
B_CAS = LIT["cassini"]["b_min_rsun"] * R_SUN
geo = dict(cassini=(R_SUN, B_CAS), llr=(R_SUN, AU), microscope=(R_E, R_ORB), clocks=(R_SUN, AU))
route_b = {}
for X in (0.13, XI, 0.17):
    a2x = 2 * X * f_p
    pred = dict(cassini=2 * a2x / (1 + a2x), llr=2 * a2x / (1 + a2x), microscope=2 * X * (f_Ti - f_Pt), clocks=2 * X * f_p)
    rb = {}
    for k, (R, r) in geo.items():
        need = BOUND[k] / pred[k]
        lam = lam_max(lambda L: S(R, r, L) * (1 + r / L), need)
        # Mentat checks the root: suppression at lambda_max equals the required factor
        x = R / lam
        chk = calc(f"exp(-({r}-{R})/{lam})*3*(({x}-1)+exp(-2*{x})*({x}+1))/(2*{x}**3)*(1+{r}/{lam})/({need})",
                   f"route b check {k} xi={X}: S(lam_max)/need")
        rb[k] = dict(lambda_max_m=lam, suppression_needed=need, mentat_check_ratio=chk)
    rb["dipole"] = dict(lambda_max_m=calc(f"{C}*{Pb}/(2*pi)", "dipole: lambda below which a field cannot radiate at omega_orb"),
                        note="massive field radiates only at omega > c/lambda; below this range the dipole row passes trivially")
    rb["binding_row"] = min((k for k in rb if isinstance(rb[k], dict)), key=lambda k: rb[k]["lambda_max_m"])
    route_b[str(X)] = rb

# (c) notation: sigma == -2 xi Phi/c^2, non-dynamical, universal -> no prediction on any row (by construction).
# Caveat: a universal MASS-ONLY rescale m -> m e^sigma (light untouched) is still a universal repulsive force 2 xi:
gam_c = calc(f"4*{xi}/(1-2*{xi})", "route c caveat: gamma-1 if e^sigma rescales all masses but not light")
route_c = dict(statement="sigma == -2 xi Phi/c^2 as universal, non-dynamical notation; drops the sigma-invariant m_e/alpha_EM/bond rules",
               rows="all five pass BY CONSTRUCTION (no observable)",
               caveat_gamma_minus_1_mass_only_universal=gam_c,
               caveat_log10_excess=lg(f"{gam_c}/{BOUND['cassini']}", "log10 excess route c mass-only"),
               caveat="passes only if e^sigma is a change of units (all dimensionful constants together, Duff 2002); "
                      "a mass-only universal rescale is excluded by Cassini")

# ════════════════════════════════════════════════════════════════════════
# 6. JSON
# ════════════════════════════════════════════════════════════════════════
res = dict(
    id="U1", date="2026-10-10",
    prereg=dict(engine="Pbdot_engine within 2 % of Peters (by construction); > 5 % = nbody bug",
                physics="coded sigma passes only if ALL of |gamma-1|<4.6e-5, |eta_N|<2.2e-4, eta(Ti,Pt)<5.4e-15, "
                        "|beta_LPI|<9e-7, (dalpha)^2<3.6e-6; predicted NO-GO on all five, excesses 4.0/3.3/11.1/5.5/>=3 orders"),
    engine=dict(J1738=dict(Pb_s=Pb, m_p_kg=m1, m_c_kg=m2, a_m=a_rel, v_rel_m_s=v_rel, E_kepler_J=E_kep,
                           PN_parameter=pn_param, r_s_pulsar_m=rs_p, r_s_companion_m=rs_c),
                dlnPb_dE=dlnP_dE, Pbdot_peters_mentat=pbdot_peters,
                E0_engine_vs_kepler_rel=(main["E0_J"] - E_kep) / abs(E_kep), Pbdot_peters_vs_freire_GR_rel=peters_vs_lit,
                runs={str(k): {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in eng.items()},
                rel_dev_vs_peters=rel, rel_dev_engine_vs_freire_GR=rel_lit,
                defect_diagnosis=dict(engine_over_peters_divided_by_a_per_m=ratio_over_a,
                                      same_at_a_over_10=ratio10_over_a, Pbdot_peters_at_a_over_10=peters10,
                                      counterfactual_r5_rel_dev_vs_peters=rel_r5,
                                      earth_sun_LGW_r4_W=P_es_r4, earth_sun_LGW_r5_W=P_es_r5),
                verdict=("PASS" if engine_pass else ("BUG" if engine_bug else "FAIL(2-5 %)"))),
    couplings=dict(XI=xi, f_p=f_p, f_p_numeric=f_p_num, f_atoms={f"Z={Z},A={A}": v for (Z, A), v in F.items()},
                   f_TA6V=f_TA6V, f_PtRh10=f_PtRh, alpha_eff2=a2, alpha_DEF=alpha, omega_BD=omega_bd,
                   gamma_minus_1_attractive=gam_attr, gamma_minus_1_repulsive=gam_rep),
    gate_rows=rows, dipole_rows=dip_rows, dipole_coef_per_dalpha2=coef, dalpha_repo_crude=dalpha_repo,
    dipole_corrected_bounds=dict(dalpha2_1sigma=da2_1s, dalpha2_2sigma=da2_2s,
                                 note="Pbdot_xs = +2.0(+3.7/-3.6) fs/s and Pbdot_D < 0, so the -3.6 side with the +2.0 central value bounds it"),
    dipole_robustness=dict(rsR_min_for_nogo=rsR_min, R_max_pulsar_m=R_max_ns,
                           note="trace-charge reading M_sigma/M_T = 1-(3/(5-n)) r_s/R, n=1; WD charge ~1 [DERIVED, normalisation SPECULATIVE]"),
    overall_verdict=overall, all_five_nogo=all_nogo,
    routes=dict(a_long_range_small_xi=route_a, b_massive_sigma_at_captains_xi=route_b, c_notation=route_c,
                chosen=None, note="recorded, not chosen (the Captain's decision; design section 2.5 item 6)"),
    mentat_diagnostics=[dict(file="sigma_ground/mcp/tools/symbolic.py", line=142,
                             defect="differentiate_expr parses the expression with sp.sympify before creating the variable "
                                    "Symbol, so variable 'E' (also I, N, S, O, Q) becomes a sympy constant and the derivative is "
                                    "silently 0", evidence=f"d(E**2)/dE returned {dE2!r}")],
    mentat_msun_note=dict(lookup_constant_M_sun=MSUN, solar_system_body_sun_mass=M_SUN_BODY,
                          rel=(MSUN - M_SUN_BODY) / M_SUN_BODY),
    companion_files=dict(precision_diagnostic="u1_gw_precision_diag.json (u1_gw_precision_diag.py)",
                         gain_scan="u1_gain_scan.json (u1_gain_scan.py)", series="u1_series.json"),
    lit=LIT, mentat_calls=CALLS[0], sources=LOG, runtime_s=time.time() - t_start,
)
json.dump(res, open(os.path.join(OUT, "u1_results.json"), "w", encoding="utf-8"), indent=1, default=str)
json.dump({k: v["series"] for k, v in eng.items()}, open(os.path.join(OUT, "u1_series.json"), "w"), indent=0)
print(json.dumps(dict(engine=res["engine"]["rel_dev_vs_peters"], diag=res["engine"]["defect_diagnosis"], engine_verdict=res["engine"]["verdict"],
                      Pbdot_engine_fs=main["Pbdot_energy_endpoint"] * 1e15, Pbdot_timing_fs=main["Pbdot_timing"] * 1e15,
                      peters_fs=pbdot_peters * 1e15,
                      rows=[(r["row"], r["prediction"], r["log10_excess"], r["verdict"]) for r in rows],
                      dip=[(r["dalpha"], r["Pbdot_D_fs"], r["verdict_design_dalpha2"], r["verdict_x1_sketch_abs_lt_3p7fs"],
                            r["verdict_x1_text_within_minus3p6fs"], r["verdict_corrected_2sigma"]) for r in dip_rows],
                      overall=overall, route_a=route_a,
                      route_b={k: {kk: (vv["lambda_max_m"] if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in route_b.items()},
                      route_c=gam_c, calls=CALLS[0], runtime=res["runtime_s"]), indent=1, default=str))
