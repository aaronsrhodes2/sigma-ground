"""U9 repair (v2) — independent-integrator check of the orbit results, and the reference laws that give the frame test
its discriminating power.  Written by the repair agent; criteria fixed in this docstring BEFORE the script was run.

Integrator: scipy DOP853 (adaptive 8th order, rtol 1e-13), NOT the engine's Forest-Ruth, so the O(dt) Forest-Ruth bias
with velocity-dependent kicks is absent.  Mercury-Sun pair, Mentat leaves, same initial data as u9_orbits.py
(pericentre on +x, perihelion velocity on +y), 10 orbits, apocentre events, osculating Laplace-Runge-Lenz vector.
Each law's precession is its LRL slope minus the Newtonian slope IN THE SAME FRAME, in units of u = pi G M/(c^2 p).

Laws (accelerations of both bodies):
  newton       engine NBodySystem.compute_accelerations, PhysicsToggles()                    (engine, unmodified)
  gr_engine    engine, PhysicsToggles(gr_1pn=True)                                           (engine, unmodified)
  eih_engine   engine, PhysicsToggles(eih_cross=True)                                        (engine, unmodified)
  eih_ref      eih_engine MINUS 2 x the engine's own velocity cross-term, i.e. the engine's EIH with only the sign of
               (r_ij.(4v_i - 3v_j))(v_i - v_j) restored to n_ij = x_i - x_j  [reference law, scratch what-if; the engine
               is NOT changed; the cross term is recomputed here from nbody.py:457-459 with the engine's gm and _c2]
  gr_ref       a_N - (a_gr_engine - a_N): gr_1pn with its whole 1PN correction negated (outward n) [scratch what-if]
Validation before the runs: on a 1 kg test body around a resting Sun, eih_ref and gr_ref must fit the textbook
test-particle form K[(A GM/r - B v^2) n + C (n.v) v] with (A,B,C) = (4,1,4) to <= 1e-6 (two-body extras ~ m/M).

PRE-REGISTERED CRITERIA (fixed before running; predictions are u9_gauss.json, first-order Gauss, Mentat integrate_expr):
  D1  engine laws: gr_engine and eih_engine precession/u equal the Gauss predictions within 1e-4 relative in all four
      frames (rest, x, y, z; V = 30 km/s), and de/u within 1e-3 absolute.  If met, the Forest-Ruth results of the
      original deliverable are integrator-independent.
  D2  discriminating power: eih_ref gives +6.000 u within 1e-3 relative in ALL four frames and |de/u| <= 1e-3.
      If eih_ref also changes by > 1 % under an in-plane boost, the frame test cannot separate a correct law from a
      broken one and the eih_cross frame-dependence finding is withdrawn.
  D3  gr_ref equals the Gauss 'gr_textbook' predictions (+6, +6.384, +35.312, +6 u; de_x +6.054 u) within 1e-4 / 1e-3.
  D4  gr_1pn's frame dependence is independent of its sign defect: |ratio_y(gr_engine) - ratio_y(gr_ref)| <= 1e-4,
      ratio_y = precession(y)/precession(rest).  If met, the gr_1pn x5.9 change is [BY CONSTRUCTION] (absolute v_i),
      not evidence of a further defect.
Also measured (no criterion): the fixture of test_nbody.py::test_eih_two_body_reduces_to_schwarzschild_at_leading_order
(|a_eih|/|a_1pn| - 1, against its 1e-4 tolerance).
Usage: python -X utf8 u9_dop853_v2.py OUT.json GAUSS.json
"""
import json, math, sys, time
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
import sigma_ground.field.interface.nbody as nb
from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles
from sigma_ground.field.interface.rolling_analysis import _ORBITAL

OUT, GAUSS = sys.argv[1], sys.argv[2]
gauss = json.load(open(GAUSS, encoding="utf-8"))
SRC = {}


def leaf(tool, args, key):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


G = leaf("lookup_constant", {"name": "G"}, "G")
C = leaf("lookup_constant", {"name": "c"}, "c")
AU = leaf("lookup_constant", {"name": "au"}, "au")
sun = leaf("solar_system_body", {"body_name": "sun"}, "sun")
mer = leaf("solar_system_body", {"body_name": "mercury"}, "mercury")
M_S, R_S, M_M, R_M = sun["mass_kg"], sun["radius_m"], mer["mass_kg"], mer["radius_m"]
A_M = mer["semimajor_axis_au"] * AU
E_M = _ORBITAL["Mercury"][1]
SRC["e_mercury"] = {"value": E_M, "source": "sigma_ground.field.interface.rolling_analysis._ORBITAL['Mercury'] (JPL fact sheet)"}
SRC["engine_c2"] = {"value": nb._c2, "source": "sigma_ground.field.interface.nbody._c2 (engine's own c^2)"}
MU = G * (M_S + M_M)
P_KEP = 2 * math.pi * math.sqrt(A_M ** 3 / MU)
U = math.pi * MU / (C * C * A_M * (1 - E_M ** 2))
TOG = {"newton": PhysicsToggles(), "gr": PhysicsToggles(gr_1pn=True), "eih": PhysicsToggles(eih_cross=True)}


def engine(xs, vs, ms, phys):
    bl = [CelestialBody(m, x.copy(), v.copy(), 1.0, 0.0) for m, x, v in zip(ms, xs, vs)]
    return NBodySystem(bl, toggles=TOG[phys]).compute_accelerations()


def engine_cross(xs, vs, ms):
    """The engine's EIH velocity cross-term, recomputed from nbody.py:457-459 (r_ij = x_j - x_i, engine gm, engine _c2)."""
    bl = [CelestialBody(m, x.copy(), v.copy(), 1.0, 0.0) for m, x, v in zip(ms, xs, vs)]
    acc = np.zeros((len(bl), 3))
    for i, bi in enumerate(bl):
        for j, bj in enumerate(bl):
            if i == j:
                continue
            rij = bj.position_m - bi.position_m
            r = math.sqrt(rij @ rij)
            cross = float(rij @ (4.0 * bi.velocity_m_s - 3.0 * bj.velocity_m_s)) / nb._c2
            acc[i] += (bj.gm_m3_s2 / (r * r * r)) * cross * (bi.velocity_m_s - bj.velocity_m_s)
    return acc


def law_acc(law, xs, vs, ms):
    if law == "newton":
        return engine(xs, vs, ms, "newton")
    if law == "gr_engine":
        return engine(xs, vs, ms, "gr")
    if law == "eih_engine":
        return engine(xs, vs, ms, "eih")
    if law == "eih_ref":
        return engine(xs, vs, ms, "eih") - 2.0 * engine_cross(xs, vs, ms)
    if law == "gr_ref":
        an = engine(xs, vs, ms, "newton")
        return an - (engine(xs, vs, ms, "gr") - an)
    raise KeyError(law)


# ── validation: test-particle coefficient fit of the two reference laws (and the engine laws) ──
rng = np.random.default_rng(20261010)
fits = {}
for law in ("gr_engine", "eih_engine", "gr_ref", "eih_ref"):
    rows, rhs = [], []
    for _ in range(12):
        r = A_M * rng.uniform(0.6, 1.4)
        nvec = rng.normal(size=3); nvec /= np.linalg.norm(nvec)
        v = rng.normal(size=3) * 4.0e4
        xs = [np.zeros(3), r * nvec]; vs = [np.zeros(3), v]; ms = [M_S, 1.0]
        d = law_acc(law, xs, vs, ms)[1] - engine(xs, vs, ms, "newton")[1]
        gm = CelestialBody(M_S, np.zeros(3), np.zeros(3), 1.0, 0.0).gm_m3_s2   # engine GM of the Sun
        K = gm / (r * r * nb._c2)
        cols = np.stack([K * (gm / r) * nvec, -K * (v @ v) * nvec, K * (nvec @ v) * v], axis=1)
        rows.append(cols); rhs.append(d)
    Amat = np.concatenate(rows); bvec = np.concatenate(rhs)
    coef, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
    res = float(np.linalg.norm(Amat @ coef - bvec) / np.linalg.norm(bvec))
    fits[law] = {"A": float(coef[0]), "B": float(coef[1]), "C": float(coef[2]), "rel_residual": res}
    print("fit", law, fits[law], flush=True)

# ── the test_nbody fixture (test_eih_two_body_reduces_to_schwarzschild_at_leading_order), values copied from the test ──
from sigma_ground.field.interface import test_nbody as tnb   # import only, to read its M_SUN/AU/R_SUN constants
b = lambda: [CelestialBody(tnb.M_SUN, np.zeros(3), np.zeros(3), tnb.R_SUN, 0.02),
             CelestialBody(3.3e23, np.array([0.4 * tnb.AU, 0.0, 0.0]), np.array([0.0, 4.7e4, 0.0]), 2.44e6, 0.45)]
a1 = NBodySystem(b(), toggles=PhysicsToggles(gr_1pn=True)).compute_accelerations()[1]
a2 = NBodySystem(b(), toggles=PhysicsToggles(eih_cross=True)).compute_accelerations()[1]
an = NBodySystem(b(), toggles=PhysicsToggles()).compute_accelerations()[1]
fixture = {"ratio_minus_1": float(np.linalg.norm(a2) / np.linalg.norm(a1) - 1), "test_tolerance": 1e-4,
           "gr_1pn_radial_corr_over_aN": float((a1 - an) @ an / (an @ an)),
           "eih_cross_radial_corr_over_aN": float((a2 - an) @ an / (an @ an)),
           "note": "n.v = 0 at this state; the cross term vanishes, the radial 1PN terms of the two layers have opposite signs"}
print("fixture", fixture, flush=True)

# ── orbits ──
FR = {"rest": np.zeros(3), "x": np.array([3.0e4, 0, 0]), "y": np.array([0, 3.0e4, 0]), "z": np.array([0, 0, 3.0e4])}
N_ORB = 10
MS = [M_S, M_M]


def run(law, frame):
    rp = A_M * (1 - E_M); vp = math.sqrt(MU * (1 + E_M) / rp)
    rvec, vvec = np.array([rp, 0, 0.0]), np.array([0, vp, 0.0])
    fS, fM = M_M / (M_S + M_M), M_S / (M_S + M_M)
    V = FR[frame]
    y0 = np.concatenate([-fS * rvec, fM * rvec, -fS * vvec + V, fM * vvec + V])

    def rhs(t, y):
        xs = [y[0:3], y[3:6]]; vs = [y[6:9], y[9:12]]
        return np.concatenate([y[6:], law_acc(law, xs, vs, MS).ravel()])

    def apo(t, y):
        return float((y[3:6] - y[0:3]) @ (y[9:12] - y[6:9]))
    apo.direction = -1
    t0 = time.time()
    sol = solve_ivp(rhs, (0, (N_ORB + 0.25) * P_KEP), y0, method="DOP853", rtol=1e-13, atol=1e-6, events=apo)
    om, ee = [], []
    for y in sol.y_events[0]:
        rr = y[3:6] - y[0:3]; vv = y[9:12] - y[6:9]
        L = np.cross(rr, vv); ev = np.cross(vv, L) / MU - rr / math.sqrt(rr @ rr)
        om.append(math.atan2(ev[1], ev[0])); ee.append(float(np.linalg.norm(ev[:2])))
    om = np.unwrap(np.array(om)); idx = np.arange(len(om))
    return {"law": law, "frame": frame, "n_apo": len(om), "domega_per_orbit": float(np.polyfit(idx, om, 1)[0]),
            "de_per_orbit": float(np.polyfit(idx, ee, 1)[0]), "nfev": int(sol.nfev), "wall_s": time.time() - t0}


runs = {}
for law in ("newton", "gr_engine", "eih_engine", "eih_ref", "gr_ref"):
    for fr in FR:
        r = run(law, fr)
        runs[f"{law}_{fr}"] = r
        print(json.dumps(r), flush=True)
prec = {}
for k, r in runs.items():
    if r["law"] == "newton":
        continue
    bnm = runs[f"newton_{r['frame']}"]
    prec[k] = {"over_u": (r["domega_per_orbit"] - bnm["domega_per_orbit"]) / U,
               "de_over_u": (r["de_per_orbit"] - bnm["de_per_orbit"]) / U}
PRED = {"gr_engine": "gr_engine", "eih_engine": "eih_engine", "gr_ref": "gr_textbook", "eih_ref": "eih_textbook"}
for k, pr in prec.items():
    law, fr = k.rsplit("_", 1)
    pr["pred_over_u"] = gauss["pred_over_u"][f"{PRED[law]}_{fr}"]
    pr["de_pred_over_u"] = gauss["de_pred_over_u"][f"{PRED[law]}_{fr}"]
    pr["rel_dev"] = pr["over_u"] / pr["pred_over_u"] - 1
    pr["de_abs_dev"] = pr["de_over_u"] - pr["de_pred_over_u"]
    print(f"{k:16s} dω/u {pr['over_u']:+10.5f} pred {pr['pred_over_u']:+10.5f} dev {pr['rel_dev']:+.1e}"
          f"  de/u {pr['de_over_u']:+9.5f} pred {pr['de_pred_over_u']:+9.5f}", flush=True)
frames = list(FR)
crit = {
    "D1_engine_matches_gauss": all(abs(prec[f"{l}_{f}"]["rel_dev"]) <= 1e-4 and abs(prec[f"{l}_{f}"]["de_abs_dev"]) <= 1e-3
                                   for l in ("gr_engine", "eih_engine") for f in frames),
    "D2_eih_ref_plus6_all_frames": all(abs(prec[f"eih_ref_{f}"]["over_u"] / 6.0 - 1) <= 1e-3
                                       and abs(prec[f"eih_ref_{f}"]["de_over_u"]) <= 1e-3 for f in frames),
    "D3_gr_ref_matches_gauss_textbook": all(abs(prec[f"gr_ref_{f}"]["rel_dev"]) <= 1e-4
                                            and abs(prec[f"gr_ref_{f}"]["de_abs_dev"]) <= 1e-3 for f in frames),
}
ry = {l: prec[f"{l}_y"]["over_u"] / prec[f"{l}_rest"]["over_u"] for l in ("gr_engine", "gr_ref", "eih_engine", "eih_ref")}
crit["D4_gr_frame_dependence_sign_independent"] = abs(ry["gr_engine"] - ry["gr_ref"]) <= 1e-4
crit["ratio_y_over_rest"] = ry
crit["validation_ref_laws_textbook_4_1_4"] = all(abs(fits[l][q] - t) <= 1e-6 * 4
                                                 for l in ("gr_ref", "eih_ref") for q, t in (("A", 4), ("B", 1), ("C", 4)))
print(json.dumps(crit, indent=1), flush=True)
out = {"sources": SRC, "u_rad_per_orbit": U, "P_kepler_s": P_KEP, "coefficient_fits": fits, "test_fixture": fixture,
       "runs": runs, "precession": prec, "criteria": crit,
       "integrator": "scipy DOP853, rtol 1e-13, atol 1e-6, apocentre events, 10 orbits, per-frame Newtonian baseline"}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
