"""U9 part C — the Mercury anchor and the frame test on whole orbits.

PRE-REGISTERED (written before any orbit was integrated):
  anchor     engine perihelion advance (run minus the Newtonian run, same dt) = +42.98"/cy [LIT Will 2014, Living Rev.
             Relativ. 17, 4, the GR value for gamma = beta = 1] and = 6 pi G(M+m)/(c^2 a (1-e^2)) per orbit from Mentat
             leaves, within 1 %, for gr_1pn and for eih_cross.  My expectation after the instantaneous sign probe
             (engine coefficients gr_1pn (A,B,C) = (-4,-1,-4), eih_cross (4,1,-4)) and the Gauss secular integral
             d(omega)/orbit = (-A + 2B + 2C) pi G M/(c^2 p): gr_1pn -> -42.98"/cy, eih_cross -> -71.6"/cy.
  frame test a post-Galilean (Lorentz) invariant 1PN law gives the same relative-orbit precession in the rest frame and in
             frames boosted by 30 km/s (x = perihelion direction, y = perihelion velocity, z = orbit normal), within 1 %
             (physical difference O(V^2/c^2) ~ 1e-8). Expected: z invariant for both engine laws; in-plane boosts
             frame-dependent for gr_1pn (absolute v_i, a preferred frame) and for eih_cross (velocity cross-term sign).
  control    Newtonian run boosted along x reproduces the rest-frame Newtonian precession (Galilean covariance of the
             discrete map); dt-halving for gr_1pn rest (1200 s vs 600 s) agrees within 0.5 %.
Usage: python -X utf8 u9_orbits.py OUT.json run1 run2 ...   (run = <physics>_<frame>[_dt<seconds>])
"""
import json, math, sys, time
import numpy as np

sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles
from sigma_ground.field.interface.rolling_analysis import _ORBITAL

OUT = sys.argv[1]
RUNS = sys.argv[2:]
N_ORB = 20
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
MU = G * (M_S + M_M)
P_KEP = 2 * math.pi * math.sqrt(A_M ** 3 / MU)
TOG = {"newton": PhysicsToggles(), "gr": PhysicsToggles(gr_1pn=True), "eih": PhysicsToggles(eih_cross=True)}
DIRS = {"rest": np.zeros(3), "x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
V0 = 3.0e4


def run(name):
    parts = name.split("_")
    phys, frame = parts[0], parts[1]
    dt = float(parts[2][2:]) if len(parts) > 2 else 600.0
    rp = A_M * (1 - E_M)
    vp = math.sqrt(MU * (1 + E_M) / rp)
    rvec, vvec = np.array([rp, 0, 0.0]), np.array([0, vp, 0.0])
    fS, fM = M_M / (M_S + M_M), M_S / (M_S + M_M)
    V = V0 * DIRS[frame]
    sys_ = NBodySystem([CelestialBody(M_S, -fS * rvec, -fS * vvec + V, R_S, 0.0),
                        CelestialBody(M_M, fM * rvec, fM * vvec + V, R_M, 0.0)], toggles=TOG[phys])
    nsteps = int(round(N_ORB * P_KEP / dt)) + 3
    r_prev2 = r_prev = None
    st_prev = None
    apo = []          # (t, omega, e) at apocentres
    t0 = time.time()
    for k in range(nsteps):
        sys_.forest_ruth_step(dt)
        b0, b1 = sys_.bodies
        rr = b1.position_m - b0.position_m
        vv = b1.velocity_m_s - b0.velocity_m_s
        r = math.sqrt(rr @ rr)
        if r_prev2 is not None and r_prev > r_prev2 and r_prev >= r:
            rr_, vv_, t_ = st_prev
            L = np.cross(rr_, vv_)
            evec = np.cross(vv_, L) / MU - rr_ / math.sqrt(rr_ @ rr_)
            apo.append((t_, math.atan2(evec[1], evec[0]), float(np.linalg.norm(evec[:2])), float(evec[2])))
        r_prev2, r_prev = r_prev, r
        st_prev = (rr, vv, sys_.time)
    apo = np.array(apo)
    idx = np.arange(len(apo))
    om = np.unwrap(apo[:, 1])
    slope, icpt = np.polyfit(idx, om, 1)
    resid = om - (slope * idx + icpt)
    eslope = np.polyfit(idx, apo[:, 2], 1)[0]
    period = float(np.mean(np.diff(apo[:, 0])))
    return {"name": name, "physics": phys, "frame": frame, "V_m_s": float(np.linalg.norm(V)), "dt_s": dt,
            "n_apocentres": int(len(apo)), "domega_per_orbit_rad": float(slope),
            "domega_fit_rms_rad": float(np.sqrt(np.mean(resid ** 2))), "de_per_orbit": float(eslope),
            "e_first": float(apo[0, 2]), "ez_max": float(np.max(np.abs(apo[:, 3]))),
            "period_s": period, "wall_s": time.time() - t0}


out = {"sources": SRC, "P_kepler_s": P_KEP, "a_m": A_M, "e": E_M, "runs": {}}
for nm in RUNS:
    res = run(nm)
    out["runs"][nm] = res
    print(json.dumps(res), flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
