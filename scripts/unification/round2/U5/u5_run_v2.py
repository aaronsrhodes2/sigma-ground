"""U5 v2 worker (identical physics to u5_run.py; reads leaves_v2.json, writes runs_v2/): one Sun-Mercury Forest-Ruth run on the repo's own N-body engine.

usage (repo root as cwd):  python -X utf8 u5_run.py <outdir> <mode> <dt_s> <orbits>
mode: A  Newtonian
      B  include_gr=True (engine gr_1pn branch)
      C  Sun.sigma_field = XI * r_s / r   (field-equation normalisation), updated every step
      D  Sun.sigma_field = sigma_from_potential(r, M_sun) = XI * r_s / (2 r)  (inventory normalisation)
      E  toggles = PhysicsToggles(eih_cross=True)  (engine EIH branch)
Every physical number is read from leaves.json (Mentat values + sources) or from the engine itself.
No engine file is modified.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles
from sigma_ground.inventory.core.sigma import XI, sigma_from_potential

outdir, mode, dt, orbits = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
L = json.load(open(os.path.join(outdir, "leaves_v2.json")))
sun, mer = L["sun"]["value"], L["mercury"]["value"]
G, rs, a, e = L["G"]["value"], L["r_s_sun"]["value"], L["a_mercury_m"]["value"], L["e_mercury"]["value"]
M, m = sun["mass_kg"], mer["mass_kg"]
mu = G * (M + m)                       # two-body Newtonian mu for the LRL vector (unscaled)

rp = a * (1 - e)
vp = math.sqrt(G * M * (1 + e) / (a * (1 - e)))   # as in the design sketch
bodies = [CelestialBody(M, np.zeros(3), np.zeros(3), sun["radius_m"], 0.0),
          CelestialBody(m, np.array([rp, 0.0, 0.0]), np.array([0.0, vp, 0.0]), mer["radius_m"], 0.0)]
if mode == "E":
    S = NBodySystem(bodies, softening_m=0.0, toggles=PhysicsToggles(eih_cross=True))
else:
    S = NBodySystem(bodies, softening_m=0.0, include_gr=(mode == "B"))

def sig_of(rr):
    if mode == "C":
        return XI * rs / rr
    if mode == "D":
        return sigma_from_potential(rr, M)
    return 0.0

def rel_state():
    x = S.bodies[1].position_m - S.bodies[0].position_m
    v = S.bodies[1].velocity_m_s - S.bodies[0].velocity_m_s
    return x, v

def lrl_angle(x, v):
    h = np.cross(x, v)
    A = np.cross(v, h) - mu * x / np.linalg.norm(x)
    return math.atan2(A[1], A[0]), float(np.linalg.norm(A) / mu)

# force-level probe at t = 0 (perihelion): engine extra acceleration on Mercury relative to Newtonian
probe = None
if mode in ("B", "E", "C", "D"):
    if mode in ("C", "D"):
        S.bodies[0] = S.bodies[0].replace(sigma_field=sig_of(rp))
    a_mode = S.compute_accelerations()
    S0 = NBodySystem([b.replace(sigma_field=0.0) for b in S.bodies], softening_m=0.0)
    a_newt = S0.compute_accelerations()
    x, v = rel_state()
    da = (a_mode[1] - a_mode[0]) - (a_newt[1] - a_newt[0])       # relative-acceleration perturbation
    r = float(np.linalg.norm(x)); c = L["c"]["value"]
    if mode in ("B", "E"):
        # standard harmonic-gauge 1PN test-body EOM (x from Sun to body):  GM/(c^2 r^3) [ (4GM/r - v^2) x + 4 (x.v) v ]
        std = G * M / (c**2 * r**3) * ((4 * G * M / r - v @ v) * x + 4 * (x @ v) * v)
    else:
        # first-order sigma force: extra attraction G M sigma / r^2 toward the Sun
        std = -G * M * sig_of(r) * x / r**3
    probe = {"engine_da": da.tolist(), "reference_da": std.tolist(),
             "cos_angle": float(da @ std / (np.linalg.norm(da) * np.linalg.norm(std))),
             "magnitude_ratio": float(np.linalg.norm(da) / np.linalg.norm(std))}

peri_lrl, peri_sketch, peri_interp, peri_t, ecc = [], [], [], [], []
r_hist, th_hist, t_hist, st_hist = [], [], [], []   # last three steps
r2 = r1 = None
prev_state = None
n_steps = 0
t0 = time.time()
while len(peri_lrl) < orbits:
    x, v = rel_state(); rr = float(np.linalg.norm(x))
    if mode in ("C", "D"):
        S.bodies[0] = S.bodies[0].replace(sigma_field=sig_of(rr))
    th = math.atan2(x[1], x[0])
    if r2 is not None and r1 < r2 and r1 <= rr:
        # sketch estimator verbatim: atan2 of the CURRENT position (one step after the discrete minimum)
        peri_sketch.append(th)
        # primary: LRL direction at the discrete minimum step (previous state)
        ang, ec = lrl_angle(*prev_state)
        peri_lrl.append(ang); ecc.append(ec)
        # parabola through (t-2, t-1, t) for r; vertex time; quadratic interpolation of unwrapped angle
        tt = np.array(t_hist[-2:] + [S.time]); rv = np.array(r_hist[-2:] + [rr])
        thv = np.unwrap(np.array(th_hist[-2:] + [th]))
        tc = tt - tt[1]
        cr = np.polyfit(tc, rv, 2); ts = -cr[1] / (2 * cr[0])
        ct = np.polyfit(tc, thv, 2); peri_interp.append(float(np.polyval(ct, ts)))
        peri_t.append(float(tt[1] + ts))
    r2, r1 = r1, rr
    prev_state = (x.copy(), v.copy())
    r_hist = (r_hist + [rr])[-2:]; th_hist = (th_hist + [th])[-2:]; t_hist = (t_hist + [S.time])[-2:]
    S.forest_ruth_step(dt); n_steps += 1

def slope(y):
    y = np.unwrap(np.array(y)); k = np.arange(len(y))
    p, cov = np.polyfit(k, y, 1, cov=True)
    res = y - np.polyval(p, k)
    return float(p[0]), float(math.sqrt(cov[0, 0])), float(np.std(res))

res = {"mode": mode, "dt_s": dt, "orbits": orbits, "n_steps": n_steps, "wall_s": time.time() - t0,
       "lrl": slope(peri_lrl), "sketch": slope(peri_sketch), "interp": slope(peri_interp),
       "peri_lrl": peri_lrl, "peri_sketch": peri_sketch, "peri_interp": peri_interp, "peri_t": peri_t,
       "ecc_lrl": ecc, "anomalistic_period_s": float(np.polyfit(np.arange(len(peri_t)), peri_t, 1)[0]),
       "probe": probe, "final_sigma_sun": float(S.bodies[0].sigma_field)}
json.dump(res, open(os.path.join(outdir, "runs_v2", f"run_{mode}_dt{int(dt)}_n{orbits}.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("mode", "dt_s", "orbits", "n_steps", "wall_s", "lrl", "sketch", "interp", "anomalistic_period_s", "probe")}, indent=1))
