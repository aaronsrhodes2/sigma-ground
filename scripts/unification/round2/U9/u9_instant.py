"""U9 part A + B — Galilean boost of the instantaneous accelerations (design procedure, steps 1-3).

PRE-REGISTERED (before any run; design entry U9 of misc/unification/design_synthesis.json):
  design criterion  SEPARATE if Newtonian and SPH |da|/|a| <= 1e-12 while the 1PN da matches the analytic
                    O(V.v/c^2) form  (GM/r^2c^2)[-2(V.v) r^ + 4(r^.V) v + 4(r^.v) V]  to 10 %;
                    SAME if all three |da|/|a| lie within a factor 10 of each other.
  my prediction     (a) Newtonian da == 0 bit-exactly (the Newtonian layer reads positions only)  [BY CONSTRUCTION]
                    (b) the implemented 1PN term is a QUADRATIC polynomial in v, so at V = 30 km/s and
                        v ~ 40-59 km/s the first-order form misses the exact da by ~V/2v ~ 25-38 %
                        -> the design's 10 % first-order criterion is predicted to FAIL for in-plane boosts;
                        corrected criterion: engine da == first + second order (Mentat series, exact) to <= 1e-6,
                        and the first-order mismatch -> 0 linearly as V -> 0.
                    (c) SPH: only the artificial viscosity reads velocities, and only as v_i - v_j, so da is pure
                        roundoff ~ 2^-53 * V/|v_ij| * a_visc/|a|; with the program's standard cloud (internal
                        speeds ~0.05 m/s) and V = 1 km/s this is ~1e-12..1e-11 — the design's 1e-12 line may be
                        crossed by roundoff alone; corrected criterion: da/|a| proportional to V (roundoff scaling),
                        <= 10x the measured floor eps*V/|dv_min|, and identically 0 with alpha_visc = 0.
  sign probe        fit the engine's 1PN correction to K[(A GM/r - B v^2) n + C (n.v) v] (n = outward unit vector
                    from attractor to body). Textbook Schwarzschild/PPN harmonic gauge: (A,B,C) = (4,1,4) [DERIVED from
                    the PPN Lagrangian; LIT Wikipedia EIH test-particle limit]. From code reading I expect
                    gr_1pn = (-4,-1,-4) and eih_cross = (4,1,-4).
Usage: python -X utf8 u9_instant.py OUT.json
"""
import copy, json, math, os, sys, time
import numpy as np
import sympy as sp

sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles
from sigma_ground.field.interface.rolling_analysis import _ORBITAL
from sigma_ground.field.interface.surface import MATERIALS
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum
from sigma_ground.dynamics.continuum import (_sph_density_sum, _pressure_isothermal, _sph_pressure_force,
                                             _sph_viscous_force, _buoyancy_force)
from sigma_ground.dynamics.vec import Vec3

OUT = sys.argv[1] if len(sys.argv) > 1 else "u9_instant.json"
SRC = {}
t_start = time.time()


def leaf(tool, args, key):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


G = leaf("lookup_constant", {"name": "G"}, "G")
C = leaf("lookup_constant", {"name": "c"}, "c")
AU = leaf("lookup_constant", {"name": "au"}, "au")
sun = leaf("solar_system_body", {"body_name": "sun"}, "sun")
mer = leaf("solar_system_body", {"body_name": "mercury"}, "mercury")
M_S, R_S = sun["mass_kg"], sun["radius_m"]
M_M, R_M = mer["mass_kg"], mer["radius_m"]
A_M = mer["semimajor_axis_au"] * AU
E_M = _ORBITAL["Mercury"][1]
SRC["e_mercury"] = {"tool": "repo engine data", "value": E_M,
                    "source": "sigma_ground.field.interface.rolling_analysis._ORBITAL['Mercury'] (JPL Planetary Fact Sheets)"}
MU = G * (M_S + M_M)
C2 = C * C

# ── Mentat symbolic: exact boost expansion of the implemented gr_1pn bracket (nbody.py:463-477) ──
# bracket(v) = (4 mu/r - v.v) rh + 4 (rh.v) v, rh = code's unit vector (body -> attractor); v -> v + s V
sym_series = {}
syms = sp.symbols("mu r nx ny nz vx vy vz Vx Vy Vz s")
for k, comp in zip("xyz", ("vx+s*Vx", "vy+s*Vy", "vz+s*Vz")):
    nk = "n" + k
    expr = (f"(4*mu/r - ((vx+s*Vx)**2+(vy+s*Vy)**2+(vz+s*Vz)**2))*{nk} + "
            f"4*(nx*(vx+s*Vx)+ny*(vy+s*Vy)+nz*(vz+s*Vz))*({comp})")
    val, src = mentat("series_expansion", {"expression": expr, "variable": "s", "point": "0", "order": 4})
    SRC[f"series_{k}"] = {"tool": "series_expansion", "args": {"expression": expr, "variable": "s"},
                          "value": val, "source": src}
    poly = sp.Poly(sp.sympify(val, locals={n.name: n for n in syms}), syms[-1])
    c1 = poly.coeff_monomial(syms[-1])
    c2 = poly.coeff_monomial(syms[-1] ** 2)
    sym_series[k] = (sp.lambdify(syms[:-1], c1, "math"), sp.lambdify(syms[:-1], c2, "math"))
# design-literal first-order form, checked symbolically against Mentat's first-order coefficient
design_first = {"x": "-2*(Vx*vx+Vy*vy+Vz*vz)*nx + 4*(nx*Vx+ny*Vy+nz*Vz)*vx + 4*(nx*vx+ny*vy+nz*vz)*Vx"}
chk, chk_src = mentat("simplify_expr", {"expression": f"({design_first['x']}) - ("
                                       + str(sp.Poly(sp.sympify(SRC['series_x']['value'], locals={n.name: n for n in syms}), syms[-1]).coeff_monomial(syms[-1])) + ")"})
SRC["design_vs_series_first_order_x"] = {"tool": "simplify_expr", "value": chk, "source": chk_src}


def model_dbracket(rh, v, V, order):
    """first-order (order=1) or exact (order=2) change of the implemented bracket under v -> v+V."""
    out = np.zeros(3)
    for idx, k in enumerate("xyz"):
        f1, f2 = sym_series[k]
        args = (0.0, 1.0, rh[0], rh[1], rh[2], v[0], v[1], v[2], V[0], V[1], V[2])
        out[idx] = f1(*args) + (f2(*args) if order == 2 else 0.0)
    return out


# ── Mercury-like pair at true anomaly f (barycentric, zero momentum) ──
def pair_state(f_deg):
    f = math.radians(f_deg)
    p = A_M * (1 - E_M ** 2)
    r = p / (1 + E_M * math.cos(f))
    rvec = r * np.array([math.cos(f), math.sin(f), 0.0])
    vfac = math.sqrt(MU / p)
    vvec = vfac * np.array([-math.sin(f), E_M + math.cos(f), 0.0])
    xs, xm = -M_M / (M_S + M_M) * rvec, M_S / (M_S + M_M) * rvec
    vs, vm = -M_M / (M_S + M_M) * vvec, M_S / (M_S + M_M) * vvec
    return xs, vs, xm, vm


def bodies(xs, vs, xm, vm, V=np.zeros(3), m_test=None):
    return [CelestialBody(M_S, xs.copy(), vs + V, R_S, 0.0),
            CelestialBody(m_test if m_test else M_M, xm.copy(), vm + V, R_M, 0.0)]


TOG = {"newton": PhysicsToggles(), "gr_1pn": PhysicsToggles(gr_1pn=True), "eih_cross": PhysicsToggles(eih_cross=True)}


def acc(bl, name):
    return NBodySystem(bl, toggles=TOG[name]).compute_accelerations()


V0 = 3.0e4
rng = np.random.default_rng(9)
rdir = rng.normal(size=3); rdir /= np.linalg.norm(rdir)
DIRS = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0]), "rand": rdir}
PHASES = [0.0, 60.0, 180.0, 250.0]

rows = []
for fdeg in PHASES:
    xs, vs, xm, vm = pair_state(fdeg)
    for dname, dvec in DIRS.items():
        V = V0 * dvec
        row = {"f_deg": fdeg, "dir": dname, "V_m_s": V0}
        for name in TOG:
            a0 = acc(bodies(xs, vs, xm, vm), name)
            aV = acc(bodies(xs, vs, xm, vm, V), name)
            da = aV - a0
            row[name] = {"rel_da_mercury": float(np.linalg.norm(da[1]) / np.linalg.norm(a0[1])),
                         "rel_da_sun": float(np.linalg.norm(da[0]) / np.linalg.norm(a0[0])),
                         "bit_identical": bool(np.array_equal(aV, a0))}
            if name == "gr_1pn":
                m_out = {}
                for bi, (x_i, v_i, x_j, gm_j) in enumerate(((xs, vs, xm, G * M_M), (xm, vm, xs, G * M_S))):
                    rij = x_j - x_i; r = np.linalg.norm(rij); rh = rij / r
                    K = gm_j / (r * r * C2)
                    d1 = K * model_dbracket(rh, v_i, V, 1)
                    d2 = K * model_dbracket(rh, v_i, V, 2)
                    ne = np.linalg.norm(da[bi])
                    m_out["sun" if bi == 0 else "mercury"] = {
                        "mismatch_first_order_vector": float(np.linalg.norm(da[bi] - d1) / ne),
                        "mismatch_first_order_magnitude": float(abs(np.linalg.norm(d1) - ne) / ne),
                        "mismatch_exact": float(np.linalg.norm(da[bi] - d2) / ne),
                        "da_over_aN": float(ne / (gm_j / r ** 2)),
                        "four_Vv_over_c2": float(4 * V0 * np.linalg.norm(v_i) / C2)}
                row["gr_1pn_model"] = m_out
        rows.append(row)

# ── V scan (f = 60 deg, V along y and along x) ──
scan = {}
xs, vs, xm, vm = pair_state(60.0)
Vs = np.logspace(-1, math.log10(3e4), 22)
for dname in ("x", "y"):
    sc = []
    for Vm in Vs:
        V = Vm * DIRS[dname]
        a0n = acc(bodies(xs, vs, xm, vm), "newton"); aVn = acc(bodies(xs, vs, xm, vm, V), "newton")
        a0g = acc(bodies(xs, vs, xm, vm), "gr_1pn"); aVg = acc(bodies(xs, vs, xm, vm, V), "gr_1pn")
        a0e = acc(bodies(xs, vs, xm, vm), "eih_cross"); aVe = acc(bodies(xs, vs, xm, vm, V), "eih_cross")
        dg = aVg[1] - a0g[1]
        rij = xs - xm; r = np.linalg.norm(rij); rh = rij / r; K = G * M_S / (r * r * C2)
        d1 = K * model_dbracket(rh, vm, V, 1); d2 = K * model_dbracket(rh, vm, V, 2)
        sc.append({"V": float(Vm),
                   "newton": float(np.linalg.norm(aVn[1] - a0n[1]) / np.linalg.norm(a0n[1])),
                   "gr_1pn": float(np.linalg.norm(dg) / np.linalg.norm(a0g[1])),
                   "eih_cross": float(np.linalg.norm(aVe[1] - a0e[1]) / np.linalg.norm(a0e[1])),
                   "first_order_mismatch": float(np.linalg.norm(dg - d1) / np.linalg.norm(dg)),
                   "exact_mismatch": float(np.linalg.norm(dg - d2) / np.linalg.norm(dg))})
    scan[dname] = sc

# ── sign probe: fit (A,B,C) of the engine's 1PN correction, test body around a resting Sun ──
probe = {}
rng2 = np.random.default_rng(91)
for name in ("gr_1pn", "eih_cross"):
    Arows, b = [], []
    for k in range(12):
        r = A_M * (0.6 + 0.8 * rng2.random())
        th = 2 * math.pi * rng2.random()
        n = np.array([math.cos(th), math.sin(th), 0.0])
        vmag = math.sqrt(G * M_S / r) * (0.5 + rng2.random())
        ph = 2 * math.pi * rng2.random()
        v = vmag * np.array([math.cos(ph), math.sin(ph), 0.3 * (rng2.random() - 0.5)])
        bl = [CelestialBody(M_S, np.zeros(3), np.zeros(3), R_S, 0.0),
              CelestialBody(1.0, r * n, v, 1.0, 0.0)]          # 1 kg test body: exact test-particle limit
        corr = acc(bl, name)[1] - acc(bl, "newton")[1]
        K = G * M_S / (r * r * C2)
        cols = [K * (G * M_S / r) * n, -K * float(v @ v) * n, K * float(n @ v) * v]
        for comp in range(3):
            Arows.append([cols[0][comp], cols[1][comp], cols[2][comp]]); b.append(corr[comp])
    Am, bm = np.array(Arows), np.array(b)
    coef, *_ = np.linalg.lstsq(Am, bm, rcond=None)
    resid = float(np.linalg.norm(Am @ coef - bm) / np.linalg.norm(bm))
    probe[name] = {"A": float(coef[0]), "B": float(coef[1]), "C": float(coef[2]), "fit_rel_residual": resid,
                   "textbook": [4, 1, 4]}

# ── SPH cloud (program standard: M = 1e12 kg, R = 5 km, water ice, isothermal, h = 1.5 (V/N)^1/3) ──
N, MC, RC = 64, 1.0e12, 5000.0
rho_ice = MATERIALS["water_ice"]["density_kg_m3"]
SRC["rho_water_ice"] = {"tool": "repo engine data", "value": rho_ice,
                        "source": "sigma_ground.field.interface.surface.MATERIALS['water_ice']"}
cs = 0.4 * math.sqrt(G * MC / RC)
hS = 1.5 * (4 / 3 * math.pi * RC ** 3 / N) ** (1 / 3)
eps = 0.02 * RC
cloud = seed_rotating_cloud(N, cloud_mass_kg=MC, cloud_radius_m=RC, omega_rad_s=1.0e-5,
                            particle_density_kg_m3=rho_ice, jitter_frac=0.02, seed=20261010)


def sph_acc(Vb, alpha=1.0, parts=False):
    sc = to_continuum(cloud, material_key="water_ice", G=G, gravity_softening_m=eps, eos="isothermal",
                      sound_speed_m_s=cs, smoothing_h=hS)
    ps = sc.continuum_parcels()
    for p in ps:
        p.velocity = Vec3(p.velocity.x + Vb[0], p.velocity.y + Vb[1], p.velocity.z + Vb[2])
    grav = _sph_density_sum(ps, hS, gravity_G=G, softening=eps)
    for p in ps:
        _pressure_isothermal(p, cs)
    ap = _sph_pressure_force(ps, hS)
    av = _sph_viscous_force(ps, hS, alpha, c_override=cs)
    g0 = sc.gravity
    tot, visc = [], []
    for i, p in enumerate(ps):
        ab = _buoyancy_force(p, sc.rho_0, sc.T_0, g0)
        tot.append([g0.x + ap[i].x + av[i].x + ab.x + grav[i].x, g0.y + ap[i].y + av[i].y + ab.y + grav[i].y,
                    g0.z + ap[i].z + av[i].z + ab.z + grav[i].z])
        visc.append([av[i].x, av[i].y, av[i].z])
    if parts:
        dvmin = min(math.sqrt((ps[i].velocity.x - ps[j].velocity.x) ** 2 + (ps[i].velocity.y - ps[j].velocity.y) ** 2
                              + (ps[i].velocity.z - ps[j].velocity.z) ** 2)
                    for i in range(N) for j in ps[i].neighbors)
        nnb = sum(len(p.neighbors) for p in ps) / N
        return np.array(tot), np.array(visc), dvmin, nnb
    return np.array(tot), np.array(visc)


a0, v0s, dvmin, nnb = sph_acc(np.zeros(3), parts=True)
amax = float(np.max(np.linalg.norm(a0, axis=1)))
sph = {"N": N, "cloud_mass_kg": MC, "cloud_radius_m": RC, "h_m": hS, "softening_m": eps, "c_s_m_s": cs,
       "mean_neighbours": nnb, "min_pair_dv_m_s": dvmin, "max_|a|": amax,
       "visc_share_max": float(np.max(np.linalg.norm(v0s, axis=1)) / amax), "boost": {}}
for dname, dvec in DIRS.items():
    aV, _ = sph_acc(1.0e3 * dvec)
    d = np.linalg.norm(aV - a0, axis=1)
    sph["boost"][dname] = {"V_m_s": 1.0e3, "max_rel_da": float(d.max() / amax),
                           "median_rel_da": float(np.median(d / np.linalg.norm(a0, axis=1))),
                           "bit_identical": bool(np.array_equal(aV, a0))}
a0nv, _ = sph_acc(np.zeros(3), alpha=0.0)
aVnv, _ = sph_acc(1.0e3 * DIRS["rand"], alpha=0.0)
sph["no_viscosity_bit_identical"] = bool(np.array_equal(aVnv, a0nv))
sph_scan = []
for Vm in np.logspace(-3, 6, 19):
    aV, _ = sph_acc(Vm * DIRS["rand"])
    d = float(np.max(np.linalg.norm(aV - a0, axis=1)) / amax)
    sph_scan.append({"V": float(Vm), "max_rel_da": d,
                     "roundoff_floor_est": float(2.0 ** -53 * Vm / dvmin * sph["visc_share_max"])})
sph["scan_rand_dir"] = sph_scan

res = {"experiment": "U9 part A+B (instantaneous Galilean boost)", "sources": SRC,
       "choices": {"V_nbody_m_s": V0, "phases_deg": PHASES, "dirs": {k: v.tolist() for k, v in DIRS.items()},
                   "sph_cloud": "M=1e12 kg, R=5 km, N=64, omega=1e-5 rad/s, jitter 0.02, seed 20261010, "
                                "c_s = 0.4 sqrt(GM/R), h = 1.5 (V/N)^(1/3), eps = 0.02 R, alpha_visc = 1"},
       "nbody_rows": rows, "nbody_scan": scan, "sign_probe": probe, "sph": sph,
       "runtime_s": time.time() - t_start}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({"probe": probe, "sph_boost": sph["boost"], "no_visc": sph["no_viscosity_bit_identical"],
                  "visc_share": sph["visc_share_max"], "dvmin": dvmin, "nnb": nnb}, indent=1))
for r in rows:
    print(r["f_deg"], r["dir"], {k: (r[k]["rel_da_mercury"], r[k]["bit_identical"]) for k in TOG},
          r["gr_1pn_model"]["mercury"])
print("series simplify check:", chk)
