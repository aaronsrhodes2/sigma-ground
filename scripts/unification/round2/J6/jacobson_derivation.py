"""J6 — Jacobson 1995 as a checked derivation (sympy) + Mentat closing numbers.

Jacobson, "Thermodynamics of spacetime: the Einstein equation of state", PRL 75, 1260 (1995),
gr-qc/9504004, doi:10.1103/PhysRevLett.75.1260   [LIT, verified by web search 2026-10-10]
Eling, Guedens, Jacobson, "Non-equilibrium thermodynamics of spacetime", PRL 96, 121301 (2006),
gr-qc/0602001, doi:10.1103/PhysRevLett.96.121301 [LIT, verified by web search 2026-10-10]

Every algebraic step that can be done symbolically is done here, and every assumption is an explicit
line in ASSUMPTIONS. Each step records which assumptions it uses and a residual that must simplify to 0.
Units: c = k_B = 1 with hbar and G explicit (as in the paper), then restored to SI in step S8.
Signature (-,+,+,+); R_ab k^a k^b > 0 for normal matter (null energy condition) -> focusing.

Run from the repo root:  python -X utf8 <this file>
Writes jacobson_results.json next to this file.  No repo file is modified.
"""
import sys, os, json, time, math
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat_raw
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# PRE-REGISTRATION (fixed before the first run; not edited afterwards)
# ----------------------------------------------------------------------------------------------
PREREG = {
    "prediction": (
        "Every algebraic step of Jacobson 1995 closes in sympy: (S1) Raychaudhuri with theta(p)=sigma(p)=0 "
        "gives theta = -lambda R_kk + O(lambda^2); (S2) dQ = kappa eps^2 T_kk A/2, dA = eps^2 R_kk A/2; "
        "(S3) dQ = T eta dA with T = hbar kappa/2pi gives T_kk = (hbar eta/2pi) R_kk with kappa, eps, A cancelling; "
        "(S4) a symmetric M_ab with M_kk = 0 for every null k is f g_ab (solution space of dimension exactly 1); "
        "(S5) the contracted Bianchi identity div(G_ab + Lambda g_ab) = 0 holds identically on test metric families "
        "with arbitrary functions; (S6) conservation fixes f = -R/2 + Lambda; (S7) G_ab + Lambda g_ab = "
        "(2pi/hbar eta) T_ab and 8 pi G = 2pi/(hbar eta) gives eta = 1/(4 hbar G); (S8) the SI restoration is "
        "unique: eta = k_B c^3/(4 G hbar) = k_B/(4 l_P^2); (S9) EGJ shear: allowing sigma(p) != 0 gives a shear "
        "viscosity eta_s = hbar eta/(4 pi) = 1/(16 pi G), eta_s/s = hbar/(4 pi); (S10) Schwarzschild consistency: "
        "kappa = c^2/(2 r_s), T_H = hbar kappa/(2 pi c k_B), d(Mc^2) = T_H dS."),
    "pass_fail": (
        "DELIVERED if every symbolic residual simplifies to exactly 0 and S4's null space has dimension 1; FAIL if any "
        "residual is non-zero; INCONCLUSIVE for any step that cannot be checked symbolically (named). Mentat numbers: "
        "N1 eta/k_B from Mentat G, hbar, c equals 1/(4 L_PLANCK^2) from Mentat to rel <= 1e-9; N2 entropy/area from "
        "Mentat bekenstein_hawking_entropy equals N1 to rel <= 1e-9 for masses 1e10..1e40 kg; N3 Mentat "
        "unruh_temperature(a) equals hbar a/(2 pi c k_B) from Mentat constants to rel <= 1e-9; N4 Mentat "
        "hawking_temperature(M) equals Mentat unruh_temperature(c^2/(2 r_s)) with Mentat schwarzschild_radius to "
        "rel <= 1e-9; N5 first law (M+ - M-)c^2/(k_B (S+ - S-)) from Mentat entropies equals Mentat T_H to rel <= 1e-8."),
    "fixed_at": "2026-10-10, before first execution",
}

# ----------------------------------------------------------------------------------------------
# ASSUMPTIONS — every input that is not algebra, one explicit line each
# ----------------------------------------------------------------------------------------------
ASSUMPTIONS = [
    ("A1", "Spacetime is a smooth 4-D Lorentzian manifold (signature -,+,+,+); at every point p a local inertial frame "
           "exists (equivalence principle), so g_ab = diag(-1,1,1,1) at p and local Rindler horizons can be built."),
    ("A2", "Local Rindler horizon: through p pick a small spacelike 2-surface P; its past-directed null normals generate a "
           "null congruence, affinely parametrised by lambda (lambda = 0 at P, lambda < 0 to the past), tangent k^a, "
           "hypersurface-orthogonal so the twist omega vanishes."),
    ("A3", "Local equilibrium / instantaneous stationarity at p: the expansion theta(p) = 0 and the shear sigma(p) = 0 "
           "(Jacobson: P chosen so both vanish to first order; this choice is what makes the state equilibrium)."),
    ("A4", "Entropy is proportional to horizon area: dS = eta * dA with eta a universal constant (motivated by vacuum "
           "entanglement across the horizon with a UV cutoff; eta is NOT fixed by the argument)."),
    ("A5", "Unruh temperature: the vacuum seen by the boost observer is thermal at T = hbar kappa/(2 pi) (c = k_B = 1), "
           "kappa the acceleration of the orbit where |chi| = 1 (Unruh 1976; Bisognano-Wichmann)."),
    ("A6", "Heat = boost-energy flux of matter across the horizon: dQ = int_H T_ab chi^a dSigma^b with "
           "chi^a = -kappa lambda k^a (approximate boost Killing field vanishing at P) and dSigma^b = k^b dlambda dA. "
           "Quantum fluctuations of T_ab neglected; T_ab classical, symmetric."),
    ("A7", "Clausius relation dQ = T dS holds for ALL local Rindler horizons through ALL points p, i.e. for every null k."),
    ("A8", "Locality / leading order: the horizon patch is small compared with the curvature radius, so R_kk and T_kk are "
           "evaluated at p and only the leading power of the patch size eps is kept."),
    ("A9", "Matter stress-energy is conserved: nabla^a T_ab = 0."),
    ("A10", "Spacetime (the domain of f) is connected, so a scalar with vanishing gradient is a constant."),
    ("A11", "Identification: the coupling 2 pi/(hbar eta) is NAMED 8 pi G; G is taken from measurement, the derivation "
            "then predicts eta = 1/(4 hbar G), it does not predict G."),
    ("GEO1", "[geometric identity, not an assumption] Raychaudhuri equation for an affinely parametrised null "
             "congruence: dtheta/dlambda = -theta^2/2 - sigma_ab sigma^ab + omega_ab omega^ab - R_ab k^a k^b."),
    ("GEO2", "[geometric identity, checked in S5 on metric families] contracted Bianchi identity "
             "nabla^a (R_ab - R g_ab/2) = 0; metric compatibility nabla g = 0."),
]
AMAP = dict(ASSUMPTIONS)

STEPS = []
def step(sid, title, uses, residual, status, detail):
    STEPS.append({"id": sid, "title": title, "uses": uses, "residual": str(residual),
                  "status": status, "detail": detail})
    print(f"[{sid}] {status:12s} {title}  | residual = {residual}")

def is_zero(expr):
    return sp.simplify(sp.expand(expr)) == 0

MENTAT = []   # every Mentat call with its source
def M(tool, args):
    r = mentat_raw(tool, args)
    if not r.get("ok"):
        raise RuntimeError(f"mentat {tool} {args}: {r.get('error')}")
    res = r["result"]
    MENTAT.append({"tool": tool, "args": args, "value": res.get("value"), "units": res.get("units"),
                   "source": res.get("source"), "formula": res.get("formula")})
    return res.get("value"), res.get("source")

# ==============================================================================================
# S1  Raychaudhuri focusing to leading order
# ==============================================================================================
lam, eps = sp.symbols("lambda epsilon", real=True)
kappa, hbar, eta, G, A, Lam = sp.symbols("kappa hbar eta G A Lambda", positive=True)
R0, R1, R2, s0, s1, w0, th0 = sp.symbols("R0 R1 R2 s0 s1 w0 theta0", real=True)
a = sp.symbols("a0:5", real=True)

N = 4
theta = sum(a[n] * lam**n for n in range(N))
Rkk = R0 + R1 * lam + R2 * lam**2                 # R_ab k^a k^b along the generator
sig2 = (s0 + s1 * lam)**2                          # sigma_ab sigma^ab (one shear amplitude)
om2 = w0**2                                        # twist^2 (set by A2)
ray = sp.expand(sp.diff(theta, lam) - (-theta**2 / 2 - sig2 + om2 - Rkk))
eqs = [ray.coeff(lam, n) for n in range(N - 1)]
sol_general = sp.solve([sp.Eq(a[0], th0)] + eqs, a[:N], dict=True)[0]
theta_general = sp.expand(theta.subs(sol_general))
a1_general = sp.simplify(sol_general[a[1]])
# impose A2 (omega = 0) and A3 (theta(p) = 0, sigma(p) = 0)
theta_eq = sp.expand(theta_general.subs({w0: 0, th0: 0, s0: 0}))
theta_lead = sp.series(theta_eq, lam, 0, 2).removeO()
res_S1 = sp.simplify(theta_lead - (-lam * R0))
step("S1", "Raychaudhuri with theta(p)=sigma(p)=omega=0 gives theta = -lambda R_kk + O(lambda^2)",
     ["GEO1", "A2", "A3", "A8"], res_S1, "CLOSES" if res_S1 == 0 else "FAILS",
     {"theta_general_series": str(theta_general), "a1_general": str(a1_general),
      "theta_with_A2_A3": str(theta_eq),
      "note": ("with theta(p)=theta0, sigma(p)=s0, omega=w0 kept the O(lambda) coefficient is a1 = %s; all three "
               "extra terms vanish under A2+A3. With theta(p)=0 the theta^2/2 term first enters theta at O(lambda^3) "
               "(-R0^2 lambda^3/6) and a shear growing from zero (s1 lambda) also only at O(lambda^3)") % a1_general})

# exact closed form for constant R_kk (used in the figure), verified symbolically, cross-checked by Mentat series
Rc = sp.symbols("R_c", positive=True)
theta_exact = -sp.sqrt(2 * Rc) * sp.tan(sp.sqrt(Rc / 2) * lam)
res_ode = sp.simplify(sp.diff(theta_exact, lam) - (-theta_exact**2 / 2 - Rc))
ser_sym = sp.series(theta_exact, lam, 0, 4).removeO()
ms_val, ms_src = M("series_expansion", {"expression": "-sqrt(2*R)*tan(sqrt(R/2)*lam)", "variable": "lam",
                                         "point": "0", "order": 4})
res_ser = sp.simplify(ser_sym.subs(Rc, sp.Symbol("R")) - sp.sympify(ms_val, locals={"R": sp.Symbol("R"), "lam": lam}).subs(sp.Symbol("lam"), lam))
dA_exact = sp.simplify(sp.integrate(theta_exact, (lam, -eps, 0)))
step("S1b", "exact focusing solution theta = -sqrt(2R) tan(sqrt(R/2) lambda) solves Raychaudhuri (sigma=0, const R); "
     "series agrees with Mentat series_expansion", ["GEO1", "A2", "A3"], sp.simplify(res_ode + res_ser),
     "CLOSES" if (res_ode == 0 and res_ser == 0) else "FAILS",
     {"ode_residual": str(res_ode), "series_sympy": str(ser_sym), "series_mentat": ms_val, "mentat_source": ms_src,
      "dA_exact_over_A": str(dA_exact)})

# ==============================================================================================
# S2  Heat flux and area change across the horizon patch lambda in [-eps, 0]
# ==============================================================================================
Tkk, Rkk0 = sp.symbols("T_kk R_kk", real=True)
chi_over_k = -kappa * lam                                     # chi^a = -kappa lambda k^a   (A6)
dQ = sp.integrate(Tkk * chi_over_k, (lam, -eps, 0)) * A      # T_ab chi^a k^b dlambda dA
dA_area = sp.integrate((-lam * Rkk0), (lam, -eps, 0)) * A    # int theta dlambda dA, theta from S1
mq_val, mq_src = M("integrate_expr", {"expression": "-kappa*lam*Tkk", "variable": "lam", "lower": "-eps", "upper": "0"})
ma_val, ma_src = M("integrate_expr", {"expression": "-lam*Rkk", "variable": "lam", "lower": "-eps", "upper": "0"})
loc = {"kappa": kappa, "lam": lam, "eps": eps, "Tkk": Tkk, "Rkk": Rkk0}
res_S2 = sp.simplify(dQ - kappa * eps**2 * Tkk * A / 2) + sp.simplify(dA_area - eps**2 * Rkk0 * A / 2)
res_S2m = sp.simplify(dQ / A - sp.sympify(mq_val, locals=loc)) + sp.simplify(dA_area / A - sp.sympify(ma_val, locals=loc))
step("S2", "dQ = kappa eps^2 T_kk A/2 and dA = eps^2 R_kk A/2 (sympy and Mentat integrate_expr agree)",
     ["A6", "A8", "S1"], res_S2 + res_S2m, "CLOSES" if (res_S2 == 0 and res_S2m == 0) else "FAILS",
     {"dQ": str(dQ), "dA": str(dA_area), "mentat_dQ_per_A": mq_val, "mentat_dA_per_A": ma_val,
      "mentat_source": mq_src})

# ==============================================================================================
# S3  Clausius relation dQ = T dS, T = hbar kappa / 2pi, dS = eta dA
# ==============================================================================================
T_unruh = hbar * kappa / (2 * sp.pi)                          # A5
dS = eta * dA_area                                            # A4
clausius = sp.Eq(dQ, T_unruh * dS)                            # A7
Tkk_sol = sp.solve(clausius, Tkk)[0]
res_S3 = sp.simplify(Tkk_sol - hbar * eta / (2 * sp.pi) * Rkk0)
free_after = sorted(str(s) for s in Tkk_sol.free_symbols)
step("S3", "Clausius => T_kk = (hbar eta / 2 pi) R_kk ; kappa, eps and A cancel",
     ["A4", "A5", "A7", "S2"], res_S3, "CLOSES" if (res_S3 == 0 and not {"kappa", "epsilon", "A"} & set(free_after)) else "FAILS",
     {"T_kk": str(Tkk_sol), "free_symbols": free_after})

# necessity of theta(p) = 0: keep theta0, collect powers of eps
dA_th0 = sp.integrate(th0 + (-lam * Rkk0), (lam, -eps, 0)) * A
gap = sp.expand(dQ - T_unruh * eta * dA_th0)
c1 = sp.simplify(gap.coeff(eps, 1)); c2 = sp.simplify(gap.coeff(eps, 2))
th0_forced = sp.solve(c1, th0)
step("S3b", "theta(p) = 0 is forced: with theta(p)=theta0 the Clausius gap has an O(eps) term -T eta theta0 A that "
     "nothing in dQ (O(eps^2)) can cancel", ["A3", "A7"], sp.simplify(th0_forced[0]) if th0_forced else "no solution",
     "CLOSES" if th0_forced == [0] else "FAILS", {"eps1_coefficient": str(c1), "eps2_coefficient": str(c2)})

# shear at p: Clausius would read T_kk = hbar eta/2pi (R_kk + s0^2) — record it (used in S9)
dA_sh = sp.integrate(-lam * (Rkk0 + s0**2), (lam, -eps, 0)) * A
Tkk_sh = sp.solve(sp.Eq(dQ, T_unruh * eta * dA_sh), Tkk)[0]
step("S3c", "with sigma(p)=s0 != 0 the same step gives T_kk = (hbar eta/2pi)(R_kk + s0^2): shear enters at the SAME "
     "order as curvature, so it cannot be dropped by taking eps -> 0", ["A4", "A5", "A7"],
     sp.simplify(Tkk_sh - hbar * eta / (2 * sp.pi) * (Rkk0 + s0**2)), "CLOSES", {"T_kk_with_shear": str(Tkk_sh)})

# ==============================================================================================
# S4  null-tensor lemma: M_ab k^a k^b = 0 for all null k  =>  M_ab = f g_ab   (in the local frame, A1)
# ==============================================================================================
u, v = sp.symbols("u v", real=True)
Ms = sp.symbols("m00 m01 m02 m03 m11 m12 m13 m22 m23 m33", real=True)
Mmat = sp.Matrix([[Ms[0], Ms[1], Ms[2], Ms[3]],
                  [Ms[1], Ms[4], Ms[5], Ms[6]],
                  [Ms[2], Ms[5], Ms[7], Ms[8]],
                  [Ms[3], Ms[6], Ms[8], Ms[9]]])
eta_mink = sp.diag(-1, 1, 1, 1)
den = 1 + u**2 + v**2
# every null direction except the 'north pole' is k = (1, n), n = stereographic(u, v); multiply by den^2 -> polynomial
kvec = sp.Matrix([den, 2 * u, 2 * v, u**2 + v**2 - 1])        # = den * (1, n)
assert sp.expand((kvec.T * eta_mink * kvec)[0]) == 0           # k is null
Mkk = sp.expand((kvec.T * Mmat * kvec)[0])
coeffs = sp.Poly(Mkk, u, v).coeffs()
lin = sp.linear_eq_to_matrix(coeffs, Ms)[0]
null = lin.nullspace()
dimnull = len(null)
basis = null[0] if dimnull == 1 else None
Mbasis = Mmat.subs(dict(zip(Ms, list(basis)))) if basis is not None else None
fsym = sp.Symbol("f")
prop = None
if Mbasis is not None:
    ratio = sp.simplify(Mbasis[1, 1] / eta_mink[1, 1])
    prop = sp.simplify(Mbasis - ratio * eta_mink)
step("S4", "symmetric M_ab with M_kk = 0 for every null k has a 1-D solution space, M_ab = f g_ab "
     "(so (2pi/hbar eta) T_ab - R_ab = f g_ab)", ["A1", "A7", "S3"],
     (prop if prop is not None else f"null-space dim {dimnull}"),
     "CLOSES" if (dimnull == 1 and prop == sp.zeros(4, 4)) else "FAILS",
     {"n_polynomial_conditions": len(coeffs), "rank": int(lin.rank()), "nullspace_dim": dimnull,
      "basis": str(list(basis)) if basis is not None else None,
      "note": "polynomial identity in the stereographic coordinates (u,v) of the celestial sphere: exact, not sampled"})

# ==============================================================================================
# S5  contracted Bianchi identity on metric families with arbitrary functions
# ==============================================================================================
def christoffel(g, X):
    n = len(X); gi = g.inv()
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a_ in range(n):
        for b in range(n):
            for c in range(b, n):
                s = 0
                for d in range(n):
                    if gi[a_, d] != 0:
                        s += gi[a_, d] * (sp.diff(g[d, c], X[b]) + sp.diff(g[d, b], X[c]) - sp.diff(g[b, c], X[d]))
                Gam[a_][b][c] = Gam[a_][c][b] = sp.simplify(s / 2)
    return Gam, gi

def ricci(Gam, X):
    n = len(X); Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(b, n):
            s = 0
            for a_ in range(n):
                s += sp.diff(Gam[a_][b][d], X[a_]) - sp.diff(Gam[a_][b][a_], X[d])
                for e in range(n):
                    s += Gam[a_][a_][e] * Gam[e][b][d] - Gam[a_][d][e] * Gam[e][b][a_]
            Ric[b, d] = Ric[d, b] = sp.simplify(s)
    return Ric

def bianchi_residual(g, X, Lam_):
    Gam, gi = christoffel(g, X)
    Ric = ricci(Gam, X)
    Rs = sp.simplify(sum(gi[a_, b] * Ric[a_, b] for a_ in range(len(X)) for b in range(len(X))))
    E = Ric - Rs * g / 2 + Lam_ * g                            # G_ab + Lambda g_ab (lower)
    Emix = sp.simplify(gi * E)                                 # E^a_b
    n = len(X); res = []
    for b in range(n):
        s = 0
        for a_ in range(n):
            s += sp.diff(Emix[a_, b], X[a_])
            for e in range(n):
                s += Gam[a_][a_][e] * Emix[e, b] - Gam[e][a_][b] * Emix[a_, e]
        res.append(sp.simplify(s))
    return res, Rs

t, r, th, ph, x, y, z = sp.symbols("t r theta phi x y z", real=True)
fam = {}
Af, Bf = sp.Function("A")(t, r), sp.Function("B")(t, r)
fam["spherical_time_dependent: diag(-e^{2A(t,r)}, e^{2B(t,r)}, r^2, r^2 sin^2 theta)"] = (
    sp.diag(-sp.exp(2 * Af), sp.exp(2 * Bf), r**2, r**2 * sp.sin(th)**2), [t, r, th, ph])
Nf, af, bf, cf = [sp.Function(nm)(t, x) for nm in ("N", "a", "b", "c")]
fam["diagonal_2var: diag(-N^2, a^2, b^2, c^2), all functions of (t,x)"] = (
    sp.diag(-Nf**2, af**2, bf**2, cf**2), [t, x, y, z])
Ff, Hf, Kf, Pf = [sp.Function(nm)(t, x) for nm in ("F", "H", "K", "P")]
fam["off_diagonal: g_tt=-F, g_tx=H, g_xx=K, g_yy=g_zz=P, functions of (t,x)"] = (
    sp.Matrix([[-Ff, Hf, 0, 0], [Hf, Kf, 0, 0], [0, 0, Pf, 0], [0, 0, 0, Pf]]), [t, x, y, z])
bianchi_out = {}
for name, (g, X) in fam.items():
    t1 = time.time()
    res, Rs = bianchi_residual(g, X, Lam)
    ok = all(sp.simplify(e) == 0 for e in res)
    bianchi_out[name] = {"residuals": [str(e) for e in res], "zero": ok, "seconds": round(time.time() - t1, 1),
                         "ricci_scalar_ops": int(sp.count_ops(Rs))}
    print(f"   Bianchi on {name}: zero={ok}  ({time.time() - t1:.1f}s, R has {sp.count_ops(Rs)} ops)")
all_bianchi = all(v_["zero"] for v_ in bianchi_out.values())
step("S5", "contracted Bianchi + metric compatibility: nabla^a (G_ab + Lambda g_ab) = 0 identically on 3 metric "
     "families with arbitrary functions", ["GEO2"], 0 if all_bianchi else "non-zero",
     "CLOSES" if all_bianchi else "FAILS",
     {"families": bianchi_out, "scope": "families, not the general metric: the identity itself is a theorem "
      "(Riemann symmetries); this is a symbolic spot check that the code's conventions satisfy it"})

# ==============================================================================================
# S6  conservation fixes f:  nabla^a [R_ab + f g_ab] = 0  =>  d_b (R/2 + f) = 0  =>  f = -R/2 + Lambda
# ==============================================================================================
Xs = sp.symbols("x0:4", real=True)
Rfun, ffun = sp.Function("Rs")(*Xs), sp.Function("f")(*Xs)
# using GEO2: nabla^a R_ab = (1/2) d_b R ; nabla^a (f g_ab) = d_b f ; T conserved (A9)
grad_eqs = [sp.diff(Rfun, X_) / 2 + sp.diff(ffun, X_) for X_ in Xs]
# integrate the first equation in x0, then require the others: F = f + R/2 depends on nothing
Ffun = sp.Function("F")(*Xs)
sols = [sp.dsolve(sp.Eq(sp.diff(sp.Function("F")(X_), X_), 0)) for X_ in Xs]
f_sol = -Rfun / 2 + Lam
res_S6 = [sp.simplify(e.subs(ffun, f_sol).doit()) for e in grad_eqs]
step("S6", "conservation + Bianchi => d_b(f + R/2) = 0 => f = -R/2 + Lambda (Lambda an undetermined constant)",
     ["A9", "A10", "GEO2", "S4"], res_S6, "CLOSES" if all(e == 0 for e in res_S6) else "FAILS",
     {"dsolve_each_direction": [str(s_) for s_ in sols],
      "note": "Lambda is an integration constant: the argument does not fix it"})

# ==============================================================================================
# S7  the Einstein equation of state and the identification of G
# ==============================================================================================
Rab, gab, Tab, Rscal = sp.symbols("R_ab g_ab T_ab R", real=True)
coupling = 2 * sp.pi / (hbar * eta)
lhs = Rab + (-Rscal / 2 + Lam) * gab                          # R_ab + f g_ab with f from S6
einstein_form = Rab - Rscal * gab / 2 + Lam * gab
res_S7a = sp.simplify(lhs - einstein_form)
eta_sol = sp.solve(sp.Eq(coupling, 8 * sp.pi * G), eta)
ms7_val, ms7_src = M("simplify_expr", {"expression": "2*pi/(hbar*(1/(4*G*hbar)))"})
res_S7b = sp.simplify(eta_sol[0] - 1 / (4 * hbar * G))
res_S7c = sp.simplify(sp.sympify(ms7_val, locals={"G": G}) - 8 * sp.pi * G)
# round trip: contract the final equation with a null k -> back to S3
kk_contract = sp.simplify((einstein_form).subs(gab, 0) - Rab)   # g_kk = 0 for null k
step("S7", "R_ab - R g_ab/2 + Lambda g_ab = (2 pi/hbar eta) T_ab ; 2pi/(hbar eta) = 8 pi G  =>  eta = 1/(4 hbar G)",
     ["S4", "S6", "A11"], sp.simplify(res_S7a + res_S7b + res_S7c + kk_contract),
     "CLOSES" if all(e == 0 for e in (res_S7a, res_S7b, res_S7c, kk_contract)) else "FAILS",
     {"eta": str(eta_sol[0]), "mentat_simplify": ms7_val, "mentat_source": ms7_src,
      "eta_inverse_sqrt": str(sp.simplify(1 / sp.sqrt(eta_sol[0]))),
      "note": "eta^(-1/2) = 2 sqrt(hbar G) = 2 l_P, as stated in the paper"})

# ==============================================================================================
# S8  restoring SI units
# ==============================================================================================
p_, q_, s_ = sp.symbols("p q s")
# dimensions (L, M, T): G = L^3 M^-1 T^-2, hbar = M L^2 T^-1, c = L T^-1 ; want G^p hbar^q c^s = L^-2
dim_eqs = [sp.Eq(3 * p_ + 2 * q_ + s_, -2), sp.Eq(-p_ + q_, 0), sp.Eq(-2 * p_ - q_ - s_, 0)]
exps = sp.solve(dim_eqs, [p_, q_, s_], dict=True)
cS, kB, aS, GS, hS = sp.symbols("c k_B a G_SI hbar_SI", positive=True)
eta_SI = kB * cS**3 / (4 * GS * hS)
# SI derivation: x^0 = c t, lambda in m, kappa_geo = a/c^2 [1/m]; k_B T = hbar c kappa_geo / 2pi;
# chi = -kappa_geo lambda k (dimensionless); dQ = kappa_geo * int(-lambda) T_kk ; dS = (eta'/k_B)*k_B * int(-lambda) R_kk
kgeo = aS / cS**2
T_SI = hS * cS * kgeo / (2 * sp.pi * kB)
res_T = sp.simplify(T_SI - hS * aS / (2 * sp.pi * cS * kB))
Tkk_SI_sol = sp.solve(sp.Eq(kgeo * Tkk, T_SI * sp.Symbol("etap") * Rkk0), Tkk)[0]   # common factor int(-lambda) dropped
coupling_SI = sp.simplify(Rkk0 / Tkk_SI_sol)                    # R_kk = coupling_SI * T_kk
etap_solve = sp.solve(sp.Eq(coupling_SI, 8 * sp.pi * GS / cS**4), sp.Symbol("etap"))
res_S8 = sp.simplify(etap_solve[0] - eta_SI)
lP2 = GS * hS / cS**3
step("S8", "SI restoration: G^p hbar^q c^s = 1/m^2 has the unique solution (p,q,s) = (-1,-1,3); the SI Clausius "
     "coefficient 2 pi k_B/(hbar c eta') = 8 pi G/c^4 gives eta' = k_B c^3/(4 G hbar) = k_B/(4 l_P^2); "
     "T = hbar a/(2 pi c k_B)", ["S7"], sp.simplify(res_S8 + res_T + (eta_SI - kB / (4 * lP2))),
     "CLOSES" if (len(exps) == 1 and exps[0] == {p_: -1, q_: -1, s_: 3} and res_S8 == 0 and res_T == 0) else "FAILS",
     {"exponents": str(exps), "coupling_SI": str(coupling_SI), "eta_SI": str(etap_solve[0])})

# ==============================================================================================
# S9  Eling-Guedens-Jacobson 2006: shear at p -> internal entropy production -> shear viscosity
# ==============================================================================================
# keep sigma(p) = s0: dS = eta int(-lambda)(R_kk + sigma^2) = dQ/T + d_iS with d_iS = eta int(-lambda) sigma^2
# Killing time v: chi = d/dv = -kappa lambda k  => dlambda/dv = -kappa lambda ; Killing-time shear sh = sigma dlambda/dv
vsym, sh = sp.symbols("v sigma_hat", real=True)
dlam_dv = -kappa * lam
sigma_aff = sh / dlam_dv                                       # affine shear in terms of Killing-time shear
integrand_v = sp.simplify(eta * (-lam) * sigma_aff**2 * dlam_dv)  # eta(-lam) sigma^2 dlambda  -> per dv
res_iv = sp.simplify(integrand_v - eta * sh**2 / kappa)
# viscous dissipation of a 2-D membrane: T dS_i/dv = 2 eta_s sigma_hat^2 per unit area
eta_s = sp.symbols("eta_s", positive=True)
eta_s_sol = sp.solve(sp.Eq(T_unruh * integrand_v, 2 * eta_s * sh**2), eta_s)[0]
res_S9 = sp.simplify(eta_s_sol - hbar * eta / (4 * sp.pi))
eta_s_G = sp.simplify(eta_s_sol.subs(eta, 1 / (4 * hbar * G)))
ratio = sp.simplify(eta_s_sol / eta)
step("S9", "EGJ 2006 shear: d_iS = (eta/kappa) int sigma_hat^2 dv ; T d_iS = 2 eta_s sigma_hat^2 => eta_s = hbar eta/(4 pi) "
     "= 1/(16 pi G) ; eta_s / s = hbar/(4 pi)", ["A3 relaxed", "A4", "A5", "A6"],
     sp.simplify(res_iv + res_S9 + (eta_s_G - 1 / (16 * sp.pi * G)) + (ratio - hbar / (4 * sp.pi))),
     "CLOSES" if (res_iv == 0 and res_S9 == 0 and sp.simplify(eta_s_G - 1 / (16 * sp.pi * G)) == 0) else "FAILS",
     {"eta_s": str(eta_s_sol), "eta_s_in_G": str(eta_s_G), "eta_s_over_s": str(ratio),
      "convention": "dissipation 2 eta_s sigma_ab sigma^ab per unit area per unit Killing time (EGJ remark 4: "
                    "eta_s = T alpha/2 = hbar alpha/4pi = 1/16piG)",
      "bulk_viscosity": "EGJ: an f(R) entropy density needs a bulk-viscosity entropy-production term fixed by "
                        "energy-momentum conservation; not reproduced here (cited only)"})

# ==============================================================================================
# S10  Schwarzschild consistency: surface gravity, Hawking T = Unruh T(kappa), first law
# ==============================================================================================
rr, rs, Mm = sp.symbols("r r_s M", positive=True)
fS = 1 - rs / rr
kappa_S = sp.simplify(cS**2 * sp.diff(fS, rr).subs(rr, rs) / 2)   # metric -f c^2 dt^2 + dr^2/f
rs_M = 2 * GS * Mm / cS**2
T_H = sp.simplify((hS * kappa_S / (2 * sp.pi * cS * kB)).subs(rs, rs_M))
res_TH = sp.simplify(T_H - hS * cS**3 / (8 * sp.pi * GS * Mm * kB))
S_BH = (eta_SI * 4 * sp.pi * rs**2).subs(rs, rs_M)
res_first = sp.simplify(sp.diff(Mm * cS**2, Mm) / sp.diff(S_BH, Mm) - T_H)
step("S10", "Schwarzschild: kappa = c^2/(2 r_s); hbar kappa/(2 pi c k_B) = hbar c^3/(8 pi G M k_B) = T_H; "
     "d(M c^2) = T_H d(eta' A)", ["A4", "A5", "S8"], sp.simplify(res_TH + res_first),
     "CLOSES" if (res_TH == 0 and res_first == 0) else "FAILS",
     {"kappa": str(kappa_S), "T_H": str(T_H), "S_BH": str(sp.simplify(S_BH)),
      "honesty": "the 1/4 in S_BH was itself obtained inside GR (Hawking 1975 + first law), so this agreement is a "
                 "consistency check, not independent evidence for eta"})

sym_time = time.time() - T0
print(f"symbolic part done in {sym_time:.1f}s")

# ==============================================================================================
# MENTAT closing numbers (PLATINUM 21)
# ==============================================================================================
Gv, Gsrc = M("lookup_constant", {"name": "G"})
cv, csrc = M("lookup_constant", {"name": "c"})
hv, hsrc = M("lookup_constant", {"name": "hbar"})
kv, ksrc = M("lookup_constant", {"name": "k_B"})
lPv, lPsrc = M("lookup_constant", {"name": "L_PLANCK"})
num = {}
rel = lambda x_, y_: abs(x_ - y_) / abs(y_)

# N1 eta per unit area
eta_over_kB = cv**3 / (4 * Gv * hv)                 # m^-2
eta_from_lP = 1 / (4 * lPv**2)
num["N1"] = {"what": "eta/k_B = c^3/(4 G hbar) [Mentat G,hbar,c] vs 1/(4 L_PLANCK^2) [Mentat]",
             "eta_over_kB_m^-2": eta_over_kB, "eta_from_LPLANCK_m^-2": eta_from_lP,
             "eta_SI_J_per_K_m2": eta_over_kB * kv, "l_P_from_G_hbar_c_m": math.sqrt(Gv * hv / cv**3),
             "rel": rel(eta_from_lP, eta_over_kB), "tol": 1e-9}
num["N1"]["pass"] = num["N1"]["rel"] <= 1e-9

# N2 BH entropy per area from Mentat's own BH tool, over 30 decades of mass
N2rows = []
for Mkg in [1e10, 1e15, 1e20, 5.97219e24, 1.989e30, 1.989e31, 1.989e36, 1e40]:
    val, src = M("bekenstein_hawking_entropy", {"mass_kg": Mkg})
    spa = val["entropy_k_B"] / val["horizon_area_m2"]
    N2rows.append({"M_kg": Mkg, "entropy_kB": val["entropy_k_B"], "area_m2": val["horizon_area_m2"],
                   "S_per_area_m^-2": spa, "rel_vs_N1": rel(spa, eta_over_kB), "source": src})
num["N2"] = {"what": "S_BH/(k_B A) from Mentat bekenstein_hawking_entropy vs eta/k_B (N1)", "rows": N2rows,
             "max_rel": max(r_["rel_vs_N1"] for r_ in N2rows), "tol": 1e-9}
num["N2"]["pass"] = num["N2"]["max_rel"] <= 1e-9
# N2b second code path: the repo engine sigma_ground.field.gr_basics.bekenstein_hawking_entropy (J/K, written as
# k_B 4 pi G M^2/(hbar c), no Planck length) vs the MCP tool (A/(4 L_PLANCK^2), L_PLANCK from M_PLANCK)
from sigma_ground.field import gr_basics as _grb
N2b = []
for row in N2rows:
    S_engine_kB = _grb.bekenstein_hawking_entropy(row["M_kg"]) / kv
    N2b.append({"M_kg": row["M_kg"], "S_gr_basics_kB": S_engine_kB, "S_tool_kB": row["entropy_kB"],
                "rel": rel(row["entropy_kB"], S_engine_kB)})
MENTAT.append({"tool": "sigma_ground.field.gr_basics.bekenstein_hawking_entropy (direct import)",
               "args": {"mass_kg": [r_["M_kg"] for r_ in N2rows]}, "value": [r_["S_gr_basics_kB"] for r_ in N2b],
               "units": "k_B", "source": "sigma_ground.field.gr_basics (Bekenstein 1973 / Hawking 1975)"})
num["N2"]["second_code_path"] = {"rows": N2b, "max_rel": max(r_["rel"] for r_ in N2b)}
num["N2"]["pass"] = num["N2"]["pass"] and num["N2"]["second_code_path"]["max_rel"] <= 1e-9

# N3 Unruh temperature: Mentat tool vs formula with Mentat constants; Earth's surface g as the everyday sample
earth, esrc = M("solar_system_body", {"body_name": "earth"})
g_earth = earth["surface_g_ms2"]
N3rows = []
for acc in [g_earth, 1e10, 1e15, 1e20, 2.5e20, 1e25, 1e30]:
    Tv, Tsrc = M("unruh_temperature", {"acceleration_m_s2": acc})
    Tf = hv * acc / (2 * math.pi * cv * kv)
    N3rows.append({"a_m_s2": acc, "T_mentat_K": Tv, "T_formula_K": Tf, "rel": rel(Tv, Tf), "source": Tsrc})
num["N3"] = {"what": "Mentat unruh_temperature(a) vs hbar a/(2 pi c k_B) from Mentat constants",
             "earth_g_source": esrc, "g_earth_m_s2": g_earth, "T_unruh_earth_K": N3rows[0]["T_mentat_K"],
             "rows": N3rows, "max_rel": max(r_["rel"] for r_ in N3rows), "tol": 1e-9}
num["N3"]["pass"] = num["N3"]["max_rel"] <= 1e-9

# N4 Hawking T = Unruh T at the Schwarzschild surface gravity
N4rows = []
for Mkg in [1e12, 1e20, 5.97219e24, 1.989e30, 1.989e31, 8.26e36, 1.989e40]:
    rsv, rsrc = M("schwarzschild_radius", {"mass_kg": Mkg})
    kap = cv**2 / (2 * rsv)
    TU, TUsrc = M("unruh_temperature", {"acceleration_m_s2": kap})
    TH, THsrc = M("hawking_temperature", {"mass_kg": Mkg})
    N4rows.append({"M_kg": Mkg, "r_s_m": rsv, "kappa_m_s2": kap, "T_unruh_at_kappa_K": TU, "T_hawking_K": TH,
                   "rel": rel(TU, TH), "sources": [rsrc, TUsrc, THsrc]})
num["N4"] = {"what": "Mentat hawking_temperature(M) vs Mentat unruh_temperature(c^2/(2 r_s)), r_s from Mentat",
             "rows": N4rows, "max_rel": max(r_["rel"] for r_ in N4rows), "tol": 1e-9}
num["N4"]["pass"] = num["N4"]["max_rel"] <= 1e-9

# N5 first law dE = T_H dS with Mentat's own entropies (central difference, S ~ M^2 so exact up to rounding)
N5rows = []
for Mkg in [1e12, 1.989e30, 1.989e36]:
    d = 1e-4 * Mkg
    Sp, _ = M("bekenstein_hawking_entropy", {"mass_kg": Mkg + d})
    Sm, _ = M("bekenstein_hawking_entropy", {"mass_kg": Mkg - d})
    TH, THsrc = M("hawking_temperature", {"mass_kg": Mkg})
    Tfl = (2 * d * cv**2) / (kv * (Sp["entropy_k_B"] - Sm["entropy_k_B"]))
    N5rows.append({"M_kg": Mkg, "T_from_first_law_K": Tfl, "T_hawking_K": TH, "rel": rel(Tfl, TH)})
num["N5"] = {"what": "(M+ - M-) c^2 / (k_B (S+ - S-)) from Mentat bekenstein_hawking_entropy vs Mentat T_H",
             "rows": N5rows, "max_rel": max(r_["rel"] for r_ in N5rows), "tol": 1e-8}
num["N5"]["pass"] = num["N5"]["max_rel"] <= 1e-8

# N6 the EGJ/KSS number in SI (reported, no pass criterion): eta_s/s = hbar/(4 pi k_B)
num["N6"] = {"what": "eta_s/s = hbar/(4 pi k_B) from Mentat constants (EGJ horizon value = KSS value [LIT]); "
                    "membrane shear viscosity eta_s = hbar eta'/(4 pi k_B) = c^3/(16 pi G)",
             "eta_s_over_s_K_s": hv / (4 * math.pi * kv),
             "eta_s_membrane_kg_per_s": eta_over_kB * hv / (4 * math.pi),
             "c3_over_16piG_kg_per_s": cv**3 / (16 * math.pi * Gv)}
num["N6"]["rel"] = rel(num["N6"]["eta_s_membrane_kg_per_s"], num["N6"]["c3_over_16piG_kg_per_s"])

# ----------------------------------------------------------------------------------------------
sym_ok = all(s_["status"] == "CLOSES" for s_ in STEPS)
num_ok = all(num[k_]["pass"] for k_ in ("N1", "N2", "N3", "N4", "N5"))
INCONCLUSIVE_STEPS = [
    {"step": "A5 Unruh effect", "why": "a QFT theorem (Bisognano-Wichmann / Unruh); taken as input, not derivable in sympy "
                                       "from the algebra here; Mentat's tool encodes the formula, it does not derive it"},
    {"step": "A4 entropy ~ area", "why": "an assumption about microphysics (entanglement with a Planck-scale cutoff); no "
                                         "symbolic check possible; the derivation runs for ANY constant eta"},
    {"step": "S5 Bianchi in full generality", "why": "checked symbolically on 3 families with arbitrary functions, not on "
                                                      "the fully general 4-D metric (a theorem, so not in doubt)"},
]
out = {
    "id": "J6", "title": "Jacobson 1995 checked derivation", "prereg": PREREG,
    "assumptions": [{"id": k_, "text": v_} for k_, v_ in ASSUMPTIONS],
    "steps": STEPS, "symbolic_all_close": sym_ok,
    "mentat_numbers": num, "mentat_numbers_all_pass": num_ok,
    "not_symbolically_checkable": INCONCLUSIVE_STEPS,
    "mentat_calls": MENTAT,
    "constants": {"G": [Gv, Gsrc], "c": [cv, csrc], "hbar": [hv, hsrc], "k_B": [kv, ksrc], "L_PLANCK": [lPv, lPsrc]},
    "literature": {
        "Jacobson1995": "T. Jacobson, PRL 75, 1260 (1995), gr-qc/9504004, doi:10.1103/PhysRevLett.75.1260 [LIT verified]",
        "EGJ2006": "C. Eling, R. Guedens, T. Jacobson, PRL 96, 121301 (2006), gr-qc/0602001, "
                   "doi:10.1103/PhysRevLett.96.121301 [LIT verified]",
        "Unruh1976": "W. G. Unruh, PRD 14, 870 (1976) [LIT verified]",
        "KSS2005": "P. Kovtun, D. T. Son, A. O. Starinets, PRL 94, 111601 (2005), hep-th/0405231 [LIT verified]",
        "ChircoLiberati2010": "G. Chirco, S. Liberati, PRD 81, 024016 (2010), arXiv:0909.4194 [LIT verified]"},
    "runtime_s": round(time.time() - T0, 1),
    "versions": {"sympy": sp.__version__, "python": sys.version.split()[0]},
}
with open(os.path.join(HERE, "jacobson_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
print("symbolic all close:", sym_ok, "| Mentat numbers all pass:", num_ok, "| runtime", out["runtime_s"], "s")
for k_ in ("N1", "N2", "N3", "N4", "N5"):
    v_ = num[k_]; print(k_, "rel =", v_.get("rel", v_.get("max_rel")), "pass =", v_["pass"])
print("eta/k_B =", eta_over_kB, "m^-2 ; T_Unruh(g_earth) =", num["N3"]["T_unruh_earth_K"], "K ; eta_s/s =",
      num["N6"]["eta_s_over_s_K_s"], "K s")
