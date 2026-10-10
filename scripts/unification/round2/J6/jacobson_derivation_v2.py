"""J6 v2 — Jacobson 1995 as a checked derivation (sympy) + Mentat closing numbers.  REPAIRED after 3 skeptic reviews.

Jacobson, "Thermodynamics of spacetime: the Einstein equation of state", PRL 75, 1260 (1995),
gr-qc/9504004, doi:10.1103/PhysRevLett.75.1260   [LIT]
Eling, Guedens, Jacobson (EGJ), "Non-equilibrium thermodynamics of spacetime", PRL 96, 121301 (2006),
gr-qc/0602001, doi:10.1103/PhysRevLett.96.121301 [LIT]

What changed from v1 (each item answers a skeptic's required change; see "repairs" in the JSON):
  * every step's status is computed from its residual (v1 hard-coded S3c as "CLOSES");
  * S6 now SOLVES for the trace coefficient alpha in f = -alpha R + Lambda on a metric family (v1 substituted its answer);
  * S7 takes its coupling from S3's solved coefficient (v1 hard-coded 2 pi/(hbar eta));
  * S8 and S10 use the derived eta' and the same Unruh factor as S3, so a mutation propagates;
  * every step carries a tag ([VERIFIED numerics] / [BY CONSTRUCTION] / [LIT reproduced] / [DERIVED]) and lists its
    live checks and its by-construction parts separately;
  * a mutation harness (env J6_MUTATION = none | unruh_x2 | chi_sign) re-runs everything with a broken input so the
    reader can see which checks can fail;
  * diagnostic D1 (NOT a step of Jacobson 1995, outside the pre-registered count): the Clausius bookkeeping at the next
    order in the patch size, with Einstein's equation imposed along the generator;
  * every mass is fetched from Mentat or labelled as an arbitrary sample input.

Units: c = k_B = 1 with hbar and G explicit (as in the paper), then restored to SI in step S8.
Signature (-,+,+,+); R_ab k^a k^b > 0 for normal matter (null energy condition) -> focusing.
Run from the repo root:  python -X utf8 <this file>     No repo file is modified.
"""
import sys, os, json, time, math
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat_raw
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# MUTATION HARNESS (controls only; the pre-registered verdict is read from the unmutated run)
# ----------------------------------------------------------------------------------------------
MUT = os.environ.get("J6_MUTATION", "none")
assert MUT in ("none", "unruh_x2", "chi_sign"), MUT
UNRUH_DEN = sp.pi if MUT == "unruh_x2" else 2 * sp.pi      # T = hbar kappa / UNRUH_DEN   (A5; true value 2 pi)
CHI_SIGN = +1 if MUT == "chi_sign" else -1                  # chi^a = CHI_SIGN kappa lambda k^a (A6; true sign -1)
OUTNAME = "jacobson_results_v2.json" if MUT == "none" else f"jacobson_results_v2_mut_{MUT}.json"
print(f"=== J6 v2, mutation = {MUT} ===")

# ----------------------------------------------------------------------------------------------
# PRE-REGISTRATION (verbatim from v1, fixed 2026-10-10 before the first v1 run; NOT edited for v2)
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
    "fixed_at": "2026-10-10, before first execution of v1; carried into v2 unchanged",
    "scope_note_v2": ("The prediction concerns the steps of Jacobson 1995, which is a leading-order argument (A8). "
                      "Diagnostic D1 (next order) is not one of those steps and is reported outside the verdict."),
}

# ----------------------------------------------------------------------------------------------
# ASSUMPTIONS — every input that is not algebra, one explicit line each (v2 wording)
# ----------------------------------------------------------------------------------------------
ASSUMPTIONS = [
    ("A1", "[geometry input] Spacetime is a smooth 4-D Lorentzian manifold (signature -,+,+,+); at every point p a "
           "local inertial frame exists (equivalence principle), so g_ab = diag(-1,1,1,1) at p and local Rindler "
           "horizons can be built."),
    ("A2", "[construction] Local Rindler horizon: through p pick a small spacelike 2-surface P; its past-directed null "
           "normals generate a null congruence, affinely parametrised by lambda (lambda = 0 at p, lambda < 0 to the "
           "past), tangent k^a, hypersurface-orthogonal so the twist omega vanishes."),
    ("A3", "[construction, not a restriction on spacetime] P is chosen so that the expansion theta and the shear sigma "
           "of the inside congruence vanish at p. Jacobson 1995: 'It is always possible to choose P through p so that "
           "the expansion and shear vanish in a first order neighborhood of p.' Jacobson calls such a horizon "
           "'instantaneously stationary (in local equilibrium) at p'; EGJ 2006 remark 4 later withdrew equilibrium as "
           "the reason for sigma(p) = 0 ('This is not valid however, since the shear defined with respect to Killing "
           "parameter would in any case vanish')."),
    ("A4", "[physical assumption] Entropy is proportional to horizon area: dS = eta dA with eta a universal constant "
           "(motivated by vacuum entanglement across the horizon with a UV cutoff; eta is NOT fixed by the argument). "
           "With a non-constant entropy density (f(R)), EGJ show theta(p) != 0 is forced instead (S3b detail)."),
    ("A5", "[QFT input] Unruh temperature: the vacuum seen by the boost observer is thermal at T = hbar kappa/(2 pi) "
           "(c = k_B = 1), kappa the acceleration of the orbit where |chi| = 1 (Unruh 1976; Bisognano-Wichmann)."),
    ("A6", "[physical assumption] Heat = boost-energy flux of matter across the horizon: dQ = int_H T_ab chi^a dSigma^b "
           "with chi^a = -kappa lambda k^a (approximate boost Killing field vanishing at p) and dSigma^b = k^b dlambda "
           "dA. T_ab classical and symmetric; its quantum fluctuations are neglected. EGJ 2006 remark 5: the "
           "approximate Killing vector is ambiguous at O(x^3) while its lowest-order part is O(x), so the heat flux is "
           "fuzzy at relative order (patch size)^2/L_c^2."),
    ("A7", "[the postulate; carries the physics] The Clausius relation dQ = T dS holds as an EQUALITY (no internal "
           "entropy production) for every local Rindler horizon through every point p, i.e. for every null k, on "
           "horizons whose expansion and shear vanish at p (A3). theta(p) = 0 is also necessary for the equality "
           "(S3b, EGJ remark 4); sigma(p) = 0 is the condition for no internal entropy production (S3c, S9, EGJ "
           "remark 4)."),
    ("A8", "[approximation] Leading order: the horizon patch is small compared with the curvature radius; R_kk and "
           "T_kk are evaluated at p and the relation is required only at the leading power of the patch size eps. It "
           "does NOT hold at the next order even with Einstein's equation imposed (diagnostic D1)."),
    ("A9", "[physical assumption] Matter stress-energy is conserved: nabla^a T_ab = 0."),
    ("A10", "[topology] Spacetime (the domain of f) is connected, so a scalar with vanishing gradient is a constant."),
    ("A11", "[identification, not a prediction] The coupling 2 pi/(hbar eta) is NAMED 8 pi G with G taken from "
            "measurement; identifying the two gives eta = 1/(4 hbar G). eta (horizon entropy density) is not "
            "independently measured, so nothing is predicted."),
    ("GEO1", "[geometric identity] Raychaudhuri equation for an affinely parametrised null congruence: "
             "dtheta/dlambda = -theta^2/2 - sigma_ab sigma^ab + omega_ab omega^ab - R_ab k^a k^b."),
    ("GEO2", "[geometric identity, spot-checked in S5 and used live in S6] contracted Bianchi identity "
             "nabla^a (R_ab - R g_ab/2) = 0; metric compatibility nabla g = 0."),
]

STEPS = []
def step(sid, title, uses, residual, ok, detail, tag, live, by_construction):
    """status is ALWAYS computed from ok (a boolean built from residuals)."""
    status = "CLOSES" if ok else "FAILS"
    STEPS.append({"id": sid, "title": title, "uses": uses, "residual": str(residual), "status": status,
                  "tag": tag, "live_checks": live, "by_construction_parts": by_construction, "detail": detail})
    print(f"[{sid}] {status:7s} {tag:22s} {title[:90]}  | residual = {str(residual)[:120]}")

def z(expr):
    return sp.simplify(sp.expand(expr)) == 0

MENTAT = []   # every Mentat call with its source
def M(tool, args):
    r = mentat_raw(tool, args)
    if not r.get("ok"):
        raise RuntimeError(f"mentat {tool} {args}: {r.get('error')}")
    res = r["result"]
    MENTAT.append({"tool": tool, "args": args, "value": res.get("value"), "units": res.get("units"),
                   "source": res.get("source"), "formula": res.get("formula"), "kind": "mcp_tool_call"})
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
Rkk = R0 + R1 * lam + R2 * lam**2
sig2 = (s0 + s1 * lam)**2
om2 = w0**2
ray = sp.expand(sp.diff(theta, lam) - (-theta**2 / 2 - sig2 + om2 - Rkk))
eqs = [ray.coeff(lam, n) for n in range(N - 1)]
sol_general = sp.solve([sp.Eq(a[0], th0)] + eqs, a[:N], dict=True)[0]
theta_general = sp.expand(theta.subs(sol_general))
a1_general = sp.simplify(sol_general[a[1]])
theta_eq = sp.expand(theta_general.subs({w0: 0, th0: 0, s0: 0}))
theta_lead = sp.series(theta_eq, lam, 0, 2).removeO()
res_S1 = sp.simplify(theta_lead - (-lam * R0))
res_S1_a1 = sp.simplify(a1_general - (-R0 - s0**2 - th0**2 / 2 + w0**2))
step("S1", "Raychaudhuri with theta(p)=sigma(p)=omega=0 gives theta = -lambda R_kk + O(lambda^2)",
     ["GEO1", "A2", "A3", "A8"], res_S1, z(res_S1) and z(res_S1_a1),
     {"theta_general_series": str(theta_general), "a1_general": str(a1_general), "theta_with_A2_A3": str(theta_eq)},
     "[VERIFIED numerics]", ["order-by-order solution of Raychaudhuri (general a1 compared with the textbook form)"], [])

Rc = sp.symbols("R_c", positive=True)
theta_exact = -sp.sqrt(2 * Rc) * sp.tan(sp.sqrt(Rc / 2) * lam)
res_ode = sp.simplify(sp.diff(theta_exact, lam) - (-theta_exact**2 / 2 - Rc))
ser_sym = sp.series(theta_exact, lam, 0, 4).removeO()
ms_val, ms_src = M("series_expansion", {"expression": "-sqrt(2*R)*tan(sqrt(R/2)*lam)", "variable": "lam",
                                         "point": "0", "order": 4})
res_ser = sp.simplify(ser_sym.subs(Rc, sp.Symbol("R")) -
                      sp.sympify(ms_val, locals={"R": sp.Symbol("R"), "lam": lam}))
dA_exact = sp.simplify(sp.integrate(theta_exact, (lam, -eps, 0)))
dA_exact_ratio_series = sp.series(dA_exact / (Rc * eps**2 / 2), eps, 0, 4).removeO()
step("S1b", "exact focusing solution theta = -sqrt(2R) tan(sqrt(R/2) lambda) solves Raychaudhuri (sigma=0, const R); "
     "series agrees with Mentat series_expansion", ["GEO1", "A2", "A3"], sp.simplify(res_ode + res_ser),
     z(res_ode) and z(res_ser),
     {"ode_residual": str(res_ode), "series_sympy": str(ser_sym), "series_mentat": ms_val, "mentat_source": ms_src,
      "dA_exact_over_A": str(dA_exact), "dA_exact_over_leading_series": str(dA_exact_ratio_series)},
     "[VERIFIED numerics]", ["ODE residual of the closed form", "sympy series vs Mentat series_expansion"], [])

# ==============================================================================================
# S2  Heat flux and area change across the horizon patch lambda in [-eps, 0]
# ==============================================================================================
Tkk, Rkk0 = sp.symbols("T_kk R_kk", real=True)
chi_over_k = CHI_SIGN * kappa * lam                            # chi^a = -kappa lambda k^a   (A6)
dQ = sp.integrate(Tkk * chi_over_k, (lam, -eps, 0)) * A
dA_area = sp.integrate((-lam * Rkk0), (lam, -eps, 0)) * A
chi_str = f"{CHI_SIGN}*kappa*lam*Tkk"
mq_val, mq_src = M("integrate_expr", {"expression": chi_str, "variable": "lam", "lower": "-eps", "upper": "0"})
ma_val, ma_src = M("integrate_expr", {"expression": "-lam*Rkk", "variable": "lam", "lower": "-eps", "upper": "0"})
loc = {"kappa": kappa, "lam": lam, "eps": eps, "Tkk": Tkk, "Rkk": Rkk0}
r2a = sp.simplify(dQ - kappa * eps**2 * Tkk * A / 2); r2b = sp.simplify(dA_area - eps**2 * Rkk0 * A / 2)
r2m = sp.simplify(dQ / A - sp.sympify(mq_val, locals=loc)) + sp.simplify(dA_area / A - sp.sympify(ma_val, locals=loc))
step("S2", "dQ = kappa eps^2 T_kk A/2 and dA = eps^2 R_kk A/2 (sympy and Mentat integrate_expr agree)",
     ["A6", "A8", "S1"], r2a + r2b + r2m, z(r2a) and z(r2b) and z(r2m),
     {"dQ": str(dQ), "dA": str(dA_area), "mentat_dQ_per_A": mq_val, "mentat_dA_per_A": ma_val,
      "mentat_source": [mq_src, ma_src]},
     "[VERIFIED numerics]", ["dQ vs Jacobson's kappa eps^2 T_kk A/2", "dA vs eps^2 R_kk A/2",
                             "sympy vs Mentat integrate_expr"], [])

# ==============================================================================================
# S3  Clausius relation dQ = T dS, T = hbar kappa / 2pi, dS = eta dA
# ==============================================================================================
T_unruh = hbar * kappa / UNRUH_DEN                             # A5
dS = eta * dA_area                                             # A4
Tkk_sol = sp.solve(sp.Eq(dQ, T_unruh * dS), Tkk)[0]            # A7
res_S3 = sp.simplify(Tkk_sol - hbar * eta / (2 * sp.pi) * Rkk0)
free_after = sorted(str(s) for s in Tkk_sol.free_symbols)
step("S3", "Clausius => T_kk = (hbar eta / 2 pi) R_kk ; kappa, eps and A cancel",
     ["A4", "A5", "A7", "S2"], res_S3, z(res_S3) and not ({"kappa", "epsilon", "A"} & set(free_after)),
     {"T_kk": str(Tkk_sol), "free_symbols": free_after},
     "[VERIFIED numerics]", ["solved T_kk vs Jacobson's (hbar eta/2pi) R_kk", "kappa, eps, A absent"], [])

# S3b: theta(p) = theta0 kept, theta from the full Raychaudhuri series (theta0^2/2 term included), sigma = omega = 0
theta_th0 = sp.expand(theta_general.subs({w0: 0, s0: 0, s1: 0, R1: 0, R2: 0, R0: Rkk0}))
dA_th0 = sp.integrate(theta_th0, (lam, -eps, 0)) * A
gap = sp.expand(dQ - T_unruh * eta * dA_th0)
c1 = sp.factor(gap.coeff(eps, 1)); c2 = sp.factor(gap.coeff(eps, 2))
c2_expected = -A * kappa * (2 * Rkk0 * eta * hbar - 4 * sp.pi * Tkk + eta * hbar * th0**2) / (8 * sp.pi)
th0_forced = sp.solve(c1, th0)
# f(R)-type entropy density eta*f along the generator, f = 1 + f1 lambda: EGJ eq. (10) (theta f + fdot)(p) = 0
f1 = sp.symbols("f1", real=True)
dS_f = eta * sp.integrate(theta_th0 * (1 + f1 * lam) + f1, (lam, -eps, 0)) * A
c1_f = sp.expand(dQ - T_unruh * dS_f).coeff(eps, 1)
th0_forced_f = sp.solve(c1_f, th0)
ok_S3b = (th0_forced == [0]) and (th0_forced_f == [-f1])
step("S3b", "theta(p) = 0 is forced (constant eta): with theta(p)=theta0 the Clausius gap has an O(eps) term "
     "-T eta theta0 A that nothing in dQ (O(eps^2)) can cancel; with entropy density eta(1 + f1 lambda) the same step "
     "forces theta0 = -f1 instead (EGJ eq. 10)", ["A3", "A4", "A7"],
     sp.simplify(th0_forced[0]) if th0_forced else "no solution", ok_S3b,
     {"eps1_coefficient": str(c1),
      "eps2_coefficient_complete": str(c2),
      "eps2_coefficient_matches_full_form": bool(z(c2 - c2_expected)),
      "eps2_note": "complete through eps^2: includes the theta0^2/2 Raychaudhuri term (v1 stored a truncated form)",
      "f_R_variant_theta0_forced": str(th0_forced_f),
      "credit": "EGJ 2006 remark 4: 'We cannot allow for expansion at p, since without the derivative of f to balance "
                "the expansion, there would be a first order term in the entropy change not matched by the heat flux "
                "term.' This is a checked reproduction of that remark, not a new finding."},
     "[LIT reproduced]", ["O(eps) coefficient solved for theta0 (constant eta -> 0)",
                          "f(R)-type density -> theta0 = -fdot (EGJ eq. 10)"], [])

# S3c: shear at p; status computed from the residual (v1 hard-coded it)
dA_sh = sp.integrate(-lam * (Rkk0 + s0**2), (lam, -eps, 0)) * A
Tkk_sh = sp.solve(sp.Eq(dQ, T_unruh * eta * dA_sh), Tkk)[0]
res_S3c = sp.simplify(Tkk_sh - hbar * eta / (2 * sp.pi) * (Rkk0 + s0**2))
sa, sb = sp.symbols("s_a s_b", real=True)
two_P = sp.simplify(Tkk_sh.subs(s0, sa) - Tkk_sh.subs(s0, sb))           # same k, two surfaces P with different shear
res_twoP = sp.simplify(two_P - hbar * eta * (sa**2 - sb**2) / (2 * sp.pi))
step("S3c", "with sigma(p)=s0 != 0 the Clausius equality gives T_kk = (hbar eta/2pi)(R_kk + s0^2): shear enters at "
     "the SAME order as curvature; two surfaces P with the same k but different s0 cannot both satisfy it, so the "
     "equality can be postulated only on one class of P (sigma(p) = 0, A3/A7) or the shear term is booked as "
     "entropy production (S9)", ["A4", "A5", "A7"], res_S3c + res_twoP, z(res_S3c) and z(res_twoP),
     {"T_kk_with_shear": str(Tkk_sh), "two_P_same_k_difference": str(two_P),
      "credit": "EGJ 2006 remark 4: 'the d k^a part [of the Clausius relation] is satisfied only if the shear "
                "vanishes. However, nonzero shear at p is allowed if we include an internal entropy production term'. "
                "Checked reproduction; the two-P form of the argument is from the physics skeptic."},
     "[LIT reproduced]", ["solved T_kk with shear vs (hbar eta/2pi)(R_kk + s0^2)", "two-P difference"], [])

# ==============================================================================================
# S4  null-tensor lemma: M_ab k^a k^b = 0 for all null k  =>  M_ab = f g_ab   (in the local frame, A1)
# ==============================================================================================
u, v = sp.symbols("u v", real=True)
Ms = sp.symbols("m00 m01 m02 m03 m11 m12 m13 m22 m23 m33", real=True)
Mmat = sp.Matrix([[Ms[0], Ms[1], Ms[2], Ms[3]], [Ms[1], Ms[4], Ms[5], Ms[6]],
                  [Ms[2], Ms[5], Ms[7], Ms[8]], [Ms[3], Ms[6], Ms[8], Ms[9]]])
eta_mink = sp.diag(-1, 1, 1, 1)
def null_lemma(kvec, vars_):
    assert sp.expand((kvec.T * eta_mink * kvec)[0]) == 0
    Mkk = sp.expand((kvec.T * Mmat * kvec)[0])
    coeffs = sp.Poly(Mkk, *vars_).coeffs()
    lin = sp.linear_eq_to_matrix(coeffs, Ms)[0]
    return lin, lin.nullspace(), len(coeffs)
den = 1 + u**2 + v**2
lin, null, ncond = null_lemma(sp.Matrix([den, 2 * u, 2 * v, u**2 + v**2 - 1]), (u, v))
dimnull = len(null)
basis = null[0] if dimnull == 1 else None
prop = None
if basis is not None:
    Mbasis = Mmat.subs(dict(zip(Ms, list(basis))))
    prop = sp.simplify(Mbasis - sp.simplify(Mbasis[1, 1] / eta_mink[1, 1]) * eta_mink)
# negative control: null directions confined to one plane (v = 0) -> the lemma must NOT hold
lin_pl, null_pl, _ = null_lemma(sp.Matrix([1 + u**2, 2 * u, 0, u**2 - 1]), (u,))
step("S4", "symmetric M_ab with M_kk = 0 for every null k has a 1-D solution space, M_ab = f g_ab "
     "(so (2pi/hbar eta) T_ab - R_ab = f g_ab)", ["A1", "A7", "S3"],
     (prop if prop is not None else f"null-space dim {dimnull}"),
     dimnull == 1 and prop == sp.zeros(4, 4) and len(null_pl) > 1,
     {"n_polynomial_conditions": ncond, "rank": int(lin.rank()), "nullspace_dim": dimnull,
      "basis": str(list(basis)) if basis is not None else None,
      "negative_control_planar_null_k": {"rank": int(lin_pl.rank()), "nullspace_dim": len(null_pl),
                                         "reading": "with k confined to one plane the solution space is larger than "
                                                    "g_ab: the lemma needs ALL null directions (A7 'for every k')"},
      "note": "polynomial identity in the stereographic coordinates (u,v) of the celestial sphere: exact, not sampled"},
     "[VERIFIED numerics]", ["null-space dimension 1 with basis g_ab", "planar negative control has dimension > 1"], [])

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

def div_mixed(Emix, Gam, X):
    """nabla_a E^a_b for a mixed (1,1) tensor given as a matrix Emix[a, b]."""
    n = len(X); out = []
    for b in range(n):
        s = 0
        for a_ in range(n):
            s += sp.diff(Emix[a_, b], X[a_])
            for e in range(n):
                s += Gam[a_][a_][e] * Emix[e, b] - Gam[e][a_][b] * Emix[a_, e]
        out.append(s)
    return out

def geometry(g, X):
    Gam, gi = christoffel(g, X)
    Ric = ricci(Gam, X)
    Rs = sp.simplify(sum(gi[a_, b] * Ric[a_, b] for a_ in range(len(X)) for b in range(len(X))))
    return Gam, gi, Ric, Rs

t, r, th, ph, x, y, zc = sp.symbols("t r theta phi x y z", real=True)
fam = {}
Af, Bf = sp.Function("A")(t, r), sp.Function("B")(t, r)
FAM1 = "spherical_time_dependent: diag(-e^{2A(t,r)}, e^{2B(t,r)}, r^2, r^2 sin^2 theta)"
fam[FAM1] = (sp.diag(-sp.exp(2 * Af), sp.exp(2 * Bf), r**2, r**2 * sp.sin(th)**2), [t, r, th, ph])
Nf, af, bf, cf = [sp.Function(nm)(t, x) for nm in ("N", "a", "b", "c")]
fam["diagonal_2var: diag(-N^2, a^2, b^2, c^2), all functions of (t,x)"] = (
    sp.diag(-Nf**2, af**2, bf**2, cf**2), [t, x, y, zc])
Ff, Hf, Kf, Pf = [sp.Function(nm)(t, x) for nm in ("F", "H", "K", "P")]
fam["off_diagonal: g_tt=-F, g_tx=H, g_xx=K, g_yy=g_zz=P, functions of (t,x)"] = (
    sp.Matrix([[-Ff, Hf, 0, 0], [Hf, Kf, 0, 0], [0, 0, Pf, 0], [0, 0, 0, Pf]]), [t, x, y, zc])
bianchi_out = {}; GEOM = {}
for name, (g, X) in fam.items():
    t1 = time.time()
    Gam, gi, Ric, Rs = geometry(g, X)
    GEOM[name] = (g, X, Gam, gi, Ric, Rs)
    E = Ric - Rs * g / 2 + Lam * g
    res = [sp.simplify(e) for e in div_mixed(sp.simplify(gi * E), Gam, X)]
    ok = all(e == 0 for e in res)
    bianchi_out[name] = {"residuals": [str(e) for e in res], "zero": ok, "seconds": round(time.time() - t1, 1),
                         "ricci_scalar_ops": int(sp.count_ops(Rs))}
    print(f"   Bianchi on {name}: zero={ok}  ({time.time() - t1:.1f}s, R has {sp.count_ops(Rs)} ops)")
all_bianchi = all(v_["zero"] for v_ in bianchi_out.values())
step("S5", "contracted Bianchi + metric compatibility: nabla^a (G_ab + Lambda g_ab) = 0 identically on 3 metric "
     "families with arbitrary functions", ["GEO2"], 0 if all_bianchi else "non-zero", all_bianchi,
     {"families": bianchi_out, "scope": "families, not the general metric: the identity itself is a theorem "
      "(Riemann symmetries); this is a symbolic spot check that the code's conventions satisfy it"},
     "[VERIFIED numerics]", ["divergence computed from Christoffels/Ricci of each family"], [])

# ==============================================================================================
# S6  conservation fixes f.  v2: the trace coefficient is SOLVED, not substituted.
#   (i)  on FAM1, with F(t,r) an arbitrary function: nabla^a (R_ab + F g_ab) = d_b (F + R/2) identically
#   (ii) on FAM1: nabla^a (R_ab - alpha R g_ab) = 0 for all A(t,r), B(t,r)  =>  solve for alpha (expect 1/2, unique)
#   (iii) d_b(F + R/2) = 0 with spacetime connected (A10) => F = -R/2 + Lambda   [BY CONSTRUCTION / topology]
# ==============================================================================================
t6 = time.time()
g1, X1, Gam1, gi1, Ric1, Rs1 = GEOM[FAM1]
Fx = sp.Function("F")(t, r)
divRF = div_mixed(sp.simplify(gi1 * (Ric1 + Fx * g1)), Gam1, X1)
res_i = [sp.simplify(divRF[b] - sp.diff(Fx + Rs1 / 2, X1[b])) for b in range(4)]
alpha = sp.symbols("alpha")
divR = div_mixed(sp.simplify(gi1 * Ric1), Gam1, X1)
divRg = div_mixed(sp.simplify(gi1 * (Rs1 * g1)), Gam1, X1)
alpha_sols = {}
for b in range(4):
    c0 = sp.simplify(divR[b]); c1b = sp.simplify(divRg[b])
    if c1b != 0:
        alpha_sols[str(X1[b])] = str(sp.simplify(c0 / c1b))     # c0 - alpha c1b = 0  ->  alpha = c0/c1b
alpha_vals = set(alpha_sols.values())
alpha_S6 = sp.Rational(1, 2) if alpha_vals == {"1/2"} else None
Xs = sp.symbols("x0:4", real=True)
Rfun, ffun = sp.Function("Rs")(*Xs), sp.Function("f")(*Xs)
grad_eqs = [sp.diff(Rfun, X_) / 2 + sp.diff(ffun, X_) for X_ in Xs]
f_sol = -(alpha_S6 if alpha_S6 is not None else alpha) * Rfun + Lam
res_iii = [sp.simplify(e.subs(ffun, f_sol).doit()) for e in grad_eqs]
ok_S6 = all(e == 0 for e in res_i) and alpha_S6 == sp.Rational(1, 2) and len(alpha_sols) >= 1 \
    and all(e == 0 for e in res_iii)
step("S6", "conservation + Bianchi => nabla^a(R_ab + f g_ab) = d_b(f + R/2) = 0 => f = -R/2 + Lambda; the 1/2 is "
     "solved for on a metric family, Lambda is an undetermined constant", ["A9", "A10", "GEO2", "S4"],
     [str(e) for e in res_i] + [f"alpha={alpha_sols}"], ok_S6,
     {"family": FAM1, "div(R_ab + F g_ab) - d_b(F + R/2)": [str(e) for e in res_i],
      "alpha_solved_per_component": alpha_sols, "seconds": round(time.time() - t6, 1),
      "note": "Lambda is an integration constant: the argument does not fix it"},
     "[VERIFIED numerics] + [BY CONSTRUCTION]",
     ["div(R_ab + F g_ab) = d_b(F + R/2) with F an arbitrary function (family 1)",
      "alpha solved from div(R_ab - alpha R g_ab) = 0 (family 1)"],
     ["integration d_b(F + R/2) = 0 => F + R/2 = Lambda (A10, topology), checked by substitution"])

# ==============================================================================================
# S7  the Einstein equation of state; coupling taken from S3's solved coefficient (v1 hard-coded it)
# ==============================================================================================
Rab, gab, Tab, Rscal = sp.symbols("R_ab g_ab T_ab R", real=True)
coupling = sp.simplify(Rkk0 / Tkk_sol)                         # from S3:  R_kk = coupling * T_kk
lhs = Rab + (-(alpha_S6 if alpha_S6 is not None else alpha) * Rscal + Lam) * gab   # R_ab + f g_ab, f from S6
einstein_form = Rab - Rscal * gab / 2 + Lam * gab
res_S7a = sp.simplify(lhs - einstein_form)
eta_any = sp.Symbol("eta_any")                                  # no sign assumption, so a broken input shows up
eta_sol = [e_.subs(eta_any, eta) for e_ in sp.solve(sp.Eq(coupling.subs(eta, eta_any), 8 * sp.pi * G), eta_any)]
eta_sol = eta_sol or [sp.nan]                                    # A11: name the coupling 8 pi G
res_S7b = sp.simplify(eta_sol[0] - 1 / (4 * hbar * G))        # vs Jacobson's printed value [LIT]
ms7_val, ms7_src = M("simplify_expr", {"expression": f"2*pi/(hbar*({sp.sstr(eta_sol[0])}))"})
res_S7c = sp.simplify(sp.sympify(ms7_val, locals={"G": G, "hbar": hbar}) - 8 * sp.pi * G)
kk_contract = sp.simplify((einstein_form).subs(gab, 0) - Rab)   # g_kk = 0 for null k
step("S7", "R_ab - R g_ab/2 + Lambda g_ab = (coupling from S3) T_ab ; identifying the coupling with 8 pi G gives "
     "eta = 1/(4 hbar G)", ["S3", "S4", "S6", "A11"],
     sp.simplify(res_S7a + res_S7b + res_S7c + kk_contract),
     all(z(e) for e in (res_S7a, res_S7b, res_S7c, kk_contract)),
     {"coupling_from_S3": str(coupling), "eta": str(eta_sol[0]), "mentat_simplify_2pi_over_hbar_eta": ms7_val,
      "mentat_source": ms7_src, "eta_inverse_sqrt": str(sp.simplify(1 / sp.sqrt(eta_sol[0]))),
      "note": "eta^(-1/2) = 2 sqrt(hbar G) = 2 l_P, as stated in the paper; eta follows from an identification (A11)"},
     "[VERIFIED numerics] + [BY CONSTRUCTION]",
     ["eta solved from S3's coupling vs 1/(4 hbar G)", "Mentat simplify of 2pi/(hbar eta) at the solved eta"],
     ["Einstein form given S6's alpha (algebraic rearrangement)", "null contraction g_kk = 0 (round trip)",
      "naming the coupling 8 pi G (A11)"])

# ==============================================================================================
# S8  restoring SI units (same Unruh factor and chi sign as S3, so mutations propagate)
# ==============================================================================================
p_, q_, s_ = sp.symbols("p q s")
dim_eqs = [sp.Eq(3 * p_ + 2 * q_ + s_, -2), sp.Eq(-p_ + q_, 0), sp.Eq(-2 * p_ - q_ - s_, 0)]
exps = sp.solve(dim_eqs, [p_, q_, s_], dict=True)
cS, kB, aS, GS, hS = sp.symbols("c k_B a G_SI hbar_SI", positive=True)
etap = sp.Symbol("etap")
eta_SI_lit = kB * cS**3 / (4 * GS * hS)                       # Jacobson value restored [LIT]
kgeo = aS / cS**2
T_SI = hS * cS * kgeo / (UNRUH_DEN * kB)
res_T = sp.simplify(T_SI - hS * aS / (2 * sp.pi * cS * kB))
dQ_SI = sp.integrate(Tkk * CHI_SIGN * kgeo * lam, (lam, -eps, 0))
dS_SI = etap * sp.integrate(-lam * Rkk0, (lam, -eps, 0))
Tkk_SI_sol = sp.solve(sp.Eq(dQ_SI, T_SI * dS_SI), Tkk)[0]
coupling_SI = sp.simplify(Rkk0 / Tkk_SI_sol)
etap_solve = sp.solve(sp.Eq(coupling_SI, 8 * sp.pi * GS / cS**4), etap)
etap_derived = etap_solve[0] if etap_solve else sp.nan
res_S8 = sp.simplify(etap_derived - eta_SI_lit)
lP2 = GS * hS / cS**3
res_lP = sp.simplify(eta_SI_lit - kB / (4 * lP2))                 # identity by definition of l_P
step("S8", "SI restoration: G^p hbar^q c^s = 1/m^2 has the unique solution (p,q,s) = (-1,-1,3); the SI Clausius "
     "coefficient set to 8 pi G/c^4 gives eta' = k_B c^3/(4 G hbar) = k_B/(4 l_P^2); T = hbar a/(2 pi c k_B)",
     ["S7", "A5", "A6"], sp.simplify(res_S8 + res_T + res_lP),
     len(exps) == 1 and exps[0] == {p_: -1, q_: -1, s_: 3} and z(res_S8) and z(res_T) and z(res_lP),
     {"exponents": str(exps), "coupling_SI": str(coupling_SI), "eta_SI_derived": str(etap_derived)},
     "[VERIFIED numerics] + [BY CONSTRUCTION]",
     ["dimensional exponents unique", "eta' solved from the SI Clausius relation", "T_SI vs hbar a/(2 pi c k_B)"],
     ["eta' = k_B/(4 l_P^2) given l_P^2 = G hbar/c^3 (definition)"])

# ==============================================================================================
# S9  Eling-Guedens-Jacobson 2006: shear at p -> internal entropy production -> shear viscosity
# ==============================================================================================
sh = sp.symbols("sigma_hat", real=True)
dlam_dv = CHI_SIGN * kappa * lam                               # chi = d/dv
sigma_aff = sh / dlam_dv
integrand_v = sp.simplify(eta * (-lam) * sigma_aff**2 * dlam_dv)
res_iv = sp.simplify(integrand_v - eta * sh**2 / kappa)
eta_s = sp.symbols("eta_s")
eta_s_sol = sp.solve(sp.Eq(T_unruh * integrand_v, 2 * eta_s * sh**2), eta_s)[0]   # de Groot-Mazur: 2 eta_s sigma^2
res_S9 = sp.simplify(eta_s_sol - hbar * eta / (4 * sp.pi))
eta_s_G = sp.simplify(eta_s_sol.subs(eta, 1 / (4 * hbar * G)))
ratio = sp.simplify(eta_s_sol / eta)
ratio_free = sorted(str(s_) for s_ in ratio.free_symbols)
step("S9", "EGJ 2006 shear: d_iS = (eta/kappa) int sigma_hat^2 dv ; T d_iS = 2 eta_s sigma_hat^2 => eta_s = hbar eta/(4 pi) "
     "= 1/(16 pi G) ; eta_s / s = hbar/(4 pi), independent of eta and G", ["A3 relaxed", "A4", "A5", "A6"],
     sp.simplify(res_iv + res_S9 + (eta_s_G - 1 / (16 * sp.pi * G)) + (ratio - hbar / (4 * sp.pi))),
     z(res_iv) and z(res_S9) and z(eta_s_G - 1 / (16 * sp.pi * G)) and z(ratio - hbar / (4 * sp.pi))
     and ratio_free == ["hbar"],
     {"eta_s": str(eta_s_sol), "eta_s_in_G": str(eta_s_G), "eta_s_over_s": str(ratio),
      "eta_s_over_s_free_symbols": ratio_free,
      "convention_input": "dissipation 2 eta_s sigma_ab sigma^ab per unit area per unit Killing time (de Groot-Mazur, "
                          "EGJ ref. [9]); EGJ remark 4 states eta_s = T alpha/2 = hbar alpha/4pi = 1/16piG",
      "reading": "eta_s/s = hbar/4pi does not depend on eta or G: it is fixed by the boost temperature hbar/2pi and "
                 "the factor 2 in the dissipation convention. It equals the KSS number because KSS's number is itself "
                 "a horizon value (membrane paradigm); the equality is an identity, not contact with measured matter.",
      "bulk_viscosity": "EGJ: an f(R) entropy density needs a bulk-viscosity entropy-production term fixed by "
                        "energy-momentum conservation; not reproduced here (cited only)"},
     "[LIT reproduced] + [BY CONSTRUCTION]",
     ["Killing-time entropy production", "eta_s vs EGJ's hbar eta/4pi"],
     ["eta_s/s = hbar/4pi independent of eta and G (fixed by T = hbar kappa/2pi and the convention factor 2)"])

# ==============================================================================================
# S10  Schwarzschild consistency (first law uses the eta' DERIVED in S8)
# ==============================================================================================
rr, rs, Mm = sp.symbols("r r_s M", positive=True)
fS = 1 - rs / rr
kappa_S = sp.simplify(cS**2 * sp.diff(fS, rr).subs(rr, rs) / 2)
rs_M = 2 * GS * Mm / cS**2
T_H = sp.simplify((hS * kappa_S / (UNRUH_DEN * cS * kB)).subs(rs, rs_M))
res_TH = sp.simplify(T_H - hS * cS**3 / (8 * sp.pi * GS * Mm * kB))     # vs Hawking's formula [LIT]
S_BH = (etap_derived * 4 * sp.pi * rs**2).subs(rs, rs_M)
res_first = sp.simplify(sp.diff(Mm * cS**2, Mm) / sp.diff(S_BH, Mm) - T_H)
step("S10", "Schwarzschild: kappa = c^2/(2 r_s); hbar kappa/(2 pi c k_B) = hbar c^3/(8 pi G M k_B) = T_H; "
     "d(M c^2) = T_H d(eta' A) with eta' from S8", ["A4", "A5", "S8"], sp.simplify(res_TH + res_first),
     z(res_TH) and z(res_first),
     {"kappa": str(kappa_S), "T_H": str(T_H), "S_BH": str(sp.simplify(S_BH)),
      "first_law_residual": str(res_first), "hawking_formula_residual": str(res_TH),
      "honesty": "the 1/4 in S_BH was itself obtained inside GR (Hawking 1975 + first law), so this agreement is a "
                 "consistency check, not independent evidence for eta; the first law alone closes for ANY Unruh "
                 "factor (it is internal consistency), only the comparison with Hawking's formula can fail"},
     "[VERIFIED numerics]", ["kappa from the metric", "T_H vs Hawking's formula"],
     ["first law closes for any Unruh factor used consistently in S8 and S10"])

# ==============================================================================================
# D1  DIAGNOSTIC (not a step of Jacobson 1995; outside the pre-registered count): the next order in eps.
#     R_kk = R0 + R1 lambda, T_kk = T0 + T1 lambda along the generator, theta from Raychaudhuri (theta(p)=sigma=0).
# ==============================================================================================
T0s, T1s = sp.symbols("T0 T1", real=True)
th_D = sp.expand(theta_general.subs({w0: 0, th0: 0, s0: 0, s1: 0, R2: 0}))
dA_D = sp.integrate(th_D, (lam, -eps, 0)) * A
dQ_D = sp.integrate(CHI_SIGN * kappa * lam * (T0s + T1s * lam), (lam, -eps, 0)) * A
gap_D = sp.expand(dQ_D - T_unruh * eta * dA_D)
einstein_along = {T0s: hbar * eta * R0 / (2 * sp.pi), T1s: hbar * eta * R1 / (2 * sp.pi)}
gD2 = sp.factor(gap_D.coeff(eps, 2).subs(einstein_along))
gD3 = sp.factor(gap_D.coeff(eps, 3).subs(einstein_along))
gD3_expected = -A * R1 * eta * hbar * kappa / (12 * sp.pi)
T1_needed = sp.solve(gap_D.coeff(eps, 3), T1s)
# integrand form (EGJ match integrands at O(lambda)): heat integrand vs T x entropy integrand at O(lambda^2)
heat_int = CHI_SIGN * kappa * lam * (T0s + T1s * lam)
ent_int = T_unruh * eta * th_D
int_gap = sp.expand(heat_int - ent_int).subs(einstein_along)
int_gap_l1 = sp.factor(sp.expand(int_gap).coeff(lam, 1)); int_gap_l2 = sp.factor(sp.expand(int_gap).coeff(lam, 2))
dA_De = sp.expand(dA_D)
rel_dA_gradient = sp.simplify(dA_De.coeff(eps, 3) * eps / dA_De.coeff(eps, 2))
# for the figure: relative Clausius gaps, each isolating one neglected effect, Einstein imposed (R0 = 1 units)
g_, x_ = sp.symbols("g x", positive=True)       # g = R1/R0^(3/2), x = eps sqrt(R0)
lead_gap_grad = sp.simplify(((dQ_D - T_unruh * eta * sp.integrate(-R0 * lam - R1 * lam**2 / 2, (lam, -eps, 0)) * A)
                             / (T_unruh * eta * R0 * eps**2 * A / 2)).subs(einstein_along).subs({R1: g_, R0: 1})
                            .subs(eps, x_))
rel_gap_theta2 = sp.simplify((dA_exact / (Rc * eps**2 / 2) - 1).subs({Rc: 1, eps: x_}))
rel_gap_shear = sp.simplify((Tkk_sh / (hbar * eta * Rkk0 / (2 * sp.pi))) - 1)
D1 = {
    "what": "next-order Clausius bookkeeping with Einstein's equation imposed along the whole generator",
    "gap_eps2_with_Einstein": str(gD2), "gap_eps3_with_Einstein": str(gD3),
    "gap_eps3_matches_skeptic": bool(z(gD3 - gD3_expected)) if MUT == "none" else None,
    "T1_needed_for_eps3_closure": str(T1_needed),
    "integrand_gap_lambda1": str(int_gap_l1), "integrand_gap_lambda2": str(int_gap_l2),
    "relative_dA_correction_from_gradient": str(rel_dA_gradient),
    "relative_dA_correction_from_theta2": str(sp.series(dA_exact / (Rc * eps**2 / 2), eps, 0, 4).removeO() - 1),
    "killing_vector_fuzz": "relative O(eps^2/L_c^2) [LIT, EGJ 2006 remark 5]",
    "figure_expressions": {"gradient_rel_gap(x,g)": str(lead_gap_grad), "theta2_rel_gap(x)": str(rel_gap_theta2),
                           "shear_rel_gap(s0,R_kk)": str(rel_gap_shear)},
    "reading": ("At leading order the gap is 0 (that is S3). At the next order it is -A R1 eta hbar kappa eps^3/(12 pi) "
                "even with Einstein's equation imposed along the generator; exact Clausius at O(eps^3) would need "
                "T1 = hbar eta R1/(4 pi), contradicting Einstein. Jacobson 1995 and EGJ 2006 claim the relation only at "
                "lowest order ('to lowest order in lambda', EGJ eq. 6), so this limits the scope of the statement and "
                "is not a defect of the argument. The largest neglected term is the gradient one, relative size "
                "eps R1/(3 R0), first order in eps; theta^2/2 is eps^2 R/12 and the Killing-vector fuzz eps^2/L_c^2."),
    "tag": "[DERIVED] (sympy); credit: could-it-fail skeptic",
}
print("D1 gap eps^3 with Einstein:", gD3, "| T1 needed:", T1_needed, "| rel dA gradient:", rel_dA_gradient)

sym_time = time.time() - T0
print(f"symbolic part done in {sym_time:.1f}s")

# ==============================================================================================
# MENTAT closing numbers (PLATINUM 21). Every mass is fetched from Mentat or labelled an arbitrary sample input.
# ==============================================================================================
Gv, Gsrc = M("lookup_constant", {"name": "G"})
cv, csrc = M("lookup_constant", {"name": "c"})
hv, hsrc = M("lookup_constant", {"name": "hbar"})
kv, ksrc = M("lookup_constant", {"name": "k_B"})
lPv, lPsrc = M("lookup_constant", {"name": "L_PLANCK"})
MSUN_c, MSUN_c_src = M("lookup_constant", {"name": "M_SUN"})
sun, sun_src = M("solar_system_body", {"body_name": "sun"})
earth, esrc = M("solar_system_body", {"body_name": "earth"})
MSUN_b = sun["mass_kg"]; MEARTH = earth["mass_kg"]; g_earth = earth["surface_g_ms2"]
ARB = "arbitrary round sample input (no physical claim)"
MASS = {  # label -> (kg, provenance)
    "1e10 kg": (1e10, ARB), "1e12 kg": (1e12, ARB), "1e15 kg": (1e15, ARB), "1e20 kg": (1e20, ARB),
    "Earth": (MEARTH, f"Mentat solar_system_body earth ({esrc})"),
    "Sun (lookup_constant M_SUN)": (MSUN_c, f"Mentat lookup_constant M_SUN ({MSUN_c_src})"),
    "Sun (solar_system_body)": (MSUN_b, f"Mentat solar_system_body sun ({sun_src})"),
    "10 M_SUN": (10 * MSUN_c, "10 x Mentat lookup_constant M_SUN"),
    "1e6 M_SUN": (1e6 * MSUN_c, "1e6 x Mentat lookup_constant M_SUN"),
    "1e10 M_SUN": (1e10 * MSUN_c, "1e10 x Mentat lookup_constant M_SUN"),
    "1e40 kg": (1e40, ARB),
}
num = {}
rel = lambda x_, y_: abs(x_ - y_) / abs(y_)

eta_over_kB = cv**3 / (4 * Gv * hv)
eta_from_lP = 1 / (4 * lPv**2)
num["N1"] = {"what": "eta/k_B = c^3/(4 G hbar) [Mentat G,hbar,c] vs 1/(4 L_PLANCK^2) [Mentat]",
             "eta_over_kB_m^-2": eta_over_kB, "eta_from_LPLANCK_m^-2": eta_from_lP,
             "eta_SI_J_per_K_m2": eta_over_kB * kv, "l_P_from_G_hbar_c_m": math.sqrt(Gv * hv / cv**3),
             "rel": rel(eta_from_lP, eta_over_kB), "tol": 1e-9}
num["N1"]["pass"] = num["N1"]["rel"] <= 1e-9

N2rows = []
for lab in ["1e10 kg", "1e15 kg", "1e20 kg", "Earth", "Sun (lookup_constant M_SUN)", "Sun (solar_system_body)",
            "10 M_SUN", "1e6 M_SUN", "1e40 kg"]:
    Mkg, prov = MASS[lab]
    val, src = M("bekenstein_hawking_entropy", {"mass_kg": Mkg})
    spa = val["entropy_k_B"] / val["horizon_area_m2"]
    N2rows.append({"label": lab, "M_kg": Mkg, "mass_provenance": prov, "entropy_kB": val["entropy_k_B"],
                   "area_m2": val["horizon_area_m2"], "S_per_area_m^-2": spa, "rel_vs_N1": rel(spa, eta_over_kB),
                   "source": src})
num["N2"] = {"what": "S_BH/(k_B A) from Mentat bekenstein_hawking_entropy vs eta/k_B (N1)", "rows": N2rows,
             "max_rel": max(r_["rel_vs_N1"] for r_ in N2rows), "tol": 1e-9}
from sigma_ground.field import gr_basics as _grb
N2b = []
for row in N2rows:
    S_engine_kB = _grb.bekenstein_hawking_entropy(row["M_kg"]) / kv
    N2b.append({"M_kg": row["M_kg"], "S_gr_basics_kB": S_engine_kB, "S_tool_kB": row["entropy_kB"],
                "rel": rel(row["entropy_kB"], S_engine_kB)})
MENTAT.append({"tool": "sigma_ground.field.gr_basics.bekenstein_hawking_entropy", "kind": "direct_import_record",
               "args": {"mass_kg": [r_["M_kg"] for r_ in N2rows]}, "value": [r_["S_gr_basics_kB"] for r_ in N2b],
               "units": "k_B", "source": "sigma_ground.field.gr_basics (Bekenstein 1973 / Hawking 1975)"})
num["N2"]["second_code_path"] = {"rows": N2b, "max_rel": max(r_["rel"] for r_ in N2b),
                                 "note": "shares G, C, HBAR, K_B with the tool; differs only in algebraic form"}
num["N2"]["pass"] = num["N2"]["max_rel"] <= 1e-9 and num["N2"]["second_code_path"]["max_rel"] <= 1e-9

N3rows = []
for acc, prov in [(g_earth, f"Earth surface g, Mentat solar_system_body earth ({esrc})"), (1e10, ARB), (1e15, ARB),
                  (1e20, ARB), (2.5e20, ARB), (1e25, ARB), (1e30, ARB)]:
    Tv, Tsrc = M("unruh_temperature", {"acceleration_m_s2": acc})
    Tf = hv * acc / (2 * math.pi * cv * kv)
    N3rows.append({"a_m_s2": acc, "a_provenance": prov, "T_mentat_K": Tv, "T_formula_K": Tf, "rel": rel(Tv, Tf),
                   "source": Tsrc})
num["N3"] = {"what": "Mentat unruh_temperature(a) vs hbar a/(2 pi c k_B) from Mentat constants",
             "earth_g_source": esrc, "g_earth_m_s2": g_earth, "T_unruh_earth_K": N3rows[0]["T_mentat_K"],
             "rows": N3rows, "max_rel": max(r_["rel"] for r_ in N3rows), "tol": 1e-9}
num["N3"]["pass"] = num["N3"]["max_rel"] <= 1e-9

N4rows = []
for lab in ["1e12 kg", "1e20 kg", "Earth", "Sun (lookup_constant M_SUN)", "Sun (solar_system_body)", "10 M_SUN",
            "1e6 M_SUN", "1e10 M_SUN"]:
    Mkg, prov = MASS[lab]
    rsv, rsrc = M("schwarzschild_radius", {"mass_kg": Mkg})
    kap = cv**2 / (2 * rsv)
    TU, TUsrc = M("unruh_temperature", {"acceleration_m_s2": kap})
    TH, THsrc = M("hawking_temperature", {"mass_kg": Mkg})
    N4rows.append({"label": lab, "M_kg": Mkg, "mass_provenance": prov, "r_s_m": rsv, "kappa_m_s2": kap,
                   "T_unruh_at_kappa_K": TU, "T_hawking_K": TH, "rel": rel(TU, TH), "sources": [rsrc, TUsrc, THsrc]})
num["N4"] = {"what": "Mentat hawking_temperature(M) vs Mentat unruh_temperature(c^2/(2 r_s)), r_s from Mentat",
             "rows": N4rows, "max_rel": max(r_["rel"] for r_ in N4rows), "tol": 1e-9}
num["N4"]["pass"] = num["N4"]["max_rel"] <= 1e-9
TH_sun = {r_["label"]: r_["T_hawking_K"] for r_ in N4rows if r_["label"].startswith("Sun")}
num["solar_mass_note"] = {
    "M_SUN_lookup_constant_kg": [MSUN_c, MSUN_c_src], "M_sun_solar_system_body_kg": [MSUN_b, sun_src],
    "relative_difference": rel(MSUN_c, MSUN_b), "T_H_K": TH_sun,
    "reading": "Mentat holds two solar masses; T_H(1 M_sun) = 6.168e-8 K with M_SUN = 1.989e30 kg and 6.169e-8 K with "
               "solar_system_body 1.98892e30 kg; quoted as 6.17e-8 K"}

N5rows = []
for lab in ["1e12 kg", "Sun (lookup_constant M_SUN)", "Sun (solar_system_body)", "1e6 M_SUN"]:
    Mkg, prov = MASS[lab]
    d = 1e-4 * Mkg
    Sp, _ = M("bekenstein_hawking_entropy", {"mass_kg": Mkg + d})
    Sm, _ = M("bekenstein_hawking_entropy", {"mass_kg": Mkg - d})
    TH, THsrc = M("hawking_temperature", {"mass_kg": Mkg})
    Tfl = (2 * d * cv**2) / (kv * (Sp["entropy_k_B"] - Sm["entropy_k_B"]))
    N5rows.append({"label": lab, "M_kg": Mkg, "mass_provenance": prov, "T_from_first_law_K": Tfl, "T_hawking_K": TH,
                   "rel": rel(Tfl, TH)})
num["N5"] = {"what": "(M+ - M-) c^2 / (k_B (S+ - S-)) from Mentat bekenstein_hawking_entropy vs Mentat T_H",
             "rows": N5rows, "max_rel": max(r_["rel"] for r_ in N5rows), "tol": 1e-8}
num["N5"]["pass"] = num["N5"]["max_rel"] <= 1e-8

num["N6"] = {"what": "eta_s/s = hbar/(4 pi k_B) from Mentat constants; membrane eta_s = c^3/(16 pi G). [BY CONSTRUCTION]: "
                    "independent of eta and G; equals the KSS number because KSS's number is itself a horizon value",
             "eta_s_over_s_K_s": hv / (4 * math.pi * kv),
             "eta_s_membrane_kg_per_s": eta_over_kB * hv / (4 * math.pi),
             "c3_over_16piG_kg_per_s": cv**3 / (16 * math.pi * Gv)}
num["N6"]["rel"] = rel(num["N6"]["eta_s_membrane_kg_per_s"], num["N6"]["c3_over_16piG_kg_per_s"])

# ----------------------------------------------------------------------------------------------
sym_ok = all(s_["status"] == "CLOSES" for s_ in STEPS)
num_ok = all(num[k_]["pass"] for k_ in ("N1", "N2", "N3", "N4", "N5"))
NOT_SYMBOLIC = [
    {"step": "A5 Unruh effect", "why": "a QFT theorem (Bisognano-Wichmann / Unruh); taken as input; Mentat's tool "
                                       "encodes the formula, it does not derive it"},
    {"step": "A4 entropy ~ area", "why": "an assumption about microphysics; the derivation runs for ANY constant eta"},
    {"step": "S5 Bianchi in full generality", "why": "checked on 3 families with arbitrary functions, not on the fully "
                                                      "general 4-D metric (a theorem, so not in doubt)"},
]
ENGINE_FLAGS = [
    {"file": "sigma_ground/field/gr_basics.py", "line": 122,
     "flag": "citation: hawking_temperature docstring cites 'Hawking 1974, Comm. Math. Phys. 43, 199'; CMP 43, 199 is "
             "the 1975 paper (EGJ 2006 reference [3]: 'S. W. Hawking, Commun. Math. Phys. 43, 199 (1975)'). Same "
             "pairing in the module header at line 10. The MCP source strings 'Hawking 1974' (gr.py:100, "
             "frontier.py:56) propagate it. Citation hygiene, not a numerical defect."},
    {"file": "sigma_ground/mcp/tools/gr.py", "line": 89,
     "flag": "provenance label: hawking_temperature (lines 89-106) gives source 'sigma_ground.field.gr_basics "
             "(Hawking 1974)' but re-implements the formula inline (line 96) instead of calling gr_basics; value "
             "identical. Label issue, not a value defect."},
    {"file": "sigma_ground/field/constants.py", "line": 427,
     "flag": f"two solar masses: lookup_constant M_SUN = {MSUN_c} kg (constants.py:427) vs solar_system_body sun = "
             f"{MSUN_b} kg (materia/bodies.py:46, source label '{sun_src}'); relative difference "
             f"{rel(MSUN_c, MSUN_b):.3e}. Two sources of truth for one quantity."},
    {"file": "sigma_ground/mcp/tools/frontier.py", "line": 24,
     "flag": "hygiene: _consts() hard-codes c = 299792458.0 instead of importing constants.C (value identical)."},
    {"file": "sigma_ground/field/constants.py", "line": 322,
     "flag": "naming hazard: ETA = 0.412 (DESI cosmic entanglement fraction) collides with the standard symbol for "
             "Jacobson's entropy density."},
]
n_mcp = sum(1 for c_ in MENTAT if c_.get("kind") == "mcp_tool_call")
n_direct = sum(1 for c_ in MENTAT if c_.get("kind") == "direct_import_record")
out = {
    "id": "J6", "version": "v2 (repair)", "mutation": MUT, "title": "Jacobson 1995 checked derivation",
    "prereg": PREREG,
    "assumptions": [{"id": k_, "text": v_} for k_, v_ in ASSUMPTIONS],
    "steps": STEPS, "symbolic_all_close": sym_ok, "n_steps": len(STEPS),
    "n_steps_close": sum(1 for s_ in STEPS if s_["status"] == "CLOSES"),
    "diagnostic_D1_next_order": D1,
    "mentat_numbers": num, "mentat_numbers_all_pass": num_ok,
    "not_symbolically_checkable": NOT_SYMBOLIC, "engine_flags": ENGINE_FLAGS,
    "mentat_calls": MENTAT, "mentat_call_counts": {"mcp_tool_calls": n_mcp, "direct_import_records": n_direct,
                                                   "total_records": len(MENTAT)},
    "constants": {"G": [Gv, Gsrc], "c": [cv, csrc], "hbar": [hv, hsrc], "k_B": [kv, ksrc], "L_PLANCK": [lPv, lPsrc],
                  "M_SUN": [MSUN_c, MSUN_c_src], "M_sun_body": [MSUN_b, sun_src], "M_earth": [MEARTH, esrc],
                  "g_earth": [g_earth, esrc]},
    "literature": {
        "Jacobson1995": "T. Jacobson, PRL 75, 1260 (1995), gr-qc/9504004, doi:10.1103/PhysRevLett.75.1260",
        "EGJ2006": "C. Eling, R. Guedens, T. Jacobson, PRL 96, 121301 (2006), gr-qc/0602001, "
                   "doi:10.1103/PhysRevLett.96.121301 (remarks 4 and 5 quoted from the arXiv text)",
        "Unruh1976": "W. G. Unruh, PRD 14, 870 (1976)",
        "KSS2005": "P. Kovtun, D. T. Son, A. O. Starinets, PRL 94, 111601 (2005), hep-th/0405231",
        "ChircoLiberati2010": "G. Chirco, S. Liberati, PRD 81, 024016 (2010), arXiv:0909.4194",
        "Hawking1975": "S. W. Hawking, Commun. Math. Phys. 43, 199 (1975) (as cited in EGJ 2006 ref. [3])"},
    "runtime_s": round(time.time() - T0, 1), "symbolic_runtime_s": round(sym_time, 1),
    "versions": {"sympy": sp.__version__, "python": sys.version.split()[0]},
}
with open(os.path.join(HERE, OUTNAME), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"[{MUT}] steps closing {out['n_steps_close']}/{out['n_steps']} | Mentat numbers all pass: {num_ok} | "
      f"runtime {out['runtime_s']} s | Mentat records {n_mcp}+{n_direct}")
for k_ in ("N1", "N2", "N3", "N4", "N5"):
    v_ = num[k_]; print(k_, "rel =", v_.get("rel", v_.get("max_rel")), "pass =", v_["pass"])
print("eta/k_B =", eta_over_kB, "m^-2 ; T_Unruh(g_earth) =", num["N3"]["T_unruh_earth_K"], "K ; T_H(sun) =", TH_sun)
