"""A2 v2 (repair round) — sigma action options for the Captain's decisions C1-C6 (misc/MENTAT_GAPS_UNIFICATION.md).

v2 repairs (three skeptic reviews): B5/B6/D2/E3/F2 retagged [BY CONSTRUCTION] and B5/B6 shown to be tautologies;
B4m (light is GR also for a massive sigma); C3 (ghost test around a background sigma0: O4 is ghost-free only near 0);
G normalisation (bare vs observed) for every option; O5 per-test sources; Earth-Moon composition indicator for LLR;
Eot-Wash finite-range indicator; O7 per-bound values; check kinds (which checks could fail).
The runner's file a2_sigma_action.py is kept unchanged.

Every variation and sign check the memo relies on, in sympy, plus the per-option numbers against the
section-6 bounds of misc/UNIFICATION_PROGRAM.md (leaf values from Mentat, f_i from the repo's own sigma
mass decomposition, bounds [LIT] with verified citations).

Conventions (stated once, used everywhere):
  signature (-,+,+,+); Riemann R^a_bcd = d_c G^a_db - d_d G^a_cb + G^a_ce G^e_db - G^a_de G^e_cb; R_bd = R^a_bad (MTW).
  Einstein-Hilbert action +(M/2) sqrt(-g) R with M = c^4/(8 pi G); c = 1 in the symbolic part.
  Weak field: ds^2 = -(1 + 2 Phi) dt^2 + (1 - 2 Psi) dx.dx ; in GR Phi = Psi = -G M / r.
  Dust (rest-mass density rho, at rest): L_m = -rho (1 + f sigma) sqrt(1 + 2 Phi)   (point particle -m(sigma) dtau).

The option family (Jordan frame for leptons and photons):
  L = sqrt(-g) [ (M/2)(1 + 2 zeta sigma) R  -  (M/2) kappa g^{mn} d_m sigma d_n sigma  -  (M/2) m2 sigma^2 ]  +  L_m[g; m_i (1 + f_i sigma)]
  zeta = 0            -> "sigma T" coupling only (the mass response is the only coupling; Einstein frame)
  f = 0               -> "sigma R" coupling only (matter minimal in the Jordan frame)
  both nonzero        -> mixed
  kappa sign          -> kinetic sign (the ghost test is on kappa_E = kappa + 6 zeta^2, derived below)
  m2                  -> massless (0) or massive (range lambda)
  flat box            -> replace g^{mn} sqrt(-g) by eta^{mn} in the kinetic term (checked at 1PN in part D)

Run from the repo root:  python -X utf8 -B <this file>
"""
import json, math, os, sys, time

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"D:/Aaron/development/sigma-ground"
T0 = time.time()

CHECKS = []


def check(name, expr, tag, claim, note=""):
    """Record a symbolic identity: passes iff expr simplifies to exactly 0."""
    r = sp.simplify(sp.expand(expr))
    ok = bool(r == 0)
    CHECKS.append(dict(name=name, claim=claim, residual=str(r), passed=ok, tag=tag, note=note))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "   residual = " + str(r)))
    return ok


# ════════════════════════════════════════════════════════════════════════════════════════
# Part A — geometry
# ════════════════════════════════════════════════════════════════════════════════════════

def christoffel(g, ginv, X):
    n = len(X)
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                s = 0
                for d in range(n):
                    if ginv[a, d] != 0:
                        s += ginv[a, d] * (sp.diff(g[d, c], X[b]) + sp.diff(g[d, b], X[c]) - sp.diff(g[b, c], X[d]))
                G[a][b][c] = s / 2
    return G


def ricci_scalar(g, X):
    n = len(X)
    ginv = sp.diag(*[1 / g[i, i] for i in range(n)])          # diagonal metrics only
    G = christoffel(g, ginv, X)
    R = 0
    for b in range(n):
        d = b                                                    # g^{bd} diagonal
        Rbd = 0
        for a in range(n):
            Rbd += sp.diff(G[a][d][b], X[a]) - sp.diff(G[a][a][b], X[d])
            for e in range(n):
                Rbd += G[a][a][e] * G[e][d][b] - G[a][d][e] * G[e][a][b]
        R += ginv[b, b] * Rbd
    return R


def order2(expr, eps):
    """Coefficient of eps^2 in the Taylor expansion of expr."""
    return sp.diff(expr, eps, 2).subs(eps, 0) / 2


def EL(L, funcs, X):
    """Euler-Lagrange expressions (dL/df - d_i dL/d(d_i f) + d_ij dL/d(d_ij f)), one per field."""
    # explicit variational derivative (sympy's euler_equations silently drops identically-zero equations):
    #   E_F = dL/dF + sum_alpha (-1)^|alpha| D_alpha ( dL/d(D_alpha F) )
    out = []
    for F in funcs:
        derivs = sorted([dv for dv in L.atoms(sp.Derivative) if dv.expr == F], key=lambda dv: -dv.derivative_count)
        syms = {dv: sp.Dummy() for dv in derivs}
        fs = sp.Dummy()
        rep = dict(syms)
        rep[F] = fs
        Ls = L.xreplace(rep)
        back = {v: kk for kk, v in syms.items()}
        back[fs] = F
        res = sp.diff(Ls, fs).xreplace(back)
        for dv, s in syms.items():
            term = sp.diff(Ls, s).xreplace(back)
            for v, cnt in dv.variable_count:
                term = sp.diff(term, v, cnt)
            res += (-1) ** dv.derivative_count * term
        out.append(sp.expand(res))
    return out


eps = sp.Symbol("epsilon")
Mpl = sp.Symbol("M", positive=True)            # M = c^4 / (8 pi G)
kappa, zeta, m2 = sp.symbols("kappa zeta m2", real=True)
fA, fB = sp.symbols("f_A f_B", real=True)
k = sp.Symbol("k", positive=True)
rho_hat = sp.Symbol("rhohat", positive=True)

# ════════════════════════════════════════════════════════════════════════════════════════
# Part B — static weak field: variation of the full family, point source solved in Fourier space
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part B: static weak-field variation ...", flush=True)
x, y, z = sp.symbols("x y z", real=True)
X3 = (x, y, z)
Phi, Psi, sig, rho = [sp.Function(nm)(*X3) for nm in ("Phi", "Psi", "sigma", "rho")]
tcoord = sp.Symbol("t", real=True)
g_static = sp.diag(-(1 + 2 * eps * Phi), 1 - 2 * eps * Psi, 1 - 2 * eps * Psi, 1 - 2 * eps * Psi)
R_static = ricci_scalar(g_static, (tcoord, x, y, z))
sqrtg = sp.sqrt((1 + 2 * eps * Phi) * (1 - 2 * eps * Psi) ** 3)
grad2_sig = sum(sp.diff(sig, v) ** 2 for v in X3)
L_full = (sqrtg * (Mpl / 2) * (1 + 2 * zeta * eps * sig) * R_static
          - sqrtg * (Mpl / 2) * kappa * eps ** 2 * grad2_sig / (1 - 2 * eps * Psi)
          - sqrtg * (Mpl / 2) * m2 * eps ** 2 * sig ** 2
          - eps * rho * (1 + fA * eps * sig) * sp.sqrt(1 + 2 * eps * Phi))
L2 = sp.expand(order2(L_full, eps))
eqs = EL(L2, [Phi, Psi, sig], X3)

# Fourier: every field ~ hat * exp(i k x) (the equations are isotropic and linear, a plane wave along x suffices)
Ph, Ps, Sg = sp.symbols("Phihat Psihat sigmahat")
pw = sp.exp(sp.I * k * x)
subsF = {Phi: Ph * pw, Psi: Ps * pw, sig: Sg * pw, rho: rho_hat * pw}
lin = [sp.simplify(e.subs(subsF).doit() / pw) for e in eqs]
sol = sp.solve(lin, [Ph, Ps, Sg], dict=True)[0]
PhK, PsK, SgK = [sp.simplify(sol[v]) for v in (Ph, Ps, Sg)]
print("  Phi(k) =", PhK, "\n  Psi(k) =", PsK, "\n  sigma(k) =", SgK, flush=True)

kE = kappa + 6 * zeta ** 2
PhiN = -rho_hat / (2 * Mpl * k ** 2)            # Newton: Lap Phi = 4 pi G rho  ->  Phi(k) = -4 pi G rho/k^2 = -rho/(2 M k^2)

# B0: GR limit (zeta = f = 0) reproduces Newton and gamma = 1 (sign convention check)
check("B0a_GR_limit_Newton", PhK.subs({zeta: 0, fA: 0}) - PhiN, "[VERIFIED numerics]",
      "with zeta = f = 0 the variation gives Lap Phi = 4 pi G rho (attractive gravity: sign conventions are right)")
check("B0b_GR_limit_Psi_equals_Phi", (PsK - PhK).subs({zeta: 0, fA: 0}), "[VERIFIED numerics]", "GR: Psi = Phi (PPN gamma = 1)")

# massless family
Ph0, Ps0, Sg0 = [e.subs(m2, 0) for e in (PhK, PsK, SgK)]
xi_eff = sp.simplify(Sg0 / (-2 * PhiN))         # sigma = -2 xi_eff Phi_N  <=>  sigma = 2 xi_eff G M / r
check("B1_xi_eff", xi_eff - (zeta - fA) / kE, "[DERIVED]",
      "static sigma = 2 xi_eff G M/(r c^2) with xi_eff = (zeta - f_A)/(kappa + 6 zeta^2)")
# test body B: potential energy per unit mass = Phi + f_B sigma  (from -m_B(1 + f_B sigma) sqrt(1 + 2 Phi))
GeffAB = sp.simplify((Ph0 + fB * Sg0) / PhiN)
aAaB = 2 * (fA - zeta) * (fB - zeta) / kE
check("B2_fifth_force", GeffAB - (1 + aAaB), "[DERIVED]",
      "G_AB/G = 1 + alpha_A alpha_B, alpha_A alpha_B = 2 (f_A - zeta)(f_B - zeta)/(kappa + 6 zeta^2)")
gam = sp.simplify((Ph0 + Ps0) / (Ph0 + fB * Sg0) - 1)  # light: Phi + Psi ; Kepler: Phi + f_B sigma
check("B3_PPN_gamma", gam - 1 - (-2 * aAaB / (1 + aAaB)), "[DERIVED]",
      "gamma_AB - 1 = -2 alpha_A alpha_B/(1 + alpha_A alpha_B) (Damour & Esposito-Farese form, non-universal)")
check("B4m_light_is_GR_massive", sp.simplify((PhK + PsK) - 2 * PhiN), "[DERIVED]",
      "with a mass term too, light sees exactly GR's Phi + Psi: a short-range sigma leaves Cassini's gamma at 1 once lambda << AU")   # REPAIR (O7 per-bound values)
check("B4_light_is_GR", sp.simplify((Ph0 + Ps0) - 2 * PhiN),
      "[DERIVED]", "light deflection (Phi + Psi) is exactly GR's: the scalar is invisible to light at this order (conformal invariance)")
# composition invariance: differential acceleration between B and C depends only on sigma's amplitude
fC = sp.Symbol("f_C", real=True)
eta_BC = sp.simplify(((Ph0 + fB * Sg0) - (Ph0 + fC * Sg0)) / PhiN)
check("B5_EP_eta_action_independent", eta_BC - (-2 * xi_eff * (fB - fC)), "[BY CONSTRUCTION]",   # REPAIR: retagged
      "eta(B,C) = -2 xi_eff (f_B - f_C) for every kappa, zeta: composition tests see only xi_eff and Delta f")
# clocks: d ln mu = f_mu sigma ; d ln mu / dPhi_N = -2 xi_eff f_mu (by the definition of xi_eff; no kappa or zeta left)
fmu = sp.Symbol("f_mu", real=True)
check("B6_clock_action_independent", sp.simplify(fmu * Sg0 / PhiN) - (-2 * xi_eff * fmu), "[BY CONSTRUCTION]",   # REPAIR: retagged
      "(c^2/mu) d mu/d Phi_N = -2 xi_eff f_mu for every kappa, zeta (Phi_N built with the bare G of the action)")
# REPAIR: B5/B6 cannot fail. Repeat them with an ARBITRARY sigma(k) and Phi(k) (no field equation at all).
S_arb, P_arb = sp.Function("S_arb")(k), sp.Function("P_arb")(k)
xi_arb = S_arb / (-2 * PhiN)
r5 = sp.simplify(((P_arb + fB * S_arb) - (P_arb + fC * S_arb)) / PhiN - (-2 * xi_arb * (fB - fC)))
r6 = sp.simplify(fmu * S_arb / PhiN - (-2 * xi_arb * fmu))
_taut = bool(r5 == 0 and r6 == 0)
CHECKS.append(dict(name="B5B6_tautology_demo", claim="B5 and B6 hold for an arbitrary sigma(k), Phi(k): they follow from the definition "
                   "xi_eff = sigma/(-2 Phi_N) and cannot fail (passed = 'is a tautology')", residual=f"B5: {r5}; B6: {r6}", passed=_taut,
                   tag="[BY CONSTRUCTION]", note="the only content is that composition tests measure grad sigma; this restates program section 6"))
print(("PASS " if _taut else "FAIL ") + "B5B6_tautology_demo (residuals vanish for an arbitrary sigma: B5 and B6 are by construction)")

# B7: sign table at the repo's sign (xi_eff > 0) with f > 0
#   pure sigma-T: zeta = 0 -> xi_eff = -f/kappa > 0 needs kappa < 0 (ghost); alpha^2 = 2 f^2/kappa < 0 (repulsive)
xi_s, f_s = sp.symbols("xi f", positive=True)
kap_T = sp.solve(sp.Eq(xi_eff.subs({zeta: 0, fA: f_s}), xi_s), kappa)[0]
check("B7a_pureT_repo_sign_needs_ghost", kap_T - (-f_s / xi_s), "[DERIVED]",
      "pure sigma-T with sigma > 0 near matter: kappa = -f/xi < 0 (ghost)")
check("B7b_pureT_alpha2", sp.simplify(aAaB.subs({zeta: 0, fA: f_s, fB: f_s, kappa: kap_T})) - (-2 * xi_s * f_s), "[DERIVED]",
      "pure sigma-T at the repo sign: alpha^2 = -2 xi f (repulsive fifth force, the program's 'repulsive' finding)")
check("B7c_pureT_healthy_alpha2", sp.simplify(aAaB.subs({zeta: 0, fA: f_s, fB: f_s, kappa: f_s / xi_s})) - 2 * xi_s * f_s, "[DERIVED]",
      "pure sigma-T healthy (kappa = f/xi): alpha^2 = +2 xi f, sigma = -2 xi G M/(r c^2) < 0 near matter")
#   mixed: xi_eff > 0 with kappa_E > 0 needs zeta > f ; then alpha^2 = 2 (zeta - f) xi
check("B7d_mixed_alpha2", sp.simplify(aAaB.subs({fA: f_s, fB: f_s}) - (-2 * (f_s - zeta) * xi_eff.subs(fA, f_s))), "[DERIVED]",
      "alpha^2 = -2 (f - zeta) xi_eff: at the repo sign (xi_eff > 0) healthy iff zeta > f")
#   pure sigma-R: f = 0 -> xi_eff = zeta/kappa_E, alpha^2 = 2 zeta xi_eff
check("B7e_pureR", sp.simplify(aAaB.subs({fA: 0, fB: 0}) - 2 * zeta * xi_eff.subs(fA, 0)), "[DERIVED]",
      "pure sigma-R: alpha^2 = 2 zeta xi_eff (Cassini bounds zeta, not xi_eff)")

# B8: massive case — Einstein-frame range lambda^2 = kappa_E/m2 ; Yukawa amplitude alpha_A alpha_B
SgM = sp.simplify(SgK)
check("B8a_massive_sigma_pole", sp.simplify(SgM - (zeta - fA) * rho_hat / (Mpl * (kE * k ** 2 + m2))), "[DERIVED]",
      "sigma(k) = (zeta - f) rho/(M (kappa_E k^2 + m2)): one scalar pole at k^2 = -m2/kappa_E")
GeffM = sp.simplify((PhK + fB * SgK) / PhiN)
yuk = sp.simplify(GeffM - 1)
check("B8b_massive_yukawa_amplitude", sp.simplify(yuk - aAaB * kE * k ** 2 / (kE * k ** 2 + m2)), "[DERIVED]",
      "G_AB(k)/G = 1 + alpha_A alpha_B k^2/(k^2 + m2/kappa_E): Newton + Yukawa with the massless amplitude")

# B9: Green's functions (r > 0): (Lap - mu^2) e^{-mu r}/r = 0 ; (Lap + mu^2) cos(mu r)/r = 0
r = sp.Symbol("r", positive=True)
mu = sp.Symbol("mu", positive=True)
lap_r = lambda F: sp.diff(r * F, r, 2) / r
check("B9a_yukawa_green", sp.simplify(lap_r(sp.exp(-mu * r) / r) - mu ** 2 * sp.exp(-mu * r) / r), "[DERIVED]",
      "m2/kappa_E > 0: static Green's function e^{-r/lambda}/r (short range)")
check("B9b_tachyonic_green", sp.simplify(lap_r(sp.cos(mu * r) / r) + mu ** 2 * sp.cos(mu * r) / r), "[DERIVED]",
      "m2/kappa_E < 0: static Green's function cos(r/lambda)/r (oscillating, no screening)")

# ════════════════════════════════════════════════════════════════════════════════════════
# Part C — ghost criterion: time-dependent quadratic action, linearised Weyl shift decouples sigma
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part C: Weyl-shift decoupling (t, x) ...", flush=True)
t_, x_ = sp.symbols("t x1", real=True)
X2 = (t_, x_)
P2, S2, s2, r2 = [sp.Function(nm)(*X2) for nm in ("Phi2", "Psi2", "sig2", "rho2")]
Pt, St = [sp.Function(nm)(*X2) for nm in ("PhiT", "PsiT")]
y2, z2 = sp.symbols("y2 z2", real=True)
g_t = sp.diag(-(1 + 2 * eps * P2), 1 - 2 * eps * S2, 1 - 2 * eps * S2, 1 - 2 * eps * S2)
R_t = ricci_scalar(g_t, (t_, x_, y2, z2))
sqrtg_t = sp.sqrt((1 + 2 * eps * P2) * (1 - 2 * eps * S2) ** 3)
kin_t = (-sp.diff(s2, t_) ** 2 / (1 + 2 * eps * P2) + sp.diff(s2, x_) ** 2 / (1 - 2 * eps * S2))   # g^{mn} d sigma d sigma
LJ = (sqrtg_t * (Mpl / 2) * (1 + 2 * zeta * eps * s2) * R_t - sqrtg_t * (Mpl / 2) * kappa * eps ** 2 * kin_t
      - eps * r2 * (1 + fA * eps * s2) * sp.sqrt(1 + 2 * eps * P2))
LJ2 = sp.expand(order2(LJ, eps))
LGR2 = sp.expand(LJ2.subs({zeta: 0, kappa: 0, fA: 0}).subs(s2, 0).doit())      # pure GR + minimal dust
LJ2_shift = sp.expand(LJ2.subs({P2: Pt - zeta * s2, S2: St + zeta * s2}).doit())
LGR2_T = sp.expand(LGR2.subs({P2: Pt, S2: St}).doit())
target = (LGR2_T + (Mpl / 2) * kE * (sp.diff(s2, t_) ** 2 - sp.diff(s2, x_) ** 2) - r2 * (fA - zeta) * s2)
diffL = sp.expand(LJ2_shift - target)
print("  terms in (shifted Jordan - target):", len(sp.Add.make_args(diffL)), flush=True)
els = EL(diffL, [Pt, St, s2], X2)
# negative control: the same test with the naive coefficient kappa (no 6 zeta^2) must FAIL
target_wrong = target - (Mpl / 2) * 6 * zeta ** 2 * (sp.diff(s2, t_) ** 2 - sp.diff(s2, x_) ** 2)
els_wrong = EL(sp.expand(LJ2_shift - target_wrong), [s2], X2)
ctrl = sp.simplify(els_wrong[0])
CHECKS.append(dict(name="C0_negative_control_naive_kappa", claim="the decoupling test rejects the naive kinetic coefficient kappa (control: must be NONZERO)",
                   residual=str(ctrl), passed=bool(ctrl != 0), tag="[control]", note="shows the C1 test can fail"))
print(("PASS " if ctrl != 0 else "FAIL ") + "C0_negative_control_naive_kappa  residual = " + str(ctrl), flush=True)
for nm, e in zip(("PhiT", "PsiT", "sigma"), els):
    check(f"C1_decoupling_EL_{nm}", e, "[DERIVED]",
          "Jordan quadratic action = GR(PhiT, PsiT) + (M/2)(kappa + 6 zeta^2)(sigma_t^2 - sigma_x^2) - rho (f - zeta) sigma, up to a total derivative",
          note="the Euler-Lagrange operator annihilates the difference, so the difference is a total derivative")
CHECKS.append(dict(name="C2_ghost_rule", claim="sigma's kinetic energy is (M/2) kappa_E sigma_t^2: ghost iff kappa_E = kappa + 6 zeta^2 < 0; "
                   "Einstein-frame matter coupling beta = f - zeta", residual="0 (follows from C1)", passed=all(c["passed"] for c in CHECKS if c["name"].startswith("C1")),
                   tag="[DERIVED]", note="Brans-Dicke's omega > -3/2 is the same rule (P11)"))

# REPAIR (O4 is ghost-free only near sigma = 0): repeat the C1 decoupling test around a constant background sigma0.
#   linear completion F = 1 + 2 zeta sigma : F0 = 1 + 2 zeta sigma0, F'/(2F) = zeta/F0, kinetic energy (M/2)(kappa + 6 zeta^2/F0) sigma_t^2
#   exponential F = exp(2 zeta sigma)      : F0 = exp(2 zeta sigma0), F'/(2F) = zeta,   kinetic energy (M/2)(kappa + 6 zeta^2 F0) sigma_t^2
#   F(sigma0 + d) = F0 + F' d + F''/2 d^2 ; the d^2 R term is cubic around flat space but is kept to show it drops out.
print("Part C3: ghost test around a background sigma0 ...", flush=True)
s0 = sp.Symbol("sigma0", real=True)
grav_T = sp.expand(LGR2_T.subs(r2, 0))
matt_T = sp.expand(LGR2_T - grav_T)                     # = -rho PhiT
box_s = sp.diff(s2, t_) ** 2 - sp.diff(s2, x_) ** 2
C3_RES = {}
for comp, Ffun in (("linF", 1 + 2 * zeta * s0), ("expF", sp.exp(2 * zeta * s0))):
    F0_ = Ffun
    F1_ = sp.diff(Ffun, s0)
    F2_ = sp.diff(Ffun, s0, 2)
    shift = sp.simplify(F1_ / (2 * F0_))                # Weyl-shift coefficient F'/(2F)
    LJb = (sqrtg_t * (Mpl / 2) * (F0_ + F1_ * eps * s2 + F2_ / 2 * eps ** 2 * s2 ** 2) * R_t
           - sqrtg_t * (Mpl / 2) * kappa * eps ** 2 * kin_t - eps * r2 * (1 + fA * s0 + fA * eps * s2) * sp.sqrt(1 + 2 * eps * P2))
    LJb2 = sp.expand(order2(LJb, eps))
    LJb2_shift = sp.expand(LJb2.subs({P2: Pt - shift * s2, S2: St + shift * s2}).doit())
    kin_coef = sp.simplify(kappa + 6 * shift ** 2 * F0_)          # predicted: kappa + 6 zeta^2/F0 (lin) or kappa + 6 zeta^2 F0 (exp)
    targetb = (F0_ * grav_T + (1 + fA * s0) * matt_T + (Mpl / 2) * kin_coef * box_s - r2 * (fA - (1 + fA * s0) * shift) * s2)
    elsb = EL(sp.expand(LJb2_shift - targetb), [Pt, St, s2], X2)
    for nm, e in zip(("PhiT", "PsiT", "sigma"), elsb):
        check(f"C3_{comp}_background_EL_{nm}", sp.cancel(sp.together(e)), "[DERIVED]",
              f"around a constant sigma0 ({comp}) sigma's kinetic energy is (M/2)({kin_coef}) sigma_t^2: ghost iff that is < 0")
    wrong = sp.expand(LJb2_shift - (targetb - (Mpl / 2) * (kin_coef - (kappa + 6 * zeta ** 2)) * box_s))
    ctrlb = sp.cancel(sp.together(EL(wrong, [s2], X2)[0]))
    CHECKS.append(dict(name=f"C3_{comp}_negative_control_sigma0_independent", claim="the background test rejects the sigma0-independent "
                       "coefficient kappa + 6 zeta^2 (control: must be NONZERO)", residual=str(ctrlb), passed=bool(ctrlb != 0), tag="[control]",
                       note="shows the C3 test can fail"))
    print(("PASS " if ctrlb != 0 else "FAIL ") + f"C3_{comp}_negative_control  residual = {ctrlb}", flush=True)
    C3_RES[comp] = str(kin_coef)

# ════════════════════════════════════════════════════════════════════════════════════════
# Part D — covariant versus flat box at 1PN (static)
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part D: covariant box at 1PN ...", flush=True)
gm = sp.Symbol("gamma_ppn", real=True)
U = sp.Function("U")(*X3)
sD = sp.Function("s")(*X3)
gD = sp.diag(-(1 + 2 * eps * U), 1 - 2 * gm * eps * U, 1 - 2 * gm * eps * U, 1 - 2 * gm * eps * U)
sqrtgD = sp.sqrt(-gD.det())
boxD = sum(sp.diff(sqrtgD * (1 / gD[i + 1, i + 1]) * sp.diff(sD, X3[i]), X3[i]) for i in range(3)) / sqrtgD
box1 = sp.series(boxD, eps, 0, 2).removeO()
lapS = sum(sp.diff(sD, v, 2) for v in X3)
gradUgradS = sum(sp.diff(U, v) * sp.diff(sD, v) for v in X3)
check("D1_covariant_box_1PN", sp.expand(box1 - ((1 + 2 * gm * eps * U) * lapS + (1 - gm) * eps * gradUgradS)), "[DERIVED]",
      "static covariant box = (1 + 2 gamma Phi/c^2) Lap sigma + (1 - gamma) grad Phi . grad sigma/c^2 + O(Phi^2); the flat box drops both")
# the covariant kinetic term at quadratic order equals the flat one (difference is cubic): linear bounds cannot tell them apart
kin_cov = sqrtgD * sum((1 / gD[i + 1, i + 1]) * sp.diff(sD, X3[i]) ** 2 for i in range(3))
check("D2_cov_equals_flat_at_quadratic_order", sp.series(kin_cov, eps, 0, 1).removeO() - sum(sp.diff(sD, v) ** 2 for v in X3), "[BY CONSTRUCTION]",
      "sqrt(-g) g^{ij} d sigma d sigma = (grad sigma)^2 + O(eps): every LINEAR-order prediction is identical for both boxes "
      "(says nothing about 1PN beta, lunar ranging or strong-field pulsar terms)",
      note="REPAIR: retagged; the zeroth-order term of sqrt(-g) g^ij is delta^ij, so this cannot fail")

# ════════════════════════════════════════════════════════════════════════════════════════
# Part E — what the sigma-T source is: the trace eps - 3P (free gas, scale-only EOS)
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part E: trace-source identities ...", flush=True)
p, m = sp.symbols("p m", positive=True)
E = sp.sqrt(p ** 2 + m ** 2)
check("E1_free_particle_trace", m * sp.diff(E, m) - (E - 3 * p ** 2 / (3 * E)), "[DERIVED]",
      "m dE/dm = E - 3 (p^2/3E): for ANY momentum distribution of free particles, m d eps/dm |_{distribution} = eps - 3P")
Lam, n = sp.symbols("Lambda n", positive=True)
gfun = sp.Function("g")
eps_L = Lam ** 4 * gfun(n / Lam ** 3)
P_L = n * sp.diff(eps_L, n) - eps_L
check("E2_scale_only_EOS_trace", sp.simplify(Lam * sp.diff(eps_L, Lam) - (eps_L - 3 * P_L)), "[DERIVED]",
      "if Lambda_QCD is the only scale, Lambda d eps/d Lambda |_n = eps - 3P for ANY equation of state (the trace identity)",
      note="REPAIR: uses the T = 0 relation P = n d eps/dn - eps")
rd, fd, sd = sp.symbols("rho_d f_d sigma_d", real=True)
Ld = -rd * (1 + fd * sd)
check("E3_dust_coupling_is_fT", sp.diff(Ld, sd) - fd * (-rd), "[BY CONSTRUCTION]",
      "dust: dL_m/d sigma = f T with T = -rho c^2: the mass response IS a sigma-T coupling with charge f")

# ════════════════════════════════════════════════════════════════════════════════════════
# Part F — reciprocity: action versus the postulate (universal source, non-universal response)
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part F: reciprocity ...", flush=True)
xa, xb = sp.symbols("x_a x_b", real=True)
Gs, mA_, mB_, xi_p = sp.symbols("G m_A m_B xi_p", positive=True)
dist = sp.sqrt((xa - xb) ** 2)
# postulate: sigma_X(x) = 2 xi G m_X / |x - x_X| (universal source), force on Y = -m_Y f_Y grad sigma_X (c = 1)
sigA = lambda X_: 2 * xi_p * Gs * mA_ / sp.sqrt((X_ - xa) ** 2)
sigB = lambda X_: 2 * xi_p * Gs * mB_ / sp.sqrt((X_ - xb) ** 2)
XX = sp.Symbol("XX", real=True)
F_on_B = -mB_ * fB * sp.diff(sigA(XX), XX).subs(XX, xb)
F_on_A = -mA_ * fA * sp.diff(sigB(XX), XX).subs(XX, xa)
net = sp.simplify((F_on_A + F_on_B).subs({xa: 0, xb: 1}))
check("F1_postulate_violates_third_law", net - 2 * xi_p * Gs * mA_ * mB_ * (fB - fA), "[DERIVED]",
      "postulate: F_A + F_B = 2 xi G m_A m_B (f_B - f_A)/r^2 != 0 -> no action, no momentum conservation unless source charge = response charge",
      note="the program's 'Candidate A' cannot come from any Lagrangian when f differs between bodies")
# action: L_int = G alpha_A alpha_B m_A m_B / |x_a - x_b| -> forces equal and opposite (translation invariance)
aA, aB = sp.symbols("alpha_A alpha_B", real=True)
Lint = Gs * aA * aB * mA_ * mB_ / dist
check("F2_action_obeys_third_law", sp.simplify(sp.diff(Lint, xa) + sp.diff(Lint, xb)), "[BY CONSTRUCTION]",
      "from an action the scalar forces are equal and opposite for any couplings")

# ════════════════════════════════════════════════════════════════════════════════════════
# Part G — Brans-Dicke and DEF maps
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part G: Brans-Dicke / DEF maps ...", flush=True)
om = sp.Symbol("omega", positive=True)
a_bd = 1 / (2 * om + 3)
check("G1_BD_gamma", (1 + om) / (2 + om) - (1 - 2 * a_bd / (1 + a_bd)), "[DERIVED]",
      "Brans-Dicke gamma = (1 + omega)/(2 + omega) = 1 - 2 alpha^2/(1 + alpha^2), alpha^2 = 1/(2 omega + 3)")
# pure sigma-R as Brans-Dicke: phi = 1 + 2 zeta sigma, (M/2) kappa (d sigma)^2 = (M/2)(omega/phi)(d phi)^2 at phi = 1
om_R = kappa / (4 * zeta ** 2)
check("G2_pureR_is_BD", sp.simplify(1 / (2 * om_R + 3) - aAaB.subs({fA: 0, fB: 0})), "[DERIVED]",
      "pure sigma-R = Brans-Dicke with omega = kappa/(4 zeta^2); alpha^2 = 1/(2 omega + 3) matches the static solve")
# DEF 1PN beta: alpha(phi) = d ln A/d phi, beta0 = d alpha/d phi; phi = int sqrt(kappa_E(s)/2) ds
s_ = sp.Symbol("s", real=True)
fq = sp.Symbol("q", positive=True)  # QCD share at sigma = 0 (f)
lnA = sp.log((1 - fq) + fq * sp.exp(s_)) - sp.Rational(1, 2) * sp.log(1 + 2 * zeta * s_)   # Einstein-frame ln m: m_bare + m_QCD e^s, times F^{-1/2}
Fs = 1 + 2 * zeta * s_
kE_s = kappa / Fs + sp.Rational(3, 2) * (sp.diff(Fs, s_) / Fs) ** 2        # Einstein-frame kinetic coefficient as a function of sigma
dphi_ds = sp.sqrt(kE_s / 2)
alpha_s = sp.diff(lnA, s_) / dphi_ds
beta0 = sp.simplify((sp.diff(alpha_s, s_) / dphi_ds).subs(s_, 0))
alpha0 = sp.simplify(alpha_s.subs(s_, 0))
check("G3_alpha0", sp.simplify(alpha0 ** 2 - aAaB.subs({fA: fq, fB: fq})), "[DERIVED]",
      "alpha_0^2 from the nonlinear Einstein-frame map equals the linear static solve (consistency)")
beta0_expected = (2 / kE) * ((fq * (1 - fq) + 2 * zeta ** 2) - (fq - zeta) * (-2 * zeta * kappa - 24 * zeta ** 3) / (2 * kE))
check("G4_beta0", sp.simplify(beta0 - beta0_expected), "[DERIVED]",
      "beta_0 = (2/kappa_E)[f(1-f) + 2 zeta^2 - (f - zeta) kappa_E'/(2 kappa_E)], kappa_E' = -2 zeta kappa - 24 zeta^3")

# G5: robustness of beta_0 to the nonlinear completion: F = exp(2 zeta sigma) instead of 1 + 2 zeta sigma
Fe = sp.exp(2 * zeta * s_)
lnA_e = sp.log((1 - fq) + fq * sp.exp(s_)) - sp.Rational(1, 2) * sp.log(Fe)
kE_e = kappa / Fe + sp.Rational(3, 2) * (sp.diff(Fe, s_) / Fe) ** 2
dphi_e = sp.sqrt(kE_e / 2)
alpha_e = sp.diff(lnA_e, s_) / dphi_e
beta0_expF = sp.simplify((sp.diff(alpha_e, s_) / dphi_e).subs(s_, 0))
check("G5_alpha0_same_for_expF", sp.simplify(alpha_e.subs(s_, 0) ** 2 - alpha0 ** 2), "[DERIVED]",
      "alpha_0 does not depend on the nonlinear completion of F (only beta_0 does)")

SYMBOLIC = dict(
    Phi_k=str(PhK), Psi_k=str(PsK), sigma_k=str(SgK), xi_eff=str(xi_eff), G_AB_over_G=str(GeffAB), gamma_minus_1=str(gam - 1),
    alpha0=str(alpha0), beta0=str(beta0), kappa_E="kappa + 6*zeta**2",
)
print("symbolic part done in %.0f s" % (time.time() - T0), flush=True)

# ════════════════════════════════════════════════════════════════════════════════════════
# Part H — numbers: every option against every section-6 bound
# ════════════════════════════════════════════════════════════════════════════════════════
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)
from mentat_client import mentat                                       # noqa: E402
from sigma_ground.inventory.core.sigma import XI, proton_mass_kg, neutron_mass_kg, three_measures_atom   # noqa: E402

SRC = {}


def leaf(tool, args, key=None, label=None):
    v, s = mentat(tool, args)
    if key is not None:
        v = v[key]
    SRC[label or f"{tool}{args}"] = s
    return v


Gv = leaf("lookup_constant", {"name": "G"}, label="G")
Cv = leaf("lookup_constant", {"name": "c"}, label="c")
HBAR = leaf("lookup_constant", {"name": "hbar"}, label="hbar")
EV = leaf("lookup_constant", {"name": "electron_volt"}, label="electron_volt")
AU = leaf("lookup_constant", {"name": "au"}, label="au")
sun = leaf("solar_system_body", {"body_name": "sun"}, label="sun")
earth = leaf("solar_system_body", {"body_name": "earth"}, label="earth")
M_SUN, R_SUN, R_E = sun["mass_kg"], sun["radius_m"], earth["radius_m"]

iso = json.load(open(os.path.join(REPO, "sigma_ground", "inventory", "data", "isotopes.json"), encoding="utf-8"))


def isotope(Z, A):
    for rr in iso:
        if rr["Z"] == Z and rr["A"] == A:
            return rr
    raise KeyError((Z, A))


d = 1e-6


def f_atom(Z, A):
    rr = isotope(Z, A)
    if A == 1:
        be = 0.0
        SRC[f"BE Z={Z} A={A}"] = "no nucleus to bind (single proton)"
    else:
        be = leaf("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": rr["atomic_mass_u"]},
                  key="binding_energy_MeV", label=f"BE Z={Z} A={A}")
    mm = lambda s: three_measures_atom(Z, A - Z, be, s)["atom_stable_mass_kg"]
    return (math.log(mm(d)) - math.log(mm(-d))) / (2 * d)


f_mu = (math.log(proton_mass_kg(d)) - math.log(proton_mass_kg(-d))) / (2 * d)
f_n = (math.log(neutron_mass_kg(d)) - math.log(neutron_mass_kg(-d))) / (2 * d)
F = {k_: f_atom(*k_) for k_ in [(1, 1), (2, 4), (4, 9), (6, 12), (8, 16), (13, 27), (22, 48), (23, 51), (26, 56), (45, 103), (78, 195)]}
f_TA6V = 0.90 * F[(22, 48)] + 0.06 * F[(13, 27)] + 0.04 * F[(23, 51)]   # [LIT approx] TA6V and PtRh10 mass fractions, as in sigma_exclusion.py
f_PtRh = 0.90 * F[(78, 195)] + 0.10 * F[(45, 103)]
df_micro = abs(f_TA6V - f_PtRh)
df_ew = abs(F[(4, 9)] - F[(22, 48)])                                     # Eot-Wash Be-Ti, one isotope each (stated approximation)
f_bar = F[(26, 56)]                                                      # a representative 'rock/iron' QCD share for the fine-tuning corner

# bounds, 2 sigma two-sided and conservative (|central| + 2 err), the program's convention   [LIT, verified 2026-10-10]
B = dict(
    clock=abs(7e-8) + 2 * 45e-8,                                        # Lange et al. 2021 PRL 126, 011102: (c^2/mu) dmu/dPhi = 7(45)e-8
    micro=abs(-1.5e-15) + 2 * math.hypot(2.3e-15, 1.5e-15),             # Touboul et al. 2022 PRL 129, 121102
    wd_G2938=abs(-5.8e-5) + 2 * math.hypot(3.8e-5, 0.3e-5),             # Bagdonaite et al. 2014 PRL 113, 123002, phi = 1.9e-4 (as encoded in sigma_exclusion.py)
    cassini=abs(2.1e-5) + 2 * 2.3e-5,                                   # Bertotti, Iess & Tortora 2003 Nature 425, 374
    nordtvedt=abs(-0.2e-4) + 2 * 1.1e-4,                                # Hofmann & Mueller 2018 CQG 35, 035015
    eotwash=abs(0.3e-13) + 2 * 1.8e-13,                                 # Schlamminger et al. 2008 PRL 100, 041101 (Be-Ti toward Earth)
    pulsar_dalpha=2e-3,                                                 # Freire et al. 2012 MNRAS 423, 3328, as summarised in Wex 2014 arXiv:1402.5594 (95 %)
)
PHI_WD = 1.9e-4
WD_DATA = [("GD133", 1.2e-4, -2.7e-5, math.hypot(4.7e-5, 0.2e-5)),     # Bagdonaite et al. 2014 (verified): (-2.7 +- 4.7 +- 0.2)e-5
           ("G29-38", 1.9e-4, -5.8e-5, math.hypot(3.8e-5, 0.3e-5))]    # (-5.8 +- 3.8 +- 0.3)e-5; phi values as encoded in sigma_exclusion.py


def predictions(kap, zet, f_src, label, frame, ghost_note, xi_scale=None):
    """All long-range (massless) predictions for one option. f_src: QCD share of the source/test matter."""
    kE_ = kap + 6 * zet ** 2
    beta = f_src - zet
    xi_e = (zet - f_src) / kE_
    a2 = 2 * beta ** 2 / kE_                                            # alpha_A alpha_B for like bodies
    gam1 = -2 * a2 / (1 + a2)
    b0 = float(beta0.subs({kappa: kap, zeta: zet, fq: f_src}))
    beta1 = 0.5 * b0 * a2 / (1 + a2) ** 2
    etaN = 4 * beta1 - gam1
    omega = (1 / a2 - 3) / 2 if a2 != 0 else float("inf")
    mass_response = f_src != 0
    clock = abs(2 * xi_e * f_mu) if mass_response else 0.0
    micro = abs(2 * xi_e * df_micro) if mass_response else 0.0
    wd = abs(2 * xi_e * f_mu * PHI_WD) if mass_response else 0.0
    ew = abs(2 * xi_e * df_ew) if mass_response else 0.0
    # |Delta alpha| from composition only (neutron matter vs carbon); for a ghost (kappa_E < 0) the dipole flux has the
    # same magnitude and the opposite sign (negative-energy radiation), so the magnitude is reported
    dalpha_comp = abs(F[(6, 12)] - f_n) * math.sqrt(2 / abs(kE_)) if mass_response else 0.0
    pred = dict(clock=clock, micro=micro, wd_G2938=wd, cassini=abs(gam1), nordtvedt=abs(etaN), eotwash=ew, pulsar_dalpha_composition_only=dalpha_comp)
    # the one bound where the SIGN of sigma matters: white-dwarf H2, signed Delta mu/mu = f_mu sigma = 2 xi_eff f_mu phi
    wd_signed = {}
    for nm, phi, val, err in WD_DATA:
        pr = 2 * xi_e * f_mu * phi if mass_response else 0.0
        wd_signed[nm] = dict(phi=phi, predicted=pr, measured=val, err=err, tension_sigma=(pr - val) / err)
    ratio = {kk: (pred[kk] / B[kk] if kk in B else None) for kk in pred}
    ratio["pulsar_dalpha_composition_only"] = pred["pulsar_dalpha_composition_only"] / B["pulsar_dalpha"]
    # REPAIR (G normalisation): xi_eff is defined against the action's bare G. Every composition test is normalised to an
    # OBSERVED potential or acceleration (GM_sun from ephemerides, g at MICROSCOPE / Eot-Wash, the white dwarf's observed mass),
    # which is (1 + alpha_A alpha_B) times the bare one, so at fixed bare xi_eff the observed coefficients carry 1/(1 + alpha^2).
    obs = 1 / (1 + a2)
    ratio_obs = dict(ratio)
    for kk in ("clock", "micro", "wd_G2938", "eotwash"):
        ratio_obs[kk] = ratio[kk] * obs
    return dict(xi_obs=xi_e * obs, obs_norm_factor=obs, exclusion_factor_observed_potential=ratio_obs,
                label=label, frame=frame, ghost=ghost_note, kappa=kap, zeta=zet, f=f_src, kappa_E=kE_, beta=beta, xi_eff=xi_e,
                sigma_sign_near_matter=("+" if xi_e > 0 else "-" if xi_e < 0 else "0"),
                alpha2=a2, fifth_force=("attractive" if a2 > 0 else "repulsive" if a2 < 0 else "none"),
                gamma_minus_1=gam1, beta_minus_1=beta1, beta0=b0, eta_N=etaN, omega_BD=omega,
                omega_jordan_BD=(kap / (4 * zet ** 2) if zet != 0 else None),   # Jordan-frame Brans-Dicke omega of the sigma-R sector; ghost iff < -3/2 (when f = 0)
                predictions=pred, exclusion_factor=ratio, white_dwarf_signed=wd_signed)


f0 = f_mu    # nucleon QCD share (0.990) as the representative f for sources and test bodies in the universal sector
XI_FIELD = XI                       # sigma = 2 xi G M/(r c^2)  (field-equation reading)
XI_INV = XI / 2                     # sigma = xi G M/(r c^2)    (inventory reading) == xi_eff = xi/2
OPT = {}
OPT["O1_T_healthy"] = predictions(f0 / XI_FIELD, 0.0, f0, "sigma-T, healthy (kappa = f/xi)", "Einstein (leptons, photons); non-universal dilaton", "no ghost")
OPT["O2_T_ghost"] = predictions(-f0 / XI_FIELD, 0.0, f0, "sigma-T, ghost (kappa = -f/xi), the repo's sign", "Einstein; non-universal", "GHOST (kappa_E < 0)")
OPT["O1n_T_healthy_inv_norm"] = predictions(f0 / XI_INV, 0.0, f0, "sigma-T healthy, inventory normalisation (xi_eff = xi/2)", "Einstein", "no ghost")
# pure sigma-R: matter minimal; xi_eff = zeta/kappa_E fixed to XI; zeta = 1 (O(1) coupling) and zeta at the Cassini edge
for zz, tag in ((1.0, "zeta1"), (None, "zeta_cassini")):
    if zz is None:
        a2max = B["cassini"] / (2 - B["cassini"])                           # |gamma-1| = 2a/(1+a) at the bound
        zz = a2max / (2 * XI_FIELD)
    kap_ = zz / XI_FIELD - 6 * zz ** 2
    OPT[f"O3_R_{tag}"] = predictions(kap_, zz, 0.0, f"sigma-R only, zeta = {zz:.3g} (kappa = {kap_:.3g})", "Jordan; universal (Brans-Dicke)",
                                     "no ghost" if kap_ + 6 * zz ** 2 > 0 else "GHOST")
# mixed sigma-R + sigma-T at the repo sign: zeta > f, xi_eff = XI
for zz, tag in ((2 * f0, "zeta2f"), (f0 + 1e-4, "zeta_f_plus_1e-4")):
    kE_needed = (zz - f0) / XI_FIELD
    kap_ = kE_needed - 6 * zz ** 2
    OPT[f"O4_RT_{tag}"] = predictions(kap_, zz, f0, f"sigma-R + sigma-T, zeta = {zz:.4g} (kappa = {kap_:.3g})", "Jordan (leptons); mixed",
                                      "no ghost (kappa < 0 but kappa_E > 0)" if kE_needed > 0 else "GHOST")
# the conformal-cancellation corner: literal coefficient zeta/kappa = XI but zeta = f_bar so the QCD share barely sources sigma
zz = f_bar
kap_ = zz / XI_FIELD
cc = predictions(kap_, zz, F[(1, 1)], "sigma-R + sigma-T, zeta = f(Fe-56): hydrogen-rich source (Sun-like)", "Jordan; mixed", "no ghost")
cc_e = predictions(kap_, zz, F[(8, 16)], "same corner, oxygen-rich source (Earth-like, O-16 as proxy)", "Jordan; mixed", "no ghost")
OPT["O5_cancel_Hsource"] = cc
OPT["O5_cancel_Osource"] = cc_e


# beta_0 at the tuned O4 corner for the two completions of F (does the 1/kappa_E growth survive?)
_o = OPT["O4_RT_zeta_f_plus_1e-4"]
BETA0_COMPLETION = dict(
    point=dict(kappa=_o["kappa"], zeta=_o["zeta"], f=_o["f"], kappa_E=_o["kappa_E"]),
    beta0_linearF=float(beta0.subs({kappa: _o["kappa"], zeta: _o["zeta"], fq: _o["f"]})),
    beta0_expF=float(beta0_expF.subs({kappa: _o["kappa"], zeta: _o["zeta"], fq: _o["f"]})),
)
_a2 = _o["alpha2"]; _g1 = _o["gamma_minus_1"]
BETA0_COMPLETION["eta_N_expF"] = 4 * 0.5 * BETA0_COMPLETION["beta0_expF"] * _a2 / (1 + _a2) ** 2 - _g1
BETA0_COMPLETION["LLR_factor_expF"] = abs(BETA0_COMPLETION["eta_N_expF"]) / B["nordtvedt"]
print("beta0 completion check:", BETA0_COMPLETION)

# ---------- range: MICROSCOPE lambda limit for each normalisation (Yukawa form factor of the program) ----------
def S(Rb, rr, lam):
    xx = Rb / lam
    if xx < 1e-3:
        return math.exp(-rr / lam) * (1 + xx * xx / 10)
    ex = -(rr - Rb) / lam
    if ex < -700:
        return 0.0
    return math.exp(ex) * 3 * ((xx - 1) + math.exp(-2 * xx) * (xx + 1)) / (2 * xx ** 3)


R_ORBIT = R_E + 710e3       # MICROSCOPE altitude [LIT approx, as in sigma_exclusion.py]


def eta_micro(xi_e, lam):
    return 2 * xi_e * df_micro * S(R_E, R_ORBIT, lam) * (1 + R_ORBIT / lam)


def lam_max(xi_e):
    lo, hi = 1.0, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eta_micro(xi_e, mid) > B["micro"]:
            hi = mid
        else:
            lo = mid
    return lo


RANGE = {f"xi_eff={v:.4f}": lam_max(v) for v in (0.13, XI_INV, 0.1571, XI_FIELD, 0.17, 2 * XI_FIELD)}
lam_grid = [10 ** (-5 + 0.05 * i) for i in range(281)]      # REPAIR: 10 um .. 1e9 m
curve_field = [eta_micro(XI_FIELD, L_) / B["micro"] for L_ in lam_grid]
curve_inv = [eta_micro(XI_INV, L_) / B["micro"] for L_ in lam_grid]
from sigma_ground.field.constants import LAMBDA_QCD_MEV   # noqa: E402  (the repo's PAPER point m_sigma = Lambda_QCD)
SRC["Lambda_QCD_MeV"] = "sigma_ground.field.constants.LAMBDA_QCD_MEV = %s (PDG reference, repo constant)" % LAMBDA_QCD_MEV
lam_qcd = HBAR * Cv / (LAMBDA_QCD_MEV * 1e6 * EV)

# ════════════════════════════════════════════════════════════════════════════════════════
# Part I (v2 repairs) — G normalisation, O5 per-test sources, LLR composition indicator, finite-range Eot-Wash
# indicator, O7 per-bound values, ghost onset around a background sigma, check kinds
# ════════════════════════════════════════════════════════════════════════════════════════
print("Part I: v2 repairs ...", flush=True)
V2 = {}

# ---- I1 which G normalises xi ----
# (a) bare: xi_eff = sigma / (2 G_bare M / r c^2); the option table above (reproduces program section 6). Each option also carries
#     exclusion_factor_observed_potential = composition factors x 1/(1 + alpha^2) (tests normalised to observed potentials).
# (b) observed: xi = 0.1582 read as sigma = 2 xi G_obs M / r c^2 with CODATA G (the repo's usage). Then
#     xi_obs = xi_eff/(1 + alpha^2) = (zeta - f)/(kappa_E + 2 (f - zeta)^2)  ->  kappa_E = (zeta - f)/xi_obs - 2 (f - zeta)^2.


def kappa_for_obs(zet, f_src, xi_obs):
    return (zet - f_src) / xi_obs - 2 * (f_src - zet) ** 2 - 6 * zet ** 2


OBSR = {}
OBSR["O1_T_healthy"] = predictions(kappa_for_obs(0.0, f0, -XI_FIELD), 0.0, f0, "sigma-T healthy, xi = 0.1582 as the observed amplitude", "Einstein", "")
OBSR["O2_T_ghost"] = predictions(kappa_for_obs(0.0, f0, XI_FIELD), 0.0, f0, "sigma-T ghost (repo sign), xi observed", "Einstein", "")
OBSR["O1n_T_healthy_inv_norm"] = predictions(kappa_for_obs(0.0, f0, -XI_INV), 0.0, f0, "sigma-T healthy, xi/2 observed", "Einstein", "")
OBSR["O3_R_zeta1"] = predictions(kappa_for_obs(1.0, 0.0, XI_FIELD), 1.0, 0.0, "sigma-R only, zeta = 1, xi observed", "Jordan", "")
OBSR["O4_RT_zeta2f"] = predictions(kappa_for_obs(2 * f0, f0, XI_FIELD), 2 * f0, f0, "sigma-R + sigma-T, zeta = 2f, xi observed", "Jordan", "")
OBSR["O4_RT_zeta_f_plus_1e-4"] = predictions(kappa_for_obs(f0 + 1e-4, f0, XI_FIELD), f0 + 1e-4, f0, "tuned corner, xi observed", "Jordan", "")
for kk_, o_ in OBSR.items():
    o_["ghost"] = "GHOST (kappa_E < 0)" if o_["kappa_E"] < 0 else "no ghost"
    target = (XI_INV if "inv" in kk_ else XI_FIELD) * (1 if o_["xi_eff"] > 0 else -1)
    assert abs(o_["xi_obs"] - target) < 1e-12, (kk_, o_["xi_obs"], target)
V2["observed_xi_reading"] = OBSR

NORM = {}
for kk_ in ("O1_T_healthy", "O2_T_ghost", "O1n_T_healthy_inv_norm", "O3_R_zeta1", "O4_RT_zeta2f", "O4_RT_zeta_f_plus_1e-4"):
    a_, b_ = OPT[kk_], OBSR[kk_]
    NORM[kk_] = dict(
        bare_xi=dict(xi_eff=a_["xi_eff"], xi_obs=a_["xi_obs"], obs_norm_factor=a_["obs_norm_factor"], alpha2=a_["alpha2"],
                     gamma_minus_1=a_["gamma_minus_1"], eta_N=a_["eta_N"],
                     clock=a_["exclusion_factor"]["clock"], micro=a_["exclusion_factor"]["micro"],
                     clock_vs_observed_potential=a_["exclusion_factor_observed_potential"]["clock"],
                     micro_vs_observed_g=a_["exclusion_factor_observed_potential"]["micro"],
                     cassini=a_["exclusion_factor"]["cassini"], nordtvedt=a_["exclusion_factor"]["nordtvedt"]),
        observed_xi=dict(xi_eff=b_["xi_eff"], xi_obs=b_["xi_obs"], kappa=b_["kappa"], kappa_E=b_["kappa_E"], alpha2=b_["alpha2"],
                         gamma_minus_1=b_["gamma_minus_1"], eta_N=b_["eta_N"], omega_BD=b_["omega_BD"],
                         clock=b_["exclusion_factor_observed_potential"]["clock"], micro=b_["exclusion_factor_observed_potential"]["micro"],
                         cassini=b_["exclusion_factor"]["cassini"], nordtvedt=b_["exclusion_factor"]["nordtvedt"],
                         # the white-dwarf potential phi is an observed one, so the prediction is 2 xi_obs f_mu phi
                         white_dwarf_G2938_tension_sigma=(2 * b_["xi_obs"] * f_mu * 1.9e-4 - (-5.8e-5)) / math.hypot(3.8e-5, 0.3e-5)))
V2["normalisation_table"] = NORM

# ---- I2 O5 per-test source: each test is sourced by its own body ----
zz5 = F[(26, 56)]
kap5 = zz5 / XI_FIELD
cc_c = predictions(kap5, zz5, F[(6, 12)], "O5 corner, carbon source (white-dwarf C/O core, C-12 proxy)", "Jordan; mixed", "no ghost")
OPT["O5_cancel_Csource"] = cc_c
V2["O5_per_test_source"] = dict(
    source_by_test=dict(clock="H-1 (Sun)", cassini="H-1 (Sun)", nordtvedt="H-1 (Sun)", micro="O-16 (Earth proxy)", eotwash="O-16 (Earth proxy)",
                        wd_G2938="C-12 (white-dwarf core proxy)", pulsar_dalpha_composition_only="independent of the source"),
    xi_eff=dict(H=cc["xi_eff"], O=cc_e["xi_eff"], C=cc_c["xi_eff"]),
    exclusion_factor=dict(clock=cc["exclusion_factor"]["clock"], cassini=cc["exclusion_factor"]["cassini"], nordtvedt=cc["exclusion_factor"]["nordtvedt"],
                          micro=cc_e["exclusion_factor"]["micro"], eotwash=cc_e["exclusion_factor"]["eotwash"],
                          wd_G2938=cc_c["exclusion_factor"]["wd_G2938"],
                          pulsar_dalpha_composition_only=cc["exclusion_factor"]["pulsar_dalpha_composition_only"]),
    white_dwarf_signed=cc_c["white_dwarf_signed"])

# ---- I3 lunar laser ranging: the Earth-Moon composition (EP) term, for options with a mass response ----
for zA in ((12, 24), (14, 28)):
    F[zA] = f_atom(*zA)
f_sil = (24 * F[(12, 24)] + 28 * F[(14, 28)] + 48 * F[(8, 16)]) / 100.0   # MgSiO3 by nucleon number: a stoichiometric silicate proxy
df_core = F[(26, 56)] - f_sil
B["llr_composition_EM"] = abs(-3e-14) + 2 * 5e-14   # [LIT] Hofmann & Mueller 2018 CQG 35, 035015 abstract (Crossref, verified 2026-10-10)
LLRC = {}
for kk_, o_ in OPT.items():
    if o_["f"] == 0:
        LLRC[kk_] = dict(eta_EM_per_unit_iron_fraction_difference=0.0, iron_fraction_difference_threshold=None,
                         note="no mass response: no composition term")
        continue
    per = 2 * abs(o_["xi_eff"]) * abs(df_core)
    LLRC[kk_] = dict(eta_EM_per_unit_iron_fraction_difference=per, iron_fraction_difference_threshold=B["llr_composition_EM"] / per,
                     note="excluded unless the Earth's and Moon's iron mass fractions differ by less than the threshold (Sun-sourced, bare xi_eff)")
V2["llr_composition"] = dict(f_silicate_MgSiO3_proxy=f_sil, f_Fe56=F[(26, 56)], delta_f_iron_minus_silicate=df_core,
                             bound_2sigma=B["llr_composition_EM"], per_option=LLRC,
                             citation="Hofmann & Mueller 2018, CQG 35, 035015: Delta(m_g/m_i)_EM = (-3 +- 5)e-14 (abstract via Crossref, verified 2026-10-10)")

# ---- I4 Eot-Wash Be-Ti toward Earth at finite range: smooth uniform Earth indicator (not a bound) ----


def ew_surface(xi_e, lam, dens=1.0):
    return dens * 2 * abs(xi_e) * df_ew * S(R_E, R_E, lam) * (1 + R_E / lam) / B["eotwash"]


lam_star = RANGE["xi_eff=%.4f" % XI_FIELD]


def ew_crossing(xi_e, dens=1.0):
    lo, hi = 1e-9, lam_star
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if ew_surface(xi_e, mid, dens) > 1:
            hi = mid
        else:
            lo = mid
    return hi


EWF = dict(lambda_star_m=lam_star, vertical_suppression_at_lambda_star=S(R_E, R_E, lam_star) * (1 + R_E / lam_star),
           halfspace_check_1p5_lambda_over_R=1.5 * lam_star / R_E,
           ratio_at_lambda_star=dict(uniform=ew_surface(XI_FIELD, lam_star), half_density=ew_surface(XI_FIELD, lam_star, 0.5),
                                     tenth_density=ew_surface(XI_FIELD, lam_star, 0.1), inventory_norm_uniform=ew_surface(XI_INV, lam_star)),
           indicator_reaches_1_at_lambda_m=dict(uniform=ew_crossing(XI_FIELD), tenth_density=ew_crossing(XI_FIELD, 0.1)),
           validity="Schlamminger et al. 2008 (arXiv:0712.0607, verified) report equivalence-principle limits for ranges from 1 m to infinity; "
                    "a torsion balance senses the horizontal part of the field, set by local topography for lambda << R_E, so this smooth-Earth "
                    "vertical indicator is an order-of-magnitude guide, not a bound, and says nothing below 1 m",
           curve=[ew_surface(XI_FIELD, L_) for L_ in lam_grid])
V2["eotwash_finite_range_indicator"] = EWF

# ---- I5 O7 (massive) per-bound values ----


def sun_clock_factor(lam):        # clock signal over Earth's orbit relative to the massless value (Yukawa of a ball, at 1 AU)
    return S(R_SUN, AU, lam) * (1 + AU / lam)


def o7_row(parent, lam):
    o_ = OPT[parent]
    return dict(lambda_m=lam, parent=parent, alpha2_yukawa_strength=o_["alpha2"],
                clock=o_["exclusion_factor"]["clock"] * sun_clock_factor(lam),
                micro=eta_micro(abs(o_["xi_eff"]), lam) / B["micro"],
                eotwash_finite_range_indicator=(ew_surface(o_["xi_eff"], lam) if lam >= 1.0 else None),
                wd_G2938_upper_bound=o_["exclusion_factor"]["wd_G2938"],
                cassini=0.0, nordtvedt=0.0, llr_composition=0.0, pulsar_dalpha=0.0,
                reasons=dict(cassini="B4m: light sees GR's Phi + Psi for any m2, and the ephemeris GM at 1 AU >> lambda carries no Yukawa: gamma = 1",
                             nordtvedt="the Sun's sigma at 1 AU and the Earth's at the Moon are suppressed by exp(-r/lambda)",
                             pulsar_dalpha="a scalar of range lambda radiates only above the frequency c/lambda; binary orbits are many orders slower",
                             wd="the Yukawa suppression is <= 1, so the factor is at most the massless value",
                             eotwash="smooth-Earth indicator (I4), not a bound; None below 1 m where Schlamminger 2008 does not reach"))


V2["O7_per_bound"] = {
    "O1_parent_at_lambda_star": o7_row("O1_T_healthy", lam_star),
    "O2_parent_at_lambda_star": o7_row("O2_T_ghost", lam_star),
    "O4_zeta2f_parent_at_lambda_star": o7_row("O4_RT_zeta2f", lam_star),
    "O1_parent_at_1m": o7_row("O1_T_healthy", 1.0),
    "O1_parent_at_paper_point": o7_row("O1_T_healthy", lam_qcd),
}
V2["isl_lit"] = ("[LIT] Lee, Adelberger, Cook, Fleischer & Heckel 2020, PRL 124, 101101 (arXiv:2002.11761, verified 2026-10-10): any "
                 "gravitational-strength (|alpha| = 1) Yukawa is limited to ranges < 38.6 um (95 %). Every option with a mass response predicts a "
                 "near-universal Yukawa of strength alpha^2 = 0.16-0.31 (0.46 in the observed-xi reading); the limit for that strength is not read "
                 "off the published curve here.")

# ---- I6 ghost onset around a background sigma (C3 coefficients) for the options with zeta != 0 ----
roots_lin, s_lin = mentat("solve_equation", {"equation": "kap*F + 6*zz**2", "variable": "F"})
roots_exp, s_exp = mentat("solve_equation", {"equation": "kap + 6*zz**2*exp(2*zz*s)", "variable": "s"})
SRC["ghost onset, linear F: solve(kap*F + 6*zz**2, F)"] = s_lin
SRC["ghost onset, exponential F: solve(kap + 6*zz**2*exp(2*zz*s), s)"] = s_exp
kap_sym, zz_sym = sp.symbols("kap zz")
LAND = {"Sun surface": 2 * XI_FIELD * Gv * M_SUN / (R_SUN * Cv ** 2),
        "white dwarf GD133 photosphere": 2 * XI_FIELD * 1.2e-4,       # phi [LIT] Bagdonaite 2014, as encoded in sigma_exclusion.py
        "white dwarf G29-38 photosphere": 2 * XI_FIELD * 1.9e-4,
        "horizon (r = r_s)": XI_FIELD,
        "sigma_conv = -ln xi": -math.log(XI)}


def K_lin(kp, ze, sg):
    return kp + 6 * ze ** 2 / (1 + 2 * ze * sg)


def K_exp(kp, ze, sg):
    return kp + 6 * ze ** 2 * math.exp(2 * ze * sg)


ONSET = {}
for kk_ in ("O4_RT_zeta2f", "O4_RT_zeta_f_plus_1e-4", "O5_cancel_Hsource", "O3_R_zeta1"):
    o_ = OPT[kk_]
    kp, ze = o_["kappa"], o_["zeta"]
    Fstar = complex(sp.N(sp.sympify(roots_lin[0]).subs({kap_sym: kp, zz_sym: ze})))
    sig_lin = (Fstar.real - 1) / (2 * ze) if kp < 0 else None                      # kappa > 0: never a ghost for F > 0
    sig_exp = None
    for r_ in roots_exp:
        v_ = complex(sp.N(sp.sympify(r_).subs({kap_sym: kp, zz_sym: ze})))
        if abs(v_.imag) < 1e-12:
            sig_exp = v_.real
    ONSET[kk_] = dict(kappa=kp, zeta=ze, K0=K_lin(kp, ze, 0.0),
                      ghost_for_sigma_above_linearF=sig_lin, ghost_for_sigma_below_expF=sig_exp,
                      check_K_at_onset_linF=(K_lin(kp, ze, sig_lin) if sig_lin is not None else None),
                      ghost_at_landmark_linF={nm: bool(K_lin(kp, ze, v) < 0) for nm, v in LAND.items()},
                      ghost_at_landmark_expF={nm: bool(K_exp(kp, ze, v) < 0) for nm, v in LAND.items()})
sig_grid = [10 ** (-8 + 0.025 * i) for i in range(361)]
V2["ghost_onset"] = dict(landmarks=LAND, per_option=ONSET,
                         curves={kk_: dict(lin=[K_lin(OPT[kk_]["kappa"], OPT[kk_]["zeta"], s_) / OPT[kk_]["kappa_E"] for s_ in sig_grid],
                                           exp=[K_exp(OPT[kk_]["kappa"], OPT[kk_]["zeta"], s_) / OPT[kk_]["kappa_E"] for s_ in sig_grid])
                                 for kk_ in ("O4_RT_zeta2f", "O4_RT_zeta_f_plus_1e-4")},
                         sigma_grid=sig_grid,
                         formula="sigma's kinetic energy around a constant sigma0 is (M/2) K sigma_t^2 with K = kappa + 6 zeta^2/F0 (F = 1 + 2 zeta sigma) "
                                 "or kappa + 6 zeta^2 F0 (F = exp(2 zeta sigma)) [C3]; ghost iff K < 0")

# ---- I7 which checks could fail ----


def kind_of(name):
    if "negative_control" in name:
        return "negative control (must come out nonzero; shows the test it guards can fail)"
    p = name.split("_")[0]
    if p in ("B0a", "B0b", "B1", "B2", "B3", "B4", "B4m", "B8a", "B8b", "C1", "C3", "D1", "E1", "E2", "F1", "G2", "G3", "G4", "G5"):
        return "could fail on physics grounds (an independent variation, solve or cross-route consistency)"
    if p in ("B7a", "B7b", "B7c", "B7d", "B7e", "G1"):
        return "algebraic substitution into an earlier result (could fail only through a typo)"
    if p in ("B9a", "B9b"):
        return "textbook identity (could fail only through a typo)"
    if p in ("B5", "B6", "B5B6", "D2", "E3", "F2"):
        return "tautology / by construction (cannot fail)"
    if p == "C2":
        return "summary record of C1"
    return "unclassified"


KINDS_OUT = {c["name"]: kind_of(c["name"]) for c in CHECKS}
for c in CHECKS:
    c["can_fail"] = KINDS_OUT[c["name"]]
V2["check_kind_counts"] = {kd: sum(1 for v in KINDS_OUT.values() if v == kd) for kd in sorted(set(KINDS_OUT.values()))}

V2["prereg_notes"] = dict(
    P5="prereg.json printed +0.911 for the ghost gamma - 1, from alpha^2 rounded to 0.313; the exact alpha^2 = 0.31336 gives +0.9127. The pass "
       "criterion is the symbolic identity B3, so P5 passes; the printed expectation was a rounding, recorded here, not moved.",
    P6="B5/B6 hold for any sigma amplitude (B5B6_tautology_demo): P6 is true by construction at fixed bare xi_eff. Against observed potentials the "
       "composition coefficients carry 1/(1 + alpha^2), which depends on kappa and zeta; P6's 'no action choice moves them' holds exactly only at "
       "fixed observed amplitude.",
    P8_D2="D2 (flat = covariant at quadratic order) is a tautology; it covers linear-order bounds only.",
    E2="E2 uses the T = 0 relation P = n d eps/dn - eps.",
    P1_local="P1 (ghost iff kappa + 6 zeta^2 < 0) is the criterion at sigma = 0; around a background sigma0 it becomes K(sigma0) < 0 (C3).")

V2["summary"] = dict(
    alpha2_O1=OPT["O1_T_healthy"]["alpha2"],
    composition_excl_bare=dict(clock=OPT["O1_T_healthy"]["exclusion_factor"]["clock"], micro=OPT["O1_T_healthy"]["exclusion_factor"]["micro"]),
    composition_excl_vs_observed_potential={kk_: (NORM[kk_]["bare_xi"]["clock_vs_observed_potential"], NORM[kk_]["bare_xi"]["micro_vs_observed_g"])
                                            for kk_ in ("O1_T_healthy", "O2_T_ghost", "O4_RT_zeta2f", "O4_RT_zeta_f_plus_1e-4")},
    observed_xi_universal={kk_: (NORM[kk_]["observed_xi"]["alpha2"], NORM[kk_]["observed_xi"]["gamma_minus_1"]) for kk_ in ("O1_T_healthy", "O2_T_ghost")},
    O5_per_test=V2["O5_per_test_source"]["exclusion_factor"],
    O3_cassini_edge=dict(zeta=OPT["O3_R_zeta_cassini"]["zeta"], omega=OPT["O3_R_zeta_cassini"]["omega_BD"],
                         max_factor=max(v for v in OPT["O3_R_zeta_cassini"]["exclusion_factor"].values() if v is not None)),
    ghost_onset={kk_: (ONSET[kk_]["ghost_for_sigma_above_linearF"], ONSET[kk_]["ghost_for_sigma_below_expF"]) for kk_ in ONSET},
    landmarks=LAND,
    eotwash_finite=dict(at_lambda_star=EWF["ratio_at_lambda_star"], reaches_1=EWF["indicator_reaches_1_at_lambda_m"]),
    llr_composition_threshold_O1=LLRC["O1_T_healthy"]["iron_fraction_difference_threshold"],
    llr_composition_threshold_O5H=LLRC["O5_cancel_Hsource"]["iron_fraction_difference_threshold"],
    O7_at_lambda_star=V2["O7_per_bound"]["O1_parent_at_lambda_star"],
    n_checks=len(CHECKS), n_passed=sum(c["passed"] for c in CHECKS))

out = dict(
    item="A2 sigma action options memo (v2, repair round)", prereg_file=os.path.join(HERE, "prereg.json"),
    conventions="MTW (-,+,+,+); EH +(M/2)sqrt(-g)R, M = c^4/8piG; ds2 = -(1+2Phi)dt2 + (1-2Psi)dx2",
    checks=CHECKS, n_checks=len(CHECKS), n_passed=sum(c["passed"] for c in CHECKS), symbolic=SYMBOLIC,
    mentat_sources=SRC, xi_repo=XI, f_mu=f_mu, f_neutron=f_n, f_atoms={f"Z={a},A={b}": v for (a, b), v in F.items()},
    f_TA6V=f_TA6V, f_PtRh10=f_PtRh, delta_f_microscope=df_micro, delta_f_BeTi=df_ew,
    bounds_2sigma=B, bound_citations=dict(
        clock="Lange et al. 2021, PRL 126, 011102 (arXiv:2010.06620)", micro="Touboul et al. 2022, PRL 129, 121102 (arXiv:2209.15487)",
        wd_G2938="Bagdonaite et al. 2014, PRL 113, 123002 (arXiv:1409.1000)", cassini="Bertotti, Iess & Tortora 2003, Nature 425, 374 (doi:10.1038/nature01997)",
        nordtvedt="Hofmann & Mueller 2018, CQG 35, 035015 (doi:10.1088/1361-6382/aa8f7a)", eotwash="Schlamminger et al. 2008, PRL 100, 041101 (arXiv:0712.0607)",
        pulsar_dalpha="Freire et al. 2012, MNRAS 423, 3328 (arXiv:1205.1450); |alpha_p - alpha_c| < 2e-3 as summarised in Wex 2014 arXiv:1402.5594"),
    options=OPT, beta0_completion_check=BETA0_COMPLETION, range_lambda_max_m=RANGE, lambda_paper_m=lam_qcd,
    microscope_curve=dict(lam_m=lam_grid, ratio_field=curve_field, ratio_inventory=curve_inv),
    v2=V2, check_kinds=KINDS_OUT, C3_kinetic_coefficients=C3_RES,
    runtime_s=time.time() - T0,
)
with open(os.path.join(HERE, "a2_results_v2.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
print("checks passed: %d / %d" % (out["n_passed"], out["n_checks"]))
for kk, v in OPT.items():
    print(f"{kk:28s} kE={v['kappa_E']:+.3g} xi_eff={v['xi_eff']:+.4f} a2={v['alpha2']:+.3g} g-1={v['gamma_minus_1']:+.3g} etaN={v['eta_N']:+.3g} "
          + " ".join(f"{a}={b:.2g}" for a, b in v["exclusion_factor"].items() if b is not None))
print("lambda_max:", {a: round(b) for a, b in RANGE.items()})
print("v2 summary:", json.dumps(V2["summary"], indent=1, default=float))
print("runtime %.0f s" % (time.time() - T0))
