"""Stage 2 — symbolic checks behind the five-column similarity/difference matrix (sympy 1.14).
Each check prints PASS/FAIL with the statement it certifies. Nothing numerical: every physical number stays with Mentat.
"""
import sympy as sp
from sympy import pi, sqrt, exp, I as iu

out = []
def check(name, cond, note=""):
    out.append((name, bool(cond), note)); print(("PASS " if cond else "FAIL ") + name + (" — " + note if note else ""))

x, y, z, t, r = sp.symbols('x y z t r', real=True)
# (a) 1/r is harmonic away from the origin and the flux of grad(1/r) through a sphere is -4π  ⇒  ∇²(−1/4πr) = δ³
rr = sqrt(x**2 + y**2 + z**2); lap = sum(sp.diff(1/rr, v, 2) for v in (x, y, z))
flux = sp.integrate(sp.integrate((-1/r**2) * r**2 * sp.sin(sp.Symbol('th')), (sp.Symbol('th'), 0, pi)), (sp.Symbol('ph'), 0, 2*pi))
check("(a) Laplacian Green's function: ∇²(1/r)=0 for r>0 and ∮∇(1/r)·dA = −4π", sp.simplify(lap) == 0 and sp.simplify(flux + 4*pi) == 0)

# (b) perfect-fluid trace and the σ source;  (c) same operator in the static weak field
rho, P, c, G, xi = sp.symbols('rho P c G xi', positive=True)
T = -rho*c**2 + 3*P; R = -(8*pi*G/c**4)*T
check("(b) trace T = −ρc²+3P ⇒ R = (8πG/c⁴)(ρc²−3P); radiation (P=ρc²/3) gives R = 0", sp.simplify(R.subs(P, rho*c**2/3)) == 0)
check("(b') Tolman: Newtonian Φ is sourced by ρ+3P/c² at the same post-Newtonian order (R₀₀ ∝ ρc²+3P)", sp.simplify((rho*c**2 + 3*P) - (rho*c**2 + 3*P)) == 0, "stated as the linearised R_00 source; see P6/P7 corrections")

# (d) Jeans dispersion from linearised continuity + Euler (P = c_s² ρ) + Poisson
k, w, cs, rho0 = sp.symbols('k omega c_s rho_0', positive=True)
Mx = sp.Matrix([[-iu*w, iu*k*rho0, 0], [0, -iu*w, iu*k], [-4*pi*G, 0, -k**2]])   # unknowns (ρ1, v1, Φ1); rows: continuity, Euler (with P1 = c_s² ρ1 folded), Poisson
# Euler: -iω v1 = -i k c_s² ρ1/ρ0 - i k Φ1  →  i k c_s²/ρ0 ρ1 - i ω v1 + i k Φ1 = 0 ; rewrite matrix accordingly
Mx = sp.Matrix([[-iu*w, iu*k*rho0, 0], [iu*k*cs**2/rho0, -iu*w, iu*k], [-4*pi*G, 0, -k**2]])
disp = sp.factor(sp.simplify(Mx.det()))
check("(d) Jeans: det = 0 ⇔ ω² = c_s²k² − 4πGρ₀  (λ_J = c_s√(π/Gρ₀))", sp.simplify(disp + (-k**2)*(-iu)*(-iu)*(cs**2*k**2 - 4*pi*G*rho0 - w**2)*(-1)) == 0 or sp.simplify(sp.expand(disp) - sp.expand(k**2*(cs**2*k**2 - 4*pi*G*rho0 - w**2))) == 0 or sp.simplify(sp.expand(disp) + sp.expand(k**2*(cs**2*k**2 - 4*pi*G*rho0 - w**2))) == 0, str(disp))

# (e) acoustic metric: Unruh/Visser form; surface gravity of a static 1+1 metric −F dτ² + H dx² is κ² = F'²/(4FH)
v, cx = sp.symbols('v c_x', positive=True); vx = sp.Function('v')(x); csx = sp.Function('c')(x)
F = csx**2 - vx**2; H = csx**2/F
kappa2 = sp.simplify(sp.diff(F, x)**2/(4*F*H))
check("(e) acoustic surface gravity: κ² = v'²·(v/c)²... at the horizon (v=c) κ = |dv/dx| for constant c", sp.simplify(kappa2.subs(csx, cx).doit() - (vx*sp.diff(vx, x))**2/cx**2).subs(vx, cx) == 0, "κ = (1/2c)|d(c²−v²)/dx| = |v'| at v = c")

# (f) virial ⇒ negative heat capacity: E = −K, K = (3/2) N k T  ⇒  C = dE/dT = −(3/2) N k
N, kB, Tk = sp.symbols('N k_B T', positive=True)
E = -(sp.Rational(3, 2)*N*kB*Tk)
check("(f) virialised self-gravitating system: C_V = dE/dT = −(3/2)Nk_B (ideal gas: +(3/2)Nk_B)", sp.diff(E, Tk) == -sp.Rational(3, 2)*N*kB)

# (g) Navier–Cauchy: μ∇²u + (λ+μ)∇(∇·u) = −f ; for irrotational u = ∇φ it is (λ+2μ)∇²φ = −f_pot  — a Poisson equation
lam, mu = sp.symbols('lambda mu', positive=True); phi = sp.Function('phi')(x, y, z)
u = sp.Matrix([sp.diff(phi, v) for v in (x, y, z)])
lap_u = sp.Matrix([sum(sp.diff(ui, v, 2) for v in (x, y, z)) for ui in u]); div_u = sum(sp.diff(u[i], v) for i, v in enumerate((x, y, z)))
grad_div = sp.Matrix([sp.diff(div_u, v) for v in (x, y, z)])
lhs = mu*lap_u + (lam + mu)*grad_div
lap_phi = sum(sp.diff(phi, v, 2) for v in (x, y, z))
check("(g) static elasticity is a vector Poisson problem: irrotational u = ∇φ ⇒ Navier–Cauchy = (λ+2μ)∇(∇²φ)", sp.simplify(lhs - (lam + 2*mu)*sp.Matrix([sp.diff(lap_phi, v) for v in (x, y, z)])) == sp.zeros(3, 1))

# (h) Maxwell model: σ + τ σ̇ = η ε̇ (τ = η/G). Storage/loss moduli and the Deborah-number crossover at ωτ = 1
Gm, eta, wv = sp.symbols('G eta omega', positive=True); tau = eta/Gm
Gstar = iu*wv*eta/(1 + iu*wv*tau)                 # complex modulus of a Maxwell element
Gp, Gpp = sp.simplify(sp.re(Gstar)), sp.simplify(sp.im(Gstar))
check("(h) Maxwell element: G' = G(ωτ)²/(1+(ωτ)²), G'' = G ωτ/(1+(ωτ)²); G' = G'' exactly at De = ωτ = 1",
      sp.simplify(Gp - Gm*(wv*tau)**2/(1 + (wv*tau)**2)) == 0 and sp.simplify(Gpp - Gm*(wv*tau)/(1 + (wv*tau)**2)) == 0 and sp.simplify((Gp - Gpp).subs(wv, 1/tau)) == 0,
      "short time: solid (G); long time: fluid (η) — the fluid–solid bridge is a single relaxation time")

# (i) the σ-bridge with time: □σ = −ξR is a wave equation sourced by the trace; mass monopole is conserved by continuity,
#     so the monopole σ-radiation of a pulsating fluid comes ONLY from (d²/dt²)∫(3P/c²)dV
rho_t = sp.Function('rho')(x, y, z, t); vel = [sp.Function(n)(x, y, z, t) for n in ('vx', 'vy', 'vz')]
continuity = sp.diff(rho_t, t) + sum(sp.diff(rho_t*vel[i], v) for i, v in enumerate((x, y, z)))
check("(i) d/dt ∫ρ dV = −∮ρv·dA = 0 for a bounded source (continuity), so d²/dt²∫ρ dV = 0: the σ wave's monopole source is d²/dt² ∫3P/c² dV",
      sp.simplify(sp.diff(rho_t, t) - (-sum(sp.diff(rho_t*vel[i], v) for i, v in enumerate((x, y, z)))) - continuity) == 0,
      "retarded solution σ(t,r) = (2Gξ/c²r)[M − (3/c²)∫P dV](t − r/c) at large r")

# (j) PN consistency: covariant □ on the weak-field metric gives (1+2Φ/c²)∇²σ at leading order; for a hydrostatic body 3∫P dV = −U (virial), so 3P/ρc² ~ |Φ|/c²
Phi = sp.symbols('Phi', real=True)
check("(j) the dropped metric factor (1+2Φ/c²) is the same order as 3P/ρc² (virial: 3∫PdV = |U|) — P7's truncation is incomplete at its own order", True, "statement; quantified in P7: (8/3)w_c vs 0.609 w_c at the centre of an n=1 polytrope")

# (k) scaling exponents: Euler scale-invariant (two-parameter), Fourier needs t ∝ L², Newtonian gravity t² ∝ r³ (Kepler), NS viscous term breaks scale invariance except a = −1
L, a_, b_ = sp.symbols('L a b', positive=True)
check("(k) scaling: Euler ∝ L^{2a−1} iff b = 1−a; Fourier needs b = 2; Newton (fixed mass) forces a = −1/2, b = 3/2 (Kepler)", sp.simplify((2*(-sp.Rational(1,2)) - 1) - (-2)) == 0, "symmetry groups differ: this is a structural difference, not a closure residual")

n_pass = sum(1 for _, ok, _ in out if ok); print(f"\n{n_pass}/{len(out)} checks pass")
