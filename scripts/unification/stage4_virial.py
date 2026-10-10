"""Stage 4 — L2, "same equation": the virial identity on the repo's SPH engine, measured exactly at t = 0, and the cause
of the missing pressure found.

Exact SPH identity (derived here; the symmetric Monaghan force makes it exact, not approximate). With
a_i = −Σ_j m_j (P_i/ρ_i² + P_j/ρ_j²) ∇_i W_ij and the pair antisymmetry ∇_j W_ji = −∇_i W_ij,

    Σ_i m_i r_i·a_i  =  −Σ_i m_i (P_i/ρ_i²) Σ_j m_j r_ij·∇_i W_ij  =  Σ_i m_i (P_i/ρ_i²) D_i ,   D_i := −Σ_{j≠i} m_j r_ij·∇_i W_ij ≥ 0.

For the isothermal closure P_i = ρ_i c_s²:  Σ m r·a_press = c_s² Σ_i m_i D_i/ρ_i, so the pressure-virial efficiency is

    x = Σ_i m_i D_i/ρ_i / (3M)                 (exact; independent of c_s by construction)

In the continuum, D = 3ρ exactly (∫ −r·∇W dV = 3∫W dV = 3). On particles, two things break it:
  (i)  the free surface: a particle within 2h of the edge sees an incomplete kernel, so D_i < 3ρ_i;
  (ii) the self-term: the SPH density sum includes the particle itself, ρ_i = m_i W(0,h) + Σ_{j≠i} m_j W_ij, while D_i
       has no j = i term (r_ii = 0). For a uniform interior, E[D_i] = 3(ρ_i − m_i W(0,h)), so
           x_interior = 1 − m W(0,h)/ρ  =  1/(1 + 32/(3 N_nb))   for the cubic spline (W(0) = 1/πh³),
       where N_nb is the full-support neighbour count n·(4/3)π(2h)³. This is an estimator property, not physics.
With h = 1.5 (V/N)^{1/3}, N_nb(full) = (4/3)π·8·1.5³ = 113.1, so x_interior → 1/(1 + 32/339) = 0.914 — a PERSISTENT deficit,
which is the "x_∞ ≈ 0.92" model the verification could not distinguish from convergence to 1. This stage measures it.

Gravity term: −Σ m r·a_grav vs |U₀| (unsoftened) and vs the softened |U₀,ε|.
Leaf constant G from Mentat (lookup_constant, sigma_ground.field.constants). Cold uniform cloud M = 1e12 kg, R = 5 km.
Usage: python -X utf8 scripts/unification/stage4_virial.py [out.json]
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum
from sigma_ground.dynamics.continuum import _sph_density_sum, _sph_pressure_force, _pressure_isothermal
from sigma_ground.dynamics.fluid.kernel import W, grad_W

G, G_src = mentat("lookup_constant", {"name": "G"})
M, R = 1.0e12, 5000.0
H_MULT = 1.5

def exact_t0(N, seed, f=0.4, h_mult=H_MULT, eps_frac=0.02):
    bodies = seed_rotating_cloud(N, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                 particle_density_kg_m3=920.0, jitter_frac=0.0, seed=seed)
    h = h_mult * (4 / 3 * math.pi * R ** 3 / N) ** (1 / 3)
    cs = f * math.sqrt(G * M / R)
    eps = eps_frac * R
    scene = to_continuum(bodies, material_key="water_ice", G=G, gravity_softening_m=eps,
                         eos="isothermal", sound_speed_m_s=cs, smoothing_h=h)
    ps = scene.continuum_parcels()
    grav = _sph_density_sum(ps, h, gravity_G=G, softening=eps)
    for p in ps:
        _pressure_isothermal(p, cs)
    apress = _sph_pressure_force(ps, h)
    m = [p.mass for p in ps]; Mt = sum(m)
    cx = sum(mi * p.position.x for mi, p in zip(m, ps)) / Mt
    cy = sum(mi * p.position.y for mi, p in zip(m, ps)) / Mt
    cz = sum(mi * p.position.z for mi, p in zip(m, ps)) / Mt
    rv = [(p.position.x - cx, p.position.y - cy, p.position.z - cz) for p in ps]
    vir_p = sum(mi * (r[0] * apress[i].x + r[1] * apress[i].y + r[2] * apress[i].z) for i, (mi, r) in enumerate(zip(m, rv)))
    vir_g = sum(mi * (r[0] * grav[i].x + r[1] * grav[i].y + r[2] * grav[i].z) for i, (mi, r) in enumerate(zip(m, rv)))
    U = U_soft = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = ps[i].position.x - ps[j].position.x; dy = ps[i].position.y - ps[j].position.y; dz = ps[i].position.z - ps[j].position.z
            r2 = dx * dx + dy * dy + dz * dz
            U -= G * m[i] * m[j] / math.sqrt(r2); U_soft -= G * m[i] * m[j] / math.sqrt(r2 + eps * eps)
    # identity pieces
    W0 = W(0.0, h)
    D = []; self_frac = []; interior = []
    for i, p in enumerate(ps):
        Di = 0.0
        for j in p.neighbors:
            q = ps[j]
            dx = p.position.x - q.position.x; dy = p.position.y - q.position.y; dz = p.position.z - q.position.z
            gx, gy, gz = grad_W(dx, dy, dz, h)
            Di -= q.mass * (dx * gx + dy * gy + dz * gz)
        D.append(Di); self_frac.append(p.mass * W0 / p.sph_density)
        rr = math.sqrt(sum(c * c for c in rv[i])); interior.append(rr < R - 2 * h)
    x_measured = vir_p / (3 * Mt * cs * cs)
    x_identity = sum(mi * Di / p.sph_density for mi, Di, p in zip(m, D, ps)) / (3 * Mt)
    n_int = sum(interior)
    x_int = (sum(mi * Di / p.sph_density for mi, Di, p, inn in zip(m, D, ps, interior) if inn) /
             (3 * sum(mi for mi, inn in zip(m, interior) if inn))) if n_int else None
    self_int = (sum(sf for sf, inn in zip(self_frac, interior) if inn) / n_int) if n_int else None
    nnb = sum(len(p.neighbors) for p in ps) / N
    nnb_full = (4 / 3) * math.pi * (2 * h) ** 3 * N / (4 / 3 * math.pi * R ** 3)
    return dict(N=N, seed=seed, h_mult=h_mult, two_h_over_R=2 * h / R, mean_neighbours=nnb, full_support_neighbours=nnb_full,
                x_measured=x_measured, x_identity=x_identity, identity_residual=x_measured - x_identity,
                n_interior=n_int, x_interior=x_int, predicted_interior_1_minus_self=(1 - self_int) if self_int is not None else None,
                predicted_uniform=1 / (1 + 32 / (3 * nnb_full)),
                gravity_over_U=-vir_g / abs(U), gravity_over_Usoft=-vir_g / abs(U_soft),
                U_over_analytic=abs(U) / (0.6 * G * M * M / R))

if __name__ == "__main__":
    plan = [(64, range(1, 7)), (128, range(1, 7)), (256, range(1, 5)), (512, range(1, 4)), (1024, range(1, 3)), (2048, range(1, 2))]
    rows = []
    for N, seeds in plan:
        for s in seeds:
            r = exact_t0(N, s); rows.append(r)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # h scan at fixed N: the self-term predicts x_interior = 1/(1+32/(3 N_nb)) rising with h; the free surface pulls x down
    for hm in (1.0, 1.25, 1.5, 2.0, 2.5):
        r = exact_t0(512, 1, h_mult=hm); r["h_scan"] = True; rows.append(r)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "misc/unification/stage4_virial.json"
    json.dump(dict(G_source=G_src, rows=rows), open(out, "w"), indent=1)
    print("saved", out)
