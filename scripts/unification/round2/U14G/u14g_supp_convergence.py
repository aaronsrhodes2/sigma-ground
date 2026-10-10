"""U14G supplementary (NOT pre-registered): does the corrected SPH conduction operator converge to the
Fourier rate as k h -> 0, while the engine stays a factor 1/rho_sph below it at every resolution?

Validates the reference used in u14g_defect_check.py (its Q_ref = 0.91 at k h = 0.94 should be
discretisation, not a missing factor).  Read-only against the repo.
"""
import sys, os, math, json

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (ContinuumParcel, _heat_conduction, _sph_density_sum,
                                             material_properties)
from sigma_ground.dynamics.fluid.kernel import smoothing_length

MAT, T0, DX, NY, NZ = "iron", 300.0, 0.01, 7, 7
props = material_properties(MAT, T0)
kappa0, cp0, rho0 = (props["thermal_conductivity_W_mK"], props["specific_heat_J_kgK"],
                     props["density_kg_m3"])
H = smoothing_length(DX ** 3)
R = DX * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

rows = []
for nlam in (8, 12, 16, 24):
    lam = nlam * DX
    kw = 2 * math.pi / lam
    nx = max(16, nlam + 8)
    ps, idx = [], {}
    for ix in range(nx):
        for iy in range(NY):
            for iz in range(NZ):
                p = ContinuumParcel(MAT, radius=R, temperature=T0 + math.sin(kw * ix * DX),
                                    position=Vec3(ix * DX, iy * DX, iz * DX))
                idx[(ix, iy, iz)] = len(ps); ps.append(p)
    for p in ps:
        p.update_properties()
    _sph_density_sum(ps, H)
    eng = _heat_conduction(ps, H)
    inner = [idx[(ix, 3, 3)] for ix in range(3, nx - 3)]
    an = [-(kappa0 / (rho0 * cp0)) * kw * kw * math.sin(kw * ps[i].position.x) for i in inner]
    e = [eng[i] for i in inner]
    ref = [eng[i] * ps[i].sph_density for i in inner]      # = eq.(17)/c_p, identity shown to 2.5e-13 in main run
    s2 = sum(a * a for a in an)
    q_eng = sum(x * a for x, a in zip(e, an)) / s2
    q_ref = sum(x * a for x, a in zip(ref, an)) / s2
    rho_m = sum(ps[i].sph_density for i in inner) / len(inner)
    rows.append({"lambda_over_dx": nlam, "kh": kw * H, "nx": nx, "Q_engine": q_eng, "Q_reference": q_ref,
                 "rho_sph_mean": rho_m, "Q_engine_times_rho_sph": q_eng * rho_m})
    print(f"lambda={nlam}dx kh={kw*H:.3f}: Q_ref={q_ref:.4f} Q_eng={q_eng:.4e} Q_eng*rho={q_eng*rho_m:.4f}")

# k h -> 0 limit of the corrected operator on this lattice: Brookshaw second moment
# M = sum_j V_j x_j^2 r|W'(r)|/(r^2 + 0.01 h^2), V_j = m/rho_sph (exact operator gives 1 for T = x^2/2)
from sigma_ground.dynamics.fluid.kernel import W as _W, grad_W as _gW
n = 4
m_par = ps[0].mass
rho_inf = sum(m_par * _W(math.sqrt((a*DX)**2 + (b*DX)**2 + (c*DX)**2), H)
              for a in range(-n, n+1) for b in range(-n, n+1) for c in range(-n, n+1))
M = 0.0
for a in range(-n, n+1):
    for b in range(-n, n+1):
        for c in range(-n, n+1):
            if a == b == c == 0:
                continue
            x, y, z = a*DX, b*DX, c*DX
            gx, gy, gz = _gW(x, y, z, H)
            rdg = x*gx + y*gy + z*gz
            M += (m_par / rho_inf) * x * x * (-rdg) / (x*x + y*y + z*z + 0.01*H*H)
M_no_eta = 0.0
for a in range(-n, n+1):
    for b in range(-n, n+1):
        for c in range(-n, n+1):
            if a == b == c == 0:
                continue
            x, y, z = a*DX, b*DX, c*DX
            gx, gy, gz = _gW(x, y, z, H)
            M_no_eta += (m_par / rho_inf) * x * x * (-(x*gx + y*gy + z*gz)) / (x*x + y*y + z*z)
print(f"lattice moment M = {M:.4f} (eta^2 included), {M_no_eta:.4f} (eta^2 = 0); rho_inf = {rho_inf:.2f}")

json.dump({"rows": rows, "lattice_moment_M_kh_to_0": M, "lattice_moment_M_no_eta": M_no_eta,
           "rho_sph_infinite_lattice": rho_inf, "note": "supplementary, not pre-registered; reference = engine x rho_sph,i "
           "(the per-parcel identity established in u14g_results.json D2.a to 2.5e-13)"},
          open(os.path.join(OUT, "u14g_supp_convergence.json"), "w", encoding="utf8"), indent=1)
