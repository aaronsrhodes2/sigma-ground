"""U7 — post-hoc check of the engine's SPH pressure operator against the linearised dispersion (u7_sph_dispersion.py):
impose a small longitudinal standing displacement u_x = U sin(k x) on the 27x11x11 bar (surface-layer IC masses),
evaluate the engine's own density sum, Tait pressure and pressure force once (the same internal functions
continuum_step calls), and read omega^2(k) = -a_x/u_x on interior axis parcels. Compare with
(m^2 K/rho0^3) S(k)^2 and with c_Tait^2 k^2. Static operator check — no time stepping, no viscosity."""
import json, math, sys
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (ContinuumParcel, ContinuumScene, material_properties, _sph_density_sum,
                                             _pressure_from_eos, _sph_pressure_force)
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
disp = json.load(open(f"{OUT}/sph_dispersion.json"))["water_ice"]
mat = "water_ice"; nx, ny, nz = 27, 11, 11; s = 0.01; h = 1.5 * s
pr = material_properties(mat, 300.0); rho0, K = pr["density_kg_m3"], pr["bulk_modulus_Pa"]; c_t = math.sqrt(K / rho0)
masses = np.load(f"{OUT}/masses_bar27_water_ice.npy")
ijk = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
cx, cy, cz = nx // 2, ny // 2, nz // 2
from sigma_ground.dynamics.fluid.kernel import grad_W, W
R = 4; L = [(i, j, k) for i in range(-R, R + 1) for j in range(-R, R + 1) for k in range(-R, R + 1)]
gx = np.array([grad_W(i * s, j * s, k * s, h)[0] if (i, j, k) != (0, 0, 0) else 0.0 for i, j, k in L])
xl = np.array([i * s for i, j, k in L])
mb = rho0 / sum(W(s * math.sqrt(i * i + j * j + k * k), h) for i, j, k in L)

def accel(ks_, U):
    ps = []
    for n_, (i, j, k) in enumerate(ijk):
        x = (i - cx) * s
        p = ContinuumParcel(mat, 0.005, temperature=300.0, position=Vec3(x + U * math.sin(ks_ / s * x), (j - cy) * s, (k - cz) * s))
        p.mass = float(masses[n_]); ps.append(p)
    for p in ps:
        p.update_properties()
    _sph_density_sum(ps, h)
    for p in ps:
        _pressure_from_eos(p)
    a = _sph_pressure_force(ps, h)
    return ps, a

res = {}
for ks_ in (0.2, 0.4, 0.8, 1.2):
    U = 1e-4 * s
    ps0, a0 = accel(ks_, 0.0)
    ps1, a1 = accel(ks_, U)
    rows = []
    for d in range(-5, 6):
        n_ = ijk.index((cx + d, cy, cz)); x = d * s
        u = U * math.sin(ks_ / s * x)
        if abs(math.sin(ks_ / s * x)) < 0.3:
            continue
        rows.append(-(a1[n_].x - a0[n_].x) / u)
    w2 = float(np.median(rows))
    S = float(np.sum(gx * np.sin(ks_ / s * xl)))
    w2_pred = mb ** 2 * K / rho0 ** 3 * S ** 2
    res[f"{ks_:g}"] = dict(omega2_engine=w2, omega2_dispersion=w2_pred, omega2_cTait_k2=c_t ** 2 * (ks_ / s) ** 2,
                           ratio_engine_to_dispersion=w2 / w2_pred, vphase_engine_over_c_tait=math.sqrt(max(w2, 0)) / (ks_ / s) / c_t,
                           spread=float(np.std(rows) / abs(w2)))
json.dump(res, open(f"{OUT}/sph_operator_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
