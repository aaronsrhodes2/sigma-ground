"""U7 — linearised dispersion of the engine's SPH operators on the simple-cubic lattice (s = 2h/3), longitudinal
plane wave along x, uniform rest state P = 0: from continuum_step's summation density, linear Tait pressure and
symmetric pressure force (artificial viscosity omitted: it is off for alpha = 0 and is dissipative, not dispersive):
    omega^2 = (m^2 K / rho0^3) S(k)^2,   S(k) = sum_l dW/dx(r_l) sin(k x_l)
using the engine's own kernel gradient (sigma_ground.dynamics.fluid.kernel.grad_W) and the bulk-tuned mass.
Long-wave limit -> c_Tait = sqrt(K/rho0) when the lattice sum of x dW/dx equals -1/s^3. [DERIVED, numerics]"""
import json, math, sys
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.fluid.kernel import W, grad_W
from sigma_ground.dynamics.continuum import material_properties
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
s = 0.01; h = 1.5 * s
res = {}
for mat in ("water_ice", "iron"):
    pr = material_properties(mat, 300.0); rho0, K = pr["density_kg_m3"], pr["bulk_modulus_Pa"]
    R = 4; L = [(i, j, k) for i in range(-R, R + 1) for j in range(-R, R + 1) for k in range(-R, R + 1)]
    m = rho0 / sum(W(s * math.sqrt(i * i + j * j + k * k), h) for i, j, k in L)
    gx = np.array([grad_W(i * s, j * s, k * s, h)[0] if (i, j, k) != (0, 0, 0) else 0.0 for i, j, k in L])
    xl = np.array([i * s for i, j, k in L])
    ks = np.linspace(1e-3, math.pi / s, 400)
    S = np.array([np.sum(gx * np.sin(kk * xl)) for kk in ks])
    omega = m * math.sqrt(K / rho0 ** 3) * np.abs(S)
    c_t = math.sqrt(K / rho0)
    vph = omega / ks; vg = np.gradient(omega, ks)
    res[mat] = dict(c_tait=c_t, c_L=pr["sound_speed_m_s"], longwave_ratio=float(vph[0] / c_t),
                    sum_x_dWdx_times_s3=float(np.sum(gx * xl) * s ** 3), vg_max_over_c_tait=float(vg.max() / c_t),
                    k_at_vg_max_times_s=float(ks[np.argmax(vg)] * s), vph_max_over_c_tait=float(vph.max() / c_t),
                    k_s=ks[::8].tolist(), vph_over_c=(vph[::8] / c_t).tolist(), vg_over_c=(vg[::8] / c_t).tolist())
json.dump(res, open(f"{OUT}/sph_dispersion.json", "w"), indent=1)
print(json.dumps({m: {k: v for k, v in d.items() if not isinstance(v, list)} for m, d in res.items()}, indent=1))
