"""U6 smoke: discrete monopole conservation of the two source discretisations, static monopole vs Mentat, timing."""
import sys, time
sys.path.insert(0, r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U6")
import numpy as np
import u6_sigma_wave as U

s = U.Source(8000, "cell", "mass", 0.05, 1.0)
sp = U.Source(8000, "point", "mass", 0.05, 1.0)
print("lambda, cell monopole rel. change, point monopole rel. change")
for lam in (0.95, 0.97, 0.999, 1.0, 1.001, 1.03, 1.05):
    mc = s.mass_cell(lam).sum() / s.Mcell0.sum() - 1
    xx = np.maximum(sp.rj / (U.a * lam), 1e-12)
    rho = U.RHO_C * np.where(sp.rj < U.R * lam, np.sin(xx) / xx, 0.0) / lam**3
    rho0 = U.RHO_C * np.where(sp.rj < U.R, np.sin(np.maximum(sp.rj / U.a, 1e-12)) / np.maximum(sp.rj / U.a, 1e-12), 0.0)
    mp = (sp.rj**2 * rho).sum() / (sp.rj**2 * rho0).sum() - 1
    print(lam, mc, mp)
print("M grid (cell)", 4 * np.pi * s.Mcell0.sum(), " M Mentat", U.L["M_kg"]["value"])
print("sum Q (static v_far)", (U.COEF * (U.C**2 * s.Mcell0 - 3 * s.Pcell)).sum(), " v_static Mentat", U.L["v_static_m"]["value"])
t = time.time(); U.run(8000, "cell", "mass", 0.08, nsteps=20000); print("cell mass N8000 20000 steps", round(time.time() - t, 2), "s")
t = time.time(); U.run(16000, "cell", "pressure", 0.08, nsteps=20000); print("cell pressure N16000 20000 steps", round(time.time() - t, 2), "s")
t = time.time(); U.run(16000, "point", "mass", 0.08, nsteps=20000); print("point mass N16000 20000 steps", round(time.time() - t, 2), "s")
