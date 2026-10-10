"""U8 post-hoc check v2 (NOT pre-registered; requested by the physics skeptic).

The 200-seed run (u8_posthoc_ratio.py) gave |U_engine|/U_Mentat above (N-1)/N at all three N
(z = +1.7, +1.2, +1.0; same sign, Stouffer z ~ 2.3).  Is that sampling noise?
Larger samples on a fresh seed range (70001+): N = 32 x 4000, N = 64 x 2000, N = 256 x 500.
Stated before running: if |z| > 3 at any N, the claim "E|U_engine| = (N-1)/N x U_Mentat" fails.
The +/- reported is the standard error of the mean (SD / sqrt(n_seeds)); the per-seed SD is
reported separately.  Unsoftened pair sum from phase_e.diagnostics; Mentat leaves re-fetched.
"""
import sys, os, json, math, time
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
import numpy as np
from mentat_client import mentat
from sigma_ground.dynamics.seeding import seed_rotating_cloud
from sigma_ground.dynamics.phase_e import diagnostics

HERE = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8"
T0 = time.time()
G, sG = mentat("lookup_constant", {"name": "G"})
RHO, sR = mentat("density", {"material": "iron"})
G, RHO = float(G), float(RHO)
out = {"label": "POST HOC, not pre-registered (v2, skeptic-requested)",
       "criterion_stated_before_running": "fail if |z| > 3 at any N against (N-1)/N",
       "pm_is": "standard error of the mean = SD/sqrt(n_seeds)",
       "seed_range_start": 70001,
       "mentat_sources": {"G": {"tool": "lookup_constant", "args": {"name": "G"}, "value": G, "source": sG},
                          "rho_iron": {"tool": "density", "args": {"material": "iron"}, "value": RHO, "source": sR}},
       "per_N": {}}
for N, nseed in ((32, 4000), (64, 2000), (256, 500)):
    R = (3.0 * N / (4.0 * math.pi * RHO)) ** (1.0 / 3.0)
    Um, sU = mentat("gravitational_binding_energy", {"mass_kg": float(N), "radius_m": R})
    Um = float(Um)
    rat = []
    for sd in range(70001, 70001 + nseed):
        b = seed_rotating_cloud(N, cloud_mass_kg=float(N), cloud_radius_m=R, omega_rad_s=0.0,
                                particle_density_kg_m3=RHO, seed=sd)
        u = diagnostics([x.mass_kg for x in b], [x.position_m for x in b], [x.velocity_m_s for x in b], G)["pe_j"]
        rat.append(abs(u) / Um)
    rat = np.array(rat)
    se = float(rat.std(ddof=1) / math.sqrt(len(rat)))
    out["per_N"][str(N)] = {"mentat_U": {"tool": "gravitational_binding_energy",
                                         "args": {"mass_kg": float(N), "radius_m": R}, "value": Um, "source": sU},
                            "n_seeds": nseed, "mean": float(rat.mean()), "sd_per_seed": float(rat.std(ddof=1)),
                            "se_of_mean": se, "expected_(N-1)/N": (N - 1) / N,
                            "z": float((rat.mean() - (N - 1) / N) / se),
                            "fails_|z|>3": bool(abs((rat.mean() - (N - 1) / N) / se) > 3)}
    print(N, {k: v for k, v in out["per_N"][str(N)].items() if k != "mentat_U"},
          f"t={time.time()-T0:.0f}s", flush=True)
zs = [v["z"] for v in out["per_N"].values()]
out["stouffer_z"] = float(sum(zs) / math.sqrt(len(zs)))
out["outcome"] = "FAIL" if any(v["fails_|z|>3"] for v in out["per_N"].values()) else "PASS"
out["runtime_s"] = time.time() - T0
json.dump(out, open(os.path.join(HERE, "u8_posthoc_ratio_v2.json"), "w", encoding="utf8"), indent=1)
print("stouffer", out["stouffer_z"], out["outcome"])
