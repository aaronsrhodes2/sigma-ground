"""U8 post-hoc check (NOT pre-registered): is the 10-seed mean of |U_engine|/U_Mentat really
below the iid expectation (N-1)/N at N = 256, 512 (2-2.5 SE), or is it right-skew sampling?
200 seeds per N, unsoftened pair sum from phase_e.diagnostics.  Mentat leaves re-fetched."""
import sys, os, json, math
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
import numpy as np
from mentat_client import mentat
from sigma_ground.dynamics.seeding import seed_rotating_cloud
from sigma_ground.dynamics.phase_e import diagnostics

HERE = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8"
G, sG = mentat("lookup_constant", {"name": "G"})
RHO, sR = mentat("density", {"material": "iron"})
out = {"sources": {"G": [G, sG], "rho_iron": [RHO, sR]}, "per_N": {}}
for N in (32, 256, 512):
    R = (3.0 * N / (4.0 * math.pi * RHO)) ** (1.0 / 3.0)
    Um, sU = mentat("gravitational_binding_energy", {"mass_kg": float(N), "radius_m": R})
    rat = []
    for sd in range(1001, 1201):
        b = seed_rotating_cloud(N, cloud_mass_kg=float(N), cloud_radius_m=R, omega_rad_s=0.0,
                                particle_density_kg_m3=RHO, seed=sd)
        u = diagnostics([x.mass_kg for x in b], [x.position_m for x in b], [x.velocity_m_s for x in b], G)["pe_j"]
        rat.append(abs(u) / Um)
    rat = np.array(rat)
    out["per_N"][str(N)] = {"mentat_U": [Um, sU], "n_seeds": len(rat), "mean": float(rat.mean()),
                            "se": float(rat.std(ddof=1) / math.sqrt(len(rat))), "median": float(np.median(rat)),
                            "skew": float(((rat - rat.mean()) ** 3).mean() / rat.std() ** 3),
                            "expected_(N-1)/N": (N - 1) / N,
                            "z": float((rat.mean() - (N - 1) / N) / (rat.std(ddof=1) / math.sqrt(len(rat))))}
    print(N, out["per_N"][str(N)], flush=True)
json.dump(out, open(os.path.join(HERE, "u8_posthoc_ratio.json"), "w", encoding="utf8"), indent=1)
