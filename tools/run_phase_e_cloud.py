"""Phase E4: the rotating-cloud collapse comparison — the textbook case.

    python tools/run_phase_e_cloud.py

Same seed, both engines, ~1.2 free-fall times. The n-body side collapses
pressureless (violent relaxation, caustics); the SPH side has isothermal
pressure at c_s chosen NEAR the Jeans-support scale (c_s² ~ G·M/R), so it
should halt or slow its collapse where the swarm cannot. The residual on
the radial distribution is the whole point — this is the scenario where
the two descriptions are SUPPOSED to part ways dramatically.

Scaled scenario (disclosed Choice): 1e12 kg cloud, 5 km radius — dense
enough that t_ff is hours at real CODATA G, not megayears.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sigma_ground.dynamics.phase_e import run_comparison
from sigma_ground.dynamics.seeding import G_CODATA, seed_rotating_cloud

CLOUD = dict(cloud_mass_kg=1.0e12, cloud_radius_m=5000.0,
             omega_rad_s=1.0e-5,               # mild rotation: ω·t_ff ~ 0.5
             particle_density_kg_m3=920.0,     # icy grains
             jitter_frac=0.02)                 # break perfect symmetry
N = 100
SEED = 2026_07_30

bodies = seed_rotating_cloud(N, seed=SEED, **CLOUD)
rho_mean = CLOUD["cloud_mass_kg"] / ((4.0 / 3.0) * math.pi
                                     * CLOUD["cloud_radius_m"] ** 3)
t_ff = math.sqrt(3.0 * math.pi / (32.0 * G_CODATA * rho_mean))
# near-Jeans sound speed: c_s² ~ G·M/R marks full pressure support; run a
# bit below it so the SPH side still contracts, just far slower
c_jeans = math.sqrt(G_CODATA * CLOUD["cloud_mass_kg"]
                    / CLOUD["cloud_radius_m"])
c_s = 0.7 * c_jeans

n_steps = 800
dt = 1.2 * t_ff / n_steps
print(f"cloud: {N} particles, t_ff {t_ff/3600:.2f} h, c_jeans "
      f"{c_jeans:.3f} m/s, c_s {c_s:.3f} m/s, dt {dt:.1f} s, "
      f"{n_steps} steps = 1.2 t_ff")

t0 = time.perf_counter()
report = run_comparison(
    bodies, dt=dt, n_steps=n_steps, sample_every=20,
    material_key='water_ice', sound_speed_m_s=c_s,
    softening_m=100.0,
    label=f"rotating cloud collapse N={N} seed={SEED}")
report["choices"].append({"description": "seed geometry (scaled scenario)",
                          "value": {**CLOUD, "n": N, "seed": SEED}})
report["facts"].append({"description": "free-fall time", "value": t_ff,
                        "unit": "s"})
elapsed = time.perf_counter() - t0

out = os.path.join(os.path.dirname(__file__), "..", "misc",
                   "PHASE_E_CLOUD_REPORT.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

s0, s1 = report["samples"][0], report["samples"][-1]
print(f"done in {elapsed:.1f}s -> {os.path.normpath(out)}")
print(f"{'':14}{'nbody':>14}{'sph':>14}{'dfrac':>10}")
for k in ("e_total_j", "r_med_m", "speed_std_m_s", "nn_mean_m"):
    d = s1["residual"].get("dfrac_" + k, float("nan"))
    print(f"{k:14}{s1['nbody'][k]:14.4e}{s1['sph'][k]:14.4e}{d:10.2%}")
print(f"collapse: nbody r_med {s0['nbody']['r_med_m']:.0f} -> "
      f"{s1['nbody']['r_med_m']:.0f} m; sph r_med "
      f"{s0['sph']['r_med_m']:.0f} -> {s1['sph']['r_med_m']:.0f} m")
for eng in ("nbody", "sph"):
    drift = (s1[eng]["e_total_j"] - s0[eng]["e_total_j"]) / abs(s0[eng]["e_total_j"])
    print(f"{eng} energy drift over the run: {drift:+.3%}")
