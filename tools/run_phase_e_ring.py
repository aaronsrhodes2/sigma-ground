"""Phase E4: the scaled-ring comparison run — real, repeatable, on disk.

    python tools/run_phase_e_ring.py

Runs the same seeded annulus through both engines for two mid-ring orbits
and writes misc/PHASE_E_RING_REPORT.json. Scaled scenario (a disclosed
Choice, nothing pretends to BE Saturn): a 5e15 kg central body — comet-
cluster scale — with a 6-12 km icy annulus, chosen so orbital periods are
hours, not years, at real CODATA G.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sigma_ground.dynamics.phase_e import run_comparison
from sigma_ground.dynamics.seeding import G_CODATA, seed_ring_annulus

RING = dict(central_mass_kg=5.0e15, central_radius_m=800.0,
            r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
            particle_density_kg_m3=920.0,      # water ice — ring matter
            velocity_dispersion_frac=0.01)     # dynamically cold, not perfect
N = 100
SEED = 2026_07_30

bodies = seed_ring_annulus(N, seed=SEED, **RING)
r_mid = 0.5 * (RING["r_inner_m"] + RING["r_outer_m"])
T = 2 * math.pi * math.sqrt(r_mid ** 3 / (G_CODATA * RING["central_mass_kg"]))
v_kep = math.sqrt(G_CODATA * RING["central_mass_kg"] / r_mid)
dt = T / 400.0
n_steps = 800                                  # two mid-ring orbits

print(f"ring: {N} particles, mid-ring period {T/3600:.2f} h, "
      f"dt {dt:.1f} s, {n_steps} steps = 2 orbits")
t0 = time.perf_counter()
report = run_comparison(
    bodies, dt=dt, n_steps=n_steps, sample_every=20,
    material_key='water_ice',                  # the ring matter itself
    sound_speed_m_s=RING["velocity_dispersion_frac"] * v_kep,
    softening_m=50.0,
    label=f"scaled ring annulus N={N} seed={SEED}")
report["choices"].append({"description": "seed geometry (scaled scenario)",
                          "value": {**RING, "n": N, "seed": SEED}})
elapsed = time.perf_counter() - t0

out = os.path.join(os.path.dirname(__file__), "..", "misc",
                   "PHASE_E_RING_REPORT.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

s0, s1 = report["samples"][0], report["samples"][-1]
print(f"done in {elapsed:.1f}s -> {os.path.normpath(out)}")
print(f"{'':14}{'nbody':>14}{'sph':>14}{'dfrac':>10}")
for k in ("e_total_j", "r_med_m", "speed_std_m_s", "nn_mean_m"):
    d = s1["residual"].get("dfrac_" + k, float("nan"))
    print(f"{k:14}{s1['nbody'][k]:14.4e}{s1['sph'][k]:14.4e}{d:10.2%}")
for eng in ("nbody", "sph"):
    drift = (s1[eng]["e_total_j"] - s0[eng]["e_total_j"]) / abs(s0[eng]["e_total_j"])
    print(f"{eng} energy drift over 2 orbits: {drift:+.3%}")
