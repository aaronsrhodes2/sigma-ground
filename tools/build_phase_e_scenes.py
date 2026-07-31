"""Build the Phase E viewer scenes — both scenarios, both engines.

    python tools/build_phase_e_scenes.py

Re-runs each comparison with record_frames=True (same seeds and Choices as
the report tools) and writes four bundles into radiance/web/data/:

    phase_e_ring_nbody / phase_e_ring_sph     (variant slider: engine)
    phase_e_cloud_nbody / phase_e_cloud_sph   (variant slider: engine)
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sigma_ground.dynamics.phase_e import run_comparison
from sigma_ground.dynamics.seeding import (
    G_CODATA, seed_ring_annulus, seed_rotating_cloud)
from sigma_ground.radiance.phase_e_export import (
    comparison_to_bundles, kepler_display_radius)

DATA = os.path.join(os.path.dirname(__file__), "..",
                    "sigma_ground", "radiance", "web", "data")
SEED = 2026_07_30
N = 100

RING = dict(central_mass_kg=5.0e15, central_radius_m=800.0,
            r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
            particle_density_kg_m3=920.0, velocity_dispersion_frac=0.01)
CLOUD = dict(cloud_mass_kg=1.0e12, cloud_radius_m=5000.0,
             omega_rad_s=1.0e-5, particle_density_kg_m3=920.0,
             jitter_frac=0.02)


def _save(bundles):
    for slug, bundle in bundles.items():
        path = os.path.join(DATA, slug + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=1)
        print(f"  -> {os.path.normpath(path)}")


t0 = time.perf_counter()

# ── ring: two mid-ring orbits ──
bodies = seed_ring_annulus(N, seed=SEED, **RING)
r_mid = 0.5 * (RING["r_inner_m"] + RING["r_outer_m"])
T = 2 * math.pi * math.sqrt(r_mid ** 3 / (G_CODATA * RING["central_mass_kg"]))
v_kep = math.sqrt(G_CODATA * RING["central_mass_kg"] / r_mid)
print("ring comparison (2 orbits)…")
rep = run_comparison(
    bodies, dt=T / 400.0, n_steps=800, sample_every=10,
    material_key='water_ice',
    sound_speed_m_s=RING["velocity_dispersion_frac"] * v_kep,
    softening_m=50.0, record_frames=True,
    label=f"scaled ring annulus N={N} seed={SEED}")
_save(comparison_to_bundles(
    rep, bodies, slug_base="phase_e_ring",
    title_base="Phase E ring: fluid vs gravity",
    material_key="water_ice",
    display_radius_m=kepler_display_radius(RING["r_outer_m"])))

# ── cloud: 1.2 free-fall times ──
bodies = seed_rotating_cloud(N, seed=SEED, **CLOUD)
rho = CLOUD["cloud_mass_kg"] / ((4/3) * math.pi * CLOUD["cloud_radius_m"]**3)
t_ff = math.sqrt(3.0 * math.pi / (32.0 * G_CODATA * rho))
c_j = math.sqrt(G_CODATA * CLOUD["cloud_mass_kg"] / CLOUD["cloud_radius_m"])
print("cloud comparison (1.2 t_ff)…")
rep = run_comparison(
    bodies, dt=1.2 * t_ff / 800, n_steps=800, sample_every=10,
    material_key='water_ice', sound_speed_m_s=0.7 * c_j,
    softening_m=100.0, record_frames=True,
    label=f"rotating cloud collapse N={N} seed={SEED}")
_save(comparison_to_bundles(
    rep, bodies, slug_base="phase_e_cloud",
    title_base="Phase E cloud collapse: fluid vs gravity",
    material_key="water_ice",
    display_radius_m=kepler_display_radius(CLOUD["cloud_radius_m"])))

print(f"done in {time.perf_counter() - t0:.1f}s")
