"""Stage 8 — gas expansion and the first law on the repo's own fluid engine (after the 2026-10-10 energy-equation fix).

Setup: a uniform sphere of N₂ gas (ideal EOS, γ from field.interface.gas.heat_capacity_ratio, M from molecular_mass_kg),
at rest, 300 K, released into vacuum — free expansion. No gravity, no conduction (scene.heat_conduction = False, stated).
Predictions decided before the run:
  (1) FIRST LAW: K + U_int is conserved — the PdV work the gas does on itself becomes bulk kinetic energy. Exact to
      roundoff with the pairwise booking (Σ mᵢΔuᵢ = −ΔK for the internal forces).
  (2) ADIABAT: an expanding gas cools as T/T₀ = (ρ/ρ₀)^{γ−1} parcel by parcel, while the kernels still overlap (the engine
      has one global smoothing length, so the test stops at moderate expansion). Viscosity is off for separating pairs, so
      the flow is reversible: the parcel entropy s = c_v ln(T/ρ^{γ−1}) should stay constant up to SPH estimator error.
  (3) ARROW: a colliding gas (two clouds, head-on) must GAIN entropy (viscous/shock heating, Σ m Δs > 0) while still
      conserving K + U_int — the second law appearing on the same engine whose expansion was reversible.
"""
import json, math, os, random
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step, _sph_density_sum
from sigma_ground.field.interface.gas import heat_capacity_ratio, molecular_mass_kg
from sigma_ground.field.constants import N_AVOGADRO, R_GAS

GAM = heat_capacity_ratio("N2", 300.0)
M_MOL = molecular_mass_kg("N2") * N_AVOGADRO
CV = R_GAS / (M_MOL * (GAM - 1.0))

def sphere(N, R, centre=(0.0, 0.0, 0.0), v=(0.0, 0.0, 0.0), seed=4, T=300.0):
    rng = random.Random(seed); ps = []
    while len(ps) < N:
        x, y, z = [rng.uniform(-R, R) for _ in range(3)]
        if x * x + y * y + z * z <= R * R:
            ps.append(ContinuumParcel("water_ice", 0.05, temperature=T,
                                      position=Vec3(centre[0] + x, centre[1] + y, centre[2] + z), velocity=Vec3(*v)))
    return ps

def scene_of(ps, h):
    sc = ContinuumScene(ps, gravity=Vec3(0, 0, 0), ground=False, smoothing_h=h, eos="ideal",
                        ideal_molar_mass_kg_mol=M_MOL, ideal_gamma=GAM)
    sc.heat_conduction = False
    return sc

def entropy(ps, h):
    _sph_density_sum(ps, h)
    return [CV * (math.log(p.temperature) - (GAM - 1) * math.log(p.sph_density)) for p in ps]

out = dict(gamma=GAM, molar_mass=M_MOL, c_v=CV)

# ---- (1) + (2) free expansion -------------------------------------------------------------
N, R = 400, 1.0
h = 1.5 * (4 / 3 * math.pi * R ** 3 / N) ** (1 / 3)
ps = sphere(N, R); sc = scene_of(ps, h)
cs = math.sqrt(GAM * R_GAS * 300.0 / M_MOL); dt = 0.15 * h / cs
_sph_density_sum(ps, h)
rho0 = [p.sph_density for p in ps]; T0 = [p.temperature for p in ps]; s0 = entropy(ps, h)
e0 = sc.energy_ledger(); hist = []
r0 = sorted(math.sqrt(p.position.x ** 2 + p.position.y ** 2 + p.position.z ** 2) for p in ps)[N // 2]
k = 0
while True:
    continuum_step(sc, dt); k += 1
    rmed = sorted(math.sqrt(p.position.x ** 2 + p.position.y ** 2 + p.position.z ** 2) for p in ps)[N // 2]
    if k % 5 == 0:
        e = sc.energy_ledger(); hist.append(dict(t=k * dt, r_med_ratio=rmed / r0, K=e["kinetic"], U=e["internal"], total=e["total"]))
    if rmed > 1.6 * r0 or k > 400:
        break
s1 = entropy(ps, h)
interior0 = [math.sqrt(p.position.x ** 2 + p.position.y ** 2 + p.position.z ** 2) for p in ps]
pairs = [(p.sph_density / r0_, p.temperature / t0_) for p, r0_, t0_ in zip(ps, rho0, T0)]
ds = [b - a for a, b in zip(s0, s1)]
e1 = sc.energy_ledger()
mean_ds = sum(p.mass * d for p, d in zip(ps, ds)) / sum(p.mass for p in ps)
out["expansion"] = dict(N=N, steps=k, r_med_ratio=hist[-1]["r_med_ratio"] if hist else None,
                        K_final=e1["kinetic"], U_int_final=e1["internal"], ledger_drift=e1["total"] - e0["total"],
                        ledger_drift_rel_to_K=(e1["total"] - e0["total"]) / e1["kinetic"],
                        mass_weighted_mean_ds_over_cv=mean_ds / CV, ds_over_cv_min=min(ds) / CV, ds_over_cv_max=max(ds) / CV,
                        rho_T_pairs=pairs[::4], history=hist)

# ---- (3) collision: two clouds head-on ----------------------------------------------------
N2, R2 = 200, 0.6
h2 = 1.5 * (4 / 3 * math.pi * R2 ** 3 / N2) ** (1 / 3)
v_in = 1.5 * cs
ps2 = sphere(N2, R2, centre=(-0.75, 0, 0), v=(v_in, 0, 0), seed=5) + sphere(N2, R2, centre=(0.75, 0, 0), v=(-v_in, 0, 0), seed=6)
sc2 = scene_of(ps2, h2)
dt2 = 0.1 * h2 / (cs + v_in)
s20 = entropy(ps2, h2); e20 = sc2.energy_ledger()
for _ in range(int(0.9 / (v_in * dt2))):
    continuum_step(sc2, dt2)
s21 = entropy(ps2, h2); e21 = sc2.energy_ledger()
dS = sum(p.mass * (b - a) for p, a, b in zip(ps2, s20, s21))
out["collision"] = dict(N=2 * N2, mach=v_in / cs, K0=e20["kinetic"], K1=e21["kinetic"], U_int=e21["internal"],
                        ledger_drift_rel_to_K0=(e21["total"] - e20["total"]) / e20["kinetic"],
                        total_entropy_change_J_per_K=dS, T_max=max(p.temperature for p in ps2))
os.makedirs("misc/unification", exist_ok=True)
json.dump(out, open("misc/unification/stage8_gas_expansion.json", "w"), indent=1)
print(json.dumps({k_: (v if k_ not in ("expansion",) else {kk: vv for kk, vv in v.items() if kk not in ("rho_T_pairs", "history")})
                  for k_, v in out.items()}, indent=1))
