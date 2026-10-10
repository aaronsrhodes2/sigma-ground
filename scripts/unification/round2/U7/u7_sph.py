"""U7 — SPH leg: response of a Tait lattice to a displaced parcel, on the repo engine (unmodified).

Usage: python -X utf8 u7_sph.py <material> <nx> <ny> <nz> <nsteps> <tag> <amp_frac> [dt_factor] [alpha_visc] [masses.npy] [point|plane]
  amp_frac = 0 -> reference run (no displacement); else the centre parcel (point) or the whole centre y-z plane
  of parcels (plane: a 1-D plane-wave source, added after the point runs showed near-field reshaping) is
  displaced by amp_frac * s along +x.
Lattice spacing s = 2h/3 (design), h = 1.5 s, s = 1 cm (scale-free: no gravity; a disclosed Choice).
Masses: uniform bulk-tuned (summation density = rho0 in the bulk) unless a masses file is given (surface-layer IC).
eos='tait', gravity 0, ground=False, alpha_visc (default 1, the engine default), dt = cfl_timestep(t=0 scene)*dt_factor.
Every parcel's position and velocity are saved after every step.
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step, cfl_timestep, material_properties
from sigma_ground.dynamics.fluid.kernel import W

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"

def bulk_mass(rho0, s, h):
    R = int(math.ceil(2 * h / s)) + 1
    tot = sum(W(s * math.sqrt(i * i + j * j + k * k), h)
              for i in range(-R, R + 1) for j in range(-R, R + 1) for k in range(-R, R + 1))
    return rho0 / tot

def main():
    a = sys.argv
    material, nx, ny, nz, nsteps, tag, amp = a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6], float(a[7])
    dt_factor = float(a[8]) if len(a) > 8 else 1.0
    alpha_visc = float(a[9]) if len(a) > 9 else 1.0
    mfile = a[10] if len(a) > 10 and a[10] != "none" else None
    source = a[11] if len(a) > 11 else "point"
    s = 0.01; h = 1.5 * s
    props = material_properties(material, 300.0)
    rho0, K, G = props["density_kg_m3"], props["bulk_modulus_Pa"], props["shear_modulus_Pa"]
    mb = bulk_mass(rho0, s, h)
    ijk = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    masses = np.load(mfile) if mfile else np.full(len(ijk), mb)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    parcels = []
    for n_, (i, j, k) in enumerate(ijk):
        x, y, z = (i - cx) * s, (j - cy) * s, (k - cz) * s
        if (source == "point" and (i, j, k) == (cx, cy, cz)) or (source == "plane" and i == cx):
            x += amp * s
        m = float(masses[n_])
        rad = (3.0 * max(m, 1e-300) / (4.0 * math.pi * rho0)) ** (1.0 / 3.0)
        p = ContinuumParcel(material, rad, temperature=300.0, position=Vec3(x, y, z), velocity=Vec3(0, 0, 0))
        p.mass = m          # exact assignment (avoids the r^3 round trip); PhysicsParcel stores mass as a plain attribute
        parcels.append(p)
    sc = ContinuumScene(parcels, gravity=Vec3(0, 0, 0), ground=False, smoothing_h=h, eos="tait")
    dt = cfl_timestep(sc) * dt_factor
    ps = sc.continuum_parcels()
    V = np.zeros((nsteps + 1, len(ps), 3)); X = np.zeros((nsteps + 1, len(ps), 3))
    X[0] = [(p.position.x, p.position.y, p.position.z) for p in ps]
    t0 = time.time()
    for n in range(1, nsteps + 1):
        continuum_step(sc, dt, alpha_visc=alpha_visc)
        V[n] = [(p.velocity.x, p.velocity.y, p.velocity.z) for p in ps]
        X[n] = [(p.position.x, p.position.y, p.position.z) for p in ps]
        if n == 1:
            rho1 = np.array([p.sph_density for p in ps])
            print(f"{tag}: step 1 in {time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(f"{OUT}/sph_{tag}.npz", V=V, X=X, ijk=np.array(ijk), centre=np.array([cx, cy, cz]), rho_step1=rho1,
                        masses=masses)
    meta = dict(material=material, dims=[nx, ny, nz], nsteps=nsteps, s=s, h=h, dt=dt, dt_factor=dt_factor,
                alpha_visc=alpha_visc, amp_frac=amp, source=source, rho0=rho0, K=K, G=G, c_L_cascade=props["sound_speed_m_s"],
                c_tait=math.sqrt(K / rho0), m_bulk=mb, masses_file=mfile, wall_seconds=time.time() - t0,
                T_max_end=max(p.temperature for p in ps), T_min_end=min(p.temperature for p in ps))
    json.dump(meta, open(f"{OUT}/sph_{tag}.json", "w"), indent=1)
    print(f"{tag}: done in {meta['wall_seconds']:.0f}s", flush=True)

if __name__ == "__main__":
    main()
