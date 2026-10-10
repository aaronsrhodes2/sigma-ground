"""U3 worker — one seed: Plummer sphere (design sketch's sampler), relax, then five injection branches from the SAME state.

Engine: sigma_ground.field.interface.nbody.NBodySystem.step (velocity-Verlet, Plummer-softened forces), CelestialBody.replace.
Diagnostics use the SOFTENED Hamiltonian the integrator conserves (NBodySystem.total_energy() is the unsoftened 1/r sum,
a filed PHYSICS_GAP):  U_s = -sum G m_i m_j / sqrt(r^2+eps^2),  W_s = sum r_ij.F_ij = -sum G m_i m_j r^2/(r^2+eps^2)^(3/2).
Lagrange-Jacobi: (1/2) d^2I/dt^2 = 2K + W_s.

Usage (from the sigma-ground repo root):  python -X utf8 u3_worker.py SEED [SMOKE_STEPS]
Writes seed<SEED>_relax.npz, seed<SEED>_branch_<delta>.npz, seed<SEED>_branches.jsonl in this folder.
"""
import json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem  # noqa: E402

L = json.load(open(os.path.join(HERE, "mentat_leaves.json"), encoding="utf-8"))
G = float(L["G"]["value"])                                  # Mentat lookup_constant(G)
TR_OVER_TC = float(L["trelax_over_tcross_N128"]["value"])   # Mentat simplify_expr 0.1 N / ln N at N = 128

SEED = int(sys.argv[1])
SMOKE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N, M, b = 128, 1.0e12, 5000.0
EPS = 0.02 * b
tc = math.sqrt(b ** 3 / (G * M))      # the design's t_cross (sketch definition)
dt = tc / 300
tr = TR_OVER_TC * tc
N_RELAX = 3000 if not SMOKE else SMOKE                       # 10 t_cross
N_PRE = int(3 * tr / dt)                                     # pre-registered branch length (2374)
N_PRE_RUN = N_PRE if not SMOKE else SMOKE
N_EXT = int(6 * tr / dt) if not SMOKE else 2 * SMOKE         # extended branch length (4749)
DELTAS = (0.0, 0.10, -0.10, 0.05, -0.05)
TAG = "smoke_" if SMOKE else ""


def plummer(seed):  # byte-identical to the design's script_sketch sampler (Plummer 1911; Aarseth, Henon & Wielen 1974)
    rng = random.Random(seed); m = M / N; P = []; V = []
    for _ in range(N):
        r = b / math.sqrt(rng.random() ** (-2 / 3) - 1); ct = 2 * rng.random() - 1; st = math.sqrt(1 - ct * ct); ph = 2 * math.pi * rng.random()
        P.append([r * st * math.cos(ph), r * st * math.sin(ph), r * ct])
        ve = math.sqrt(2 * G * M) * (r * r + b * b) ** -0.25
        while True:
            q, g = rng.random(), 0.1 * rng.random()
            if g < q * q * (1 - q * q) ** 3.5:
                break
        v = q * ve; ct = 2 * rng.random() - 1; st = math.sqrt(1 - ct * ct); ph = 2 * math.pi * rng.random()
        V.append([v * st * math.cos(ph), v * st * math.sin(ph), v * ct])
    P, V = np.array(P), np.array(V); P -= P.mean(0); V -= V.mean(0)
    return [CelestialBody(m, P[i], V[i], 1.0, 0.0) for i in range(N)]


IU = np.triu_indices(N, 1)
EYE = np.eye(N, dtype=bool)
FIELDS = ("K", "U_s", "W_s", "I", "Idot", "n_bound", "K_bound", "r_half")


def diag(s):
    pos = np.array([x.position_m for x in s.bodies]); vel = np.array([x.velocity_m_s for x in s.bodies])
    m = np.array([x.mass_kg for x in s.bodies]); Mt = m.sum()
    xc = (m[:, None] * pos).sum(0) / Mt; vc = (m[:, None] * vel).sum(0) / Mt
    x = pos - xc; v = vel - vc
    K = 0.5 * float((m * (v ** 2).sum(1)).sum())
    d = pos[:, None, :] - pos[None, :, :]; r2 = (d ** 2).sum(-1)
    r2p = r2[IU]; mm = np.outer(m, m)[IU]; e2 = EPS * EPS
    U_s = -G * float((mm / np.sqrt(r2p + e2)).sum())
    W_s = -G * float((mm * r2p / (r2p + e2) ** 1.5).sum())
    I = float((m * (x ** 2).sum(1)).sum()); Idot = 2.0 * float((m * (x * v).sum(1)).sum())
    inv = 1.0 / np.sqrt(r2 + e2); inv[EYE] = 0.0
    phi = -G * (inv * m[None, :]).sum(1)
    bound = 0.5 * (v ** 2).sum(1) + phi < 0
    K_b = 0.5 * float((m[bound] * (v[bound] ** 2).sum(1)).sum())
    r_half = float(np.median(np.sqrt((x ** 2).sum(1))))
    return (K, U_s, W_s, I, Idot, int(bound.sum()), K_b, r_half)


def run(s, nsteps, label):
    rec = np.zeros((nsteps + 1, len(FIELDS)))
    rec[0] = diag(s); t0 = time.time()
    for k in range(nsteps):
        s.step(dt)
        rec[k + 1] = diag(s)
        if (k + 1) % 500 == 0:
            E = rec[k + 1, 0] + rec[k + 1, 1]; E0 = rec[0, 0] + rec[0, 1]
            print(f"[seed {SEED} {label}] step {k+1}/{nsteps} 2K/|W|={2*rec[k+1,0]/abs(rec[k+1,2]):.4f} "
                  f"|dE/E|={abs(E/E0-1):.2e} nb={int(rec[k+1,5])} {(time.time()-t0)/(k+1):.3f}s/step", flush=True)
    return rec


if __name__ == "__main__":
    T0 = time.time()
    s = NBodySystem(plummer(SEED), softening_m=EPS)
    relax = run(s, N_RELAX, "relax")
    np.savez(os.path.join(HERE, f"{TAG}seed{SEED}_relax.npz"), rec=relax, fields=np.array(FIELDS), dt=dt, tc=tc, tr=tr,
             pos=np.array([x.position_m for x in s.bodies]), vel=np.array([x.velocity_m_s for x in s.bodies]))
    base = list(s.bodies)
    m = np.array([x.mass_kg for x in base]); vel = np.array([x.velocity_m_s for x in base])
    vc = (m[:, None] * vel).sum(0) / m.sum()
    K0 = relax[-1, 0]; E_before = relax[-1, 0] + relax[-1, 1]
    jl = open(os.path.join(HERE, f"{TAG}seed{SEED}_branches.jsonl"), "a", encoding="utf-8")
    # Phase 1: every branch to the pre-registered length first (so the primary result exists early)
    systems, recs = {}, {}
    for d in DELTAS:
        bodies = [x.replace(velocity_m_s=vc + (1.0 + d) * (x.velocity_m_s - vc)) for x in base]
        sb = NBodySystem(bodies, softening_m=EPS)
        rec = run(sb, N_PRE_RUN, f"delta={d:+.2f} pre")
        systems[d], recs[d] = sb, rec
        np.savez(os.path.join(HERE, f"{TAG}seed{SEED}_branch_{d:+.2f}.npz"), rec=rec, fields=np.array(FIELDS))
        E_after = rec[0, 0] + rec[0, 1]
        row = dict(seed=SEED, delta=d, K0=K0, E_before=E_before, E_after=E_after,
                   dE_formula=((1 + d) ** 2 - 1) * K0, dE_measured=E_after - E_before,
                   E_end_pre=rec[-1, 0] + rec[-1, 1], wall_s=round(time.time() - T0))
        jl.write(json.dumps(row) + "\n"); jl.flush()
        print("BRANCH " + json.dumps(row), flush=True)
    # Phase 2: continue the same five systems (state = bodies, fixed dt) to the extended length
    for d in DELTAS:
        rec2 = run(systems[d], N_EXT - N_PRE_RUN, f"delta={d:+.2f} ext")
        rec = np.vstack([recs[d], rec2[1:]])
        np.savez(os.path.join(HERE, f"{TAG}seed{SEED}_branch_{d:+.2f}.npz"), rec=rec, fields=np.array(FIELDS))
        print(f"EXTENDED seed {SEED} delta {d:+.2f} rows {len(rec)} wall {time.time()-T0:.0f}s", flush=True)
    print(f"DONE seed {SEED} wall {time.time()-T0:.0f}s", flush=True)
