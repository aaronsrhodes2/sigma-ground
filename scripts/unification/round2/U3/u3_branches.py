"""U3 branches, lockstep — continues one seed from its saved relaxed state (seed<S>_relax.npz written by u3_worker.py).

Same physics as u3_worker.py phase 1/2 (same engine, dt, eps, deltas, diagnostics), but the five branches advance in
lockstep chunks so every branch always has the same length: the pre-registered window (2374 steps) completes for all
five at once, and the extension can be stopped at any chunk boundary (create a file named STOP in this folder) with all
branches equally long. The relaxed bodies are rebuilt from the saved positions/velocities (masses M/N, radius 1, k2 0),
which is the whole state of a Newtonian NBodySystem; the rebuild is checked against the saved diagnostics.

Usage (from the sigma-ground repo root):  python -X utf8 u3_branches.py SEED
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]           # worker reads SEED from argv[1]; no smoke
sys.path.insert(0, HERE)
import u3_worker as W                            # noqa: E402  (imports constants, sampler, diag; does not run __main__)
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem  # noqa: E402

SEED = W.SEED
CHUNK = 125
z = np.load(os.path.join(HERE, f"seed{SEED}_relax.npz"))
relax = z["rec"]; pos = z["pos"]; vel = z["vel"]
base = [CelestialBody(W.M / W.N, pos[i].copy(), vel[i].copy(), 1.0, 0.0) for i in range(W.N)]
chk = W.diag(NBodySystem(base, softening_m=W.EPS))
assert np.allclose(np.array(chk), relax[-1], rtol=1e-12, atol=0), (chk, relax[-1])
m = np.array([x.mass_kg for x in base]); vc = (m[:, None] * vel).sum(0) / m.sum()
K0 = relax[-1, 0]; E_before = relax[-1, 0] + relax[-1, 1]

systems, recs = {}, {}
for d in W.DELTAS:
    bodies = [x.replace(velocity_m_s=vc + (1.0 + d) * (x.velocity_m_s - vc)) for x in base]
    systems[d] = NBodySystem(bodies, softening_m=W.EPS)
    recs[d] = [W.diag(systems[d])]

T0 = time.time(); n = 0; pre_written = False
jl_path = os.path.join(HERE, f"seed{SEED}_branches.jsonl")
if os.path.exists(jl_path):
    os.remove(jl_path)
while n < W.N_EXT:
    if os.path.exists(os.path.join(HERE, "STOP")) and n >= W.N_PRE:
        print(f"STOP file seen at n = {n}", flush=True); break
    nstep = min(CHUNK, W.N_EXT - n)
    if n < W.N_PRE:
        nstep = min(nstep, W.N_PRE - n)
    for d in W.DELTAS:
        s = systems[d]
        for _ in range(nstep):
            s.step(W.dt); recs[d].append(W.diag(s))
    n += nstep
    for d in W.DELTAS:
        np.savez(os.path.join(HERE, f"seed{SEED}_branch_{d:+.2f}.npz"), rec=np.array(recs[d]), fields=np.array(W.FIELDS))
    r0 = recs[0.0][-1]
    print(f"[seed {SEED}] n = {n} ({n*W.dt/W.tr:.2f} t_relax) 2K/|W|(0) = {2*r0[0]/abs(r0[2]):.4f} "
          f"{(time.time()-T0)/(5*n):.3f} s/step", flush=True)
    if n >= W.N_PRE and not pre_written:
        with open(jl_path, "w", encoding="utf-8") as jl:
            for d in W.DELTAS:
                rec = recs[d]
                row = dict(seed=SEED, delta=d, K0=K0, E_before=E_before, E_after=rec[0][0] + rec[0][1],
                           dE_formula=((1 + d) ** 2 - 1) * K0, dE_measured=rec[0][0] + rec[0][1] - E_before,
                           E_end_pre=rec[W.N_PRE][0] + rec[W.N_PRE][1], wall_s=round(time.time() - T0))
                jl.write(json.dumps(row) + "\n")
        pre_written = True
        print(f"PRE DONE seed {SEED} wall {time.time()-T0:.0f}s", flush=True)
print(f"DONE seed {SEED} n = {n} wall {time.time()-T0:.0f}s", flush=True)
