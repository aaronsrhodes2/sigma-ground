"""U2 repair round - SPH side checks (same seed, scenes and engine as run_sph.py; dt = t_ff/600, T = 1 t_ff).

(1) Cache-invalidation caveat: the alpha = 0 echo with and without clearing continuum_step's velocity-Verlet cache
    (scene._vv_accel / _vv_pairs) at the reversal, for eos = 'isothermal' and eos = 'ideal'.
(2) Cloud size along the forward leg (every 30 steps): median radius, mass-weighted rms radius, 90th-percentile
    radius about the centre of mass, and the mass-weighted mean ln(rho_sph), for alpha = 0 and 1, both EOS.
Only script-side state is touched; no engine file is modified."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
from run_sph import build_scene
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import continuum_step, _sph_density_sum
from sigma_ground.dynamics.phase_e import _sph_state

N_PER = 600


def size_stats(sc, ps, ms, R):
    _, p, _ = _sph_state(sc)
    p = np.array(p)
    r = np.linalg.norm(p - np.average(p, axis=0, weights=ms), axis=1)
    _sph_density_sum(ps, sc.h)
    lnrho = float(np.average([math.log(q.sph_density) for q in ps], weights=ms))
    return {"r_median_R": float(np.median(r) / R), "r_rms_mass_R": float(np.sqrt(np.average(r * r, weights=ms)) / R),
            "r_p90_R": float(np.percentile(r, 90) / R), "mean_ln_rho_sph": lnrho}


def run(S, eos, alpha, invalidate=True, back=True):
    R, tff = S["R"], S["tff"]
    dt = tff / N_PER
    sc = build_scene(S, eos)
    ps = sc.continuum_parcels()
    ms = np.array([p.mass for p in ps])
    _, p0, _ = _sph_state(sc)
    p0 = np.array(p0)
    track = [dict(t_tff=0.0, **size_stats(sc, ps, ms, R))]
    t0 = time.time()
    for k in range(1, N_PER + 1):
        continuum_step(sc, dt, alpha_visc=alpha)
        if k % 30 == 0:
            track.append(dict(t_tff=k / N_PER, **size_stats(sc, ps, ms, R)))
    out = {"eos": eos, "alpha": alpha, "invalidate_cache_at_reversal": invalidate, "n_per_tff": N_PER,
           "size_track": track}
    if back:
        for p in ps:
            p.velocity = Vec3(-p.velocity.x, -p.velocity.y, -p.velocity.z)
        if invalidate:
            sc._vv_accel = None
            sc._vv_pairs = None
        for _ in range(N_PER):
            continuum_step(sc, dt, alpha_visc=alpha)
        out["eps_rec_R"] = C.rms_rel(_sph_state(sc)[1], p0, R)
    out["wall_s"] = time.time() - t0
    print(json.dumps({k: v for k, v in out.items() if k != "size_track"} | {"size_T": track[-1]}), flush=True)
    return out


if __name__ == "__main__":
    S = C.setup(64)
    runs = []
    for eos in ("isothermal", "ideal"):
        for inv in (True, False):
            runs.append(run(S, eos, 0.0, invalidate=inv))
        runs.append(run(S, eos, 1.0, back=False))
    with open(os.path.join(C.HERE, "res_sph_v2_side.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "leaves_G": S["leaves"]["G"],
                   "choices": {k: S[k] for k in ("N", "M", "R", "cs", "h", "eps", "tff")}}, f, indent=1)
    print("done")
