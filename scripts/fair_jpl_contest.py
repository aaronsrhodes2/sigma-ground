"""The fair JPL contest: same initial state, different dynamics.

For each JPL vintage V (DE403 ... DE432, kernels at D:/datasets/jpl/vintages) and each start epoch E at or after V's release, take V's OWN state of every body at E
(its forecast of that instant), propagate it with OUR N-body engine, and compare at E + 1..5 yr with
    eV : V's own continuation of that state, against DE440 (what JPL's old dynamics did to its own initial state),
    eO : our propagation, against DE440 (what our dynamics do to the same initial state),
    dyn: our propagation against V's own continuation (how far apart the two dynamics are, given identical initial conditions),
    e0 : the initial offset |V(E) - DE440(E)|, shared by both (neither dynamics can remove it).
If our dynamics are closer to the truth than the old ones, eO < eV; if the same, eO ~ eV; dyn isolates the dynamics disagreement. Truth is DE440s, a fit whose data
end about 2020: start epochs stop at 2015 so every target is <= 2020.

Model (reduced, validated against the full shoot-out): Sun, Mercury, Venus, Earth, Moon, Mars, and the giants and Pluto as SYSTEM barycentres (planet + its moons' mass),
which is what every JPL kernel contains. Same force layers as the shoot-out's headline stack (full EIH, zonal harmonics, tides, solar pressure), same engine, DE440 GM values
(masses are not part of the initial state), fourth-order Forest-Ruth at ``--dt-days``. No asteroids (neither does the shoot-out).

``--validate`` runs the control: start from DE440 itself and compare with the r2 shoot-out's inner-planet and Moon errors, to show the reduced model is faithful.

usage: python scripts/fair_jpl_contest.py --out DIR [--workers 12] [--dt-days 0.25] [--validate] [--shootout r2.json]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

V_DIR = Path("D:/datasets/jpl/vintages")
TRUTH = Path("D:/datasets/jpl/de440s.bsp")
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
KM = 1e3
DAY = 86400.0
CHAINS = {"Sun": ((0, 10),), "Mercury": ((0, 1), (1, 199)), "Venus": ((0, 2), (2, 299)), "Earth": ((0, 3), (3, 399)), "Moon": ((0, 3), (3, 301)),
          "Mars": ((0, 4),), "Jupiter": ((0, 5),), "Saturn": ((0, 6),), "Uranus": ((0, 7),), "Neptune": ((0, 8),), "Pluto": ((0, 9),)}
ORDER = list(CHAINS)
SYSTEMS = ("Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")           # barycentres: planet + its moons


def jd_of(y, m, d):
    import datetime as dt
    return dt.date(y, m, d).toordinal() + 1721424.5


def state(kernel, name, jd):
    """SSB-frame ICRF position (km) and velocity (km/day) of a body in a kernel, at TDB Julian date jd."""
    p, v = 0.0, 0.0
    for c, t in CHAINS[name]:
        pp, vv = kernel[c, t].compute_and_differentiate(jd)
        p, v = p + pp, v + vv
    return np.asarray(p, dtype=float), np.asarray(v, dtype=float)


def pos(kernel, name, jd):
    p = 0.0
    for c, t in CHAINS[name]:
        p = p + kernel[c, t].compute(jd)
    return np.asarray(p, dtype=float)


def build_masses(de: dict, key: str):
    """GM (m^3/s^2) per reduced body from the DE440 fixture, giants and Pluto summed over their moons."""
    snap = de["snapshots"][key]
    gm = {b["name"]: b["gm_km3_s2"] * 1e9 for b in snap["bodies"]}
    anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
    out = {}
    for n in ORDER:
        if n in SYSTEMS:
            out[n] = gm[n] + sum(g for m, g in gm.items() if anchor.get(m) == n)
        else:
            out[n] = gm[n]
    return out


def run_one(args):
    """One propagation: returns {body: {lead_years: error numbers}}. Runs in a worker process."""
    vname, vfile, key, e_jd, dt_days, leads = args[:6]
    toggle_names = args[6] if len(args) > 6 else None
    import sigma_ground.field.interface.rolling_shootout as rs
    from jplephem.spk import SPK
    from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem
    from sigma_ground.field.constants import G
    snap_small = args[7] if len(args) > 7 else None
    de = {"snapshots": {key: snap_small}} if snap_small is not None else json.loads(FIXTURE.read_text())   # workers get one snapshot, not the whole fixture
    kv = SPK.open(str(V_DIR / vfile)) if vfile else SPK.open(str(TRUTH))
    kt = SPK.open(str(TRUTH))
    gm = build_masses(de, key)
    snap = de["snapshots"][key]
    full, names, sun_idx, _ = rs._build_bodies_at_snapshot(key, de, [p for p in rs.canonical_predictors(dt_days) if p.name == "de440_stack"][0])
    byname = {n: b for n, b in zip(names, full)}
    pred = [p for p in rs.canonical_predictors(dt_days) if p.name == "de440_stack"][0]
    bodies, e0 = [], {}
    for n in ORDER:
        p, v = state(kv, n, e_jd)
        base = byname[n]
        if n in SYSTEMS:                                    # a barycentre carries no oblateness or tides of its own here
            cb = CelestialBody(mass_kg=gm[n] / G, position_m=p * KM, velocity_m_s=v * KM / DAY, radius_m=base.radius_m, love_number_k2=0.0,
                               area_m2=base.area_m2, reflectivity=base.reflectivity, j2=0.0, j3=0.0, j4=0.0, pole_axis_unit=base.pole_axis_unit)
        else:
            cb = CelestialBody(mass_kg=gm[n] / G, position_m=p * KM, velocity_m_s=v * KM / DAY, radius_m=base.radius_m,
                               love_number_k2=base.love_number_k2, area_m2=base.area_m2, reflectivity=base.reflectivity, j2=base.j2, j3=base.j3,
                               j4=base.j4, pole_axis_unit=base.pole_axis_unit)
        bodies.append(cb)
        e0[n] = float(np.linalg.norm(p - pos(kt, n, e_jd)))
    asteroid_states = args[8] if len(args) > 8 else None
    if asteroid_states:                                         # heavy asteroids as massive bodies (states from scripts/make_asteroid_states.py)
        for nm, d in asteroid_states.items():
            sx = d["state"]
            bodies.append(CelestialBody(mass_kg=d["gm_km3_s2"] * 1e9 / G, position_m=np.array(sx[:3]) * KM, velocity_m_s=np.array(sx[3:]) * KM / DAY,
                                        radius_m=d["radius_km"] * KM, love_number_k2=0.0, area_m2=0.0, reflectivity=0.0, j2=0.0, j3=0.0, j4=0.0,
                                        pole_axis_unit=np.array([0.0, 0.0, 1.0])))
    sun_i = ORDER.index("Sun")
    toggles = pred.toggles
    if toggle_names is not None:
        import dataclasses
        toggles = type(pred.toggles)(**{f.name: (f.name in toggle_names) for f in dataclasses.fields(pred.toggles) if isinstance(getattr(pred.toggles, f.name), bool)})
    c22 = args[11] if len(args) > 11 else None
    if c22:                                                      # candidate C3a: the Moon's triaxial figure (sectoral C22), long axis locked to the Earth
        from sigma_ground.field.interface.lunar_figure import LockedMoonSystem
        system = LockedMoonSystem(bodies, toggles=toggles, solar_luminosity_W=rs._L_SUN_W if toggles.srp else 0.0, parent_attractor_indices=None,
                                  moon_index=ORDER.index("Moon"), earth_index=ORDER.index("Earth"), c22=float(c22))
    else:
        system = NBodySystem(bodies, toggles=toggles, solar_luminosity_W=rs._L_SUN_W if toggles.srp else 0.0, parent_attractor_indices=None)
    out = {n: {} for n in ORDER if n != "Sun"}
    pole_days = args[9] if len(args) > 9 else None
    rtn_wanted = bool(args[10]) if len(args) > 10 else False
    pole_idx = [(i, n) for i, n in enumerate(ORDER) if pole_days and n in rs._IAU_POLE_MODELS and bodies[i].j2 != 0.0]   # bodies whose zonal axis matters
    pole_every = max(1, int(round(pole_days / dt_days))) if pole_days else 0
    steps_done = 0

    def refresh_poles(jd):
        for i, n in pole_idx:
            system.bodies[i] = system.bodies[i].replace(pole_axis_unit=rs._iau_pole_unit(n, jd))

    t_now = e_jd
    kend = min(kv[0, 3].end_jd, kt[0, 3].end_jd)                 # a vintage kernel can end before the lead does (DE403s ends in 2011)
    for k in leads:
        target = e_jd + k * 365.25
        if target > kend:
            break
        n_steps = max(1, int(round((target - t_now) / dt_days)))
        dts = (target - t_now) * DAY / n_steps
        jd_cur = t_now
        for _ in range(n_steps):
            if pole_every and steps_done % pole_every == 0:          # candidate C2: spin axes follow the IAU models instead of staying fixed at the start epoch
                refresh_poles(jd_cur)
            system.forest_ruth_step(dts)
            steps_done += 1
            jd_cur += dts / DAY
        t_now = target
        sun_p = system.bodies[sun_i].position_m / KM
        for i, n in enumerate(ORDER):
            if n == "Sun":
                continue
            ours = system.bodies[i].position_m / KM - sun_p
            v_own = pos(kv, n, target) - pos(kv, "Sun", target)
            truth = pos(kt, n, target) - pos(kt, "Sun", target)
            out[n][k] = {"eO": float(np.linalg.norm(ours - truth)), "eV": float(np.linalg.norm(v_own - truth)),
                         "dyn": float(np.linalg.norm(ours - v_own)), "e0": e0[n]}
            if n == "Moon" and rtn_wanted:                            # the gap as radial / along-track / cross-track about the Earth, in the vintage's own orbit frame
                pm, vm = state(kv, "Moon", target)
                pe, ve = state(kv, "Earth", target)
                r_g, v_g = pm - pe, vm - ve
                ours_geo = system.bodies[i].position_m / KM - system.bodies[ORDER.index("Earth")].position_m / KM
                rh = r_g / np.linalg.norm(r_g)
                nh = np.cross(r_g, v_g) / np.linalg.norm(np.cross(r_g, v_g))
                th = np.cross(nh, rh)
                d = ours_geo - r_g
                out[n][k]["rtn_km"] = [float(d @ rh), float(d @ th), float(d @ nh)]
    return {"vintage": vname, "start": key, "jd": e_jd, "bodies": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dt-days", type=float, default=0.25)
    ap.add_argument("--validate", action="store_true", help="control: start from DE440 itself")
    ap.add_argument("--shootout", type=Path, default=None, help="r2 rolling shoot-out json for the validation comparison")
    ap.add_argument("--worker", nargs=2, metavar=("JOB", "RESULT"), help=argparse.SUPPRESS)
    ap.add_argument("--asteroids", type=Path, default=None, help="asteroid_perturber_states_v2physics.json: add the heavy asteroids as massive bodies")
    ap.add_argument("--pole-update-days", type=float, default=None, help="candidate C2: refresh the IAU spin axes of the zonal bodies every this many days")
    ap.add_argument("--sample-days", type=float, default=None, help="sample every this many days (residual atlas) instead of yearly; also records the Moon gap as R/T/N")
    ap.add_argument("--lunar-c22", type=float, default=None, help="candidate C3a: add the Moon's C22 figure term with this coefficient (e.g. 2.2382e-5)")
    ap.add_argument("--cases", nargs="*", default=None, help="explicit VINTAGE:start cases, e.g. DE405:j2001 DE421:j2009 (overrides --only-vintage/--starts)")
    ap.add_argument("--only-vintage", nargs="*", default=None)
    ap.add_argument("--starts", nargs="*", default=None, help="restrict start snapshots, e.g. j2005 j2010")
    a = ap.parse_args()
    if a.worker:
        spec = json.loads(Path(a.worker[0]).read_text())
        res = run_one(tuple(spec["args"]))
        tmp = Path(a.worker[1] + ".part")
        tmp.write_text(json.dumps(res))
        tmp.replace(a.worker[1])
        return 0
    from vintage_deviance_mesh import VINTAGES
    de = json.loads(FIXTURE.read_text())
    keys = sorted(k for k in de["snapshots"] if k.startswith("j") and 2000 <= int(k[1:]) <= 2015)
    if a.starts:
        keys = [k for k in keys if k in a.starts]
    ast = json.loads(a.asteroids.read_text()) if a.asteroids else None
    def ast_for(key):
        return {n: {"gm_km3_s2": d["gm_km3_s2"], "radius_km": d["radius_km"], "state": d["states"][key]} for n, d in ast["asteroids"].items()} if ast else None
    leads = [1, 2, 3, 4, 5] if not a.sample_days else [round(i * a.sample_days / 365.25, 9) for i in range(1, int(5 * 365.25 / a.sample_days) + 1)]
    jobs = []
    if a.validate:
        for k in keys:
            jobs.append(("DE440(control)", None, k, de["snapshots"][k]["epoch"]["jd_tdb"], a.dt_days, leads, None, de["snapshots"][k], ast_for(k), a.pole_update_days, bool(a.sample_days), a.lunar_c22))
    else:
        import datetime as dt
        wanted = {tuple(c.split(":")) for c in a.cases} if a.cases else None
        for name, f, rel in VINTAGES:
            if wanted is None and a.only_vintage and name not in a.only_vintage:
                continue
            for k in (sorted({kk for v, kk in wanted if v == name}) if wanted else keys):
                if wanted and (name, k) not in wanted:
                    continue
                jd = de["snapshots"][k]["epoch"]["jd_tdb"]
                if jd >= rel.toordinal() + 1721424.5:
                    jobs.append((name, f, k, jd, a.dt_days, leads, None, de["snapshots"][k], ast_for(k), a.pole_update_days, bool(a.sample_days), a.lunar_c22))
    tag = "validate" if a.validate else "contest"
    a.out.mkdir(parents=True, exist_ok=True)
    part = a.out / f"fair_contest_{tag}.jsonl"                  # one finished propagation per line: a killed run resumes where it stopped
    results = [json.loads(ln) for ln in part.read_text().splitlines() if ln.strip()] if part.exists() else []
    done = {(r["vintage"], r["start"]) for r in results}
    todo = [j for j in jobs if (j[0], j[2]) not in done]
    print(f"{len(jobs)} propagations ({len(done)} already done), {a.workers} workers, dt = {a.dt_days} d", flush=True)
    # One subprocess per propagation, retried on failure: a single killed worker must not take the whole contest down
    # (a ProcessPool breaks entirely when one worker dies; the shoot-out hit silent worker deaths twice).
    import subprocess
    t0 = time.time()
    jobdir = a.out / "jobs"
    jobdir.mkdir(exist_ok=True)
    queue = []
    for j in todo:
        tagname = f"{j[0]}_{j[2]}".replace("(", "").replace(")", "")
        (jobdir / f"{tagname}.job.json").write_text(json.dumps({"args": list(j)}))
        queue.append([j, jobdir / f"{tagname}.job.json", jobdir / f"{tagname}.result.json", 0])
    running, finished = [], 0
    while queue or running:
        while queue and len(running) < a.workers:
            item = queue.pop(0)
            item[3] += 1
            proc = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(item[1]), str(item[2])],
                                    stdout=subprocess.DEVNULL, stderr=open(str(item[1]) + ".err", "a"))
            running.append((proc, item, time.time()))
        time.sleep(5)
        for entry in list(running):
            proc, item, t1 = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            if proc.returncode == 0 and item[2].exists():
                r = json.loads(item[2].read_text())
                results.append(r)
                with open(part, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                finished += 1
                print(f"[{time.time() - t0:6.0f}s] done {item[0][0]}/{item[0][2]} in {time.time() - t1:.0f}s ({finished}/{len(todo)})", flush=True)
            elif item[3] < 3:
                print(f"[{time.time() - t0:6.0f}s] FAILED {item[0][0]}/{item[0][2]} exit code {proc.returncode} after {time.time() - t1:.0f}s, attempt {item[3]}/3; retrying", flush=True)
                queue.insert(0, item)
            else:
                print(f"[{time.time() - t0:6.0f}s] GAVE UP {item[0][0]}/{item[0][2]} exit code {proc.returncode}", flush=True)
    (a.out / f"fair_contest_{tag}.json").write_text(json.dumps({"dt_days": a.dt_days, "results": results}, indent=0))
    # ---- summary ---------------------------------------------------------------------------------------------------------
    summ = {}
    for n in [b for b in ORDER if b != "Sun"]:
        summ[n] = {}
        for k in (1, 2, 3, 4, 5):
            rows = [r["bodies"][n][k] if k in r["bodies"][n] else r["bodies"][n].get(str(k)) for r in results]
            rows = [x for x in rows if x is not None]
            if not rows:
                continue
            summ[n][str(k)] = {"eO_med": statistics.median(x["eO"] for x in rows), "eV_med": statistics.median(x["eV"] for x in rows),
                               "dyn_med": statistics.median(x["dyn"] for x in rows), "e0_med": statistics.median(x["e0"] for x in rows),
                               "ours_wins": sum(1 for x in rows if x["eO"] < x["eV"]), "n": len(rows)}
    (a.out / f"fair_contest_{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    print("\nmedian km at +1 / +3 / +5 yr:   ours eO | vintage eV | dynamics gap | initial offset e0   (ours wins = runs where eO < eV)")
    for n, d in summ.items():
        line = []
        for k in ("1", "3", "5"):
            if k in d:
                x = d[k]
                line.append(f"+{k}: {x['eO_med']:8.2f} | {x['eV_med']:8.2f} | {x['dyn_med']:8.2f} | {x['e0_med']:8.2f} ({x['ours_wins']}/{x['n']})")
        print(f"{n:8s} " + "   ".join(line))
    if a.validate and a.shootout:
        d = json.loads(a.shootout.read_text())
        AU = 149597870.7
        ref = {}
        for s in d["samples"]:
            if s["predictor"] == "de440_stack" and s.get("error_au") is not None:
                k = int(round((s["sample_jd"] - s["window_start_jd"]) / 365.25))
                if k >= 1:
                    ref.setdefault((s["body"], k), []).append(s["error_au"] * AU)
        print("\nvalidation: reduced model vs the full shoot-out headline stack (median km over windows), same start epochs not required")
        for n in ("Mercury", "Venus", "Earth", "Mars", "Moon"):
            for k in (1, 3, 5):
                if (n, k) in ref and str(k) in summ.get(n, {}):
                    print(f"  {n:8s} +{k} yr  reduced {summ[n][str(k)]['eO_med']:8.3f}   full {statistics.median(ref[(n, k)]):8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
