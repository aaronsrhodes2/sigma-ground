"""ms/step vs. entity count, for every step-callable physics backend.

Four backends, four builders, one measurement discipline: median of 5
timed runs (best-of-5 median, not mean — Windows wall-clock noise is
real and one-sided). Every builder is deterministic (seeded), so a
baseline is reproducible up to timing noise alone.

    python -m sigma_ground.engine.bench.harness            # measure, print
    python -m sigma_ground.engine.bench.harness --save      # + write baseline
    python -m sigma_ground.engine.bench.harness --check     # measure, compare
                                                              # to the committed
                                                              # baseline, exit 1
                                                              # on >20% regression

No timing number existed anywhere in the repo before this (confirmed by
grep during the 2026-08-03 audit) — every native-porting decision in the
game-engine plan is gated on what this file measures, not on guesswork.
"""
from __future__ import annotations

import json
import os
import statistics
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.join(_HERE, "baselines")
_BASELINE_PATH = os.path.join(_BASELINE_DIR, "harness_baseline.json")
_REGRESSION_FACTOR = 1.20          # >20% slower than baseline = fail --check

REPS = 5                            # best-of-5 median per measurement


# ── minimal rigid-body material (mirrors tests/test_actuator.py's _ConstDensity) ──
class _ConstDensity:
    def __init__(self, rho=2700.0):
        self._rho = rho

    def density_at_sigma(self, sigma):
        return self._rho


def _time_it(fn, reps=REPS):
    """Median wall-clock seconds for one call to fn(), best-of-`reps`."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ── backend 1: StackSim (radiance.push_off — the planar god-hand stack) ──────
def _build_stacksim():
    from sigma_ground.deckard.scene import build_stack
    from sigma_ground.radiance.push_off import StackSim, make_lookups
    built = build_stack(["1700s table", "plate", "Doughnut"], allow_llm=False)
    mu, e = make_lookups(1)
    sim = StackSim(built, mu_lookup=mu, e_lookup=e)
    sim.vx[1] = 2.2
    return sim


def bench_stacksim(steps=500):
    """Fixed 3-body stack (its real use case is a small stack, many steps,
    not a large-N sweep) — one measurement, not a sweep."""
    sim = _build_stacksim()

    def run():
        s = _build_stacksim()
        for _ in range(steps):
            s.step(1.0 / 960.0)
    ms = 1000.0 * _time_it(run) / steps
    return [{"n": 3, "steps": steps, "ms_per_step": ms}]


# ── backend 2: dynamics.stepper (rigid 6-DOF, all-pairs contacts) ───────────
def _build_rigid_scene(n, seed=0):
    import random
    from sigma_ground.dynamics.parcel import PhysicsParcel
    from sigma_ground.dynamics.scene import PhysicsScene
    from sigma_ground.dynamics.vec import Vec3
    rng = random.Random(seed)
    parcels = []
    for i in range(n):
        x = rng.uniform(-2.0, 2.0)
        z = rng.uniform(-2.0, 2.0)
        y = 1.0 + i * 0.3
        parcels.append(PhysicsParcel(
            0.08, _ConstDensity(2700.0),
            position=Vec3(x, y, z),
            velocity=Vec3(rng.uniform(-0.5, 0.5), 0.0, rng.uniform(-0.5, 0.5)),
            label=f"p{i}"))
    return PhysicsScene(parcels)


def bench_rigid_stepper(ns=(5, 10, 25, 50), steps=100):
    from sigma_ground.dynamics import stepper
    out = []
    for n in ns:
        def run(n=n):
            scene = _build_rigid_scene(n)
            for _ in range(steps):
                stepper.step(scene, dt=1.0 / 960.0, sub_steps=1)
        ms = 1000.0 * _time_it(run) / steps
        out.append({"n": n, "steps": steps, "ms_per_step": ms})
    return out


# ── backend 3: field.interface.nbody (pure gravity, symplectic) ─────────────
def _build_nbody(n, seed=0):
    from sigma_ground.dynamics.seeding import seed_ring_annulus, to_nbody
    bodies = seed_ring_annulus(
        n, seed=seed,
        central_mass_kg=5.0e15, central_radius_m=800.0,
        r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
        particle_density_kg_m3=920.0)
    return to_nbody(bodies, softening_m=50.0)


def bench_nbody(ns=(10, 25, 50, 100), steps=50):
    out = []
    for n in ns:
        def run(n=n):
            sys_ = _build_nbody(n)
            for _ in range(steps):
                sys_.step(1.0)
        ms = 1000.0 * _time_it(run) / steps
        out.append({"n": n + 1, "steps": steps, "ms_per_step": ms})  # +1: central body
    return out


# ── backend 4: dynamics.continuum (SPH, self-gravity + isothermal EOS) ──────
def _build_continuum(n, seed=0):
    from sigma_ground.dynamics.seeding import seed_ring_annulus, to_continuum, G_CODATA
    bodies = seed_ring_annulus(
        n, seed=seed,
        central_mass_kg=5.0e15, central_radius_m=800.0,
        r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
        particle_density_kg_m3=920.0)
    return to_continuum(bodies, material_key='water_ice', G=G_CODATA,
                        gravity_softening_m=50.0, eos="isothermal",
                        sound_speed_m_s=1.0)


def bench_continuum(ns=(10, 25, 50, 100), steps=20):
    from sigma_ground.dynamics.continuum import continuum_step
    out = []
    for n in ns:
        def run(n=n):
            scene = _build_continuum(n)
            for _ in range(steps):
                continuum_step(scene, 1.0)
        ms = 1000.0 * _time_it(run) / steps
        out.append({"n": n + 1, "steps": steps, "ms_per_step": ms})
    return out


BACKENDS = {
    "stacksim": bench_stacksim,
    "rigid_stepper": bench_rigid_stepper,
    "nbody": bench_nbody,
    "continuum": bench_continuum,
}


def run_all():
    return {name: fn() for name, fn in BACKENDS.items()}


def save_baseline(results, path=_BASELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


def load_baseline(path=_BASELINE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_regression(new, baseline, factor=_REGRESSION_FACTOR):
    """Compare new results to a committed baseline. Returns a list of
    regression strings (empty = clean). Matches by (backend, n)."""
    regressions = []
    for backend, rows in new.items():
        base_rows = {r["n"]: r["ms_per_step"] for r in baseline.get(backend, [])}
        for row in rows:
            n = row["n"]
            if n not in base_rows:
                continue                            # new n, nothing to compare
            base_ms = base_rows[n]
            new_ms = row["ms_per_step"]
            if base_ms > 0 and new_ms > base_ms * factor:
                regressions.append(
                    f"{backend} n={n}: {new_ms:.4f} ms/step vs baseline "
                    f"{base_ms:.4f} ms/step ({new_ms / base_ms:.2f}x)")
    return regressions


def _print_table(results):
    for backend, rows in results.items():
        print(f"\n{backend}")
        print(f"  {'n':>6}  {'steps':>6}  {'ms/step':>10}")
        for r in rows:
            print(f"  {r['n']:>6}  {r['steps']:>6}  {r['ms_per_step']:>10.4f}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true",
                   help="write results as the committed baseline")
    p.add_argument("--check", action="store_true",
                   help="compare against the committed baseline, exit 1 on regression")
    args = p.parse_args(argv)

    results = run_all()
    _print_table(results)

    if args.save:
        save_baseline(results)
        print(f"\nbaseline written -> {_BASELINE_PATH}")
    if args.check:
        baseline = load_baseline()
        regressions = check_regression(results, baseline)
        if regressions:
            print(f"\nREGRESSIONS (>{int((_REGRESSION_FACTOR - 1) * 100)}% slower):")
            for r in regressions:
                print(f"  {r}")
            raise SystemExit(1)
        print("\nno regressions vs. baseline")
    return results


if __name__ == "__main__":
    main()
