"""Double-run state-hash equality — the determinism gate.

Every integrator in this repo requires fixed dt for symplecticity (the
n-body class docstring says so explicitly), and the whole engine plan
depends on deterministic replay (save = {seed, command log}; replay must
reproduce bit-identically). This module is the proof: build the same
scenario twice from the same seed, step both the same number of times,
and assert the resulting state is EXACT — not close, not statistically
similar. Serial pure-Python/numpy arithmetic with no threads and no
nondeterministic reduction order is bit-reproducible by construction; if
it ever isn't, something (uncontrolled RNG, iteration-order-dependent
dict/set use, an accidental thread) has broken that guarantee, and this
is the tripwire.

    python -m sigma_ground.engine.bench.determinism
"""
from __future__ import annotations

import hashlib
import struct


def _floats_to_hash(floats) -> str:
    """A stable, order-sensitive hash over a flat sequence of floats."""
    h = hashlib.sha256()
    for x in floats:
        h.update(struct.pack("<d", float(x)))
    return h.hexdigest()


def _state_stacksim(sim):
    out = []
    for i in range(sim.n):
        out += [sim.dx[i], sim.dz[i], sim.vx[i], sim.vz[i]]
    return out


def _state_rigid(scene):
    out = []
    for p in scene.parcels:
        out += [p.position.x, p.position.y, p.position.z,
               p.velocity.x, p.velocity.y, p.velocity.z]
    return out


def _state_nbody(sys_):
    out = []
    for b in sys_.bodies:
        out += list(b.position_m) + list(b.velocity_m_s)
    return out


def _state_continuum(scene):
    out = []
    for p in scene.parcels:
        out += [p.position.x, p.position.y, p.position.z,
               p.velocity.x, p.velocity.y, p.velocity.z,
               p.temperature]
    return out


def check_stacksim(steps=300):
    from .harness import _build_stacksim
    hashes = []
    for _ in range(2):
        sim = _build_stacksim()
        for _ in range(steps):
            sim.step(1.0 / 960.0)
        hashes.append(_floats_to_hash(_state_stacksim(sim)))
    return hashes[0] == hashes[1], hashes


def check_rigid(n=10, steps=100):
    from sigma_ground.dynamics import stepper
    from .harness import _build_rigid_scene
    hashes = []
    for _ in range(2):
        scene = _build_rigid_scene(n, seed=7)
        for _ in range(steps):
            stepper.step(scene, dt=1.0 / 960.0, sub_steps=1)
        hashes.append(_floats_to_hash(_state_rigid(scene)))
    return hashes[0] == hashes[1], hashes


def check_nbody(n=20, steps=50):
    from .harness import _build_nbody
    hashes = []
    for _ in range(2):
        sys_ = _build_nbody(n, seed=7)
        for _ in range(steps):
            sys_.step(1.0)
        hashes.append(_floats_to_hash(_state_nbody(sys_)))
    return hashes[0] == hashes[1], hashes


def check_continuum(n=20, steps=20):
    from sigma_ground.dynamics.continuum import continuum_step
    from .harness import _build_continuum
    hashes = []
    for _ in range(2):
        scene = _build_continuum(n, seed=7)
        for _ in range(steps):
            continuum_step(scene, 1.0)
        hashes.append(_floats_to_hash(_state_continuum(scene)))
    return hashes[0] == hashes[1], hashes


CHECKS = {
    "stacksim": check_stacksim,
    "rigid_stepper": check_rigid,
    "nbody": check_nbody,
    "continuum": check_continuum,
}


def main():
    all_ok = True
    for name, fn in CHECKS.items():
        ok, hashes = fn()
        status = "DETERMINISTIC" if ok else "*** NON-DETERMINISTIC ***"
        print(f"{name:16} {status}  ({hashes[0][:12]}... vs {hashes[1][:12]}...)")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)
    print("\nall backends deterministic")


if __name__ == "__main__":
    main()
