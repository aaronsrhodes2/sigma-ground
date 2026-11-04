"""S-B (C35): body-years per minute of the whole-sky batch engine on this machine, with the full physics (1PN, the 343 sources, the planets), at the main-belt step.

    python scripts/batch_engine_bench.py [--bodies 10000] [--days 365] [--step 1.0]

The bodies are the 343 catalogue asteroids' states, cloned with small random offsets (a benchmark of cost, not of accuracy: the accuracy test is tests/test_batch_particles.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bodies", type=int, default=10000)
    ap.add_argument("--days", type=float, default=365.25)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--no-sources", action="store_true")
    a = ap.parse_args()
    import mpc_live_score as m
    from sigma_ground.field.interface import batch_particles as bp
    from sigma_ground.field.interface.prescribed_sources import PrescribedSources
    e = m.Engine(print)
    base = e.nodes[0]                                                       # (343, 6) at W0
    rng = np.random.default_rng(1)
    reps = int(np.ceil(a.bodies / len(base)))
    y0 = np.tile(base, (reps, 1))[: a.bodies] * (1.0 + 1e-3 * rng.standard_normal((a.bodies, 6)))
    t = e.pp.to_prescribed_sources(e.nodes, e.gm_all, 1.0)
    src = None if a.no_sources else PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)
    t0 = time.time()
    tn, nodes = bp.propagate_batch(y0, m.W0, m.W0 + a.days, e.kernel, e.gmp, sources=src, sources_epoch_jd=m.W0, step_days=a.step)
    dt = time.time() - t0
    by = a.bodies * a.days / 365.25
    print(f"{a.bodies} bodies x {a.days:.0f} d at {a.step} d step{' (no sources)' if a.no_sources else ''}: {dt:.1f} s -> {by / dt * 60:,.0f} body-years per minute; nodes {nodes.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
