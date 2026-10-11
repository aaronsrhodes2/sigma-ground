"""Internal bridge to jplephem -- SGEphemeris's exact-DE440 kernel tier.

NOT public API.  SGEphemeris.sg_position() serves a query from the first
tier that holds a value for that epoch, in the order DE440 snapshot >
SSBM forecast > this kernel > Keplerian fit; so this bridge is consulted
off the DE440 snapshot and forecast epochs, ahead of the Kepler fit.

query_jplephem returns what de440s holds: for Mars..Pluto that is the
planetary SYSTEM BARYCENTRE, not the planet centre (SGEphemeris converts
it; see SGEphemeris._jplephem_position).

The external library (jplephem) is not modified in any way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_KERNEL_CACHE: dict = {}
# Golden rule: external-provider datasets live under D:/datasets/, not a
# per-user profile directory -- a fresh machine/clone can find every kernel
# in one place instead of each project re-fetching to its own cache.
_KERNEL_DIR   = Path("D:/datasets/jpl")

# NAIF body codes for JPL DE440, as before 2026-10-10.  Kept for
# scripts/run_long_horizon_sweep.py, which imports it and chains Earth and
# Moon itself through kernel[3, code]; other bodies there use kernel[0, code],
# so Mercury and Venus stay barycentre codes 1 and 2.  SGEphemeris uses
# _NAIF_CHAINS below.
_NAIF_CODES: dict[str, int] = {
    "Sun":     10,
    "Mercury":  1,   # Mercury barycentre
    "Venus":    2,
    "Earth":    399,
    "Moon":     301,
    "Mars":     4,
    "Jupiter":  5,
    "Saturn":   6,
    "Uranus":   7,
    "Neptune":  8,
    "Pluto":    9,
}

# NAIF segment chains for JPL DE440s: a body's SSB position is the sum of the
# segments along its chain.  de440s holds SSB -> planetary-system barycentres
# (1-9), SSB -> Sun (10), EMB (3) -> Earth (399) and Moon (301), and the
# trivial Mercury/Venus barycentre -> centre segments.  It has no planet-centre
# segments for Mars..Pluto, so those are SYSTEM BARYCENTRES.  Measured
# |barycentre - planet centre| against the DE440 fixture (Horizons N99
# centres) over its 27 epochs 2000-2026: Mars < 1 m, Jupiter 4-217 km,
# Saturn 272-308 km, Uranus 352-1163 km, Neptune 74 km, Pluto 2131-2132 km.
_NAIF_CHAINS: dict[str, tuple[tuple[int, int], ...]] = {
    "Sun":     ((0, 10),),
    "Mercury": ((0, 1), (1, 199)),
    "Venus":   ((0, 2), (2, 299)),
    "Earth":   ((0, 3), (3, 399)),
    "Moon":    ((0, 3), (3, 301)),
    "Mars":    ((0, 4),),
    "Jupiter": ((0, 5),),
    "Saturn":  ((0, 6),),
    "Uranus":  ((0, 7),),
    "Neptune": ((0, 8),),
    "Pluto":   ((0, 9),),
}
_CHAIN_BY_LOWER = {k.lower(): v for k, v in _NAIF_CHAINS.items()}


def _load_kernel(name: str = "de440s"):
    if name in _KERNEL_CACHE:
        return _KERNEL_CACHE[name]
    from jplephem.spk import SPK
    candidates = [
        _KERNEL_DIR / f"{name}.bsp",
        Path(f"{name}.bsp"),
    ]
    for path in candidates:
        if path.exists():
            kernel = SPK.open(str(path))
            _KERNEL_CACHE[name] = kernel
            return kernel
    raise FileNotFoundError(
        f"jplephem kernel {name}.bsp not found. "
        f"Download from https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ "
        f"and place in {_KERNEL_DIR}."
    )


def query_jplephem(body_name: str, jd: float) -> np.ndarray | None:
    """Return SSB-equatorial position (km) via jplephem, or None on failure."""
    try:
        kernel = _load_kernel("de440s")
    except Exception:
        try:
            kernel = _load_kernel("de440")
        except Exception:
            return None

    chain = _CHAIN_BY_LOWER.get(str(body_name).lower())
    if chain is None:
        return None

    try:
        # jplephem compute() returns the position in km.  Earth and Moon need
        # two segments (SSB -> EMB -> body): kernel[0, 399] does not exist.
        pos = np.zeros(3)
        for center, target in chain:
            pos = pos + np.asarray(kernel[center, target].compute(jd), dtype=float)
        return pos
    except Exception:
        return None
