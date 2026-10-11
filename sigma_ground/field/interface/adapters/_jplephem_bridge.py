"""Internal bridge to jplephem -- used only by SGEphemeris as tier-4 fallback.

NOT public API.  SGEphemeris.sg_position() calls this only after all
internal tiers (DE440 fixture, SSBM forecast, Keplerian) have failed.

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

# NAIF segment chains for JPL DE440s: a body's SSB position is the sum of the
# segments along its chain.  de440s holds SSB -> planetary-system barycentres
# (1-9), SSB -> Sun (10), EMB (3) -> Earth (399) and Moon (301), and the
# trivial Mercury/Venus barycentre -> centre segments.  It has no planet-centre
# segments for Mars..Pluto, so those are SYSTEM BARYCENTRES (the planet centre
# is within ~300 km for Saturn; see the fixture's bary_to_center_offsets).
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
