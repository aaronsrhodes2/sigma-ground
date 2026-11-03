"""Where the big local data sets live (the JPL kernel and Horizons copies), the same way on this PC, in a container and on skippy-core.

Order of lookup: the environment variable ``SIGMA_DATASETS`` (set it to the data root; the Mentat container on skippy-core sets ``/datasets``), then ``D:/datasets`` (the PC), ``/srv/datasets`` (skippy-core itself),
``/datasets``. A path is used only if it exists, so a tool can say "the kernel is not installed here" instead of failing on a made-up path.

The data are mirrored from the PC to skippy-core by SkippyTel's tools/sync_datasets.sh (rsync over the LAN).
"""
from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = ("D:/datasets", "/srv/datasets", "/datasets")
KERNEL_NAME = "de440s.bsp"
TRUTH_KERNEL_NAME = "de442/de442.bsp"      # JPL's latest fit (May 2024): the "reality" every prediction is scored against (2026-11-02)


def datasets_root() -> Path | None:
    env = os.environ.get("SIGMA_DATASETS")
    for c in ([env] if env else []) + list(_CANDIDATES):
        if c and Path(c).is_dir():
            return Path(c)
    return None


def jpl_dir() -> Path | None:
    root = datasets_root()
    d = root / "jpl" if root else None
    return d if d is not None and d.is_dir() else None


def de440_kernel() -> Path | None:
    """The DE440s kernel (positions 1849-2150) if it is installed here, else None.

    DE440 is the model's START state and the source of the perturber paths (the fixtures and the sealed forecast were built from it); it is not the truth any more, see truth_kernel()."""
    d = jpl_dir()
    f = d / KERNEL_NAME if d else None
    return f if f is not None and f.is_file() else None


def truth_kernel() -> tuple[Path | None, str]:
    """The kernel predictions are SCORED against, and its name: ``SIGMA_TRUTH_KERNEL`` if set, else DE442 (jpl/de442/de442.bsp, JPL's May 2024 fit with four more years of
    Mars and Juno ranging), else DE440 with the name saying so, else (None, "none").

    DE442 - DE440 heliocentric: inner planets <= 0.02 km over 2000-2026, Mars <= 0.08, Jupiter 3-15, Uranus 364 -> 1141, Neptune 60 -> 336 km. So the inner system is known
    to tens of metres and an outer-planet error below that spread cannot be told from the truth's own uncertainty."""
    env = os.environ.get("SIGMA_TRUTH_KERNEL")
    if env:
        p = Path(env)
        return (p, p.stem.upper()) if p.is_file() else (None, "none")
    d = jpl_dir()
    if d is not None and (d / TRUTH_KERNEL_NAME).is_file():
        return d / TRUTH_KERNEL_NAME, "DE442"
    f = de440_kernel()
    return (f, "DE440 (fallback: DE442 not installed)") if f else (None, "none")
