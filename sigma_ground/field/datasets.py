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
    """The DE440s kernel (positions 1849-2150) if it is installed here, else None."""
    d = jpl_dir()
    f = d / KERNEL_NAME if d else None
    return f if f is not None and f.is_file() else None
