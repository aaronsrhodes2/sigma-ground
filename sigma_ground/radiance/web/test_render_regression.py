"""R0 pixel-regression net — the pytest face of tools/render_regression.py.

DOUBLY environment-gated (skip honestly, never fail on environment — the
test_camera.py node-absent precedent):

* ``SG_RENDER_REGRESSION=1`` must be set: the check opens REAL browser tabs
  (one per baseline) against a real GPU — a default pytest run must never pop
  eleven browser windows.
* the committed baselines manifest must exist for this checkout.

The heavy lifting (server spawn on an EPHEMERAL port, capture, Tier-1 exact /
Tier-2 advisory comparison, verdict table) all lives in the runner; this shim
just surfaces its exit code to pytest.
"""
import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MANIFEST = pathlib.Path(__file__).resolve().parent / "baselines" / "manifest.json"


@pytest.mark.skipif(
    os.environ.get("SG_RENDER_REGRESSION") != "1",
    reason="browser/GPU-gated — opens real browser windows; run "
           "tools/render_regression.py --check, or set SG_RENDER_REGRESSION=1")
@pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason="no render baselines committed for this configuration")
def test_render_baselines_hold():
    r = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "render_regression.py"),
         "--check"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
