"""camera.js's headless math tripwire — the node test its header promises.

camera.js was written with ZERO DOM listeners and ZERO fetch precisely so
it loads under plain node; test_camera.node.js drives the real SGCamera
against an independently rewritten float64 reference (transition
continuity to 1e-9 relative, the distance-proportional speed law, the
step-cap no-tunnel invariant, cam_move merge/decay, orbit-mode
bit-identity with the E3 formulas, and follow's live-pose contract).

Environment-gated (skip, never fail, when node is absent) — the same
honesty class as the network-gated ephemeris test: the checks need a JS
engine, and the repo's zero-dependency doctrine means we don't vendor one.
"""
import os
import shutil
import subprocess

import pytest

_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "test_camera.node.js")


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not installed — camera.js math checks "
                           "need a JS engine (environment-gated)")
def test_camera_controller_math_via_node():
    r = subprocess.run([shutil.which("node"), _JS],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"camera.js node checks failed:\n{r.stdout}\n{r.stderr}")
    assert "passed" in r.stdout
