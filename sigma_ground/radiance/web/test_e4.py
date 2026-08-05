"""E4 client stack's headless node tripwire (net.js/audio.js/hud.js).

net.js, audio.js and hud.js keep their pure cores DOM-free precisely so
they load under plain node; test_e4.node.js drives the real SGWorldNet
against an injected fake websocket (the wsFactory seam), the real
deriveVoice/mediumOf gain-and-vacuum truth table, and the real hud.js
data-spec renderer including the NOVEL-spec LLM-composability proof.

Environment-gated (skip, never fail, when node is absent) — the exact
test_camera.py house pattern: the checks need a JS engine, and the
repo's zero-dependency doctrine means we don't vendor one.
"""
import os
import shutil
import subprocess

import pytest

_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "test_e4.node.js")


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not installed — the E4 client-stack "
                           "checks need a JS engine (environment-gated)")
def test_e4_client_stack_via_node():
    r = subprocess.run([shutil.which("node"), _JS],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"E4 node checks failed:\n{r.stdout}\n{r.stderr}")
    assert "passed" in r.stdout
