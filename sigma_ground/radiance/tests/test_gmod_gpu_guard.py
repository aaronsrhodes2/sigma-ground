"""The launcher waits for a calm GPU before a hands-off game run (a window starved of GPU time froze with the 'not responding' spinner on 2026-10-12,
while a training run held the GPU at 100%). A played session only gets a warning. Tested with a fake nvidia-smi on the PATH."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GMOD = Path(__file__).resolve().parents[3] / "stage" / "gmod"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _fake_gpu(tmp_path: Path, percent: int) -> dict:
    exe = tmp_path / "nvidia-smi"
    exe.write_text(f"#!/usr/bin/env bash\necho {percent}\n", encoding="utf-8")
    exe.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    return env


def _run(env: dict, call: str, **extra) -> subprocess.CompletedProcess:
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    script = (GMOD / "gmod_ctl.sh").read_text(encoding="utf-8")
    start = script.index("gpu_util()")
    end = script.index("\nstart() {", start)
    code = script[start:end] + "\n" + call + '\necho "rc=$?"\n'
    return subprocess.run([BASH, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)


def test_a_hands_off_run_refuses_when_the_gpu_stays_busy(tmp_path):
    r = _run(_fake_gpu(tmp_path, 95), "gpu_calm 1", GMOD_GPU_MAX=70, GMOD_GPU_WAIT=0)
    assert "rc=3" in r.stdout and "not starting a hands-off run" in r.stderr


def test_a_hands_off_run_starts_when_the_gpu_is_calm(tmp_path):
    r = _run(_fake_gpu(tmp_path, 30), "gpu_calm 1", GMOD_GPU_MAX=70, GMOD_GPU_WAIT=0)
    assert "rc=0" in r.stdout and r.stderr.strip() == ""


def test_a_played_session_is_only_warned(tmp_path):
    r = _run(_fake_gpu(tmp_path, 95), "gpu_calm 0", GMOD_GPU_MAX=70)
    assert "rc=0" in r.stdout and "warning" in r.stderr


def test_the_check_can_be_switched_off_and_a_missing_gpu_is_fine(tmp_path):
    r = _run(_fake_gpu(tmp_path, 100), "gpu_calm 1", GMOD_GPU_MAX=0)
    assert "rc=0" in r.stdout
    empty = {**os.environ, "PATH": str(tmp_path / "nowhere")}                    # no nvidia-smi at all: nothing to wait for
    r2 = subprocess.run([BASH, "-c", "source <(sed -n '/^gpu_util()/,/^start() {/p' " + str(GMOD / "gmod_ctl.sh").replace("\\", "/") + " | sed '$d'); gpu_calm 1; echo rc=$?"],
                        env={**empty, "PATH": str(Path(BASH).parent)}, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert "rc=0" in r2.stdout


def test_start_waits_for_hands_off_runs_only():
    text = (GMOD / "gmod_ctl.sh").read_text(encoding="utf-8")
    assert 'if [ "$quiet" -eq 1 ] || [ "${GMOD_MOUSE:-0}" = "0" ]; then gpu_calm 1 || return $?; else gpu_calm 0; fi' in text
