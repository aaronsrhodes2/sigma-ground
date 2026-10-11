"""Tests for the one-command voice launcher (stage/gmod/holodeck/play.py), its shell wrapper, the addon packaging script and the
fidelity tool's bash lookup.

Everything is faked: no game, no microphone, no network, no model. The control script (`play.exp_run.ctl`), the bridge and the
recogniser are replaced.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOLODECK = ROOT / "stage" / "gmod" / "holodeck"
GMOD = ROOT / "stage" / "gmod"
sys.path.insert(0, str(HOLODECK))
sys.path.insert(0, str(GMOD / "experiments"))

import fidelity  # noqa: E402
import play as P  # noqa: E402


class Done:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout = rc, out


class FakeSrv:
    def __init__(self):
        self.down = False

    def shutdown(self):
        self.down = True

    def server_close(self):
        pass


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A fake game. ctl("start") records what tool.txt held, and (like the game's addon) answers by writing tool_skippy_play.json;
    `state["consume"]` says whether the addon also deletes tool.txt, `state["answer"]` whether it answers at all."""
    data = tmp_path / "mentat"
    data.mkdir()
    monkeypatch.setattr(P, "DATA", data)
    monkeypatch.setattr(P, "CONSOLE_LOG", tmp_path / "console.log")
    calls = []
    state = {"running": False, "armed": None, "srv": FakeSrv(), "consume": True, "answer": True, "tool_result": {}, "polls": 1,
             "tool_present_while_running": []}

    def ctl(*args):
        calls.append(args)
        if args[0] == "status":
            if state["running"]:
                state["tool_present_while_running"].append((data / "tool.txt").exists())
                state["polls"] -= 1
                if state["polls"] < 0:
                    state["running"] = False
            return Done(0 if state["running"] else 1)
        if args[0] == "start":
            state["armed"] = json.loads((data / "tool.txt").read_text(encoding="utf-8")) if (data / "tool.txt").exists() else None
            state["running"] = True
            if state["consume"]:
                (data / "tool.txt").unlink(missing_ok=True)
            if state["answer"]:
                (data / "tool_skippy_play.json").write_text(json.dumps(state["tool_result"]), encoding="utf-8")
            return Done(0)
        if args[0] == "stop":
            state["running"] = False
        return Done(0)

    monkeypatch.setattr(P.exp_run, "ctl", ctl)
    monkeypatch.setattr(P.skippy_bridge, "serve", lambda host, port: state["srv"])
    monkeypatch.setattr(P, "tailscale_ip", lambda: "100.1.2.3")
    monkeypatch.setattr(P, "preflight", lambda text_only, host: {"problems": [], "notes": [], "mic": False, "device": "", "alt": "", "host": host, "whisper": True})
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    monkeypatch.setattr(P, "wait_for", lambda path, seconds, alive=None: path.exists())
    monkeypatch.delenv("SKIPPY_BRIDGE_HOST", raising=False)
    monkeypatch.delenv("GMOD_EXTRA_ARGS", raising=False)
    return data, calls, state


# ---------------------------------------------------------------------------------------------- preflight
def test_preflight_reports_each_missing_piece(monkeypatch):
    def refuse(url, timeout=4.0):
        raise OSError("refused")
    monkeypatch.setattr(P, "fetch_json", refuse)
    monkeypatch.setattr(P.exp_run, "ctl", lambda *a: Done(0))                  # a game is already running
    monkeypatch.setattr(P.shutil, "which", lambda n: None)
    found = P.preflight(False, "")
    assert found["whisper"] is False and found["mic"] is False
    assert any("Tailscale" in p for p in found["problems"]) and any("already running" in p for p in found["problems"])
    assert any("SkippyTel" in n for n in found["notes"]) and any("Ollama" in n for n in found["notes"]) and any("ffmpeg" in n for n in found["notes"])


def test_preflight_picks_the_first_microphone_and_is_clean_when_everything_is_up(monkeypatch):
    monkeypatch.setattr(P, "fetch_json", lambda url, timeout=4.0: {"models": [{"name": P.skippy_bridge.MODEL}]} if "tags" in url else {"status": "ok"})
    monkeypatch.setattr(P.exp_run, "ctl", lambda *a: Done(1))
    monkeypatch.setattr(P.shutil, "which", lambda n: "ffmpeg")
    monkeypatch.setattr(P.skippy_ears, "dshow_audio_devices", lambda: [("Mic One", "alt1"), ("Mic Two", "")])
    found = P.preflight(False, "100.1.2.3")
    assert found["problems"] == [] and found["mic"] and found["device"] == "Mic One" and found["alt"] == "alt1" and found["whisper"]
    assert P.preflight(True, "100.1.2.3")["mic"] is False, "text-only never looks for a microphone"


# ---------------------------------------------------------------------------------------------- arming and cleaning the tool file
def test_the_tool_is_armed_with_the_bridge_and_a_fresh_token_and_nothing_is_left_behind(world):
    data, calls, state = world
    assert P.main(["--text-only"]) == 0
    armed = state["armed"]
    assert armed["script"] == "mentat/tools/skippy_play.lua" and armed["args"]["bridge"].startswith("http://100.1.2.3:")
    assert len(armed["args"]["token"]) >= 16 and armed["args"]["key"] == "x" and armed["args"]["debug"] is False
    assert not (data / "tool.txt").exists()
    assert state["srv"].down, "the bridge is shut down when the game closes"


def test_the_token_file_is_deleted_as_soon_as_the_game_has_answered_while_it_keeps_running(world):
    data, calls, state = world
    state["consume"] = False                                                   # a game that answered but left tool.txt in place
    state["polls"] = 3
    assert P.main(["--text-only"]) == 0
    assert state["tool_present_while_running"] and not any(state["tool_present_while_running"]), \
        "tool.txt (and the token in it) must be gone before the session loop, not only at exit"


def test_the_token_file_is_removed_when_the_start_crashes(world, monkeypatch):
    data, calls, state = world

    def boom(*a):
        if a[0] == "start":
            raise RuntimeError("the control script exploded")
        return Done(1)
    monkeypatch.setattr(P.exp_run, "ctl", boom)
    with pytest.raises(RuntimeError):
        P.main(["--text-only"])
    assert not (data / "tool.txt").exists(), "armed before the start, so a crash must still clean it"
    assert state["srv"].down


def test_a_game_that_never_answers_leaves_no_token_file(world):
    data, calls, state = world
    state["answer"], state["consume"] = False, False
    assert P.main(["--text-only"]) == 1
    assert not (data / "tool.txt").exists()
    assert state["srv"].down


def test_a_game_that_reports_an_error_from_the_tool_fails_cleanly(world):
    data, calls, state = world
    state["tool_result"] = {"error": "needs the game client"}
    assert P.main(["--text-only"]) == 1
    assert not (data / "tool.txt").exists()


def test_the_docstring_documents_that_the_token_is_short_lived():
    doc = P.__doc__
    assert "short-lived" in doc and "tool.txt" in doc and "deleted" in doc
    assert "exists only in this process" not in doc, "the old claim was false: the token touched the disk"


# ---------------------------------------------------------------------------------------------- arguments
def test_arguments_reach_the_armed_tool_and_the_game(world):
    data, calls, state = world
    P.main(["--text-only", "--key", "v", "--map", "gm_flatgrass", "--players", "1", "--tts", "sapi"])
    assert state["armed"]["args"]["key"] == "v"
    assert ("start", "gm_flatgrass") in calls and P.skippy_bridge.TTS_MODE == "sapi"
    assert os.environ["GMOD_EXTRA_ARGS"] == "+maxplayers 2", "a listen server needs two slots for voice chat"


def test_a_bad_tts_choice_is_refused_by_the_parser(world):
    with pytest.raises(SystemExit):
        P.main(["--tts", "bogus"])


def test_the_bridge_host_can_be_forced_from_the_environment(world, monkeypatch):
    data, calls, state = world
    monkeypatch.setenv("SKIPPY_BRIDGE_HOST", "100.9.9.9")
    P.main(["--text-only"])
    assert state["armed"]["args"]["bridge"].startswith("http://100.9.9.9:")


def test_a_problem_found_by_preflight_stops_before_anything_starts(world, monkeypatch):
    data, calls, state = world
    monkeypatch.setattr(P, "preflight", lambda t, h: {"problems": ["no Tailscale address"], "notes": [], "mic": False, "device": "", "alt": "", "host": h})
    assert P.main(["--text-only"]) == 1
    assert not any(c[0] == "start" for c in calls) and not (data / "tool.txt").exists()


def test_the_selftest_needs_whisper_and_ffmpeg(world, monkeypatch):
    data, calls, state = world
    monkeypatch.setattr(P, "preflight", lambda t, h: {"problems": [], "notes": [], "mic": False, "device": "", "alt": "", "host": h, "whisper": False})
    assert P.main(["--selftest-voice"]) == 1 and not calls
    monkeypatch.setattr(P, "preflight", lambda t, h: {"problems": [], "notes": [], "mic": False, "device": "", "alt": "", "host": h, "whisper": True})
    monkeypatch.setattr(P.shutil, "which", lambda n: None)
    assert P.main(["--selftest-voice"]) == 1 and not calls


def test_the_selftest_run_passes_its_tts_and_turns_its_verdict_into_the_exit_code(world, monkeypatch):
    data, calls, state = world
    monkeypatch.setattr(P.shutil, "which", lambda n: "ffmpeg")
    seen = []
    monkeypatch.setattr(P, "selftest_voice", lambda request, api, tts="auto": seen.append((request, tts)) or False)
    assert P.main(["--selftest-voice", "add a lamp", "--tts", "elevenlabs", "--stop-after"]) == 1
    assert seen == [("add a lamp", "elevenlabs")]
    assert state["armed"]["args"]["debug"] is True and ("stop",) in calls, "--stop-after closes the game"
    monkeypatch.setattr(P, "selftest_voice", lambda request, api, tts="auto": True)
    assert P.main(["--selftest-voice", "--stop-after"]) == 0


# ---------------------------------------------------------------------------------------------- the self-test verdict
def _verdict(request, got, console, played=True, **kw):
    return P.selftest_verdict(request, got, console, played, **kw)


BUILD = "[holodeck] (for A) Two couches, in a row.\n"


def test_the_selftest_passes_when_the_text_matches_and_the_game_built_what_was_asked():
    ok, why = _verdict("add two couches", "Skippy, add two couches.", BUILD)
    assert ok, why
    assert _verdict("add two couches", "add two couches", "[holodeck] (for A) Added.\n", count_before=0, count_after=2)[0], "counts, when known, are the proof"


def test_a_mishear_fails_even_when_the_game_answered_and_built_something():
    ok, why = _verdict("add two couches", "Skippy, ad to couches", "[holodeck] (for A) Three couches, in a row.\n")
    assert not ok and "heard" in why


@pytest.mark.parametrize("line", [
    "You already have 1 things standing, which is your limit of 1. Clear some first.",
    "That is as many props as I will keep standing (400). Clear some first.",
    "There was nothing in that to add.",
    "I don't have that in my library, so I added nothing.",
    "I could not find room for that.",
    "There is nothing recent to go by.",
])
def test_a_refusal_or_an_error_sentence_is_not_a_build(line):
    ok, why = _verdict("add two couches", "Skippy, add two couches", f"[holodeck] (for A) {line}\n")
    assert not ok and "nothing was built" in why


def test_an_answer_that_never_names_the_thing_asked_for_is_not_a_build():
    ok, _ = _verdict("add two couches", "add two couches", "[holodeck] (for A) Removed a lamp.\n")
    assert not ok


def test_no_new_props_is_not_a_build_even_if_the_game_said_something_cheerful():
    ok, why = _verdict("add two couches", "add two couches", BUILD, count_before=2, count_after=2)
    assert not ok and "nothing was built" in why


def test_no_answer_at_all_fails():
    ok, why = _verdict("add two couches", "add two couches", "")
    assert not ok and "never answered" in why


def test_the_voice_must_have_played():
    ok, why = _verdict("add two couches", "add two couches", BUILD, played=False)
    assert not ok and "voice" in why


def test_asking_for_elevenlabs_fails_when_another_voice_spoke():
    sapi = BUILD + "[holodeck] Skippy's voice is now: sapi\n"
    ok, why = _verdict("add two couches", "add two couches", sapi, tts="elevenlabs")
    assert not ok and "sapi" in why and "elevenlabs" in why
    ok, why = _verdict("add two couches", "add two couches", BUILD, tts="elevenlabs")
    assert not ok and "not reported" in why
    assert _verdict("add two couches", "add two couches", BUILD + "[holodeck] Skippy's voice is now: elevenlabs\n", tts="elevenlabs")[0]
    assert _verdict("add two couches", "add two couches", sapi, tts="auto")[0], "auto accepts either voice"


def test_the_heard_check_ignores_case_punctuation_numerals_plurals_and_the_wake_word():
    assert P.heard_request("Add two couches", "Skippy, add two couches.")
    assert P.heard_request("add 2 couches", "Skippy add two couches")
    assert P.heard_request("add a couch", "hey Skippy, add the couches")
    assert not P.heard_request("add two couches", "Skippy, put a lamp")
    assert not P.heard_request("add two couches", "")
    assert not P.heard_request("add two couches", None)


# ---------------------------------------------------------------------------------------------- shell scripts
def _bash():
    for cand in (os.environ.get("GMOD_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(not BASH, reason="needs bash")
FAKE_GMAD = """#!/usr/bin/env bash
out=""
while [ $# -gt 0 ]; do case "$1" in -out) out="$2"; shift 2;; *) shift;; esac; done
echo archive > "$out"; exit 0
"""


def _fwd(p):
    return str(p).replace("\\", "/")


def _package(tmp_path, env_extra, cwd=None):
    root = tmp_path / "pk"
    (root / "addon" / "lua" / "autorun").mkdir(parents=True, exist_ok=True)
    (root / "addon" / "lua" / "autorun" / "a.lua").write_text("-- x\n", encoding="utf-8")
    shutil.copy(GMOD / "package_addon.sh", root / "package_addon.sh")
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([BASH, _fwd(root / "package_addon.sh"), _fwd(tmp_path / "out")], capture_output=True, text=True, env=env,
                          cwd=str(cwd or tmp_path), stdin=subprocess.DEVNULL, timeout=60)


@needs_bash
def test_gmad_given_as_a_bare_program_name_is_found_on_the_path(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "gmad").write_text(FAKE_GMAD, encoding="utf-8", newline="\n")
    (bindir / "gmad").chmod(0o755)
    r = _package(tmp_path, {"GMAD": "gmad", "PATH": _fwd(bindir) + os.pathsep + os.environ["PATH"]})
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "mentat-stage.gma").read_text().strip() == "archive"


@needs_bash
def test_a_bare_gmad_that_is_not_on_the_path_is_reported_as_missing_even_with_a_file_of_that_name_here(tmp_path):
    (tmp_path / "gmad-not-on-path").write_text(FAKE_GMAD, encoding="utf-8", newline="\n")
    (tmp_path / "gmad-not-on-path").chmod(0o755)
    r = _package(tmp_path, {"GMAD": "gmad-not-on-path"})
    assert r.returncode == 1 and "gmad-not-on-path" in r.stderr and "not found" in r.stderr
    assert "refused" not in r.stderr, "not the misleading 'gmad refused the addon'"


@needs_bash
def test_gmad_given_as_a_path_still_works_and_a_missing_path_is_named(tmp_path):
    fake = tmp_path / "fake_gmad.sh"
    fake.write_text(FAKE_GMAD, encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    assert _package(tmp_path, {"GMAD": _fwd(fake)}).returncode == 0
    r = _package(tmp_path, {"GMAD": _fwd(tmp_path / "nope.exe")})
    assert r.returncode == 1 and "nope.exe" in r.stderr and "GMAD" in r.stderr


@needs_bash
def test_skippy_play_sh_parses_and_hands_over_to_play_py():
    assert subprocess.run([BASH, "-n", _fwd(GMOD / "skippy_play.sh")], capture_output=True, timeout=30).returncode == 0
    assert "holodeck/play.py" in (GMOD / "skippy_play.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------- fidelity.py --verify-densities on Windows
def test_the_density_check_resolves_git_bash_not_the_wsl_launcher(monkeypatch, tmp_path):
    fake = tmp_path / "bash.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("GMOD_BASH", str(fake))
    cmd = fidelity.mentat_density_cmd("wood_oak")
    assert cmd[0] == str(fake)
    assert cmd[1].endswith("tools/mentat_call.sh") and cmd[2] == "density" and json.loads(cmd[3]) == {"material": "wood_oak"}
    assert (ROOT / "tools" / "mentat_call.sh").exists()
    monkeypatch.delenv("GMOD_BASH")
    resolved = fidelity._bash()
    if sys.platform == "win32":
        assert "system32" not in resolved.lower().replace("\\", "/"), "a bare bash on Windows is the WSL stub"
        assert Path(resolved).exists()


def test_verify_densities_says_so_when_the_bash_launcher_itself_failed(monkeypatch, capsys):
    specs = {"densities": {"values": {"wood_oak": {"kg_m3": 770.0}}}}

    class R:
        stdout, stderr, returncode = "", "WSL ERROR: execvpe(/bin/bash) failed", 1
    monkeypatch.setattr(fidelity.subprocess, "run", lambda *a, **k: R())
    assert fidelity.verify_densities(specs) == 1
    out = capsys.readouterr().out
    assert "bash launcher failed" in out and "no answer from Mentat" not in out


def test_verify_densities_compares_what_mentat_answers(monkeypatch, capsys):
    specs = {"densities": {"values": {"wood_oak": {"kg_m3": 770.0}, "glass": {"kg_m3": 2500.0}}}}

    def run(cmd, **k):
        name = json.loads(cmd[3])["material"]

        class R:
            stdout = json.dumps({"result": {"value": 770.0 if name == "wood_oak" else 2400.0}})
            stderr, returncode = "", 0
        return R()
    monkeypatch.setattr(fidelity.subprocess, "run", run)
    assert fidelity.verify_densities(specs) == 1
    out = capsys.readouterr().out
    assert "wood_oak: Mentat 770.0 kg/m3, specs 770.0 ok" in out and "glass" in out and "MISMATCH" in out
