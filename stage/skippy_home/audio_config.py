"""Config for the audio router: which Windows endpoint is 'here', which is 'upstairs', and the groups built from them.

The real file is private (stage/skippy_home/private/audio_config.json, gitignored); audio_config.example.json is the tracked
template. Names, never IP addresses (the PC's address is DHCP). Missing file -> the built-in defaults below, with a warning.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"
CONFIG_PATH = PRIVATE / "audio_config.json"
EXAMPLE_PATH = HERE / "audio_config.example.json"

RATE = 48000

OUTPUT_DEFAULTS = {
    "device": "",              # substring of the Windows endpoint name (see `devices`)
    "hostapi": "WASAPI",       # substring of the host API name
    "latency_ms": 100.0,       # how long after we hand over a sample it is audible (calibrate.py measures this)
    "delay_ms": None,          # explicit extra delay; None = computed (the slowest output gets 0, faster ones wait)
    "lead_in_ms": 250.0,       # near-silent noise before the first samples so a sleeping link wakes before the music
    "volume": 0.4,             # 0..1
    "kind": "wired",           # wired | bluetooth | hdmi: informational, picks defaults when latency is unmeasured
}

DEFAULT_CONFIG = {
    "default_target": "here",
    "outputs": {
        "here": {"device": "EPSON PJ (NVIDIA High Definition Audio)", "kind": "hdmi", "latency_ms": 60, "lead_in_ms": 300},
        "upstairs": {"device": "Speakers (Enclave Audio)", "kind": "bluetooth", "latency_ms": 200, "lead_in_ms": 400},
    },
    "groups": {"everywhere": ["here", "upstairs"]},
    "aliases": {
        "here": ["downstairs", "pc", "desk", "this room", "hdmi", "computer"],
        "upstairs": ["up stairs", "enclave", "bluetooth"],
        "everywhere": ["every where", "all rooms", "whole house", "both", "everything", "all speakers"],
    },
    "api": {"host": "127.0.0.1", "port": 8790},
    "buffer_s": 1.0,           # how far ahead of each output the decoder runs
    "idle_close_s": 45,        # close output streams this long after the queue ends (wake lead-in is paid again)
    "reconnect_s": 5,
    "max_gain_volume": 0.9,    # volume above this is refused (a loud mistake in the house is worse than a quiet one)
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("outputs",):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: Path | str | None = None) -> tuple[dict, list[str]]:
    """Return (config, notes). Notes say what fell back to a default so nothing is silently guessed."""
    notes: list[str] = []
    p = Path(path) if path else CONFIG_PATH
    over: dict = {}
    if p.exists():
        try:
            over = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            raise ValueError(f"config {p} is not valid JSON: {e}") from e
    else:
        notes.append(f"no private config at {p}: using built-in defaults (copy audio_config.example.json there and edit)")
    cfg = _merge(DEFAULT_CONFIG, over)
    if "outputs" in over:
        cfg["outputs"] = copy.deepcopy(over["outputs"])
        cfg["groups"] = copy.deepcopy(over["groups"]) if "groups" in over else {"everywhere": list(over["outputs"])}
    outs = {}
    for name, o in cfg["outputs"].items():
        if not isinstance(o, dict) or not o.get("device"):
            raise ValueError(f"output {name!r} needs a 'device'")
        full = {**OUTPUT_DEFAULTS, **o}
        if "latency_ms" not in o:
            full["latency_ms"] = {"bluetooth": 200.0, "hdmi": 60.0}.get(full["kind"], 40.0)
            notes.append(f"output {name}: latency_ms not set, assuming {full['latency_ms']:.0f} ms for a {full['kind']} link; run calibrate.py")
        if not 0.0 <= float(full["volume"]) <= 1.0:
            raise ValueError(f"output {name}: volume must be 0..1")
        outs[name] = full
    cfg["outputs"] = outs
    for g, members in cfg["groups"].items():
        if g in outs:
            raise ValueError(f"group {g!r} has the same name as an output")
        for m in members:
            if m not in outs:
                raise ValueError(f"group {g!r} names unknown output {m!r}")
    if cfg["default_target"] not in outs and cfg["default_target"] not in cfg["groups"]:
        raise ValueError(f"default_target {cfg['default_target']!r} is not an output or group")
    return cfg, notes


def target_names(cfg: dict, target: str) -> list[str]:
    """Expand a target (output or group name) to output names."""
    if target in cfg["outputs"]:
        return [target]
    if target in cfg["groups"]:
        return list(cfg["groups"][target])
    raise KeyError(f"unknown target {target!r}; known: {sorted(list(cfg['outputs']) + list(cfg['groups']))}")


def effective_delays(cfg_outputs: dict, names: list[str]) -> dict[str, float]:
    """delay_ms per output so that all of `names` are audible together: slowest link gets 0, faster ones wait the difference.
    An explicit delay_ms in config overrides the computed one."""
    if not names:
        return {}
    slowest = max(float(cfg_outputs[n]["latency_ms"]) for n in names)
    out = {}
    for n in names:
        d = cfg_outputs[n].get("delay_ms")
        out[n] = float(d) if d is not None else slowest - float(cfg_outputs[n]["latency_ms"])
    return out


def start_prefix_ms(cfg_outputs: dict, names: list[str]) -> dict[str, float]:
    """Silence (near-silent lead-in) to put in front of the first sample of each output so the music STARTS together on all of them:
    prefix_i + latency_i is the same for all, and prefix_i >= lead_in_i (the link needs time to wake). An explicit delay_ms output
    gets lead_in + delay_ms instead (the Captain's manual override)."""
    auto = [n for n in names if cfg_outputs[n].get("delay_ms") is None]
    big = max((float(cfg_outputs[n]["latency_ms"]) + float(cfg_outputs[n]["lead_in_ms"]) for n in auto), default=0.0)
    out = {}
    for n in names:
        o = cfg_outputs[n]
        out[n] = float(o["lead_in_ms"]) + float(o["delay_ms"]) if o.get("delay_ms") is not None else big - float(o["latency_ms"])
    return out
