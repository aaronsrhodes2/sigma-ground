"""Tweaks: the hand-written rules that stand in for model competence, one switch each.

Before the router was fine-tuned, a modest model needed help: regexes that answer a
question without it, a shorter tool list, renamed parameters, hints in the prompt. Each
of those is a tweak. Once a model is TRAINED to do the job, its tweak must be switched
off in the same run, or the two overlap (the classifier answers before the model is asked,
or an alias hides a wrong parameter name the model should have got right) and the run no
longer says what the model can do.

This module is the one place a run's tweaks are decided:

    tweaks.on("cascade.astro")          # call sites ask; True = the tweak is working
    tweaks.configure(profile="trained") # or off=[...], on=[...]; names are checked
    tweaks.snapshot()                   # {"profile", "off", "on"} -- stored in every run record

Profiles (a profile only ever turns off a tweak whose replacement is declared):
    legacy      nothing changed -- every result so far
    trained     what a trained router replaces: the 15 hidden tools are shown, parameters
                are not renamed for it, a made-up parameter is an error instead of a silent
                default. The deterministic cascade stays as the safety net.
    model-only  trained + the whole cascade off: the model faces all 150 questions alone.

`tools/tweak_pairs.py` checks a training set against a profile: a tweak may be off only
where the data teaches the tools it used to emit. Environment (read at import, so
subprocesses agree): MENTAT_TWEAKS=<profile>, MENTAT_TWEAKS_OFF=a,b, MENTAT_TWEAKS_ON=c.
A misspelled name is an error, never a silent no-op.
"""
from __future__ import annotations

import ast
import fnmatch
import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Tweak:
    name: str
    kind: str                     # cascade | surface | args | check | capability | loop
    what: str                     # what it does for a model that cannot do it alone
    covers: tuple[str, ...] = ()  # tools the training data must teach before it may go (more are scanned)
    default: bool = True          # True: working unless switched off; False: opt-in check
    keep_reason: str = ""         # non-empty: stays on even in a trained run, and why


CASCADE_MODULES = {
    "frontier": "frontier_classifier", "new_tools": "new_tools_classifier",
    "refusal": "refusal_classifier", "clarification": "clarification_classifier",
    "conversion": "conversion_classifier", "gr": "gr_classifier",
    "cosmology": "cosmology_classifier", "math": "math_classifier",
    "thermo": "thermo_classifier", "modern": "modern_classifier",
    "classical_intro": "classical_intro_classifier", "optics": "optics_classifier",
    "astro": "astro_classifier", "semantic": "interpreter_demo",
}
# Tools a classifier can reach that its source does not spell out as a string literal.
CASCADE_ALSO = {
    "frontier": ("bekenstein_hawking_entropy", "entanglement_channel", "gravitational_binding_energy",
                 "unruh_temperature", "baryon_vs_disc", "entanglements_to_pop_bubble",
                 "holographic_matching_mass"),
    "new_tools": ("orbital_velocity", "orbital_period", "gravitational_force", "orbital_raise_energy",
                  "nuclear_binding_energy", "coulomb_force", "de_broglie_from_kinetic_energy",
                  "energy_power_time"),
    "refusal": ("<decline>",),
    "clarification": ("request_clarification",),
    "conversion": ("convert_units",),
    # The astro classifier computes in the harness (Wien's law, sqrt(GM/r), pc -> ly); the
    # plans that replace it call these library tools instead.
    "astro": ("named_star", "solar_system_body", "light_travel_time", "blackbody_peak_wavelength",
              "orbital_velocity", "energy_power_time", "blackbody_total_power"),
}
HIDDEN = ("baryon_vs_disc", "bekenstein_hawking_entropy", "coulomb_force",
          "de_broglie_from_kinetic_energy", "energy_power_time", "entanglement_channel",
          "entanglements_to_pop_bubble", "gravitational_binding_energy", "gravitational_force",
          "holographic_matching_mass", "nuclear_binding_energy", "orbital_period",
          "orbital_raise_energy", "orbital_velocity", "unruh_temperature")

# Tools with no inputs whose single possible question IS a benchmark question ("how old is the
# universe?"): the benchmark guard forbids training on them, so they cannot be taught and
# their classifier stays for them. Measured, not assumed: the guard holds 4, 2, 2, 2 and 1
# benchmark instances of these tools; generating 30 phrasings of the last kept none.
GUARD_BLOCKED = ("age_of_universe", "critical_density", "eta_value_report", "mond_a0_constant",
                 "holographic_matching_mass")

REGISTRY: dict[str, Tweak] = {}


def _add(t: Tweak) -> None:
    REGISTRY[t.name] = t


for _g, _m in CASCADE_MODULES.items():
    _add(Tweak(f"cascade.{_g}", "cascade",
               f"{_m}: answers matching questions before the model is asked",
               covers=CASCADE_ALSO.get(_g, ())))   # the pairing tool adds what the source reaches
_add(Tweak("hidden_tools", "surface",
           "HIDDEN_FROM_LLM: 15 tools are left out of the model's tool list and reached only by a classifier",
           covers=HIDDEN))
_add(Tweak("param_aliases", "args",
           "param_aliases.PARAM_ALIASES: renames a made-up parameter (gravity_ms2) to the real one"))
_add(Tweak("body_name_chain", "args",
           "param_aliases.resolve_body_name_chain: body_name='sun' becomes solar_system_body -> mass_kg",
           covers=("solar_system_body", "named_star")))
_add(Tweak("strict_params", "check",
           "a call with a parameter the tool does not have is an error (pydantic would drop it and use the default)",
           default=False))
_add(Tweak("structured_results", "capability",
           "router path reads a symbolic-string or list result (roots, derivatives, integrals, matrices) as the answer, "
           "as the cascade's math executor does, and names a rejected call by its cause. Off = 'returned no value' "
           "(every earlier run). A prerequisite for retiring cascade.math, not a help for the model",
           default=False))
_add(Tweak("pattern_hints", "loop",
           "_PATTERN_HINTS: question phrases mapped to a domain at the top of the chain-mode prompt",
           keep_reason="chain mode only; the router prompt carries no index"))
_add(Tweak("answer_key_hint", "loop",
           "chain-mode fallback prefers the benchmark's expected tool (primary_tool_expected): the answer key",
           keep_reason="chain mode only; measurement integrity -- switch it off to measure honestly"))

PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "legacy": {"off": (), "on": ()},
    "trained": {"off": ("hidden_tools", "param_aliases", "body_name_chain"), "on": ("strict_params",)},
    "model-only": {"off": ("hidden_tools", "param_aliases", "body_name_chain", "cascade.*"),
                   "on": ("strict_params",)},
}

_state: dict[str, bool] = {}
_profile = "legacy"


def expand(patterns) -> list[str]:
    """Names matching the patterns ("cascade.*" is a wildcard); an unknown name is an error."""
    out: list[str] = []
    for p in patterns:
        hit = fnmatch.filter(REGISTRY, p.strip())
        if not hit:
            raise ValueError(f"unknown tweak {p!r}; known: {', '.join(REGISTRY)}")
        out += [h for h in hit if h not in out]
    return out


def reset() -> None:
    global _profile
    _profile = "legacy"
    _state.clear()
    _state.update({n: t.default for n, t in REGISTRY.items()})


def configure(profile: str | None = None, off=(), on=()) -> None:
    """Start from `profile` (default legacy), then apply `off` and `on` on top."""
    global _profile
    reset()
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"unknown tweak profile {profile!r}; known: {', '.join(PROFILES)}")
        _profile = profile
        for n in expand(PROFILES[profile]["off"]):
            _state[n] = False
        for n in expand(PROFILES[profile]["on"]):
            _state[n] = True
    for n in expand(off):
        _state[n] = False
    for n in expand(on):
        _state[n] = True


def _split(value: str) -> list[str]:
    return [x for x in re.split(r"[,\s]+", value) if x]


def configure_from_env() -> None:
    configure(os.environ.get("MENTAT_TWEAKS") or None,
              _split(os.environ.get("MENTAT_TWEAKS_OFF", "")),
              _split(os.environ.get("MENTAT_TWEAKS_ON", "")))


def on(name: str) -> bool:
    return _state[name]            # KeyError for a name nobody registered: on purpose


def snapshot() -> dict:
    """What differs from the defaults, for the run record and the comparison tools."""
    return {"profile": _profile,
            "off": sorted(n for n, t in REGISTRY.items() if t.default and not _state[n]),
            "on": sorted(n for n, t in REGISTRY.items() if not t.default and _state[n])}


def label(snap: dict) -> str:
    if not snap["off"] and not snap["on"]:
        return "legacy"
    return f"{snap['profile']}[-{len(snap['off'])} +{len(snap['on'])}]"


@functools.lru_cache(maxsize=1)
def server_catalogue() -> dict[str, tuple[str, ...]]:
    """Tool name -> parameter names, read from the @server.tool() functions in server.py: the
    real signatures, with no server to start (the manifest's `inputs` lists can lag them)."""
    tree = ast.parse((HERE.parent / "server.py").read_text(encoding="utf-8"))
    out: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))                 and any(ast.unparse(d).startswith("server.tool") for d in node.decorator_list):
            a = node.args
            out[node.name] = tuple(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
    return out


def emitted_tools(group: str, catalogue: set[str]) -> set[str]:
    """Catalogue tools a cascade group can reach: its source's string literals plus CASCADE_ALSO."""
    src = (HERE / f"{CASCADE_MODULES[group]}.py").read_text(encoding="utf-8")
    found = set(re.findall(r"""["']([A-Za-z_][A-Za-z0-9_]*)["']""", src)) & catalogue
    return found | set(CASCADE_ALSO.get(group, ()))


reset()
configure_from_env()
