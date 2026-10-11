"""Router fine-tune dataset generator — verified (question -> tool call) pairs.

The Qwen fine-tune this feeds is a ROUTER, not a physicist: it reads a
question and emits ONE JSON tool plan; Mentat runs the plan and writes the
answer (the run_sigma_ground finalize="chain" contract). So every training
target here is a tool call, a request_clarification call, or a decline --
never a number.

Pipeline (all deterministic except the optional teacher pass):
  1. Tool surface: spawn the Mentat MCP server and list the tools the harness
     shows the model (HIDDEN_FROM_LLM removed). Its hash goes in the manifest
     so a trained adapter can be checked against the live surface later.
  2. Sample inputs per tool from the parameter NAME (mass_kg -> kilograms,
     wavelength_m -> metres, rendered as "550 nm" with the label in metres);
     string keys come from probing the tool itself (most lookups answer a bad
     key with their list of valid ones).
  3. Render a template question from the manifest's own keyword phrases.
  4. VERIFY: run the gold call through the real server; keep it only if it
     returns a value (the harness's own success test, _tool_call_value).
  5. Second opinion: if a deterministic classifier routes the question to a
     DIFFERENT tool, the label is ambiguous -- drop it.
  6. Contamination guard: drop anything resembling a benchmark question
     (text overlap, or the same tool with the same answer/numbers) -- the
     benchmark corpora stay a clean test set.
  7. Negatives: a required input removed -> request_clarification; everyday
     non-physics / metaphor questions -> decline.
  8. Optional teacher pass (--teacher-model, needs Ollama): rewrite each
     stilted template question naturally; a rewrite is kept only if it keeps
     every number, unit and named key and still passes steps 5-6.
  9. Split train/val by seed group (a seed's rewrites never straddle splits)
     and write chat-format JSONL + manifest.json.

Run:
  python -m sigma_ground.mcp.benchmark.router_dataset --per-tool 40
  python -m sigma_ground.mcp.benchmark.router_dataset --per-tool 40 \\
      --teacher-model qwen2.5:32b --variants 3
Writes: local-cache/router_dataset/{train,val}.jsonl + manifest.json
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sigma_ground.mcp.benchmark.run_sigma_ground import _tool_call_value

HERE = Path(__file__).parent
REPO = HERE.parents[2]
DEFAULT_OUT = REPO / "local-cache" / "router_dataset"
GENERATOR_VERSION = 1

ROUTER_SYSTEM_PROMPT = (
    "You are Mentat's router. Read the user's question and reply with ONE JSON "
    "object and nothing else:\n"
    '  {"tool": "<tool name>", "args": {...}} to call a library tool. Convert '
    "every input to the unit its parameter name states (mass_kg in kilograms, "
    "wavelength_m in metres).\n"
    '  {"tool": "request_clarification", "args": {"variable": "<parameter>", '
    '"question": "<what to ask>"}} when a required input is missing. Never '
    "invent it.\n"
    '  {"decline": "<reason>"} when it is not a question about a physical '
    "system.\n"
    "Never compute or recall the answer yourself; the tool does that."
)


# ── tool surface ─────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict

    @property
    def params(self) -> dict[str, dict]:
        return self.schema.get("properties") or {}

    @property
    def required(self) -> list[str]:
        return list(self.schema.get("required") or [])


def surface_hash(tools: list[ToolSpec]) -> str:
    """Stable hash of the visible tool surface (names + input schemas)."""
    blob = json.dumps(sorted([t.name, t.schema] for t in tools),
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _param_type(spec: dict) -> str:
    """'number' | 'integer' | 'string' | other, looking through Optional[...]."""
    if spec.get("type"):
        return spec["type"]
    kinds = [s.get("type") for s in spec.get("anyOf", []) if s.get("type") != "null"]
    return kinds[0] if kinds else "unknown"


# ── quantities: parameter name -> range + how a person would write it ────

@dataclass(frozen=True)
class Unit:
    lo: float
    hi: float
    renderings: tuple[tuple[str, float], ...]  # (display, factor to the PARAM's unit)
    log: bool = True
    integer: bool = False


def _u(lo, hi, *renderings, log=True, integer=False):
    return Unit(lo, hi, tuple(renderings), log, integer)


# Matched against the END of a numeric parameter's name (`_` + suffix),
# longest suffix first; case-sensitive so `_M` (molar) is not `_m` (metres).
UNIT_SUFFIXES: dict[str, Unit] = {
    "m_s2":       _u(0.5, 30, ("m/s²", 1), log=False),
    "rad_s":      _u(0.1, 1e3, ("rad/s", 1)),
    "m_s":        _u(1, 3e4, ("m/s", 1), ("km/s", 1e3), ("km/h", 1 / 3.6)),
    "kg_m3":      _u(500, 2e4, ("kg/m³", 1), ("g/cm³", 1e3)),
    "kg_per_mol": _u(2e-3, 0.3, ("kg/mol", 1)),
    "mol_kg":     _u(0.01, 2, ("mol/kg", 1)),
    "mol_l":      _u(1e-4, 2, ("mol/L", 1), ("mM", 1e-3)),
    "pa_s":       _u(1e-5, 10, ("Pa·s", 1)),
    "m2_s":       _u(1e-9, 1e-4, ("m²/s", 1)),
    "w_m2":       _u(1, 1e15, ("W/m²", 1)),
    "m2_w":       _u(1e-20, 1e-17, ("m²/W", 1)),
    "J_m3":       _u(1e-10, 1e10, ("J/m³", 1)),
    "v_m":        _u(1, 1e7, ("V/m", 1), ("kV/m", 1e3)),
    "m2":         _u(1e-4, 10, ("m²", 1), ("cm²", 1e-4)),
    "m3":         _u(1e-4, 10, ("m³", 1), ("litres", 1e-3)),
    "fm2":        _u(0.1, 100, ("fm²", 1)),
    "kg":         _u(1e-3, 1e4, ("kg", 1), ("g", 1e-3), ("tonnes", 1e3)),
    "km":         _u(100, 1e5, ("km", 1)),
    "nm":         _u(0.1, 100, ("nm", 1), ("Å", 0.1)),
    "pm":         _u(30, 300, ("pm", 1), log=False),
    "m":          _u(1e-3, 1e4, ("m", 1), ("km", 1e3), ("cm", 1e-2), ("mm", 1e-3),
                     ("µm", 1e-6), ("nm", 1e-9)),
    "meters":     _u(1e-3, 1e4, ("m", 1), ("km", 1e3), ("cm", 1e-2)),
    "s":          _u(1e-3, 1e5, ("s", 1), ("ms", 1e-3), ("minutes", 60), ("hours", 3600)),
    "k":          _u(50, 2000, ("K", 1)),
    "hz":         _u(1, 1e18, ("Hz", 1), ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9),
                     ("THz", 1e12)),
    "deg":        _u(5, 85, ("degrees", 1), log=False),
    "ohm":        _u(1, 1e6, ("ohms", 1), ("kilohms", 1e3)),
    "v":          _u(0.1, 1e3, ("V", 1), ("mV", 1e-3), ("kV", 1e3)),
    "a":          _u(1e-3, 100, ("A", 1), ("mA", 1e-3)),
    "f":          _u(1e-12, 1e-3, ("F", 1), ("µF", 1e-6), ("nF", 1e-9), ("pF", 1e-12)),
    "h":          _u(1e-6, 1, ("H", 1), ("mH", 1e-3), ("µH", 1e-6)),
    "pa":         _u(1e3, 1e7, ("Pa", 1), ("kPa", 1e3), ("MPa", 1e6), ("GPa", 1e9)),
    "j":          _u(1e-3, 1e9, ("J", 1), ("kJ", 1e3), ("MJ", 1e6)),
    "joules":     _u(1e-3, 1e9, ("J", 1), ("kJ", 1e3), ("MJ", 1e6)),
    "ev":         _u(0.01, 1e6, ("eV", 1), ("keV", 1e3), ("MeV", 1e6), ("meV", 1e-3)),
    "eV":         _u(0.01, 1e6, ("eV", 1), ("keV", 1e3), ("MeV", 1e6), ("meV", 1e-3)),
    "w":          _u(1, 1e6, ("W", 1), ("kW", 1e3), ("MW", 1e6)),
    "watts":      _u(1e24, 1e30, ("W", 1)),
    "n":          _u(0.1, 1e4, ("N", 1), ("kN", 1e3)),
    "t":          _u(0.01, 10, ("T", 1), ("mT", 1e-3)),
    "tesla":      _u(0.01, 10, ("T", 1), ("mT", 1e-3)),
    "c":          _u(1e-19, 1e-3, ("C", 1), ("µC", 1e-6), ("nC", 1e-9)),
    "amu":        _u(1, 250, ("amu", 1)),
    "mL":         _u(1, 1e3, ("mL", 1)),
    "M":          _u(1e-4, 2, ("M", 1), ("mM", 1e-3)),
    "debye":      _u(0.1, 5, ("D", 1), log=False),
}

# Exact-name overrides: a better range, or a unit for a name with no suffix.
NAME_UNITS: dict[str, Unit] = {
    # metres
    "wavelength_m":        _u(1e-10, 1, *UNIT_SUFFIXES["m"].renderings),
    "design_wavelength_m": _u(4e-7, 2e-6, ("nm", 1e-9), ("µm", 1e-6)),
    "height_m":            _u(0.5, 500, ("m", 1), ("cm", 1e-2)),
    "radius_m":            _u(1e-3, 1e8, ("m", 1), ("km", 1e3), ("cm", 1e-2)),
    "distance_m":          _u(1, 1e13, ("m", 1), ("km", 1e3)),
    "separation_m":        _u(1e-5, 1e-2, ("mm", 1e-3), ("µm", 1e-6), ("m", 1)),
    "thickness_m":         _u(1e-6, 1e-2, ("mm", 1e-3), ("µm", 1e-6)),
    "slit_separation_m":   _u(1e-6, 1e-3, ("mm", 1e-3), ("µm", 1e-6)),
    "slit_width_m":        _u(1e-7, 1e-4, ("µm", 1e-6), ("mm", 1e-3)),
    "grating_spacing_m":   _u(1e-7, 1e-5, ("µm", 1e-6), ("nm", 1e-9)),
    "screen_distance_m":   _u(0.5, 5, ("m", 1), ("cm", 1e-2), log=False),
    "object_distance_m":   _u(0.05, 5, ("m", 1), ("cm", 1e-2)),
    "image_distance_m":    _u(0.05, 5, ("m", 1), ("cm", 1e-2)),
    "focal_length_m":      _u(0.02, 2, ("m", 1), ("cm", 1e-2), ("mm", 1e-3)),
    "roughness_m":         _u(1e-9, 1e-5, ("nm", 1e-9), ("µm", 1e-6)),
    "core_thickness_m":    _u(1e-6, 1e-4, ("µm", 1e-6)),
    "r_inner_m":           _u(1e-3, 0.5, ("mm", 1e-3), ("cm", 1e-2), ("m", 1)),
    "r_outer_m":           _u(1e-3, 0.5, ("mm", 1e-3), ("cm", 1e-2), ("m", 1)),
    "sphere_inner_m":      _u(1e-3, 0.5, ("mm", 1e-3), ("cm", 1e-2), ("m", 1)),
    "sphere_outer_m":      _u(1e-3, 0.5, ("mm", 1e-3), ("cm", 1e-2), ("m", 1)),
    # kilograms
    "central_mass_kg":     _u(1e22, 1e31, ("kg", 1)),
    "ion_mass_kg":         _u(1e-27, 5e-25, ("kg", 1)),
    "molecular_mass_kg":   _u(1e-27, 5e-25, ("kg", 1)),
    "rest_mass_kg":        _u(1e-31, 1e3, ("kg", 1)),
    # kelvin
    "critical_temp_k":     _u(1, 100, ("K", 1)),
    "electron_temperature_k": _u(1e3, 1e8, ("K", 1)),
    "ambient_temperature_k":  _u(250, 320, ("K", 1), log=False),
    "t_kelvin":            _u(50, 2000, ("K", 1)),
    "t_hot":               _u(300, 2000, ("K", 1)),
    "t_cold":              _u(200, 2000, ("K", 1)),
    # velocities a relativity tool can use (c = 2.998e8 m/s)
    "u_m_s":               _u(1e6, 2.9e8, ("m/s", 1), ("km/s", 1e3)),
    "v_m_s":               _u(1e6, 2.9e8, ("m/s", 1), ("km/s", 1e3)),
    # pascals
    "pressure_pa":         _u(1e4, 1e7, ("Pa", 1), ("kPa", 1e3), ("MPa", 1e6)),
    "e1_pa":               _u(1e9, 4e11, ("GPa", 1e9)),
    "e2_pa":               _u(1e9, 4e11, ("GPa", 1e9)),
    "bulk_modulus1_pa":    _u(1e9, 4e11, ("GPa", 1e9)),
    "bulk_modulus2_pa":    _u(1e9, 4e11, ("GPa", 1e9)),
    "yield1_pa":           _u(5e7, 2e9, ("MPa", 1e6)),
    "yield2_pa":           _u(5e7, 2e9, ("MPa", 1e6)),
    "applied_stress":      _u(1e6, 1e9, ("Pa", 1), ("MPa", 1e6)),
    # electron-volts / joules / watts / amps
    "barrier_height_eV":   _u(0.1, 10, ("eV", 1)),
    "particle_energy_eV":  _u(0.01, 10, ("eV", 1)),
    "spin_orbit_constant_ev": _u(1e-4, 0.1, ("eV", 1), ("meV", 1e-3)),
    "activation_energy_eV":   _u(0.1, 3, ("eV", 1)),
    "energy_joules":       _u(1, 1e15, ("J", 1), ("kJ", 1e3), ("MJ", 1e6)),
    "energy_j":            _u(1, 1e15, ("J", 1), ("kJ", 1e3), ("MJ", 1e6)),
    "saturation_current_a": _u(1e-15, 1e-9, ("A", 1)),
    "applied_voltage":     _u(0.05, 1, ("V", 1), log=False),
    "fluid_density_kg_m3": _u(500, 2000, ("kg/m³", 1)),
    "molar_mass_kg":       _u(2e-3, 0.3, ("kg/mol", 1)),
    "density1":            _u(1e3, 2e4, ("kg/m³", 1)),
    "density2":            _u(1e3, 2e4, ("kg/m³", 1)),
    # number densities (per m³ / per m²)
    "donor_density_m3":    _u(1e20, 1e25, ("per m³", 1)),
    "acceptor_density_m3": _u(1e20, 1e25, ("per m³", 1)),
    "donor_density":       _u(1e20, 1e25, ("per m³", 1)),
    "acceptor_density":    _u(1e20, 1e25, ("per m³", 1)),
    "electron_density_m3": _u(1e10, 1e26, ("per m³", 1)),
    "number_density_m3":   _u(1e20, 1e28, ("per m³", 1)),
    "dislocation_density": _u(1e10, 1e16, ("per m²", 1)),
    "luminosity_watts":    _u(1e24, 1e30, ("W", 1)),
    "t_celsius":           _u(-50, 500, ("°C", 1), log=False),
}

# Dimensionless (or unit-implied) parameters: (lo, hi, log, integer).
PLAIN_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "refractive_index": (1.0, 2.5, False, False), "n_core": (1.4, 1.7, False, False),
    "n_clad": (1.3, 1.45, False, False), "n0": (1.3, 2.5, False, False),
    "n_low": (1.3, 1.6, False, False), "n_high": (1.8, 2.6, False, False),
    "n_dense": (1.4, 2.5, False, False), "n_rare": (1.0, 1.35, False, False),
    "n": (1, 7, False, True), "n3": (1, 7, False, True),
    "n_initial": (2, 7, False, True), "n_final": (1, 6, False, True),
    "n_pairs": (2, 30, False, True), "n_moles": (0.1, 10, True, False),
    "n_atoms": (2, 20, False, True), "atomic_number": (1, 92, False, True),
    "emissivity": (0.05, 1, False, False), "gamma": (1.1, 1.67, False, False),
    "degrees_of_freedom": (3, 7, False, True),
    "friction_coefficient": (0.05, 0.9, False, False),
    "restitution": (0.0, 1.0, False, False), "strain": (1e-4, 0.02, True, False),
    "plastic_strain": (1e-3, 0.2, True, False), "initial_strain": (1e-4, 1e-2, True, False),
    "fraction1": (0.05, 0.95, False, False), "fraction_b": (0.05, 0.95, False, False),
    "percent": (1, 99, False, False), "ph": (1, 13, False, False),
    "van_t_hoff_i": (1, 4, False, True), "electrons": (1, 4, False, True),
    "d_electron_count": (0, 10, False, True), "oxidation_state": (1, 4, False, True),
    "electron_domains": (2, 6, False, True), "lone_pairs": (0, 3, False, True),
    "dielectric_constant": (1, 100, True, False), "epsilon_r": (1, 100, True, False),
    "drag_coefficient": (0.1, 1.5, False, False), "nu1": (0.2, 0.45, False, False),
    "nu2": (0.2, 0.45, False, False), "asperity_slope": (0.01, 0.2, True, False),
    "reaction_quotient": (1e-3, 1e3, True, False),
    "ratio_base_over_acid": (0.1, 10, True, False), "tnt_amount": (0.1, 100, True, False),
    "intensity_max": (0.5, 1.0, False, False), "intensity_min": (0.0, 0.5, False, False),
    "initial_brightness": (1, 100, True, False), "grover_n_qubits": (2, 10, False, True),
    "qaoa_n_nodes": (3, 8, False, True), "order": (1, 6, False, True),
    "lambda_cation": (5e-3, 3.5e-2, True, False), "lambda_anion": (5e-3, 3.5e-2, True, False),
    "current_density": (1, 1e4, True, False),
    "exchange_current_density": (1e-6, 10, True, False),
    "rate_constant": (1e-5, 10, True, False), "prefactor": (1e6, 1e13, True, False),
    "target_rate": (1e-5, 10, True, False), "value": (0.1, 1e3, True, False),
}

# Values that only come in steps. `total_j` is the quantum number J, not
# joules: read as energy it once sampled J = 3e8, and the tool enumerated all
# 6e8 m_j values until the server was OOM-killed at 13.7 GB.
DISCRETE: dict[str, list[float]] = {
    "total_j": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    "total_angular_momentum_j": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
}

# n1/n2 are refractive indices in optics tools, quantum numbers elsewhere.
_OPTICS_TOOL = re.compile(r"snell|refract|fresnel|critical|brewster|reflect|tir", re.I)

# (larger, smaller): sampled independently, then swapped into order.
ORDERED_PAIRS = [("t_hot_k", "t_cold_k"), ("hot_temperature_k", "cold_temperature_k"),
                 ("t_hot", "t_cold"), ("n_initial", "n_final"), ("n_dense", "n_rare"),
                 ("r_outer_m", "r_inner_m"), ("sphere_outer_m", "sphere_inner_m"),
                 ("n_high", "n_low"), ("n_core", "n_clad"),
                 ("intensity_max", "intensity_min"),
                 ("final_volume_mL", "initial_volume_mL")]

# Nouns for parameter names whose own words read badly.
NOUNS = {"g": "gravitational acceleration", "e_field": "electric field",
         "b_field": "magnetic field", "b": "magnetic field", "n_moles": "amount",
         "t_hot": "hot reservoir temperature", "t_cold": "cold reservoir temperature",
         "u": "velocity u", "v": "velocity v", "ph": "pH", "nu1": "Poisson ratio 1",
         "nu2": "Poisson ratio 2", "n_initial": "initial level n",
         "n_final": "final level n", "n": "quantum number n",
         "van_t_hoff_i": "van 't Hoff factor", "L": "length",
         "total_j": "total angular momentum quantum number J",
         "total_angular_momentum_j": "total angular momentum quantum number J"}


# Masses for gravity/astro tools are stars and planets, not bowling balls.
_ASTRO_TOOL = re.compile(r"schwarzschild|black_hole|hawking|bekenstein|photon_sphere|"
                         r"isco|redshift|escape|orbital|surface_gravity|tidal|roche", re.I)


def _astro_mass() -> Unit:
    from sigma_ground.field.constants import M_SUN_KG
    return _u(1e22, 1e32, ("kg", 1), ("solar masses", M_SUN_KG))


def _unit_for(param: str, tool: str) -> Unit | None:
    if param == "mass_kg" and _ASTRO_TOOL.search(tool):
        return _astro_mass()
    if param in NAME_UNITS:
        return NAME_UNITS[param]
    for suffix in sorted(UNIT_SUFFIXES, key=len, reverse=True):
        if param.endswith("_" + suffix):
            if suffix == "m3" and "density" in param:
                return _u(1e20, 1e28, ("per m³", 1))
            return UNIT_SUFFIXES[suffix]
    return None


def _noun(param: str, unit: Unit | None) -> str:
    stem = param
    if unit is not None:
        for suffix in sorted(UNIT_SUFFIXES, key=len, reverse=True):
            if stem.endswith("_" + suffix):
                stem = stem[: -len(suffix) - 1]
                break
    stem = re.sub(r"_key(_\d)?$", r"\1", stem)
    if stem in NOUNS:
        return NOUNS[stem]
    return re.sub(r"(\D)(\d)$", r"\1 \2", stem).replace("_", " ").strip()


def _nice(x: float) -> str:
    """3 significant figures, no float noise: 5.5e-07, 120, 0.0315."""
    if x == 0:
        return "0"
    s = f"{x:.3g}"
    if "e" in s:
        mant, exp = s.split("e")
        return f"{mant}e{int(exp)}"
    return s


@dataclass
class Arg:
    """One sampled input: its label value (in the PARAM's unit) and how the
    question writes it. `unit_text`/`key_text` are what a rewrite must keep."""
    name: str
    value: Any
    text: str
    noun: str
    unit_text: str = ""
    key_text: str = ""


def render_quantity(value: float, unit: Unit, rng: random.Random,
                    integer: bool = False) -> tuple[str, float, float, str]:
    """Pick a rendering. Returns (text, label_value, shown_number, unit_text):
    the label is recomputed from the SHOWN number, so text and label agree
    exactly whatever the rounding."""
    if integer:
        v = int(round(value))
        disp = unit.renderings[0][0] if unit.renderings else ""
        return (f"{v} {disp}".strip(), v, float(v), disp)
    fits = [(d, f) for d, f in unit.renderings if 0.1 <= abs(value) / f < 1000]
    choices = fits or [unit.renderings[0]]
    disp, factor = (choices[0] if rng.random() < 0.5 else rng.choice(choices))
    shown = _nice(value / factor)
    return (f"{shown} {disp}", _clean(float(shown) * factor), float(shown), disp)


def _clean(x: float) -> float:
    """16600000.000000002 -> 16600000.0 (12 significant digits)."""
    return float(f"{x:.12g}")


def _sample_range(lo, hi, log, rng):
    if log and lo > 0:
        return 10 ** rng.uniform(math.log10(lo), math.log10(hi))
    return rng.uniform(lo, hi)


def sample_numeric(param: str, ptype: str, tool: str,
                   rng: random.Random) -> Arg | None:
    """A plausible value for a numeric parameter, or None if unknown."""
    integer = ptype == "integer"
    if param in DISCRETE:
        v = rng.choice(DISCRETE[param])
        return Arg(param, v, _nice(v), NOUNS.get(param) or _noun(param, None))
    if param in ("n1", "n2"):
        if _OPTICS_TOOL.search(tool):
            lo, hi, log, integer = 1.0, 2.5, False, False
        else:
            lo, hi, log, integer = 1, 6, False, True
        rng_val = _sample_range(lo, hi, log, rng)
        return _plain_arg(param, rng_val, integer)
    if param == "t_celsius":
        v = round(rng.uniform(-50, 500))
        return Arg(param, float(v), f"{v} °C", "temperature", "°C")
    unit = _unit_for(param, tool)
    if unit is not None:
        integer = integer or unit.integer
        val = _sample_range(unit.lo, unit.hi, unit.log, rng)
        noun = _noun(param, unit)
        # some kelvin inputs read better in Celsius; label stays kelvin
        if unit.renderings == UNIT_SUFFIXES["k"].renderings and val < 1200 \
                and rng.random() < 0.25:
            c = round(val - 273.15)
            return Arg(param, _clean(c + 273.15), f"{c} °C", noun, "°C")
        text, label, shown, utext = render_quantity(val, unit, rng, integer)
        return Arg(param, label, text, noun, utext)
    if param in PLAIN_RANGES:
        lo, hi, log, pint_ = PLAIN_RANGES[param]
        return _plain_arg(param, _sample_range(lo, hi, log, rng), integer or pint_)
    return None


def _plain_arg(param: str, val: float, integer: bool) -> Arg:
    noun = NOUNS.get(param) or _noun(param, None)
    if integer:
        v = int(round(val))
        return Arg(param, v, str(v), noun)
    shown = _nice(val)
    return Arg(param, float(shown), shown, noun)


# ── string inputs: probed vocabularies + curated symbolic pools ──────────

_LIST_RE = re.compile(r"\[([^\[\]]*'[^\[\]]*)\]")
_AVAILABLE_RE = re.compile(r"(?:available|known|options|one of)\s*:\s*([^\]\n.]+)", re.I)


def parse_vocab(text: str) -> list[str]:
    """Valid keys a tool lists when handed a bad one: "Available: ['a', 'b']",
    "Needs a metal in ['a', 'b']", or "Some available: a, b, c, ..."."""
    for m in _LIST_RE.finditer(text):
        try:
            items = ast.literal_eval("[" + m.group(1) + "]")
        except (ValueError, SyntaxError):
            continue
        keys = [i for i in items if isinstance(i, str) and i.strip()]
        if keys:
            return keys
    m = _AVAILABLE_RE.search(text)
    if m:
        keys = [k.strip().strip("'\"") for k in m.group(1).split(",")]
        return [k for k in keys if k and k != "..." and re.fullmatch(r"[\w\- ]+", k)]
    return []


# Symbolic tools take expressions, not keys; the server still verifies them.
SYMBOLIC: dict[str, dict[str, list]] = {
    "integrate_expr":     {"expression": ["x**2 + 3*x", "sin(x)*cos(x)", "x*exp(x)",
                                          "1/(1 + x**2)", "x**3 - 2*x + 1"],
                           "variable": ["x"]},
    "differentiate_expr": {"expression": ["x**3 - 2*x + 1", "sin(x)**2", "exp(-x**2)",
                                          "log(x)*x", "(x + 1)**4"],
                           "variable": ["x"]},
    "simplify_expr":      {"expression": ["sin(x)**2 + cos(x)**2", "(x**2 - 1)/(x - 1)",
                                          "(x**2 + 2*x + 1)/(x + 1)"]},
    "factor_expression":  {"expression": ["x**2 - 5*x + 6", "x**3 - 1", "x**2 - 9",
                                          "2*x**2 + 7*x + 3"]},
    "expand_expression":  {"expression": ["(x + 2)**3", "(x - 1)*(x + 4)",
                                          "(2*x + 1)**2"]},
    "compute_limit":      {"expression": ["sin(x)/x", "(1 - cos(x))/x**2",
                                          "(exp(x) - 1)/x"]},
    "series_expansion":   {"expression": ["exp(x)", "sin(x)", "log(1 + x)", "cos(x)"]},
    "summation":          {"expression": ["1/n**2", "1/2**n", "1/(n*(n + 1))"]},
    "laplace_transform":  {"expression": ["exp(-2*t)", "t**2", "sin(3*t)"]},
    "fourier_transform":  {"expression": ["exp(-x**2)"]},
    "solve_equation":     {"equation": ["x**2 - 5*x + 6", "2*x + 3 - 11", "x**3 - 8",
                                        "x**2 - 2"],
                           "variable": ["x"]},
    "gradient":           {"scalar_field": ["x**2*y + z", "x*y*z", "x**2 + y**2 + z**2"]},
    "divergence":         {"vector_field": ["x*y, y*z, z*x", "x**2, y**2, z**2"]},
    "curl":               {"vector_field": ["y, -x, 0", "x*y, y*z, z*x"]},
    "matrix_determinant": {"matrix": ["[[1, 2], [3, 4]]", "[[2, 0, 1], [1, 3, 2], [1, 1, 1]]"]},
    "matrix_eigenvalues": {"matrix": ["[[2, 0], [0, 3]]", "[[4, 1], [2, 3]]"]},
    "matrix_inverse":     {"matrix": ["[[1, 2], [3, 4]]", "[[2, 1], [1, 1]]"]},
    "matrix_multiply":    {"matrix_a": ["[[1, 2], [3, 4]]"], "matrix_b": ["[[0, 1], [1, 0]]"]},
    "solve_linear_system": {"matrix_a": ["[[2, 1], [1, 3]]"], "vector_b": ["[5, 10]"]},
}
_SYMBOLIC_PHRASE = {"expression": "{v}", "equation": "{v} = 0", "scalar_field": "{v}",
                    "vector_field": "({v})", "variable": "with respect to {v}",
                    "matrix": "the matrix {v}", "matrix_a": "matrix A = {v}",
                    "matrix_b": "matrix B = {v}", "vector_b": "b = {v}"}

# Strings the tools resolve themselves; probing can't list these.
CURATED_STRINGS: dict[str, list[tuple[str, str]]] = {
    # (label value, how the question says it)
    "name":     [("speed_of_light", "the speed of light"), ("G", "the gravitational constant"),
                 ("planck_constant", "Planck's constant"), ("electron_mass", "the electron mass"),
                 ("boltzmann_constant", "the Boltzmann constant"),
                 ("avogadro_constant", "Avogadro's number"),
                 ("elementary_charge", "the elementary charge"),
                 ("proton_mass", "the proton mass")],
    "particle": [("electron", "an electron"), ("proton", "a proton"), ("neutron", "a neutron")],
}

# convert_units: (from, to, how a person writes from, to, lo, hi)
CONVERSIONS = [("eV", "joule", "eV", "joules", 0.1, 1e6),
               ("joule", "eV", "joules", "eV", 1e-20, 1e-15),
               ("km", "mile", "km", "miles", 1, 1e4), ("mile", "km", "miles", "km", 1, 1e4),
               ("light_year", "m", "light years", "metres", 0.1, 100),
               ("parsec", "light_year", "parsecs", "light years", 0.1, 1e3),
               ("AU", "km", "AU", "km", 0.1, 50), ("atm", "Pa", "atm", "pascals", 0.1, 100),
               ("calorie", "joule", "calories", "joules", 1, 1e4),
               ("kWh", "joule", "kWh", "joules", 0.1, 1e3),
               ("MeV", "joule", "MeV", "joules", 0.1, 1e4),
               ("solar_mass", "kg", "solar masses", "kg", 0.1, 100),
               ("pound", "kg", "pounds", "kg", 1, 1e3), ("foot", "m", "feet", "metres", 1, 1e4),
               ("mph", "m/s", "mph", "m/s", 1, 500),
               ("degree", "radian", "degrees", "radians", 1, 360),
               ("amu", "kg", "amu", "kg", 1, 250)]


@dataclass
class Vocab:
    """String keys per (tool, param), from probing; plus per-param fallbacks
    merged across tools (material_key appears in 11 tools; one lists keys)."""
    by_tool: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    by_param: dict[str, list[str]] = field(default_factory=dict)

    def get(self, tool: str, param: str) -> list[str]:
        return self.by_tool.get((tool, param)) or self.by_param.get(param) or []


def sample_string(tool: str, param: str, vocab: Vocab, rng: random.Random,
                  n_string_params: int) -> Arg | None:
    if tool in SYMBOLIC and param in SYMBOLIC[tool]:
        v = rng.choice(SYMBOLIC[tool][param])
        text = _SYMBOLIC_PHRASE.get(param, "{v}").format(v=v)
        return Arg(param, v, text, param, key_text=v)
    if param in CURATED_STRINGS:
        v, said = rng.choice(CURATED_STRINGS[param])
        return Arg(param, v, said, param, key_text=said)
    keys = vocab.get(tool, param)
    if not keys:
        return None
    v = rng.choice(keys)
    said = v.replace("_", " ")
    noun = _noun(param, None)
    text = said if n_string_params == 1 else f"{noun} {said}"
    return Arg(param, v, text, noun, key_text=said)


# ── question rendering ───────────────────────────────────────────────────

_QUESTION_WORDS = {"how", "what", "what's", "whats", "why", "will", "if", "can",
                   "does", "is", "when", "where", "which", "who", "should"}
_IMPERATIVE = {"look", "find", "convert", "list", "compute", "calculate", "show",
               "give", "get", "estimate", "determine", "solve", "integrate",
               "differentiate", "simplify", "factor", "expand", "parse", "check",
               "predict", "simulate", "make", "build", "run", "tell", "count"}
_BAD_KEYWORD = re.compile(r"[=^\\/(){}_\d]")   # formulas, symbols, years, bare numbers
# A fragment ("drop it from", "crush it to a black hole") is only usable as a
# question; as a noun it makes "What is the drop it from?".
_FRAGMENT = re.compile(r"\b(it|i|my|me|this|that)\b|\b(from|to|of|for|with|in|on|at|by)$",
                       re.I)
_ENDS_IN_PREPOSITION = re.compile(r"\b(from|to|of|for|with|in|on|at|by|about)$", re.I)

TEMPLATES = {
    "noun":       ["What is the {kw} for {args}?", "Find the {kw} for {args}.",
                   "Calculate the {kw} for {args}.", "{Args}: what's the {kw}?",
                   "I need the {kw} for {args}."],
    "noun0":      ["What is the {kw}?", "Give me the {kw}.", "{Kw}?"],
    "question":   ["{Kw}? {Args}.", "{Args}: {kw}?"],
    "question0":  ["{Kw}?"],
    "imperative": ["{Kw} for {args}.", "Please {kw} for {args}."],
    "imperative0": ["{Kw}.", "Please {kw}."],
}
# Tools whose input IS the thing asked about ("the speed of light"), where a
# generic intent would contradict it ("the gravitational constant G for ...").
OWN_TEMPLATES = {
    "lookup_constant": ["What is the value of {args}?", "Look up {args}.",
                        "What's {args} in SI units?", "Give me {args}."],
}


def intent_kind(kw: str) -> str:
    first = kw.split()[0].lower()
    if first in _QUESTION_WORDS:
        return "question"
    if first in _IMPERATIVE:
        return "imperative"
    return "noun"


def intent_phrases(tool: ToolSpec, manifest_entry: dict | None) -> list[str]:
    """Natural-language intents for a tool, from the manifest's own keywords."""
    raw: list[str] = []
    if manifest_entry:
        raw += manifest_entry.get("keywords_formal") or []
        raw += manifest_entry.get("keywords_colloquial") or []
    if not raw:
        from sigma_ground.mcp.tool_keywords import TOOL_KEYWORDS
        raw = list(TOOL_KEYWORDS.get(tool.name, []))
    out = []
    for k in raw:
        k = k.strip().rstrip("?.")
        if len(k) < 5 or _BAD_KEYWORD.search(k) or re.fullmatch(r"[A-Z0-9 \-]{1,6}", k):
            continue
        if intent_kind(k) != "question" and _FRAGMENT.search(k):
            continue
        if k.lower() not in (o.lower() for o in out):
            out.append(k)
    return out or [tool.name.replace("_", " ")]


def drop_contradicting(kws: list[str], keys: list[str]) -> list[str]:
    """Drop intents that name one of the tool's own keys ("voltage of a copper
    and zinc battery" would contradict cathode=silver); keep at least one."""
    def names_key(kw: str) -> bool:
        for k in keys:
            said = k.replace("_", " ")
            flags = 0 if len(said) <= 2 else re.I
            if re.search(rf"(?<!\w){re.escape(said)}(?!\w)", kw, flags):
                return True
        return False
    kept = [k for k in kws if not names_key(k)]
    return kept or kws[:1]


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _arg_phrase(a: Arg) -> str:
    """'mass of 9.44 tonnes' for a quantity; the key itself ('copper',
    'x**2 + 3*x') for a string input."""
    if isinstance(a.value, str):
        return a.text
    return f"{a.noun} of {a.text}"


def render_question(kw: str, args: list[Arg], template_index: int,
                    tool: str = "") -> str:
    phrases = [_arg_phrase(x) for x in args]
    tail = " ".join(p for p in phrases if p.startswith("with respect to"))
    a = (_join([p for p in phrases if not p.startswith("with respect to")])
         + (" " + tail if tail else "")).strip()
    if tool in OWN_TEMPLATES and args:
        pool = OWN_TEMPLATES[tool]
    elif args and _ENDS_IN_PREPOSITION.search(kw):
        pool = ["{Kw} {args}?"]            # "how strong is gravity on" + "Mercury"
    else:
        if not args:                       # nothing follows: "how strong is gravity?"
            kw = _ENDS_IN_PREPOSITION.sub("", kw).strip()
        pool = TEMPLATES[intent_kind(kw) + ("" if args else "0")]
    t = pool[template_index % len(pool)]
    q = t.format(kw=kw, Kw=kw[:1].upper() + kw[1:], args=a,
                 Args=a[:1].upper() + a[1:])
    return re.sub(r"\s+", " ", q).replace(" ,", ",").replace("..", ".").strip()


# ── verification ─────────────────────────────────────────────────────────

_META_FIELDS = {"inputs", "source", "notes", "procedure", "provenance_tag", "formula",
                "units", "uncertainty"}


def has_content(v: Any) -> bool:
    """A real answer, not an empty shell: elastic_analysis on a bad key returns
    a dict of all-None fields, playground_status with nothing open returns
    empty lists."""
    if v is None:
        return False
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple)):
        return any(has_content(x) for x in v)
    if isinstance(v, dict):
        return any(has_content(x) for x in v.values())
    return True


async def execute(session, name: str, args: dict, question: str = "",
                  timeout_s: float = 15.0) -> dict | None:
    """Run the gold call. Scalar answers go through the harness's own success
    test; others (expressions, catalogues, procedure step lists) must carry
    real content."""
    try:
        res = await asyncio.wait_for(session.call_tool(name, args), timeout_s)
    except Exception:
        return None
    if getattr(res, "isError", False):
        return None
    text = "\n".join(p.text for p in (res.content or []) if getattr(p, "text", None))
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        return None
    got = _tool_call_value({"name": name, "args": args, "result_text": text}, question)
    if got is None:
        if "error" in str(parsed.get("source")).lower():
            return None
        body = (parsed["value"] if "value" in parsed else
                {k: v for k, v in parsed.items() if k not in _META_FIELDS})
        if not has_content(body):
            return None
        shown = body if isinstance(body, str) else json.dumps(body, default=str)
        got = (shown[:200], parsed.get("units") or "")
    value = got[0] if isinstance(got[0], (int, float, str)) else str(got[0])
    return {"value": value, "units": got[1], "source": parsed.get("source")}


def classifier_route(question: str) -> tuple[str | None, str | None]:
    """What the deterministic cascade (run_sigma_ground order) would do:
    ("tool", name) | ("decline", None) | ("clarify", None) | (None, None)."""
    from sigma_ground.mcp.benchmark import (
        astro_classifier, clarification_classifier, classical_intro_classifier,
        conversion_classifier, cosmology_classifier, frontier_classifier,
        gr_classifier, math_classifier, modern_classifier, new_tools_classifier,
        optics_classifier, refusal_classifier, thermo_classifier)
    cascade = [
        (frontier_classifier.classify_for_frontier, "tool"),
        (new_tools_classifier.classify_for_new_tools, "tool"),
        (refusal_classifier.classify_for_refusal, "decline"),
        (clarification_classifier.classify_for_clarification, "clarify"),
        (conversion_classifier.classify_for_conversion, "convert"),
        (gr_classifier.classify_for_gr, "tool"),
        (cosmology_classifier.classify_for_cosmology, "tool"),
        (math_classifier.classify_for_math, "tool"),
        (thermo_classifier.classify_for_thermo, "tool"),
        (modern_classifier.classify_for_modern, "tool"),
        (classical_intro_classifier.classify_for_classical_intro, "tool"),
        (optics_classifier.classify_for_optics, "tool"),
        (astro_classifier.classify_for_astro, "tool"),
    ]
    for fn, kind in cascade:
        try:
            m = fn(question)
        except Exception:
            continue
        if m is None:
            continue
        if kind == "convert":
            return "tool", "convert_units"
        if kind == "tool":
            return "tool", getattr(m, "tool", None)
        return kind, None
    return None, None


def classifier_verdict(question: str, plan: dict) -> str:
    """'agrees' | 'silent' | 'conflict' for a labelled plan."""
    kind, tool = classifier_route(question)
    if kind is None:
        return "silent"
    want = plan.get("tool")
    if kind == "tool" and tool == want and want != "request_clarification":
        return "agrees"
    if kind in ("decline", "clarify") and (plan.get("decline")
                                          or want == "request_clarification"):
        return "agrees"
    return "conflict"


# ── contamination guard ──────────────────────────────────────────────────

_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text: str) -> list[float]:
    t = re.sub(r"(\d)\s*[×x\*]\s*10\s*\^\s*\{?([-+]?\d+)\}?", r"\1e\2", text)
    out = []
    for tok in _NUM_RE.findall(t):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w.]+", " ", text.lower())).strip()


def _trigrams(text: str) -> frozenset:
    t = f"  {_norm(text)}  "
    return frozenset(t[i:i + 3] for i in range(len(t) - 2))


class BenchmarkGuard:
    """Keeps the benchmark corpora a clean test set: drops a generated
    question that resembles one (trigram Jaccard), or that asks the same
    tool for the same answer / with the same numbers."""

    def __init__(self, texts: list[str], instances: list[tuple[str, float, tuple]],
                 threshold: float = 0.5):
        self.grams = [(_trigrams(t), t) for t in texts]
        self.instances = instances
        self.threshold = threshold

    @classmethod
    def from_repo(cls, bench_dir: Path = HERE, threshold: float = 0.5,
                  exclude: tuple = ()):
        """Every question file in the benchmark folder counts as test data, which
        includes router_hardval.json: training data that resembles a hard-set
        question is dropped too. Pass exclude=("router_hardval.json",) to check
        the hard set itself against the benchmark proper."""
        texts, instances = [], []
        for path in sorted(bench_dir.glob("*.json")):
            if "ground_truth" in path.name or path.name in exclude:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if not isinstance(data, list):
                continue
            for q in data:
                if not isinstance(q, dict) or not isinstance(q.get("question"), str):
                    continue
                texts.append(q["question"])
                tool, val = q.get("primary_tool_expected"), q.get("expected_value")
                if tool and isinstance(val, (int, float)):
                    instances.append((tool, float(val),
                                      tuple(sorted(numbers_in(q["question"])))))
        texts += _routing_corpus_strings(REPO / "misc" / "routing_corpus.py")
        return cls(texts, instances, threshold)

    def check(self, question: str, tool: str | None = None,
              value: Any = None) -> str | None:
        g = _trigrams(question)
        for bg, text in self.grams:
            inter = len(g & bg)
            if inter and inter / len(g | bg) >= self.threshold:
                return f"text overlap with benchmark: {text[:60]!r}"
        if tool:
            nums = sorted(numbers_in(question))
            for btool, bval, bnums in self.instances:
                if btool != tool:
                    continue
                # Same answer = same instance only when no numbers pin the
                # instance down (a keyed lookup like "density of gold", or a
                # fixed constant). With numbers, the rule below decides:
                # doppler_shift_factor is ~1.000 for every ordinary speed.
                if not nums and isinstance(value, (int, float)) and bval \
                        and math.isfinite(bval) and abs(value - bval) <= 5e-3 * abs(bval):
                    return f"same tool+answer as a benchmark question ({tool}={bval})"
                if nums and all(any(abs(n - b) <= 1e-9 * max(1, abs(b)) for b in bnums)
                                for n in nums):
                    return f"same tool+numbers as a benchmark question ({tool})"
        return None


def _routing_corpus_strings(path: Path) -> list[str]:
    """NEGATIVES / HELDOUT question strings from misc/routing_corpus.py, read
    with ast (importing it runs its harness)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("NEGATIVES", "HELDOUT")
                for t in node.targets):
            for c in ast.walk(node.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) \
                        and " " in c.value:
                    out.append(c.value)
    return out


# ── negatives ────────────────────────────────────────────────────────────

DECLINE_SEEDS = [
    "Who wrote Pride and Prejudice?", "What's a good recipe for banana bread?",
    "Can you recommend a fantasy novel?", "How do I reset my router password?",
    "What time zone is Tokyo in?", "Translate 'good morning' into Spanish.",
    "Write a haiku about autumn leaves.", "Who won the 1998 World Cup?",
    "What's the best way to learn guitar?", "How many episodes are in season two?",
    "Is it rude to text during dinner?", "What should I name my new puppy?",
    "Summarize the plot of Hamlet.", "How do I fold a fitted sheet?",
    "What's the population of Canada?", "Suggest a birthday gift for my dad.",
    "My code throws a null pointer exception, help.", "Which stocks should I buy?",
    "Our sales took a quantum leap this year, write a memo about it.",
    "Her career has real momentum, how do I congratulate her?",
    "The housing market is overheating, should I sell?",
    "My motivation has zero inertia on Mondays, any tips?",
    "His speech had a lot of gravity, what made it work?",
    "The team chemistry is great, how do we keep it?",
    "My phone battery drains fast when I play games, which app is to blame?",
    "The project is gaining traction, what's the next milestone?",
    "Rumours spread at the speed of light in our office, how do I stop them?",
    "What's the half-life of a meme on social media?",
    "How much energy do I need to get through a Monday?",
    "What's the escape velocity from a boring meeting?",
]
DECLINE_REASON = "not a question about a physical system"


# ── teacher rewrites ─────────────────────────────────────────────────────

TEACHER_PROMPT = (
    "Rewrite the user's question the way a curious person would naturally ask "
    "it. Keep every number exactly as written (same digits, same notation) and "
    "keep every unit symbol and every named substance exactly as written. Do "
    "not add numbers, do not answer, do not explain. Reply with the rewritten "
    "question only, on one line."
)


def ollama_teacher(url: str, model: str) -> Callable[[str, int], str]:
    import httpx                          # already a benchmark extra
    # One client for the whole pass: a fresh httpx.post per call cost ~1 s of
    # client setup, plus ~2 s per new connection to "localhost" on Windows
    # (IPv6 first), against ~0.7 s for the rewrite itself.
    client = httpx.Client(timeout=120)

    def rewrite(question: str, k: int) -> str:
        r = client.post(f"{url.rstrip('/')}/api/chat", json={
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": TEACHER_PROMPT},
                         {"role": "user", "content": question}],
            "options": {"temperature": 0.9, "seed": k},
        })
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "") or ""
    return rewrite


def clean_rewrite(text: str) -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    return line.strip().strip('"“”\'').strip()


def check_rewrite(original: str, rewrite: str, args: list[Arg]) -> str | None:
    """None if the rewrite keeps the label true, else why not."""
    if not rewrite or len(rewrite) > 400:
        return "empty or too long"
    if _norm(rewrite) == _norm(original):
        return "unchanged"
    if Counter(numbers_in(rewrite)) != Counter(numbers_in(original)):
        return "numbers changed"
    low = rewrite.lower()
    for a in args:
        if a.unit_text:
            # the unit must still follow ITS number: "20.6 g" -> "20.6 kg" is
            # a 1000x label error that a bare `"g" in text` test would pass
            num = re.escape(a.text.split(" ", 1)[0])
            if not re.search(rf"(?<![\d.]){num}\s*-?\s*{re.escape(a.unit_text)}(?![A-Za-z])",
                             rewrite):
                return f"unit {a.unit_text!r} changed"
        if a.key_text and a.key_text.lower() not in low \
                and a.key_text.replace(" ", "").lower() not in low.replace(" ", ""):
            return f"{a.key_text!r} dropped"
    return None


# ── records ──────────────────────────────────────────────────────────────

@dataclass
class Example:
    question: str
    plan: dict
    group: str
    source: str                     # template | rewrite | clarify | decline
    args: list[Arg] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": self.question},
                {"role": "assistant",
                 "content": json.dumps(self.plan, ensure_ascii=False)},
            ],
            "meta": {"group": self.group, "source": self.source, **self.meta},
        }


def split_of(group: str, val_pct: int) -> str:
    h = int(hashlib.sha1(group.encode()).hexdigest(), 16)
    return "val" if h % 100 < val_pct else "train"


# ── the build ────────────────────────────────────────────────────────────

@dataclass
class BuildReport:
    drops: Counter = field(default_factory=Counter)
    drop_samples: dict[str, list[str]] = field(default_factory=dict)
    tool_drops: dict[str, Counter] = field(default_factory=dict)
    gaps: dict[str, str] = field(default_factory=dict)
    per_tool: Counter = field(default_factory=Counter)
    classifier: Counter = field(default_factory=Counter)


async def probe_vocab(session, tools: list[ToolSpec], rng: random.Random,
                      manifest: dict[str, dict]) -> Vocab:
    """Ask each string-keyed tool for its keys by handing it a bad one.
    Only REQUIRED strings are probed: an optional one has a working default,
    and junk in it can be expensive (a garbage simon_hidden_string kept
    quantum_algorithm_analysis busy, blocking the server, for minutes)."""
    vocab = Vocab()
    for t in tools:
        strs = [p for p in t.required if _param_type(t.params[p]) == "string"
                and t.name not in SYMBOLIC and p not in CURATED_STRINGS]
        if not strs:
            continue
        args = {}
        for p in t.required:
            spec = t.params[p]
            if _param_type(spec) == "string":
                args[p] = "__probe__"
            else:
                a = sample_numeric(p, _param_type(spec), t.name, rng)
                args[p] = a.value if a else 1.0
        try:
            res = await asyncio.wait_for(session.call_tool(t.name, args), 15)
            text = "\n".join(x.text for x in (res.content or []) if getattr(x, "text", None))
        except Exception:
            text = ""
        keys = parse_vocab(text)
        # fallback: the manifest's input docs often quote examples ('water')
        inputs = (manifest.get(t.name) or {}).get("inputs") or {}
        for p in strs:
            ks = keys or re.findall(r"'([\w\-]+)'", str(inputs.get(p, "")))
            if ks:
                vocab.by_tool[(t.name, p)] = ks
                merged = vocab.by_param.setdefault(p, [])
                merged += [k for k in ks if k not in merged]
    return vocab


def sample_args(t: ToolSpec, vocab: Vocab, rng: random.Random,
                optional_rate: float = 0.25) -> tuple[list[Arg] | None, str]:
    """Inputs for one call, or (None, why) when a required one can't be made."""
    n_str = sum(1 for p in t.params if _param_type(t.params[p]) == "string")
    if t.name == "convert_units":
        frm, to, fsay, tsay, lo, hi = rng.choice(CONVERSIONS)
        shown = _nice(_sample_range(lo, hi, True, rng))
        return [Arg("value", float(shown), f"{shown} {fsay}", "value", key_text=fsay),
                Arg("from_units", frm, "", "from", key_text=""),
                Arg("to_units", to, f"to {tsay}", "to", key_text=tsay)], ""
    args: list[Arg] = []
    for p, spec in t.params.items():
        required = p in t.required
        if not required and ("default" not in spec or rng.random() > optional_rate):
            continue
        ptype = _param_type(spec)
        if ptype in ("number", "integer"):
            a = sample_numeric(p, ptype, t.name, rng)
        elif ptype == "string":
            a = sample_string(t.name, p, vocab, rng, n_str)
        else:
            a = None
        if a is None:
            if required:
                return None, f"can't sample required {ptype} '{p}'"
            continue
        args.append(a)
    by_name = {a.name: a for a in args}
    for big, small in ORDERED_PAIRS:
        a, b = by_name.get(big), by_name.get(small)
        if a and b and isinstance(a.value, (int, float)) and a.value < b.value:
            a.value, b.value = b.value, a.value
            a.text, b.text = b.text, a.text
            a.unit_text, b.unit_text = b.unit_text, a.unit_text
        if a and b and a.value == b.value and isinstance(a.value, int):
            return None, "ordered pair collided"
    return args, ""


def _plan_args(args: list[Arg]) -> dict:
    return {a.name: a.value for a in args}


async def build(session, tools: list[ToolSpec], manifest: dict[str, dict], *,
                per_tool: int = 40, seed: int = 0, clarify_rate: float = 0.1,
                guard: BenchmarkGuard | None = None,
                teacher: Callable[[str, int], str] | None = None, variants: int = 0,
                log: Callable[[str], None] = print
                ) -> tuple[list[Example], BuildReport]:
    report = BuildReport()
    vocab = await probe_vocab(session, tools, random.Random(seed), manifest)
    examples: list[Example] = []
    seen: set[str] = set()

    def drop(reason: str, question: str, detail: str = "", tool: str = "") -> None:
        report.drops[reason] += 1
        if tool:
            report.tool_drops.setdefault(tool, Counter())[reason] += 1
        samples = report.drop_samples.setdefault(reason, [])
        if len(samples) < 8:
            samples.append(f"{question}  [{detail}]" if detail else question)

    def admit(ex: Example) -> bool:
        key = _norm(ex.question)
        tool = ex.meta.get("tool", "")
        if key in seen:
            report.drops["duplicate"] += 1
            if tool:
                report.tool_drops.setdefault(tool, Counter())["duplicate"] += 1
            return False
        verdict = classifier_verdict(ex.question, ex.plan)
        if verdict == "conflict":
            drop("classifier conflict", ex.question,
                 f"-> {classifier_route(ex.question)}", tool)
            return False
        if guard is not None:
            why = guard.check(ex.question, ex.plan.get("tool"),
                              ex.meta.get("verified", {}).get("value"))
            if why:
                drop("benchmark overlap", ex.question, why, tool)
                return False
        report.classifier[verdict] += 1
        ex.meta["classifier"] = verdict
        seen.add(key)
        examples.append(ex)
        return True

    for t in tools:
        if t.name == "request_clarification":
            continue
        rng = random.Random(f"{seed}:{t.name}")
        keys = [k for p in t.params for k in vocab.get(t.name, p)]
        kws = drop_contradicting(intent_phrases(t, manifest.get(t.name)), keys)
        made, attempts, last_gap = 0, 0, ""
        while made < per_tool and attempts < per_tool * 4:
            attempts += 1
            args, why = sample_args(t, vocab, rng)
            if args is None:
                if "can't sample" in why:
                    last_gap = why
                    break
                continue
            kw, ti = rng.choice(kws), rng.randrange(100)
            q = render_question(kw, args, ti, t.name)
            if t.name == "convert_units":
                q = f"Convert {args[0].text} {args[2].text}." if ti % 2 else \
                    f"How many {args[2].key_text} is {args[0].text}?"
            plan = {"tool": t.name, "args": _plan_args(args)}
            got = await execute(session, t.name, plan["args"], q)
            if got is None:
                drop("call returned no value", f"{t.name}({json.dumps(plan['args'])})",
                     tool=t.name)
                continue
            group = f"{t.name}:{made}:{seed}"
            ex = Example(q, plan, group, "template", args,
                         {"tool": t.name, "verified": got, "intent": kw})
            if not admit(ex):
                continue
            made += 1
            report.per_tool[t.name] += 1
            # missing-input negative: drop one required, defaultless number
            droppable = [a for a in args if a.name in t.required
                         and "default" not in t.params[a.name]
                         and isinstance(a.value, (int, float)) and not a.key_text]
            # a proper-noun intent ("WiFi wavelength") may imply the missing
            # input, so it can't teach "ask, don't guess"
            if droppable and not re.search(r"[A-Z]", kw[1:]) \
                    and rng.random() < clarify_rate:
                gone = rng.choice(droppable)
                rest = [a for a in args if a is not gone]
                cq = render_question(kw, rest, ti, t.name)
                cplan = {"tool": "request_clarification",
                         "args": {"variable": gone.name,
                                  "question": f"What is the {gone.noun}?"}}
                admit(Example(cq, cplan, group, "clarify", rest,
                              {"tool": t.name, "missing": gone.name, "intent": kw}))
        if made == 0:
            why_not = report.tool_drops.get(t.name)
            report.gaps[t.name] = last_gap or (
                ", ".join(f"{r} x{n}" for r, n in why_not.most_common())
                if why_not else "no example survived the checks")
        log(f"  {t.name:40s} {made:3d}" + (f"   ({report.gaps[t.name]})"
                                           if t.name in report.gaps else ""))

    for i, q in enumerate(DECLINE_SEEDS):
        admit(Example(q, {"decline": DECLINE_REASON}, f"decline:{i}", "decline"))

    if teacher is not None and variants > 0:
        base = [e for e in examples if e.source in ("template", "clarify", "decline")]
        log(f"teacher rewrites: {len(base)} seeds x {variants}")
        for e in base:
            for k in range(variants):
                try:
                    rw = clean_rewrite(teacher(e.question, seed * 1000 + k))
                except Exception:
                    report.drops["teacher error"] += 1
                    continue
                why = check_rewrite(e.question, rw, e.args)
                if why:
                    report.drops[f"rewrite: {why}"] += 1
                    continue
                admit(Example(rw, e.plan, e.group, "rewrite", e.args,
                              {**e.meta, "rewrite_of": e.question}))
    return examples, report


def write_dataset(examples: list[Example], report: BuildReport, out: Path,
                  tools: list[ToolSpec], val_pct: int, settings: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (out / "train.jsonl").open("w", encoding="utf-8") as ftr, \
            (out / "val.jsonl").open("w", encoding="utf-8") as fva:
        for e in examples:
            split = split_of(e.group, val_pct)
            (fva if split == "val" else ftr).write(
                json.dumps(e.to_record(), ensure_ascii=False, default=str) + "\n")
            counts[split] += 1
            counts[e.source] += 1
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool_surface_hash": surface_hash(tools),
        "visible_tools": len(tools),
        "tools_with_examples": len(report.per_tool),
        "settings": settings,
        "counts": dict(counts),
        "classifier": dict(report.classifier),
        "drops": dict(report.drops),
        "drop_samples": report.drop_samples,
        "drops_by_tool": {t: dict(c) for t, c in sorted(report.tool_drops.items())},
        "per_tool": dict(sorted(report.per_tool.items())),
        "gaps": dict(sorted(report.gaps.items())),
        "system_prompt": ROUTER_SYSTEM_PROMPT,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


async def visible_tools(session) -> list[ToolSpec]:
    """Exactly the surface the harness shows the model."""
    from sigma_ground.mcp.benchmark.new_tools_classifier import HIDDEN_FROM_LLM
    resp = await session.list_tools()
    return [ToolSpec(t.name, t.description or "", t.inputSchema or {})
            for t in resp.tools if t.name not in HIDDEN_FROM_LLM]


def manifest_by_tool() -> dict[str, dict]:
    from sigma_ground.mcp.manifest import get_manifest
    m = get_manifest().value
    return {e["name"]: e for e in (m.get("primary") or []) + (m.get("extended") or [])}


def server_params(mem_gb: float = 0):
    """Spawn the Mentat server; on POSIX optionally under an address-space
    ceiling, so a tool that runs away on a sampled input raises MemoryError
    (returned as a tool error) instead of taking the machine's memory and
    getting the whole run OOM-killed."""
    from mcp import StdioServerParameters
    code = "import runpy; runpy.run_module('sigma_ground.mcp.server', run_name='__main__')"
    if mem_gb and sys.platform != "win32":
        lim = int(mem_gb * 2 ** 30)
        code = f"import resource; resource.setrlimit(resource.RLIMIT_AS, ({lim}, {lim})); " + code
    return StdioServerParameters(command=sys.executable, args=["-c", code],
                                 env=dict(os.environ))


async def _amain(args) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = server_params(args.server_mem_gb)
    teacher = (ollama_teacher(args.teacher_url, args.teacher_model)
               if args.teacher_model else None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await visible_tools(session)
            if args.tools:
                wanted = set(args.tools.split(","))
                tools = [t for t in tools if t.name in wanted]
            print(f"{len(tools)} tools; generating up to {args.per_tool} each")
            examples, report = await build(
                session, tools, manifest_by_tool(), per_tool=args.per_tool,
                seed=args.seed, clarify_rate=args.clarify_rate,
                guard=None if args.no_guard else BenchmarkGuard.from_repo(),
                teacher=teacher,
                variants=args.variants)
    settings = {k: v for k, v in vars(args).items() if k != "out"}
    manifest = write_dataset(examples, report, args.out, tools, args.val_pct, settings)
    print(f"\nWrote {args.out}: {manifest['counts']}")
    print(f"tools with examples: {manifest['tools_with_examples']}/{len(tools)}; "
          f"gaps: {len(manifest['gaps'])}; drops: {manifest['drops']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--per-tool", type=int, default=40,
                   help="verified template questions per tool (before rewrites)")
    p.add_argument("--tools", default="", help="comma-separated subset (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--clarify-rate", type=float, default=0.1,
                   help="fraction of seeds that also yield a missing-input example")
    p.add_argument("--teacher-model", default="",
                   help="Ollama model that rewrites template questions naturally "
                        "(e.g. qwen2.5:32b); omit to skip the teacher pass")
    p.add_argument("--teacher-url", default="http://localhost:11434")
    p.add_argument("--variants", type=int, default=3,
                   help="teacher rewrites attempted per seed")
    p.add_argument("--no-guard", action="store_true",
                   help="keep examples that resemble benchmark questions. Use for a "
                        "router you will DEPLOY (the guard otherwise drops every "
                        "example of a few zero-argument tools); never report "
                        "benchmark scores from a model trained this way")
    p.add_argument("--server-mem-gb", type=float, default=6,
                   help="address-space ceiling for the spawned MCP server (POSIX; "
                        "0 = none)")
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
