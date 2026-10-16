"""One vocabulary for tool parameter names: the canonical names, the old names that are still accepted, and the helpers every consumer shares.

misc/CONTRACT_RENAMES.md proposed collapsing 310 parameter names that say the same thing (`material`, `material_key`, `metal_key`; `t_hot`,
`hot_temperature_k`, `t_hot_k`; `b_tesla`, `magnetic_field_t`, `b_field_t` ...). The Captain approved it on 2026-10-15 ("sure"). The Python functions keep
their own parameter names; the contract (mcp/contract.py) shows the canonical names in the schema, accepts the old ones as aliases, and answers with the
canonical ones. This module is the only place the map lives, so the schema, the training-data builder, the lint and the router harness cannot drift apart.

A rename is applied to a tool only when it is unambiguous there: the new name is not already one of that tool's parameters, and two old names never
collapse into one (wetting_analysis keeps `solid_key` and `liquid_key`: they are two roles, not two spellings).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

# old name -> canonical name, for every tool that has the old name
RENAMES: dict[str, str] = {
    # a material from the table
    "material_key": "material", "metal_key": "material",
    # two materials
    "material_1": "material_a", "material_2": "material_b",
    "material_key_1": "material_a", "material_key_2": "material_b",
    "mat_a": "material_a", "mat_b": "material_b",
    "metal_a": "material_a", "metal_b": "material_b",
    "mat_n": "material_n", "mat_p": "material_p",
    "semiconductor_key": "semiconductor",
    # hot / cold reservoir temperatures
    "t_hot": "t_hot_k", "t_cold": "t_cold_k",
    "hot_temperature_k": "t_hot_k", "cold_temperature_k": "t_cold_k",
    # the same quantity with the unit in the name, one spelling each
    "energy_eV": "energy_ev", "energy_joules": "energy_j",
    "b_tesla": "b_field_t", "magnetic_field_t": "b_field_t",
    "molar_mass_kg": "molar_mass_kg_per_mol",
    "incident_angle_deg": "incidence_angle_deg",
}

# old name -> canonical name, for ONE tool (the old name means something else, or needs the tool to read right, elsewhere)
TOOL_RENAMES: dict[str, dict[str, str]] = {
    "semiconductor_junction_analysis": {"sc_key": "semiconductor"},
    "natural_convection_analysis": {"gas_key": "material_a", "gas_key_2": "material_b"},      # the two gases of a binary diffusivity
}


def rename_map(tool: str, params: Iterable[str]) -> dict[str, str]:
    """{old: canonical} for the parameters of `tool` that are renamed. `params` are the function's own (old-spelling) parameter names."""
    names = list(params)
    table = {**RENAMES, **TOOL_RENAMES.get(tool, {})}
    wanted = {old: table[old] for old in names if old in table and table[old] != old}
    claims = Counter(wanted.values())
    return {old: new for old, new in wanted.items() if new not in names and claims[new] == 1}


def inverse(mapping: Mapping[str, str]) -> dict[str, str]:
    return {new: old for old, new in mapping.items()}


def canonical_args(tool: str, args: Mapping, params: Iterable[str]) -> dict:
    """A plan's arguments with old parameter names replaced by the canonical ones (values untouched; an argument already canonical wins)."""
    rm = rename_map(tool, params)
    out: dict = {}
    for k, v in args.items():
        new = rm.get(k, k)
        if new != k and new in args:
            continue
        out[new] = v
    return out


def accepted_old_names(tool: str, canonical_params: Iterable[str]) -> set[str]:
    """The old names the server still accepts for `tool`, worked out from the canonical schema alone (what a client that only sees the schema knows)."""
    names = set(canonical_params)
    table = {**RENAMES, **TOOL_RENAMES.get(tool, {})}
    return {old for old, new in table.items() if new in names and old not in names and old != new}
