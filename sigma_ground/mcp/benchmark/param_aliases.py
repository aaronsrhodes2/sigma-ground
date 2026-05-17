"""Tool-argument alias normalization for the benchmark runner.

When Qwen invents a parameter name like `gravity_ms2` instead of the
real `g_m_s2`, the MCP server's pydantic layer silently drops the
unknown kwarg and uses the default. The free-fall-on-Moon question
became free-fall-on-Earth that way. This module catches that BEFORE
the MCP call: we know each tool's real param names (from list_tools),
so we rename common Qwen-isms to the canonical form.

Algorithm per kwarg:
  1. Look up its canonical form in PARAM_ALIASES.
  2. If the canonical form is in the target tool's real params AND the
     original alias is NOT, rename. (Avoids overriding intentional
     same-named params on tools that legitimately use the alias name.)
  3. Otherwise leave alone (pydantic will then catch it as an error,
     which Qwen can see and correct).
"""

from __future__ import annotations


# Common Qwen-isms that map cleanly to canonical sigma-ground param names.
# Each entry is alias -> canonical. The runner only applies a rename if
# the canonical name actually exists in the target tool's signature.
PARAM_ALIASES: dict[str, str] = {
    # ===== Gravity / acceleration =====
    "gravity_ms2": "g_m_s2",
    "gravity_m_s2": "g_m_s2",
    "gravity": "g_m_s2",
    "grav_m_s2": "g_m_s2",
    "g": "g_m_s2",  # only renamed if the tool has g_m_s2 and not g

    # ===== Speed / velocity =====
    "initial_velocity_m_s": "initial_speed_m_s",
    "initial_velocity": "initial_speed_m_s",
    "v0_m_s": "initial_speed_m_s",
    "v0": "initial_speed_m_s",
    "velocity_m_s": "speed_m_s",
    "velocity": "speed_m_s",
    "v_m_s": "speed_m_s",  # only renamed if target tool has speed_m_s
    "v": "speed_m_s",

    # Relativistic velocity addition uses u_m_s, v_m_s -- but Qwen uses v1, v2.
    # NOTE: We can't simply rename v1->u_m_s globally; it's specific to
    # the relativistic_velocity_addition tool. The runner's context-aware
    # logic handles this: 'v1' is only renamed to 'u_m_s' if u_m_s is in
    # the target tool's params (which it is, only for that tool).
    "v1": "u_m_s",
    "v2": "v_m_s",

    # ===== Angles =====
    "launch_angle_degrees": "launch_angle_deg",
    "angle_degrees": "launch_angle_deg",
    "angle_deg_launch": "launch_angle_deg",
    "theta_deg": "launch_angle_deg",
    "theta_degrees": "launch_angle_deg",
    "theta": "launch_angle_deg",
    "incidence_angle_degrees": "incidence_angle_deg",
    "angle_of_incidence_deg": "incidence_angle_deg",

    # ===== Mass =====
    "mass": "mass_kg",
    "m_kg": "mass_kg",
    "m": "mass_kg",  # ambiguous; only renamed if target has mass_kg
    "M": "mass_kg",
    "mass_planet": "mass_planet_kg",
    "mass_object": "mass_object_kg",
    "M_planet": "mass_planet_kg",
    "m_object": "mass_object_kg",

    # ===== Lengths / distances =====
    "altitude_m": "height_m",
    "h": "height_m",  # only renamed if target has height_m
    "height": "height_m",
    "distance": "distance_m",
    "r": "radius_m",
    "radius": "radius_m",

    # ===== Time =====
    "t_s": "time_s",
    "time": "time_s",

    # ===== Energy =====
    "energy": "energy_j",
    "E": "energy_j",

    # ===== Electrical =====
    "voltage": "voltage_v",
    "current": "current_a",
    "resistance": "resistance_ohm",
    "R": "resistance_ohm",
    "capacitance": "capacitance_f",
    "C": "capacitance_f",
    "inductance": "inductance_h",
    "L": "inductance_h",

    # ===== Frequency / wavelength =====
    "frequency": "frequency_hz",
    "f": "frequency_hz",
    "f_hz": "frequency_hz",
    "wavelength": "wavelength_m",
    "lambda": "wavelength_m",
    "lambda_m": "wavelength_m",

    # ===== Temperature =====
    "T": "temperature_k",
    "temp_k": "temperature_k",
    "temperature": "temperature_k",
    "temp_c": "temperature_c",
    "temp_celsius": "temperature_c",

    # ===== Refractive index =====
    "n1": "n_from",
    "n2": "n_to",
    "n_incident": "n_from",
    "n_transmitted": "n_to",

    # ===== Quantum / atomic =====
    "Z": "atomic_number",
    "n": "n",  # principal quantum number; many tools use n directly
    "principal_quantum_number": "n",

    # ===== Body / element / star name =====
    "name": "body_name",  # generic; renamed only if target has body_name
    "planet_name": "body_name",
    "planet": "body_name",
    "body": "body_name",
    "object_name": "body_name",
    "element": "element_symbol",
    "atom": "element_symbol",
    "symbol": "element_symbol",
    "star": "star_name",
    "material_name": "material",
}


def normalize_kwargs(kwargs: dict, real_params: set[str]) -> tuple[dict, list[str]]:
    """Rename Qwen-style aliases to the target tool's canonical params.

    Parameters
    ----------
    kwargs : dict
        The arguments Qwen passed.
    real_params : set[str]
        The actual parameter names the target tool's signature accepts
        (from list_tools' inputSchema.properties).

    Returns
    -------
    (normalized_kwargs, renames)
        normalized_kwargs is a NEW dict (the input is not mutated).
        renames is a list of "alias -> canonical" strings for logging.
    """
    out: dict = {}
    renames: list[str] = []
    for k, v in kwargs.items():
        if k in real_params:
            # Already the canonical name -- leave alone
            out[k] = v
            continue
        canonical = PARAM_ALIASES.get(k)
        if canonical is None or canonical == k:
            # No alias or alias points to itself; pass through (pydantic
            # will reject if it's actually unknown, or accept if it is).
            out[k] = v
            continue
        if canonical in real_params:
            # The canonical form IS what the tool wants -- rename it.
            out[canonical] = v
            renames.append(f"{k}->{canonical}")
            continue
        # Canonical not in real_params either; pass through.
        out[k] = v
    return out, renames
