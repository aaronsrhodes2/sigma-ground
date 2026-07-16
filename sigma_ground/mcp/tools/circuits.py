"""Electrical circuits: Ohm's law, capacitance, RC/RL/RLC time constants.

Pure formulas. Combine with materials.py for resistivity-of-copper-style
lookups (deferred to future expansion).
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


def ohms_law_voltage(current_a: float, resistance_ohm: float) -> ToolResult:
    """V = I R."""
    if resistance_ohm < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="resistance_ohm must be non-negative",
                           inputs={"current_a": current_a,
                                   "resistance_ohm": resistance_ohm})
    V = current_a * resistance_ohm
    return ToolResult(
        value=V,
        units="V",
        source="sigma-ground (Ohm's law)",
        formula="V = I R",
        inputs={"current_a": current_a, "resistance_ohm": resistance_ohm},
    )


def ohms_law_current(voltage_v: float, resistance_ohm: float) -> ToolResult:
    """I = V / R."""
    if resistance_ohm <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="resistance_ohm must be positive",
                           inputs={"voltage_v": voltage_v,
                                   "resistance_ohm": resistance_ohm})
    I = voltage_v / resistance_ohm
    return ToolResult(
        value=I,
        units="A",
        source="sigma-ground (Ohm's law)",
        formula="I = V / R",
        inputs={"voltage_v": voltage_v, "resistance_ohm": resistance_ohm},
    )


def electrical_power(voltage_v: float, current_a: float) -> ToolResult:
    """P = V I = I^2 R = V^2 / R."""
    P = voltage_v * current_a
    return ToolResult(
        value=P,
        units="W",
        source="sigma-ground (electrical power)",
        formula="P = V I",
        inputs={"voltage_v": voltage_v, "current_a": current_a},
        notes="Equivalent forms: P = I^2 R = V^2 / R.",
    )


def energy_power_time(power_w: float | None = None,
                        time_s: float | None = None,
                        energy_j: float | None = None) -> ToolResult:
    """Solve E = P · t for whichever value is missing. Provide exactly two.

    Collapses the "5 kW heater for 1 hour = how many joules?" and
    "5 W LED to dissipate 1 kJ = how long?" questions into one tool —
    the LLM just supplies the two it knows and leaves the unknown as None.

    Examples:
      energy_power_time(power_w=5000, time_s=3600)      # -> energy_j
      energy_power_time(power_w=5, energy_j=1000)       # -> time_s
      energy_power_time(time_s=3600, energy_j=1.8e7)    # -> power_w
    """
    known = [x is not None for x in (power_w, time_s, energy_j)]
    if sum(known) != 2:
        return ToolResult(value=None, source="invalid input",
                           notes="Provide exactly two of power_w, time_s, energy_j.",
                           inputs={"power_w": power_w, "time_s": time_s,
                                   "energy_j": energy_j})
    if energy_j is None:
        val, units, solved = power_w * time_s, "J", "energy_j"
        formula = "E = P t"
    elif time_s is None:
        if power_w == 0:
            return ToolResult(value=None, source="invalid input",
                               notes="power_w is zero.", inputs={})
        val, units, solved = energy_j / power_w, "s", "time_s"
        formula = "t = E / P"
    else:  # power_w is None
        if time_s == 0:
            return ToolResult(value=None, source="invalid input",
                               notes="time_s is zero.", inputs={})
        val, units, solved = energy_j / time_s, "W", "power_w"
        formula = "P = E / t"
    return ToolResult(
        value=val, units=units,
        source="sigma-ground (energy = power × time)",
        formula=formula,
        inputs={"power_w": power_w, "time_s": time_s, "energy_j": energy_j,
                "solved_for": solved},
        notes=f"Solved for {solved}. 1 hour = 3600 s; 1 kWh = 3.6e6 J.",
    )


def power_dissipation_resistor(current_a: float,
                                 resistance_ohm: float) -> ToolResult:
    """P = I^2 R. Power dissipated as heat in a resistor."""
    if resistance_ohm < 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"current_a": current_a,
                                   "resistance_ohm": resistance_ohm})
    P = current_a ** 2 * resistance_ohm
    return ToolResult(
        value=P,
        units="W",
        source="sigma-ground (Joule heating)",
        formula="P = I^2 R",
        inputs={"current_a": current_a, "resistance_ohm": resistance_ohm},
    )


def parallel_plate_capacitance(area_m2: float, separation_m: float,
                                 dielectric_constant: float = 1.0) -> ToolResult:
    """C = epsilon_0 epsilon_r A / d."""
    if area_m2 <= 0 or separation_m <= 0 or dielectric_constant < 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"area_m2": area_m2,
                                   "separation_m": separation_m,
                                   "dielectric_constant": dielectric_constant})
    from sigma_ground.field.constants import EPS_0
    C = EPS_0 * dielectric_constant * area_m2 / separation_m
    return ToolResult(
        value=C,
        units="F",
        source="sigma-ground (parallel-plate capacitor)",
        formula="C = eps_0 eps_r A / d",
        inputs={"area_m2": area_m2, "separation_m": separation_m,
                "dielectric_constant": dielectric_constant},
        notes=("Vacuum gap: eps_r=1. Common dielectrics: paper~3, glass~5-10, "
                "water~80, vacuum ferroelectric~1000+. Edge effects ignored "
                "(plate dimensions much larger than gap)."),
    )


def rc_time_constant(resistance_ohm: float, capacitance_f: float) -> ToolResult:
    """tau = R C. Time for voltage to reach 63% of asymptotic value."""
    if resistance_ohm <= 0 or capacitance_f <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"resistance_ohm": resistance_ohm,
                                   "capacitance_f": capacitance_f})
    tau = resistance_ohm * capacitance_f
    return ToolResult(
        value=tau,
        units="s",
        source="sigma-ground (RC circuit analysis)",
        formula="tau = R C",
        inputs={"resistance_ohm": resistance_ohm,
                "capacitance_f": capacitance_f},
        notes=("After 1 tau, capacitor is 63% charged. After 5 tau, "
                "effectively fully charged (>99%). Same formula for "
                "discharge: 63% discharged at t=tau."),
    )


def rl_time_constant(resistance_ohm: float, inductance_h: float) -> ToolResult:
    """tau = L / R for an RL circuit."""
    if resistance_ohm <= 0 or inductance_h <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"resistance_ohm": resistance_ohm,
                                   "inductance_h": inductance_h})
    tau = inductance_h / resistance_ohm
    return ToolResult(
        value=tau,
        units="s",
        source="sigma-ground (RL circuit)",
        formula="tau = L / R",
        inputs={"resistance_ohm": resistance_ohm,
                "inductance_h": inductance_h},
        notes="Current reaches 63% of asymptotic value in 1 tau.",
    )


def rlc_resonant_frequency(inductance_h: float,
                             capacitance_f: float) -> ToolResult:
    """f = 1 / (2 pi sqrt(L C)). RLC undamped resonance.

    Returns the frequency in Hz (what "resonant frequency" means in
    ordinary usage). Angular frequency omega_0 = 2 pi f is in notes.
    """
    if inductance_h <= 0 or capacitance_f <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"inductance_h": inductance_h,
                                   "capacitance_f": capacitance_f})
    omega = 1.0 / math.sqrt(inductance_h * capacitance_f)
    freq_hz = omega / (2.0 * math.pi)
    return ToolResult(
        value=freq_hz,
        units="Hz",
        source="sigma-ground (RLC series circuit resonance)",
        formula="f = 1 / (2 pi sqrt(L C))",
        inputs={"inductance_h": inductance_h,
                "capacitance_f": capacitance_f},
        notes=(f"Angular frequency: omega_0 = {omega:.3e} rad/s. Resonant "
                f"frequency where impedance is purely resistive. Tuner "
                f"circuits select stations by sweeping this."),
    )


def em_wave_wavelength(frequency_hz: float,
                         refractive_index: float = 1.0) -> ToolResult:
    """Wavelength of EM wave: lambda = c / (n f).

    In vacuum n=1, in matter n>1 reduces wavelength.
    """
    if frequency_hz <= 0 or refractive_index <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"frequency_hz": frequency_hz,
                                   "refractive_index": refractive_index})
    from sigma_ground.field.constants import C
    lam = C / (refractive_index * frequency_hz)
    return ToolResult(
        value=lam,
        units="m",
        source="sigma-ground (EM wave dispersion in medium)",
        formula="lambda = c / (n f)",
        inputs={"frequency_hz": frequency_hz,
                "refractive_index": refractive_index},
    )


def em_wave_frequency(wavelength_m: float,
                        refractive_index: float = 1.0) -> ToolResult:
    """Frequency from wavelength: f = c / (n lambda)."""
    if wavelength_m <= 0 or refractive_index <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"wavelength_m": wavelength_m,
                                   "refractive_index": refractive_index})
    from sigma_ground.field.constants import C
    f = C / (refractive_index * wavelength_m)
    return ToolResult(
        value=f,
        units="Hz",
        source="sigma-ground (EM wave dispersion)",
        formula="f = c / (n lambda)",
        inputs={"wavelength_m": wavelength_m,
                "refractive_index": refractive_index},
    )
