"""Procedures — named, code-defined multi-step scientific routines.

A Procedure is a CANONICAL WORKFLOW scientists run repeatedly, expressed as a
fixed sequence of library-tool calls: "compute r_s, then the Hawking
temperature, then the entropy, then the evaporation time." Because the
SEQUENCE lives in code — not in the LLM's head — the steps can never be
mis-ordered, mis-chained, or mis-interpreted. The model's only job is to
recognize WHICH procedure the user wants and hand it the inputs; the
procedure does the rest, deterministically.

Each step records its own value, formula, and source, so a procedure is also
its own worked example ("show your work"). This is the layer ABOVE single
tools: tools are one arithmetic step; procedures are validated multi-step
recipes built from them.

To add one: write a function that calls library tools in order, appending a
Step per stage, and register it in PROCEDURES.
"""
from __future__ import annotations

from dataclasses import dataclass

_C = 299792458.0
_EV_J = 1.602176634e-19
_H = 6.62607015e-34


@dataclass
class Step:
    label: str
    value: object
    units: str = ""
    formula: str = ""
    source: str = ""


@dataclass
class ProcedureResult:
    name: str
    inputs: dict
    steps: list           # list[Step]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "procedure": self.name,
            "inputs": self.inputs,
            "steps": [vars(s) for s in self.steps],
            "result": vars(self.steps[-1]) if self.steps else None,
            "summary": self.summary,
        }


def _from_tool(label, tr, formula=None, units=None):
    """Build a Step from a library ToolResult (keeps its provenance)."""
    return Step(label, getattr(tr, "value", tr),
                units if units is not None else (getattr(tr, "units", "") or ""),
                formula or (getattr(tr, "formula", "") or ""),
                getattr(tr, "source", "") or "")


# ── PROCEDURE: black-hole thermodynamic profile ─────────────────────────
def black_hole_profile(mass_kg: float) -> ProcedureResult:
    """r_s → Hawking T → Bekenstein-Hawking entropy → evaporation time.

    The standard "everything about this black hole" cascade. One input
    (mass), four library-validated outputs, in the only correct order.
    """
    from sigma_ground.mcp.tools.gr import (schwarzschild_radius,
                                            hawking_evaporation_time)
    from sigma_ground.mcp.tools.frontier import bekenstein_hawking_entropy
    s: list[Step] = []
    rs = schwarzschild_radius(mass_kg)
    s.append(_from_tool("Schwarzschild radius r_s", rs))
    bk = bekenstein_hawking_entropy(mass_kg).value
    s.append(Step("Hawking temperature", bk["hawking_temperature_K"], "K",
                   "T = ħc³/(8πGMk_B)", "sigma-ground (Hawking 1974)"))
    s.append(Step("Bekenstein-Hawking entropy", bk["entropy_k_B"], "k_B",
                   "S = A/(4 L_p²)", "sigma-ground (Bekenstein 1973)"))
    ev = hawking_evaporation_time(mass_kg)
    s.append(_from_tool("Evaporation time", ev))
    return ProcedureResult("black_hole_profile", {"mass_kg": mass_kg}, s,
                            summary=f"Thermodynamic profile of a {mass_kg:.3g} kg black hole.")


# ── PROCEDURE: full photon properties ───────────────────────────────────
def photon_spectrum(wavelength_m: float) -> ProcedureResult:
    """λ → frequency → energy (J) → energy (eV) → momentum.

    One wavelength, the whole photon. Every downstream value is derived from
    the previous, in code, so the chain is exact.
    """
    from sigma_ground.mcp.tools.atomic import photon_energy_from_wavelength
    s: list[Step] = []
    s.append(Step("Wavelength", wavelength_m, "m", "(input)", "user"))
    freq = _C / wavelength_m
    s.append(Step("Frequency", freq, "Hz", "f = c / λ", "sigma-ground"))
    e = photon_energy_from_wavelength(wavelength_m)
    s.append(_from_tool("Energy", e))
    e_ev = e.value / _EV_J
    s.append(Step("Energy", e_ev, "eV", "E / e", "sigma-ground"))
    p = e.value / _C
    s.append(Step("Momentum", p, "kg·m/s", "p = E / c = h / λ", "sigma-ground"))
    return ProcedureResult("photon_spectrum", {"wavelength_m": wavelength_m}, s,
                            summary=f"Full properties of a {wavelength_m*1e9:.1f} nm photon.")


# ── PROCEDURE: relativistic particle cascade ────────────────────────────
def relativistic_particle(kinetic_energy_eV: float,
                           particle: str = "electron") -> ProcedureResult:
    """KE → γ → total energy → momentum → de Broglie wavelength.

    The relativistic-kinematics chain a physicist runs for any fast particle.
    Particles: electron, proton, neutron, alpha, muon (one CODATA rest mass
    per particle for every step: the de Broglie tool's table).

    FIXED 2026-10-10 (A43): unknown particles and "alpha" silently used the
    electron's rest energy, and the wavelength step used a different mass
    from gamma and pc; pc = sqrt(E^2 - (mc^2)^2) cancelled catastrophically
    for slow particles (a 100 neV neutron's pc was 17 % high). Now unknown
    particles and KE <= 0 or non-finite raise ValueError; pc is
    sqrt(KE (KE + 2 mc^2)) and lambda = h c / pc from that same pc.

    Raises:
        ValueError: unknown particle, or kinetic_energy_eV not positive and finite.
    """
    import math
    from sigma_ground.mcp.tools.atomic import _PARTICLE_REST_MASS_KG
    key = str(particle or "").strip().lower()
    key = {"alpha particle": "alpha", "helium nucleus": "alpha",
           "helium-4 nucleus": "alpha"}.get(key, key)
    m_kg = _PARTICLE_REST_MASS_KG.get(key)
    if m_kg is None:
        raise ValueError(f"unknown particle {particle!r}: known particles are "
                         f"{sorted(_PARTICLE_REST_MASS_KG)}")
    if not (isinstance(kinetic_energy_eV, (int, float))
            and math.isfinite(kinetic_energy_eV) and kinetic_energy_eV > 0):
        raise ValueError(f"kinetic_energy_eV must be positive and finite "
                         f"(got {kinetic_energy_eV!r})")
    KE = float(kinetic_energy_eV)
    rest_eV = m_kg * _C * _C / _EV_J                 # m c^2 in eV
    s: list[Step] = []
    s.append(Step("Kinetic energy", KE, "eV", "(input)", "user"))
    s.append(Step("Rest energy m c²", rest_eV, "eV", "m c² / e",
                  "CODATA 2018 mass (mcp.tools.atomic)"))
    gamma = 1.0 + KE / rest_eV
    s.append(Step("Lorentz factor γ", gamma, "", "γ = 1 + KE/(m c²)", "sigma-ground"))
    total = KE + rest_eV
    s.append(Step("Total energy", total, "eV", "E = KE + m c²", "sigma-ground"))
    pc = math.sqrt(KE * (KE + 2.0 * rest_eV))      # = √(E² − (mc²)²), no cancellation
    s.append(Step("Momentum (pc)", pc, "eV", "pc = √(KE (KE + 2 m c²))", "sigma-ground"))
    lam = _H * _C / (pc * _EV_J)
    s.append(Step("de Broglie wavelength", lam, "m", "λ = h / p = h c / (pc)",
                  "sigma-ground (de Broglie 1924)"))
    return ProcedureResult("relativistic_particle",
                            {"kinetic_energy_eV": kinetic_energy_eV, "particle": key}, s,
                            summary=f"Relativistic profile of a {KE:g} eV {key}.")


# ── PROCEDURE: projectile trajectory ────────────────────────────────────
def projectile_trajectory(initial_speed_m_s: float,
                           launch_angle_deg: float) -> ProcedureResult:
    """Time of flight → range → max height (no air resistance).

    The canonical intro-mechanics three-step: one launch, three outputs.
    """
    from sigma_ground.mcp.tools.kinematics import (projectile_flight_time,
        projectile_range, projectile_max_height)
    s: list[Step] = [
        Step("Initial speed", initial_speed_m_s, "m/s", "(input)", "user"),
        Step("Launch angle", launch_angle_deg, "deg", "(input)", "user"),
    ]
    s.append(_from_tool("Time of flight",
                         projectile_flight_time(initial_speed_m_s, launch_angle_deg)))
    s.append(_from_tool("Range",
                         projectile_range(initial_speed_m_s, launch_angle_deg)))
    s.append(_from_tool("Max height",
                         projectile_max_height(initial_speed_m_s, launch_angle_deg)))
    return ProcedureResult("projectile_trajectory",
                            {"initial_speed_m_s": initial_speed_m_s,
                             "launch_angle_deg": launch_angle_deg}, s,
                            summary=(f"Trajectory of a projectile launched at "
                                      f"{initial_speed_m_s} m/s, {launch_angle_deg}°."))


# ── PROCEDURE: stellar blackbody ────────────────────────────────────────
def stellar_blackbody(temperature_k: float) -> ProcedureResult:
    """Wien peak wavelength → Stefan-Boltzmann surface flux → peak photon energy.

    The standard "characterize a blackbody / star surface" cascade.
    """
    from sigma_ground.mcp.tools.thermodynamics import (blackbody_peak_wavelength,
                                                        blackbody_total_power)
    s: list[Step] = [Step("Temperature", temperature_k, "K", "(input)", "user")]
    wl = blackbody_peak_wavelength(temperature_k)
    s.append(_from_tool("Peak wavelength (Wien)", wl))
    s.append(_from_tool("Surface flux (Stefan-Boltzmann)",
                         blackbody_total_power(temperature_k, 1.0)))
    e_peak = _H * _C / wl.value / _EV_J
    s.append(Step("Peak photon energy", e_peak, "eV", "E = h c / λ_peak", "sigma-ground"))
    return ProcedureResult("stellar_blackbody", {"temperature_k": temperature_k}, s,
                            summary=f"Blackbody profile at {temperature_k} K.")


PROCEDURES = {
    "black_hole_profile": black_hole_profile,
    "photon_spectrum": photon_spectrum,
    "relativistic_particle": relativistic_particle,
    "projectile_trajectory": projectile_trajectory,
    "stellar_blackbody": stellar_blackbody,
}
