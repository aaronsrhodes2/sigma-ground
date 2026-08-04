"""
Propulsion — rocket-equation kinematics and a curated real-engine catalog.

STANDALONE physics module (P0a of the game-engine plan): the Tsiolkovsky
rocket equation and its inverse, the Isp <-> exhaust-velocity convention,
mass-flow rate from thrust, and a sourced table of real propulsion-system
specific impulses. Two thin adapters translate a burn into the closed
artificial-force taxonomy (dynamics.mechanisms.applied) so a caller can turn
"burn the engine" into a disclosed AppliedForce without duplicating the
physics. No ECS component, no fuel-tank state machine, no engine-loop
integration lives here — those are later phases (E1+).

Tsiolkovsky rocket equation (Tsiolkovsky, 1903):
  FIRST_PRINCIPLES: conservation of momentum for a body that continuously
  ejects mass at a constant velocity relative to itself. Integrating
  dp = 0 over the burn (m dv = -v_e dm, rocket frame) gives
    delta_v = v_e * ln(m0 / m1)
  Tsiolkovsky, K.E. (1903), "Исследование мировых пространств реактивными
  приборами" ("The Exploration of Cosmic Space by Means of Reaction
  Devices"), Nauchnoye Obozreniye. Standard modern derivation/citation:
  Sutton, G.P. & Biblarz, O., "Rocket Propulsion Elements," 9th ed.
  (Wiley, 2016), Eq. 3-33.

Specific impulse <-> exhaust velocity (the Isp-in-seconds convention):
  FIRST_PRINCIPLES: Isp is DEFINED as v_e / g0, where g0 is the
  internationally defined STANDARD (sea-level, "one-gee") gravitational
  acceleration, a fixed constant by convention -- NOT the local gravity
  the engine actually fires in. The seconds unit falls out because
  Isp = v_e / g0 has units of (m/s)/(m/s^2) = s. This is why a vacuum
  ion engine's Isp is quoted in seconds even though it never touches
  Earth's surface: g0 is a units-conversion constant, not a physical
  input to the engine. Standard gravity g0 = 9.80665 m/s^2 was fixed by
  the 3rd CGPM (1901) and is used by this convention throughout
  astronautics (Sutton & Biblarz, 9th ed., Eq. 2-14 and surrounding text;
  also NIST SP 811 / ISO 80000-3 for the defined value of g0 itself).

Thrust / mass-flow relation:
  FIRST_PRINCIPLES: thrust is the momentum flux of the exhaust jet,
  F = v_e * mdot (ideal nozzle, matched or negligible pressure-thrust
  term dropped). Sutton & Biblarz, 9th ed., Eq. 3-2 restricted to its
  momentum-thrust term.

Doctrine: every number below is a FACT (a real tested/flown engine's
published Isp, cited), a DEDUCTION (the four functions, each a traceable
formula chain from Tsiolkovsky/Sutton & Biblarz), or would be an explicit
Choice if invented -- nothing here is invented. See
sigma_ground/dynamics/mechanisms/choice.py for the Choice provenance
pattern this module deliberately does NOT need: every catalog entry
traces to a specific tested or flown engine, so none of them are Choices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ...dynamics.mechanisms.applied import AppliedForce

# ── the Isp <-> exhaust-velocity convention constant ───────────────────────
# FACT: internationally defined standard gravity, fixed by the 3rd CGPM
# (1901) at exactly 9.80665 m/s^2 (NIST SP 811; ISO 80000-3). This is a
# DEFINED constant, not a measurement of local gravity anywhere -- it is
# the units-conversion constant the Isp-in-seconds convention is built on
# (see module docstring).
STANDARD_GRAVITY_M_S2 = 9.80665


# ═══════════════════════════════════════════════════════════════════════════
# § 1. Rocket-equation kinematics
# ═══════════════════════════════════════════════════════════════════════════

def tsiolkovsky_delta_v(exhaust_velocity_m_s: float, initial_mass_kg: float,
                        final_mass_kg: float) -> float:
    """Delta-v = v_e * ln(m0 / m1) -- the Tsiolkovsky rocket equation.

    FIRST_PRINCIPLES: conservation of momentum, constant exhaust velocity
    relative to the rocket. Tsiolkovsky (1903); Sutton & Biblarz, "Rocket
    Propulsion Elements," 9th ed., Eq. 3-33.

    Parameters
    ----------
    exhaust_velocity_m_s : effective exhaust velocity v_e, m/s (> 0)
    initial_mass_kg : wet mass m0 (propellant + dry), kg
    final_mass_kg : dry mass m1 after the burn, kg (0 < m1 <= m0)

    Returns
    -------
    delta-v in m/s.
    """
    if exhaust_velocity_m_s <= 0:
        raise ValueError("exhaust_velocity_m_s must be > 0")
    if initial_mass_kg <= 0 or final_mass_kg <= 0:
        raise ValueError("initial_mass_kg and final_mass_kg must be > 0")
    if final_mass_kg > initial_mass_kg:
        raise ValueError(
            "final_mass_kg cannot exceed initial_mass_kg -- a rocket "
            "cannot gain mass by burning propellant "
            f"(got m0={initial_mass_kg}, m1={final_mass_kg})")
    return exhaust_velocity_m_s * math.log(initial_mass_kg / final_mass_kg)


def exhaust_velocity_from_isp(isp_s: float,
                              g0: float = STANDARD_GRAVITY_M_S2) -> float:
    """v_e = Isp * g0 -- effective exhaust velocity from specific impulse.

    FIRST_PRINCIPLES / DEDUCTION: Isp is defined as v_e / g0 with g0 the
    STANDARD (defined) gravitational acceleration, not local gravity --
    see module docstring for why Isp is quoted in seconds regardless of
    where the engine fires. g0 defaults to the internationally defined
    standard gravity (3rd CGPM, 1901); pass a different g0 only to
    reproduce a source that used a non-standard rounding, never to model
    "firing on the Moon" (that would be a physics error -- g0 here is a
    units constant, not the local field).
    """
    if isp_s <= 0:
        raise ValueError("isp_s must be > 0")
    if g0 <= 0:
        raise ValueError("g0 must be > 0")
    return isp_s * g0


def mass_flow_rate(thrust_n: float, exhaust_velocity_m_s: float) -> float:
    """mdot = F / v_e -- propellant mass flow rate from thrust.

    FIRST_PRINCIPLES: thrust is the momentum flux of the exhaust jet,
    F = v_e * mdot (Sutton & Biblarz, 9th ed., Eq. 3-2, momentum-thrust
    term; ideal nozzle, matched/negligible pressure-thrust).
    """
    if thrust_n <= 0:
        raise ValueError("thrust_n must be > 0")
    if exhaust_velocity_m_s <= 0:
        raise ValueError("exhaust_velocity_m_s must be > 0")
    return thrust_n / exhaust_velocity_m_s


def propellant_mass_for_delta_v(delta_v_m_s: float,
                                exhaust_velocity_m_s: float,
                                initial_mass_kg: float) -> float:
    """Propellant mass required for a target delta-v -- the inverse
    Tsiolkovsky relation.

    DEDUCTION: solve delta_v = v_e * ln(m0/m1) for m1, then propellant
    mass = m0 - m1:
      m1 = m0 * exp(-delta_v / v_e)
      m_propellant = m0 - m1 = m0 * (1 - exp(-delta_v / v_e))
    """
    if delta_v_m_s < 0:
        raise ValueError("delta_v_m_s must be >= 0")
    if exhaust_velocity_m_s <= 0:
        raise ValueError("exhaust_velocity_m_s must be > 0")
    if initial_mass_kg <= 0:
        raise ValueError("initial_mass_kg must be > 0")
    final_mass_kg = initial_mass_kg * math.exp(
        -delta_v_m_s / exhaust_velocity_m_s)
    return initial_mass_kg - final_mass_kg


# ═══════════════════════════════════════════════════════════════════════════
# § 2. Curated propulsion-system catalog -- every entry a cited Fact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PropulsionSystem:
    """One catalog entry -- provenance disclosed per-entry, same doctrine
    as Choice/Fact/Plug: every number here traces to a source.

    provenance is "fact" for every entry in PROPULSION_SYSTEMS below --
    each is a specific tested or flown engine with a published specific
    impulse. A category is allowed to add an invented (Choice) entry
    later, but none exists here: adding one would need a disclosed
    Choice object, not a bare number in this table (see
    dynamics.mechanisms.choice.Choice).
    """
    category: str
    name: str
    isp_s: float
    provenance: str
    source: str
    note: str = ""

    def __post_init__(self):
        if self.provenance not in ("fact", "choice"):
            raise ValueError(
                f"provenance must be 'fact' or 'choice', got "
                f"{self.provenance!r}")
        if self.isp_s <= 0:
            raise ValueError(f"{self.name}: isp_s must be > 0")

    @property
    def exhaust_velocity_m_s(self) -> float:
        return exhaust_velocity_from_isp(self.isp_s)

    def cite(self) -> str:
        tail = f" ({self.note})" if self.note else ""
        return f"{self.name} [{self.category}]: Isp={self.isp_s}s{tail} -- {self.source}"


# Every isp_s figure below is the ADVERSARIALLY-VERIFIED research data
# supplied for this table (sourced, cross-checked); nothing here is
# re-researched or invented. Citation wording refined for specificity but
# the underlying source and figure are unchanged from the verified input.
PROPULSION_SYSTEMS: dict[str, PropulsionSystem] = {
    "rs25": PropulsionSystem(
        category="chemical_bipropellant",
        name="RS-25 (Space Shuttle Main Engine / SLS core stage engine)",
        isp_s=452.3,
        provenance="fact",
        source="NASA fact sheet 'Space Launch System RS-25 Core Stage "
               "Engine' (nasa.gov/wp-content/uploads/2021/10/"
               "0080_rs-25_fact_sheet_12092020_3.pdf), corroborated by "
               "Aerojet Rocketdyne/L3Harris RS-25 propulsion system spec "
               "sheet",
        note="vacuum Isp, LH2/LOX bipropellant, flown on Space Shuttle "
             "and SLS Artemis I"),
    "srb": PropulsionSystem(
        category="chemical_solid",
        name="Space Shuttle Solid Rocket Booster (SRB)",
        isp_s=269.0,
        provenance="fact",
        source="NASA NTRS technical document 'Solid Rocket Booster (SRB)' "
               "(ntrs.nasa.gov/api/citations/20120001534/downloads/"
               "20120001534.pdf); cross-checked against Astronautix SRB "
               "entry (astronautix.com/s/srb.html)",
        note="vacuum Isp, PBAN solid propellant, flown on "
             "Space Shuttle"),
    "aj10_137": PropulsionSystem(
        category="chemical_bipropellant",
        name="Aerojet AJ10-137 (Apollo Service Propulsion System engine)",
        isp_s=314.5,
        provenance="fact",
        source="NASA Technical Reports Server (NTRS), 'Aerojet - AJ10-137 "
               "Apollo Service Module Engine' technical report "
               "(ntrs.nasa.gov/search.jsp?R=20100027319); figure also "
               "given in enginehistory.org 'Rocket Propulsion Evolution' "
               "section 9.31",
        note="vacuum Isp, N2O4/Aerozine-50 hypergolic bipropellant, flown "
             "on the Apollo Service Module"),
    "nstar": PropulsionSystem(
        category="electric_ion",
        name="NSTAR ion engine",
        isp_s=3100.0,
        provenance="fact",
        source="NASA/JPL Dawn mission technical documentation (NASA "
               "Technical Reports Server, ntrs.nasa.gov, 'Ion Propulsion "
               "System' status reports); also astronautix.com/n/nstar.html",
        note="xenon gridded ion thruster, flown on Deep Space 1 and Dawn"),
    "next_c": PropulsionSystem(
        category="electric_ion",
        name="NEXT-C ion engine (NASA Evolutionary Xenon Thruster "
             "- Commercial)",
        isp_s=4190.0,
        provenance="fact",
        source="NASA press materials on the DART mission technology "
               "demonstration payload; NASA Glenn Research Center NEXT-C "
               "program documentation (see also Universe Today coverage "
               "of NEXT-C on DART and NTRS conference papers, e.g. IEPC "
               "papers on NEXT-C status)",
        note="xenon gridded ion thruster, flown as a technology "
             "demonstration payload on DART"),
    "pps1350": PropulsionSystem(
        category="electric_hall_effect",
        name="PPS-1350 Hall-effect thruster",
        isp_s=1650.0,
        provenance="fact",
        source="ESA Science Portal (sci.esa.int, SMART-1 electric "
               "propulsion pages); Safran/Snecma PPS-1350 datasheet "
               "(safran-group.com); Wikipedia PPS-1350 article citing "
               "ESA/Safran specs",
        note="xenon Hall-effect thruster, flown on ESA SMART-1"),
    "nerva_nrx_a6": PropulsionSystem(
        category="nuclear_thermal",
        name="NERVA NRX-A6 (Rover/NERVA program, Nevada Test Site, "
             "Jackass Flats)",
        isp_s=869.0,
        provenance="fact",
        source="Finseth, J.L., 'Rover Nuclear Rocket Engine Program: "
               "Overview of Rover Engine Tests,' NASA Contractor Report "
               "NASA CR-184270 (1991), Table C-1 (Reactor Test Summary), "
               "NRX A6 row, ideal vacuum specific impulse column (869 s)",
        note="hydrogen propellant, solid-core nuclear thermal, "
             "ground-tested only (never flown)"),
    "nerva_pewee1": PropulsionSystem(
        category="nuclear_thermal",
        name="NERVA Pewee-1 (Rover program, Nevada Test Site, Jackass "
             "Flats)",
        isp_s=901.0,
        provenance="fact",
        source="Finseth, J.L., 'Rover Nuclear Rocket Engine Program: "
               "Overview of Rover Engine Tests,' NASA Contractor Report "
               "NASA CR-184270 (1991), pp. 78-83",
        note="hydrogen propellant, solid-core nuclear thermal, highest "
             "Isp achieved in the Rover/NERVA program; ground-tested only"),
    "nerva_xe_prime": PropulsionSystem(
        category="nuclear_thermal",
        name="NERVA XE-Prime (flight-configuration prototype "
             "'Experimental Engine')",
        isp_s=710.0,
        provenance="fact",
        source="Finseth, J.L., NASA Contractor Report NASA CR-184270 "
               "(1991), Table C-2 (XE PRIME entry); program history "
               "corroborated in Westinghouse Astronuclear Laboratory "
               "retrospectives (e.g., 'Westinghouse Engineer,' 1974)",
        note="hydrogen propellant, solid-core nuclear thermal, tested in "
             "flight-like configuration; program cancelled 1973, never "
             "flew"),
    "vasimr_vx200ss": PropulsionSystem(
        category="electromagnetic_plasma",
        name="VASIMR (Variable Specific Impulse Magnetoplasma Rocket) "
             "-- VX-200SS 200 kWe design/test point, argon propellant",
        isp_s=5000.0,
        provenance="fact",
        source="Ad Astra Rocket Company / NASA Advanced Space Propulsion "
               "Laboratory ground-test program (F. Chang Diaz et al., "
               "VX-200SS 200 kWe full-power test campaign, published "
               "thruster-efficiency and Isp results, ~2013-2015)",
        note="argon propellant, electromagnetic plasma thruster, "
             "ground-tested only (never flown) -- the measured VX-200SS "
             "test-campaign figure, not a higher speculative number "
             "sometimes quoted for un-flown conceptual VASIMR variants"),
}


# ═══════════════════════════════════════════════════════════════════════════
# § 3. Adapters into the closed AppliedForce taxonomy (Golden Rule 7)
# ═══════════════════════════════════════════════════════════════════════════

def _unit_vector(direction) -> tuple[float, float, float]:
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 0:
        raise ValueError("direction_unit_vector must be nonzero")
    return (x / n, y / n, z / n)


def impulse_burn_applied_force(target: str, mass_before_kg: float,
                               delta_v_m_s: float, direction_unit_vector,
                               exhaust_velocity_m_s: float, *,
                               t0: float = 0.0,
                               label: str | None = None) -> AppliedForce:
    """An instantaneous rocket burn as a member of the closed
    artificial-force taxonomy (dynamics.mechanisms.applied.AppliedForce).

    Idiom: J = m * delta_v (same J=m*v0 pattern as push_off.py's
    ``_push_af``) -- valid for an IMPULSE by definition (duration == 0):
    the burn is modeled as fast enough that the mass lost to propellant
    during the burn itself is negligible for the momentum bookkeeping
    (mass_before_kg stands in for the whole instant). The propellant
    consumed to PRODUCE that delta-v is a separate question -- answered
    by the inverse Tsiolkovsky relation (propellant_mass_for_delta_v) and
    disclosed in ``variables`` alongside the wrench: propulsion is never
    a free force.

    Parameters
    ----------
    target : body label the burn acts on
    mass_before_kg : the body's mass immediately before the burn, kg
    delta_v_m_s : the velocity change delivered, m/s (may be negative --
        a retro-burn; the impulse vector then points opposite direction)
    direction_unit_vector : [x, y, z], normalized internally
    exhaust_velocity_m_s : v_e of the engine doing the burn, m/s -- from
        exhaust_velocity_from_isp(isp_s) or PROPULSION_SYSTEMS[...]
    """
    if mass_before_kg <= 0:
        raise ValueError("mass_before_kg must be > 0")
    dx, dy, dz = _unit_vector(direction_unit_vector)
    propellant_kg = propellant_mass_for_delta_v(
        abs(delta_v_m_s), exhaust_velocity_m_s, mass_before_kg)
    j = mass_before_kg * delta_v_m_s
    lbl = label or f"rocket burn (impulse) on the {target}"
    return AppliedForce(
        "impulse", target, label=lbl,
        vector=[j * dx, j * dy, j * dz], t0=t0,
        variables={
            "delta_v_m_s": float(delta_v_m_s),
            "exhaust_velocity_m_s": float(exhaust_velocity_m_s),
            "mass_before_kg": float(mass_before_kg),
            "propellant_mass_consumed_kg": round(propellant_kg, 6),
        })


def sustained_burn_applied_force(target: str, thrust_n: float,
                                 duration_s: float, direction_unit_vector,
                                 exhaust_velocity_m_s: float, *,
                                 t0: float = 0.0,
                                 label: str | None = None) -> AppliedForce:
    """A sustained rocket burn (steady thrust over a time window) as a
    member of the closed artificial-force taxonomy
    (dynamics.mechanisms.applied.AppliedForce).

    vector = thrust_n * direction, duration = duration_s -- a plain
    FORCE wrench. Propellant consumed over the window is disclosed via
    mass_flow_rate(thrust_n, v_e) * duration_s: propulsion is never a
    free force.

    Parameters
    ----------
    target : body label the thrust acts on
    thrust_n : steady thrust magnitude, N (> 0)
    duration_s : how long the thrust is sustained, s (> 0)
    direction_unit_vector : [x, y, z], normalized internally
    exhaust_velocity_m_s : v_e of the engine doing the burn, m/s
    """
    if thrust_n <= 0:
        raise ValueError("thrust_n must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    dx, dy, dz = _unit_vector(direction_unit_vector)
    mdot = mass_flow_rate(thrust_n, exhaust_velocity_m_s)
    propellant_kg = mdot * duration_s
    lbl = label or f"rocket burn (sustained thrust) on the {target}"
    return AppliedForce(
        "force", target, label=lbl,
        vector=[thrust_n * dx, thrust_n * dy, thrust_n * dz],
        t0=t0, duration=duration_s,
        variables={
            "thrust_n": float(thrust_n),
            "exhaust_velocity_m_s": float(exhaust_velocity_m_s),
            "mass_flow_rate_kg_s": round(mdot, 6),
            "propellant_mass_consumed_kg": round(propellant_kg, 6),
        })


__all__ = [
    "STANDARD_GRAVITY_M_S2",
    "tsiolkovsky_delta_v",
    "exhaust_velocity_from_isp",
    "mass_flow_rate",
    "propellant_mass_for_delta_v",
    "PropulsionSystem",
    "PROPULSION_SYSTEMS",
    "impulse_burn_applied_force",
    "sustained_burn_applied_force",
]
