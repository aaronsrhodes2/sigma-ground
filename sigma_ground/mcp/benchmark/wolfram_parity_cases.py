"""Frozen Wolfram-Alpha parity cases for Mentat.

Each case pairs a Mentat computation (`compute`, returns a float in `units`)
with the value Wolfram Alpha returned for `wq`, captured via the Wolfram MCP
on 2026-06-05, plus a relative tolerance. `test_wolfram_parity.py` calls each
`compute()` and asserts it matches the frozen `oracle` within `tol` -- a
deterministic drift guard with NO network dependency (the Wolfram MCP is
absent in CI).

oracle_kind:
  nl       -- Wolfram's natural-language engine returned the whole answer
  atomic   -- Wolfram returned the underlying constant / arithmetic
              (textbook formula + Wolfram-evaluated constant or math), used
              where Wolfram's NL had no entity for the full scenario
  textbook -- Wolfram had no entity at all; oracle is a standard textbook value

This is a STRATIFIED CONFIDENCE SAMPLE across domains, not a proof of all of
Mentat's ~1000 covered functions. It is biased toward quantities with a single
unambiguous numeric answer. See misc/WOLFRAM_PARITY.md.
"""
from __future__ import annotations

import math

# ── physical constants used to set up cases (CODATA-ish; only for inputs) ──
AMU = 1.66053906660e-27
EV = 1.602176634e-19
C = 2.99792458e8
M_E = 9.1093837015e-31

CAPTURED = "2026-06-05"


def _v(x):
    """Pull a number (or dict) out of a ToolResult / dict / float."""
    if hasattr(x, "value"):
        return x.value
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x


# ── Mentat module imports (consume only; never mutate) ──
from sigma_ground.field.interface import liquid_water as W
from sigma_ground.field.interface import fluid_surface as FS
from sigma_ground.field import relativity as R
from sigma_ground.field import scale as SC
from sigma_ground.field.interface import surface as S
from sigma_ground.mcp.tools import thermodynamics as TH
from sigma_ground.field.interface import thermoelectric as TE
from sigma_ground.mcp.tools import atomic as AT
from sigma_ground.field.interface import quantum as Q
from sigma_ground.field import electrodynamics as ED
from sigma_ground.field.interface import electronics as EL
from sigma_ground.mcp.tools import optics as OP
from sigma_ground.mcp.tools import nuclear as NU
from sigma_ground.field.interface import statistical as ST
from sigma_ground.field.interface import gas as GAS
from sigma_ground.field.interface import acid_base as AB
from sigma_ground.mcp.tools import kinematics as KIN
from sigma_ground.mcp.tools import orbital as ORB

_SIG = W.water_surface_tension(293.15)
_RHO = W.water_density(293.15)
_KE_E = 0.5 * M_E * (1e6) ** 2


def _C(id_, domain, compute, units, oracle, tol, kind, wq):
    return dict(id=id_, domain=domain, compute=compute, units=units,
                oracle=oracle, tol=tol, oracle_kind=kind, wq=wq)


CASES = [
    # ── original 14 (9 domains) ──
    _C("water_density_20C", "fluids", lambda: W.water_density(293.15),
       "kg/m^3", 998.2, 0.01, "nl", "density of water at 20 C"),
    _C("water_surface_tension_20C", "fluids", lambda: W.water_surface_tension(293.15),
       "N/m", 0.07274, 0.02, "nl", "surface tension of water at 20 C"),
    _C("water_viscosity_20C", "fluids", lambda: W.water_viscosity(293.15),
       "Pa s", 0.001, 0.03, "nl", "dynamic viscosity of water at 20 C (Wolfram rounds to 1.00 cP)"),
    _C("capillary_gravity_min_speed", "waves", lambda: FS.capillary_gravity_min_phase_speed(_SIG, _RHO)[0],
       "m/s", 0.23, 0.05, "textbook", "minimum phase speed of water surface waves (Lighthill)"),
    _C("lorentz_factor_0.9c", "relativity", lambda: R.lorentz_factor(0.9 * C),
       "dimensionless", 2.294, 0.01, "nl", "Lorentz factor at 0.9 c"),
    _C("electron_rest_energy_MeV", "particle", lambda: R.rest_energy(M_E) / EV / 1e6,
       "MeV", 0.5109989461, 0.01, "nl", "electron rest energy in MeV"),
    _C("schwarzschild_radius_sun", "gr", lambda: SC.schwarzschild_radius(1.98892e30),
       "m", 2953.25, 0.01, "nl", "Schwarzschild radius of the Sun"),
    _C("copper_density", "materials", lambda: S.MATERIALS["copper"]["density_kg_m3"],
       "kg/m^3", 8960.0, 0.02, "nl", "density of copper"),
    _C("wien_peak_5778K", "radiation", lambda: _v(TH.blackbody_peak_wavelength(5778.0)) * 1e9,
       "nm", 501.5, 0.01, "nl", "Wien's law peak wavelength for blackbody at 5778 K"),
    _C("speed_of_sound_air_20C", "acoustics", lambda: _v(TH.speed_of_sound_in_ideal_gas(293.15, 1.4, 0.02896)),
       "m/s", 343.093, 0.02, "nl", "speed of sound in dry air at 20 C"),
    _C("carnot_600_300", "thermo", lambda: TE.carnot_efficiency(600.0, 300.0),
       "dimensionless", 0.5, 0.01, "nl", "Carnot efficiency hot 600 K cold 300 K"),
    _C("hydrogen_ionization_eV", "atomic", lambda: abs(_v(AT.hydrogen_like_energy_level(1, 1))),
       "eV", 13.605693123, 0.01, "nl", "Rydberg energy in eV"),
    _C("photon_energy_500nm_eV", "optics", lambda: _v(AT.photon_energy_from_wavelength(500e-9)) / EV,
       "eV", 2.48, 0.01, "nl", "photon energy of 500 nm light in eV"),
    _C("de_broglie_electron_1e6", "quantum", lambda: _v(Q.de_broglie_wavelength(M_E, _KE_E)) * 1e9,
       "nm", 0.7273895159, 0.01, "nl", "de Broglie wavelength of electron at 1e6 m/s"),

    # ── new 20 (EM, circuits, optics, nuclear, gases, chemistry, astro, mechanics) ──
    _C("coulomb_force_1C_1m", "em", lambda: ED.coulomb_force(1.0, 1.0, 1.0),
       "N", 8.98755179e9, 0.01, "atomic", "Coulomb's constant (= F for q1=q2=1C, r=1m)"),
    _C("parallel_plate_C", "circuits", lambda: EL.parallel_plate_capacitance(1.0, 1e-3, 1.0),
       "F", 8.85418782e-9, 0.01, "atomic", "eps0 * 1 m^2 / 1 mm in farads"),
    _C("capacitor_energy", "circuits", lambda: EL.energy_stored(1e-6, 100.0),
       "J", 0.005, 0.01, "nl", "energy stored in a 1 uF capacitor at 100 V"),
    _C("electron_cyclotron_freq_1T", "em", lambda: ED.cyclotron_frequency(EV, M_E, 1.0) / (2 * math.pi),
       "Hz", 2.79924898e10, 0.02, "atomic", "e * 1 T / (2 pi m_e) in Hz"),
    _C("snell_air_water_30deg", "optics", lambda: _v(OP.snells_law_refraction_angle(1.0, 1.33, 30.0)),
       "deg", 22.08, 0.02, "nl", "Snell's law refraction angle n1=1 n2=1.33 incidence 30 deg"),
    _C("thin_lens_focal", "optics", lambda: _v(OP.thin_lens_focal_length(0.30, 0.15)),
       "m", 0.10, 0.02, "nl", "thin lens object 0.30 m image 0.15 m focal length"),
    _C("grating_600lpmm_500nm", "optics", lambda: _v(OP.diffraction_grating_angle(500e-9, 1.0 / 600e3, 1)),
       "deg", 17.46, 0.02, "atomic", "arcsin(0.3) deg (d sin th = m lambda, 600 lpmm, 500 nm)"),
    _C("fe56_binding_per_nucleon", "nuclear", lambda: _v(NU.nuclear_binding_energy(26, 30))["binding_per_nucleon_MeV"],
       "MeV", 8.79036, 0.03, "nl", "binding energy per nucleon of iron-56 (Mentat SEMF vs measured)"),
    _C("rms_speed_N2_300K", "gases", lambda: ST.rms_speed(28 * AMU, 300.0),
       "m/s", 516.83, 0.02, "nl", "rms speed of a nitrogen molecule at 300 K"),
    _C("ideal_gas_pressure", "gases", lambda: _v(TH.ideal_gas_pressure(1.0, 273.15, 0.0224)),
       "Pa", 101400.0, 0.02, "nl", "pressure of 1 mol ideal gas at 273.15 K in 0.0224 m^3"),
    _C("ideal_gas_molar_volume_STP", "gases", lambda: _v(TH.ideal_gas_volume(1.0, 273.15, 100000.0)),
       "m^3", 0.02271, 0.02, "atomic", "1 mol * R * 273.15 K / 100 kPa (IUPAC STP)"),
    _C("most_probable_speed_O2_300K", "gases", lambda: _v(TH.maxwell_boltzmann_most_probable_speed(300.0, 32 * AMU)),
       "m/s", 394.837, 0.02, "atomic", "sqrt(2 k 300 / (32 u)) m/s"),
    _C("molar_mass_water", "chemistry", lambda: GAS.MOLECULES["H2O"]["molecular_mass_amu"],
       "g/mol", 18.015, 0.01, "nl", "molar mass of water"),
    _C("molar_mass_co2", "chemistry", lambda: GAS.MOLECULES["CO2"]["molecular_mass_amu"],
       "g/mol", 44.009, 0.01, "nl", "molar mass of carbon dioxide"),
    _C("pH_0.01M_HCl", "chemistry", lambda: AB.pH_strong_acid(0.01),
       "dimensionless", 2.00, 0.02, "nl", "pH of 0.01 molar HCl"),
    _C("escape_velocity_earth", "astro", lambda: _v(KIN.escape_velocity_classical(5.972e24, 6.371e6)),
       "m/s", 11180.0, 0.01, "nl", "escape velocity of Earth"),
    _C("projectile_range_20_45", "mechanics", lambda: _v(KIN.projectile_range(20.0, 45.0)),
       "m", 40.79, 0.02, "nl", "range of a projectile launched at 20 m/s at 45 deg"),
    _C("gravitational_force_1kg_1m", "astro", lambda: _v(ORB.gravitational_force(1.0, 1.0, 1.0)),
       "N", 6.674e-11, 0.01, "atomic", "Newtonian gravitational constant (= F for 1kg,1kg,1m)"),
    _C("orbital_period_1AU_sun", "astro", lambda: _v(ORB.orbital_period(semimajor_axis_au=1.0, central_body="sun")),
       "s", 3.155814954e7, 0.02, "atomic", "sidereal year in seconds (a=1 AU, Kepler III)"),
    _C("circular_orbit_v_400km", "astro", lambda: _v(KIN.circular_orbit_velocity(5.972e24, 6.771e6)),
       "m/s", 7673.0, 0.02, "nl", "circular orbital speed at 400 km altitude above Earth"),
]


def evaluate():
    """Return list of result dicts: mentat, oracle, rel_err, passed, ..."""
    out = []
    for c in CASES:
        try:
            mentat = float(c["compute"]())
            rel = abs(mentat - c["oracle"]) / abs(c["oracle"]) if c["oracle"] else abs(mentat)
            passed = rel <= c["tol"]
            err = None
        except Exception as e:  # pragma: no cover - surfaced as a failure
            mentat, rel, passed, err = None, None, False, f"{type(e).__name__}: {e}"
        out.append({**{k: c[k] for k in ("id", "domain", "units", "oracle", "tol", "oracle_kind", "wq")},
                    "mentat": mentat, "rel_err": rel, "passed": passed, "error": err})
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rows = evaluate()
    print(f"{'id':30} {'domain':10} {'mentat':>14} {'oracle':>14} {'rel%':>8} {'kind':8} ok")
    print("-" * 100)
    for r in rows:
        m = f"{r['mentat']:.6g}" if r["mentat"] is not None else (r["error"] or "ERR")
        re_ = f"{100 * r['rel_err']:.3f}" if r["rel_err"] is not None else "-"
        print(f"{r['id']:30} {r['domain']:10} {m:>14} {r['oracle']:>14.6g} {re_:>8} {r['oracle_kind']:8} {'Y' if r['passed'] else 'N'}")
    n = len(rows); ok = sum(r["passed"] for r in rows)
    print(f"\nPASS {ok}/{n} ({100*ok/n:.1f}%) within tolerance | captured {CAPTURED}")
