"""Hand-written recipe families for router v2: multi-step chains and the weak single-call families.

router_dataset.py builds one-call examples from tool keywords, so its questions are
stiff and it cannot express "find the orbital speed, then the kinetic energy". A
recipe is the opposite: a few natural question templates written by hand, a plan
(one to three calls, "$n" referring to an earlier call's value), and the ranges its
quantities are sampled from. Every plan is run against the real MCP server; the
example is kept only if every call returns a value inside the recipe's bounds, the
benchmark guard passes on every call, and no deterministic classifier would have
sent the question elsewhere.

Chains follow tool inputs and outputs and the physics (an orbital speed feeds a
kinetic energy), not benchmark items. The recipes were written from the tool
descriptions; a few of their tool pairs coincide with the way some benchmark questions
are solved (satellite kinetic energy, TNT equivalent of a fission, mass from TNT), so
`PAIRS_SHARED_WITH_BENCHMARK` lists them and the manifest reports their share.

    python -m sigma_ground.mcp.benchmark.router_recipes --per-recipe 70 --out local-cache/router_recipes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigma_ground.field.constants import M_SUN_KG
from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_plan as RP

U = D._u

# ── quantities ───────────────────────────────────────────────────────────

KG_OBJ = U(1, 5e4, ("kg", 1), ("tonnes", 1e3))
KG_CAR = U(300, 5e4, ("kg", 1), ("tonnes", 1e3))
KG_SMALL = U(0.05, 500, ("kg", 1), ("g", 1e-3))
KG_SAT = U(50, 2e4, ("kg", 1), ("tonnes", 1e3))
KG_PLANET = U(1e22, 2e27, ("kg", 1))
KG_STAR_OR_PLANET = U(1e22, 2e30, ("kg", 1), ("solar masses", M_SUN_KG))
KG_PARTICLE = U(1e-31, 1e-25, ("kg", 1))
KG_FUEL = U(1e-3, 50, ("kg", 1), ("g", 1e-3))
KG_CONVERTED = U(1e-6, 10, ("kg", 1), ("g", 1e-3), ("mg", 1e-6))
M_ORBIT = U(7e6, 5e12, ("m", 1), ("km", 1e3), ("million km", 1e9))
M_PLANET_R = U(1e6, 2e8, ("m", 1), ("km", 1e3))
M_HEIGHT = U(1, 3e3, ("m", 1), ("km", 1e3))
M_FALL = U(2, 500, ("m", 1))
M_DIST = U(0.02, 5, ("m", 1), ("cm", 1e-2))
M_FOCAL = U(0.01, 1, ("m", 1), ("cm", 1e-2), ("mm", 1e-3))
M_WAVELENGTH = U(1e-8, 1e-3, ("nm", 1e-9), ("µm", 1e-6))
MS_SPEED = U(1, 3e4, ("m/s", 1), ("km/s", 1e3), ("km/h", 1 / 3.6))
MS_PARTICLE = U(1e4, 1e7, ("m/s", 1), ("km/s", 1e3))
TEMP_K = U(20, 1e4, ("K", 1))
TEMP_STAR = U(2500, 3e4, ("K", 1))
WATT = U(1, 5e3, ("W", 1), ("kW", 1e3))
JOULE = U(1, 1e7, ("J", 1), ("kJ", 1e3), ("MJ", 1e6))
VOLT = U(1, 400, ("V", 1))
OHM = U(1, 1e5, ("Ω", 1), ("kΩ", 1e3))
AMP = U(1e-3, 50, ("A", 1), ("mA", 1e-3))
PERCENT = U(1, 95, ("percent", 1), log=False)
SOLAR = U(0.3 * M_SUN_KG, 60 * M_SUN_KG, ("solar masses", M_SUN_KG))
LEVEL = U(2, 8, ("", 1), log=False, integer=True)


@dataclass(frozen=True)
class Q:
    """A numeric slot: sampled from `unit`, rendered the way a person writes it."""
    unit: D.Unit
    integer: bool = False


@dataclass(frozen=True)
class C:
    """A choice slot: (text in the question, value in the plan) pairs."""
    options: tuple


def S(slot: str):    # plan argument = the slot's value
    return ("s", slot)


def R(n: int, field: str | None = None):    # plan argument = "$n", or "$n.field" of a lookup result
    return ("r", n) if field is None else ("rf", (n, field))


def K(slot: str, index: int):    # plan argument = one element of a choice slot's value (a tuple)
    return ("k", (slot, index))


def L(value):        # plan argument = a literal
    return ("l", value)


@dataclass
class Recipe:
    id: str
    family: str                       # chain | percent | convert | atomic | astro
    slots: dict
    templates: list
    steps: list                       # [(tool, {param: S/R/L})]
    bounds: dict = field(default_factory=dict)      # step number -> (lo, hi) on its value
    per: int = 0                      # 0: use the command-line default
    distractors: tuple = ()           # slots the question states but the plan does not use

    def used_slots(self) -> set:
        used = {a[1] for _, args in self.steps for a in args.values() if a[0] == "s"}
        return used | {a[1][0] for _, args in self.steps for a in args.values() if a[0] == "k"}


RECIPES: list = []


def add(*a, **k) -> Recipe:
    r = Recipe(*a, **k)
    RECIPES.append(r)
    return r


def chain(id, slots, templates, steps, bounds=None, per=0):
    return add(id, "chain", slots, templates, steps, bounds or {}, per)


# ── chains: two or three calls, "$n" carries a value forward ─────────────

ORBIT = [("circular_orbit_velocity", {"central_mass_kg": S("M"), "radius_m": S("r")})]
ORBIT_SLOTS = {"M": Q(KG_STAR_OR_PLANET), "r": Q(M_ORBIT)}
ORBIT_BOUNDS = {1: (50, 3e7)}

chain("orbit_kinetic_energy", {**ORBIT_SLOTS, "m": Q(KG_SAT)}, [
    "A {m} satellite circles a body of mass {M} at an orbital radius of {r}. What is its kinetic energy?",
    "How much kinetic energy does a {m} spacecraft have while orbiting a {M} planet at a radius of {r}?",
    "Satellite mass {m}, circular orbit of radius {r} around a mass of {M}: find its kinetic energy.",
    "I have a {m} probe in a circular orbit {r} from the centre of a {M} world. How much energy is it carrying in its motion?",
    "What's the orbital kinetic energy of a {m} object going around a {M} body at {r} from its centre?",
    "A {m} moon moves in a circle of radius {r} about a {M} planet. Kinetic energy?",
], ORBIT + [("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": R(1)})], ORBIT_BOUNDS)

chain("orbit_momentum", {**ORBIT_SLOTS, "m": Q(KG_SAT)}, [
    "What is the orbital momentum of a {m} satellite circling a {M} body at a radius of {r}?",
    "A {m} spacecraft is in a circular orbit of radius {r} around a mass of {M}. How much momentum does it have?",
    "Momentum of a {m} moon going round a {M} planet at {r} from the centre?",
    "Find the linear momentum of a {m} probe orbiting at {r} about a {M} object.",
    "A {m} body orbits a {M} star on a circle of radius {r}; what is its momentum?",
], ORBIT + [("momentum", {"mass_kg": S("m"), "velocity_m_s": R(1)})], ORBIT_BOUNDS)

FALL = [("free_fall_velocity", {"height_m": S("h")})]
chain("drop_impact_energy", {"h": Q(M_FALL), "m": Q(KG_OBJ)}, [
    "A {m} rock is dropped from {h}. How much kinetic energy does it have when it hits the ground?",
    "What kinetic energy does a {m} object carry on impact after falling {h}?",
    "A {m} crate slides off a ledge and falls {h} to the floor. How much energy does it arrive with?",
    "Dropping a {m} weight from a height of {h}: what's its kinetic energy just before landing?",
    "A {m} stone falls freely for {h}. Kinetic energy at the bottom?",
], FALL + [("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": R(1)})], {1: (1, 3e3)})

chain("drop_impact_momentum", {"h": Q(M_FALL), "m": Q(KG_OBJ)}, [
    "A {m} rock is dropped from {h}. What is its momentum just before it hits the ground?",
    "How much momentum does a {m} object have after falling {h} from rest?",
    "A {m} hammer falls {h} onto a nail. What momentum does it deliver?",
    "Momentum on impact of a {m} ball dropped from {h}?",
], FALL + [("momentum", {"mass_kg": S("m"), "velocity_m_s": R(1)})], {1: (1, 3e3)})

chain("escape_energy", {"M": Q(KG_PLANET), "R": Q(M_PLANET_R), "m": Q(KG_SAT)}, [
    "What is the minimum kinetic energy a {m} probe needs to escape a planet of mass {M} and radius {R}?",
    "A {m} rocket must leave a world of mass {M} and radius {R} for good. How much kinetic energy does it need at the surface?",
    "How much kinetic energy must a {m} spacecraft have to get away from a body of {M} with a radius of {R}?",
    "Escape energy for a {m} lander on a {M} planet whose radius is {R}?",
    "Planet: mass {M}, radius {R}. What kinetic energy lets a {m} payload escape it?",
], [("escape_velocity", {"mass_kg": S("M"), "radius_m": S("R")}),
    ("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": R(1)})], {1: (100, 3e7)})

chain("thermal_de_broglie", {"T": Q(TEMP_K), "mm": Q(KG_PARTICLE)}, [
    "What is the de Broglie wavelength of a molecule of mass {mm} moving at the most probable speed in a gas at {T}?",
    "A gas at {T} has molecules of mass {mm}. Find the thermal de Broglie wavelength at the most probable speed.",
    "How long is the matter wave of a typical {mm} molecule in a {T} gas?",
    "At {T}, what wavelength goes with a {mm} molecule travelling at the most probable speed?",
    "Thermal de Broglie wavelength for molecules of {mm} at {T}?",
], [("maxwell_boltzmann_most_probable_speed", {"temperature_k": S("T"), "molecular_mass_kg": S("mm")}),
    ("de_broglie_wavelength", {"mass_kg": S("mm"), "velocity_m_s": R(1)})], {1: (1, 5e4)})

chain("mass_to_tnt", {"m": Q(KG_CONVERTED)}, [
    "If {m} of matter were completely converted to energy, how many tons of TNT would that equal?",
    "Annihilate {m} of mass entirely. What is the TNT equivalent in tons?",
    "How much TNT, in tons, has the same energy as the rest energy of {m}?",
    "Convert {m} of mass completely into energy: express the result as tons of TNT.",
    "The full mass-energy of {m}, in tons of TNT?",
], [("mass_to_energy", {"mass_kg": S("m")}), ("joules_to_TNT", {"energy_joules": R(1)})])

chain("mass_to_tnt_kilotons", {"m": Q(KG_CONVERTED)}, [
    "Completely converting {m} of matter into energy releases how many kilotons of TNT?",
    "What is the yield, in kilotons, of turning {m} of mass entirely into energy?",
    "Total conversion of {m} of mass: how many kilotons of TNT is that?",
    "Express the rest-mass energy of {m} in kilotons of TNT.",
], [("mass_to_energy", {"mass_kg": S("m")}),
    ("joules_to_TNT", {"energy_joules": R(1), "unit": L("kt")})])

chain("mass_to_ev", {"m": Q(KG_PARTICLE)}, [
    "What is the rest energy, in electronvolts, of a particle of mass {m}?",
    "A particle weighs {m}. How many electronvolts is its rest-mass energy?",
    "Express the mass-energy of {m} in eV.",
    "How much energy, in eV, is locked up in a {m} particle at rest?",
], [("mass_to_energy", {"mass_kg": S("m")}), ("joules_to_eV", {"energy_joules": R(1)})])

chain("kinetic_energy_tnt", {"m": Q(KG_CAR), "v": Q(MS_SPEED)}, [
    "A {m} vehicle is moving at {v}. How many tons of TNT equal its kinetic energy?",
    "Express the kinetic energy of a {m} object travelling at {v} as a TNT equivalent in tons.",
    "A {m} asteroid fragment hits at {v}. What is the impact energy in tons of TNT?",
    "What mass of TNT, in tons, would release the same energy as a {m} body moving at {v}?",
], [("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": S("v")}),
    ("joules_to_TNT", {"energy_joules": R(1)})])

chain("kinetic_energy_ev", {"m": Q(KG_PARTICLE), "v": Q(MS_PARTICLE)}, [
    "An electron-sized particle of mass {m} moves at {v}. What is its kinetic energy in electronvolts?",
    "Kinetic energy in eV of a {m} particle travelling at {v}?",
    "A particle of {m} is going {v}. Give its kinetic energy in electronvolts.",
    "How many eV of kinetic energy does a {m} particle have at {v}?",
], [("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": S("v")}),
    ("joules_to_eV", {"energy_joules": R(1)})])

chain("fission_tnt", {"m": Q(KG_FUEL)}, [
    "Fission {m} of uranium-235 completely. How many tons of TNT is the energy equivalent to?",
    "How much TNT, in tons, matches the energy from fully splitting {m} of U-235?",
    "A bomb fissions {m} of uranium-235. Give its yield as tons of TNT.",
    "Energy from the complete fission of {m} of U-235, expressed in tons of TNT?",
], [("fission_energy_release", {"mass_kg": S("m"), "isotope": L("U235")}),
    ("joules_to_TNT", {"energy_joules": R(1)})])

chain("fission_tnt_megatons", {"m": Q(KG_FUEL)}, [
    "A device completely fissions {m} of plutonium-239. What is the yield in megatons of TNT?",
    "Fully splitting {m} of Pu-239 releases how many megatons of TNT?",
    "Express the energy of fissioning {m} of plutonium-239 in megatons.",
], [("fission_energy_release", {"mass_kg": S("m"), "isotope": L("Pu239")}),
    ("joules_to_TNT", {"energy_joules": R(1), "unit": L("MT")})])

TNT_UNITS = C((("kiloton", "kt"), ("megaton", "MT"), ("ton", "ton")))
chain("tnt_to_mass", {"t": Q(U(0.1, 900, ("", 1), log=True)), "u": TNT_UNITS}, [
    "How much mass is converted to energy in a {t}-{u} blast?",
    "A {t} {u} explosion converts how much mass into energy?",
    "What mass disappears in a {t}-{u} nuclear explosion?",
    "Find the mass equivalent of the energy released by {t} {u}s of TNT.",
], [("tnt_to_joules", {"tnt_amount": S("t"), "unit": S("u")}),
    ("energy_to_mass", {"energy_j": R(1)})])

chain("hydrogen_photon_energy", {"ni": Q(LEVEL, True), "nf": Q(U(1, 4, ("", 1), log=False, integer=True), True)}, [
    "What is the energy of the photon emitted when a hydrogen atom drops from level n = {ni} to n = {nf}?",
    "Hydrogen falls from the n = {ni} state to n = {nf}. How much energy does the emitted photon carry?",
    "Photon energy for the hydrogen transition {ni} → {nf}?",
    "A hydrogen electron jumps down from n = {ni} to n = {nf}; find the energy of the light given off.",
], [("rydberg_hydrogen_wavelength", {"n_initial": S("ni"), "n_final": S("nf")}),
    ("photon_energy_from_wavelength", {"wavelength_m": R(1)})], {1: (1e-8, 1e-4)})

chain("hydrogen_photon_ev", {"ni": Q(LEVEL, True), "nf": Q(U(1, 4, ("", 1), log=False, integer=True), True)}, [
    "How many electronvolts does the photon carry when hydrogen drops from n = {ni} to n = {nf}?",
    "Give the energy, in eV, of the light emitted in the hydrogen transition from level {ni} to level {nf}.",
    "A hydrogen atom goes from n = {ni} to n = {nf}. What is the photon's energy in electronvolts?",
], [("rydberg_hydrogen_wavelength", {"n_initial": S("ni"), "n_final": S("nf")}),
    ("photon_energy_from_wavelength", {"wavelength_m": R(1)}),
    ("joules_to_eV", {"energy_joules": R(2)})], {1: (1e-8, 1e-4)})

chain("blackbody_peak_frequency", {"T": Q(TEMP_STAR)}, [
    "At what frequency does a blackbody at {T} radiate most strongly?",
    "A star's surface is at {T}. What is the frequency of its peak emission?",
    "Find the peak emission frequency of a perfect radiator at {T}.",
    "What frequency corresponds to the peak of the thermal spectrum at {T}?",
], [("blackbody_peak_wavelength", {"temperature_k": S("T")}),
    ("em_wave_frequency", {"wavelength_m": R(1)})])

chain("blackbody_peak_photon", {"T": Q(TEMP_STAR)}, [
    "What is the energy of a photon at the peak wavelength of a blackbody at {T}?",
    "A glowing object sits at {T}. How much energy does a photon at its peak emission carry?",
    "Photon energy at the Wien peak for {T}?",
    "Find the energy of a typical photon from a thermal source at {T}, taken at the spectrum's peak.",
], [("blackbody_peak_wavelength", {"temperature_k": S("T")}),
    ("photon_energy_from_wavelength", {"wavelength_m": R(1)})])

chain("thermal_energy_ev", {"T": Q(TEMP_K)}, [
    "What is the average thermal energy of a molecule at {T}, in electronvolts?",
    "At {T}, how many eV of thermal energy does each molecule carry?",
    "Mean thermal energy per molecule at {T}, expressed in eV?",
    "Express the thermal energy of one molecule at {T} in electronvolts.",
], [("thermal_energy_per_molecule", {"temperature_k": S("T")}),
    ("joules_to_eV", {"energy_joules": R(1)})])

chain("photon_ev", {"w": Q(M_WAVELENGTH)}, [
    "How many electronvolts does a photon of wavelength {w} carry?",
    "Light with a wavelength of {w}: what is each photon's energy in eV?",
    "Express the photon energy for {w} radiation in electronvolts.",
    "A photon has a wavelength of {w}. Give its energy in eV.",
], [("photon_energy_from_wavelength", {"wavelength_m": S("w")}),
    ("joules_to_eV", {"energy_joules": R(1)})])

chain("lens_image_magnification", {"d": Q(M_DIST), "f": Q(M_FOCAL)}, [
    "An object sits {d} in front of a lens of focal length {f}. How much is the image magnified?",
    "A lens has a focal length of {f}. With the object {d} away, what is the magnification?",
    "Focal length {f}, object distance {d}: what magnification does the lens give?",
    "How large is the image, relative to the object, for a {f} lens with the object {d} away?",
], [("thin_lens_image_distance", {"object_distance_m": S("d"), "focal_length_m": S("f")}),
    ("lens_magnification", {"object_distance_m": S("d"), "image_distance_m": R(1)})], {1: (1e-4, 1e3)})

chain("ohm_current_power", {"V": Q(VOLT), "R": Q(OHM)}, [
    "A {V} supply is connected across a {R} resistor. How much power does the circuit draw?",
    "What power is dissipated when {V} is applied to a resistance of {R}?",
    "A {R} heating element runs on {V}. How many watts is that?",
    "Find the electrical power for a {V} source driving a {R} load.",
], [("ohms_law_current", {"voltage_v": S("V"), "resistance_ohm": S("R")}),
    ("electrical_power", {"voltage_v": S("V"), "current_a": R(1)})])

chain("ohm_voltage_power", {"I": Q(AMP), "R": Q(OHM)}, [
    "A current of {I} flows through a {R} resistor. How much power does it use?",
    "What power does a {R} resistor take when it carries {I}?",
    "{I} passes through a resistance of {R}. Find the electrical power.",
    "A {R} load carries {I}; how many watts does it consume?",
], [("ohms_law_voltage", {"current_a": S("I"), "resistance_ohm": S("R")}),
    ("electrical_power", {"voltage_v": R(1), "current_a": S("I")})])

chain("kinetic_energy_percent", {"m": Q(KG_CAR), "v": Q(MS_SPEED), "p": Q(PERCENT)}, [
    "A {m} vehicle moving at {v} loses {p} of its kinetic energy as heat in braking. How much energy is that?",
    "A {m} body travels at {v}. {p} of its kinetic energy is absorbed by a barrier. How much energy does the barrier take?",
    "What is {p} of the kinetic energy of a {m} object moving at {v}?",
    "An engine recovers {p} of the kinetic energy of a {m} train going {v}. How much is recovered?",
], [("kinetic_energy", {"mass_kg": S("m"), "velocity_m_s": S("v")}),
    ("percent_of", {"percent": S("p"), "value": R(1)})])

chain("mass_energy_percent", {"m": Q(KG_CONVERTED), "p": Q(PERCENT)}, [
    "A reactor captures {p} of the energy from fully converting {m} of mass. How much energy is captured?",
    "What is {p} of the rest energy of {m} of matter?",
    "Only {p} of the mass-energy of {m} can be collected. How much energy is that?",
], [("mass_to_energy", {"mass_kg": S("m")}), ("percent_of", {"percent": S("p"), "value": R(1)})])

chain("lift_energy_waste", {"m": Q(KG_OBJ), "h": Q(M_HEIGHT), "p": Q(PERCENT)}, [
    "A hoist lifts a {m} load {h}; {p} of the gravitational energy gained must be dissipated as heat in the cable. How much heat?",
    "Lifting {m} through {h} gives it gravitational potential energy; {p} of that amount is lost to friction. How many joules is that?",
    "What is {p} of the potential energy a {m} mass gains when raised {h}?",
], [("gravitational_potential_energy", {"mass_kg": S("m"), "height_m": S("h")}),
    ("percent_of", {"percent": S("p"), "value": R(1)})])

chain("fission_electricity", {"m": Q(KG_FUEL), "p": Q(PERCENT)}, [
    "A reactor turns {p} of the fission energy into electricity. How much electrical energy comes from fully fissioning {m} of uranium-235?",
    "Fully fissioning {m} of U-235 releases some energy; {p} becomes electricity. How much is that in joules?",
    "How many joules of electricity does a plant make from {m} of uranium-235 if its efficiency is {p}?",
], [("fission_energy_release", {"mass_kg": S("m"), "isotope": L("U235")}),
    ("percent_of", {"percent": S("p"), "value": R(1)})])

# ── single calls: the weak families ──────────────────────────────────────

def single(id, family, slots, templates, tool, args, per=0, bounds=None):
    return add(id, family, slots, templates, [(tool, args)], bounds or {}, per)


single("percent_of_power", "percent", {"p": Q(PERCENT), "P": Q(WATT)}, [
    "A {P} heater is {p} efficient at turning electricity into useful heat. How many watts of heat is that?",
    "A solar panel receives {P} of sunlight and converts {p} of it to electricity. How much electrical power does it produce?",
    "What is {p} of {P}?",
    "An LED turns {p} of its {P} input into visible light. How many watts of light is that?",
    "Only {p} of a {P} motor's input becomes mechanical output. What is the output power?",
    "If {p} of {P} is lost as heat, how much power is lost?",
], "percent_of", {"percent": S("p"), "value": S("P")})

single("percent_of_energy", "percent", {"p": Q(PERCENT), "E": Q(JOULE)}, [
    "A battery holds {E} and {p} of it is used up. How much energy was used?",
    "What is {p} of {E}?",
    "A process converts {p} of {E} of input energy into useful work. How much work is that?",
    "{p} of the {E} supplied is dissipated as heat. How much heat is that?",
], "percent_of", {"percent": S("p"), "value": S("E")})

single("percent_of_mass", "percent", {"p": Q(PERCENT), "m": Q(KG_OBJ)}, [
    "What is {p} of {m}?",
    "A rocket burns {p} of its {m} of fuel. How much fuel is burned?",
    "{p} of a {m} sample evaporates. How much mass is that?",
    "How much is {p} of a {m} load?",
], "percent_of", {"percent": S("p"), "value": S("m")})

single("percent_of_length", "percent", {"p": Q(PERCENT), "L": Q(M_HEIGHT)}, [
    "What is {p} of {L}?",
    "A rod of length {L} shrinks by {p}. By how many metres?",
    "A runner has covered {p} of a {L} course. How far is that?",
    "{p} of a {L} cable is buried. How much cable is buried?",
], "percent_of", {"percent": S("p"), "value": S("L")})


def convert(id, unit_from, unit_to, target_word, unit, contexts):
    base = ["Convert {v} to " + target_word + ".", "How many " + target_word + " is {v}?",
            "Express {v} in " + target_word + ".", "What is {v} in " + target_word + "?"]
    return single(id, "convert", {"v": Q(unit)}, base + contexts, "convert_units",
                  {"value": S("v"), "from_units": L(unit_from), "to_units": L(unit_to)}, per=40)


convert("convert_mev_ev", "MeV", "eV", "electronvolts", U(0.1, 900, ("MeV", 1)),
        ["A cosmic-ray proton arrives with {v} of energy. Give that in electronvolts.",
         "A detector records a particle at {v}; what is that in eV?"])
convert("convert_gev_mev", "GeV", "MeV", "megaelectronvolts", U(0.1, 900, ("GeV", 1)),
        ["An accelerator beam carries {v} per particle. Express that in MeV."])
convert("convert_kev_j", "keV", "J", "joules", U(0.1, 900, ("keV", 1)),
        ["An X-ray photon has {v} of energy. How many joules is that?"])
convert("convert_ev_j", "eV", "J", "joules", U(0.1, 900, ("eV", 1)),
        ["An electron gains {v} crossing a gap. What is that in joules?"])
convert("convert_mev_j", "MeV", "J", "joules", U(0.1, 900, ("MeV", 1)),
        ["A nuclear decay releases {v}. How many joules is that?"])
convert("convert_kmh_ms", "km/h", "m/s", "metres per second", U(1, 1500, ("km/h", 1)),
        ["A train travels at {v}. How fast is that in SI units?"])
convert("convert_mph_ms", "mph", "m/s", "metres per second", U(1, 700, ("mph", 1)),
        ["A car is doing {v}. What is that in metres per second?"])
convert("convert_atm_pa", "atm", "Pa", "pascals", U(0.01, 500, ("atm", 1)),
        ["A tank is pressurised to {v}. What is that in pascals?"])
convert("convert_kwh_j", "kWh", "J", "joules", U(0.1, 5000, ("kWh", 1)),
        ["A household uses {v} in a month. How many joules is that?"])

ELEMENTS = C((("helium", "He"), ("lithium", "Li"), ("beryllium", "Be"), ("boron", "B"),
              ("carbon", "C"), ("nitrogen", "N"), ("oxygen", "O"), ("fluorine", "F"),
              ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"), ("aluminium", "Al"),
              ("silicon", "Si"), ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"),
              ("argon", "Ar"), ("potassium", "K"), ("calcium", "Ca"), ("hydrogen", "H")))
single("ionization_energy", "atomic", {"el": ELEMENTS}, [
    "How much energy does it take to pull the outermost electron off a {el} atom?",
    "What is the first ionization energy of {el}?",
    "Energy needed to ionize a neutral {el} atom?",
    "How tightly is the outer electron of {el} bound, in electronvolts?",
    "How many electronvolts does it take to strip one electron from {el}?",
    "What does it cost, energetically, to remove one electron from {el}?",
], "first_ionization_energy", {"element_symbol": S("el")}, per=60)

single("level_hydrogen", "atomic", {"n": Q(U(1, 8, ("", 1), log=False, integer=True), True)}, [
    "What is the energy of the n = {n} level of hydrogen?",
    "Hydrogen's electron sits in level {n}. What is its energy?",
    "Energy of the n = {n} orbit in a hydrogen atom, in eV?",
    "Find the binding energy level of hydrogen for n = {n}.",
], "hydrogen_like_energy_level", {"n": S("n")}, per=30)

ION = C((("singly ionized helium", 2), ("doubly ionized lithium", 3), ("triply ionized beryllium", 4)))
single("level_hydrogen_like", "atomic", {"n": Q(U(1, 6, ("", 1), log=False, integer=True), True), "ion": ION}, [
    "What is the energy of the n = {n} level of {ion}?",
    "Find the energy of the n = {n} state in {ion}.",
    "{ion}: how much energy is the n = {n} level?",
    "A one-electron ion, {ion}, has its electron at n = {n}. What energy does that correspond to?",
], "hydrogen_like_energy_level", {"n": S("n"), "atomic_number": S("ion")}, per=40)

ASTRO = {
    "schwarzschild_radius": [
        "What is the Schwarzschild radius of a black hole of {M}?",
        "How big is the event horizon of a {M} black hole?",
        "A black hole has {M}. Find its Schwarzschild radius.",
        "If a star of {M} collapsed to a black hole, how large would the horizon be?",
        "Horizon radius for {M} of collapsed matter?"],
    "isco_radius": [
        "Where is the innermost stable circular orbit around a black hole of {M}?",
        "What is the ISCO radius for a {M} black hole?",
        "How close can a stable orbit get to a {M} black hole?",
        "Find the innermost stable orbit radius for a black hole with {M}."],
    "photon_sphere_radius": [
        "At what radius can light orbit a black hole of {M}?",
        "What is the photon sphere radius of a {M} black hole?",
        "Find the radius of the photon sphere around {M} of collapsed matter.",
        "How far from a {M} black hole does light travel in a circle?"],
    "hawking_temperature": [
        "What is the Hawking temperature of a black hole of {M}?",
        "How hot does a {M} black hole glow from Hawking radiation?",
        "Find the Hawking temperature for a black hole with {M}.",
        "A black hole has {M}; what temperature does it radiate at?"],
    "hawking_evaporation_time": [
        "How long would a black hole of {M} take to evaporate by Hawking radiation?",
        "What is the evaporation time of a {M} black hole?",
        "Find the lifetime of a {M} black hole against Hawking evaporation.",
        "If nothing falls in, when does a {M} black hole finally evaporate?"],
    "procedure_black_hole_profile": [
        "Give me a full profile of a black hole of {M}.",
        "Describe a {M} black hole: horizon, photon sphere, innermost orbit, temperature.",
        "What are the main properties of a black hole with {M}?",
        "Profile a {M} black hole."],
}
for tool, tpls in ASTRO.items():
    single(f"solar_{tool}", "astro", {"M": Q(SOLAR)}, tpls, tool, {"mass_kg": S("M")}, per=90)


# ── generation ───────────────────────────────────────────────────────────

# Tool pairs that are also how some benchmark questions are solved. Reported, not hidden.
PAIRS_SHARED_WITH_BENCHMARK = {
    ("circular_orbit_velocity", "kinetic_energy"),
    ("fission_energy_release", "joules_to_TNT"),
    ("tnt_to_joules", "energy_to_mass"),
}


def _sample_slot(name: str, spec, rng: random.Random) -> D.Arg:
    if isinstance(spec, C):
        text, value = rng.choice(spec.options)
        return D.Arg(name, value, text, name, key_text=text if isinstance(text, str) else "")
    val = D._sample_range(spec.unit.lo, spec.unit.hi, spec.unit.log, rng)
    text, label, _, utext = D.render_quantity(val, spec.unit, rng, spec.integer)
    return D.Arg(name, label, text.strip(), name, utext)


def build_plan(recipe: Recipe, args: dict) -> list:
    steps = []
    for tool, spec in recipe.steps:
        a = {}
        for param, (kind, x) in spec.items():
            a[param] = (args[x].value if kind == "s" else f"${x}" if kind == "r"
                        else f"${x[0]}.{x[1]}" if kind == "rf" else args[x[0]].value[x[1]]
                        if kind == "k" else x)
        steps.append({"tool": tool, "args": a})
    return steps


def render(recipe: Recipe, args: dict, template: str) -> str:
    q = template.format(**{k: v.text for k, v in args.items()})
    return re.sub(r"\s+", " ", q).replace(" ,", ",").strip()


async def run_plan(session, steps: list, question: str, bounds: dict):
    """Execute the plan; returns the list of step results or a drop reason."""
    values, results = [], []
    for i, s in enumerate(steps, start=1):
        try:
            a = RP.resolve_args(s["args"], values)
        except RP.PlanError as e:
            return None, f"{s['tool']}: bad reference ({e})"
        res = await D.execute(session, s["tool"], a, question, timeout_s=120, keep_raw=True)
        if res is None:
            return None, f"{s['tool']} returned no value"
        v = res["value"]
        if isinstance(res.get("raw"), dict):
            v = res["raw"]          # a lookup: a later call reads one field of it ("$1.mass_kg")
            if any(x is None for x in v.values()):
                return None, f"{s['tool']} result has empty fields"       # e.g. a material key the tool does not know
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            if i < len(steps):
                return None, f"{s['tool']} value is not a number"
        elif i in bounds and not (bounds[i][0] <= v <= bounds[i][1]):
            return None, f"step {i} value {v:.3g} outside {bounds[i]}"
        values.append(v)
        results.append({"tool": s["tool"], "args": a, "value": v, "units": res.get("units")})
    return results, ""


async def generate(session, per_default: int, seed: int, guard, only: set | None = None):
    """-> (examples, report). An example is a dict ready to write (seeds.jsonl)."""
    examples, drops, per_recipe = [], Counter(), Counter()
    seen: set = set()
    await D.execute(session, "convert_units", {"value": 1.0, "from_units": "eV", "to_units": "J"},
                    "", timeout_s=900)               # a fresh server's first call imports pint: minutes on a busy CPU
    for recipe in RECIPES:
        if only and recipe.id not in only:
            continue
        missing = [t for t in recipe.templates if not all("{%s}" % s in t for s in recipe.used_slots())]
        assert not missing, (recipe.id, "template lacks a slot the plan uses", missing[:1])
        for t in recipe.templates:
            assert set(re.findall(r"\{(\w+)\}", t)) <= set(recipe.slots), (recipe.id, "unknown slot", t)
        rng = random.Random(f"{seed}:{recipe.id}")
        want, made, attempts = recipe.per or per_default, 0, 0
        while made < want and attempts < want * 6:
            attempts += 1
            args = {n: _sample_slot(n, spec, rng) for n, spec in recipe.slots.items()}
            q = render(recipe, args, rng.choice(recipe.templates))
            key = D._norm(q)
            if key in seen:
                drops["duplicate"] += 1
                continue
            steps = build_plan(recipe, args)
            results, why = await run_plan(session, steps, q, recipe.bounds)
            if results is None:
                drops["outside bounds" if why.startswith("step") else
                      "not a number" if "not a number" in why else "call returned no value"] += 1
                continue
            if guard is not None:
                bad = next((g for g in (guard.check(q, r["tool"], r["value"]) for r in results) if g), None)
                if bad:
                    drops["benchmark overlap"] += 1
                    continue
            verdict = D.classifier_verdict(q, {"tool": steps[-1]["tool"]})
            if verdict == "conflict":
                drops["classifier conflict"] += 1
                continue
            seen.add(key)
            made += 1
            examples.append({
                "question": q, "plan": RP.make_plan(steps), "group": f"recipe:{recipe.id}:{made}:{seed}",
                "meta": {"source": "recipe", "family": recipe.family, "recipe": recipe.id,
                         "steps": len(steps), "classifier": verdict,
                         "verified": [{"tool": r["tool"], "value": r["value"], "units": r["units"]}
                                      for r in results]},
                "args_meta": [{"name": a.name, "value": a.value, "text": a.text,
                               "unit_text": a.unit_text, "key_text": a.key_text}
                              for a in args.values()],
            })
        per_recipe[recipe.id] = made
        print(f"  {recipe.id:34s} {made:4d}/{want}", flush=True)
    return examples, {"drops": dict(drops), "per_recipe": dict(per_recipe)}


def write_seeds(examples: list, report: dict, out: Path, seed: int, per: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "seeds.jsonl").open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    chains = Counter(len(e["plan"]["steps"]) if "steps" in e["plan"] else 1 for e in examples)
    pairs = Counter(tuple(s["tool"] for s in e["plan"]["steps"]) for e in examples if "steps" in e["plan"])
    shared = sum(n for p, n in pairs.items() if p in PAIRS_SHARED_WITH_BENCHMARK)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "per_recipe_default": per,
        "recipes": len(RECIPES), "seeds": len(examples), "calls_per_plan": dict(chains),
        "families": dict(Counter(e["meta"]["family"] for e in examples)),
        "tool_pairs": {" > ".join(p): n for p, n in pairs.most_common()},
        "pairs_shared_with_benchmark": {"pairs": sorted(" > ".join(p) for p in PAIRS_SHARED_WITH_BENCHMARK),
                                        "seeds": shared, "share_of_chain_seeds":
                                        round(shared / max(1, sum(pairs.values())), 3)},
        **report,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


async def _amain(args) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    async with stdio_client(D.server_params(args.server_mem_gb)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            guard = None if args.no_guard else D.BenchmarkGuard.from_repo()
            only = set(args.only.split(",")) if args.only else None
            print(f"{len(RECIPES)} recipes")
            examples, report = await generate(session, args.per_recipe, args.seed, guard, only)
    manifest = write_seeds(examples, report, args.out, args.seed, args.per_recipe)
    print(f"\nWrote {args.out}: {manifest['seeds']} seeds; calls per plan {manifest['calls_per_plan']}; "
          f"drops {manifest['drops']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_recipes")
    p.add_argument("--per-recipe", type=int, default=70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="", help="comma-separated recipe ids")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
