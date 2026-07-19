"""Materia verb manifest — the CONTRACT between the translator and the verbs.

This is the seam between the two tracks: the Mentat track authors physics verbs
(scenarios); the Materia translator routes natural language to them and chains
them. Neither needs to know the other's internals — they meet here. Register a
verb in this table and the translator can route AND chain it with no translator
code change.

A verb entry:
    "verb_name": {
        "description": "<one line for the LLM>",
        "answers":     ["<trigger phrase>", ...],   # what questions it answers
        "slots":       { <slot>: <slot spec>, ... }, # the inputs it takes
        "outputs":     ["<field>", ...],             # named results (set below)
    }

A slot spec:
    {
        "unit":      "m" | "m/s" | "material" | "mach" | "",
        "default":   <value>,                 # used if neither filled nor bound
        "aliases":   ["<nl cue>", ...],        # words the extractor latches onto
        "bind_from": "<output_field>",         # OPTIONAL — auto-chain: thread this
    }                                          #   slot from a prior step that
                                               #   OUTPUTS <output_field>.

`bind_from` is the whole chaining contract: a descent verb whose altitude slot
declares bind_from="apex_altitude_m" auto-wires to any prior step that produces
an apex. Declare it once on the verb; the translator does the threading.
"""
from __future__ import annotations

# A slot: the unit it's in, a default, the NL cues the extractor latches onto,
# and optionally `bind_from` — the prior-step output to thread it from.
_DROP_SLOTS = {
    "material_key": {"unit": "material", "default": "iron",
                     "aliases": ["made of", "ball of", "sphere of"]},
    "radius_m": {"unit": "m", "default": 0.05,
                 "aliases": ["radius", "sphere", "ball", "diameter"]},
    "drop_altitude_m": {"unit": "m", "default": 10000.0,
                        "aliases": ["from", "altitude", "height", "dropped from"],
                        "bind_from": "apex_altitude_m"},
}

VERB_MANIFEST = {
    "terminal_velocity_drop": {
        "description": "How fast a solid sphere is moving when it hits the "
                       "ground, dropped from altitude through the air.",
        "answers": ["how fast", "impact speed", "terminal velocity",
                    "how hard does it hit"],
        "slots": _DROP_SLOTS,
    },
    "drag_heating_drop": {
        "description": "How much a falling body heats up from air drag — the "
                       "temperature rise as drag dissipates its energy.",
        "answers": ["how hot", "does it heat up", "temperature rise",
                    "does it cook", "does it burn", "does it melt"],
        "slots": _DROP_SLOTS,
    },
    "high_altitude_descent": {
        "description": "A body dropped from the stratosphere/space: does it go "
                       "supersonic in thin air and then slow down as the air "
                       "thickens? Skydiver / free-fall / parachute from altitude.",
        "answers": ["does it slow down", "does it go supersonic falling",
                    "skydiver from space", "parachute from the stratosphere"],
        "slots": {
            "start_altitude_m": {"unit": "m", "default": 35000.0,
                                 "aliases": ["from", "altitude", "jump from"],
                                 "bind_from": "apex_altitude_m"},
            "payload_mass_kg": {"unit": "kg", "default": 118.0, "aliases": ["mass"]},
            "drag_area_m2": {"unit": "m^2", "default": 0.28, "aliases": ["area"]},
            "cd": {"unit": "", "default": 0.70, "aliases": ["drag coefficient"]},
        },
    },
    "supersonic_projectile": {
        "description": "A projectile fired faster than sound: how its drag "
                       "coefficient spikes through the transonic zone and how it "
                       "decelerates back below the sound barrier.",
        "answers": ["supersonic", "breaks the sound barrier", "mach",
                    "bullet drag", "transonic"],
        "slots": {
            "launch_mach": {"unit": "mach", "default": 2.5,
                            "aliases": ["mach", "times the speed of sound"]},
            "mass_kg": {"unit": "kg", "default": 0.02, "aliases": ["mass"]},
            "diameter_m": {"unit": "m", "default": 0.01, "aliases": ["diameter",
                                                                     "caliber"]},
        },
    },
    "vertical_launch": {
        "description": "Throw/launch a sphere straight up at a speed; how high "
                       "it reaches (its apex). Feeds a descent verb.",
        "answers": ["throw it up", "launch straight up", "how high does it go"],
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["made of", "ball of"]},
            "radius_m": {"unit": "m", "default": 0.05, "aliases": ["radius",
                                                                   "ball", "sphere"]},
            "launch_speed_m_s": {"unit": "m/s", "default": 300.0,
                                 "aliases": ["thrown at", "launched at", "up at"]},
        },
    },
    "orbital_velocity": {
        "description": "Orbital speed and period of a circular orbit at a given "
                       "altitude around a body (Earth, Moon, Mars, …).",
        "answers": ["orbital velocity", "orbital speed", "how fast to orbit",
                    "orbital period"],
        "slots": {
            "central_body": {"unit": "body", "default": "earth",
                             "aliases": ["around", "above"]},
            "altitude_m": {"unit": "m", "default": 400000.0,
                           "aliases": ["altitude", "at", "above"]},
            "semimajor_axis_au": {"unit": "au", "default": None,
                                  "aliases": ["au", "astronomical units"]},
        },
    },
    "material_profile": {
        "description": "Characterize a material — density, sound speed, specific "
                       "heat, conductivity, the elastic moduli, restitution.",
        "answers": ["properties of", "young's modulus of", "how stiff",
                    "material properties", "characterize"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to steel_mild. Require a named
        # material.
        "material_required": True,
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
            "temperature_k": {"unit": "K", "default": 293.15,
                              "aliases": ["at", "temperature"]},
        },
    },
    "structural_response": {
        "description": "Structural integrity of a material under load: elastic "
                       "moduli, fracture toughness, fatigue life, plastic flow.",
        "answers": ["fracture toughness", "fatigue life", "yield strength",
                    "stress-strain"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to steel_mild. Require a named
        # material.
        "material_required": True,
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
        },
    },
    "thermal_response": {
        "description": "How a material responds to heat: thermal expansion, "
                       "melting/phase data, radiated power.",
        "answers": ["thermal expansion", "melting point of", "how much does it "
                    "expand", "latent heat"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to steel_mild. Require a named
        # material.
        "material_required": True,
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
        },
    },
    "contact_conduction": {
        "description": "A hot block set on a cold slab: per-cell heat flow "
                       "across the material interface (harmonic-mean face "
                       "conductivity), in room air (IRT) or interstellar "
                       "vacuum (ISM). Renderable as a temperature field.",
        "answers": ["how fast does heat flow between them", "does the slab "
                    "warm up", "heat crossing the interface"],
        "slots": {
            "hot_material": {"unit": "material", "default": "iron",
                             "aliases": ["hot"]},
            "cold_material": {"unit": "material", "default": "copper",
                              "aliases": ["cold", "onto", "on a"]},
            "T_hot": {"unit": "K", "default": 900.0,
                      "aliases": ["at", "temperature"]},
            "environment": {"unit": "preset", "default": "IRT",
                            "aliases": ["in"]},
        },
    },
    "rotational_dynamics": {
        "description": "Rigid-body rotation: moments of inertia for standard "
                       "shapes, angular momentum, rotational kinetic energy.",
        "answers": ["moment of inertia", "angular momentum", "rotational kinetic "
                    "energy"],
        "slots": {
            "mass_kg": {"unit": "kg", "default": 1.0, "aliases": ["mass"]},
            "radius_m": {"unit": "m", "default": 0.1, "aliases": ["radius"]},
        },
    },
    "material_full_profile": {
        "description": "Exhaustively characterize a material — every property "
                       "suite sigma-ground exposes (thermal, elastic, magnetic, "
                       "acoustic, electrochemical, surface, …).",
        "answers": ["all properties of", "full profile of",
                    "everything about the material"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to steel_mild and report a
        # confidently-wrong full property suite. Require a named material.
        "material_required": True,
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
        },
    },
    "quantum_report": {
        "description": "Quantum & atomic physics: spectra, quantum wells, "
                       "tunneling, crystal-field splitting, superconductivity.",
        "answers": ["atomic spectra", "energy levels", "quantum well",
                    "superconductor", "tunneling"],
        "slots": {
            "element_Z": {"unit": "Z", "default": 1, "aliases": ["element"]},
        },
    },
}

# Each verb's named outputs — what a later step can read from it. Used by the
# planner's deterministic auto-binder to thread physical state forward.
_VERB_OUTPUTS = {
    "terminal_velocity_drop": ["impact_speed_m_s", "max_speed_m_s",
                               "terminal_velocity_m_s", "fall_time_s"],
    "drag_heating_drop": ["delta_T_K", "peak_T_K", "dissipation_J",
                          "impact_speed_m_s", "temperature_history", "glows"],
    "contact_conduction": ["heat_crossed_J", "interface_k_W_mK",
                           "peak_T_end_K", "glows"],
    "high_altitude_descent": ["max_speed_m_s", "max_mach", "landing_speed_m_s",
                              "mach1_altitude_m"],
    "supersonic_projectile": ["launch_speed_m_s", "distance_to_subsonic_m",
                              "final_mach"],
    "vertical_launch": ["apex_altitude_m", "time_to_apex_s"],
    "orbital_velocity": ["orbital_velocity_m_s", "orbital_period_s", "altitude_m"],
    "material_profile": ["density_kg_m3", "youngs_modulus_Pa",
                         "specific_heat_J_kgK", "sound_velocity_m_s"],
    "structural_response": ["youngs_modulus_Pa", "shear_modulus_Pa",
                            "bulk_modulus_Pa"],
    "thermal_response": ["radiated_power_W_m2"],
    "rotational_dynamics": ["moment_of_inertia_sphere", "angular_momentum",
                            "rotational_ke_J"],
    "material_full_profile": ["suites_computed"],
    "quantum_report": ["suites_computed"],
}
for _v, _o in _VERB_OUTPUTS.items():
    VERB_MANIFEST[_v]["outputs"] = _o


# Trigger keywords the translator routes on. A plain string matches as a
# substring; a string starting with \b is treated as a word-boundary regex
# (so "mach" doesn't fire on "machine"). `exclusive` verbs own their phrasing —
# a match returns ONLY that verb; non-exclusive verbs can co-match (a question
# that is both "how fast" and "how hot" routes to both drop verbs). Keeping
# triggers here means a new verb is routable purely by its manifest entry.
SPEED_TRIGGERS = ["how fast", "speed", "velocity", "terminal", "impact",
                  "hit the ground", "hits the ground", "how hard", "m/s",
                  "mph", "km/h", "kph", "land at", "how quick", "quick",
                  "splat", "splats", "smack", "smacks", "wallop"]
HEAT_TRIGGERS = ["heat", "hot", "warm", "cook", "burn", "melt", "glow",
                 "temperature", "thermal", "scorch", "fry", "char", "ignite"]
_VERB_TRIGGERS = {
    "terminal_velocity_drop": (SPEED_TRIGGERS, False),
    "drag_heating_drop": (HEAT_TRIGGERS, False),
    "high_altitude_descent": (["parachute", "skydiv", "drogue", "stratosphere",
                               "from space", "from orbit", "reentry",
                               "re-entry"], True),
    "supersonic_projectile": (["supersonic", "sound barrier", "transonic",
                               r"\bmach\b"], True),
    "vertical_launch": ([], False),   # chain-only; reached via the launch template
    "orbital_velocity": (["orbital velocity", "orbital speed", "orbital period",
                          "geostationary", "geosynchronous", "low earth orbit",
                          "in orbit", r"\borbit\b", r"\borbits\b", "around the sun",
                          "revolve", "revolves", "revolution"], True),
    "material_profile": (["properties of", "characterize", "material properties",
                          "young's modulus", "youngs modulus", "elastic modulus",
                          "shear modulus", "bulk modulus", "tensile strength",
                          "specific heat of", "thermal conductivity of"], True),
    "structural_response": (["fracture toughness", "fatigue life",
                             "fatigue strength", "yield strength",
                             "ultimate tensile", "stress-strain",
                             "plastic deformation", "fracture mechanics",
                             "stress intensity", "factor of safety"], True),
    "thermal_response": (["thermal expansion", "thermal strain", "thermal stress",
                          "expansion coefficient", "coefficient of expansion",
                          "melting point of", "latent heat",
                          "how much does it expand"], True),
    "contact_conduction": (["conduction", "hot cube on", "hot block on",
                            "touching", "press a hot", "heat flow between",
                            "heat flowing between", "on a cold"], True),
    "rotational_dynamics": (["moment of inertia", "angular momentum",
                             "rotational kinetic energy", "rotational inertia",
                             "flywheel inertia"], True),
    "material_full_profile": (["all properties of", "every property of",
                               "full profile of", "complete profile of",
                               "full material profile", "characterize everything"],
                              True),
    "quantum_report": (["crystal field", "crystal-field", "ligand field"], True),
}
for _v, (_t, _ex) in _VERB_TRIGGERS.items():
    VERB_MANIFEST[_v]["triggers"] = _t
    VERB_MANIFEST[_v]["exclusive"] = _ex


# ── Domain verbs (the coverage campaign) — registered for routing here ──────
# Each entry carries DISTINCTIVE triggers (a match returns ONLY this verb), a
# couple of example questions (few-shot for the LLM *and* the validation corpus),
# and slots (material_key where the verb takes one; everything else defaults in
# the verb, so routing to the verb alone already yields a valid answer). The
# example questions are the "questions and answers" that both drive and verify
# the matching — see misc/routing_corpus.py.
_MAT = {"material_key": {"unit": "material", "default": "iron",
                         "aliases": ["of", "for"]}}
_ELEM = {"element_Z": {"unit": "Z", "default": 1, "aliases": ["element", "Z="]}}

_ROUTABLE_NEW = {
    # ── electromagnetism ──
    "charged_particle": {
        "description": "Electric & magnetic forces on a charged particle — "
                       "Coulomb's law, point fields, cyclotron motion, Lorentz "
                       "force, EM-wave energy.",
        "triggers": ["coulomb force", "coulomb's law", "electric field",
                     "electric potential", "cyclotron", "lorentz force",
                     "charged particle", "force between two charges",
                     "two charges", "point charge"],
        "examples": ["coulomb force between two 1 microcoulomb charges 1 cm apart",
                     "cyclotron frequency of an electron in a 1 tesla field"],
        "slots": {}},
    "transmission_line": {
        "description": "Transmission-line EM — coaxial / twisted-pair / Möbius "
                       "conductor inductance, characteristic impedance, skin "
                       "depth, field cancellation.",
        "triggers": ["coaxial", "transmission line", "twisted pair",
                     "characteristic impedance", "skin depth", "mobius conductor",
                     "möbius conductor", "cable inductance"],
        "examples": ["characteristic impedance of a coaxial cable",
                     "skin depth of copper at 1 MHz"],
        "slots": _MAT},
    "optical_fiber": {
        "description": "Fiber-optic & nonlinear photonics — numerical aperture, "
                       "mode count, single-mode cutoff, Bragg stacks, Kerr / "
                       "second-harmonic.",
        "triggers": ["optical fiber", "fibre optic", "fiber optic",
                     "numerical aperture", "single mode fiber", "bragg",
                     "second harmonic", "kerr effect"],
        "examples": ["numerical aperture of an optical fiber",
                     "how many modes does an optical fiber support"],
        "slots": {}},
    "optical_dispersion": {
        "description": "Optical dispersion & colour of a material — refractive "
                       "index, metal reflectance, Drude scattering, plasma "
                       "frequency, dielectric colour.",
        "triggers": ["refractive index", "reflectance of", "drude",
                     "optical dispersion", "cauchy", "metal colour",
                     "metal color"],
        "examples": ["reflectance of copper", "refractive index of glass"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to iron/steel_mild and report a
        # confidently-wrong refractive index. Require a named material.
        "material_required": True,
        "slots": _MAT},
    "semiconductor_device": {
        "description": "Semiconductor device physics — band gap, carrier "
                       "concentration & mobility, p-n junction built-in voltage "
                       "and depletion width, diode current.",
        "triggers": ["semiconductor", "band gap", "diode", "p-n junction",
                     "pn junction", "depletion width", "carrier concentration",
                     "fermi level"],
        "examples": ["band gap of silicon", "depletion width of a pn junction"],
        "slots": {"sc_key": {"unit": "material", "default": "silicon",
                             "aliases": ["of", "for"]}}},
    "magnetic_material": {
        "description": "Magnetic & dielectric response of a material — "
                       "ferromagnetism, susceptibility, saturation "
                       "magnetization; dielectric breakdown and stored energy.",
        "triggers": ["magnetic susceptibility", "ferromagnetic",
                     "saturation magnetization", "dielectric breakdown",
                     "breakdown field", "diamagnetic", "paramagnetic"],
        "examples": ["is iron ferromagnetic",
                     "dielectric breakdown field of quartz"],
        "slots": _MAT},
    "magnetic_hysteresis": {
        "description": "Ferromagnetic hysteresis — the B-H loop, coercivity, "
                       "anhysteretic magnetization, Curie-Weiss susceptibility, "
                       "energy lost per cycle.",
        "triggers": ["hysteresis", "b-h loop", "bh loop", "coercivity",
                     "hysteresis loss", "curie-weiss", "magnetic loss",
                     "energy loss per cycle"],
        "examples": ["hysteresis loop of iron",
                     "magnetic energy loss per cycle of a transformer core"],
        "slots": _MAT},
    "piezoelectric_material": {
        "description": "Piezoelectric response — charge/voltage from stress, "
                       "strain from field, electromechanical coupling, harvested "
                       "energy.",
        "triggers": ["piezoelectric", "piezo", "quartz frequency",
                     "electromechanical coupling"],
        "examples": ["piezoelectric voltage of PZT under stress",
                     "electromechanical coupling of quartz"],
        "slots": {"material_key": {"unit": "material", "default": "PZT4",
                                   "aliases": ["of", "for"]}}},
    "thermoelectric": {
        "description": "Thermoelectric transport — Seebeck-derived figure of "
                       "merit ZT, electrical conductivity, Carnot bound, "
                       "thermocouple voltage.",
        "triggers": ["thermoelectric", "seebeck", "peltier",
                     "figure of merit", "thermocouple", "figure of merit zt"],
        "examples": ["thermoelectric figure of merit of bismuth telluride",
                     "seebeck voltage of a thermocouple"],
        "slots": _MAT},
    # ── mechanics & motion ──
    "projectile_motion": {
        "description": "Projectile & inclined-plane kinematics — range, max "
                       "height, time of flight, launch at an angle, motion on a "
                       "ramp.",
        "triggers": ["projectile range", "projectile motion", "range of a",
                     "launch angle", "trajectory", "at an angle", "how far does",
                     "thrown at an angle", "ballistic", "fired at an angle",
                     "maximum height of", "degree angle", "degrees"],
        "examples": ["range of a projectile launched at 45 degrees",
                     "how far does a ball thrown at 30 m/s travel"],
        "slots": {}},
    "collision_dynamics": {
        "description": "Energy, momentum & collisions — kinetic / potential / "
                       "rotational energy, elastic and inelastic collision "
                       "outcomes, energy dissipated.",
        "triggers": ["collision", "collide", "elastic collision",
                     "inelastic collision", "conservation of momentum",
                     "two balls collide", "head-on", "crash into"],
        "examples": ["elastic collision of two balls",
                     "energy lost in an inelastic collision"],
        "slots": {}},
    "rolling_object": {
        "description": "A body rolling without slipping — moments of inertia, "
                       "rolling acceleration down an incline, speed from a "
                       "height, rolling distance.",
        "triggers": ["rolling without slipping", "rolls down", "rolling down",
                     "down an incline", "down a ramp", "rolling acceleration",
                     "rolling speed"],
        "examples": ["how fast does a sphere roll down a ramp",
                     "rolling acceleration of a cylinder on an incline"],
        "slots": {}},
    "relativistic_motion": {
        "description": "Special relativity — Lorentz factor, time dilation, "
                       "length contraction, relativistic energy / momentum, "
                       "relativistic Doppler.",
        "triggers": ["relativistic", "lorentz factor", "time dilation",
                     "length contraction", "speed of light", "special "
                     "relativity", "near light speed"],
        "examples": ["time dilation at 0.9c",
                     "relativistic energy of a proton near light speed"],
        "slots": {}},
    "thermal_statistics": {
        "description": "Statistical mechanics of a gas — Maxwell-Boltzmann "
                       "speeds, Boltzmann factors & entropy, partition function, "
                       "Fermi-Dirac / Bose-Einstein.",
        "triggers": ["maxwell-boltzmann", "boltzmann distribution",
                     "boltzmann factor", "partition function", "fermi-dirac",
                     "bose-einstein", "average speed of", "statistical mechanics",
                     "most probable speed", "rms speed", "thermal speed",
                     "root mean square speed"],
        "examples": ["average speed of nitrogen molecules at 300 K",
                     "maxwell-boltzmann distribution of a gas"],
        "slots": {}},
    "elastic_solid": {
        "description": "Linear elasticity of a solid — Lamé parameters, elastic "
                       "moduli, P-wave modulus, strain energy, yield check.",
        "triggers": ["lame parameter", "lamé parameter", "p-wave modulus",
                     "strain energy", "elastic modulus of", "shear modulus of",
                     "poisson ratio"],
        "examples": ["shear modulus of steel", "Lamé parameters of aluminum"],
        # The material IS the answer here (same reasoning as acoustics) — with
        # none named the verb would default to steel_mild/iron and report
        # confidently-wrong moduli. Require a named material; else decline.
        "material_required": True,
        "slots": _MAT},
    "acoustics": {
        "description": "Acoustic waves in and across materials — longitudinal / "
                       "transverse sound speed, acoustic impedance, reflection / "
                       "transmission, resonance.",
        "triggers": ["speed of sound", "acoustic impedance", "sound speed",
                     "acoustic", "ultrasound", "wave speed in",
                     "sound travel through", "how fast does sound"],
        "examples": ["speed of sound in steel", "acoustic impedance of water"],
        # The material IS the answer here; with none named the verb would default
        # to iron and report a confidently-wrong sound speed ("at sea level" →
        # iron 5942 m/s). Require a named material — else decline (→ Q&A).
        "material_required": True,
        "slots": _MAT},
    "tribology": {
        "description": "Friction, wear & adhesion at a sliding contact — Archard "
                       "wear, wear rate, work of adhesion, contact angle.",
        "triggers": ["wear rate", "archard", "sliding wear", "friction",
                     "work of adhesion", "tribolog", "abrasion"],
        "examples": ["wear rate of steel sliding on iron",
                     "work of adhesion between two metals"],
        # The material pair IS the answer (same reasoning as acoustics) — with
        # none named the verb would silently default to steel/iron. Require a
        # named material; else decline.
        "material_required": True,
        "slots": _MAT},
    "composite_material": {
        "description": "Composite-material mechanics — stiffness & density by "
                       "rule-of-mixtures and Halpin-Tsai, foam scaling, thermal "
                       "expansion.",
        "triggers": ["composite", "fiber reinforced", "fibre reinforced",
                     "carbon fiber", "carbon fibre", "halpin-tsai",
                     "rule of mixtures"],
        "examples": ["stiffness of a carbon-fiber composite",
                     "modulus of a fiber-reinforced composite"],
        "slots": {"composite_key": {"unit": "material",
                                    "default": "cfrp_unidirectional",
                                    "aliases": ["of", "for"]}}},
    "metallurgy": {
        "description": "Microstructure-dependent strength — Hall-Petch yield vs "
                       "grain size, polycrystal & Taylor hardening, grain "
                       "growth, alloy prediction.",
        "triggers": ["grain size", "hall-petch yield", "hall petch yield",
                     "hall-petch", "hall petch", "grain boundary",
                     "dislocation density", "annealing", "alloy properties",
                     "microstructure"],
        "examples": ["Hall-Petch yield strength vs grain size",
                     "how grain size affects steel strength"],
        "slots": _MAT},
    "viscoelastic_material": {
        "description": "Viscoelastic polymer response — Maxwell / Kelvin-Voigt "
                       "creep and stress relaxation, storage & loss moduli, loss "
                       "tangent.",
        "triggers": ["viscoelastic", "creep of", "stress relaxation",
                     "maxwell model", "kelvin-voigt", "loss tangent",
                     "storage modulus"],
        "examples": ["creep of a polymer under load",
                     "stress relaxation of a viscoelastic material"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to iron/steel_mild. Require a
        # named material (or class word: "polymer", "material", ...).
        "material_required": True,
        "slots": _MAT},
    "fluid_flow": {
        "description": "Viscous fluid flow — pipe (Poiseuille) flow, Stokes drag "
                       "& settling, Reynolds number, boundary layer.",
        "triggers": ["reynolds number", "stokes drag", "poiseuille",
                     "pipe flow", "viscous flow", "boundary layer",
                     "terminal settling", "turbulent", "laminar flow",
                     "flow in this pipe", "flow in a pipe"],
        "examples": ["reynolds number of water flowing in a pipe",
                     "stokes drag on a settling sphere"],
        "slots": {}},
    "fluid_dynamics": {
        "description": "Liquid-phase fluid properties — dynamic & kinematic "
                       "viscosity, surface tension.",
        "triggers": ["viscosity of", "surface tension", "kinematic viscosity",
                     "how viscous"],
        "examples": ["viscosity of ethanol at 20 C", "surface tension of water"],
        "slots": {"liquid_key": {"unit": "material", "default": "water",
                                 "aliases": ["of", "for"]}}},
    "diffusion_transport": {
        "description": "Mass diffusion & transport — Einstein-Stokes "
                       "diffusivity, Fick's first and second laws, "
                       "interdiffusion.",
        "triggers": ["diffusion coefficient", "fick's", "ficks",
                     "einstein-stokes", "diffusivity", "interdiffusion",
                     "how fast does it diffuse"],
        "examples": ["diffusion coefficient of a particle in water",
                     "Fick's first law flux"],
        "slots": {}},
    # ── chemistry ──
    "chemistry_lab": {
        "description": "Acid-base & solution chemistry — pH, Ka/Kb, buffers, "
                       "titration, plus ideal-gas transport and galvanic cells.",
        "triggers": ["ph of", "acid", "base", "buffer", "titration",
                     "henderson", "pka", "pkb", "acidic", "alkaline"],
        "examples": ["pH of 0.1 M acetic acid",
                     "titration curve of a weak acid with a strong base"],
        "slots": {}},
    "solution_chemistry": {
        "description": "Electrolyte solutions — Debye-Hückel activity & "
                       "screening, colligative shifts (boiling-point elevation, "
                       "freezing-point depression), Faraday electrolysis.",
        "triggers": ["solubility", "electrolyte", "debye-huckel", "debye-hückel",
                     "osmotic pressure", "boiling point elevation",
                     "freezing point depression", "electrolysis", "faraday"],
        "examples": ["boiling point elevation of salt water",
                     "how much metal is deposited by electrolysis"],
        "slots": {}},
    "chemical_reaction": {
        "description": "Reaction kinetics & thermodynamics — Arrhenius rate, "
                       "equilibrium constant & Gibbs energy, activation energy.",
        "triggers": ["reaction rate", "arrhenius", "activation energy",
                     "equilibrium constant", "gibbs energy of reaction",
                     "rate constant", "reaction enthalpy"],
        "examples": ["arrhenius reaction rate at 350 K",
                     "equilibrium constant of a reaction"],
        "slots": {}},
    "molecular_bond": {
        "description": "Chemical-bond properties — bond energy (Pauling), length, "
                       "force constant, dipole, hybridization, reduced mass.",
        "triggers": ["bond energy", "bond length", "bond dipole",
                     "chemical bond", "hybridization", "bond order",
                     "pauling"],
        "examples": ["C-O bond energy", "hybridization of carbon in methane"],
        "slots": {}},
    "intermolecular_forces": {
        "description": "Hydrogen bonding & intermolecular forces — H-bond "
                       "energy, estimated boiling point & vaporization enthalpy.",
        "triggers": ["hydrogen bond", "intermolecular", "van der waals",
                     "vaporization enthalpy", "why does water boil"],
        "examples": ["hydrogen bond energy in water",
                     "intermolecular forces in water"],
        "slots": {}},
    "corrosion_attack": {
        "description": "Corrosion & oxidation — corrosion rate, galvanic "
                       "potential & series, oxide growth over a duration, "
                       "Pilling-Bedworth ratio; cited soil/pH environment "
                       "regime (qualitative).",
        "triggers": ["corrosion", "corrode", "corroding", "corroded", "rust",
                     "galvanic", "pilling-bedworth", "oxidation rate",
                     "oxide growth"],
        "examples": ["corrosion rate of iron",
                     "zinc rod in alkaline soil corroding over 5 years"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to iron and report confidently
        # wrong electrochemistry. Require a named material.
        "material_required": True,
        "slots": {**_MAT,
                  "duration_s": {"unit": "s", "default": 3.15e7,
                                 "aliases": ["over", "for", "after", "years",
                                             "months", "weeks", "days"]},
                  "environment": {"unit": "environment", "default": None,
                                  "aliases": ["soil", "buried", "alkaline",
                                              "acidic", "seawater",
                                              "aerated"]}}},
    "combustion_flow": {
        "description": "Combustion with porous flow — heat of combustion & flame "
                       "temperature, soot, Darcy flow, Kozeny-Carman "
                       "permeability.",
        "triggers": ["combustion", "flame temperature", "soot",
                     "darcy flow", "porous flow", "kozeny-carman",
                     "adiabatic flame"],
        "examples": ["adiabatic flame temperature of burning carbon",
                     "flame temperature of a smouldering column"],
        "slots": {}},
    "fuel_ignition": {
        "description": "Combustion-ignition properties of a fuel — adiabatic "
                       "flame temperature, autoignition, flash point, ignition "
                       "delay, flammability.",
        "triggers": ["ignition", "flash point", "autoignition",
                     "flammable", "flammability", "ignition delay",
                     "catch fire"],
        "examples": ["flash point of a fuel", "autoignition temperature of methane"],
        "slots": {}},
    # ── nuclear & atomic ──
    "nuclear_decay": {
        "description": "Nuclear & radioactive properties — alpha/beta half-lives "
                       "& decay constants, Q-values, activity, plus the parent "
                       "element's structure.",
        "triggers": ["radioactive", "half-life", "half life", "decay constant",
                     "alpha decay", "beta decay", "isotope", "becquerel",
                     "geiger-nuttall", "activity of", "decay", "radon", "radium"],
        "examples": ["half-life of uranium-238", "alpha decay Q-value of U238"],
        "slots": _ELEM},
    "nucleon_masses": {
        "description": "Nucleon & nuclear masses — proton and neutron rest "
                       "masses, the QCD scale, QCD/Higgs mass fractions, nuclear "
                       "binding.",
        "triggers": ["proton mass", "neutron mass", "nucleon", "qcd scale",
                     "nuclear binding energy", "mass of a proton",
                     "mass of a neutron", "rest energy of", "rest mass"],
        "examples": ["what is the mass of a proton",
                     "nuclear binding energy of helium-4"],
        "slots": {}},
    "stellar_fusion": {
        "description": "Stellar nucleosynthesis — proton-proton chain & CNO "
                       "cycle energy rates, crossover temperature, Gamow peak.",
        "triggers": ["nucleosynthesis", "stellar fusion", "proton-proton chain",
                     "pp chain", "cno cycle", "gamow", "fusion in the sun",
                     "stellar burning", "sun produce energy", "powers the sun",
                     "fusion in stars"],
        "examples": ["energy rate of the proton-proton chain in the sun",
                     "CNO cycle energy generation"],
        "slots": {}},
    "hydrogen_spectrum": {
        "description": "Hydrogen-like atomic spectra — energy levels, Balmer / "
                       "Lyman / Paschen series, transitions, fine structure.",
        "triggers": ["hydrogen spectrum", "hydrogen atom", "balmer", "lyman",
                     "paschen", "rydberg", "spectral lines",
                     "energy levels of hydrogen", "atomic spectrum",
                     "emission spectrum", "hydrogen-alpha", "hydrogen alpha"],
        "examples": ["Balmer series of hydrogen",
                     "energy levels of a hydrogen atom"],
        "slots": _ELEM},
    "atomic_coupling": {
        "description": "Angular-momentum coupling in atoms — term symbols, "
                       "Hund's-rule ground state, allowed J, Clebsch-Gordan, "
                       "spin-orbit.",
        "triggers": ["term symbol", "spin-orbit", "spin orbit",
                     "clebsch-gordan", "hund's rule", "hunds rule",
                     "lande g", "russell-saunders"],
        "examples": ["term symbols of a carbon atom",
                     "spin-orbit splitting of an energy level"],
        "slots": {}},
    # ── quantum ──
    "matter_wave": {
        "description": "Quantum matter waves — de Broglie wavelength, double-slit "
                       "interference, diffraction, which-path bounds.",
        "triggers": ["de broglie", "matter wave", "double slit",
                     "double-slit", "wavelength of an electron",
                     "electron diffraction", "interference fringes"],
        "examples": ["de Broglie wavelength of a 100 eV electron",
                     "double-slit interference of electrons"],
        "slots": {}},
    "quantum_dot": {
        "description": "Quantum confinement — particle-in-a-box levels, the Brus "
                       "quantum-dot gap, confinement energy, density of states.",
        "triggers": ["quantum dot", "particle in a box",
                     "particle-in-a-box", "quantum well", "confinement energy",
                     "brus", "nanocrystal"],
        "examples": ["energy levels of a quantum dot",
                     "particle in a box ground state energy"],
        "slots": {}},
    "quantum_tunneling": {
        "description": "Quantum tunneling — barrier transmission & WKB, decay "
                       "constant, resonant levels, field emission, STM, Gamow.",
        "triggers": ["tunneling", "tunnelling", "tunnel through", "barrier "
                     "transmission", "wkb", "field emission", "scanning "
                     "tunneling", "tunnel diode"],
        "examples": ["tunneling probability through a barrier",
                     "WKB transmission through a potential barrier"],
        "slots": {}},
    "quantum_gates": {
        "description": "Quantum computing — build a multi-qubit circuit, apply "
                       "gates (Hadamard, CNOT, Toffoli…), measure probabilities "
                       "and entanglement.",
        "triggers": ["qubit", "quantum gate", "bell state", "cnot", "hadamard",
                     "quantum circuit", "entanglement entropy", "quantum "
                     "computer"],
        "examples": ["build a Bell state", "apply a Hadamard then a CNOT gate"],
        "slots": {}},
    "quantum_algorithms_demo": {
        "description": "Quantum algorithms — Deutsch-Jozsa, Bernstein-Vazirani, "
                       "Grover, QFT, phase estimation, Shor, VQE, Ising/"
                       "Heisenberg ground states.",
        "triggers": ["quantum algorithm", "grover", "shor", "quantum fourier",
                     "deutsch-jozsa", "bernstein-vazirani", "vqe",
                     "phase estimation"],
        "examples": ["run Grover's search algorithm",
                     "Shor's algorithm to factor 15"],
        "slots": {}},
    "condensed_matter": {
        "description": "Strongly-correlated electron matter — Fermi energy, "
                       "Hubbard parameters & ground state, Mott metal-insulator "
                       "transition.",
        "triggers": ["fermi energy", "hubbard", "mott", "correlated electron",
                     "metal-insulator", "metal insulator transition"],
        "examples": ["Fermi energy of a metal", "Mott metal-insulator transition"],
        # The material IS the answer (same reasoning as acoustics) — with none
        # named the verb would silently default to iron/steel_mild. Require a
        # named material (or class word: "metal", ...).
        "material_required": True,
        "slots": _MAT},
    # ── other domains ──
    "plasma_physics": {
        "description": "Plasma parameters — plasma & cyclotron frequencies, "
                       "Debye length & number, Alfvén speed, Coulomb logarithm, "
                       "plasma beta.",
        "triggers": ["plasma", "debye length", "plasma frequency", "alfven",
                     "alfvén", "coulomb logarithm", "plasma beta", "gyroradius",
                     "larmor radius", "cyclotron radius"],
        "examples": ["plasma frequency of a 1e19 density plasma",
                     "Debye length of a plasma at 10 eV"],
        "slots": {}},
    "black_hole": {
        "description": "Black-hole thermodynamics — Schwarzschild radius, Hawking "
                       "temperature / luminosity / lifetime, entropy, ISCO, "
                       "photon sphere, tidal force.",
        "triggers": ["black hole", "hawking", "schwarzschild", "event horizon",
                     "photon sphere", "isco", "bekenstein", "tidal force near"],
        "examples": ["Hawking temperature of a 10 solar-mass black hole",
                     "Schwarzschild radius of a black hole"],
        "slots": {"solar_masses": {"unit": "Msun", "default": 10.0,
                                   "aliases": ["solar mass", "solar masses",
                                               "sun masses"]}}},
    "superconductor": {
        "description": "Superconductivity — BCS gap & coherence length, "
                       "Ginzburg-Landau parameter & type, London penetration, "
                       "critical fields, McMillan Tc.",
        "triggers": ["superconductor", "superconducting", "bcs gap",
                     "critical field", "london penetration", "cooper pair",
                     "meissner", "type ii", "type-ii", "mcmillan"],
        "examples": ["BCS gap of niobium",
                     "upper critical field of a superconductor"],
        "slots": {"sc_key": {"unit": "material", "default": "niobium",
                             "aliases": ["of", "for"]}}},
    "atmospheric_profile": {
        "description": "Standard-atmosphere state & transport — density / "
                       "pressure vs altitude, humidity & dew point, specific "
                       "heats.",
        "triggers": ["atmosphere", "air density", "dew point", "humidity",
                     "standard atmosphere", "air pressure", "at altitude",
                     "on the summit", "on top of everest"],
        "examples": ["air density at 5000 m altitude", "dew point at 50% humidity"],
        # An ambient report — it must NOT hijack a falling-object sim that merely
        # mentions the setting ("drop a ball ... in standard atmosphere"). The
        # translator skips it when the query is a drop scenario.
        "ambient_report": True,
        "slots": {}},
    "water_state": {
        "description": "Anomalous properties of liquid water vs temperature — "
                       "density (4 °C max), viscosity, surface tension, heat "
                       "capacity, ice fraction.",
        "triggers": ["density of water", "properties of water", "water viscosity",
                     "viscosity of water", "boiling point of water",
                     "water density", "water heat capacity"],
        "examples": ["density of water at 20 C", "viscosity of water vs temperature"],
        "slots": {}},
    "subsurface_scattering": {
        "description": "Subsurface light transport in translucent materials — "
                       "absorption & scattering lengths, diffusion length, "
                       "diffuse reflectance, BSSRDF.",
        "triggers": ["subsurface scattering", "translucent", "skin scattering",
                     "diffuse reflectance", "bssrdf", "light through skin"],
        "examples": ["subsurface scattering of skin",
                     "subsurface scattering in marble"],
        "slots": {"material_key": {"unit": "material",
                                   "default": "skin_caucasian",
                                   "aliases": ["of", "for", "in"]}}},
    "organic_material": {
        "description": "Organic & biological materials — alkane boiling points "
                       "and combustion enthalpy, anisotropic bone and wood "
                       "mechanics.",
        "triggers": ["alkane", "hydrocarbon", "bone", "wood strength",
                     "boiling point of octane", "combustion enthalpy of"],
        "examples": ["boiling point of octane", "mechanical strength of bone"],
        "slots": {}},
    "asteroid_shape": {
        "description": "Shape & surface gravity of a small body — triaxial "
                       "ellipsoid volume & mean radius, axis ratios, surface "
                       "gravity, escape velocity.",
        "triggers": ["asteroid", "ellipsoid volume", "small body",
                     "axis ratio", "surface gravity of an asteroid"],
        "examples": ["surface gravity of an asteroid",
                     "mean radius of a triaxial asteroid"],
        "slots": {}},
    # ── Phase-2 ledger drains (questions the system used to refuse) ──
    "planetary_surface": {
        "description": "Surface conditions of a solar-system body: surface "
                       "gravity, escape velocity, mass and radius.",
        "triggers": ["surface gravity", "gravity on", "gravity of mars",
                     "gravity on the moon", "escape velocity from",
                     "how strong is gravity"],
        "examples": ["what is the surface gravity of Mars",
                     "escape velocity from the Moon"],
        "slots": {"body": {"unit": "body", "default": "earth",
                           "aliases": ["of", "on", "from"]}}},
    "mass_energy": {
        "description": "Mass-energy equivalence E=mc² — rest energy of a mass, "
                       "or the energy released by converting mass.",
        "triggers": ["e=mc", "e = mc", "mass-energy", "mass energy",
                     "rest energy", "converted to energy", "mass to energy",
                     "energy from converting", "annihilat", "kg of mass"],
        "examples": ["how much energy is in 1 kg of mass",
                     "how much mass is converted to energy in a 1-megaton bomb"],
        "slots": {"mass_kg": {"unit": "kg", "default": 1.0,
                              "aliases": ["kg", "kilogram", "mass"]},
                  "energy_j": {"unit": "J", "default": None,
                               "aliases": ["megaton", "kiloton", "joule"]}}},
    "drop_object": {
        "description": "Drop a NAMED real object (anvil, piano, frying pan…) — "
                       "asks the Deckard shape researcher for its real "
                       "mass/shape, then simulates the fall.",
        "triggers": [],          # custom-routed in the translator (named object)
        "examples": ["drop a steel anvil from 10 km",
                     "how fast does a piano hit the ground"],
        "slots": {"object_name": {"unit": "object", "default": None,
                                  "aliases": ["drop a", "drop an", "drop the"]},
                  "drop_altitude_m": {"unit": "m", "default": 1000.0}}},
    "actuate_hand_tool": {
        "description": "Open and close a two-jaw hand tool (pliers, tongs, "
                       "scissors, tweezers, nutcracker…) — the pivot is "
                       "DISCOVERED from real dodoxel contact geometry "
                       "(deckard.hand_tools + infer_dodoxel_joints), not "
                       "declared, then driven by a SOLVED RevoluteJoint.",
        "triggers": [],          # custom-routed in the translator (named tool)
        "examples": ["simulate a pair of pliers being opened and closed",
                     "open and close a pair of tongs over time"],
        "exclusive": True,
        "answers": ["simulate a pair of pliers being opened and closed",
                    "open and close a pair of tongs over time"],
        "outputs": ["pivot_type", "total_mass_kg"],
        "slots": {"tool_name": {"unit": "object", "default": None,
                                "aliases": ["open and close", "actuate"]},
                  "n_cycles": {"unit": "count", "default": 3.0}}},
    "strike_bell": {
        "description": "A hanging bell struck by a thrown stone: ring "
                       "frequency from field.interface.acoustics."
                       "ring_frequency, propagated through earth "
                       "atmosphere at the real speed of sound.",
        "triggers": [],          # custom-routed in the translator
        "examples": ["simulate a hanging bell being struck by a stone",
                     "strike a bronze bell with a rock, what does it sound "
                     "like"],
        "exclusive": True,
        "answers": ["simulate a hanging bell being struck by a stone"],
        "outputs": ["ring_frequency_hz", "arrival_time_s"],
        "slots": {"bell_material": {"unit": "material", "default": "iron"},
                  "bell_diameter_m": {"unit": "m", "default": 0.15},
                  "stone_mass_kg": {"unit": "kg", "default": 0.2},
                  "observer_distance_m": {"unit": "m", "default": 50.0}}},
}
for _v, _e in _ROUTABLE_NEW.items():
    _e["exclusive"] = True
    _e.setdefault("answers", _e["examples"])
    _e.setdefault("outputs", ["functions_called"])
    VERB_MANIFEST[_v] = _e

# Verbs that SIMULATE MOTION. When a question is an ACTION (fired/thrown/
# dropped/jumps...), the translator lets these outrank property-report verbs
# in the exclusive-trigger contest — "fire a slug at Mach 2" must not route
# to acoustics just because "speed of sound" is a longer trigger string.
for _v in ("terminal_velocity_drop", "drag_heating_drop",
           "high_altitude_descent", "supersonic_projectile",
           "vertical_launch", "drop_object", "projectile_motion"):
    if _v in VERB_MANIFEST:
        VERB_MANIFEST[_v]["motion"] = True

# Example questions for the ORIGINAL verbs too, so the corpus covers all of them.
_ORIG_EXAMPLES = {
    "terminal_velocity_drop": ["how fast does a 5 cm steel ball hit the ground "
                               "from 10 km", "impact speed of a dropped cannonball"],
    "drag_heating_drop": ["does an iron sphere heat up falling from 30 km",
                          "how hot does a meteorite get from drag"],
    "high_altitude_descent": ["does a skydiver go supersonic jumping from the "
                              "stratosphere", "parachute descent from 35 km"],
    "supersonic_projectile": ["how does a bullet's drag change going supersonic",
                              "a projectile at mach 2.5"],
    "vertical_launch": ["throw a steel ball straight up at 300 m/s how high",
                        "launch a marble upward how high does it go"],
    "orbital_velocity": ["orbital velocity at 400 km above Earth",
                         "how fast does the Moon orbit"],
    "material_profile": ["properties of copper", "young's modulus of titanium"],
    "structural_response": ["fracture toughness of steel", "fatigue life of aluminum"],
    "thermal_response": ["thermal expansion of aluminum", "melting point of copper"],
    "contact_conduction": ["put a hot iron cube on a cold copper slab",
                           "how fast does heat flow between iron and copper"],
    "rotational_dynamics": ["moment of inertia of a solid sphere",
                            "angular momentum of a spinning flywheel"],
    "material_full_profile": ["all properties of steel", "full profile of copper"],
    "quantum_report": ["crystal field splitting of an iron ion",
                       "crystal-field stabilization energy"],
}
for _v, _ex in _ORIG_EXAMPLES.items():
    VERB_MANIFEST[_v]["examples"] = _ex


def manifest_for_prompt(with_examples: bool = True) -> str:
    """Render the manifest as a compact verb list for the LLM system prompt.
    With `with_examples`, append one example question per verb — the few-shot
    anchor that makes the LLM's verb match reliable across the full catalogue.
    """
    lines = []
    for verb, m in VERB_MANIFEST.items():
        slots = ", ".join(f"{s}" for s in m["slots"])
        line = f"- {verb}({slots}): {m['description']}"
        ex = m.get("examples")
        if with_examples and ex:
            line += f'   e.g. "{ex[0]}"'
        lines.append(line)
    return "\n".join(lines)
