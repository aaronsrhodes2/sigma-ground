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
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
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
        "slots": {
            "material_key": {"unit": "material", "default": "steel_mild",
                             "aliases": ["of", "for"]},
        },
    },
}

# Each verb's named outputs — what a later step can read from it. Used by the
# planner's deterministic auto-binder to thread physical state forward.
_VERB_OUTPUTS = {
    "terminal_velocity_drop": ["impact_speed_m_s", "max_speed_m_s",
                               "terminal_velocity_m_s", "fall_time_s"],
    "drag_heating_drop": ["delta_T_K", "peak_T_K", "dissipation_J",
                          "impact_speed_m_s"],
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
                               "free fall", "free-fall", "freefall",
                               "from space"], True),
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
    "rotational_dynamics": (["moment of inertia", "angular momentum",
                             "rotational kinetic energy", "rotational inertia",
                             "flywheel inertia"], True),
    "material_full_profile": (["all properties of", "every property of",
                               "full profile of", "complete profile of",
                               "full material profile", "characterize everything"],
                              True),
}
for _v, (_t, _ex) in _VERB_TRIGGERS.items():
    VERB_MANIFEST[_v]["triggers"] = _t
    VERB_MANIFEST[_v]["exclusive"] = _ex


def manifest_for_prompt() -> str:
    """Render the manifest as a compact verb list for the LLM system prompt."""
    lines = []
    for verb, m in VERB_MANIFEST.items():
        slots = ", ".join(f"{s}" for s in m["slots"])
        lines.append(f"- {verb}({slots}): {m['description']}")
    return "\n".join(lines)
