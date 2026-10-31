"""Degeneracy probe: derived material properties must actually vary.

Born from a real incident (2026-07-30): field/interface/friction.py's
friction_coefficient reads as a well-structured, correctly-cited Bowden-Tabor
implementation -- and returns exactly 1.000 for every material pair that can
be formed. Its own docstring even works through why the terms cancel. Code
review cannot catch that class of defect: every line is defensible, the math
is right, and the model is simply the wrong regime (clean metals in vacuum)
for everyday matter. The only instrument that catches it is EXECUTION across
the vocabulary: a "derived" property that returns one constant for
everything is not deriving anything.

The probe evaluates each single-material property over the whole
surface.MATERIALS vocabulary and counts distinct values (3 significant
figures). The KNOWN_DEGENERATE table is a two-way ratchet:
  - a function NOT listed must produce >= MIN_DISTINCT distinct values --
    a new degeneracy fails the build instead of shipping silently;
  - a function listed must STILL be degenerate as documented -- fixing one
    forces removing its entry, so the table never rots into folklore.
"""
import math

import pytest

from sigma_ground.field.interface.surface import MATERIALS

MIN_EVALS = 5          # below this the vocabulary sample is too thin to judge
MIN_DISTINCT = 3       # a real material property should separate at least this much


def _sig3(x):
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + 2)


def _distinct(fn):
    vals = []
    for m in sorted(MATERIALS):
        try:
            vals.append(float(fn(m)))
        except Exception:
            continue
    return len(vals), len({_sig3(v) for v in vals})


def _probes():
    from sigma_ground.field.interface import (mechanical, plasticity, stress,
                                              hardness, thermal, texture,
                                              acoustics, impact, friction,
                                              surface)
    return {
        "mechanical.youngs_modulus": mechanical.youngs_modulus,
        "mechanical.shear_modulus": mechanical.shear_modulus,
        "mechanical.bulk_modulus": mechanical.bulk_modulus,
        "mechanical.theoretical_shear_strength": mechanical.theoretical_shear_strength,
        "plasticity.yield_stress": plasticity.yield_stress,
        "stress.fracture_toughness": stress.fracture_toughness,
        "hardness.vickers_hardness": hardness.vickers_hardness,
        "thermal.thermal_conductivity": thermal.thermal_conductivity,
        "thermal.specific_heat_j_kg_K": thermal.specific_heat_j_kg_K,
        "thermal.emissivity": thermal.emissivity,
        "texture.microfacet_roughness": texture.microfacet_roughness,
        "texture.specular_fraction": texture.specular_fraction,
        "acoustics.longitudinal_wave_speed": acoustics.longitudinal_wave_speed,
        "surface.surface_energy": lambda m: surface.surface_energy_at_sigma(m, 0.0),
        "impact.coefficient_of_restitution":
            lambda m: impact.coefficient_of_restitution(m, velocity=1.0, radius_m=0.02),
        "friction.friction_coefficient_self":
            lambda m: friction.friction_coefficient(m, m),
    }


# name -> (documented distinct count, why it is degenerate / near-degenerate).
# An entry here is a DISCLOSED defect or limitation, not an endorsement.
KNOWN_DEGENERATE = {
    # Bowden-Tabor adhesive model, CLEAN METALS IN VACUUM: the terms cancel to
    # ~1.0 for every pair (its own docstring derives why). Wrong regime for
    # everyday matter (real engineering mu spans ~0.03-1.0). This is why
    # materia.engine.make_material_friction keeps derived=False by default.
    "friction.friction_coefficient_self": (1, "vacuum-regime constant ~1.0"),
    # Rayleigh criterion goes binary at optical wavelengths: crystalline
    # surfaces (sub-nm thermal roughness) -> 1.0, amorphous class constant
    # (0.4 roughness) -> 0.15. The crystalline/amorphous split is real
    # physics; the two-bucket collapse is the honest resolution limit of the
    # model, disclosed in _emergent_matter's roughness_origin.
    "texture.specular_fraction": (2, "crystalline/amorphous binary"),
}


@pytest.mark.parametrize("name", sorted(_probes()))
def test_derived_property_varies_across_the_vocabulary(name):
    fn = _probes()[name]
    n_evals, n_distinct = _distinct(fn)
    if n_evals < MIN_EVALS:
        pytest.skip(f"{name}: only {n_evals} materials evaluate — sample too thin")
    if name in KNOWN_DEGENERATE:
        documented, why = KNOWN_DEGENERATE[name]
        assert n_distinct == documented, (
            f"{name} was documented degenerate ({documented} distinct: {why}) "
            f"but now yields {n_distinct} — if it was fixed, remove its "
            f"KNOWN_DEGENERATE entry so the ratchet tightens")
    else:
        assert n_distinct >= MIN_DISTINCT, (
            f"{name} returns only {n_distinct} distinct value(s) across "
            f"{n_evals} materials — a 'derived' property that barely varies "
            f"is likely a wrong-regime model (see friction_coefficient's "
            f"story at the top of this file); document it in "
            f"KNOWN_DEGENERATE with a reason, or fix the derivation")
