"""Provenance / anti-hallucination guards for Deckard.

These lock in the property the live demo proved: the LLM supplies only
dimensionless proportions, while every *physical* number — densities, mass,
centre of mass, the self-check — comes from our own data, geometry kernel, and
SDF integrator. If the pipeline ever regressed to trusting LLM-supplied physics
(or to a hard-coded/hallucinated answer), these fail.

Deterministic: the LLM is stubbed with a payload that carries material NAMES but
no densities — exactly the contract the real prompt enforces — so nothing here
touches the network.
"""
import json
import math

from sigma_ground.deckard import compile
from sigma_ground.deckard.schema import Fact
from sigma_ground.deckard.researcher import research_spec
from sigma_ground.deckard.sources import density_of

# A vessel the model might propose: geometry + material NAMES only, NO densities.
_LLM_PAYLOAD = {
    "kind": "layered_vessel",
    "geometry": {"outer_radius_m": 0.035, "height_m": 0.12, "wall_m": 0.003,
                 "glaze_m": 0.0, "base_m": 0.005, "fill_fraction": 0.8},
    "layers": [{"name": "glaze", "material": "glass"},
               {"name": "ceramic", "material": "glass"},
               {"name": "water", "material": "liquid water"}],
    "notes": "stubbed drinking glass",
}


def _spec():
    return research_spec("drinking glass", ask=lambda n: json.dumps(_LLM_PAYLOAD),
                         model="stub")


def test_llm_payload_carries_no_physics():
    # The contract: the model proposes proportions + material names, never physics.
    assert "density" not in json.dumps(_LLM_PAYLOAD).lower()


def test_densities_come_from_our_data_not_the_llm():
    spec = _spec()
    for L in spec.layers:
        grounded = density_of(L.material)
        assert grounded is not None, L.material
        # the spec's density IS our file's value, cited to our files — not estimated
        assert L.density.value == grounded.value
        assert not L.density.estimated
        assert ("materials.json" in L.density.source
                or "surface.MATERIALS" in L.density.source)


def test_llm_dimensions_are_flagged_estimated():
    # LLM-supplied geometry must never masquerade as measured.
    assert all(f.estimated for f in _spec().geometry.values())


def test_compiled_mass_equals_independent_hand_math():
    spec = _spec()
    g = {k: spec.geometry[k].value for k in spec.geometry}
    R_cer = g["outer_radius_m"] - g["glaze_m"]
    R_in = R_cer - g["wall_m"]
    h_fill = g["fill_fraction"] * (g["height_m"] - g["base_m"])
    rho_body = density_of(next(L.material for L in spec.layers if L.name == "ceramic")).value
    rho_water = density_of("liquid water").value
    v_body = (math.pi * R_cer**2 * g["height_m"]
              - math.pi * R_in**2 * (g["height_m"] - g["base_m"]))
    v_water = math.pi * R_in**2 * h_fill
    mass_hand = v_body * rho_body + v_water * rho_water
    cup = compile(spec, resolution=56)
    # mass = (real geometry formula) x (our-data densities), not an LLM number
    assert abs(cup.mass_kg - mass_hand) < 1e-6


def test_independent_sdf_integrator_agrees_but_is_not_a_copy():
    v = compile(_spec(), resolution=56).validation
    assert v["passed"], v["note"]
    # the grid integrator sampled material_at and matched the analytic mass...
    assert v["mass_residual"] < 0.05
    # ...via a different method, so a small non-zero discretisation gap remains
    # (a faked/copied value would be bit-identical, residual exactly 0).
    assert 0.0 < v["mass_residual"]


def test_construct_uses_the_real_geometry_kernel():
    cup = compile(_spec(), resolution=48)
    assert type(cup.composed).__module__ == "sigma_ground.kernel.csg"
    assert type(cup.composed).__name__ == "ComposedSDF"
    # the SDF is genuinely queryable geometry, not a stored scalar
    assert cup.material_at(0.0, 0.0, 0.05) in {"water", "ceramic", "air", "glaze"}


def test_grounding_matches_whole_words_not_substrings():
    # A name that merely *contains* a material's letters must not mis-ground:
    # "keratin" must never resolve to "tin" (7310 kg/m³), "hair" never to "air".
    # A wrong cited density is worse than an honest estimate.
    for word, sub in [("keratin", "tin"), ("hair", "air"), ("marigold", "gold")]:
        sub_fact = density_of(sub, allow_web=False)
        assert sub_fact is not None, sub                 # the short material IS in our data
        got = density_of(word, allow_web=False)
        assert got is None or got.value != sub_fact.value, (word, sub, got)
    # ...but real whole-word containment still grounds a verbose phrase to its core
    gold = density_of("gold", allow_web=False)
    phrase = density_of("gold ring band", allow_web=False)
    assert phrase is not None and phrase.value == gold.value


def test_mass_is_recomputed_from_inputs_not_memorized():
    base = compile(_spec(), resolution=48)
    spec2 = _spec()
    spec2.geometry["outer_radius_m"] = Fact(
        spec2.geometry["outer_radius_m"].value * 1.5, "perturb", "", 0.5)
    bigger = compile(spec2, resolution=48)
    assert bigger.mass_kg > base.mass_kg * 1.3   # grew through real geometry
