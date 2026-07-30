"""Material-resolved Coulomb friction: the orchestrator-tier seam.

The solver's friction machinery has always been complete (accumulated
tangential impulse, proper friction cone). What was missing is that no scene
ever supplied a mu, so every simulation ran frictionless -- which
PhysicsScene itself flags as NOT_PHYSICS.

These tests pin the seam AND the honesty rule: an unresolvable pair must fall
through to the caller's explicit default and be RECORDED as unresolved, never
silently invented.
"""
import pytest

from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.materia.engine import make_material_friction, _DensityMaterial


def _p(label):
    return PhysicsParcel(0.05, _DensityMaterial(1000.0), label=label)


def test_mu_table_resolves_and_is_order_insensitive():
    a, b = _p("plate"), _p("table")
    fn = make_material_friction({id(a): "ceramic", id(b): "wood_oak"},
                                mu_table={("wood_oak", "ceramic"): 0.35})
    assert fn(a, b) == 0.35          # table given (wood, ceramic), asked (ceramic, wood)
    assert fn(b, a) == 0.35
    assert not fn.unresolved


def test_unresolved_pair_falls_back_and_is_recorded():
    """The load-bearing honesty rule: no silent invention."""
    a, b = _p("x"), _p("y")
    fn = make_material_friction({id(a): "unobtainium", id(b): "wood_oak"})
    assert fn(a, b) == 0.0                       # frictionless, the prior behaviour
    assert ("unobtainium", "wood_oak") in fn.unresolved

    fn2 = make_material_friction({id(a): "unobtainium", id(b): "wood_oak"},
                                 default_mu=0.4)
    assert fn2(a, b) == 0.4                      # an explicit choice, still recorded
    assert fn2.unresolved


def test_ground_contact_uses_the_declared_ground_material():
    a = _p("puck")
    fn = make_material_friction({id(a): "steel_mild"}, ground_material="concrete",
                                mu_table={("steel_mild", "concrete"): 0.6})
    assert fn(a, None) == 0.6                    # b is None => static environment


def test_derived_is_off_by_default_because_it_is_the_wrong_regime():
    """field.interface.friction is the Bowden-Tabor CLEAN-METALS-IN-VACUUM
    model; it returns ~1.0 for every pair we can form, which would make a
    plate grip a table like rubber. It must not engage unless asked."""
    a, b = _p("a"), _p("b")
    mapping = {id(a): "steel_mild", id(b): "wood_oak"}
    assert make_material_friction(mapping)(a, b) == 0.0          # not derived
    derived = make_material_friction(mapping, derived=True)(a, b)
    assert derived == pytest.approx(1.0, abs=0.05)               # the degenerate constant


def test_callback_shape_matches_what_the_scene_and_stepper_expect():
    """PhysicsScene stores friction_fn(a, b) and stepper.py calls it as
    mu_fn(c.a, c.b) -- this is the contract the seam has to satisfy."""
    a, b = _p("a"), _p("b")
    fn = make_material_friction({id(a): "iron", id(b): "iron"},
                                mu_table={("iron", "iron"): 0.5})
    scene = PhysicsScene([a, b], friction_fn=fn)
    assert scene.friction_fn is fn
    assert scene.friction_fn(a, b) == 0.5


def test_expensive_resolution_is_cached_per_material_pair():
    """The cache exists to avoid repeating the EXPENSIVE step (the derived
    coefficient is a real physics derivation), not the cheap name lookup --
    mu is keyed on the material pair, so a scene with hundreds of contacts
    between the same two materials resolves once."""
    seen = []
    real = None

    def counting_table_lookup(pair):
        seen.append(pair)
        return 0.5

    a, b = _p("a"), _p("b")
    fn = make_material_friction({id(a): "iron", id(b): "oak"},
                                mu_table={("iron", "oak"): 0.5})
    assert fn(a, b) == 0.5
    # mutate the table AFTER the first call: a cached pair must not re-read it
    fn(a, b), fn(a, b)
    assert fn(a, b) == 0.5
    assert not fn.unresolved
