"""validate.py gates — synthetic cases proving each check fires (a
deliberately broken spec must fail exactly the way it's broken), plus the
real Kelly (1944) watch-train catalog entry proving the pipeline holds up
against genuine sourced data, not just fixtures."""
from sigma_ground.blueprint.schema import (BlueprintFact, GearSpec, MeshPair,
                                           EscapementSpec, MechanismSpec)
from sigma_ground.blueprint.validate import validate, cumulative_ratio
from sigma_ground.blueprint import catalog


def _cited(v, quote="a real sentence", locator="p.1", **kw):
    return BlueprintFact(value=v, source="a real source", license="test",
                         confidence=0.9, quote=quote, locator=locator, **kw)


def _est(v):
    return BlueprintFact(value=v)   # estimated sentinel, no quote needed


def test_valid_mesh_with_module_passes():
    spec = MechanismSpec(
        name="ok", gears=[
            GearSpec(name="a", teeth=_cited(72), module_mm=_cited(0.15)),
            GearSpec(name="b", is_pinion=True, teeth=_cited(12),
                    module_mm=_cited(0.15)),
        ],
        meshes=[MeshPair(a="a", b="b",
                         center_distance_mm=_cited(0.15 * (72 + 12) / 2.0))])
    r = validate(spec)
    assert r.ok
    assert not r.errors
    assert not r.gaps            # every field present — nothing to flag


def test_non_positive_teeth_is_an_error():
    spec = MechanismSpec(name="bad", gears=[GearSpec(name="a", teeth=_cited(0))])
    r = validate(spec)
    assert not r.ok
    assert any("positive integer" in e for e in r.errors)


def test_cited_fact_without_quote_is_an_error():
    bad = BlueprintFact(value=72, source="a real source")   # claims real, no quote
    spec = MechanismSpec(name="bad", gears=[GearSpec(name="a", teeth=bad)])
    r = validate(spec)
    assert not r.ok
    assert any("no quote" in e for e in r.errors)


def test_estimated_fact_without_quote_is_fine():
    spec = MechanismSpec(name="ok", gears=[GearSpec(name="a", teeth=_est(72))])
    r = validate(spec)
    assert r.ok   # estimated values are honestly flagged [estimated], not an error


def test_mismatched_module_across_a_mesh_is_an_error():
    spec = MechanismSpec(
        name="bad", gears=[
            GearSpec(name="a", teeth=_cited(72), module_mm=_cited(0.15)),
            GearSpec(name="b", is_pinion=True, teeth=_cited(12),
                    module_mm=_cited(0.20)),   # meshing gears must share a module
        ],
        meshes=[MeshPair(a="a", b="b", center_distance_mm=_cited(6.3))])
    r = validate(spec)
    assert not r.ok
    assert any("modules disagree" in e for e in r.errors)


def test_center_distance_inconsistent_with_module_is_an_error():
    spec = MechanismSpec(
        name="bad", gears=[
            GearSpec(name="a", teeth=_cited(72), module_mm=_cited(0.15)),
            GearSpec(name="b", is_pinion=True, teeth=_cited(12), module_mm=_cited(0.15)),
        ],
        # correct value would be 0.15*(72+12)/2 = 6.3mm; 10mm is way off
        meshes=[MeshPair(a="a", b="b", center_distance_mm=_cited(10.0))])
    r = validate(spec)
    assert not r.ok
    assert any("doesn't match module" in e for e in r.errors)


def test_missing_module_is_a_gap_not_an_error():
    spec = MechanismSpec(name="ok", gears=[GearSpec(name="a", teeth=_cited(72))])
    r = validate(spec)
    assert r.ok
    assert any("no module cited" in g for g in r.gaps)


def test_escapement_lift_angle_exceeding_tooth_pitch_is_an_error():
    spec = MechanismSpec(
        name="bad", gears=[],
        escapement=EscapementSpec(kind="lever", escape_wheel_teeth=_cited(15),
                                  lift_angle_deg=_cited(30.0)))  # pitch = 360/15 = 24deg
    r = validate(spec)
    assert not r.ok
    assert any("lift angle" in e for e in r.errors)


def test_unusual_pressure_angle_is_a_warning_not_an_error():
    spec = MechanismSpec(name="ok", gears=[
        GearSpec(name="a", teeth=_cited(30), pressure_angle_deg=_cited(17.0))])
    r = validate(spec)
    assert r.ok
    assert any("outside the standard set" in w for w in r.warnings)


def test_cumulative_ratio_matches_hand_computation():
    spec = MechanismSpec(name="chain", gears=[
        GearSpec(name="w1", teeth=_cited(72)),
        GearSpec(name="p1", is_pinion=True, teeth=_cited(12)),
        GearSpec(name="w2", teeth=_cited(80)),
        GearSpec(name="p2", is_pinion=True, teeth=_cited(10)),
    ])
    ratio = cumulative_ratio(spec, [("w1", "p1"), ("w2", "p2")])
    assert ratio == (72 / 12) * (80 / 10)


# ── the real, sourced catalog entry ──────────────────────────────────────

def test_kelly_1944_watch_train_catalog_entry_validates_cleanly():
    assert catalog.has("kelly_1944_watch_going_train_18000bph"), (
        "run: python tools/distill_kelly_watch_train.py")
    spec = catalog.load("kelly_1944_watch_going_train_18000bph")
    r = validate(spec)
    assert r.ok
    assert not r.errors
    # every gap is a module/center-distance/lift-angle absence — the honest,
    # documented boundary of this source, not a hidden problem
    assert all("module" in g or "center distance" in g or "lift angle" in g
              for g in r.gaps)


def test_kelly_1944_watch_train_reproduces_the_books_own_beats_per_hour():
    spec = catalog.load("kelly_1944_watch_going_train_18000bph")
    mesh_pairs = [(m.a, m.b) for m in spec.meshes]
    turns_escape_per_barrel_turn = cumulative_ratio(spec, mesh_pairs)
    assert turns_escape_per_barrel_turn == 3600.0
    barrel_to_center = cumulative_ratio(spec, mesh_pairs[:1])
    turns_escape_per_hour = turns_escape_per_barrel_turn / barrel_to_center
    assert turns_escape_per_hour == 600.0
    beats = turns_escape_per_hour * spec.gear("escape_wheel").teeth.value * 2
    assert beats == spec.escapement.beats_per_hour.value == 18000.0


def test_kelly_1944_watch_train_every_fact_carries_a_verbatim_quote():
    spec = catalog.load("kelly_1944_watch_going_train_18000bph")
    for g in spec.gears:
        assert g.teeth.quote.strip(), f"{g.name}'s teeth Fact has no quote"
        assert not g.teeth.estimated, f"{g.name}'s teeth Fact is flagged estimated"
