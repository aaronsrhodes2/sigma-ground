"""Blueprint schema round-trip — mirrors deckard's ConstructSpec markdown
gate, extended for BlueprintFact's mandatory quote/locator."""
from sigma_ground.blueprint.schema import (BlueprintFact, GearSpec, MeshPair,
                                           SpringSpec, EscapementSpec,
                                           MechanismSpec, emit_markdown,
                                           parse_markdown)


def _fact(v, **kw):
    return BlueprintFact(value=v, source="test source", license="test",
                         confidence=0.9, quote="a quoted sentence",
                         locator="p.1", **kw)


def _sample_spec():
    return MechanismSpec(
        name="test_mechanism",
        gears=[
            GearSpec(name="wheel_a", teeth=_fact(72),
                    module_mm=_fact(0.15), tooth_form=_fact("involute")),
            GearSpec(name="pinion_b", is_pinion=True, teeth=_fact(12),
                    module_mm=_fact(0.15)),
        ],
        meshes=[MeshPair(a="wheel_a", b="pinion_b",
                         center_distance_mm=_fact(6.3))],
        # lists, not tuples: JSON has no tuple type, so a tuple value would
        # come back as a list after the markdown round-trip anyway — this
        # mirrors deckard.Fact's own value semantics, not a schema bug
        spring=SpringSpec(torque_curve=[_fact([0, 1.2]), _fact([5, 0.8])]),
        escapement=EscapementSpec(kind="lever", escape_wheel_teeth=_fact(15),
                                  beats_per_hour=_fact(18000)),
        sources=[{"name": "test source", "license": "test", "url": "http://x",
                 "locator": "p.1"}],
        notes="a test mechanism",
    )


def test_blueprint_fact_round_trips_through_dict():
    f = _fact(42)
    f2 = BlueprintFact.from_dict(f.to_dict())
    assert f2 == f


def test_blueprint_fact_estimated_sentinel():
    f = BlueprintFact(value=1.0)
    assert f.estimated
    assert "[estimated]" in f.cite()
    cited = _fact(1.0)
    assert not cited.estimated
    assert "[estimated]" not in cited.cite()
    assert "p.1" in cited.cite()


def test_mechanism_spec_round_trips_through_payload():
    spec = _sample_spec()
    spec2 = MechanismSpec.from_payload(spec.to_payload())
    assert spec2.to_payload() == spec.to_payload()


def test_mechanism_spec_round_trips_through_markdown():
    spec = _sample_spec()
    md = emit_markdown(spec)
    spec2 = parse_markdown(md)
    assert spec2.to_payload() == spec.to_payload()
    # the human-facing prose must carry the citation quote itself, not just
    # the machine-readable json block — a reviewer skimming the markdown
    # should be able to verify a number without opening the payload
    assert "a quoted sentence" in md
    assert md.count("a quoted sentence") >= 2   # appears in prose AND json


def test_markdown_prose_shows_quote_for_every_cited_fact():
    spec = _sample_spec()
    md = emit_markdown(spec)
    # every non-estimated Fact's quote text appears somewhere before the
    # canonical payload fence starts (i.e. in the human-readable section)
    prose, _, _ = md.partition("## Canonical payload")
    assert "a quoted sentence" in prose
