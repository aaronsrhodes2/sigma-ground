"""Choice doctrine gates -- provenance for assistant-invented variables.

The doctrine (2026-07-16, triggered by a real miss: a clock dial material
was picked without checking the materials table for a better fit): any
value Mentat picks rather than looks up or derives must say so, out loud,
in the scene -- what was picked, what table was checked, what else was on
offer.
"""
from sigma_ground.dynamics.mechanisms.choice import Choice


def test_choice_records_description_value_and_defaults_adjustable():
    c = Choice("clock dial material", "aluminum")
    assert c.value == "aluminum"
    assert c.adjustable
    d = c.to_dict()
    assert d["description"] == "clock dial material"
    assert d["value"] == "aluminum"
    assert d["adjustable"] is True
    assert "CHOSEN" in c.cite()


def test_choice_carries_what_was_checked_and_the_alternatives():
    c = Choice("clock dial material", "ceramic_alumina",
              checked="sigma_ground.field.interface.surface.MATERIALS",
              alternatives=["aluminum", "iron", "copper", "bone", "glass"],
              reason="closest to a real dial's ivory/enamel look")
    d = c.to_dict()
    assert d["checked"] == "sigma_ground.field.interface.surface.MATERIALS"
    assert "aluminum" in d["alternatives"]
    cite = c.cite()
    assert "checked:" in cite
    assert "alternatives:" in cite
    assert "ceramic_alumina" in cite


def test_choice_can_be_marked_non_adjustable():
    c = Choice("a cosmetic render-only leaf thickness", 0.004,
              adjustable=False)
    assert not c.to_dict()["adjustable"]
