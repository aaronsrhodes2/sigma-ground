"""record_clock's dial material -- the concrete case that triggered the
Choice doctrine (2026-07-16): a material picked without checking the
materials table for a better fit. Gates that the fix is real, not cosmetic.
"""
from sigma_ground.radiance.trajectory import record_clock


def test_dial_material_choice_is_recorded_and_checked_the_real_table():
    out = record_clock(t_max=0.5)
    scene = out["scene"]
    assert "choices" in scene
    # record_clock's full Choice retrofit covers more than the dial now
    # (materials, geometry estimates, spring constants, layout, hands) --
    # this test only pins the dial entry specifically, by description
    dial_choices = [c for c in scene["choices"]
                    if c["description"] == "clock dial material"]
    assert len(dial_choices) == 1
    c = dial_choices[0]
    assert c["value"] == "ceramic_alumina"
    assert "MATERIALS" in c["checked"]
    assert "aluminum" in c["alternatives"]           # what was passed over
    assert c["adjustable"]
    assert "CHOSEN" in scene["source"]
    assert "ceramic_alumina" in scene["source"]
    # the material must actually be baked and used by a leaf, not just cited
    assert "ceramic_alumina" in scene["materials"]
    dial_leaves = [l for l in scene["csg_leaves"]
                   if l.get("material") == "ceramic_alumina"]
    assert len(dial_leaves) == 1
    assert dial_leaves[0]["shape"]["type"] == "Cylinder"
