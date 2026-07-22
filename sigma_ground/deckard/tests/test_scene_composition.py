"""Multi-object scene composition — sigma_ground.deckard.scene.

The Captain's target prompt needs THREE independently-researched objects
stacked with an explicit spatial relationship ("a doughnut ... sitting on a
plate, which is flat on top of a ... table"), not one Deckard shape. These
tests prove the stacking primitive (``stack_constructs``/``compose_scene``)
places each object's bottom flush on the previous object's top — reusing
construct.py's own whole-object anchor semantics — for both of Deckard's
construct kinds (a ``layered_vessel`` cup and a ``parts``-kit table), and
that the exact doughnut/plate/table chain compiles end-to-end.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

import pytest

from sigma_ground.deckard.scene import (
    SceneItem, stack_constructs, compose_scene, _object_anchor, _style_hint,
)
from sigma_ground.radiance.scene_export import scene_spec_to_sdf
from sigma_ground.dynamics.vec import Vec3

_EPS = 1e-6   # metres — anchor-mate tolerance (float roundtrip, not physical slop)


def _assert_flush(prev_bbox, this_bbox):
    """The next object's bottom anchor must equal the previous object's top
    anchor exactly (to float tolerance) in x, y, AND z — no gap, no overlap."""
    top = _object_anchor(prev_bbox, "top")
    bottom = _object_anchor(this_bbox, "bottom")
    for a, b, axis in zip(top, bottom, "xyz"):
        assert abs(a - b) < _EPS, f"{axis}: prev-top={a} != this-bottom={b}"


# ── 1. two-object stack: "a mug on a table" ─────────────────────────────────
# Deterministic/offline (allow_llm=False): "table" resolves through the real
# PartNet exemplar path, "mug" through the frozen layered_vessel catalog —
# the two DIFFERENT Construct kinds Deckard has, so this also proves stacking
# works uniformly across both (translation happens post-compile, on whichever
# shapes construct.py produced).
MUG_ON_TABLE = stack_constructs(["table", "mug"], allow_llm=False, resolution=48)


def test_mug_bottom_sits_flush_on_table_top_no_gap_no_overlap():
    table_bbox, mug_bbox = (o["bbox"] for o in MUG_ON_TABLE.validation["objects"])
    _assert_flush(table_bbox, mug_bbox)


def test_mug_on_table_merges_to_one_identified_construct():
    assert MUG_ON_TABLE.identified is True
    assert MUG_ON_TABLE.mass_kg > 0.0
    # every part/layer name is namespaced per object — no collisions even
    # though "table" and "mug"'s own internal part names could otherwise clash
    names = [L.name for L in MUG_ON_TABLE.layers]
    assert len(names) == len(set(names))
    assert any(n.startswith("table0__") for n in names)
    assert any(n.startswith("mug1__") for n in names)


def test_mug_on_table_bbox_is_the_union_of_both_objects():
    table_bbox, mug_bbox = (o["bbox"] for o in MUG_ON_TABLE.validation["objects"])
    (x0, x1), (y0, y1), (z0, z1) = MUG_ON_TABLE.bbox
    assert z0 == pytest.approx(table_bbox[2][0])
    assert z1 == pytest.approx(mug_bbox[2][1])


def test_mug_on_table_exports_to_one_renderable_scene():
    scene = compose_scene(["table", "mug"], allow_llm=False, resolution=48)
    assert scene is not None
    assert scene["identified"] is True
    assert len(scene["csg_leaves"]) > 1
    sdf, material_at = scene_spec_to_sdf(scene)
    assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)


def test_first_item_keeps_its_own_placement_unshifted():
    """The base of the stack (index 0) is never translated."""
    solo_table = stack_constructs(["table"], allow_llm=False, resolution=48)
    table_bbox = MUG_ON_TABLE.validation["objects"][0]["bbox"]
    assert table_bbox == solo_table.bbox


# ── 2. the exact Captain/Patrick chain: doughnut on plate on 1700s table ───
# "doughnut" and "plate" were both frozen into Deckard's catalog by earlier
# LLM-research runs this session (a_doughnut.md/doughnut.md/plate.md), so
# allow_llm=True resolves instantly here (a catalog hit, no network) — but
# the test still exercises the real allow_llm=True path end-to-end, matching
# how the chat route (ask_deckard/front_door) will actually call this.
def test_doughnut_plate_table_chain_compiles_without_error():
    construct = stack_constructs(
        [SceneItem("1700s table"), SceneItem("plate"), SceneItem("doughnut")],
        allow_llm=True, resolution=48)
    assert construct.identified is True
    assert construct.mass_kg > 0.0

    table_bbox, plate_bbox, doughnut_bbox = (
        o["bbox"] for o in construct.validation["objects"])
    _assert_flush(table_bbox, plate_bbox)
    _assert_flush(plate_bbox, doughnut_bbox)


def test_doughnut_plate_table_chain_exports_to_one_scene():
    scene = compose_scene(
        [SceneItem("1700s table"), SceneItem("plate"), SceneItem("doughnut")],
        allow_llm=True, resolution=48)
    assert scene is not None
    assert scene["identified"] is True
    assert len(scene["csg_leaves"]) >= 3     # >= one leaf per object, at least
    sdf, material_at = scene_spec_to_sdf(scene)
    assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)


# ── 3. the "1700s table" style convention ───────────────────────────────────
def test_style_word_maps_to_a_wood_material_hint():
    assert _style_hint("a 1700s table") == "oak"
    assert _style_hint("an antique chest") == "oak"
    assert _style_hint("a plain table") is None      # no style word -> no hint


def test_style_hint_only_applied_when_caller_gave_no_explicit_hint():
    """An explicit material_hint always wins over the style-word convention —
    the convention SELECTS a default, it never overrides a real choice."""
    hinted = stack_constructs(
        [SceneItem("1700s table", material_hint="steel")],
        allow_llm=False, resolution=48)
    assert any(L.material == "steel" for L in hinted.layers)
    assert not any(L.material == "wood_oak" for L in hinted.layers)


# ── 4. honest refusal — never a fabricated stand-in ─────────────────────────
def test_ungroundable_object_in_the_chain_refuses_the_whole_stack():
    with pytest.raises(ValueError):
        stack_constructs(
            ["table", "a nonexistent flibbertigibbet contraption"],
            allow_llm=False, resolution=48)


def test_compose_scene_returns_none_on_refusal_not_a_fake_render():
    scene = compose_scene(
        ["table", "a nonexistent flibbertigibbet contraption"],
        allow_llm=False, resolution=48)
    assert scene is None
