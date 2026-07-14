"""Lane 1 (actuation epic): parts survive the pipeline.

Part identity through voxelize — REGION labels (one (part, material) pair per
label id), per-part mass/CoM/inertia from the same sums as the totals,
part-pair interface geometry (the A-graph substrate), the hierarchy reader,
fill-as-a-part, and the bake cache. numpy/trimesh/scipy are the opt-in
[shapes] deps; the chair test additionally needs the local PartNet data.
"""
import os

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from sigma_ground.deckard.voxelize import voxelize, construct_from_field

_PARTNET = r"D:/datasets/shapenet/PartNet/data_v0/44164"
_HAS_CHAIR = os.path.isdir(_PARTNET)


def _two_boxes():
    """Two 4 cm cubes touching on a face — the minimal two-part body."""
    a = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    a.apply_translation((0.02, 0.02, 0.02))            # x ∈ [0, 0.04]
    b = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    b.apply_translation((0.06, 0.02, 0.02))            # x ∈ [0.04, 0.08]
    return a, b


def test_part_masses_sum_exactly_and_coms_compose():
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    assert [p["name"] for p in f.parts] == ["left", "right"]
    total = sum(p["mass_kg"] for p in f.parts)
    assert total == pytest.approx(f.mass_kg, rel=1e-12)      # same sums, exact
    # mass-weighted part CoMs reproduce the body CoM
    com = np.zeros(3)
    for p in f.parts:
        com += p["mass_kg"] * np.asarray(p["com_m"])
    com /= f.mass_kg
    assert np.allclose(com, f.com_m, atol=1e-12)
    vol = sum(p["volume_m3"] for p in f.parts)
    assert vol == pytest.approx(f.volume_m3, rel=1e-12)


def test_part_id_round_trips_from_occupied_cells():
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    c = construct_from_field("pair", f)
    # sample the two parts' interiors: part_id_at must name them
    assert c.part_id_at(0.02, 0.02, 0.02) == 0               # left cube centre
    assert c.part_id_at(0.06, 0.02, 0.02) == 1               # right cube centre
    assert c.part_id_at(0.5, 0.5, 0.5) == -1                 # void
    assert c.material_at(0.02, 0.02, 0.02) == "iron"          # material view intact
    assert c.material_at(0.06, 0.02, 0.02) == "copper"
    assert [(p["name"], p["material"]) for p in c.parts] == \
        [("left", "iron"), ("right", "copper")]


def test_part_interface_geometry_names_the_shared_face():
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    assert f.part_interfaces and (0, 1) in f.part_interfaces
    g = f.part_interfaces[(0, 1)]
    # the shared face is the x=0.04 plane, a 4x4 cm patch
    assert g["centroid_m"][0] == pytest.approx(0.04, abs=0.004)
    assert g["centroid_m"][1] == pytest.approx(0.02, abs=0.004)
    assert abs(g["principal_dir"][0]) < 0.2                  # patch lies in the yz plane
    assert g["area_m2"] == pytest.approx(0.04 * 0.04, rel=0.35)  # stair-step tolerance
    # material-pair interfaces (the heat view) still report the same contact
    assert ("copper", "iron") in f.interfaces


def test_legacy_two_tuple_input_is_unchanged_and_flagged():
    a, b = _two_boxes()
    legacy = voxelize([(a, "iron"), (b, "iron")], pitch=0.004)
    assert legacy.materials == ["air", "iron"]               # one region per material
    assert [p["flags"] for p in legacy.parts] == [("unsegmented",)]
    # the same geometry as 3-tuples sharing one (part, material) region
    merged = voxelize([(a, "iron", "body"), (b, "iron", "body")], pitch=0.004)
    assert np.array_equal(legacy.label_grid, merged.label_grid)
    assert legacy.mass_kg == pytest.approx(merged.mass_kg, rel=1e-12)


def test_fill_becomes_its_own_flagged_part():
    """A hand-built open container (numpy label grid → _finalize): the fill
    must arrive as its OWN flagged part and the part ledger must stay exact."""
    from sigma_ground.deckard.voxelize import fill_cavity, _finalize
    N, t = 20, 3                                              # grid, wall cells
    label = np.zeros((N, N, N), dtype=np.int32)
    s = slice(4, 16)
    label[s, s, 4:4 + t] = 1                                  # floor
    label[s, slice(4, 4 + t), 4:14] = 1                       # four walls, open top
    label[s, slice(13, 16), 4:14] = 1
    label[slice(4, 4 + t), s, 4:14] = 1
    label[slice(13, 16), s, 4:14] = 1
    f = _finalize(label, ["air", "ceramic"], 0.003, (0.0, 0.0, 0.0),
                  lambda n: 2400.0, 1.0,
                  regions=[("vessel", "ceramic", [1], ())])
    filled, rec = fill_cavity(f, "water")
    assert rec["filled_m3"] > 0
    names = [p["name"] for p in filled.parts]
    assert names[0] == "vessel" and names[-1] == "fill:water"
    assert "fluid_approximated_as_rigid" in filled.parts[-1]["flags"]
    total = sum(p["mass_kg"] for p in filled.parts)
    assert total == pytest.approx(filled.mass_kg, rel=1e-12)


def test_bake_cache_round_trip_is_array_equal(tmp_path, monkeypatch):
    from sigma_ground.deckard import voxel_cache
    monkeypatch.setattr(voxel_cache, "_ROOT", tmp_path)
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    voxel_cache.save_cached_field("test1", 0.004, f)
    g = voxel_cache.load_cached_field("test1", 0.004)
    assert g is not None
    assert np.array_equal(f.label_grid, g.label_grid)
    assert np.allclose(f.sdf_grid, g.sdf_grid, atol=1e-7)    # f32 round-trip
    assert f.parts == g.parts
    assert f.part_of_label == g.part_of_label
    assert f.part_interfaces == g.part_interfaces
    assert voxel_cache.load_cached_field("absent", 0.004) is None


def test_a_graph_all_weld_default_with_flags():
    """P2: contacting pairs weld by default (conf 0.8, flagged) — the graph is
    exactly yesterday's rigid behavior, stated per pair."""
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    c = construct_from_field("pair", f)
    art = c.articulation
    assert art is not None
    assert [p.name for p in art.parts] == ["left", "right"]
    assert len(art.joints) == 1
    j = art.joints[0]
    assert (j.a, j.b, j.type) == (0, 1, "weld")
    assert j.confidence == 0.8 and "default_weld" in j.flags
    assert j.contact_area_m2 > 0
    assert art.mobility["available"] is False            # honest absence note


def test_a_graph_round_trip_identity():
    from sigma_ground.deckard.articulate import Articulation
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    c = construct_from_field("pair", f)
    d = c.articulation.to_dict()
    back = Articulation.from_dict(d)
    assert back.to_dict() == d                           # loss-free round trip


def test_rule_hypothesis_proposed_then_downgraded():
    """A wheel-on-frame contact PROPOSES a revolute at low confidence with the
    axis from the measured interface; effective_joints(0.75) downgrades it to
    weld with the flag — nothing moves on a guess."""
    from sigma_ground.deckard.articulate import infer_joints, effective_joints
    a, b = _two_boxes()
    f = voxelize([(a, "steel_mild", "wheel"), (b, "steel_mild", "frame")],
                 pitch=0.004)
    art = infer_joints(f.parts, f.part_interfaces)
    j = art.joints[0]
    assert j.type == "revolute" and j.confidence == 0.5
    assert "rule_hypothesis" in j.flags and "axis_estimated" in j.flags
    assert j.axis_dir is not None                        # SELECTED from geometry
    eff = effective_joints(art, min_confidence=0.75)
    assert eff[0].type == "weld"
    assert "downgraded_low_confidence" in eff[0].flags
    # explicit intent lowers the bar: the proposal survives at 0.4
    assert effective_joints(art, min_confidence=0.4)[0].type == "revolute"


def test_scene_bundle_carries_the_a_graph_additively():
    from sigma_ground.radiance.scene_export import construct_to_scene
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    c = construct_from_field("pair", f)
    spec = construct_to_scene(c)
    assert "articulation" in spec
    assert spec["articulation"]["joints"][0]["type"] == "weld"
    # additive: the geometry contract is untouched
    assert spec["csg_leaves"] and spec["materials"]


def test_distilled_annotation_overrides_inferred_weld(tmp_path, monkeypatch):
    """The oracle wins: a distilled mobility file upgrades the pair's weld to
    an annotated revolute at conf 0.95; unresolved names drop, flagged."""
    import json as _json
    from sigma_ground.deckard import articulate
    monkeypatch.setattr(articulate, "_ART_DIR", tmp_path)
    (tmp_path / "test9.json").write_text(_json.dumps({
        "model_id": "m", "source": "PartNet-Mobility (test fixture)",
        "joints": [
            {"a": "left", "b": "right", "type": "revolute",
             "axis_origin": [0.04, 0.02, 0.02], "axis_dir": [0, 0, 1],
             "limits": [0.0, 1.5708], "confidence": 0.95,
             "flags": ["partnet_mobility"]},
            {"a": "ghost", "b": "right", "type": "weld"},   # unresolvable → dropped
        ]}), encoding="utf-8")
    a, b = _two_boxes()
    f = voxelize([(a, "iron", "left"), (b, "copper", "right")], pitch=0.004)
    art = articulate.acquire("test9", f.parts, f.part_interfaces)
    assert art.mobility["available"] is True
    j = art.joints[0]
    assert j.type == "revolute" and j.confidence == 0.95
    assert j.limits == (0.0, 1.5708)
    assert "partnet_mobility" in j.flags
    assert j.contact_area_m2 > 0                          # measured area attached
    assert "ghost" in art.mobility["note"]                # the drop is visible


@pytest.mark.skipif(not _HAS_CHAIR, reason="local PartNet data not present")
def test_chair_44164_parts_survive_identify():
    """The epic's headline gate: a real chair identifies with NAMED parts."""
    from sigma_ground import deckard
    c = deckard.identify("chair", fidelity="voxel")
    assert getattr(c, "parts", None), "chair lost its parts"
    names = sorted({p["name"].split("_")[0] for p in c.parts})
    assert set(names) >= {"arm", "back", "base", "seat"}      # result.json's truth
    total = sum(p["mass_kg"] for p in c.parts)
    assert total == pytest.approx(c.mass_kg, rel=1e-9)
    assert c.part_interfaces, "no part-pair contacts scanned"
    # seat touches base and back (the chair's joint substrate exists)
    by_name = {p["part_id"]: p["name"] for p in c.parts}
    touching = {tuple(sorted((by_name[a], by_name[b])))
                for a, b in c.part_interfaces}
    assert any("seat" in t for pair in touching for t in pair)
