"""Two-jaw hand-tool dodoxel builder gates. The pivot's joint type/axis are
DISCOVERED by infer_dodoxel_joints from real contact geometry -- these tests
confirm the discovery, not a hardcoded declaration."""
import math

import pytest

from sigma_ground.deckard.hand_tools import (build_hand_tool_field,
                                             list_hand_tools, TOOL_ALIASES)


def test_pliers_pivot_is_discovered_as_a_revolute_on_z():
    field, joint_info, meta = build_hand_tool_field("pliers")
    assert joint_info["type"] == "revolute"
    assert joint_info["axis_family"] == "axis"
    assert abs(joint_info["axis"][2]) == pytest.approx(1.0, abs=1e-6)
    assert joint_info["snap_error_deg"] < 1.0
    assert len(field.parts) == 2
    assert field.parts[0]["mass_kg"] > 0 and field.parts[1]["mass_kg"] > 0


def test_every_catalog_tool_discovers_a_clean_revolute():
    for tool in list_hand_tools():
        field, joint_info, meta = build_hand_tool_field(tool)
        assert joint_info["type"] == "revolute", f"{tool} read as weld"
        assert joint_info["snap_error_deg"] < 1.0, tool


def test_aliases_resolve_to_the_same_catalog_entry():
    f1, j1, m1 = build_hand_tool_field("pliers")
    f2, j2, m2 = build_hand_tool_field("needle-nose pliers")
    assert m1["tool"] == m2["tool"] == "pliers"
    assert set(TOOL_ALIASES.values()) <= set(list_hand_tools())


def test_unknown_tool_name_raises_rather_than_silently_defaulting():
    with pytest.raises(ValueError):
        build_hand_tool_field("a wibbly wobbly gripper thing")


def test_geometry_is_a_choice_with_a_stated_reason():
    field, joint_info, meta = build_hand_tool_field("pliers")
    geom = meta["choices"][0]
    assert "no blueprint mechanism pins a real product" in geom["description"]
    assert geom["reason"]
