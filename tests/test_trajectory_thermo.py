"""The per-body thermal trajectory contract, end to end.

Frames may carry ``temperature_k`` per body (radiance/trajectory.py docstring);
``poses_at`` lerps it like a pose channel; ``bake_frame_temperatures`` gives the
Python renderer a ground-truth still; ``record_fall_thermal`` re-integrates a
drag-heated fall and REFUSES to render if its ΔT disagrees with the scenario's
(>2%); the front door routes "does an iron sphere heat up falling from 30 km"
to a thermal offer whose "yes" saves a temperature-carrying bundle.
"""
import math

import pytest

from sigma_ground.radiance.trajectory import poses_at, bake_frame_temperatures
from sigma_ground.radiance.thermal_record import record_fall_thermal


def _bundle(temps=(300.0, 400.0)):
    """A minimal 2-frame, 1-body trajectory bundle (T ramp optional)."""
    frames = []
    for i, t_k in enumerate(temps):
        body = {"pos": [0.0, 10.0 - 5.0 * i, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}
        if t_k is not None:
            body["temperature_k"] = t_k
        frames.append({"t_sim": float(i), "bodies": [body]})
    scene = {"name": "t", "bodies": [{"pivot": [0, 0, 0], "label": "iron"}],
             "csg_leaves": [
                 {"op": "add", "material": "iron", "body": 0,
                  "shape": {"type": "Sphere", "center": [0, 0, 0], "radius": 0.05}},
                 {"op": "add", "material": "concrete",     # static — never baked
                  "shape": {"type": "Box", "center": [0, -0.1, 0],
                            "x": 1.0, "y": 0.1, "z": 1.0}}],
             "materials": {"iron": {"color_rgb": [0.9, 0.9, 0.9]},
                           "concrete": {"color_rgb": [0.5, 0.5, 0.5]}},
             "bbox": [[-1, 1], [-1, 11], [-1, 1]],
             "camera": {"target": [0, 5, 0], "orbit_radius": 3.0}}
    return {"scene": scene, "kind": "trajectory",
            "trajectory": {"frames": frames, "t_end_s": float(len(temps) - 1),
                           "suggested_rate": 1.0, "body_labels": ["iron"]}}


def test_poses_at_lerps_temperature_like_a_pose_channel():
    tr = _bundle()["trajectory"]
    mid = poses_at(tr, 0.5)[0]
    assert mid["temperature_k"] == pytest.approx(350.0)   # playback lerp
    assert mid["pos"][1] == pytest.approx(7.5)            # pos still lerps
    assert abs(math.sqrt(sum(v * v for v in mid["quat"])) - 1) < 1e-9
    assert poses_at(tr, -1.0)[0]["temperature_k"] == 300.0   # clamped ends
    assert poses_at(tr, 9.0)[0]["temperature_k"] == 400.0


def test_poses_at_holds_when_one_endpoint_lacks_temperature():
    tr = _bundle(temps=(300.0, None))["trajectory"]
    assert poses_at(tr, 0.5)[0]["temperature_k"] == 300.0    # hold, don't invent
    tr2 = _bundle(temps=(None, None))["trajectory"]
    assert "temperature_k" not in poses_at(tr2, 0.5)[0]      # absent stays absent


def test_bake_frame_temperatures_overrides_only_body_leaves():
    bundle = _bundle()
    baked = bake_frame_temperatures(bundle, 0.5)
    assert baked["csg_leaves"][0]["temperature_k"] == pytest.approx(350.0)
    assert "temperature_k" not in baked["csg_leaves"][1]     # static leaf untouched
    assert "temperature_k" not in bundle["scene"]["csg_leaves"][0]  # deep copy


def test_verify_artifacts_accepts_and_rejects_frame_temperatures(tmp_path):
    import json
    import sys
    sys.path.insert(0, r"D:\Aaron\development\sigma-ground\tools")
    import verify_artifacts as va

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_bundle()), encoding="utf-8")
    assert va.verify(good) == []                              # T ramp accepted
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(_bundle(temps=(None, None))), encoding="utf-8")
    assert va.verify(plain) == []                             # temps optional
    bad_doc = _bundle()
    bad_doc["trajectory"]["frames"][0]["bodies"][0]["temperature_k"] = float("nan")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_doc).replace("NaN", '"NaN"'), encoding="utf-8")
    assert any("temperature_k" in p for p in va.verify(bad))  # junk refused


def test_record_fall_thermal_frames_heat_monotonically():
    out = record_fall_thermal("iron", 0.05, 500.0, frame_dt=0.1)
    frames = out["trajectory"]["frames"]
    temps = [f["bodies"][0]["temperature_k"] for f in frames]
    assert all(t is not None for t in temps)
    assert temps[0] == pytest.approx(288.15)
    assert all(b >= a - 1e-9 for a, b in zip(temps, temps[1:]))   # drag only ADDS heat
    assert temps[-1] > temps[0]                                   # it warmed
    assert out["thermal"] is True and out["kind"] == "trajectory"
    v = out["trajectory"]["validation"]
    assert v["delta_T_final_K"] == pytest.approx(temps[-1] - 288.15, abs=1e-2)
    assert "adiabatic upper bound" in v["body_fraction_flag"]


def test_record_fall_thermal_refuses_disagreeing_physics():
    with pytest.raises(ValueError, match="thermal cross-check failed"):
        record_fall_thermal("iron", 0.05, 500.0, expected_delta_T_K=999.0)


def test_drag_heating_drop_history_matches_its_own_integral():
    from sigma_ground.materia.scenarios import drag_heating_drop
    r = drag_heating_drop("iron", radius_m=0.05, drop_altitude_m=2_000.0)
    hist = r.outputs["temperature_history"]
    temps = [h["T_K"] for h in hist]
    assert all(b >= a - 1e-9 for a, b in zip(temps, temps[1:]))   # monotone
    h = r.outputs["render_handle"]
    assert h["kind"] == "sphere_thermal"
    # the handle's cross-check target IS the history's final ΔT (same integral)
    assert temps[-1] - h["T0"] == pytest.approx(h["expected_delta_T_K"], rel=1e-6)
    assert isinstance(r.outputs["glows"], bool)


def test_front_door_two_turn_thermal_render():
    """'does an iron sphere heat up falling from 30 km' → thermal offer →
    'yes' → a saved bundle whose frames carry T and whose cross-check landed."""
    import json
    import os
    from sigma_ground.mcp.front_door import dispatch, Session

    s = Session()
    e1 = dispatch("does an iron sphere heat up falling from 30 km?",
                  use_llm=False, session=s)
    assert e1["intent"] == "simulate" and e1["can_render"] is True
    assert s.render_handle and s.render_handle["kind"] == "sphere_thermal"
    assert "temperature" in e1["text"].lower()               # the thermal offer
    e2 = dispatch("yes", use_llm=False, session=s)
    assert e2["intent"] == "render" and e2["saved"]["slug"]
    assert os.path.exists(e2["saved"]["path"])
    bundle = json.load(open(e2["saved"]["path"], encoding="utf-8"))
    frames = bundle["trajectory"]["frames"]
    temps = [f["bodies"][0].get("temperature_k") for f in frames]
    assert all(t is not None for t in temps)
    assert temps[-1] > temps[0] + 100.0                      # it genuinely heated
    v = bundle["trajectory"]["validation"]
    assert v["thermal_residual"] is not None and v["thermal_residual"] < 0.02
    assert "peak" in e2["text"].lower()                      # announce cites peak T
