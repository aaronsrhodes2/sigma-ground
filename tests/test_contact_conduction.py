"""contact_conduction + the windward flagship — voxel-interface energy transfer
end to end: verb physics (effusivity bracket, energy ledgers), the mixed-pair
control comparison, manifest routing, the front-door two-turn flow, and the
flagship's per-cell windward field riding the 30 km fall.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.dynamics.fields.heat import diffuse_fvm
from sigma_ground.field.interface.thermal import (thermal_conductivity,
                                                  heat_capacity_volumetric)


def _pair_gain(mat_left, mat_right, seconds=2.0):
    """ΔE of the right (cold) half of an insulated 2-material bar — the
    control apparatus for the mixed-vs-pure interface comparison."""
    dx = 0.005
    shape = (24, 4, 4)
    left = np.zeros(shape, dtype=bool); left[:12] = True
    k = np.where(left, thermal_conductivity(mat_left),
                 thermal_conductivity(mat_right))
    rc = np.where(left, heat_capacity_volumetric(mat_left),
                  heat_capacity_volumetric(mat_right))
    T = np.where(left, 900.0, 300.0)
    Tf, _ = diffuse_fvm(T, k, rc, dx, total_time=seconds)
    return float(((Tf - T) * rc)[~left].sum()) * dx ** 3


def test_mixed_pair_flux_sits_between_pure_controls():
    """iron|copper interface transfer must land BETWEEN iron|iron and
    copper|copper — the harmonic-mean face never invents conductivity."""
    fe_cu = _pair_gain("iron", "copper")
    fe_fe = _pair_gain("iron", "iron")
    cu_cu = _pair_gain("copper", "copper")
    lo, hi = sorted((fe_fe, cu_cu))
    assert lo < fe_cu < hi


def test_verb_validation_passes_in_both_environments():
    from sigma_ground.materia import contact_conduction
    r = contact_conduction(contact_time_s=6.0)         # small window → fast
    v = r.validation
    assert v["passed"] is True
    lo, hi = sorted(v["interface_T_measured_K"])
    slack = 0.25 * (900.0 - 293.15)
    assert lo - slack <= v["interface_T_predicted_K"] <= hi + slack
    assert v["dE_cold_J"] > 0                          # heat crossed, early window
    h = r.outputs["render_handle"]
    assert h["kind"] == "conduction_field" and len(h["T_frames"]) == 8

    r2 = contact_conduction(environment="ISM", contact_time_s=6.0)
    assert r2.validation["passed"] is True
    assert r2.validation["energy_ledger_ok"] is True   # E0−E1 == radiated (exact)
    assert r2.validation["radiated_J"] > 0


def test_manifest_routes_the_example_sentences():
    from sigma_ground import materia
    for txt in ("put a hot iron cube on a cold copper slab",
                "how fast does heat flow between iron and copper"):
        spec = materia.translate(txt, use_qwen=False)
        assert spec.is_runnable()
        assert [st.verb for st in spec.steps] == ["contact_conduction"], txt


def test_front_door_two_turn_conduction_render(tmp_path):
    import json
    import os
    from sigma_ground.mcp.front_door import dispatch, Session
    s = Session()
    e1 = dispatch("put a hot iron cube on a cold copper slab",
                  use_llm=False, session=s)
    assert e1["intent"] == "simulate"
    assert s.render_handle and s.render_handle["kind"] == "conduction_field"
    assert "per-cell temperature field" in e1["text"]  # the kind-aware offer
    e2 = dispatch("yes", use_llm=False, session=s)
    assert e2["intent"] == "render" and os.path.exists(e2["saved"]["path"])
    b = json.load(open(e2["saved"]["path"], encoding="utf-8"))
    assert b["scene"]["theater"] == "room"             # IRT → the room theater
    with_fields = [l for l in b["scene"]["csg_leaves"] if l.get("fields")]
    assert len(with_fields) == 2                       # cube AND slab sample the grid
    assert b["scene"]["field_samples"]                 # the not-faked check ships
    assert len(b["trajectory"]["frames"]) == 8         # keyframes drive playback


def test_windward_flagship_field_rides_the_fall():
    """record_fall_thermal(windward_field=True): the per-cell field carries the
    SAME f·q energy as the bulk frames (adiabatic — matching the f=1 flag), so
    its ceiling exceeds the bulk (face-concentrated) and its floor is ambient."""
    from sigma_ground.radiance.thermal_record import record_fall_thermal
    out = record_fall_thermal("iron", 0.05, 500.0, windward_field=True,
                              frame_dt=0.1)
    leaf = out["scene"]["csg_leaves"][0]
    f = leaf["fields"]["temperature_k"]
    v = out["trajectory"]["validation"]
    bulk_end = out["trajectory"]["frames"][-1]["bodies"][0]["temperature_k"]
    assert v["windward_deposited_J"] > 0
    assert f["t_max"] >= bulk_end - 1.0                # face ceiling ≥ bulk
    assert f["t_min"] == pytest.approx(288.15, abs=2.0)
    assert out["scene"]["field_samples"]
    # bookkeeping: deposited == the bulk model's f·Q to a few percent (frame
    # rounding + max(0,Δq) clipping are the only differences)
    from sigma_ground.field.interface.thermal import specific_heat_j_kg_K
    mass = out["scene"]["physics"]["mass_kg"]
    q_bulk = (bulk_end - 288.15) * mass * specific_heat_j_kg_K("iron", 288.15)
    assert v["windward_deposited_J"] == pytest.approx(q_bulk, rel=0.05)
