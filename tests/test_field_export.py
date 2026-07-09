"""The per-cell field contract: encode → decode → sample, exactly.

u8-xfast quantization over one shared [t_min, t_max]; x-fastest pack (the
sdf_b64 convention); trilinear sampling on the DECODED payload; keyframe-indexed
field_samples ground truth; verify_artifacts schema + replay; and the Python
renderer hook (scene_from_spec's per-cell precedence + keyframe lerp).
"""
import json
import sys

import pytest

np = pytest.importorskip("numpy")

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance.scene_export import (
    field_spec_from_grid, field_spec_from_thermal, decode_field_keyframe,
    field_trilinear, field_samples, scene_from_spec,
)


def _ramp(nx=6, ny=5, nz=4, lo=300.0, hi=900.0):
    """An ASYMMETRIC-dims x-ramp — the transpose trap: any axis mix-up in the
    pack/decode shows up as hundreds of kelvin, not roundoff."""
    x = np.linspace(lo, hi, nx).reshape(nx, 1, 1)
    return np.broadcast_to(x, (nx, ny, nz)).copy()


def test_roundtrip_is_exact_on_asymmetric_dims():
    T = _ramp()
    f = field_spec_from_grid([T], [0.0], voxel_size=0.01)
    raw = decode_field_keyframe(f, 0)
    nx, ny, nz = f["grid"]["dims"]
    assert (nx, ny, nz) == (6, 5, 4)
    scale = (f["t_max"] - f["t_min"]) / 255.0
    worst = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                got = f["t_min"] + raw[(k * ny + j) * nx + i] * scale
                worst = max(worst, abs(got - T[i, j, k]))
    assert worst <= scale / 2 + 1e-9              # exactly quantization, no more


def test_trilinear_matches_cell_centres_and_clamps_outside():
    T = _ramp()
    vs = 0.01
    f = field_spec_from_grid([T], [0.0], voxel_size=vs)
    raw = decode_field_keyframe(f, 0)
    nx = f["grid"]["dims"][0]
    scale = (f["t_max"] - f["t_min"]) / 255.0
    # a cell centre: (i-(n-1)/2)*vs on each axis
    p = [(2 - (nx - 1) / 2) * vs, (1 - (5 - 1) / 2) * vs, (3 - (4 - 1) / 2) * vs]
    assert field_trilinear(raw, f, p) == pytest.approx(T[2, 1, 3], abs=scale / 2 + 1e-9)
    # far outside the box on -x → clamps to the edge cell (the ambient rind)
    left = field_trilinear(raw, f, [-1.0, 0.0, 0.0])
    q_left = raw[(0 * 5 + 2) * nx + 0]            # i=0 column value at (j=2,k=... )
    assert left == pytest.approx(f["t_min"] + q_left * scale, abs=scale)


def test_degenerate_uniform_field_guard():
    T = np.full((4, 4, 4), 500.0)
    f = field_spec_from_grid([T], [0.0], voxel_size=0.01)
    assert f["t_max"] == f["t_min"] + 1.0         # guard: never a zero range
    raw = decode_field_keyframe(f, 0)
    assert field_trilinear(raw, f, [0, 0, 0]) == pytest.approx(500.0, abs=0.01)


def test_shared_range_spans_all_keyframes():
    cold, hot = _ramp(lo=300, hi=400), _ramp(lo=300, hi=1500)
    f = field_spec_from_grid([cold, hot], [0.0, 2.0], voxel_size=0.01)
    assert f["t_min"] == pytest.approx(300.0) and f["t_max"] == pytest.approx(1500.0)
    with pytest.raises(ValueError):               # non-increasing times refused
        field_spec_from_grid([cold, hot], [2.0, 2.0], voxel_size=0.01)


def test_thermal_field_duck_typing():
    class FakeThermal:                            # .T/.dx/.T_ambient is the seam
        def __init__(self, T):
            self.T, self.dx, self.T_ambient = T, 0.005, 288.15
    f = field_spec_from_thermal([FakeThermal(_ramp())], [0.0], source="test")
    assert f["grid"]["voxel_size"] == 0.005
    assert f["outside_value"] == pytest.approx(288.15)
    assert f["source"] == "test"


def _field_scene(times=(0.0, 2.0)):
    """A sphere leaf carrying a 2-keyframe field: uniform 400 K → 1400 K."""
    grids = [np.full((6, 5, 4), t) for t in (400.0, 1400.0)][:len(times)]
    f = field_spec_from_grid(grids, list(times), voxel_size=0.05)
    leaf = {"op": "add", "material": "iron",
            "shape": {"type": "Sphere", "center": [0, 0, 0], "radius": 0.05},
            "fields": {"temperature_k": f}}
    return {"csg_leaves": [leaf],
            "materials": {"iron": {"color_rgb": [0.9, 0.9, 0.9]}},
            "field_samples": field_samples(0, f)}


def test_scene_from_spec_field_beats_scalar_and_lerps_keyframes():
    spec = _field_scene()
    spec["csg_leaves"][0]["temperature_k"] = 9999.0   # the scalar must LOSE
    centre = Vec3(0.0, 0.0, 0.0)
    s0 = scene_from_spec(spec, t_sim=0.0)
    s1 = scene_from_spec(spec, t_sim=1.0)             # midway between keyframes
    s2 = scene_from_spec(spec, t_sim=99.0)            # clamped to the last
    assert s0.temperature_at(centre) == pytest.approx(400.0, abs=4.0)
    assert s1.temperature_at(centre) == pytest.approx(900.0, abs=4.0)   # lerp
    assert s2.temperature_at(centre) == pytest.approx(1400.0, abs=4.0)


def test_verify_artifacts_field_rules(tmp_path):
    sys.path.insert(0, r"D:\Aaron\development\sigma-ground\tools")
    import verify_artifacts as va

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_field_scene()), encoding="utf-8")
    assert va.verify(good) == []
    # fields without field_samples → refused (not-faked is mandatory)
    naked = _field_scene(); naked.pop("field_samples")
    p = tmp_path / "naked.json"; p.write_text(json.dumps(naked), encoding="utf-8")
    assert any("field_samples" in x for x in va.verify(p))
    # a corrupted payload byte → the replay catches it
    bad = _field_scene()
    import base64
    kf = bad["csg_leaves"][0]["fields"]["temperature_k"]["keyframes"][0]
    raw = bytearray(base64.b64decode(kf["values_b64"])); raw[0] ^= 0xFF
    kf["values_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    p = tmp_path / "bad.json"; p.write_text(json.dumps(bad), encoding="utf-8")
    assert any("replay off" in x for x in va.verify(p))
