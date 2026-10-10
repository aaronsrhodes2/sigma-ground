"""Offline tests for the Mentat GMod stage (Phase 0): the exporter and the PURE-Lua core.

The Lua modules (`stage/gmod/addon/lua/mentat/{units,interp,shapes_core}.lua`) have no
GMod globals, so they run here under LuaJIT (via `lupa`, the same VM family GMod uses)
and are checked against INDEPENDENT Python oracles written in this file, not against
copies of the code under test. Nothing here proves the in-game behaviour; that is the
job of `stage/gmod/addon/lua/mentat/autotest.lua` + `verify_autotest.py`.

Skipped automatically when `lupa` is not installed (it is an optional, test-only dep).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from sigma_ground.radiance import export_gmod
from sigma_ground.radiance.trajectory import poses_at

lupa = pytest.importorskip("lupa", reason="lupa (LuaJIT binding) not installed; Lua parity tests skipped")
try:  # same VM family as Garry's Mod
    from lupa import luajit21 as _lua_mod
except Exception:  # pragma: no cover
    from lupa import lua51 as _lua_mod

REPO = Path(__file__).resolve().parents[3]
LUA_DIR = REPO / "stage" / "gmod" / "addon" / "lua" / "mentat"
HU_PER_M = 1 / 0.01905


def _rt():
    rt = _lua_mod.LuaRuntime(unpack_returned_tuples=True)
    # Interpreter-only: once a loop is hot LuaJIT starts tracing and aborts traces via an
    # internal Windows SEH exception (code 0xe24c4a02) that Python's faulthandler prints as
    # "fatal" although execution continues. The arithmetic is identical either way.
    rt.execute("jit.off()")
    return rt


def _mod(rt, name):
    return rt.eval("function(p) return dofile(p) end")((LUA_DIR / f"{name}.lua").as_posix())


def _py(t):
    """lupa table -> nested python (arrays become lists, maps become dicts)."""
    if _lua_mod.lua_type(t) != "table":
        return t
    keys = list(t.keys())
    # LuaJIT numbers are doubles, so array indices come back as 1.0, 2.0, ...
    if keys and all(isinstance(k, (int, float)) and float(k).is_integer() for k in keys) \
            and sorted(int(k) for k in keys) == list(range(1, len(keys) + 1)):
        return [_py(t[i]) for i in range(1, len(keys) + 1)]
    return {k: _py(t[k]) for k in keys}


def _t(rt, x):
    """Python list/tuple (nested ok) -> Lua table. Lua code indexes from 1, so a bare
    Python list handed to Lua would be read through Python's 0-based __getitem__."""
    if isinstance(x, (list, tuple)):
        return rt.table_from([_t(rt, v) for v in x])
    return x


def _load_scenario(rt, text):
    fn = rt.eval("function(s) return assert(loadstring(s))() end")
    return fn(text)


# --------------------------------------------------------------- independent oracles
def _rot(q):
    """Rotation matrix by Rodrigues from axis-angle (independent of the Lua formula)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-15:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ang = 2 * math.atan2(s, w)
    ax = (x / s, y / s, z / s)
    c, sn = math.cos(ang), math.sin(ang)
    K = [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]
    KK = [[sum(K[i][k] * K[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [[(1 if i == j else 0) + sn * K[i][j] + (1 - c) * KK[i][j] for j in range(3)] for i in range(3)]


def _src(up, v):
    return [v[0], -v[2], v[1]] if up == "y" else list(v)


def _rand_quat(rng):
    q = [rng.gauss(0, 1) for _ in range(4)]
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


# ------------------------------------------------------------------------ units.lua
def test_world_frames_are_proper_rotations():
    rt = _rt()
    U = _mod(rt, "units")
    for up in ("y", "z"):
        W = _py(U.WORLD[up])
        det = (W[0][0] * (W[1][1] * W[2][2] - W[1][2] * W[2][1])
               - W[0][1] * (W[1][0] * W[2][2] - W[1][2] * W[2][0])
               + W[0][2] * (W[1][0] * W[2][1] - W[1][1] * W[2][0]))
        assert det == pytest.approx(1.0)


def test_unit_constant_matches_source_definition():
    U = _mod(_rt(), "units")
    assert U.HU_PER_M == pytest.approx(1 / 0.01905)      # 1 HU = 0.75 in = 0.01905 m
    assert 16 * 0.01905 == pytest.approx(0.3048)         # 16 HU = 1 foot (sanity of the constant)


def test_skillet_quaternion_stands_the_cylinder_up_in_source():
    """The bundle's body quat (-90 deg about x) maps the cylinder axis (+z local)
    to sigma +Y; in Source that must be world +Z (up)."""
    rt = _rt()
    U = _mod(rt, "units")
    f, l, u = U.entity_axes("y", _t(rt, [-math.sqrt(0.5), 0, 0, math.sqrt(0.5)]))
    assert _py(u) == pytest.approx([0, 0, 1], abs=1e-12)
    assert _py(f) == pytest.approx([1, 0, 0], abs=1e-12)   # local x stays world x
    assert _py(l) == pytest.approx([0, 1, 0], abs=1e-12)   # right-handed: x * y = z in Source


@pytest.mark.parametrize("up", ["y", "z"])
def test_entity_axes_match_independent_rotation(up):
    rt = _rt()
    U = _mod(rt, "units")
    rng = random.Random(7)
    for _ in range(200):
        q = _rand_quat(rng)
        R = _rot(q)
        f, l, u = (_py(a) for a in U.entity_axes(up, _t(rt, q)))
        for got, col in ((f, 0), (l, 1), (u, 2)):
            want = _src(up, [R[0][col], R[1][col], R[2][col]])
            assert got == pytest.approx(want, abs=1e-9)


def test_angles_roundtrip_through_source_angle_matrix():
    rt = _rt()
    U = _mod(rt, "units")
    rng = random.Random(11)
    for _ in range(500):
        p, y, r = rng.uniform(-85, 85), rng.uniform(-179, 179), rng.uniform(-179, 179)
        f, l, u = U.axes_from_angles(p, y, r)
        p2, y2, r2 = U.angles_from_axes(f, l, u)
        assert (p2, y2, r2) == pytest.approx((p, y, r), abs=1e-7)


def test_source_angle_matrix_known_values():
    """Hand-checked Source conventions: yaw 90 faces +Y (left becomes -X); pitch 90 faces DOWN."""
    U = _mod(_rt(), "units")
    f, l, u = (_py(a) for a in U.axes_from_angles(0, 90, 0))
    assert f == pytest.approx([0, 1, 0], abs=1e-12) and l == pytest.approx([-1, 0, 0], abs=1e-12)
    f, l, u = (_py(a) for a in U.axes_from_angles(90, 0, 0))
    assert f == pytest.approx([0, 0, -1], abs=1e-12)


def test_world_pos_and_inverse():
    rt = _rt()
    U = _mod(rt, "units")
    x, y, z = U.world_pos("y", _t(rt, [0.0, 3.0, 0.0]), HU_PER_M, 0.0)
    assert (x, y, z) == pytest.approx((0, 0, 3 * HU_PER_M))
    x, y, z = U.world_pos("y", _t(rt, [1.0, 2.0, 3.0]), HU_PER_M, 0.5)
    assert (x, y, z) == pytest.approx((1 * HU_PER_M, -3 * HU_PER_M, 1.5 * HU_PER_M))
    back = U.sigma_pos("y", x, y, z, HU_PER_M, 0.5)
    assert tuple(back) == pytest.approx((1.0, 2.0, 3.0))
    x, y, z = U.world_pos("z", _t(rt, [1.0, 2.0, 3.0]), HU_PER_M, 0.5)
    assert (x, y, z) == pytest.approx((1 * HU_PER_M, 2 * HU_PER_M, 2.5 * HU_PER_M))


def test_angular_velocity_axis_angle():
    rt = _rt()
    U = _mod(rt, "units")
    th = 0.5                                              # rad about +z over 0.1 s -> 5 rad/s
    q0 = _t(rt, [0, 0, 0, 1])
    q1 = [0, 0, math.sin(th / 2), math.cos(th / 2)]
    assert tuple(U.angular_velocity(q0, _t(rt, q1), 0.1)) == pytest.approx((0, 0, 5.0), abs=1e-9)
    q1n = _t(rt, [-c for c in q1])                        # same rotation, opposite sign: shortest arc
    assert tuple(U.angular_velocity(q0, q1n, 0.1)) == pytest.approx((0, 0, 5.0), abs=1e-9)
    assert tuple(U.angular_velocity(q0, q0, 0.1)) == (0, 0, 0)


# ---------------------------------------------------------------------- shapes_core.lua
def test_cylinder_mesh_is_closed_outward_and_matches_polygon_volume():
    rt = _rt()
    S = _mod(rt, "shapes_core")
    r, h, seg = 0.15, 0.12, 24
    c = S.cylinder(r, h)
    verts = _py(c.verts)
    # exact volume of an n-gon prism, written independently of the Lua code
    want = 0.5 * seg * r * r * math.sin(2 * math.pi / seg) * h
    assert S.mesh_volume(c.verts) == pytest.approx(want, rel=1e-12)
    assert 0.985 * math.pi * r * r * h < want < math.pi * r * r * h      # a 24-gon is ~1.1 % under a circle
    assert len(verts) % 3 == 0
    hull = _py(c.hull)
    assert len(hull) == 2 * seg
    assert max(abs(p[2]) for p in hull) == pytest.approx(h / 2)
    assert max(math.hypot(p[0], p[1]) for p in hull) == pytest.approx(r)


def test_all_primitives_have_outward_winding_and_unit_normals():
    rt = _rt()
    S = _mod(rt, "shapes_core")
    for prim, analytic in ((S.cylinder(0.3, 0.5), None), (S.box(1.0, 2.0, 3.0), 6.0), (S.sphere(0.7), None)):
        verts = _py(prim.verts)
        for v in verts:
            assert math.sqrt(v[3] ** 2 + v[4] ** 2 + v[5] ** 2) == pytest.approx(1.0, abs=1e-9)
        for k in range(0, len(verts), 3):
            a, b, c = verts[k], verts[k + 1], verts[k + 2]
            e1 = [b[i] - a[i] for i in range(3)]
            e2 = [c[i] - a[i] for i in range(3)]
            n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
            area2 = math.sqrt(sum(x * x for x in n))
            assert area2 > 1e-12, "degenerate triangle emitted"
            centroid = [(a[i] + b[i] + c[i]) / 3 for i in range(3)]
            # outward: geometric normal points away from the (origin-centred) shape
            assert sum(n[i] * centroid[i] for i in range(3)) > 0
            # and agrees with the stored vertex normal at the first vertex
            assert sum(n[i] * a[3 + i] for i in range(3)) > 0
        vol = S.mesh_volume(prim.verts)
        assert vol > 0
        if analytic:
            assert vol == pytest.approx(analytic)


def test_sphere_volume_is_close_below_analytic():
    S = _mod(_rt(), "shapes_core")
    v = S.mesh_volume(S.sphere(1.0).verts)
    assert 0.93 * (4 / 3 * math.pi) < v < 4 / 3 * math.pi


def test_build_body_puts_the_pivot_at_the_origin_and_scales_to_hu():
    rt = _rt()
    S = _mod(rt, "shapes_core")
    body = rt.table_from({
        "pivot": rt.table_from([0.0, 0.0, 0.044]),
        "shapes": rt.table_from([rt.table_from({
            "type": "Cylinder", "center": rt.table_from([0.0, 0.0, 0.06]),
            "radius": 0.15, "height": 0.12, "color": rt.table_from([0.7, 0.7, 0.7])})]),
    })
    g = S.build_body(body, HU_PER_M)
    mins, maxs = _py(g.mins), _py(g.maxs)
    assert mins[2] == pytest.approx((0.0 - 0.044) * HU_PER_M)    # base is 0.044 m BELOW the pivot
    assert maxs[2] == pytest.approx((0.12 - 0.044) * HU_PER_M)
    assert maxs[0] == pytest.approx(0.15 * HU_PER_M)
    assert g.volume_m3 == pytest.approx(math.pi * 0.15 ** 2 * 0.12)


# ------------------------------------------------------------------------ exporter
def _skillet():
    bundle, name, sha, path = export_gmod.load_bundle("fall_skillet")
    text, report = export_gmod.export_scenario(bundle, name, "radiance/web/data/fall_skillet.json", sha)
    return bundle, text, report


def test_exporter_decisions_for_fall_skillet():
    bundle, text, rep = _skillet()
    assert rep["up"] == "y" and rep["up_source"] == "camera.up"
    assert rep["floor_m"] == 0.0                                 # stage slab top: -0.06 + 0.12/2
    assert rep["shapes"] == [["Cylinder"]]
    joined = " ".join(rep["notes"])
    assert "subtract leaf 2" in joined and "zero size" in joined and "inside leaf 0" in joined
    assert not rep["warnings"]


def test_exported_lua_is_bit_exact_with_the_bundle():
    bundle, text, rep = _skillet()
    rt = _rt()
    scn = _py(_load_scenario(rt, text))
    frames = bundle["trajectory"]["frames"]
    assert scn["format"] == "mentat-scenario/1" and scn["stride"] == 7 and scn["up"] == "y"
    assert scn["times"] == [f["t_sim"] for f in frames]          # exact float equality
    track = scn["bodies"][0]["track"]
    want = []
    for f in frames:
        b = f["bodies"][0]
        want += list(b["pos"]) + list(b["quat"])
    assert track == want
    body = scn["bodies"][0]
    assert body["pivot"] == bundle["scene"]["bodies"][0]["pivot"]
    assert body["mass_kg"] == bundle["scene"]["physics"]["mass_kg"]
    assert body["surfaceprop"] == "metal"
    shape = body["shapes"][0]
    assert shape["type"] == "Cylinder" and shape["radius"] == 0.15 and shape["height"] == 0.12
    assert shape["center"] == [0.0, 0.0, 0.06]


@pytest.mark.parametrize("name", ["fall_skillet", "drop", "clatter", "tip"])
def test_lua_interpolation_matches_python_poses_at(name):
    try:
        bundle, _, sha, _ = export_gmod.load_bundle(name)
        text, _ = export_gmod.export_scenario(bundle, name, name, sha)
    except export_gmod.ExportError as e:                          # unsupported shapes are reported, never faked
        pytest.skip(f"{name}: {e}")
    rt = _rt()
    I = _mod(rt, "interp")
    scn = _load_scenario(rt, text)
    t0, t1 = bundle["trajectory"]["frames"][0]["t_sim"], bundle["trajectory"]["frames"][-1]["t_sim"]
    rng = random.Random(3)
    times = [t0 - 0.5, t0, t1, t1 + 0.5] + [rng.uniform(t0, t1) for _ in range(300)]
    for t in times:
        want = poses_at(bundle["trajectory"], t)
        for k, body in enumerate(want):
            got = I.sample_body(scn, scn.bodies[k + 1], t)
            assert tuple(got.pos.values()) == pytest.approx(tuple(body["pos"]), abs=1e-12)
            assert tuple(got.quat.values()) == pytest.approx(tuple(body["quat"]), abs=1e-12)


def test_lua_velocity_is_the_segment_slope_with_ballistic_continuation():
    bundle, text, _ = _skillet()
    rt = _rt()
    I = _mod(rt, "interp")
    scn = _load_scenario(rt, text)
    fr = bundle["trajectory"]["frames"]
    t = 0.33                                                     # inside segment [0.31, 0.36]
    a, b = fr[6]["bodies"][0]["pos"], fr[7]["bodies"][0]["pos"]
    dt = fr[7]["t_sim"] - fr[6]["t_sim"]
    vx, vy, vz = I.velocity(scn, scn.bodies[1], t)
    assert (vx, vy, vz) == pytest.approx(tuple((b[i] - a[i]) / dt for i in range(3)))
    assert vy < 0                                                # falling
    # after the recording ends (still moving): continue the last segment, do not freeze
    a, b = fr[-2]["bodies"][0]["pos"], fr[-1]["bodies"][0]["pos"]
    dt = fr[-1]["t_sim"] - fr[-2]["t_sim"]
    assert tuple(I.velocity(scn, scn.bodies[1], 99.0)) == pytest.approx(tuple((b[i] - a[i]) / dt for i in range(3)))


def test_up_axis_detection_and_cross_check():
    z_bundle = {"scene": {"camera": {"up": [0, 0, 1]}}, "trajectory": {"frames": []}}
    assert export_gmod.detect_up(z_bundle)[:2] == ("z", "camera.up")
    g_bundle = {"scene": {"camera": {"up": [0, 1, 0]}, "physics_env": {"gravity_dir": [0, 0, -1]}},
                "trajectory": {"frames": []}}
    assert export_gmod.detect_up(g_bundle)[:2] == ("z", "physics_env.gravity_dir")
    # camera says Z-up but the only body falls along Y: must warn, not stay silent
    liar = {"scene": {"camera": {"up": [0, 0, 1]}},
            "trajectory": {"frames": [{"t_sim": 0, "bodies": [{"pos": [0, 3, 0]}]},
                                      {"t_sim": 1, "bodies": [{"pos": [0, 0, 0]}]}]}}
    up, src, warns = export_gmod.detect_up(liar)
    assert up == "z" and warns and "verify the orientation" in warns[0]


def test_real_bundles_that_lie_about_up_are_flagged():
    for name in ("falling_aluminum_ball", "falling_lead_ball"):
        bundle, *_ = export_gmod.load_bundle(name)
        up, src, warns = export_gmod.detect_up(bundle)
        assert warns, f"{name}: expected an up-axis disagreement warning"


def test_unsupported_shapes_are_an_error_not_a_fake():
    bundle, *_ = export_gmod.load_bundle("vox_mug")          # a Voxel-only body
    with pytest.raises(export_gmod.ExportError, match="no supported shape"):
        export_gmod.export_scenario(bundle, "vox_mug")


def test_duplicate_final_frames_are_collapsed_and_reported():
    bundle, name, sha, _ = export_gmod.load_bundle("gear_train")
    text, rep = export_gmod.export_scenario(bundle, name, name, sha)
    assert any("exact-duplicate frame" in n for n in rep["notes"])
    assert rep["frames"] == len(bundle["trajectory"]["frames"]) - 1
    # two DIFFERENT poses at one timestamp is ambiguous and must not export
    bad = json.loads(json.dumps(bundle))
    bad["trajectory"]["frames"][-1]["bodies"][0]["pos"][0] += 1.0
    with pytest.raises(export_gmod.ExportError, match="ambiguous"):
        export_gmod.export_scenario(bad, "gear_train_bad")


def test_generated_lua_strings_never_use_lua51_invalid_escapes():
    s = export_gmod._str('a"b\\c\n\x01 é')
    assert "\\u" not in s and "\\001" in s and "é" in s


# ------------------------------------------------- angular velocity handed to the engine (LOCAL frame)
@pytest.mark.parametrize("up", ["y", "z"])
def test_local_angvel_is_the_world_spin_projected_on_the_body_axes(up):
    """The engine takes deg/s about the body's own forward/left/up (measured by the probe). The oracle is
    local = R^T * omega in the sigma frame, because rotating both vectors into Source preserves dot products."""
    rt = _rt()
    U = _mod(rt, "units")
    rng = random.Random(21)
    for _ in range(200):
        q = _rand_quat(rng)
        w = [rng.uniform(-9, 9) for _ in range(3)]
        R = _rot(q)
        want = [math.degrees(sum(R[r][c] * w[r] for r in range(3))) for c in range(3)]
        got = tuple(U.local_angvel_deg(up, _t(rt, w), _t(rt, q)))
        assert got == pytest.approx(tuple(want), abs=1e-9)


def test_local_angvel_differs_from_the_world_vector_when_the_body_is_tilted():
    """The bug this guards: passing the WORLD vector. A body tilted 90 deg about x, spinning about
    the world vertical, must be told to spin about its own (rotated) axis."""
    rt = _rt()
    U = _mod(rt, "units")
    q = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]           # +90 deg about sigma x
    got = tuple(U.local_angvel_deg("z", _t(rt, [0, 0, 1.0]), _t(rt, q)))
    # rolled +90 deg about x, the body's local +y axis points along world +z, so a world-z spin is a local +y spin
    assert got == pytest.approx((0, math.degrees(1.0), 0), abs=1e-9)
    world = tuple(U.world_vec("z", _t(rt, [0, 0, 1.0])).values())
    assert world == pytest.approx((0, 0, 1.0))                           # what the old code sent


# ----------------------------------------------------------------- exporter guards (no Lua needed)
def test_scenario_names_are_validated():
    bundle, name, sha, _ = export_gmod.load_bundle("fall_skillet")
    for bad in ("has space", "dot.name", "../escape", "", "ünï"):
        with pytest.raises(export_gmod.ExportError, match="must match"):
            export_gmod.export_scenario(bundle, bad)
    export_gmod.export_scenario(bundle, "ok_name-1")


def test_non_object_json_is_a_clean_error_not_a_traceback(tmp_path):
    f = tmp_path / "list_bundle.json"
    f.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(export_gmod.ExportError, match="not a bundle"):
        export_gmod.load_bundle(f)
    assert export_gmod.main([str(f), "--report-only"]) == 1             # CLI reports it, does not crash
    with pytest.raises(export_gmod.ExportError, match="not a bundle"):
        export_gmod.export_scenario([1, 2], "x")


def test_a_scenario_too_big_for_clients_is_refused_then_fixable_by_decimation():
    bundle, name, sha, _ = export_gmod.load_bundle("phase_e_ring_nbody")
    with pytest.raises(export_gmod.ExportError, match="compresses to"):
        export_gmod.export_scenario(bundle, name, name, sha)
    text, rep = export_gmod.export_scenario(bundle, name, name, sha, decimate=2)
    assert rep["client_bytes_lzma"] <= export_gmod.CLIENT_LZMA_LIMIT
    assert any("decimated by 2" in n for n in rep["notes"])
    n_all = len(bundle["trajectory"]["frames"])
    assert rep["frames"] == (n_all + 1) // 2 + (0 if (n_all - 1) % 2 == 0 else 1) or rep["frames"] < n_all
    _, rep2 = export_gmod.export_scenario(bundle, name, name, sha, allow_large=True)
    assert any("exported anyway" in w for w in rep2["warnings"])


def test_decimation_always_keeps_the_first_and_last_frame():
    notes = {"notes": []}
    frames = [{"t_sim": float(i)} for i in range(10)]
    assert [f["t_sim"] for f in export_gmod._decimate(frames, 4, notes)] == [0.0, 4.0, 8.0, 9.0]
    assert export_gmod._decimate(frames, 1, notes) is frames


def test_scenarios_that_cannot_fit_the_source_map_are_flagged_and_carry_their_extent():
    bundle, name, sha, _ = export_gmod.load_bundle("falling_copper_ball")
    text, rep = export_gmod.export_scenario(bundle, name, name, sha)
    assert any("spans" in w and "Source" in w for w in rep["warnings"]) and rep["span_m"] > 900
    assert "bbox_m = {" in text and "max_body_radius_m" in text
    _, skillet = export_gmod.export_scenario(*_skillet_args())
    assert not any("spans" in w for w in skillet["warnings"])


def _skillet_args():
    b, n, sha, _ = export_gmod.load_bundle("fall_skillet")
    return b, n, n, sha


def test_mass_outside_the_physobj_range_is_reported():
    bundle, name, sha, _ = export_gmod.load_bundle("mach_2_projectile")      # a 0.02 kg slug
    _, rep = export_gmod.export_scenario(bundle, name, name, sha)
    assert any("outside GMod" in n and "clamped" in n for n in rep["notes"])


def test_a_hollow_bundle_exported_as_a_solid_hull_is_called_out():
    """The skillet bundle is an open vessel (4 mm wall); Havok will treat the hull as solid."""
    _, rep = export_gmod.export_scenario(*_skillet_args())
    note = next(n for n in rep["notes"] if "SOLID hull" in n)
    assert "41% off" in note or "% off" in note


def test_every_surface_property_the_exporter_can_emit_is_one_the_game_knows():
    """util.GetSurfaceIndex confirmed these in the real game; 'ceramic' returned -1 (an unknown name silently
    degrades to the default material)."""
    assert "ceramic" not in export_gmod.VERIFIED_SURFACEPROPS
    emitted = {prop for prop, _ in export_gmod._SURFACE_HINTS} | {"default"}
    assert emitted <= export_gmod.VERIFIED_SURFACEPROPS
    assert export_gmod._surfaceprop("coffee cup", {"ceramic", "glaze"}) == "pottery"
    assert export_gmod._surfaceprop("dinner plate", set()) == "porcelain"
    assert export_gmod._surfaceprop("cast iron skillet", {"ceramic"}) == "metal"      # label wins over layer names
    for name in ("fall_cup", "falling_coffee_cup", "fall_glass", "fall_skillet"):
        bundle, n, sha, _ = export_gmod.load_bundle(name)
        text, _ = export_gmod.export_scenario(bundle, n, n, sha)
        for line in text.splitlines():
            if "surfaceprop" in line:
                assert line.split('"')[1] in export_gmod.VERIFIED_SURFACEPROPS, (name, line)


# --------------------------------------------------------------- experiment prop geometry (ring + build_prop)
def test_ring_is_a_closed_outward_surface_with_the_polygon_annulus_volume():
    rt = _rt()
    S = _mod(rt, "shapes_core")
    R, r, h, seg = 0.12, 0.10, 0.03, 48
    ring = S.ring(R, r, h, seg)
    want = 0.5 * seg * math.sin(2 * math.pi / seg) * (R * R - r * r) * h   # outer minus inner chord polygon, extruded
    assert S.mesh_volume(ring.verts) == pytest.approx(want, rel=1e-9)
    hulls = _py(ring.hulls)
    assert len(hulls) == seg and all(len(hu) == 8 for hu in hulls)          # one convex 8-point prism per segment
    for v in _py(ring.verts):
        assert math.sqrt(v[3] ** 2 + v[4] ** 2 + v[5] ** 2) == pytest.approx(1.0, abs=1e-9)


def test_build_prop_scales_to_hu_and_reports_the_analytic_volume():
    rt = _rt()
    S = _mod(rt, "shapes_core")
    scale = HU_PER_M
    cases = (
        ("sphere", {"radius": 0.04}, 4 / 3 * math.pi * 0.04 ** 3, 0.04),
        ("cylinder", {"radius": 0.1, "height": 0.05, "seg": 64}, math.pi * 0.1 ** 2 * 0.05, 0.1),
        ("ring", {"outer_radius": 0.1, "inner_radius": 0.09, "height": 0.05, "seg": 48},
         math.pi * (0.1 ** 2 - 0.09 ** 2) * 0.05, 0.1),
    )
    for shape, dims, vol, extent in cases:
        d = rt.table_from({k: v for k, v in dims.items()})
        g = S.build_prop(shape, d, scale)
        assert g.volume_m3 == pytest.approx(vol)
        assert max(abs(c) for c in (g.maxs[1], g.maxs[2])) == pytest.approx(extent * scale, rel=0.02)
    box = S.build_prop("box", rt.table_from({"size": rt.table_from([1.0, 2.0, 0.5])}), scale)
    assert box.maxs[2] == pytest.approx(1.0 * scale)
    assert box.volume_m3 == pytest.approx(1.0)
    with pytest.raises(Exception, match="unknown prop shape"):
        S.build_prop("blob", rt.table_from({}), scale)
