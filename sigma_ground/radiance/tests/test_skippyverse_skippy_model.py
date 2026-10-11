"""Tests for stage/skippyverse/skippy_model.py (the Skippy print kit as avatar payloads). Offline: synthetic STL/3MF files built in tmp; the
heavy test on the real kit runs only if the private kit has been extracted (env SKIPPY_KIT or stage/skippyverse/private/kit)."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import zipfile
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[3] / "stage"
sys.path.insert(0, str(STAGE))

from skippyverse import skippy_model as sm  # noqa: E402

# ------------------------------------------------------------------------------------------------------------ synthetic data


def box(x0, x1, y0, y1, z0, z1):
    """12 outward CCW triangles of an axis-aligned box: (list of 8 verts, list of 12 index triples)."""
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]       # index = x + 2y + 4z
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6),          # z-, z+
         (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3),          # y-, y+
         (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]          # x-, x+
    return v, f


def stl_bytes(tris):
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(tris)))
    for t in tris:
        out += struct.pack("<12fH", 0, 0, 0, *[c for p in t for c in p], 0)
    return bytes(out)


def stl_ascii(tris):
    s = "solid x\n"
    for t in tris:
        s += "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {p[0]} {p[1]} {p[2]}\n" for p in t) + "endloop\nendfacet\n"
    return (s + "endsolid x\n").encode()


def tris_of(v, f):
    return [tuple(v[i] for i in t) for t in f]


# part name -> (box, default extruder, {triangle index: paint_color})
PAINT_RED_ZPLUS = {2: "1C", 3: "1C"}                 # triangles 2,3 = the +z face of the head box: extruder 4 (red)
LAYOUT = {                                           # source frame, y up
    "Skippy_can": (box(-1, 1, 0, 2, -1, 1), 1, {}),
    "Skippy_jacket": (box(-1.2, 1.2, -0.5, 1.5, -1.2, 1.2), 2, {0: "0C", 5: "8"}),
    "Skippy_head_keyed": (box(-0.8, 0.8, 2, 3, -0.8, 0.8), 1, PAINT_RED_ZPLUS),
    "Skippy_hat": (box(-1, 1, 3, 4, -1, 1), 2, {1: "0C"}),
    "Skippy_hand_left": (box(1.5, 2, 1, 2.5, -0.2, 0.2), 1, {}),       # taller: the pointing hand
    "Skippy_hand_right": (box(-2, -1.5, 0.5, 1, -0.2, 0.2), 1, {}),
}
OFFSET = (1.0, -2.0, 3.0)                            # a source_offset: 3MF mesh = STL - offset


def model_xml(vs, fs, paint):
    s = '<?xml version="1.0"?>\n<model><resources><object id="1"><mesh><vertices>\n'
    s += "".join(f'<vertex x="{x - OFFSET[0]}" y="{y - OFFSET[1]}" z="{z - OFFSET[2]}"/>\n' for x, y, z in vs)
    s += "</vertices><triangles>\n"
    for k, t in enumerate(fs):
        pc = f' paint_color="{paint[k]}"' if k in paint else ""
        s += f'<triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}"{pc}/>\n'
    return (s + "</triangles></mesh></object></resources></model>").encode()


def make_kit(root: Path, drop_jacket_tri=None):
    """A tiny kit: Individual Parts/*.stl and Skippy_AMS_Coloured.3mf. `drop_jacket_tri`: a triangle the 3MF lacks (degenerate removed)."""
    (root / "Individual Parts").mkdir(parents=True)
    main = '<?xml version="1.0"?><model><resources>'
    cfg = "<config>"
    files = {}
    for i, (name, ((vs, fs), ext, paint)) in enumerate(LAYOUT.items(), start=1):
        (root / "Individual Parts" / f"{name}.stl").write_bytes(stl_bytes(tris_of(vs, fs)))
        fs3, paint3 = list(fs), dict(paint)
        if name == "Skippy_jacket" and drop_jacket_tri is not None:
            k = drop_jacket_tri
            fs3 = fs3[:k] + fs3[k + 1:]
            paint3 = {(j if j < k else j - 1): c for j, c in paint.items() if j != k}
        files[f"3D/Objects/object_{i}.model"] = model_xml(vs, fs3, paint3)
        main += f'<object id="{i * 2}"><components><component p:path="/3D/Objects/object_{i}.model" objectid="1"/></components></object>'
        cfg += (f'<object id="{i * 2}"><metadata key="name" value="{name}.stl"/><metadata key="extruder" value="{ext}"/>'
                f'<part id="1"><metadata key="source_offset_x" value="{OFFSET[0]}"/><metadata key="source_offset_y" value="{OFFSET[1]}"/>'
                f'<metadata key="source_offset_z" value="{OFFSET[2]}"/></part></object>')
    main += "</resources></model>"
    proj = {"filament_colour": ["#C0C0C0", "#0000A0", "#FCEF03", "#FF0000", "#000000", "#FFFFFF"]}
    with zipfile.ZipFile(root / "Skippy_AMS_Coloured.3mf", "w") as z:
        z.writestr("3D/3dmodel.model", main)
        z.writestr("Metadata/model_settings.config", cfg + "</config>")
        z.writestr("Metadata/project_settings.config", json.dumps(proj))
        for k, v in files.items():
            z.writestr(k, v)
    return root


# ------------------------------------------------------------------------------------------------------------ STL


def test_stl_binary_and_ascii_agree():
    v, f = box(0, 1, 0, 2, 0, 3)
    t = tris_of(v, f)
    a, b = sm.read_stl(stl_bytes(t)), sm.read_stl(stl_ascii(t))
    assert len(a) == 12 * 9 and list(a) == pytest.approx(list(b))
    assert max(a) == 3 and min(a) == 0


def test_stl_binary_with_solid_header_still_binary():
    v, f = box(0, 1, 0, 1, 0, 1)
    d = bytearray(stl_bytes(tris_of(v, f)))
    d[:5] = b"solid"                                    # exporters do this; the exact size decides
    assert len(sm.read_stl(bytes(d))) == 12 * 9


def test_stl_rejects_garbage():
    with pytest.raises(sm.ModelError):
        sm.read_stl(b"hello world")


# ------------------------------------------------------------------------------------------------------------ paint_color


@pytest.mark.parametrize("code,expect", [
    ("", {0: 1.0}), ("4", {1: 1.0}), ("8", {2: 1.0}), ("0C", {3: 1.0}), ("1C", {4: 1.0}), ("2C", {5: 1.0}), ("3C", {6: 1.0}),
])
def test_paint_leaf_states(code, expect):
    assert sm.decode_paint(code) == expect


def test_paint_split_tree():
    # read last char first: '3' = split into 4, then children '4' (state 1), '8' (state 2), '0C' = 3 (extended), '4'
    # string (reversed reading order): nibbles 3,4,8,C,0,4 -> text "40C843"
    sh = sm.decode_paint("40C843")
    assert sh == pytest.approx({1: 0.5, 2: 0.25, 3: 0.25})
    assert sm.dominant_state("40C843") == 1
    # a 2-way split with an extended leaf: nibbles 1,8,4 -> '481'; halves state 2 and state 1 -> tie broken to the lower state
    assert sm.decode_paint("481") == pytest.approx({2: 0.5, 1: 0.5})


@pytest.mark.parametrize("bad", ["3", "43", "C", "44"])
def test_paint_malformed(bad):
    with pytest.raises(sm.ModelError):
        sm.decode_paint(bad)


def test_object_states_with_removed_triangle(tmp_path):
    kit_dir = make_kit(tmp_path / "kit", drop_jacket_tri=3)
    kit = sm.read_3mf(kit_dir / "Skippy_AMS_Coloured.3mf")
    assert kit["filaments"][2] == "#FCEF03" and kit["objects"]["Skippy_hat.stl"][0]["extruder"] == 2
    flat = sm.read_stl(kit_dir / "Individual Parts" / "Skippy_jacket.stl")
    st, unmatched = sm.object_states(kit, "Skippy_jacket.stl", flat, max_unmatched=0.1)
    assert unmatched == 1 and len(st) == 12
    assert st[0] == 3 and st[5] == 2 and st[1] == 2           # paint "0C" -> yellow(3); "8" -> 2; unpainted -> the object's extruder 2
    with pytest.raises(sm.ModelError):                         # the default tolerance (0.1 %) refuses a 1-in-12 mismatch
        sm.object_states(kit, "Skippy_jacket.stl", flat)


def test_object_states_refuses_a_misaligned_mesh(tmp_path):
    kit_dir = make_kit(tmp_path / "kit")
    kit = sm.read_3mf(kit_dir / "Skippy_AMS_Coloured.3mf")
    other = sm.read_stl(kit_dir / "Individual Parts" / "Skippy_hat.stl")
    with pytest.raises(sm.ModelError):
        sm.object_states(kit, "Skippy_jacket.stl", other, max_unmatched=0.05)


# ------------------------------------------------------------------------------------------------------------ mesh ops


def sphere(n=24, m=48, r=1.0):
    verts = [(0, 0, r)] + [(r * math.sin(math.pi * i / n) * math.cos(2 * math.pi * j / m), r * math.sin(math.pi * i / n) * math.sin(2 * math.pi * j / m),
                            r * math.cos(math.pi * i / n)) for i in range(1, n) for j in range(m)] + [(0, 0, -r)]
    ring = lambda i, j: 1 + (i - 1) * m + (j % m)  # noqa: E731
    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(m)]
    for i in range(1, n - 1):
        for j in range(m):
            faces += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)), (ring(i, j), ring(i + 1, j + 1), ring(i, j + 1))]
    last = len(verts) - 1
    faces += [(last, ring(n - 1, j + 1), ring(n - 1, j)) for j in range(m)]
    return verts, faces


def flat_of(verts, faces):
    from array import array
    a = array("f")
    for f in faces:
        for i in f:
            a.extend(verts[i])
    return a


def test_decimate_hits_budget_and_keeps_shape_and_colour():
    v, f = sphere()
    cols = [(255, 0, 0) if v[f[k][0]][0] > 0.5 else (0, 0, 255) for k in range(len(f))]       # a red cap on +x
    flat = flat_of(v, f)
    dv, df, dc = sm.decimate(flat, [cols[k] for k in range(len(f))], 300)
    assert 150 < len(df) <= 320
    assert max(p[0] for p in dv) == pytest.approx(1.0, abs=0.06) and min(p[2] for p in dv) == pytest.approx(-1.0, abs=0.06)
    red = [dv[i] for fi, t in enumerate(df) if dc[fi] == (255, 0, 0) for i in t]
    assert red and sum(p[0] for p in red) / len(red) > 0.6        # the cap is still on +x
    # orientation kept: outward normals (sphere, centre at origin)
    outward = sum(1 for t in df if sm._dot(sm._area_normal(*[dv[i] for i in t]), [sum(dv[i][k] for i in t) for k in range(3)]) > 0)
    assert outward > 0.95 * len(df)


def test_decimate_small_mesh_is_untouched():
    v, f = box(0, 1, 0, 1, 0, 1)
    dv, df, _ = sm.decimate(flat_of(v, f), [(1, 2, 3)] * 12, 100)
    assert len(df) == 12 and len(dv) == 8


def test_shade_splits_creases_and_smooths_curves():
    v, f = box(0, 1, 0, 1, 0, 1)
    sv, sn, sf = sm.shade(v, f)
    assert len(sv) == 24 and all(abs(math.sqrt(sum(c * c for c in n)) - 1) < 1e-9 for n in sn)    # a cube keeps its 6 faces' normals: 3 per corner
    v, f = sphere(12, 24)
    sv, sn, sf = sm.shade(v, f)
    assert len(sv) == len(v)                                                                    # smooth: no splits


# ------------------------------------------------------------------------------------------------------------ frame, pack


@pytest.mark.parametrize("front", [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)])
def test_frame_is_a_proper_rotation_front_to_x(front):
    f = sm.frame_matrix(front)
    fx, fz = front
    ln = math.hypot(fx, fz)
    assert f((fx / ln, 0, fz / ln)) == pytest.approx((1, 0, 0), abs=1e-9)
    assert f((0, 1, 0)) == pytest.approx((0, 0, 1))
    ex, ey = f((1, 0, 0)), f((0, 0, 1))
    # determinant of the basis images of source x, y, z must be +1 (no mirroring)
    cols = [f((1, 0, 0)), f((0, 1, 0)), f((0, 0, 1))]
    det = sm._dot(cols[0], sm._cross(cols[1], cols[2]))
    assert det == pytest.approx(1.0)
    assert ex is not None and ey is not None


def test_pack2_roundtrip_and_cap():
    v, f = sphere(10, 20)
    sv, sn, sf = sm.shade(v, f)
    cols = [(10 * (i % 20), 5, 250) for i in range(len(sf))]
    blob, info = sm.pack2(sv, sn, sf, cols)
    assert blob[:1] == bytes([0x5D]) and info["bytes"] == len(blob) < 256 * 1024
    uv, un, uf, uc = sm.unpack2(blob)
    assert len(uv) == len(sv) and [tuple(t) for t in uf] == [tuple(t) for t in sf] and [tuple(c) for c in uc] == cols
    assert max(abs(a - b) for p, q in zip(uv, sv) for a, b in zip(p, q)) < 1e-4
    assert max(abs(a - b) for p, q in zip(un, sn) for a, b in zip(p, q)) < 0.01
    # SVM1 readers must refuse it rather than misread it
    with pytest.raises(sm.meshpack.M.MeasureError):
        sm.meshpack.unpack(blob)


def test_pack2_vertex_limit():
    with pytest.raises(sm.ModelError):
        sm.pack2([(0, 0, 0)] * 70000, [(0, 0, 1)] * 70000, [(0, 1, 2)], [(1, 1, 1)])


# ------------------------------------------------------------------------------------------------------------ assembly (synthetic kit)


def test_build_synthetic_kit_assembles_scales_and_pivots(tmp_path):
    kit = make_kit(tmp_path / "kit")
    out = tmp_path / "out"
    man, blobs, geo = sm.build(kit, out, height_m=0.6, progress=lambda *_: None)
    # scale: the whole figure is 0.6 m tall, standing on z = 0, centred on the footprint
    zs = [p[2] for r in geo for p in geo[r][0]]
    xs = [p[0] for r in geo for p in geo[r][0]]
    assert min(zs) == pytest.approx(0.0, abs=1e-6) and max(zs) == pytest.approx(0.6, abs=1e-6)
    assert man["figure_height_m"] == pytest.approx(0.6, abs=1e-3) and man["scale_m_per_mm"] == pytest.approx(0.6 / 4.5)
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0, abs=0.05)
    # front: the red head face was on source +z, so it must now face +X
    hv, hf, hc, _ = geo["head"]
    red = [hv[i] for k, t in enumerate(hf) for i in t if hc[k] == (255, 0, 0)]
    assert red and sum(p[0] for p in red) / len(red) > 0.5 * max(p[0] for p in hv)
    # colours come from the 3MF: jacket navy + yellow, can silver
    assert {tuple(c) for c in geo["jacket"][2]} == {(0, 0, 160), (252, 239, 3)} and {tuple(c) for c in geo["can"][2]} == {(192, 192, 192)}
    # pivots: the pointing hand is the taller one; its tip is its highest vertex; the hat top is the highest point of the figure
    piv = man["pivots"]
    assert piv["hand_point"]["part"] == "hand_a" and piv["hand_rest"]["part"] == "hand_b"
    assert piv["hand_point"]["tip"][2] == pytest.approx(max(p[2] for p in geo["hand_a"][0]))
    assert piv["hand_point"]["wrist"][2] < piv["hand_point"]["tip"][2]
    assert piv["hat"]["top"][2] == pytest.approx(0.6, abs=1e-3) and piv["head"]["pivot"][2] == pytest.approx(min(p[2] for p in hv), abs=1e-3)
    # files and budgets
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{r}.svm2" for r in sm.PARTS] + ["manifest.json"])
    assert json.loads((out / "manifest.json").read_text())["parts"][0]["origin"] == [0, 0, 0]
    for r, blob in blobs.items():
        assert len(blob) < 256 * 1024
        uv, _, uf, _ = sm.unpack2(blob)
        assert len(uv) <= 65535 and len(uf) <= sm.PARTS[r][1] * 2


def test_build_refuses_kit_without_a_red_face(tmp_path):
    kit = make_kit(tmp_path / "kit")
    # strip the head paint: no tongue, no way to know the front
    import shutil
    work = tmp_path / "k2"
    shutil.copytree(kit, work)
    z = zipfile.ZipFile(work / "Skippy_AMS_Coloured.3mf")
    items = {n: z.read(n) for n in z.namelist()}
    z.close()
    items["3D/Objects/object_3.model"] = items["3D/Objects/object_3.model"].replace(b' paint_color="1C"', b"")
    with zipfile.ZipFile(work / "Skippy_AMS_Coloured.3mf", "w") as zz:
        for n, d in items.items():
            zz.writestr(n, d)
    with pytest.raises(sm.ModelError):
        sm.build(work, None, progress=lambda *_: None)


def test_render_draws_something():
    pytest.importorskip("PIL")
    v, f = box(-1, 1, -1, 1, 0, 2)
    im = sm.render([(v, f, [(200, 50, 50)] * 12)], (30, 20), size=(64, 64), ss=1)
    raw = im.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    assert len(set(px)) > 2 and (120, 120, 128) in px


# ------------------------------------------------------------------------------------------------------------ the real kit (optional)


def _real_kit():
    for c in (os.environ.get("SKIPPY_KIT"), STAGE / "skippyverse" / "private" / "kit" / "Skippy the Magnificent"):
        if c and (Path(c) / "Skippy_AMS_Coloured.3mf").exists():
            return Path(c)
    return None


@pytest.mark.skipif(_real_kit() is None, reason="private Skippy kit not present")
def test_real_kit_budget_and_colours(tmp_path):
    man, blobs, geo = sm.build(_real_kit(), tmp_path, progress=lambda *_: None)
    assert 8000 <= sum(p["tris"] for p in man["parts"]) <= 12500
    assert all(p["verts"] <= 65535 and p["bytes"] < 256 * 1024 for p in man["parts"])
    assert {tuple(c) for c in geo["head"][2]} >= {(255, 0, 0), (192, 192, 192)}
    assert 0.55 < man["figure_height_m"] < 0.65
