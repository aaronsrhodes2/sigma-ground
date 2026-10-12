"""Tests for stage/skippyverse/meshpack.py: a model becomes a small payload with the right size, on the floor, z up. Offline, synthetic mesh."""
from __future__ import annotations

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[3] / "stage"
sys.path.insert(0, str(STAGE))

from skippyverse import meshpack  # noqa: E402

# a box 2 wide (x), 3 tall (glTF y, so z after the axis turn), 1 deep (glTF z, so -y after), centred on the origin, 12 triangles
_V = [(x, y, z) for x in (-1, 1) for y in (-1.5, 1.5) for z in (-0.5, 0.5)]
_I = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]


def _gltf(tmp_path, *, node_extra=None, colour=(1.0, 0.0, 0.0, 1.0)):
    pos = b"".join(struct.pack("<3f", *v) for v in _V)
    idx = b"".join(struct.pack("<H", i) for i in _I)
    blob = pos + idx
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}, {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-1, -1.5, -0.5], "max": [1, 1.5, 0.5]},
            {"bufferView": 1, "componentType": 5123, "count": len(_I), "type": "SCALAR"}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(colour)}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "nodes": [{"mesh": 0, **(node_extra or {})}],
        "scenes": [{"nodes": [0]}],
    }
    p = tmp_path / "box.gltf"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_box_comes_out_the_right_size_z_up_and_on_the_floor(tmp_path):
    blob, info = meshpack.pack(_gltf(tmp_path), target_tris=100)
    verts, faces, cols = meshpack.unpack(blob)
    xs, ys, zs = [v[0] for v in verts], [v[1] for v in verts], [v[2] for v in verts]
    assert max(xs) - min(xs) == pytest.approx(2.0, abs=1e-3)
    assert max(ys) - min(ys) == pytest.approx(1.0, abs=1e-3)           # glTF depth turned to y
    assert max(zs) - min(zs) == pytest.approx(3.0, abs=1e-3)           # glTF height turned to z
    assert min(zs) == pytest.approx(0.0, abs=1e-3)                      # on the floor
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0, abs=1e-3) and (min(ys) + max(ys)) / 2 == pytest.approx(0.0, abs=1e-3)
    assert len(faces) == 12 and info["tris_in"] == 12


def test_node_transform_is_applied_before_packing(tmp_path):
    blob, _ = meshpack.pack(_gltf(tmp_path, node_extra={"scale": [2, 2, 2]}), target_tris=100)
    verts, _, _ = meshpack.unpack(blob)
    assert max(v[2] for v in verts) - min(v[2] for v in verts) == pytest.approx(6.0, abs=1e-3)


def test_base_colour_survives_and_is_display_encoded(tmp_path):
    _, _, cols = meshpack.unpack(meshpack.pack(_gltf(tmp_path, colour=(1.0, 0.0, 0.0, 1.0)), target_tris=100)[0])
    assert all(c == (255, 0, 0) for c in cols)


def test_simplification_hits_the_target_and_keeps_the_size(tmp_path):
    # a dense strip of 4000 triangles along x: the packed version is under the target and as long as the original
    n = 2000
    tris = []
    for i in range(n):
        x0, x1 = i * 0.01, (i + 1) * 0.01
        tris.append(((x0, 0, 0), (x1, 0, 0), (x0, 0, 0.1), (200, 200, 200)))
        tris.append(((x1, 0, 0), (x1, 0, 0.1), (x0, 0, 0.1), (200, 200, 200)))
    verts, faces, _ = meshpack.simplify(tris, 300)
    assert 0 < len(faces) <= 300
    assert max(v[0] for v in verts) - min(v[0] for v in verts) > 0.9 * (n * 0.01) * 0.95


def test_payload_is_small_and_round_trips_the_magic(tmp_path):
    blob, info = meshpack.pack(_gltf(tmp_path), target_tris=100)
    assert info["bytes"] == len(blob) < 600
    assert zlib.decompress(blob)[:4] == b"SVM1"
    with pytest.raises(Exception):
        meshpack.unpack(zlib.compress(b"JUNKJUNKJUNK" * 4))


# ------------------------------------------------------------------------------------------------------- Gazebo Fuel SDF primitives, NASA scale
SDF_SHELF = """<sdf version="1.5"><model name="Shelf"><static>true</static><link name="link"><pose>0 0 0 0 0 0</pose>
<visual name="back"><pose>0 0.005 0.6 0 0 0</pose><geometry><box><size>0.9 0.01 1.2</size></box></geometry>
<material><diffuse>1.0 0.0 0.0</diffuse></material></visual>
<collision name="c"><geometry><box><size>9 9 9</size></box></geometry></collision>
<visual name="side"><pose>0.45 -0.195 0.6 0 0 0</pose><geometry><box><size>0.02 0.4 1.2</size></box></geometry></visual>
</link></model></sdf>"""


def _sdf(tmp_path, text=SDF_SHELF):
    p = tmp_path / "model.sdf"
    p.write_text(text, encoding="utf-8")
    return p


def test_an_sdf_of_boxes_packs_at_the_size_measure_reports_on_the_floor(tmp_path):
    from skippyverse import measure as M
    p = _sdf(tmp_path)
    m = M.measure_sdf(p, tmp_path)
    blob, info = meshpack.pack(p, target_tris=500)
    verts, faces, cols = meshpack.unpack(blob)
    ext = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
    assert ext == pytest.approx(list(m.size_whd()), abs=2e-3), "the picture is as big as the size the card was given"
    assert min(v[2] for v in verts) == pytest.approx(0, abs=1e-6) and info["tris_in"] == 24, "two boxes (collisions are not drawn)"
    assert (255, 0, 0) in set(cols), "the visual's diffuse colour survives"


def test_sdf_box_faces_point_outward(tmp_path):
    p = _sdf(tmp_path, SDF_SHELF.replace("0.45 -0.195 0.6", "0 0 0.6").replace("0.9 0.01 1.2", "1 1 1").replace("0.02 0.4 1.2", "1 1 1"))
    tris = meshpack.triangles(p)
    cen = (0.0, 0.0, 0.6)
    assert tris
    for a, b, c, _ in tris:
        n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
             (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        mid = tuple((a[i] + b[i] + c[i]) / 3 - cen[i] for i in range(3))
        assert sum(n[i] * mid[i] for i in range(3)) > 0, "counter-clockwise seen from outside, like glTF"


def test_sdf_cylinder_and_sphere_faces_point_outward_and_have_no_degenerate_triangles(tmp_path):
    for geo in ("<cylinder><radius>0.3</radius><length>1</length></cylinder>", "<sphere><radius>0.5</radius></sphere>"):
        tris = meshpack.triangles(_sdf(tmp_path, f'<sdf><model name="m"><link name="l"><visual name="v"><geometry>{geo}</geometry></visual></link></model></sdf>'))
        for a, b, c, _ in tris:
            n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                 (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            mid = [(a[i] + b[i] + c[i]) / 3 for i in range(3)]
            assert sum(x * x for x in n) > 1e-12, "a degenerate (zero-area) triangle"
            assert sum(n[i] * mid[i] for i in range(3)) > 0, geo


def test_sdf_cylinder_and_sphere_are_drawn_a_mesh_visual_is_refused(tmp_path):
    from skippyverse import measure as M
    cyl = '<sdf><model name="m"><link name="l"><visual name="v"><geometry><cylinder><radius>0.1</radius><length>0.5</length></cylinder></geometry></visual>' \
          '<visual name="s"><pose>0 0 0.5 0 0 0</pose><geometry><sphere><radius>0.1</radius></sphere></geometry></visual></link></model></sdf>'
    verts, faces, _ = meshpack.unpack(meshpack.pack(_sdf(tmp_path, cyl), target_tris=2000)[0])
    assert max(v[2] for v in verts) - min(v[2] for v in verts) == pytest.approx(0.85, abs=2e-3)       # cylinder -0.25..0.25, sphere to 0.6
    assert max(v[0] for v in verts) - min(v[0] for v in verts) == pytest.approx(0.2, abs=2e-3)
    mesh = '<sdf><model name="m"><link name="l"><visual name="v"><geometry><mesh><uri>model://m/x.dae</uri></mesh></geometry></visual></link></model></sdf>'
    with pytest.raises(M.MeasureError):
        meshpack.pack(_sdf(tmp_path, mesh), target_tris=100)
    assert meshpack.sdf_drawable(_sdf(tmp_path, cyl)) is True
    assert meshpack.sdf_drawable(_sdf(tmp_path, mesh)) is False
    assert meshpack.sdf_drawable(tmp_path / "nothing.sdf") is False


def test_pack_applies_a_scale_to_every_coordinate(tmp_path):
    verts, _, _ = meshpack.unpack(meshpack.pack(_gltf(tmp_path), target_tris=100, scale=0.5)[0])
    assert max(v[2] for v in verts) - min(v[2] for v in verts) == pytest.approx(1.5, abs=1e-3)


# ------------------------------------------------------------------------------------------------------- Collada and OBJ meshes (Gazebo Fuel's usual form)
def _dae(tmp_path, *, up="Z_UP", unit=1.0, node='<translate>0 0 1</translate>'):
    # a 2 x 1 x 3 box (x, y, z) as 12 triangles; <triangles> with VERTEX at offset 0 and NORMAL at offset 1 (so the stride is 2)
    pts = [(x, y, z) for x in (0, 2) for y in (0, 1) for z in (0, 3)]
    pos = " ".join(str(c) for p in pts for c in p)
    idx = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    p = " ".join(f"{i} 0" for i in idx)
    text = f"""<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><asset><unit meter="{unit}"/><up_axis>{up}</up_axis></asset>
<library_geometries><geometry id="g"><mesh><source id="pos"><float_array id="pa" count="24">{pos}</float_array>
<technique_common><accessor source="#pa" count="8" stride="3"/></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
<triangles count="12"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#n" offset="1"/><p>{p}</p></triangles></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node id="n1">{node}<instance_geometry url="#g"/></node></visual_scene></library_visual_scenes></COLLADA>"""
    f = tmp_path / "m.dae"
    f.write_text(text, encoding="utf-8")
    return f


def test_a_collada_box_packs_at_its_measured_size_with_unit_axis_and_node_transform_applied(tmp_path):
    from skippyverse import measure as M
    for kw in ({}, {"up": "Y_UP"}, {"unit": 0.01}):
        f = _dae(tmp_path, **kw)
        m = M.measure_dae(f)
        verts, faces, _ = meshpack.unpack(meshpack.pack(f, target_tris=100)[0])
        ext = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
        assert len(faces) == 12 and ext == pytest.approx(list(m.size_whd()), abs=1e-3), kw
        assert min(v[2] for v in verts) == pytest.approx(0, abs=1e-6)


def test_a_collada_polylist_is_fanned_into_triangles(tmp_path):
    f = _dae(tmp_path)
    t = f.read_text(encoding="utf-8")
    quad = '<polylist count="1"><input semantic="VERTEX" source="#v" offset="0"/><vcount>4</vcount><p>0 1 3 2</p></polylist>'
    f.write_text(t[:t.index("<triangles")] + quad + t[t.index("</mesh>"):], encoding="utf-8")
    assert len(meshpack.triangles(f)) == 2


def test_an_obj_and_an_sdf_mesh_visual_with_scale_and_pose(tmp_path):
    from skippyverse import measure as M
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "x.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 2\nf 1 2 3 4\nf 1 2 5\n", encoding="utf-8")
    sdf = ('<sdf><model name="m"><link name="l"><visual name="v"><pose>0 0 1 0 0 0</pose><geometry><mesh><uri>model://m/meshes/x.obj</uri>'
           '<scale>2 2 2</scale></mesh></geometry></visual></link></model></sdf>')
    p = _sdf(tmp_path, sdf)
    assert meshpack.sdf_drawable(p) is True
    verts, faces, _ = meshpack.unpack(meshpack.pack(p, target_tris=100)[0])
    ext = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
    assert len(faces) == 3 and ext == pytest.approx(list(M.measure_sdf(p, tmp_path).size_whd()), abs=1e-3) and ext == pytest.approx([2, 2, 4], abs=1e-3)
    assert meshpack.sdf_drawable(_sdf(tmp_path, sdf.replace("x.obj", "missing.obj"))) is False
    assert meshpack.sdf_drawable(_sdf(tmp_path, sdf.replace("x.obj", "x.stl"))) is False, "STL is not read"


# ------------------------------------------------------------------------------------------------------- compressed glTF (NASA's Apollo Lunar Module is Draco)
def _draco_gltf(tmp_path):
    p = _gltf(tmp_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["extensionsUsed"] = doc["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    doc["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {"POSITION": 0}}}
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_a_draco_compressed_gltf_is_not_readable_and_says_why(tmp_path):
    from skippyverse import measure as M
    assert meshpack.gltf_readable(_gltf(tmp_path)) is True
    d = _draco_gltf(tmp_path)
    assert meshpack.gltf_readable(d) is False
    with pytest.raises(M.MeasureError, match="Draco"):
        meshpack.pack(d, target_tris=100)
    assert meshpack.gltf_readable(tmp_path / "missing.glb") is False


# ------------------------------------------------------------------------------------------------------- Draco decoding (DracoPy, optional)
def _glb(doc, blob):
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


def _draco_glb(tmp_path, buffers, *, node_extra=None, name="d.glb", colour=(1.0, 0.0, 0.0, 1.0)):
    """A GLB whose primitives are Draco buffers (one primitive per buffer in `buffers`), all under one node."""
    blob, views, prims = b"", [], []
    for i, b in enumerate(buffers):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(b)})
        blob += b + b"\0" * (-len(b) % 4)
        prims.append({"attributes": {"POSITION": i}, "material": 0, "extensions": {"KHR_draco_mesh_compression": {"bufferView": i, "attributes": {"POSITION": 0}}}})
    doc = {"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_draco_mesh_compression"], "extensionsRequired": ["KHR_draco_mesh_compression"],
           "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
           "accessors": [{"componentType": 5126, "count": 8, "type": "VEC3", "min": [-1, -1.5, -0.5], "max": [1, 1.5, 0.5]} for _ in buffers],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(colour)}}],
           "meshes": [{"primitives": prims}], "nodes": [{"mesh": 0, **(node_extra or {})}], "scenes": [{"nodes": [0]}]}
    p = tmp_path / name
    p.write_bytes(_glb(doc, blob))
    return p


def _draco_box(bits, shift=(0.0, 0.0, 0.0)):
    import numpy as np
    dp = pytest.importorskip("DracoPy")
    v = np.array([(x + shift[0], y + shift[1], z + shift[2]) for x, y, z in _V], dtype=np.float32)
    f = np.array(_I, dtype=np.uint32).reshape(-1, 3)
    return dp.encode(v, f, quantization_bits=bits)


@pytest.mark.parametrize("bits", [8, 14, 30])
def test_a_draco_box_packs_to_the_right_size_triangles_and_colour(tmp_path, bits):
    from skippyverse import measure as M
    p = _draco_glb(tmp_path, [_draco_box(bits)])
    assert meshpack.gltf_readable(p) is True
    m = M.measure_file(p)
    assert list(m.size_whd()) == pytest.approx([2.0, 1.0, 3.0], abs=0.05) and not m.notes
    blob, info = meshpack.pack(p, target_tris=100)
    verts, faces, cols = meshpack.unpack(blob)
    ext = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
    assert info["tris_in"] == 12 and len(faces) == 12
    assert ext == pytest.approx([2.0, 1.0, 3.0], abs=0.05) and min(v[2] for v in verts) == pytest.approx(0.0, abs=1e-3)
    assert set(cols) == {(255, 0, 0)}


def test_draco_with_two_primitives_and_a_node_scale(tmp_path):
    from skippyverse import measure as M
    p = _draco_glb(tmp_path, [_draco_box(14), _draco_box(14, shift=(4.0, 0.0, 0.0))], node_extra={"scale": [2, 2, 2]})
    assert list(M.measure_file(p).size_whd()) == pytest.approx([12.0, 2.0, 6.0], abs=0.1), "x spans -1..5 doubled"
    _, info = meshpack.pack(p, target_tris=1000)
    assert info["tris_in"] == 24


def test_a_corrupt_draco_buffer_is_refused_cleanly(tmp_path):
    from skippyverse import measure as M
    good = _draco_box(14)
    for bad in (good[: len(good) // 2], b"DRACO" + b"\xff" * 40, b"not draco at all"):
        p = _draco_glb(tmp_path, [bad], name="bad.glb")
        assert meshpack.gltf_readable(p) is False
        assert "Draco" in meshpack.gltf_unreadable_why(p)
        with pytest.raises(M.MeasureError, match="Draco"):
            meshpack.pack(p, target_tris=100)
        assert list(M.measure_file(p).size_whd()) == pytest.approx([2.0, 1.0, 3.0]), "the size falls back to the accessor min/max"


def test_draco_error_paths_without_the_package(tmp_path, monkeypatch):
    """Runs with or without DracoPy: a fixture buffer, the decoder made unavailable."""
    from skippyverse import measure as M
    monkeypatch.setitem(sys.modules, "DracoPy", None)               # import DracoPy -> ImportError
    assert M.draco_available() is False
    p = _draco_glb(tmp_path, [b"DRACO\0\0\0\0\0\0\0"])
    assert meshpack.gltf_readable(p) is False
    assert "decoder" in meshpack.gltf_unreadable_why(p) and "DracoPy" in meshpack.gltf_unreadable_why(p)
    with pytest.raises(M.MeasureError, match="not installed"):
        meshpack.pack(p, target_tris=100)
    assert list(M.measure_file(p).size_whd()) == pytest.approx([2.0, 1.0, 3.0]), "still sized from the accessor min/max"
    assert meshpack.gltf_readable(_gltf(tmp_path)) is True, "plain glTF is unaffected"


def test_a_draco_primitive_without_a_buffer_view_is_refused(tmp_path):
    from skippyverse import measure as M
    p = _draco_glb(tmp_path, [b"x" * 8])
    doc, _ = M._load_gltf(p)
    doc["meshes"][0]["primitives"][0]["extensions"]["KHR_draco_mesh_compression"] = {}
    with pytest.raises(M.MeasureError, match="bufferView"):
        M.decode_draco_primitive(doc, [b""], doc["meshes"][0]["primitives"][0])
