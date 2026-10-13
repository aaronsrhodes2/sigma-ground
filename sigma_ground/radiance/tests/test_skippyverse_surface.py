"""The look pass on found objects (2026-10-12): colour from textures without shipping them, material cues, colour regions (a palette), smooth normals
with creases, the SVM3 payload, and the texture downloads. Offline and synthetic: a 2 x 3 x 1 box with a two-colour texture, a Collada box with a
material and a submesh, a glTF with metal/rough maps. Fail-first claims are marked."""
from __future__ import annotations

import base64
import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "stage"
sys.path.insert(0, str(STAGE))

from skippyverse import acquire as A  # noqa: E402
from skippyverse import meshpack  # noqa: E402
from skippyverse import surface as S  # noqa: E402
from test_skippyverse_acquire import Fake, found_card  # noqa: E402

# the same box as the meshpack tests: 2 wide, 3 tall (glTF y), 1 deep; 12 triangles
_V = [(x, y, z) for x in (-1, 1) for y in (-1.5, 1.5) for z in (-0.5, 0.5)]
_I = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]


def png(colours, size=8):
    """A size x size image, left half colours[0], right half colours[1] (display RGB)."""
    im = Image.new("RGBA", (size, size))
    for x in range(size):
        for y in range(size):
            im.putpixel((x, y), tuple(colours[0 if x < size // 2 else 1]) + (255,))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def textured_gltf(tmp_path, tex, *, uv_of=lambda x, y, z: (0.1 if x < 0 else 0.9, 0.5), pbr=None, extra_mat=None, with_uv=True):
    pos = b"".join(struct.pack("<3f", *v) for v in _V)
    uv = b"".join(struct.pack("<2f", *uv_of(*v)) for v in _V)
    idx = b"".join(struct.pack("<H", i) for i in _I)
    blob = pos + uv + idx
    attrs = {"POSITION": 0}
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}, {"buffer": 0, "byteOffset": len(pos) + len(uv), "byteLength": len(idx)}]
    acc = [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-1, -1.5, -0.5], "max": [1, 1.5, 0.5]},
           {"bufferView": 1, "componentType": 5123, "count": len(_I), "type": "SCALAR"}]
    if with_uv:
        views.append({"buffer": 0, "byteOffset": len(pos), "byteLength": len(uv)})
        acc.append({"bufferView": 2, "componentType": 5126, "count": 8, "type": "VEC2"})
        attrs["TEXCOORD_0"] = 2
    (tmp_path / "textures").mkdir(exist_ok=True)
    (tmp_path / "textures" / "diff.png").write_bytes(tex)
    mat = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, **(pbr or {})}, **(extra_mat or {})}
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
           "bufferViews": views, "accessors": acc, "materials": [mat], "textures": [{"source": 0}], "images": [{"uri": "textures/diff.png"}],
           "meshes": [{"primitives": [{"attributes": attrs, "indices": 1, "material": 0}]}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}]}
    p = tmp_path / "m.gltf"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


RED, BLUE = (200, 30, 30), (30, 60, 200)


# ------------------------------------------------------------------------------------------------------------- texture sampling
def test_texture_sampling_is_bilinear_and_wraps():
    t = S.Texture(png([RED, BLUE], 8), max_side=8)
    assert t.sample(0.1, 0.5)[:3] == pytest.approx(RED, abs=1.5)
    assert t.sample(0.9, 0.5)[:3] == pytest.approx(BLUE, abs=1.5)
    assert t.sample(1.1, 0.5)[:3] == pytest.approx(RED, abs=1.5), "wraps (repeat)"
    mid = t.sample(0.5, 0.5)
    assert min(RED[0], BLUE[0]) < mid[0] < max(RED[0], BLUE[0]), "between the two halves it blends"


def test_texture_is_reduced_to_a_small_image():
    big = Image.new("RGB", (1024, 1024), (10, 200, 10))
    b = io.BytesIO()
    big.save(b, "PNG")
    t = S.Texture(b.getvalue(), max_side=128)
    assert max(t.w, t.h) <= 128 and t.sample(0.3, 0.3)[:3] == pytest.approx((10, 200, 10), abs=1.5)


def test_a_broken_texture_is_none_not_an_error():
    assert S.load_texture(b"not an image") is None and S.load_texture(None) is None


def test_the_factor_tints_the_texture_in_linear_light():
    t = S.Texture(png([(255, 255, 255), (255, 255, 255)]), max_side=8)
    assert S.tri_colour((1, 1, 1), t, [(0.2, 0.5)] * 3) == (255, 255, 255)
    half = S.tri_colour((0.5, 0.5, 0.5), t, [(0.2, 0.5)] * 3)
    assert half == (S.to_display(0.5),) * 3
    assert S.tri_colour((0.5, 0.0, 0.0), None, None) == (S.to_display(0.5), 0, 0), "no texture: the factor alone (the old behaviour)"


# ------------------------------------------------------------------------------------------------------------- the glTF colour (FAIL FIRST: it was the factor only)
def test_a_textured_gltf_takes_its_colours_from_the_texture(tmp_path):
    p = textured_gltf(tmp_path, png([RED, BLUE]))
    cols = meshpack.unpack(meshpack.pack(p, 100)[0])[2]
    assert set(cols) <= {RED, BLUE} or all(min(abs(c[i] - r[i]) for r in (RED, BLUE) for i in range(3)) <= 3 for c in cols)
    assert any(c[0] > 150 for c in cols) and any(c[2] > 150 for c in cols), "both halves of the texture reach the payload (the old payload was one grey)"


def test_a_missing_texture_file_falls_back_to_the_base_colour_factor(tmp_path):
    p = textured_gltf(tmp_path, png([RED, BLUE]), pbr={"baseColorFactor": [0.0, 1.0, 0.0, 1.0]})
    (tmp_path / "textures" / "diff.png").unlink()
    cols = meshpack.unpack(meshpack.pack(p, 100)[0])[2]
    assert set(cols) == {(0, 255, 0)}


def test_the_metal_rough_texture_and_factors_become_the_material_cue(tmp_path):
    mr = Image.new("RGB", (4, 4), (0, 40, 255))                      # G = rough 0.16, B = metal 1.0
    b = io.BytesIO()
    mr.save(b, "PNG")
    p = textured_gltf(tmp_path, png([RED, RED]))
    doc = json.loads(p.read_text())
    doc["images"].append({"uri": "textures/mr.png"})
    doc["textures"].append({"source": 1})
    doc["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 1}
    (tmp_path / "textures" / "mr.png").write_bytes(b.getvalue())
    p.write_text(json.dumps(doc))
    d = meshpack.unpack3(meshpack.pack(p, 100, fmt="svm3")[0])
    spec, metal, emit, alpha = d["palette"][0][3:]
    assert metal > 230 and spec > 190 and emit == 0 and alpha == 255


@pytest.mark.parametrize("metal,rough,more_spec_than,less_than", [(1.0, 0.2, 150, None), (0.0, 0.9, None, 40), (0.0, 0.3, 90, None)])
def test_pbr_cue_orders_a_polished_metal_a_plastic_and_a_matte(metal, rough, more_spec_than, less_than):
    spec, m, emit, alpha = S.pbr_cue(metal, rough)
    assert (more_spec_than is None or spec > more_spec_than) and (less_than is None or spec < less_than)
    assert m == round(255 * metal) and emit == 0 and alpha == 255


def test_unknown_metal_counts_as_zero_and_emissive_is_carried(tmp_path):
    assert S.pbr_cue(None, 0.5)[1] == 0
    p = textured_gltf(tmp_path, png([RED, RED]), extra_mat={"emissiveFactor": [1.0, 0.8, 0.0]})
    pal = meshpack.unpack3(meshpack.pack(p, 100, fmt="svm3")[0])["palette"]
    assert pal[0][5] > 150 and pal[0][0] > 200, "a glowing material: emit set, colour = the emissive colour"


def test_an_alpha_blended_material_is_see_through_and_a_cutout_loses_its_holes(tmp_path):
    tex = Image.new("RGBA", (8, 8), (255, 255, 255, 80))
    b = io.BytesIO()
    tex.save(b, "PNG")
    p = textured_gltf(tmp_path, b.getvalue(), extra_mat={"alphaMode": "BLEND"})
    pal = meshpack.unpack3(meshpack.pack(p, 100, fmt="svm3")[0])["palette"]
    assert all(e[6] < 120 for e in pal)
    p2 = textured_gltf(tmp_path, b.getvalue(), extra_mat={"alphaMode": "MASK", "alphaCutoff": 0.5})
    with pytest.raises(Exception):
        meshpack.pack(p2, 100, fmt="svm3")                           # every texel is below the cutoff: nothing left to draw


# ------------------------------------------------------------------------------------------------------------- palette
def test_palette_keeps_few_regions_and_separates_materials():
    items = [((200, 30, 30), (200, 255, 0, 255), 1.0)] * 20 + [((205, 28, 33), (200, 255, 0, 255), 1.0)] * 20 + [((30, 60, 200), (20, 0, 0, 255), 1.0)] * 30
    pal, idx = S.build_palette(items, 2)
    assert len(pal) == 2 and len(idx) == 70
    assert idx[0] == idx[25] and idx[0] != idx[60], "the two reds share a region, the blue is its own"
    assert pal[idx[0]][4] > 200 and pal[idx[60]][4] == 0, "a metal red and a plain blue stay different materials"


def test_palette_never_exceeds_its_limit():
    items = [((i * 7 % 256, i * 13 % 256, i * 29 % 256), (i % 256, 0, 0, 255), 1.0) for i in range(500)]
    pal, idx = S.build_palette(items, 24)
    assert len(pal) <= 24 and all(0 <= i < len(pal) for i in idx)


# ------------------------------------------------------------------------------------------------------------- normals
def _unit(n):
    return math.sqrt(sum(c * c for c in n))


def test_a_box_keeps_hard_edges_and_a_sphere_is_smooth():
    cube_v = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    cube_f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    v, n, f = S.smooth_normals(cube_v, cube_f)
    assert len(v) == 24, "each corner is split three ways: no pillow shading on a box"
    assert all(abs(_unit(x) - 1) < 1e-6 for x in n)
    assert all(max(abs(c) for c in x) > 0.999 for x in n), "every normal is along an axis"
    # a UV sphere: neighbouring faces differ by < 40 degrees, so vertices are shared and normals point outward
    nl, nz = 10, 14
    pts, faces = [], []
    for i in range(1, nl):
        for j in range(nz):
            th, ph = math.pi * i / nl, 2 * math.pi * j / nz
            pts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    for i in range(nl - 2):
        for j in range(nz):
            a, b = i * nz + j, i * nz + (j + 1) % nz
            c, d = (i + 1) * nz + j, (i + 1) * nz + (j + 1) % nz
            faces += [(a, c, d), (a, d, b)]
    v, n, f = S.smooth_normals(pts, faces)
    assert len(v) == len(pts), "no creases on a smooth surface"
    for p, q in zip(v, n):
        assert sum(p[k] * q[k] for k in range(3)) > 0.95


# ------------------------------------------------------------------------------------------------------------- the SVM3 payload
def test_svm3_round_trips_and_is_smaller_than_it_looks(tmp_path):
    p = textured_gltf(tmp_path, png([RED, BLUE]))
    blob, info = meshpack.pack(p, 100, "lzma", fmt="svm3")
    assert info["format"] == "svm3" and info["tris_out"] == 12 and info["verts"] == 24 and info["bytes"] < 400
    raw = meshpack.unlzma_alone(blob)
    assert raw[:4] == b"SVM3"
    d = meshpack.unpack3(blob)
    assert len(d["verts"]) == 24 and len(d["faces"]) == 12 and len(d["normals"]) == 24 and len(d["face_pal"]) == 12
    ext = [max(v[i] for v in d["verts"]) - min(v[i] for v in d["verts"]) for i in range(3)]
    assert ext == pytest.approx([2.0, 1.0, 3.0], abs=1e-3)
    assert min(v[2] for v in d["verts"]) == pytest.approx(0, abs=1e-6)
    # the normal of a vertex is the outward axis of its face: check on the face that uses it
    for f in d["faces"]:
        v = [d["verts"][i] for i in f]
        n = [d["normals"][i] for i in f]
        e1 = [v[1][k] - v[0][k] for k in range(3)]
        e2 = [v[2][k] - v[0][k] for k in range(3)]
        fn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
        assert sum(fn[k] * n[0][k] for k in range(3)) > 0, "normals agree with the winding (outward)"


def test_svm1_stays_the_default_and_still_unpacks(tmp_path):
    p = textured_gltf(tmp_path, png([RED, BLUE]))
    blob, info = meshpack.pack(p, 100, "lzma")
    assert meshpack.unlzma_alone(blob)[:4] == b"SVM1" and "format" not in info
    verts, faces, cols = meshpack.unpack(blob)
    assert len(faces) == 12 and len(cols) == 12


# ------------------------------------------------------------------------------------------------------------- Collada materials and submeshes
DAE = """<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><asset><unit meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_images><image id="img"><init_from>tex.png</init_from></image></library_images>
<library_materials><material id="red-material"><instance_effect url="#red"/></material><material id="tex-material"><instance_effect url="#tex"/></material></library_materials>
<library_effects>
<effect id="red"><profile_COMMON><technique sid="common"><phong><diffuse><color>0.8 0.1 0.1 1</color></diffuse><specular><color>0.9 0.9 0.9 1</color></specular><shininess><float>60</float></shininess></phong></technique></profile_COMMON></effect>
<effect id="tex"><profile_COMMON><newparam sid="s-surface"><surface type="2D"><init_from>img</init_from></surface></newparam><newparam sid="s-sampler"><sampler2D><source>s-surface</source></sampler2D></newparam>
<technique sid="common"><lambert><diffuse><texture texture="s-sampler" texcoord="UV"/></diffuse></lambert></technique></profile_COMMON></effect></library_effects>
<library_geometries>
<geometry id="a-mesh" name="PartA"><mesh><source id="pa"><float_array id="paa" count="9">0 0 0 1 0 0 0 1 0</float_array><technique_common><accessor source="#paa" count="3" stride="3"/></technique_common></source>
<source id="ta"><float_array id="taa" count="6">0.1 0.5 0.1 0.5 0.1 0.5</float_array><technique_common><accessor source="#taa" count="3" stride="2"/></technique_common></source>
<vertices id="va"><input semantic="POSITION" source="#pa"/></vertices>
<triangles material="redsym" count="1"><input semantic="VERTEX" source="#va" offset="0"/><input semantic="TEXCOORD" source="#ta" offset="1" set="0"/><p>0 0 1 1 2 2</p></triangles></mesh></geometry>
<geometry id="b-mesh" name="PartB"><mesh><source id="pb"><float_array id="pbb" count="9">0 0 2 1 0 2 0 1 2</float_array><technique_common><accessor source="#pbb" count="3" stride="3"/></technique_common></source>
<source id="tb"><float_array id="tbb" count="6">0.9 0.5 0.9 0.5 0.9 0.5</float_array><technique_common><accessor source="#tbb" count="3" stride="2"/></technique_common></source>
<vertices id="vb"><input semantic="POSITION" source="#pb"/></vertices>
<triangles material="texsym" count="1"><input semantic="VERTEX" source="#vb" offset="0"/><input semantic="TEXCOORD" source="#tb" offset="1" set="0"/><p>0 0 1 1 2 2</p></triangles></mesh></geometry>
</library_geometries>
<library_visual_scenes><visual_scene id="s">
<node id="A" name="PartA"><instance_geometry url="#a-mesh"><bind_material><technique_common><instance_material symbol="redsym" target="#red-material"/></technique_common></bind_material></instance_geometry></node>
<node id="B" name="PartB"><instance_geometry url="#b-mesh"><bind_material><technique_common><instance_material symbol="texsym" target="#tex-material"/></technique_common></bind_material></instance_geometry></node>
</visual_scene></library_visual_scenes></COLLADA>"""


def dae_file(tmp_path):
    (tmp_path / "tex.png").write_bytes(png([RED, BLUE]))
    f = tmp_path / "m.dae"
    f.write_text(DAE, encoding="utf-8")
    return f


def test_collada_materials_give_colours_and_gloss_and_textures_are_sampled(tmp_path):
    f = dae_file(tmp_path)
    tris = meshpack.surface_triangles(f)
    assert len(tris) == 2
    by_z = {round(t[0][2], 3): t for t in tris}
    red_tri, tex_tri = by_z[0.0], by_z[2.0]
    assert red_tri[3] == (S.to_display(0.8), S.to_display(0.1), S.to_display(0.1)) and red_tri[4][0] > 100, "a diffuse colour and a shiny specular"
    assert tex_tri[3][2] > 150 and tex_tri[3][0] < 100, "the texture (blue on the right half) at that triangle's UVs, not grey"
    assert tex_tri[4][0] < 60, "a lambert material has a weak highlight"


def test_a_submesh_name_draws_only_that_part(tmp_path):
    f = dae_file(tmp_path)
    assert len(meshpack.mesh_file_triangles(f, meshpack.M.IDENT, None, "PartA")) == 1
    assert len(meshpack.mesh_file_triangles(f, meshpack.M.IDENT, None, "B")) == 1
    assert len(meshpack.mesh_file_triangles(f, meshpack.M.IDENT, None, None)) == 2
    assert meshpack.mesh_file_triangles(f, meshpack.M.IDENT, None, "PartA")[0][0][2] == 0.0


def test_an_sdf_with_one_visual_per_submesh_is_not_drawn_ten_times_over(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "m.dae").write_text(DAE, encoding="utf-8")
    (tmp_path / "meshes" / "tex.png").write_bytes(png([RED, BLUE]))
    vis = "".join(f"<visual name='v{n}'><geometry><mesh><uri>meshes/m.dae</uri><submesh><name>{n}</name></submesh></mesh></geometry></visual>" for n in ("PartA", "PartB"))
    sdf = tmp_path / "model.sdf"
    sdf.write_text(f"<sdf version='1.5'><model name='m'><link name='l'>{vis}</link></model></sdf>", encoding="utf-8")
    assert len(meshpack.surface_triangles(sdf)) == 2, "(before: every visual drew the whole file, so 4)"


def test_a_white_collada_with_a_loose_gazebo_texture_uses_that_texture(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "materials" / "textures").mkdir(parents=True)
    (tmp_path / "materials" / "textures" / "Thing_Diffuse.tga.orig").write_bytes(png([RED, RED]))
    white = DAE.replace("0.8 0.1 0.1 1", "1 1 1 1").replace("<texture texture=\"s-sampler\" texcoord=\"UV\"/>", "<color>1 1 1 1</color>")
    (tmp_path / "meshes" / "m.dae").write_text(white, encoding="utf-8")
    tris = meshpack.surface_triangles(tmp_path / "meshes" / "m.dae")
    assert all(t[3][0] > 150 and t[3][2] < 100 for t in tris), "the shipped diffuse image colours a model whose Collada says plain white"


# ------------------------------------------------------------------------------------------------------------- downloads: textures, and a bin from the wrong resolution
def ph_gltf(*, buf_len, with_tex=True):
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": buf_len, "uri": "x.bin"}], "bufferViews": [], "accessors": [],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicRoughnessTexture": {"index": 2}}, "normalTexture": {"index": 0}}],
           "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
           "images": [{"uri": "textures/x_nor_1k.jpg"}, {"uri": "textures/x_diff_1k.jpg"}, {"uri": "textures/x_arm_1k.jpg"}],
           "meshes": [], "nodes": [], "scenes": [{"nodes": []}]}
    return json.dumps(doc).encode()


def ph_routes(bin_len=64, listed_bin_url="https://dl.example/8k/x.bin", good_bin=b"B" * 64):
    gl = ph_gltf(buf_len=len(good_bin))
    files = {"gltf": {"1k": {"gltf": {"url": "https://dl.example/1k/x_1k.gltf", "size": len(gl), "include": {
        "x.bin": {"url": listed_bin_url, "size": len(good_bin)},
        "textures/x_nor_1k.jpg": {"url": "https://dl.example/n.jpg", "size": 5},
        "textures/x_diff_1k.jpg": {"url": "https://dl.example/d.jpg", "size": 7},
        "textures/x_arm_1k.jpg": {"url": "https://dl.example/a.jpg", "size": 6}}}}}}
    return {"https://api.polyhaven.com/files/x": json.dumps(files).encode(), "https://dl.example/1k/x_1k.gltf": gl,
            "https://dl.example/1k/x.bin": good_bin, "https://dl.example/8k/x.bin": b"W" * bin_len,
            "https://dl.example/d.jpg": b"DIFFUSE", "https://dl.example/a.jpg": b"ARMARM"}


def ph_card():
    return found_card(source="Poly Haven", url="https://polyhaven.com/a/x", cid="found.poly-haven.x")


def test_polyhaven_fetches_the_colour_and_metal_rough_images_but_not_the_normal_map(tmp_path):
    f = Fake(ph_routes())
    info = A.acquire(ph_card(), f, tmp_path)
    d = Path(info["dir"])
    assert (d / "textures" / "x_diff_1k.jpg").read_bytes() == b"DIFFUSE" and (d / "textures" / "x_arm_1k.jpg").is_file()
    assert "https://dl.example/n.jpg" not in f.calls and not (d / "textures" / "x_nor_1k.jpg").exists()
    assert info["textures_checked"] is True


def test_a_bin_from_another_resolution_is_replaced_from_the_gltfs_own_folder(tmp_path):
    """FAIL FIRST: Poly Haven's listing for cannon_01 names the 8k .bin for the 1k glTF (1198792 B against the glTF's 1143768): indices ran past the
    vertex list and the object never came. The glTF's declared buffer length now decides, and the sibling file in the glTF's folder is tried."""
    f = Fake(ph_routes(bin_len=99))
    info = A.acquire(ph_card(), f, tmp_path)
    assert (Path(info["dir"]) / "x.bin").read_bytes() == b"B" * 64 and "https://dl.example/1k/x.bin" in f.calls
    with pytest.raises(A.AcquireError, match="declares"):
        A.acquire(ph_card(), Fake(ph_routes(bin_len=99, good_bin=b"B" * 64) | {"https://dl.example/1k/x.bin": b"C" * 70}), tmp_path / "two")


def test_ensure_textures_tops_up_an_old_cache_once(tmp_path):
    card = ph_card()
    routes = ph_routes()
    asset = A.acquire(card, Fake(routes), tmp_path)
    d = Path(asset["dir"])
    for f in (d / "textures").glob("*"):
        f.unlink()                                                    # a download from before the look pass
    info = json.loads((d / "acquired.json").read_text())
    info.pop("textures_checked")
    (d / "acquired.json").write_text(json.dumps(info))
    f2 = Fake(routes)
    assert A.ensure_textures(card, asset, f2, tmp_path) > 0
    assert (d / "textures" / "x_diff_1k.jpg").is_file()
    f3 = Fake({})
    assert A.ensure_textures(card, asset, f3, tmp_path) == 0 and f3.calls == [], "asked once; acquired.json remembers"


def test_fuel_zip_keeps_its_diffuse_images_now(tmp_path):
    card = found_card(source="Gazebo Fuel", url="https://fuel.gazebosim.org/O/models/N", cid="found.fuel.n")
    url = "https://fuel.gazebosim.org/1.0/O/models/N.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("model.sdf", "<sdf/>")
        z.writestr("meshes/a.dae", "<COLLADA/>")
        z.writestr("materials/textures/diffuse.png", png([RED, BLUE]))
        z.writestr("materials/textures/huge.png", b"x" * (A.TEXTURE_MAX + 10))
    info = A.acquire(card, Fake({url: buf.getvalue()}), tmp_path)
    d = Path(info["dir"])
    assert (d / "materials/textures/diffuse.png").is_file() and not (d / "materials/textures/huge.png").exists()
