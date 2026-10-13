"""stage/skippyverse acquire.py + measure.py + resolver rung 2. Fully offline: every download goes through an injected fetch function
and the meshes are tiny synthetic ones with known answers (a 2 x 1 x 3 box in each format)."""
from __future__ import annotations

import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "stage"
sys.path.insert(0, str(STAGE))

from skippyverse import acquire as A  # noqa: E402
from skippyverse import cards as C  # noqa: E402
from skippyverse import index as IX  # noqa: E402
from skippyverse import measure as M  # noqa: E402
from skippyverse import resolve as R  # noqa: E402

# a 2 (x) by 1 (y) by 3 (z) box, as 8 corners
BOX = [(x, y, z) for x in (0.0, 2.0) for y in (0.0, 1.0) for z in (0.0, 3.0)]


def approx(a, b, tol=1e-5):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)


# ------------------------------------------------------------------------------------------------------------- synthetic builders
def glb(nodes, mesh_pts=BOX, *, loose_minmax=None, scene_nodes=(0,), external_bin=None):
    flat = struct.pack(f"<{3 * len(mesh_pts)}f", *[c for p in mesh_pts for c in p])
    lo = loose_minmax[0] if loose_minmax else [min(p[i] for p in mesh_pts) for i in range(3)]
    hi = loose_minmax[1] if loose_minmax else [max(p[i] for p in mesh_pts) for i in range(3)]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(scene_nodes)}], "nodes": nodes,
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(mesh_pts), "type": "VEC3", "min": lo, "max": hi}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(flat)}],
           "buffers": [{"byteLength": len(flat)}]}
    if external_bin:
        doc["buffers"][0]["uri"] = external_bin
        return json.dumps(doc).encode(), flat
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    b = flat + b"\0" * (-len(flat) % 4)
    body = struct.pack("<II", len(j), 0x4E4F534A) + j + struct.pack("<II", len(b), 0x004E4942) + b
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def quat_y(deg):
    h = math.radians(deg) / 2
    return [0, math.sin(h), 0, math.cos(h)]


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def obj_box():
    return "\n".join(f"v {x} {y} {z}" for x, y, z in BOX) + "\nf 1 2 3\n"


def dae(unit="0.01", up="Z_UP", node=""):
    verts = " ".join(f"{x} {y} {z}" for x, y, z in BOX)
    return f"""<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="u" meter="{unit}"/><up_axis>{up}</up_axis></asset>
<library_geometries><geometry id="g"><mesh><source id="s"><float_array id="fa" count="24">{verts}</float_array>
<technique_common><accessor source="#fa" count="8" stride="3"/></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#s"/></vertices></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="vs"><node id="n">{node}<instance_geometry url="#g"/></node></visual_scene></library_visual_scenes></COLLADA>"""


# ================================================================================================================ measure: glTF
def test_glb_box_is_y_up_so_depth_is_z_and_height_is_y(tmp_path):
    m = M.measure_file(write(tmp_path, "b.glb", glb([{"mesh": 0}])))
    assert approx(m.size_whd(), (2, 3, 1)) and m.unit_known and m.up_in == "y"


def test_glb_node_rotation_swaps_the_horizontal_extents(tmp_path):
    m = M.measure_file(write(tmp_path, "b.glb", glb([{"mesh": 0, "rotation": quat_y(90)}])))
    assert approx(m.size_whd(), (3, 2, 1))


def test_glb_45_degree_rotation_uses_the_vertices_not_the_loose_minmax(tmp_path):
    loose = ([-50, -50, -50], [50, 50, 50])
    m = M.measure_file(write(tmp_path, "b.glb", glb([{"mesh": 0, "rotation": quat_y(45)}], loose_minmax=loose)))
    assert approx(m.size_whd(), (5 / math.sqrt(2), 5 / math.sqrt(2), 1))


def test_glb_scale_and_translation_nested_in_parents(tmp_path):
    nodes = [{"children": [1], "scale": [2, 2, 2], "translation": [10, 0, 0]}, {"mesh": 0, "scale": [1, 3, 1]}]
    m = M.measure_file(write(tmp_path, "b.glb", glb(nodes)))
    assert approx(m.size_whd(), (4, 6, 6))


def test_gltf_with_external_bin_and_column_major_matrix(tmp_path):
    mat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 8, 9, 1]       # translation only, column-major
    js, binary = glb([{"mesh": 0, "matrix": mat}], external_bin="b.bin")
    write(tmp_path, "b.bin", binary)
    m = M.measure_file(write(tmp_path, "b.gltf", js))
    assert approx(m.size_whd(), (2, 3, 1))


def test_normalise_matrix_puts_the_box_on_the_floor_centred(tmp_path):
    m = M.measure_file(write(tmp_path, "b.glb", glb([{"mesh": 0, "translation": [10, 20, 30]}])))
    pts = [M.mat_apply(m.normalise_matrix(), p) for p in [(10 + x, 20 + y, 30 + z) for x, y, z in BOX]]
    # the glTF box translated by (10,20,30); canonical (x, -z, y): floor is the min canonical z
    zs = [p[2] for p in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert min(zs) == pytest.approx(0, abs=1e-9) and max(zs) == pytest.approx(1)
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0, abs=1e-9) and (min(ys) + max(ys)) / 2 == pytest.approx(0, abs=1e-9)


def test_truncated_glb_is_a_measure_error(tmp_path):
    data = glb([{"mesh": 0}])
    with pytest.raises(M.MeasureError):
        M.measure_file(write(tmp_path, "t.glb", data[:20]))


# ================================================================================================================ measure: OBJ STL DAE
def test_obj_box_units_unknown(tmp_path):
    m = M.measure_file(write(tmp_path, "b.obj", obj_box()))
    assert approx(m.size_whd(), (2, 1, 3)) and not m.unit_known


def test_stl_ascii_and_binary(tmp_path):
    tris = [(BOX[0], BOX[1], BOX[2]), (BOX[4], BOX[5], BOX[7]), (BOX[3], BOX[6], BOX[7]), (BOX[0], BOX[4], BOX[6])]
    ascii_ = "solid s\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n" for t in tris) + "endsolid\n"
    binary = b"\0" * 80 + struct.pack("<I", len(tris)) + b"".join(struct.pack("<12fH", 0, 0, 0, *[c for v in t for c in v], 0) for t in tris)
    assert approx(M.measure_file(write(tmp_path, "a.stl", ascii_)).size_whd(), (2, 1, 3))
    assert approx(M.measure_file(write(tmp_path, "b.stl", binary)).size_whd(), (2, 1, 3))


def test_dae_unit_is_applied_and_z_up_kept(tmp_path):
    m = M.measure_file(write(tmp_path, "b.dae", dae("0.01", "Z_UP")))
    assert approx(m.size_whd(), (0.02, 0.01, 0.03)) and m.unit_known and m.unit_m == 0.01


def test_dae_y_up_is_rotated_to_z_up(tmp_path):
    m = M.measure_file(write(tmp_path, "b.dae", dae("1", "Y_UP")))
    assert approx(m.size_whd(), (2, 3, 1))


def test_dae_node_rotation_and_scale(tmp_path):
    node = '<rotate>0 0 1 90</rotate><scale>1 1 2</scale>'
    m = M.measure_file(write(tmp_path, "b.dae", dae("1", "Z_UP", node)))
    assert approx(m.size_whd(), (1, 2, 6))


# ================================================================================================================ measure: SDF
def sdf(body, pose=""):
    return f'<?xml version="1.0"?><sdf version="1.5"><model name="m">{pose}<link name="l">{body}</link></model></sdf>'


def visual(geom, pose=""):
    return f"<visual name=\"v\">{pose}<geometry>{geom}</geometry></visual>"


def test_sdf_scale_tag_is_applied(tmp_path):
    write(tmp_path, "b.obj", obj_box())
    p = write(tmp_path, "model.sdf", sdf(visual("<mesh><uri>model://m/b.obj</uri><scale>2 2 2</scale></mesh>")))
    m = M.measure_sdf(p, tmp_path)
    assert approx(m.size_whd(), (4, 2, 6)) and m.unit_known


def test_sdf_dae_unit_and_pose_rotation(tmp_path):
    write(tmp_path, "b.dae", dae("0.01"))
    pose = "<pose>0 0 0 0 0 1.5707963267948966</pose>"
    p = write(tmp_path, "model.sdf", sdf(visual("<mesh><uri>b.dae</uri></mesh>", pose)))
    assert approx(M.measure_sdf(p, tmp_path).size_whd(), (0.01, 0.02, 0.03))


def test_sdf_union_of_two_visuals_with_translation(tmp_path):
    write(tmp_path, "b.obj", obj_box())
    v1 = visual("<mesh><uri>b.obj</uri></mesh>")
    v2 = visual("<box><size>1 1 1</size></box>", "<pose>10 0 0 0 0 0</pose>")
    m = M.measure_sdf(write(tmp_path, "model.sdf", sdf(v1 + v2)), tmp_path)
    assert approx(m.size_whd(), (10.5, 1.5, 3.5))


def test_sdf_with_include_or_unknown_geometry_is_refused(tmp_path):
    with pytest.raises(M.MeasureError):
        M.measure_sdf(write(tmp_path, "a.sdf", sdf("<include><uri>model://x</uri></include>")), tmp_path)
    with pytest.raises(M.MeasureError):
        M.measure_sdf(write(tmp_path, "b.sdf", sdf(visual("<plane><normal>0 0 1</normal></plane>"))), tmp_path)


# ================================================================================================================ acquire
def found_card(source="Poly Haven", url="https://polyhaven.com/a/barrel_03", lic="CC0", size=None, asset=None, cid=None, name="barrel"):
    prov = {"source": source, "license": lic, "author": "A", "url": url}
    if asset:
        prov["asset"] = asset
    return C.make_card(id=cid or f"found.{C.slug(source)}.x", name=name, size=size, category="misc",
                       visual={"class": "found", "style": "x", "blocky": False, "lods": []}, provenance=prov,
                       mass=C.prior_mass(size) if size else None, flags=["front_unknown"])


class Fake:
    """An injected fetch that serves a dict url -> bytes and records every call."""

    def __init__(self, routes):
        self.routes, self.calls, self.limits = routes, [], []

    def __call__(self, url, *, max_bytes):
        self.calls.append(url)
        self.limits.append(max_bytes)
        if url not in self.routes:
            raise AssertionError(f"unexpected fetch {url}")
        return self.routes[url]


def ph_routes(nodes=None):
    gl, binary = glb(nodes or [{"mesh": 0}], external_bin="barrel_03.bin")
    files = {"gltf": {"1k": {"gltf": {"url": "https://dl.example/barrel_03_1k.gltf", "size": len(gl), "include": {
        "barrel_03.bin": {"url": "https://dl.example/barrel_03.bin", "size": len(binary)},
        "textures/barrel_03_diff_1k.jpg": {"url": "https://dl.example/tex.jpg", "size": 5}}}}}}
    return {"https://api.polyhaven.com/files/barrel_03": json.dumps(files).encode(),
            "https://dl.example/barrel_03_1k.gltf": gl, "https://dl.example/barrel_03.bin": binary}


def test_licence_gate_blocks_before_any_fetch(tmp_path):
    f = Fake({})
    for lic, src in [("Valve-SSA", "Poly Haven"), ("CC-BY-NC-4.0", "Poly Haven"), ("CC0", "ShapeNet")]:
        card = found_card(source=src, lic=lic)
        with pytest.raises(A.AcquireRefused):
            A.acquire(card, f, tmp_path)
    assert f.calls == []


def test_source_without_a_downloader_is_refused(tmp_path):
    with pytest.raises(A.AcquireRefused, match="no downloader"):
        A.acquire(found_card(source="Kenney", url="https://kenney.nl/x"), Fake({}), tmp_path)


def test_polyhaven_fetches_gltf_and_bin_and_no_texture_the_gltf_does_not_use(tmp_path):
    f = Fake(ph_routes())
    info = A.acquire(found_card(), f, tmp_path)
    assert "https://dl.example/tex.jpg" not in f.calls
    assert set(f.calls) == {"https://api.polyhaven.com/files/barrel_03", "https://dl.example/barrel_03_1k.gltf", "https://dl.example/barrel_03.bin"}
    assert (Path(info["dir"]) / "barrel_03.bin").is_file() and info["bytes"] > 0 and not info["from_cache"]


def test_second_acquire_is_a_cache_hit_with_no_fetch(tmp_path):
    card = found_card()
    A.acquire(card, Fake(ph_routes()), tmp_path)
    f2 = Fake({})
    info = A.acquire(card, f2, tmp_path)
    assert f2.calls == [] and info["from_cache"] and info["bytes"] == 0


def test_an_interrupted_download_is_not_a_cache_hit(tmp_path):
    card = found_card()
    d = A._safe_dir(tmp_path, card["id"])
    d.mkdir(parents=True)
    (d / "barrel_03_1k.gltf").write_text("{}")                      # partial leftovers, no acquired.json marker
    f = Fake(ph_routes())
    A.acquire(card, f, tmp_path)
    assert f.calls


def test_nasa_url_is_quoted_and_the_file_must_be_a_glb(tmp_path):
    path = "3D Models/Space Shuttle (B)/Space Shuttle (B).glb"
    card = found_card(source="NASA 3D Resources", url="https://github.com/nasa/x", asset=path, cid="found.nasa.sh")
    url = A.nasa_url(path)
    assert url == "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/Space%20Shuttle%20%28B%29/Space%20Shuttle%20%28B%29.glb"
    with pytest.raises(A.AcquireError, match="git-lfs"):
        A.acquire(card, Fake({url: b"version https://git-lfs.github.com/spec/v1\n"}), tmp_path / "a")
    info = A.acquire(card, Fake({url: glb([{"mesh": 0}])}), tmp_path / "b")
    assert info["mesh"].endswith(".glb")


def test_nasa_path_is_recovered_from_the_listing_for_old_cards(tmp_path):
    path = "3D Models/Hammer/Hammer.glb"
    tree = tmp_path / "tree.json"
    tree.write_text(json.dumps({"tree": [{"type": "blob", "path": path, "size": 1}]}))
    card = found_card(source="NASA 3D Resources", cid="found.nasa-3d-resources." + C.slug(path))
    assert A.nasa_path(card, tree) == path


def test_fuel_url_pattern_and_zip_handling(tmp_path):
    card = found_card(source="Gazebo Fuel", url="https://fuel.gazebosim.org/OpenRobotics/models/Fire%20hydrant", cid="found.fuel.h")
    url = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Fire%20hydrant.zip"
    assert A.fuel_url(*A.fuel_ref(card)) == url
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("model.sdf", sdf(visual("<mesh><uri>model://h/meshes/b.dae</uri></mesh>")))
        z.writestr("meshes/b.dae", dae("0.01"))
        z.writestr("materials/textures/t.png", b"x" * 100)
    info = A.acquire(card, Fake({url: buf.getvalue()}), tmp_path)
    assert info["sdf"] == "model.sdf" and (Path(info["dir"]) / "materials/textures/t.png").is_file(), "images are kept since the look pass (colours are sampled at pack time)"


def test_zip_slip_is_refused(tmp_path):
    card = found_card(source="Gazebo Fuel", url="https://fuel.gazebosim.org/O/models/N", cid="found.fuel.n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.sdf", "x")
    with pytest.raises(A.AcquireError, match="unsafe"):
        A.acquire(card, Fake({A.fuel_url("O", "N"): buf.getvalue()}), tmp_path)
    assert not (tmp_path / "evil.sdf").exists()


def test_size_limit_is_40mb_unless_forced(tmp_path):
    f = Fake(ph_routes())
    A.acquire(found_card(cid="found.a"), f, tmp_path / "1")
    assert set(f.limits) == {40 * 1024 * 1024}
    f2 = Fake(ph_routes())
    A.acquire(found_card(cid="found.b"), f2, tmp_path / "2", force=True)
    assert min(f2.limits) > 40 * 1024 * 1024


def test_default_fetch_refuses_on_content_length_and_streamed_overflow(monkeypatch):
    class Resp(io.BytesIO):
        def __init__(self, data, length):
            super().__init__(data)
            self.headers = {"Content-Length": length}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(A.urllib.request, "urlopen", lambda req, timeout=0: Resp(b"x" * 10, "999999999"))
    with pytest.raises(A.TooLarge):
        A.default_fetch("https://e.invalid/x")
    monkeypatch.setattr(A.urllib.request, "urlopen", lambda req, timeout=0: Resp(b"x" * 5000, None))
    with pytest.raises(A.TooLarge):
        A.default_fetch("https://e.invalid/x", max_bytes=1000)
    assert A.default_fetch("https://e.invalid/x", max_bytes=10_000) == b"x" * 5000


# ================================================================================================================ acquire_and_measure
def test_polyhaven_measured_size_replaces_a_listing_size_and_flags_a_disagreement(tmp_path):
    listing = {"w": 2.0, "d": 3.0, "h": 1.0, "basis": "sourced", "source": "listing"}
    r = A.acquire_and_measure(found_card(size=listing), Fake(ph_routes()), tmp_path)
    s = r["card"]["size_m"]
    assert s["basis"] == "measured" and (s["w"], s["d"], s["h"]) == pytest.approx((2, 3, 1)) and "measured_size_disagrees_with_listing" not in r["card"]["flags"]
    r2 = A.acquire_and_measure(found_card(size={**listing, "w": 1.0}, cid="found.o"), Fake(ph_routes()), tmp_path / "o")
    assert "measured_size_disagrees_with_listing" in r2["card"]["flags"]


def test_updated_card_has_mass_prior_guessed_front_unknown_and_validates(tmp_path):
    r = A.acquire_and_measure(found_card(), Fake(ph_routes()), tmp_path)
    c = r["card"]
    assert c["mass"]["basis"] == "guessed" and c["front"]["source"] == "unknown" and "size_unknown" not in c["flags"]
    assert C.validate_card(c) == [] and "size measured" in c["say"]
    assert c["visual"]["local_asset"]["mesh"] and "local_asset" not in C.strip_for_web(c)["visual"]


def test_pose_matrix_reports_floor_origin_and_up_axis(tmp_path):
    r = A.acquire_and_measure(found_card(), Fake(ph_routes([{"mesh": 0, "translation": [5, 5, 5]}])), tmp_path)
    pose = r["pose"]
    assert pose["up"] == "+Z" and len(pose["matrix"]) == 16
    zs = [M.mat_apply(pose["matrix"], (5 + x, 5 + y, 5 + z))[2] for x, y, z in BOX]
    assert min(zs) == pytest.approx(0, abs=1e-9)


def nasa_card():
    path = "3D Models/Thing/Thing.glb"
    return found_card(source="NASA 3D Resources", url="https://github.com/nasa/x", asset=path, cid="found.nasa.thing", name="thing"), A.nasa_url(path)


def test_nasa_size_stays_guessed_without_a_table_entry(tmp_path):
    card, url = nasa_card()
    r = A.acquire_and_measure(card, Fake({url: glb([{"mesh": 0}])}), tmp_path, nasa_sizes={})
    assert r["card"]["size_m"]["basis"] == "guessed" and "scale_arbitrary" in r["card"]["flags"] and "size is a guess" in r["card"]["say"]


def test_nasa_unverified_table_entry_is_ignored_and_verified_one_is_sourced(tmp_path):
    card, url = nasa_card()
    ent = {"longest_m": 30.0, "source_url": "https://nasa.example/fact", "verified": False}
    r = A.acquire_and_measure(card, Fake({url: glb([{"mesh": 0}])}), tmp_path / "a", nasa_sizes={"Thing": ent})
    assert r["card"]["size_m"]["basis"] == "guessed"
    ent["verified"] = True
    r = A.acquire_and_measure(card, Fake({url: glb([{"mesh": 0}])}), tmp_path / "b", nasa_sizes={"Thing": ent})
    s = r["card"]["size_m"]
    assert s["basis"] == "sourced" and max(s["w"], s["d"], s["h"]) == pytest.approx(30.0) and s["w"] / s["h"] == pytest.approx(2.0)
    assert "https://nasa.example/fact" in s["source"]


def test_a_guessed_download_never_overwrites_a_stronger_listing_size(tmp_path):
    card, url = nasa_card()
    card["size_m"] = {"w": 1.0, "d": 1.0, "h": 1.0, "basis": "sourced", "source": "listing"}
    r = A.acquire_and_measure(card, Fake({url: glb([{"mesh": 0}])}), tmp_path, nasa_sizes={})
    assert r["card"]["size_m"]["basis"] == "sourced" and "download_size_weaker_than_listing" in r["card"]["flags"]


def test_fuel_measures_with_sdf_scale_as_measured(tmp_path):
    card = found_card(source="Gazebo Fuel", url="https://fuel.gazebosim.org/O/models/N", cid="found.fuel.n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("model.sdf", sdf(visual("<mesh><uri>model://N/m.obj</uri><scale>2 2 2</scale></mesh>")))
        z.writestr("m.obj", obj_box())
    r = A.acquire_and_measure(card, Fake({A.fuel_url("O", "N"): buf.getvalue()}), tmp_path)
    s = r["card"]["size_m"]
    assert s["basis"] == "measured" and (s["w"], s["d"], s["h"]) == pytest.approx((4, 2, 6))


def test_unmeasurable_download_raises_acquire_error(tmp_path):
    routes = ph_routes()
    routes["https://dl.example/barrel_03.bin"] = b"\0" * 4                 # too short for the accessor
    with pytest.raises(A.AcquireError):
        A.acquire_and_measure(found_card(), Fake(routes), tmp_path)


# ================================================================================================================ resolver rung 2
def sized(w, d, h, basis="measured"):
    return {"w": w, "d": d, "h": h, "basis": basis, "source": "t"}


@pytest.fixture()
def ix(tmp_path):
    cs = [found_card(cid="f.hydrant", name="fire hydrant", url="https://polyhaven.com/a/hydrant"),
          found_card(cid="f.barrel", name="barrel", size=sized(0.6, 0.6, 0.9, "sourced"), url="https://polyhaven.com/a/barrel_03"),
          C.make_card(id="v.couch", name="couch", category="seating", size=sized(2, 1, 1), front={"yaw_deg": 0.0, "source": "curated-by-eye"},
                      visual={"class": "curated", "style": "x", "blocky": False}, provenance={"source": "Half-Life 2 (Valve)", "license": "Valve-SSA"})]
    IX.build_index(cs, tmp_path / "ix", synonym_groups=[])
    return IX.load_index(tmp_path / "ix")


def fake_acquirer(size, calls, basis="measured"):
    def acq(card):
        calls.append(card["id"])
        c = json.loads(json.dumps(card))
        c["size_m"] = sized(*size, basis=basis)
        c["flags"] = [f for f in c["flags"] if f != "size_unknown"]
        c["mass"] = C.prior_mass(c["size_m"])
        c["say"] = C.honesty_clause(c)
        return {"card": c, "pose": {"matrix": list(M.IDENT)}}
    return acq


def test_default_resolve_stays_offline_and_returns_a_fetch_plan(ix):
    r = R.resolve("a fire hydrant", ix)
    assert r["status"] == "needs_size" and r["fetched"] is False
    p = r["plan"][0]
    assert p["action"] == "fetch" and p["executed"] is False and p["source"] == "Poly Haven" and p["card_id"] == "f.hydrant"


def test_acquirer_executes_rung_2_and_tells_the_truth(ix):
    calls = []
    r = R.resolve("a fire hydrant", ix, acquirer=fake_acquirer((0.4, 0.4, 0.9), calls))
    assert calls == ["f.hydrant"] and r["fetched"] and r["rung"] == "2" and r["status"] == "exact"
    assert r["card"]["size_m"]["basis"] == "measured" and r["pose"]["matrix"]
    assert "I don't have a fire hydrant on the shelf yet. Fetching a model from Poly Haven..." in r["sentence"]
    assert "size measured" in r["sentence"] and "weight is a guess" in r["sentence"]


def test_guessed_fetched_size_is_labelled_a_guess_in_the_sentence(ix):
    r = R.resolve("a fire hydrant", ix, acquirer=fake_acquirer((0.4, 0.4, 0.9), [], basis="guessed"))
    assert "size is a guess" in r["sentence"]


def test_failed_fetch_is_reported_not_hidden(ix):
    def boom(card):
        raise A.AcquireError("network down")
    r = R.resolve("a fire hydrant", ix, acquirer=boom)
    assert r["status"] == "fetch_failed" and r["refusal"] == "fetch_failed" and "network down" in r["sentence"] and r["fetched"] is False


def test_sized_found_card_without_a_local_asset_is_fetched_only_when_an_acquirer_is_given(ix):
    calls = []
    assert R.resolve("a barrel", ix)["fetched"] is False
    r = R.resolve("a barrel", ix, acquirer=fake_acquirer((0.62, 0.62, 0.95), calls))
    assert calls == ["f.barrel"] and r["fetched"] and r["card"]["size_m"]["basis"] == "measured"


def test_an_implausible_measured_size_is_not_placed(ix):
    r = R.resolve("a fire hydrant", ix, acquirer=fake_acquirer((900, 900, 900), []))
    assert r["status"] == "untrusted_size" and "looks wrong" in r["sentence"]


def test_non_found_or_non_servable_cards_are_never_passed_to_the_acquirer(ix):
    calls = []
    r = R.resolve("a couch", ix, acquirer=fake_acquirer((1, 1, 1), calls))
    assert calls == [] and r["fetched"] is False and r["status"] == "exact"
