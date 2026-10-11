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
