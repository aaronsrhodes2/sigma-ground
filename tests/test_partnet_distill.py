"""PartNet geometry distiller — offline, synthetic data_v0 fixtures only.

Covers: label cleaning, hierarchy walking (node instances + subtree objs),
OBJ bbox streaming with the PartNet-Y-up -> Deckard-Z-up axis map, category
aggregation (freq / median count / size, z, r fractions), primitive
classification, the whole-object filter, and shipped-table loader round-trip.
"""
import importlib.util
import json
import pathlib


def _mod():
    p = pathlib.Path(__file__).resolve().parents[1] / "tools" / "distill_partnet.py"
    spec = importlib.util.spec_from_file_location("distill_partnet", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _box_obj(path, lo, hi):
    """An 8-vertex axis-aligned box in PARTNET coords (y-up)."""
    lines = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                lines.append(f"v {x} {y} {z}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _chair_model(root: pathlib.Path, anno: str):
    """One synthetic chair (PartNet y-up): 4 low legs, a seat, a back."""
    md = root / anno
    (md / "objs").mkdir(parents=True)
    hierarchy = [{
        "name": "chair", "children": [
            {"name": "chair_back", "objs": ["b"]},
            {"name": "chair_seat", "objs": ["s"]},
            {"name": "chair_base", "children": [
                {"name": "leg", "objs": [f"l{i}"]} for i in range(4)
            ]},
        ],
    }]
    (md / "result.json").write_text(json.dumps(hierarchy), encoding="utf-8")
    (md / "meta.json").write_text(json.dumps(
        {"anno_id": anno, "model_cat": "Chair"}), encoding="utf-8")
    # geometry: object spans y_pn 0..1 (becomes z_dk 0..1)
    _box_obj(md / "objs" / "s.obj", (-0.5, 0.4, -0.5), (0.5, 0.5, 0.5))     # seat
    _box_obj(md / "objs" / "b.obj", (-0.5, 0.5, -0.5), (0.5, 1.0, -0.4))    # back
    corners = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
    for i, (cx, cz) in enumerate(corners):                                   # 4 legs, LOW
        _box_obj(md / "objs" / f"l{i}.obj",
                 (cx - 0.05, 0.0, cz - 0.05), (cx + 0.05, 0.4, cz + 0.05))
    return md


def test_clean_strips_prefix_and_grouping_suffixes():
    m = _mod()
    assert m._clean("chair_base", "chair") == "base"
    assert m._clean("leg_set", "chair") == "leg"
    assert m._clean("tabletop", "table") == "tabletop"


def test_obj_bbox_maps_partnet_y_up_to_deckard_z_up(tmp_path):
    m = _mod()
    p = tmp_path / "t.obj"
    _box_obj(p, (-1.0, 0.0, -2.0), (1.0, 5.0, 2.0))     # tall in PartNet y
    lo, hi = m._obj_bbox(p)
    assert hi[2] - lo[2] == 5.0                          # tall in Deckard z
    assert hi[1] - lo[1] == 4.0                          # PartNet z -> Deckard y


def test_aggregate_counts_freq_and_fractions(tmp_path):
    m = _mod()
    m._MIN_GEOM = 3                                      # fixture-scale threshold
    root = tmp_path / "data_v0"
    dirs = [_chair_model(root, str(1000 + i)) for i in range(3)]
    agg = m.aggregate_category(dirs, "chair", {d.name for d in dirs})
    assert agg["n_models"] == 3 and agg["n_geom"] == 3
    parts = {p["name"]: p for p in agg["parts"]}
    assert parts["leg"]["count"] == 4                    # four leaf instances
    assert parts["leg"]["freq"] == 1.0
    assert parts["leg"]["z_frac"] < -0.10                # legs are LOW (axis map right)
    assert parts["leg"]["shape"] == "cylinder"           # rod aspect + label prior
    assert parts["seat"]["count"] == 1
    assert abs(parts["seat"]["size_frac"][0] - 1.0) < 0.01   # full width
    assert parts["seat"]["size_frac"][2] <= 0.11             # thin slab in z
    assert parts["back"]["z_frac"] > 0.10                # back is HIGH
    # the 'base' group spans the legs but is not the whole object -> kept or
    # dropped by ranking, but NEVER misplaced if present
    if "base" in parts and "z_frac" in parts["base"]:
        assert parts["base"]["z_frac"] < 0.0


def test_whole_object_wrapper_is_filtered(tmp_path):
    m = _mod()
    m._MIN_GEOM = 3                                      # fixture-scale threshold
    root = tmp_path / "data_v0"
    dirs = []
    for i in range(3):
        md = root / str(i)
        (md / "objs").mkdir(parents=True)
        hierarchy = [{"name": "mug", "children": [
            {"name": "regular_mug", "children": [           # subtype wrapper = whole object
                {"name": "body", "objs": ["a"]},
            ]},
        ]}]
        (md / "result.json").write_text(json.dumps(hierarchy), encoding="utf-8")
        _box_obj(md / "objs" / "a.obj", (0, 0, 0), (1, 1, 1))
        dirs.append(md)
    agg = m.aggregate_category(dirs, "mug", {d.name for d in dirs})
    names = [p["name"] for p in agg["parts"]]
    # the subtype WRAPPER (an internal node spanning the object) is filtered…
    assert "regular_mug" not in names
    # …but the LEAF body that spans the object is a REAL part (a bottle is
    # mostly one part) and must be kept, fractions intact
    assert "body" in names
    body = next(p for p in agg["parts"] if p["name"] == "body")
    assert min(body["size_frac"]) >= 0.85


def test_classify_prim_rod_slab_and_label_override():
    m = _mod()
    assert m._classify_prim([0.05, 0.05, 0.5], "strut") == "cylinder"   # rod
    assert m._classify_prim([0.9, 0.8, 0.05], "panel") == "box"         # slab
    assert m._classify_prim([0.4, 0.4, 0.4], "blade") == "outline"      # label wins


def test_shipped_table_round_trips_through_the_loader(tmp_path, monkeypatch):
    m = _mod()
    entry = [{"object": "chair", "aliases": [], "n_models": 3, "n_geom": 3,
              "parts": [{"name": "leg", "shape": "cylinder", "count": 4,
                         "freq": 1.0, "size_frac": [0.1, 0.1, 0.4],
                         "z_frac": -0.3, "r_frac": 1.1}],
              "source": "PartNet (Mo et al. 2019) — aggregate medians",
              "license": "test"}]
    p = tmp_path / "compositions.json"
    p.write_text(json.dumps(entry), encoding="utf-8")
    from sigma_ground.deckard.sources import composition
    monkeypatch.setattr(composition, "_JSON", p)
    composition._table.cache_clear()
    try:
        got = composition.composition_of("a chair")
        assert got is not None
        parts = got[0]
        assert parts[0]["name"] == "leg" and parts[0]["count"] == 4
        # enriched keys must SURVIVE the loader (passthrough)
        assert parts[0].get("freq") == 1.0
        assert parts[0].get("z_frac") == -0.3
        assert parts[0].get("size_frac") == [0.1, 0.1, 0.4]
    finally:
        composition._table.cache_clear()
