"""The ShapeNetSem distiller's aggregation — offline, synthetic rows only.

The fetch step is access-gated (and network); these tests cover the pure
distillation: per-name medians, cm->m conversion, the REAL '\\,'-escaped dims
separator, junk-row rejection, the min-sample threshold, multi-name
(category + wnlemmas + synset alias) contribution, and material-prior blending.
"""
import importlib.util
import pathlib


def _mod():
    p = pathlib.Path(__file__).resolve().parents[1] / "tools" / "distill_shapenetsem.py"
    spec = importlib.util.spec_from_file_location("distill_shapenetsem", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_aggregate_takes_per_name_medians_in_meters():
    m = _mod()
    rows = [
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "8\\,8\\,10"},
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "10\\,10\\,12"},
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "12\\,12\\,14"},
    ]
    agg, _ = m.aggregate(rows)
    assert agg["mug"]["dims_m"] == [0.10, 0.10, 0.12]          # cm -> m, median row
    assert agg["mug"]["size_m"] == 0.12                        # longest median extent
    assert agg["mug"]["n"] == 3


def test_escaped_and_plain_dims_separators_both_parse():
    m = _mod()
    assert m._split_dims("111.97\\,84.16\\,96.13") == [111.97, 84.16, 96.13]
    assert m._split_dims("30,20,20") == [30.0, 20.0, 20.0]


def test_names_below_min_sample_threshold_are_dropped():
    m = _mod()
    rows = [
        {"category": "Rare", "wnlemmas": "", "aligned.dims": "10\\,10\\,10"},
        {"category": "Rare", "wnlemmas": "", "aligned.dims": "12\\,12\\,12"},
    ]
    agg, _ = m.aggregate(rows)
    assert agg == {}                                           # n=2 < _MIN_N=3


def test_junk_rows_are_skipped_not_fatal():
    m = _mod()
    good = [{"category": "Box", "wnlemmas": "", "aligned.dims": "30\\,20\\,20"}] * 3
    junk = [
        {"category": "Box", "wnlemmas": "", "aligned.dims": ""},                  # missing
        {"category": "Box", "wnlemmas": "", "aligned.dims": "a\\,b\\,c"},         # non-numeric
        {"category": "Box", "wnlemmas": "", "aligned.dims": "-5\\,10\\,10"},      # non-positive
        {"category": "Box", "wnlemmas": "", "aligned.dims": "20000\\,10\\,10"},   # 200 m
    ]
    agg, _ = m.aggregate(good + junk)
    assert agg["box"]["n"] == 3


def test_row_contributes_to_category_lemmas_and_synset_aliases():
    m = _mod()
    rows = [{"category": "1Shelves", "wnlemmas": "",
             "aligned.dims": "80\\,30\\,180"}] * 3
    agg, name_cats = m.aggregate(rows, aliases={"1shelves": ["shelf"]})
    assert set(agg) == {"1shelves", "shelf"}                   # alias gets the data too
    assert agg["shelf"]["size_m"] == 1.8
    assert name_cats["shelf"] == {"1shelves": 3}               # provenance for materials


def test_attribute_pseudo_categories_are_ignored():
    m = _mod()
    rows = [{"category": "Speaker,_Attributes", "wnlemmas": "loudspeaker",
             "aligned.dims": "43\\,60\\,32"}] * 3
    agg, _ = m.aggregate(rows)
    assert "_attributes" not in agg and "speaker" in agg and "loudspeaker" in agg


def test_material_priors_blend_by_sample_count():
    m = _mod()
    # 'shelf' fed by two categories with different wood ratios: 3 samples of one,
    # 1 of the other -> count-weighted blend.
    name_cats = {"shelf": {"1shelves": 3, "2shelves": 1}}
    ratios = {"1shelves": {"wood": 0.8, "metal": 0.2},
              "2shelves": {"wood": 0.4, "glass": 0.6}}
    blended = m._blend_materials(name_cats, ratios)["shelf"]
    assert abs(blended["wood"] - 0.7) < 1e-9                   # (0.8*3 + 0.4*1)/4
    assert abs(blended["glass"] - 0.15) < 1e-9                 # (0.6*1)/4
    assert list(blended)[0] == "wood"                          # sorted by ratio
