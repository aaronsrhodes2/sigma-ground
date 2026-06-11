"""The ShapeNetSem distiller's aggregation — offline, synthetic rows only.

The fetch step is access-gated (and network); these tests cover the pure
distillation: per-category medians, cm->m conversion, junk-row rejection,
the min-sample threshold, and multi-name (category + wnlemmas) contribution.
"""
import importlib.util
import pathlib


def _mod():
    p = pathlib.Path(__file__).resolve().parents[1] / "tools" / "distill_shapenetsem.py"
    spec = importlib.util.spec_from_file_location("distill_shapenetsem", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_aggregate_takes_per_category_medians_in_meters():
    m = _mod()
    rows = [
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "8,8,10", "weight": "0.30"},
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "10,10,12", "weight": "0.35"},
        {"category": "Mug", "wnlemmas": "mug", "aligned.dims": "12,12,14", "weight": "0.40"},
    ]
    agg = m.aggregate(rows)
    assert agg["mug"]["dims_m"] == [0.10, 0.10, 0.12]          # cm -> m, median row
    assert agg["mug"]["size_m"] == 0.12                        # longest median extent
    assert agg["mug"]["weight_kg"] == 0.35
    assert agg["mug"]["n"] == 3


def test_categories_below_min_sample_threshold_are_dropped():
    m = _mod()
    rows = [
        {"category": "Rare", "wnlemmas": "", "aligned.dims": "10,10,10", "weight": "1"},
        {"category": "Rare", "wnlemmas": "", "aligned.dims": "12,12,12", "weight": "1"},
    ]
    assert m.aggregate(rows) == {}                             # n=2 < _MIN_N=3


def test_junk_rows_are_skipped_not_fatal():
    m = _mod()
    good = [{"category": "Box", "wnlemmas": "", "aligned.dims": "30,20,20", "weight": "1.0"}] * 3
    junk = [
        {"category": "Box", "wnlemmas": "", "aligned.dims": "", "weight": "1"},          # missing
        {"category": "Box", "wnlemmas": "", "aligned.dims": "a,b,c", "weight": "1"},     # non-numeric
        {"category": "Box", "wnlemmas": "", "aligned.dims": "-5,10,10", "weight": "1"},  # non-positive
        {"category": "Box", "wnlemmas": "", "aligned.dims": "20000,10,10", "weight": "1"},  # 200 m
        {"category": "Box", "wnlemmas": "", "aligned.dims": "30,20,20", "weight": "oops"},  # bad weight only
    ]
    agg = m.aggregate(good + junk)
    assert agg["box"]["n"] == 4                                # 3 good + the bad-weight row
    assert agg["box"]["weight_kg"] == 1.0


def test_row_contributes_to_both_category_and_wordnet_lemmas():
    m = _mod()
    rows = [{"category": "LaptopComputer", "wnlemmas": "laptop,notebook computer",
             "aligned.dims": "34,24,3", "weight": "1.8"}] * 3
    agg = m.aggregate(rows)
    assert set(agg) == {"laptopcomputer", "laptop", "notebook computer"}
    assert agg["laptop"]["size_m"] == 0.34
