"""Head-noun tie-break between equally specific matches.

"a desk lamp" fits two ShapeNetSem keys ('desk', 'lamp') and two exemplar categories (the tables
alias 'desk'; the lamps are 'lamp') with ONE word each. The first-listed one used to win, so a
desk lamp got the DESK's size (1.27 x 0.77 x 0.67 m), the desk's wood composition, and (by file
order) a lamp layout stretched to a desk: 1.33 m tall, 99 kg. English noun compounds put the head
last: a desk LAMP is a lamp. Offline.
"""
import pytest

from sigma_ground.deckard import compile
from sigma_ground.deckard.research import _exemplar_spec
from sigma_ground.deckard.sources import aliases, composition, exemplar, shapenetsem, sizes

_LAMP_IDS = {"14982", "13639", "13794", "13889", "13963", "16690", "16926", "17433"}


def test_head_noun_helpers():
    assert aliases.head_noun("a desk lamp") == "lamp"
    assert aliases.head_noun("The Coffee-Table") == "table"
    assert aliases.head_noun("") == "" and aliases.head_noun(None) == ""
    assert aliases.holds_head({"desk", "lamp"}, "lamp")
    assert aliases.holds_head({"lamp"}, "lamps") and aliases.holds_head({"box"}, "boxes")
    assert not aliases.holds_head({"desk"}, "lamp")
    assert not aliases.holds_head({"desk"}, "")


def test_a_desk_lamp_has_a_lamps_size_and_materials_not_a_desks():
    lamp = shapenetsem.dims_of("a lamp")[0]
    desk = shapenetsem.dims_of("a desk")[0]
    assert shapenetsem.dims_of("a desk lamp")[0] == lamp != desk
    mats = shapenetsem.materials_of("a desk lamp")[0]
    assert mats.get("wood", 0.0) < 0.99                                  # the desk is 99.7% wood
    assert mats == shapenetsem.materials_of("a lamp")[0]
    assert shapenetsem.size_of("a desk lamp")[0] == shapenetsem.size_of("a lamp")[0]


def test_a_desk_lamp_pools_the_lamps_not_the_tables():
    pool = exemplar._category_pool("a desk lamp")
    assert pool and {e[4] for e in pool} == _LAMP_IDS
    # ... and "a DIFFERENT desk lamp" can never hand back a table, however many are excluded
    for k in range(1, 9):
        got = exemplar.exemplar_of("a desk lamp", exclude=sorted(_LAMP_IDS)[:k])
        assert got is not None and got[3] in _LAMP_IDS


def test_the_same_tie_break_in_composition_and_sizes():
    assert composition.composition_of("a desk lamp") == composition.composition_of("a lamp")
    assert composition.composition_of("a desk lamp") != composition.composition_of("a desk")
    assert sizes._local("a desk lamp") == sizes._local("a lamp")


def test_the_compiled_desk_lamp_is_a_lamp():
    spec = _exemplar_spec("a desk lamp")
    assert spec is not None and spec.identified
    assert any(s.get("anno_id") in _LAMP_IDS for s in spec.sources)
    c = compile(spec, resolution=40)
    ext = [abs(b[1] - b[0]) for b in c.bbox]
    assert max(ext) < 1.0                                                # was 1.33 m
    assert c.mass_kg < 25.0                                              # was 99 kg


def test_unambiguous_names_resolve_as_before():
    chair = {e[4] for e in exemplar._category_pool("a chair")}
    assert {e[4] for e in exemplar._category_pool("a wooden chair")} == chair
    table = {e[4] for e in exemplar._category_pool("a table")}
    assert {e[4] for e in exemplar._category_pool("a coffee table")} == table      # 'coffee table' is no exemplar key
    assert exemplar._category_pool("a desk") and {e[4] for e in exemplar._category_pool("a desk")} == table
    assert shapenetsem.dims_of("a coffee table")[0] == shapenetsem.dims_of("coffee table")[0]
    assert shapenetsem.dims_of("a wooden chair")[0] == shapenetsem.dims_of("a chair")[0]
    # the longer key still beats a shorter one that holds the head ('coffee table' > 'table')
    assert len(shapenetsem._words("coffee table")) == 2
    assert shapenetsem._entry("a coffee table") is shapenetsem._doc()["categories"]["coffee table"]


# ── the rule itself, on a synthetic table (no dependence on the shipped data) ──────────────────
def _fake_doc(*keys):
    return {"categories": {k: {"n": 9, "size_m": 1.0, "dims_m": [1, 1, 1], "marker": k} for k in keys}}


@pytest.mark.parametrize("query, expect", [
    ("alpha beta", "beta"),            # both fit with one word; the head noun (beta) wins although listed second
    ("beta alpha", "alpha"),
    ("alpha beta gamma", "alpha"),     # neither holds the head: first-listed, exactly as before
])
def test_tie_break_on_a_synthetic_table(monkeypatch, query, expect):
    monkeypatch.setattr(shapenetsem, "_doc", lambda: _fake_doc("alpha", "beta"))
    assert shapenetsem._entry(query)["marker"] == expect


def test_a_longer_key_beats_the_head_noun(monkeypatch):
    monkeypatch.setattr(shapenetsem, "_doc", lambda: _fake_doc("beta", "alpha gamma"))
    assert shapenetsem._entry("alpha gamma beta")["marker"] == "alpha gamma"          # 2 words > 1 word holding the head
