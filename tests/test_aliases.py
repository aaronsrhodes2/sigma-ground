"""WordNet alias expansion (ShapeNet taxonomy) widens the sources' miss paths.

Runs against the SHIPPED shape_aliases.json plus synthetic monkeypatched group
tables. A query phrased with one lemma resolves entries keyed under another.
"""
import json

from sigma_ground.deckard.sources import aliases, shapenetsem, typical_size_of


def test_shipped_table_expands_common_lemma_groups():
    got = aliases.expand("cellphone")
    assert "cellular telephone" in got or "mobile phone" in got
    assert aliases.expand("zxqwerty gizmo") == set()
    assert "cellphone" not in aliases.expand("cellphone")     # never itself


def test_articles_are_normalized():
    assert aliases.expand("a cellphone") == aliases.expand("cellphone")


def test_sem_entry_resolves_via_alias():
    # 'sofa' and 'couch' are one WordNet group; the Sem table knows 'couch'
    direct = shapenetsem.size_of("couch")
    via_alias = shapenetsem.size_of("sofa")
    assert direct is not None
    assert via_alias is not None and via_alias[0] == direct[0]


def test_typical_size_falls_through_aliases(monkeypatch):
    p = aliases._JSON
    monkeypatch.setattr(aliases, "_JSON", p)                  # shipped table
    got = typical_size_of("lounge")                           # couch group lemma
    assert got is not None
