"""The ShapeNetSem-distilled source — measured-world size + material grounding.

Runs against the SHIPPED fact-table (inventory/data/shapenetsem_sizes.json),
fully offline. Covers: name resolution (exact / word-containment), plausible
human-scale sizes, the typical_size_of fallback chain (curated wins, Sem fills
the long tail), material priors, and the researcher's material hint line.
"""
from sigma_ground.deckard.sources import shapenetsem, typical_size_of


def test_distilled_table_resolves_common_objects_at_plausible_sizes():
    for name, lo, hi in (("chair", 0.5, 1.5), ("couch", 1.5, 3.0),
                         ("mug", 0.05, 0.3), ("table", 0.6, 2.0)):
        got = shapenetsem.size_of(name)
        assert got is not None, name
        size, src, lic = got
        assert lo <= size <= hi, f"{name}: {size} m outside [{lo}, {hi}]"
        assert "ShapeNetSem" in src and "median of" in src
        assert "non-commercial" in lic


def test_word_containment_matches_phrases():
    direct = shapenetsem.size_of("bookcase")
    phrased = shapenetsem.size_of("a tall wooden bookcase")
    assert direct is not None and phrased is not None
    assert phrased[0] == direct[0]                       # same entry via containment


def test_typical_size_of_falls_back_to_shapenetsem():
    # 'bookcase' is NOT in the curated object_sizes table -> Sem supplies it
    got = typical_size_of("bookcase")
    assert got is not None and "ShapeNetSem" in got[1]
    # the curated table still wins where it has an answer (toaster, 0.30 m)
    got = typical_size_of("toaster")
    assert got is not None and abs(got[0] - 0.30) < 1e-9


def test_material_priors_and_hint():
    mats, src, lic = shapenetsem.materials_of("mug")
    assert mats.get("ceramic", 0) > 0.5                  # mugs are mostly ceramic
    h = shapenetsem.hint("mug")
    assert "ceramic" in h and "ShapeNetSem" in h
    assert shapenetsem.hint("zxqwerty contraption") == ""    # unknown -> no hint


def test_sem_material_density_vocabulary():
    d = shapenetsem.density_of("ceramic")
    assert d is not None and 1500 < d < 4000             # kg/m3, converted from g/cm3
    assert shapenetsem.density_of("unobtanium") is None
