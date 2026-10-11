"""Catalog aliases match WHOLE WORDS, never substrings.

The holodeck asks Deckard for "a clothes cupboard" when "a wardrobe" is unknown. The curated
alias "cup" used to match the letters c-u-p inside "cupboard", so a wardrobe was resolved to the
frozen coffee-cup spec (8 cm, 0.6 kg, identified, audit verdict 'verified'). Offline.
"""
import pytest

from sigma_ground.deckard import catalog, compile, research
from sigma_ground.deckard.schema import ConstructSpec
from sigma_ground.deckard.sources import exemplar


@pytest.mark.parametrize("name", [
    "a clothes cupboard", "a cupboard", "cupboard", "a cupcake", "a hiccup", "a mugshot",
    "a teacupboard", "a cupola", "a wardrobe", "a bookcase", "a tv",
])
def test_an_alias_is_not_found_inside_another_word(name):
    assert catalog.slug_for(name) is None


@pytest.mark.parametrize("name", [
    "a ceramic mug", "a teacup", "coffee cup", "a big coffee mug", "the cup", "a cup of coffee",
    "two mugs", "three coffee cups", "A CERAMIC MUG", "cup", "mug",
])
def test_whole_word_aliases_still_hit(name):
    assert catalog.slug_for(name) == "coffee_cup"


def test_only_the_last_word_of_an_alias_may_be_plural():
    assert catalog.slug_for("two coffee mugs") == "coffee_cup"
    assert catalog._alias_in("coffee cup", {"coffee", "cups"})
    assert not catalog._alias_in("coffee cup", {"coffees", "cup"})       # 'coffee' must be whole and singular
    assert not catalog._alias_in("cup", {"cupboard"})
    assert not catalog._alias_in("mug", {"mugshot"})


def test_a_cupboard_is_not_in_the_catalog():
    assert catalog.lookup("a cupboard") is None
    assert catalog.lookup("a clothes cupboard") is None
    assert catalog.lookup("a ceramic mug") is not None          # the real hit is unchanged


def test_a_clothes_cupboard_is_researched_as_a_cupboard_not_a_coffee_cup():
    spec = research("a clothes cupboard", allow_llm=False)
    assert isinstance(spec, ConstructSpec)
    assert spec.identified and spec.kind == "composite"          # a real cabinet layout, not the cup kit
    c = compile(spec, resolution=40)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    assert max(x1 - x0, y1 - y0, z1 - z0) > 0.8                  # furniture-sized (the cup was 0.095 m)
    assert c.mass_kg > 20.0                                      # not the 0.6 kg cup


def test_the_frozen_broken_bookcase_is_gone_and_a_bookcase_resolves_live():
    """catalog/a_bookcase.md (a qwen entry: eight 2.9 cm planks, 0.42 x 1.30 x 0.08 m, 4 kg,
    lying on its back) is retired. A frozen entry would shadow the live path forever; the cabinet
    exemplar (alias 'bookcase') gives a real layout at the ShapeNetSem size."""
    assert "a_bookcase" not in catalog.available()
    assert catalog.lookup("a bookcase") is None
    spec = research("a bookcase", allow_llm=False)
    assert spec.identified and spec.kind == "composite"
    c = compile(spec, resolution=40)
    ext = sorted(abs(b[1] - b[0]) for b in c.bbox)
    assert ext[0] > 0.2 and 1.0 < ext[2] < 1.6                   # deep enough to be a case, 1-1.6 m tall
    assert 10.0 < c.mass_kg < 150.0                              # 30-50 kg real; 4 kg was the bug


def test_the_curated_catalog_entries_are_untouched():
    for slug in ("coffee_cup", "hammer", "anvil", "a_dumbbell", "feather"):
        assert slug in catalog.available()


# ── the exemplar aliases (data): names the holodeck asks for that PartNet files under another name ──
_ALIASES = {
    "cabinet": ["bookcase", "bookshelf", "book shelf", "wardrobe", "closet", "armoire", "shelf"],
    "monitor": ["tv", "television"],
    "vase": ["plant", "houseplant", "potted plant", "flower pot"],
}


@pytest.mark.parametrize("category, aliases", sorted(_ALIASES.items()))
def test_every_variant_of_a_category_carries_its_aliases(category, aliases):
    """The pool is chosen file by file from each file's own alias list: one file without the alias
    would silently drop out of the variant pool."""
    import json
    files = sorted(exemplar._DIR.glob(f"{category}*.json"))
    assert len(files) >= 8
    for f in files:
        have = json.loads(f.read_text(encoding="utf-8"))["aliases"]
        for a in aliases:
            assert a in have, f"{f.name} lacks the alias {a!r}"


@pytest.mark.parametrize("name, category", [
    (f"a {a}", cat) for cat, al in sorted(_ALIASES.items()) for a in al])
def test_the_alias_names_pool_the_right_category(name, category):
    pool = exemplar._category_pool(name)
    assert pool and all(category in e[0] for e in pool)
    assert len(pool) >= 8                                       # the whole variant pool, not one file
    got = exemplar.exemplar_of(name)
    assert got is not None and got[3] in {e[4] for e in pool}


def test_an_alias_does_not_steal_a_longer_or_head_matching_name():
    """'a closet door' is a door (head noun), not a cabinet; 'a plant pot' is still a pot."""
    assert {e[4] for e in exemplar._category_pool("a closet door")} == {e[4] for e in exemplar._category_pool("a door")}
    assert {e[4] for e in exemplar._category_pool("a plant pot")} == {e[4] for e in exemplar._category_pool("a pot")}
