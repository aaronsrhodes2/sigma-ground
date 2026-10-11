"""An exemplar alias identifies a name only as its HEAD noun, never as a modifier.

The first repair added bare nouns (plant, houseplant, tv, television, shelf, bookshelf, closet, wardrobe,
armoire) as aliases of the vase / monitor / cabinet exemplars, and the lookup matched an alias on ANY word of the
query: 'a power plant' came back as a 62 kg flowerpot, 'a tv stand' as a 49 kg TV, 'a shelf bracket' as a
13-part cabinet, 'a tea pot' as an 83 kg flowerpot, all identified=True. Now the alias (or category name) must
name the phrase's head noun (aliases.names_head), and the aliases that are the head of well-known OTHER things
(power PLANT, tea POT, water CLOSET, heart MONITOR, bottle CAP, tea BAG ...) also need neutral company.

And a compound resolves ONE object: layout, size and materials come from the same exemplar category, so 'a bed
frame' is a bed (never the 32 cm picture frame the 'frame' ShapeNetSem key is) and 'a closet door' a door
(never closet-sized). Offline.
"""
import importlib

import pytest

from sigma_ground.deckard import compile, research
from sigma_ground.deckard.sources import aliases, exemplar, shapenetsem

R = importlib.import_module("sigma_ground.deckard.research")

# ── names that must NOT identify an exemplar category ─────────────────────────────────────────
FALSE_FRIENDS = [
    # plant / houseplant / potted plant
    "a power plant", "a nuclear power plant", "a chemical plant", "a plant cell", "a plant stand",
    "a potted plant stand", "a houseplant fertilizer", "a houseplant pest",
    # tv / television / monitor / screen / display
    "a tv stand", "a tv remote", "a tv antenna", "a tv dinner", "a tv guide", "a tv tray", "a television stand",
    "a television antenna", "a monitor stand", "a monitor lizard", "a heart monitor", "a baby monitor",
    "a sun screen", "a window screen", "a projector screen", "a display case", "a display stand",
    # shelf / bookshelf / closet / wardrobe / armoire / dresser / cupboard
    "a shelf bracket", "a shelf life", "a bookshelf speaker", "a bookshelf stereo", "a wardrobe trunk",
    "a wardrobe malfunction", "a closet organizer", "a closet rod", "a water closet", "an armoire key",
    "a dresser drawer",
    # pot
    "a tea pot", "a coffee pot", "a pot roast", "a pot belly stove", "a pot holder", "a chamber pot",
    "a cooking pot", "a hot pot", "a jam pot", "a pot pie", "a flower pot stand",
    # accessories / parts of an exemplar category
    "a chair cushion", "a chair leg", "a chair lift", "a door handle", "a door knob", "a door mat",
    "a bed bug", "a bed sheet", "a bed linen", "a table cloth", "a table salt", "a table tennis ball",
    "a desk calendar", "a desk pad", "a lamp shade", "a lamp post", "a lamp oil", "a clock tower",
    "a mug shot", "a knife block", "a knife edge", "a bowl cut", "a hat rack", "a hat trick",
    "a fridge magnet", "a trash can lid", "a bin liner", "a tap dance", "a tap root", "a bottle cap",
    "a cap gun", "a tea bag", "a bean bag", "a sleeping bag", "a bag of chips", "a punching bag",
    "a laptop bag", "a laptop stand", "a microwave popcorn", "a scissors lift", "a dishwasher detergent",
    "a vase handle", "an earphone case", "a bottle opener", "a bottle rack", "a faucet handle",
]

# ── names that MUST identify it (and which category) ──────────────────────────────────────────
TRUE_MATCHES = [
    ("a plant", "vase"), ("a houseplant", "vase"), ("a potted plant", "vase"), ("a house plant", "vase"),
    ("a green plant", "vase"), ("a small plant", "vase"), ("a large potted plant", "vase"),
    ("an indoor plant", "vase"), ("an artificial plant", "vase"), ("a fern plant", "vase"),
    ("a pot", "vase"), ("a flower pot", "vase"), ("a plant pot", "vase"), ("a clay pot", "vase"),
    ("a terracotta pot", "vase"), ("a ceramic flower pot", "vase"), ("a vase", "vase"), ("a flower vase", "vase"),
    ("a tv", "monitor"), ("a television", "monitor"), ("a tv set", "monitor"), ("a television set", "monitor"),
    ("a flat screen tv", "monitor"), ("a big tv", "monitor"), ("a monitor", "monitor"),
    ("a computer monitor", "monitor"), ("a pc monitor", "monitor"), ("a display", "monitor"),
    ("a screen", "monitor"), ("a computer screen", "monitor"),
    ("a shelf", "cabinet"), ("a bookshelf", "cabinet"), ("a bookcase", "cabinet"), ("a book shelf", "cabinet"),
    ("a wooden bookshelf", "cabinet"), ("a tall bookcase", "cabinet"), ("a closet", "cabinet"),
    ("a wardrobe", "cabinet"), ("a clothes closet", "cabinet"), ("a walk-in closet", "cabinet"),
    ("a built-in wardrobe", "cabinet"), ("an armoire", "cabinet"), ("a wooden wardrobe", "cabinet"),
    ("a clothes cupboard", "cabinet"), ("a cupboard", "cabinet"), ("a dresser", "cabinet"),
    ("a cabinet", "cabinet"), ("a storage furniture", "cabinet"), ("a bathroom cabinet", "cabinet"),
    ("a bed", "bed"), ("a bed frame", "bed"), ("a metal bed frame", "bed"), ("a bunk bed", "bed"),
    ("a chair", "chair"), ("a wooden chair", "chair"), ("an office chair", "chair"), ("a chair with arms", "chair"),
    ("a table", "table"), ("a desk", "table"), ("a coffee table", "table"),
    ("a lamp", "lamp"), ("a desk lamp", "lamp"), ("a floor lamp", "lamp"),
    ("a refrigerator", "refrigerator"), ("a fridge", "refrigerator"), ("a fridge freezer", "refrigerator"),
    ("a microwave", "microwave"), ("a microwave oven", "microwave"),
    ("a laptop", "laptop"), ("a laptop computer", "laptop"),
    ("a trash can", "trash can"), ("a garbage can", "trash can"), ("a trash bin", "trash can"), ("a bin", "trash can"),
    ("a door", "door"), ("a closet door", "door"), ("a cupboard door", "door"),
    ("a mug", "mug"), ("a coffee mug", "mug"), ("a pair of scissors", "scissors"),
    ("a hat", "hat"), ("a baseball cap", "hat"), ("a handbag", "bag"), ("a leather handbag", "bag"),
    ("a shoulder bag", "bag"), ("a headphone", "earphone"), ("headphones", "earphone"),
    ("a bottle", "bottle"), ("a bowl", "bowl"), ("a clock", "clock"), ("a kitchen tap", "faucet"),
    ("a dishwasher", "dishwasher"), ("a knife", "knife"),
]


def test_the_lists_are_big_enough():
    assert len(FALSE_FRIENDS) >= 40 and len(TRUE_MATCHES) >= 40
    assert len(set(FALSE_FRIENDS)) == len(FALSE_FRIENDS) and len({n for n, _ in TRUE_MATCHES}) == len(TRUE_MATCHES)
    assert not set(FALSE_FRIENDS) & {n for n, _ in TRUE_MATCHES}


@pytest.mark.parametrize("name", FALSE_FRIENDS)
def test_a_false_friend_is_not_identified_as_the_alias(name):
    assert exemplar.category_of(name) is None, f"{name!r} was taken for a {exemplar.category_of(name)!r}"
    assert exemplar._category_pool(name) == []
    assert exemplar.exemplar_of(name) is None


@pytest.mark.parametrize("name, category", TRUE_MATCHES)
def test_a_true_match_pools_its_category(name, category):
    assert exemplar.category_of(name) == category
    pool = exemplar._category_pool(name)
    assert pool and all(e[6] == category for e in pool)
    assert exemplar.exemplar_of(name) is not None


@pytest.mark.parametrize("name", ["a power plant", "a plant stand", "a plant cell", "a tv stand", "a tv remote",
                                  "a shelf bracket", "a tea pot", "a pot roast", "a pot belly stove",
                                  "a bookshelf speaker", "a wardrobe trunk", "a water closet"])
def test_the_audit_cases_are_not_identified_through_research(name):
    spec = research(name, allow_llm=False)
    assert not spec.identified, f"{name}: identified={spec.identified}, kind={spec.kind}"
    assert not any("representative real model" in s.get("name", "") for s in spec.sources)


def test_the_head_noun_helpers():
    assert aliases.head_noun("a chair with arms") == "chair"
    assert aliases.head_noun("a pair of scissors") == "scissors" and aliases.head_noun("a set of red chairs") == "chairs"
    assert aliases.head_noun("a walk-in closet") == "closet"
    assert aliases.head_noun("a lamp, tall") == "lamp"
    assert aliases.names_head({"tv"}, "a tv set") and not aliases.names_head({"tv"}, "a tv stand")
    assert aliases.names_head({"plant"}, "a potted plant") and not aliases.names_head({"plant"}, "a plant cell")
    assert aliases.names_head({"chair"}, "two wooden chairs")
    assert not aliases.names_head(set(), "") and not aliases.names_head({"x"}, "")


def test_a_polysemous_alias_needs_neutral_company():
    assert exemplar.category_of("a small green plant") == "vase"
    assert exemplar.category_of("a power plant") is None and exemplar.category_of("a steel plant") is None
    assert exemplar.category_of("a pot") == "vase" and exemplar.category_of("a tea pot") is None
    assert exemplar.category_of("a closet") == "cabinet" and exemplar.category_of("a water closet") is None


# ── one object: size and materials follow the layout that was chosen ──────────────────────────
def _extent(c):
    return sorted(abs(b[1] - b[0]) for b in c.bbox)


@pytest.mark.parametrize("compound, base", [("a bed frame", "a bed"), ("a metal bed frame", "a bed"),
                                            ("a microwave oven", "a microwave"), ("a laptop computer", "a laptop"),
                                            ("a fridge freezer", "a fridge"), ("a potted plant", "a plant")])
def test_a_compound_alias_is_sized_as_its_category(compound, base):
    a, b = R._object_dims(compound), R._object_dims(base)
    assert a is not None and b is not None
    assert a[0] == b[0], f"{compound} {a[0]} vs {base} {b[0]}"


def test_a_bed_frame_is_not_a_picture_frame():
    anchor = exemplar.anchor_words("a bed frame")
    assert anchor == {"bed"}
    assert shapenetsem.dims_of("a bed frame", anchor=anchor)[0] == shapenetsem.dims_of("a bed")[0]
    assert shapenetsem.dims_of("a bed frame", anchor=anchor)[0] != shapenetsem.dims_of("a frame")[0]
    c = compile(research("a bed frame", allow_llm=False), resolution=32)
    d = compile(research("a bed", allow_llm=False), resolution=32)
    assert max(_extent(c)) > 1.5 and _extent(c) == pytest.approx(_extent(d))      # was 0.32 x 0.15 x 0.33 m, 0.4 kg
    assert c.mass_kg == pytest.approx(d.mass_kg)


def test_a_door_is_a_door_whatever_it_belongs_to():
    extents = {}
    for n in ("a door", "a closet door", "a cupboard door"):
        c = compile(research(n, allow_llm=False), resolution=32)
        extents[n] = _extent(c)
        assert 1.8 < max(_extent(c)) < 2.4, n                                    # an 80 in leaf, not closet-sized or 0.3 m
        assert 0.7 < _extent(c)[1] < 1.1, n
        assert 10.0 < c.mass_kg < 60.0, n                                        # was 82.6 kg closet-sized
    assert extents["a closet door"] == pytest.approx(extents["a door"])
    spec = research("a closet door", allow_llm=False)
    assert any("standard interior door" in s.get("name", "") for s in spec.sources)       # the size is cited


def test_an_accessory_compound_keeps_its_own_head_noun():
    """'a chair frame' used to be the chair archetype built at the 'frame' size, 'a monitor stand' a monitor at the
    'stand' size. They are not the exemplar object at all now, and are flagged, never a confident fake."""
    for name in ("a chair frame", "a monitor stand", "a laptop stand", "a plant stand", "a tv stand"):
        spec = research(name, allow_llm=False)
        assert not spec.identified, name
        assert not any("representative real model" in s.get("name", "") for s in spec.sources), name
