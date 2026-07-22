"""The SCATTER convention (schema.Part.scatter / construct._expand_scatter):
multiplying ONE template Part into N seeded, surface-placed, single-material
instances -- and its target application, a doughnut with frosting (conform)
and rainbow sprinkles (scatter). Deterministic, offline.

See the convention doc on ``sigma_ground.deckard.schema.Part.scatter`` for the
full rationale/schema, and ``construct._expand_scatter`` for the placement
mechanism (ray-march from the host's centre -- reuses ``_march_gap``, the
same technique conform's auto-seat already uses).
"""
import math

from sigma_ground.deckard import compile
from sigma_ground.deckard.schema import ConstructSpec, Part, Fact, ESTIMATED
from sigma_ground.deckard.sources import density_of


# ── generic scatter mechanics (host-agnostic; a sphere host is the simplest
#    surface to reason the math out on) ─────────────────────────────────────

def _sprinkled_ball(count=18, seed=3, region="top", **extra):
    ball = Part("ball", "sphere", {"radius_m": Fact(0.03)}, "steel", density_of("steel"))
    grain = Part("grain", "ellipsoid",
                 {"rx_m": Fact(0.001), "ry_m": Fact(0.001), "rz_m": Fact(0.001)},
                 "grain_a", Fact(1550.0, "t", "", 0.5),
                 scatter=dict(on="ball", count=count, seed=seed, region=region,
                              materials=["grain_a", "grain_b", "grain_c"], **extra))
    return ConstructSpec(name="sprinkled-ball", kind="composite", parts=[ball, grain])


def test_scatter_produces_exactly_the_requested_count():
    c = compile(_sprinkled_ball(count=18), resolution=28)
    # 1 host layer + 18 scattered instances, no more, no less
    assert len(c.layers) == 19
    grain_layers = [L for L in c.layers if L.name.startswith("grain_")]
    assert len(grain_layers) == 18


def test_scatter_cycles_the_full_materials_palette():
    c = compile(_sprinkled_ball(count=18, seed=11), resolution=28)
    used = {L.material for L in c.layers if L.name.startswith("grain_")}
    # with 18 instances and a 3-colour palette, a seeded draw should exercise
    # every colour at least once (not a hard guarantee for tiny N, but this
    # N/palette ratio makes an all-but-impossible flake)
    assert used == {"grain_a", "grain_b", "grain_c"}


def test_scatter_same_seed_is_bit_exact_reproducible():
    a = compile(_sprinkled_ball(count=12, seed=42), resolution=24)
    b = compile(_sprinkled_ball(count=12, seed=42), resolution=24)
    pos_a = [tuple(L.name for L in a.layers)]
    centers_a = [i.shape.center for i in _items_of(a)]
    centers_b = [i.shape.center for i in _items_of(b)]
    assert centers_a == centers_b


def test_scatter_different_seed_moves_the_instances():
    a = compile(_sprinkled_ball(count=12, seed=1), resolution=24)
    b = compile(_sprinkled_ball(count=12, seed=2), resolution=24)
    centers_a = [i.shape.center for i in _items_of(a)]
    centers_b = [i.shape.center for i in _items_of(b)]
    assert centers_a != centers_b


def _items_of(construct):
    """The scattered grain leaves' underlying CSG leaves, in insertion order
    (skips the host) -- used to compare raw placement, not just labels."""
    return [leaf for leaf, _op in construct.composed._leaves if leaf.material.startswith("grain_")]


def test_scatter_region_top_stays_in_the_upper_cap():
    c = compile(_sprinkled_ball(count=24, seed=5, region="top"), resolution=24)
    for leaf in _items_of(c):
        cx, cy, cz = leaf.shape.center
        r = math.sqrt(cx * cx + cy * cy + cz * cz)
        assert cz / r > 0.0   # every instance in the +z half, matching region="top"


def test_scatter_region_all_reaches_the_lower_hemisphere_too():
    c = compile(_sprinkled_ball(count=40, seed=5, region="all"), resolution=24)
    zs = [leaf.shape.center[2] for leaf in _items_of(c)]
    assert min(zs) < 0.0 and max(zs) > 0.0   # both hemispheres hit at N=40


def test_scatter_embed_frac_controls_how_deep_instances_sit():
    # embed_frac=0 => centres sit just OUTSIDE the host surface (fully proud);
    # embed_frac=1 => centres sit ON the host surface (fully sunk).
    proud = compile(_sprinkled_ball(count=10, seed=9, embed_frac=0.0), resolution=24)
    sunk = compile(_sprinkled_ball(count=10, seed=9, embed_frac=1.0), resolution=24)
    r_proud = [math.sqrt(sum(v * v for v in leaf.shape.center)) for leaf in _items_of(proud)]
    r_sunk = [math.sqrt(sum(v * v for v in leaf.shape.center)) for leaf in _items_of(sunk)]
    # proud instances' centres sit ~1 grain-radius further from the ball centre
    assert min(r_proud) > max(r_sunk)


def test_scatter_unknown_host_raises():
    ball = Part("ball", "sphere", {"radius_m": Fact(0.03)}, "steel", density_of("steel"))
    grain = Part("grain", "sphere", {"radius_m": Fact(0.001)}, "sugar",
                 Fact(1550.0, "t", "", 0.5), scatter=dict(on="nonexistent", count=5))
    spec = ConstructSpec(name="bad-scatter", kind="composite", parts=[ball, grain])
    try:
        compile(spec, resolution=16)
        assert False, "expected a ValueError for an unknown scatter host"
    except ValueError as e:
        assert "unknown host" in str(e)


def test_scatter_count_zero_is_a_no_op():
    ball = Part("ball", "sphere", {"radius_m": Fact(0.03)}, "steel", density_of("steel"))
    grain = Part("grain", "sphere", {"radius_m": Fact(0.001)}, "sugar",
                 Fact(1550.0, "t", "", 0.5), scatter=dict(on="ball", count=0))
    spec = ConstructSpec(name="zero-scatter", kind="composite", parts=[ball, grain])
    c = compile(spec, resolution=16)
    assert len(c.layers) == 1   # just the ball; the template contributes nothing


# ── the actual target: a doughnut with frosting (conform) + rainbow
#    sprinkles (scatter) ────────────────────────────────────────────────────

# Sprinkle size: NO citable dimension entry exists in this codebase's sources/
# package for confectionery sprinkles/jimmies (checked: dimensions.json has no
# food/confection category; a grep of sources/ for "sprinkle"/"candy"/
# "confection" returns nothing). ~3mm x ~1mm is commonly cited by candy
# suppliers for classic "jimmies", but that figure is NOT independently
# verified against a primary source this session -- flagged ESTIMATED
# (Fact.source sentinel), matching this codebase's Fact/estimated convention,
# never presented as a measurement.
_SPRINKLE_SIZE_NOTE = (
    "estimated -- no citable confectionery-sprinkle dimension exists in "
    "sources/ this session; ~3mm length x ~1mm diameter is a commonly-cited "
    "'jimmies' size (candy-supplier descriptions), NOT independently "
    "verified against a primary source here -- treat as an approximation."
)

# Icing/glaze density: no cited confectionery-icing density exists locally
# either. Icing is mostly sucrose in a thin water paste, so the cited
# crystalline-sucrose density (inventory/data/materials.json) is used as an
# approximation; real icing is somewhat less dense (entrained water/air) --
# also flagged, not measured.
_ICING_DENSITY_NOTE = (
    "approximated as cited crystalline sucrose density (inventory/data/"
    "materials.json 'Sucrose', 1550 kg/m3) -- icing/glaze is mostly sucrose "
    "in a thin water paste; real icing bulk density is somewhat lower "
    "(entrained water/air), not independently measured this session."
)

_RAINBOW = ["sprinkle_red", "sprinkle_orange", "sprinkle_yellow",
           "sprinkle_green", "sprinkle_blue", "sprinkle_violet"]


def _build_doughnut(sprinkle_count=24, include_sprinkles=True):
    """A torus dough body + a frosting shell (conform) + a scatter of
    rainbow sprinkles. Dough dimensions reuse the SAME values Deckard's own
    researcher already produced for "Doughnut" this session
    (sigma_ground/deckard/catalog/doughnut.md) rather than re-guessing a
    second, silently-different set of numbers for the same object.
    """
    dough = Part(
        "dough", "torus",
        {"major_radius_m": Fact(0.03763, ESTIMATED, "", 0.55),
         "minor_radius_m": Fact(0.007526, ESTIMATED, "", 0.55)},
        "leavened dough", Fact(1000.0, ESTIMATED, "", 0.2),
    )

    # frosting: a CONCENTRIC, slightly-fatter torus that CONFORMS to the
    # dough -- the EXISTING mechanism (construct._make_conform) carves away
    # the overlap, leaving a uniform coat shell over the whole dough torus.
    # Simplification, flagged: real doughnuts are icing-dipped mostly on
    # TOP, not all the way around -- a directional partial coat is a v2
    # extension (needs a _Clipped-style half-space cut, not built this
    # session), same "flag the roadmap" discipline as research._tree_spec.
    coat_m = 0.0015   # 1.5mm icing coat -- ESTIMATE, no cited source
    frosting = Part(
        "frosting", "torus",
        {"major_radius_m": Fact(0.03763, ESTIMATED, "", 0.55),
         "minor_radius_m": Fact(0.007526 + coat_m, ESTIMATED, "1.5mm coat, ESTIMATE", 0.3)},
        "icing", Fact(1550.0, ESTIMATED, _ICING_DENSITY_NOTE, 0.3),
        conform="dough",
    )

    parts = [dough, frosting]
    if include_sprinkles:
        sprinkle = Part(
            "sprinkle", "ellipsoid",
            {"rx_m": Fact(0.0005, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2),
             "ry_m": Fact(0.0005, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2),
             "rz_m": Fact(0.0015, ESTIMATED, _SPRINKLE_SIZE_NOTE, 0.2)},
            _RAINBOW[0], density_of("sucrose"),
            scatter=dict(on="frosting", count=sprinkle_count, materials=_RAINBOW,
                        seed=7, region="top", embed_frac=0.4, size_jitter=0.15),
        )
        parts.append(sprinkle)

    return ConstructSpec(
        name="doughnut with frosting and rainbow sprinkles", kind="composite",
        identified=True, parts=parts,
        sources=[
            {"name": "sigma_ground/deckard/catalog/doughnut.md (this session's "
                     "own Deckard research) -- dough torus dimensions reused "
                     "verbatim", "license": ""},
            {"name": "inventory/data/materials.json 'Sucrose' -- cited icing/"
                     "sprinkle density basis", "license": ""},
            {"name": _SPRINKLE_SIZE_NOTE, "license": ""},
            {"name": _ICING_DENSITY_NOTE, "license": ""},
        ],
        notes=("Frosting is a full uniform coat (all the way around the torus, "
              "not just the top) -- a documented simplification, see the "
              "frosting Part's construction comment for the v2 roadmap. "
              "Sprinkles are a SCATTER of 1 template ellipsoid across the "
              "frosting's top-facing region, materials cycling a 6-colour "
              "'rainbow' palette (see schema.Part.scatter's density note: "
              "every instance shares the template's cited sucrose density; "
              "only the colour LABEL varies per instance)."),
    )


def test_frosting_conforms_to_the_dough_torus():
    spec = _build_doughnut(include_sprinkles=False)
    c = compile(spec, resolution=48)
    assert c.validation["mode"] == "conforming" and c.validation["passed"]
    conforms = c.validation.get("conforms") or []
    assert any(x["part"] == "frosting" and x["to"] == "dough" for x in conforms)
    # the frosting yielded real matter to the dough -- a genuine congruent
    # interface, not a zero-area point-touch (would be flagged "unmated")
    assert not c.validation.get("unmated")
    # frosting mass is small relative to the dough (a thin coat, not a
    # second doughnut's worth of icing)
    dough_layer = next(L for L in c.layers if L.name == "dough")
    frosting_layer = next(L for L in c.layers if L.name == "frosting")
    assert 0.0 < frosting_layer.mass_kg < dough_layer.mass_kg


def test_full_doughnut_compiles_with_sane_mass_and_bbox():
    spec = _build_doughnut(sprinkle_count=24)
    c = compile(spec, resolution=40)
    assert c.validation["passed"], c.validation["note"]

    # part count: 1 dough + 1 frosting + 24 sprinkles
    assert len(c.layers) == 26
    sprinkle_layers = [L for L in c.layers if L.name.startswith("sprinkle_")]
    assert len(sprinkle_layers) == 24
    assert {L.material for L in sprinkle_layers} == set(_RAINBOW)

    # sane mass: a real doughnut is roughly 40-60 g; a torus this size at
    # ~1000 kg/m3 (dough) plus a thin sugar coat should land in a physically
    # plausible ballpark, not zero and not kilograms
    assert 0.005 < c.mass_kg < 0.5

    # sane bounding size: overall diameter on the order of the dough torus
    # (major+minor)*2 -- a few centimetres, not metres or micrometres
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    span_x, span_y, span_z = x1 - x0, y1 - y0, z1 - z0
    assert 0.05 < span_x < 0.15
    assert 0.05 < span_y < 0.15
    assert 0.01 < span_z < 0.05

    # centre of mass stays near the torus's own symmetry axis (on-axis
    # object; the seeded sprinkle scatter shouldn't drag it off meaningfully)
    cx, cy, cz = c.com_m
    assert math.sqrt(cx * cx + cy * cy) < 0.01
