"""Tests for ecs.py — generational handles, mandatory provenanced Matter,
deterministic slot reuse.

Plain pytest functions, matching sigma_ground/field/interface/
test_propulsion.py style. Backend-free: no deckard build — a tiny stub
construct (mass_kg/com_m/layers/bbox/source, duck-typed exactly like
matter_from_construct consumes them) is all the matter factory needs.
"""
import pytest

from sigma_ground.engine.ecs import (
    LEDGER_KEYS,
    PROVENANCE_TAGS,
    ROOM_TEMPERATURE_K,
    Binding,
    GameTags,
    Geometry,
    Matter,
    StaleEntityError,
    World,
    matter_from_construct,
)


# ── the stub construct (duck-typed; no deckard import) ─────────────────────

class _StubLayer:
    def __init__(self, name, material, mass_kg, source):
        self.name = name
        self.material = material
        self.mass_kg = mass_kg
        self.source = source


class _StubConstruct:
    mass_kg = 1.25
    com_m = (0.0, 0.0, 0.05)
    bbox = ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1))
    source = "stub: hand-set values for unit tests"
    layers = [_StubLayer("body", "stub_stuff", 1.25, "stub source")]


def _matter():
    return matter_from_construct("stub", _StubConstruct())


# ═══════════════════════════════════════════════════════════════════════════
# § 1. create / get roundtrip and the mandatory-Matter rule
# ═══════════════════════════════════════════════════════════════════════════

def test_create_get_roundtrip():
    w = World()
    m = _matter()
    eid = w.create("stub", m, geometry=Geometry(_StubConstruct(),
                                                _StubConstruct.bbox))
    ent = w.get(eid)
    assert ent.label == "stub"
    assert ent.matter is m
    assert ent.geometry.bbox == ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1))
    assert ent.binding is None
    assert isinstance(ent.tags, GameTags) and ent.tags.tags == ()
    assert w.alive(eid)
    assert len(w) == 1


def test_matter_is_mandatory():
    w = World()
    with pytest.raises(TypeError):
        w.create("no-matter")               # missing positional Matter
    with pytest.raises(TypeError):
        w.create("bad-matter", {"mass_kg": 1.0})   # a dict is not Matter


# ═══════════════════════════════════════════════════════════════════════════
# § 2. provenance enforcement — unprovenanced Matter cannot exist
# ═══════════════════════════════════════════════════════════════════════════

def _matter_kwargs():
    """A valid Matter's constructor kwargs, ready to be perturbed."""
    m = _matter()
    return dict(mass_kg=m.mass_kg, com_m=m.com_m, composition=m.composition,
                temperature_k=m.temperature_k,
                internal_energy_j=m.internal_energy_j,
                energy_ledger=dict(m.energy_ledger),
                provenance=dict(m.provenance))


def test_provenance_enforced():
    # baseline sanity: the unperturbed kwargs construct fine
    Matter(**_matter_kwargs())

    # missing provenance key
    kw = _matter_kwargs()
    del kw["provenance"]["mass_kg"]
    with pytest.raises(ValueError):
        Matter(**kw)

    # EXTRA provenance key (over-disclosure of a nonexistent field is also
    # a coverage error — the key set is exact)
    kw = _matter_kwargs()
    kw["provenance"]["colour"] = "FACT: not a Matter field"
    with pytest.raises(ValueError):
        Matter(**kw)

    # untagged provenance string
    kw = _matter_kwargs()
    kw["provenance"]["mass_kg"] = "measured somewhere, trust me"
    with pytest.raises(ValueError):
        Matter(**kw)

    # wrong ledger key set
    kw = _matter_kwargs()
    del kw["energy_ledger"]["thermal"]
    with pytest.raises(ValueError):
        Matter(**kw)
    kw = _matter_kwargs()
    kw["energy_ledger"]["magic"] = 0.0
    with pytest.raises(ValueError):
        Matter(**kw)

    # nonpositive mass / temperature
    kw = _matter_kwargs()
    kw["mass_kg"] = 0.0
    with pytest.raises(ValueError):
        Matter(**kw)
    kw = _matter_kwargs()
    kw["temperature_k"] = -1.0
    with pytest.raises(ValueError):
        Matter(**kw)


def test_matter_from_construct_tags():
    m = _matter()
    assert m.mass_kg == 1.25
    assert m.com_m == (0.0, 0.0, 0.05)
    assert m.composition == (("body", "stub_stuff", 1.25, "stub source"),)
    assert m.temperature_k == ROOM_TEMPERATURE_K
    # facts on the measured/derived fields
    assert m.provenance["mass_kg"].startswith("FACT:")
    assert m.provenance["com_m"].startswith("FACT:")
    assert m.provenance["composition"].startswith("FACT:")
    # temperature is a REAL Choice object's cite() — "CHOSEN: ..." exactly
    assert m.provenance["temperature_k"].startswith("CHOSEN:")
    assert m.provenance["internal_energy_j"].startswith("CHOSEN:")
    # ledger: kinetic is a Deduction, the model-scope zeros are Facts
    assert m.provenance["energy_ledger.kinetic"].startswith("DEDUCTION:")
    for key in ("rotational", "thermal", "deformation", "chemical"):
        assert m.provenance[f"energy_ledger.{key}"].startswith("FACT:")
    # exact key coverage: every field + every ledger channel, nothing more
    expected = {"mass_kg", "com_m", "composition", "temperature_k",
                "internal_energy_j"} | {f"energy_ledger.{k}"
                                        for k in LEDGER_KEYS}
    assert set(m.provenance) == expected
    for text in m.provenance.values():
        assert text.startswith(PROVENANCE_TAGS)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. generational invalidation — the reason the ECS exists
# ═══════════════════════════════════════════════════════════════════════════

def test_remove_invalidates_handle():
    w = World()
    eid = w.create("stub", _matter())
    w.remove(eid)
    assert not w.alive(eid)
    with pytest.raises(StaleEntityError):
        w.get(eid)
    assert len(w) == 0


def test_slot_reuse_bumps_generation():
    w = World()
    old = w.create("first", _matter())
    w.remove(old)
    new = w.create("second", _matter())
    # LIFO reuse: same slot index, HIGHER generation
    assert new.index == old.index
    assert new.generation == old.generation + 1
    # the old handle stays dead even though the slot is live again
    with pytest.raises(StaleEntityError):
        w.get(old)
    assert not w.alive(old)
    assert w.get(new).label == "second"


def test_double_remove_raises():
    w = World()
    eid = w.create("stub", _matter())
    w.remove(eid)
    with pytest.raises(StaleEntityError):
        w.remove(eid)


def test_forged_out_of_range_handle_raises():
    from sigma_ground.engine.ecs import EntityId
    w = World()
    w.create("stub", _matter())
    with pytest.raises(StaleEntityError):
        w.get(EntityId(99, 0))              # index never allocated
    with pytest.raises(StaleEntityError):
        w.get(EntityId(0, 7))               # generation never minted
    assert not w.alive(EntityId(99, 0))


def test_bound_entity_remove_refused():
    # Backend-bound matter cannot be deleted behind the backend's back —
    # release is a Region operation (and StackSimRegion honestly has none).
    w = World()
    eid = w.create("stub", _matter(), binding=Binding("stacksim", 1))
    with pytest.raises(ValueError):
        w.remove(eid)
    assert w.alive(eid)                     # the refusal changed nothing


def test_entities_order_deterministic():
    w = World()
    a = w.create("a", _matter())
    b = w.create("b", _matter())
    c = w.create("c", _matter())
    assert w.entities() == (a, b, c)        # slot order, not hash order
    w.remove(b)
    assert w.entities() == (a, c)           # dead slot skipped
    d = w.create("d", _matter())            # reuses b's slot (LIFO)
    assert d.index == b.index
    assert w.entities() == (a, d, c)        # still slot order
    assert len(w) == 3


def test_gametags_tuple():
    t = GameTags(("pushable", "crockery"))
    assert t.tags == ("pushable", "crockery")
    assert isinstance(t.tags, tuple)        # ordered + immutable, never a set
    assert GameTags().tags == ()
