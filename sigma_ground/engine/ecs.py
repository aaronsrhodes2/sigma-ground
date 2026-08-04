"""ECS core — generational entity handles + cold components (engine phase E1).

The repo's current identity scheme is bare list indices (StackSim bodies,
parcel lists, nbody arrays) with NO remove path anywhere — and that is
exactly the defect this module exists to fix. An ``EntityId`` is a
(index, generation) pair: removing an entity bumps its slot's generation,
so every handle minted before the removal dies loudly (``StaleEntityError``)
instead of silently resolving to whatever matter reused the slot. Silent
staleness is the bug class generational handles exist to surface, so ``get``
raises rather than returning None; ``alive()`` is the single non-throwing
probe.

Components are COLD state only — identity, provenance, geometry references,
backend bindings. HOT state (positions, velocities) lives inside the physics
backend that owns it (see ``Binding`` and engine.region); the ECS never
mirrors it per frame. Physics stays in ONE place — the plan's non-negotiable.

Doctrine, enforced at the type level ("deduce, don't infer"):

* ``Matter`` is MANDATORY on every world entity, and every one of its fields
  carries a provenance string tagged FACT:/DEDUCTION:/CHOSEN:/PLUG: —
  unprovenanced matter cannot be constructed at all (``Matter.__post_init__``
  rejects missing tags, missing keys, AND extra keys).
* Cameras, waypoints and UI are engine citizens, NOT world entities. They
  get their own registry when E3 builds them and never enter ``World`` —
  which is precisely why ``Matter`` is a mandatory positional argument of
  ``World.create`` with no escape hatch: nothing without matter belongs here,
  and nothing here can be missing it.

DESIGN CALL (disclosed): MaterialComposition is folded into
``Matter.composition`` for E1 as plain (name, material, mass_kg, source)
4-tuples read from the Construct's layers; the full ``Layer`` objects stay
reachable via ``Geometry.construct.layers``. A separate component earns its
keep only when something consumes composition independently of mass — no E1
system does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..dynamics.mechanisms.choice import Choice

# The closed provenance vocabulary. "CHOSEN:" (not "CHOICE:") because that is
# what dynamics.mechanisms.choice.Choice.cite() actually emits — a Matter
# field whose value is a real Choice stores that cite() string verbatim.
PROVENANCE_TAGS = ("FACT:", "DEDUCTION:", "CHOSEN:", "PLUG:")

# The per-tick energy ledger's key set — closed, like applied.KINDS. A new
# energy channel is a doctrine decision, not a dict key someone slips in.
LEDGER_KEYS = ("kinetic", "rotational", "thermal", "deformation", "chemical")

ROOM_TEMPERATURE_K = 293.15    # 20 degC — the disclosed default temperature Choice

# Exactly the provenance keys every Matter must carry: one per scalar field
# plus one per ledger channel. Missing AND extra keys are both rejected.
_PROVENANCE_KEYS = frozenset(
    ("mass_kg", "com_m", "composition", "temperature_k", "internal_energy_j")
) | frozenset("energy_ledger." + k for k in LEDGER_KEYS)


class StaleEntityError(LookupError):
    """A handle whose slot was removed (or never existed) was dereferenced.

    Raised, never silently None-d: a stale handle resolving to reused matter
    is the exact failure mode generational handles exist to make impossible.
    """


@dataclass(frozen=True)
class EntityId:
    """Generational handle — NEVER a bare list index (the plan's mandate).

    Frozen, hence hashable, hence usable in a Region's membership frozenset.
    """
    index: int
    generation: int


@dataclass
class Matter:
    """MANDATORY on every world entity — cold Facts/Choices only, never hot state.

    ``provenance`` maps every field name (and every ``energy_ledger.<key>``)
    to a tagged string; ``__post_init__`` enforces exact coverage and tagging,
    so a Matter with an invented, undisclosed value cannot exist.
    """
    mass_kg: float
    com_m: tuple            # (x, y, z) composed-pose centre of mass — a Fact
    composition: tuple      # ((layer_name, material, mass_kg, source), ...) —
                            # plain 4-tuples from Construct.layers (see the
                            # module docstring's disclosed design call)
    temperature_k: float
    internal_energy_j: float
    energy_ledger: dict     # keys exactly LEDGER_KEYS
    provenance: dict        # {"mass_kg": "FACT: ...", "energy_ledger.kinetic": ...}

    def __post_init__(self):
        if not self.mass_kg > 0:
            raise ValueError(f"Matter.mass_kg must be positive, got {self.mass_kg!r}")
        if not self.temperature_k > 0:
            raise ValueError(
                f"Matter.temperature_k must be positive (kelvin), got "
                f"{self.temperature_k!r}")
        if set(self.energy_ledger) != set(LEDGER_KEYS):
            raise ValueError(
                f"energy_ledger keys must be exactly {sorted(LEDGER_KEYS)}, "
                f"got {sorted(self.energy_ledger)} — the ledger is a closed "
                f"set, like applied.KINDS")
        got = set(self.provenance)
        if got != _PROVENANCE_KEYS:
            missing = sorted(_PROVENANCE_KEYS - got)
            extra = sorted(got - _PROVENANCE_KEYS)
            raise ValueError(
                f"Matter.provenance must cover every field exactly once — "
                f"missing: {missing}, extra: {extra}")
        for key, text in self.provenance.items():
            if not (isinstance(text, str) and text.startswith(PROVENANCE_TAGS)):
                raise ValueError(
                    f"provenance[{key!r}] must be a str starting with one of "
                    f"{PROVENANCE_TAGS}, got {text!r} — every value is a "
                    f"Fact, a Deduction, a disclosed Choice, or a Plug; "
                    f"never invented silently")


@dataclass
class Geometry:
    """The shape reference: the deckard Construct plus its bbox.

    ``construct`` is duck-typed (never imported) — the ECS stays a tier-3
    consumer of tier-2 matter without a compile-time edge. ``bbox`` is copied
    to nested float tuples so no caller can mutate shared geometry through it.
    """
    construct: object
    bbox: tuple             # ((x0,x1),(y0,y1),(z0,z1)), nested float tuples

    def __post_init__(self):
        self.bbox = tuple(tuple(float(v) for v in pair) for pair in self.bbox)


@dataclass(frozen=True)
class Binding:
    """Which physics backend OWNS this entity's HOT state.

    The ECS never mirrors positions/velocities per frame: reads go through
    the owning Region on demand (Region.extract). ``index`` is the backend's
    own local body index — meaningful only to that backend.
    """
    backend: str            # e.g. "stacksim"
    index: int


@dataclass
class GameTags:
    """Free-form string tags (may be empty).

    A tuple, NOT a set/list default: str-hash randomization makes set
    iteration order vary per process, and the determinism doctrine wants
    ordered immutable containers on anything that might ever be serialized.
    """
    tags: tuple = ()


@dataclass
class Entity:
    """One world entity's component bundle (internal storage record)."""
    label: str              # the body name an AppliedForce.target must match
    matter: Matter
    geometry: "Geometry | None"
    binding: "Binding | None"
    tags: GameTags


class World:
    """The WORLD registry: everything here has Matter, full stop.

    Cameras/waypoints/UI are engine citizens (E3+, their own registry) and
    NEVER enter World — that is why ``Matter`` is a mandatory positional
    argument with no default. Slot reuse is LIFO (deterministic — no
    hash-order anywhere near identity), and a removed slot's generation is
    bumped IMMEDIATELY so old handles die before any reuse can occur.
    """

    def __init__(self):
        self._slots: list = []      # Entity | None per slot
        self._gens: list = []       # generation counter per slot
        self._free: list = []       # freed slot indices, LIFO

    def create(self, label: str, matter: Matter, *, geometry=None,
               binding=None, tags=None) -> EntityId:
        """Register one world entity; returns its generational handle."""
        if not isinstance(label, str) or not label:
            raise TypeError(f"label must be a non-empty str, got {label!r}")
        if not isinstance(matter, Matter):
            raise TypeError(
                f"every world entity carries Matter (the mandatory component "
                f"— see the module docstring), got {type(matter).__name__}")
        if geometry is not None and not isinstance(geometry, Geometry):
            raise TypeError(f"geometry must be Geometry or None, got "
                            f"{type(geometry).__name__}")
        if binding is not None and not isinstance(binding, Binding):
            raise TypeError(f"binding must be Binding or None, got "
                            f"{type(binding).__name__}")
        if tags is None:
            tags = GameTags()
        elif not isinstance(tags, GameTags):
            raise TypeError(f"tags must be GameTags or None, got "
                            f"{type(tags).__name__}")
        ent = Entity(label, matter, geometry, binding, tags)
        if self._free:
            idx = self._free.pop()          # LIFO — deterministic reuse order
            self._slots[idx] = ent
        else:
            idx = len(self._slots)
            self._slots.append(ent)
            self._gens.append(0)
        return EntityId(idx, self._gens[idx])

    def get(self, eid: EntityId) -> Entity:
        """Dereference a handle; StaleEntityError on anything but a live one."""
        if not isinstance(eid, EntityId):
            raise TypeError(f"get() takes an EntityId, got {type(eid).__name__}")
        if not (0 <= eid.index < len(self._slots)) \
                or self._gens[eid.index] != eid.generation \
                or self._slots[eid.index] is None:
            raise StaleEntityError(
                f"stale or unknown entity handle (index={eid.index}, "
                f"generation={eid.generation}) — the slot was removed, "
                f"reused, or never allocated")
        return self._slots[eid.index]

    def alive(self, eid) -> bool:
        """The single non-throwing liveness probe."""
        return (isinstance(eid, EntityId)
                and 0 <= eid.index < len(self._slots)
                and self._gens[eid.index] == eid.generation
                and self._slots[eid.index] is not None)

    def remove(self, eid: EntityId) -> None:
        """Remove a live, UNBOUND entity; its handle (and any copy) goes stale.

        An entity whose ``binding`` is not None is refused with ValueError:
        backend-bound matter cannot be deleted behind the backend's back —
        release is a Region operation, and StackSimRegion honestly has none
        (its body set is fixed at construction).
        """
        ent = self.get(eid)                 # StaleEntityError on dead handles
        if ent.binding is not None:
            raise ValueError(
                f"entity {ent.label!r} is bound to backend "
                f"{ent.binding.backend!r} — backend-bound matter cannot be "
                f"deleted behind the backend's back; release is a Region "
                f"operation, and this backend's body set is fixed")
        self._slots[eid.index] = None
        self._gens[eid.index] += 1          # handles die BEFORE any reuse
        self._free.append(eid.index)

    def entities(self) -> tuple:
        """Live handles in slot order (deterministic; dead slots skipped)."""
        return tuple(EntityId(i, self._gens[i])
                     for i, slot in enumerate(self._slots) if slot is not None)

    def __len__(self) -> int:
        return sum(1 for slot in self._slots if slot is not None)


def matter_from_construct(name: str, construct, *,
                          temperature_k: float = ROOM_TEMPERATURE_K) -> Matter:
    """Build a fully-provenanced Matter from a real deckard Construct.

    Duck-typed on ``.mass_kg`` / ``.com_m`` / ``.layers`` (+ optional
    ``.source``) — this module never imports deckard, so a deckard rename
    surfaces here as an AttributeError at the call site, not as a hidden
    layering edge.

    Provenance, per field:
    * mass_kg / com_m — Facts: closed-form mass and analytic CoM from the
      Construct at its composed pose.
    * composition — Fact: per-layer 4-tuples read from Construct.layers.
    * temperature_k — a REAL disclosed Choice object (room temperature by
      default; this Region has no thermal state, StackSim refuses SOURCE),
      stored as its ``cite()`` string, which starts "CHOSEN:".
    * internal_energy_j / energy_ledger — zeros whose provenance strings say
      exactly why each channel is zero (model scope, or "E1 has no tick loop
      — per-tick ledger updating is owned by E2's WorldHost").
    """
    mass = float(construct.mass_kg)
    com = tuple(float(v) for v in construct.com_m)
    composition = tuple(
        (str(L.name), str(L.material), float(L.mass_kg), str(L.source))
        for L in construct.layers)
    src = str(getattr(construct, "source", "") or "").strip()
    src_tail = f" ({src})" if src else ""

    temp_choice = Choice(
        f"temperature of the {name}",
        float(temperature_k),
        checked="this Region has no thermal state — StackSim refuses SOURCE, "
                "and no thermal sim consumes this value in E1",
        reason="room temperature; matter must carry a temperature even where "
               "no thermal process runs, so the value is a disclosed pick, "
               "not a hidden default",
        adjustable=True)

    provenance = {
        "mass_kg": f"FACT: deckard Construct {name!r} closed-form total "
                   f"mass{src_tail}",
        "com_m": f"FACT: deckard Construct {name!r} analytic centre of mass "
                 f"at its composed pose{src_tail}",
        "composition": "FACT: per-layer (name, material, mass_kg, source) "
                       "4-tuples read from Construct.layers",
        "temperature_k": temp_choice.cite(),
        "internal_energy_j": "CHOSEN: ledger baseline zero at scenario "
                             "start; deltas only",
        "energy_ledger.kinetic":
            "DEDUCTION: 0 at rest at scenario start; the live value is a "
            "Region read (StackSim.energy()); per-tick updating is owned by "
            "E2's WorldHost — E1 has no tick loop",
        "energy_ledger.rotational":
            "FACT: SIMPLIFIED_MODEL — planar StackSim has no rotational "
            "DOF; 0 by model scope",
        "energy_ledger.thermal":
            "FACT: SIMPLIFIED_MODEL — no thermal state in this Region "
            "(SOURCE refused); 0 by model scope",
        "energy_ledger.deformation":
            "FACT: SIMPLIFIED_MODEL — no deformation in StackSim; 0 by "
            "model scope",
        "energy_ledger.chemical":
            "FACT: SIMPLIFIED_MODEL — no chemistry in this Region; 0 by "
            "model scope",
    }
    return Matter(
        mass_kg=mass,
        com_m=com,
        composition=composition,
        temperature_k=float(temp_choice.value),
        internal_energy_j=0.0,
        energy_ledger={k: 0.0 for k in LEDGER_KEYS},
        provenance=provenance)


__all__ = [
    "PROVENANCE_TAGS", "LEDGER_KEYS", "ROOM_TEMPERATURE_K",
    "StaleEntityError", "EntityId", "Matter", "Geometry", "Binding",
    "GameTags", "Entity", "World", "matter_from_construct",
]
