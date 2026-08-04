"""Tests for region.py — the StackSimRegion adapter, gated by the GOLDEN
REPLAY: the same scenario through the engine and through bare StackSim must
be BYTE-IDENTICAL. Exact ``==`` on the raw float lists, never approx — the
adapter wraps the same code, so exact equality is achievable and anything
less means the engine layer perturbed physics.

Plain pytest functions, house style of sigma_ground/field/interface/
test_propulsion.py. The stack is the proven offline scenario from
engine/bench/harness.py: build_stack(["1700s table", "plate", "Doughnut"],
allow_llm=False) — deterministic, no network, no LLM.

The ``built`` list is built ONCE (module-scope fixture) and SHARED by both
sims — verified safe: StackSim copies com into its own lists at
construction and never mutates Constructs; sharing removes build variance
from the gate and halves runtime (engine/bench/determinism.py separately
proves double-builds are bit-equal).
"""
import json

import pytest

from sigma_ground.dynamics.mechanisms.applied import AppliedForce, KINDS
from sigma_ground.dynamics.seeding import SeedBody
from sigma_ground.engine.ecs import StaleEntityError, World, \
    matter_from_construct
from sigma_ground.engine.region import (
    STACK_LIVE_KINDS,
    RegionRefusal,
    StackSimRegion,
)
from sigma_ground.radiance.push_off import StackSim, make_lookups

DT = 1.0 / 960.0
N_STEPS = 1920          # 2.0 s — crosses slide, edge, ballistic and landing

# ONE schedule, TWO interpreters (bare + engine) — a single table cannot
# drift between the two sides. Events at step k are applied BEFORE step k.
SCHEDULE = (
    (0,   "impulse", "plate",    {"dv": 2.2}),
    (240, "force",   "Doughnut", {"fx_n": 0.8, "duration_s": 0.25}),
    (480, "tilt",    None,       {"angle_deg": 6.0, "duration_s": 0.5}),
    (700, "hold",    "Doughnut", {"on": True}),
    (900, "hold",    "Doughnut", {"on": False}),
)

# One lookup pair, shared by both sims (stateless closures over copied dicts).
MU_LOOKUP, E_LOOKUP = make_lookups(1)


@pytest.fixture(scope="module")
def built():
    from sigma_ground.deckard.scene import build_stack
    return build_stack(["1700s table", "plate", "Doughnut"], allow_llm=False)


@pytest.fixture()
def region(built):
    return StackSimRegion(World(), built, mu_lookup=MU_LOOKUP,
                          e_lookup=E_LOOKUP)


def _eid(region, name):
    return region.entity_ids[region.sim.names.index(name)]


# ═══════════════════════════════════════════════════════════════════════════
# § 1. THE GATE — golden replay, byte-identical
# ═══════════════════════════════════════════════════════════════════════════

def _bare_dispatch(bare, kind, target, p):
    """The bare interpreter: the backend's own entry points, raw floats."""
    idx = bare.names.index(target) if target is not None else None
    if kind == "impulse":
        bare.apply_impulse(idx, p["dv"])
    elif kind == "force":
        bare.apply_force(idx, p["fx_n"], p["duration_s"])
    elif kind == "tilt":
        bare.apply_tilt(p["angle_deg"], p["duration_s"])
    elif kind == "hold":
        bare.hold(idx, p["on"])


def _engine_af(built, kind, target, p):
    """The engine interpreter: the SAME event as a closed-taxonomy
    AppliedForce with exact-float parameter equivalents (point/twist None)."""
    if kind == "impulse":
        mass = dict(built)[target].mass_kg
        return AppliedForce(
            "impulse", target,
            vector=[mass * p["dv"], 0.0, 0.0],      # unrounded N*s disclosure
            variables={"push_speed_m_s": p["dv"]})  # the routed exact dv
    if kind == "force":
        return AppliedForce("force", target, vector=[p["fx_n"], 0.0, 0.0],
                            duration=p["duration_s"])
    if kind == "tilt":
        return AppliedForce("tilt", "world",
                            variables={"angle_deg": p["angle_deg"]},
                            duration=p["duration_s"])
    if kind == "hold":
        return AppliedForce("hold", target, variables={"on": p["on"]})
    raise AssertionError(f"schedule kind {kind!r} has no interpreter")


def test_golden_replay_byte_identical(built):
    """The phase gate. WHY byte-identity holds (the three load-bearing
    invariants):

    1. apply-then-step at index k on BOTH sides means sim.t (the identical
       accumulated sequence of += DT) is bit-identical at every apply call;
       apply_force's t_end = t + duration and apply_tilt's (t0, t1) window
       therefore match bit-for-bit.
    2. region.apply is a synchronous pass-through: same StackSim methods,
       same float objects, same call order — no queue, no reorder, no
       arithmetic on the test path (dv comes verbatim from
       variables["push_speed_m_s"]; force/tilt/hold floats pass straight
       through; float() on a float is value-identity).
    3. region.step(dt) is exactly one sim.step(dt): the engine layer
       performs ZERO arithmetic on hot state.

    Anything less than exact == means the engine perturbed physics — which
    is what this gate exists to catch.
    """
    bare = StackSim(built, mu_lookup=MU_LOOKUP, e_lookup=E_LOOKUP)
    world = World()
    reg = StackSimRegion(world, built, mu_lookup=MU_LOOKUP,
                         e_lookup=E_LOOKUP)
    eid_by_name = {reg.sim.names[i]: reg.entity_ids[i]
                   for i in range(reg.sim.n)}
    assert reg.sim is not bare              # two objects, ONE physics impl

    for k in range(N_STEPS):
        for (at, kind, target, p) in SCHEDULE:
            if at != k:
                continue
            _bare_dispatch(bare, kind, target, p)       # bare FIRST
            reg.apply(eid_by_name[target] if target is not None else None,
                      _engine_af(built, kind, target, p))
        bare.step(DT)
        reg.step(DT)
        s = reg.sim
        # exact list == — the gate, NEVER approx
        assert bare.dx == s.dx and bare.dz == s.dz, f"pos diverged at {k}"
        assert bare.vx == s.vx and bare.vz == s.vz, f"vel diverged at {k}"
        assert bare.support == s.support, f"support diverged at {k}"
        assert bare.t == s.t, f"t diverged at {k}"
        assert json.dumps(bare.frame(), sort_keys=True) == \
            json.dumps(s.frame(), sort_keys=True), \
            f"frame diverged at step {k}"


# ═══════════════════════════════════════════════════════════════════════════
# § 2. honest refusals — capability, not caller error
# ═══════════════════════════════════════════════════════════════════════════

def test_source_refused(region):
    af = AppliedForce("source", "plate", power_w=50.0, duration=1.0)
    with pytest.raises(RegionRefusal, match="thermal"):
        region.apply(_eid(region, "plate"), af)


def test_drive_refused(region):
    af = AppliedForce("drive", "plate", variables={"torque_nm": 1.0})
    with pytest.raises(RegionRefusal, match="joint"):
        region.apply(_eid(region, "plate"), af)


def test_spawn_refused(region):
    neutral = SeedBody(mass_kg=0.5, position_m=(0.0, 0.0, 1.0),
                       velocity_m_s=(0.0, 0.0, 0.0), radius_m=0.1,
                       label="intruder")
    with pytest.raises(RegionRefusal, match="fixed at construction"):
        region.spawn(neutral)


def test_base_body_refused_as_target(region):
    base_name = region.sim.names[0]
    af = AppliedForce("impulse", base_name,
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(RegionRefusal, match="base"):
        region.apply(region.entity_ids[0], af)


def test_base_body_extract_allowed(region):
    # reading is observation — always free, even on the base
    body = region.extract(region.entity_ids[0])
    assert isinstance(body, SeedBody)
    assert body.label == region.sim.names[0]


def test_tilt_with_body_handle_refused(region):
    af = AppliedForce("tilt", "world", variables={"angle_deg": 5.0},
                      duration=1.0)
    with pytest.raises(RegionRefusal, match="whole scene"):
        region.apply(_eid(region, "plate"), af)


def test_body_kind_with_none_refused(region):
    af = AppliedForce("impulse", "plate",
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(RegionRefusal, match="body handle"):
        region.apply(None, af)


def test_t0_scheduling_refused(region):
    af = AppliedForce("impulse", "plate", t0=0.5,
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(RegionRefusal, match="immediate"):
        region.apply(_eid(region, "plate"), af)


def test_zero_duration_force_refused(region):
    af = AppliedForce("force", "plate", vector=[1.0, 0.0, 0.0])  # duration 0
    with pytest.raises(RegionRefusal, match="positive window"):
        region.apply(_eid(region, "plate"), af)


def test_offplane_vector_refused(region):
    af = AppliedForce("impulse", "plate", vector=[0.0, 1.0, 0.0],
                      variables={"push_speed_m_s": 1.0})
    with pytest.raises(RegionRefusal, match="planar"):
        region.apply(_eid(region, "plate"), af)


def test_twist_refused(region):
    af = AppliedForce("impulse", "plate", vector=[1.0, 0.0, 0.0],
                      twist=[0.0, 1.0, 0.0],
                      variables={"push_speed_m_s": 1.0})
    with pytest.raises(RegionRefusal, match="rotational"):
        region.apply(_eid(region, "plate"), af)


def test_point_refused(region):
    # an application point is unrepresentable for every body kind here
    for af in (
        AppliedForce("impulse", "plate", vector=[1.0, 0.0, 0.0],
                     point=[0.0, 0.0, 0.0],
                     variables={"push_speed_m_s": 1.0}),
        AppliedForce("force", "plate", vector=[1.0, 0.0, 0.0],
                     duration=0.5, point=[0.0, 0.0, 0.0]),
        AppliedForce("hold", "plate", point=[0.0, 0.0, 0.0],
                     variables={"on": True}),
    ):
        with pytest.raises(RegionRefusal, match="point"):
            region.apply(_eid(region, "plate"), af)


def test_tilt_axis_key_refused(region):
    af = AppliedForce("tilt", "world",
                      variables={"angle_deg": 5.0, "axis": [0.0, 1.0, 0.0]},
                      duration=1.0)
    with pytest.raises(RegionRefusal, match="tilt axis"):
        region.apply(None, af)


def test_target_label_mismatch_refused(region):
    af = AppliedForce("impulse", "Doughnut",
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(RegionRefusal, match="must name"):
        region.apply(_eid(region, "plate"), af)


def test_non_appliedforce_typeerror(region):
    with pytest.raises(TypeError):
        region.apply(_eid(region, "plate"),
                     {"kind": "impulse", "target": "plate"})


def test_foreign_entity_refused(region, built):
    # a live world entity that is NOT bound to this region
    spare = region.world.create("spare",
                                matter_from_construct("spare", built[1][1]))
    af = AppliedForce("impulse", "spare",
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(RegionRefusal, match="not bound"):
        region.apply(spare, af)


def test_stale_handle_raises_stale_error(region, built):
    # create an UNBOUND world entity, remove it, then route through the
    # region: world.get runs BEFORE region membership, so ANY stale handle
    # dies as StaleEntityError, never as a membership refusal
    spare = region.world.create("spare",
                                matter_from_construct("spare", built[1][1]))
    region.world.remove(spare)
    af = AppliedForce("impulse", "spare",
                      variables={"push_speed_m_s": 1.0},
                      vector=[1.0, 0.0, 0.0])
    with pytest.raises(StaleEntityError):
        region.apply(spare, af)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. contract plumbing — routing completeness, fixed dt, neutral reads
# ═══════════════════════════════════════════════════════════════════════════

def test_route_table_covers_kinds(region):
    # a 7th taxonomy kind must fail LOUDLY at region construction; this
    # asserts the table tracks applied.KINDS exactly
    assert set(region._route) == set(KINDS)
    assert STACK_LIVE_KINDS < set(KINDS)


def test_fixed_dt_guard(region):
    region.step(DT)
    region.step(DT)                          # same dt: fine
    with pytest.raises(ValueError, match="fixed dt"):
        region.step(2 * DT)
    with pytest.raises(ValueError):
        region.step(0.0)                     # nonpositive dt refused too


def test_extract_is_engine_neutral(region):
    eid = _eid(region, "plate")
    i = region.sim.names.index("plate")
    body = region.extract(eid)
    assert isinstance(body, SeedBody)
    # plain floats, no engine types
    assert isinstance(body.mass_kg, float)
    assert all(isinstance(v, float) for v in body.position_m)
    assert all(isinstance(v, float) for v in body.velocity_m_s)
    # position == com + displacement; velocity == (vx, 0, vz)
    s = region.sim
    assert body.position_m == (s.com[i][0] + s.dx[i], s.com[i][1],
                               s.com[i][2] + s.dz[i])
    assert body.velocity_m_s == (s.vx[i], 0.0, s.vz[i])
    assert body.label == "plate"
    # displacement flows through after motion
    region.apply(eid, AppliedForce("impulse", "plate",
                                   vector=[1.0, 0.0, 0.0],
                                   variables={"push_speed_m_s": 2.2}))
    for _ in range(96):                      # 0.1 s
        region.step(DT)
    moved = region.extract(eid)
    assert moved.position_m == (s.com[i][0] + s.dx[i], s.com[i][1],
                                s.com[i][2] + s.dz[i])
    assert moved.position_m[0] > body.position_m[0]


def test_matter_provenance_on_real_constructs(region, built):
    from sigma_ground.engine.ecs import PROVENANCE_TAGS
    for i, (name, construct) in enumerate(built):
        ent = region.world.get(region.entity_ids[i])
        assert ent.label == name
        # mass is the Construct's own Fact, exactly — no copy drift
        assert ent.matter.mass_kg == construct.mass_kg
        assert ent.matter.com_m == tuple(construct.com_m)
        for text in ent.matter.provenance.values():
            assert text.startswith(PROVENANCE_TAGS)
        assert ent.matter.provenance["temperature_k"].startswith("CHOSEN:")
        assert ent.binding.backend == "stacksim"
        assert ent.binding.index == i
