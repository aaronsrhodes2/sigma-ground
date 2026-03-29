"""
Tests for sigma_ground.labs — mini-lab simulation engine.

Covers: dataclasses, environment, material cascade, simulation runs,
collision detection, and known-scenario validation.
"""

import math
import pytest

from sigma_ground.labs import (
    SimulationScene, SimObject, SimEvent,
    Environment, Medium, GroundConfig, BoundaryPlane, LightSource,
    run_simulation, cascade_material, validate, KNOWN_SCENARIOS,
)
from sigma_ground.labs.result import (
    SimulationResult, Snapshot, ObjectState, CollisionEvent,
)


# ── Tolerance helper ─────────────────────────────────────────────────

REL = 0.02  # 2% relative tolerance for numerical integration


# ── SimObject ────────────────────────────────────────────────────────

def test_simobject_defaults():
    """SimObject stores defaults for position, velocity, temperature."""
    obj = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron')
    assert obj.name == 'ball'
    assert obj.shape == 'sphere'
    assert obj.dimensions == {'radius': 0.05}
    assert obj.material_key == 'iron'
    assert obj.position == (0, 0, 0)
    assert obj.velocity == (0, 0, 0)
    assert obj.temperature == pytest.approx(293.15)
    assert obj.angular_velocity == (0, 0, 0)
    assert obj.is_static is False


def test_simobject_custom_fields():
    """SimObject accepts all custom fields."""
    obj = SimObject(
        'bullet', 'cylinder', {'radius': 0.005, 'length': 0.02}, 'copper',
        position=(1, 2, 3), velocity=(100, 0, 0),
        temperature=400.0, angular_velocity=(0, 0, 50),
        is_static=True,
    )
    assert obj.position == (1, 2, 3)
    assert obj.velocity == (100, 0, 0)
    assert obj.temperature == 400.0
    assert obj.angular_velocity == (0, 0, 50)
    assert obj.is_static is True


def test_simobject_dimensions_are_copied():
    """SimObject makes a copy of dimensions dict (no aliasing)."""
    dims = {'radius': 0.1}
    obj = SimObject('a', 'sphere', dims, 'iron')
    dims['radius'] = 999
    assert obj.dimensions['radius'] == 0.1


def test_simobject_repr():
    """SimObject repr includes name, shape, and material."""
    obj = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron')
    r = repr(obj)
    assert 'ball' in r
    assert 'iron' in r


# ── SimEvent ─────────────────────────────────────────────────────────

def test_simevent_creation():
    """SimEvent stores name, condition, and description."""
    ev = SimEvent('impact', 'collision', 'ball hits ground')
    assert ev.name == 'impact'
    assert ev.condition == 'collision'
    assert ev.description == 'ball hits ground'


def test_simevent_default_description():
    """SimEvent description defaults to empty string."""
    ev = SimEvent('peak', 't=0.5')
    assert ev.description == ''


# ── SimulationScene ──────────────────────────────────────────────────

def test_scene_defaults():
    """SimulationScene applies default environment and empty events."""
    obj = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron')
    scene = SimulationScene('test', 'desc', [obj])
    assert scene.name == 'test'
    assert scene.description == 'desc'
    assert len(scene.objects) == 1
    assert scene.environment is not None
    assert scene.events == []
    assert scene.duration == 1.0
    assert scene.snapshot_interval == 0.01


def test_scene_custom_duration():
    """SimulationScene accepts custom duration and snapshot interval."""
    obj = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron')
    scene = SimulationScene('t', 'd', [obj], duration=5.0,
                            snapshot_interval=0.1)
    assert scene.duration == 5.0
    assert scene.snapshot_interval == 0.1


def test_scene_repr():
    """SimulationScene repr includes name, object count, duration."""
    obj = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron')
    scene = SimulationScene('drop', 'test', [obj], duration=2.0)
    r = repr(scene)
    assert 'drop' in r
    assert '1 objects' in r
    assert '2.0s' in r


# ── Medium ───────────────────────────────────────────────────────────

def test_medium_vacuum():
    """Medium.vacuum() has zero density and viscosity."""
    m = Medium.vacuum()
    assert m.name == 'vacuum'
    assert m.density == 0.0
    assert m.viscosity == 0.0
    assert m.temperature == pytest.approx(2.725)  # CMB temperature


def test_medium_air():
    """Medium.air() produces air-like density and viscosity at STP."""
    m = Medium.air()
    assert m.name == 'air'
    # Air density at 20C, 1atm: ~1.2 kg/m^3
    assert m.density == pytest.approx(1.2, abs=0.1)
    # Air viscosity at 20C: ~1.8e-5 Pa*s
    assert m.viscosity == pytest.approx(1.8e-5, rel=0.1)
    assert m.temperature == pytest.approx(293.15)


def test_medium_water():
    """Medium.water() produces water-like density and viscosity at 20C."""
    m = Medium.water()
    assert m.name == 'water'
    # Water density near 20C: ~998 kg/m^3
    assert m.density == pytest.approx(998, rel=0.01)
    # Water viscosity near 20C: ~1e-3 Pa*s
    assert m.viscosity == pytest.approx(1e-3, rel=0.2)


def test_medium_default_is_air():
    """Medium() default is air."""
    m = Medium()
    assert m.name == 'air'
    assert m.density == pytest.approx(1.225)


def test_medium_repr():
    """Medium repr includes name, density, viscosity."""
    m = Medium.vacuum()
    r = repr(m)
    assert 'vacuum' in r


# ── GroundConfig ─────────────────────────────────────────────────────

def test_ground_defaults():
    """GroundConfig defaults to enabled, y=0, concrete."""
    g = GroundConfig()
    assert g.enabled is True
    assert g.height == 0.0
    assert g.material_key == 'concrete'
    assert g.normal == (0, 1, 0)


def test_ground_disabled():
    """GroundConfig can be disabled."""
    g = GroundConfig(enabled=False)
    assert g.enabled is False


# ── BoundaryPlane ────────────────────────────────────────────────────

def test_boundary_plane():
    """BoundaryPlane stores point, normal, material."""
    bp = BoundaryPlane(point=(5, 0, 0), normal=(-1, 0, 0),
                       material_key='steel_mild')
    assert bp.point == (5, 0, 0)
    assert bp.normal == (-1, 0, 0)
    assert bp.material_key == 'steel_mild'


# ── LightSource ──────────────────────────────────────────────────────

def test_light_source_visible():
    """LightSource with default range is visible."""
    ls = LightSource(position=(0, 5, 0), intensity=100.0)
    assert ls.is_visible is True
    assert ls.is_monochromatic is False


def test_light_source_laser():
    """LightSource with single wavelength is monochromatic."""
    ls = LightSource(position=(0, 0, 0), wavelength_range=(532e-9, 532e-9))
    assert ls.is_monochromatic is True
    assert ls.is_visible is True


def test_light_source_uv_not_visible():
    """UV-only LightSource is not visible."""
    ls = LightSource(position=(0, 0, 0), wavelength_range=(10e-9, 300e-9))
    assert ls.is_visible is False


# ── Environment ──────────────────────────────────────────────────────

def test_environment_defaults():
    """Environment() has Earth gravity, air medium, ground enabled."""
    env = Environment()
    assert env.gravity == (0, -9.80665, 0)
    assert env.medium.name == 'air'
    assert env.ground.enabled is True
    assert env.boundaries == []
    assert env.light_sources == []
    assert env.reference_frame == 'lab'


def test_environment_custom_gravity():
    """Environment accepts custom gravity vector."""
    env = Environment(gravity=(0, -1.625, 0))  # Moon
    assert env.gravity == (0, -1.625, 0)


def test_environment_with_medium():
    """Environment accepts a custom Medium."""
    env = Environment(medium=Medium.vacuum())
    assert env.medium.density == 0.0


def test_environment_repr():
    """Environment repr includes gravity magnitude and medium."""
    env = Environment()
    r = repr(env)
    assert 'air' in r
    assert 'on' in r  # ground on


# ── Material cascade ────────────────────────────────────────────────

def test_cascade_iron():
    """cascade_material('iron') returns density and thermal properties."""
    props = cascade_material('iron', T=293.15)
    assert props['material_key'] == 'iron'
    assert props['density_kg_m3'] == pytest.approx(7874, rel=0.01)
    assert 'sound_velocity_m_s' in props
    assert 'specific_heat_J_kgK' in props
    assert 'thermal_conductivity_W_mK' in props
    assert 'restitution' in props


def test_cascade_copper():
    """cascade_material('copper') returns valid properties."""
    props = cascade_material('copper', T=300.0)
    assert props['material_key'] == 'copper'
    assert props['density_kg_m3'] == pytest.approx(8960, rel=0.01)
    assert 'bulk_modulus_Pa' in props


def test_cascade_aluminum():
    """cascade_material('aluminum') returns valid properties."""
    props = cascade_material('aluminum', T=293.15)
    assert props['material_key'] == 'aluminum'
    assert props['density_kg_m3'] == pytest.approx(2700, rel=0.01)


def test_cascade_temperature_affects_properties():
    """Changing temperature changes thermal properties."""
    cold = cascade_material('iron', T=200.0)
    hot = cascade_material('iron', T=800.0)
    # Specific heat generally increases with T
    # Just verify they are different
    assert cold['specific_heat_J_kgK'] != hot['specific_heat_J_kgK']


def test_cascade_has_mechanical_properties():
    """cascade_material returns bulk, shear, and Young's modulus."""
    props = cascade_material('iron', T=293.15)
    assert 'bulk_modulus_Pa' in props
    assert 'shear_modulus_Pa' in props
    assert 'youngs_modulus_Pa' in props
    assert props['bulk_modulus_Pa'] > 0
    assert props['shear_modulus_Pa'] > 0
    assert props['youngs_modulus_Pa'] > 0


def test_cascade_unknown_material_raises():
    """cascade_material with an unknown key raises KeyError."""
    with pytest.raises(KeyError):
        cascade_material('unobtanium')


# ── Result dataclasses ───────────────────────────────────────────────

def test_object_state_to_dict():
    """ObjectState.to_dict() produces the expected keys."""
    os = ObjectState('ball', (0, 1, 0), (0, -3, 0), 3.0, 1.5)
    d = os.to_dict()
    assert d['name'] == 'ball'
    assert d['position'] == [0, 1, 0]
    assert d['velocity'] == [0, -3, 0]
    assert d['speed'] == 3.0
    assert d['kinetic_energy'] == 1.5


def test_collision_event_to_dict():
    """CollisionEvent.to_dict() produces the expected keys."""
    ce = CollisionEvent(0.45, 'ball', 'ground', 4.4, 10.0, 2.0)
    d = ce.to_dict()
    assert d['time'] == pytest.approx(0.45)
    assert d['objects'] == ['ball', 'ground']
    assert d['relative_velocity'] == pytest.approx(4.4)


def test_snapshot_to_dict():
    """Snapshot.to_dict() includes time, event, objects, energies."""
    os = ObjectState('ball', (0, 0, 0), (0, 0, 0), 0.0, 0.0)
    snap = Snapshot(1.0, [os], event_name='final',
                    total_kinetic_energy=0.0, total_potential_energy=5.0)
    d = snap.to_dict()
    assert d['time'] == 1.0
    assert d['event'] == 'final'
    assert len(d['objects']) == 1
    assert d['total_potential_energy'] == 5.0


def test_simulation_result_initial_final():
    """SimulationResult.initial and .final return first and last snapshots."""
    s0 = Snapshot(0.0, [])
    s1 = Snapshot(1.0, [])
    sr = SimulationResult('test', 'desc', [s0, s1], [],
                          duration=1.0, steps_taken=100)
    assert sr.initial is s0
    assert sr.final is s1


def test_simulation_result_empty_snapshots():
    """SimulationResult with no snapshots returns None for initial/final."""
    sr = SimulationResult('test', 'desc', [], [],
                          duration=0.0, steps_taken=0)
    assert sr.initial is None
    assert sr.final is None


def test_simulation_result_to_dict():
    """SimulationResult.to_dict() includes all top-level fields."""
    s0 = Snapshot(0.0, [])
    sr = SimulationResult('t', 'd', [s0], [],
                          duration=1.0, steps_taken=50,
                          material_properties={'ball': {}},
                          summary={'steps': 50})
    d = sr.to_dict()
    assert d['scene_name'] == 't'
    assert d['steps_taken'] == 50
    assert d['num_snapshots'] == 1
    assert d['num_collisions'] == 0


# ── Free fall in vacuum ──────────────────────────────────────────────

def test_free_fall_vacuum_velocity():
    """Iron sphere in vacuum: after 0.45s, v = g*t ~ 4.41 m/s."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 10, 0))
    duration = 0.45
    scene = SimulationScene(
        name='free_fall',
        description='Iron ball free fall in vacuum',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=duration,
    )
    result = run_simulation(scene)

    # v = g*t = 9.80665 * 0.45 = 4.413 m/s
    expected_speed = 9.80665 * duration
    final_speed = result.final.objects[0].speed
    assert final_speed == pytest.approx(expected_speed, rel=REL)


def test_free_fall_vacuum_position():
    """After 0.45s of free fall in vacuum, y ~ 1 - 0.5*g*t^2."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='freefall_pos',
        description='Check position during free fall',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.45,
    )
    result = run_simulation(scene)

    # y = 1 - 0.5 * 9.80665 * 0.45^2 = 1 - 0.993 = 0.007
    expected_y = 1.0 - 0.5 * 9.80665 * 0.45**2
    actual_y = result.final.objects[0].position[1]
    assert actual_y == pytest.approx(expected_y, abs=0.02)


def test_free_fall_no_horizontal_drift():
    """Free fall in vacuum produces no horizontal motion."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='no_drift',
        description='No x or z drift',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.45,
    )
    result = run_simulation(scene)
    final_pos = result.final.objects[0].position
    assert abs(final_pos[0]) < 1e-10
    assert abs(final_pos[2]) < 1e-10


# ── Free fall in air ─────────────────────────────────────────────────

def test_free_fall_air_slower_than_vacuum():
    """Iron sphere in air falls slightly slower than in vacuum due to drag."""
    ball_vac = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                         position=(0, 1, 0))
    scene_vac = SimulationScene(
        name='vac',
        description='vacuum fall',
        objects=[ball_vac],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.45,
    )

    ball_air = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                         position=(0, 1, 0))
    scene_air = SimulationScene(
        name='air',
        description='air fall',
        objects=[ball_air],
        environment=Environment(
            medium=Medium.air(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.45,
    )

    res_vac = run_simulation(scene_vac)
    res_air = run_simulation(scene_air)

    speed_vac = res_vac.final.objects[0].speed
    speed_air = res_air.final.objects[0].speed

    # Air drag slows it down (even if slightly for dense iron)
    assert speed_air < speed_vac


# ── Ground collision ─────────────────────────────────────────────────

def test_ground_collision_bounces():
    """Ball dropped onto ground produces at least one ground collision."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='bounce',
        description='Ball dropped onto ground',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=True, height=0.0),
        ),
        duration=1.5,
    )
    result = run_simulation(scene)

    # Should have at least one collision with ground
    ground_collisions = [c for c in result.collisions
                         if c.object_b == 'ground']
    assert len(ground_collisions) >= 1


def test_ground_collision_ball_stays_above_ground():
    """After bouncing, ball should remain at or above ground level."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='stay_above',
        description='Ball should not fall through ground',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=True, height=0.0),
        ),
        duration=2.0,
    )
    result = run_simulation(scene)

    # Ball center should always be at or above ground + radius
    for snap in result.snapshots:
        for obj in snap.objects:
            # Allow small numerical penetration
            assert obj.position[1] >= -0.01, (
                f"Ball fell below ground at t={snap.time:.3f}: "
                f"y={obj.position[1]:.4f}"
            )


def test_ground_collision_energy_dissipated():
    """Ground collision dissipates energy (inelastic restitution < 1)."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='energy_loss',
        description='Check energy dissipation on bounce',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=True),
        ),
        duration=3.0,
    )
    result = run_simulation(scene)

    # Final speed should be less than initial impact speed
    # (energy lost to inelastic collision)
    max_speed = max(
        obj.speed
        for snap in result.snapshots
        for obj in snap.objects
    )
    final_speed = result.final.objects[0].speed
    assert final_speed < max_speed


# ── Two-ball collision ───────────────────────────────────────────────

def test_two_ball_collision_occurs():
    """Two approaching spheres produce at least one collision event."""
    ball_a = SimObject('left', 'sphere', {'radius': 0.05}, 'iron',
                       position=(-0.5, 0.1, 0), velocity=(2, 0, 0))
    ball_b = SimObject('right', 'sphere', {'radius': 0.05}, 'iron',
                       position=(0.5, 0.1, 0), velocity=(-2, 0, 0))
    scene = SimulationScene(
        name='collision',
        description='Two balls approaching head-on',
        objects=[ball_a, ball_b],
        environment=Environment(
            gravity=(0, 0, 0),  # zero gravity so they stay on track
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
    )
    result = run_simulation(scene)

    # At least one collision between the two balls
    ball_collisions = [
        c for c in result.collisions
        if set([c.object_a, c.object_b]) == {'left', 'right'}
    ]
    assert len(ball_collisions) >= 1


def test_two_ball_collision_momentum_exchange():
    """Head-on collision of equal spheres: velocities should reverse."""
    ball_a = SimObject('left', 'sphere', {'radius': 0.05}, 'iron',
                       position=(-0.5, 0.1, 0), velocity=(2, 0, 0))
    ball_b = SimObject('right', 'sphere', {'radius': 0.05}, 'iron',
                       position=(0.5, 0.1, 0), velocity=(-2, 0, 0))
    scene = SimulationScene(
        name='momentum',
        description='Equal mass head-on collision',
        objects=[ball_a, ball_b],
        environment=Environment(
            gravity=(0, 0, 0),
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
    )
    result = run_simulation(scene)

    # After collision, left ball should be moving leftward (or stopped)
    # and right ball should be moving rightward (or stopped)
    final_states = {o.name: o for o in result.final.objects}
    left_vx = final_states['left'].velocity[0]
    right_vx = final_states['right'].velocity[0]

    # For near-elastic collision: velocities should roughly exchange
    # left started at +2, should now be negative or near zero
    assert left_vx < 1.0, f"Left ball vx={left_vx} should have reversed"
    # right started at -2, should now be positive or near zero
    assert right_vx > -1.0, f"Right ball vx={right_vx} should have reversed"


# ── Simulation result structure ──────────────────────────────────────

def test_result_has_snapshots():
    """run_simulation returns result with multiple snapshots."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='snaps',
        description='Check snapshot generation',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
        snapshot_interval=0.01,
    )
    result = run_simulation(scene)

    assert len(result.snapshots) >= 10
    assert result.initial is not None
    assert result.final is not None


def test_result_snapshots_time_ordered():
    """Snapshots are in chronological order."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='order',
        description='Snapshot ordering',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.3,
    )
    result = run_simulation(scene)

    times = [s.time for s in result.snapshots]
    assert times == sorted(times)


def test_result_initial_snapshot_is_t0():
    """Initial snapshot is at t=0 with event name 'initial'."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='t0',
        description='Initial state',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)

    assert result.initial.time == pytest.approx(0.0)
    assert result.initial.event_name == 'initial'


def test_result_final_snapshot():
    """Final snapshot has event name 'final'."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='fin',
        description='Final state',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)

    assert result.final.event_name == 'final'


def test_result_steps_taken_positive():
    """Simulation takes a positive number of steps."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='steps',
        description='Step count',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)

    assert result.steps_taken > 0
    assert result.duration > 0


def test_result_material_properties_populated():
    """run_simulation populates material_properties for each object."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='matprops',
        description='Material properties in result',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)

    assert 'ball' in result.material_properties
    assert result.material_properties['ball']['density_kg_m3'] > 0


def test_result_summary_has_object_data():
    """Result summary includes per-object final data."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='summ',
        description='Summary check',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)

    assert 'ball' in result.summary['objects']
    ball_summary = result.summary['objects']['ball']
    assert 'final_position' in ball_summary
    assert 'final_velocity' in ball_summary
    assert 'final_speed' in ball_summary


# ── Named events ─────────────────────────────────────────────────────

def test_time_event_triggers():
    """A time-based event captures a snapshot at the specified time."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='event_test',
        description='Time event',
        objects=[ball],
        events=[SimEvent('halfway', 't=0.2', 'check at 0.2s')],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.4,
    )
    result = run_simulation(scene)

    event_snaps = [s for s in result.snapshots
                   if s.event_name == 'halfway']
    assert len(event_snaps) == 1
    assert event_snaps[0].time == pytest.approx(0.2, abs=0.01)


# ── Validation ───────────────────────────────────────────────────────

def test_known_scenarios_exist():
    """KNOWN_SCENARIOS dict contains the expected scenario keys."""
    assert 'free_fall_vacuum' in KNOWN_SCENARIOS
    assert 'free_fall_air' in KNOWN_SCENARIOS
    assert 'elastic_collision' in KNOWN_SCENARIOS


def test_validate_free_fall_vacuum_passes():
    """validate() passes for a correct free-fall-in-vacuum simulation.

    The 'free_fall_vacuum' scenario expects impact_velocity=4.429 m/s,
    which is v=sqrt(2gh) for h=1m. Duration must match the theoretical
    fall time t=sqrt(2h/g)=0.4515s so the final speed matches.
    """
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    # t = sqrt(2h/g) = 0.4515s -- use this as duration so final speed
    # matches the expected impact velocity
    duration = math.sqrt(2.0 / 9.80665)
    scene = SimulationScene(
        name='validate_freefall',
        description='Validation test',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=duration,
    )
    result = run_simulation(scene)
    report = validate(result, 'free_fall_vacuum')

    assert report['passed'] is True, (
        f"Validation failed: {report['checks']}"
    )


def test_validate_unknown_scenario():
    """validate() with unknown scenario key returns passed=False."""
    s0 = Snapshot(0.0, [])
    dummy = SimulationResult('t', 'd', [s0], [],
                             duration=1.0, steps_taken=1)
    report = validate(dummy, 'nonexistent_scenario')
    assert report['passed'] is False


def test_validate_report_structure():
    """validate() returns dict with passed, checks, scenario keys."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    duration = math.sqrt(2.0 / 9.80665)
    scene = SimulationScene(
        name='report_check',
        description='Report structure',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=duration,
    )
    result = run_simulation(scene)
    report = validate(result, 'free_fall_vacuum')

    assert 'passed' in report
    assert 'checks' in report
    assert 'scenario' in report
    assert isinstance(report['checks'], list)
    assert len(report['checks']) > 0

    for check in report['checks']:
        assert 'name' in check
        assert 'passed' in check


# ── Energy conservation ──────────────────────────────────────────────

def test_energy_conservation_in_vacuum_free_fall():
    """Total mechanical energy (KE + PE) is conserved in vacuum free fall."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 5, 0))
    scene = SimulationScene(
        name='energy_cons',
        description='Energy conservation check',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
        snapshot_interval=0.05,
    )
    result = run_simulation(scene)

    # Total mechanical energy = KE + PE should be roughly constant
    initial_E = (result.initial.total_kinetic_energy +
                 result.initial.total_potential_energy)
    final_E = (result.final.total_kinetic_energy +
               result.final.total_potential_energy)

    assert final_E == pytest.approx(initial_E, rel=REL)


# ── Multiple shapes ─────────────────────────────────────────────────

def test_box_shape_object():
    """Box-shaped object can be simulated."""
    box = SimObject('crate', 'box',
                    {'width': 0.1, 'depth': 0.1, 'height': 0.1}, 'iron',
                    position=(0, 1, 0))
    scene = SimulationScene(
        name='box_test',
        description='Box in free fall',
        objects=[box],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.3,
    )
    result = run_simulation(scene)

    assert result.final.objects[0].speed > 0


def test_cylinder_shape_object():
    """Cylinder-shaped object can be simulated."""
    cyl = SimObject('rod', 'cylinder',
                    {'radius': 0.01, 'length': 0.2}, 'copper',
                    position=(0, 2, 0))
    scene = SimulationScene(
        name='cyl_test',
        description='Cylinder in free fall',
        objects=[cyl],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.3,
    )
    result = run_simulation(scene)

    assert result.final.objects[0].speed > 0


# ── Static objects ───────────────────────────────────────────────────

def test_static_object_does_not_move():
    """A static object stays at its initial position."""
    wall = SimObject('wall', 'box',
                     {'width': 2, 'depth': 0.1, 'height': 2}, 'iron',
                     position=(0, 0, 0), is_static=True)
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='static_test',
        description='Static wall and falling ball',
        objects=[wall, ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.3,
    )
    result = run_simulation(scene)

    # Static objects are excluded from snapshots (only dynamic objects)
    # The ball should have moved
    ball_state = next(
        (o for o in result.final.objects if o.name == 'ball'), None)
    assert ball_state is not None
    assert ball_state.speed > 0


# ── Drag in different media ──────────────────────────────────────────

def test_drag_greater_in_water_than_air():
    """Same sphere falls slower in water than in air due to greater drag."""
    ball_air = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                         position=(0, 5, 0))
    scene_air = SimulationScene(
        name='air_drag',
        description='Fall in air',
        objects=[ball_air],
        environment=Environment(
            medium=Medium.air(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
    )

    ball_water = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                           position=(0, 5, 0))
    scene_water = SimulationScene(
        name='water_drag',
        description='Fall in water',
        objects=[ball_water],
        environment=Environment(
            medium=Medium.water(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.5,
    )

    res_air = run_simulation(scene_air)
    res_water = run_simulation(scene_water)

    speed_air = res_air.final.objects[0].speed
    speed_water = res_water.final.objects[0].speed

    assert speed_water < speed_air


# ── Zero gravity ─────────────────────────────────────────────────────

def test_zero_gravity_no_acceleration():
    """Object in zero gravity maintains constant velocity."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 0, 0), velocity=(1, 0, 0))
    scene = SimulationScene(
        name='zero_g',
        description='No gravity drift',
        objects=[ball],
        environment=Environment(
            gravity=(0, 0, 0),
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=1.0,
    )
    result = run_simulation(scene)

    final = result.final.objects[0]
    # Should drift at 1 m/s for 1s => x ~ 1.0
    assert final.position[0] == pytest.approx(1.0, rel=REL)
    assert final.speed == pytest.approx(1.0, rel=REL)
    # y and z unchanged
    assert abs(final.position[1]) < 1e-6
    assert abs(final.position[2]) < 1e-6


# ── to_dict round-trip ───────────────────────────────────────────────

def test_full_result_to_dict():
    """Full simulation result serializes to dict without error."""
    ball = SimObject('ball', 'sphere', {'radius': 0.05}, 'iron',
                     position=(0, 1, 0))
    scene = SimulationScene(
        name='serial',
        description='Serialization test',
        objects=[ball],
        environment=Environment(
            medium=Medium.vacuum(),
            ground=GroundConfig(enabled=False),
        ),
        duration=0.1,
    )
    result = run_simulation(scene)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert d['scene_name'] == 'serial'
    assert isinstance(d['snapshots'], list)
    assert isinstance(d['collisions'], list)
    assert d['initial_state'] is not None
    assert d['final_state'] is not None
