"""
Runner — builds, steps, and captures snapshots from a SimulationScene.

This is the orchestrator. It:
  1. Calls builder.build() to get a PhysicsScene + forces callback
  2. Steps the simulation forward in time
  3. Captures snapshots at regular intervals and at named events
  4. Returns a SimulationResult with full history
"""

import math
from ..dynamics.stepper import step
from ..dynamics.vec import Vec3
from .builder import build
from .result import (
    SimulationResult, Snapshot, ObjectState, CollisionEvent,
)


def _capture_snapshot(physics_scene, time, event_name=None, gravity=None):
    """Capture the current state of all parcels as a Snapshot."""
    objects = []
    total_ke = 0.0
    total_pe = 0.0

    for p in physics_scene.parcels:
        if p.is_static:
            continue

        speed = p.velocity.length()
        ke = p.kinetic_energy()
        total_ke += ke

        # Gravitational PE relative to ground (PE = mgh along gravity direction)
        if gravity is not None:
            g_mag = gravity.length()
            if g_mag > 1e-12:
                # Height = position projected along -gravity direction
                g_hat = gravity * (-1.0 / g_mag)
                h = p.position.dot(g_hat)
                pe = p.mass * g_mag * h
                total_pe += pe

        objects.append(ObjectState(
            name=p.label,
            position=(p.position.x, p.position.y, p.position.z),
            velocity=(p.velocity.x, p.velocity.y, p.velocity.z),
            speed=speed,
            kinetic_energy=ke,
        ))

    return Snapshot(
        time=time,
        objects=objects,
        event_name=event_name,
        total_kinetic_energy=total_ke,
        total_potential_energy=total_pe,
    )


def _detect_collisions(physics_scene, time, prev_ke):
    """Detect collisions by checking for energy changes and proximity.

    Simple approach: if two dynamic parcels overlap, record a collision.
    """
    from ..dynamics.collision import sphere_sphere_collision

    collisions = []
    parcels = physics_scene.parcels
    n = len(parcels)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = parcels[i], parcels[j]
            is_col, penetration, n_hat = sphere_sphere_collision(p1, p2)
            if is_col:
                v_rel = abs((p1.velocity - p2.velocity).dot(n_hat))
                if v_rel > 0.01:  # threshold to avoid rest-contact noise
                    e = min(p1.restitution, p2.restitution)
                    inv_m = p1.inv_mass + p2.inv_mass
                    if inv_m > 0:
                        j_impulse = (1 + e) * v_rel / inv_m
                    else:
                        j_impulse = 0
                    ke_before = 0.5 * v_rel**2 / inv_m if inv_m > 0 else 0
                    ke_after = ke_before * e**2
                    collisions.append(CollisionEvent(
                        time=time,
                        object_a=p1.label,
                        object_b=p2.label,
                        relative_velocity=v_rel,
                        impulse=j_impulse,
                        energy_dissipated=ke_before - ke_after,
                    ))

    # Ground collisions
    if physics_scene.ground is not None:
        gnd = physics_scene.ground
        from ..dynamics.collision import sphere_plane_collision
        for p in parcels:
            if p.is_static:
                continue
            is_col, pen, n_vec = sphere_plane_collision(p, gnd.point, gnd.normal)
            if is_col:
                v_rel = abs(p.velocity.dot(gnd.normal))
                if v_rel > 0.01:
                    collisions.append(CollisionEvent(
                        time=time,
                        object_a=p.label,
                        object_b='ground',
                        relative_velocity=v_rel,
                        impulse=(1 + gnd.restitution) * v_rel * p.mass,
                        energy_dissipated=0.5 * p.mass * v_rel**2 * (1 - gnd.restitution**2),
                    ))

    return collisions


def _check_event(event, physics_scene, time):
    """Check if a SimEvent condition is triggered.

    Returns True if the event just triggered.
    """
    cond = event.condition

    # Time-based: "t=0.5"
    if cond.startswith('t='):
        t_target = float(cond[2:])
        # Check if we just crossed the target time
        return abs(time - t_target) < 0.005

    # Collision-based: "collision" or "collision:ball,ground"
    if cond.startswith('collision'):
        parcels = physics_scene.parcels
        from ..dynamics.collision import sphere_sphere_collision
        for i in range(len(parcels)):
            for j in range(i + 1, len(parcels)):
                is_col, _, _ = sphere_sphere_collision(parcels[i], parcels[j])
                if is_col:
                    if ':' in cond:
                        names = cond.split(':')[1].split(',')
                        if (parcels[i].label in names and
                                parcels[j].label in names):
                            return True
                    else:
                        return True
        return False

    # Rest: all objects have velocity < threshold
    if cond == 'rest':
        for p in physics_scene.parcels:
            if not p.is_static and p.velocity.length() > 0.01:
                return False
        return True

    return False


def run_simulation(scene):
    """Execute a simulation scene end-to-end.

    Args:
        scene: SimulationScene instance.

    Returns:
        SimulationResult with snapshots, collisions, and summary.
    """
    # Build the physics scene
    physics_scene, forces_cb, material_props = build(scene)

    gravity = physics_scene.gravity
    snapshots = []
    all_collisions = []
    steps_taken = 0
    next_snap = 0.0

    # Track which events have fired (fire each only once)
    events_fired = set()

    # Capture initial state
    snapshots.append(_capture_snapshot(physics_scene, 0.0, 'initial', gravity))
    prev_ke = physics_scene.total_kinetic_energy()

    # Step loop
    t = 0.0
    max_steps = int(scene.duration / 0.0001) + 10000  # safety limit
    step_count = 0

    while t < scene.duration and step_count < max_steps:
        dt = step(physics_scene, dt_max=0.005, sub_steps=4,
                  external_forces=forces_cb)
        t = physics_scene.time
        step_count += 1
        steps_taken += 1

        # Detect collisions
        frame_collisions = _detect_collisions(physics_scene, t, prev_ke)
        all_collisions.extend(frame_collisions)

        # Regular snapshots
        if t >= next_snap:
            snapshots.append(_capture_snapshot(physics_scene, t,
                                               gravity=gravity))
            next_snap += scene.snapshot_interval

        # Named events
        for event in scene.events:
            if event.name not in events_fired:
                if _check_event(event, physics_scene, t):
                    events_fired.add(event.name)
                    snapshots.append(_capture_snapshot(
                        physics_scene, t, event.name, gravity))

        prev_ke = physics_scene.total_kinetic_energy()

    # Capture final state
    snapshots.append(_capture_snapshot(physics_scene, t, 'final', gravity))

    # Build summary
    initial = snapshots[0]
    final = snapshots[-1]

    summary = {
        'duration': t,
        'steps': steps_taken,
        'total_collisions': len(all_collisions),
        'objects': {},
    }

    for obj_state in final.objects:
        init_state = next(
            (o for o in initial.objects if o.name == obj_state.name), None)
        summary['objects'][obj_state.name] = {
            'final_position': list(obj_state.position),
            'final_velocity': list(obj_state.velocity),
            'final_speed': obj_state.speed,
            'initial_speed': init_state.speed if init_state else 0,
        }

    return SimulationResult(
        scene_name=scene.name,
        description=scene.description,
        snapshots=snapshots,
        collisions=all_collisions,
        duration=t,
        steps_taken=steps_taken,
        material_properties=material_props,
        summary=summary,
    )
