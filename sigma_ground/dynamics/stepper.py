"""
Leapfrog integrator for rigid-body dynamics.

Integration scheme: Leapfrog (Störmer-Verlet)
==============================================
Leapfrog is symplectic — it conserves a modified energy (shadow Hamiltonian)
exactly at each step, unlike Euler which accumulates energy error.

Half-step velocity update → full-step position update → half-step velocity:
  v(t + dt/2) = v(t) + a(t) × dt/2
  x(t + dt)   = x(t) + v(t + dt/2) × dt
  a(t + dt)   = F(x(t+dt)) / m          [compute new forces at new position]
  v(t + dt)   = v(t + dt/2) + a(t+dt) × dt/2

For constant gravity (no position-dependent forces beyond collision), this
simplifies in the rigid-body case to standard leapfrog.

FIRST_PRINCIPLES: symplectic integration is required for any Hamiltonian
system. Euler integration violates the symplectic structure → energy drifts
unboundedly. Leapfrog is O(dt²) accurate and time-reversible. This matches
the physical system at the cost of requiring the half-step velocity.

Reference: Leimkuhler & Reich (2004) "Simulating Hamiltonian Dynamics"
           Cambridge University Press, §4.1.

CFL timestep constraint
=======================
  dt_max = 0.4 × r_min / v_max

Where r_min is the smallest object radius and v_max is the largest speed.
This ensures no object travels more than 40% of its own radius per step,
preventing tunneling through thin objects.

The 0.4 factor is the CFL coefficient. For rigid-body simulations with
inelastic collisions, 0.3–0.5 is standard. We use 0.4.
NOT_PHYSICS: the exact coefficient is a numerical parameter, not a physical one.

Reference: Courant, Friedrichs & Lewy (1928). The continuous version limits
wave propagation to sub-cell distances. We adapt it to bounding-sphere cells.

Collision ordering
==================
All O(n²) pairs are tested per step. For small scenes (< 50 objects) this is
acceptable. For large scenes, replace with a spatial hash (see fluid/sph.py).

Gravity
=======
  a_gravity = scene.gravity   (constant Vec3, typically Vec3(0, -g, 0))
  F = m × a_gravity
  v += a_gravity × dt

σ-dependence
============
  mass m(σ) already encoded in each parcel.mass via density_at_sigma(σ).
  The acceleration a = F/m = g is σ-INVARIANT for gravity (equivalence
  principle — all masses fall at the same rate regardless of σ).
  Only the momentum p = m(σ)v and kinetic energy ½m(σ)v² change with σ.
"""

import math
from .collision import collect_pair_contacts, collect_ground_contacts
from .solver import solve_velocity, project_positions
from .quat import quat_step


_CFL_COEFF = 0.40      # NOT_PHYSICS — numerical stability parameter


def _cfl_dt(scene, dt_max):
    """Compute CFL-limited timestep.

    dt = min(dt_max, CFL × r_min / max(v_max, eps))

    Returns dt in seconds.
    """
    r_min = float('inf')
    v_max = 0.0
    for p in scene.parcels:
        if not p.is_static:
            r_min = min(r_min, p.radius)
            v = p.velocity.length()
            v_max = max(v_max, v)

    if r_min == float('inf') or v_max < 1e-12:
        return dt_max

    dt_cfl = _CFL_COEFF * r_min / v_max
    return min(dt_max, dt_cfl)


def step(scene, dt=None, dt_max=0.01, sub_steps=4, external_forces=None):
    """Advance the scene by one timestep.

    Uses leapfrog integration with CFL-limited substeps.

    Args:
        scene:     PhysicsScene to advance.
        dt:        Desired timestep in seconds. None → use CFL-computed dt.
        dt_max:    Maximum timestep per call (default 10 ms).
        sub_steps: Number of internal substeps per call.
                   Finer resolution improves collision accuracy.
                   Default: 4 (so each substep dt = dt/4).
        external_forces: Optional callback f(parcel) → Vec3 returning
                   additional force on each parcel (drag, buoyancy, etc.).
                   Applied as acceleration a = F/m during each substep.

    Returns:
        actual_dt (float): total elapsed time this call.

    Modifies:
        scene.parcels[*].position and .velocity in place.
        scene.time incremented.
    """
    if dt is None:
        dt = _cfl_dt(scene, dt_max)
    else:
        dt = min(dt, dt_max)

    sub_dt = dt / sub_steps

    for _ in range(sub_steps):
        _leapfrog_sub_step(scene, sub_dt, external_forces)

    scene.time += dt
    return dt


def step_to(scene, t_end, dt_max=0.01, sub_steps=4, callback=None,
            external_forces=None):
    """Advance the scene until scene.time >= t_end.

    Args:
        scene:    PhysicsScene.
        t_end:    Target simulation time (s).
        dt_max:   Maximum timestep per call.
        sub_steps: Substeps per call.
        callback: Optional function(scene, frame_index) called after each step.
                  Useful for recording frame data.
        external_forces: Optional callback f(parcel) → Vec3 for extra forces.

    Returns:
        history: list of (time, [(label, pos, vel), …]) per frame.
    """
    history = []
    frame   = 0
    while scene.time < t_end:
        remaining = t_end - scene.time
        dt = min(dt_max, remaining)
        step(scene, dt=dt, sub_steps=sub_steps,
             external_forces=external_forces)
        snapshot = _snapshot(scene)
        history.append((scene.time, snapshot))
        if callback:
            callback(scene, frame)
        frame += 1

    return history


def _apply_external(p, external_forces, half_dt):
    """One half-kick of the external-force callback on parcel ``p``.

    The callback may return either a Vec3 FORCE (the original contract —
    drag, buoyancy: applied at the CoM, no torque) or a (force, torque)
    TUPLE for distributed loads whose resultant does not pass through the
    CoM (wind on pitched blades, paddle drag): the torque is applied as an
    angular impulse tau*dt/2 through the parcel's real inverse inertia,
    keeping the same half-kick placement (and thus the same symplectic
    structure) as the linear part. Before this, the stepper had NO path
    from an external force to a torque — a body could be pushed but never
    externally spun, which is exactly what natural drives (wind against a
    windmill's blades) need."""
    out = external_forces(p)
    if isinstance(out, tuple):
        F_ext, tau_ext = out
    else:
        F_ext, tau_ext = out, None
    p.velocity = p.velocity + F_ext * (half_dt / p.mass)
    if tau_ext is not None and (tau_ext.x or tau_ext.y or tau_ext.z):
        p.apply_angular_impulse(tau_ext * half_dt)


def _leapfrog_sub_step(scene, dt, external_forces=None):
    """Single leapfrog substep.

    1. Half-step velocity update (gravity + external forces)
    2. Full-step position update
    3. Collision detection and impulse response
    4. Second half-step velocity

    For constant gravity the full leapfrog collapses to:
      v_half = v + a × dt/2
      x_new  = x + v_half × dt
      [resolve collisions at x_new — this modifies v_half]
      v_new  = v_half + a × dt/2

    External forces (drag, buoyancy, etc.) are applied as accelerations
    a_ext = F_ext(parcel) / m during both half-steps, maintaining the
    symplectic structure of the integrator. A callback may also return
    (force, torque) — see _apply_external.
    """
    g = scene.gravity

    # ── Step 1: half-step velocity from gravity + external forces ─────────
    for p in scene.parcels:
        if p.is_static:
            continue
        half_dv = g * (dt * 0.5)
        p.velocity = p.velocity + half_dv

        if external_forces is not None and p.mass > 0 and p.mass != float('inf'):
            _apply_external(p, external_forces, dt * 0.5)

    # ── Step 1.5: pre-drift constraint solve (manifold contacts + joints) ────
    # The drift-then-solve order (legacy, kept for bit-parity) leaks the
    # half-kick velocity into a full drift before any constraint can object:
    #   - for a box resting under friction that is a g·dt tangential creep
    #     per substep, which reads as false sliding;
    #   - for a pendulum it is a radial velocity component whose deletion
    #     (plus the projection's PE work) drains real energy every step.
    # Solving the ALREADY-overlapping manifold contacts on the half-kicked
    # velocity BEFORE drift (velocities only, no position pass) kills the
    # creep. Joints run their SHAKE rows here — position error solved
    # THROUGH the pre-drift velocity so the drift lands ON the constraint
    # manifold (first half of the RATTLE constrained leapfrog; a measured
    # secular sink otherwise: plain post-drift deletion drained the double
    # pendulum ~0.24 J/s). Legacy center-line rows are excluded on purpose:
    # sphere scenes keep the exact legacy sequence, and a separating legacy
    # row would have skipped itself anyway.
    joints = getattr(scene, "constraints", None) or []
    for j in joints:
        reset = getattr(j, "reset_substep", None)
        if reset is not None:
            reset()                        # motor impulse caps span the substep
    pre = []
    if scene.ground is not None:
        pre = [c for c in collect_ground_contacts(
                   scene.parcels, scene.ground,
                   restitution_fn=getattr(scene, "restitution_fn", None))
               if c.kind == "manifold"]
    iters = getattr(scene, "solver_iterations", 10)
    if pre or joints:
        mu_fn = getattr(scene, "friction_fn", None)
        fr = ((lambda c: mu_fn(c.a, c.b)) if mu_fn is not None else None)
        solve_velocity(pre, joints, dt, iterations=iters, friction_fn=fr,
                       shake=True)

    # ── Step 2: full-step position update (drift: linear + angular) ──────────
    # Angular drift by KDK placement: ω was half-kicked by any contact/joint
    # impulses of the PREVIOUS substep; the quaternion advances on the full
    # step. quat_step renormalizes (unit-norm stable over 1e5 steps — gated).
    for p in scene.parcels:
        if p.is_static:
            continue
        p.position = p.position + p.velocity * dt
        w = p.angular_velocity
        if w.x != 0.0 or w.y != 0.0 or w.z != 0.0:
            p.orientation = list(quat_step(p.orientation,
                                           (w.x, w.y, w.z), dt))

    # ── Step 3: constraint resolution (the ONE enforcement pass) ─────────────
    # Contacts are COLLECTED from the drifted poses, then handed with the
    # scene's joints to dynamics/solver.py (sequential impulses + split
    # position projection). Center-line sphere rows carry kind="legacy" and
    # reproduce the old resolve_* impulses exactly (r×n̂=0, skip-when-
    # separating, single 0.8·pen position pass) — gated at 1e-12.
    parcels = scene.parcels
    restitution_fn = getattr(scene, "restitution_fn", None)
    contacts = collect_pair_contacts(parcels, restitution_fn=restitution_fn)
    if scene.ground is not None:
        contacts.extend(collect_ground_contacts(
            parcels, scene.ground, restitution_fn=restitution_fn))
    # Joints join this pass only when contacts exist (contact↔joint coupling,
    # e.g. a hinged lid resting on a table); without contacts the RATTLE
    # stages own the joints — re-solving them here would re-introduce the
    # radial-velocity-deletion sink the SHAKE stage exists to remove.
    if contacts:
        mu_fn = getattr(scene, "friction_fn", None)
        friction = ((lambda c: mu_fn(c.a, c.b)) if mu_fn is not None
                    else None)
        solve_velocity(contacts, joints, dt, iterations=iters,
                       friction_fn=friction)
        # Positions: contacts always; joints only here, where contact
        # impulses can knock anchors apart. In a joint-only scene SHAKE
        # already landed the drift ON the constraint (anchor error ~1e-7 m
        # measured) — projecting the residual was a measured energy
        # polluter, so the teleport is reserved for contact events.
        project_positions(
            contacts, joints,
            iterations=getattr(scene, "projection_iterations", 3))

    # ── Step 4: second half-step velocity ─────────────────────────────────────
    for p in scene.parcels:
        if p.is_static:
            continue
        half_dv = g * (dt * 0.5)
        p.velocity = p.velocity + half_dv

        if external_forces is not None and p.mass > 0 and p.mass != float('inf'):
            _apply_external(p, external_forces, dt * 0.5)

    # ── Step 5: RATTLE velocity projection (joints only) ─────────────────────
    # The second half of the constrained leapfrog: remove the constraint-
    # violating velocity components from the FULL-step velocity (after the
    # second half-kick). Paired with the SHAKE stage this keeps the energy
    # error bounded instead of secular.
    if joints:
        solve_velocity([], joints, dt, iterations=iters)


def _snapshot(scene):
    """Capture current (label, position, velocity) for each parcel."""
    return [(p.label, Vec3(p.position.x, p.position.y, p.position.z),
             Vec3(p.velocity.x, p.velocity.y, p.velocity.z))
            for p in scene.parcels]


# ── Import fix for _snapshot's Vec3 usage ────────────────────────────────────
from .vec import Vec3
