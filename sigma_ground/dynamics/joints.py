"""Typed articulation constraints — the A-graph's joints, enforced by the
ONE stepper (dynamics/solver.py orchestrates; each joint solves itself
through the shared body state).

Joint types (the Holodeck grammar's verbs of connection):
  WeldJoint      — locks all 6 relative DOF (rigid attachment).
  RevoluteJoint  — hinge: locks 5 DOF, leaves rotation about one axis;
                   optional angle limits (rad) and a torque-capped motor.
  PrismaticJoint — slider: locks 5 DOF, leaves translation along one axis;
                   optional travel limits (m). (Motors deferred — plan Q6.)

Protocol (called by solver.solve_velocity / solver.project_positions):
  begin(dt)          — cache pose-dependent geometry for this substep and
                       reset accumulated motor/limit impulses. Poses are
                       FIXED during the velocity pass, so anchors, axes and
                       effective-mass blocks are computed once.
  solve_velocity(dt) — one Gauss–Seidel sweep: remove constraint-violating
                       relative velocity via impulses (direct 3×3/2×2 block
                       solves — exact for a single joint, iterated across
                       many).
  project()          — split-impulse position/orientation projection from
                       CURRENT poses (velocities untouched — no Baumgarte
                       energy pollution).

Conventions:
  - All "relative" quantities are b-relative-to-a: c_dot = v_b − v_a at the
    anchor, w_rel = ω_b − ω_a, and a RevoluteJoint's angle()/motor speed s
    satisfy s = dθ/dt.
  - b=None means the WORLD: an immovable anchor/orientation reference
    (inv_mass = 0, I⁻¹ = 0, identity orientation). Author joints PARENT
    FIRST (a=parent, b=child) so angle()/travel() read as "child relative
    to parent"; with b=None the world plays the CHILD, so a body sliding
    +n̂ reads travel() < 0 and a body spinning +n̂ reads dθ/dt < 0.
  - Anchors and axes are given in WORLD coordinates AT CONSTRUCTION TIME and
    stored in each body's local frame (the joint rides with the bodies).
  - Impulse convention mirrors solver.Contact: +P/+L to b, −P/−L to a.

Energy accounting:
  - Constraint rows are workless in the converged limit (they remove
    relative velocity along locked DOF) — the "joints inject ≈ 0 energy"
    ledger gate is measurable per test.
  - Motors LOG their work: motor_work_j accumulates Σ λ·(s_mid) per applied
    impulse, so the gate ΔE ≤ ∫τ·ω dt is checkable from the ledger.

FIRST_PRINCIPLES: constraint impulses via effective-mass (Schur complement)
blocks — Newton–Euler with ideal bilateral constraints. NOT_PHYSICS pieces:
Gauss–Seidel iteration count and the position-projection style are numerical
choices (same doctrine as collision.py's flagged slop/β constants).
"""
import math

from .vec import Vec3
from .quat import (qrot, qrot_inv, quat_mul, quat_conj, quat_normalize,
                   quat_step, twist_angle, solve3)

_IDENTITY = (0.0, 0.0, 0.0, 1.0)


# ── shared body-state helpers (b=None → the world) ──────────────────────────

def _orient(p):
    return p.orientation if p is not None else _IDENTITY


def _ang_vel(p):
    return p.angular_velocity if p is not None else Vec3(0.0, 0.0, 0.0)


def _pt_vel(p, r):
    """Velocity of the material point at world offset r from p's CoM."""
    if p is None:
        return Vec3(0.0, 0.0, 0.0)
    w = p.angular_velocity
    return Vec3(p.velocity.x + w.y * r.z - w.z * r.y,
                p.velocity.y + w.z * r.x - w.x * r.z,
                p.velocity.z + w.x * r.y - w.y * r.x)


def _dv_unit(a, ra, b, rb, d):
    """Δ(v_b − v_a) at the anchors for a unit impulse d (+ to b, − to a)."""
    ima = a.inv_mass if a is not None else 0.0
    imb = b.inv_mass if b is not None else 0.0
    dv = d * (ima + imb)
    if a is not None and not a.is_static:
        dv = dv + a.inv_inertia_apply(ra.cross(d)).cross(ra)
    if b is not None and not b.is_static:
        dv = dv + b.inv_inertia_apply(rb.cross(d)).cross(rb)
    return dv


def _k3_point(a, ra, b, rb):
    """3×3 effective-mass matrix for the anchor-point constraint (rows)."""
    c0 = _dv_unit(a, ra, b, rb, Vec3(1.0, 0.0, 0.0))
    c1 = _dv_unit(a, ra, b, rb, Vec3(0.0, 1.0, 0.0))
    c2 = _dv_unit(a, ra, b, rb, Vec3(0.0, 0.0, 1.0))
    return ((c0.x, c1.x, c2.x),
            (c0.y, c1.y, c2.y),
            (c0.z, c1.z, c2.z))


def _iinv_sum(a, b, d):
    """(I_a⁻¹ + I_b⁻¹)·d, world frame."""
    out = Vec3(0.0, 0.0, 0.0)
    if a is not None and not a.is_static:
        out = out + a.inv_inertia_apply(d)
    if b is not None and not b.is_static:
        out = out + b.inv_inertia_apply(d)
    return out


def _k3_ang(a, b):
    """3×3 effective-mass matrix for the angular-lock constraint (rows)."""
    c0 = _iinv_sum(a, b, Vec3(1.0, 0.0, 0.0))
    c1 = _iinv_sum(a, b, Vec3(0.0, 1.0, 0.0))
    c2 = _iinv_sum(a, b, Vec3(0.0, 0.0, 1.0))
    return ((c0.x, c1.x, c2.x),
            (c0.y, c1.y, c2.y),
            (c0.z, c1.z, c2.z))


def _apply_point(a, ra, b, rb, P):
    """Anchor impulse: +P to b, −P to a (the b-relative-to-a convention)."""
    if a is not None and not a.is_static:
        a.velocity = a.velocity - P * a.inv_mass
        a.apply_angular_impulse(ra.cross(P) * -1.0)
    if b is not None and not b.is_static:
        b.velocity = b.velocity + P * b.inv_mass
        b.apply_angular_impulse(rb.cross(P))


def _apply_ang(a, b, L):
    """Pure angular impulse: +L to b, −L to a."""
    if a is not None and not a.is_static:
        a.apply_angular_impulse(L * -1.0)
    if b is not None and not b.is_static:
        b.apply_angular_impulse(L)


def _rotate(p, dth):
    """Rotate p's orientation by the small world-frame rotation vector dth."""
    p.orientation = list(quat_step(p.orientation,
                                   (dth.x, dth.y, dth.z), 1.0))


def _rotvec(q):
    """Rotation vector (axis·angle, world frame) of a quaternion, shortest arc."""
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return Vec3(2.0 * x, 2.0 * y, 2.0 * z)   # small-angle limit
    ang = 2.0 * math.atan2(n, w)
    s = ang / n
    return Vec3(x * s, y * s, z * s)


def _unit(v):
    l = v.length()
    if l < 1e-30:
        raise ValueError("joint axis must be non-zero")
    return v * (1.0 / l)


def _predict(p, dt):
    """(position, orientation) of p after drifting dt under CURRENT
    velocities — the pose the upcoming drift stage will produce."""
    pos = p.position + p.velocity * dt
    w = p.angular_velocity
    if w.x != 0.0 or w.y != 0.0 or w.z != 0.0:
        q = quat_step(p.orientation, (w.x, w.y, w.z), dt)
    else:
        q = p.orientation
    return pos, q


def _tangent_basis(n):
    """Two unit tangents orthogonal to n (deterministic choice)."""
    ref = Vec3(1.0, 0.0, 0.0) if abs(n.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
    t1 = _unit(n.cross(ref))
    t2 = n.cross(t1)
    return t1, t2


# ── base: shared anchor bookkeeping + the point (ball) constraint ────────────

class _JointBase:
    def __init__(self, a, b, anchor_world):
        if a is None:
            raise ValueError("joint body a must be a parcel (b may be None)")
        self.a = a
        self.b = b
        pa = anchor_world - a.position
        self.la = qrot_inv(a.orientation, (pa.x, pa.y, pa.z))
        if b is not None:
            pb = anchor_world - b.position
            self.lb = qrot_inv(b.orientation, (pb.x, pb.y, pb.z))
        else:
            self.lb = None
            self.anchor_w = Vec3(anchor_world.x, anchor_world.y,
                                 anchor_world.z)
        # rest relative orientation q_a⁻¹ ⊗ q_b, captured at construction
        self.q0_rel = quat_normalize(
            quat_mul(quat_conj(a.orientation), _orient(b)))

    def _geom(self):
        """(ra, wa, rb, wb) — world lever arms and anchor points, CURRENT."""
        a, b = self.a, self.b
        ta = qrot(a.orientation, self.la)
        ra = Vec3(ta[0], ta[1], ta[2])
        wa = a.position + ra
        if b is not None:
            tb = qrot(b.orientation, self.lb)
            rb = Vec3(tb[0], tb[1], tb[2])
            wb = b.position + rb
        else:
            rb = Vec3(0.0, 0.0, 0.0)
            wb = self.anchor_w
        return ra, wa, rb, wb

    def _q_rel(self):
        return quat_mul(quat_conj(self.a.orientation), _orient(self.b))

    def _solve_point_velocity(self, ra, rb):
        a, b = self.a, self.b
        cd = _pt_vel(b, rb) - _pt_vel(a, ra)
        K = self._k_pt
        P = solve3(K, (-cd.x, -cd.y, -cd.z))
        _apply_point(a, ra, b, rb, Vec3(P[0], P[1], P[2]))

    def _predicted_errors(self, dt):
        """(anchor error, orientation error) at the PREDICTED post-drift
        poses under the CURRENT velocities. Iterating impulses against this
        NONLINEAR error is SHAKE proper: the drift lands ON the constraint
        manifold (to iteration tolerance), so no position teleport is needed
        — the teleport was a measured energy polluter."""
        a, b = self.a, self.b
        pa, qa = _predict(a, dt)
        ta = qrot(qa, self.la)
        wa = pa + Vec3(ta[0], ta[1], ta[2])
        if b is not None:
            pb, qb = _predict(b, dt)
            tb = qrot(qb, self.lb)
            wb = pb + Vec3(tb[0], tb[1], tb[2])
        else:
            qb = _IDENTITY
            wb = self.anchor_w
        err = wb - wa
        q_target = quat_mul(qa, self.q0_rel)
        phi = _rotvec(quat_mul(qb, quat_conj(q_target)))
        return err, phi, qa, qb

    def _solve_point_shake(self, err, dt):
        """SHAKE point row: impulse chosen so the anchors COINCIDE after the
        upcoming drift: K·P = −err_pred/dt (K and lever arms frozen at the
        begin pose — a Jacobian approximation; the fixed point still
        satisfies the exact predicted constraint)."""
        a, b = self.a, self.b
        ra, rb = self._ra, self._rb
        inv = 1.0 / dt
        P = solve3(self._k_pt, (-err.x * inv,
                                -err.y * inv,
                                -err.z * inv))
        _apply_point(a, ra, b, rb, Vec3(P[0], P[1], P[2]))

    def _project_point(self):
        """Move poses so the anchors coincide (split-impulse: no velocity)."""
        a, b = self.a, self.b
        ra, wa, rb, wb = self._geom()
        err = wb - wa
        if err.length() < 1e-12:
            return
        K = _k3_point(a, ra, b, rb)
        P = solve3(K, (-err.x, -err.y, -err.z))
        P = Vec3(P[0], P[1], P[2])
        if not a.is_static:
            a.position = a.position - P * a.inv_mass
            _rotate(a, a.inv_inertia_apply(ra.cross(P)) * -1.0)
        if b is not None and not b.is_static:
            b.position = b.position + P * b.inv_mass
            _rotate(b, b.inv_inertia_apply(rb.cross(P)))

    def _orientation_error(self):
        """World rotation vector by which b is over-rotated vs. its target
        orientation q_a ⊗ q0_rel (zero when the joint is at rest pose)."""
        q_target = quat_mul(self.a.orientation, self.q0_rel)
        q_err = quat_mul(_orient(self.b), quat_conj(q_target))
        return _rotvec(q_err)

    def _project_orientation(self, phi):
        """Distribute the orientation error phi by inverse inertia."""
        a, b = self.a, self.b
        if phi.length() < 1e-12:
            return
        K = _k3_ang(a, b)
        L = solve3(K, (phi.x, phi.y, phi.z))
        L = Vec3(L[0], L[1], L[2])
        if not a.is_static:
            _rotate(a, a.inv_inertia_apply(L))
        if b is not None and not b.is_static:
            _rotate(b, b.inv_inertia_apply(L) * -1.0)


# ── weld ─────────────────────────────────────────────────────────────────────

class WeldJoint(_JointBase):
    """Rigid attachment: anchor points coincide AND relative orientation is
    frozen at its construction value. Six locked DOF."""

    def begin(self, dt):
        ra, wa, rb, wb = self._geom()
        self._ra, self._rb = ra, rb
        self._k_pt = _k3_point(self.a, ra, self.b, rb)
        self._k_ang = _k3_ang(self.a, self.b)

    def solve_velocity(self, dt):
        a, b = self.a, self.b
        self._solve_point_velocity(self._ra, self._rb)
        w_rel = _ang_vel(b) - _ang_vel(a)
        L = solve3(self._k_ang, (-w_rel.x, -w_rel.y, -w_rel.z))
        _apply_ang(a, b, Vec3(L[0], L[1], L[2]))

    def solve_shake(self, dt):
        a, b = self.a, self.b
        err, phi, _, _ = self._predicted_errors(dt)
        self._solve_point_shake(err, dt)
        inv = 1.0 / dt
        L = solve3(self._k_ang, (-phi.x * inv,
                                 -phi.y * inv,
                                 -phi.z * inv))
        _apply_ang(a, b, Vec3(L[0], L[1], L[2]))

    def project(self):
        self._project_point()
        self._project_orientation(self._orientation_error())


# ── revolute (hinge) ─────────────────────────────────────────────────────────

class RevoluteJoint(_JointBase):
    """Hinge about a fixed axis (stored in a's frame): 5 locked DOF.

    limits:   (lo, hi) angle window in RADIANS about the construction pose
              (angle() reads 0 at construction; s = dθ/dt = (ω_b − ω_a)·n̂).
              Limit rows only brake (inelastic stop, no restitution).
    motor:    motor_speed (rad/s target for s) + motor_max_torque (N·m cap);
              the impulse per substep is clamped to ±τ_max·dt and the work
              is logged in motor_work_j (the ΔE ≤ ∫τ·ω dt ledger gate).
    """

    def __init__(self, a, b, anchor_world, axis_world, limits=None,
                 motor_speed=None, motor_max_torque=0.0):
        super().__init__(a, b, anchor_world)
        ax = _unit(axis_world)
        self.axis_a = qrot_inv(a.orientation, (ax.x, ax.y, ax.z))
        self.limits = limits
        self.motor_speed = motor_speed
        self.motor_max_torque = float(motor_max_torque)
        self.motor_work_j = 0.0
        self._lam_motor = 0.0

    def reset_substep(self):
        """Called ONCE per substep by the stepper: the motor's impulse cap
        (±τ_max·dt) spans the whole substep even though the solver runs a
        pre-drift and a post-drift velocity pass (begin() runs per pass)."""
        self._lam_motor = 0.0

    def angle(self):
        """Hinge angle vs. the construction pose (rad); dθ/dt = (ω_b−ω_a)·n̂."""
        q_d = quat_mul(self._q_rel(), quat_conj(self.q0_rel))
        return twist_angle(q_d, self.axis_a)

    def _axis_world(self):
        t = qrot(self.a.orientation, self.axis_a)
        return Vec3(t[0], t[1], t[2])

    def begin(self, dt):
        a, b = self.a, self.b
        ra, wa, rb, wb = self._geom()
        self._ra, self._rb = ra, rb
        self._k_pt = _k3_point(a, ra, b, rb)
        n = self._axis_world()
        self._n = n
        t1, t2 = _tangent_basis(n)
        self._t1, self._t2 = t1, t2
        self._k11 = t1.dot(_iinv_sum(a, b, t1))
        self._k12 = t1.dot(_iinv_sum(a, b, t2))
        self._k22 = t2.dot(_iinv_sum(a, b, t2))
        self._k_axis = n.dot(_iinv_sum(a, b, n))
        self._theta = self.angle()
        self._lam_lo = 0.0
        self._lam_hi = 0.0

    def solve_velocity(self, dt):
        a, b = self.a, self.b
        self._solve_point_velocity(self._ra, self._rb)

        # swing lock: kill relative rotation ⊥ axis (2×2 block)
        n, t1, t2 = self._n, self._t1, self._t2
        w_rel = _ang_vel(b) - _ang_vel(a)
        c1, c2 = w_rel.dot(t1), w_rel.dot(t2)
        det = self._k11 * self._k22 - self._k12 * self._k12
        if abs(det) > 1e-30:
            l1 = (-c1 * self._k22 + c2 * self._k12) / det
            l2 = (-c2 * self._k11 + c1 * self._k12) / det
            _apply_ang(a, b, t1 * l1 + t2 * l2)

        k = self._k_axis
        if k <= 1e-30:
            return

        # motor: drive s → motor_speed under the torque cap
        if self.motor_speed is not None and self.motor_max_torque > 0.0:
            s = (_ang_vel(b) - _ang_vel(a)).dot(n)
            dl = (self.motor_speed - s) / k
            cap = self.motor_max_torque * dt
            new = min(max(self._lam_motor + dl, -cap), cap)
            dl = new - self._lam_motor
            self._lam_motor = new
            if dl:
                _apply_ang(a, b, n * dl)
                self.motor_work_j += dl * (s + 0.5 * dl * k)   # λ·s_mid

        # limits: brake-only rows at the window edges (no restitution)
        if self.limits is not None:
            lo, hi = self.limits
            th = self._theta
            if th <= lo:
                s = (_ang_vel(b) - _ang_vel(a)).dot(n)
                new = max(self._lam_lo - s / k, 0.0)
                dl = new - self._lam_lo
                self._lam_lo = new
                if dl:
                    _apply_ang(a, b, n * dl)
            elif th >= hi:
                s = (_ang_vel(b) - _ang_vel(a)).dot(n)
                new = min(self._lam_hi - s / k, 0.0)
                dl = new - self._lam_hi
                self._lam_hi = new
                if dl:
                    _apply_ang(a, b, n * dl)

    def solve_shake(self, dt):
        """SHAKE rows against the PREDICTED post-drift pose: point + swing
        errors solved through the pre-drift velocity; limits land the
        predicted angle inelastically ON the boundary; the motor drives as
        usual (velocity target, substep-capped)."""
        a, b = self.a, self.b
        err, phi, qa, qb = self._predicted_errors(dt)
        self._solve_point_shake(err, dt)
        inv = 1.0 / dt

        # swing: kill the predicted ⊥-axis orientation error
        n, t1, t2 = self._n, self._t1, self._t2
        c1 = phi.dot(t1) * inv
        c2 = phi.dot(t2) * inv
        det = self._k11 * self._k22 - self._k12 * self._k12
        if abs(det) > 1e-30:
            l1 = (-c1 * self._k22 + c2 * self._k12) / det
            l2 = (-c2 * self._k11 + c1 * self._k12) / det
            _apply_ang(a, b, t1 * l1 + t2 * l2)

        k = self._k_axis
        if k <= 1e-30:
            return

        if self.motor_speed is not None and self.motor_max_torque > 0.0:
            s = (_ang_vel(b) - _ang_vel(a)).dot(n)
            dl = (self.motor_speed - s) / k
            cap = self.motor_max_torque * dt
            new = min(max(self._lam_motor + dl, -cap), cap)
            dl = new - self._lam_motor
            self._lam_motor = new
            if dl:
                _apply_ang(a, b, n * dl)
                self.motor_work_j += dl * (s + 0.5 * dl * k)

        if self.limits is not None:
            lo, hi = self.limits
            # predicted hinge angle from the predicted quaternions
            q_d = quat_mul(quat_mul(quat_conj(qa), qb),
                           quat_conj(self.q0_rel))
            th_pred = twist_angle(q_d, self.axis_a)
            s = (_ang_vel(b) - _ang_vel(a)).dot(n)
            if th_pred < lo:
                target = s + (lo - th_pred) * inv  # land ON the boundary
                new = max(self._lam_lo + (target - s) / k, 0.0)
                dl = new - self._lam_lo
                self._lam_lo = new
                if dl:
                    _apply_ang(a, b, n * dl)
            elif th_pred > hi:
                target = s + (hi - th_pred) * inv
                new = min(self._lam_hi + (target - s) / k, 0.0)
                dl = new - self._lam_hi
                self._lam_hi = new
                if dl:
                    _apply_ang(a, b, n * dl)

    def project(self):
        self._project_point()
        # swing error only: remove the ⊥-axis part of the orientation error
        # (the twist DOF is free). NOTE: with anisotropic inertia the 3×3
        # distribution can leak a small twist — bounded by the next pass.
        n = self._axis_world()
        phi = self._orientation_error()
        phi_perp = phi - n * phi.dot(n)
        self._project_orientation(phi_perp)
        # limits: rotate the twist back inside the window
        if self.limits is not None:
            th = self.angle()
            lo, hi = self.limits
            d = 0.0
            if th < lo:
                d = lo - th
            elif th > hi:
                d = hi - th
            if d:
                a, b = self.a, self.b
                wa_s = (n.dot(a.inv_inertia_apply(n))
                        if not a.is_static else 0.0)
                wb_s = (n.dot(b.inv_inertia_apply(n))
                        if b is not None and not b.is_static else 0.0)
                tot = wa_s + wb_s
                if tot > 1e-30:
                    if wa_s:
                        _rotate(a, n * (-d * wa_s / tot))
                    if wb_s:
                        _rotate(b, n * (d * wb_s / tot))


# ── gear coupling (kinematic rate constraint between two hinges) ─────────────

class GearCouplingJoint:
    """A rigid gear-mesh RATIO between two RevoluteJoints' hinge rates:
    s_a + ratio·s_b = 0, where s = dθ/dt in each joint's OWN convention
    (see RevoluteJoint / module docstring — with b=None the world plays the
    child, so a wheel's own world-frame spin and its joint's s carry a
    built-in sign flip).

    NOT a simulated tooth mesh: real gear teeth transmit force through
    contact geometry this constraint does not model at all — it enforces
    the RATIO directly, the way a real mesh's ratio is a kinematic fact
    once teeth engage, not a force computed from tooth profiles. Involute
    geometry (when it arrives) is for RENDERING and mass properties; this
    row is what actually keeps two rendered wheels turning in lockstep.

    ratio is SIGNED and must be discovered empirically for a given axis/
    construction convention (tests/test_joints.py does this the same way
    Phase 0 discovered the motor's own sign convention) — do not assume
    ratio = +teeth_a/teeth_b without checking against the actual axes.

    Scope limit (NOT_PHYSICS): the effective mass assumes joint_a and
    joint_b do not share a body (independent shafts, the Phase 1 case —
    each wheel hinged directly to the world). Two joints sharing a body
    would need cross inertia terms this scalar row does not carry.

    No project(): this is a rate constraint like RevoluteJoint's own motor
    row (which also has no position-domain component) — re-solved fresh
    every substep, so the two hinges' RATE ratio never drifts, though the
    absolute PHASE could accumulate slow error under stress a position
    pass would correct. Acceptable for a kinematic coupling; skip until a
    scene actually needs it (e.g. a geared train also resting on a table).
    """

    def __init__(self, joint_a, joint_b, ratio):
        self.ja = joint_a
        self.jb = joint_b
        self.ratio = float(ratio)

    def _solve(self, dt):
        ja, jb, ratio = self.ja, self.jb, self.ratio
        na, nb = ja._n, jb._n
        ka, kb = ja._k_axis, jb._k_axis
        if ka <= 1e-30 or kb <= 1e-30:
            return
        k = ka + ratio * ratio * kb
        if k <= 1e-30:
            return
        sa = (_ang_vel(ja.b) - _ang_vel(ja.a)).dot(na)
        sb = (_ang_vel(jb.b) - _ang_vel(jb.a)).dot(nb)
        lam = -(sa + ratio * sb) / k
        _apply_ang(ja.a, ja.b, na * lam)
        _apply_ang(jb.a, jb.b, nb * (ratio * lam))

    def solve_velocity(self, dt):
        # begin() on ja/jb (called by solver.solve_velocity before ANY
        # joint's solve_velocity/solve_shake, this joint included) has
        # already cached ja._n/._k_axis and jb._n/._k_axis for this pass.
        self._solve(dt)

    def solve_shake(self, dt):
        # A pure rate constraint, like the motor row: no predicted-pose
        # error to solve against, so SHAKE and the live-velocity pass are
        # the same math (mirrors RevoluteJoint's own motor row, which is
        # identical in solve_velocity and solve_shake).
        self._solve(dt)

    def project(self):
        # Deliberate no-op, but the METHOD must exist: the stepper's
        # contact path calls project() on EVERY joint whenever any contact
        # exists anywhere in the scene (solver.project_positions), so a
        # missing method is a crash the moment a coupled scene also touches
        # something. The rate constraint has no position-domain component
        # to project (see class docstring).
        pass


# ── prismatic (slider) ───────────────────────────────────────────────────────

class PrismaticJoint(_JointBase):
    """Slider along a fixed axis (stored in a's frame): 5 locked DOF —
    relative rotation fully locked, translation confined to the axis.

    limits: (lo, hi) travel window in METERS for
            s = (anchor_b − anchor_a)·n̂ (0 at construction). Brake-only.
    (Motors deferred — plan Q6: prismatic motors are a later stone.)
    """

    def __init__(self, a, b, anchor_world, axis_world, limits=None):
        super().__init__(a, b, anchor_world)
        ax = _unit(axis_world)
        self.axis_a = qrot_inv(a.orientation, (ax.x, ax.y, ax.z))
        self.limits = limits

    def _axis_world(self):
        t = qrot(self.a.orientation, self.axis_a)
        return Vec3(t[0], t[1], t[2])

    def travel(self):
        """Slide distance s along the axis vs. the construction pose (m)."""
        ra, wa, rb, wb = self._geom()
        return (wb - wa).dot(self._axis_world())

    def begin(self, dt):
        a, b = self.a, self.b
        ra, wa, rb, wb = self._geom()
        self._ra, self._rb = ra, rb
        n = self._axis_world()
        self._n = n
        t1, t2 = _tangent_basis(n)
        self._t1, self._t2 = t1, t2
        d1 = _dv_unit(a, ra, b, rb, t1)
        d2 = _dv_unit(a, ra, b, rb, t2)
        self._k11 = t1.dot(d1)
        self._k12 = t1.dot(d2)
        self._k22 = t2.dot(d2)
        self._k_axis = n.dot(_dv_unit(a, ra, b, rb, n))
        self._k_ang = _k3_ang(a, b)
        self._s = (wb - wa).dot(n)
        self._lam_lo = 0.0
        self._lam_hi = 0.0

    def solve_velocity(self, dt):
        a, b = self.a, self.b
        ra, rb = self._ra, self._rb

        # full angular lock (a slider transmits ALL torque)
        w_rel = _ang_vel(b) - _ang_vel(a)
        L = solve3(self._k_ang, (-w_rel.x, -w_rel.y, -w_rel.z))
        _apply_ang(a, b, Vec3(L[0], L[1], L[2]))

        # translation ⊥ axis: kill it (2×2 block on the tangent plane)
        n, t1, t2 = self._n, self._t1, self._t2
        cd = _pt_vel(b, rb) - _pt_vel(a, ra)
        c1, c2 = cd.dot(t1), cd.dot(t2)
        det = self._k11 * self._k22 - self._k12 * self._k12
        if abs(det) > 1e-30:
            l1 = (-c1 * self._k22 + c2 * self._k12) / det
            l2 = (-c2 * self._k11 + c1 * self._k12) / det
            _apply_point(a, ra, b, rb, t1 * l1 + t2 * l2)

        # travel limits: brake-only rows along the axis
        if self.limits is not None and self._k_axis > 1e-30:
            lo, hi = self.limits
            s, k = self._s, self._k_axis
            if s <= lo:
                sd = (_pt_vel(b, rb) - _pt_vel(a, ra)).dot(n)
                new = max(self._lam_lo - sd / k, 0.0)
                dl = new - self._lam_lo
                self._lam_lo = new
                if dl:
                    _apply_point(a, ra, b, rb, n * dl)
            elif s >= hi:
                sd = (_pt_vel(b, rb) - _pt_vel(a, ra)).dot(n)
                new = min(self._lam_hi - sd / k, 0.0)
                dl = new - self._lam_hi
                self._lam_hi = new
                if dl:
                    _apply_point(a, ra, b, rb, n * dl)

    def solve_shake(self, dt):
        """SHAKE rows against the PREDICTED post-drift pose: orientation +
        ⊥-axis errors solved through the pre-drift velocity; limits land the
        predicted travel inelastically ON the boundary."""
        a, b = self.a, self.b
        ra, rb = self._ra, self._rb
        err, phi, qa, qb = self._predicted_errors(dt)
        inv = 1.0 / dt

        L = solve3(self._k_ang, (-phi.x * inv,
                                 -phi.y * inv,
                                 -phi.z * inv))
        _apply_ang(a, b, Vec3(L[0], L[1], L[2]))

        n, t1, t2 = self._n, self._t1, self._t2
        s_pred = err.dot(n)
        ep = err - n * s_pred
        c1 = ep.dot(t1) * inv
        c2 = ep.dot(t2) * inv
        det = self._k11 * self._k22 - self._k12 * self._k12
        if abs(det) > 1e-30:
            l1 = (-c1 * self._k22 + c2 * self._k12) / det
            l2 = (-c2 * self._k11 + c1 * self._k12) / det
            _apply_point(a, ra, b, rb, t1 * l1 + t2 * l2)

        if self.limits is not None and self._k_axis > 1e-30:
            lo, hi = self.limits
            k = self._k_axis
            sd = (_pt_vel(b, rb) - _pt_vel(a, ra)).dot(n)
            if s_pred < lo:
                target = sd + (lo - s_pred) * inv  # land ON the boundary
                new = max(self._lam_lo + (target - sd) / k, 0.0)
                dl = new - self._lam_lo
                self._lam_lo = new
                if dl:
                    _apply_point(a, ra, b, rb, n * dl)
            elif s_pred > hi:
                target = sd + (hi - s_pred) * inv
                new = min(self._lam_hi + (target - sd) / k, 0.0)
                dl = new - self._lam_hi
                self._lam_hi = new
                if dl:
                    _apply_point(a, ra, b, rb, n * dl)

    def project(self):
        a, b = self.a, self.b
        # orientation lock first (the axis rides on a's orientation)
        self._project_orientation(self._orientation_error())
        # then confine the anchor offset to the axis (remove ⊥ error,
        # clamp the along-axis travel into the limit window)
        ra, wa, rb, wb = self._geom()
        n = self._axis_world()
        err = wb - wa
        s = err.dot(n)
        corr = err - n * s                     # ⊥ part must vanish
        if self.limits is not None:
            lo, hi = self.limits
            if s < lo:
                corr = corr + n * (s - lo)     # push s up to lo
            elif s > hi:
                corr = corr + n * (s - hi)     # pull s down to hi
        if corr.length() < 1e-12:
            return
        K = _k3_point(a, ra, b, rb)
        P = solve3(K, (-corr.x, -corr.y, -corr.z))
        P = Vec3(P[0], P[1], P[2])
        if not a.is_static:
            a.position = a.position - P * a.inv_mass
            _rotate(a, a.inv_inertia_apply(ra.cross(P)) * -1.0)
        if b is not None and not b.is_static:
            b.position = b.position + P * b.inv_mass
            _rotate(b, b.inv_inertia_apply(rb.cross(P)))


__all__ = ["WeldJoint", "RevoluteJoint", "PrismaticJoint"]
