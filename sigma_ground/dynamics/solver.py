"""Sequential-impulse constraint solver — the ONE clock's enforcement pass.

Velocity-level Gauss–Seidel with accumulated, clamped impulses, then a direct
position projection (split-impulse style — velocities untouched, so no
Baumgarte energy pollution). This generalizes the stepper's existing collision
block: for a center-line sphere contact (r×n̂ = 0) one iteration reproduces
``collision.py``'s ``j = −(1+e)·v_rel/(1/m₁+1/m₂)`` EXACTLY — legacy behavior
is unchanged by construction, and the legacy "skip if separating" semantics
are preserved for legacy-kind contacts (bit-identity with the shipped suites).

Iteration order is FIXED (determinism): joints → contacts (normal) → friction.
Joint types live in ``joints.py`` and enforce themselves through the same
body state; this module owns contacts and the orchestration.

Per-impulse work is auditable: joints are workless rows (the ledger gate
"joints inject ≈ 0 energy" is measurable); contacts only ever dissipate.

NOT_PHYSICS constants (inherited from collision.py, flagged there): position
slop 1 mm, correction fraction 0.8 per pass.
"""
from __future__ import annotations

from .vec import Vec3

_SLOP = 0.001        # NOT_PHYSICS — resting-overlap allowance (collision.py's)
_BETA = 0.80         # NOT_PHYSICS — position correction fraction per pass


class Contact:
    """One contact row between parcel ``a`` and parcel-or-environment ``b``.

    ``normal`` points from a toward b (the legacy convention — for a ground
    contact b is None and the normal faces INTO the floor); ``point`` is the
    world contact location; ``kind`` selects semantics:
      "legacy"   — center-line sphere contact: skip when separating (exact
                   mirror of collision.py, bit-identity), no angular terms
                   asserted by geometry (r×n̂=0 holds by construction).
      "manifold" — offset contact point (a box corner on the floor): full SI
                   row with resting normal force (λ ≥ 0 holds bodies apart at
                   v≈0), torque-aware, restitution only on real impacts.
    ``depth_now`` (optional callable) re-evaluates penetration from the
    CURRENT pose for the position pass.
    """
    __slots__ = ("a", "b", "point", "normal", "depth", "kind", "e",
                 "vn0", "lam_n", "lam_t", "t_dir", "depth_now")

    def __init__(self, a, b, point, normal, depth, kind="manifold", e=0.5,
                 depth_now=None):
        self.a = a
        self.b = b                       # None → static environment (ground)
        self.point = point
        self.normal = normal
        self.depth = depth
        self.kind = kind
        self.e = e
        self.vn0 = 0.0
        self.lam_n = 0.0
        self.lam_t = 0.0
        self.t_dir = None
        self.depth_now = depth_now


def _vel_at(p, r):
    """Velocity of the material point at world offset r from the CoM."""
    w = p.angular_velocity
    return Vec3(p.velocity.x + w.y * r.z - w.z * r.y,
                p.velocity.y + w.z * r.x - w.x * r.z,
                p.velocity.z + w.x * r.y - w.y * r.x)


def _rel_normal_velocity(c):
    """(v_a − v_b)·n̂ at the contact point — POSITIVE when approaching
    (legacy semantics: normal a→b, bodies closing)."""
    ra = c.point - c.a.position
    va = _vel_at(c.a, ra)
    if c.b is not None:
        rb = c.point - c.b.position
        vb = _vel_at(c.b, rb)
    else:
        vb = Vec3(0, 0, 0)
    rel = va - vb
    return rel.dot(c.normal), rel


def _k_along(c, d):
    """Effective mass denominator for a unit impulse along direction d."""
    ra = c.point - c.a.position
    k = c.a.inv_mass
    ra_x_d = ra.cross(d)
    ia = c.a.inv_inertia_apply(ra_x_d)
    k += d.dot(ia.cross(ra))
    if c.b is not None:
        rb = c.point - c.b.position
        k += c.b.inv_mass
        rb_x_d = rb.cross(d)
        ib = c.b.inv_inertia_apply(rb_x_d)
        k += d.dot(ib.cross(rb))
    return k


def _apply(c, d, lam):
    """Apply impulse λ·d̂ pushing a and b APART (a gets −, b gets +),
    matching the legacy sign convention for j<0 on approach."""
    P = d * lam
    a, b = c.a, c.b
    if not a.is_static:
        a.velocity = a.velocity + P * a.inv_mass
        ra = c.point - a.position
        a.apply_angular_impulse(ra.cross(P))
    if b is not None and not b.is_static:
        b.velocity = b.velocity - P * b.inv_mass
        rb = c.point - b.position
        b.apply_angular_impulse(rb.cross(P) * -1.0)


def solve_velocity(contacts, joints, dt, *, iterations=10, friction_fn=None,
                   shake=False):
    """The velocity pass: capture v_n⁰, then Gauss–Seidel over
    joints → contact normals → friction, with accumulated clamped impulses.

    shake=True runs the joints' SHAKE rows instead (position error solved
    THROUGH the pre-drift velocity, so the drift lands on the constraint
    manifold — the first half of the RATTLE constrained-leapfrog scheme;
    plain velocity deletion was a measured secular energy sink)."""
    live = []
    for c in contacts:
        vn, _ = _rel_normal_velocity(c)
        c.vn0 = vn
        if c.kind == "legacy" and vn <= 0.0:
            continue                     # legacy mirror: separating → untouched
        live.append(c)

    # Legacy mirror: with only center-line rows (r×n̂=0), no joints and no
    # friction, ONE Gauss–Seidel pass IS collision.py's closed-form impulse —
    # extra passes would only re-apply float residuals (≈1e-16·v) and break
    # the bit-identity gate for no physical gain.
    if (not joints and friction_fn is None
            and all(c.kind == "legacy" for c in live)):
        iterations = 1

    # Joints cache pose-dependent geometry once per substep (poses are FIXED
    # during the velocity pass — only velocities change between iterations)
    # and reset their accumulated motor/limit impulses.
    for j in joints:
        begin = getattr(j, "begin", None)
        if begin is not None:
            begin(dt)

    for _ in range(iterations):
        for j in joints:
            if shake:
                j.solve_shake(dt)
            else:
                j.solve_velocity(dt)
        for c in live:
            vn, _ = _rel_normal_velocity(c)
            # restitution only against a REAL approach speed; a resting manifold
            # contact (vn0≈0) gets a pure non-penetration row (λ ≥ 0)
            target = c.e * max(c.vn0, 0.0)
            k = _k_along(c, c.normal)
            if k <= 1e-30:
                continue
            dlam = (vn + target) / k     # impulse magnitude to remove approach
            new = max(c.lam_n + dlam, 0.0)
            dl = new - c.lam_n
            c.lam_n = new
            if dl:
                _apply(c, c.normal, -dl)  # −n̂ on a = push a away from b
        if friction_fn is not None:
            for c in live:
                mu = friction_fn(c)
                if mu <= 0.0 or c.lam_n <= 0.0:
                    continue
                vn, rel = _rel_normal_velocity(c)
                tv = rel - c.normal * vn
                tl = tv.length()
                if tl < 1e-12 and c.t_dir is None:
                    continue
                if tl >= 1e-12:
                    c.t_dir = tv * (1.0 / tl)
                t = c.t_dir
                k = _k_along(c, t)
                if k <= 1e-30:
                    continue
                dlam = rel.dot(t) / k
                hi = mu * c.lam_n        # Coulomb cone (single μ, flagged in scene)
                new = min(max(c.lam_t + dlam, -hi), hi)
                dl = new - c.lam_t
                c.lam_t = new
                if dl:
                    _apply(c, t, -dl)


def project_positions(contacts, joints, *, iterations=3):
    """Direct position/orientation projection — velocities untouched.

    Contacts re-evaluate their CURRENT depth each pass and shift positions
    along the normal by the legacy slop/β rule (linear only — the legacy
    convention); joints project their own errors through their Jacobians.
    """
    for it in range(iterations):
        for j in joints:
            j.project()
        for c in contacts:
            if c.kind == "legacy" and it > 0:
                continue                 # legacy mirror: ONE 0.8·pen pass, not β^n
            a, b = c.a, c.b
            if b is not None:
                # sphere-pair depth from current centers (legacy formula)
                delta = b.position - a.position
                dist = delta.length()
                min_d = a.radius + b.radius
                pen = min_d - dist
                if pen <= 0.0 or dist < 1e-9:
                    continue
                n = delta * (1.0 / dist)
                inv_sum = a.inv_mass + b.inv_mass
                if inv_sum < 1e-30:
                    continue
                if c.kind == "legacy" and c.vn0 <= 0.0:
                    continue             # legacy mirror: separating → no fix
                corr = max(pen - _SLOP, 0.0) * _BETA / inv_sum
                if not a.is_static:
                    a.position = a.position - n * (corr * a.inv_mass)
                if not b.is_static:
                    b.position = b.position + n * (corr * b.inv_mass)
            else:
                # environment contact: re-evaluate the point's depth below the
                # plane through the CURRENT pose (offset points rotate with q)
                pen = c.depth_now() if c.depth_now is not None else c.depth
                if pen <= 0.0:
                    continue
                if c.kind == "legacy" and c.vn0 <= 0.0:
                    continue             # legacy mirror: separating → no fix
                corr = max(pen - _SLOP, 0.0) * _BETA
                if not c.a.is_static:
                    # normal faces a→ground (down); the fix lifts a OUT: −n̂
                    c.a.position = c.a.position - c.normal * corr


__all__ = ["Contact", "solve_velocity", "project_positions"]
