"""
PhysicsScene — a collection of parcels with shared gravity and boundaries.

The scene owns:
  - A list of PhysicsParcels (dynamic and static objects)
  - Gravity vector (default: -9.80665 m/s² in Y)
  - Ground plane (optional, default: Y=0 with restitution)
  - Elapsed simulation time

The scene does NOT own the renderer. It is rendered by passing parcel
positions to the entangler externally.

Standard gravity
  g = 9.80665 m/s² (exact, BIPM standard since 1901)
  F_grav = m × g (downward)
  FIRST_PRINCIPLES: Newtonian gravity. Relativistic corrections negligible
  at these scales (they enter at 10⁻⁹ level for surface gravity).
  σ-dependence: m(σ) scales → F_grav(σ) scales. Heavier matter falls faster
  in absolute terms but acceleration g is invariant (equivalence principle).

Ground plane
  By default at Y=0, normal (0,1,0), with restitution = 0.50.
  The ground is effectively infinite mass (inv_mass=0).
  To disable ground: pass ground=False.
"""

from .vec import Vec3


_G_STANDARD = 9.80665   # m/s², BIPM standard


class GroundPlane:
    """An infinite rigid plane — the floor."""

    def __init__(self, y=0.0, normal=None, restitution=0.50):
        self.point       = Vec3(0, y, 0)
        self.normal      = normal if normal is not None else Vec3(0, 1, 0)
        self.restitution = restitution

    def __repr__(self):
        return f"GroundPlane(y={self.point.y:.3f}, e={self.restitution:.2f})"


class PhysicsScene:
    """A scene in which matter evolves under physics.

    Args:
        parcels:     list of PhysicsParcel.
        gravity:     Vec3 gravitational acceleration (m/s²). Default: -g ĵ.
        ground:      GroundPlane or False to disable. Default: Y=0 plane.
        time:        starting simulation time (s). Default: 0.
    """

    def __init__(self, parcels, gravity=None, ground=None, time=0.0,
                 constraints=None, restitution_fn=None, friction_fn=None,
                 solver_iterations=10, projection_iterations=3):
        self.parcels = list(parcels)
        self.gravity = (gravity if gravity is not None
                        else Vec3(0, -_G_STANDARD, 0))
        self.time    = float(time)

        if ground is None:
            self.ground = GroundPlane()
        elif ground is False:
            self.ground = None
        else:
            self.ground = ground

        # ── articulation (the ONE clock's constraint set) ──
        # constraints: dynamics.joints objects enforced every substep.
        # restitution_fn(parcel_a, parcel_b|None) -> e and
        # friction_fn(parcel_a, parcel_b|None) -> mu are CALLBACKS the
        # orchestrator (materia) builds over material names — the
        # make_atmospheric_drag tier-crossing idiom, so dynamics stays
        # material-name-free. Defaults preserve legacy behavior exactly:
        # e = min(restitutions), mu = 0 (frictionless — NOT_PHYSICS, flagged:
        # a friction_fn should be injected wherever sliding matters).
        self.constraints = list(constraints) if constraints else []
        self.restitution_fn = restitution_fn
        self.friction_fn = friction_fn
        # Gauss–Seidel budget (NOT_PHYSICS — numerical convergence knobs;
        # raise for stiff joint chains, e.g. a whipping double pendulum)
        self.solver_iterations = int(solver_iterations)
        self.projection_iterations = int(projection_iterations)

    def total_kinetic_energy(self):
        """Sum of ½mv² + ½ωᵀIω for all dynamic parcels (J). (The rotational
        term is 0 for every pre-articulation scene — ω was never integrated.)"""
        return sum(p.kinetic_energy() + p.rotational_ke() for p in self.parcels)

    def total_angular_momentum(self, about=None):
        """World-frame angular momentum about ``about`` (default: the origin):
        Σ [ I·ω + r×(m·v) ] over dynamic parcels."""
        o = about if about is not None else Vec3(0, 0, 0)
        L = Vec3(0, 0, 0)
        for p in self.parcels:
            if p.is_static:
                continue
            L = L + p.angular_momentum()
            r = p.position - o
            L = L + r.cross(p.momentum())
        return L

    def total_momentum(self):
        """Vector sum of mv for all dynamic parcels (kg·m/s)."""
        px = py = pz = 0.0
        for p in self.parcels:
            mom = p.momentum()
            px += mom.x; py += mom.y; pz += mom.z
        return Vec3(px, py, pz)

    def add(self, parcel):
        """Add a parcel to the scene."""
        self.parcels.append(parcel)

    def dynamic_parcels(self):
        """Return only non-static parcels."""
        return [p for p in self.parcels if not p.is_static]

    def __repr__(self):
        n_dyn = sum(1 for p in self.parcels if not p.is_static)
        return (f"PhysicsScene({len(self.parcels)} parcels, "
                f"{n_dyn} dynamic, t={self.time:.4f}s)")
