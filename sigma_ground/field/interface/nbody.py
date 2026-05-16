"""N-body gravitational integrator for SSBM physics.

Two integration schemes:
  * step()             — velocity-Verlet, 2nd-order symplectic, fast
  * forest_ruth_step() — Forest-Ruth (1990), 4th-order symplectic

4th-order symplectic is required for:
  - Long integrations of chaotic orbits (Halley's comet, NEAs)
  - Preserving symplecticity while resolving perihelion passages
  - Cases where the 2nd-order Verlet energy drift accumulates visibly

Physics included (all deriving from local_library constants):
  - Pairwise Newtonian gravity (point masses)
  - σ-field mass scaling via local_library.scale.scale_ratio
  - Optional 1PN Schwarzschild post-Newtonian correction (GR time dilation)
  - Zonal J₂ quadrupole acceleration (rotational oblateness / static tides)
  - Tidal deformation: r(θ) = R₀[1 + ε₂ P₂(cos θ)]
  - Love number k₂ tidal response
  - Gravitational wave energy loss (Peters 1964 formula)
  - Solar radiation pressure (opt-in via solar_luminosity_W + area_m2)

All constants from sigma_ground.field.constants.  No magic numbers.
  G, C sourced from measured CODATA values in constants.py.
  Forest-Ruth coefficients are exact mathematics (Forest & Ruth 1990).
  1PN coefficients (4, 4) are exact from GR Schwarzschild solution.
  P₂(cos θ) = ½(3cos²θ - 1): exact Legendre polynomial.

References
----------
  Forest & Ruth (1990): "Fourth-order symplectic integration",
      Physica D: Nonlinear Phenomena 43(1), 105-117.
  Soffel et al. (2003): "The IAU 2000 resolutions for astrometry ...",
      AJ 126(6), 2687-2706.  [1PN EOM, solar-system limit]
  Peters (1964): GW inspiral formula, Phys. Rev. 136, B1224.
  Wisdom & Holman (1991): Symplectic integration, AJ 102(4), 1528-1538.
  Darwin (1879) / Love (1911): Tidal deformation.
  Vallado (2013) §9.4: J₂ zonal-harmonic acceleration, exact vector form.
  Montenbruck & Gill (2000) §3.5: Earth oblateness in orbit propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace as _replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..constants import G as _G_SI, C as _C_SI, L_SUN_W as _L_SUN_W
from ..scale import scale_ratio as _scale_ratio
from ..bounds import check_sigma as _check_sigma, Safety as _Safety

# ── derived (not magic) constants ─────────────────────────────────────────
_G   = _G_SI                  # m³ kg⁻¹ s⁻²
_c   = _C_SI                  # m/s
_c2  = _c * _c                # m²/s²
_c4  = _c2 * _c2              # m⁴/s⁴
_4PI = 4.0 * math.pi          # exact

# Forest-Ruth integrator coefficients (exact — Forest & Ruth 1990 §2)
# θ = 1 / (2 − ∛2)  ≈ 1.351207191959657
_FR_THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_FR_C     = [_FR_THETA / 2.0,                # c₁ = c₄
             (1.0 - _FR_THETA) / 2.0,        # c₂ = c₃
             (1.0 - _FR_THETA) / 2.0,
             _FR_THETA / 2.0]
_FR_D     = [_FR_THETA,                      # d₁ = d₃
             1.0 - 2.0 * _FR_THETA,          # d₂  (NEGATIVE — unavoidable at 4th order)
             _FR_THETA]


# ═══════════════════════════════════════════════════════════════════════════
# § 0. PHYSICS TOGGLES — ablation filters for the force budget
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsToggles:
    """Boolean flags that gate each force-computation layer in NBodySystem.

    Toggles gate computation, not data. Per-body parameters (e.g. body.j2,
    body.area_m2) further determine whether a body contributes once its
    physics class is enabled.

    All flags default False — a freshly-constructed PhysicsToggles produces
    pure Newtonian gravity. The NBodySystem __init__ backward-compat shim
    auto-builds toggles from legacy kwargs (include_gr, solar_luminosity_W)
    so existing callers behave bit-identically.

    Each filter is independently flippable, which is the basis of the
    ablation methodology: if turning one on produces a fingerprint shift
    that overshoots truth, the same toggle can be flipped off to bisect
    the contribution and find the over-correction.

    Borrowed-from-textbook flags are marked BORROWED in inline comments.
    These are pending SSBM σ-field re-derivation per GOLDEN_RULES Rule 2.
    Grep for "BORROWED FROM TEXTBOOK" to find every line awaiting derivation.
    """

    gr_1pn:      bool = False   # single-body 1PN Schwarzschild (Soffel 2003 -- BORROWED)
    gr_2pn:      bool = False   # single-body 2PN c⁻⁴ correction (Will 1993 -- BORROWED)
    eih_cross:   bool = False   # full N-body 1PN EIH cross-terms (-- BORROWED, stretch)
    srp:         bool = False   # classical EM radiation pressure (σ-independent at SS scales)
    j2_zonal:    bool = False   # rotational oblateness quadrupole (Vallado §9.4 -- BORROWED)
    j3_zonal:    bool = False   # zonal J₃ (asymmetric, e.g. Earth's pear shape -- BORROWED)
    j4_zonal:    bool = False   # zonal J₄ (gas giants -- BORROWED)
    tidal_force: bool = False   # mutual tidal bulge as effective quadrupole (OURS, not borrowed)
    gw_damping:  bool = False   # GW energy loss (Peters 1964, binaries only -- BORROWED)
    # Future toggles to be added when their physics is implemented:
    #   yarkovsky:        bool = False    # NEA thermal recoil
    #   retarded_gravity: bool = False    # full speed-of-light propagation


# ═══════════════════════════════════════════════════════════════════════════
# § 1. TIDAL DEFORMATION FIELD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TidalDeformationField:
    """Analytic tidal deformation model: r(θ) = R₀[1 + ε₂ P₂(cos θ)].

    Parameters
    ----------
    body_radius_m : float
        Undeformed body radius R₀ (meters).
    epsilon2 : float
        Tidal deformation amplitude ε₂ = (k₂/2)(Mc/Mb)(R/d)³.
    tidal_direction : NDArray
        Unit vector pointing toward primary tidal source (shape (3,)).
    """

    body_radius_m: float
    epsilon2: float
    tidal_direction: NDArray[np.float64]

    def evaluate_at_angle(self, theta: float) -> float:
        """Relative tidal deformation amplitude δr/R₀ at polar angle θ.

        P₂(cos θ) = ½(3cos²θ − 1) — exact Legendre polynomial.
        """
        p2 = 0.5 * (3.0 * math.cos(theta) ** 2 - 1.0)
        return self.epsilon2 * p2

    def max_deformation(self) -> float:
        """Maximum deformation amplitude ε₂ (at θ = 0 or π)."""
        return self.epsilon2


# ═══════════════════════════════════════════════════════════════════════════
# § 2. CELESTIAL BODY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CelestialBody:
    """A single body in the n-body system.

    Immutable; use .replace() to evolve.

    σ-field coupling (SSBM): effective gravitational mass scales as
        M_eff = M_rest × e^σ
    implemented via local_library.scale.scale_ratio(sigma_field).
    This is the canonical SSBM mass-gravity coupling.

    Attributes
    ----------
    mass_kg         : rest mass (kg)
    position_m      : Cartesian position (m), shape (3,)
    velocity_m_s    : Cartesian velocity (m/s), shape (3,)
    radius_m        : mean radius (m)
    love_number_k2  : dimensionless tidal Love number (0.3-0.5 stars, ~0 BHs)
    sigma_field     : σ value (dimensionless); 0 = standard Newtonian
    area_m2         : cross-sectional area (m²) for solar radiation pressure;
                      0 = SRP disabled for this body (default, safe for planets)
    reflectivity    : radiation pressure coefficient CR in [0, 1];
                      0 = perfect absorber, 1 = perfect mirror (2× pressure).
                      Typical rocky body: ~0.1.  Solar sail: ~1.
    j2              : zonal J₂ quadrupole coefficient (dimensionless);
                      0 = perfect sphere (default).  Positive = oblate (Earth-like).
                      Typical solar-system values:
                        Sun:     2.0e-7   Earth:  1.083e-3   Jupiter:  1.470e-2
                        Saturn:  1.629e-2 Uranus: 3.510e-3   Neptune:  3.539e-3
                      Sectorial (m≠0) terms NOT modelled.
    j3              : zonal J₃ coefficient (asymmetric, e.g. Earth's pear shape).
                      Default 0.0.  Earth: -2.5e-6.  Most bodies: 0 by symmetry.
    j4              : zonal J₄ coefficient (next-order oblate correction).
                      Default 0.0.  Earth: -1.6e-6.  Jupiter: -5.9e-4.  Saturn: -9.2e-4.
    pole_axis_unit  : unit vector along rotational symmetry axis, shape (3,).
                      Default: (0, 0, 1) — z-axis aligned (good for ecliptic/ICRS
                      where most planets' poles are within ~30° of z-axis).
                      WRONG for Uranus (97° obliquity), Pluto (122°), and Triton
                      (retrograde) — callers should supply real IAU pole vectors.
                      Auto-normalized in __post_init__ if not unit length.
    """

    mass_kg:        float
    position_m:     NDArray[np.float64]
    velocity_m_s:   NDArray[np.float64]
    radius_m:       float
    love_number_k2: float
    sigma_field:    float = 0.0
    area_m2:        float = 0.0   # cross-sectional area for solar radiation pressure (m²)
    reflectivity:   float = 0.0   # radiation pressure coefficient CR: 0=absorber, 1=mirror
    j2:             float = 0.0   # zonal quadrupole coefficient (oblateness)
    j3:             float = 0.0   # zonal J₃ coefficient (north-south asymmetry)
    j4:             float = 0.0   # zonal J₄ coefficient (next-order oblateness)
    pole_axis_unit: NDArray[np.float64] | None = None  # default → +z axis in __post_init__

    def __post_init__(self) -> None:
        if self.position_m.shape != (3,):
            raise ValueError(f"position_m must be (3,), got {self.position_m.shape}")
        if self.velocity_m_s.shape != (3,):
            raise ValueError(f"velocity_m_s must be (3,), got {self.velocity_m_s.shape}")
        # σ-field validation: bounds.check_sigma classifies SAFE/EDGE/WALL/BEYOND.
        # For solar-system bodies σ ≈ 0, deeply SAFE. The check exists so that any
        # hand-edited body straying into the cavitation regime (σ → σ_conv) gets
        # flagged loudly rather than silently producing nonsense via math.exp(σ).
        sigma_status = _check_sigma(self.sigma_field)
        if sigma_status["status"] == _Safety.BEYOND:
            raise ValueError(
                f"sigma_field={self.sigma_field} is BEYOND the SSBM domain "
                f"(σ_conv ≈ {sigma_status.get('clamped')}). {sigma_status['note']}"
            )
        # Pole axis: default to +z, auto-normalize if user supplied a non-unit vector
        if self.pole_axis_unit is None:
            object.__setattr__(self, "pole_axis_unit", np.array([0.0, 0.0, 1.0]))
        else:
            if self.pole_axis_unit.shape != (3,):
                raise ValueError(f"pole_axis_unit must be (3,), got {self.pole_axis_unit.shape}")
            norm = float(np.linalg.norm(self.pole_axis_unit))
            if norm == 0.0:
                raise ValueError("pole_axis_unit must not be the zero vector")
            if abs(norm - 1.0) > 1e-10:
                object.__setattr__(self, "pole_axis_unit", self.pole_axis_unit / norm)

    @property
    def gm_m3_s2(self) -> float:
        """GM in m³/s² with σ-field scaling.

        G is sourced from sigma_ground.field.constants (CODATA measured).
        σ scaling: M_eff = M × e^σ via sigma_ground.field.scale.scale_ratio,
        which provides overflow/underflow guards at the σ field boundaries
        (clamps to inf/0 outside ±709 to avoid math domain errors).
        """
        return _G * self.mass_kg * _scale_ratio(self.sigma_field)

    def kinetic_energy(self) -> float:
        """½mv² in Joules."""
        return 0.5 * self.mass_kg * float(np.dot(self.velocity_m_s, self.velocity_m_s))

    def momentum(self) -> NDArray[np.float64]:
        """Linear momentum mv (kg·m/s)."""
        return self.mass_kg * self.velocity_m_s

    def angular_momentum_about_origin(self) -> NDArray[np.float64]:
        """L = r × p (kg·m²/s)."""
        return np.cross(self.position_m, self.momentum())

    def replace(self, **kw) -> "CelestialBody":
        return _replace(self, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. N-BODY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class NBodySystem:
    """N-body gravitational integrator with tidal deformation.

    Two integration methods:
      step()             Velocity-Verlet, 2nd-order symplectic. Fast.
                         Use for well-resolved inner-planet / moon systems.
      forest_ruth_step() Forest-Ruth, 4th-order symplectic. 3× cost.
                         Required for chaotic high-eccentricity orbits
                         (comets, NEAs, binary pulsars) where 2nd-order
                         energy drift accumulates over many periods.

    Forest-Ruth maintains EXACT symplecticity for fixed dt, so energy and
    angular momentum are conserved to machine precision over arbitrary time.
    (Variable dt breaks symplecticity in BOTH schemes — use fixed dt.)

    Parameters
    ----------
    bodies              : initial body configuration
    softening_m         : Plummer softening length (m), default 0
    toggles             : PhysicsToggles instance gating each force layer.
                          If None, built automatically from legacy kwargs
                          (include_gr, solar_luminosity_W) with j2_zonal
                          auto-enabled when any body has non-zero j2.
    solar_luminosity_W  : solar luminosity for radiation pressure (W).
                          A *parameter*, not a toggle — gates SRP only
                          indirectly (toggles.srp must also be True for
                          SRP to be computed).  Set to L_SUN_W (3.828e26 W,
                          IAU 2015) to enable.  Assumes bodies[0] is the
                          radiation source.  SRP only applies to bodies
                          with area_m2 > 0.
    include_gr          : LEGACY kwarg. Sets toggles.gr_1pn when no
                          explicit toggles= is given. Default False.

    Backward compatibility
    ----------------------
    Existing callers passing include_gr=True and/or solar_luminosity_W>0
    produce bit-identical accelerations to pre-toggles behavior, because
    the shim builds equivalent PhysicsToggles automatically:

        toggles = PhysicsToggles(
            gr_1pn = include_gr,
            srp    = (solar_luminosity_W > 0),
            j2_zonal = any body has j2 != 0,
        )
    """

    def __init__(
        self,
        bodies:             Sequence[CelestialBody],
        softening_m:        float = 0.0,
        toggles:            PhysicsToggles | None = None,
        solar_luminosity_W: float = 0.0,
        parent_attractor_indices: list[int | None] | None = None,
        # Legacy kwargs (deprecated, accepted for backward compatibility):
        include_gr:         bool  = False,
    ) -> None:
        self.bodies             = list(bodies)
        self.softening_m        = softening_m
        self.solar_luminosity_W = solar_luminosity_W
        self._time              = 0.0
        # parent_attractor_indices[i] = index of body i's dominant Newton
        # attractor (e.g. Earth -> Sun, Mimas -> Saturn), or None if i has
        # no fast-pair partner. Used by respa_step to split the Hamiltonian
        # by force type. None for all bodies disables RESPA-style fast/slow
        # separation; respa_step degenerates to a slow-only kick scheme.
        if parent_attractor_indices is None:
            parent_attractor_indices = [None] * len(self.bodies)
        if len(parent_attractor_indices) != len(self.bodies):
            raise ValueError(
                f"parent_attractor_indices length {len(parent_attractor_indices)} "
                f"must match bodies length {len(self.bodies)}"
            )
        self.parent_attractor_indices = list(parent_attractor_indices)
        if toggles is None:
            # Backward-compat: build PhysicsToggles from legacy kwargs +
            # auto-detect of per-body J2 data so existing callers behave
            # bit-identically to pre-toggles code.
            j2_data_present = any(b.j2 != 0.0 for b in self.bodies)
            toggles = PhysicsToggles(
                gr_1pn=include_gr,
                srp=(solar_luminosity_W > 0.0),
                j2_zonal=j2_data_present,
            )
        self.toggles = toggles
        # Legacy attribute access (some callers / tests inspect .include_gr directly).
        self.include_gr = self.toggles.gr_1pn

    @property
    def time(self) -> float:
        """Current simulation time (seconds)."""
        return self._time

    # ── accelerations ─────────────────────────────────────────────────────

    def compute_accelerations(self) -> NDArray[np.float64]:
        """Pairwise gravitational accelerations, shape (N, 3) in m/s².

        Each force layer is gated by an entry in self.toggles. Per-body
        parameters (j2, j3, j4, love_number_k2, area_m2) further determine
        whether a given body contributes once its physics class is enabled.

        Layers (in order applied):
          - Newtonian            (always on)
          - 1PN Schwarzschild GR (toggles.gr_1pn)              [BORROWED]
          - 2PN c⁻⁴ correction   (toggles.gr_2pn)              [BORROWED]
          - J₂ zonal quadrupole  (toggles.j2_zonal)            [BORROWED]
          - J₃ zonal             (toggles.j3_zonal)            [BORROWED]
          - J₄ zonal             (toggles.j4_zonal)            [BORROWED]
          - Tidal mutual force   (toggles.tidal_force)         (OURS — built from
                                                                compute_tidal_deformation)
          - Solar radiation pres (toggles.srp + solar_luminosity_W > 0)

        Sigma-mass scaling (M_eff = M·e^σ) is applied in body.gm_m3_s2 via
        scale.scale_ratio() — present at every layer.
        """
        n   = len(self.bodies)
        acc = np.zeros((n, 3), dtype=np.float64)
        t   = self.toggles

        # ── EIH pre-pass: Newtonian a_j and potential Φ_i needed by EIH ──────
        # eih_cross is mutually exclusive with gr_1pn: both model the 1PN
        # contribution; eih_cross is the full N-body form, gr_1pn the
        # single-body Schwarzschild approximation. If both are on, eih_cross
        # wins and gr_1pn is silently skipped to avoid double-counting.
        a_newton: NDArray[np.float64] | None = None
        phi:      NDArray[np.float64] | None = None
        if t.eih_cross:
            a_newton = np.zeros((n, 3), dtype=np.float64)
            phi      = np.zeros(n, dtype=np.float64)
            for ii in range(n):
                for kk in range(n):
                    if ii == kk:
                        continue
                    r_diff = (self.bodies[kk].position_m
                              - self.bodies[ii].position_m)
                    r_sq_p = float(np.dot(r_diff, r_diff)) + self.softening_m**2
                    r_p    = math.sqrt(r_sq_p)
                    if r_p == 0.0:
                        continue
                    gm_k   = self.bodies[kk].gm_m3_s2
                    a_newton[ii] += gm_k * r_diff / (r_sq_p * r_p)
                    phi[ii]      += gm_k / r_p

        # ── Newtonian + 1PN GR + 2PN GR (per-pair loop) ───────────────────
        for i in range(n):
            vi  = self.bodies[i].velocity_m_s
            vi2 = float(np.dot(vi, vi))

            for j in range(n):
                if i == j:
                    continue
                r_ij = self.bodies[j].position_m - self.bodies[i].position_m
                r_sq = float(np.dot(r_ij, r_ij)) + self.softening_m ** 2
                r    = math.sqrt(r_sq)
                if r == 0.0:
                    continue

                gm_j = self.bodies[j].gm_m3_s2
                acc[i] += gm_j * r_ij / (r_sq * r)   # Newtonian (always on)

                if t.eih_cross:
                    # ┌──────────────────────────────────────────────────────┐
                    # │ BORROWED FROM TEXTBOOK -- PENDING SSBM DERIVATION    │
                    # │ Full N-body 1PN Einstein-Infeld-Hoffmann equations   │
                    # │ in the GR specialization (β = γ = 1).                │
                    # │                                                      │
                    # │ Source: Will 1993 "Theory and Experiment in          │
                    # │   Gravitational Physics" Box 6.2 (eq. 6.4.4),        │
                    # │   equivalent to Soffel et al. 2003 §8 (the IAU 2000  │
                    # │   resolutions form). This is the canonical force     │
                    # │   model used by JPL DE440 (Park et al. 2021).        │
                    # │                                                      │
                    # │ Form with vec_ij = x_j - x_i:                        │
                    # │   a_i = Σ_j (μ_j vec_ij / r_ij³) × (1 + A_ij)        │
                    # │        + Σ_j (μ_j / (c² r_ij³)) ×                    │
                    # │           (vec_ij·(4v_i - 3v_j)) × (v_i - v_j)       │
                    # │        + (7/(2c²)) Σ_j (μ_j a_j^N / r_ij)            │
                    # │ where                                                │
                    # │   A_ij = -(4/c²) Φ_i  -(1/c²) Φ_j                    │
                    # │         + v_i²/c² + 2 v_j²/c² - (4/c²)(v_i·v_j)      │
                    # │         - (3/(2c²)) ((vec_ij·v_j)/r_ij)²             │
                    # │         + (1/(2c²)) (vec_ij·a_j^N)                   │
                    # │   Φ_i = Σ_{k≠i} μ_k / r_ik                           │
                    # │                                                      │
                    # │ Replaces single-body gr_1pn when both are on.        │
                    # │ Pending σ-field re-derivation of full N-body 1PN.    │
                    # └──────────────────────────────────────────────────────┘
                    vj  = self.bodies[j].velocity_m_s
                    vj2 = float(np.dot(vj, vj))

                    r_dot_vj = float(np.dot(r_ij, vj))
                    A_ij = (
                        - (4.0 / _c2) * phi[i]
                        - (1.0 / _c2) * phi[j]
                        + vi2 / _c2
                        + 2.0 * vj2 / _c2
                        - (4.0 / _c2) * float(np.dot(vi, vj))
                        - (3.0 / (2.0 * _c2)) * (r_dot_vj / r) ** 2
                        + (1.0 / (2.0 * _c2)) * float(np.dot(r_ij, a_newton[j]))
                    )
                    acc[i] += gm_j * r_ij * A_ij / (r_sq * r)

                    B_ij = 4.0 * vi - 3.0 * vj
                    cross = float(np.dot(r_ij, B_ij)) / _c2
                    acc[i] += (gm_j / (r_sq * r)) * cross * (vi - vj)

                    acc[i] += (7.0 / (2.0 * _c2)) * gm_j * a_newton[j] / r

                elif t.gr_1pn:
                    # ┌──────────────────────────────────────────────────────┐
                    # │ BORROWED FROM TEXTBOOK -- PENDING SSBM DERIVATION    │
                    # │ Source: Soffel et al. 2003, eq. 10.12                │
                    # │ (single-body 1PN Schwarzschild, solar-system limit)  │
                    # │ Coefficients 4 and 4 are EXACT from the Schwarzschild│
                    # │ metric. Replace this block when σ-field-derived 1PN  │
                    # │ is available in sigma_ground.field.gr_basics.        │
                    # └──────────────────────────────────────────────────────┘
                    r_hat  = r_ij / r
                    rdotv  = float(np.dot(r_hat, vi))
                    factor = gm_j / (r_sq * _c2)
                    acc[i] += factor * (
                        (4.0 * gm_j / r - vi2) * r_hat + 4.0 * rdotv * vi
                    )

                if t.gr_2pn:
                    # ┌──────────────────────────────────────────────────────┐
                    # │ BORROWED FROM TEXTBOOK -- LEADING TERM ONLY          │
                    # │ Source: Will 1993 "Theory and Experiment in          │
                    # │ Gravitational Physics" §4.4 (test-particle 2PN in    │
                    # │ Schwarzschild), Soffel 2003 eq. 10.12 at c⁻⁴.        │
                    # │                                                      │
                    # │ STATUS: LEADING RADIAL TERM with the correct c⁻⁴     │
                    # │ Schwarzschild coefficient 9:                         │
                    # │   a_2PN_leading = +9 (GM/r)² × (1/c⁴) × (1/r²) × r̂   │
                    # │                 = 9 GM × (GM/(rc²))² / r² × r̂        │
                    # │ Earlier version had a placeholder coefficient 1; the │
                    # │ correct leading-order radial coefficient from the    │
                    # │ Schwarzschild metric expansion is 9.                 │
                    # │                                                      │
                    # │ NOT YET INCLUDED (full 2PN has additional terms):    │
                    # │   - v_i² mixed terms ~ (GM/r) × v² × r̂/c⁴            │
                    # │   - (r̂·v_i)² terms                                   │
                    # │   - Cross-velocity-radial couplings                  │
                    # │ The leading radial dominates for slow orbits         │
                    # │ (v/c < 0.05); add velocity terms for compact binary  │
                    # │ work. Per SSBM derivation roadmap (GOLDEN_RULES §2)  │
                    # │ this whole block awaits σ-modified geodesic.         │
                    # └──────────────────────────────────────────────────────┘
                    r_hat     = r_ij / r
                    gm_over_r = gm_j / r          # m²/s²
                    pn_param  = gm_over_r / _c2   # dimensionless (GM/(rc²))
                    # Leading radial 2PN: +9 GM³/(r⁴ c⁴) × r̂
                    #   = 9 × (GM/r²) × (GM/(rc²))² × r̂
                    a_2pn_mag = 9.0 * (gm_j / r_sq) * pn_param * pn_param
                    acc[i] += a_2pn_mag * r_hat

        # ── J₂ zonal quadrupole (rotational oblateness) ───────────────────
        if t.j2_zonal:
            # ┌──────────────────────────────────────────────────────────────┐
            # │ BORROWED FROM TEXTBOOK -- PENDING SSBM DERIVATION            │
            # │ Source: Vallado 2013 "Fundamentals of Astrodynamics" §9.4    │
            # │ Vector form:                                                 │
            # │   a_J₂ = (3 G M_j R_j² J₂_j / (2 r⁵))                        │
            # │          × [(5(r·n̂)²/r² − 1) × r − 2(r·n̂) × n̂]              │
            # │ where r = pos_i − pos_j (j → i) and n̂ is j's pole axis.      │
            # │ SSBM should derive J₂ from the σ-field response of a         │
            # │ rotating mass distribution. Until that derivation lands,     │
            # │ we use measured J₂ values from NASA Planetary Fact Sheets.   │
            # └──────────────────────────────────────────────────────────────┘
            for j in range(n):
                bj = self.bodies[j]
                if bj.j2 == 0.0 or bj.radius_m == 0.0:
                    continue
                gm_j   = bj.gm_m3_s2
                R_j_sq = bj.radius_m * bj.radius_m
                j2_j   = bj.j2
                n_hat  = bj.pole_axis_unit

                for i in range(n):
                    if i == j:
                        continue
                    r_vec = self.bodies[i].position_m - bj.position_m
                    r_sq  = float(np.dot(r_vec, r_vec)) + self.softening_m ** 2
                    if r_sq == 0.0:
                        continue
                    r        = math.sqrt(r_sq)
                    r_dot_n  = float(np.dot(r_vec, n_hat))
                    cos2     = (r_dot_n * r_dot_n) / r_sq
                    coef     = 1.5 * gm_j * j2_j * R_j_sq / (r_sq * r_sq * r)
                    acc[i] += coef * ((5.0 * cos2 - 1.0) * r_vec
                                      - 2.0 * r_dot_n * n_hat)

        # ── J₃ zonal (north-south asymmetric quadrupole) ──────────────────
        if t.j3_zonal:
            # ┌──────────────────────────────────────────────────────────────┐
            # │ DERIVED FORMULA, MEASURED PER-BODY INPUT                     │
            # │ Source: gradient of the J₃ term in the Legendre expansion    │
            # │ of the gravitational potential:                              │
            # │   Φ_J₃ = +(GM J₃ R³/r⁴) × P₃(cos θ)                          │
            # │   P₃(c) = (5c³ - 3c)/2                                       │
            # │   a_J₃ = -∇Φ_J₃                                              │
            # │ Derivation by hand gives the vector form:                    │
            # │   a_J₃ = (GM J₃ R³)/(2 r⁵) × [(3 − 15s²) n̂ + 5s(7s² − 3) r̂] │
            # │ where s = (r·n̂)/r and r is the vector from the oblate body  │
            # │ to the test particle. Verified by limits:                    │
            # │   equator (s=0): radial term vanishes, only n̂ component     │
            # │   pole    (s=1): combine to outward force for J₃ < 0         │
            # │ The PER-BODY J₃ values come from spacecraft tracking         │
            # │ (e.g. Earth J₃ ≈ -2.5e-6) -- they are measured, not guessed. │
            # └──────────────────────────────────────────────────────────────┘
            for j in range(n):
                bj = self.bodies[j]
                if bj.j3 == 0.0 or bj.radius_m == 0.0:
                    continue
                gm_j  = bj.gm_m3_s2
                R3    = bj.radius_m ** 3
                j3_j  = bj.j3
                n_hat = bj.pole_axis_unit

                for i in range(n):
                    if i == j:
                        continue
                    r_vec = self.bodies[i].position_m - bj.position_m
                    r_sq  = float(np.dot(r_vec, r_vec)) + self.softening_m ** 2
                    if r_sq == 0.0:
                        continue
                    r       = math.sqrt(r_sq)
                    r_dot_n = float(np.dot(r_vec, n_hat))
                    s       = r_dot_n / r                   # = cos θ
                    # Compact vector form derived above:
                    #   a_J₃ = (GM J₃ R³)/(2 r⁵) × [(3 − 15s²) n̂ + 5s(7s² − 3) r̂]
                    # Using r̂ = r_vec/r:
                    coef     = 0.5 * gm_j * j3_j * R3 / (r_sq * r_sq * r)   # GM J₃ R³ / (2 r⁵)
                    n_factor = 3.0 - 15.0 * s * s
                    r_factor = 5.0 * s * (7.0 * s * s - 3.0)
                    acc[i] += coef * (n_factor * n_hat + r_factor * (r_vec / r))

        # ── J₄ zonal (next-order oblateness) ──────────────────────────────
        if t.j4_zonal:
            # ┌──────────────────────────────────────────────────────────────┐
            # │ DERIVED FORMULA, MEASURED PER-BODY INPUT                     │
            # │ Source: gradient of the J₄ term in the Legendre expansion:   │
            # │   Φ_J₄ = +(GM J₄ R⁴/r⁵) × P₄(cos θ)                          │
            # │   P₄(c) = (35c⁴ − 30c² + 3)/8                                │
            # │   a_J₄ = -∇Φ_J₄                                              │
            # │ Derivation by hand gives the vector form:                    │
            # │   a_J₄ = (5 GM J₄ R⁴)/(8 r⁶)                                 │
            # │          × [3(21s⁴ − 14s² + 1) r̂ + 4s(3 − 7s²) n̂]            │
            # │ where s = (r·n̂)/r and r is the vector from the oblate body  │
            # │ to the test particle. PER-BODY J₄ values are measured        │
            # │ (Earth -1.6e-6, Jupiter -5.87e-4, Saturn -9.15e-4).          │
            # └──────────────────────────────────────────────────────────────┘
            for j in range(n):
                bj = self.bodies[j]
                if bj.j4 == 0.0 or bj.radius_m == 0.0:
                    continue
                gm_j  = bj.gm_m3_s2
                R4    = bj.radius_m ** 4
                j4_j  = bj.j4
                n_hat = bj.pole_axis_unit

                for i in range(n):
                    if i == j:
                        continue
                    r_vec = self.bodies[i].position_m - bj.position_m
                    r_sq  = float(np.dot(r_vec, r_vec)) + self.softening_m ** 2
                    if r_sq == 0.0:
                        continue
                    r       = math.sqrt(r_sq)
                    r_dot_n = float(np.dot(r_vec, n_hat))
                    s       = r_dot_n / r
                    # Compact vector form derived above:
                    #   a_J₄ = (5 GM J₄ R⁴)/(8 r⁶)
                    #          × [3(21s⁴ − 14s² + 1) r̂ + 4s(3 − 7s²) n̂]
                    coef     = 0.625 * gm_j * j4_j * R4 / (r_sq * r_sq * r_sq)  # 5/8 GM J₄ R⁴/r⁶
                    r_factor = 3.0 * (21.0 * s**4 - 14.0 * s**2 + 1.0)
                    n_factor = 4.0 * s * (3.0 - 7.0 * s * s)
                    acc[i] += coef * (r_factor * (r_vec / r) + n_factor * n_hat)

        # ── Mutual tidal force (OURS — built from compute_tidal_deformation) ──
        if t.tidal_force:
            # For each body i with non-zero Love number, it is tidally
            # deformed by every other body j. The deformation produces an
            # *effective* J₂-like quadrupole on body i along the i-j axis
            # with amplitude ε₂ = (k₂/2)(M_j/M_i)(R_i/d_ij)³.
            #
            # The resulting acceleration on a third body k (and the back-
            # reaction on j) is computed using the same J₂ vector formula
            # we already use above, with J₂_eff = ε₂ and pole_axis = unit
            # vector from i to j.
            #
            # NOT BORROWED: this uses our existing TidalDeformationField
            # geometry + our J₂ formula. The math is local to this project.
            #
            # Caveat: the bulge is treated as static (rigid response, no
            # dissipation lag). Adding the lag is a separate (and σ-derivable)
            # extension.
            for i in range(n):
                bi = self.bodies[i]
                if bi.love_number_k2 == 0.0 or bi.radius_m == 0.0:
                    continue
                pos_i  = bi.position_m
                R_i_sq = bi.radius_m * bi.radius_m
                M_i    = bi.mass_kg
                if M_i == 0.0:
                    continue
                gm_i = bi.gm_m3_s2

                # Find the dominant perturber of body i (maximum tidal magnitude
                # ε₂ ∝ M_j / d_ij³). For the SS this is overwhelmingly the
                # Sun for planets and the parent planet for moons.
                best_j = -1
                best_eps2 = 0.0
                best_d2 = None
                for j in range(n):
                    if j == i:
                        continue
                    bj = self.bodies[j]
                    d_vec = bj.position_m - pos_i
                    d_sq  = float(np.dot(d_vec, d_vec))
                    if d_sq == 0.0:
                        continue
                    d3    = d_sq * math.sqrt(d_sq)
                    eps2  = (bi.love_number_k2 / 2.0) * (bj.mass_kg / M_i) * (R_i_sq * bi.radius_m) / d3
                    if eps2 > best_eps2:
                        best_eps2 = eps2
                        best_j    = j
                        best_d2   = d_sq
                if best_j < 0 or best_eps2 == 0.0:
                    continue

                # Effective J₂ along the i->(perturber) axis; apply the same
                # vector form used for permanent J₂ above to body i's effect
                # on every body k (including back-reaction on the perturber).
                bj    = self.bodies[best_j]
                d_vec = bj.position_m - pos_i
                d     = math.sqrt(best_d2)  # |d_vec|
                n_hat_tidal = d_vec / d
                j2_eff = best_eps2

                for k in range(n):
                    if k == i:
                        continue
                    r_vec = self.bodies[k].position_m - pos_i
                    r_sq  = float(np.dot(r_vec, r_vec)) + self.softening_m ** 2
                    if r_sq == 0.0:
                        continue
                    r        = math.sqrt(r_sq)
                    r_dot_n  = float(np.dot(r_vec, n_hat_tidal))
                    cos2     = (r_dot_n * r_dot_n) / r_sq
                    coef     = 1.5 * gm_i * j2_eff * R_i_sq / (r_sq * r_sq * r)
                    acc[k] += coef * ((5.0 * cos2 - 1.0) * r_vec
                                      - 2.0 * r_dot_n * n_hat_tidal)

        # ── Solar radiation pressure ──────────────────────────────────────
        # Gated by both toggles.srp AND solar_luminosity_W > 0 (need a value
        # for L_sun, not just permission). Classical EM; σ-independent at SS scales.
        if t.srp and self.solar_luminosity_W > 0.0 and n >= 2:
            sun_pos = self.bodies[0].position_m
            for i in range(1, n):
                body = self.bodies[i]
                if body.area_m2 <= 0.0 or body.mass_kg <= 0.0:
                    continue
                r_vec = body.position_m - sun_pos     # points away from Sun
                r_sq  = float(np.dot(r_vec, r_vec))
                if r_sq == 0.0:
                    continue
                r     = math.sqrt(r_sq)
                # F_SRP = L A (1+CR) / (4π r² c)
                f_srp = (self.solar_luminosity_W * body.area_m2
                         * (1.0 + body.reflectivity)
                         / (_4PI * r_sq * _c))
                acc[i] += (f_srp / body.mass_kg) * (r_vec / r)

        return acc

    # ── velocity-Verlet (2nd-order symplectic) ─────────────────────────────

    def step(self, dt: float, include_gw_loss: bool | None = None) -> None:
        """Advance dt using velocity-Verlet (leapfrog).

        Symplectic, O(dt²) global error per orbit.  Two force evaluations.
        For well-sampled orbits (≥ 88 steps/period) this is sufficient.

        For highly-eccentric orbits where timestep adaption is needed,
        use forest_ruth_step() with a fixed small dt instead.

        Parameters
        ----------
        dt              : time step in seconds
        include_gw_loss : LEGACY kwarg. If None (default), use
                          self.toggles.gw_damping. If True/False, override
                          the system toggle for this step only.
        """
        n   = len(self.bodies)
        acc = self.compute_accelerations()

        vel_half = [self.bodies[i].velocity_m_s + 0.5 * dt * acc[i]
                    for i in range(n)]
        pos_new  = [self.bodies[i].position_m + dt * vel_half[i]
                    for i in range(n)]

        for i in range(n):
            self.bodies[i] = self.bodies[i].replace(position_m=pos_new[i])

        acc_new = self.compute_accelerations()
        vel_new = [vel_half[i] + 0.5 * dt * acc_new[i] for i in range(n)]

        # GW damping resolution: legacy kwarg overrides toggle if given,
        # else fall through to the PhysicsToggles flag.
        apply_gw = (include_gw_loss if include_gw_loss is not None
                    else self.toggles.gw_damping)
        if apply_gw and n == 2:
            vel_new = self._apply_gw_damping(vel_new, dt)

        for i in range(n):
            self.bodies[i] = self.bodies[i].replace(velocity_m_s=vel_new[i])

        self._time += dt

    # ── Forest-Ruth (4th-order symplectic) ────────────────────────────────

    def forest_ruth_step(self, dt: float) -> None:
        """Advance dt using 4th-order Forest-Ruth symplectic integration.

        Forest & Ruth (1990): 7 sub-steps (4 drifts, 3 kicks), 3 force
        evaluations.  Exact symplecticity for fixed dt — energy and angular
        momentum conserved to machine precision over arbitrary time.

        Coefficients (exact mathematics, not approximations):
            θ  = 1 / (2 − ∛2) ≈ 1.3512
            c₁ = c₄ = θ/2
            c₂ = c₃ = (1−θ)/2
            d₁ = d₃ = θ
            d₂       = 1−2θ  ← NEGATIVE (unavoidable for 4th-order, 3-force)

        Use this for:
          • Halley's comet (e=0.967, perihelion ~0.586 AU)
          • 67P, Apophis, any chaotic NEA or comet
          • Long binary pulsar integrations
        with fixed dt ≈ 1 hr near perihelion (tune until energy drift < 1e-8).

        Warning: do NOT use variable dt with this (or any symplectic) scheme.
        Variable dt breaks the symplectic condition, causing slow energy drift.
        """
        c = _FR_C
        d = _FR_D

        def _drift(frac: float) -> None:
            self.bodies = [
                b.replace(position_m=b.position_m + frac * dt * b.velocity_m_s)
                for b in self.bodies
            ]

        def _kick(frac: float) -> None:
            acc = self.compute_accelerations()
            self.bodies = [
                b.replace(velocity_m_s=b.velocity_m_s + frac * dt * acc[i])
                for i, b in enumerate(self.bodies)
            ]

        _drift(c[0])  # drift θ/2
        _kick(d[0])   # kick θ
        _drift(c[1])  # drift (1−θ)/2
        _kick(d[1])   # kick 1−2θ  (negative — bodies briefly backtrack)
        _drift(c[2])  # drift (1−θ)/2
        _kick(d[2])   # kick θ
        _drift(c[3])  # drift θ/2

        self._time += dt

    # ── RESPA-style force decomposition (per-body dt by force type) ─────
    def compute_fast_accelerations(self) -> NDArray[np.float64]:
        """Newtonian accelerations for fast parent-child pairs only.

        For each body i with a parent attractor p (set via the
        parent_attractor_indices NBodySystem constructor argument), this
        returns the Newtonian acceleration on i from p, AND the reciprocal
        back-reaction on p from i (Newton's third law).

        Bodies without a parent contribute zero to the fast acceleration.
        Bodies that ARE parents (e.g. Saturn, parent of multiple moons)
        accumulate back-reactions from each of their children.

        Used by respa_step (Tuckerman 1992 / Wisdom-Holman style symplectic
        multi-timestep) to separate fast-varying forces (close binary
        Newton) from slow-varying ones (planet-Sun perturbations, J₂/J₃/J₄,
        EIH cross-terms, tides, SRP).

        Returns
        -------
        ndarray of shape (N, 3) in m/s².
        """
        n = len(self.bodies)
        acc = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            p_idx = self.parent_attractor_indices[i]
            if p_idx is None or p_idx == i:
                continue
            b_i = self.bodies[i]
            b_p = self.bodies[p_idx]
            r_vec = b_p.position_m - b_i.position_m
            r_sq = float(np.dot(r_vec, r_vec)) + self.softening_m ** 2
            if r_sq == 0.0:
                continue
            r = math.sqrt(r_sq)
            # Acceleration on i toward parent: F = G M_p / r²  -> a = GM_p r̂ / r²
            gm_p = b_p.gm_m3_s2
            acc[i] += gm_p * r_vec / (r_sq * r)
            # Reciprocal acceleration on parent from i: opposite direction.
            gm_i = b_i.gm_m3_s2
            acc[p_idx] -= gm_i * r_vec / (r_sq * r)
        return acc

    def compute_slow_accelerations(self) -> NDArray[np.float64]:
        """Full acceleration minus fast-pair Newton.

        This is the "rest" of the dynamics: planet-Sun, planet-planet,
        moon-Sun, J₂/J₃/J₄ zonals, EIH cross-terms, mutual tides, SRP.
        Sampled at the macro dt by respa_step.

        Returns
        -------
        ndarray of shape (N, 3) in m/s².
        """
        return self.compute_accelerations() - self.compute_fast_accelerations()

    # ── RESPA step: symplectic multi-timestep (slow outer, fast inner) ──
    def respa_step(self, dt: float, n_substeps: int) -> None:
        """Symplectic multi-timestep: slow forces at dt, fast forces at dt/N.

        ⚠ EXPERIMENTAL — Works for systems where the parent-pair Newton
        force genuinely dominates over the rest of the dynamics (2-body
        Earth-Moon: validated, 12x better than uniform-coarse). FAILS for
        cases where the "slow" force is comparable to the "fast" force
        (3-body Sun-Earth-Moon: Moon's Sun-pull is ~5.9e-3 m/s², Earth-pull
        is ~2.7e-3 m/s² -- they're the same order, so the Strang splitting
        introduces large errors). The fix is Wisdom-Holman-style Kepler
        splitting (analytic two-body inside the inner block), which is a
        substantial implementation. See misc/respa_attempt_2026-05-16.md.

        Implements the Tuckerman, Berne, Martyna 1992 RESPA scheme:

            exp(dt L) = exp(dt/2 L_slow) [exp(dt/N L_fast)]^N exp(dt/2 L_slow)

        with the inner block run as N steps of leapfrog (velocity-Verlet)
        over the fast Hamiltonian (T + U_fast). This decomposition is by
        FORCE TYPE: U_fast = Newtonian potential of declared fast pairs;
        U_slow = everything else. Set the fast pairs via the
        parent_attractor_indices argument to NBodySystem.

        Order: 2nd-order globally (outer Strang on slow forces), but the
        inner leapfrog at dt/N gives the fast forces 2nd-order resolution
        at the fast orbital timescale. For our solar-system application,
        this is sufficient -- the slow forces (Sun on moon, zonals, GR)
        are slowly varying over fast-moon orbital periods.

        Symplectic for fixed dt and fixed n_substeps: long-term energy
        bounded, no secular drift. The earlier forest_ruth_step_hierarchical
        attempt failed precisely because it was NOT symplectic; this method
        is the correct fix.

        Parameters
        ----------
        dt          : macro time step in seconds (slow forces sampled here)
        n_substeps  : inner leapfrog substeps for fast forces
                       (effective fast force resolution = dt / n_substeps)
        """
        if n_substeps < 1:
            raise ValueError(f"n_substeps={n_substeps} must be >= 1")
        n = len(self.bodies)

        # 1. Half-kick using slow forces
        acc_slow = self.compute_slow_accelerations()
        for i in range(n):
            b = self.bodies[i]
            self.bodies[i] = b.replace(
                velocity_m_s=b.velocity_m_s + 0.5 * dt * acc_slow[i]
            )

        # 2. N inner leapfrog steps using fast forces only.
        # Standard velocity-Verlet on the fast Hamiltonian:
        #   v(t+dt/2)   = v(t)    + 0.5 dt a_fast(x(t))
        #   x(t+dt)     = x(t)    + dt v(t+dt/2)
        #   v(t+dt)     = v(t+dt/2) + 0.5 dt a_fast(x(t+dt))
        # We collapse adjacent half-kicks across substeps where possible
        # by tracking that the post-substep state has v = v_half + ...
        # For simplicity, run each substep as a full velocity-Verlet step.
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            acc_fast = self.compute_fast_accelerations()
            # Half-kick
            for i in range(n):
                b = self.bodies[i]
                self.bodies[i] = b.replace(
                    velocity_m_s=b.velocity_m_s + 0.5 * sub_dt * acc_fast[i]
                )
            # Drift
            for i in range(n):
                b = self.bodies[i]
                self.bodies[i] = b.replace(
                    position_m=b.position_m + sub_dt * b.velocity_m_s
                )
            # Half-kick at new positions
            acc_fast = self.compute_fast_accelerations()
            for i in range(n):
                b = self.bodies[i]
                self.bodies[i] = b.replace(
                    velocity_m_s=b.velocity_m_s + 0.5 * sub_dt * acc_fast[i]
                )

        # 3. Half-kick using slow forces (recomputed at new positions)
        acc_slow = self.compute_slow_accelerations()
        for i in range(n):
            b = self.bodies[i]
            self.bodies[i] = b.replace(
                velocity_m_s=b.velocity_m_s + 0.5 * dt * acc_slow[i]
            )

        self._time += dt

    # ── selective Forest-Ruth: update only a subset of bodies ────────────
    def _selective_fr_step(self, dt: float, body_indices: list[int]) -> None:
        """Forest-Ruth on a subset of bodies. Other bodies remain fixed.

        Used by forest_ruth_step_hierarchical to integrate fast bodies
        through micro-steps while slow bodies are held at their interpolated
        positions. Forces at each kick are computed using the FULL current
        state, but only the listed bodies have their position/velocity
        updated. Does NOT advance self._time -- the caller manages it.
        """
        c = _FR_C
        d = _FR_D

        def _drift_partial(frac: float) -> None:
            for i in body_indices:
                b = self.bodies[i]
                self.bodies[i] = b.replace(
                    position_m=b.position_m + frac * dt * b.velocity_m_s
                )

        def _kick_partial(frac: float) -> None:
            acc = self.compute_accelerations()
            for i in body_indices:
                b = self.bodies[i]
                self.bodies[i] = b.replace(
                    velocity_m_s=b.velocity_m_s + frac * dt * acc[i]
                )

        _drift_partial(c[0])
        _kick_partial(d[0])
        _drift_partial(c[1])
        _kick_partial(d[1])
        _drift_partial(c[2])
        _kick_partial(d[2])
        _drift_partial(c[3])

    # ── Hierarchical Forest-Ruth: per-body dt ────────────────────────────
    def forest_ruth_step_hierarchical(
        self,
        dt: float,
        fast_indices: list[int],
        n_substeps: int,
    ) -> None:
        """Macro-step at dt, substepping fast bodies at dt/n_substeps.

        ⚠ KNOWN BROKEN 2026-05-16 ⚠
        This body-split operator scheme has a correctness bug: the slow-
        body advancement under the frozen-fast-body assumption produces
        a wrong slow trajectory (because the fast body's gravity is
        modelled as constant during the macro step, not time-averaged
        over its orbit). The fast substeps then run on an incorrect
        slow-body trajectory, compounding the error.

        Demonstrated by tests in test_nbody.py:
          - 2-body Earth-Moon hier(1d/0.1d) is 8.5x WORSE than uniform 1d
          - 3-body Sun-Earth-Moon similar pattern
        See misc/hierarchical_dt_known_bug_2026-05-16.md for details.

        The correct fix is symplectic multi-timestep (Tuckerman 1992
        RESPA) split by FORCE TYPE not body: half-kick from slow forces,
        N small kicks from fast forces, half-kick from slow forces.
        Deferred to a future session.

        Motivation: with a single global dt, fast moons (Mimas at 0.94d,
        Phobos at 0.32d) under-resolve at dt=0.1d but Forest-Ruth's
        shadow-Hamiltonian phase drift makes a global dt=0.02d REGRESS
        them (see misc/dt_tradeoff_verdict_2026-05-15.md). The right fix
        is per-body dt: slow bodies use dt=0.1d, fast bodies use a fraction.

        Algorithm:
          1. Snapshot all body states.
          2. Advance SLOW bodies one full Forest-Ruth step at dt, with
             fast bodies frozen at their start positions. (Fast bodies
             perturb slow bodies very weakly: Mimas/Saturn mass ratio is
             ~6e-8; over 0.1d the displacement bias is negligible.)
          3. Reset fast bodies to their start state. Save slow-body end state.
          4. Run n_substeps Forest-Ruth steps for fast bodies at dt/n_substeps,
             with slow bodies linearly interpolated between their start and
             end positions/velocities at each substep.

        NOT symplectic: the operator-split between slow and fast loses
        Forest-Ruth's exact symplecticity. For multi-year integrations the
        secular energy drift is bounded by the slow-fast perturbation
        magnitudes -- in the solar system that's ~1e-8 per outer step, so
        accumulated drift over 5y stays sub-percent for slow bodies.

        Parameters
        ----------
        dt           : macro step in seconds (slow bodies advance by this)
        fast_indices : indices into self.bodies to substep
        n_substeps   : substeps per macro step for fast bodies
                       (effective fast-body dt = dt / n_substeps)
        """
        if n_substeps < 1:
            raise ValueError(f"n_substeps={n_substeps} must be >= 1")
        if not fast_indices:
            self.forest_ruth_step(dt)
            return
        if n_substeps == 1:
            self.forest_ruth_step(dt)
            return

        n = len(self.bodies)
        fast_set = set(fast_indices)
        slow_indices = [i for i in range(n) if i not in fast_set]

        # 1. Snapshot
        initial = [(b.position_m.copy(), b.velocity_m_s.copy())
                   for b in self.bodies]

        # 2. Advance slow bodies one full step (fast bodies frozen)
        self._selective_fr_step(dt, slow_indices)
        slow_end = [(self.bodies[i].position_m.copy(),
                     self.bodies[i].velocity_m_s.copy())
                    for i in range(n)]

        # 3. Reset fast bodies to their start state (slow bodies stay at end)
        for i in fast_indices:
            self.bodies[i] = self.bodies[i].replace(
                position_m=initial[i][0].copy(),
                velocity_m_s=initial[i][1].copy(),
            )

        # 4. Substep fast bodies. At each substep, set slow bodies to their
        #    linearly-interpolated start-of-substep state, then run a fast-only
        #    Forest-Ruth at sub_dt.
        sub_dt = dt / n_substeps
        for sub in range(n_substeps):
            alpha = sub / n_substeps  # start-of-substep interpolation parameter
            for i in slow_indices:
                interp_pos = (1.0 - alpha) * initial[i][0] + alpha * slow_end[i][0]
                interp_vel = (1.0 - alpha) * initial[i][1] + alpha * slow_end[i][1]
                self.bodies[i] = self.bodies[i].replace(
                    position_m=interp_pos, velocity_m_s=interp_vel,
                )
            self._selective_fr_step(sub_dt, fast_indices)

        # 5. Finalize slow bodies at their full-step end state
        for i in slow_indices:
            self.bodies[i] = self.bodies[i].replace(
                position_m=slow_end[i][0], velocity_m_s=slow_end[i][1],
            )

        self._time += dt

    # ── conserved quantities ────────────────────────────────────────────────

    def total_energy(self) -> float:
        """Total mechanical energy (kinetic + gravitational potential) in J."""
        ke = sum(b.kinetic_energy() for b in self.bodies)
        pe = 0.0
        for i, bi in enumerate(self.bodies):
            for j, bj in enumerate(self.bodies):
                if j <= i:
                    continue
                r = float(np.linalg.norm(bi.position_m - bj.position_m))
                if r > 0:
                    pe -= _G * bi.mass_kg * bj.mass_kg / r
        return ke + pe

    def total_momentum(self) -> NDArray[np.float64]:
        """Total linear momentum (kg·m/s)."""
        return sum(b.momentum() for b in self.bodies)  # type: ignore[return-value]

    def total_angular_momentum(self) -> NDArray[np.float64]:
        """Total angular momentum about origin (kg·m²/s)."""
        return sum(b.angular_momentum_about_origin() for b in self.bodies)  # type: ignore[return-value]

    # ── tidal deformation ────────────────────────────────────────────────

    def compute_tidal_deformation(
        self, body_idx: int, perturber_idx: int,
    ) -> TidalDeformationField:
        """Tidal deformation of body_idx due to perturber_idx.

        ε₂ = (k₂/2)(M_c/M_b)(R_b/d)³  —  derived from Love (1911).
        """
        body      = self.bodies[body_idx]
        perturber = self.bodies[perturber_idx]
        r_ij      = perturber.position_m - body.position_m
        d         = float(np.linalg.norm(r_ij))
        if d == 0.0:
            raise ValueError("Bodies are at the same position")
        tidal_dir = r_ij / d
        ratio     = perturber.mass_kg / body.mass_kg if body.mass_kg > 0 else 0.0
        eps2      = (body.love_number_k2 / 2.0) * ratio * (body.radius_m / d) ** 3
        return TidalDeformationField(
            body_radius_m=body.radius_m,
            epsilon2=eps2,
            tidal_direction=tidal_dir,
        )

    def roche_limit(self, primary_idx: int, satellite_idx: int) -> float:
        """Roche limit d = R_p × (2 M_p/M_s)^(1/3) (rigid satellite).

        Exact formula for a rigid satellite (Roche 1847).
        """
        primary   = self.bodies[primary_idx]
        satellite = self.bodies[satellite_idx]
        if satellite.mass_kg == 0:
            return float("inf")
        return primary.radius_m * (2.0 * primary.mass_kg / satellite.mass_kg) ** (1.0 / 3.0)

    # ── GW energy loss ───────────────────────────────────────────────────

    def _apply_gw_damping(
        self,
        vel_new:  list[NDArray[np.float64]],
        dt:       float,
    ) -> list[NDArray[np.float64]]:
        """Reduce binary velocities by GW energy loss (Peters 1964).

        Only meaningful for compact object binaries; ignored for planets.
        """
        b0, b1 = self.bodies[0], self.bodies[1]
        r_vec  = b1.position_m - b0.position_m
        r      = float(np.linalg.norm(r_vec))
        if r == 0.0:
            return vel_new
        mu_red = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg)
        M_tot  = b0.mass_kg + b1.mass_kg
        # Peters formula: dE/dt = -(32/5) G⁴ m1² m2² M / (c⁵ r⁴)
        # Coefficients 32/5 are exact from quadrupole formula (Peters 1964).
        dE_dt  = -(32.0 / 5.0) * _G**4 * b0.mass_kg**2 * b1.mass_kg**2 * M_tot / (
            _c**5 * r**4
        )
        v_rel  = vel_new[1] - vel_new[0]
        v_mag  = float(np.linalg.norm(v_rel))
        if v_mag == 0.0:
            return vel_new
        ke_rel = 0.5 * mu_red * v_mag**2
        if ke_rel == 0.0:
            return vel_new
        damp   = 1.0 + dE_dt * dt / ke_rel
        damp   = max(0.0, damp)
        v_cm   = (b0.mass_kg * vel_new[0] + b1.mass_kg * vel_new[1]) / M_tot
        # CM-frame decomposition: v_i = v_cm ± (m_other/M) * v_rel
        # b0 gets -m_b1/M * v_rel;  b1 gets +m_b0/M * v_rel
        vel_new[0] = v_cm - math.sqrt(damp) * b1.mass_kg / M_tot * v_rel
        vel_new[1] = v_cm + math.sqrt(damp) * b0.mass_kg / M_tot * v_rel
        return vel_new
