"""
Wave-particle duality ellipse (Khatiwada-Qian 2025).

Generalises the Englert-Greenberger-Yasin (1996) complementarity bound
    D² + V² ≤ 1
into a tighter equality on a 2-parameter surface:
    V²/γ² + D² = 1     (Khatiwada-Qian 2025, arXiv:2505.21443v1)

where:
    V = fringe visibility         (interference contrast)
    D = distinguishability        (which-path information extracted)
    γ = marginal coherence        (overlap |⟨M₁|M₂⟩| between the two
                                   environmental "notepad" states that
                                   record which-path info)

Concurrence (Wootters 1998, adapted for this state family):
    C = 2·|c₁·c₂|·√(1 − γ²)

where c₁, c₂ are the path-amplitude coefficients, |c₁|² + |c₂|² = 1.

──────────────────────────────────────────────────────────────────────────────
RELATION TO SIGMA-GROUND
──────────────────────────────────────────────────────────────────────────────

The paper's concurrence C = 2|c₁c₂|·√(1−γ²) is identically the product
of the Schmidt coefficients 2·s₁·s₂ of the bipartite (path ⊗ environment)
state.  This gives a direct bridge between the paper's γ and
sigma-ground's existing schmidt_coefficients() output without any new
machinery — see gamma_from_schmidt() below.

At γ = 1 (default), the ellipse collapses to the Englert circle and
every function reduces to its pre-existing sigma-ground form.  γ is an
open input knob: the σ→γ mapping is intentionally NOT wired here.
Leave γ free until empirical correlation is measured.
"""

import math


def duality_ellipse_visibility(D, gamma=1.0):
    """Maximum fringe visibility given distinguishability D and coherence γ.

    V_max = γ · √(1 − D²)

    FIRST_PRINCIPLES: Khatiwada-Qian 2025 Eq. 8 (rearranged from
    V²/γ² + D² = 1).  At γ=1 this is the Englert saturation
    V = √(1 − D²); at γ=0 (orthogonal environment notepads) V = 0
    regardless of D — the environment has erased all path coherence.

    Args:
        D:     distinguishability in [0, 1]
        gamma: marginal coherence γ = |⟨M₁|M₂⟩| in [0, 1], default 1.0

    Returns:
        maximum visibility V in [0, γ]
    """
    if D < 0.0 or D > 1.0:
        raise ValueError("D must be in [0, 1]")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return gamma * math.sqrt(max(0.0, 1.0 - D * D))


def concurrence_from_coherence(c1, c2, gamma):
    """Wootters concurrence of the (path ⊗ environment) state.

    C = 2 · |c₁ · c₂| · √(1 − γ²)

    FIRST_PRINCIPLES: Khatiwada-Qian 2025 Eq. 11.  This is the
    path-system-to-environment entanglement measure; it vanishes when
    γ=1 (environment notepads identical, state separable) and is
    maximised when γ=0 (orthogonal notepads, fully entangled).

    Args:
        c1:    path-1 amplitude (real or complex), |c₁|² + |c₂|² = 1
        c2:    path-2 amplitude
        gamma: marginal coherence γ in [0, 1]

    Returns:
        concurrence C in [0, 1]
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return 2.0 * abs(complex(c1) * complex(c2)) * math.sqrt(max(0.0, 1.0 - gamma * gamma))


def duality_ellipse_holds(V, D, gamma, tol=1e-10):
    """Check whether (V, D, γ) lies on or inside the duality ellipse.

        V²/γ² + D² ≤ 1 + tol

    At γ = 1 this reduces exactly to the Englert circle test.  The
    ellipse equality V²/γ² + D² = 1 is saturated by pure states of the
    form |Ψ⟩ = c₁|path₁⟩|M₁⟩ + c₂|path₂⟩|M₂⟩ with |⟨M₁|M₂⟩| = γ.

    Args:
        V:     visibility in [0, 1]
        D:     distinguishability in [0, 1]
        gamma: marginal coherence in (0, 1]
        tol:   floating-point tolerance (default 1e-10)

    Returns:
        True if the ellipse bound is satisfied.
    """
    if gamma <= 0.0:
        # Degenerate limit: V must be exactly 0 (modulo tol).
        return V <= tol and D * D <= 1.0 + tol
    return (V * V) / (gamma * gamma) + D * D <= 1.0 + tol


def gamma_from_concurrence(c1, c2, C):
    """Invert the concurrence relation to recover γ.

        γ = √(1 − (C / (2·|c₁·c₂|))²)

    FIRST_PRINCIPLES: algebraic inverse of concurrence_from_coherence.
    Useful when C has been computed independently (e.g., from a reduced
    density matrix) and γ must be extracted for ellipse consistency
    checks.

    Args:
        c1: path-1 amplitude
        c2: path-2 amplitude
        C:  concurrence in [0, 2|c₁c₂|]

    Returns:
        γ in [0, 1]

    Raises:
        ValueError if |c₁·c₂| = 0 (γ is undefined; any value satisfies C=0).
    """
    prod = abs(complex(c1) * complex(c2))
    if prod < 1e-15:
        raise ValueError("|c1·c2| = 0: gamma is undefined (any γ gives C=0)")
    ratio = C / (2.0 * prod)
    return math.sqrt(max(0.0, 1.0 - ratio * ratio))


def gamma_from_schmidt(state, subsystem_qubits):
    """Derive γ from a state vector via its Schmidt decomposition.

    For a bipartite (path ⊗ environment) state with partition
    *subsystem_qubits* = [path qubit index], the Schmidt coefficients
    s₁ ≥ s₂ satisfy:

        2·s₁·s₂ = C = 2·|c₁·c₂|·√(1 − γ²)

    Solving for γ (with |c₁|² = s₁² + (possible M-admixture), |c₂|²
    analogous) gives the environment-marginal coherence as a derived
    quantity — no free parameter.

    This is the Schmidt bridge: the paper's γ is NOT a new independent
    quantity, it is sigma-ground's existing entanglement structure
    expressed in path-basis language.

    Args:
        state:            full state vector (list of complex amplitudes)
        subsystem_qubits: list of qubit indices that play the role of
                          "path"; the rest are the environment/marginal.

    Returns:
        γ in [0, 1]

    Raises:
        ValueError if the state's reduced path marginals are pure-path
        (|c₁·c₂| = 0), in which case γ is undefined.
    """
    # Import locally to avoid a hard import cycle at module load time.
    from sigma_ground.field.interface.quantum_output import schmidt_coefficients

    s = schmidt_coefficients(state, subsystem_qubits)
    if len(s) < 2:
        return 1.0
    s1, s2 = s[0], s[1]

    # Reduced path density matrix has diagonal entries |c_i|² after the
    # environment is traced out.  For states of the form
    #   |Ψ⟩ = c₁|path₁⟩|M₁⟩ + c₂|path₂⟩|M₂⟩,
    # the path-marginal eigenvalues equal s_i² AND also equal |c_i|²
    # (the Schmidt basis coincides with the path basis when γ=1 or when
    # both M-vectors share that basis), so we read |c_i|² = s_i² and
    # recover γ via concurrence inversion.
    c1_sq, c2_sq = s1 * s1, s2 * s2
    prod = math.sqrt(max(0.0, c1_sq * c2_sq))
    if prod < 1e-15:
        raise ValueError(
            "reduced |c1·c2| = 0: gamma is undefined (state is a path eigenstate)"
        )
    concurrence = 2.0 * s1 * s2
    ratio = concurrence / (2.0 * prod)
    return math.sqrt(max(0.0, 1.0 - ratio * ratio))


# ──────────────────────────────────────────────────────────────────────────
#  Phase G — candidate σ → γ derivations
# ──────────────────────────────────────────────────────────────────────────
#
# Four forms connecting the sigma-ground scale field σ to the paper's
# environment marginal coherence γ.  All four satisfy γ(σ=0) = 1
# (laboratory limit, matches matter-shaper empirical) and are monotonic
# non-increasing in σ.  They differ in where γ terminates at σ = σ_conv
# (the matter-conversion horizon).
#
#   mode='linear'    γ(σ) = 1 − (1−Θ)·(σ/σ_conv)
#                    Terminates at Θ = η^(1/3) ≈ 0.7461 (H4 signature).
#
#   mode='exp'       γ(σ) = Θ + (1−Θ)·exp(−σ/σ_conv)
#                    Smooth exponential decay toward Θ; value at σ_conv
#                    is Θ + (1−Θ)/e ≈ 0.8395.
#
#   mode='cbrt'      γ(σ) = (1 − (1−η)·σ/σ_conv)^(1/3)
#                    Cube-root interpolation.  Argument equals 1 at σ=0
#                    (γ=1) and equals η at σ=σ_conv (γ=Θ).  Reads as
#                    "fraction of fossil entanglement remaining, per
#                    gravitational dimension".
#
#   mode='sigma_coh' γ(σ) = 1 − (η/2)·(σ/σ_conv)       ← DEFAULT
#                    Directly mirrors the existing sigma_coherence
#                    formula σ_eff = σ − (η/2)·σ, reinterpreting the
#                    fractional gravitational damping σ_eff/σ as the
#                    quantum marginal coherence γ.  Terminates at
#                    1 − η/2 ≈ 0.7924.  Cleanest "missing link" form:
#                    one mechanism underlies both gravitational
#                    compression and quantum coherence.
#
# The default is 'sigma_coh' because it reuses existing sigma-ground
# physics without introducing a new functional form.  All four modes
# are exported so Phase G/H empirical data can pick the best fit.


def coherence_gamma_from_sigma(sigma, mode='sigma_coh'):
    """Derive γ from the sigma-ground scale field σ.

    FIRST_PRINCIPLES: four candidate derivations connecting sigma-ground's
    gravitational σ to the Khatiwada-Qian marginal coherence γ.  All
    satisfy γ(σ=0) = 1 (laboratory regime → Englert saturation) and are
    monotonic non-increasing in σ.  See module docstring for the forms.

    Args:
        sigma: sigma-ground scale field value in [0, σ_conv]
        mode:  one of 'linear', 'exp', 'cbrt', 'sigma_coh' (default).

    Returns:
        γ in [0, 1] — the marginal coherence predicted for this σ.

    Raises:
        ValueError on unknown mode or σ outside [0, σ_conv].
    """
    # Imported locally to keep this module free of heavy cross-deps at load.
    from sigma_ground.field.constants import (
        ETA, SIGMA_CONV, THETA_ENTANGLEMENT_PER_DIM,
    )

    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if sigma > SIGMA_CONV + 1e-12:
        raise ValueError(f"sigma={sigma} exceeds SIGMA_CONV={SIGMA_CONV}; model undefined past conversion")

    x = sigma / SIGMA_CONV  # normalised curvature in [0, 1]
    Theta = THETA_ENTANGLEMENT_PER_DIM

    if mode == 'linear':
        return 1.0 - (1.0 - Theta) * x
    if mode == 'exp':
        return Theta + (1.0 - Theta) * math.exp(-sigma / SIGMA_CONV)
    if mode == 'cbrt':
        arg = 1.0 - (1.0 - ETA) * x
        return max(0.0, arg) ** (1.0 / 3.0)
    if mode == 'sigma_coh':
        # γ(σ) = σ_eff / σ_local when σ_mean = 0, which reduces to
        # 1 − η/2 · (σ/σ_conv).  Reuses the existing damping ratio.
        return 1.0 - (ETA / 2.0) * x

    if mode == 'csl_linear':
        # CSL α=1 (standard Continuous Spontaneous Localisation).
        # arXiv:2501.17637 proves only α=1 and α=1/2 survive compoundation-
        # invariance and Markovian-feedback tests; α≥3/2 excluded.
        # γ = Θ^(σ/σ_conv) = exp(−κ·x), κ = −ln(Θ), terminates at Θ.
        # [THEORETICAL]
        kappa = -math.log(Theta)  # ≈ 0.293
        return math.exp(-kappa * x)

    if mode == 'csl_psl':
        # PSL α=1/2 (Poissonian Spontaneous Localisation).
        # Same theoretical constraint as csl_linear (arXiv:2501.17637).
        # γ = Θ^√(σ/σ_conv) = exp(−κ·√x) — same endpoint Θ, steeper
        # initial decay due to √x diverging faster near x=0.
        # [THEORETICAL]
        kappa = -math.log(Theta)  # ≈ 0.293
        return math.exp(-kappa * math.sqrt(x))

    if mode == 'dp':
        # Diósi–Penrose gravitational self-energy decoherence.
        # arXiv:2406.18494 (Donadi et al., NJP 26, 113004, 2024).
        # Collapse timescale τ = ℏ/ΔE_grav; mapping σ/σ_conv ↔ ΔE/ΔE_horizon
        # gives γ = exp(−σ/σ_conv), terminating at 1/e ≈ 0.368 — the only
        # candidate below the η-derived floor [0.746, 0.839].
        # [SPECULATIVE]
        return math.exp(-x)

    raise ValueError(
        f"unknown mode '{mode}'; expected one of "
        "'linear', 'exp', 'cbrt', 'sigma_coh', 'csl_linear', 'csl_psl', 'dp'"
    )
