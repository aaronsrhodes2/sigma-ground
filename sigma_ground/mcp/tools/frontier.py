"""Frontier physics — black-hole thermodynamics, holography, entanglement counting.

These closed-form tools make the boundary-of-physics questions callable:
Bekenstein-Hawking entropy, the holographic thread/pixel count of a surface,
the entanglements-to-pop-a-bubble count, the baryon-vs-horizon-capacity
crossover, and gravitational binding energy. Every one is one line of
geometry grounded in the library's G, c, hbar, L_p — no solvers.

Provenance: G, C, HBAR, L_PLANCK, PROTON_TOTAL_MEV from
sigma_ground.field.constants.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult

_MeV_J = 1.602176634e-13


def _consts():
    import sigma_ground.field.constants as C
    c = 299792458.0
    mp = C.PROTON_TOTAL_MEV * _MeV_J / c**2
    return C.G, c, C.HBAR, C.L_PLANCK, C.K_B, mp


def bekenstein_hawking_entropy(mass_kg: float) -> ToolResult:
    """Black-hole entropy, temperature, horizon area, and holographic thread count.

    S = A / (4 L_p²) in units of k_B (this is also the max number of
    entanglement threads / Planck pixels the horizon can hold).

    Returns a dict: {entropy_k_B, horizon_area_m2, schwarzschild_radius_m,
    hawking_temperature_K, thread_count}.
    """
    G, c, hbar, Lp, kB, _ = _consts()
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg})
    r_s = 2 * G * mass_kg / c**2
    A = 4 * math.pi * r_s**2
    Lp2 = Lp * Lp
    S = A / (4 * Lp2)                       # entropy in k_B (= thread count)
    T_H = hbar * c**3 / (8 * math.pi * G * mass_kg * kB)
    return ToolResult(
        value={
            "entropy_k_B": S,
            "thread_count": S,
            "horizon_area_m2": A,
            "schwarzschild_radius_m": r_s,
            "hawking_temperature_K": T_H,
        },
        units="",
        source="sigma-ground (Bekenstein 1973 / Hawking 1974)",
        formula="S = A/(4 L_p²) = k_B c³ A/(4 G ħ)",
        inputs={"mass_kg": mass_kg},
        notes=("entropy_k_B is also the max entanglement threads the horizon "
                "can hold (one per ~4 Planck pixels). Solar mass ~10^77."),
    )


def entanglements_to_pop_bubble(radius_m: float) -> ToolResult:
    """Number of entanglement threads to saturate (pop) a bubble of radius R.

    Each thread pierces the surface at one Planck pixel; the surface holds
    N = π R² / L_p² of them. Saturating the pixels is the collapse/reform
    threshold. The smallest possible bubble (R = L_p) pops at N = π ≈ 3 —
    the irreducible quantum of cavitation.

    Returns the thread count (dimensionless).
    """
    G, c, hbar, Lp, kB, _ = _consts()
    if radius_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"radius_m": radius_m})
    N = math.pi * radius_m**2 / (Lp * Lp)
    return ToolResult(
        value=N, units="",
        source="sigma-ground (holographic thread saturation)",
        formula="N = π R² / L_p²",
        inputs={"radius_m": radius_m},
        notes=("Threads to pop a bubble of radius R. Smallest bubble "
                "(R=L_p) pops at N=π≈3 — the quantum of cavitation. "
                "= the bubble's Bekenstein-Hawking pixel count."),
    )


def holographic_matching_mass() -> ToolResult:
    """The black-hole mass where baryon count equals horizon pixel count.

    Baryons scale as M; the horizon disc scales as M². They cross at one
    mass: M = ħ c / (4π G m_p) ≈ 2.25e10 kg (a mountain crushed smaller
    than a proton). Above it the horizon has capacity to spare; below it
    the matter overflows the horizon (the information-paradox regime).
    """
    G, c, hbar, Lp, kB, mp = _consts()
    M = hbar * c / (4 * math.pi * G * mp)
    r_s = 2 * G * M / c**2
    return ToolResult(
        value=M, units="kg",
        source="sigma-ground (baryon-count = horizon-pixel crossover)",
        formula="M = ħ c / (4π G m_p)",
        inputs={},
        notes=(f"r_s = {r_s:.3e} m (sub-proton). Above: disc has room; "
                f"below: matter overflows the horizon."),
    )


def baryon_vs_disc(mass_kg: float) -> ToolResult:
    """Compare a black hole's baryon count to its horizon pixel count.

    Returns {n_baryons, n_disc_pixels, ratio_disc_over_baryons, regime}.
    regime = 'horizon has room' (disc > baryons) or 'matter overflows'
    (baryons > disc) — the latter is the sub-matching-mass info puzzle.
    """
    G, c, hbar, Lp, kB, mp = _consts()
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg})
    r_s = 2 * G * mass_kg / c**2
    A = 4 * math.pi * r_s**2
    n_disc = A / (4 * Lp * Lp)
    n_bary = mass_kg / mp
    ratio = n_disc / n_bary
    regime = "horizon has room to spare" if ratio >= 1 else "matter overflows the horizon"
    return ToolResult(
        value={
            "n_baryons": n_bary,
            "n_disc_pixels": n_disc,
            "ratio_disc_over_baryons": ratio,
            "regime": regime,
        },
        units="",
        source="sigma-ground (holographic baryon accounting)",
        formula="n_disc = A/(4 L_p²);  n_baryons = M/m_p",
        inputs={"mass_kg": mass_kg},
        notes=("Baryons ~ M, disc ~ M², so they cross at the matching mass "
                "(~2.25e10 kg). Below it, baryon info overflows the horizon."),
    )


def gravitational_binding_energy(mass_kg: float, radius_m: float) -> ToolResult:
    """Gravitational binding energy of a uniform sphere: U = (3/5) G M² / R.

    The energy released assembling the mass from infinity into a sphere of
    radius R — i.e., the energy your 'thread compression' liberates as a
    star contracts.
    """
    G, c, hbar, Lp, kB, _ = _consts()
    if mass_kg <= 0 or radius_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg, "radius_m": radius_m})
    U = 0.6 * G * mass_kg**2 / radius_m
    return ToolResult(
        value=U, units="J",
        source="sigma-ground (uniform-sphere self-gravity)",
        formula="U = (3/5) G M² / R",
        inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        notes="Binding energy of a uniform sphere; released on contraction.",
    )


def unruh_temperature(acceleration_m_s2: float) -> ToolResult:
    """Unruh temperature seen by an accelerated observer: T = ħ a / (2π c k_B)."""
    G, c, hbar, Lp, kB, _ = _consts()
    if acceleration_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"acceleration_m_s2": acceleration_m_s2})
    T = hbar * acceleration_m_s2 / (2 * math.pi * c * kB)
    return ToolResult(
        value=T, units="K",
        source="sigma-ground (Unruh 1976)",
        formula="T = ħ a / (2π c k_B)",
        inputs={"acceleration_m_s2": acceleration_m_s2},
        notes="The vacuum looks thermal to an accelerated observer.",
    )


def entanglement_channel(scenario: str = "") -> ToolResult:
    """What an entangled pair can and cannot do as a channel.

    Encodes the hard facts so the switchboard never mis-answers "can
    entangled particles communicate faster than light?" (the answer is
    NO — the no-communication theorem). Returns a dict whose `verdict`
    is tailored to the scenario:

      - communication / signaling / FTL  → NO (no-communication theorem)
      - key / QKD / cryptography         → YES, a shared SECRET KEY (Ekert)
      - CHSH / Bell / Tsirelson maximum  → 2√2 ≈ 2.828 (quantum bound)

    Entanglement is a *correlation* resource, not a *communication*
    channel: it shares private randomness, never a message. Any usable
    message needs a separate classical channel at ≤ c.
    """
    s = (scenario or "").lower()
    chsh_quantum = 2.0 * math.sqrt(2.0)      # Tsirelson bound ≈ 2.828
    facts = {
        "can_transmit_information": False,
        "can_signal_faster_than_light": False,
        "no_communication_theorem": (
            "Local measurement outcomes are random and the marginal "
            "statistics on one side are independent of the partner's "
            "measurement choice — so entanglement alone transmits zero bits."),
        "shared_resource": (
            "shared secret randomness: a private correlated key, the basis "
            "of quantum key distribution (Ekert E91)."),
        "chsh_classical_max": 2.0,
        "chsh_quantum_max": chsh_quantum,
        "monogamy": (
            "If A is maximally entangled with B it cannot also be entangled "
            "with C — which is why an eavesdropper is detectable."),
        "requires_classical_channel": (
            "To USE the correlation (teleportation, key agreement) a separate "
            "classical channel at ≤ c is mandatory."),
    }
    if any(k in s for k in ("key", "qkd", "secret", "crypto", "encrypt", "secure")):
        verdict = ("YES — entanglement can establish a shared SECRET KEY between "
                    "two parties (quantum key distribution, Ekert E91). It shares "
                    "private correlated randomness, not a message; any eavesdropper "
                    "is detectable via monogamy. No information is sent by the "
                    "entanglement itself.")
        primary = "shared secret key (QKD)"
    elif any(k in s for k in ("chsh", "tsirelson", "bell value", "bell inequality",
                                "maximum", "max value", "violat", "correlation bound")):
        verdict = (f"CHSH ≤ 2 for any classical (local-hidden-variable) theory; "
                    f"quantum mechanics allows up to 2√2 ≈ {chsh_quantum:.4f} "
                    f"(the Tsirelson bound). A value above 2 proves the correlation "
                    f"is nonclassical — but even the maximum violation cannot signal.")
        primary = chsh_quantum
    else:                                      # communication / FTL / signaling
        verdict = ("NO — entangled particles cannot communicate or send "
                    "information, faster than light or otherwise (the "
                    "no-communication theorem). They share private correlated "
                    "randomness, not a signal; extracting any message requires a "
                    "separate classical channel at ≤ c.")
        primary = False
    return ToolResult(
        value={"verdict": verdict, "primary": primary, **facts},
        units="",
        source=("sigma-ground (no-communication theorem; Bell 1964; "
                "Tsirelson 1980; Ekert 1991)"),
        formula="CHSH: |S| ≤ 2 (classical) < |S| ≤ 2√2 (quantum); no-signaling",
        inputs={"scenario": scenario},
        notes=("Entanglement is a correlation resource, not a communication "
                "channel. Faster-than-light signaling via entanglement is "
                "impossible; the only transferable thing is shared secret "
                "randomness, and using it needs a classical channel at ≤ c."),
    )
