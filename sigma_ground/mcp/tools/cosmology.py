"""Cosmology tools: Hubble radius, HDE, MOND classifier, dark energy.

Wraps sigma_ground.field.interface.cosmology, which holds the physics
implementations plus the DESI Union3 dark-energy integration.

Mentat reports standard, observation-anchored cosmology: the Holographic
Dark Energy (HDE) c^2 parameter is taken straight from the DESI 2024
Union3 fit. No speculative interpretation is layered on top.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


def hubble_radius() -> ToolResult:
    """Current-epoch Hubble radius R_H = c / H_0 in meters.

    Uses H_0 from sigma_ground.field.constants (Planck 2018 base value).
    """
    from sigma_ground.field.interface.cosmology import hubble_radius as _hr
    R_H = _hr()
    return ToolResult(
        value=R_H,
        units="m",
        source="sigma-ground (Planck 2018 H_0)",
        formula="R_H = c / H_0",
        notes=(f"~{R_H/3.086e22:.2f} Gpc. The Hubble radius is the distance "
                f"at which the cosmological recession velocity equals c."),
    )


def hde_dark_energy_density(c_squared: float | None = None,
                              L_meters: float | None = None) -> ToolResult:
    """Holographic Dark Energy density: rho_DE = 3 c^2 M_Pl^2 c^4 / (8 pi G L^2).

    Parameters
    ----------
    c_squared : float | None
        HDE parameter c^2. Defaults to the DESI 2024 Union3 fit (~0.4122).
    L_meters : float | None
        IR cutoff length. Defaults to Hubble radius.

    Returns
    -------
    ToolResult with energy density in J/m^3.
    """
    if c_squared is not None and c_squared < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="c_squared must be non-negative",
                           inputs={"c_squared": c_squared, "L_meters": L_meters})
    if L_meters is not None and L_meters <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="L_meters must be positive",
                           inputs={"c_squared": c_squared, "L_meters": L_meters})
    from sigma_ground.field.interface.cosmology import hde_rho_de
    rho = hde_rho_de(c_squared=c_squared, L=L_meters)
    return ToolResult(
        value=rho,
        units="J/m^3",
        source="sigma-ground (HDE; Li 2004, DESI 2024 anchor)",
        formula="rho_DE = 3 c^2 M_Pl^2 c^4 / (8 pi G L^2)",
        inputs={"c_squared": c_squared, "L_meters": L_meters},
        notes=("Holographic Dark Energy model (Li 2004). c^2 defaults to the "
                "DESI 2024 Union3 fit (~0.4122); L defaults to the Hubble "
                "radius."),
    )


def eta_desi_band_check(dataset: str = "dr2") -> ToolResult:
    """Does the adopted HDE c^2 fall within the DESI Union3 1-sigma band?

    Parameters
    ----------
    dataset : str
        "dr2" (default) or "dr3" (placeholder, not yet lifted).

    Returns
    -------
    ToolResult with `value` = dict {in_band, low, high, eta}.
    """
    from sigma_ground.field.interface.cosmology import eta_in_desi_union3_band
    if dataset not in ("dr2", "dr3"):
        return ToolResult(value=None, source="invalid input",
                           notes="dataset must be 'dr2' or 'dr3'",
                           inputs={"dataset": dataset})
    r = eta_in_desi_union3_band(dataset)
    return ToolResult(
        value=r,
        source="sigma-ground (DESI 2024 Union3 HDE fit, arXiv:2411.08639)",
        inputs={"dataset": dataset},
        notes=("Consistency check of the adopted Holographic Dark Energy c^2 "
                "against the central DESI 2024 Union3 fit."),
    )


def mond_regime_classifier(acceleration_m_s2: float) -> ToolResult:
    """Classify an acceleration as Newtonian / MOND-regime / transition.

    Uses Milgrom's a_0 ~ 1.2e-10 m/s^2. Above ~10 a_0, gravity is
    Newtonian; below ~0.1 a_0, deep MOND regime.

    Parameters
    ----------
    acceleration_m_s2 : float
        Magnitude of the gravitational acceleration in m/s^2.

    Returns
    -------
    ToolResult with `value` in {"newtonian", "mond", "transition"}.
    """
    if acceleration_m_s2 < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="acceleration_m_s2 must be non-negative",
                           inputs={"acceleration_m_s2": acceleration_m_s2})
    from sigma_ground.field.interface.cosmology import newtonian_regime
    regime = newtonian_regime(acceleration_m_s2)
    return ToolResult(
        value=regime,
        source="sigma-ground (Milgrom MOND, arXiv:2511.05632)",
        inputs={"acceleration_m_s2": acceleration_m_s2},
        notes=("Newtonian gravity well-tested above ~1e-9 m/s^2. "
                "Galaxy outskirts probe the transition (~1e-10 m/s^2). "
                "MOND is one phenomenological resolution; dark matter "
                "is another. The classifier just flags which regime "
                "your input is in."),
    )


def mond_a0_constant() -> ToolResult:
    """Milgrom's a_0 constant in m/s^2, with the cosmological coincidence check."""
    from sigma_ground.field.interface.cosmology import a0_mond_from_cosmological_scale
    r = a0_mond_from_cosmological_scale()
    return ToolResult(
        value=r["a0_measured"],
        units="m/s^2",
        source="sigma-ground (Milgrom 1983; SPARC 2016 confirmation)",
        formula="a_0 ~ c H_0 / (2 pi)",
        inputs={},
        notes=("Empirical MOND constant. The cosmological coincidence: "
                f"a_0 ~ c H_0/(2 pi) = {r['a0_predicted']:.3e} m/s^2 "
                f"vs measured {r['a0_measured']:.3e} m/s^2 "
                f"(ratio: {r['ratio']:.3f})."),
    )


def critical_density() -> ToolResult:
    """Cosmological critical density rho_crit = 3 H_0^2 / (8 pi G)."""
    import math
    from sigma_ground.field.constants import H0, G
    rho_crit = 3.0 * H0 * H0 / (8.0 * math.pi * G)
    return ToolResult(
        value=rho_crit,
        units="kg/m^3",
        source="sigma-ground (FLRW critical density, Planck 2018 H_0)",
        formula="rho_crit = 3 H_0^2 / (8 pi G)",
        inputs={},
        notes=(f"~{rho_crit*1e29:.3f}e-29 kg/m^3. Density at which the "
                f"universe would be spatially flat. Current best estimate "
                f"matches this closely (Omega_total ~ 1)."),
    )


def age_of_universe(mode: str = "lcdm") -> ToolResult:
    """Age of the universe -- two DISTINCT, both-legitimate quantities:

    mode="lcdm" (default): the TRUE flat-LambdaCDM age, integrating the
      Friedmann equation with matter + dark energy:
        t(a=1) = (2 / (3 H0 sqrt(OmegaLambda))) * asinh(sqrt(OmegaLambda/Omega_m))
      Planck 2018 parameters give ~13.80 Gyr, matching the standard
      "13.787 Gyr" figure to <0.1%. This is what "how old is the universe"
      means in standard cosmology.
    mode="hubble": the Hubble time t_H = 1/H_0 -- a DIFFERENT quantity (the
      age a universe with NO deceleration/acceleration would have), used
      as a quick order-of-magnitude estimate. Differs from the true age by
      an O(1) factor; kept available under its own mode since some
      questions genuinely ask for it by name ("Hubble time").
    """
    from sigma_ground.field.constants import H0
    tH_sec = 1.0 / H0
    tH_gyr = tH_sec / (1e9 * 365.25 * 86400.0)

    if mode == "hubble":
        return ToolResult(
            value=tH_sec,
            units="s",
            source="sigma-ground (Planck 2018 H_0)",
            formula="t_H = 1 / H_0",
            inputs={"mode": mode},
            notes=(f"Hubble time = {tH_gyr:.2f} Gyr. The TRUE age of the "
                    f"universe in LambdaCDM is ~13.80 Gyr (Planck 2018) -- "
                    f"call age_of_universe(mode='lcdm') for that."),
        )

    import math
    from sigma_ground.field.constants import OMEGA_M, OMEGA_LAMBDA
    t_sec = ((2.0 / (3.0 * H0 * math.sqrt(OMEGA_LAMBDA)))
             * math.asinh(math.sqrt(OMEGA_LAMBDA / OMEGA_M)))
    t_gyr = t_sec / (1e9 * 365.25 * 86400.0)
    return ToolResult(
        value=t_sec,
        units="s",
        source="sigma-ground (Planck 2018 flat LambdaCDM)",
        formula="t = (2/(3 H0 sqrt(OL))) asinh(sqrt(OL/Om))",
        inputs={"mode": mode},
        notes=(f"True LambdaCDM age = {t_gyr:.3f} Gyr (Planck 2018: "
                f"Omega_m={OMEGA_M}, Omega_Lambda={OMEGA_LAMBDA}). "
                f"The pure Hubble time t_H=1/H0 = {tH_gyr:.2f} Gyr is a "
                f"DIFFERENT quantity -- call age_of_universe(mode='hubble') "
                f"for that."),
    )


def eta_value_report() -> ToolResult:
    """The adopted Holographic Dark Energy c^2 parameter, with provenance.

    Returns the DESI 2024 Union3 HDE c^2 value and its source. This is a
    standard observational input to the HDE dark-energy model.
    """
    from sigma_ground.field.constants import ETA, ETA_UNCERTAINTY_1SIGMA, C_HDE_UNION3
    return ToolResult(
        value=ETA,
        units="dimensionless",
        source="sigma-ground via DESI 2024 Union3 HDE fit (arXiv:2411.08639)",
        provenance_tag="EMPIRICAL-INPUT",
        uncertainty=ETA_UNCERTAINTY_1SIGMA,
        formula=f"ETA = c^2_DESI_Union3 = {C_HDE_UNION3}^2",
        inputs={},
        notes=("Empirical-input Holographic Dark Energy c^2, anchored at the "
                "central DESI 2024 Union3 HDE fit. 1-sigma uncertainty = "
                "2 c sigma_c = 0.036."),
    )
