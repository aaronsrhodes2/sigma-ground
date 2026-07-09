"""Electronics / solid-state tools for the Sigma Ground MCP.

Thin wrappers over sigma_ground.field.interface.electronics — metal transport,
semiconductor properties, p-n junctions, capacitance. Verified against textbook
values before exposure (copper ρ=1.68e-8 Ω·m; Si E_g=1.12 eV; Cu mean free path
39 nm; Si V_bi=0.74 V). Metal-only and semiconductor-only functions validate the
key class and decline gracefully on the wrong material kind.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface.electronics"


def _decline(notes: str, **inputs) -> dict[str, Any]:
    return ToolResult(value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
                      notes=notes, inputs=inputs).to_dict()


def _ok(value, units, formula, notes="", **inputs) -> dict[str, Any]:
    return ToolResult(value=value, units=units, source=_SRC,
                      provenance_tag="DERIVED", formula=formula, notes=notes,
                      inputs=inputs).to_dict()


def _metals():
    from sigma_ground.field.interface.electronics import METAL_TRANSPORT
    return METAL_TRANSPORT


def _semis():
    from sigma_ground.field.interface.electronics import SEMICONDUCTORS
    return SEMICONDUCTORS


def _metal_guard(metal_key: str):
    """Return a decline dict unless metal_key is a *pure* metal, else None.

    'silicon' lives in BOTH METAL_TRANSPORT and SEMICONDUCTORS (materia uses its
    bulk properties), so the metal Drude / free-electron models would otherwise
    hand back a confidently-wrong number for it. Redirect semiconductors to the
    semiconductor_* / pn_* tools rather than answer with the wrong model.
    """
    if metal_key in _semis():
        return _decline(f"{metal_key!r} is a semiconductor; the metal transport "
                        f"models (Drude / free-electron) don't apply — use the "
                        f"semiconductor_band_gap / intrinsic_carrier_density / "
                        f"pn_* tools instead", metal_key=metal_key)
    if metal_key not in _metals():
        pure = sorted(m for m in _metals() if m not in _semis())
        return _decline(f"Needs a metal in {pure}; got {metal_key!r}",
                        metal_key=metal_key)
    return None


# ── metal transport ─────────────────────────────────────────────────────
def electrical_resistivity(metal_key: str, temperature_k: float = 300.0) -> dict[str, Any]:
    """Electrical resistivity of a metal (Ω·m). e.g. copper, aluminum, gold, tungsten."""
    from sigma_ground.field.interface.electronics import resistivity as f
    g = _metal_guard(metal_key)
    if g is not None:
        return g
    try:
        v = f(metal_key, float(temperature_k))
    except Exception as e:
        return _decline(f"resistivity failed: {e}", metal_key=metal_key)
    return _ok(v, "Ω·m", "electron–phonon transport ρ(T)",
               metal_key=metal_key, temperature_k=temperature_k)


def carrier_mobility(metal_key: str, temperature_k: float = 300.0) -> dict[str, Any]:
    """Drude carrier mobility of a METAL (m²/V·s). Metals only (not semiconductors)."""
    from sigma_ground.field.interface.electronics import carrier_mobility as f
    g = _metal_guard(metal_key)
    if g is not None:
        return g
    try:
        v = f(metal_key, float(temperature_k))
    except Exception as e:
        return _decline(f"carrier_mobility failed: {e}", metal_key=metal_key)
    return _ok(v, "m^2/(V·s)", "μ = σ/(n e) (Drude)",
               metal_key=metal_key, temperature_k=temperature_k)


def hall_coefficient(metal_key: str) -> dict[str, Any]:
    """Hall coefficient of a metal (m³/C), free-electron model R_H = −1/(n e)."""
    from sigma_ground.field.interface.electronics import hall_coefficient as f
    g = _metal_guard(metal_key)
    if g is not None:
        return g
    try:
        v = f(metal_key)
    except Exception as e:
        return _decline(f"hall_coefficient failed: {e}", metal_key=metal_key)
    return _ok(v, "m^3/C", "R_H = −1/(n e)", metal_key=metal_key)


def electron_mean_free_path(metal_key: str, temperature_k: float = 300.0) -> dict[str, Any]:
    """Electron mean free path in a metal (m). e.g. copper ≈ 39 nm at 300 K."""
    from sigma_ground.field.interface.electronics import mean_free_path as f
    g = _metal_guard(metal_key)
    if g is not None:
        return g
    try:
        v = f(metal_key, float(temperature_k))
    except Exception as e:
        return _decline(f"mean_free_path failed: {e}", metal_key=metal_key)
    return _ok(v, "m", "ℓ = v_F τ", metal_key=metal_key, temperature_k=temperature_k)


def free_electron_density(metal_key: str) -> dict[str, Any]:
    """Conduction-electron number density of a metal (m⁻³). e.g. copper 8.5e28."""
    from sigma_ground.field.interface.electronics import free_electron_density as f
    g = _metal_guard(metal_key)
    if g is not None:
        return g
    try:
        v = f(metal_key)
    except Exception as e:
        return _decline(f"free_electron_density failed: {e}", metal_key=metal_key)
    return _ok(v, "m^-3", "n = Z ρ N_A / M (valence electrons)", metal_key=metal_key)


# ── semiconductors ──────────────────────────────────────────────────────
def semiconductor_band_gap(semiconductor_key: str, temperature_k: float = 300.0) -> dict[str, Any]:
    """Band gap of a semiconductor (eV). e.g. silicon 1.12, germanium, gallium_arsenide."""
    from sigma_ground.field.interface.electronics import band_gap as f
    if semiconductor_key not in _semis():
        return _decline(f"Needs a semiconductor in {sorted(_semis())}; "
                        f"got {semiconductor_key!r}", semiconductor_key=semiconductor_key)
    try:
        v = f(semiconductor_key, float(temperature_k))
    except Exception as e:
        return _decline(f"band_gap failed: {e}", semiconductor_key=semiconductor_key)
    return _ok(v, "eV", "E_g(T) (Varshni)",
               semiconductor_key=semiconductor_key, temperature_k=temperature_k)


def intrinsic_carrier_density(semiconductor_key: str, temperature_k: float = 300.0) -> dict[str, Any]:
    """Intrinsic carrier concentration n_i (m⁻³) of a semiconductor."""
    from sigma_ground.field.interface.electronics import intrinsic_carrier_concentration as f
    if semiconductor_key not in _semis():
        return _decline(f"Needs a semiconductor in {sorted(_semis())}; "
                        f"got {semiconductor_key!r}", semiconductor_key=semiconductor_key)
    try:
        v = f(semiconductor_key, float(temperature_k))
    except Exception as e:
        return _decline(f"intrinsic_carrier failed: {e}", semiconductor_key=semiconductor_key)
    return _ok(v, "m^-3", "n_i = √(N_c N_v) exp(−E_g/2kT)",
               semiconductor_key=semiconductor_key, temperature_k=temperature_k)


def pn_built_in_voltage(semiconductor_key: str, donor_density_m3: float,
                        acceptor_density_m3: float, temperature_k: float = 300.0) -> dict[str, Any]:
    """Built-in voltage of a p-n junction (V): V_bi = (kT/e) ln(N_A N_D / n_i²)."""
    from sigma_ground.field.interface.electronics import built_in_voltage as f
    if semiconductor_key not in _semis():
        return _decline(f"Needs a semiconductor in {sorted(_semis())}; "
                        f"got {semiconductor_key!r}", semiconductor_key=semiconductor_key)
    try:
        v = f(semiconductor_key, float(donor_density_m3), float(acceptor_density_m3),
              float(temperature_k))
    except Exception as e:
        return _decline(f"built_in_voltage failed: {e}", semiconductor_key=semiconductor_key)
    return _ok(v, "V", "V_bi = (kT/e) ln(N_A N_D / n_i²)",
               semiconductor_key=semiconductor_key,
               donor_density_m3=donor_density_m3, acceptor_density_m3=acceptor_density_m3)


def depletion_width(semiconductor_key: str, donor_density_m3: float,
                    acceptor_density_m3: float, applied_voltage_v: float = 0.0,
                    temperature_k: float = 300.0) -> dict[str, Any]:
    """Depletion-region width of a p-n junction (m)."""
    from sigma_ground.field.interface.electronics import depletion_width as f
    if semiconductor_key not in _semis():
        return _decline(f"Needs a semiconductor in {sorted(_semis())}; "
                        f"got {semiconductor_key!r}", semiconductor_key=semiconductor_key)
    try:
        v = f(semiconductor_key, float(donor_density_m3), float(acceptor_density_m3),
              float(applied_voltage_v), float(temperature_k))
    except Exception as e:
        return _decline(f"depletion_width failed: {e}", semiconductor_key=semiconductor_key)
    return _ok(v, "m", "W = √(2ε(V_bi−V)/e · (1/N_A + 1/N_D))",
               semiconductor_key=semiconductor_key)


# ── junctions & capacitance ─────────────────────────────────────────────
def diode_current(saturation_current_a: float, voltage_v: float,
                  temperature_k: float = 300.0) -> dict[str, Any]:
    """Shockley diode current (A): I = I₀(exp(eV/kT) − 1)."""
    from sigma_ground.field.interface.electronics import diode_current as f
    try:
        v = f(float(saturation_current_a), float(voltage_v), float(temperature_k))
    except Exception as e:
        return _decline(f"diode_current failed: {e}")
    return _ok(v, "A", "I = I₀(exp(eV/kT) − 1)",
               saturation_current_a=saturation_current_a, voltage_v=voltage_v)


def parallel_plate_capacitance(area_m2: float, separation_m: float,
                               relative_permittivity: float = 1.0) -> dict[str, Any]:
    """Parallel-plate capacitance (F): C = ε₀ε_r A / d."""
    from sigma_ground.field.interface.electronics import parallel_plate_capacitance as f
    try:
        v = f(float(area_m2), float(separation_m), float(relative_permittivity))
    except Exception as e:
        return _decline(f"parallel_plate_capacitance failed: {e}")
    return _ok(v, "F", "C = ε₀ε_r A / d",
               area_m2=area_m2, separation_m=separation_m,
               relative_permittivity=relative_permittivity)
