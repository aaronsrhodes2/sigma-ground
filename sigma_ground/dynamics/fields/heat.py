"""FVM heat diffusion on a voxel field — the field-physics LAW (first taste).

Explicit finite-volume discretization of Fourier's heat equation ∂T/∂t = α∇²T on
a regular cubic grid: each cell exchanges heat with its 6 neighbours ∝ ΔT. Pure
law — it takes a per-cell thermal-diffusivity grid α (m²/s) built by the
orchestrator from the material physics (α = k/(ρ·cp)) and steps the temperature
field. A WHOLE-GRID sweep, independent of the camera.

numpy/scipy are imported lazily so `dynamics` stays import-clean.

Honest limit: the explicit per-cell ``α·∇²T`` form conserves energy exactly for a
UNIFORM material (the common-interface conductivity equals α); across a
material↔material interface the flux should use the harmonic-mean conductivity —
not yet done, so mixed-material heat is a first-order approximation (flagged). The
momentum/fluid field laws are the deferred grand-direction arc.
"""
from __future__ import annotations


def stable_dt(alpha_max, dx, safety=0.2):
    """CFL-stable timestep for explicit 3-D diffusion: dt ≤ dx²/(6·α_max)."""
    if alpha_max <= 0:
        return float("inf")
    return safety * dx * dx / (6.0 * alpha_max)


def diffuse(T, alpha, dx, *, total_time, ambient_mask=None, ambient_T=None,
            max_substeps=200000):
    """Diffuse a temperature field ``T`` (K) for ``total_time`` (s).

    Args:
        T:      3-D ndarray of temperatures (K).
        alpha:  3-D ndarray of thermal diffusivity (m²/s) per cell (same shape).
        dx:     cell edge length (m).
        ambient_mask / ambient_T: cells held at a fixed temperature each step
            (Dirichlet — e.g. void = ambient air). If None, the grid edges are
            insulated (Neumann, zero-flux) and total energy is conserved.

    Returns a NEW stepped temperature array. The step count is chosen CFL-safe.
    """
    import numpy as np
    import scipy.ndimage as ndi

    T = np.array(T, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    amax = float(alpha.max()) if alpha.size else 0.0
    dt = stable_dt(amax, dx)
    if not np.isfinite(dt) or dt <= 0 or total_time <= 0:
        return T
    n = min(max(int(np.ceil(total_time / dt)), 1), int(max_substeps))
    dt = total_time / n
    inv_dx2 = 1.0 / (dx * dx)
    if ambient_mask is not None and ambient_T is not None:
        T[ambient_mask] = ambient_T
    for _ in range(n):
        lap = ndi.laplace(T, mode="nearest") * inv_dx2        # ∇²T (zero-flux edges)
        T = T + dt * alpha * lap
        if ambient_mask is not None and ambient_T is not None:
            T[ambient_mask] = ambient_T
    return T


def thermal_energy(T, vol_heat_capacity, dx):
    """Total thermal energy Σ (ρ·cp)·T·cell_volume (J) — for conservation checks.

    ``vol_heat_capacity`` is the volumetric heat capacity ρ·cp (J/m³·K) per cell.
    """
    import numpy as np
    return float((np.asarray(vol_heat_capacity) * np.asarray(T)).sum()) * dx ** 3


__all__ = ["diffuse", "stable_dt", "thermal_energy"]
