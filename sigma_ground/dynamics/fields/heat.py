"""FVM heat diffusion on a voxel field — the field-physics LAW (first taste).

Explicit finite-volume discretization of Fourier's heat equation ∂T/∂t = α∇²T on
a regular cubic grid: each cell exchanges heat with its 6 neighbours ∝ ΔT. Pure
law — it takes a per-cell thermal-diffusivity grid α (m²/s) built by the
orchestrator from the material physics (α = k/(ρ·cp)) and steps the temperature
field. A WHOLE-GRID sweep, independent of the camera.

numpy/scipy are imported lazily so `dynamics` stays import-clean.

Two solvers:
  ``diffuse``     — the original per-cell ``α·∇²T`` form. Exact for a UNIFORM
                    material; across a material↔material interface it is a
                    first-order approximation (kept as the fast path).
  ``diffuse_fvm`` — the conservative face-flux form with HARMONIC-MEAN face
                    conductivity (the series thermal resistance of the two
                    half-cells meeting at a face — the same law as
                    ``field.interface.thermal.contact_conductance``). Face
                    updates are antisymmetric, so insulated total energy is
                    conserved to float roundoff even across material
                    interfaces. Supports a Dirichlet ambient (air bath) and an
                    explicit Stefan-Boltzmann radiative surface loss (vacuum),
                    with an exact radiated-energy ledger.

The momentum/fluid field laws are the deferred grand-direction arc.
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


def diffuse_fvm(T, k, rho_cp, dx, *, total_time, ambient_mask=None,
                ambient_T=None, radiative=None, max_substeps=200000):
    """Diffuse ``T`` (K) for ``total_time`` (s) — conservative face-flux FVM.

    Pure law: grids in, grids out — no material names. The orchestrator (the
    materia tier) builds ``k``/``rho_cp`` from its material physics.

    Args:
        T:       3-D ndarray of temperatures (K).
        k:       3-D ndarray of thermal conductivity (W/(m·K)) per cell.
                 Void cells carry k=0 → no conduction across a void face
                 (the harmonic mean vanishes when either side is 0).
        rho_cp:  3-D ndarray of volumetric heat capacity (J/(m³·K)) per cell.
                 Cells with rho_cp=0 are inert (void) — they never change.
        dx:      cell edge length (m).
        ambient_mask / ambient_T:
                 Dirichlet air bath — masked cells are held at ``ambient_T``
                 (a scalar, or a per-cell array for a spatial profile) each
                 substep. At a solid↔ambient face the flux uses the SOLID's
                 own conductivity (ghost-cell-same-material convention).
                 APPROXIMATION (flagged): a well-stirred ambient — the rate
                 limit is the solid's own conduction, an UPPER BOUND on
                 convective loss; a real film coefficient h is not derivable
                 from first principles here, so we refuse to invent one.
        radiative:
                 Stefan-Boltzmann surface loss (vacuum media), a dict:
                   surface_faces — 3-D int ndarray, exposed faces per cell
                                   (0–6; the orchestrator counts them, so the
                                   law stays geometry-agnostic),
                   emissivity    — scalar or per-cell ndarray ε,
                   T_env         — environment temperature (K), e.g. 2.725.
                 Applied per substep: dT = −dt·ε·σ_SB·(T⁴−T_env⁴)·n_faces /
                 (rho_cp·dx). APPROXIMATION (flagged): explicit-Euler T⁴ term;
                 voxel stair-step area (dx² per exposed face — the same
                 convention as the deckard free-surface scan); view factor 1
                 to the environment (no self-viewing concavities).

    Returns:
        (T_new, info) — info = {"radiated_J", "substeps", "dt"}. The radiated
        ledger is exact bookkeeping of the applied radiative ΔT, so an
        insulated-in-vacuum run satisfies E0 − E1 == radiated_J to roundoff.
    """
    import numpy as np

    T = np.array(T, dtype=float)
    k = np.asarray(k, dtype=float)
    rho_cp = np.asarray(rho_cp, dtype=float)
    solid = rho_cp > 0.0
    info = {"radiated_J": 0.0, "substeps": 0, "dt": 0.0}
    if total_time <= 0 or not solid.any():
        return T, info

    amb = None
    if ambient_mask is not None and ambient_T is not None:
        amb = np.asarray(ambient_mask, dtype=bool)
        amb_T = np.asarray(ambient_T, dtype=float)

        def _pin(arr):
            arr[amb] = amb_T[amb] if amb_T.ndim else amb_T
    else:
        def _pin(arr):
            pass

    rad = None
    if radiative is not None:
        from ...field.constants import STEFAN_BOLTZMANN
        n_faces = np.asarray(radiative["surface_faces"], dtype=float)
        eps = np.asarray(radiative.get("emissivity", 1.0), dtype=float)
        T_env = float(radiative.get("T_env", 0.0))
        rad_mask = solid & (n_faces > 0)
        if rad_mask.any():
            rad = (n_faces, eps, T_env, rad_mask, STEFAN_BOLTZMANN)

    # ── CFL-safe timestep: conduction bound, then the radiative bound ──
    alpha_max = float((k[solid] / rho_cp[solid]).max())
    dt = stable_dt(alpha_max, dx)
    if rad is not None:
        n_faces, eps, T_env, rad_mask, SB = rad
        # linearized radiative rate 4εσT³·n/(ρcp·dx); harmonic-mean-safe bound
        T_ref = max(float(T.max()), T_env, 1.0)
        eps_max = float(eps.max()) if eps.ndim else float(eps)
        rate = (4.0 * eps_max * SB * T_ref ** 3
                * float(n_faces[rad_mask].max())
                / (float(rho_cp[rad_mask].min()) * dx))
        if rate > 0.0:
            dt = min(dt, 0.2 / rate)
    if not np.isfinite(dt) or dt <= 0:
        return T, info
    n = min(max(int(np.ceil(total_time / dt)), 1), int(max_substeps))
    dt = total_time / n
    info["substeps"], info["dt"] = n, dt

    # Face conductivity: harmonic mean = the series resistance of the two
    # half-cells meeting at the face (same law as thermal.contact_conductance).
    # Precompute per axis — k never changes during the run.
    axes = []
    ndim = T.ndim
    for ax in range(ndim):
        lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(ndim))
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(ndim))
        kA, kB = k[lo], k[hi]
        den = kA + kB
        k_face = np.where(den > 0.0, 2.0 * kA * kB / np.where(den > 0.0, den, 1.0), 0.0)
        if amb is not None:
            # Ghost-cell convention: a solid face against a pinned-ambient cell
            # conducts with the solid's OWN k (the bath is well-stirred, not a
            # second resistor) — reproducing the legacy ``diffuse`` algebra.
            aA, aB = amb[lo], amb[hi]
            k_face = np.where(aA ^ aB, np.where(aA, kB, kA), k_face)
        axes.append((lo, hi, k_face))

    inv_cap = np.where(solid, 1.0 / np.where(solid, rho_cp, 1.0), 0.0) / (dx * dx)
    cell_vol = dx ** 3
    radiated = 0.0
    _pin(T)
    for _ in range(n):
        dE = np.zeros_like(T)
        for lo, hi, k_face in axes:
            F = k_face * (T[hi] - T[lo])       # (flux·dx) through each face
            dE[lo] += F                        # antisymmetric pair → exact
            dE[hi] -= F                        #   conservation by construction
        T = T + dt * dE * inv_cap              # void (inv_cap=0) stays inert
        if rad is not None:
            n_faces, eps, T_env, rad_mask, SB = rad
            dT_rad = np.zeros_like(T)
            dT_rad[rad_mask] = (dt * SB * (T[rad_mask] ** 4 - T_env ** 4)
                                * (eps[rad_mask] if eps.ndim else eps)
                                * n_faces[rad_mask]
                                / (rho_cp[rad_mask] * dx))
            T = T - dT_rad
            # exact ledger: the energy this ΔT removed, same arrays, same dtype
            radiated += float((dT_rad[rad_mask] * rho_cp[rad_mask]).sum()) * cell_vol
        _pin(T)
    info["radiated_J"] = radiated
    return T, info


def thermal_energy(T, vol_heat_capacity, dx):
    """Total thermal energy Σ (ρ·cp)·T·cell_volume (J) — for conservation checks.

    ``vol_heat_capacity`` is the volumetric heat capacity ρ·cp (J/m³·K) per cell.
    """
    import numpy as np
    return float((np.asarray(vol_heat_capacity) * np.asarray(T)).sum()) * dx ** 3


__all__ = ["diffuse", "diffuse_fvm", "stable_dt", "thermal_energy"]
