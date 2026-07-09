"""Spatial drag-heating — a temperature FIELD over a body, not one bulk scalar.

Drag heating is a windward phenomenon: the air slams the leading face, so the
energy lands there (Newtonian impact heating ∝ cosθ from the stagnation point),
then conducts inward through the solid while the body sheds heat to the passing
air. A reentry sphere is therefore white-hot on its leading face and dark on its
trailing side — a thing a single ``peak_T`` can never show, but a per-cell field
can.

This builds that field on a voxel grid and evolves it with the real physics:
  - α = k / (ρ·cp), straight from ``field.interface.thermal`` (grounded);
  - the CITED total drag-dissipation energy deposited on the windward surface
    cells, weighted by max(0, n̂·v̂) (Newtonian/Lees cosine law);
  - Fourier conduction via ``dynamics.fields.heat.diffuse``, with the surrounding
    void held at ambient (the body loses surface heat to the air it falls through).

Energy is NOT conserved by design — the body sheds heat to ambient air, which is
the physical reality of a falling hot body. The total DEPOSITED energy is the
cited drag dissipation; where it ends up is the physics.

Tier: materia (3) reading dynamics.fields (1) + field.interface (1) — downward,
legal. numpy/scipy are imported lazily so the module stays import-clean.
"""
from __future__ import annotations

import math


class ThermalField:
    """A per-cell temperature grid centred on the origin, with point sampling.

    Cell (i,j,k) sits at ``(i-(n-1)/2)·dx`` on each axis (metres). ``sample`` does
    nearest-cell lookup in that frame and falls back to ambient outside the grid.
    """
    __slots__ = ("T", "dx", "n", "solid", "T_ambient", "windward_T",
                 "trailing_T", "radius_m")

    def __init__(self, T, dx, n, solid, T_ambient, windward_T, trailing_T,
                 radius_m=0.05):
        self.T = T
        self.dx = float(dx)
        self.n = int(n)
        self.solid = solid
        self.T_ambient = float(T_ambient)
        self.windward_T = float(windward_T)
        self.trailing_T = float(trailing_T)
        self.radius_m = float(radius_m)

    def sample(self, x, y, z):
        """Temperature (K) at a physical position in the field's centred frame."""
        h = (self.n - 1) / 2.0
        i = int(round(x / self.dx + h))
        j = int(round(y / self.dx + h))
        k = int(round(z / self.dx + h))
        if 0 <= i < self.n and 0 <= j < self.n and 0 <= k < self.n:
            return float(self.T[i, j, k])
        return self.T_ambient


def windward_heating_field(radius_m, material_key, *, windward_T,
                           spread_time_s, velocity_dir=(0.0, -1.0, 0.0),
                           T_ambient=288.15, n=30):
    """Temperature field for a sphere drag-heated on its windward face.

    The aerodynamic heating is SUSTAINED (the body keeps slamming into air), so
    the windward surface is HELD at the heating temperature — a Newtonian cosθ
    profile from ``windward_T`` at the stagnation point down to ambient at the
    shoulder. Conduction (Fourier, the ``dynamics.fields.heat`` law) carries that
    heat inward as a thin rind over ``spread_time_s``; the trailing surface and
    the surrounding air stay at ambient. The result is a leading-hot/trailing-cold
    field, not a uniform bulk temperature.

    Args:
        radius_m:      sphere radius (m).
        material_key:  field-library material (→ α = k/ρcp).
        windward_T:    stagnation-point temperature (K) — the scenario's cited
                       drag-heating ``peak_T_K`` (a conservative floor; the true
                       recovery temperature is higher).
        spread_time_s: heating-phase duration the rind conducts over.
        velocity_dir:  body velocity through the air (default downward → leading
                       face is −y).
        T_ambient:     surrounding air / interior start temperature (K).
        n:             grid cells per axis.

    Returns a ThermalField.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from ..field.interface.thermal import (thermal_conductivity,
                                           heat_capacity_volumetric)
    from ..dynamics.fields.heat import stable_dt

    R = float(radius_m)
    half = R * 1.18                                   # a margin of void around it
    dx = 2.0 * half / n
    axis = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    rr = np.sqrt(X * X + Y * Y + Z * Z)
    solid = rr <= R
    if not solid.any():                               # grid too coarse for this R
        solid[n // 2, n // 2, n // 2] = True
    void = ~solid

    # material physics → diffusivity (grounded)
    k = float(thermal_conductivity(material_key, T=T_ambient))
    rhocp = float(heat_capacity_volumetric(material_key, T=T_ambient))
    alpha_val = k / rhocp if rhocp > 0 else 0.0
    alpha = np.where(solid, alpha_val, 0.0)

    # windward surface, cosθ from the stagnation point (n̂ ≈ radial, dotted v̂)
    surface = solid & ~ndi.binary_erosion(solid)
    vhat = np.asarray(velocity_dir, float)
    vhat = vhat / (float(np.linalg.norm(vhat)) or 1.0)
    safe = np.where(rr > 0, rr, 1.0)
    cos = (X / safe) * vhat[0] + (Y / safe) * vhat[1] + (Z / safe) * vhat[2]
    held = surface & (cos > 0.05)                     # leading hemisphere
    if not held.any():
        held = surface

    # Dirichlet windward profile: stagnation at windward_T, ambient at the shoulder
    profile = np.full((n, n, n), float(T_ambient))
    profile[held] = T_ambient + (float(windward_T) - T_ambient) * np.clip(cos[held], 0.0, 1.0)

    # explicit Fourier conduction (∂T/∂t = α∇²T), re-imposing the held windward
    # profile and the ambient air each step. Same stencil as dynamics.fields.heat;
    # we inline it only because that solver holds ONE scalar Dirichlet value and we
    # need a per-cell windward PROFILE plus the ambient void.
    T = profile.copy()
    dt = stable_dt(alpha_val, dx)
    if not math.isfinite(dt) or dt <= 0 or spread_time_s <= 0:
        Tf = T
    else:
        nsteps = max(1, min(int(math.ceil(spread_time_s / dt)), 200000))
        dt = spread_time_s / nsteps
        inv_dx2 = 1.0 / (dx * dx)
        Tf = T
        for _ in range(nsteps):
            lap = ndi.laplace(Tf, mode="nearest") * inv_dx2
            Tf = Tf + dt * alpha * lap
            Tf[held] = profile[held]                  # sustained windward heating
            Tf[void] = float(T_ambient)               # air stays ambient

    lead = surface & (cos > 0.6)
    trail = surface & (cos < -0.6)
    windward_T_out = float(Tf[lead].max()) if lead.any() else float(Tf[solid].max())
    trailing_T_out = float(Tf[trail].mean()) if trail.any() else float(T_ambient)
    return ThermalField(Tf, dx, n, solid, T_ambient, windward_T_out, trailing_T_out,
                        radius_m=R)


def material_grids(label_grid, materials, *, T_ref=293.15):
    """Label grid + material-name list → (k, rho_cp, emissivity) per-cell grids.

    The bridge from a labeled voxel body (a synthetic grid, or a Deckard
    ``VoxelField``'s ``label_grid``/``materials``) to the pure-law solver
    ``dynamics.fields.heat.diffuse_fvm``: every property comes from
    ``field.interface.thermal`` per material. ``materials[0]`` must be the
    void ("air"/"void") → k=0, rho_cp=0 (inert cells).

    APPROXIMATION (flagged): k, ρ·cp and ε are evaluated once at ``T_ref`` and
    FROZEN for the run — no k(T) feedback during diffusion. First-order honest;
    the temperature-coupled property sweep is a later rung.
    """
    import numpy as np
    from ..field.interface.thermal import (thermal_conductivity,
                                           heat_capacity_volumetric, emissivity)
    label = np.asarray(label_grid)
    k = np.zeros(label.shape, dtype=float)
    rho_cp = np.zeros(label.shape, dtype=float)
    eps = np.zeros(label.shape, dtype=float)
    for idx, name in enumerate(materials):
        if idx == 0:
            continue                               # void: k=0, rho_cp=0 → inert
        cells = label == idx
        if not cells.any():
            continue
        k[cells] = float(thermal_conductivity(name, T=T_ref))
        rho_cp[cells] = float(heat_capacity_volumetric(name, T=T_ref))
        eps[cells] = float(emissivity(name, T=T_ref))
    return k, rho_cp, eps


def exposed_faces(solid_mask):
    """Void-adjacent faces per cell (0–6) — the radiating surface counter.

    Same 6-neighbour convention as deckard's free-surface scan (dx² per exposed
    face, stair-step area), so surface totals agree across the stack. Faces on
    the domain boundary are NOT counted — pad the grid with a void margin, as
    every builder here (and deckard's voxelizer) already does.
    """
    import numpy as np
    s = np.asarray(solid_mask, dtype=bool)
    n = np.zeros(s.shape, dtype=float)
    for ax in range(s.ndim):
        lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(s.ndim))
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(s.ndim))
        n[lo] += (s[lo] & ~s[hi])                  # solid's +ax face meets void
        n[hi] += (s[hi] & ~s[lo])                  # solid's −ax face meets void
    return n


def boundary_from_env(env, solid_mask, *, emissivity_grid=None):
    """Translate an atmosphere preset / ``physics_env`` into solver boundary kwargs.

    ``env`` is a preset name ("STP" | "IRT" | "ISM") or a dict carrying at least
    ``medium`` and ``temperature_k`` (a theater's ``physics_env`` qualifies —
    the SAME table stages the scene and bounds the physics, by design).

      air    → Dirichlet: the void held at ambient (diffuse_fvm's flagged
               well-stirred-ambient convection stand-in).
      vacuum → radiative-only: k_face to void is already 0 (harmonic mean with
               k_void=0), plus a Stefan-Boltzmann sink on every exposed face.
    """
    import numpy as np
    if isinstance(env, str):
        from ..field.interface.atmosphere import atmosphere_preset
        env = atmosphere_preset(env)
    T_amb = float(env.get("temperature_k", 293.15))
    solid = np.asarray(solid_mask, dtype=bool)
    if env.get("medium", "air") == "air":
        return {"ambient_mask": ~solid, "ambient_T": T_amb}
    return {"radiative": {
        "surface_faces": exposed_faces(solid),
        "emissivity": 1.0 if emissivity_grid is None else emissivity_grid,
        "T_env": T_amb}}


def evolve_contact_field(label_grid, materials, dx, T_init, *, total_time,
                         env, n_keyframes=8, T_ref=None):
    """Evolve a labeled multi-material body's temperature field and FREEZE
    keyframes — the per-cell orchestrator for "energy transfer between voxel
    interfaces".

    One function, any labeled grid: a synthetic cube-on-slab, or a Deckard
    ``VoxelField``'s (label_grid, materials, voxel_size) — same call, zero new
    physics. Properties come from ``material_grids`` (k, ρ·cp, ε per cell);
    boundaries from ``boundary_from_env`` (air → Dirichlet ambient; vacuum →
    Stefan-Boltzmann per exposed face); the law is
    ``dynamics.fields.heat.diffuse_fvm`` — harmonic-mean face conductivity, so
    the iron↔copper interface flux is the series-resistance physics, exactly.

    Returns {"times_s", "T_frames" (float32 grids, FROZEN), "E_frames_J",
             "radiated_J", "k_by_material", "ambient_T", "solid_mask"}.
    """
    import numpy as np
    from ..dynamics.fields.heat import diffuse_fvm, thermal_energy
    from ..field.interface.thermal import thermal_conductivity
    if isinstance(env, str):
        from ..field.interface.atmosphere import atmosphere_preset
        env = atmosphere_preset(env)
    label = np.asarray(label_grid)
    solid = label > 0
    T_ref = float(T_ref if T_ref is not None else env.get("temperature_k", 293.15))
    k, rho_cp, eps = material_grids(label, materials, T_ref=T_ref)
    bc = boundary_from_env(env, solid, emissivity_grid=eps)
    if n_keyframes < 2:
        raise ValueError("need at least 2 keyframes (start + end)")
    T = np.array(T_init, dtype=float)
    times = [total_time * i / (n_keyframes - 1) for i in range(n_keyframes)]
    frames = [T.astype(np.float32).copy()]
    energies = [thermal_energy(T, rho_cp, dx)]
    radiated = 0.0
    for t0, t1 in zip(times, times[1:]):
        T, info = diffuse_fvm(T, k, rho_cp, dx, total_time=t1 - t0, **bc)
        radiated += info["radiated_J"]
        frames.append(T.astype(np.float32).copy())
        energies.append(thermal_energy(T, rho_cp, dx))
    return {"times_s": times, "T_frames": frames, "E_frames_J": energies,
            "radiated_J": radiated, "ambient_T": float(env["temperature_k"]),
            "solid_mask": solid,
            "k_by_material": {m: float(thermal_conductivity(m, T=T_ref))
                              for m in materials[1:]}}


def summarize(field):
    """One-line provenance/summary of a ThermalField."""
    return (f"windward {field.windward_T:.0f} K vs trailing {field.trailing_T:.0f} K "
            f"(Fourier conduction, Newtonian cosθ deposit; "
            f"dynamics.fields.heat + field.interface.thermal)")


__all__ = ["ThermalField", "windward_heating_field", "material_grids",
           "exposed_faces", "boundary_from_env", "evolve_contact_field",
           "summarize"]
