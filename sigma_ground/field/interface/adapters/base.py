"""SGPhysicsMixin -- the core sigma-ground adapter primitive.

Any class that inherits this mixin gains sg_ methods backed by SSBM
physics.  External library objects can be wrapped via SGAdapter.

Convention
----------
    sg_<method>     SSBM-patched version of an equivalent external method.
                    Uses DE440-measured GMs, CODATA constants, Forest-Ruth
                    integration, and SSBM EOS densities -- never the
                    external library's own approximations.

Why sg_ not override?
    External library code is NEVER modified.  Patching happens outward:
    callers migrate from lib.method() to sg.method() one call at a time.
    The external library remains available as a fallback and for comparison.

Example
-------
    # Before: astropy physics (CODATA, STP density, RK45 integrator)
    import astropy.constants as const
    GM = const.GM_sun.value

    # After: SSBM physics (DE440, EOS density, Forest-Ruth)
    from sigma_ground.field.interface.adapters import SGConstants
    GM = SGConstants.GM_sun
"""

from __future__ import annotations

import math
from typing import Any


class SGPhysicsMixin:
    """Mixin adding SSBM-physics sg_ methods to any class.

    Inherit alongside any class (including external library classes) to
    add SSBM-patched alternatives.  All sg_ methods are self-contained;
    they do not call super() and do not depend on the mixed-in class.
    """

    # ── gravitational parameters ─────────────────────────────────────────

    def sg_gm(self, body_name: str) -> float:
        """GM in m³/s² from DE440 (names are case-insensitive).

        Source: JPL SSD Astrodynamic Parameters (astro_par), from Park et al.
        2021, AJ 161, 105 -- orbital.DE440_GM.  Mars..Pluto are SYSTEM GMs
        (planet + satellites), the mass a heliocentric orbit feels; the
        planet-only GMs (central mass of a satellite orbit) are in
        orbital.PLANET_GM.  FIXED 2026-10-10: Sun and Earth were DE430
        values, Mercury a pre-DE430 value (+10 ppm), and Mars..Pluto
        planet-only GMs (Pluto -10.9 %), all labelled DE440.
        """
        from sigma_ground.field.interface.orbital import DE440_GM
        by_lower = {k.lower(): v for k, v in DE440_GM.items()}
        km3 = by_lower.get(str(body_name).lower())
        if km3 is None:
            raise KeyError(
                f"No DE440 GM for {body_name!r}. "
                f"Available: {list(DE440_GM.keys())}"
            )
        return km3 * 1e9  # km³/s² → m³/s²

    def sg_mass(self, body_name: str) -> float:
        """Mass in kg derived from DE440 GM and CODATA G.

        m = GM / G  — both from measured sources, not approximations.
        """
        from sigma_ground.field.constants import G
        return self.sg_gm(body_name) / G

    # ── density ─────────────────────────────────────────────────────────

    def sg_density(self, material_name: str, T_K: float = 293.15,
                   P_Pa: float = 101_325.0) -> float:
        """Density in kg/m³ of a solid at temperature T_K and pressure P_Pa.

        d ln rho = -beta dT + dP/K, integrated with constant coefficients:

            rho = rho_0 * exp(-3 alpha (T - T_0) + (P - P_0) / K)

        rho_0: measured density (surface.MATERIALS) at T_0 = 293.15 K
        (water_ice: 273.15 K) and P_0 = 1 atm; alpha: measured linear
        expansion coefficient at 300 K (thermal_expansion.EXPANSION_DATA);
        K: the library's bulk modulus (thermal_expansion.bulk_modulus).
        Bound: holding alpha and K constant errs by < ~0.5 % for
        |T - T_0| <= 300 K and |P - P_0| <= 1 GPa in metals (alpha(T)
        follows the Debye heat capacity; K' ~ 5).  Solid-solid phase
        changes are not modelled; above the melting point
        (phase_transition.PHASE_DATA) a ValueError is raised, since no
        liquid equation of state is available.
        FIXED 2026-10-10: both import targets of the previous version were
        missing, so every call raised ImportError.

        Parameters
        ----------
        material_name : MATERIALS key ('iron', 'water_ice') or element symbol
                        ('Fe', 'Cu'); case-insensitive
        T_K : temperature in Kelvin
        P_Pa : pressure in Pascals
        """
        from sigma_ground.field.interface.surface import MATERIALS
        from sigma_ground.field.interface import thermal_expansion as _te
        from sigma_ground.field.interface.phase_transition import PHASE_DATA

        symbols = {
            "fe": "iron", "cu": "copper", "al": "aluminum", "au": "gold",
            "si": "silicon", "w": "tungsten", "ni": "nickel", "ti": "titanium",
            "pb": "lead", "ag": "silver", "zn": "zinc", "pt": "platinum",
        }
        name = str(material_name).strip().lower()
        key = symbols.get(name, name)
        if key not in MATERIALS:
            raise KeyError(
                f"No density data for {material_name!r}. Known: "
                f"{sorted(MATERIALS)} or symbols {sorted(s.capitalize() for s in symbols)}"
            )
        rho_0 = float(MATERIALS[key]["density_kg_m3"])
        T_0 = 273.15 if key == "water_ice" else 293.15
        P_0 = 101_325.0
        T_K, P_Pa = float(T_K), float(P_Pa)
        if not (math.isfinite(T_K) and T_K > 0 and math.isfinite(P_Pa)):
            raise ValueError(f"need T_K > 0 and finite P_Pa, got {T_K}, {P_Pa}")
        T_melt = PHASE_DATA.get(key, {}).get("T_melt_K")
        if T_melt is not None and T_melt >= T_0 and T_K > T_melt:
            raise ValueError(
                f"{key} melts at {T_melt} K; T_K = {T_K} K is above it and no "
                f"liquid equation of state is available"
            )
        if T_K == T_0 and P_Pa == P_0:
            return rho_0
        alpha = _te.EXPANSION_DATA[key]["alpha_linear_per_K"]
        K = _te.bulk_modulus(key, 0.0)
        return rho_0 * math.exp(-3.0 * alpha * (T_K - T_0) + (P_Pa - P_0) / K)

    # ── integration ───────────────────────────────────────────────────────

    def sg_propagate_orbit(
        self,
        bodies: list,
        dt_s: float,
        n_steps: int,
        method: str = "forest_ruth",
        include_gr: bool = False,
    ) -> list:
        """Propagate N bodies using Forest-Ruth (default) or Verlet.

        Drop-in replacement for scipy.integrate orbit propagators, using
        the SSBM 4th-order symplectic integrator.

        Parameters
        ----------
        bodies : list of CelestialBody from sigma_ground.field.interface.nbody
        dt_s   : fixed timestep (seconds) -- do NOT vary; symplecticity requires fixed dt
        n_steps : number of steps
        method  : 'forest_ruth' (4th-order) or 'verlet' (2nd-order)
        include_gr : opt in to the 1PN Schwarzschild correction (nbody gr_1pn; a
                     single-body form, valid in the attractor's rest frame).
                     Default False since 2026-10-10: plain Newtonian, like the
                     scipy propagators this replaces.

        Returns
        -------
        list of CelestialBody at t0 + n_steps * dt_s
        """
        from sigma_ground.field.interface.nbody import NBodySystem
        if method not in ("forest_ruth", "verlet"):
            raise ValueError(
                f"Unknown method {method!r}. Choose 'forest_ruth' or 'verlet'."
            )
        system  = NBodySystem(bodies, include_gr=include_gr)
        step_fn = (system.forest_ruth_step if method == "forest_ruth"
                   else system.step)
        for _ in range(n_steps):
            step_fn(dt_s)
        return system.bodies


class SGAdapter:
    """Wraps any external library object, adding sg_ methods from SGPhysicsMixin.

    The wrapped object is accessible via .lib for non-patched operations;
    all sg_ calls use SSBM physics regardless of what the external library
    would return.

    Parameters
    ----------
    obj : external library object to wrap (astropy Body, jplephem kernel, etc.)

    Example
    -------
        from astropy.coordinates import get_body_barycentric
        from sigma_ground.field.interface.adapters import SGAdapter

        astropy_sun = get_body_barycentric('sun', time)
        wrapped = SGAdapter(astropy_sun)
        wrapped.sg_gm('Sun')          # DE440, not CODATA
        wrapped.lib.xyz               # fall through to astropy
    """

    def __init__(self, obj: Any) -> None:
        self.lib = obj  # unmodified external object

    def __getattr__(self, name: str) -> Any:
        # 'lib' itself is missing on an instance rebuilt without __init__
        # (copy, pickle); looking it up via self.lib would recurse forever.
        if name == "lib":
            raise AttributeError(name)
        if name.startswith("sg_"):
            raise AttributeError(
                f"SGAdapter has no SSBM method {name!r}. "
                f"Did you mean to add it to SGPhysicsMixin?"
            )
        return getattr(self.lib, name)

    # Inject all SGPhysicsMixin methods
    sg_gm             = SGPhysicsMixin.sg_gm
    sg_mass           = SGPhysicsMixin.sg_mass
    sg_density        = SGPhysicsMixin.sg_density
    sg_propagate_orbit = SGPhysicsMixin.sg_propagate_orbit
