"""
Environment — the sandbox conditions for a simulation.

Defines:
  Medium      — what the simulation takes place in (air, water, vacuum)
  GroundConfig — ground plane parameters
  BoundaryPlane — additional walls/ceilings
  LightSource — light source for optical calculations
  Environment — the complete environmental context

All values are in SI units. No physics imports at module level.
"""


class Medium:
    """The medium the simulation takes place in.

    Args:
        name:        Identifier ("air", "water", "vacuum").
        density:     Medium density in kg/m^3.
        viscosity:   Dynamic viscosity in Pa*s.
        temperature: Medium temperature in K.
    """

    def __init__(self, name='air', density=1.225, viscosity=1.81e-5,
                 temperature=293.15):
        self.name = name
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.temperature = float(temperature)

    @classmethod
    def air(cls, T=293.15, P=101325.0):
        """Air at given temperature and pressure.

        Uses sigma-ground's atmosphere module for derived properties.
        """
        from ..field.interface.atmosphere import air_density
        rho = air_density(T, P)
        # Sutherland's law for air viscosity
        # MEASURED: Sutherland constant S = 110.4 K, ref viscosity at 291.15 K
        mu_ref = 1.827e-5  # Pa*s at 291.15 K
        T_ref = 291.15
        S = 110.4
        mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
        return cls(name='air', density=rho, viscosity=mu, temperature=T)

    @classmethod
    def water(cls, T=293.15):
        """Water at given temperature.

        Uses sigma-ground's fluid module where available.
        """
        # Water density: roughly 998 kg/m^3 at 20C
        # Slight temperature dependence (quadratic around 4C max)
        rho = 998.2 - 0.06 * (T - 293.15)  # linear approx near 20C
        try:
            from ..field.interface.fluid import liquid_viscosity
            mu = liquid_viscosity('water', T)
        except (ImportError, KeyError):
            # Fallback: Arrhenius-type for water
            import math
            mu = 1.002e-3 * math.exp(1808 * (1/T - 1/293.15))
        return cls(name='water', density=rho, viscosity=mu, temperature=T)

    @classmethod
    def vacuum(cls):
        """Hard vacuum — no medium interactions."""
        return cls(name='vacuum', density=0.0, viscosity=0.0, temperature=2.725)

    def __repr__(self):
        return (f"Medium({self.name!r}, rho={self.density:.3f} kg/m³, "
                f"mu={self.viscosity:.2e} Pa·s)")


class GroundConfig:
    """Ground plane configuration.

    Args:
        enabled:      Whether the ground exists.
        height:       Ground height in meters (default: 0).
        material_key: Material of the ground surface (for friction/restitution).
        normal:       Outward normal direction as (nx, ny, nz).
                      Default: (0, 1, 0) = Y-up.
    """

    def __init__(self, enabled=True, height=0.0, material_key='concrete',
                 normal=(0, 1, 0)):
        self.enabled = enabled
        self.height = float(height)
        self.material_key = material_key
        self.normal = tuple(normal)

    def __repr__(self):
        return (f"GroundConfig(y={self.height}, "
                f"material={self.material_key!r})")


class BoundaryPlane:
    """An additional boundary surface (wall, ceiling).

    Args:
        point:        A point on the plane (x, y, z) in meters.
        normal:       Outward normal (nx, ny, nz) pointing into the scene.
        material_key: Material of the boundary surface.
    """

    def __init__(self, point, normal, material_key='steel_mild'):
        self.point = tuple(point)
        self.normal = tuple(normal)
        self.material_key = material_key

    def __repr__(self):
        return (f"BoundaryPlane(point={self.point}, "
                f"normal={self.normal})")


class LightSource:
    """A light source for optical/thermal calculations.

    Args:
        position:         Position (x, y, z) in meters.
        intensity:        Power in W/m^2 at 1 meter from source.
        wavelength_range: (min_wavelength, max_wavelength) in meters.
                          Default: visible range (380-780 nm).
                          Single wavelength (laser): set min == max.
                          UV: (10e-9, 380e-9)
                          IR: (780e-9, 1e-3)
                          Full spectrum: (10e-9, 1e-3)
        color_temperature: Optional color temperature in K (for broadband).
                          If set, overrides wavelength_range for rendering.
    """

    def __init__(self, position, intensity=1.0,
                 wavelength_range=(380e-9, 780e-9),
                 color_temperature=None):
        self.position = tuple(position)
        self.intensity = float(intensity)
        self.wavelength_range = tuple(wavelength_range)
        self.color_temperature = color_temperature

    @property
    def is_monochromatic(self):
        """True if this is a single-wavelength (laser-like) source."""
        lo, hi = self.wavelength_range
        return abs(hi - lo) < 1e-12

    @property
    def is_visible(self):
        """True if wavelength range overlaps visible spectrum."""
        lo, hi = self.wavelength_range
        return lo < 780e-9 and hi > 380e-9

    def __repr__(self):
        lo_nm = self.wavelength_range[0] * 1e9
        hi_nm = self.wavelength_range[1] * 1e9
        if self.is_monochromatic:
            return f"LightSource(laser, {lo_nm:.0f}nm, {self.intensity}W/m²)"
        return (f"LightSource({lo_nm:.0f}-{hi_nm:.0f}nm, "
                f"{self.intensity}W/m²)")


class Environment:
    """Complete environmental context for a simulation.

    Args:
        gravity:         Gravitational acceleration as (gx, gy, gz) in m/s^2.
                         Default: Earth surface gravity, Y-down.
        medium:          Medium instance (air, water, vacuum).
        ground:          GroundConfig instance.
        boundaries:      List of BoundaryPlane instances.
        light_sources:   List of LightSource instances.
        reference_frame: Coordinate frame ("lab" or "com" for center-of-mass).
    """

    def __init__(self, gravity=(0, -9.80665, 0), medium=None,
                 ground=None, boundaries=None, light_sources=None,
                 reference_frame='lab'):
        self.gravity = tuple(gravity)
        self.medium = medium if medium is not None else Medium()
        self.ground = ground if ground is not None else GroundConfig()
        self.boundaries = list(boundaries) if boundaries else []
        self.light_sources = list(light_sources) if light_sources else []
        self.reference_frame = reference_frame

    def __repr__(self):
        g_mag = sum(c**2 for c in self.gravity) ** 0.5
        return (f"Environment(g={g_mag:.2f}m/s², "
                f"medium={self.medium.name}, "
                f"ground={'on' if self.ground.enabled else 'off'})")
