"""
Medium attenuation — applies environmental effects to weapon params before impact.
"""
import math

ENVIRONMENTS = {
    'vacuum': {
        'label': 'Vacuum (space)',
        'density_kg_m3': 1e-20,
        'kinetic_drag_coeff': 0.0,
        'laser_abs_coeff_per_m': 0.0,
        'plasma_dispersal': 0.0,
        'description': 'Zero attenuation. What you fire is what hits.',
    },
    'atmosphere_earth': {
        'label': 'Earth Atmosphere',
        'density_kg_m3': 1.225,
        'kinetic_drag_coeff': 0.47,
        'laser_abs_coeff_per_m': 0.003,   # ~0.3%/m for visible, much worse for IR
        'plasma_dispersal': 0.15,
        'description': 'Drag, thermal blooming, plasma dispersal.',
    },
    'atmosphere_venus': {
        'label': 'Dense Atmosphere (Venus)',
        'density_kg_m3': 65.0,
        'kinetic_drag_coeff': 0.47,
        'laser_abs_coeff_per_m': 0.08,
        'plasma_dispersal': 0.6,
        'description': 'Extreme drag. Lasers degrade fast. Plasma scattered.',
    },
    'underwater': {
        'label': 'Underwater',
        'density_kg_m3': 1000.0,
        'kinetic_drag_coeff': 0.47,
        'laser_abs_coeff_per_m': 0.07,    # ~10% per meter for visible
        'plasma_dispersal': 0.99,
        'description': 'Kinetic drag dominant. Lasers absorbed in meters. Plasma quenched instantly.',
    },
    'ism': {
        'label': 'Interstellar Medium',
        'density_kg_m3': 1.7e-21,
        'kinetic_drag_coeff': 0.0,
        'laser_abs_coeff_per_m': 0.0,
        'plasma_dispersal': 0.0,
        'description': 'Effectively vacuum at combat distances.',
    },
}


def attenuate(weapon: dict, environment: str, distance_m: float) -> dict:
    """Return a modified copy of weapon params after traversing the medium."""
    env = ENVIRONMENTS[environment]
    w = dict(weapon)

    if distance_m <= 0:
        return w

    wtype = weapon['type']

    if wtype == 'kinetic':
        # Drag: v_final from energy loss to aero drag
        v0 = weapon['velocity_ms']
        rho = env['density_kg_m3']
        Cd = env['kinetic_drag_coeff']
        r = weapon.get('radius_m', 0.01)
        m = weapon['mass_kg']
        A = math.pi * r * r
        # Simple deceleration via drag work: KE_loss = 0.5 * Cd * rho * A * v² * distance
        # Approximate: v_f = v0 * exp(-k * d) where k = 0.5 * Cd * rho * A / m
        k = 0.5 * Cd * rho * A / max(m, 1e-30)
        v_final = v0 * math.exp(-k * distance_m)
        w['velocity_ms'] = max(v_final, 0.0)

    elif wtype == 'laser':
        # Beer-Lambert through medium
        alpha = env['laser_abs_coeff_per_m']
        power_fraction = math.exp(-alpha * distance_m)
        w['power_w'] = weapon['power_w'] * power_fraction

    elif wtype == 'plasma':
        # Disperse by 1/r² plus medium quench factor
        quench = 1.0 - env['plasma_dispersal']
        inv_sq = 1.0 / max(1.0, distance_m ** 2)
        # Normalize: at 1m it's full, then falls off
        factor = quench * min(1.0, inv_sq * 1.0)
        w['temperature_ev'] = weapon['temperature_ev'] * max(factor, 0.01)
        w['pressure_pa'] = weapon['pressure_pa'] * max(factor, 0.01)

    elif wtype == 'explosive':
        # Standoff is baked into the explosive model itself
        pass

    elif wtype == 'particle':
        # Near-vacuum: negligible attenuation at combat distances
        rho = env['density_kg_m3']
        # Stopping power ∝ rho * distance (Bethe regime)
        # Only matters in atmosphere/water at >100m
        stopping_fraction = rho * distance_m * 1e-4  # rough
        w['energy_mev'] = weapon['energy_mev'] * max(0.01, 1.0 - stopping_fraction)

    elif wtype == 'gravitational':
        # Gravitational effects don't attenuate through matter
        pass

    return w
