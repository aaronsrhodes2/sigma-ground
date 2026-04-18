"""
Laser physics — Beer-Lambert absorption + thermal heating per layer.

Absorption coefficients (alpha, m^-1) are estimated from material class:
- Metals: ~1e7 (skin depth ~100nm, essentially opaque)
- Ceramics: ~1e4 to 1e5 (partially transparent in IR)
- Polymers/rubber: ~1e3 to 1e4
- Glass/ice: ~1e1 to 1e3 (depends on wavelength)

If absorbed energy heats the layer above melting point: layer is ablated.
"""
import math
from ..materials import get_material, MELTING_POINTS_K

# Absorption coefficients per material (m^-1)
_ALPHA = {
    'iron': 7e7, 'copper': 6e7, 'aluminum': 1.5e8, 'gold': 8e7,
    'silicon': 1e6, 'tungsten': 5e7, 'nickel': 7e7, 'titanium': 5e7,
    'steel_mild': 7e7, 'lead': 8e7, 'silver': 1e8, 'platinum': 5e7,
    'depleted_uranium': 6e7, 'rubber': 2e4, 'plastic_abs': 1e4,
    'glass': 100, 'concrete': 5e4, 'granite': 3e4, 'ceramic_alumina': 2e5,
    'water_ice': 10, 'wood_oak': 5e4, 'bone': 2e4,
    'carbon_fiber': 4e7, 'kevlar': 1e4,
}

# Wavelength correction — metals reflect well in IR, absorb better in UV
# laser_wavelength_nm < 400 (UV): alpha * 2
# laser_wavelength_nm > 1000 (IR): alpha * 0.3 for metals
def _effective_alpha(mat_name: str, wavelength_nm: float) -> float:
    base = _ALPHA.get(mat_name, 1e6)
    mat = get_material(mat_name)
    is_metal = mat['density_kg_m3'] > 2000 and mat['melting_point_k'] > 500
    if is_metal:
        if wavelength_nm < 400:
            return base * 2.0
        elif wavelength_nm > 1000:
            return base * 0.3  # metals reflect IR better
    return base


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Laser interaction with one armor layer.

    weapon keys: type, power_w, wavelength_nm, pulse_duration_s, spot_radius_m
    Returns: penetrated, temperature_k, damage_depth_m, energy_deposited_j,
             ablated (bool), weapon_out
    """
    power = weapon['power_w']
    wavelength = weapon.get('wavelength_nm', 1064.0)
    duration = weapon.get('pulse_duration_s', 1.0)
    spot_r = weapon.get('spot_radius_m', 0.01)

    if power <= 0:
        return _no_effect(thickness_m, 'No laser power')

    target = get_material(layer_material)
    alpha = _effective_alpha(layer_material, wavelength)
    spot_area = math.pi * spot_r * spot_r

    # Total energy in pulse
    total_energy_j = power * duration

    # Beer-Lambert: energy absorbed in this layer
    # I(x) = I0 * exp(-alpha * x)
    # Absorbed fraction: 1 - exp(-alpha * L)
    absorbed_fraction = 1.0 - math.exp(-alpha * thickness_m)
    energy_absorbed = total_energy_j * absorbed_fraction
    energy_transmitted = total_energy_j - energy_absorbed

    # Heat the irradiated volume
    # Volume = spot_area * effective_depth (min of thickness and 1/alpha)
    effective_depth = min(thickness_m, 1.0 / max(alpha, 1.0))
    volume = spot_area * effective_depth
    heat_cap = target['heat_cap_vol']

    delta_T = energy_absorbed / max(volume * heat_cap, 1e-30)
    temperature_k = 300.0 + delta_T
    melting_k = target['melting_point_k']

    # Thermal conduction into adjacent material reduces peak temperature
    cond = target['thermal_cond']  # W/(m·K)
    # Simple: conduction loss fraction ≈ cond * duration / (rho * Cp * thickness^2)
    cond_loss_frac = cond * duration / max(heat_cap / target['density_kg_m3'] * thickness_m ** 2, 1e-30)
    cond_loss_frac = min(0.9, cond_loss_frac)
    temperature_k = 300.0 + delta_T * (1.0 - cond_loss_frac)

    ablated = temperature_k >= melting_k * 1.2

    if ablated:
        # Layer vaporized — full penetration, laser exits weakened
        power_out = power * (1.0 - absorbed_fraction)
        w_out = dict(weapon)
        w_out['power_w'] = power_out
        return {
            'penetrated': True,
            'temperature_k': temperature_k,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': energy_absorbed,
            'ablated': True,
            'notes': f'ABLATED at {temperature_k:.0f}K (melt={melting_k:.0f}K), abs={alpha:.1e}/m',
            'weapon_out': w_out,
        }

    penetrated = temperature_k >= melting_k

    if penetrated:
        w_out = dict(weapon)
        w_out['power_w'] = power * (1.0 - absorbed_fraction)
        damage_depth = thickness_m
        verdict = f'melted through at {temperature_k:.0f}K'
    else:
        w_out = None
        damage_depth = effective_depth * (temperature_k - 300) / max(melting_k - 300, 1)
        verdict = f'heated to {temperature_k:.0f}K (melt={melting_k:.0f}K)'

    return {
        'penetrated': penetrated,
        'temperature_k': temperature_k,
        'damage_depth_m': damage_depth,
        'energy_deposited_j': energy_absorbed,
        'ablated': False,
        'notes': verdict,
        'weapon_out': w_out,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'ablated': False, 'notes': notes, 'weapon_out': None,
    }
