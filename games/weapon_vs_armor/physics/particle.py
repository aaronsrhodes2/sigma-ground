"""
Particle beam physics — relativistic penetration via Bethe-Bloch stopping power.

Simplified Bethe stopping power:
    -dE/dx ≈ K * Z_target * rho / (A * beta^2) * [ln(2*m_e*c^2*beta^2*gamma^2/I) - beta^2]

For the game: approximate as stopping_power ∝ rho * (Z/A) / beta^2
Range ≈ E_MeV / (rho * k_material)  [g/cm^2 → m]

Particle types: proton, neutron, electron
- Neutrons: no charge, much longer range (nuclear stopping only, ~10× proton range)
- Electrons: light, scatter easily, shorter range than protons at same energy
- Protons: standard
"""
import math
from ..materials import get_material

C_LIGHT = 299_792_458.0
M_PROTON_MEV = 938.272  # MeV/c^2
M_ELECTRON_MEV = 0.511  # MeV/c^2

# Approximate stopping coefficient k [MeV cm^2/g] at 100 MeV (Bethe plateau)
_K_STOP = {
    'proton':   2.0,
    'electron': 1.8,
    'neutron':  0.2,  # nuclear stopping only, much longer range
}


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Particle beam interaction with one armor layer.

    weapon keys: energy_mev, particle ('proton'/'neutron'/'electron'), current_a
    """
    E_mev = weapon['energy_mev']
    particle = weapon.get('particle', 'proton')
    current_a = weapon.get('current_a', 1e-6)

    if E_mev <= 0:
        return _no_effect(thickness_m, 'No beam energy')

    target = get_material(layer_material)
    rho = target['density_kg_m3']
    rho_gcm3 = rho / 1000.0

    k = _K_STOP[particle]
    # Range in g/cm² ≈ E_mev / (k * rho_gcm3) ... (simplified Bragg-Kleeman scaling)
    # More accurate: range scales roughly as E^1.77 for protons
    m_particle = M_PROTON_MEV if particle in ('proton', 'neutron') else M_ELECTRON_MEV
    beta2 = 1.0 - (m_particle / (m_particle + E_mev)) ** 2
    beta = math.sqrt(max(beta2, 1e-10))

    # Range (CSDA approx): R [g/cm²] ≈ E / (k * rho_gcm3 / beta^2 * correction)
    R_g_cm2 = E_mev * beta**2 / (k * rho_gcm3 + 0.001)
    range_m = R_g_cm2 / (rho_gcm3 * 100)  # convert to meters

    thickness_g_cm2 = rho_gcm3 * thickness_m * 100  # convert m → g/cm²

    # Energy loss in this layer (Bragg curve approximation)
    if range_m <= 1e-9:
        E_remaining = 0.0
        energy_deposited_j = E_mev * 1.602e-13  # full deposition
    elif range_m < thickness_m:
        # Particle stops inside this layer (Bragg peak)
        frac_to_stop = range_m / thickness_m
        E_remaining = 0.0
        energy_deposited_j = E_mev * 1.602e-13
    else:
        # Particle transits
        frac_deposited = thickness_g_cm2 / max(R_g_cm2, 1e-6)
        energy_deposited_j = E_mev * 1.602e-13 * frac_deposited
        E_remaining = E_mev * (1.0 - frac_deposited)

    # Beam power = current [A] * energy [V equivalent]
    # For dose: multiply by current / charge_per_particle
    E_per_particle = E_mev * 1.602e-13  # J
    e_charge = 1.602e-19
    particles_per_sec = current_a / e_charge
    power_deposited = energy_deposited_j * particles_per_sec  # W (per second of irradiation)

    duration = weapon.get('duration_s', 0.5)
    total_energy = power_deposited * duration

    # Heat deposition in the Bragg peak volume
    vol = math.pi * (weapon.get('beam_radius_m', 0.005)) ** 2 * min(thickness_m, range_m)
    delta_T = total_energy / max(vol * target['heat_cap_vol'], 1e-30)
    temperature_k = 300.0 + delta_T
    melting_k = target['melting_point_k']

    penetrated = E_remaining > 0

    if penetrated:
        w_out = dict(weapon)
        w_out['energy_mev'] = E_remaining
        return {
            'penetrated': True,
            'temperature_k': temperature_k,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': total_energy,
            'notes': f'{particle} beam: range={range_m:.3f}m, E_rem={E_remaining:.1f}MeV, T={temperature_k:.0f}K',
            'weapon_out': w_out,
        }

    return {
        'penetrated': False,
        'temperature_k': temperature_k,
        'damage_depth_m': min(range_m, thickness_m),
        'energy_deposited_j': total_energy,
        'notes': f'{particle} stopped: range={range_m:.3f}m < {thickness_m:.3f}m, T={temperature_k:.0f}K',
        'weapon_out': None,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'notes': notes, 'weapon_out': None,
    }
