"""
Plasma torch physics — thermal flux + pressure wave per layer.

Plasma at temperature T_eV deposits thermal energy:
    q = n * k_B * T * v_plasma  [W/m²]

Simplified for the game: we work with total energy budget =
    temperature_ev * k_B * pressure_pa * contact_area * duration_s

Bond failure check: local sigma rises as material heats toward melting.
"""
import math
from ..materials import get_material, MELTING_POINTS_K
from sigma_ground.field.constants import K_B, SIGMA_CONV

EV_TO_J = 1.602176634e-19


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Plasma interaction with one armor layer.

    weapon keys: temperature_ev, pressure_pa, duration_s, contact_radius_m
    """
    T_ev = weapon['temperature_ev']
    P = weapon['pressure_pa']
    duration = weapon.get('duration_s', 0.1)
    contact_r = weapon.get('contact_radius_m', 0.05)

    if T_ev <= 0 or P <= 0:
        return _no_effect(thickness_m, 'No plasma')

    target = get_material(layer_material)
    area = math.pi * contact_r * contact_r

    # Thermal energy flux: q = P * v_thermal where v ~ sqrt(k_B*T/m_ion)
    # Simplified: total energy = plasma_pressure * area * duration
    # (dimensional: Pa * m² * s = N/m² * m² * s = N*s = J... no, Pa*m²*s = J/m * s)
    # Better: thermal flux q ~ 0.5 * n * k_B * T * <v> ≈ P * sqrt(k_B*T / (2*pi*m_proton))
    # For game: just use energy_budget = T_ev[eV] * P[Pa] * area * duration * scale_factor
    m_proton = 1.673e-27
    v_thermal = math.sqrt(2 * T_ev * EV_TO_J / m_proton)
    energy_flux = 0.5 * P * v_thermal  # W/m²
    total_energy = energy_flux * area * duration

    # Heat the near-surface volume (plasma doesn't penetrate deep, absorbed at surface)
    # Effective depth: thermal diffusion length = sqrt(kappa * t) where kappa = k/(rho*cp)
    kappa = target['thermal_cond'] / (target['density_kg_m3'] * target['heat_cap_vol'] / target['density_kg_m3'])
    diff_depth = math.sqrt(max(kappa * duration, 1e-12))
    vol = area * min(thickness_m, diff_depth)

    delta_T = total_energy / max(vol * target['heat_cap_vol'], 1e-30)
    temperature_k = 300.0 + delta_T
    melting_k = target['melting_point_k']

    # Pressure wave: compare to bulk modulus
    bulk = target['bulk_modulus_pa']
    yield_s = target['yield_stress_pa']
    pressure_failed = P > yield_s * 3  # 3× yield = catastrophic

    if pressure_failed or temperature_k >= melting_k:
        # Layer fails — plasma punches through (weakened)
        frac_absorbed = min(0.9, thickness_m / max(diff_depth, 1e-10))
        w_out = dict(weapon)
        w_out['temperature_ev'] *= (1.0 - frac_absorbed)
        w_out['pressure_pa'] *= (1.0 - frac_absorbed)
        reason = 'pressure failure' if pressure_failed else f'melted at {temperature_k:.0f}K'
        return {
            'penetrated': True,
            'temperature_k': temperature_k,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': total_energy * frac_absorbed,
            'notes': f'PENETRATED via {reason}',
            'weapon_out': w_out,
        }

    damage_depth = min(thickness_m, diff_depth * (temperature_k - 300) / max(melting_k - 300, 1))
    return {
        'penetrated': False,
        'temperature_k': temperature_k,
        'damage_depth_m': damage_depth,
        'energy_deposited_j': total_energy,
        'notes': f'Heated to {temperature_k:.0f}K (melt={melting_k:.0f}K), P={P:.2e}Pa vs Y={yield_s:.2e}Pa',
        'weapon_out': None,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'notes': notes, 'weapon_out': None,
    }
