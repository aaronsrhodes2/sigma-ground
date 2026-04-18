"""
Explosive physics — Hopkinson-Cranz blast wave + thermal pulse.

Kingery-Bulmash scaled distance: Z = r / W^(1/3)  [m/kg^(1/3)]
Peak overpressure (simplified Sadovsky):
    P_peak = (0.84/Z + 2.7/Z^2 + 7.2/Z^3) MPa  for Z in [0.5, 40]

At Z < 0.1 (contact): catastrophic destruction.
"""
import math
from ..materials import get_material

TNT_ENERGY_J_PER_KG = 4.6e6  # J/kg


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Explosive interaction with one armor layer.

    weapon keys: tnt_kg, standoff_m (0 = contact), detonation_face (0=first layer)
    """
    tnt_kg = weapon['tnt_kg']
    standoff = weapon.get('standoff_m', 0.0)

    if tnt_kg <= 0:
        return _no_effect(thickness_m, 'No explosive')

    target = get_material(layer_material)

    # Total energy
    total_energy = tnt_kg * TNT_ENERGY_J_PER_KG

    # Effective distance includes standoff + accumulated armor thickness already traversed
    dist = standoff + weapon.get('_depth_traversed_m', 0.0)

    # Scaled distance
    W_third = tnt_kg ** (1.0 / 3.0)
    Z = (dist + 0.001) / W_third  # avoid div by 0

    # Peak overpressure (MPa)
    if Z < 0.1:
        P_pa = 1e9  # near-contact: >1 GPa, total destruction
    elif Z < 40:
        P_mpa = 0.84 / Z + 2.7 / Z**2 + 7.2 / Z**3
        P_pa = P_mpa * 1e6
    else:
        P_pa = 1000.0  # ~1 kPa at large distance, negligible

    yield_s = target['yield_stress_pa']
    bulk = target['bulk_modulus_pa']

    # Thermal pulse: ~1/3 of explosive energy is thermal
    thermal_energy = total_energy * 0.33
    area_est = math.pi * (0.5) ** 2  # 0.5m blast radius estimate
    vol = area_est * min(thickness_m, 0.1)
    delta_T = thermal_energy / max(vol * target['heat_cap_vol'], 1e-30)
    temperature_k = 300.0 + delta_T
    melting_k = target['melting_point_k']

    # Failure: P > yield_stress * 3 or P > bulk_modulus * 0.01 (elastic limit)
    failed = P_pa > yield_s * 2.0

    if failed or temperature_k >= melting_k:
        w_out = dict(weapon)
        # Attenuate blast after penetration
        w_out['tnt_kg'] = tnt_kg * 0.4
        w_out['_depth_traversed_m'] = dist + thickness_m
        reason = f'blast P={P_pa:.2e}Pa > 2×yield={yield_s*2:.2e}Pa' if failed else f'thermal {temperature_k:.0f}K'
        return {
            'penetrated': True,
            'temperature_k': temperature_k,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': total_energy * 0.5,
            'notes': f'BLAST THROUGH: {reason}, Z={Z:.2f}',
            'weapon_out': w_out,
        }

    damage_depth = thickness_m * min(1.0, P_pa / (yield_s * 2.0))
    return {
        'penetrated': False,
        'temperature_k': temperature_k,
        'damage_depth_m': damage_depth,
        'energy_deposited_j': total_energy * 0.3,
        'notes': f'Blast contained: P={P_pa:.2e}Pa, Z={Z:.2f}, T={temperature_k:.0f}K',
        'weapon_out': None,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'notes': notes, 'weapon_out': None,
    }
