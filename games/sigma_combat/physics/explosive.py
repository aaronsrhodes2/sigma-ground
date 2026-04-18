"""
Explosive physics — two modes:

1. Pure blast (HE, thermobaric):
   Hopkinson-Cranz scaled distance Z = r / W^(1/3)
   Sadovsky overpressure: P = (0.84/Z + 2.7/Z² + 7.2/Z³) MPa

2. Shaped charge / HEAT jet (weapon['shaped_charge'] = True):
   Munroe-effect copper jet at ~8 km/s.
   Penetration capacity = SHAPED_CHARGE_FACTOR * W^(1/3)  [m RHA-equiv]
   Per layer: resistance = thickness * (yield_stress / RHA_YIELD) * _HEAT_MULT
   Gaps disrupt jet coherence (2× their length in RHA-equiv).
"""
import math
from ..materials import get_material

TNT_ENERGY_J_PER_KG = 4.6e6   # J/kg
RHA_YIELD_PA        = 1.2e9   # reference yield of Rolled Homogeneous Armour steel
SHAPED_CHARGE_FACTOR = 0.35   # m RHA-equiv per kg^(1/3) TNT — calibrated to RPG-7/ATGM data

# Materials that disrupt/resist HEAT jets better than RHA-equiv predicts
_HEAT_MULT = {
    'ceramic_alumina':  4.0,   # brittle conoid fracture breaks jet coherence
    'tungsten':         2.5,   # very high density; jet erodes rapidly
    'depleted_uranium': 2.0,
}


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """Explosive interaction with one armor layer."""
    if weapon.get('shaped_charge', False):
        return _shaped_charge_interact(weapon, layer_material, thickness_m)
    return _blast_interact(weapon, layer_material, thickness_m)


def _shaped_charge_interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Munroe-effect HEAT jet — erosion model.

    Jet capacity starts at SHAPED_CHARGE_FACTOR * W^(1/3) meters of RHA-equivalent.
    Each layer consumes capacity = thickness * (yield / RHA_YIELD) * _HEAT_MULT.
    Air gaps cost 2× their length (disrupts jet coherence).
    """
    tnt_kg = weapon['tnt_kg']
    jet_cap = weapon.get('_jet_remaining_m',
                         SHAPED_CHARGE_FACTOR * tnt_kg ** (1.0 / 3.0))

    if jet_cap <= 0:
        return _no_effect(thickness_m, 'HEAT jet exhausted')

    # Gap: coherence disruption (not material resistance)
    if layer_material == 'gap':
        loss = thickness_m * 2.0
        w_out = dict(weapon)
        w_out['_jet_remaining_m'] = max(0.0, jet_cap - loss)
        return {
            'penetrated': True,
            'temperature_k': 300.0,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': 0.0,
            'notes': f'HEAT jet crosses gap, coherence loss {loss*1000:.0f}mm RHA-eq',
            'weapon_out': w_out,
        }

    target = get_material(layer_material)
    mult = _HEAT_MULT.get(layer_material, 1.0)
    rha_equiv = thickness_m * (target['yield_stress_pa'] / RHA_YIELD_PA) * mult

    if rha_equiv <= jet_cap:
        # Jet punches through
        w_out = dict(weapon)
        w_out['_jet_remaining_m'] = jet_cap - rha_equiv
        energy = tnt_kg * TNT_ENERGY_J_PER_KG * min(1.0, rha_equiv / (jet_cap + 1e-10))
        vol = math.pi * (0.005 ** 2) * thickness_m  # ~5mm jet diameter
        dT = energy / max(vol * target['heat_cap_vol'], 1e-30)
        return {
            'penetrated': True,
            'temperature_k': 300.0 + dT,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': energy,
            'notes': (f'HEAT jet: {rha_equiv*1000:.0f}mm RHA-eq consumed, '
                      f'{w_out["_jet_remaining_m"]*1000:.0f}mm remaining'),
            'weapon_out': w_out,
        }
    else:
        # Jet stops inside this layer
        depth = thickness_m * (jet_cap / max(rha_equiv, 1e-10))
        energy = tnt_kg * TNT_ENERGY_J_PER_KG * 0.8
        vol = math.pi * (0.005 ** 2) * max(depth, 1e-4)
        dT = energy / max(vol * target['heat_cap_vol'], 1e-30)
        return {
            'penetrated': False,
            'temperature_k': 300.0 + dT,
            'damage_depth_m': depth,
            'energy_deposited_j': energy,
            'notes': (f'HEAT jet stopped: {jet_cap*1000:.0f}mm capacity '
                      f'vs {rha_equiv*1000:.0f}mm needed'),
            'weapon_out': None,
        }


def _blast_interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """Original Hopkinson-Cranz blast model."""
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
