"""
Gravitational / exotic weapon — tidal force spike + sigma-field disruptor.

Tidal force: ΔF = 2 G M m dr / r^3
Applied as differential stress across armor thickness: sigma_tidal = 2GM*rho*dr/r^3

Sigma-field spike: locally elevates sigma toward SIGMA_CONV, weakening bond
strength. At sigma = SIGMA_CONV, yield_stress → 0.
"""
import math
from ..materials import get_material
from sigma_ground.field.constants import G, SIGMA_CONV
from sigma_ground.field.interface.hardness import yield_stress as sg_yield_stress

G_NEWT = G  # 6.674e-11


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Exotic weapon interaction with one armor layer.

    weapon keys:
      tidal_gradient_ms2  — differential gravity across 1m (m/s²/m)
      sigma_spike         — extra sigma added to material (0 to SIGMA_CONV)
    """
    tidal_grad = weapon.get('tidal_gradient_ms2', 0.0)
    sigma_spike = weapon.get('sigma_spike', 0.0)

    if tidal_grad <= 0 and sigma_spike <= 0:
        return _no_effect(thickness_m, 'No exotic effect')

    target = get_material(layer_material)
    rho = target['density_kg_m3']

    # Tidal stress across layer thickness
    # ΔF/A = rho * tidal_gradient * thickness
    tidal_stress = rho * tidal_grad * thickness_m

    # Sigma-field modified yield stress
    # At sigma=0: normal yield. At sigma=SIGMA_CONV: yield → 0.
    sigma_effective = min(sigma_spike, SIGMA_CONV * 0.999)
    sigma_ratio = 1.0 - (sigma_effective / SIGMA_CONV) ** 2
    base_yield = target['yield_stress_pa']
    effective_yield = base_yield * max(0.001, sigma_ratio)

    # Combined failure check
    failed = tidal_stress > effective_yield

    notes_parts = []
    if tidal_grad > 0:
        notes_parts.append(f'tidal_stress={tidal_stress:.2e}Pa vs yield={effective_yield:.2e}Pa')
    if sigma_spike > 0:
        notes_parts.append(f'σ-spike={sigma_spike:.3f} (σ_conv={SIGMA_CONV:.3f}), yield reduced to {sigma_ratio*100:.0f}%')

    if failed:
        w_out = dict(weapon)
        w_out['tidal_gradient_ms2'] = tidal_grad * 0.7  # attenuates through armor
        w_out['sigma_spike'] = sigma_spike * 0.5
        return {
            'penetrated': True,
            'temperature_k': 300.0,  # exotic weapons don't primarily heat
            'damage_depth_m': thickness_m,
            'energy_deposited_j': tidal_stress * thickness_m * math.pi * 0.05**2,
            'notes': 'EXOTIC BREACH: ' + '; '.join(notes_parts),
            'weapon_out': w_out,
        }

    damage_depth = thickness_m * min(1.0, tidal_stress / max(effective_yield, 1.0))
    return {
        'penetrated': False,
        'temperature_k': 300.0,
        'damage_depth_m': damage_depth,
        'energy_deposited_j': 0.0,
        'notes': 'Contained: ' + '; '.join(notes_parts),
        'weapon_out': None,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'notes': notes, 'weapon_out': None,
    }
