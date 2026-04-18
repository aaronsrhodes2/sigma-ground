"""
Main simulation solver — orchestrates per-layer loop, returns DamageResult.

Usage:
    result = solve(weapon, armor_layers, environment='vacuum', distance_m=100)
    print(result.verdict)
"""
from dataclasses import dataclass, field
from typing import List, Optional

from . import kinetic, laser, plasma, explosive, particle, gravitational
from .environment import attenuate

_MODULES = {
    'kinetic':      kinetic,
    'laser':        laser,
    'plasma':       plasma,
    'explosive':    explosive,
    'particle':     particle,
    'gravitational': gravitational,
}


@dataclass
class LayerState:
    material: str
    thickness_m: float
    temperature_k: float = 300.0
    penetrated: bool = False
    damage_depth_m: float = 0.0
    energy_deposited_j: float = 0.0
    notes: str = ''

    @property
    def damage_fraction(self) -> float:
        return min(1.0, self.damage_depth_m / max(self.thickness_m, 1e-10))

    @property
    def is_destroyed(self) -> bool:
        return self.penetrated or self.damage_fraction >= 0.99


@dataclass
class DamageResult:
    layers: List[LayerState]
    layers_penetrated: int
    total_layers: int
    verdict: str
    weapon_type: str
    environment: str
    summary_lines: List[str] = field(default_factory=list)

    @property
    def breached(self) -> bool:
        return self.layers_penetrated >= self.total_layers


def solve(weapon: dict, armor_layers: list,
          environment: str = 'vacuum', distance_m: float = 100.0) -> DamageResult:
    """
    Simulate weapon through armor stack.

    weapon: dict with 'type' key + type-specific params
    armor_layers: list of {'material': str, 'thickness_m': float}
                  or {'gap': True, 'thickness_m': float} for air gaps
    environment: key from environment.ENVIRONMENTS
    distance_m: range from weapon origin to armor face
    """
    wtype = weapon.get('type', 'kinetic')
    mod = _MODULES.get(wtype)
    if mod is None:
        raise ValueError(f"Unknown weapon type: {wtype}")

    # Environment attenuation before impact
    w = attenuate(weapon, environment, distance_m)

    states = []
    layers_penetrated = 0
    summary = []

    for i, layer_def in enumerate(armor_layers):
        if layer_def.get('gap', False):
            # Air/vacuum gap — propagate weapon, no material interaction
            gap_t = layer_def['thickness_m']
            w = attenuate(w, 'atmosphere_earth' if environment == 'atmosphere_earth' else 'vacuum', gap_t)
            states.append(LayerState(
                material='gap',
                thickness_m=gap_t,
                notes='gap (no material)',
            ))
            continue

        mat = layer_def['material']
        thick = layer_def['thickness_m']

        result = mod.interact(w, mat, thick)

        state = LayerState(
            material=mat,
            thickness_m=thick,
            temperature_k=result.get('temperature_k', 300.0),
            penetrated=result.get('penetrated', False),
            damage_depth_m=result.get('damage_depth_m', 0.0),
            energy_deposited_j=result.get('energy_deposited_j', 0.0),
            notes=result.get('notes', ''),
        )
        states.append(state)

        line = f'Layer {i+1} [{mat} {thick*1000:.1f}mm]: {result.get("notes","")}'
        summary.append(line)

        if state.penetrated:
            layers_penetrated += 1
            if result.get('weapon_out') is not None:
                w = result['weapon_out']
            else:
                break
        else:
            break

    total = sum(1 for s in states if s.material != 'gap')
    if layers_penetrated >= total:
        verdict = 'ARMOR BREACHED'
    elif layers_penetrated == 0:
        verdict = 'ARMOR HOLDS — first layer stopped it'
    else:
        verdict = f'ARMOR HOLDS — stopped at layer {layers_penetrated + 1} of {total}'

    return DamageResult(
        layers=states,
        layers_penetrated=layers_penetrated,
        total_layers=total,
        verdict=verdict,
        weapon_type=wtype,
        environment=environment,
        summary_lines=summary,
    )
