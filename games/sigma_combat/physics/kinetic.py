"""
Kinetic penetrator physics — Poncelet + Birkhoff hydrodynamic transition.

Poncelet (low velocity, rigid penetrator):
    x_max = m/(2B) * ln(1 + B*v0^2 / A)
    A = yield_stress * pi * r^2   (static resistance)
    B = rho_target * pi * r^2     (inertial resistance)

Birkhoff (high velocity, hydrodynamic regime, v >> material sound speed):
    x_max = L * sqrt(rho_penetrator / rho_target)

Transition at v ~ 2 * sqrt(yield_stress / rho_target) (Hugoniot elastic limit).
"""
import math
from ..materials import get_material

C_LIGHT = 299_792_458.0


def _sound_speed(mat_name: str) -> float:
    props = get_material(mat_name)
    # c_s = sqrt(bulk_modulus / density)
    return math.sqrt(props['bulk_modulus_pa'] / props['density_kg_m3'])


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Compute kinetic penetrator interaction with one armor layer.

    weapon keys: type, mass_kg, velocity_ms, radius_m, material
    Returns dict with: penetrated, velocity_out_ms, temperature_k,
                       damage_depth_m, energy_deposited_j, notes
    """
    v0 = weapon['velocity_ms']
    m = weapon['mass_kg']
    r = weapon.get('radius_m', 0.01)
    pen_mat = weapon.get('material', 'tungsten')

    if v0 <= 0:
        return _stopped(0.0, 0.0, thickness_m, 'No velocity')

    target = get_material(layer_material)
    pen = get_material(pen_mat)

    rho_t = target['density_kg_m3']
    rho_p = pen['density_kg_m3']
    Y = target['yield_stress_pa']
    area = math.pi * r * r

    # Relativistic kinetic energy
    beta = v0 / C_LIGHT
    if beta >= 1.0:
        beta = 0.9999
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    KE0 = (gamma - 1.0) * m * C_LIGHT * C_LIGHT

    # Sound speed transition: above this, hydrodynamic regime
    c_s = _sound_speed(layer_material)
    v_hel = 2.0 * math.sqrt(Y / rho_t)  # Hugoniot Elastic Limit velocity

    if v0 > 5 * c_s:
        # Hydrodynamic / hypervelocity: Birkhoff penetration
        # Length estimate: assume L/r = 10 (long-rod approximation)
        L_pen = 10.0 * r
        x_max = L_pen * math.sqrt(rho_p / rho_t)
        regime = 'hypervelocity (Birkhoff)'
        # Energy partition: most goes to crater + heat
        energy_deposited = KE0 * min(1.0, thickness_m / x_max)
    else:
        # Poncelet regime
        A = Y * area
        B = rho_t * area
        if A <= 0 or B <= 0:
            x_max = thickness_m + 1  # pass through
            energy_deposited = KE0 * 0.1
            regime = 'no resistance'
        else:
            denom = B * v0 * v0 / A
            if denom <= 0:
                x_max = 0.0
            else:
                x_max = (m / (2 * B)) * math.log1p(denom)
            regime = 'Poncelet'
            energy_deposited = KE0 * min(1.0, thickness_m / max(x_max, 1e-10))

    # Heating: energy deposited heats a cylindrical volume
    vol = area * min(thickness_m, x_max)
    heat_cap = target['heat_cap_vol']  # J/(m³·K)
    delta_T = energy_deposited / max(vol * heat_cap, 1e-30)
    temperature_k = 300.0 + delta_T

    if x_max >= thickness_m:
        # Penetrated — compute exit velocity
        # Remaining KE after Poncelet deceleration
        if 'Poncelet' in regime:
            A = Y * area
            B = rho_t * area
            # v_exit^2 = (v0^2 + A/B) * exp(-2*B*thickness/m) - A/B
            exponent = -2 * B * thickness_m / m
            v_exit_sq = (v0 * v0 + A / B) * math.exp(exponent) - A / B
            v_exit = math.sqrt(max(0.0, v_exit_sq))
        else:
            # Hydrodynamic: proportional energy loss
            frac_remaining = max(0.0, 1.0 - thickness_m / x_max)
            v_exit = v0 * math.sqrt(frac_remaining)

        w_out = dict(weapon)
        w_out['velocity_ms'] = v_exit
        return {
            'penetrated': True,
            'velocity_out_ms': v_exit,
            'temperature_k': temperature_k,
            'damage_depth_m': thickness_m,
            'energy_deposited_j': energy_deposited,
            'notes': f'{regime}, x_max={x_max:.3f}m, T={temperature_k:.0f}K',
            'weapon_out': w_out,
        }
    else:
        return _stopped(x_max, temperature_k, thickness_m,
                        f'{regime}, stopped at {x_max:.3f}m, T={temperature_k:.0f}K')


def _stopped(depth, temp, thickness, notes):
    return {
        'penetrated': False,
        'velocity_out_ms': 0.0,
        'temperature_k': temp,
        'damage_depth_m': depth,
        'energy_deposited_j': 0.0,
        'notes': notes,
        'weapon_out': None,
    }
