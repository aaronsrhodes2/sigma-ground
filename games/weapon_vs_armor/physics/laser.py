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


# metals with no measured n+k of their own take their base metal's (steels are iron)
_NK_PROXY = {'steel_mild': 'iron'}


def absorptivity(mat_name: str, wavelength_nm: float):
    """``(fraction of the beam the surface takes in, source)``. A metal with measured n+ik: Kirchhoff, 1 − R at the laser's
    wavelength (``thermal_emission.emissivity``: iron 0.46, aluminium 0.10 at 1064 nm). Anything else: 1, a gap, as the
    game always assumed (sigma-ground's block() emissivity for alumina, carbon fibre and steel reads 0.01, not usable)."""
    from sigma_ground.field.interface import thermal_emission as te
    key = _NK_PROXY.get(mat_name, mat_name)
    if key in te.MEASURED_NK:
        how = "Kirchhoff 1-R, measured n+ik" + (f" of {key}" if key != mat_name else "")
        return te.emissivity(key, wavelength_nm * 1e-9), how
    return 1.0, "no measured reflectance: all absorbed (gap)"


CW_MIN_S = 1e-3   # a pulse this long or longer burns through by melting (the melt is blown or flows out), not vaporising


def ablation_front(mat_name: str, duration_s: float = 1.0):
    """``(J per m³ to remove, front temperature K, source)``: what removing one cubic metre of the layer costs.

    Energy balance, every number from ``field.interface.block`` (measured first):
      - a continuous beam (``duration_s`` ≥ ``CW_MIN_S``: laser cutting, HEL burn-through) melts its way through, the melt
        driven out by recoil pressure, gas or airflow: H = ρ·c_p·(T_m − 300) + ρ·L_fus, front at the melting point;
      - a short pulse vaporises: H = ρ·c_p·(T_b − 300) + ρ·L_fus + ρ·L_vap, front at the boiling point.
    A material with no latent heats in sigma-ground (kevlar, rubber, composites: they char or decompose) takes the
    sensible heat to its melting point only, a lower bound, and says so.
    """
    from sigma_ground.field.interface.block import block
    p = block(mat_name)["properties"]
    rho, cvol = p["density"]["value"], p["heat_capacity_volumetric"]["value"]
    if duration_s >= CW_MIN_S and "latent_heat_fusion" in p and "melting_point" in p:
        T = p["melting_point"]["value"]
        return cvol * (T - 300.0) + rho * p["latent_heat_fusion"]["value"], T, \
            "melted through (block(): melting point, latent heat of fusion)"
    if "boiling_point" in p and "latent_heat_vaporization" in p:
        T = p["boiling_point"]["value"]
        H = cvol * (T - 300.0) + rho * (p.get("latent_heat_fusion", {}).get("value", 0.0)
                                        + p["latent_heat_vaporization"]["value"])
        return H, T, "vaporised (block(): boiling point, latent heats)"
    T = p.get("melting_point", {}).get("value") or MELTING_POINTS_K.get(mat_name, 1000.0)
    return cvol * max(T - 300.0, 1.0), T, "ablated at its melting point, sensible heat only (no latent heats: a lower bound)"


def interact(weapon: dict, layer_material: str, thickness_m: float) -> dict:
    """
    Laser interaction with one armor layer: an energy balance.

    weapon keys: type, power_w, wavelength_nm, pulse_duration_s, spot_radius_m
    Returns: penetrated, temperature_k, damage_depth_m, energy_deposited_j,
             ablated (bool), weapon_out

    The surface takes in a fraction of the beam (``absorptivity``: measured for metals) and the layer absorbs that
    by Beer-Lambert; that energy first heats the spot down to the thermal diffusion length
    (√(4·κ·t), or 1/α if light reaches deeper): if that stays under the melting point, the layer only warms. Past it,
    the beam drills: depth = absorbed energy / (spot area × H), H from ``ablation_front`` (sigma-ground's data: melt-through
    for a continuous beam, vaporisation for a short pulse), and the crater's wall sits at the front's temperature. A layer drilled through passes the rest of the pulse on.
    (Before 2026-10-31 all the energy went into 1/α, ~50 nm of steel, giving ~10⁹ K and whole layers gone.)
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
    total_energy_j = power * duration
    A, a_src = absorptivity(layer_material, wavelength)
    absorbed = total_energy_j * A * (1.0 - math.exp(-alpha * thickness_m))

    heat_cap = target['heat_cap_vol']
    kappa = target['thermal_cond'] / max(heat_cap, 1e-30)            # thermal diffusivity, m²/s
    heated_depth = min(thickness_m, max(1.0 / max(alpha, 1.0), math.sqrt(4.0 * kappa * duration)))
    temperature_k = 300.0 + absorbed / max(spot_area * heated_depth * heat_cap, 1e-30)
    melting_k = target['melting_point_k']

    if temperature_k < melting_k:
        return {
            'penetrated': False, 'temperature_k': temperature_k, 'damage_depth_m': 0.0,
            'energy_deposited_j': absorbed, 'ablated': False,
            'notes': f'heated to {temperature_k:.0f}K (melt={melting_k:.0f}K)', 'weapon_out': None,
        }

    H, front_k, how = ablation_front(layer_material, duration)
    depth = absorbed / max(spot_area * H, 1e-30)
    if depth >= thickness_m:
        used = thickness_m * spot_area * H
        w_out = dict(weapon)
        w_out['power_w'] = max(0.0, total_energy_j - used / max(A, 1e-9)) / max(duration, 1e-30)
        return {
            'penetrated': True, 'temperature_k': front_k, 'damage_depth_m': thickness_m,
            'energy_deposited_j': used, 'ablated': True,
            'notes': f'drilled through ({how}, {H:.3g} J/m3, front {front_k:.0f}K; absorbed {A:.2f}: {a_src})',
            'weapon_out': w_out,
        }
    return {
        'penetrated': False, 'temperature_k': front_k, 'damage_depth_m': depth,
        'energy_deposited_j': absorbed, 'ablated': True,
        'notes': f'drilled {depth * 1000:.2f} mm ({how}, front {front_k:.0f}K)', 'weapon_out': None,
    }


def _no_effect(thickness_m, notes):
    return {
        'penetrated': False, 'temperature_k': 300.0,
        'damage_depth_m': 0.0, 'energy_deposited_j': 0.0,
        'ablated': False, 'notes': notes, 'weapon_out': None,
    }
