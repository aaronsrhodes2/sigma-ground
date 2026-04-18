"""
Material adapter — wraps sigma-ground interface modules into a simple dict
per material, pre-computing all properties needed by the physics solvers.
"""

from sigma_ground.field.interface.surface import MATERIALS as _BASE
from sigma_ground.field.interface.hardness import yield_stress, vickers_hardness
from sigma_ground.field.interface.thermal import (
    thermal_conductivity, heat_capacity_volumetric, blackbody_color,
    debye_temperature,
)
from sigma_ground.field.interface.mechanical import bulk_modulus

# Known melting points (K) — authoritative lookup
MELTING_POINTS_K = {
    'iron': 1811, 'copper': 1358, 'aluminum': 933, 'gold': 1337,
    'silicon': 1687, 'tungsten': 3695, 'nickel': 1728, 'titanium': 1941,
    'steel_mild': 1700, 'lead': 600, 'silver': 1235, 'platinum': 2041,
    'depleted_uranium': 1408, 'rubber': 420, 'plastic_abs': 500,
    'glass': 1400, 'concrete': 1600, 'granite': 1400, 'ceramic_alumina': 2345,
    'water_ice': 273, 'wood_oak': 573, 'bone': 1200, 'carbon_fiber': 3800,
    'kevlar': 773,
}

# Base colors for 2D cross-section rendering (RGB, 0-255)
MATERIAL_COLORS = {
    'iron':           (120, 110, 100),
    'copper':         (184, 115,  51),
    'aluminum':       (200, 200, 210),
    'gold':           (218, 165,  32),
    'silicon':        (100, 120, 160),
    'tungsten':       ( 55,  55,  65),
    'nickel':         (150, 150, 160),
    'titanium':       (115, 145, 170),
    'steel_mild':     (100,  95,  90),
    'lead':           ( 80,  80,  95),
    'silver':         (192, 192, 200),
    'platinum':       (180, 185, 190),
    'depleted_uranium':(70, 100,  70),
    'rubber':         ( 40,  40,  40),
    'plastic_abs':    (200, 180, 160),
    'glass':          (180, 220, 240),
    'concrete':       (160, 155, 150),
    'granite':        (130, 110, 110),
    'ceramic_alumina':(230, 220, 200),
    'water_ice':      (200, 230, 255),
    'wood_oak':       (140, 100,  60),
    'bone':           (240, 230, 200),
    'carbon_fiber':   ( 30,  30,  35),
    'kevlar':         (210, 190, 120),
}

MATERIAL_DESCRIPTIONS = {
    'iron':           'Iron — classic structural metal',
    'copper':         'Copper — excellent conductor, soft',
    'aluminum':       'Aluminum — lightweight structural',
    'gold':           'Gold — very dense, corrosion-proof',
    'silicon':        'Silicon — brittle semiconductor',
    'tungsten':       'Tungsten — densest metal, 3695 K melt',
    'nickel':         'Nickel — tough, corrosion-resistant',
    'titanium':       'Titanium — strong, lightweight, 1941 K',
    'steel_mild':     'Steel (mild) — structural workhorse',
    'lead':           'Lead — very dense, soft, cheap',
    'silver':         'Silver — best conductor, costly',
    'platinum':       'Platinum — ultra-dense, inert',
    'depleted_uranium':'Depleted uranium — dense, self-sharpening',
    'rubber':         'Rubber — elastic, low melt point',
    'plastic_abs':    'ABS plastic — lightweight, cheap',
    'glass':          'Glass — brittle, moderate hardness',
    'concrete':       'Concrete — cheap, brittle in tension',
    'granite':        'Granite — hard rock, compressive',
    'ceramic_alumina':'Alumina ceramic — very hard, brittle',
    'water_ice':      'Water ice — melts at 273 K',
    'wood_oak':       'Oak wood — combusts ~573 K',
    'bone':           'Bone — biological composite',
    'carbon_fiber':   'Carbon fiber — light, very strong',
    'kevlar':         'Kevlar — ballistic fiber, tough',
}

# Display groups for the armor picker
MATERIAL_GROUPS = {
    'Metals': ['iron', 'copper', 'aluminum', 'gold', 'nickel', 'titanium',
               'steel_mild', 'lead', 'silver', 'platinum', 'depleted_uranium', 'tungsten'],
    'Ceramics & Rock': ['silicon', 'glass', 'concrete', 'granite', 'ceramic_alumina'],
    'Composites & Polymers': ['carbon_fiber', 'kevlar', 'rubber', 'plastic_abs', 'bone'],
    'Exotic': ['water_ice', 'wood_oak'],
}

_cache = {}


def get_material(name: str) -> dict:
    """Return a pre-computed property dict for a material name."""
    if name in _cache:
        return _cache[name]

    base = _BASE[name]
    props = {
        'name':              name,
        'density_kg_m3':     base['density_kg_m3'],
        'melting_point_k':   MELTING_POINTS_K[name],
        'yield_stress_pa':   yield_stress(name),
        'vickers_hv':        vickers_hardness(name),
        'thermal_cond':      thermal_conductivity(name, 300.0),
        'heat_cap_vol':      heat_capacity_volumetric(name, 300.0),  # J/(m³·K)
        'bulk_modulus_pa':   bulk_modulus(name),
        'color':             MATERIAL_COLORS[name],
        'description':       MATERIAL_DESCRIPTIONS[name],
    }
    _cache[name] = props
    return props


def all_materials() -> list:
    return list(_BASE.keys())


def temperature_color(mat_name: str, temperature_k: float) -> tuple:
    """Blend material base color with damage color based on temperature."""
    mp = MELTING_POINTS_K[mat_name]
    base = MATERIAL_COLORS[mat_name]

    if temperature_k < 300:
        return base

    # Fraction toward melting
    frac = min(1.0, (temperature_k - 300) / max(1, mp - 300))

    if temperature_k >= mp * 1.5:
        # Vaporized — bright white/black
        return (20, 20, 20)

    if temperature_k >= mp:
        # Molten — deep orange/red
        return (200, 80, 20)

    # Heating up: blend base → orange → near-white
    bb = blackbody_color(max(800, temperature_k))
    bb_rgb = (int(bb[0] * 255), int(bb[1] * 255), int(bb[2] * 255))
    r = int(base[0] * (1 - frac) + bb_rgb[0] * frac)
    g = int(base[1] * (1 - frac) + bb_rgb[1] * frac)
    b = int(base[2] * (1 - frac) + bb_rgb[2] * frac)
    return (min(255, r), min(255, g), min(255, b))
