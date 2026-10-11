"""
Asteroid Data — triaxial ellipsoid models from spacecraft encounters.

Each asteroid is defined by three semi-axes (a ≥ b ≥ c), mass,
bulk density, rotation period, and spin state.

Dimensions and masses from mission data (OSIRIS-REx, Hayabusa,
Hayabusa2, NEAR Shoemaker, Dawn). No empirical fits imported —
all derived properties computed from our model using:

    σ(r) = ξ × Φ(r) / c²

where Φ(r) is the gravitational potential computed from the
body's own mass distribution.
"""

import math
from .constants import G, C

# ── Asteroid Registry ─────────────────────────────────────────────────

BENNU = {
    'name': 'Bennu',
    'mission': 'OSIRIS-REx (2018-2021)',
    # Semi-axes in meters (a ≥ b ≥ c): the best-fit ellipsoid of the
    # OSIRIS-REx shape model, Barnouin et al. 2019, Nat. Geosci. 12, 247,
    # Table 1: (252.78 ± 0.05) × (246.20 ± 0.09) × (228.69 ± 0.12) m.
    # Its volume, 0.0596 km³, is 0.97× the shape-model volume 0.0615 km³
    # (same paper), so M/V is 1.03× the measured 1190 kg/m³.  The full
    # extents of the spinning-top shape are larger: (563.87 × 536.09 ×
    # 498.42) ± 0.17 m (Barnouin et al. 2019, as quoted in NAIF
    # bennu_v14.tpc); the pre-arrival Design Reference Asteroid (Hergenrother
    # et al. 2014, arXiv:1409.4704, from the Nolan et al. 2013 radar model)
    # gave 565 × 535 × 508 m.
    # FIXED 2026-10-10: the values were halved (131.25 × 129.25 × 124.65 m;
    # implied density 6.95× the table's).
    'a_m': 252.78,
    'b_m': 246.20,
    'c_m': 228.69,
    'mass_kg': 7.329e10,
    'density_kg_m3': 1190,
    'rotation_period_h': 4.2905,
    'spin_state': 'retrograde, slight wobble',
    'shape_desc': 'Diamond / spinning top',
    'surface': 'Carbonaceous, rubble pile',
    'color_primary': '#3a3a3a',    # dark carbon
    'color_accent': '#5a4a3a',     # slight brown
    'spectral_type': 'B-type',
}

RYUGU = {
    'name': 'Ryugu',
    'mission': 'Hayabusa2 (2018-2019)',
    # Equatorial radius 502 ± 2 m and polar radius 438 ± 2 m of the
    # Hayabusa2 shape model (Watanabe et al. 2019, Science 364, 268).  These
    # are the radii of the spinning top's equatorial ridge and poles, so the
    # spheroid through them holds 0.462 km³, 1.23× the shape-model volume
    # 0.377 ± 0.005 km³ (same paper): M/V is 0.82× the measured 1190 kg/m³
    # and (abc)^(1/3) = 480 m vs the volume-equivalent 448 m.
    # FIXED 2026-10-10: the values were halved (252 × 251 × 237 m; implied
    # density 6.02× the table's).
    'a_m': 502.0,
    'b_m': 502.0,
    'c_m': 438.0,
    'mass_kg': 4.50e11,
    'density_kg_m3': 1190,
    'rotation_period_h': 7.6326,
    'spin_state': 'retrograde',
    'shape_desc': 'Spinning top / oblate diamond',
    'surface': 'Carbonaceous, rubble pile, boulders',
    'color_primary': '#2e2e2e',
    'color_accent': '#4a3e30',
    'spectral_type': 'Cb-type',
}

ITOKAWA = {
    'name': 'Itokawa',
    'mission': 'Hayabusa (2005)',
    'a_m': 535 / 2,      # very elongated
    'b_m': 294 / 2,
    'c_m': 209 / 2,
    'mass_kg': 3.51e10,
    'density_kg_m3': 1950,
    'rotation_period_h': 12.132,
    'spin_state': 'tumbling (non-principal axis)',
    'shape_desc': 'Peanut / contact binary',
    'surface': 'S-type silicate, smooth + rough halves',
    'color_primary': '#8a7a6a',    # pale tan
    'color_accent': '#6a6050',
    'spectral_type': 'S-type',
}

EROS = {
    'name': 'Eros',
    'mission': 'NEAR Shoemaker (2000-2001)',
    'a_m': 34400 / 2,    # 34.4 km × 11.2 km × 11.2 km
    'b_m': 11200 / 2,
    'c_m': 11200 / 2,
    'mass_kg': 6.687e15,
    'density_kg_m3': 2670,
    'rotation_period_h': 5.270,
    'spin_state': 'principal axis',
    'shape_desc': 'Elongated, curved, saddle shape',
    'surface': 'Silicate regolith, craters, boulders',
    'color_primary': '#9a8a70',
    'color_accent': '#7a6a5a',
    'spectral_type': 'S-type',
}

VESTA = {
    'name': 'Vesta',
    'mission': 'Dawn (2011-2012)',
    'a_m': 572600 / 2,   # 572.6 × 557.2 × 446.4 km
    'b_m': 557200 / 2,
    'c_m': 446400 / 2,
    'mass_kg': 2.59076e20,
    'density_kg_m3': 3456,
    'rotation_period_h': 5.342,
    'spin_state': 'principal axis',
    'shape_desc': 'Oblate, giant south pole basin (Rheasilvia)',
    'surface': 'Basaltic (differentiated), HED meteorites',
    'color_primary': '#a0a0a0',
    'color_accent': '#c0b8a0',
    'spectral_type': 'V-type',
}

CERES = {
    'name': 'Ceres',
    'mission': 'Dawn (2015-2018)',
    'a_m': 964400 / 2,   # 964.4 × 964.2 × 891.8 km
    'b_m': 964200 / 2,
    'c_m': 891800 / 2,
    'mass_kg': 9.3835e20,
    'density_kg_m3': 2162,
    'rotation_period_h': 9.074,
    'spin_state': 'principal axis',
    'shape_desc': 'Nearly round (dwarf planet), bright spots (Occator)',
    'surface': 'Hydrated minerals, ammoniated clays, water ice',
    'color_primary': '#707070',
    'color_accent': '#909080',
    'spectral_type': 'C-type',
}


ALL_ASTEROIDS = {
    'bennu': BENNU,
    'ryugu': RYUGU,
    'itokawa': ITOKAWA,
    'eros': EROS,
    'vesta': VESTA,
    'ceres': CERES,
}


# ── Derived Properties (from our model, not imported) ─────────────────

def ellipsoid_volume(a, b, c):
    """Volume of a triaxial ellipsoid: V = (4/3)πabc"""
    return (4.0 / 3.0) * math.pi * a * b * c


def mean_radius(a, b, c):
    """Geometric mean radius: R_mean = (abc)^(1/3)"""
    return (a * b * c) ** (1.0 / 3.0)


def surface_gravity(body):
    """Surface gravity at the tip of the longest axis.

    g = GM/a²  (along the a-axis, minimum gravity)

    We compute all three axis gravities — the variation IS the wobble.
    No external formula; this is Newton directly.

    PHYSICS_GAP: point-mass GM/r² at each tip, non-rotating.  A homogeneous
    ellipsoid's own field differs at the tips (audit 2026-10-10: Eros g_a
    3× low, g_c 2.3× high), and the centrifugal ω²a is 42 % (Eros), 12 %
    (Vesta) and 55 % (Bennu, at a = 252.78 m) of g at the equator.  Not
    fixed here.
    """
    M = body['mass_kg']
    return {
        'g_a': G * M / body['a_m']**2,  # at tip of longest axis (weakest)
        'g_b': G * M / body['b_m']**2,  # at tip of middle axis
        'g_c': G * M / body['c_m']**2,  # at tip of shortest axis (strongest)
        'g_mean': G * M / mean_radius(body['a_m'], body['b_m'], body['c_m'])**2,
        'g_ratio_a_c': body['c_m']**2 / body['a_m']**2,  # how much gravity varies
    }


def escape_velocity(body):
    """Escape velocity from the surface at each axis tip.

    v_esc = sqrt(2GM/r) — derived from energy conservation, no imports.
    """
    M = body['mass_kg']
    return {
        'v_a': math.sqrt(2 * G * M / body['a_m']),
        'v_b': math.sqrt(2 * G * M / body['b_m']),
        'v_c': math.sqrt(2 * G * M / body['c_m']),
    }


def axis_ratios(body):
    """Shape ratios — how far from spherical.

    b/a = 1 and c/a = 1 → perfect sphere.
    Lower ratios → more elongated/oblate.
    """
    a = body['a_m']
    return {
        'b_over_a': body['b_m'] / a,
        'c_over_a': body['c_m'] / a,
        'oblateness': 1.0 - body['c_m'] / a,
        'elongation': 1.0 - body['b_m'] / a,
        'is_nearly_spherical': (body['b_m'] / a > 0.95 and body['c_m'] / a > 0.85),
        'is_contact_binary': body['b_m'] / a < 0.65,
    }
