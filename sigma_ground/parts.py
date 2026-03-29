"""
Standard parts catalog — engineering shapes as analytic primitives.

Every standard part decomposes into a small list of {Sphere, Torus, Cone,
Cylinder, Box} primitives.  A 5000-triangle bolt mesh (~60 KB) becomes
3 primitives (~200 bytes) with exact volume, inertia, and surface area.

Three catalogs:
  PIPES       — ANSI B36.10 / B36.19 standard pipe sizes
  ISO_BOLTS   — ISO 724 metric fasteners
  W_BEAMS     — AISC wide-flange structural steel sections

Each catalog entry stores only the defining dimensions.  Structure builders
(`pipe_structure`, `bolt_structure`, `beam_structure`) turn those dimensions
into a Structure of analytic primitives at a requested length.

Conversion pipeline:
  `convert_to_primitives(description)` accepts a dict of external shape data
  and returns the best analytic decomposition with a quality grade.

  Quality grades:
    excellent  (>= 0.95 volume efficiency)
    good       (0.85 - 0.95)
    fair       (0.70 - 0.85)
    poor       (< 0.70) — original data preserved, reason given

Boundary agreement:
  After conversion, we sample points on the target shape's boundary and
  measure how close our primitives' surfaces come.  If volume efficiency
  AND boundary agreement are both high, the external data is redundant
  and can be safely discarded.

  Metrics:
    volume_efficiency  — did we capture the right amount of matter?
    boundary_score     — do our surfaces match the original shape's outline?
    can_discard        — True if both metrics pass thresholds

Shape budget:
  The number of primitives allowed scales with the complexity of the
  external data: budget = max(3, source_bytes / 300).  A 60 KB mesh
  gets 200 primitives.  A 2 KB spec gets 6.

All dimensions in SI (metres).  Pure Python, zero dependencies.
"""

import math
import random
from .shapes import Sphere, Cylinder, Box, Cone, Torus, Structure


# ── Pipes — ANSI B36.10 (carbon steel) ──────────────────────────────
# NPS (Nominal Pipe Size) → outer diameter and wall thickness.
# Source: ASME/ANSI B36.10M
# Key: 'NPS_{size}_SCH{schedule}'

PIPES = {
    # NPS ⅛" to 2"
    'NPS_0.125_SCH40':  {'od_m': 0.01029, 'wall_m': 0.00175, 'desc': '⅛" Sch 40'},
    'NPS_0.25_SCH40':   {'od_m': 0.01372, 'wall_m': 0.00218, 'desc': '¼" Sch 40'},
    'NPS_0.375_SCH40':  {'od_m': 0.01715, 'wall_m': 0.00231, 'desc': '⅜" Sch 40'},
    'NPS_0.5_SCH40':    {'od_m': 0.02134, 'wall_m': 0.00287, 'desc': '½" Sch 40'},
    'NPS_0.75_SCH40':   {'od_m': 0.02667, 'wall_m': 0.00287, 'desc': '¾" Sch 40'},
    'NPS_1_SCH40':      {'od_m': 0.03340, 'wall_m': 0.00338, 'desc': '1" Sch 40'},
    'NPS_1.25_SCH40':   {'od_m': 0.04216, 'wall_m': 0.00356, 'desc': '1¼" Sch 40'},
    'NPS_1.5_SCH40':    {'od_m': 0.04826, 'wall_m': 0.00368, 'desc': '1½" Sch 40'},
    'NPS_2_SCH40':      {'od_m': 0.06033, 'wall_m': 0.00391, 'desc': '2" Sch 40'},
    # NPS 2½" to 6"
    'NPS_2.5_SCH40':    {'od_m': 0.07303, 'wall_m': 0.00516, 'desc': '2½" Sch 40'},
    'NPS_3_SCH40':      {'od_m': 0.08890, 'wall_m': 0.00549, 'desc': '3" Sch 40'},
    'NPS_3.5_SCH40':    {'od_m': 0.10160, 'wall_m': 0.00574, 'desc': '3½" Sch 40'},
    'NPS_4_SCH40':      {'od_m': 0.11430, 'wall_m': 0.00602, 'desc': '4" Sch 40'},
    'NPS_5_SCH40':      {'od_m': 0.14130, 'wall_m': 0.00655, 'desc': '5" Sch 40'},
    'NPS_6_SCH40':      {'od_m': 0.16828, 'wall_m': 0.00711, 'desc': '6" Sch 40'},
    # NPS 8" to 24"
    'NPS_8_SCH40':      {'od_m': 0.21910, 'wall_m': 0.00823, 'desc': '8" Sch 40'},
    'NPS_10_SCH40':     {'od_m': 0.27305, 'wall_m': 0.00927, 'desc': '10" Sch 40'},
    'NPS_12_SCH40':     {'od_m': 0.32385, 'wall_m': 0.01048, 'desc': '12" Sch 40'},
    'NPS_14_SCH40':     {'od_m': 0.35560, 'wall_m': 0.01118, 'desc': '14" Sch 40'},
    'NPS_16_SCH40':     {'od_m': 0.40640, 'wall_m': 0.01270, 'desc': '16" Sch 40'},
    'NPS_18_SCH40':     {'od_m': 0.45720, 'wall_m': 0.01422, 'desc': '18" Sch 40'},
    'NPS_20_SCH40':     {'od_m': 0.50800, 'wall_m': 0.01524, 'desc': '20" Sch 40'},
    'NPS_24_SCH40':     {'od_m': 0.60960, 'wall_m': 0.01749, 'desc': '24" Sch 40'},
    # Schedule 80 (heavy wall) — common sizes
    'NPS_0.5_SCH80':    {'od_m': 0.02134, 'wall_m': 0.00363, 'desc': '½" Sch 80'},
    'NPS_1_SCH80':      {'od_m': 0.03340, 'wall_m': 0.00455, 'desc': '1" Sch 80'},
    'NPS_2_SCH80':      {'od_m': 0.06033, 'wall_m': 0.00554, 'desc': '2" Sch 80'},
    'NPS_4_SCH80':      {'od_m': 0.11430, 'wall_m': 0.00846, 'desc': '4" Sch 80'},
    'NPS_6_SCH80':      {'od_m': 0.16828, 'wall_m': 0.01097, 'desc': '6" Sch 80'},
    'NPS_8_SCH80':      {'od_m': 0.21910, 'wall_m': 0.01270, 'desc': '8" Sch 80'},
    'NPS_10_SCH80':     {'od_m': 0.27305, 'wall_m': 0.01524, 'desc': '10" Sch 80'},
    'NPS_12_SCH80':     {'od_m': 0.32385, 'wall_m': 0.01753, 'desc': '12" Sch 80'},
}


# ── Fasteners — ISO 724 metric bolts ────────────────────────────────
# Source: ISO 724 (thread), ISO 4014/4017 (hex bolt head)
# Dimensions: major diameter, pitch (coarse), head height, head diameter
# (across flats, approximated as circumscribed circle for cylinder model)

ISO_BOLTS = {
    'M1.6': {'major_d_m': 0.0016, 'pitch_m': 0.00035, 'head_h_m': 0.0011, 'head_d_m': 0.0035},
    'M2':   {'major_d_m': 0.0020, 'pitch_m': 0.00040, 'head_h_m': 0.0014, 'head_d_m': 0.0045},
    'M2.5': {'major_d_m': 0.0025, 'pitch_m': 0.00045, 'head_h_m': 0.0018, 'head_d_m': 0.0055},
    'M3':   {'major_d_m': 0.0030, 'pitch_m': 0.00050, 'head_h_m': 0.0020, 'head_d_m': 0.0062},
    'M4':   {'major_d_m': 0.0040, 'pitch_m': 0.00070, 'head_h_m': 0.0028, 'head_d_m': 0.0078},
    'M5':   {'major_d_m': 0.0050, 'pitch_m': 0.00080, 'head_h_m': 0.0035, 'head_d_m': 0.0093},
    'M6':   {'major_d_m': 0.0060, 'pitch_m': 0.00100, 'head_h_m': 0.0040, 'head_d_m': 0.0112},
    'M8':   {'major_d_m': 0.0080, 'pitch_m': 0.00125, 'head_h_m': 0.0053, 'head_d_m': 0.0147},
    'M10':  {'major_d_m': 0.0100, 'pitch_m': 0.00150, 'head_h_m': 0.0065, 'head_d_m': 0.0181},
    'M12':  {'major_d_m': 0.0120, 'pitch_m': 0.00175, 'head_h_m': 0.0078, 'head_d_m': 0.0210},
    'M14':  {'major_d_m': 0.0140, 'pitch_m': 0.00200, 'head_h_m': 0.0090, 'head_d_m': 0.0243},
    'M16':  {'major_d_m': 0.0160, 'pitch_m': 0.00200, 'head_h_m': 0.0100, 'head_d_m': 0.0270},
    'M18':  {'major_d_m': 0.0180, 'pitch_m': 0.00250, 'head_h_m': 0.0120, 'head_d_m': 0.0303},
    'M20':  {'major_d_m': 0.0200, 'pitch_m': 0.00250, 'head_h_m': 0.0130, 'head_d_m': 0.0340},
    'M22':  {'major_d_m': 0.0220, 'pitch_m': 0.00250, 'head_h_m': 0.0140, 'head_d_m': 0.0370},
    'M24':  {'major_d_m': 0.0240, 'pitch_m': 0.00300, 'head_h_m': 0.0150, 'head_d_m': 0.0403},
    'M27':  {'major_d_m': 0.0270, 'pitch_m': 0.00300, 'head_h_m': 0.0170, 'head_d_m': 0.0456},
    'M30':  {'major_d_m': 0.0300, 'pitch_m': 0.00350, 'head_h_m': 0.0190, 'head_d_m': 0.0505},
    'M36':  {'major_d_m': 0.0360, 'pitch_m': 0.00400, 'head_h_m': 0.0230, 'head_d_m': 0.0609},
}


# ── Structural steel — AISC W-shapes (wide flange) ──────────────────
# Source: AISC Steel Construction Manual, 16th Edition
# Key: 'W{depth_in}x{weight_plf}'
# Dimensions: nominal depth, flange width, web thickness, flange thickness

W_BEAMS = {
    'W4x13':   {'depth_m': 0.1063, 'flange_w_m': 0.1036, 'web_t_m': 0.0074, 'flange_t_m': 0.0089},
    'W6x9':    {'depth_m': 0.1499, 'flange_w_m': 0.0991, 'web_t_m': 0.0036, 'flange_t_m': 0.0058},
    'W6x15':   {'depth_m': 0.1524, 'flange_w_m': 0.1524, 'web_t_m': 0.0058, 'flange_t_m': 0.0066},
    'W8x10':   {'depth_m': 0.2007, 'flange_w_m': 0.0998, 'web_t_m': 0.0046, 'flange_t_m': 0.0053},
    'W8x18':   {'depth_m': 0.2070, 'flange_w_m': 0.1334, 'web_t_m': 0.0053, 'flange_t_m': 0.0074},
    'W8x31':   {'depth_m': 0.2032, 'flange_w_m': 0.2032, 'web_t_m': 0.0076, 'flange_t_m': 0.0109},
    'W10x12':  {'depth_m': 0.2515, 'flange_w_m': 0.1014, 'web_t_m': 0.0048, 'flange_t_m': 0.0058},
    'W10x22':  {'depth_m': 0.2591, 'flange_w_m': 0.1461, 'web_t_m': 0.0061, 'flange_t_m': 0.0084},
    'W10x49':  {'depth_m': 0.2534, 'flange_w_m': 0.2534, 'web_t_m': 0.0089, 'flange_t_m': 0.0160},
    'W12x14':  {'depth_m': 0.3023, 'flange_w_m': 0.1016, 'web_t_m': 0.0048, 'flange_t_m': 0.0058},
    'W12x26':  {'depth_m': 0.3099, 'flange_w_m': 0.1651, 'web_t_m': 0.0058, 'flange_t_m': 0.0096},
    'W12x50':  {'depth_m': 0.3099, 'flange_w_m': 0.2054, 'web_t_m': 0.0096, 'flange_t_m': 0.0163},
    'W14x22':  {'depth_m': 0.3490, 'flange_w_m': 0.1270, 'web_t_m': 0.0053, 'flange_t_m': 0.0084},
    'W14x30':  {'depth_m': 0.3521, 'flange_w_m': 0.1715, 'web_t_m': 0.0066, 'flange_t_m': 0.0094},
    'W14x61':  {'depth_m': 0.3531, 'flange_w_m': 0.2540, 'web_t_m': 0.0094, 'flange_t_m': 0.0160},
    'W16x26':  {'depth_m': 0.3988, 'flange_w_m': 0.1397, 'web_t_m': 0.0064, 'flange_t_m': 0.0089},
    'W16x36':  {'depth_m': 0.4026, 'flange_w_m': 0.1778, 'web_t_m': 0.0076, 'flange_t_m': 0.0107},
    'W18x35':  {'depth_m': 0.4501, 'flange_w_m': 0.1524, 'web_t_m': 0.0076, 'flange_t_m': 0.0102},
    'W18x50':  {'depth_m': 0.4572, 'flange_w_m': 0.1905, 'web_t_m': 0.0091, 'flange_t_m': 0.0145},
    'W21x44':  {'depth_m': 0.5253, 'flange_w_m': 0.1651, 'web_t_m': 0.0084, 'flange_t_m': 0.0112},
    'W21x62':  {'depth_m': 0.5334, 'flange_w_m': 0.2096, 'web_t_m': 0.0102, 'flange_t_m': 0.0157},
    'W24x55':  {'depth_m': 0.5994, 'flange_w_m': 0.1778, 'web_t_m': 0.0099, 'flange_t_m': 0.0127},
    'W24x84':  {'depth_m': 0.6121, 'flange_w_m': 0.2286, 'web_t_m': 0.0117, 'flange_t_m': 0.0196},
}


# ── Structure builders ──────────────────────────────────────────────

def pipe_structure(key, length, material='steel_mild'):
    """Build a Structure for a standard pipe.

    Decomposes to outer Cylinder (material) + inner Cylinder (air).

    Args:
        key: pipe designation, e.g. 'NPS_1_SCH40'
        length: pipe length in metres
        material: wall material name

    Returns:
        Structure with volume efficiency ~0.99+
    """
    spec = PIPES.get(key)
    if spec is None:
        raise KeyError(f"Unknown pipe: {key}. Available: {sorted(PIPES.keys())}")

    od = spec['od_m']
    wall = spec['wall_m']
    r_outer = od / 2.0
    r_inner = r_outer - wall
    length = float(length)

    # Target volume = annular cross-section × length
    target_vol = math.pi * (r_outer ** 2 - r_inner ** 2) * length

    s = Structure(target_volume=target_vol)
    s.add(Cylinder(r_outer, length), material)
    s.add(Cylinder(r_inner, length, center=(0, 0, 0)), 'air')

    return s


def bolt_structure(key, length, material='steel_mild'):
    """Build a Structure for a standard ISO metric bolt.

    Decomposes to head Cylinder + shank Cylinder.

    Args:
        key: bolt designation, e.g. 'M6'
        length: shank length in metres (not including head)
        material: bolt material name

    Returns:
        Structure with volume efficiency ~0.98+
    """
    spec = ISO_BOLTS.get(key)
    if spec is None:
        raise KeyError(f"Unknown bolt: {key}. Available: {sorted(ISO_BOLTS.keys())}")

    head_r = spec['head_d_m'] / 2.0
    head_h = spec['head_h_m']
    shank_r = spec['major_d_m'] / 2.0
    length = float(length)

    # Target volume = head cylinder + shank cylinder
    target_vol = (math.pi * head_r ** 2 * head_h +
                  math.pi * shank_r ** 2 * length)

    s = Structure(target_volume=target_vol)

    # Head sits on top of shank
    head_center_z = length + head_h / 2.0
    s.add(Cylinder(head_r, head_h, center=(0, 0, head_center_z)), material)
    # Shank
    s.add(Cylinder(shank_r, length, center=(0, 0, length / 2.0)), material)

    return s


def beam_structure(key, length, material='steel_mild'):
    """Build a Structure for a standard AISC W-beam.

    Decomposes to web Box + 2 flange Boxes.
    I-beam cross-section: two horizontal flanges connected by vertical web.

    Args:
        key: beam designation, e.g. 'W12x26'
        length: beam length in metres (along beam axis = x)
        material: beam material name

    Returns:
        Structure with volume efficiency ~0.99+
    """
    spec = W_BEAMS.get(key)
    if spec is None:
        raise KeyError(f"Unknown beam: {key}. Available: {sorted(W_BEAMS.keys())}")

    d = spec['depth_m']
    bf = spec['flange_w_m']
    tw = spec['web_t_m']
    tf = spec['flange_t_m']
    length = float(length)

    # Web height (between flanges)
    web_h = d - 2.0 * tf

    # Target volume = web + 2 flanges
    target_vol = (tw * web_h * length +
                  2.0 * bf * tf * length)

    s = Structure(target_volume=target_vol)

    # Web: thin vertical plate along beam length
    s.add(Box(length, tw, web_h, center=(length / 2, 0, d / 2)), material)

    # Top flange
    top_z = d - tf / 2.0
    s.add(Box(length, bf, tf, center=(length / 2, 0, top_z)), material)

    # Bottom flange
    bot_z = tf / 2.0
    s.add(Box(length, bf, tf, center=(length / 2, 0, bot_z)), material)

    return s


def list_parts(category=None):
    """List available standard parts.

    Args:
        category: 'pipe', 'fastener', 'beam', or None for all.

    Returns:
        dict of {key: description} for matching parts.
    """
    result = {}

    if category is None or category == 'pipe':
        for k, v in sorted(PIPES.items()):
            result[k] = v['desc']

    if category is None or category == 'fastener':
        for k, v in sorted(ISO_BOLTS.items()):
            result[k] = f"ISO metric bolt {k} (d={v['major_d_m']*1000:.1f}mm)"

    if category is None or category == 'beam':
        for k, v in sorted(W_BEAMS.items()):
            result[k] = f"AISC {k} (depth={v['depth_m']*1000:.0f}mm)"

    return result


# ── Shape budget ────────────────────────────────────────────────────

def shape_budget_from_source(source_bytes):
    """Compute primitive budget from external data size.

    Scales with complexity: a 60 KB mesh gets ~200 primitives,
    a 2 KB spec gets ~6.  Minimum budget is 3 (head + body + detail).

    Args:
        source_bytes: size of the external shape data in bytes.

    Returns:
        int — maximum number of primitives to use.
    """
    return max(3, int(source_bytes / 300))


# ── Boundary sampling ───────────────────────────────────────────────
# Generate points on the surface of known shape types for verification.

def _sample_cylinder_surface(radius, height, center, n):
    """Sample n points uniformly on a cylinder's surface."""
    cx, cy, cz = center
    points = []
    # Lateral surface: 2/3 of points, end caps: 1/3
    n_lateral = max(1, int(n * 0.67))
    n_cap = max(1, (n - n_lateral) // 2)

    for i in range(n_lateral):
        theta = 2.0 * math.pi * i / n_lateral
        z = cz + (random.random() - 0.5) * height
        points.append((cx + radius * math.cos(theta),
                        cy + radius * math.sin(theta), z))

    half_h = height / 2.0
    for cap_z in (cz - half_h, cz + half_h):
        for i in range(n_cap):
            r = radius * math.sqrt(random.random())
            theta = 2.0 * math.pi * random.random()
            points.append((cx + r * math.cos(theta),
                            cy + r * math.sin(theta), cap_z))

    return points


def _sample_box_surface(x, y, z, center, n):
    """Sample n points uniformly on a box's surface."""
    cx, cy, cz = center
    hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
    points = []
    # Area of each face pair
    areas = [y * z, x * z, x * y]
    total = 2.0 * sum(areas)
    n_per = [max(1, int(n * 2.0 * a / total)) for a in areas]

    for i in range(n_per[0]):  # yz faces (x = ±hx)
        for sign in (-1, 1):
            py = cy + (random.random() - 0.5) * y
            pz = cz + (random.random() - 0.5) * z
            points.append((cx + sign * hx, py, pz))

    for i in range(n_per[1]):  # xz faces (y = ±hy)
        for sign in (-1, 1):
            px = cx + (random.random() - 0.5) * x
            pz = cz + (random.random() - 0.5) * z
            points.append((px, cy + sign * hy, pz))

    for i in range(n_per[2]):  # xy faces (z = ±hz)
        for sign in (-1, 1):
            px = cx + (random.random() - 0.5) * x
            py = cy + (random.random() - 0.5) * y
            points.append((px, py, cz + sign * hz))

    return points


def sample_boundary(description, n_points=200):
    """Generate boundary sample points from a shape description.

    These points lie on the TRUE surface of the described shape.
    After conversion to primitives, we test whether our primitives'
    surfaces pass through these same points (boundary_agreement).

    Args:
        description: dict with 'type' and dimensions.
        n_points: number of surface points to generate.

    Returns:
        list of (x, y, z) tuples on the target boundary.
        Empty list if the shape type can't be sampled.
    """
    shape_type = description.get('type', '').lower()
    c = (0.0, 0.0, 0.0)  # default center

    if shape_type in ('pipe', 'tube'):
        od = description.get('outer_diameter_m')
        wall = description.get('wall_thickness_m')
        length = description.get('length_m', 1.0)
        if od is None or wall is None:
            return []
        r_outer = od / 2.0
        r_inner = r_outer - wall
        # Sample on BOTH surfaces: outer cylinder + inner cylinder
        pts = _sample_cylinder_surface(r_outer, length, c, n_points // 2)
        pts += _sample_cylinder_surface(r_inner, length, c, n_points // 2)
        return pts

    elif shape_type in ('bolt', 'screw', 'fastener'):
        iso_key = description.get('iso_key')
        if iso_key and iso_key in ISO_BOLTS:
            spec = ISO_BOLTS[iso_key]
            d = description.get('diameter_m') or spec['major_d_m']
            head_d = description.get('head_diameter_m', spec['head_d_m'])
            head_h = description.get('head_height_m', spec['head_h_m'])
        else:
            d = description.get('diameter_m') or description.get('major_diameter_m')
            head_d = description.get('head_diameter_m', d * 1.8 if d else None)
            head_h = description.get('head_height_m', d * 0.65 if d else None)
        length = description.get('length_m', 0.03)
        if d is None:
            return []
        # Sample on shank + head
        shank_c = (0, 0, length / 2.0)
        head_c = (0, 0, length + head_h / 2.0)
        pts = _sample_cylinder_surface(d / 2.0, length, shank_c, n_points * 2 // 3)
        pts += _sample_cylinder_surface(head_d / 2.0, head_h, head_c, n_points // 3)
        return pts

    elif shape_type in ('beam', 'i-beam', 'w-beam'):
        depth = description.get('depth_m')
        flange_w = description.get('flange_width_m')
        web_t = description.get('web_thickness_m')
        flange_t = description.get('flange_thickness_m')
        length = description.get('length_m', 1.0)
        if not all([depth, flange_w, web_t, flange_t]):
            return []
        web_h = depth - 2.0 * flange_t
        # Sample on web + 2 flanges
        web_c = (length / 2, 0, depth / 2)
        top_c = (length / 2, 0, depth - flange_t / 2)
        bot_c = (length / 2, 0, flange_t / 2)
        pts = _sample_box_surface(length, web_t, web_h, web_c, n_points // 3)
        pts += _sample_box_surface(length, flange_w, flange_t, top_c, n_points // 3)
        pts += _sample_box_surface(length, flange_w, flange_t, bot_c, n_points // 3)
        return pts

    elif shape_type in ('cylinder', 'rod', 'shaft'):
        r = description.get('radius_m') or (description.get('diameter_m', 0) / 2.0)
        h = description.get('height_m') or description.get('length_m', 1.0)
        if not r:
            return []
        return _sample_cylinder_surface(r, h, c, n_points)

    elif shape_type in ('sphere', 'ball'):
        r = description.get('radius_m') or (description.get('diameter_m', 0) / 2.0)
        if not r:
            return []
        pts = []
        for i in range(n_points):
            # Fibonacci sphere sampling (uniform on sphere)
            golden = (1.0 + math.sqrt(5.0)) / 2.0
            theta = 2.0 * math.pi * i / golden
            phi = math.acos(1.0 - 2.0 * (i + 0.5) / n_points)
            pts.append((r * math.sin(phi) * math.cos(theta),
                         r * math.sin(phi) * math.sin(theta),
                         r * math.cos(phi)))
        return pts

    elif shape_type in ('box', 'plate', 'block'):
        x = description.get('x_m') or description.get('width_m', 1.0)
        y = description.get('y_m') or description.get('depth_m', 1.0)
        z = description.get('z_m') or description.get('height_m') or description.get('thickness_m', 1.0)
        return _sample_box_surface(x, y, z, c, n_points)

    return []


# ── Conversion pipeline ─────────────────────────────────────────────

def _quality_grade(efficiency):
    """Map volume efficiency to quality grade string."""
    if efficiency >= 0.95:
        return 'excellent'
    elif efficiency >= 0.85:
        return 'good'
    elif efficiency >= 0.70:
        return 'fair'
    else:
        return 'poor'


def _convert_pipe(desc):
    """Convert pipe description to primitives."""
    od = desc.get('outer_diameter_m')
    wall = desc.get('wall_thickness_m')
    length = desc.get('length_m', 1.0)
    material = desc.get('material', 'steel_mild')

    if od is None or wall is None:
        return None, "missing_dimensions: need outer_diameter_m and wall_thickness_m"

    r_outer = od / 2.0
    r_inner = r_outer - wall
    if r_inner <= 0:
        return None, "invalid_dimensions: wall thickness exceeds radius"

    target_vol = math.pi * (r_outer ** 2 - r_inner ** 2) * length
    s = Structure(target_volume=target_vol)
    s.add(Cylinder(r_outer, length), material)
    s.add(Cylinder(r_inner, length), 'air')
    return s, None


def _convert_bolt(desc):
    """Convert bolt description to primitives."""
    # Resolve ISO key if provided
    iso_key = desc.get('iso_key')
    if iso_key and iso_key in ISO_BOLTS:
        spec = ISO_BOLTS[iso_key]
        d = desc.get('diameter_m') or spec['major_d_m']
        head_d = desc.get('head_diameter_m', spec['head_d_m'])
        head_h = desc.get('head_height_m', spec['head_h_m'])
    else:
        d = desc.get('diameter_m') or desc.get('major_diameter_m')
        head_d = desc.get('head_diameter_m', d * 1.8 if d else None)
        head_h = desc.get('head_height_m', d * 0.65 if d else None)
    length = desc.get('length_m', 0.03)
    material = desc.get('material', 'steel_mild')

    if d is None:
        return None, "missing_dimensions: need diameter_m or iso_key"

    r_shank = d / 2.0
    r_head = head_d / 2.0

    target_vol = (math.pi * r_head ** 2 * head_h +
                  math.pi * r_shank ** 2 * length)
    s = Structure(target_volume=target_vol)
    s.add(Cylinder(r_head, head_h, center=(0, 0, length + head_h / 2)), material)
    s.add(Cylinder(r_shank, length, center=(0, 0, length / 2)), material)
    return s, None


def _convert_beam(desc):
    """Convert I-beam description to primitives."""
    depth = desc.get('depth_m')
    flange_w = desc.get('flange_width_m')
    web_t = desc.get('web_thickness_m')
    flange_t = desc.get('flange_thickness_m')
    length = desc.get('length_m', 1.0)
    material = desc.get('material', 'steel_mild')

    if not all([depth, flange_w, web_t, flange_t]):
        return None, "missing_dimensions: need depth_m, flange_width_m, web_thickness_m, flange_thickness_m"

    web_h = depth - 2.0 * flange_t
    target_vol = (web_t * web_h * length + 2.0 * flange_w * flange_t * length)

    s = Structure(target_volume=target_vol)
    s.add(Box(length, web_t, web_h, center=(length / 2, 0, depth / 2)), material)
    s.add(Box(length, flange_w, flange_t, center=(length / 2, 0, depth - flange_t / 2)), material)
    s.add(Box(length, flange_w, flange_t, center=(length / 2, 0, flange_t / 2)), material)
    return s, None


def _convert_cylinder(desc):
    """Convert simple cylinder description."""
    r = desc.get('radius_m') or (desc.get('diameter_m', 0) / 2.0)
    h = desc.get('height_m') or desc.get('length_m', 1.0)
    material = desc.get('material', 'steel_mild')

    if not r:
        return None, "missing_dimensions: need radius_m or diameter_m"

    target_vol = math.pi * r ** 2 * h
    s = Structure(target_volume=target_vol)
    s.add(Cylinder(r, h), material)
    return s, None


def _convert_sphere(desc):
    """Convert simple sphere description."""
    r = desc.get('radius_m') or (desc.get('diameter_m', 0) / 2.0)
    material = desc.get('material', 'steel_mild')

    if not r:
        return None, "missing_dimensions: need radius_m or diameter_m"

    target_vol = (4.0 / 3.0) * math.pi * r ** 3
    s = Structure(target_volume=target_vol)
    s.add(Sphere(r), material)
    return s, None


def _convert_box(desc):
    """Convert simple box/plate description."""
    x = desc.get('x_m') or desc.get('width_m') or desc.get('length_m', 1.0)
    y = desc.get('y_m') or desc.get('depth_m', 1.0)
    z = desc.get('z_m') or desc.get('height_m') or desc.get('thickness_m', 1.0)
    material = desc.get('material', 'steel_mild')

    target_vol = x * y * z
    s = Structure(target_volume=target_vol)
    s.add(Box(x, y, z), material)
    return s, None


# Converter dispatch
_CONVERTERS = {
    'pipe': _convert_pipe,
    'tube': _convert_pipe,
    'bolt': _convert_bolt,
    'screw': _convert_bolt,
    'fastener': _convert_bolt,
    'beam': _convert_beam,
    'i-beam': _convert_beam,
    'w-beam': _convert_beam,
    'cylinder': _convert_cylinder,
    'rod': _convert_cylinder,
    'shaft': _convert_cylinder,
    'sphere': _convert_sphere,
    'ball': _convert_sphere,
    'box': _convert_box,
    'plate': _convert_box,
    'block': _convert_box,
    'cone': lambda d: (_convert_cone(d)),
}


def _convert_cone(desc):
    """Convert simple cone description."""
    r = desc.get('radius_m') or (desc.get('diameter_m', 0) / 2.0)
    h = desc.get('height_m', 1.0)
    material = desc.get('material', 'steel_mild')

    if not r:
        return None, "missing_dimensions: need radius_m or diameter_m"

    target_vol = (1.0 / 3.0) * math.pi * r ** 2 * h
    s = Structure(target_volume=target_vol)
    s.add(Cone(r, h), material)
    return s, None


def convert_to_primitives(description, source_bytes=None):
    """Convert an external shape description to analytic primitives.

    Attempts to decompose any shape description into a Structure of
    analytic primitives (Sphere, Cylinder, Cone, Box, Torus).  Then
    verifies the conversion by sampling the target boundary and
    measuring how well our primitives match.

    If both volume_efficiency and boundary_score are high, the external
    data is redundant and can_discard is set to True.

    Args:
        description: dict with at least 'type' key and relevant dimensions.
            Example: {'type': 'pipe', 'outer_diameter_m': 0.05,
                      'wall_thickness_m': 0.003, 'length_m': 1.0}
        source_bytes: size of external data in bytes (for shape budget).
            If provided, sets the primitive count limit.

    Returns:
        dict with:
            structure: Structure instance (or None if conversion failed)
            volume_efficiency: float (0.0 to 1.0+)
            primitive_count: int
            conversion_quality: 'excellent'/'good'/'fair'/'poor'
            boundary: dict with max/mean/rms deviation, score, grade
            can_discard: bool — True if conversion is good enough to
                         safely discard the external source data
            shape_budget: int — max primitives allowed (from source_bytes)
            storage_ratio: float — original bytes / primitive bytes
            original_data: the input description (preserved until can_discard)
            provenance: source tag if provided
            notes: failure reason or None
    """
    shape_type = description.get('type', '').lower().strip()

    # Shape budget from source data size
    budget = shape_budget_from_source(source_bytes) if source_bytes else None

    converter = _CONVERTERS.get(shape_type)

    if converter is None:
        return {
            'structure': None,
            'volume_efficiency': 0.0,
            'primitive_count': 0,
            'conversion_quality': 'poor',
            'boundary': {'score': 0.0, 'grade': 'poor', 'n_points': 0,
                         'max_deviation_m': 0, 'mean_deviation_m': 0,
                         'rms_deviation_m': 0},
            'can_discard': False,
            'shape_budget': budget,
            'storage_ratio': 0.0,
            'original_data': description,
            'provenance': description.get('source', 'unknown'),
            'notes': f"unknown_shape_type: '{shape_type}' — no converter available",
        }

    structure, error = converter(description)

    if structure is None:
        return {
            'structure': None,
            'volume_efficiency': 0.0,
            'primitive_count': 0,
            'conversion_quality': 'poor',
            'boundary': {'score': 0.0, 'grade': 'poor', 'n_points': 0,
                         'max_deviation_m': 0, 'mean_deviation_m': 0,
                         'rms_deviation_m': 0},
            'can_discard': False,
            'shape_budget': budget,
            'storage_ratio': 0.0,
            'original_data': description,
            'provenance': description.get('source', 'unknown'),
            'notes': error,
        }

    eff = structure.volume_efficiency
    vol_quality = _quality_grade(eff)

    # Boundary agreement — sample the target and test our primitives
    sample_pts = sample_boundary(description, n_points=200)
    boundary = structure.boundary_agreement(sample_pts)

    # Combined quality: worst of volume and boundary grades
    grade_order = {'exact': 4, 'excellent': 3, 'good': 2, 'fair': 1, 'poor': 0}
    vol_rank = grade_order.get(vol_quality, 0)
    bnd_rank = grade_order.get(boundary['grade'], 0)
    combined_rank = min(vol_rank, bnd_rank)
    rank_to_grade = {4: 'exact', 3: 'excellent', 2: 'good', 1: 'fair', 0: 'poor'}
    combined_quality = rank_to_grade[combined_rank]

    # Composed SDF sanity check — verify interior/exterior consistency
    csdf = structure.composed_sdf()
    solid_check = True
    if sample_pts:
        # Points on the boundary should be near SDF=0
        # Points just inside should have negative SDF
        for px, py, pz in sample_pts[:20]:
            sdf_val = csdf.sdf(px, py, pz)
            if abs(sdf_val) > boundary.get('max_deviation_m', 0) * 2 + 0.001:
                solid_check = False
                break

    # Can we safely discard the external data?
    # Both volume and boundary must be excellent (or exact)
    can_discard = (eff >= 0.95 and boundary['score'] >= 0.95 and solid_check)

    # Storage ratio: how much we're saving
    # Estimate primitive storage: ~80 bytes per primitive (type + dims + center)
    prim_bytes = structure.shape_count * 80
    if source_bytes and source_bytes > 0:
        storage_ratio = source_bytes / max(prim_bytes, 1)
    else:
        storage_ratio = 0.0

    return {
        'structure': structure,
        'volume_efficiency': eff,
        'primitive_count': structure.shape_count,
        'conversion_quality': combined_quality,
        'boundary': boundary,
        'can_discard': can_discard,
        'shape_budget': budget,
        'storage_ratio': storage_ratio,
        'original_data': description,
        'provenance': description.get('source', 'unknown'),
        'notes': None,
    }
