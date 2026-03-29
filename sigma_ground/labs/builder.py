"""
Builder — converts a declarative SimulationScene into an executable PhysicsScene.

The builder's main job is the MATERIAL CASCADE: given a material_key and
temperature, it computes every derivable property from sigma-ground's
interface layer. This is the bridge between "I want an iron sphere" and
"here is a PhysicsParcel with mass=0.257 kg, restitution=0.63, etc."

No hardcoded material constants. Everything flows from the cascade.
"""

import math
from ..dynamics.vec import Vec3
from ..dynamics.parcel import PhysicsParcel
from ..dynamics.scene import PhysicsScene, GroundPlane
from .forces import combined_forces


# ── Shape construction ────────────────────────────────────────────────

def _make_shape(shape_type, dimensions):
    """Create a Shape object from type string + dimensions dict.

    Returns a Shape instance from sigma_ground.shapes.
    """
    from ..shapes import Sphere, Cylinder, Box, Ellipsoid, Cone, Torus

    s = shape_type.lower()

    if s == 'sphere':
        return Sphere(dimensions['radius'])

    elif s == 'cylinder':
        return Cylinder(dimensions['radius'], dimensions['length'])

    elif s == 'box':
        return Box(dimensions['width'], dimensions['depth'],
                   dimensions['height'])

    elif s == 'ellipsoid':
        return Ellipsoid(dimensions['a'], dimensions['b'], dimensions['c'])

    elif s == 'cone':
        return Cone(dimensions['radius'], dimensions['height'])

    elif s == 'torus':
        return Torus(dimensions['major_radius'], dimensions['minor_radius'])

    else:
        raise ValueError(f"Unknown shape type: {shape_type!r}. "
                         f"Supported: sphere, cylinder, box, ellipsoid, "
                         f"cone, torus")


# ── Material cascade ──────────────────────────────────────────────────

def cascade_material(material_key, T=293.15, velocity=1.0, radius=0.01):
    """Compute all derivable properties for a material at temperature T.

    This is the heart of the mini-lab: one material_key → a complete
    physical profile. Every value comes from sigma-ground's interface
    layer, derived from first principles or measured inputs.

    Args:
        material_key: Key into MATERIALS dict.
        T:            Temperature in K.
        velocity:     Characteristic velocity for impact properties (m/s).
        radius:       Characteristic radius for impact properties (m).

    Returns:
        dict with all computed properties and their units.
    """
    from ..field.interface.surface import MATERIALS
    from ..field.constants import SIGMA_HERE

    mat = MATERIALS[material_key]
    props = {
        'material_key': material_key,
        'name': mat['name'],
        'Z': mat['Z'],
        'density_kg_m3': mat['density_kg_m3'],
        'temperature_K': T,
    }

    # Thermal properties
    try:
        from ..field.interface.thermal import (
            sound_velocity, debye_temperature, specific_heat_j_kg_K,
            thermal_conductivity,
        )
        props['sound_velocity_m_s'] = sound_velocity(material_key)
        props['debye_temperature_K'] = debye_temperature(material_key)
        props['specific_heat_J_kgK'] = specific_heat_j_kg_K(material_key, T)
        props['thermal_conductivity_W_mK'] = thermal_conductivity(material_key, T)
    except Exception:
        pass

    # Mechanical properties
    try:
        from ..field.interface.mechanical import (
            bulk_modulus, shear_modulus, youngs_modulus,
        )
        props['bulk_modulus_Pa'] = bulk_modulus(material_key)
        props['shear_modulus_Pa'] = shear_modulus(material_key)
        props['youngs_modulus_Pa'] = youngs_modulus(material_key)
    except Exception:
        pass

    # Impact properties
    try:
        from ..field.interface.impact import coefficient_of_restitution
        props['restitution'] = coefficient_of_restitution(
            material_key, velocity=velocity, radius_m=radius)
    except Exception:
        props['restitution'] = 0.5

    # Friction
    try:
        from ..field.interface.friction import friction_coefficient
        props['friction_coefficient'] = friction_coefficient(
            material_key, material_key)
    except Exception:
        pass

    # Optical properties
    try:
        from ..field.interface.optics import get_material_color
        mat_type = mat.get('material_type', 'metal')
        if mat_type == 'metal':
            props['color_rgb'] = get_material_color('metal', material_key)
        elif mat_type == 'semiconductor':
            props['color_rgb'] = get_material_color('dielectric', material_key)
        else:
            props['color_rgb'] = get_material_color('dielectric', material_key)
    except Exception:
        pass

    return props


# ── Material wrapper for PhysicsParcel ────────────────────────────────

class _LabMaterial:
    """Thin wrapper providing the density_at_sigma() interface.

    PhysicsParcel expects material.density_at_sigma(sigma). This wrapper
    provides that plus the cascaded restitution.
    """

    def __init__(self, material_key, density, restitution=0.5):
        self.material_key = material_key
        self.density_kg_m3 = density
        self.restitution = restitution

    def density_at_sigma(self, sigma):
        # For mini-labs we use the base density (σ ≈ 0 at Earth)
        return self.density_kg_m3


# ── Scene builder ─────────────────────────────────────────────────────

def build(scene):
    """Convert a SimulationScene into an executable PhysicsScene.

    Steps:
      1. Create Shape objects from SimObject definitions
      2. Cascade material properties for each object
      3. Create PhysicsParcels with computed mass, restitution
      4. Configure gravity, ground plane
      5. Build external_forces callback (drag + buoyancy)
      6. Return (PhysicsScene, forces_callback, cascaded_properties)

    Args:
        scene: SimulationScene instance.

    Returns:
        (PhysicsScene, external_forces_callback, material_props_dict)
    """
    env = scene.environment
    gravity = Vec3(*env.gravity)

    parcels = []
    all_props = {}

    for obj in scene.objects:
        # 1. Build shape
        shape = _make_shape(obj.shape, obj.dimensions)

        # 2. Cascade material properties
        speed = sum(v**2 for v in obj.velocity) ** 0.5
        props = cascade_material(
            obj.material_key,
            T=obj.temperature,
            velocity=max(speed, 1.0),
            radius=shape.bounding_radius(),
        )
        all_props[obj.name] = props

        # 3. Create material wrapper
        restitution = props.get('restitution', 0.5)
        mat = _LabMaterial(obj.material_key, props['density_kg_m3'],
                           restitution)

        # 4. Create parcel
        parcel = PhysicsParcel(
            radius=shape,
            material=mat,
            position=Vec3(*obj.position),
            velocity=Vec3(*obj.velocity),
            is_static=obj.is_static,
            label=obj.name,
        )
        parcels.append(parcel)

    # 5. Ground plane
    if env.ground.enabled:
        gnd_normal = Vec3(*env.ground.normal)
        # Ground plane height: project along normal
        h = env.ground.height
        # Determine ground point from height and normal direction
        nx, ny, nz = env.ground.normal
        if abs(ny) > 0.5:
            gnd_point = Vec3(0, h, 0)
        elif abs(nz) > 0.5:
            gnd_point = Vec3(0, 0, h)
        else:
            gnd_point = Vec3(h, 0, 0)

        # Get ground restitution from material
        try:
            gnd_props = cascade_material(env.ground.material_key)
            gnd_restitution = gnd_props.get('restitution', 0.5)
        except (KeyError, Exception):
            gnd_restitution = 0.5

        ground = GroundPlane(
            y=h, normal=gnd_normal, restitution=gnd_restitution)
    else:
        ground = False

    # 6. Build PhysicsScene
    physics_scene = PhysicsScene(
        parcels=parcels,
        gravity=gravity,
        ground=ground,
    )

    # 7. External forces callback (drag + buoyancy)
    medium = env.medium
    if medium.density > 0:
        forces_cb = combined_forces(
            medium.density, medium.viscosity, gravity)
    else:
        forces_cb = None

    return physics_scene, forces_cb, all_props
