"""Deckard general primitive kit — solid objects from sphere/cylinder/box/cone.

Each single primitive's analytic mass must equal ``density × shape.volume()``,
the independent SDF integrator must agree (self-check), and material/density
queries must resolve real geometry. Deterministic, offline.
"""
import math

import pytest

from sigma_ground.deckard import compile
from sigma_ground.deckard.schema import ConstructSpec, Part, Fact
from sigma_ground.deckard.sources import density_of


def _solid(shape, dims, material):
    part = Part("body", shape, {k: Fact(v, "test", "", 0.5) for k, v in dims.items()},
                material, density_of(material))
    return ConstructSpec(name=f"{material} {shape}", kind="composite", identified=True,
                         parts=[part], sources=[{"name": "test", "license": ""}])


CASES = [
    ("cylinder", {"radius_m": 0.005, "height_m": 0.30}, "steel",
     lambda d: math.pi * d["radius_m"] ** 2 * d["height_m"]),
    ("sphere", {"radius_m": 0.008}, "glass",
     lambda d: 4.0 / 3.0 * math.pi * d["radius_m"] ** 3),
    ("box", {"x_m": 0.016, "y_m": 0.016, "z_m": 0.016}, "aluminium",
     lambda d: d["x_m"] * d["y_m"] * d["z_m"]),
]


@pytest.mark.parametrize("shape,dims,material,vol", CASES)
def test_primitive_mass_is_density_times_volume(shape, dims, material, vol):
    c = compile(_solid(shape, dims, material), resolution=48)
    assert abs(c.mass_kg - density_of(material).value * vol(dims)) < 1e-9   # analytic = ρ·V
    assert c.validation["passed"], c.validation["note"]                    # integrator agrees
    assert c.validation["mass_residual"] < 0.05


def test_primitive_density_is_grounded_not_estimated():
    spec = _solid("sphere", {"radius_m": 0.01}, "glass")
    assert not spec.parts[0].density.estimated          # cited from our data
    c = compile(spec, resolution=40)
    assert c.density_by_label["body"] == density_of("glass").value


def test_primitive_queries_resolve_real_geometry():
    c = compile(_solid("box", {"x_m": 0.02, "y_m": 0.02, "z_m": 0.02}, "aluminium"),
                resolution=40)
    assert c.material_at(0.0, 0.0, 0.0) == "body"
    assert c.density_at(0.0, 0.0, 0.0) == density_of("aluminium").value
    assert c.material_at(0.05, 0.0, 0.0) is None         # outside the solid


def test_unsupported_shape_raises():
    spec = ConstructSpec(name="weird", kind="composite",
                         parts=[Part("p", "dodecahedron", {"r": Fact(1.0)}, "steel", Fact(1000.0))])
    with pytest.raises(ValueError):
        compile(spec, resolution=16)
