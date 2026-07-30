"""Per-cell scalar state on a voxel grid.

Before this slot existed a cell could say only "which material am I"
(label_grid) and "how far am I from the surface" (sdf_grid). There was
nowhere to put a number that VARIES across the body -- temperature, stress,
damage, phase fraction -- so every such quantity had to remain a whole-body
constant. That is why a conduction solver could produce a temperature grid
while no voxel body was able to carry one.
"""
import numpy as np
import pytest

from sigma_ground.kernel.voxel import Voxel


def _ball(n=9, r=3.0):
    """A small sphere SDF grid centred in an n^3 box, 1 cm cells."""
    c = (n - 1) / 2.0
    idx = np.indices((n, n, n)).astype(float)
    d = np.sqrt(sum((idx[a] - c) ** 2 for a in range(3))) - r
    return d * 0.01


def test_voxel_without_fields_is_unchanged():
    """Additive: a voxel built the old way carries an empty registry, and
    asking for a field returns the caller's default -- never a silent 0.0,
    which would read as 'measured cold' rather than 'not simulated'."""
    v = Voxel(_ball(), 0.01)
    assert v.scalar_fields == {}
    assert v.field_at("temperature_k", 0.0, 0.0, 0.0) is None
    assert v.field_at("temperature_k", 0.0, 0.0, 0.0, default=293.15) == 293.15


def test_voxel_carries_and_reads_back_a_per_cell_field():
    n = 9
    g = _ball(n)
    # a real gradient: hot on one face, cold on the other
    T = np.zeros((n, n, n))
    for i in range(n):
        T[i, :, :] = 300.0 + 50.0 * i
    v = Voxel(g, 0.01, scalar_fields={"temperature_k": T})
    assert set(v.scalar_fields) == {"temperature_k"}
    lo = v.field_at("temperature_k", *[c - 0.04 for c in v.center][:1] + list(v.center[1:]))
    hi = v.field_at("temperature_k", v.center[0] + 0.04, v.center[1], v.center[2])
    assert hi > lo                                    # the gradient survives the round trip
    assert v.field_at("temperature_k", *v.center) == pytest.approx(300.0 + 50.0 * 4)


def test_field_at_is_nearest_cell_and_bounded():
    """Nearest-cell, matching material_at rather than the renderer's trilinear
    read -- a physics query wants the cell's own value, not a blend across a
    boundary it may be straddling. Outside the grid yields the default."""
    n = 5
    T = np.arange(n ** 3, dtype=float).reshape((n, n, n))
    v = Voxel(_ball(n), 0.01, scalar_fields={"temperature_k": T})
    assert v.field_at("temperature_k", *v.center) == T[2, 2, 2]
    assert v.field_at("temperature_k", 99.0, 0.0, 0.0, default=-1.0) == -1.0
    assert v.field_at("no_such_quantity", *v.center, default="absent") == "absent"


def test_multiple_quantities_coexist():
    """The registry is free-form: physics can carry a quantity whether or not
    anything knows how to DRAW it yet (only temperature_k has a renderer)."""
    n = 5
    g = _ball(n)
    v = Voxel(g, 0.01, scalar_fields={
        "temperature_k": np.full((n, n, n), 400.0),
        "von_mises_pa": np.full((n, n, n), 1.2e8),
        "damage": np.zeros((n, n, n)),
    })
    assert set(v.scalar_fields) == {"temperature_k", "von_mises_pa", "damage"}
    assert v.field_at("von_mises_pa", *v.center) == pytest.approx(1.2e8)
    assert v.field_at("damage", *v.center) == 0.0


def test_voxelfield_passes_scalar_fields_through_to_voxel():
    """VoxelField.to_voxel() must not drop the per-cell state on the floor."""
    from sigma_ground.deckard.voxelize import VoxelField
    n = 5
    T = np.full((n, n, n), 350.0)
    vf = VoxelField(sdf_grid=_ball(n), label_grid=np.ones((n, n, n), dtype=int),
                    voxel_size=0.01, center=(0.0, 0.0, 0.0), materials=["air", "iron"],
                    volume_m3=1e-4, mass_kg=0.8, com_m=(0.0, 0.0, 0.0),
                    inertia_kgm2=(1e-5, 1e-5, 1e-5), interfaces={}, free_surfaces={},
                    watertight_frac=1.0, confidence=0.9,
                    scalar_fields={"temperature_k": T})
    v = vf.to_voxel()
    assert v.field_at("temperature_k", 0.0, 0.0, 0.0) == pytest.approx(350.0)


def test_voxelfield_defaults_to_no_fields():
    """A field built without per-cell state behaves exactly as before."""
    from sigma_ground.deckard.voxelize import VoxelField
    n = 5
    vf = VoxelField(sdf_grid=_ball(n), label_grid=np.ones((n, n, n), dtype=int),
                    voxel_size=0.01, center=(0.0, 0.0, 0.0), materials=["air", "iron"],
                    volume_m3=1e-4, mass_kg=0.8, com_m=(0.0, 0.0, 0.0),
                    inertia_kgm2=(1e-5, 1e-5, 1e-5), interfaces={}, free_surfaces={},
                    watertight_frac=1.0, confidence=0.9)
    assert vf.scalar_fields is None
    assert vf.to_voxel().scalar_fields == {}
