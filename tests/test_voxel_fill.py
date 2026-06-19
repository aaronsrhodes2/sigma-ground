"""D-vox: fill a real cavity under gravity + reconcile-and-report.

The SHAPE is authoritative (the Captain's rule): we never place more liquid than
the cavity holds. ``fill_cavity`` finds the gravity-trapped void (the cells poured
liquid rests in — bottom-first, capped at the rim), fills to ``min(requested,
capacity)``, and reports requested vs capacity vs actually-filled so Mentat amends
the sim. Tested on a synthetic glass cup (a labeled grid), so no PartNet data is
needed; numpy/scipy are the opt-in [shapes] deps.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.deckard.voxelize import (
    _finalize, fill_cavity, _trapped_cavity, construct_from_field,
)

PITCH = 0.01                                   # 1 cm cells → one cell = 1 cm³
_DENS = {"glass": 2500.0, "water": 1000.0, "mercury": 13534.0}


def _density(name):
    return _DENS.get(name, 1000.0)


def _cup():
    """An open glass cup: solid walls + floor, a hollow interior open at the top,
    with a void margin all around so liquid can spill over the rim. Interior cavity
    is x6:8, y6:8, z4:12 = 2×2×8 = 32 cells = 32 cm³ at 1 cm pitch."""
    nx, ny, nz = 14, 14, 18
    label = np.zeros((nx, ny, nz), np.int32)
    label[4:10, 4:10, 2:4] = 1                 # floor slab (z2:4)
    label[4:6, 4:10, 2:12] = 1                 # -x wall (rim top z=12)
    label[8:10, 4:10, 2:12] = 1                # +x wall
    label[4:10, 4:6, 2:12] = 1                 # -y wall
    label[4:10, 8:10, 2:12] = 1                # +y wall
    lo = np.array([-nx * PITCH / 2, -ny * PITCH / 2, -nz * PITCH / 2])
    field = _finalize(label, ["air", "glass"], PITCH, lo, _density, 0.0)
    return field


def test_cavity_is_the_gravity_trapped_interior():
    field = _cup()
    mask, cells = _trapped_cavity(field.label_grid)
    assert cells == 32                                   # the 2×2×8 interior
    zs = np.argwhere(mask)[:, 2]
    assert zs.min() >= 4 and zs.max() < 12               # bottom up to (below) the rim
    assert mask.sum() * PITCH ** 3 == pytest.approx(32e-6)


def test_partial_fill_settles_at_the_bottom():
    field = _cup()
    new, recon = fill_cavity(field, "water", requested_m3=16e-6, density_of=_density)
    assert recon["filled_m3"] == pytest.approx(16e-6)    # exactly what was asked
    assert recon["fill_fraction"] == pytest.approx(0.5)
    assert recon["deferred_to_shape"] is False
    # gravity: water occupies the LOWER half of the cavity, not the top
    wid = new.materials.index("water")
    wz = np.argwhere(new.label_grid == wid)[:, 2]
    assert wz.min() == 4 and wz.max() <= 8
    # mass gained = ρ_water · V, exactly
    assert new.mass_kg - field.mass_kg == pytest.approx(1000.0 * 16e-6)


def test_brim_fill_when_no_quantity_requested():
    field = _cup()
    new, recon = fill_cavity(field, "water", density_of=_density)
    assert recon["filled_m3"] == pytest.approx(32e-6)    # the full cavity
    assert recon["fill_fraction"] == pytest.approx(1.0)
    assert recon["capacity_m3"] == pytest.approx(32e-6)


def test_overfill_defers_to_the_shape_and_reports_the_delta():
    field = _cup()
    new, recon = fill_cavity(field, "mercury", requested_m3=50e-6, density_of=_density)
    # never exceeds capacity — the shape wins
    assert recon["capacity_m3"] == pytest.approx(32e-6)
    assert recon["filled_m3"] == pytest.approx(32e-6)
    assert recon["shortfall_m3"] == pytest.approx(18e-6)
    assert recon["deferred_to_shape"] is True
    assert "deferred to shape" in recon["note"]
    # mercury actually placed → its mass shows up
    assert new.mass_kg - field.mass_kg == pytest.approx(13534.0 * 32e-6)


def test_fill_creates_the_container_liquid_interface():
    field = _cup()
    new, _ = fill_cavity(field, "mercury", requested_m3=50e-6, density_of=_density)
    # the mercury now touches the glass walls — a material-pair interface falls
    # straight out of the voxel adjacency scan (no separate contact detector)
    assert ("glass", "mercury") in new.interfaces
    assert new.interfaces[("glass", "mercury")] > 0
    # mercury also has a free top surface (mercury ↔ air)
    assert new.free_surfaces.get("mercury", 0) > 0


def test_reconciliation_surfaces_in_the_construct():
    field = _cup()
    new, recon = fill_cavity(field, "mercury", requested_m3=50e-6, density_of=_density)
    c = construct_from_field("a glass cup of mercury", new)
    vr = c.validation.get("volume_reconciliation")
    assert vr is not None
    assert vr["deferred_to_shape"] is True
    assert vr["fill_material"] == "mercury"
    assert vr["cell_quantum_m3"] == pytest.approx(PITCH ** 3)
