"""PlaygroundWorld (SKIPPYCRAFT_PLAN 10b, P2): heat, phase, liquid momentum and rigid bodies on one clock.

Done when: molten lead off a hot plate flows, a gold block sinks in it with momentum, the melt freezes where it stops,
and the books close (mass exact; thermal energy against radiated + vented + surroundings)."""
import numpy as np
import pytest

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
from sigma_ground.materia.playground import PlaygroundWorld


def _books(w, M0, E0):
    L = w.ledger
    assert w.mass_kg() == pytest.approx(M0, rel=1e-12)
    assert (w.thermal_J() + L["radiated_J"] + L["vented_J"] + L["to_surroundings_J"] - L["heat_from_motion_J"]
            == pytest.approx(E0, rel=1e-9))


@pytest.fixture(scope="module")
def melted():
    """A tungsten plate at 2200 K under a lead slab, a gold cube on the lead: 10 s."""
    field, _ = field_from_voxel_exec('box(0,0,0,11,0,11,"tungsten"); box(2,1,2,9,3,9,"lead"); box(5,4,5,6,5,6,"gold")',
                                     block_m=0.01)
    T = np.where(field.label_grid == field.materials.index("tungsten"), 2200.0, 293.15)
    w = PlaygroundWorld(field, liquid="lead", temperature_k=T, bodies=("gold", "lead"), env="ISM")
    gold = next(b for b in w.bodies if b.material == "gold")
    start = {"M": w.mass_kg(), "E": w.thermal_J(), "gold_z": gold.x[2], "plate_top_m": 0.03,
             "lead_x": np.ptp(np.argwhere(w.label == w.materials.index("lead"))[:, 0])}
    w.run(200, 0.05)
    return w, start


def test_the_hot_plate_melts_the_whole_slab(melted):
    w, s = melted
    assert w.ledger["melted_cells"] >= 192                             # every slab cell (and lead that refroze on the plate, again)
    assert not any(b.material == "lead" for b in w.bodies)


def test_the_melt_flows_out(melted):
    w, s = melted
    lead = np.argwhere(w.label == w.materials.index("lead"))
    assert np.ptp(lead[:, 0]) > s["lead_x"]                              # wider than the slab it came from


def test_gold_sinks_through_the_melt_to_the_plate(melted):
    w, s = melted
    gold = next(b for b in w.bodies if b.material == "gold")
    assert gold.x[2] < s["gold_z"] - 0.02                               # fell through the lead
    assert gold.x[2] == pytest.approx(s["plate_top_m"] + 0.01, abs=0.002)  # resting on the plate (2 cm cube)


def test_the_books_close(melted):
    w, s = melted
    _books(w, s["M"], s["E"])
    assert w.ledger["radiated_J"] > 0 and w.ledger["dissipated_J"] >= 0


@pytest.fixture(scope="module")
def frozen():
    """A 4 cm cube of lead at 700 K (molten) dropped on a cold copper plate: 5 s."""
    field, _ = field_from_voxel_exec('box(0,0,0,13,0,13,"copper"); box(5,4,5,8,7,8,"lead")', block_m=0.01)
    T = np.where(field.label_grid == field.materials.index("copper"), 293.15, 700.0)
    w = PlaygroundWorld(field, liquid="lead", temperature_k=T, env="ISM")
    start = {"M": w.mass_kg(), "E": w.thermal_J(), "particles": len(w.flip.pos)}
    w.run(100, 0.05)
    return w, start


def test_the_melt_freezes_where_it_stops(frozen):
    w, s = frozen
    assert s["particles"] > 0 and len(w.flip.pos) == 0                    # all of it froze
    lead = w.materials.index("lead")
    walls = w.static & (w.label == lead)
    assert walls.any()
    assert np.argwhere(walls)[:, 2].max() <= 4                           # a sheet on the plate (top at z = 2), not a cube
    assert w.ledger["frozen_cells"] == int(walls.sum())


def test_a_thin_frozen_sheet_is_drawn_with_its_mass(frozen):
    w, s = frozen
    lead = w.materials.index("lead")
    full = w.mat[lead]["rho"] * w.V
    drawn = int((w.drawn == lead).sum())
    assert drawn == int(round(w.Ms[w.static & (w.label == lead)].sum() / full)) == 64


def test_the_frozen_books_close(frozen):
    w, s = frozen
    _books(w, s["M"], s["E"])


def test_a_body_falls_in_air_and_stops_on_the_floor():
    field, _ = field_from_voxel_exec('box(0,0,0,5,0,5,"granite"); box(2,6,2,3,7,3,"gold")', block_m=0.05)
    w = PlaygroundWorld(field, liquid="lead", bodies=("gold",), env="ISM")  # no lead in it: a liquid that never appears
    b = w.bodies[0]
    z0 = b.x[2]
    w.run(10, 0.05)
    assert b.x[2] < z0 and b.x[2] == pytest.approx(0.20, abs=0.002)    # landed on the granite (top at 15 cm)
    assert np.allclose(b.v, 0.0, atol=1e-6)


def test_live_cards_drive_the_heat():
    """The plate radiates with its emissivity at its own temperature (tungsten: higher hot)."""
    field, _ = field_from_voxel_exec('box(0,0,0,3,0,3,"tungsten")', block_m=0.01)
    hot = PlaygroundWorld(field, liquid="tungsten", temperature_k=2500.0, env="ISM")
    cold = PlaygroundWorld(field, liquid="tungsten", temperature_k=1000.0, env="ISM")
    hot.step(0.01)
    cold.step(0.01)
    from sigma_ground.field.interface.thermal import emissivity
    sigma_T4 = (2500.0 / 1000.0) ** 4
    ratio = hot.ledger["radiated_J"] / cold.ledger["radiated_J"]
    assert ratio == pytest.approx(sigma_T4 * emissivity("tungsten", 2500.0) / emissivity("tungsten", 1000.0), rel=0.02)
