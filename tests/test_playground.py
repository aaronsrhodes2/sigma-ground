"""PlaygroundWorld (SKIPPYCRAFT_PLAN 10b, P2): heat, phase, liquid momentum and rigid bodies on one clock.

Done when: molten lead off a hot plate flows, a gold block sinks in it with momentum, the melt freezes where it stops,
and the books close (mass exact; thermal energy against radiated + vented + surroundings)."""
import numpy as np
import pytest

from sigma_ground.deckard.voxel_exec import field_from_voxel_exec
from sigma_ground.materia.playground import PlaygroundWorld


def _books(w, M0, E0):
    L = w.ledger
    assert w.mass_kg() + L["left_kg"] + L["ejected_kg"] == pytest.approx(M0, rel=1e-12)
    assert (w.thermal_J() + L["radiated_J"] + L["vented_J"] + L["to_surroundings_J"] + L["left_J"] + L["ejected_J"]
            - L["heat_from_motion_J"] - L["impact_heat_J"] == pytest.approx(E0, rel=1e-9))


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


# ── P4: a shot as an event in the world ─────────────────────────────────────────────────────────────────────────────
def _wall_and_block():
    """A granite floor, a static granite wall, and a free steel block in front of it, 1 cm cells."""
    field, _ = field_from_voxel_exec('box(0,0,0,29,0,29,"granite"); box(0,1,24,29,12,27,"granite"); '
                                     'box(10,1,10,19,10,19,"steel")', block_m=0.01)
    w = PlaygroundWorld(field, liquid="steel_mild", bodies=("steel_mild",), env="STP")
    return w


def _fixed_solve(depth_m, dp, dE, penetrated=False):
    """A stand-in for the game: every layer the shot reaches loses ``dp``, ``dE`` and is reached ``depth_m`` deep."""
    def solve(layers):
        out = []
        for L in layers:
            if L.get("gap"):
                out.append({"gap": True})
                continue
            out.append({"penetrated": penetrated, "depth_m": min(depth_m, L["thickness_m"]), "dp_Ns": dp, "dE_J": dE,
                        "temperature_k": 900.0, "notes": "fixed"})
            if not penetrated:
                break
        return out
    return solve


def test_a_shot_pushes_the_block_with_the_solves_momentum_and_heats_it_with_the_rest():
    w = _wall_and_block()
    b = w.bodies[0]
    M0, E0 = w.mass_kg(), w.thermal_J()
    start = np.array([b.x[0], w.label.shape[1] * w.pitch - 1e-4, b.x[2]])
    rec = w.impact(start, (0, -1, 0), _fixed_solve(0.03, 4.0, 5000.0), radius_m=0.004)
    assert len(rec) == 1 and rec[0]["owner"] == "body" and not rec[0]["penetrated"]
    hit = w.bodies[0]
    assert hit.mass * hit.v == pytest.approx(np.array([0.0, -4.0, 0.0]), abs=1e-9)    # momentum, along the shot
    assert rec[0]["heat_J"] == pytest.approx(5000.0 - 0.5 * 4.0 ** 2 / hit.mass, rel=1e-9)
    assert rec[0]["cells_out"] == 3                                     # a 1-cell bore, 3 cm deep
    L = w.ledger
    assert w.mass_kg() + L["ejected_kg"] == pytest.approx(M0, rel=1e-12)
    assert (w.thermal_J() + L["ejected_J"] - L["impact_heat_J"] == pytest.approx(E0, rel=1e-12))
    w.run(10, 0.05)                                                     # into the wall and still: the books hold
    _books(w, M0, E0)


def test_a_shot_into_a_wall_puts_its_momentum_into_the_ground():
    w = _wall_and_block()
    start = np.array([0.025, w.label.shape[1] * w.pitch - 1e-4, 0.08])     # past the block, onto the wall
    rec = w.impact(start, (0, -1, 0), _fixed_solve(0.02, 4.0, 5000.0), radius_m=0.004)
    assert rec[0]["owner"] == "wall" and rec[0]["heat_J"] == pytest.approx(5000.0)
    assert all(np.allclose(b.v, 0) for b in w.bodies)


def test_the_bore_is_as_hot_as_the_game_says():
    """The heated volume is the game's own (its bore, as deep as the round got): with the real solve, the hottest
    cell's rise matches the game's within the difference between its 300 K card and the live one."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))
    import shooting_range
    w = _wall_and_block()
    b = w.bodies[0]
    start = np.array([b.x[0], w.label.shape[1] * w.pitch - 1e-4, b.x[2]])
    rec = w.impact(start, (0, -1, 0), shooting_range.layer_solver("m2_browning"), shooting_range.hole_radius_m("m2_browning"))
    r = rec[0]
    assert r["dp_Ns"] == pytest.approx(0.046 * 895, rel=0.02)          # a .50 BMG round stopped: all its momentum
    hot = float(w.bodies[0].T.max())
    assert (hot - 293.15) == pytest.approx(r["game_T_K"] - 300.0, rel=0.35)
