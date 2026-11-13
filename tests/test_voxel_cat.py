"""The cat stage (materia/voxel_cat.py) and its place in the theater: what it draws follows the physics of
field/interface/schrodinger_cat.py, and the theater can show it."""
import sys
from pathlib import Path

import numpy as np
import pytest

from sigma_ground.field.interface import schrodinger_cat as sc
from sigma_ground.materia import voxel_cat as vc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))


def _run(box, ticks=vc.TICKS):
    for _ in range(ticks):
        box.step(0.1)
    return box


def test_every_material_has_a_plain_block_and_the_grid_is_the_declared_size():
    assert set(vc.LOOKS) == set(vc.MATERIALS[1:])
    b = vc.CatBox(seed=1)
    assert b.label.shape == (vc.W, vc.D, vc.H) and int(b.label.max()) < len(vc.MATERIALS)


def test_the_born_rule_draws_the_observers_outcome_with_the_samples_odds():
    outcomes = [vc.CatBox(seed=s).outcome for s in range(1000, 2000)]
    assert outcomes.count("dead") / 1000 == pytest.approx(sc.decay_probability(sc.WINDOW_S), abs=0.05)


def test_the_atom_decays_inside_the_hour_only_in_the_dead_branch():
    for s in range(30):
        b = vc.CatBox(seed=s)
        assert (b.outcome == "dead") == (b.t_decay <= sc.WINDOW_S)


def test_sealed_the_whole_box_stays_one_state_in_principle_and_is_a_mixture_in_practice():
    b = _run(vc.CatBox(seed=2))
    assert b.phase() == "test" and b.trials == (vc.TEST_TICKS + 1) * vc.TRIALS_PER_TICK
    assert b.coh_principle == pytest.approx(1.0) and b.coh_practice == pytest.approx(0.0, abs=1e-9)
    assert b.yes_in_principle == b.trials                                    # the outside test always says "superposition"
    assert b.yes_in_practice / b.trials == pytest.approx(0.5, abs=0.02)      # and in practice it cannot tell


def test_a_pinhole_tells_the_outside_and_the_principle_goes_with_it():
    b = vc.CatBox(seed=2)
    b.press("leak")
    _run(b)
    assert b.coh_principle == pytest.approx(0.0, abs=1e-9)
    assert b.yes_in_principle / b.trials == pytest.approx(0.5, abs=0.02)


def test_the_zoom_goes_through_the_air_inside_recording_the_cat():
    b = vc.CatBox(seed=2)
    seen = {}
    for _ in range(vc.TICKS):
        b.step(0.1)
        if b.phase() == "zoom":
            seen[round(b.log_t())] = b.coh_practice
    assert seen[-33] == pytest.approx(1.0, abs=1e-3)          # before the first collision the chain is coherent
    assert seen[-26] < 1e-3                                  # and the air has recorded it by 1e-26 s, a person's 0.1 s is 25 decades later
    assert seen[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_vacuum_lets_the_cat_stay_coherent_for_longer_but_its_own_light_still_records_it():
    b = vc.CatBox(seed=2)
    b.press("air")
    assert b.tau_in == pytest.approx(sc.decoherence_times(vacuum=True)["total_s"]) and b.tau_in > 1e4 * vc.CatBox(seed=2).tau_in


def test_opening_the_door_removes_the_wall_and_the_outside_learns_the_outcome():
    b = vc.CatBox(seed=3)
    door = (7, vc.JF, 4)
    assert b.label[door] == vc.M["wall"]
    _run(b, vc.HOUR_TICKS + 10)
    b.press("open")
    assert b.label[door] == 0 and b.coh_principle == pytest.approx(0.0, abs=1e-9)
    assert b.outcome.upper() in b.press("open")


def test_the_cat_is_drawn_dead_only_once_the_atom_has_decayed_in_the_dead_branch():
    b = vc.CatBox(seed=next(s for s in range(50) if vc.CatBox(seed=s).outcome == "dead"))
    assert b.label[7, 7, 3] != vc.M["cat dead"] and b.label[6, 6, 1] == vc.M["cat alive"]       # sitting
    _run(b, vc.HOUR_TICKS)
    assert b.label[6, 6, 1] == vc.M["cat dead"] and b.label[4, 4, 2] == vc.M["atom decayed"]       # on its side
    a = vc.CatBox(seed=next(s for s in range(50) if vc.CatBox(seed=s).outcome == "alive"))
    _run(a, vc.HOUR_TICKS)
    assert a.label[6, 6, 1] == vc.M["cat alive"] and a.label[4, 4, 2] == vc.M["atom"]


def test_the_stage_is_wired_into_the_theater():
    import theater
    st = theater.STAGES["cat"]
    numbers = [s["request"] for s in theater.STAGES.values()] + [n for s in theater.STAGES.values() for _, n in s.get("extra", ())]
    assert theater.STOP not in numbers and len(set(numbers)) == len(numbers)          # one button, one request number
    assert {*theater.CAT_BUTTONS, theater.CAT_INSIDE, theater.CAT_OUTSIDE} == {n for _, n in st["extra"]}
    looks = theater.cat_looks()
    _, world = theater.build("cat")
    sent = []
    th = theater.Theater(world, looks, st["origin"], sent.extend)
    th.open()
    assert th.draw() == 0 and len(sent) > 100
    ox, oy, oz = th.mc(*world.OBSERVER_CELL)                          # he stands inside the box's walls
    x0, y0, z0 = st["origin"]
    assert x0 <= ox <= x0 + 13 and y0 <= oy <= y0 + vc.BOX_H and z0 <= oz <= z0 + vc.JF
    assert all(len(c) for c in theater.cat_captions(th, world))


def test_no_desk_is_inside_the_cat_stage_and_the_cat_desk_is_clear_of_every_scene_origin():
    import theater
    st = theater.STAGES["cat"]
    _, world = theater.build("cat")
    th = theater.Theater(world, theater.cat_looks(), st["origin"], lambda c: None)
    nx, ny, nz = world.label.shape
    xs = sorted(th.mc(i, 0, 0)[0] for i in (0, nx - 1))
    ys = sorted(th.mc(0, 0, k)[1] for k in (th.k0, nz - 1))
    zs = sorted(th.mc(0, j, 0)[2] for j in (0, ny - 1))
    for other in theater.STAGES.values():
        x, y, z = other["desk"]
        for bx, by, bz in [(x + dx, y + dy, z + dz) for dx in range(-2, 12) for dy in (-1, 0) for dz in range(-4, 1)]:
            assert not (xs[0] <= bx <= xs[1] and ys[0] <= by <= ys[1] and zs[0] <= bz <= zs[1]), (bx, by, bz)
    dx, dy, dz = st["desk"]                                        # nor is the cat's desk in anyone else's grid: it stands 38 blocks off the board
    assert zs[0] - dz >= 30


def test_the_osmium_lump_at_zero_kelvin_stays_coherent_until_a_warm_observer_steps_in():
    b = vc.CatBox(seed=2)
    b.press("subject")
    assert b.subject == "osmium" and b.vacuum and b.tau_in == float("inf")
    _run(b)
    assert b.coh_practice == pytest.approx(1.0) and b.coh_principle == pytest.approx(1.0)
    assert b.yes_in_practice == b.trials                                   # even in practice the outside can see the superposition
    msg = b.set_observer(True)
    assert "5e-19" in msg and b.tau_in == pytest.approx(5.4e-19, rel=0.15)
    b.step(0.1)
    assert b.coh_practice == pytest.approx(0.0, abs=1e-9)
    assert b.set_observer(False).startswith("You are outside") and b.tau_in == float("inf")


def test_the_lump_temperature_steps_through_the_desks_values_and_a_pinhole_tells_nothing_at_zero_kelvin():
    b = vc.CatBox(seed=2)
    assert "press Subject" in b.press("temp")
    b.press("subject")
    b.press("leak")                                                        # a pinhole in a box at 0 K: no photons, nothing to leak
    assert b.tau_leak == float("inf")
    seen = []
    for _ in vc.LUMP_TEMPS_K:
        b.press("temp")
        seen.append((vc.LUMP_TEMPS_K[b.lump_t], b.tau_in))
    assert [t for t, _ in seen] == [1e-3, 1.0, 300.0, 0.0]
    assert seen[0][1] > 30 > seen[1][1] > seen[2][1] and seen[3][1] == float("inf")


def test_the_lump_is_drawn_left_then_kicked_right_in_the_branch_where_the_atom_decays():
    s = next(s for s in range(50) if vc.CatBox(seed=s).outcome == "dead")
    b = vc.CatBox(seed=s)
    b.press("subject")
    b.reset(seed=s)
    assert b.label[4, 7, 1] == vc.M["osmium"] and b.label[9, 7, 1] != vc.M["osmium"]
    _run(b, vc.HOUR_TICKS)
    assert b.label[9, 7, 1] == vc.M["osmium"] and b.label[4, 7, 1] != vc.M["osmium"]


def _ghost_cells(box, name, world):
    lo, hi = vc.GHOST_K[world], vc.GHOST_K[world] + 6
    return int((box.label[:, :, lo + 1:hi] == vc.M[name]).sum())


def test_the_ghost_view_draws_each_world_as_dense_as_its_odds():
    b = vc.CatBox(seed=2)
    b.press("subject")
    assert _ghost_cells(b, "osmium", 1) == 0 and _ghost_cells(b, "osmium", 0) == 8          # no decay yet: only the world where nothing happened
    seen = []
    for _ in range(vc.HOUR_TICKS):
        b.step(0.1)
        seen.append(_ghost_cells(b, "osmium", 1))
    assert seen == sorted(seen) and seen[-1] > 0                                             # the other world fades in as the odds grow
    assert abs(_ghost_cells(b, "osmium", 0) - _ghost_cells(b, "osmium", 1)) <= 4             # even odds: about as many in each


def test_the_ghost_views_coherence_columns_follow_the_meters():
    b = vc.CatBox(seed=2)
    _run(b)
    height = vc.GHOST_K[1] + 5 - vc.GHOST_K[0]
    gold = int((b.label[14, vc.JF, vc.GHOST_K[0]:vc.GHOST_K[0] + height] == vc.M["gold"]).sum())
    cyan = int((b.label[15, vc.JF, vc.GHOST_K[0]:vc.GHOST_K[0] + height] == vc.M["cyan"]).sum())
    assert gold == 0 and cyan == height                                       # the cat sealed: nothing in practice, everything in principle
    b.press("subject")
    _run(b)
    assert int((b.label[14, vc.JF, vc.GHOST_K[0]:vc.GHOST_K[0] + height] == vc.M["gold"]).sum()) == height   # the lump at 0 K: both full


def test_the_lump_box_is_dark_until_a_cold_lamp_is_switched_on_and_the_lamp_records_it():
    b = vc.CatBox(seed=2)
    assert b.label[7, 7, vc.BOX_H - 2] == vc.M["lamp"]                        # the cat's box is lit
    b.press("subject")
    assert b.label[7, 7, vc.BOX_H - 2] == 0 and b.tau_in == float("inf")      # the lump's is dark: nothing reaches you, you see nothing
    msg = b.press("lamp")                                                     # a femtowatt of green light
    assert b.label[7, 7, vc.BOX_H - 2] == vc.M["lamp"] and 10 < b.tau_in < 30 and "records it" in msg
    b.press("lamp")                                                           # a picowatt
    assert b.tau_in == pytest.approx(0.017, rel=0.1)
    b.press("lamp")
    b.press("lamp")                                                           # round again: off
    assert b.label[7, 7, vc.BOX_H - 2] == 0 and b.tau_in == float("inf")
    assert "always lit" in vc.CatBox(seed=2).press("lamp")
