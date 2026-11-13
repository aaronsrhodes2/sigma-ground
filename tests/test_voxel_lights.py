"""The lights-up stage (materia/voxel_lights.py): the phases, the clock, the bridge that the first photon cuts, and its place in the theater."""
import sys
from pathlib import Path

import numpy as np
import pytest

from sigma_ground.field.interface import levitated_grain as lg
from sigma_ground.materia import voxel_lights as vl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))


def _to(box, tick):
    while box.ticks < tick:
        box.step(0.1)
    return box


def _count(box, name):
    return int((box.label == vl.M[name]).sum())


def test_every_material_has_a_plain_block_and_the_phases_follow_each_other():
    assert set(vl.LOOKS) == set(vl.MATERIALS[1:])
    b = vl.LightsUp(seed=1)
    seen = []
    for _ in range(vl.TICKS):
        b.step(0.1)
        if not seen or seen[-1] != b.phase():
            seen.append(b.phase())
    assert seen == ["rest", "lift", "wait", "flight", "after"] or seen == ["lift", "wait", "flight", "after"]


def test_the_grain_rests_touching_the_floor_then_floats_in_two_places():
    b = _to(vl.LightsUp(seed=1), 30)
    assert b.label[vl.CI_REST, vl.JC, vl.FLOOR_TOP] == vl.M["stress"]                    # the contact voxel
    assert b.label[vl.CI_REST, vl.JC, vl.FLOOR_TOP + 1] == vl.M["osmium"]                # sitting on the floor
    solid = _count(b, "osmium")
    b = _to(b, vl.T_LAMP)
    assert _count(b, "bridge") > 0 and _count(b, "field") > 0 and b.label[vl.CI_REST, vl.JC, vl.FLOOR_TOP] != vl.M["stress"]
    assert 0.8 * solid < _count(b, "osmium") < 1.3 * solid                               # two half-density ghosts hold about one grain's worth of blocks
    left = int((b.label[vl.CI_LEFT - 5:vl.CI_LEFT + 6, :, :] == vl.M["osmium"]).sum())
    right = int((b.label[vl.CI_RIGHT - 5:vl.CI_RIGHT + 6, :, :] == vl.M["osmium"]).sum())
    assert abs(left - right) < 0.25 * (left + right)                                      # even odds: about as dense in each place


def test_the_lamp_is_dark_until_it_comes_on_and_almost_no_light_meets_the_grain():
    b = _to(vl.LightsUp(seed=1), vl.T_LAMP - 1)
    assert _count(b, "lamp") == 0 and not b.photons
    b = _to(b, vl.T_LAMP + 40)
    assert _count(b, "lamp") > 0 and len(b.photons) > 10
    for _ in range(60):
        b.step(0.1)
        for i, k, di, dk in b.photons:                                                    # none of the drawn photons is on the grain
            assert not (vl.CI_LEFT - 5 < i < vl.CI_LEFT + 5 and abs(k - vl.CK_FLOAT) < vl.R_BLOCKS + 1)


def test_the_real_time_clock_ends_the_wait_at_the_first_photon_then_runs_the_flight_in_nanoseconds():
    b = _to(vl.LightsUp(seed=1), vl.T_FLIGHT - 1)
    assert b.real_time_s() == pytest.approx(b.t_hit, rel=0.02)
    assert b.slow_down() == pytest.approx((1 / 6) / (b.t_hit / vl.WAIT_TICKS))
    b = _to(b, vl.T_AFTER - 1)
    assert b.real_time_s() == pytest.approx(lg.flight_time_s(1.0), rel=0.02)
    assert b.slow_down() > 1e9                                                            # billions of times slower than real


def test_the_first_photons_arrival_time_follows_the_lamps_rate():
    times = [vl.LightsUp(seed=s).t_hit for s in range(2000, 2400)]
    b = vl.LightsUp(seed=1)
    assert np.mean(times) == pytest.approx(1.0 / b.rate_hit, rel=0.15)                    # a Poisson process: the mean wait is 1 / rate
    b.press("lamp")
    assert b.rate_hit == pytest.approx(239120.5, rel=0.01)                                # a milliwatt: a thousand times the microwatt's rate


def test_the_photon_cuts_the_practice_bridge_and_leaves_the_principle_one():
    b = _to(vl.LightsUp(seed=1), vl.T_AFTER - 2)
    assert _count(b, "bridge practice") > 0 and b.coh_in_practice == 1.0
    b = _to(b, vl.TICKS)
    assert _count(b, "bridge practice") == 0 and _count(b, "bridge") > 0
    assert b.coh_in_practice == 0.0 and b.coh_in_principle == 1.0 and _count(b, "you") > 0
    assert f"{b.world} world" in b.readout() and "unchanged" in b.readout()


def test_the_stage_is_wired_into_the_theater():
    import theater
    st = theater.STAGES["lights"]
    numbers = [s["request"] for s in theater.STAGES.values()] + [n for s in theater.STAGES.values() for _, n in s.get("extra", ())]
    assert len(set(numbers)) == len(numbers) and set(theater.LIGHTS_BUTTONS) == {n for _, n in st["extra"]}
    _, world = theater.build("lights")
    sent = []
    th = theater.Theater(world, theater.lights_looks(), st["origin"], sent.extend)
    th.open()
    assert th.draw() == 0 and len(sent) > 100
    x0, y0, z0 = st["origin"]
    xs = [th.mc(i, 0, 0)[0] for i in (0, vl.W - 1)]
    for other in theater.STAGES.values():
        x, y, z = other["desk"]
        assert not (min(xs) <= x <= max(xs) and z0 <= z <= z0 + vl.D), other
    assert all(len(c) for c in theater.cat_captions(th, world, "lights"))
