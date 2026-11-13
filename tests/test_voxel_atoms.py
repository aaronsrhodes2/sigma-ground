"""The atoms stage (materia/voxel_atoms.py): the lattice and sea, one atom on a log scale, the field that flips, and its place in the theater."""
import sys
from pathlib import Path

import numpy as np
import pytest

from sigma_ground.field.interface import osmium_atom as oa
from sigma_ground.materia import voxel_atoms as va

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))


def _to(box, tick):
    while box.ticks < tick:
        box.step(0.1)
    return box


def _count(box, name, i0=0, i1=va.W):
    return int((box.label[i0:i1] == va.M[name]).sum())


def test_every_material_has_a_plain_block_and_the_phases_follow_each_other():
    assert set(va.LOOKS) == set(va.MATERIALS[1:])
    b = va.AtomicZoom(seed=1)
    seen = []
    for _ in range(va.TICKS):
        b.step(0.1)
        if not seen or seen[-1] != b.phase():
            seen.append(b.phase())
    assert seen == ["descent", "slosh", "return", "after"]


def test_the_lattice_is_hcp_layers_at_the_librarys_spacing_with_every_other_layer_offset():
    b = va.AtomicZoom(seed=1)
    sites = b._atoms
    rows = sorted({k for _, k in sites}, reverse=True)
    assert len(rows) == va.LAYERS
    a_blocks = b.lat["a_m"] / va.PITCH
    row0 = sorted(i for i, k in sites if k == rows[0])
    assert np.diff(row0).mean() == pytest.approx(a_blocks, rel=0.1)
    row1 = sorted(i for i, k in sites if k == rows[1])
    assert abs(row1[0] - row0[0]) == pytest.approx(a_blocks / 2, abs=1.0)           # the next layer sits in the hollows
    assert rows[0] - rows[1] == pytest.approx(b.lat["layer_spacing_m"] / va.PITCH, abs=1.0)


def test_the_sea_is_two_electrons_an_atom_a_couple_of_percent_of_the_blocks():
    b = va.AtomicZoom(seed=1)
    n_sea = [_count(b.__class__(seed=s), "sea", 0, 50) for s in range(5)]
    n_atoms = len(b._atoms)
    # two an atom, less the part that falls inside the atoms' own blocks (the cores are drawn solid)
    assert 0.4 * 2 * n_atoms < np.mean(n_sea) < 1.2 * 2 * n_atoms


def test_one_atom_has_a_ring_for_each_slater_group_on_a_log_scale_and_all_76_electrons():
    b = va.AtomicZoom(seed=1)
    assert len(b._radii) == len(b.groups) == 10
    assert b._radii == sorted(b._radii) and b._radii[0] == pytest.approx(3.0)
    ratios = [np.log10(g["radius_m"]) for g in b.groups]
    steps = np.diff(b._radii) / np.diff(ratios)
    assert steps == pytest.approx([4.7] * (len(steps)), rel=1e-6)                      # one log scale for every ring
    dots = _count(b, "electron", 50, 100)
    assert 50 < dots <= 76                                                              # 76 electrons (some land on the same block)


def test_the_field_flips_every_half_period_while_the_photon_is_at_the_surface():
    b = va.AtomicZoom(seed=1)
    _to(b, va.T_SLOSH + 15)
    colours = []
    for _ in range(va.TICKS_PER_PERIOD):
        b.step(0.1)
        colours.append(int(b.label[3, va.JC, va.H - 2]))
    assert set(colours) == {va.M["photon"], va.M["field"]}
    flips = sum(1 for x, y in zip(colours, colours[1:]) if x != y)
    assert 1 <= flips <= 3                                                              # one flip every half period: 4 ticks of the 8


def test_the_sea_follows_the_field_for_green_light_and_not_for_ultraviolet():
    b = _to(va.AtomicZoom(seed=1), va.T_SLOSH + 15)
    assert b.light["reflected"] and abs(b.swing()) > 0.1
    gauge = [i for i in range(5, 46) if b.label[i, va.JC, 4] == va.M["sea"]]
    assert gauge and abs(gauge[0] - 25) > 2                                           # the marker has left the middle
    b.press("colour")
    assert not b.light["reflected"]
    _to(b, va.T_SLOSH + 15)
    assert not b.light["reflected"]
    gauge = [i for i in range(5, 46) if b.label[i, va.JC, 4] == va.M["sea"]]
    assert gauge == [25]                                                               # the sea stays put: it cannot follow the ultraviolet


def test_the_clock_is_the_real_femtoseconds_and_says_how_much_it_is_slowed():
    b = _to(va.AtomicZoom(seed=1), 80)
    assert b.real_time_s() == pytest.approx(80 * 1.668e-15 / 8, rel=0.01)
    assert b.slow_down() == pytest.approx((1 / 6) / (1.668e-15 / 8), rel=0.01) and b.slow_down() > 1e14
    assert "fs" in b.readout() and "plasma frequency" in b.readout()


def test_the_stage_is_wired_into_the_theater():
    import theater
    st = theater.STAGES["atoms"]
    numbers = [s["request"] for s in theater.STAGES.values()] + [n for s in theater.STAGES.values() for _, n in s.get("extra", ())]
    assert len(set(numbers)) == len(numbers) and set(theater.ATOMS_BUTTONS) == {n for _, n in st["extra"]}
    _, world = theater.build("atoms")
    sent = []
    th = theater.Theater(world, theater.atoms_looks(), st["origin"], sent.extend)
    th.open()
    assert th.draw() == 0 and len(sent) > 100
    x0, y0, z0 = st["origin"]
    xs = [th.mc(i, 0, 0)[0] for i in (0, va.W - 1)]
    for other_name, other in theater.STAGES.items():
        x, y, z = other["desk"]
        for dx in range(-2, 14):
            for dz in range(-4, 1):
                assert not (min(xs) <= x + dx <= max(xs) and z0 <= z + dz <= z0 + va.D), (other_name, x + dx, z + dz)
    assert all(len(c) for c in theater.cat_captions(th, world, "atoms"))
