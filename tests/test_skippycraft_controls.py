"""Controls from inside the game (SKIPPYCRAFT_PLAN 10b, P6): words -> actions on the running world, through the books.

Done when: place any material, heat, drop, shoot, push; pause, step, slow motion; all of it without the console."""
import sys
from pathlib import Path

import numpy as np
import pytest

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))

import controls  # noqa: E402
from sigma_ground.deckard.voxel_exec import field_from_voxel_exec  # noqa: E402
from sigma_ground.materia.playground import PlaygroundWorld  # noqa: E402


@pytest.mark.parametrize("text, kind, extra", [
    ("pause", "pause", {}),
    ("Play", "play", {}),
    ("step 3", "step", {"n": 3}),
    ("slower", "speed", {"factor": 0.5}),
    ("speed 0.25", "speed", {"set": 0.25}),
    ("heat that to 1500 K", "heat", {"temperature_k": 1500.0}),
    ("heat this to 800 C radius 3 cm", "heat", {"temperature_k": 1073.15, "radius_m": 0.03}),
    ("cool it to 200", "heat", {"temperature_k": 200.0}),
    ("drop a gold block there", "place", {"material": "gold", "drop": True}),
    ("place a 4 cm lead block at 700 K", "place", {"material": "lead", "size_m": 0.04, "temperature_k": 700.0}),
    ("put some oak here", "place", {"material": "wood_oak"}),
    ("push it north at 2 m/s", "push", {"direction": (0, 1, 0), "speed_ms": 2.0}),
    ("shoot", "shoot", {"weapon": None}),
    ("fire the ak47", "shoot", {"weapon": "ak47"}),
    ("remove that", "remove", {}),
    ("what is that?", "what", {}),
    ("run playground", "run", {"scene": "playground"}),
    ("stop", "stop", {}),
    ("reset", "reset", {}),
])
def test_words_become_actions(text, kind, extra):
    a = controls.parse(text)
    assert a is not None and a["kind"] == kind
    for k, v in extra.items():
        assert a[k] == pytest.approx(v) if isinstance(v, float) else a[k] == v


def test_nonsense_is_not_a_command():
    assert controls.parse("hello there") is None and controls.parse("") is None


class FakeTheater:
    """The coordinate mapping of a theater at an origin, over a world."""

    def __init__(self, world, origin=(100, -60, 105)):
        import theater
        self.world = world
        self.origin = origin
        occ = np.argwhere(world.label > 0)
        self.i0, self.k0, self.jhi = int(occ[:, 0].min()), int(occ[:, 2].min()), int(occ[:, 1].max())
        self.cell = theater.Theater.cell.__get__(self)
        self.cell_float = theater.Theater.cell_float.__get__(self)
        self.mc = theater.Theater.mc.__get__(self)


def _world():
    """A granite floor with a gold cube on it, 5 mm cells, room above."""
    field, _ = field_from_voxel_exec('box(0,0,0,11,0,11,"granite"); box(5,1,5,6,2,6,"gold")', block_m=0.01, resolution=2)
    return PlaygroundWorld(field, liquid="lead", bodies=("gold",), env="ISM", room=(0, 0, 20))


def _books(w, M0, E0):
    L = w.ledger
    assert w.mass_kg() + L["left_kg"] + L["ejected_kg"] == pytest.approx(M0 + L["placed_kg"], rel=1e-12)
    assert (w.thermal_J() + L["radiated_J"] + L["vented_J"] + L["to_surroundings_J"] + L["left_J"] + L["ejected_J"]
            - L["heat_from_motion_J"] - L["impact_heat_J"] == pytest.approx(E0 + L["placed_J"] + L["injected_J"], rel=1e-9))


def _msg(th, cell, text, look=(0.0, 0.0, 1.0)):
    x, y, z = th.mc(*cell)
    return {"player": "captain", "text": text, "eye": (x + 0.5, y + 3.0, z - 4.0), "look": look, "hit": True,
            "block": (x, y, z)}


def test_the_mapping_inverts():
    w = _world()
    th = FakeTheater(w)
    for cell in ((0, 0, 0), (3, 7, 2), (11, 11, 4)):
        assert th.cell(*th.mc(*cell)) == cell
    x, y, z = th.mc(3, 7, 2)
    i, j, k = th.cell_float(x + 0.5, y + 0.5, z + 0.5)
    assert (int(i), int(j), int(k)) == (3, 7, 2)


def test_heat_place_push_remove_keep_the_books():
    w = _world()
    th = FakeTheater(w)
    s = controls.Session()
    M0, E0 = w.mass_kg(), w.thermal_J()
    gold = next(b for b in w.bodies if b.material == "gold")
    gc = tuple(gold.grid[0])
    idx = np.argwhere(w.static)                                           # the floor's top, away from the gold
    floor = (int(idx[:, 0].min()) + 2, int(idx[:, 1].min()) + 2, int(idx[:, 2].max()))
    assert "1500 K" in s.apply(_msg(th, gc, "heat that to 1500 K"), w, th)
    assert float(gold.T.min()) == 1500.0 and gold.melt.max() == 1.0          # past gold's melting point: molten
    _books(w, M0, E0)
    assert "Placed" in s.apply(_msg(th, floor, "place a 2 cm copper block"), w, th)
    assert "copper" in w.materials and any(b.material == "copper" for b in w.bodies)
    _books(w, M0, E0)
    assert "Pushed" in s.apply(_msg(th, gc, "push it north at 2 m/s"), w, th)
    assert gold.v == pytest.approx([0.0, 2.0, 0.0])
    assert "wall" in s.apply(_msg(th, floor, "push it"), w, th)
    reply = s.apply(_msg(th, gc, "what is that"), w, th)
    assert "gold" in reply and "1500 K" in reply and "liquid" in reply
    assert "Removed the gold body" in s.apply(_msg(th, gc, "remove that"), w, th)
    assert not any(b.material == "gold" for b in w.bodies)
    _books(w, M0, E0)
    assert "wall cells" in s.apply(_msg(th, floor, "remove that 1 cm"), w, th)
    _books(w, M0, E0)
    w.run(3, 0.05)
    _books(w, M0, E0)


def test_place_refuses_what_does_not_fit():
    w = _world()
    th = FakeTheater(w)
    s = controls.Session()
    gold = next(b for b in w.bodies if b.material == "gold")
    top = (int(gold.grid[:, 0].max()), int(gold.grid[:, 1].max()), int(gold.grid[:, 2].max()))
    assert "No room" in s.apply(_msg(th, top, "place a 3 m iron block"), w, th)
    assert "Which material" in s.apply(_msg(th, top, "place a block"), w, th)
    assert "wrong" not in s.apply(_msg(th, (0, 0, 0), "drop a 1 cm oak block"), w, th)


def test_time_controls_pace_the_loop():
    s = controls.Session()
    assert s.tick_allowed()
    s.apply({"text": "pause", "player": "c", "eye": (0, 0, 0), "look": (0, 0, 1), "hit": False, "block": (0, 0, 0)}, None, None)
    assert not s.tick_allowed()
    s.apply({"text": "step 2", "player": "c", "eye": (0, 0, 0), "look": (0, 0, 1), "hit": False, "block": (0, 0, 0)}, None, None)
    assert s.tick_allowed() and s.tick_allowed() and not s.tick_allowed()
    s.apply({"text": "slower", "player": "c", "eye": (0, 0, 0), "look": (0, 0, 1), "hit": False, "block": (0, 0, 0)}, None, None)
    assert s.speed == 0.5
    s.apply({"text": "play", "player": "c", "eye": (0, 0, 0), "look": (0, 0, 1), "hit": False, "block": (0, 0, 0)}, None, None)
    assert s.tick_allowed()
    r = s.apply({"text": "run strength", "player": "c", "eye": (0, 0, 0), "look": (0, 0, 1), "hit": False, "block": (0, 0, 0)}, None, None)
    assert "strength" in r and s.request == "strength"


def test_a_shot_from_where_he_stands():
    w = _world()
    th = FakeTheater(w)
    s = controls.Session()
    M0, E0 = w.mass_kg(), w.thermal_J()
    gold = next(b for b in w.bodies if b.material == "gold")
    x, y, z = th.mc(*gold.grid[0])
    # standing south of the theater (larger z), looking north (-z) at the gold block's height
    msg = {"player": "c", "text": "shoot ak47", "eye": (x + 0.5, y + 0.5, z + 30.0), "look": (0.0, 0.0, -1.0), "hit": True,
           "block": (x, y, z)}
    reply = s.apply(msg, w, th)
    assert "ak47" in reply and w.ledger["shots"] == 1 and w.shots[-1]
    _books(w, M0, E0)


def test_every_placeable_material_has_a_carrier_in_the_pack():
    import theater
    looks, _ = theater.shared_looks()
    for m in controls.PLACEABLE:
        assert (m, controls.PLACE_PITCH) in looks
