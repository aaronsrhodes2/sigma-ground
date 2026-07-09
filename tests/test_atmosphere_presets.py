"""STP / IRT / ISM named atmosphere presets — the single ambient datum.

``field.interface.atmosphere.ATMOSPHERES`` is the ONE table both Radiance
theaters (physics_env staging) and Materia (heat-solver boundary conditions)
read. A regression here means the stage and the physics have drifted apart
again — exactly the bug this table killed (theaters.plain stamped 273.15 while
every Materia engine default used the 288.15 ISA datum)."""
import pytest

from sigma_ground.field.interface.atmosphere import ATMOSPHERES, atmosphere_preset
from sigma_ground.radiance import theaters

CONTENT = {"bbox": [[-0.1, 0.1], [-0.1, 0.1], [0.0, 0.2]]}


def test_preset_values_exact():
    assert atmosphere_preset("STP") == {"medium": "air", "pressure_pa": 101325.0,
                                        "temperature_k": 288.15}
    assert atmosphere_preset("IRT") == {"medium": "air", "pressure_pa": 101325.0,
                                        "temperature_k": 293.15}
    assert atmosphere_preset("ISM") == {"medium": "vacuum", "pressure_pa": 0.0,
                                        "temperature_k": 2.725}


def test_preset_is_a_copy_and_case_insensitive():
    p = atmosphere_preset("stp")                  # case-insensitive lookup
    p["temperature_k"] = 999.0                    # caller mutation can't corrupt the table
    assert ATMOSPHERES["STP"]["temperature_k"] == 288.15


def test_unknown_preset_refuses():
    with pytest.raises(KeyError):                 # right-or-refuse: no silent default
        atmosphere_preset("mars")


def test_theater_envs_match_their_presets():
    """plain==STP (the 273.15 bug stays dead), room==IRT, deep_space==ISM."""
    for theater, preset in (("plain", "STP"), ("room", "IRT"), ("deep_space", "ISM")):
        env = theaters.stage(dict(CONTENT), theater)["physics_env"]
        assert env["atmosphere"] == preset
        for k, v in atmosphere_preset(preset).items():
            assert env[k] == v, f"{theater}.physics_env[{k!r}] drifted from {preset}"


def test_plain_datum_fixed():
    assert theaters.plain(dict(CONTENT))["physics_env"]["temperature_k"] == 288.15


def test_room_temperature_override_still_works():
    env = theaters.room(dict(CONTENT), temperature_k=310.0)["physics_env"]
    assert env["temperature_k"] == 310.0
    assert env["atmosphere"] == "IRT"


def test_void_is_a_warm_vacuum_chamber_not_ism():
    env = theaters.void(dict(CONTENT))["physics_env"]
    assert env["medium"] == "vacuum"
    assert env["temperature_k"] == 293.15         # radiatively room-temp surroundings
    assert env["atmosphere"] == "vacuum_room"     # explicitly NOT the 2.725 K ISM
