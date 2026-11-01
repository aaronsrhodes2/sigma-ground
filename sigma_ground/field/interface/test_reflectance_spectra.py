"""Measured surface reflectance (SkippyCraft P5): every value traceable to its file or its printed source."""
import numpy as np
import pytest

from sigma_ground.field.interface import reflectance_spectra as rs
from sigma_ground.field.interface.block import appearance


def test_every_ecostress_fixture_is_present_and_parses():
    for canon, (name, _) in rs.ECOSTRESS.items():
        assert (rs.FIXTURES / name).is_file(), name
        wl, r = rs.ecostress_spectrum(canon)
        assert wl[0] <= 450.0 and wl[-1] >= 650.0                       # covers the samples
        assert np.all(np.diff(wl) > 0) and 0.0 <= r.min() and r.max() <= 1.0


def test_granite_is_a_mid_grey_and_concrete_lighter_rubber_black():
    g = rs.measured_rgb("granite")["rgb"]
    c = rs.measured_rgb("concrete")["rgb"]
    r = rs.measured_rgb("rubber")["rgb"]
    assert 0.12 < min(g) and max(g) < 0.25
    assert c[1] > g[1] and max(r) < 0.08
    # the concrete file: 22.99 % at 0.520 um and 26.94 % at 0.568 um, so 550 nm interpolates between them
    assert 0.2299 < c[1] < 0.2694


def test_oak_lab_converts_exactly():
    """CIE L*a*b* -> linear sRGB, checked against the white point and a hand computation."""
    assert rs.lab_to_linear_rgb(100.0, 0.0, 0.0, "D65") == pytest.approx((1.0, 1.0, 1.0), abs=2e-3)
    rgb = rs.lab_to_linear_rgb(37.5, 10.5, 19.9, "D50")
    assert rgb == pytest.approx((0.161, 0.0835, 0.0402), abs=2e-3)        # a warm brown
    assert rs.measured_rgb("wood_oak")["rgb"] == pytest.approx(rgb)


def test_alumina_is_a_flagged_bound():
    m = rs.measured_rgb("ceramic_alumina")
    assert m["rgb"] == (0.8, 0.8, 0.8) and any("bound" in f for f in m["flags"])


@pytest.mark.parametrize("material", ["granite", "concrete", "rubber", "wood_oak", "alumina"])
def test_appearance_uses_the_measured_surface(material):
    a = appearance(material, 0.01)
    assert a["color_rgb"] is not None and "reflectance_spectra" in a["sources"]["color_rgb"]
    assert "no colour model" not in a["gaps"]
    assert a["opacity"] == 1.0 and a["metal"] is False


def test_kevlar_and_abs_stay_honest_gaps():
    for m in ("kevlar", "plastic_abs"):
        a = appearance(m, 0.01)
        assert a["color_rgb"] is None and "no colour model" in a["gaps"]
