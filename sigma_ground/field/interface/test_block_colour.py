"""block.appearance colours against published optics: Radiance's photometric ground truth (radiance/entangler/
test_photometric_ground_truth.py) ported to the shared material card every renderer reads (SkippyCraft, GMod, Mentat).

Copper: Palik (1985) d-band edge cuts blue, R > G > B. Aluminium: Rakic (1998) flat Drude reflectance, neutral silver.
Ruby: Waychunas (1988) Cr3+ absorption, red transmitted. Silicon: grey (it was drawn black). And the gaps stay gaps.
"""
import pytest

from sigma_ground.field.interface.block import appearance


def rgb(material, thickness=0.1):
    return appearance(material, thickness)["color_rgb"]


def test_copper_is_warm():
    r, g, b = rgb("copper")
    assert r > g > b
    assert r / b >= 1.5


def test_aluminium_is_neutral_silver():
    r, g, b = rgb("aluminum")
    assert abs(r - g) < 0.08 and abs(g - b) < 0.08
    assert min(r, g, b) > 0.8


def test_ruby_reddens_with_thickness():
    thin, cm, block = rgb("ruby", 0.001), rgb("ruby", 0.01), rgb("ruby", 0.1)
    assert thin[0] > 0.95 and thin[1] > 0.6               # 1 mm: pale pink
    assert cm[0] > 0.8 and cm[1] < 0.1 and cm[2] > cm[1]  # 1 cm: pink-red (the Tanabe-Sugano thin-limit hue)
    assert block[0] > 0.3 and block[1] < 0.01 and block[2] < 0.01   # 10 cm: deep red
    assert appearance("ruby", 0.001)["opacity"] < appearance("ruby", 0.1)["opacity"]


def test_emerald_transmits_green():
    r, g, b = rgb("emerald", 0.02)
    assert g > r and g > b


def test_silicon_is_grey_not_black():
    a = appearance("silicon", 0.1)
    r, g, b = a["color_rgb"]
    assert 0.25 < r < 0.5 and 0.25 < g < 0.5 and 0.25 < b < 0.5
    assert not a["metal"]
    assert "semiconductor_optics" in a["sources"]["color_rgb"]


def test_metal_rgb_refuses_a_material_with_no_free_electrons():
    from sigma_ground.field.interface import optics
    with pytest.raises(ValueError):
        optics.metal_rgb("silicon")


@pytest.mark.parametrize("material", ["gallium arsenide", "germanium"])
def test_narrow_gap_semiconductors_look_like_their_surface(material):
    a = appearance(material, 0.1)
    assert a["color_rgb"] is not None and a["reflectance_f0"] == max(a["color_rgb"])


@pytest.mark.parametrize("material", ["titanium dioxide", "malachite", "turquoise", "kevlar", "plastic_abs"])
def test_no_model_stays_a_gap(material):
    """Wide-gap semiconductors (see-through), opaque minerals (need scattering) and unmeasured organics: no colour,
    never a guess (granite, concrete, rubber and oak have measured spectra now: reflectance_spectra)."""
    a = appearance(material, 0.1)
    assert a["color_rgb"] is None
    assert "no colour model" in a["gaps"]


@pytest.mark.parametrize("material, key", [("wool", "wool_natural"), ("cotton", "cotton_white"), ("leather", "leather_brown")])
def test_organics_use_measured_spectra(material, key):
    from sigma_ground.field.interface import optics
    assert rgb(material) == pytest.approx(optics.organic_rgb(key))
