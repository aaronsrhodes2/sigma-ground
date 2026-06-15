"""material_profile(name) — names resolve to CITED/DERIVED properties, never
invented. Mercury must read as a mirror; glass must carry fracture data; an
unknown must come back flagged; every shipped catalog material must be cited.
"""
import json

from sigma_ground.field.interface.resolve import material_profile, MaterialFact, _JSON_PATH


def _is_fact(x):
    return isinstance(x, MaterialFact)


def test_mercury_is_a_reflective_liquid_metal():
    p = material_profile("mercury")
    assert p["identified"] and p["material_class"] == "liquid"
    # reflective, NOT the grey fallback — real Hg reflects ~0.74-0.79
    assert "reflectance" in p and _is_fact(p["reflectance"])
    assert p["reflectance"].value > 0.6
    assert not p["reflectance"].estimated          # derived from Fresnel n+ik
    # the liquid props travel with it
    assert p["density"].value > 13000 and "surface_tension" in p
    # alias resolves too
    assert material_profile("quicksilver")["canonical"] == "mercury"


def test_glass_carries_fracture_data_and_is_brittle():
    p = material_profile("glass")
    assert "K_Ic" in p and "sigma_UTS" in p
    assert p["is_ductile"].value is False
    assert "n_refractive" in p                      # refracts (transparent)
    # tempered glass is ~4x stronger than soda-lime, different fracture
    soda = material_profile("soda-lime glass")
    temp = material_profile("tempered glass")
    assert temp["sigma_UTS"].value > 3 * soda["sigma_UTS"].value


def test_unknown_name_is_flagged_not_fabricated():
    p = material_profile("flarbleglomp")
    assert p["identified"] is False
    assert p["density"].estimated                   # flagged, low confidence
    assert p["density"].confidence <= 0.3


def test_metal_resolves_fully_from_physics():
    p = material_profile("gold")
    for key in ("density", "E", "K", "sigma_UTS", "K_Ic", "reflectance", "color"):
        assert key in p, f"gold missing {key}"
    assert all(not p[k].estimated for k in ("density", "reflectance"))


def test_catalog_material_resolves_with_citation():
    # an everyday material the physics modules lack — comes from the cited JSON
    p = material_profile("balsa")
    assert p["identified"] and p["canonical"] == "wood_balsa"
    assert p["density"].value < 300 and not p["density"].estimated
    assert "youngs" not in p  # sanity: key is 'E', not a typo
    assert "E" in p


def test_every_catalog_material_is_cited_and_complete():
    doc = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    _STRENGTH = ("youngs_modulus_pa", "sigma_UTS_pa", "K_Ic_pa_sqrtm")
    _OPTICAL = ("n_refractive", "reflectance")
    # structural classes must carry a strength/stiffness; soft textiles (whose
    # tensile strength is weave-dependent — we won't fabricate it) carry an
    # optical instead. Every material has density + >=1 cited extra property.
    _STRUCTURAL = ("wood", "plastic", "glass", "metal", "ceramic", "stone", "foam")
    for key, rec in doc["materials"].items():
        cls = rec.get("class")
        assert "density_kg_m3" in rec, f"{key}: no density"
        has_strength = any(s in rec for s in _STRENGTH)
        has_optical = any(o in rec for o in _OPTICAL)
        has_fluid = "viscosity_pa_s" in rec or "surface_tension_n_m" in rec
        assert has_strength or has_optical or has_fluid, f"{key}: no property beyond density"
        if cls in _STRUCTURAL:
            assert has_strength, f"{key}: structural material needs a strength/stiffness"
        if cls == "fabric":
            assert has_optical, f"{key}: fabric needs an optical (reflectance)"
        # every value cell is cited (real source code + confidence present)
        for prop, cell in rec.items():
            if isinstance(cell, dict) and "v" in cell:
                assert cell.get("src") and cell.get("src") != "estimated", f"{key}.{prop} uncited"
                assert "c" in cell, f"{key}.{prop} no confidence"
