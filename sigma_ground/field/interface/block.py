"""
One block of matter: a material at a temperature, and everything else about it.

``block(material, temperature_k, mass_kg=..., volume_m3=...)`` is the single call
a voxel theater (or anyone) makes to populate a block. The material name is the
key; temperature decides its phase and state; the amount is optional. Every
returned property names its source, and what sigma-ground cannot supply is
listed under ``gaps``, never filled in.

Which source wins, per property (one answer, not three):
  1. MEASURED, from ``resolve.material_profile`` (NASA TPSX, CRC tables, the
     measured-liquids table), when it has the property;
  2. DERIVED from first principles (``thermal``: Debye/Wiedemann-Franz heat
     transport, ``thermal_emission`` emissivity), when nothing measured exists.
     These can be far off (granite's derived conductivity is ~18x its measured
     one), which is why measured comes first;
  3. a gap.
Phase points, latent heats and the volume change on melting come from
``phase_transition.PHASE_DATA``; latent heats only where it states their mole
basis (``latent_heat_J_kg``).

"water" and "water_ice" are one substance: water borrows ice's phase data (and
its derived thermal values when nothing measured exists), said in ``gaps``.

Phase points are at 1 atm; at another pressure they are not corrected here.
Thermal values are at room temperature, not at ``temperature_k``.
"""
from __future__ import annotations

# a substance written under its liquid name → the name its phase data is under
SAME_SUBSTANCE = {"water": "water_ice"}

UNITS = {
    "density": "kg/m^3", "density_solid": "kg/m^3", "density_liquid": "kg/m^3",
    "thermal_conductivity": "W/(m K)", "specific_heat": "J/(kg K)", "heat_capacity_volumetric": "J/(m^3 K)",
    "emissivity": "dimensionless", "color": "sRGB 0-1",
    "melting_point": "K", "boiling_point": "K", "latent_heat_fusion": "J/kg", "latent_heat_vaporization": "J/kg",
    "volume_change_on_melting": "fraction of solid volume",
    "E": "Pa", "G": "Pa", "K": "Pa", "poisson_ratio": "dimensionless", "sigma_UTS": "Pa", "K_Ic": "Pa m^0.5",
    "reflectance": "dimensionless", "is_ductile": "bool", "n_refractive": "dimensionless",
    "surface_tension": "N/m", "viscosity": "Pa s",
}
_RENAME = {"thermal_conductivity_w_mk": "thermal_conductivity", "specific_heat_j_kgk": "specific_heat",
           "melt_temperature_k": "melting_point"}


def _fact(value, source, confidence=None):
    return {"value": value, "source": source, "confidence": confidence}


def block(material: str, temperature_k: float = 293.15, *, mass_kg: float | None = None,
          volume_m3: float | None = None) -> dict:
    """Everything about a block of ``material`` at ``temperature_k``.

    Amount (all optional): ``mass_kg`` alone gives the volume it fills;
    ``volume_m3`` alone (a theater's block volume) gives the mass that fills it;
    both give ``fill_fraction`` = mass / (density × volume).

    Returns ``{"material", "canonical", "identified", "written_as" (the phase the
    name means: "water" liquid, "ice" solid), "temperature_k", "phase",
    "properties": {name: {value, units, source, confidence}}, "amount", "gaps"}``.
    """
    from . import thermal
    from .phase_transition import PHASE_DATA, latent_heat_J_kg
    from .resolve import material_profile

    prof = material_profile(material)
    canonical = prof["canonical"]
    data_name = SAME_SUBSTANCE.get(canonical, canonical)
    props, gaps = {}, []

    # 1. every measured/curated property the profile has
    for key, f in prof.items():
        if hasattr(f, "source") and not getattr(f, "estimated", False):
            props[_RENAME.get(key, key)] = _fact(f.value, f.source, f.confidence)
    if "density" not in props:
        gaps.append("no sourced density")
    if data_name != canonical:
        gaps.append(f"uses {data_name}'s phase data")

    # 2. derived where nothing measured
    for name, fn in (("thermal_conductivity", thermal.thermal_conductivity),
                     ("specific_heat", thermal.specific_heat_j_kg_K),
                     ("emissivity", thermal.emissivity)):
        if name in props:
            continue
        for key in dict.fromkeys((canonical, data_name)):
            try:
                props[name] = _fact(float(fn(key)), f"derived: field.interface.thermal.{fn.__name__}({key!r})", 0.4)
                break
            except (KeyError, ValueError):
                continue
        else:
            gaps.append(f"no {name.replace('_', ' ')}")
    if "density" in props and "specific_heat" in props:
        props["heat_capacity_volumetric"] = _fact(
            props["density"]["value"] * props["specific_heat"]["value"], "density x specific_heat",
            min(props["density"]["confidence"] or 1, props["specific_heat"]["confidence"] or 1))

    # 3. phase data
    written_liquid = prof.get("material_class") == "liquid"
    data = PHASE_DATA.get(data_name)
    phase = "liquid" if written_liquid else ("solid" if prof.get("identified") else None)
    if data is None:
        gaps.append("no phase data (melting/boiling/latent heats): its phase never changes")
    else:
        src = "phase_transition.PHASE_DATA (CRC Handbook)"
        Tm, Tb = float(data["T_melt_K"]), float(data["T_boil_K"])
        props["melting_point"] = _fact(Tm, src)          # PHASE_DATA owns the phase points
        props["boiling_point"] = _fact(Tb, src)
        dV = float(data.get("delta_V_fus", 0.0))
        props["volume_change_on_melting"] = _fact(dV, src)
        for kind, name in (("fus", "latent_heat_fusion"), ("vap", "latent_heat_vaporization")):
            L = latent_heat_J_kg(data_name, kind)
            if L is None:
                gaps.append(f"no {name.replace('_', ' ')}: PHASE_DATA states no mole basis")
            else:
                props[name] = _fact(L, f"{src} / molar mass of its mole ({data.get('mole')})")
        if "density" in props:
            rho = props["density"]["value"]
            solid, liquid = (rho * (1 + dV), rho) if written_liquid else (rho, rho / (1 + dV))
            props["density_solid"] = _fact(solid, "density and volume change on melting")
            props["density_liquid"] = _fact(liquid, "density and volume change on melting")
        T = float(temperature_k)
        if T > Tb:
            phase = "gas"
        elif T > Tm or (T == Tm and written_liquid):
            phase = "liquid"
        else:
            phase = "solid"
    if phase in ("liquid", "gas") and not written_liquid and "thermal_conductivity" in props:
        gaps.append(f"thermal values are the solid's: none for the {phase}")

    for name, p in props.items():
        p["units"] = UNITS.get(name)

    amount = {}
    rho = props.get("density", {}).get("value")
    if mass_kg is not None or volume_m3 is not None:
        if rho is None:
            gaps.append("amount: no density to relate mass and volume")
        elif mass_kg is not None and volume_m3 is not None:
            amount = {"mass_kg": float(mass_kg), "volume_m3": float(volume_m3),
                      "fill_fraction": float(mass_kg) / (rho * float(volume_m3))}
        elif mass_kg is not None:
            amount = {"mass_kg": float(mass_kg), "volume_m3": float(mass_kg) / rho, "fill_fraction": 1.0}
        else:
            amount = {"mass_kg": rho * float(volume_m3), "volume_m3": float(volume_m3), "fill_fraction": 1.0}

    return {"material": material, "canonical": canonical, "identified": bool(prof.get("identified")),
            "written_as": "liquid" if written_liquid else "solid", "temperature_k": float(temperature_k),
            "phase": phase, "properties": props,
            "amount": amount, "gaps": gaps}


# material → (Cauchy key, absorption key) in optics: which measured optical model a clear material is
OPTICAL_MODEL = {
    "glass": ("crown_glass", "clear"), "water": ("water", "water_blue"), "water_ice": ("ice", "clear"),
    "diamond": ("diamond", "clear"), "quartz": ("quartz", "clear"), "sapphire": ("sapphire", "clear"),
}


def appearance(material: str, thickness_m: float, temperature_k: float = 293.15) -> dict:
    """How one block of ``material``, ``thickness_m`` across, looks: what a renderer needs.

    Returns ``{"color_rgb", "transmittance_rgb", "opacity", "reflectance_f0", "metal", "smoothness",
    "sources", "gaps"}``, every value from sigma-ground's optics, or a named gap:

      - a METAL (``optics.metal_rgb``: Drude / measured n+ik → Fresnel): its colour IS its normal-incidence
        reflectance; opaque (light dies within nanometres);
      - a CLEAR dielectric (``OPTICAL_MODEL``): Cauchy n and Beer-Lambert absorption through ``thickness_m``
        (``optics.dielectric_transmission_rgb``): colour = what gets through, opacity = 1 − mean
        transmittance, F0 = the Fresnel reflectance at 550 nm. So transparency depends on block size: a 5 cm
        block of water is nearly clear, a metre of it is blue;
      - anything else: the profile's colour if it has one, opaque; else gaps.

    ``smoothness`` is 1 − √α from ``texture.microfacet_roughness`` (Beckmann α of a clean, equilibrium
    surface: an upper bound on how mirror-like a real, worked surface is).
    """
    import math
    from . import optics, texture

    b = block(material, temperature_k)
    canon, props = b["canonical"], b["properties"]
    out = {"material": material, "canonical": canon, "thickness_m": float(thickness_m), "color_rgb": None,
           "transmittance_rgb": (0.0, 0.0, 0.0), "opacity": 1.0, "reflectance_f0": None, "metal": False,
           "smoothness": None, "sources": {}, "gaps": []}
    try:
        c = tuple(float(v) for v in optics.metal_rgb(canon))
        out.update(color_rgb=c, reflectance_f0=c, metal=True)
        out["sources"]["color_rgb"] = out["sources"]["reflectance_f0"] = "optics.metal_rgb (Drude / measured n+ik, Fresnel)"
        out["sources"]["opacity"] = "metal: opaque"
    except (KeyError, ValueError):
        if canon in OPTICAL_MODEL:
            key, absorb = OPTICAL_MODEL[canon]
            T = tuple(float(v) for v in optics.dielectric_transmission_rgb(key, absorb, float(thickness_m)))
            out.update(color_rgb=T, transmittance_rgb=T, opacity=1.0 - sum(T) / 3,
                       reflectance_f0=optics.dielectric_surface_reflectance(optics.cauchy_n(key, optics.LAMBDA_G)))
            src = f"optics: Cauchy n ({key}) + Beer-Lambert ({absorb}) through {thickness_m:g} m"
            out["sources"].update(color_rgb=src, transmittance_rgb=src, opacity=src,
                                  reflectance_f0=f"optics.dielectric_surface_reflectance(n({key}, 550 nm))")
        else:
            col = props.get("color")
            if col is not None:
                out["color_rgb"] = tuple(float(v) for v in col["value"])
                out["sources"]["color_rgb"] = col["source"]
            else:
                out["gaps"].append("no colour model")
            refl = props.get("reflectance")
            if refl is not None and isinstance(refl["value"], (int, float)):
                out["reflectance_f0"] = float(refl["value"])
                out["sources"]["reflectance_f0"] = refl["source"]
            else:
                out["gaps"].append("no reflectance")
            out["sources"]["opacity"] = "no transmission model: drawn opaque"
    for key in dict.fromkeys((canon, SAME_SUBSTANCE.get(canon, canon))):
        try:
            out["smoothness"] = max(0.0, 1.0 - math.sqrt(float(texture.microfacet_roughness(key, T=temperature_k))))
            out["sources"]["smoothness"] = f"1 - sqrt(texture.microfacet_roughness({key!r}))"
            break
        except (KeyError, ValueError):
            continue
    else:
        out["gaps"].append("no surface roughness")
    return out


__all__ = ["block", "appearance", "SAME_SUBSTANCE", "UNITS", "OPTICAL_MODEL"]
