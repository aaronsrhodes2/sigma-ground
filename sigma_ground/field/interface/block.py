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
    "emissivity": "dimensionless", "emissivity_visible_550nm": "dimensionless", "color": "sRGB 0-1",
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

    # the profile's emissivity is Kirchhoff 1 - R at 550 nm (resolve.py): the VISIBLE absorptivity, right for a glow's
    # colour, wrong for heat radiated (thermal infrared). It keeps its own name; "emissivity" is the total thermal one.
    if "emissivity" in props and "Kirchhoff" in props["emissivity"]["source"]:
        props["emissivity_visible_550nm"] = props.pop("emissivity")

    # 2. derived where nothing measured
    for name, fn in (("thermal_conductivity", thermal.thermal_conductivity),
                     ("specific_heat", thermal.specific_heat_j_kg_K)):
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
    if "emissivity" not in props:                             # total thermal: Hagen-Rubens for metals, else measured
        for key in dict.fromkeys((canonical, data_name)):
            try:
                v, src, conf = thermal.emissivity_fact(key, float(temperature_k))
                props["emissivity"] = _fact(float(v), src, conf)
                break
            except KeyError:
                continue
        else:
            gaps.append("no emissivity")
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
        T = float(temperature_k)
        if T > Tb:
            phase = "gas"
        elif T > Tm or (T == Tm and written_liquid):
            phase = "liquid"
        else:
            phase = "solid"
    if phase in ("liquid", "gas") and not written_liquid and "thermal_conductivity" in props:
        gaps.append(f"thermal values are the solid's: none for the {phase}")

    # 4. at the block's own temperature (the live card)
    _at_temperature(props, gaps, canonical, data_name, float(temperature_k), phase, written_liquid,
                    props.get("melting_point", {}).get("value"))
    if data is not None and "density" in props and "density_solid" not in props:
        rho = props["density"]["value"]
        solid, liquid = (rho * (1 + dV), rho) if written_liquid else (rho, rho / (1 + dV))
        props["density_solid"] = _fact(solid, "density and volume change on melting")
        props["density_liquid"] = _fact(liquid, "density and volume change on melting")
    if "density" in props and "specific_heat" in props:
        props["heat_capacity_volumetric"] = _fact(
            props["density"]["value"] * props["specific_heat"]["value"], "density x specific_heat",
            min(props["density"]["confidence"] or 1, props["specific_heat"]["confidence"] or 1))

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


T_REF_K = 293.15     # the measured tables' temperature (CRC, TPSX: room temperature); block()'s default


def _at_temperature(props, gaps, canonical, data_name, T, phase, written_liquid, T_melt):
    """Carry ``props`` (measured near ``T_REF_K``) to the block's temperature, in place.

    Liquid water: NIST's measured values at T (``liquid_water.nist_water``: IAPWS-95). Anything else: the measured value
    times the law's own change, value(T) = value(T_REF) x law(T) / law(T_REF):
      thermal conductivity  ``thermal.thermal_conductivity`` (Wiedemann-Franz with rho(T) for metals, Umklapp phonons)
      specific heat         ``thermal.specific_heat_j_kg_K`` (Debye)
      density               thermal expansion, rho(T) = rho(T_REF) / (1 + integral of alpha(T) dT)^3
                            (``thermal_expansion.expansion_coefficient_at_T``: measured alpha, Debye C_v scaling)
    A melted block keeps its solid's laws, evaluated at the melting point (the gap above says so). No law: the room-
    temperature value stays and a gap says so.
    """
    from . import liquid_water, thermal, thermal_expansion

    if data_name == "water_ice":                     # each phase its own measured density, at T (nearest end past 273 K)
        props["density_solid"] = _fact(liquid_water.ice_table(min(T, 273.15), "density_kg_m3"),
                                       f"{liquid_water.ICE_SOURCE} at {min(T, 273.15):g} K", 0.8)
        props["density_liquid"] = _fact(liquid_water.nist_water(max(T, 275.0), "density_kg_m3"),
                                        f"{liquid_water.NIST_WATER_SOURCE} at {max(T, 275.0):g} K", 0.95)
    if data_name == "water_ice" and phase == "solid":
        for name, key in (("density", "density_kg_m3"), ("specific_heat", "cp_J_g_K"),
                          ("thermal_conductivity", "thermal_conductivity_W_m_K")):
            v = liquid_water.ice_table(T, key) * (1e3 if key == "cp_J_g_K" else 1.0)
            props[name] = _fact(v, f"{liquid_water.ICE_SOURCE} at {T:g} K", 0.8)
        gaps[:] = [g for g in gaps if g not in ("no thermal conductivity", "no specific heat")]
        return
    if data_name == "water_ice" and phase == "liquid":
        for name, key, conf in (("density", "density_kg_m3", 0.95), ("specific_heat", "cp_J_g_K", 0.95),
                                ("thermal_conductivity", "thermal_conductivity_W_m_K", 0.95),
                                ("viscosity", "viscosity_Pa_s", 0.95)):
            v = liquid_water.nist_water(T, key) * (1e3 if key == "cp_J_g_K" else 1.0)
            props[name] = _fact(v, f"{liquid_water.NIST_WATER_SOURCE} at {T:g} K", conf)
        gaps[:] = [g for g in gaps if g not in ("no thermal conductivity", "no specific heat")]
        if not liquid_water.nist_water_in_range(T):
            gaps.append("liquid water outside NIST's 275-370 K table: its nearest end")
        return
    if abs(T - T_REF_K) < 1e-9:
        return
    T_law = min(T, T_melt) if (phase in ("liquid", "gas") and not written_liquid and T_melt) else T
    keys = list(dict.fromkeys((canonical, data_name)))

    def law_ratio(fn):
        for key in keys:
            try:
                a, b = float(fn(key, T_law)), float(fn(key, T_REF_K))
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            if b > 0 and a > 0:
                return a / b, f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}({key!r})"
        return None, None

    for name, fn in (("thermal_conductivity", thermal.thermal_conductivity),
                     ("specific_heat", thermal.specific_heat_j_kg_K)):
        if name not in props:
            continue
        ratio, how = law_ratio(fn)
        if ratio is None:
            gaps.append(f"{name.replace('_', ' ')} at room temperature: no law for its change with temperature")
            continue
        p = props[name]
        props[name] = _fact(p["value"] * ratio, f"{p['source']}, x {how} at {T_law:g} K / at {T_REF_K:g} K",
                            p["confidence"])
    if "density" in props and not written_liquid:
        for key in keys:
            try:
                n, strain = 16, 0.0
                for i in range(n):                             # trapezoid on alpha(T) from T_REF to T_law
                    t0 = T_REF_K + (T_law - T_REF_K) * i / n
                    t1 = T_REF_K + (T_law - T_REF_K) * (i + 1) / n
                    strain += 0.5 * (thermal_expansion.expansion_coefficient_at_T(key, t0)
                                     + thermal_expansion.expansion_coefficient_at_T(key, t1)) * (t1 - t0)
            except KeyError:
                continue
            p = props["density"]
            props["density"] = _fact(p["value"] / (1.0 + strain) ** 3,
                                     f"{p['source']}, expanded to {T_law:g} K (thermal_expansion.expansion_coefficient_at_T"
                                     f"({key!r}): measured alpha, Debye scaling)", p["confidence"])
            break
        else:
            gaps.append("density at room temperature: no thermal expansion data")


# material → (Cauchy key, absorption key) in optics: which measured optical model a clear material is
OPTICAL_MODEL = {
    "glass": ("crown_glass", "clear"), "water": ("water", "water_blue"), "water_ice": ("ice", "clear"),
    "diamond": ("diamond", "clear"), "quartz": ("quartz", "clear"), "sapphire": ("sapphire", "clear"),
}


def appearance(material: str, thickness_m: float, temperature_k: float = 293.15) -> dict:
    """How one block of ``material``, ``thickness_m`` across, looks: what a renderer needs.

    Returns ``{"color_rgb", "transmittance_rgb", "opacity", "reflectance_f0", "metal", "smoothness",
    "n_refractive", "absorption_per_m", "sources", "gaps"}`` (the last two for a CLEAR dielectric: its index at 550 nm
    and its bulk absorption per metre in R, G, B, so a renderer can compute the Fresnel reflectance at any angle and
    the Beer-Lambert transmittance through any path length instead of reading one thickness's opacity), every value from sigma-ground's optics, or a named gap:

      - a METAL (``optics.metal_rgb``: Drude / measured n+ik → Fresnel): its colour IS its normal-incidence
        reflectance; opaque (light dies within nanometres);
      - a CLEAR dielectric (``OPTICAL_MODEL``): Cauchy n and Beer-Lambert absorption through ``thickness_m``
        (``optics.dielectric_transmission_rgb``): colour = what gets through, opacity = 1 − mean
        transmittance, F0 = the Fresnel reflectance at 550 nm. So transparency depends on block size: a 5 cm
        block of water is nearly clear, a metre of it is blue;
      - a GEM light passes through (``crystal_field.TRANSPARENT_GEMS``): measured Beer-Lambert absorption through
        ``thickness_m`` (``crystal_field.gem_transmission_rgb``), so a ruby block is pink-red at 1 cm, deep red at 10 cm;
      - a SEMICONDUCTOR opaque to all visible light (absorption edge past 700 nm: silicon, germanium, GaAs):
        ``semiconductor_optics.semiconductor_rgb`` (band gap, measured n+ik, Fresnel);
      - an ORGANIC surface with a measured spectrum (wool, cotton, leather): ``optics.organic_rgb``;
      - a MEASURED surface (granite, concrete, rubber, oak, alumina: ``reflectance_spectra``): an ECOSTRESS spectrum,
        a measured L*a*b*, or a measured bound (flagged);
      - anything else: the profile's colour if it has one, opaque; else gaps.
    (The colour models Radiance's material library used, moved here 2026-10-31 so every consumer gets them.)

    ``smoothness`` is 1 − √α from ``texture.microfacet_roughness`` (Beckmann α of a clean, equilibrium
    surface: an upper bound on how mirror-like a real, worked surface is).
    """
    import math
    from . import optics, texture

    b = block(material, temperature_k)
    canon, props = b["canonical"], b["properties"]
    out = {"material": material, "canonical": canon, "thickness_m": float(thickness_m), "color_rgb": None,
           "transmittance_rgb": (0.0, 0.0, 0.0), "opacity": 1.0, "reflectance_f0": None, "metal": False,
           "smoothness": None, "n_refractive": None, "absorption_per_m": None, "sources": {}, "gaps": []}
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
            out["n_refractive"] = float(optics.cauchy_n(key, optics.LAMBDA_G))
            out["absorption_per_m"] = tuple(float(a) for a in optics.DIELECTRIC_ABSORPTION.get(absorb, (0.5, 0.5, 0.5)))
            out["sources"]["n_refractive"] = f"optics.cauchy_n({key!r}, 550 nm)"
            out["sources"]["absorption_per_m"] = f"optics.DIELECTRIC_ABSORPTION[{absorb!r}] (Beer-Lambert, R G B per metre)"
            src = f"optics: Cauchy n ({key}) + Beer-Lambert ({absorb}) through {thickness_m:g} m"
            out["sources"].update(color_rgb=src, transmittance_rgb=src, opacity=src,
                                  reflectance_f0=f"optics.dielectric_surface_reflectance(n({key}, 550 nm))")
        elif (deriv := _derived_colour(canon, float(thickness_m), temperature_k)) is not None:
            rgb, opacity, f0, src, flags = deriv
            out.update(color_rgb=rgb, opacity=opacity)
            out["sources"]["color_rgb"] = src
            out["gaps"].extend(flags)
            if opacity < 1.0:
                out["transmittance_rgb"] = rgb
                out["sources"]["transmittance_rgb"] = out["sources"]["opacity"] = src
            else:
                out["sources"]["opacity"] = "opaque: " + src.split(" (")[0]
            if f0 is not None:
                out["reflectance_f0"] = f0
                out["sources"]["reflectance_f0"] = src
            else:
                out["gaps"].append("no reflectance")
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


# an organic surface with a measured spectrum in optics.ORGANIC_SPECTRA, by the name block() resolves
ORGANIC_SURFACE = {"wool": "wool_natural", "cotton": "cotton_white", "leather": "leather_brown"}


def _derived_colour(canon: str, thickness_m: float, temperature_k: float):
    """``(rgb, opacity, F0 or None, source, flags)`` from the gem, semiconductor, organic and measured-surface colour
    models, or None."""
    from . import crystal_field, optics, reflectance_spectra, semiconductor_optics
    if canon in crystal_field.TRANSPARENT_GEMS:
        t = tuple(float(v) for v in crystal_field.gem_transmission_rgb(canon, thickness_m))
        return t, 1.0 - sum(t) / 3, None, \
            f"crystal_field.gem_transmission_rgb (measured Beer-Lambert absorption) through {thickness_m:g} m", []
    # Not here: opaque crystal-field minerals (malachite, turquoise). crystal_field.mineral_rgb is a thin section's colour
    # (malachite pale cyan, turquoise near white); a scattering stone's body colour needs Kubelka-Munk, which sigma-ground lacks.
    # Semiconductors only when the absorption edge is past the visible (Si, Ge, GaAs): then the surface reflectance IS the
    # look. A wide-gap one (TiO2, ZnO, GaN, CdS) is see-through or a white powder; its mirror reflectance would draw it grey.
    if canon in semiconductor_optics.VARSHNI_PARAMS and semiconductor_optics.band_edge_nm(canon, temperature_k) > 700.0:
        c = tuple(float(v) for v in semiconductor_optics.semiconductor_rgb(canon, T=temperature_k))
        return c, 1.0, max(c), "semiconductor_optics.semiconductor_rgb (Varshni band gap, measured n+ik, Fresnel)", []
    if canon in ORGANIC_SURFACE:
        return tuple(float(v) for v in optics.organic_rgb(ORGANIC_SURFACE[canon])), 1.0, None, \
            f"optics.organic_rgb ({ORGANIC_SURFACE[canon]}: measured spectrophotometry)", []
    m = reflectance_spectra.measured_rgb(canon)
    if m is not None:
        return m["rgb"], 1.0, None, m["source"], m["flags"]
    return None


GLOW_SAMPLES_M = (650e-9, 550e-9, 450e-9)       # thermal_emission's red, green, blue sample wavelengths


def glow(material: str, temperature_k: float) -> dict:
    """The light a block of ``material`` gives off at ``temperature_k`` by its own heat (thermal emission).

    L(λ, T) = ε(λ) · B(λ, T): Planck's law times Kirchhoff's emissivity (``thermal_emission``). A metal with measured
    n+ik uses its own spectral emissivity (``thermal_emission.emissivity``, ε = 1 − R); any other material is a grey body
    with ``block()``'s emissivity. Returns ``{"visible", "rgb", "radiance_W_m2_sr_m", "sources", "gaps"}``:
    ``radiance`` is L at 650, 550 and 450 nm (absolute, so the brightness rises as steeply as it really does), ``power_W_m2`` is epsilon sigma T^4 and ``level`` its display tonemap in [0, 1] (``thermal_emission.glow_level``),
    ``rgb`` their ratio normalised to the largest (the glow's hue), ``visible`` whether it is past the Draper point.
    """
    from . import thermal_emission as te

    b = block(material, temperature_k)
    canon = b["canonical"]
    T = float(temperature_k)
    out = {"material": material, "temperature_k": T, "visible": te.is_visibly_glowing(T), "rgb": (0.0, 0.0, 0.0),
           "radiance_W_m2_sr_m": (0.0, 0.0, 0.0), "power_W_m2": 0.0, "level": 0.0, "sources": {}, "gaps": []}
    if canon in te.THERMAL_EMISSION_MATERIALS:
        eps = [te.emissivity(canon, lam) for lam in GLOW_SAMPLES_M]
        out["sources"]["emissivity"] = "thermal_emission.emissivity (Kirchhoff 1 - R, measured n+ik)"
    elif "emissivity_visible_550nm" in b["properties"] or "emissivity" in b["properties"]:
        e = b["properties"].get("emissivity_visible_550nm") or b["properties"]["emissivity"]
        eps = [e["value"]] * 3
        out["sources"]["emissivity"] = "grey body: block() emissivity (" + e["source"] + ")"
    else:
        eps = [1.0] * 3
        out["gaps"].append("no emissivity: drawn as a blackbody")
        out["sources"]["emissivity"] = "blackbody (gap)"
    L = tuple(e * te.planck_spectral_radiance(lam, T) for e, lam in zip(eps, GLOW_SAMPLES_M))
    out["radiance_W_m2_sr_m"] = L
    out["sources"]["radiance"] = "thermal_emission.planck_spectral_radiance (Planck 1900) x emissivity"
    out["sources"]["visible"] = "thermal_emission.is_visibly_glowing (Draper point)"
    if max(L) > 0:
        out["rgb"] = tuple(v / max(L) for v in L)
    if out["visible"]:
        from . import thermal
        try:
            out["power_W_m2"] = float(thermal.thermal_emission_power(canon, T))
            out["level"] = te.glow_level(out["power_W_m2"])
            out["sources"]["level"] = "thermal_emission.glow_level(epsilon sigma T^4): a DISPLAY tonemap (DISPLAY_GLOW_P0_W_M2), not a physics claim"
        except (KeyError, ValueError):
            out["gaps"].append("no emissivity for the emitted power: no glow brightness")
    return out


__all__ = ["block", "appearance", "glow", "SAME_SUBSTANCE", "UNITS", "OPTICAL_MODEL"]
