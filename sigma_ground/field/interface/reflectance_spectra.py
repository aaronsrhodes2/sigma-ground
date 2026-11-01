"""
Measured visible reflectance of ordinary surfaces: rock, concrete, rubber, wood, ceramic (SkippyCraft P5).

These are the materials no optical model in sigma-ground reaches: a rock or a concrete is a scattering mixture of
grains (its body colour needs a multiple-scattering treatment, Kubelka-Munk at least, which sigma-ground lacks), wood
and rubber are organic (molecular transitions: quantum chemistry). So, as ``optics.ORGANIC_SPECTRA`` does for textiles,
the colour is MEASURED, with its source kept beside it, sampled at ``SAMPLE_NM`` (red, green, blue: the same three
wavelengths ``block.glow`` uses). Three kinds of record:

  - a full spectrum from the ECOSTRESS Spectral Library (JPL; formerly the ASTER library): directional (10 degree)
    hemispherical reflectance, the file kept whole in ``fixtures/ecostress/`` and interpolated at the sample wavelengths;
  - a measured CIE L*a*b* colour (a spectrophotometer reading reduced to three numbers): converted to linear sRGB
    reflectance exactly (Lab -> XYZ at its white point -> Bradford to D65 -> the sRGB matrix);
  - a measured BOUND, when that is all a source gives (alumina: "0.8 or more across 380-700 nm"): used as the value and
    flagged, so the readout says so.

What the values are: linear reflectance (fraction of light returned) at 650, 550 and 450 nm; a renderer encodes them to
its texture format (sRGB) itself. A hemispherical reflectance includes the specular part, so no F0 is derived here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_NM = (650.0, 550.0, 450.0)
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ecostress"

ECOSTRESS_SOURCE = ("ECOSTRESS Spectral Library v1.0 (Meerdink, Hook, Roberts & Abbott 2019, Remote Sens. Environ. 230, "
                    "111196; formerly the ASTER Spectral Library, Baldridge et al. 2009), speclib.jpl.nasa.gov")
# canonical material -> (file in fixtures/ecostress, what the sample was)
ECOSTRESS = {
    "granite": ("rock.igneous.felsic.solid.all.granite_h1.jhu.becknic.spectrum.txt",
                "alkalic granite, Quincy MA, solid rock, grey medium-coarse quartz-feldspar (JHU sample Granite_H1)"),
    "concrete": ("manmade.concrete.constructionconcrete.solid.all.0598uuucnc.jhu.becknic.spectrum.txt",
                 "30-year-old runway construction concrete, light grey, weathered (NPIC sample 0598UUUCNC)"),
    "rubber": ("manmade.roofingmaterial.rubber.solid.all.0833uuurbr.jhu.becknic.spectrum.txt",
               "black roofing rubber, weathered (NPIC sample 0833UUURBR)"),
}

# canonical material -> ((L*, a*, b*), white point, observer, source)
LAB = {
    "wood_oak": ((37.5, 10.5, 19.9), "D50", "2 deg",
                 "natural oak board, Gretag SPM50 spectrophotometer, 380-730 nm in 10 nm steps, mean of three readings "
                 "(US patent 11,578,226 B2, Table 14, 'Natural Wood, Oak')"),
}

# canonical material -> (reflectance bound, applies as, source)
BOUNDS = {
    "ceramic_alumina": (0.80, "lower bound, every wavelength 380-700 nm",
                        "sintered 96% alumina (Kyocera A-476) as a white reference plate: spectral reflectance 0.8 or more "
                        "over 380-700 nm under D50 (US patent 9,267,841, Color measurement device)"),
}

# CIE 1931 2 degree white points (XYZ, Y = 1) and the Bradford adaptation to D65
_WHITE = {"D50": (0.96422, 1.0, 0.82521), "D65": (0.95047, 1.0, 1.08883)}
_BRADFORD_D50_TO_D65 = np.array([[0.9555766, -0.0230393, 0.0631636],
                                 [-0.0282895, 1.0099416, 0.0210077],
                                 [0.0122982, -0.0204830, 1.3299098]])
_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])

MEASURED_SURFACES = frozenset(ECOSTRESS) | frozenset(LAB) | frozenset(BOUNDS)


def ecostress_spectrum(canon: str):
    """``(wavelength_nm, reflectance)`` ascending, as fractions, from the material's ECOSTRESS file."""
    name, _ = ECOSTRESS[canon]
    wl, r = [], []
    with open(FIXTURES / name, encoding="utf-8", errors="replace") as f:
        data = False
        for line in f:
            s = line.strip()
            if not data:
                if s.startswith("Additional Information"):
                    data = True
                continue
            if not s:
                continue
            x, y = s.split()[:2]
            wl.append(float(x) * 1000.0)
            r.append(float(y) / 100.0)
    wl, r = np.asarray(wl), np.asarray(r)
    order = np.argsort(wl)
    return wl[order], r[order]


def sample_rgb(wavelength_nm, reflectance):
    """The spectrum at ``SAMPLE_NM``, linearly interpolated."""
    return tuple(float(np.interp(lam, wavelength_nm, reflectance)) for lam in SAMPLE_NM)


def lab_to_linear_rgb(L: float, a: float, b: float, white: str = "D50"):
    """CIE L*a*b* (CIE 1976, with the standard white) -> linear sRGB (D65) reflectance, no clipping."""
    Xn, Yn, Zn = _WHITE[white]
    fy = (L + 16.0) / 116.0
    fx, fz = a / 500.0 + fy, fy - b / 200.0

    def inv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xyz = np.array([Xn * inv(fx), Yn * inv(fy), Zn * inv(fz)])
    if white == "D50":
        xyz = _BRADFORD_D50_TO_D65 @ xyz
    return tuple(float(v) for v in _XYZ_TO_SRGB @ xyz)


def measured_rgb(canon: str):
    """``{"rgb", "source", "flags"}`` for a material with a measured record, else None."""
    if canon in ECOSTRESS:
        name, what = ECOSTRESS[canon]
        rgb = sample_rgb(*ecostress_spectrum(canon))
        return {"rgb": rgb, "flags": [],
                "source": f"reflectance_spectra: {ECOSTRESS_SOURCE}; {what}; directional-hemispherical reflectance at "
                          f"650/550/450 nm from {name}"}
    if canon in LAB:
        (L, a, b), white, obs, src = LAB[canon]
        return {"rgb": lab_to_linear_rgb(L, a, b, white), "flags": [],
                "source": f"reflectance_spectra: measured CIE L*a*b* ({L}, {a}, {b}; {white}, {obs}) -> linear sRGB; {src}"}
    if canon in BOUNDS:
        value, how, src = BOUNDS[canon]
        return {"rgb": (value, value, value), "flags": [f"{canon}: colour is a measured bound only ({how}), drawn at it"],
                "source": f"reflectance_spectra: {how}; {src}"}
    return None


__all__ = ["SAMPLE_NM", "ECOSTRESS", "LAB", "BOUNDS", "MEASURED_SURFACES", "ecostress_spectrum", "sample_rgb",
           "lab_to_linear_rgb", "measured_rgb"]
