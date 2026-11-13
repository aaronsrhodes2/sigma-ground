"""
An osmium atom, its lattice and its electron sea, and what a photon does to them: the numbers behind the "atoms" stage.

The picture the numbers support (and what is a cartoon):
  Atom        76 electrons round a nucleus 7 femtometres across. Slater's rules (1930) give each group of electrons an effective charge
              Z_eff and a mean radius n*^2 a0 / Z_eff: from 0.0007 A for the two 1s electrons to ~3.7 A for the 6s. Those radii are where
              a cloud of probability peaks, not an orbit: no electron travels a circle. The bound inner groups (everything but the outer
              6s and 5d) are held by 10s to 1000s of eV, far above a photon's 2.5 eV, and do not respond to visible light.
  Metal       the 6s electrons (2 per atom, ``element.free_electron_count``) do not belong to an atom: they are a sea of density n_e that
              fills the lattice (hcp, ``element.predict_crystal_structure``). A sea of charge sloshes at its plasma frequency
              omega_p = sqrt(n_e e^2 / (eps_0 m_e)).
  Light       a wave of angular frequency omega below omega_p cannot get into the sea: the electrons follow the field and re-radiate it
              backwards (reflection), within a skin depth c / omega_p of the surface. At atomic scale (0.5 A a block) a 500 nm wave is a
              uniform field that flips every half period (1.7 fs).
  Recoil      the momentum h / lambda is taken by the whole lattice, every one of the ~10^15 atoms of the grain sharing it.

Origin tags:
  - Electron configuration, Slater's rules, hcp structure and lattice parameter, free-electron count, atomic mass: the library
    (``element``): Slater radii are APPROXIMATIONS good to ~30-40% (the outer ones overstate the metallic radius ~2x); the predicted lattice
    parameter is 2.56 A against 2.73 A measured
  - Electron density of the sea: ``superconductivity.SUPERCONDUCTORS['osmium']['n_e_m3']`` (2 per atom of the measured density)
  - Plasma frequency, skin depth, photon frequency, momentum: FIRST_PRINCIPLES (Drude free-electron model; collisionless)
  - Nuclear radius: ``radioactive_decay.nuclear_radius_fm`` (r0 A^(1/3))

Not in it (ATOM_GAPS): the 5d electrons (6 per atom) are drawn as bound core, though in osmium they hybridise with the 6s into the bands and
carry much of the conduction; the sea's density is uniform, not the lattice's Bloch waves; the electron's response is the free-electron Drude
one (no interband transitions, which dominate osmium's colour in the visible); the 5% of light osmium absorbs is not followed into the lattice's
phonons; a displacement shown on screen is exaggerated and labelled; the nucleus is a point at this scale.
"""

import math

from ..constants import C, H_PLANCK
from . import element as _el
from . import radioactive_decay as _rd
from . import superconductivity as _sc

Z = 76
MASS_NUMBER = 192                      # the most abundant isotope (41%)
E_CHARGE = 1.602176634e-19             # C, exact
EPS_0 = 8.8541878128e-12               # F/m
M_E = 9.1093837015e-31                 # kg
EV_J = 1.602176634e-19                 # J per eV
A0_M = 0.529177210903e-10              # Bohr radius, m
ATOM_GAPS = (
    "The 5d electrons are drawn as bound core; in osmium they hybridise into the bands and carry much of the conduction.",
    "The electron sea is uniform at this scale; a real one is Bloch waves of the lattice.",
    "The response is the free-electron Drude one: osmium's interband transitions, which dominate its colour in the visible, are not in it.",
    "Slater's radii are approximate (the outer ones overstate the metallic radius about twofold); a shell's radius is where its cloud peaks, not an orbit.",
    "The predicted lattice parameter (2.56 A) is below the measured 2.73 A.",
    "The nucleus is a point at the stage's scale, and the 5% of light the metal absorbs is not followed into the lattice's vibrations.",
)

_N_STAR = _el._N_EFFECTIVE


def shell_groups(z: int = Z) -> list:
    """Slater's groups of electrons for element ``z``, innermost first: ``{label, n, electrons, zeff, radius_m, binding_ev}``.

    Each group's Z_eff follows Slater's shielding rules for an electron of that group (the library's ``slater_zeff`` does the outermost);
    its mean radius is n*^2 a0 / Z_eff and its binding energy 13.6 eV Z_eff^2 / n*^2 (hydrogen-like, an estimate)."""
    groups = _el._slater_groups(_el.aufbau_configuration(z))
    out = []
    for idx, (label, n, l_type, count) in enumerate(groups):
        if count <= 0:
            continue
        s = (count - 1) * (0.30 if label == "1s" else 0.35)
        for i in range(idx - 1, -1, -1):
            inner = groups[i][3]
            if l_type == "sp":
                s += inner * (0.85 if i == idx - 1 else 1.00)
            else:
                s += inner * 1.00
        zeff = max(z - s, 0.1)
        n_star = _N_STAR.get(n, n * 0.9)
        out.append({"label": label, "n": n, "electrons": count, "zeff": zeff, "radius_m": n_star ** 2 * A0_M / zeff,
                    "binding_ev": 13.6057 * zeff ** 2 / n_star ** 2})
    return out


def nucleus_radius_m(mass_number: int = MASS_NUMBER) -> float:
    return _rd.nuclear_radius_fm(mass_number) * 1e-15


def lattice() -> dict:
    """The crystal: its structure, the nearest-neighbour spacing a, and the spacing of the close-packed layers (hcp: a sqrt(2/3) with c/a = sqrt(8/3))."""
    a = _el.predict_lattice_parameter_m(Z)
    return {"structure": _el.predict_crystal_structure(Z), "a_m": a, "layer_spacing_m": a * math.sqrt(2.0 / 3.0),
            "atom_mass_kg": _el.atomic_mass_kg(Z), "free_electrons_per_atom": _el.free_electron_count(Z)}


def atoms_in_grain(radius_m: float, density_kg_m3: float = 22590.0) -> float:
    """How many osmium atoms a sphere of this radius holds (measured density, CRC)."""
    return density_kg_m3 * 4.0 / 3.0 * math.pi * radius_m ** 3 / _el.atomic_mass_kg(Z)


def sea() -> dict:
    """The conduction electrons: density, plasma frequency, and the skin depth within which light is turned back."""
    n_e = _sc.SUPERCONDUCTORS["osmium"]["n_e_m3"]
    w_p = math.sqrt(n_e * E_CHARGE ** 2 / (EPS_0 * M_E))
    return {"n_e_m3": n_e, "plasma_angular_hz": w_p, "plasma_energy_ev": H_PLANCK / (2 * math.pi) * w_p / EV_J,
            "skin_depth_m": C / w_p, "plasma_wavelength_m": 2 * math.pi * C / w_p}


def light(wavelength_m: float = 500e-9) -> dict:
    """A photon of this wavelength: its frequency, period, energy, and whether the sea turns it back (omega below the plasma frequency)."""
    omega = 2 * math.pi * C / wavelength_m
    s = sea()
    return {"angular_hz": omega, "period_s": 2 * math.pi / omega, "energy_ev": H_PLANCK * C / wavelength_m / EV_J,
            "momentum_kg_m_s": H_PLANCK / wavelength_m, "reflected": omega < s["plasma_angular_hz"],
            "omega_over_plasma": omega / s["plasma_angular_hz"], "wavelength_in_blocks": lambda_blocks(wavelength_m)}


def lambda_blocks(wavelength_m: float, block_m: float = 0.5e-10) -> float:
    return wavelength_m / block_m


def recoil_per_atom(wavelength_m: float, radius_m: float) -> float:
    """The momentum each atom of the grain takes when the whole lattice takes the photon's: h / (lambda N), kg m / s."""
    return H_PLANCK / wavelength_m / atoms_in_grain(radius_m)
