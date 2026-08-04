"""
Ore abundance — asteroid spectral taxonomy -> real mineral assemblage.

STANDALONE physics/chemistry module (P0b of the game-engine plan): what is
an asteroid of a given SPECTRAL CLASS actually made of, in named minerals
with mass fractions, and what would it cost (energetically) to extract a
named material from it? This is the piece the game-engine plan flagged as
genuinely missing — the bond-energy machinery (adhesion.py, stress.py) and
the mass-closure resolver (inventory/) already existed; a real ore-ABUNDANCE
model connecting "C-type asteroid" to "which minerals, how much" did not.

Taxonomy basis: the Bus-DeMeo taxonomy (DeMeo, Binzel, Slivan & Bus, 2009,
"An Extension of the Bus Asteroid Taxonomy into the Near-Infrared," Icarus
202:160-180 — a 24-class system built on 0.45-2.45 micron PCA of 371
asteroids), which itself extends Bus & Binzel (2002, Icarus 158:146-177,
visible-wavelength Bus taxonomy) and traces back to the original C/S/M/etc.
definitions in Tholen, D.J. (1984, PhD dissertation, University of Arizona).
Tholen's M-type (a spectra+albedo class: featureless red spectrum AND
0.1 < p_v < 0.3) does not have a clean Bus-DeMeo equivalent — spectra-only
classification puts most former Tholen M-types into the Bus-DeMeo X-complex
(Xc/Xk subclasses), which is spectrally DEGENERATE with E-type (enstatite)
and P-type (primitive) without an independent albedo measurement. This
module discloses that ambiguity explicitly (see `_ALIASES["x"]`) rather than
silently assuming every X-complex body is metallic.

Meteorite analogs (asteroid spectral class <-> real meteorite group is
standard, mainstream planetary science, not an invented mapping):
  C-complex (B, C, Cb, Cg subtypes)  -> CI (Ivuna-type) carbonaceous
      chondrite, per direct sample-return confirmation: Yokoyama, T. et al.
      (2023), "Samples returned from the asteroid Ryugu are similar to
      Ivuna-type carbonaceous meteorites," Science 379, eabn7850; and the
      real Bennu sample-overview paper, Lauretta, D.S., Connolly, H.C. et
      al. (2024), "Asteroid (101955) Bennu in the laboratory," Meteoritics
      & Planetary Science 59:2453-2486 (both compare their returned samples
      to CI, not CM — a citation correction applied here, see below).
  C-complex hydrated subtypes (Cgh, Ch) -> CM (Mighei-type) carbonaceous
      chondrite, per the Ch/Cgh hydration-band spectral link to CM
      mineralogy (standard meteoritics; CM chondrites are phyllosilicate-
      dominated, water/organic-bearing, distinct from CI mainly in degree
      of aqueous alteration and matrix mineralogy).
  S-complex -> ordinary chondrites (H/L/LL chemical groups), per Chapman,
      C.R. (1996), "S-type asteroids, ordinary chondrites, and space
      weathering," Meteoritics & Planetary Science 31:699-725, directly
      confirmed for the LL group by in-situ sample return: Nakamura, T. et
      al. (2011), "Itokawa dust particles: A direct link between S-type
      asteroids and ordinary chondrites," Science 333:1113-1116.
  Tholen M-type / Bus-DeMeo Xc,Xk -> iron meteorites (Fe-Ni metal,
      magmatic core-type), per Fornasier, S., Clark, B.E., Dotto, E.,
      Migliorini, A., Ockert-Bell, M., Barucci, M.A. (2010), Icarus
      210:655-673 ("iron meteorites, pallasites, and enstatite chondrites
      are the best matches to most objects in our sample... not all
      objects defined by the Tholen M-class have a pure metallic
      composition" — a direct quote, correctly attributed here; the
      original claim mis-cited this to Ockert-Bell as first author, whose
      own companion paper at the same volume/issue is Ockert-Bell, M.E. et
      al. (2010), Icarus 210:674-692, "The composition of M-type
      asteroids: Synthesis of spectroscopic and radar observations").
      Caveat (mission-confirmed, not spectral inference): many M-type
      bodies are NOT near-pure metal — Elkins-Tanton, L.T. et al. (2020),
      "Observations, Meteorites, and Models: A Preflight Assessment of the
      Composition and Formation of (16) Psyche," JGR: Planets 125,
      e2019JE006296, predicts Psyche is only ~30-60 vol% metal. This module
      keeps the classic "near-pure iron meteorite" analog as its headline
      Choice (it is what Tholen's M-type / the iron-meteorite literature
      historically means) but discloses the Psyche caveat in the M-type
      assemblage's residual ("rest") component.

CITATION CORRECTIONS APPLIED (this module was built from adversarially
verified research; several source citations in the raw research came back
"real, topical document, but does not contain this specific number/author
list" and were corrected here rather than shipped as given):
  - Dunn, T.L. et al. (2010), Meteoritics & Planetary Science 45(1):123-134
    ("...powder X-ray diffraction: 1. Modal mineral abundances") — the
    correct author list, per CrossRef DOI 10.1111/j.1945-5100.2009.01011.x,
    is Dunn, T.L., Cressey, G., McSween, H.Y., McCoy, T.J. (NOT "Dunn,
    McCoy, Sunshine, McSween," which is the byline of a different, real
    2010 Dunn-group paper in Icarus 208:789-797).
  - M-type Fe-Ni bulk metal fraction and bulk Co content: re-attributed to
    Buchwald, V.F. (1975), Handbook of Iron Meteorites (Goldstein, Scott &
    Chabot 2009, Chemie der Erde 69:293-325, is real and topical but does
    NOT itself state the bulk metal wt%; it explicitly defers to Buchwald
    1975 for exactly this kind of bulk-composition figure).
  - Ni content, chemical group IIIAB: Yang, J. & Goldstein, J.I. (2006),
    "Metallographic cooling rates of the IIIAB iron meteorites," GCA
    70(12):3197-3215 (NOT a "Chabot et al. 2006/2022" composite — no such
    single paper exists; the quoted 7.1-8.6 wt% Ni figure is Yang &
    Goldstein's Table 3).
  - Ni content, chemical group IIAB: Wasson, J.T. & Kallemeyn, G.W.
    (2002), "The IAB iron-meteorite complex," GCA 66(13):2445-2473 (not
    Wasson & Choe 2009, which discusses a different group's formation
    history, not the IIAB Ni wt% range).
  - The "unresolved matrix" residual bucket on the S-type assemblage
    absorbs high-Ca pyroxene and troilite (FeS) — both are among the six
    phases Dunn et al. (2010) itself quantifies, but no specific wt% figure
    for either was in the adversarially-verified dataset this module was
    built from, so no number is invented for them.

Every mass fraction below is either a FACT (a single number/class-average
directly stated by a cited source) or a disclosed Choice (see
dynamics.mechanisms.choice.Choice — a range midpoint, a representative
group pick, or a mass-balance residual), never an invented number presented
as measured. Every mineral is either a resolved key into
inventory/data/materials.json or field/interface/surface.py's MATERIALS, or
an explicitly flagged, named gap (see MineralComponent.gap_note) — nothing
is silently dropped or misrepresented as an existing material.

SIMPLIFIED_MODEL disclosures (stated plainly, not hidden):
  1. vol% used as mass% proxy: the best-sourced modal mineralogy in this
     dataset (King, A.J. et al. 2015, GCA 165:148-160, CI chondrites) is
     reported in VOLUME percent (PSD-XRD whole-pattern profile-stripping,
     cross-validated against Rietveld refinement — not literally a
     Rietveld analysis itself), not mass percent. Converting properly
     needs per-phase densities for every phase, which were not available
     for all phases in the verified dataset; this module uses the vol%
     figures directly as a first-order mass% proxy and discloses this
     explicitly rather than performing a partial, cherry-picked density
     correction.
  2. Extraction energy (extraction_energy_per_kg, below) uses the pure
     ELEMENT's cohesive energy (surface.py MATERIALS' cohesive_energy_ev)
     as a lower-bound proxy for "how tightly bound is this material to its
     matrix." This is explicitly NOT the thermodynamically correct
     quantity for ore reduction (that is the ORE COMPOUND's Gibbs free
     energy of formation ΔG_f — the actual bond broken during extraction,
     per the Ellingham-diagram framework: Ellingham, H.J.T. (1944), J.
     Soc. Chem. Ind. 63:125; Rankin, W.J. (2011), Minerals, Metals and
     Sustainability, CSIRO Publishing, Ch. 9). Pure-metal cohesive energy
     and ore-compound formation energy are different, only loosely
     correlated quantities (independently re-derived check: Na cohesive
     energy 1.113 eV/atom -> 4.67 MJ/kg vs NaCl's ΔH_f -> 17.9 MJ/kg per
     kg Na, a ~4x gap; Al cohesive energy 3.39 eV/atom -> 12.1 MJ/kg vs
     the actual Hall-Heroult electrolytic-reduction energy of 47-54 MJ/kg,
     a ~4x gap even against the THEORETICAL Al2O3 floor of ~31 MJ/kg —
     MDPI (2018), "Energy and Exergy Analyses of Different Aluminum
     Reduction Technologies"; ACEEE (2003), "U.S. Aluminum Production
     Energy Requirements"). Real industrial refining also includes real,
     substantial non-ideal losses this module does NOT capture: process
     inefficiency (Gutowski, T.G. et al. (2013), "The energy required to
     produce materials," Phil. Trans. R. Soc. A 371:20120003, finds even
     the best real processes run ~60% efficient against the thermodynamic
     floor) and, for low-grade ore, comminution/beneficiation energy that
     can dominate the total budget entirely (Valero, A. & Valero, A.,
     exergoecology "exergy replacement cost" work). The literature-wide
     ratio of real industrial energy to a cohesive/formation-energy floor
     ranges roughly 2x (iron/steel) to >20x (titanium), strongly boundary-
     and process-route-dependent — treat as an order-of-magnitude anchor,
     not a precise prediction. extraction_energy_per_kg's return value
     says this plainly in its own "simplified_model_caveat" field.
  3. No sigma-field dependence: mineral abundance and extraction
     energetics here are ordinary chemistry/geology (solar-system asteroid
     belt, sigma ~ 0), not SSBM's spacetime-compression physics — unlike
     surface.py/adhesion.py, this module does not thread a sigma argument.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...dynamics.mechanisms.choice import Choice
from ...inventory.data.loader import MaterialDB
from ..constants import AMU_KG, EV_TO_J
from .adhesion import work_of_adhesion
from .surface import MATERIALS as SURFACE_MATERIALS


# ═══════════════════════════════════════════════════════════════════════════
# § 1. Provenance data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MineralComponent:
    """One mineral/phase entry in a spectral class's ore assemblage.

    provenance is "fact" when mass_fraction is a number directly stated by
    a single cited source (e.g. a reported class average); "choice" when
    producing it required a judgment call (a range midpoint, a
    representative-group pick, a mass-balance residual) — disclosed via a
    real Choice object (dynamics.mechanisms.choice.Choice), same pattern
    radiance/push_off.py uses for its friction/restitution/push-speed
    values.

    Every component resolves to a real database key (material_key_surface
    into field/interface/surface.py's MATERIALS, and/or
    material_key_inventory into inventory/data/materials.json — case-
    insensitive "name" match) OR carries a non-empty gap_note naming the
    real mineral this codebase has no physical data for. Never both empty.
    """
    component_key: str
    mineral_name: str
    mass_fraction: float
    provenance: str                       # "fact" | "choice"
    source: str
    material_key_surface: str | None = None
    material_key_inventory: str | None = None
    gap_note: str | None = None
    note: str = ""
    choice: Choice | None = None

    def __post_init__(self):
        if self.provenance not in ("fact", "choice"):
            raise ValueError(
                f"{self.component_key}: provenance must be 'fact' or "
                f"'choice', got {self.provenance!r}")
        if not (0.0 <= self.mass_fraction <= 1.0 + 1e-9):
            raise ValueError(
                f"{self.component_key}: mass_fraction {self.mass_fraction} "
                f"out of [0, 1]")
        if (self.material_key_surface is None
                and self.material_key_inventory is None
                and not self.gap_note):
            raise ValueError(
                f"{self.component_key}: unresolved component (no "
                f"material_key_surface, no material_key_inventory) must "
                f"carry a gap_note naming the real mineral")
        if self.provenance == "choice" and self.choice is None:
            raise ValueError(
                f"{self.component_key}: provenance='choice' components "
                f"must carry a Choice object")

    @property
    def resolved(self) -> bool:
        return (self.material_key_surface is not None
                or self.material_key_inventory is not None)

    def to_dict(self) -> dict:
        d = {
            "component_key": self.component_key,
            "mineral_name": self.mineral_name,
            "mass_fraction": self.mass_fraction,
            "provenance": self.provenance,
            "resolved": self.resolved,
            "material_key_surface": self.material_key_surface,
            "material_key_inventory": self.material_key_inventory,
            "gap_note": self.gap_note,
            "source": self.source,
            "note": self.note,
        }
        if self.choice is not None:
            d["choice"] = self.choice.to_dict()
        return d


@dataclass(frozen=True)
class OreAssemblage:
    """A meteorite-analog mineral assemblage for one taxonomy class.

    component mass fractions must sum to 1.0 (validated at construction,
    same "If One, Then All" fail-loud style as PropulsionSystem in
    propulsion.py) — the whole body is accounted for, including whatever
    is NOT itemized by the cited sources (see each assemblage's "rest"
    component).
    """
    assemblage_key: str
    taxonomy_system: str
    meteorite_analog: str
    components: tuple[MineralComponent, ...]
    source: str
    note: str = ""

    def __post_init__(self):
        total = sum(c.mass_fraction for c in self.components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{self.assemblage_key}: component mass fractions sum to "
                f"{total!r}, not 1.0")
        keys = [c.component_key for c in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"{self.assemblage_key}: duplicate component_key in "
                f"{keys}")

    def to_dict(self) -> dict:
        return {
            "assemblage_key": self.assemblage_key,
            "taxonomy_system": self.taxonomy_system,
            "meteorite_analog": self.meteorite_analog,
            "components": [c.to_dict() for c in self.components],
            "mass_fraction_total": sum(c.mass_fraction
                                       for c in self.components),
            "source": self.source,
            "note": self.note,
        }


_BUS_DEMEO = ("Bus-DeMeo taxonomy (DeMeo, Binzel, Slivan & Bus, 2009, "
              "Icarus 202:160-180; extends Bus & Binzel 2002, Icarus "
              "158:146-177; builds on Tholen 1984 PhD dissertation, "
              "University of Arizona)")
_THOLEN_TO_X = ("Tholen (1984) M-type (spectra+albedo, 0.1<p_v<0.3), "
                "refined into the Bus-DeMeo X-complex's Xc/Xk subclasses "
                "by Bus & Binzel (2002) / DeMeo et al. (2009) — spectra "
                "alone cannot separate metallic Xc/Xk from non-metallic "
                "E-type/P-type without albedo")


# ═══════════════════════════════════════════════════════════════════════════
# § 2. C-complex: CI (Ivuna-type carbonaceous chondrite)
# ═══════════════════════════════════════════════════════════════════════════
# King, A.J., Schofield, P.F., Howard, K.T., Russell, S.S. (2015), "Modal
# mineralogy of CI and CI-like chondrites by X-ray diffraction," GCA
# 165:148-160 — PSD-XRD whole-pattern profile-stripping (cross-validated
# against, but not literally, Rietveld refinement) of CI falls Alais,
# Orgueil, Ivuna. Reports phyllosilicate 81-84 vol%, magnetite 6-10 vol%,
# sulfides 4-7 vol%, carbonates <3 vol% — used here (SIMPLIFIED_MODEL,
# see module docstring) as the mass-fraction basis.

_ci_phyllosilicate_choice = Choice(
    "CI phyllosilicate (Mg-serpentine + Fe-saponite/cronstedtite) mass "
    "fraction",
    0.825,
    checked="King et al. (2015), GCA 165:148-160: 81-84 vol% phyllosilicate "
            "across CI falls Alais/Orgueil/Ivuna",
    reason="range midpoint; vol% used directly as a mass% proxy "
           "(SIMPLIFIED_MODEL — no per-phase density correction applied)",
    adjustable=True)
_CI_PHYLLOSILICATE = MineralComponent(
    component_key="phyllosilicate",
    mineral_name="Mg-serpentine + Fe-saponite/cronstedtite (hydrous "
                 "phyllosilicate)",
    mass_fraction=_ci_phyllosilicate_choice.value,
    provenance="choice",
    source="King, A.J., Schofield, P.F., Howard, K.T., Russell, S.S. "
           "(2015), GCA 165:148-160",
    gap_note="hydrous Mg-Fe sheet silicate; no phyllosilicate entry exists "
             "in materials.json or surface.py MATERIALS",
    note="the dominant CI phase by far; volatile/water-bearing (hydroxyl "
         "sheet silicate)",
    choice=_ci_phyllosilicate_choice)

_ci_magnetite_choice = Choice(
    "CI magnetite (Fe3O4) mass fraction", 0.08,
    checked="King et al. (2015), GCA 165:148-160: 6-10 vol% magnetite",
    reason="range midpoint; vol% used as mass% proxy",
    adjustable=True)
_CI_MAGNETITE = MineralComponent(
    component_key="magnetite",
    mineral_name="Magnetite (Fe3O4)",
    mass_fraction=_ci_magnetite_choice.value,
    provenance="choice",
    source="King, A.J., Schofield, P.F., Howard, K.T., Russell, S.S. "
           "(2015), GCA 165:148-160",
    gap_note="Fe3O4 not represented; materials.json's Iron(III) Oxide is "
             "Fe2O3 (hematite) — a compositionally distinct iron oxide, "
             "not substituted here",
    choice=_ci_magnetite_choice)

_ci_sulfides_choice = Choice(
    "CI Fe-Ni sulfide (pyrrhotite/pentlandite-type) mass fraction", 0.055,
    checked="King et al. (2015), GCA 165:148-160 four-phase modal scheme: "
            "sulphides 4-7 vol% (corroborated via independent secondary "
            "sources reproducing the paper's full mineral-abundance set)",
    reason="range midpoint; vol% used as mass% proxy",
    adjustable=True)
_CI_SULFIDES = MineralComponent(
    component_key="sulfides",
    mineral_name="Fe-Ni sulfide (pyrrhotite/pentlandite-type)",
    mass_fraction=_ci_sulfides_choice.value,
    provenance="choice",
    source="King, A.J., Schofield, P.F., Howard, K.T., Russell, S.S. "
           "(2015), GCA 165:148-160",
    gap_note="no Fe-Ni sulfide (pyrrhotite/pentlandite/troilite-family) "
             "entry exists in either table",
    choice=_ci_sulfides_choice)

_ci_carbonates_choice = Choice(
    "CI carbonate (dolomite/breunnerite-type) mass fraction", 0.015,
    checked="King et al. (2015), GCA 165:148-160: carbonates <3 vol% "
            "(the number); species identification dolomite/breunnerite-"
            "type (Ca,Fe,Mg)CO3 solid solution per Fredriksson, K. & "
            "Kerridge, J.F. (1988), Meteoritics 23:35-44 (qualitative "
            "petrology only — no modal number is claimed from this "
            "source, correcting a citation/number mismatch in the raw "
            "research)",
    reason="half of the stated '<3%' upper bound, taken as a range "
           "midpoint against an implicit 0% floor",
    adjustable=True)
_CI_CARBONATES = MineralComponent(
    component_key="carbonates",
    mineral_name="Dolomite/breunnerite-type (Ca,Fe,Mg)CO3 carbonate",
    mass_fraction=_ci_carbonates_choice.value,
    provenance="choice",
    source="King, A.J. et al. (2015), GCA 165:148-160 (number); "
           "Fredriksson, K. & Kerridge, J.F. (1988), Meteoritics 23:35-44 "
           "(mineral species)",
    gap_note="(Ca,Fe,Mg)CO3 solid solution not represented; materials."
             "json's Calcium Carbonate is pure CaCO3 (no Fe/Mg), "
             "compositionally distinct, not substituted here. Gypsum "
             "(CaSO4.2H2O) deliberately excluded from this component — "
             "the meteoritics literature treats gypsum in CI/CM finds as "
             "predominantly a terrestrial weathering artifact, not an "
             "indigenous parent-body phase",
    choice=_ci_carbonates_choice)

_CI_FREE_METAL = MineralComponent(
    component_key="free_metal",
    mineral_name="Free Fe-Ni metal (kamacite/taenite)",
    mass_fraction=0.0,
    provenance="fact",
    source="Weisberg, M.K., McCoy, T.J., Krot, A.N. (2006), 'Systematics "
           "and Evaluation of Meteorite Classification,' in Meteorites and "
           "the Early Solar System II, Table 1: CI metal abundance = 0 "
           "vol%, the unique zero-metal chondrite group (vs. CM 0.1%, CO "
           "1-5%, H 8%, L 4%, LL 2%, etc.)",
    material_key_surface="iron",
    material_key_inventory="Iron",
    note="CI is uniquely metal-free among chondrite groups — resolved key "
         "given for completeness (Fe-Ni alloy; 'iron' stands in for the "
         "alloy, Ni not separately split since the fraction itself is "
         "zero); contributes nothing to recoverable mass")

_ci_rest_fraction = 1.0 - (_CI_PHYLLOSILICATE.mass_fraction
                          + _CI_MAGNETITE.mass_fraction
                          + _CI_SULFIDES.mass_fraction
                          + _CI_CARBONATES.mass_fraction
                          + _CI_FREE_METAL.mass_fraction)
_ci_rest_choice = Choice(
    "CI unresolved matrix/residual mass fraction", _ci_rest_fraction,
    checked="King et al. (2015)'s four-row modal scheme (phyllosilicate + "
            "magnetite + sulfides + carbonates) sums to ~97.5% at range "
            "midpoints, not exactly 100% (real modal-XRD measurement "
            "uncertainty)",
    reason="closes the mass balance to 1.0 exactly; also carries the CI "
           "bulk organic (macromolecular IOM + soluble fraction) carbon "
           "content, ~3.5-3.9 wt% per Alexander, C.M.O'D. et al. (2012), "
           "Science 337:721-723 (Orgueil 3.52 wt%, Ivuna 3.50 wt%, per "
           "Alexander's own Table as reproduced in a citing paper's "
           "supplementary data) — NOT summed as a separate independent "
           "component because organic matter is physically intermixed "
           "with the fine-grained phyllosilicate matrix and XRD's "
           "'phyllosilicate' bin does not cleanly separate it; itemizing "
           "both would double-count matrix mass",
    adjustable=True)
_CI_REST = MineralComponent(
    component_key="rest",
    mineral_name="Unresolved CI matrix/residual (incl. ~3.5-3.9 wt% bulk "
                 "organic carbon, intermixed with the phyllosilicate "
                 "matrix, and modal-measurement residual)",
    mass_fraction=_ci_rest_choice.value,
    provenance="choice",
    source="derived residual — see Choice.checked",
    gap_note="not itemized by King et al. (2015)'s four-row scheme; "
             "includes organic macromolecular matter (IOM), not "
             "substituted with elemental Carbon/Graphite (different "
             "bonding chemistry: C,H,N,O,S heteroatom macromolecule vs. "
             "a pure sp2/sp3 carbon lattice)",
    choice=_ci_rest_choice)

CI_ASSEMBLAGE = OreAssemblage(
    assemblage_key="CI",
    taxonomy_system=_BUS_DEMEO,
    meteorite_analog="CI (Ivuna-type) carbonaceous chondrite",
    components=(_CI_PHYLLOSILICATE, _CI_MAGNETITE, _CI_SULFIDES,
                _CI_CARBONATES, _CI_FREE_METAL, _CI_REST),
    source="Direct sample-return confirmation: Yokoyama, T. et al. (2023), "
           "Science 379, eabn7850 (Ryugu); Lauretta, D.S., Connolly, H.C. "
           "et al. (2024), Meteoritics & Planetary Science 59:2453-2486 "
           "(Bennu) — both compare returned samples to CI specifically. "
           "Modal mineralogy: King, A.J. et al. (2015), GCA 165:148-160.",
    note="Water/volatile-rich relative to S-type and M-type: the "
         "phyllosilicate (hydroxyl-bearing) fraction alone is ~82.5%, and "
         "the rest bucket carries several wt% bulk organic carbon — "
         "neither S-type nor M-type has any comparable hydrous or "
         "carbon-rich component.")


# ═══════════════════════════════════════════════════════════════════════════
# § 3. C-complex hydrated subtypes: CM (Mighei-type carbonaceous chondrite)
# ═══════════════════════════════════════════════════════════════════════════
# Only three CM-specific quantitative figures were in the verified dataset
# (calcite+gypsum, indigenous water, organic carbon); CM's own phyllosilicate
# modal wt% was NOT separately verified (unlike CI's King et al. 2015
# figure), so it is honestly left in the "rest" bucket rather than invented
# by analogy to CI.

_cm_calcite_choice = Choice(
    "CM calcite (CaCO3) mass fraction", 0.0095,
    checked="Howard, K.T. et al. (2011), GCA 75:2735-2751: calcite "
            "0-1.9 wt% (independently corroborated figure, same PSD-XRD "
            "lab/methodology as King et al. 2015)",
    reason="range midpoint",
    adjustable=True)
_CM_CALCITE = MineralComponent(
    component_key="calcite",
    mineral_name="Calcite (CaCO3)",
    mass_fraction=_cm_calcite_choice.value,
    provenance="choice",
    source="Howard, K.T. et al. (2011), GCA 75:2735-2751",
    material_key_surface=None,
    material_key_inventory="Calcium Carbonate",
    note="gypsum (CaSO4.2H2O, also reported by Howard et al. 2011 at "
         "0-0.8 wt%) is deliberately excluded — the same paper's own TGA "
         "evidence (sulfate breakdown near 120C attributed to terrestrial "
         "alteration) treats gypsum in CM finds as predominantly a "
         "terrestrial weathering artifact, not an indigenous phase",
    choice=_cm_calcite_choice)

_CM_WATER = MineralComponent(
    component_key="water",
    mineral_name="Indigenous bulk H2O (bound in phyllosilicate structure "
                 "+ minor free water)",
    mass_fraction=0.070,
    provenance="fact",
    source="Lee, M.R. et al. (2023), 'The water content of CM carbonaceous "
           "chondrite falls and finds...,' Meteoritics & Planetary "
           "Science, doi:10.1111/maps.14099: 1.9-10.5 wt% indigenous "
           "water across 20 CM falls/finds (stepwise pyrolysis, corrected "
           "for insoluble organic matter), CLASS AVERAGE 7.0 +/- 2.4 wt% "
           "— the average is used directly (not a range midpoint)",
    material_key_surface="water_ice",
    material_key_inventory="Water",
    note="chemically H2O regardless of whether bound in phyllosilicate "
         "interlayers or present as minor free water; 'water_ice' "
         "(surface.py) supplies a real hydrogen-bond cohesive energy for "
         "extraction_energy_per_kg, understanding that the true binding "
         "in a hydrated silicate is not literally crystalline ice")

_cm_organic_choice = Choice(
    "CM organic (total) carbon mass fraction", 0.02,
    checked="raw research gave ~2 wt% via Sephton (2002), Nat. Prod. Rep. "
            "19:292-311, but the specific figure's attachment to that "
            "exact paper could not be independently confirmed (secondary "
            "sources trace closely matching 0.1-5 wt% carbonaceous-"
            "chondrite figures to a DIFFERENT Sephton co-authored paper, "
            "Pearson et al. 2006, Meteoritics & Planetary Science "
            "41:1899-1918) — kept as a conservative order-of-magnitude "
            "estimate, not upgraded to a Fact",
    reason="within the well-documented literature range for carbonaceous "
           "chondrite organic carbon (~1-5 wt%)",
    adjustable=True)
_CM_ORGANIC = MineralComponent(
    component_key="organic_carbon",
    mineral_name="Organic carbon (macromolecular insoluble organic matter "
                 "+ soluble organic fraction)",
    mass_fraction=_cm_organic_choice.value,
    provenance="choice",
    source="Sephton, M.A. (2002), Nat. Prod. Rep. 19:292-311 (topically "
           "on-point review; exact figure attribution unconfirmed — see "
           "Choice.checked)",
    gap_note="complex heterogeneous C,H,N,O,S macromolecule; not "
             "substituted with elemental Carbon/Graphite (different "
             "bonding chemistry)",
    choice=_cm_organic_choice)

_cm_rest_fraction = 1.0 - (_CM_CALCITE.mass_fraction
                          + _CM_WATER.mass_fraction
                          + _CM_ORGANIC.mass_fraction)
_cm_rest_choice = Choice(
    "CM unresolved matrix/residual mass fraction", _cm_rest_fraction,
    checked="only calcite, indigenous water, and organic carbon had "
            "CM-specific verified quantitative figures in the sourced "
            "dataset; CM is qualitatively described as phyllosilicate-"
            "dominated like CI, but no CM-specific modal wt% figure "
            "(analogous to CI's King et al. 2015 number) was verified",
    reason="closes the mass balance to 1.0; deliberately NOT filled in by "
           "analogy to CI's phyllosilicate figure, to avoid presenting an "
           "uncited number as measured",
    adjustable=True)
_CM_REST = MineralComponent(
    component_key="rest",
    mineral_name="Unresolved CM matrix — phyllosilicate matrix (serpentine"
                 "/saponite family, CI-like but no CM-specific modal "
                 "figure verified), relict anhydrous silicate grains "
                 "(olivine/pyroxene chondrules), magnetite/sulfides, and "
                 "excluded gypsum",
    mass_fraction=_cm_rest_choice.value,
    provenance="choice",
    source="derived residual — see Choice.checked",
    gap_note="no CM-specific phyllosilicate modal wt% figure was in the "
             "verified dataset; not itemized further rather than invented "
             "by analogy to CI",
    choice=_cm_rest_choice)

CM_ASSEMBLAGE = OreAssemblage(
    assemblage_key="CM",
    taxonomy_system=_BUS_DEMEO,
    meteorite_analog="CM (Mighei-type) carbonaceous chondrite",
    components=(_CM_CALCITE, _CM_WATER, _CM_ORGANIC, _CM_REST),
    source="C-complex Ch/Cgh hydration-band <-> CM mineralogy link "
           "(standard meteoritics); Howard, K.T. et al. (2011), GCA "
           "75:2735-2751 (calcite/gypsum); Lee, M.R. et al. (2023), M&PS, "
           "doi:10.1111/maps.14099 (water)",
    note="Water/volatile-rich relative to S-type and M-type: 7.0 wt% "
         "indigenous, directly measured H2O content, plus 2 wt% organic "
         "carbon — neither S-type nor M-type has any water/hydration "
         "component at all.")


# ═══════════════════════════════════════════════════════════════════════════
# § 4. S-complex: ordinary chondrite (H/L/LL blend)
# ═══════════════════════════════════════════════════════════════════════════
# Dunn, T.L., Cressey, G., McSween, H.Y., McCoy, T.J. (2010), "Analysis of
# ordinary chondrites using powder X-ray diffraction: 1. Modal mineral
# abundances," Meteoritics & Planetary Science 45(1):123-134 (CORRECTED
# author list — see module docstring's CITATION CORRECTIONS section).
# Values below are H/L/LL-blended range midpoints (S-type reflectance alone
# does not resolve which chemical group a given body belongs to).

_s_olivine_choice = Choice(
    "S-type olivine ((Mg,Fe)2SiO4) mass fraction", 0.4315,
    checked="Dunn, Cressey, McSween & McCoy (2010), M&PS 45(1):123-134: "
            "29.1-57.2 wt% across 48 H/L/LL falls (group averages: H 33.0 "
            "wt%, LL 51.1 wt%)",
    reason="range midpoint; a blended H/L/LL representative value since "
           "S-type reflectance alone does not resolve chemical group",
    adjustable=True)
_S_OLIVINE = MineralComponent(
    component_key="olivine",
    mineral_name="Olivine ((Mg,Fe)2SiO4)",
    mass_fraction=_s_olivine_choice.value,
    provenance="choice",
    source="Dunn, T.L., Cressey, G., McSween, H.Y., McCoy, T.J. (2010), "
           "M&PS 45(1):123-134",
    gap_note="(Mg,Fe)2SiO4 nesosilicate not represented; materials.json's "
             "Silicon Dioxide (SiO2) is a compositionally distinct pure "
             "oxide (no Mg/Fe, different silicate structure), not "
             "substituted here",
    choice=_s_olivine_choice)

_s_pyroxene_choice = Choice(
    "S-type low-Ca pyroxene (bronzite-hypersthene) mass fraction", 0.229,
    checked="Dunn, Cressey, McSween & McCoy (2010), M&PS 45(1):123-134: "
            "17.6-28.2 wt% (H avg 25.6%, LL avg 21.2%)",
    reason="range midpoint; H/L/LL-blended representative value",
    adjustable=True)
_S_LOW_CA_PYROXENE = MineralComponent(
    component_key="low_ca_pyroxene",
    mineral_name="Low-Ca pyroxene (orthopyroxene: bronzite-hypersthene, "
                 "(Mg,Fe)SiO3)",
    mass_fraction=_s_pyroxene_choice.value,
    provenance="choice",
    source="Dunn, T.L., Cressey, G., McSween, H.Y., McCoy, T.J. (2010), "
           "M&PS 45(1):123-134",
    gap_note="(Mg,Fe)SiO3 inosilicate not represented; same SiO2-is-not-a-"
             "substitute caveat as olivine",
    choice=_s_pyroxene_choice)

_s_plagioclase_choice = Choice(
    "S-type plagioclase feldspar mass fraction", 0.093,
    checked="Dunn, Cressey, McSween & McCoy (2010), M&PS 45(1):123-134 "
            "reports plagioclase as comparatively stable across H/L/LL "
            "(tracks bulk Al2O3, not FeO/metal partitioning); the exact "
            "wt% figure (9.3% used here) could not be independently "
            "verified against the paywalled primary table",
    reason="treated as an internal estimate consistent with equilibrated "
           "ordinary-chondrite plagioclase modes (well-documented "
           "cosmochemistry range ~8-12 wt%), not asserted as a firmly-"
           "cited precise number",
    adjustable=True)
_S_PLAGIOCLASE = MineralComponent(
    component_key="plagioclase",
    mineral_name="Plagioclase feldspar (oligoclase-andesine, "
                 "(Na,Ca)AlSi3O8)",
    mass_fraction=_s_plagioclase_choice.value,
    provenance="choice",
    source="Dunn, T.L., Cressey, G., McSween, H.Y., McCoy, T.J. (2010), "
           "M&PS 45(1):123-134 (precise figure unverified — see "
           "Choice.checked)",
    gap_note="(Na,Ca)AlSi3O8 framework aluminosilicate not represented in "
             "either table",
    choice=_s_plagioclase_choice)

_s_metal_choice = Choice(
    "S-type metallic Fe-Ni (kamacite+taenite) bulk mass fraction", 0.105,
    checked="Jarosewich, E. (1990), Meteoritics 25:323-337: group means "
            "(Fe metal + Ni) approx. H 17.8%, L 8.3%, LL 3.6%; individual-"
            "analysis scatter across the paper's dataset spans "
            "approximately 2-19%",
    reason="range midpoint across the individual-analysis scatter; a "
           "blended H/L/LL representative value",
    adjustable=True)
_S_METAL = MineralComponent(
    component_key="metallic_fe_ni",
    mineral_name="Metallic Fe-Ni (kamacite + taenite alloy)",
    mass_fraction=_s_metal_choice.value,
    provenance="choice",
    source="Jarosewich, E. (1990), Meteoritics 25:323-337",
    material_key_surface="iron",
    material_key_inventory="Iron",
    note="'iron' is a dominant-mass proxy for the Fe-Ni alloy phase — Ni "
         "is a genuine several-percent minor constituent of ordinary-"
         "chondrite metal, but no OC-specific metal-phase Ni fraction was "
         "in the verified dataset (unlike M-type below, where cited "
         "chemical-group Ni data exists), so it is not separately split "
         "out here",
    choice=_s_metal_choice)

_s_minor_choice = Choice(
    "S-type minor/accessory phases (chromite + phosphates) mass fraction",
    0.01,
    checked="raw research cited Dunn et al. (2010) for chromite+phosphate "
            "abundance, but that paper's own six-phase XRD scheme "
            "(olivine, low-Ca pyroxene, high-Ca pyroxene, plagioclase, "
            "troilite, metal) does not include chromite or phosphates as "
            "resolved phases — no primary citation for this figure was "
            "confirmed",
    reason="relabeled as an internal engineering estimate (general "
           "ordinary-chondrite petrology consensus that chromite + "
           "phosphates are trace, well under 1-2 wt% combined — e.g. "
           "McSween, H.Y. (1999), Meteorites and Their Parent Planets, "
           "2nd ed., Cambridge Univ. Press, general synthesis, no page-"
           "specific figure), NOT presented as a directly cited number",
    adjustable=True)
_S_MINOR = MineralComponent(
    component_key="minor_accessory",
    mineral_name="Chromite (FeCr2O4) + phosphates (merrillite/"
                 "chlorapatite)",
    mass_fraction=_s_minor_choice.value,
    provenance="choice",
    source="general ordinary-chondrite petrology consensus (McSween "
           "1999) — see Choice.checked for the citation-integrity "
           "correction applied",
    gap_note="neither chromite nor merrillite/chlorapatite is represented "
             "in either table",
    choice=_s_minor_choice)

_s_rest_fraction = 1.0 - (_S_OLIVINE.mass_fraction
                         + _S_LOW_CA_PYROXENE.mass_fraction
                         + _S_PLAGIOCLASE.mass_fraction
                         + _S_METAL.mass_fraction
                         + _S_MINOR.mass_fraction)
_s_rest_choice = Choice(
    "S-type unresolved matrix/residual mass fraction", _s_rest_fraction,
    checked="Dunn et al. (2010) quantifies six phases total (olivine, "
            "low-Ca pyroxene, HIGH-Ca pyroxene, plagioclase, TROILITE, "
            "metal); this module has verified figures for four of the "
            "six (olivine, low-Ca pyroxene, plagioclase, metal) plus a "
            "trace-phase estimate",
    reason="closes the mass balance to 1.0; absorbs high-Ca pyroxene and "
           "troilite (FeS), both real and quantified in the source paper, "
           "but no specific wt% figure for either was in the "
           "adversarially-verified dataset this module was built from",
    adjustable=True)
_S_REST = MineralComponent(
    component_key="rest",
    mineral_name="Unresolved OC matrix (high-Ca pyroxene + troilite (FeS) "
                 "+ other accessory phases from Dunn et al. (2010)'s "
                 "six-phase scheme, un-quantified in the verified dataset)",
    mass_fraction=_s_rest_choice.value,
    provenance="choice",
    source="derived residual — see Choice.checked",
    gap_note="high-Ca pyroxene and troilite (FeS) are real, quantified by "
             "Dunn et al. (2010), but no wt% figure for either was "
             "verified for this module; not invented",
    choice=_s_rest_choice)

S_ASSEMBLAGE = OreAssemblage(
    assemblage_key="S",
    taxonomy_system=_BUS_DEMEO,
    meteorite_analog="Ordinary chondrite (H/L/LL chemical groups, blended)",
    components=(_S_OLIVINE, _S_LOW_CA_PYROXENE, _S_PLAGIOCLASE, _S_METAL,
                _S_MINOR, _S_REST),
    source="Chapman, C.R. (1996), M&PS 31:699-725; direct sample "
           "confirmation for the LL subgroup: Nakamura, T. et al. (2011), "
           "Science 333:1113-1116 (Itokawa). Modal mineralogy: Dunn, T.L., "
           "Cressey, G., McSween, H.Y., McCoy, T.J. (2010), M&PS "
           "45(1):123-134.",
    note="Silicate-dominated with a genuine, but minority, metal fraction "
         "(~10.5% here) — intermediate between CI/CM's ~0% free metal and "
         "M-type's ~90%+ metal; no water/hydration component at all, "
         "unlike CI/CM.")


# ═══════════════════════════════════════════════════════════════════════════
# § 5. Tholen M-type / Bus-DeMeo Xc,Xk: iron meteorite
# ═══════════════════════════════════════════════════════════════════════════
# Fe-Ni bulk fraction and Co re-attributed to Buchwald, V.F. (1975),
# Handbook of Iron Meteorites (the raw research's Goldstein/Scott/Chabot
# 2009 citation is real and topical but does not itself state these
# figures — it defers to Buchwald 1975). Ni-content-of-the-metal-phase uses
# the IIIAB chemical group (Yang & Goldstein 2006) as a representative
# pick, because IIIAB is documented as the largest combined magmatic group
# (Scott, E.R.D. & Wasson, J.T. (1975), Rev. Geophys. 13(4):527-546); real
# bodies vary by chemical group (e.g. IIAB ~5.3-6.6% Ni per Wasson &
# Kallemeyn 2002, GCA 66(13):2445-2473; IVB ~16-18%+ Ni per Rasmussen et
# al. 1984, GCA 48:805-813) — not distinguishable from broadband
# spectroscopy/albedo alone.

_m_fe_ni_bulk_choice = Choice(
    "M-type bulk Fe-Ni metal mass fraction", 0.925,
    checked="Buchwald, V.F. (1975), Handbook of Iron Meteorites — bulk "
            "metal content of iron meteorites, ~90-95%",
    reason="range midpoint",
    adjustable=True)
_m_ni_in_metal_choice = Choice(
    "Ni content within the bulk Fe-Ni metal phase (representative "
    "chemical group)", 0.0785,
    checked="Yang, J. & Goldstein, J.I. (2006), GCA 70(12):3197-3215, "
            "Table 3: IIIAB group Ni content 7.1-8.6 wt% (within the "
            "metal phase); IIIAB picked as representative because it is "
            "documented as the largest combined magmatic iron group "
            "(Scott, E.R.D. & Wasson, J.T., 1975, Rev. Geophys. "
            "13(4):527-546)",
    reason="range midpoint of the representative (IIIAB) group; other "
           "groups genuinely differ (IIAB ~5.3-6.6%, IVB ~16-18%+) and "
           "are not distinguishable from broadband spectroscopy alone",
    adjustable=True)
_m_fe_fraction = (_m_fe_ni_bulk_choice.value
                 * (1.0 - _m_ni_in_metal_choice.value))
_m_ni_fraction = _m_fe_ni_bulk_choice.value * _m_ni_in_metal_choice.value

_M_FE = MineralComponent(
    component_key="fe_metal",
    mineral_name="Iron content of the bulk Fe-Ni metal (kamacite/taenite)",
    mass_fraction=_m_fe_fraction,
    provenance="choice",
    source="Buchwald, V.F. (1975), Handbook of Iron Meteorites (bulk "
           "metal fraction); Yang & Goldstein (2006), GCA 70(12):3197-"
           "3215 (Ni-in-metal split, IIIAB representative)",
    material_key_surface="iron",
    material_key_inventory="Iron",
    note="bulk Fe-Ni metal fraction split into elemental Fe/Ni components "
         "using the representative (IIIAB) Ni-in-metal content — see "
         "module section docstring",
    choice=_m_fe_ni_bulk_choice)
_M_NI = MineralComponent(
    component_key="ni_metal",
    mineral_name="Nickel content of the bulk Fe-Ni metal (kamacite/"
                 "taenite)",
    mass_fraction=_m_ni_fraction,
    provenance="choice",
    source="Buchwald, V.F. (1975), Handbook of Iron Meteorites (bulk "
           "metal fraction); Yang & Goldstein (2006), GCA 70(12):3197-"
           "3215 (Ni-in-metal split, IIIAB representative)",
    material_key_surface="nickel",
    material_key_inventory="Nickel",
    choice=_m_ni_in_metal_choice)

_m_co_choice = Choice(
    "M-type bulk cobalt mass fraction", 0.005,
    checked="representative iron-meteorite Co content ~0.3-0.8 wt% (e.g. "
            "Sikhote-Alin 0.42 wt%, El Ali 0.48 wt% Co); best-traceable "
            "source is Buchwald, V.F. (1975), Handbook of Iron Meteorites "
            "+ original classification/INAA papers — NOT the "
            "'Meteoritical Bulletin database' as such (MBD entries do not "
            "typically tabulate bulk Fe/Ni/Co/P/S breakdowns; a citation-"
            "label correction applied here)",
    reason="range midpoint",
    adjustable=True)
_M_CO = MineralComponent(
    component_key="cobalt",
    mineral_name="Cobalt (siderophile, alloyed with the Fe-Ni metal)",
    mass_fraction=_m_co_choice.value,
    provenance="choice",
    source="Buchwald, V.F. (1975), Handbook of Iron Meteorites",
    material_key_surface=None,
    material_key_inventory="Cobalt",
    note="not present in surface.py MATERIALS (no cohesive_energy_ev "
         "entry), resolves via materials.json only",
    choice=_m_co_choice)

_m_p_choice = Choice(
    "M-type bulk phosphorus (schreibersite) mass fraction", 0.003,
    checked="Buchwald, V.F. (1975), Handbook of Iron Meteorites; "
            "Goldstein, Scott & Chabot (2009), Chemie der Erde 69:293-325 "
            "(general context): ~0.1-0.5 wt% P as accessory schreibersite",
    reason="range midpoint",
    adjustable=True)
_M_P = MineralComponent(
    component_key="phosphorus_schreibersite",
    mineral_name="Schreibersite ((Fe,Ni)3P)",
    mass_fraction=_m_p_choice.value,
    provenance="choice",
    source="Buchwald, V.F. (1975), Handbook of Iron Meteorites",
    gap_note="(Fe,Ni)3P metal phosphide not represented; materials.json's "
             "elemental Phosphorus (P4, molecular white-phosphorus "
             "allotrope) is a chemically distinct bonding regime, not "
             "substituted here",
    choice=_m_p_choice)

_m_s_choice = Choice(
    "M-type bulk sulfur (troilite) mass fraction", 0.0055,
    checked="Buchwald, V.F. (1975), Handbook of Iron Meteorites (bulk "
            "accessory troilite S, ~0.1-1 wt%); Goldstein, Scott & Chabot "
            "(2009)'s own S figures are for the UNDIFFERENTIATED PARENTAL "
            "core (pre-crystallization, much higher and group-dependent) "
            "— a different physical quantity, not conflated here",
    reason="range midpoint",
    adjustable=True)
_M_S = MineralComponent(
    component_key="sulfur_troilite",
    mineral_name="Troilite (FeS)",
    mass_fraction=_m_s_choice.value,
    provenance="choice",
    source="Buchwald, V.F. (1975), Handbook of Iron Meteorites",
    gap_note="FeS troilite not represented; materials.json's elemental "
             "Sulfur (S8) and Iron(II) Sulfate (FeSO4) are both "
             "chemically distinct (different bonding / oxidation state), "
             "not substituted here",
    choice=_m_s_choice)

_m_rest_fraction = 1.0 - (_M_FE.mass_fraction + _M_NI.mass_fraction
                         + _M_CO.mass_fraction + _M_P.mass_fraction
                         + _M_S.mass_fraction)
_m_rest_choice = Choice(
    "M-type unresolved matrix/residual mass fraction", _m_rest_fraction,
    checked="Fe+Ni+Co+P+S accounts for ~93.85% at the chosen "
            "representative values",
    reason="closes the mass balance to 1.0; ALSO carries the mission-"
           "confirmed caveat that this near-pure-metal figure is the "
           "classic Tholen 'iron meteorite analog,' not a universal "
           "M-type truth — Elkins-Tanton, L.T. et al. (2020), JGR "
           "Planets 125, e2019JE006296, predicts (16) Psyche specifically "
           "is only ~30-60 vol% metal (silicate-mixed), and that M-type "
           "bulk composition is inferred by spectral/radar/density "
           "analogy, not yet directly measured for a whole asteroid",
    adjustable=True)
_M_REST = MineralComponent(
    component_key="rest",
    mineral_name="Unresolved M-type matrix (silicate/oxide inclusions, "
                 "chromite, minor phases not itemized above)",
    mass_fraction=_m_rest_choice.value,
    provenance="choice",
    source="derived residual — see Choice.checked",
    gap_note="not itemized further; see Choice.reason for the Psyche "
             "partial-metal caveat this bucket also flags",
    choice=_m_rest_choice)

M_ASSEMBLAGE = OreAssemblage(
    assemblage_key="M",
    taxonomy_system=_THOLEN_TO_X,
    meteorite_analog="Iron meteorite (Fe-Ni metal, magmatic core-type)",
    components=(_M_FE, _M_NI, _M_CO, _M_P, _M_S, _M_REST),
    source="Fornasier, S., Clark, B.E., Dotto, E., Migliorini, A., "
           "Ockert-Bell, M., Barucci, M.A. (2010), Icarus 210:655-673 "
           "('iron meteorites, pallasites, and enstatite chondrites are "
           "the best matches... not all objects... have a pure metallic "
           "composition'); bulk metal/Co/P/S: Buchwald, V.F. (1975), "
           "Handbook of Iron Meteorites; Ni-in-metal: Yang & Goldstein "
           "(2006), GCA 70(12):3197-3215.",
    note="Iron-dominated relative to both C-type and S-type: combined "
         "Fe+Ni bulk metal fraction ~0.925, vs. S-type's ~0.105 and CI's "
         "0.0. Caveat: this is the classic 'iron meteorite' analog; real "
         "M-type bodies (e.g. Psyche) may be substantially less metal-"
         "rich per direct mission science — see the rest component.")


# ═══════════════════════════════════════════════════════════════════════════
# § 6. Spectral-class -> assemblage resolution
# ═══════════════════════════════════════════════════════════════════════════

ASSEMBLAGES: dict[str, OreAssemblage] = {
    "CI": CI_ASSEMBLAGE,
    "CM": CM_ASSEMBLAGE,
    "S": S_ASSEMBLAGE,
    "M": M_ASSEMBLAGE,
}

# Bus-DeMeo/Tholen spectral-class code (case-insensitive) -> (assemblage
# key, caveats). C-complex subtypes split CI (B, Cb, Cg — the least-altered
# subtypes, matching Ryugu/Bennu's direct sample-return CI match) from CM
# (Cgh, Ch — the hydrated subtypes, per the standard Ch/Cgh hydration-band
# <-> CM mineralogy link).
_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "c": ("CI", ()),
    "ci": ("CI", ()),
    "b": ("CI", ()),
    "cb": ("CI", ()),
    "cg": ("CI", ()),
    "cgh": ("CM", ()),
    "ch": ("CM", ()),
    "cm": ("CM", ()),
    "s": ("S", ()),
    "m": ("M", (
        "Tholen M-type is a spectra+albedo class (0.1<p_v<0.3); Bus-DeMeo "
        "spectra-only classification cannot itself confirm metallicity — "
        "see the M-type assemblage's taxonomy_system field.",
    )),
    "x": ("M", (
        "input was the BROAD Bus-DeMeo X-complex, not a confirmed "
        "metallic (Tholen M / Xc,Xk) subclass — X also contains E-type "
        "(enstatite, non-metallic) and P-type (primitive, non-metallic) "
        "bodies, spectrally indistinguishable from M without an "
        "independent albedo measurement (0.1<p_v<0.3 identifies M). "
        "Defaulting to the iron-meteorite/M-type assemblage; confirm "
        "albedo before trusting this for a specific named body.",
    )),
    "xc": ("M", ()),
    "xk": ("M", ()),
}


def _resolve_spectral_class(spectral_class: str) -> tuple[str, tuple[str, ...]]:
    key = str(spectral_class).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown spectral class {spectral_class!r}. Recognized "
            f"(case-insensitive): {sorted(_ALIASES)}")
    return _ALIASES[key]


# ═══════════════════════════════════════════════════════════════════════════
# § 7. Public API
# ═══════════════════════════════════════════════════════════════════════════

def ore_grade_estimate(spectral_class: str) -> dict:
    """The mineral assemblage for an asteroid of the given spectral class.

    Parameters
    ----------
    spectral_class : a Bus-DeMeo or Tholen taxonomy code, case-insensitive
        (e.g. "C", "B", "Cb", "Cg", "Cgh", "Ch", "CI", "CM", "S", "M", "X",
        "Xc", "Xk"). See _ALIASES for the full recognized set.

    Returns
    -------
    dict with the assemblage's taxonomy_system, meteorite_analog, the full
    list of mineral components (each with mass_fraction, provenance,
    resolved material key(s) or a disclosed gap_note, and Choice
    provenance where applicable), mass_fraction_total (validated ~1.0),
    and any caveats attached to the input class (e.g. the X-complex
    metallicity-ambiguity warning).
    """
    assemblage_key, caveats = _resolve_spectral_class(spectral_class)
    assemblage = ASSEMBLAGES[assemblage_key]
    result = assemblage.to_dict()
    result["input_spectral_class"] = spectral_class
    result["caveats"] = list(caveats)
    return result


def extraction_energy_per_kg(material_key: str) -> dict:
    """Deduced extraction/refining energy estimate for a material, from
    this codebase's own cohesive-energy data (surface.py MATERIALS).

    FIRST_PRINCIPLES-adjacent DEDUCTION, not a new empirical formula:
    breaking a material out of its matrix and reducing it to the pure
    element requires overcoming SOME binding energy; the cohesive energy
    per atom (energy to remove one atom from bulk to infinity, MEASURED —
    see surface.py) converted to J/kg is used as a lower-bound proxy for
    that binding energy. This module makes no claim that cohesive energy
    IS the extraction energy — see the module docstring's SIMPLIFIED_MODEL
    section §2 for the honest gap between this floor and real industrial
    refining energy (roughly 2x for iron, >20x for titanium, strongly
    process-dependent), and why the physically correct quantity is the ORE
    COMPOUND's Gibbs free energy of formation, not the pure metal's
    cohesive energy.

    Also reuses adhesion.py's work_of_adhesion (self-adhesion / "work of
    cohesion," W_AA = 2*gamma_A) as a supplementary diagnostic — same
    underlying cohesive-energy database, different (interfacial, J/m^2)
    quantity, not summed into the J/kg figure.

    Parameters
    ----------
    material_key : a key into field/interface/surface.py's MATERIALS dict
        (e.g. "iron", "nickel", "water_ice"). This function's basis
        (cohesive_energy_ev) is only carried by surface.py, not by
        inventory/data/materials.json.

    Returns
    -------
    dict with extraction_energy_j_per_kg / _mj_per_kg, the underlying
    cohesive energy, the supplementary self-adhesion figure, and an
    explicit simplified_model_caveat string.
    """
    if material_key not in SURFACE_MATERIALS:
        raise KeyError(
            f"extraction_energy_per_kg: {material_key!r} is not a "
            f"surface.py MATERIALS key (this function's basis is "
            f"cohesive_energy_ev, which only surface.py carries — "
            f"inventory/data/materials.json entries are not usable here). "
            f"Available: {sorted(SURFACE_MATERIALS)}")
    mat = SURFACE_MATERIALS[material_key]
    e_coh_ev = mat["cohesive_energy_ev"]
    atoms_per_kg = 1.0 / (mat["A"] * AMU_KG)
    e_coh_j_per_kg = e_coh_ev * EV_TO_J * atoms_per_kg
    self_adhesion_j_m2 = work_of_adhesion(material_key, material_key)
    return {
        "material_key": material_key,
        "material_name": mat["name"],
        "cohesive_energy_ev_per_atom": e_coh_ev,
        "extraction_energy_j_per_kg": e_coh_j_per_kg,
        "extraction_energy_mj_per_kg": e_coh_j_per_kg / 1.0e6,
        "self_adhesion_work_of_cohesion_j_m2": self_adhesion_j_m2,
        "basis": (
            "FIRST_PRINCIPLES-adjacent proxy: cohesive energy (measured "
            "energy to remove one atom from bulk to infinity) converted "
            "to J/kg via atoms/kg = 1/(A * AMU_KG). Reasoning: extracting "
            "a material from its matrix requires overcoming SOME binding "
            "energy, and cohesive energy is a real, measured lower-bound "
            "estimate of that binding."),
        "simplified_model_caveat": (
            "NOT the thermodynamically correct extraction-energy floor "
            "(that is the ORE COMPOUND's Gibbs free energy of formation, "
            "not the pure metal's cohesive energy — Ellingham 1944; "
            "Rankin 2011). Independently re-derived check: Al cohesive "
            "energy implies ~12.1 MJ/kg, vs. the actual Al2O3 formation-"
            "energy floor of ~31 MJ/kg, vs. real Hall-Heroult electrolytic "
            "reduction at ~47-54 MJ/kg — this function's figure "
            "underestimates real extraction energy, typically by a factor "
            "of roughly 2x (iron) to >20x (titanium), and does NOT "
            "capture real non-ideal process losses (Gutowski et al. 2013: "
            "even the best real processes run ~60% efficient against "
            "their thermodynamic floor) or comminution/beneficiation "
            "energy, which can dominate the total budget for low-grade "
            "ore (Valero & Valero exergoecology work). Treat this as an "
            "order-of-magnitude floor, not an engineering prediction."),
    }


def total_recoverable_mass(body_mass_kg: float, spectral_class: str,
                          material_key: str) -> dict:
    """How much of a named material a body of the given mass and spectral
    class actually contains, plus a full, exactly-conserved mass breakdown
    across every component in that class's assemblage (resolved and gap
    alike — nothing is dropped from the mass balance just because this
    codebase lacks physics data on it).

    Parameters
    ----------
    body_mass_kg : total body mass, kg (> 0)
    spectral_class : as in ore_grade_estimate()
    material_key : a component_key, material_key_surface, or
        material_key_inventory string from that class's assemblage
        (case-insensitive) — e.g. "phyllosilicate", "iron", "Water".

    Returns
    -------
    dict with the matched component's mass_fraction and recoverable_mass_kg,
    its resolution/gap status, and a "mass_closure" sub-dict listing every
    component's mass_kg plus total_accounted_kg / residual_kg (residual is
    ~0 by construction, since assemblage mass fractions are validated to
    sum to 1.0).
    """
    if body_mass_kg <= 0:
        raise ValueError("body_mass_kg must be > 0")
    grade = ore_grade_estimate(spectral_class)
    components = grade["components"]

    key_l = str(material_key).strip().lower()
    target = None
    for c in components:
        candidates = {c["component_key"].lower()}
        if c["material_key_surface"]:
            candidates.add(c["material_key_surface"].lower())
        if c["material_key_inventory"]:
            candidates.add(c["material_key_inventory"].lower())
        if key_l in candidates:
            target = c
            break
    if target is None:
        available = sorted(c["component_key"] for c in components)
        raise ValueError(
            f"{material_key!r} not found in the {grade['assemblage_key']} "
            f"assemblage (spectral_class={spectral_class!r}). Available "
            f"component keys: {available}")

    breakdown = []
    total_accounted = 0.0
    for c in components:
        m = body_mass_kg * c["mass_fraction"]
        total_accounted += m
        breakdown.append({
            "component_key": c["component_key"],
            "mineral_name": c["mineral_name"],
            "mass_fraction": c["mass_fraction"],
            "mass_kg": m,
            "resolved": c["resolved"],
        })

    return {
        "body_mass_kg": body_mass_kg,
        "input_spectral_class": spectral_class,
        "assemblage_key": grade["assemblage_key"],
        "material_key": material_key,
        "component_key": target["component_key"],
        "mineral_name": target["mineral_name"],
        "mass_fraction": target["mass_fraction"],
        "recoverable_mass_kg": body_mass_kg * target["mass_fraction"],
        "resolved": target["resolved"],
        "material_key_surface": target["material_key_surface"],
        "material_key_inventory": target["material_key_inventory"],
        "gap_note": target["gap_note"],
        "mass_closure": {
            "components": breakdown,
            "total_accounted_kg": total_accounted,
            "residual_kg": body_mass_kg - total_accounted,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# § 8. "If One, Then All" — validate every declared material key at import
# ═══════════════════════════════════════════════════════════════════════════

def _validate_material_resolution() -> None:
    inventory_names = {n.lower() for n in MaterialDB.get().names()}
    for assemblage in ASSEMBLAGES.values():
        for c in assemblage.components:
            if c.material_key_surface is not None:
                assert c.material_key_surface in SURFACE_MATERIALS, (
                    f"{assemblage.assemblage_key}/{c.component_key}: "
                    f"material_key_surface {c.material_key_surface!r} not "
                    f"in surface.py MATERIALS")
            if c.material_key_inventory is not None:
                assert c.material_key_inventory.lower() in inventory_names, (
                    f"{assemblage.assemblage_key}/{c.component_key}: "
                    f"material_key_inventory "
                    f"{c.material_key_inventory!r} not in "
                    f"inventory/data/materials.json")


_validate_material_resolution()


__all__ = [
    "MineralComponent",
    "OreAssemblage",
    "CI_ASSEMBLAGE",
    "CM_ASSEMBLAGE",
    "S_ASSEMBLAGE",
    "M_ASSEMBLAGE",
    "ASSEMBLAGES",
    "ore_grade_estimate",
    "extraction_energy_per_kg",
    "total_recoverable_mass",
]
