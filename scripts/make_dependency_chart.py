#!/usr/bin/env python3
"""
Generate the sigma-ground physics dependency chart.

Output: docs/dependency_chart.html  (self-contained, no external deps)

Usage:
    cd /path/to/sigma-ground
    python scripts/make_dependency_chart.py
"""

import json
from pathlib import Path

# ── Node Types ─────────────────────────────────────────────────────────
# measured  = directly measured physical constant (NIST/PDG)
# derived   = computed by formula from measured constants
# ssbm      = SSBM-specific parameter (ξ, η, σ_conv)
# higgs     = Higgs-origin mass — σ-INVARIANT (doesn't change with σ)
# qcd       = QCD-origin quantity — σ-DEPENDENT (scales with Λ_eff)
# field     = σ-field value at a location
# api       = exposed physics function

# ── Edge Types ─────────────────────────────────────────────────────────
# formula   = target derived by formula from source
# ssbm      = SSBM-specific derivation (dark energy matching etc.)
# qcd       = QCD sets/governs target
# sigma     = σ-field coupling modifies target
# uses      = API function uses this constant

LAYER_NAMES = [
    "Cosmological",
    "Fundamental Measured Constants",
    "SSBM Field Parameters",
    "Derived Fundamentals",
    "QCD / σ-Field",
    "Quark & Nucleon",
    "Nuclear & Electromagnetic",
    "Astronomical",
    "Physics API — Core Field",
    "Physics API — Interface Layer",
]

# (id, display_label, layer, node_type, tooltip_text)
NODES = [

    # ── Layer 0: Cosmological ──────────────────────────────────────────
    ("H0", "H₀", 0, "measured",
     "Hubble constant\n= 67.4 km/s/Mpc\nPlanck 2018 CMB power spectrum\nSets the cosmological energy scale"),

    # ── Layer 1: Fundamental Measured ─────────────────────────────────
    ("G",        "G",   1, "measured",
     "Gravitational constant\n= 6.67430×10⁻¹¹ m³ kg⁻¹ s⁻²\nCavendish torsion balance"),
    ("C",        "c",   1, "measured",
     "Speed of light\n= 2.99792458×10⁸ m/s  [exact by SI 2019]\nDefines the metre"),
    ("HBAR",     "ℏ",   1, "measured",
     "Reduced Planck constant\n= 1.054571817×10⁻³⁴ J·s  [exact by SI 2019]"),
    ("E_CHARGE", "e",   1, "measured",
     "Elementary charge\n= 1.602176634×10⁻¹⁹ C  [exact by SI 2019]"),
    ("EPS_0",    "ε₀",  1, "measured",
     "Vacuum permittivity\n= 8.854187817×10⁻¹² F/m  (CODATA 2018)"),
    ("MU_0",     "μ₀",  1, "measured",
     "Vacuum permeability\n= 1.25664×10⁻⁶ H/m\nDerived from SI exact values"),
    ("K_B",      "k_B", 1, "measured",
     "Boltzmann constant\n= 1.380649×10⁻²³ J/K  [exact by SI 2019]\nLinks energy to temperature"),

    # ── Layer 2: SSBM Parameters ───────────────────────────────────────
    ("XI", "ξ", 2, "ssbm",
     "Baryon fraction Ω_b/(Ω_b+Ω_c) = 0.1582\n"
     "Planck 2018.  THE single new SSBM free parameter.\n"
     "Ω_b h² = 0.02237,  Ω_c h² = 0.1200\n"
     "Drives the entire scale transition framework."),
    ("SIGMA_CONV", "σ_conv", 2, "ssbm",
     "Critical σ for nuclear bond failure\n"
     "= −ln(ξ) ≈ 1.849\n"
     "Formula derivation: σ_conv = −ln(ξ)\n"
     "The σ at which QCD binding energy overwhelms nuclear structure."),
    ("ETA", "η", 2, "ssbm",
     "Cosmic entanglement fraction = 0.4153\n"
     "DERIVED from dark energy constraint:\n"
     "  η × ρ_released = ρ_DE(observed)\n"
     "Fraction of particles with cross-hadron quantum entanglement."),

    # ── Layer 3: Derived Fundamentals ─────────────────────────────────
    ("L_PLANCK",    "l_P",     3, "derived",
     "Planck length = √(ℏG/c³) ≈ 1.616×10⁻³⁵ m\n"
     "Smallest length with physical meaning.\n"
     "Used as spatial 'effectively zero' floor."),
    ("M_PLANCK_KG", "m_P",     3, "derived",
     "Planck mass = √(ℏc/G) ≈ 2.18×10⁻⁸ kg"),
    ("ALPHA",       "α",       3, "derived",
     "Fine structure constant\n"
     "= e²/(4πε₀ℏc) ≈ 1/137.036\n"
     "Strength of electromagnetic interaction.\n"
     "σ-INVARIANT: e is Higgs-origin, ε₀ and ℏ are fundamental."),
    ("M_ELECTRON_KG","mₑ (kg)", 3, "higgs",
     "Electron mass = 9.1094×10⁻³¹ kg  (CODATA 2018)\n"
     "Higgs origin — σ-INVARIANT\n"
     "Does not change with σ-field compression."),
    ("SIGMA_FLOOR", "σ_floor", 3, "derived",
     "σ computational floor\n"
     "= l_P × H₀/c ≈ 1.18×10⁻⁶¹\n"
     "Planck/Hubble ratio.\n"
     "Smallest σ the classical field description can sustain."),
    ("M_HUBBLE_KG", "M_H",     3, "derived",
     "Hubble mass = c³/(2GH₀) ≈ 9.3×10⁵² kg\n"
     "Total mass within the Hubble radius."),

    # ── Layer 4: QCD / σ-Field ─────────────────────────────────────────
    ("LAMBDA_QCD_MEV", "Λ_QCD", 4, "qcd",
     "QCD confinement scale = 217 MeV  (PDG)\n"
     "Sets the strong force energy scale.\n"
     "σ-DEPENDENT: Λ_eff(σ) = Λ_QCD × e^σ\n"
     "~99% of nucleon mass originates here."),
    ("SIGMA_FIELD", "σ(r)", 4, "field",
     "σ-field value at location r\n"
     "= ξ·GM/(rc²)  [from Newtonian potential]\n"
     "Field equation: □σ = −ξR\n"
     "σ = 0 in flat spacetime (our reference epoch)\n"
     "σ > 0 inside matter, black holes, at Big Bang"),

    # ── Layer 5: Quark & Nucleon ───────────────────────────────────────
    ("M_UP_MEV",        "m_u",     5, "higgs",
     "Up quark bare mass = 2.16 MeV  (PDG 2020)\n"
     "Higgs origin — σ-INVARIANT\n"
     "Only ~0.2% of proton mass."),
    ("M_DOWN_MEV",      "m_d",     5, "higgs",
     "Down quark bare mass = 4.67 MeV  (PDG 2020)\n"
     "Higgs origin — σ-INVARIANT"),
    ("PROTON_BARE_MEV", "p_bare",  5, "derived",
     "Proton Higgs-origin mass\n"
     "= 2m_u + m_d = 8.99 MeV\n"
     "Only ~1% of total proton mass!\n"
     "σ-INVARIANT — set entirely by Higgs."),
    ("PROTON_TOTAL_MEV","p",       5, "measured",
     "Proton total mass = 938.272 MeV  (PDG 2022)\n"
     "~99% is QCD binding energy, only ~1% Higgs.\n"
     "Direct measurement."),
    ("PROTON_QCD_MEV",  "p_QCD",   5, "qcd",
     "Proton QCD binding mass\n"
     "= p_total − p_bare = 929.28 MeV\n"
     "~99% of proton mass — sets by gluon dynamics.\n"
     "σ-DEPENDENT: scales with Λ_QCD × e^σ"),
    ("NEUTRON_TOTAL_MEV","n",      5, "measured",
     "Neutron total mass = 939.565 MeV  (PDG 2022)\n"
     "~99% QCD binding, ~1% Higgs quark masses."),
    ("M_ELECTRON_MEV",  "mₑ (MeV)",5, "higgs",
     "Electron mass = 0.51100 MeV\n"
     "Higgs origin — σ-INVARIANT\n"
     "Not a quark; lives outside the nucleus."),

    # ── Layer 6: Nuclear & EM ──────────────────────────────────────────
    ("R0_FM",       "r₀",    6, "measured",
     "Nuclear charge radius parameter = 1.215 fm\n"
     "Hofstadter electron scattering experiments.\n"
     "R(A) = r₀ × A^(1/3)"),
    ("KE_E2_MEV_FM","ke²",   6, "derived",
     "EM coupling in nuclear units\n"
     "= e²/(4πε₀) ≈ 1.43996 MeV·fm\n"
     "Bridge between SI EM and nuclear energy scales."),
    ("A_C_MEV",     "a_C",   6, "derived",
     "Coulomb coefficient (Bethe-Weizsäcker)\n"
     "= (3/5)·ke²/r₀ ≈ 0.7111 MeV\n"
     "NOT from SEMF — derived from first principles.\n"
     "σ-INVARIANT: pure electrostatics."),
    ("N0_FM3",      "n₀",    6, "measured",
     "Nuclear saturation density = 0.16 fm⁻³\n"
     "Where nuclear attraction and repulsion balance.\n"
     "Measured from electron scattering on heavy nuclei."),
    ("E_SAT_MEV",   "E_sat", 6, "measured",
     "Binding energy per nucleon at saturation = −16 MeV\n"
     "Measured from nuclear mass spectrometry."),
    ("K_SAT_MEV",   "K",     6, "measured",
     "Nuclear incompressibility = 230 MeV\n"
     "Stiffness of nuclear matter at saturation.\n"
     "Measured from giant monopole resonances.\n"
     "σ-DEPENDENT: K(σ) = K₀ × e^σ"),
    ("J_SYM_MEV",   "J",     6, "measured",
     "Symmetry energy at saturation = 32 MeV\n"
     "Energy cost of neutron-proton imbalance.\n"
     "Measured from nuclear mass differences."),

    # ── Layer 7: Astronomical ──────────────────────────────────────────
    ("M_SUN_KG","M_☉", 7, "measured",
     "Solar mass = 1.989×10³⁰ kg\n"
     "Reference mass for stellar/galactic scales."),
    ("L_SUN_W", "L_☉", 7, "measured",
     "Solar luminosity = 3.828×10²⁶ W\n"
     "IAU 2015 nominal solar luminosity."),
    ("AU_M",    "AU",  7, "measured",
     "Astronomical unit = 1.4960×10¹¹ m\n"
     "IAU 2012 exact definition."),

    # ── Layer 8: Physics API — Core Field ─────────────────────────────
    ("api_lorentz",    "lorentz_factor\n(v)",          8, "api",
     "field.relativity\nγ = 1/√(1−v²/c²)\nUses: c"),
    ("api_rest_energy","rest_energy\n(m₀)",            8, "api",
     "field.relativity\nE₀ = m₀c²\nUses: c"),
    ("api_sr_sigma",   "sigma_time\n_dilation(σ,t₀)", 8, "api",
     "field.relativity\nt = t₀ × e^σ\nUses: σ-field (scale_ratio)"),
    ("api_coulomb",    "coulomb_force\n(q₁,q₂,r)",    8, "api",
     "field.electrodynamics\nF = ke·q₁q₂/r²\nUses: e, ε₀"),
    ("api_larmor",     "larmor_power\n(q,a)",          8, "api",
     "field.electrodynamics\nP = q²a²/(6πε₀c³)\nUses: e, ε₀, c"),
    ("api_alpha_fn",   "fine_structure\n_constant()",  8, "api",
     "field.electrodynamics\nα = e²/(4πε₀ℏc)\nUses: e, ε₀, ℏ, c → verifies α"),
    ("api_schwarz",    "schwarzschild\n_radius(M)",    8, "api",
     "field.gr_basics / scale\nr_s = 2GM/c²\nUses: G, c"),
    ("api_hawking",    "hawking_temp\n(M)",            8, "api",
     "field.gr_basics\nT_H = ℏc³/(8πGMk_B)\nUses: ℏ, c, G, k_B"),
    ("api_sigma_hz",   "sigma_at\n_horizon(M)",        8, "api",
     "field.gr_basics\nAlways = ξ/2 ≈ 0.079\nMass-independent! Uses: ξ only\nKey SSBM consistency check."),
    ("api_binding",    "binding_energy\n_mev(Z,A)",    8, "api",
     "field.binding\nBethe-Weizsäcker + QCD terms\nUses: a_C, K, n₀, J"),

    # ── Layer 9: Physics API — Interfaces ─────────────────────────────
    ("api_proton_mev","proton_mass\n_mev(σ)",         9, "api",
     "field.nucleon\np(σ) = p_bare + p_QCD × e^σ\nUses: PROTON_BARE, Λ_QCD, σ\nShows σ-dependence of nucleon mass"),
    ("api_decay",     "decay_constant\n(t½)",         9, "api",
     "field.decay\nλ = ln(2)/t½\nPure math — no physical constants needed"),
    ("api_gamow",     "gamow_factor\n(Z,Q)",          9, "api",
     "field.decay\nG = π Z_d Z_α α/β_α\nUses: α (fine structure constant)\nGamow tunneling probability"),
    ("api_sigma_decay","sigma_decay\n_shift(σ,λ)",   9, "api",
     "field.decay\nλ_eff = λ₀ × e^σ\nUses: σ-field\nDecay faster in compressed spacetime"),
    ("api_plasma_f",  "plasma_freq\n(n_e)",           9, "api",
     "field.interface.plasma\nω_p = √(n_e e²/ε₀mₑ)\nUses: e, ε₀, mₑ"),
    ("api_debye",     "debye_length\n(n_e,T)",        9, "api",
     "field.interface.plasma\nλ_D = √(ε₀k_BT/n_e e²)\nUses: ε₀, k_B, e"),
    ("api_fermi",     "fermi_dirac\n(E,μ,T)",         9, "api",
     "field.interface.statistical\n1/(e^{ΔE/kT}+1)\nUses: k_B"),
    ("api_rms",       "rms_speed\n(m,T)",             9, "api",
     "field.interface.statistical\n√(3k_BT/m)\nUses: k_B"),
    ("api_stoq",      "stoq\n(structure)",            9, "api",
     "inventory.stoq\nResolve material → quarks\nUses: p_total, n_total"),
]

# (source_id, target_id, edge_type, formula_label)
EDGES = [
    # Derived fundamentals
    ("HBAR",     "L_PLANCK",     "formula", "√(ℏG/c³)"),
    ("G",        "L_PLANCK",     "formula", ""),
    ("C",        "L_PLANCK",     "formula", ""),
    ("HBAR",     "M_PLANCK_KG",  "formula", "√(ℏc/G)"),
    ("G",        "M_PLANCK_KG",  "formula", ""),
    ("C",        "M_PLANCK_KG",  "formula", ""),
    ("E_CHARGE", "ALPHA",        "formula", "e²/(4πε₀ℏc)"),
    ("EPS_0",    "ALPHA",        "formula", ""),
    ("HBAR",     "ALPHA",        "formula", ""),
    ("C",        "ALPHA",        "formula", ""),
    ("E_CHARGE", "M_ELECTRON_KG","formula", "Higgs coupling"),
    ("L_PLANCK", "SIGMA_FLOOR",  "formula", "l_P·H₀/c"),
    ("H0",       "SIGMA_FLOOR",  "formula", ""),
    ("C",        "SIGMA_FLOOR",  "formula", ""),
    ("G",        "M_HUBBLE_KG",  "formula", "c³/(2GH₀)"),
    ("C",        "M_HUBBLE_KG",  "formula", ""),
    ("H0",       "M_HUBBLE_KG",  "formula", ""),

    # SSBM
    ("XI", "SIGMA_CONV", "ssbm", "−ln(ξ)"),
    ("XI", "ETA",        "ssbm", "dark energy constraint"),

    # σ-field
    ("XI", "SIGMA_FIELD", "formula", "σ = ξGM/rc²"),
    ("G",  "SIGMA_FIELD", "formula", ""),
    ("C",  "SIGMA_FIELD", "formula", ""),

    # QCD coupling
    ("LAMBDA_QCD_MEV", "SIGMA_FIELD",     "qcd",   "Λ_eff scales with σ"),
    ("LAMBDA_QCD_MEV", "PROTON_QCD_MEV",  "qcd",   "sets ~99% of nucleon mass"),
    ("SIGMA_FIELD",    "PROTON_QCD_MEV",  "sigma", "Λ_eff = Λ_QCD·e^σ"),
    ("SIGMA_FIELD",    "K_SAT_MEV",       "sigma", "K(σ) = K₀·e^σ"),

    # Nucleon derivation
    ("M_UP_MEV",        "PROTON_BARE_MEV", "formula", "2m_u + m_d"),
    ("M_DOWN_MEV",      "PROTON_BARE_MEV", "formula", ""),
    ("PROTON_TOTAL_MEV","PROTON_QCD_MEV",  "formula", "p_total − p_bare"),
    ("PROTON_BARE_MEV", "PROTON_QCD_MEV",  "formula", ""),

    # Nuclear EM
    ("E_CHARGE",    "KE_E2_MEV_FM", "formula", "e²/(4πε₀)"),
    ("EPS_0",       "KE_E2_MEV_FM", "formula", ""),
    ("KE_E2_MEV_FM","A_C_MEV",      "formula", "(3/5)·ke²/r₀"),
    ("R0_FM",       "A_C_MEV",      "formula", ""),

    # API uses
    ("C",              "api_lorentz",    "uses", ""),
    ("C",              "api_rest_energy","uses", ""),
    ("SIGMA_FIELD",    "api_sr_sigma",   "uses", "e^σ"),
    ("E_CHARGE",       "api_coulomb",    "uses", ""),
    ("EPS_0",          "api_coulomb",    "uses", ""),
    ("E_CHARGE",       "api_larmor",     "uses", ""),
    ("EPS_0",          "api_larmor",     "uses", ""),
    ("C",              "api_larmor",     "uses", ""),
    ("ALPHA",          "api_alpha_fn",   "uses", ""),
    ("E_CHARGE",       "api_alpha_fn",   "uses", ""),
    ("EPS_0",          "api_alpha_fn",   "uses", ""),
    ("HBAR",           "api_alpha_fn",   "uses", ""),
    ("C",              "api_alpha_fn",   "uses", ""),
    ("G",              "api_schwarz",    "uses", ""),
    ("C",              "api_schwarz",    "uses", ""),
    ("HBAR",           "api_hawking",    "uses", ""),
    ("C",              "api_hawking",    "uses", ""),
    ("G",              "api_hawking",    "uses", ""),
    ("K_B",            "api_hawking",    "uses", ""),
    ("XI",             "api_sigma_hz",   "uses", "ξ/2 always"),
    ("A_C_MEV",        "api_binding",    "uses", ""),
    ("K_SAT_MEV",      "api_binding",    "uses", ""),
    ("N0_FM3",         "api_binding",    "uses", ""),
    ("J_SYM_MEV",      "api_binding",    "uses", ""),
    ("PROTON_BARE_MEV", "api_proton_mev","uses", ""),
    ("LAMBDA_QCD_MEV",  "api_proton_mev","uses", ""),
    ("SIGMA_FIELD",     "api_proton_mev","uses", "e^σ"),
    ("ALPHA",           "api_gamow",     "uses", ""),
    ("SIGMA_FIELD",     "api_sigma_decay","uses","e^σ"),
    ("E_CHARGE",        "api_plasma_f",  "uses", ""),
    ("EPS_0",           "api_plasma_f",  "uses", ""),
    ("M_ELECTRON_KG",   "api_plasma_f",  "uses", ""),
    ("EPS_0",           "api_debye",     "uses", ""),
    ("K_B",             "api_debye",     "uses", ""),
    ("E_CHARGE",        "api_debye",     "uses", ""),
    ("K_B",             "api_fermi",     "uses", ""),
    ("K_B",             "api_rms",       "uses", ""),
    ("PROTON_TOTAL_MEV","api_stoq",      "uses", ""),
    ("NEUTRON_TOTAL_MEV","api_stoq",     "uses", ""),
]


def build_html(nodes, edges, layer_names):
    nodes_data = [
        {"id": n[0], "label": n[1], "layer": n[2], "type": n[3], "tip": n[4]}
        for n in nodes
    ]
    edges_data = [
        {"src": e[0], "tgt": e[1], "type": e[2], "label": e[3]}
        for e in edges
    ]
    graph_json = json.dumps(
        {"nodes": nodes_data, "edges": edges_data, "layers": layer_names},
        ensure_ascii=False
    )
    return HTML_TEMPLATE.replace("__GRAPH_DATA__", graph_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sigma-ground: Physics Dependency Chart</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
#header { padding: 18px 28px 10px; }
#header h1 { font-size: 1.35rem; font-weight: 600; color: #a8d8ea; letter-spacing: 0.04em; }
#header p  { font-size: 0.78rem; color: #888; margin-top: 4px; }
#legend { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 6px 28px 12px; }
.leg { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #bbb; }
.leg-dot { width: 12px; height: 12px; border-radius: 3px; border: 1.5px solid; }
.leg-line { width: 22px; height: 3px; border-radius: 2px; }
#scroll { overflow-x: auto; padding: 0 16px 24px; }
svg { display: block; }
.node rect { rx: 7; ry: 7; cursor: pointer; transition: opacity 0.15s; }
.node text { pointer-events: none; font-size: 10.5px; dominant-baseline: middle; }
.node.dim rect { opacity: 0.18; }
.node.dim text { opacity: 0.18; }
.node.hi  rect { filter: drop-shadow(0 0 6px rgba(255,255,180,0.7)); }
.edge { fill: none; transition: opacity 0.15s; }
.edge.dim { opacity: 0.06; }
.edge.hi  { opacity: 1 !important; stroke-width: 2.5px !important; }
.band { opacity: 0.07; }
.band-label { font-size: 10px; fill: #aaa; font-weight: 500; letter-spacing: 0.05em; }
#tooltip {
  position: fixed; display: none; max-width: 280px;
  background: #0d1b2a; border: 1px solid #445; border-radius: 8px;
  padding: 10px 13px; font-size: 0.76rem; line-height: 1.55;
  color: #dde; pointer-events: none; z-index: 100;
  box-shadow: 0 4px 18px rgba(0,0,0,0.6);
  white-space: pre-wrap;
}
#tooltip strong { color: #a8d8ea; display: block; margin-bottom: 4px; font-size: 0.82rem; }
</style>
</head>
<body>
<div id="header">
  <h1>sigma-ground — Physics Dependency Chart</h1>
  <p>Hover a node to trace its ancestry. Hover an edge to see the formula. Click to pin.</p>
</div>
<div id="legend">
  <span class="leg"><span class="leg-dot" style="background:#D6EAF8;border-color:#2980B9"></span>Measured</span>
  <span class="leg"><span class="leg-dot" style="background:#D5F5E3;border-color:#27AE60"></span>Derived by formula</span>
  <span class="leg"><span class="leg-dot" style="background:#FDEBD0;border-color:#E67E22"></span>SSBM parameter</span>
  <span class="leg"><span class="leg-dot" style="background:#E8DAEF;border-color:#8E44AD"></span>Higgs-origin (σ-invariant)</span>
  <span class="leg"><span class="leg-dot" style="background:#FADBD8;border-color:#C0392B"></span>QCD-origin (σ-dependent)</span>
  <span class="leg"><span class="leg-dot" style="background:#FFFDE7;border-color:#F9A825"></span>σ-field</span>
  <span class="leg"><span class="leg-dot" style="background:#F0F0F0;border-color:#607D8B"></span>API call</span>
  <span class="leg"><span class="leg-line" style="background:#2980B9"></span>Formula derivation</span>
  <span class="leg"><span class="leg-line" style="background:#E67E22;background:repeating-linear-gradient(90deg,#E67E22 0,#E67E22 5px,transparent 5px,transparent 8px)"></span>SSBM</span>
  <span class="leg"><span class="leg-line" style="background:#C0392B"></span>QCD origin</span>
  <span class="leg"><span class="leg-line" style="background:#F57F17;background:repeating-linear-gradient(90deg,#F57F17 0,#F57F17 4px,transparent 4px,transparent 6px)"></span>σ-coupling</span>
  <span class="leg"><span class="leg-line" style="background:#90A4AE;background:repeating-linear-gradient(90deg,#90A4AE 0,#90A4AE 3px,transparent 3px,transparent 6px)"></span>API uses</span>
</div>
<div id="scroll"><svg id="chart"></svg></div>
<div id="tooltip"></div>

<script>
const RAW = __GRAPH_DATA__;

const NODE_STYLE = {
  measured: {fill:"#D6EAF8", stroke:"#2980B9", text:"#154360"},
  derived:  {fill:"#D5F5E3", stroke:"#27AE60", text:"#145a32"},
  ssbm:     {fill:"#FDEBD0", stroke:"#E67E22", text:"#784212"},
  higgs:    {fill:"#E8DAEF", stroke:"#8E44AD", text:"#4a235a"},
  qcd:      {fill:"#FADBD8", stroke:"#C0392B", text:"#641e16"},
  field:    {fill:"#FFFDE7", stroke:"#F9A825", text:"#7d6608"},
  api:      {fill:"#F0F0F0", stroke:"#607D8B", text:"#263238"},
};

const EDGE_STYLE = {
  formula: {color:"#2980B9", w:1.4, dash:"none"},
  ssbm:    {color:"#E67E22", w:1.4, dash:"6,3"},
  qcd:     {color:"#C0392B", w:1.8, dash:"none"},
  sigma:   {color:"#F57F17", w:1.8, dash:"4,2"},
  uses:    {color:"#78909C", w:1.0, dash:"3,3"},
};

const BAND_COLORS = [
  "#1a3a5c","#1a2a4c","#1a2a3c","#162540","#162030",
  "#1a1a30","#161a28","#161620","#2a1a30","#2a1828",
];

const W = 1640, ROW_H = 148, TOP_PAD = 52, SIDE_PAD = 90;
const NW = 118, NH = 50;

function layout(nodes) {
  const byLayer = {};
  nodes.forEach(n => { (byLayer[n.layer] = byLayer[n.layer]||[]).push(n); });
  nodes.forEach(n => {
    const peers = byLayer[n.layer];
    const idx   = peers.indexOf(n);
    const count = peers.length;
    const usable = W - 2 * SIDE_PAD;
    n._x = count === 1 ? W/2 : SIDE_PAD + idx * usable / (count - 1);
    n._y = TOP_PAD + n.layer * ROW_H;
    n._w = NW; n._h = NH;
  });
}

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function bezier(x1,y1,x2,y2) {
  const cy = (y1+y2)/2;
  return `M${x1},${y1} C${x1},${cy} ${x2},${cy} ${x2},${y2}`;
}

let pinned = null;
const tooltip = document.getElementById("tooltip");

function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  moveTip(evt);
}
function moveTip(evt) {
  const x = evt.clientX + 14, y = evt.clientY - 10;
  tooltip.style.left = Math.min(x, window.innerWidth - 290) + "px";
  tooltip.style.top  = (y < 10 ? 10 : y) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

function render() {
  const nodes = RAW.nodes;
  const edges = RAW.edges;
  const layers = RAW.layers;
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  layout(nodes);

  const totalLayers = Math.max(...nodes.map(n=>n.layer)) + 1;
  const svgH = TOP_PAD + totalLayers * ROW_H + 40;
  const svg = document.getElementById("chart");
  svg.setAttribute("width",  W);
  svg.setAttribute("height", svgH);
  svg.setAttribute("viewBox", `0 0 ${W} ${svgH}`);

  // ── Arrowhead markers ──────────────────────────────────────────────
  const defs = svgEl("defs", {}, svg);
  const markerTypes = Object.entries(EDGE_STYLE);
  markerTypes.forEach(([type, st]) => {
    const m = svgEl("marker", {
      id:`arr-${type}`, markerWidth:8, markerHeight:6,
      refX:7, refY:3, orient:"auto"
    }, defs);
    svgEl("polygon", {
      points:"0 0, 8 3, 0 6", fill:st.color, opacity:0.85
    }, m);
  });

  // ── Layer bands ────────────────────────────────────────────────────
  const bandG = svgEl("g", {}, svg);
  for (let i = 0; i < totalLayers; i++) {
    const y = i * ROW_H;
    svgEl("rect", {
      x:0, y:TOP_PAD + y - ROW_H/2 + NH/2,
      width:W, height:ROW_H,
      fill: BAND_COLORS[i] || "#111",
      class:"band"
    }, bandG);
    const label = layers[i] || `Layer ${i}`;
    svgEl("text", {
      x:6, y: TOP_PAD + y - ROW_H/2 + NH/2 + 13,
      class:"band-label"
    }, bandG).textContent = label.toUpperCase();
  }

  // ── Edges ──────────────────────────────────────────────────────────
  const edgeG = svgEl("g", {}, svg);
  const edgeEls = [];

  edges.forEach((e,i) => {
    const src = nodeMap[e.src], tgt = nodeMap[e.tgt];
    if (!src || !tgt) return;
    const st = EDGE_STYLE[e.type] || EDGE_STYLE.uses;

    const x1 = src._x, y1 = src._y + NH/2 + 1;
    const x2 = tgt._x, y2 = tgt._y - NH/2 - 1;

    const pathD = bezier(x1,y1,x2,y2);
    const path = svgEl("path", {
      d:pathD, stroke:st.color,
      "stroke-width": st.w,
      "stroke-dasharray": st.dash === "none" ? "" : st.dash,
      "marker-end": `url(#arr-${e.type})`,
      opacity: 0.65,
      class:"edge"
    }, edgeG);

    path.dataset.src  = e.src;
    path.dataset.tgt  = e.tgt;
    path.dataset.etype = e.type;
    path.dataset.label = e.label;
    path.dataset.idx   = i;
    edgeEls.push(path);

    // Fat invisible hit area
    const hit = svgEl("path", {
      d:pathD, stroke:"transparent", "stroke-width":10, fill:"none", cursor:"pointer"
    }, edgeG);
    hit.addEventListener("mousemove", evt => {
      if (pinned) return;
      const lbl = e.label ? `<strong>${e.label}</strong>${e.src} → ${e.tgt}` : `${e.src} → ${e.tgt}`;
      showTip(evt, lbl);
    });
    hit.addEventListener("mouseleave", () => { if (!pinned) hideTip(); });
  });

  // ── Nodes ──────────────────────────────────────────────────────────
  const nodeG = svgEl("g", {}, svg);
  const nodeEls = {};

  nodes.forEach(n => {
    const st = NODE_STYLE[n.type] || NODE_STYLE.api;
    const g  = svgEl("g", {class:"node", transform:`translate(${n._x - NW/2},${n._y - NH/2})`}, nodeG);
    g.dataset.id = n.id;

    svgEl("rect", {
      width:NW, height:NH, rx:7, ry:7,
      fill:st.fill, stroke:st.stroke, "stroke-width":1.5
    }, g);

    // Multi-line label
    const lines = n.label.split("\n");
    if (lines.length === 1) {
      const t = svgEl("text", {x:NW/2, y:NH/2, "text-anchor":"middle", fill:st.text}, g);
      t.textContent = n.label;
    } else {
      const t = svgEl("text", {x:NW/2, y:NH/2 - 7, "text-anchor":"middle", fill:st.text}, g);
      t.textContent = lines[0];
      const t2 = svgEl("text", {x:NW/2, y:NH/2 + 8, "text-anchor":"middle", fill:st.text, "font-size":"9.5px"}, g);
      t2.textContent = lines[1] || "";
    }

    nodeEls[n.id] = g;

    // Hover / click
    g.addEventListener("mousemove", evt => {
      if (pinned && pinned !== n.id) return;
      highlightNode(n.id, nodes, edges, edgeEls, nodeEls);
      const tipText = `<strong>${n.label.replace(/\n/g," ")}</strong>${n.tip}`;
      showTip(evt, tipText);
      evt.stopPropagation();
    });
    g.addEventListener("mouseleave", () => {
      if (!pinned) { clearHighlight(nodes, edgeEls, nodeEls); hideTip(); }
    });
    g.addEventListener("click", evt => {
      if (pinned === n.id) {
        pinned = null;
        clearHighlight(nodes, edgeEls, nodeEls);
        hideTip();
      } else {
        pinned = n.id;
        highlightNode(n.id, nodes, edges, edgeEls, nodeEls);
      }
      evt.stopPropagation();
    });
  });

  // Click elsewhere to unpin
  svg.addEventListener("click", () => {
    if (pinned) { pinned = null; clearHighlight(nodes, edgeEls, nodeEls); hideTip(); }
  });
}

function highlightNode(id, nodes, edges, edgeEls, nodeEls) {
  // Find all connected node IDs (immediate parents + children)
  const connected = new Set([id]);
  edges.forEach(e => {
    if (e.src === id) connected.add(e.tgt);
    if (e.tgt === id) connected.add(e.src);
  });

  // Dim/highlight nodes
  nodes.forEach(n => {
    const el = nodeEls[n.id];
    if (!el) return;
    el.classList.toggle("dim", !connected.has(n.id));
    el.classList.toggle("hi",  n.id === id);
  });

  // Dim/highlight edges
  edgeEls.forEach(path => {
    const s = path.dataset.src, t = path.dataset.tgt;
    const relevant = s === id || t === id;
    path.classList.toggle("dim", !relevant);
    path.classList.toggle("hi",   relevant);
    path.style.opacity = relevant ? "1" : "";
  });
}

function clearHighlight(nodes, edgeEls, nodeEls) {
  nodes.forEach(n => {
    const el = nodeEls[n.id];
    if (el) { el.classList.remove("dim","hi"); }
  });
  edgeEls.forEach(p => {
    p.classList.remove("dim","hi");
    p.style.opacity = "";
  });
}

render();
document.addEventListener("mousemove", evt => {
  if (tooltip.style.display === "block") moveTip(evt);
});
</script>
</body>
</html>
"""


def main():
    out = Path(__file__).parent.parent / "docs" / "dependency_chart.html"
    out.parent.mkdir(exist_ok=True)
    html = build_html(NODES, EDGES, LAYER_NAMES)
    out.write_text(html, encoding="utf-8")
    n_nodes = len(NODES)
    n_edges = len(EDGES)
    print(f"Written: {out}")
    print(f"  {n_nodes} nodes across {len(LAYER_NAMES)} layers")
    print(f"  {n_edges} edges")
    print(f"  File size: {len(html)//1024} KB")
    print(f"\nOpen in browser: file:///{out.resolve()}")


if __name__ == "__main__":
    main()
