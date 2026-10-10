"""Build n1_references_v2.json from n1_references.json plus the repairs. Verification in v2 was by direct fetch of the
arXiv abstract / ar5iv full text and by Crossref metadata queries (the shared web-search budget was exhausted)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
refs = json.load(open(HERE / "n1_references.json", encoding="utf8"))
by = {r["key"]: r for r in refs}

upd = {
    "Riley2019": {"status_v2": "revisited: Vinciguerra+2024 report that adding XMM-Newton constraints disfavours the 2019 main "
                               "solutions; kept only as a soft (reported, not pass/fail) anchor"},
    "Miller2019": {"status_v2": "revisited by Vinciguerra+2024 (same data set); soft anchor only"},
    "Riley2021": {"status_v2": "superseded by Salmi+2024 (3.6 yr of NICER data); kept for comparison"},
    "Saffer2025": {"cite": "Saffer A. et al. 2025, ApJL 983, L20 — A Lower Mass Estimate for PSR J0348+0432 Based on CHIME/Pulsar Precision Timing",
                   "id": "arXiv:2412.02850; doi:10.3847/2041-8213/adc25e",
                   "value": "M_p = 1.806(37) Msun; the earlier ~2.01 Msun relied on WD-companion spectroscopy the authors argue was mis-modelled",
                   "verified": "v2: arXiv abstract fetched (comments field: 'Submitted to ApJ Letters'); journal reference ApJL 983, L20 (2025) "
                               "and DOI confirmed by a Crossref metadata query",
                   "status_v2": "one 2025 timing analysis revises J0348 below 2 Msun; independent confirmation not yet available"},
    "Bond2017": {"cite": "Bond H. E. et al. 2017, ApJ 840, 70 — The Sirius System and its Astrophysical Puzzles",
                 "id": "arXiv:1703.10625; doi:10.3847/1538-4357/aa6af8",
                 "verified": "v2: arXiv abstract fetched (M_B = 1.018 +/- 0.011 Msun); volume/page from Crossref metadata"},
    "Margueron2018": {"value": "Tables 1-2 'present estimation' (from analyses of experimental data): n_sat 0.155 +/- 0.005 fm^-3, "
                               "E_sat -15.8 +/- 0.3 MeV, K_sat 230 +/- 20 MeV, E_sym 32 +/- 2 MeV, L_sym 60 +/- 15 MeV, "
                               "K_sym -100 +/- 100 MeV, Q_sat 300 +/- 400 MeV. Table 4 'Total Average' over 50 models (Skyrme, RMF, RHF "
                               "weighted equally by type): K_sat 251 (29), L_sym 76.6 (29.2) — a model spread, not data.",
                      "verified": "v2: ar5iv full text fetched; Tables 1, 2 and 4 read",
                      "status_v2": "v1 wrongly used the Table 4 model average as 'empirical parameters'"},
    "GargColo2018": {"status_v2": "no K value in the abstract; the repo's K_SAT_MEV = 230 is cited to Margueron 2018 Table 1 instead"},
    "Abbott2018": {"value": "P(2 rho_nuc) = 3.5 (+2.7/-1.7) x 10^34 dyn/cm^2 at 90 %; rho_nuc = 2.8 x 10^14 g/cm^3 (Introduction)",
                   "verified": "v2: arXiv abstract and ar5iv full text fetched (rho_nuc definition read from the Introduction)"},
    "PodkowkaMendesPoisson2018": {"value": "C = GM/(R c^2) at which the central trace vanishes: 0.262 (+0.011/-0.017), median and 90 % "
                                           "interval over 20,000 four-parameter piecewise polytropes (Read+2009) with the low-density "
                                           "part fixed to the piecewise fit of Douchin & Haensel 2001 — a model-ensemble spread",
                                  "verified": "v2: ar5iv full text fetched (definition in the Introduction, sample size in Sec. III / Fig. 4)"},
    "Fujimoto2022": {"value": "the matter part of the trace anomaly 'may be positive definite' (no sign flip) when matched to "
                              "high-density QCD; it steeply approaches the conformal limit, explaining the sound-speed peak",
                     "verified": "v2: arXiv abstract fetched; the v1 phrase 'Delta near 0 by about 5 n_sat' is not in the abstract and is withdrawn"},
    "FarhiJaffe1984": {"status_v2": "window recomputed [DERIVED, Mentat]: B^1/4 = 144.4-162.8 MeV (massless quarks, alpha_s = 0, "
                                    "E/A(56Fe atom) = 930.41 MeV); the v1 '145-164' was secondary-sourced"},
    "OV1939": {"status_v2": "free-neutron M_max = 0.710 Msun recomputed in scratch numerics (n1_survey_v2 section 12); the Chamel+2013 "
                            "'~0.7' attribution is not in that abstract and is dropped; the OV abstract (APS) could not be fetched"},
}
for k, d in upd.items():
    by[k].update(d)

new = [
    {"key": "Vinciguerra2024", "cite": "Vinciguerra S. et al. 2024, ApJ 961, 62 — An updated mass-radius analysis of the 2017-2018 NICER data set of PSR J0030+0451",
     "id": "arXiv:2308.09469", "value": "including XMM-Newton constraints disfavours the 2019 main solutions; new solutions ~[1.4 Msun, 11.5 km] or ~[1.7 Msun, 14.5 km] depending on the model",
     "verified": "v2: arXiv abstract fetched", "url": "https://arxiv.org/abs/2308.09469"},
    {"key": "Salmi2024", "cite": "Salmi T. et al. 2024, ApJ 974, 294 — The Radius of the High-mass Pulsar PSR J0740+6620 with 3.6 yr of NICER Data",
     "id": "arXiv:2406.14466", "value": "R_eq = 12.49 (+1.28/-0.88) km, M = 2.073 +/- 0.069 Msun (68 %)",
     "verified": "v2: arXiv abstract fetched", "url": "https://arxiv.org/abs/2406.14466"},
    {"key": "Choudhury2024", "cite": "Choudhury D. et al. 2024, ApJL 971, L20 — A NICER View of the Nearest and Brightest Millisecond Pulsar: PSR J0437-4715",
     "id": "arXiv:2407.06789; doi:10.3847/2041-8213/ad5a6f", "value": "M = 1.418 +/- 0.037 Msun, R_eq = 11.36 (+0.95/-0.63) km (68 %)",
     "verified": "v2: arXiv abstract fetched; volume/page from Crossref metadata", "url": "https://arxiv.org/abs/2407.06789"},
    {"key": "HaenselPotekhin2004", "cite": "Haensel P. & Potekhin A. Y. 2004, A&A 428, 191 — Analytical representations of unified equations of state of neutron-star matter",
     "id": "arXiv:astro-ph/0408324", "value": "analytic fits of the unified FPS and SLy EOS; used here (eq. 14, Table 2 SLy) only for scratch verification of the plan's expected test outcomes",
     "verified": "v2: arXiv abstract fetched; fit coefficients as transcribed by the skeptic and reproduced DH2001 M_max 2.05 (2.047)", "url": "https://arxiv.org/abs/astro-ph/0408324"},
]
out = list(by.values()) + new
json.dump(out, open(HERE / "n1_references_v2.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(len(out), "references written")
