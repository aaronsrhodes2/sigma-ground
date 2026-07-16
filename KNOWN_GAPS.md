# Known Gaps — the greppable ledger

Per GOLDEN_RULES: every `# PHYSICS_GAP:` / `# GEOMETRY_GAP:` tag worth
tracking above file level lands here, so gaps get reviewed instead of
quietly accumulating.

## PHYSICS_GAP: electrolyte/soil corrosion kinetics not modeled (2026-07-15)

`sigma_ground/field/interface/corrosion.py` computes DRY-AIR Wagner oxidation
only. Corrosion in an electrolyte (soil, seawater, acid/caustic service) is an
electrochemical problem — pH-dependent rate, O2 mass transport, soil
resistivity cell — with no citable *general* model small enough to wire in
honestly. What exists instead: `environment_assessment()` places a metal in
its CITED regime (pH low-corrosion windows — Roetheli 1932 Zn, Whitman 1924
Fe, Pourbaix 1966 Al; ASTM G57 soil-resistivity corrosivity scale) and the
`corrosion_attack` verb states "dry-air kinetics" explicitly instead of
silently ignoring the environment words.

- Future quantitative layer: digitize Romanoff, "Underground Corrosion",
  NBS Circular 579 (1957) field data (zinc/steel mass-loss per soil type),
  and/or the Roetheli 1932 rate-vs-pH curve for zinc.
- Regression sentinels: `sigma_ground/field/interface/test_corrosion.py::
  TestEnvironment`, `sigma_ground/materia/tests/test_translator.py`
  (duration + environment extraction).

