# Parameter renames — APPLIED 2026-10-15; tool overlaps — still proposed

**Applied** (the Captain: "sure", 2026-10-15). The map is `sigma_ground/mcp/param_names.py`, the one place it lives. The tool functions keep their own parameter names;
the contract shows the canonical names in the schema and in the descriptions, **still accepts the old names** (if both are given the canonical one wins), and echoes
`inputs` under the canonical names. 47 parameters of 33 tools changed; the schema went from 311 to **290 distinct parameter names** (ratchet `distinct_param_names` in
`misc/contract_budget.json`). A rename is skipped in a tool where it would collide, so `wetting_analysis` keeps `solid_key`/`liquid_key` (two roles) and
`material_color_analysis` keeps `dye_key`/`substrate_key`. `semiconductor_junction_analysis.sc_key` became `semiconductor` (it is a semiconductor, not a superconductor, as the table below guessed).
The lint (`tools/tool_census.py`) counts old names in training data as `legacy`: the v3.1 data uses them in 7,457 of 61,392 calls, so the next data build writes the canonical names.
The router harness treats the old names as real parameters, so models trained on the old labels still run and are fairly scored.

The contract (`sigma_ground/mcp/contract.py`) fixed units, documentation and errors without changing a single parameter name: the router's training
labels (v3, v3.1) and the benchmark used the names below. Source: `misc/contract_census.json`.

## Names for the same idea (310 distinct parameter names across 232 tools)

| idea | today | proposed |
|---|---|---|
| a material from the table | `material`, `material_key`, `metal_key`, `solid_key`, `liquid_key`, `gas_key`, `substrate_key`, `counter_material` | `material` |
| two materials | `material_1`/`material_2`, `material_key_1`/`material_key_2`, `mat_a`/`mat_b`, `metal_a`/`metal_b`, `gas_key`/`gas_key_2` | `material_a`/`material_b` |
| semiconductor / superconductor | `semiconductor_key`, `sc_key`, `mat_n`/`mat_p` | `semiconductor`, `superconductor`, `material_n`/`material_p` |
| hot/cold temperature | `t_hot`/`t_cold`, `t_hot_k`/`t_cold_k`, `hot_temperature_k`/`cold_temperature_k` | `t_hot_k`/`t_cold_k` |
| the same quantity with and without its unit in the name | `energy_eV`/`energy_ev`/`energy_j`/`energy_joules`, `b_field_t`/`b_tesla`/`magnetic_field_t`, `molar_mass_kg`/`molar_mass_kg_per_mol`, `incidence_angle_deg`/`incident_angle_deg` | one spelling per quantity, the unit suffix always present |
| refractive index | `refractive_index`, `n1`/`n2`, `n_core`/`n_clad`, `n_dense`/`n_rare`, `n_high`/`n_low`, `n0` | keep (each is a different role), but `n` alone is never a refractive index |

## Tools that overlap (similar names or the same inputs): decide, merge only proven duplicates

`rydberg_hydrogen_wavelength` / `hydrogen_emission_wavelength` (same required inputs); `de_broglie_wavelength` / `de_broglie_from_kinetic_energy`;
`relativistic_energy` / `relativistic_energy_analysis`; `orbital_velocity` / `circular_orbit_velocity`; `projectile_range` / `projectile_max_height` /
`projectile_flight_time` / `projectile_analysis` / `procedure_projectile_trajectory`; `em_wave_wavelength` / `em_wave_frequency` / `photon_energy_from_*`.
Similar tools are a known cause of wrong-tool errors in small models (ToolScan, arXiv 2411.13547), and Anthropic's tool-writing guidance says the same
(consolidate overlapping tools; build only what the evaluations justify). The decision per pair needs the benchmark's per-tool usage, so it waits for the
yardstick (W6) rather than guessing.
