# Parameter renames and tool overlaps — proposed, NOT applied

The contract (`sigma_ground/mcp/contract.py`) fixed units, documentation and errors without changing a single parameter name: the router's training
labels (v3, v3.1) and the benchmark use today's names. When the labels are rebuilt for the final contract (CONTENDER_PLAN W7) the names below change
together, with the old names kept as accepted aliases for one release. Source: `misc/contract_census.json`.

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
