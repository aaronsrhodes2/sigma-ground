# sigma-ground benchmark — Improvement Plan

This file accumulates findings from `daily_job.py`. Each daily section
catalogs the failures that day, so when you sit down to work on the
benchmark you have a queue of concrete improvements to tackle.

Categories:
  - **LIBRARY GAP** — sigma-ground answered "Fitted due to incompetence".
    Build the missing tool.
  - **DISCOVERABILITY GAP** — sigma-ground has the tool but Qwen used the
    wrong one. Add keywords or pattern hints.
  - **WOLFRAM PHRASING** — Wolfram couldn't parse a question even after
    reformulation. Add a manual `wolfram_phrasing` field to the question.
  - **GEMINI HALLUCINATION** — Gemini gave a confidently-wrong answer.
    Worth keeping as evidence of bare-LLM failure modes.

---

## 2026-05-17

- LIBRARY GAP count:        1
- DISCOVERABILITY GAP count: 45
- WOLFRAM PHRASING count:    1
- GEMINI HALLUCINATION count:2

### LIBRARY GAP — build the missing tool
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
    expected tool: `kinetic_energy`; expected value: 875.0

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'light_travel_time']
- `mech_intro_008` (classical_mechanics_intro): What's the escape velocity from Earth's surface?
    expected: `escape_velocity` ; Qwen tried: ['solar_system_body', 'em_wave_wavelength']
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'light_travel_time']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['refractive_index', 'double_slit_fringe_spacing', 'single_slit_first_minimum_angle', 'light_travel_time']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['melting_point']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage', 'parallel_plate_capacitance', 'ohms_law_current', 'electrical_power', 'rc_time_constant']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['single_slit_first_minimum_angle']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'rydberg_hydrogen_wavelength', 'electrical_power', 'power_dissipation_resistor', 'list_materials']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'single_slit_first_minimum_angle']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['single_slit_first_minimum_angle']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['list_bodies', 'light_travel_time', 'rlc_resonant_frequency']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance', 'lens_magnification', 'parallel_plate_capacitance']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'photon_energy_from_frequency']
- `modern_010` (modern_physics): An electron has rest mass 9.109e-31 kg. What's its rest energy in MeV?
    expected: `mass_to_energy` ; Qwen tried: ['element_atomic_data', 'photon_energy_from_frequency']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['de_broglie_wavelength', 'light_travel_time']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'light_travel_time', 'hydrogen_like_energy_level']
- ... +25 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `qm_008` (quantum_mechanics): What's the first ionization energy of sodium?

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

