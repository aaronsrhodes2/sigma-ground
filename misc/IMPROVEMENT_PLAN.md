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

## 2026-05-17

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 2
- WOLFRAM PHRASING count:    1
- GEMINI HALLUCINATION count:2

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'light_travel_time']
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy', 'ohms_law_voltage']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `qm_008` (quantum_mechanics): What's the first ionization energy of sodium?

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

## 2026-05-17

- LIBRARY GAP count:        1
- DISCOVERABILITY GAP count: 25
- WOLFRAM PHRASING count:    1
- GEMINI HALLUCINATION count:2

### LIBRARY GAP — build the missing tool
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected tool: `maxwell_boltzmann_most_probable_speed`; expected value: 395.0

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'light_travel_time']
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy', 'ohms_law_voltage']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle', 'refractive_index']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['melting_point']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'electrical_power', 'rydberg_hydrogen_wavelength']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['single_slit_first_minimum_angle']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'photon_energy_from_wavelength', 'electrical_power']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['de_broglie_wavelength', 'em_wave_frequency']
- `qm_005` (quantum_mechanics): What's the de Broglie wavelength of a 1 keV electron?
    expected: `de_broglie_wavelength` ; Qwen tried: ['hydrogen_like_energy_level']
- `mech_adv_009` (classical_mechanics_advanced): How fast is the International Space Station going? It orbits at about 408 km altitude.
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body', 'em_wave_wavelength']
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['electrical_power', 'power_dissipation_resistor']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['electrical_power', 'power_dissipation_resistor', 'ohms_law_voltage']
- `gr_002` (general_relativity): Where is the photon sphere of a 10-solar-mass black hole?
    expected: `photon_sphere_radius` ; Qwen tried: ['hydrogen_like_energy_level', 'schwarzschild_radius']
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['light_travel_time', 'solar_system_body']
- ... +5 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `qm_008` (quantum_mechanics): What's the first ionization energy of sodium?

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

## 2026-05-17

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    1
- GEMINI HALLUCINATION count:2

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy', 'ohms_law_voltage']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['melting_point', 'refractive_index']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['single_slit_first_minimum_angle']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['light_travel_time']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance', 'de_broglie_wavelength']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power', 'power_dissipation_resistor', 'ohms_law_voltage']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'list_materials', 'light_travel_time']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'list_bodies']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `qm_008` (quantum_mechanics): What's the first ionization energy of sodium?

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

## 2026-05-17

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    1
- GEMINI HALLUCINATION count:2

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `qm_008` (quantum_mechanics): What's the first ionization energy of sodium?

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

## 2026-05-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:8

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `qm_006` (quantum_mechanics): expected 4.861e-07, gemini said 4.8628
- `cosmo_007` (cosmology): expected 0.412, gemini said Reproduction factor
- `cosmo_008` (cosmology): expected 4.35e+17, gemini said 4.36
- `astro_001` (astrophysics): expected 4.24, gemini said 1.34
- `astro_011` (astrophysics): expected 1.89813e+27, gemini said 1.898
- `astro_012` (astrophysics): expected 433.5, gemini said 323.0

---

## 2026-05-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:2

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$

---

## 2026-05-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:6

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `nuc_005` (nuclear_physics): expected 19.6, gemini said 0.02
- `math_001` (mathematical_methods): expected [-2, 2], gemini said 2.0
- `math_003` (mathematical_methods): expected cos(x), gemini said cosine of x
- `math_004` (mathematical_methods): expected [1, 2, 3], gemini said Not a physics question

---

## 2026-05-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:4

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0

---

## 2026-05-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:8

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `modern_011` (modern_physics): expected 7.275e-11, gemini said 7.27
- `qm_008` (quantum_mechanics): expected 5.139, gemini said 495.8
- `astro_011` (astrophysics): expected 1.89813e+27, gemini said 1.898
- `math_006` (mathematical_methods): expected 9460700000000000.0, gemini said 9.0

---

## 2026-05-20

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:16

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +6 more

---

## 2026-05-21

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:29

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +19 more

---

## 2026-05-22

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:39

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +29 more

---

## 2026-05-23

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:46

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +36 more

---

## 2026-05-24

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:57

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +47 more

---

## 2026-05-25

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-26

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'photon_energy_from_frequency']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current', 'rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 37
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle', 'diffraction_grating_angle', 'rydberg_hydrogen_wavelength', 'em_wave_wavelength']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'electrical_power', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_008` (modern_physics): 1 joule equals how many electronvolts?
    expected: `joules_to_eV` ; Qwen tried: ['photon_energy_from_wavelength']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- ... +17 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 44
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'ohms_law_voltage']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'ohms_law_current']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_wavelength', 'photon_energy_from_frequency', 'hydrogen_emission_wavelength']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_011` (thermodynamics_statmech): Convert 100 Celsius to Kelvin.
    expected: `temperature_celsius_to_kelvin` ; Qwen tried: ['melting_point', 'ohms_law_voltage']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- ... +24 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 41
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_012` (classical_mechanics_intro): If I launch a baseball at 40 m/s at 30 degrees above horizontal, how long is it in the air
    expected: `projectile_flight_time` ; Qwen tried: ['thin_lens_image_distance']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'ohms_law_voltage']
- `thermo_004` (thermodynamics_statmech): A heat engine runs between 600 K and 300 K. What's the maximum possible efficiency?
    expected: `carnot_efficiency` ; Qwen tried: ['ohms_law_voltage', 'rydberg_hydrogen_wavelength', 'ohms_law_current']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'photon_energy_from_wavelength', 'electrical_power', 'photon_energy_from_frequency', 'ohms_law_voltage']
- ... +21 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 37
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage', 'density', 'list_materials', 'element_atomic_data']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'ohms_law_voltage']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'solar_system_body', 'hydrogen_like_energy_level']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: ['light_travel_time']
- ... +17 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 39
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'ohms_law_voltage']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_wavelength', 'photon_energy_from_frequency', 'hydrogen_emission_wavelength']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- ... +19 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 37
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_012` (classical_mechanics_intro): If I launch a baseball at 40 m/s at 30 degrees above horizontal, how long is it in the air
    expected: `projectile_flight_time` ; Qwen tried: ['thin_lens_image_distance']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_001` (thermodynamics_statmech): What pressure does 1 mole of ideal gas exert at 0 Celsius in a 22.4 liter container?
    expected: `ideal_gas_pressure` ; Qwen tried: ['ohms_law_voltage']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'electrical_power', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'solar_system_body']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'photon_energy_from_wavelength', 'electrical_power', 'photon_energy_from_frequency']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- ... +17 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 35
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `thermo_002` (thermodynamics_statmech): What's the peak emission wavelength for a 6000 K star?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `thermo_003` (thermodynamics_statmech): What's the total power per square meter radiated by a perfect blackbody at 300 K?
    expected: `blackbody_total_power` ; Qwen tried: ['em_wave_wavelength', 'em_wave_frequency', 'photon_energy_from_wavelength', 'rydberg_hydrogen_wavelength', 'ohms_law_voltage']
- `thermo_005` (thermodynamics_statmech): What's the average thermal energy per molecule for a monatomic gas at room temperature, 30
    expected: `thermal_energy_per_molecule` ; Qwen tried: ['hydrogen_like_energy_level']
- `thermo_006` (thermodynamics_statmech): What's the most probable speed for an oxygen molecule at room temperature 300 K? Mass of O
    expected: `maxwell_boltzmann_most_probable_speed` ; Qwen tried: ['de_broglie_wavelength', 'hydrogen_like_energy_level', 'hydrogen_emission_wavelength', 'photon_energy_from_wavelength', 'photon_energy_from_frequency']
- `thermo_007` (thermodynamics_statmech): How much volume does 1 mole of gas take up at 25 Celsius and atmospheric pressure?
    expected: `ideal_gas_volume` ; Qwen tried: ['density']
- `thermo_012` (thermodynamics_statmech): The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'hydrogen_like_energy_level']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'hydrogen_like_energy_level', 'photon_energy_from_frequency']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'photon_energy_from_wavelength', 'electrical_power', 'photon_energy_from_frequency', 'ohms_law_voltage']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: ['light_travel_time']
- ... +15 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 31
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_014` (classical_mechanics_intro): What speed does a 0.5 kg ball need to have 100 joules of kinetic energy?
    expected: `solve_equation` ; Qwen tried: ['kinetic_energy', 'ohms_law_voltage']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `modern_002` (modern_physics): An astronaut travels at 0.99 c for 10 years (his time). How much time passes on Earth?
    expected: `relativistic_time_dilation` ; Qwen tried: ['thin_lens_image_distance']
- `modern_003` (modern_physics): A 1 meter rod moves at 0.6 c. How long is it as seen from the rest frame?
    expected: `relativistic_length_contraction` ; Qwen tried: ['thin_lens_image_distance']
- `modern_004` (modern_physics): How much energy is contained in 1 kg of mass via E=mc^2?
    expected: `mass_to_energy` ; Qwen tried: ['electrical_power']
- `modern_005` (modern_physics): The Sun emits about 3.828e26 watts of energy. How much mass does it convert to energy ever
    expected: `luminosity_to_mass_conversion_rate` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power', 'photon_energy_from_wavelength', 'solar_system_body']
- `modern_007` (modern_physics): How fast must I add 0.9 c to another 0.9 c (Einstein style) to get the total observed velo
    expected: `relativistic_velocity_addition` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `modern_009` (modern_physics): A 1-megaton nuclear weapon releases how many joules?
    expected: `joules_to_TNT` ; Qwen tried: ['light_travel_time', 'photon_energy_from_wavelength', 'electrical_power', 'photon_energy_from_frequency', 'hydrogen_like_energy_level']
- `modern_012` (modern_physics): If a galaxy recedes from us at 1000 km/s (0.0033 c), by what factor are its emission wavel
    expected: `doppler_shift_factor` ; Qwen tried: ['em_wave_wavelength']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: ['light_travel_time']
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- ... +11 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 22
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: ['light_travel_time']
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `astro_001` (astrophysics): How long does light from the nearest star (Proxima Centauri) take to reach us?
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `astro_004` (astrophysics): If Betelgeuse went supernova, how long would the light take to reach us?
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `astro_008` (astrophysics): What's the peak wavelength of the Sun's blackbody spectrum?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `astro_009` (astrophysics): How much energy does the Sun produce in 1 year?
    expected: `solve_equation` ; Qwen tried: ['photon_energy_from_wavelength', 'solar_system_body', 'light_travel_time', 'power_dissipation_resistor', 'ohms_law_voltage']
- `astro_010` (astrophysics): How fast is Earth moving in its orbit around the Sun?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- ... +2 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 22
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: ['light_travel_time']
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `astro_001` (astrophysics): How long does light from the nearest star (Proxima Centauri) take to reach us?
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `astro_004` (astrophysics): If Betelgeuse went supernova, how long would the light take to reach us?
    expected: `named_star` ; Qwen tried: ['light_travel_time']
- `astro_008` (astrophysics): What's the peak wavelength of the Sun's blackbody spectrum?
    expected: `blackbody_peak_wavelength` ; Qwen tried: ['rydberg_hydrogen_wavelength']
- `astro_009` (astrophysics): How much energy does the Sun produce in 1 year?
    expected: `solve_equation` ; Qwen tried: ['photon_energy_from_wavelength', 'solar_system_body', 'light_travel_time', 'power_dissipation_resistor', 'hydrogen_like_energy_level']
- ... +2 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 16
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_001` (waves_optics): Light hits water (n=1.333) from air at 30 degrees from vertical. What angle does it bend t
    expected: `snells_law_refraction_angle` ; Qwen tried: ['double_slit_fringe_spacing', 'single_slit_first_minimum_angle']
- `optics_002` (waves_optics): I'm looking up from underwater. At what angle does light from above stop being able to esc
    expected: `critical_angle_for_tir` ; Qwen tried: ['single_slit_first_minimum_angle']
- `optics_011` (waves_optics): What's the speed of sound in air at 20 degrees Celsius?
    expected: `speed_of_sound_in_ideal_gas` ; Qwen tried: ['rydberg_hydrogen_wavelength', 'de_broglie_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-29

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-30

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-05-31

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-01

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 15
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_001` (classical_mechanics_advanced): What's Jupiter's orbital velocity around the Sun (assuming a circular orbit)?
    expected: `circular_orbit_velocity` ; Qwen tried: ['solar_system_body']
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `em_adv_007` (electrodynamics_advanced): A 5 kW electric heater dissipates how much energy in 1 hour?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `em_adv_010` (electrodynamics_advanced): How long does it take a 5 watt LED to dissipate 1 kilojoule of energy?
    expected: `solve_equation` ; Qwen tried: ['power_dissipation_resistor', 'electrical_power']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['solar_system_body']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']
- `nuc_007` (nuclear_physics): How much mass is converted to energy in a 1-megaton thermonuclear explosion?
    expected: `energy_to_mass` ; Qwen tried: ['photon_energy_from_wavelength', 'light_travel_time', 'solar_system_body', 'named_star', 'density']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-01

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 22
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: ['orbital_velocity']
- `em_intro_001` (electromagnetism_intro): How much current flows through a 100 ohm resistor with 5 volts across it?
    expected: `ohms_law_current` ; Qwen tried: ['power_dissipation_resistor', 'coulomb_force', 'energy_power_time']
- `em_intro_002` (electromagnetism_intro): A 12 volt battery pushes 2 amps through a circuit. What's the power delivered?
    expected: `electrical_power` ; Qwen tried: ['photon_energy_from_frequency', 'energy_power_time']
- `em_intro_010` (electromagnetism_intro): What voltage develops across a 100 ohm resistor with 0.1 amps flowing?
    expected: `ohms_law_voltage` ; Qwen tried: ['em_wave_frequency', 'photon_energy_from_frequency']
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_005` (waves_optics): In a double-slit experiment with 633 nanometer laser light, slits 0.1 mm apart and screen 
    expected: `double_slit_fringe_spacing` ; Qwen tried: ['em_wave_wavelength', 'de_broglie_from_kinetic_energy', 'em_wave_frequency', 'de_broglie_wavelength', 'photon_energy_from_wavelength']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `mech_adv_008` (classical_mechanics_advanced): Earth's Moon orbits at about 384,000 km. What's the moon's orbital velocity around Earth?
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_adv_002` (electrodynamics_advanced): A 60 watt incandescent bulb has efficiency about 2 percent for visible light. How many wat
    expected: `electrical_power` ; Qwen tried: ['energy_power_time']
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_008` (general_relativity): At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds p
    expected: `gravitational_time_dilation` ; Qwen tried: ['orbital_velocity']
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `atom_001` (atomic_molecular): What's the atomic number of gold?
    expected: `element_atomic_data` ; Qwen tried: ['hydrogen_like_energy_level']
- `atom_006` (atomic_molecular): Iron has atomic number?
    expected: `element_atomic_data` ; Qwen tried: ['hydrogen_like_energy_level']
- `atom_008` (atomic_molecular): Uranium atomic mass?
    expected: `element_atomic_data` ; Qwen tried: ['nuclear_binding_energy']
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['kinetic_energy', 'energy_power_time']
- ... +2 more

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-01

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-01

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-02

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-04

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-05

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-06

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-07

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-08

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-09

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-10

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-11

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-12

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-13

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-14

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-15

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-16

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-17

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-18

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-19

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-20

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-21

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-22

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-23

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-24

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-25

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-26

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-27

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-28

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-29

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-06-30

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-01

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-02

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-03

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-04

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-05

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-06

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-07

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

## 2026-07-08

- LIBRARY GAP count:        0
- DISCOVERABILITY GAP count: 12
- WOLFRAM PHRASING count:    42
- GEMINI HALLUCINATION count:60

### DISCOVERABILITY GAP — keywords or pattern hint needed
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
    expected: `projectile_range` ; Qwen tried: ['double_slit_fringe_spacing']
- `mech_intro_015` (classical_mechanics_intro): If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I mov
    expected: `circular_orbit_velocity` ; Qwen tried: []
- `em_intro_012` (electromagnetism_intro): If I want a capacitor with 1 microfarad capacitance using 1 mm separation between plates i
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `optics_012` (waves_optics): Sirius B is a white dwarf about 8.6 light years away. How long does its light take to reac
    expected: `named_star` ; Qwen tried: []
- `em_adv_004` (electrodynamics_advanced): I want a 100 microfarad capacitor with 0.1 mm plates separation and a dielectric of relati
    expected: `solve_equation` ; Qwen tried: ['parallel_plate_capacitance']
- `gr_005` (general_relativity): How long would it take a stellar-mass (10 solar mass) black hole to evaporate via Hawking 
    expected: `hawking_evaporation_time` ; Qwen tried: []
- `gr_006` (general_relativity): If I drop a clock 1 km above a 10 solar mass black hole's event horizon, how slow does it 
    expected: `gravitational_time_dilation` ; Qwen tried: []
- `gr_009` (general_relativity): If I'm right at the event horizon of a black hole, what's the redshift of light I emit, as
    expected: `gravitational_redshift` ; Qwen tried: []
- `cosmo_002` (cosmology): What's the Hubble time (approximate age of the universe)?
    expected: `age_of_universe` ; Qwen tried: []
- `cosmo_004` (cosmology): Galaxy rotation curves probe accelerations of about 1e-10 m/s squared. Which regime is tha
    expected: `mond_regime_classifier` ; Qwen tried: []
- `cosmo_008` (cosmology): How old is the universe in seconds, approximately?
    expected: `age_of_universe` ; Qwen tried: []
- `nuc_003` (nuclear_physics): How much energy in eV does an alpha particle (4 amu) carry at 5 MeV?
    expected: `eV_to_joules` ; Qwen tried: ['photon_energy_from_frequency', 'kinetic_energy']

### WOLFRAM PHRASING — add manual `wolfram_phrasing` to question
- `mech_intro_001` (classical_mechanics_intro): If I drop a copper ball from 10 meters at sea level, how many seconds before it hits the g
- `mech_intro_002` (classical_mechanics_intro): I drop a steel ball from 50 meters. How fast is it moving when it hits the ground?
- `mech_intro_003` (classical_mechanics_intro): I shoot a cannonball at 100 meters per second from ground level at 45 degrees. How far doe
- `mech_intro_004` (classical_mechanics_intro): I shoot a ball at 50 m/s straight up. How high does it go?
- `mech_intro_005` (classical_mechanics_intro): What is the kinetic energy of a 70 kg person running at 5 m/s?
- `mech_intro_006` (classical_mechanics_intro): If I push a coffee cup with mass 0.2 kg across a table at 1 meter per second and the frict
- `mech_intro_007` (classical_mechanics_intro): How fast does a satellite need to orbit at 400 km altitude above Earth's surface?
- `mech_intro_009` (classical_mechanics_intro): What's the momentum of a 1500 kg car going 25 m/s?
- `mech_intro_010` (classical_mechanics_intro): How much potential energy does a 70 kg climber have at the top of a 100 m cliff?
- `mech_intro_011` (classical_mechanics_intro): I drop a ball from 10 meters on the Moon. How long does it take to land? Moon gravity is a
- ... +32 more

### GEMINI HALLUCINATION (Gemini confident, wrong)
- `mech_intro_015` (classical_mechanics_intro): expected 0.0, gemini said 3074.0
- `em_intro_003` (electromagnetism_intro): expected 8.854e-09, gemini said $8.854 \times 10^{-9} \text{ F}$
- `em_intro_007` (electromagnetism_intro): expected 545100000000000.0, gemini said $5.45 \times 10^{14}$ Hz
- `em_intro_014` (electromagnetism_intro): expected 3e+18, gemini said 3.0
- `optics_004` (waves_optics): expected 6.563e-07, gemini said 6.5647
- `optics_009` (waves_optics): expected 1.216e-07, gemini said 1.2150242
- `optics_012` (waves_optics): expected 2.667, gemini said 8.6
- `thermo_001` (thermodynamics_statmech): expected 101325, gemini said 1.01
- `thermo_002` (thermodynamics_statmech): expected 4.83e-07, gemini said $4.83 \times 10^{-7}$ m
- `thermo_003` (thermodynamics_statmech): expected 459.0, gemini said 4.59
- ... +50 more

---

