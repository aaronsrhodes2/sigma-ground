"""Keywords / formula names / theory names / attributions per tool.

This is the discoverability sidecar for the MCP manifest. Each tool's
keywords list captures:
  - The formula name        ("Stefan-Boltzmann law", "Schwarzschild radius")
  - The named theorem       ("Pythagorean theorem", "no-communication theorem")
  - Famous attribution      ("Karl Schwarzschild 1916", "Einstein 1905")
  - Common symbol forms     ("r_s", "E=mc^2", "lambda_max")
  - Topic / phenomenon      ("event horizon", "blackbody", "Hubble flow")

manifest.py merges this into _PRIMARY_TOOLS at module load, and
run_sigma_ground.py surfaces it in the TOOL INDEX so Qwen sees
"schwarzschild_radius" alongside its formal name "Schwarzschild
radius / event horizon (Karl Schwarzschild 1916)" when scanning the
list.

Keywords should be lowercased; matching is substring-based. Add
generously -- false positives are cheap, missed matches are expensive.
"""

from __future__ import annotations


TOOL_KEYWORDS: dict[str, list[str]] = {
    # ============ CONSTANTS / UNITS / SYMBOLIC ============
    "lookup_constant": [
        "physical constant", "fundamental constant", "CODATA",
        "speed of light c", "planck constant h", "boltzmann k",
        "gravitational constant G", "fine structure alpha",
    ],
    "list_constants": ["available constants", "constant catalog"],
    "convert_units": [
        "unit conversion", "convert between units", "pint",
        "meters to feet", "joules to electronvolts", "kelvin to celsius",
        "light year to meters", "parsec to meter",
    ],
    "parse_quantity": ["parse a quantity string", "value with units"],
    "solve_equation": [
        "symbolic solve", "algebra", "polynomial root",
        "quadratic formula", "cubic equation", "find x such that",
        "sympy solve",
    ],
    "integrate_expr": [
        "integral", "indefinite integral", "definite integral",
        "calculus", "antiderivative", "integration",
        "fundamental theorem of calculus",
    ],
    "differentiate_expr": [
        "derivative", "differentiate", "calculus",
        "chain rule", "rate of change",
    ],
    "simplify_expr": [
        "simplify expression", "algebraic simplification",
        "trig identities", "Pythagorean identity sin^2+cos^2=1",
    ],

    # ============ KINEMATICS ============
    "free_fall_time": [
        "free fall", "drop time", "t = sqrt(2h/g)",
        "Galileo free fall", "kinematics equation",
    ],
    "free_fall_velocity": [
        "free fall velocity", "v = sqrt(2gh)", "impact velocity",
        "terminal velocity (in vacuum)",
    ],
    "projectile_range": [
        "projectile motion", "projectile range",
        "R = v^2 sin(2 theta) / g", "cannonball range",
        "ballistic range",
    ],
    "projectile_max_height": [
        "projectile peak", "h_max = v^2 sin^2(theta) / (2g)",
        "maximum height of a projectile",
    ],
    "kinetic_energy": [
        "kinetic energy", "KE = 0.5 m v^2", "KE", "K_e",
        "energy of motion",
    ],
    "momentum": [
        "linear momentum", "p = m v", "Newton's second law in p form",
    ],
    "elastic_collision_velocities": [
        "elastic collision", "conservation of momentum",
        "1D collision", "billiard ball collision",
    ],
    "inelastic_collision_velocity": [
        "inelastic collision", "perfectly inelastic",
        "stick together collision",
    ],
    "circular_orbit_velocity": [
        "orbital velocity", "circular orbit speed",
        "v_orbit = sqrt(GM/r)", "Kepler third law derivation",
        "satellite speed",
    ],
    "escape_velocity": [
        "escape velocity", "v_esc = sqrt(2GM/r)",
        "escape from gravity well", "minimum speed to leave",
    ],
    "gravitational_potential_energy": [
        "gravitational PE", "U = -GMm/r", "PE in gravity well",
        "Newton's gravity potential", "lift to orbit energy",
    ],

    # ============ ELECTROMAGNETISM / CIRCUITS ============
    "ohms_law_voltage": [
        "Ohm's law", "V = I R", "voltage drop across resistor",
    ],
    "ohms_law_current": [
        "Ohm's law for current", "I = V / R",
    ],
    "ohms_law_resistance": [
        "Ohm's law for resistance", "R = V / I",
    ],
    "electrical_power": [
        "electrical power", "P = V I", "P = I^2 R",
        "P = V^2 / R", "Joule heating",
    ],
    "parallel_plate_capacitance": [
        "capacitor", "C = epsilon_0 A / d",
        "parallel plate capacitor", "capacitance formula",
    ],
    "rc_time_constant": [
        "RC time constant", "tau = R C", "RC circuit",
        "exponential decay in RC",
    ],
    "rl_time_constant": [
        "RL time constant", "tau = L / R", "inductor RL",
    ],
    "rlc_resonant_frequency": [
        "RLC resonance", "f_0 = 1/(2 pi sqrt(LC))",
        "LC tank circuit", "resonant frequency",
    ],
    "em_wavelength_from_frequency": [
        "EM wavelength", "lambda = c / f",
        "wavelength from frequency", "radio wavelength",
    ],
    "em_frequency_from_wavelength": [
        "EM frequency", "f = c / lambda",
        "frequency from wavelength",
    ],

    # ============ OPTICS / WAVES ============
    "snells_law_refraction_angle": [
        "Snell's law", "law of refraction",
        "n1 sin(theta1) = n2 sin(theta2)", "refraction angle",
        "Willebrord Snell 1621",
    ],
    "critical_angle_total_internal_reflection": [
        "total internal reflection", "critical angle",
        "sin(theta_c) = n2 / n1", "TIR", "fiber optic principle",
    ],
    "thin_lens_image_distance": [
        "thin lens equation", "1/f = 1/d_o + 1/d_i",
        "lens formula", "image distance", "focal length lens",
    ],
    "lens_magnification": [
        "lens magnification", "m = -d_i / d_o",
        "image magnification",
    ],
    "rydberg_hydrogen_wavelength": [
        "Rydberg formula", "Balmer series", "Lyman series",
        "Paschen series", "Brackett series", "Pfund series",
        "1/lambda = R(1/n1^2 - 1/n2^2)", "hydrogen spectral line",
        "H-alpha", "Lyman-alpha", "Johannes Rydberg 1888",
    ],
    "single_slit_first_minimum_angle": [
        "single slit diffraction", "first minimum",
        "a sin(theta) = m lambda", "single slit pattern",
    ],
    "double_slit_fringe_spacing": [
        "double slit", "Young's double-slit", "interference fringes",
        "y = lambda L / d", "Thomas Young 1801",
    ],
    "diffraction_grating_angle": [
        "diffraction grating", "d sin(theta) = m lambda",
        "grating equation", "spectral grating",
    ],
    "speed_of_sound_in_air": [
        "speed of sound", "v_sound = 331 + 0.6 T",
        "sound velocity in air",
    ],

    # ============ THERMODYNAMICS / STATMECH ============
    "ideal_gas_pressure": [
        "ideal gas law", "PV = nRT", "P from ideal gas",
        "Clapeyron equation", "perfect gas",
    ],
    "ideal_gas_volume": [
        "molar volume", "V = nRT/P", "STP volume 22.4 L",
    ],
    "blackbody_peak_wavelength": [
        "Wien's displacement law", "lambda_max T = b",
        "blackbody peak", "Planck radiation peak",
        "Wilhelm Wien 1893",
    ],
    "stefan_boltzmann_power": [
        "Stefan-Boltzmann law", "P = sigma A T^4",
        "blackbody total radiation", "Stefan's law",
        "Josef Stefan 1879 / Ludwig Boltzmann 1884",
    ],
    "carnot_efficiency": [
        "Carnot efficiency", "eta = 1 - T_c/T_h",
        "Carnot cycle", "ideal heat engine",
        "Sadi Carnot 1824", "second law thermodynamics",
    ],
    "average_thermal_energy": [
        "equipartition theorem", "(3/2) k_B T",
        "average kinetic energy of gas molecule",
    ],
    "maxwell_boltzmann_most_probable_speed": [
        "Maxwell-Boltzmann distribution",
        "v_mp = sqrt(2 k T / m)", "most probable speed",
        "molecular speed distribution",
    ],
    "temperature_celsius_to_kelvin": [
        "Convert Celsius to Kelvin", "C to K", "Celsius to K",
        "degrees Celsius to Kelvin", "T_K = T_C + 273.15",
        "what's 100 Celsius in Kelvin", "freezing point in Kelvin",
        "boiling point in Kelvin", "convert temperature",
    ],
    "temperature_kelvin_to_celsius": [
        "Convert Kelvin to Celsius", "K to C", "Kelvin to Celsius",
        "T_C = T_K - 273.15", "what is X K in Celsius",
        "convert temperature",
    ],
    "temperature_fahrenheit_to_celsius": [
        "F to C", "Fahrenheit to Celsius", "convert temperature",
    ],
    "temperature_celsius_to_fahrenheit": [
        "C to F", "Celsius to Fahrenheit", "convert temperature",
    ],

    # ============ ATOMIC / QUANTUM ============
    "element_atomic_data": [
        "atomic number", "atomic mass", "Z of", "atomic weight",
        "amu", "u", "unified atomic mass", "periodic table",
        "how many protons", "mass of element", "Mendeleev",
        "IUPAC atomic weight",
    ],
    "first_ionization_energy": [
        "ionization energy", "IE", "first ionization potential",
        "Z_eff", "remove electron energy",
    ],
    "hydrogen_like_energy_level": [
        "Bohr model", "E_n = -13.6 Z^2 / n^2 eV",
        "hydrogen energy levels", "Bohr energy",
        "Niels Bohr 1913",
    ],
    "photon_energy_from_wavelength": [
        "photon energy", "E = h c / lambda", "Einstein photon",
    ],
    "photon_energy_from_frequency": [
        "photon energy from frequency", "E = h f",
        "Planck-Einstein relation",
    ],
    "de_broglie_wavelength": [
        "de Broglie wavelength", "lambda = h / p",
        "wave-particle duality", "matter wave",
        "Louis de Broglie 1924",
    ],

    # ============ SPECIAL RELATIVITY ============
    "lorentz_factor": [
        "Lorentz factor", "gamma = 1/sqrt(1 - v^2/c^2)",
        "time dilation factor", "Hendrik Lorentz",
    ],
    "relativistic_time_dilation": [
        "time dilation", "moving clocks tick slower",
        "delta_t = gamma delta_tau", "twin paradox",
        "Einstein 1905 special relativity",
    ],
    "relativistic_length_contraction": [
        "length contraction", "Lorentz contraction",
        "L = L_0 / gamma", "ladder paradox",
    ],
    "relativistic_momentum": [
        "relativistic momentum", "p = gamma m v",
        "high-velocity momentum",
    ],
    "relativistic_energy": [
        "relativistic energy", "E = gamma m c^2",
        "total relativistic energy",
    ],
    "relativistic_velocity_addition": [
        "velocity addition formula", "u' = (u+v)/(1 + uv/c^2)",
        "Einstein velocity addition",
    ],
    "doppler_shift_factor": [
        "Doppler effect", "Doppler shift",
        "relativistic Doppler", "redshift", "blueshift",
        "Christian Doppler 1842",
    ],

    # ============ GENERAL RELATIVITY ============
    "schwarzschild_radius": [
        "Schwarzschild radius", "event horizon",
        "r_s = 2GM/c^2", "black hole radius",
        "Karl Schwarzschild 1916", "Schwarzschild solution",
    ],
    "photon_sphere_radius": [
        "photon sphere", "r = 1.5 r_s", "1.5 Schwarzschild radii",
        "unstable photon orbit", "black hole shadow",
    ],
    "isco_radius": [
        "ISCO", "innermost stable circular orbit",
        "r_ISCO = 3 r_s", "Schwarzschild ISCO",
        "accretion disk inner edge",
    ],
    "hawking_temperature": [
        "Hawking temperature", "Hawking radiation",
        "T_H = hbar c^3 / (8 pi G M k_B)",
        "Stephen Hawking 1974", "black hole evaporation temperature",
    ],
    "hawking_evaporation_time": [
        "Hawking evaporation time", "black hole lifetime",
        "tau ~ M^3", "Hawking radiation rate",
    ],
    "gravitational_time_dilation": [
        "gravitational time dilation", "GR time dilation",
        "sqrt(1 - r_s/r)", "GPS time dilation",
        "Pound-Rebka experiment", "general relativistic redshift",
    ],
    "gravitational_redshift": [
        "gravitational redshift", "Pound-Rebka",
        "z = sqrt(1 - r_s/r_emit) / sqrt(1 - r_s/r_obs) - 1",
        "redshift from gravity well",
    ],

    # ============ COSMOLOGY ============
    "hubble_radius": [
        "Hubble radius", "Hubble sphere", "c / H_0",
        "observable universe radius", "Hubble length",
    ],
    "age_of_universe": [
        "age of universe", "Hubble time", "1 / H_0",
        "13.8 billion years",
    ],
    "critical_density": [
        "critical density", "rho_crit = 3 H^2 / (8 pi G)",
        "Omega = 1 universe", "flat universe density",
    ],
    "hde_dark_energy_density": [
        "holographic dark energy", "HDE",
        "DESI dark energy", "Union3 supernova",
    ],
    "mond_regime_classifier": [
        "MOND", "modified Newtonian dynamics", "a_0 acceleration",
        "Mordehai Milgrom 1983", "galaxy rotation curves",
        "deep MOND regime", "Newtonian regime",
    ],
    "mond_a0_constant": [
        "MOND a_0", "1.2e-10 m/s^2", "MOND acceleration scale",
    ],
    "eta_value_report": [
        "eta", "η", "SSBM eta", "scale conversion factor",
        "sigma-ground eta", "DESI Union3 c^2",
    ],

    # ============ ENERGY CONVERSION (E=mc^2 family) ============
    "mass_to_energy": [
        "E=mc^2", "mass-energy equivalence", "rest mass energy",
        "Einstein 1905", "matter to energy conversion",
        "nuclear binding", "annihilation energy",
    ],
    "energy_to_mass": [
        "m = E / c^2", "mass equivalent of energy",
        "mass defect", "energy to mass",
    ],
    "joules_to_TNT": [
        "TNT equivalent", "megatons of TNT",
        "1 ton TNT = 4.184e9 J", "explosive yield",
        "nuclear yield",
    ],
    "eV_to_joules": [
        "electronvolt", "eV to J", "1 eV = 1.602e-19 J",
        "MeV", "GeV", "TeV", "particle physics energy",
    ],
    "joules_to_eV": [
        "J to eV", "joules to electronvolts", "1 J equals how many eV",
        "convert energy J to eV", "joules in electronvolts",
        "1 joule equals how many electronvolts",
        "energy unit conversion", "6.242e18",
    ],
    "luminosity_to_mass_conversion_rate": [
        "stellar mass loss to luminosity",
        "Sun mass-loss rate", "L = (dm/dt) c^2",
        "solar luminosity mass rate",
    ],

    # ============ ASTRONOMY ============
    "solar_system_body": [
        "planet data", "moon data", "Sun data",
        "NASA planetary fact sheet", "IAU 2015 nominal",
        "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto",
    ],
    "list_bodies": [
        "list solar system bodies", "available planets",
    ],
    "named_star": [
        "named star data", "bright star catalog",
        "Sirius Vega Polaris Betelgeuse Rigel Proxima",
        "Hipparcos Gaia DR3", "Bayer designation",
    ],
    "light_travel_time": [
        "light travel time", "delay = distance / c",
        "communication lag", "Earth-Sun light time 8 minutes",
        "Proxima 4.2 light years",
    ],

    # ============ MATERIALS ============
    "density": [
        "density", "rho = m/V", "kg/m^3", "specific gravity",
    ],
    "refractive_index": [
        "refractive index", "n at 589 nm", "sodium D-line",
        "speed of light in medium", "optical density",
    ],
    "melting_point": ["melting point", "T_melt", "fusion temperature"],
    "boiling_point": ["boiling point", "T_boil", "vaporization temperature"],
    "youngs_modulus": [
        "Young's modulus", "E", "elastic modulus",
        "Hooke's law constant", "stiffness", "Thomas Young 1807",
    ],
    "band_gap_ev": [
        "band gap", "semiconductor gap", "valence to conduction",
        "Si Ge GaAs gap", "Eg",
    ],
    "list_materials": ["available materials"],
    "thermal_conductivity": [
        "thermal conductivity", "k_thermal",
        "Fourier law of heat conduction", "W/(m K)",
    ],

    # ============ CIRCUITS - extended ============
    "speed_of_em_in_medium": [
        "speed of light in a medium", "v = c / n",
        "phase velocity in matter",
    ],

    # ============ OPTICS - extended ============
    "lyman_alpha_wavelength": [
        "Lyman-alpha", "121.567 nm", "hydrogen UV line",
        "Lyman series first line",
    ],
}
