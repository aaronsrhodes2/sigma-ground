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
        "MeV to eV", "how much energy in eV", "express in electronvolts",
        "amu to kg", "MeV to joules",
    ],
    "parse_quantity": ["parse a quantity string", "value with units"],
    "percent_of": [
        "N percent of", "percentage of a value", "X% of Y",
        "efficiency of", "what fraction is", "how much is N percent of",
        "efficiency for", "N percent efficient",
    ],
    "solve_equation": [
        "symbolic solve", "algebra", "polynomial root",
        "quadratic formula", "cubic equation", "find x such that",
        "sympy solve",
        # inverse-design: "given a target result, find the unknown input" --
        # deliberately scoped to this framing (not general capacitor/circuit
        # vocabulary) so plain forward lookups stay on their own tool.
        "what area do I need for a capacitor with", "design a capacitor with",
        "what plate area do I need", "solve for the plate area given",
        "how big do the plates need to be", "what value of X gives",
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
    "matrix_determinant": ["determinant", "det of matrix", "|A|"],
    "matrix_eigenvalues": ["eigenvalues", "eigenvalue", "characteristic equation",
                            "spectrum of a matrix"],
    "matrix_inverse": ["matrix inverse", "invert a matrix", "A^-1"],
    "matrix_multiply": ["matrix product", "multiply matrices", "AB"],
    "solve_linear_system": ["linear system", "solve Ax=b", "system of equations",
                             "Cramer's rule", "Gaussian elimination"],
    "compute_limit": ["limit", "as x approaches", "L'Hopital", "tends to",
                       "limit of a function"],
    "series_expansion": ["Taylor series", "Maclaurin series", "power series",
                          "series expansion"],
    "summation": ["sum of a series", "summation", "infinite series",
                   "sum from n to infinity", "geometric series"],
    "laplace_transform": ["Laplace transform", "L{f(t)}", "s-domain"],
    "fourier_transform": ["Fourier transform", "frequency domain", "spectrum"],
    "factor_expression": ["factor", "factorize", "factorization", "roots of polynomial"],
    "expand_expression": ["expand", "binomial expansion", "multiply out"],
    "solve_ode": ["differential equation", "ODE", "solve y'' ", "general solution",
                   "homogeneous equation"],
    "gradient": ["gradient", "grad", "del f", "nabla f"],
    "divergence": ["divergence", "div F", "del dot F"],
    "curl": ["curl", "del cross F", "rotation of a vector field"],

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
    "circular_orbit_velocity": [
        "orbital velocity", "circular orbit speed",
        "v_orbit = sqrt(GM/r)", "Kepler third law derivation",
        "satellite speed",
    ],
    "orbital_velocity": [
        "orbital velocity", "how fast does it orbit", "satellite speed",
        "ISS speed", "orbital speed at altitude", "Moon orbital velocity",
        "Jupiter orbital velocity", "geostationary speed",
        "how fast is X moving in its orbit", "v = sqrt(GM/r)",
    ],
    "orbital_period": [
        "orbital period", "Kepler third law", "how long is a year on",
        "year length", "T = 2 pi sqrt(a^3/GM)", "period of an orbit",
        "asteroid orbital period", "how long to orbit",
    ],
    "gravitational_force": [
        "gravitational force", "Newton's law of gravitation",
        "F = G m1 m2 / r^2", "force of gravity between two masses",
        "gravitational attraction",
    ],
    "orbital_raise_energy": [
        "energy to lift to orbit", "energy to raise a satellite",
        "lift to geosynchronous", "gravitational PE difference",
        "delta U = GMm(1/r1 - 1/r2)", "energy to reach orbit",
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
    "em_wave_wavelength": [
        "EM wavelength", "lambda = c / f",
        "wavelength from frequency", "radio wavelength",
        "WiFi wavelength", "X-ray wavelength", "gamma ray wavelength",
        "5G wavelength", "microwave wavelength",
    ],
    "em_wave_frequency": [
        "EM frequency", "f = c / lambda",
        "frequency from wavelength",
        "green light frequency", "X-ray frequency",
    ],

    # ============ OPTICS / WAVES ============
    "snells_law_refraction_angle": [
        "Snell's law", "law of refraction",
        "n1 sin(theta1) = n2 sin(theta2)", "refraction angle",
        "Willebrord Snell 1621",
    ],
    "critical_angle_for_tir": [
        "total internal reflection", "critical angle",
        "sin(theta_c) = n2 / n1", "TIR", "fiber optic principle",
        "at what angle does light stop being able to escape",
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
    "speed_of_sound_in_ideal_gas": [
        "speed of sound", "v_sound = 331 + 0.6 T",
        "sound velocity in air", "speed of sound in air at temperature",
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
    "blackbody_total_power": [
        "Stefan-Boltzmann law", "P = sigma A T^4",
        "blackbody total radiation", "Stefan's law",
        "Josef Stefan 1879 / Ludwig Boltzmann 1884",
        "total power per square meter radiated by blackbody",
    ],
    "carnot_efficiency": [
        "Carnot efficiency", "eta = 1 - T_c/T_h",
        "Carnot cycle", "ideal heat engine",
        "Sadi Carnot 1824", "second law thermodynamics",
    ],
    "thermal_energy_per_molecule": [
        "equipartition theorem", "(3/2) k_B T",
        "average kinetic energy of gas molecule",
        "average thermal energy", "thermal energy per molecule",
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
    "de_broglie_from_kinetic_energy": [
        "de Broglie wavelength of a 1 keV electron",
        "de Broglie from energy", "matter wave from kinetic energy",
        "wavelength of an electron at X eV", "lambda from KE",
        "1 keV electron wavelength", "X MeV particle wavelength",
    ],
    "nuclear_binding_energy": [
        "nuclear binding energy", "mass defect", "binding energy per nucleon",
        "how much lighter is the nucleus", "Bethe-Weizsacker", "SEMF",
        "iron binding energy", "He-4 binding", "baryon vs mass",
    ],
    "coulomb_force": [
        "Coulomb force", "Coulomb's law", "F = k q1 q2 / r^2",
        "electrostatic force", "force between two charges",
    ],
    "fission_energy_release": [
        "fission energy release", "energy from fissioning", "U-235 fissioned",
        "kg of uranium fissioned", "fully fissioned releases",
        "200 MeV per fission", "atomic bomb energy release",
    ],
    "energy_power_time": [
        "energy = power times time", "E = P t",
        "how much energy in X hours", "how long to dissipate",
        "kilowatt hour", "heater energy", "LED energy",
        "watt-hour", "5 kW for 1 hour", "joules dissipated",
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
    "bekenstein_hawking_entropy": [
        "Bekenstein-Hawking entropy", "black hole entropy",
        "S = A / 4 L_p^2", "horizon entropy", "holographic bound",
        "Jacob Bekenstein 1973", "horizon information capacity",
    ],
    "gravitational_binding_energy": [
        "gravitational binding energy", "self-gravity energy",
        "U = 3/5 G M^2 / R", "energy to assemble a sphere",
        "stellar binding energy",
    ],
    "unruh_temperature": [
        "Unruh temperature", "Unruh effect", "acceleration temperature",
        "T = hbar a / (2 pi c k_B)", "William Unruh 1976",
        "thermal vacuum of an accelerated observer",
    ],
    "entanglement_channel": [
        "entanglement communication", "faster than light", "FTL signaling",
        "no-communication theorem", "quantum key distribution", "QKD",
        "Bell inequality", "CHSH", "Tsirelson bound", "spooky action",
        "can entangled particles communicate", "EPR signaling",
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
        "HDE c^2", "dark energy c^2 parameter", "DESI Union3 c^2",
        "holographic dark energy parameter",
    ],

    # ============ MECHANICS (composite analysis tools) ============
    "collision_analysis": [
        "collision", "elastic collision", "inelastic collision",
        "coefficient of restitution", "two balls collide", "energy lost in collision",
    ],
    "work_energy_analysis": [
        "work done", "mechanical power", "potential energy", "impulse",
        "work-energy theorem", "energy of a moving object",
    ],
    "projectile_analysis": [
        "projectile", "range of a projectile", "maximum height", "time of flight",
        "launch angle", "projectile with air resistance", "how far will it fly",
    ],
    "incline_analysis": [
        "inclined plane", "ramp", "sliding down a slope", "angle of repose",
        "friction on an incline", "block on a ramp",
    ],

    # ============ TRANSPORT & STATISTICAL MECHANICS (composite) ============
    "viscous_flow_analysis": [
        "viscosity", "Reynolds number", "Stokes drag", "terminal velocity",
        "Poiseuille flow", "drag on a sphere", "laminar or turbulent",
    ],
    "diffusion_analysis": [
        "diffusion", "Fick's law", "diffusion coefficient", "Einstein-Stokes",
        "how long to diffuse", "interdiffusion",
    ],
    "statistical_distribution": [
        "Fermi-Dirac", "Bose-Einstein", "partition function", "occupation probability",
        "equipartition", "heat capacity from degrees of freedom",
    ],
    "rotational_dynamics": [
        "moment of inertia", "angular momentum", "torque", "rolling",
        "rolling down a ramp", "parallel axis theorem", "rotational kinetic energy",
    ],
    "atomic_angular_momentum": [
        "angular momentum quantum number", "spin-orbit coupling", "term symbol",
        "Lande interval", "fine structure splitting", "spin expectation value",
    ],

    # ============ MATERIALS STRENGTH + COMPOSITES (composite) ============
    "elastic_analysis": [
        "Young's modulus", "stress and strain", "Hooke's law", "shear stress",
        "strain energy", "Poisson ratio", "von Mises yield",
    ],
    "stress_failure_analysis": [
        "fracture toughness", "stress intensity", "crack", "fatigue life",
        "creep", "Paris law", "when will it break",
    ],
    "plasticity_analysis": [
        "plastic deformation", "flow stress", "work hardening", "Johnson-Cook",
        "Ludwik", "yield and beyond",
    ],
    "composite_bounds_analysis": [
        "composite material", "Voigt-Reuss-Hill", "Hashin-Shtrikman",
        "rule of mixtures", "effective modulus", "foam strength",
    ],

    # ============ PHOTONICS / OPTICS / ELECTROCERAMICS (batch 5) ============
    "optical_waveguide_analysis": [
        "optical waveguide", "slab waveguide", "V-number", "numerical aperture",
        "guided modes", "single-mode fiber", "how many modes", "core cladding",
    ],
    "photonic_bandgap_analysis": [
        "Bragg mirror", "dielectric mirror", "distributed Bragg reflector",
        "photonic bandgap", "quarter-wave stack", "stop band", "DBR reflectance",
    ],
    "nonlinear_optics_analysis": [
        "nonlinear optics", "Kerr effect", "self-focusing", "B-integral",
        "second harmonic generation", "SHG", "critical power", "n2 intensity",
    ],
    "material_color_analysis": [
        "what color is", "metal color", "color of gold", "color of copper",
        "sRGB of material", "physical color", "reflectance color", "dye color",
    ],
    "phosphor_decay_analysis": [
        "phosphor", "afterglow", "persistence", "luminescence decay",
        "glow brightness", "exponential decay of light", "decay time constant",
    ],
    "piezoelectric_actuator_analysis": [
        "piezoelectric", "piezo actuator", "converse piezoelectric effect",
        "PZT displacement", "strain from voltage", "d33", "quartz actuator",
    ],
    "dielectric_polarization_analysis": [
        "Clausius-Mossotti", "relative permittivity", "dielectric constant from polarizability",
        "polarizability", "Lorentz-Lorenz", "induced dipole permittivity",
    ],

    # ============ THERMAL SYSTEMS / MECHANICAL RESPONSE (batch 6) ============
    "thermoelectric_generator_analysis": [
        "thermoelectric", "Seebeck effect", "thermocouple voltage", "TEG",
        "Peltier", "figure of merit ZT", "Carnot efficiency", "Ioffe",
        "waste heat to electricity",
    ],
    "natural_convection_analysis": [
        "natural convection", "buoyancy", "Grashof number", "free convection",
        "hot plate rising air", "gas diffusivity", "Boussinesq", "candle plume",
    ],
    "thermal_contact_analysis": [
        "thermal contact conductance", "thermal contact resistance",
        "joint conductance", "interface conductance", "contact heat transfer",
        "pressed metal interface", "thermal interface", "TCR",
        "Cooper-Mikic-Yovanovich", "bolted joint heat transfer",
        "heat across a contact", "thermal joint resistance",
    ],
    "viscoelastic_creep_analysis": [
        "creep", "viscoelastic", "Maxwell model", "Kelvin-Voigt", "stress relaxation",
        "standard linear solid", "polymer creep", "time-dependent strain",
    ],
    "acoustic_interface_analysis": [
        "acoustic impedance", "sound reflection", "transmission coefficient",
        "Snell's law for sound", "critical angle sound", "ultrasound interface",
        "total internal reflection acoustics",
    ],

    # ============ DEVICES / QUANTUM SOLIDS (batch 7) ============
    "capacitor_analysis": [
        "capacitance", "parallel plate capacitor", "coaxial cable capacitance",
        "spherical capacitor", "energy stored in capacitor", "C = epsilon A / d",
    ],
    "hall_effect_analysis": [
        "Hall effect", "Hall voltage", "Lorentz force in conductor",
        "carrier sign", "magnetic field on current", "V_H = R_H I B / t",
    ],
    "semiconductor_junction_analysis": [
        "p-n junction", "diode", "depletion capacitance", "junction capacitance",
        "saturation current", "reverse current", "semiconductor diode",
    ],
    "superconducting_gap_analysis": [
        "superconducting gap", "BCS gap", "energy gap from Tc", "gap frequency",
        "2 Delta over h", "spectroscopic gap", "Cooper pair gap",
    ],
    "superconductor_critical_field_analysis": [
        "critical field", "upper critical field", "lower critical field",
        "Hc2", "Hc1", "Ginzburg-Landau parameter", "kappa parameter",
        "type I or type II superconductor", "Abrikosov vortex field",
        "superconducting magnet field",
    ],
    "quantum_tunneling_analysis": [
        "quantum tunneling", "WKB approximation", "tunneling probability",
        "barrier penetration", "transmission through barrier", "alpha decay tunneling",
    ],
    "quantum_box_energy_analysis": [
        "particle in a box", "infinite square well", "quantum well energy",
        "3D box energy levels", "quantum confinement", "nanoparticle energy levels",
    ],
    "band_dos_shape_analysis": [
        "density of states", "van Hove singularity", "tight binding DOS",
        "d-band filling", "pseudogap", "DOS at Fermi level",
    ],
    "magnetic_exchange_analysis": [
        "exchange coupling", "Heisenberg model", "antiferromagnet", "superexchange",
        "Goodenough-Kanamori", "spin Hamiltonian", "crystal field magnetism",
    ],

    # ============ PLASMA / EM / RELATIVITY / ATOMIC (batch 8) ============
    "plasma_parameters_analysis": [
        "Debye length", "Coulomb logarithm", "Larmor radius", "Debye number",
        "Spitzer resistivity", "plasma resistivity",
        "Debye shielding", "plasma parameter", "gyroradius", "fusion plasma",
        "resistivity of a plasma", "Spitzer-Harm", "eta parallel",
        "how conductive is a plasma",
    ],
    "electromagnetic_force_analysis": [
        "Coulomb's law", "Lorentz force", "magnetic force on charge", "qv cross B",
        "EM wave intensity", "Poynting flux", "energy density of light", "force between charges",
    ],
    "relativistic_energy_analysis": [
        "rest energy", "E=mc^2", "relativistic kinetic energy", "Lorentz factor",
        "energy momentum relation", "relativistic energy", "gamma factor energy",
    ],
    "zeeman_effect_analysis": [
        "Zeeman effect", "Zeeman splitting", "magnetic sublevels", "2j+1",
        "spectral line splitting magnetic field", "m_j states",
    ],

    # ============ TRIBOLOGY / MICROSTRUCTURE / MOLECULAR (batch 9) ============
    "friction_analysis": [
        "friction", "coefficient of friction", "Amontons law", "Bowden-Tabor",
        "adhesive friction", "ploughing", "sliding friction force",
    ],
    "wear_analysis": [
        "wear", "Archard wear", "wear rate", "material loss sliding",
        "abrasive wear", "adhesive wear", "wear volume",
    ],
    "wetting_analysis": [
        "contact angle", "wetting", "wettability", "hydrophobic", "hydrophilic",
        "Young equation", "Young-Dupre", "spreading coefficient",
        "water on glass", "water on Teflon", "mercury on glass", "beads up",
        "does it wet", "work of adhesion liquid",
    ],
    "dislocation_strengthening_analysis": [
        "work hardening", "Taylor hardening", "dislocation density", "flow stress",
        "strain hardening", "forest dislocations", "tau = alpha G b sqrt rho",
    ],
    "alloy_resistivity_analysis": [
        "alloy resistivity", "Nordheim rule", "residual resistivity", "solid solution",
        "Matthiessen", "impurity scattering", "brass cupronickel resistivity",
    ],
    "molecular_dipole_analysis": [
        "dipole moment", "molecular polarity", "bond dipole", "net dipole",
        "water dipole 1.85 D", "vector sum of dipoles", "polar molecule",
    ],
    "combustion_enthalpy_analysis": [
        "heat of combustion", "combustion enthalpy", "bond energy method",
        "Hess's law", "methane combustion", "enthalpy of reaction from bonds",
    ],

    # ============ CHEMISTRY: ACID-BASE / SOLUTION / ELECTRO / KINETICS (batch 10) ============
    "titration_analysis": [
        "titration", "pH at equivalence", "acid base titration", "buffer pH",
        "Henderson-Hasselbalch", "neutralization", "strong acid strong base",
    ],
    "acid_speciation_analysis": [
        "polyprotic acid", "alpha fraction", "speciation", "phosphoric acid species",
        "fraction of species at pH", "distribution diagram", "diprotic triprotic",
    ],
    "solution_analysis": [
        "dilution", "C1V1=C2V2", "mixing solutions", "concentration after mixing",
        "will it precipitate", "Ksp", "solubility product", "ion product",
    ],
    "electrochemistry_analysis": [
        "Tafel equation", "overpotential", "exchange current", "molar conductivity",
        "Kohlrausch", "solution conductivity", "electrolyte conductance",
    ],
    "reaction_kinetics_analysis": [
        "reaction rate", "half-life first order", "Arrhenius", "activation energy",
        "collision theory", "pre-exponential factor", "rate constant temperature",
    ],
    "radioactivity_analysis": [
        "radioactivity", "activity becquerel", "A = lambda N", "curie", "decay rate",
        "specific activity", "carbon-14 activity", "isotope activity",
    ],

    # ============ QUANTUM COMPUTING / QUANTUM (batch 11) ============
    "quantum_algorithm_analysis": [
        "Grover search", "quantum search", "QAOA", "max-cut", "Simon's algorithm",
        "quantum algorithm", "quantum speedup", "oracle", "amplitude amplification",
    ],
    "quantum_state_analysis": [
        "expectation value", "Bloch sphere", "Schmidt decomposition", "entanglement entropy",
        "Bell state", "qubit state", "Pauli expectation", "bipartite entanglement",
    ],
    "qubit_hardware_analysis": [
        "transmon", "qubit frequency", "coherence time", "T1 T2", "gate fidelity",
        "spin qubit", "NV center", "quantum dot qubit", "superconducting qubit",
    ],
    "interference_visibility_analysis": [
        "fringe visibility", "interference contrast", "double slit visibility",
        "Imax Imin", "coherence visibility", "interferometer contrast",
    ],

    # ============ ASSORTED PHYSICS (batch 12) ============
    "asteroid_analysis": [
        "asteroid", "surface gravity", "escape velocity small body", "Ceres Vesta",
        "Bennu Ryugu", "minor planet shape", "oblateness asteroid",
    ],
    "mobius_bimetallic_analysis": [
        "bimetallic strip", "Mobius loop", "thermocouple voltage", "Seebeck strip",
        "two-metal resistance", "bimetallic Seebeck",
    ],
    "hertzian_impact_analysis": [
        "coefficient of restitution", "Hertzian contact", "reduced modulus",
        "elastic impact", "bounce", "contact mechanics impact", "effective modulus",
    ],
    "holographic_dark_energy_analysis": [
        "holographic dark energy", "HDE", "dark energy density", "c squared parameter",
        "DESI dark energy", "Hubble radius cutoff", "rho dark energy",
    ],

    # ============ QUARKSUM INVENTORY (batch 13) ============
    "material_inventory_analysis": [
        "particle inventory", "how many protons", "quark count", "mass closure",
        "mass defect", "proton neutron electron count", "quarksum", "books balance",
    ],
    "constituent_behaviors_analysis": [
        "quark properties", "constituent mass", "particle behaviors", "bond summary",
        "QCD behavior", "what is in water", "molecular constituents",
    ],
    "planet_moment_of_inertia_analysis": [
        "moment of inertia factor", "C/MR^2", "polar moment planet", "core size",
        "planetary structure inertia", "differentiation", "Earth moment of inertia",
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
    "tnt_to_joules": [
        "megatons to joules", "TNT to joules", "energy of a megaton explosion",
        "how many joules in N tons of TNT", "convert TNT yield to energy",
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

    # ============ SIMULATION PLAYGROUND (conversation mode) ============
    "playground_load": [
        "load a", "start a scene", "simulate matter", "playground", "let's explore",
        "put a water molecule", "load bronze", "begin simulation", "scene",
    ],
    "playground_inspect": [
        "what's in it", "current state", "inspect", "show the bonds", "what is it now",
        "look at it", "tell me about the scene", "constituents",
    ],
    "playground_apply": [
        "heat it", "cool it", "raise the temperature", "apply pressure", "compress it",
        "apply a magnetic field", "ionize", "what if I change", "tune it", "now make it",
    ],
    "playground_simulate": [
        "drop it", "throw it", "launch it", "what happens if it falls", "orbit",
        "simulate the drop", "let it fall", "fire it",
    ],
    "playground_render": [
        "show me a render", "draw it", "visualize", "what does it look like",
        "picture of it", "render the scene",
    ],
    "playground_reset": ["reset", "start over", "undo all changes", "back to pristine"],
    "playground_status": ["what scenes are open", "session status", "list scenes"],
    "playground_clear": ["clear the scene", "drop the scene", "forget it"],
    "playground_make": [
        "make a", "make me a", "create a", "build a", "a ball of", "a cube of",
        "a sphere of", "brick", "marble", "cannonball", "ball of plutonium",
        "how big", "criticality", "critical mass",
    ],
    "request_clarification": [
        "ask the user", "I need to know", "clarify", "what value", "how big",
        "need more information", "can't guess", "which", "specify",
    ],

    # ============ FREE-SURFACE FLUID DYNAMICS (water sim) ============
    "buoyancy_analysis": [
        "will it float", "does it float or sink", "buoyancy", "Archimedes",
        "how deep does it sit", "submerged", "half sunk", "floats or sinks",
        "why does ice float", "displacement",
    ],
    "wind_wave_analysis": [
        "wind blowing across", "wind on water", "ripples", "waves on a pond",
        "wind-driven waves", "capillary waves", "surface waves", "wind stress",
        "breeze ruffling the water", "how fast do ripples move", "wavelength of ripples",
    ],

    # ============ COVERAGE SWEEP: previously zero-keyword tools ============
    "projectile_flight_time": [
        "time of flight", "how long is it in the air", "hang time",
        "flight time of a projectile",
    ],
    "friction_stopping_distance": [
        "stopping distance", "how far to stop with friction", "skid distance",
        "distance to stop sliding",
    ],
    "eta_desi_band_check": [
        "DESI band check", "within 1 sigma", "eta consistency check",
    ],
    "thin_lens_focal_length": [
        "focal length", "1/f = 1/do + 1/di", "lens equation solve for f",
        "what's the focal length",
    ],
    "power_dissipation_resistor": [
        "power dissipated in a resistor", "P = I^2 R", "heat in a resistor",
        "resistor power loss",
    ],
    "hydrogen_emission_wavelength": [
        "hydrogen spectral line", "Rydberg transition", "Balmer series",
        "Lyman series", "hydrogen emission line wavelength",
    ],
    "procedure_black_hole_profile": [
        "full black hole profile", "black hole thermodynamics cascade",
        "everything about this black hole", "horizon temperature entropy evaporation",
    ],
    "procedure_photon_spectrum": [
        "everything about this photon", "photon cascade", "wavelength frequency energy momentum",
    ],
    "procedure_relativistic_particle": [
        "relativistic particle cascade", "gamma factor and momentum from kinetic energy",
        "everything about this relativistic particle",
    ],
    "procedure_projectile_trajectory": [
        "full projectile trajectory", "range height and time of flight",
        "everything about this projectile",
    ],
    "procedure_stellar_blackbody": [
        "star as a blackbody", "Wien peak and Stefan-Boltzmann flux",
        "everything about this star's radiation",
    ],
    "run_simulation": [
        "run this simulation verb directly", "call a simulation verb with params",
    ],
    "list_simulation_scenarios": [
        "what simulations are available", "list simulation verbs",
        "available scenario types",
    ],
    "bond_energy": [
        "bond dissociation energy", "how strong is this bond", "bond strength kJ/mol",
    ],
    "bond_angle": [
        "VSEPR angle", "molecular geometry angle", "bond angle from steric number",
        "tetrahedral angle", "109.5 degrees",
    ],
    "reaction_enthalpy": [
        "heat of reaction", "delta H of reaction", "standard enthalpy of reaction",
        "is this reaction exothermic",
    ],
    "weak_acid_ph": [
        "pH of a weak acid", "pH from Ka", "acid dissociation pH",
    ],
    "buffer_ph": [
        "buffer pH", "Henderson-Hasselbalch", "pH of a buffer solution",
    ],
    "cell_potential": [
        "galvanic cell voltage", "EMF of a battery", "cell potential",
        "Daniell cell voltage", "Nernst equation cell",
    ],
    "electrolysis_mass": [
        "mass deposited by electrolysis", "Faraday's law of electrolysis",
        "electroplating mass",
    ],
    "boiling_point_elevation": [
        "boiling point elevation", "how much does salt raise the boiling point",
        "colligative boiling point",
    ],
    "freezing_point_depression": [
        "freezing point depression", "how much does salt lower the freezing point",
        "colligative freezing point", "why do we salt roads",
    ],
    "osmotic_pressure": [
        "osmotic pressure", "van't Hoff pressure", "pressure across a membrane",
    ],
    "molar_solubility": [
        "molar solubility", "solubility from Ksp", "how much dissolves",
    ],
    "electrical_resistivity": [
        "resistivity of a metal", "electrical resistivity", "how resistive is copper",
    ],
    "carrier_mobility": [
        "electron drift mobility", "carrier mobility in a metal",
    ],
    "hall_coefficient": [
        "Hall coefficient", "Hall effect constant",
    ],
    "electron_mean_free_path": [
        "electron mean free path", "how far does an electron travel before scattering",
    ],
    "free_electron_density": [
        "free electron density", "conduction electron density", "electron number density",
    ],
    "semiconductor_band_gap": [
        "band gap of a semiconductor", "silicon band gap at temperature",
        "Varshni band gap",
    ],
    "intrinsic_carrier_density": [
        "intrinsic carrier density", "n_i of a semiconductor",
    ],
    "pn_built_in_voltage": [
        "built-in voltage of a diode", "p-n junction voltage", "junction built-in potential",
    ],
    "depletion_width": [
        "depletion region width", "depletion width of a p-n junction",
    ],
    "diode_current": [
        "Shockley diode equation", "diode current from voltage", "I-V curve of a diode",
    ],
}


# ── Colloquial / layman supplement ───────────────────────────────────────────
# The entries above are FORMAL (formula names, theorems, attributions). This adds
# how a NORMAL HUMAN describes the same phenomenon, so a plain-English scenario
# routes without knowing the textbook term. Merged into TOOL_KEYWORDS at import.
# Add generously -- false positives are cheap, missed matches are expensive.
_COLLOQUIAL: dict[str, list[str]] = {
    # ---- motion / falling / throwing ----
    "free_fall_time": ["how long to fall", "how long does it take to drop",
                       "time to hit the ground", "how long before it lands",
                       "drop it from", "if I let go how long"],
    "free_fall_velocity": ["how fast when it lands", "how fast does it hit",
                           "how fast falling", "speed when it hits the ground",
                           "how fast at the bottom"],
    "projectile_range": ["how far will it go", "how far can I throw it",
                         "how far does it fly", "throw distance", "how far will it land",
                         "how far does the cannon shoot"],
    "projectile_max_height": ["how high will it go", "how high does it fly",
                              "top of the arc", "peak height", "highest point"],
    "projectile_analysis": ["throw a ball", "fire a cannon", "shoot it at an angle",
                            "launch at an angle", "kick a ball", "lob it"],
    "kinetic_energy": ["energy of a moving thing", "how much energy is it carrying",
                       "energy of motion", "how much punch"],
    "momentum": ["how much oomph", "how hard to stop it", "moving mass"],
    "collision_analysis": ["two things crash", "what happens when they collide",
                           "bumper cars", "head-on crash", "smash together", "bounce off each other"],
    "work_energy_analysis": ["how much work", "energy to push it", "push it up a hill",
                             "how much effort to move"],
    "incline_analysis": ["slide down a ramp", "push it up a slope", "ramp", "hill",
                         "will it slide down", "steep slope"],
    "rotational_dynamics": ["spinning", "how fast it spins", "spin it up",
                            "twirl", "how hard to spin"],
    "hertzian_impact_analysis": ["how much does it bounce", "bounciness",
                                 "two balls hit", "how bouncy"],
    "escape_velocity": ["how fast to escape gravity", "speed to leave the planet",
                        "how fast to fly off into space", "break free of gravity"],
    "gravitational_potential_energy": ["energy to lift it", "energy to raise it up"],
    # ---- space / planets / orbits ----
    "orbital_velocity": ["how fast does it orbit", "how fast does the moon go around",
                         "how fast is the ISS", "orbit speed"],
    "orbital_period": ["how long is a year on", "how long to go around",
                       "how long to orbit"],
    "gravitational_force": ["pull between two things", "how strong is gravity between",
                            "how hard do they attract"],
    "solar_system_body": ["tell me about mars", "facts about jupiter", "how big is earth",
                          "planet info", "how heavy is the sun",
                          "surface gravity of", "how strong is gravity on",
                          "gravity on the surface of", "how much would I weigh on",
                          "surface temperature of the sun", "how hot is the sun",
                          "temperature of the sun", "how hot is a planet"],
    "asteroid_analysis": ["jump off an asteroid", "gravity on an asteroid",
                          "how heavy on ceres", "tiny world gravity"],
    "light_travel_time": ["how long for light to get there", "how far in light years",
                          "communication delay", "how long does light take"],
    "planet_moment_of_inertia_analysis": ["what's the core like", "how is it layered inside",
                                          "is it differentiated"],
    # ---- materials / stuff ----
    "density": ["how heavy for its size", "how dense", "will it be heavy",
                "heavy or light", "mass per volume"],
    "melting_point": ["when does it melt", "how hot to melt it", "melts at"],
    "boiling_point": ["when does it boil", "how hot to boil"],
    "youngs_modulus": ["how stiff", "how springy", "how much does it stretch", "how rigid"],
    "elastic_analysis": ["will it stretch", "how much does it bend", "springiness",
                         "stretch and squish"],
    "stress_failure_analysis": ["when does it break", "will it snap", "how much before it breaks",
                                "how strong before it fails", "will it crack", "fatigue"],
    "friction_analysis": ["how slippery", "how much grip", "will it slide", "how much friction",
                          "does it skid"],
    "wear_analysis": ["how fast does it wear out", "wears down", "how long will it last rubbing"],
    "wetting_analysis": ["does it wet the surface", "beads up", "spreads out",
                         "does water stick", "is it waterproof", "contact angle"],
    "material_inventory_analysis": ["what's it made of", "how many atoms", "how many protons",
                                    "what's inside it", "how much does it weigh atom by atom"],
    "list_materials": ["what materials do you have", "what stuff can I use"],
    # ---- heat / temperature / fluids ----
    "viscous_flow_analysis": ["how thick is the fluid", "how does it flow", "flow in a pipe",
                              "honey vs water", "how runny"],
    "natural_convection_analysis": ["hot air rises", "warm air rising", "heat plume",
                                    "convection", "candle flame air"],
    "diffusion_analysis": ["how fast does it spread", "smell spreading across a room",
                           "how fast does it mix"],
    "carnot_efficiency": ["best possible engine efficiency", "how efficient can an engine be"],
    "speed_of_sound_in_ideal_gas": ["how fast is sound", "speed of sound in air"],
    "blackbody_peak_wavelength": ["what color does it glow when hot", "color of hot metal",
                                  "red hot or white hot", "what color when heated"],
    "blackbody_total_power": ["how much heat does it radiate", "how much does it glow"],
    "thermal_contact_analysis": ["heat through a joint", "how well heat passes between two surfaces"],
    "thermoelectric_generator_analysis": ["make electricity from heat", "heat to power",
                                          "thermocouple"],
    # ---- light / electricity / magnets ----
    "electrical_power": ["how much power", "how many watts", "power used"],
    "capacitor_analysis": ["store charge", "how much charge can it hold", "capacitor"],
    "hall_effect_analysis": ["voltage in a magnetic field", "magnet on a current"],
    "material_color_analysis": ["what color is", "color of gold", "what color is copper",
                                "what does it look like color"],
    "phosphor_decay_analysis": ["glow in the dark", "afterglow", "how long does it keep glowing"],
    "snells_law_refraction_angle": ["light bending", "straw looks bent in water", "bends the light"],
    "thin_lens_image_distance": ["magnifying glass", "camera lens", "where's the image"],
    "electromagnetic_force_analysis": ["force on a charge", "magnet pushes the charge",
                                       "static electricity force"],
    # ---- energy / nuclear / bangs ----
    "mass_to_energy": ["how much energy in matter", "E=mc2", "energy locked in mass"],
    "joules_to_TNT": ["how big a bang", "how many tons of TNT", "explosion size",
                      "how big an explosion"],
    "nuclear_binding_energy": ["what holds the nucleus together", "nuclear glue"],
    "fission_energy_release": ["how much bang from a kg of uranium",
                               "energy from splitting atoms", "atom bomb yield from fuel mass"],
    "radioactivity_analysis": ["how radioactive", "how hot is the radiation", "how fast does it decay",
                               "is it dangerous radioactive"],
    # ---- relativity / black holes ----
    "lorentz_factor": ["near the speed of light", "time slows down", "how much faster time"],
    "relativistic_time_dilation": ["time slows at high speed", "twins paradox", "clock runs slow"],
    "schwarzschild_radius": ["how small to be a black hole", "black hole size", "crush it to a black hole"],
    "hawking_temperature": ["does a black hole glow", "black hole temperature"],
    "gravitational_time_dilation": ["time near a black hole", "clocks slow in gravity"],
    # ---- quantum ----
    "quantum_tunneling_analysis": ["go through a wall", "tunnel through a barrier",
                                   "leak through", "pass through when it shouldn't"],
    "quantum_box_energy_analysis": ["particle in a box", "tiny box energy levels", "quantum dot"],
    "quantum_algorithm_analysis": ["quantum computer", "quantum search", "grover", "quantum speedup"],
    "de_broglie_wavelength": ["wavelength of a particle", "matter wave", "how wavy is an electron"],
    # ---- chemistry ----
    "titration_analysis": ["ph of the mix", "how acidic", "acid and base", "neutralize"],
    "solution_analysis": ["mix two solutions", "will it dissolve", "dilute it",
                          "will it precipitate", "cloudy when mixed"],
    "combustion_enthalpy_analysis": ["how much heat from burning", "burn it", "set it on fire energy"],
    "reaction_kinetics_analysis": ["how fast does it react", "reaction speed", "half life of the reaction"],
    # ---- the playground (describing/building/poking a scene) ----
    "playground_load": ["set up a", "start with a", "put a", "drop in a", "let's play with"],
    "playground_make": ["make me a", "build a", "create a", "a ball of", "a cube of",
                        "a block of", "a sphere of", "how big should it be"],
    "playground_apply": ["heat it up", "cool it down", "squeeze it", "crush it", "zap it",
                         "magnetize it", "put it under pressure", "warm it", "chill it"],
    "playground_simulate": ["drop it", "throw it", "let it fall", "what happens if",
                            "fire it", "launch it", "let it go"],
    "playground_render": ["show me", "what does it look like", "draw it", "picture of it",
                          "let me see it", "render it"],
    # ---- water-scene vocabulary (routes scene/fluid talk to the playground/sim) ----
    "simulate": ["pond", "puddle", "wind blowing across", "ripples", "waves on the water",
                 "splash", "floating", "sinks or floats", "half sunk", "water surface"],
}

for _tool, _terms in _COLLOQUIAL.items():
    TOOL_KEYWORDS.setdefault(_tool, []).extend(_terms)
