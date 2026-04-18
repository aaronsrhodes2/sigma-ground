"""
Pre-built weapon and armor presets — from flint arrowhead to sigma-field disruptor.
"""

# ── Armor presets ─────────────────────────────────────────────────────────────
ARMOR_PRESETS = {

    # ── DEFAULT ──────────────────────────────────────────────────────────────
    'default': {
        'name': 'Flint Arrow Target (7-layer leather)',
        'layers': [
            {'material': 'bone',      'thickness_m': 0.002},   # hardened bone plate
            {'material': 'kevlar',    'thickness_m': 0.003},   # linen (kevlar proxy)
            {'material': 'rubber',    'thickness_m': 0.005},   # padding
            {'material': 'bone',      'thickness_m': 0.002},
            {'material': 'kevlar',    'thickness_m': 0.003},
            {'material': 'rubber',    'thickness_m': 0.005},
            {'material': 'bone',      'thickness_m': 0.002},
        ],
    },

    # ── HISTORICAL ───────────────────────────────────────────────────────────
    'roman_lorica': {
        'name': 'Roman Lorica Segmentata',
        'layers': [
            {'material': 'iron',      'thickness_m': 0.002},
            {'material': 'kevlar',    'thickness_m': 0.005},   # linen padding
            {'material': 'iron',      'thickness_m': 0.002},
        ],
    },
    'medieval_plate': {
        'name': 'Medieval Full Plate (1400s)',
        'layers': [
            {'material': 'steel_mild','thickness_m': 0.005},
            {'material': 'kevlar',    'thickness_m': 0.010},   # gambeson padding
        ],
    },
    'wwii_helmet': {
        'name': 'WWII Steel Helmet (M1)',
        'layers': [
            {'material': 'steel_mild','thickness_m': 0.0025},
        ],
    },

    # ── MODERN BALLISTIC ─────────────────────────────────────────────────────
    'nij_iiia': {
        'name': 'NIJ Level IIIA Body Armor',
        'layers': [
            {'material': 'kevlar',    'thickness_m': 0.012},
            {'material': 'rubber',    'thickness_m': 0.003},
        ],
    },
    'nij_iv': {
        'name': 'NIJ Level IV Plate (SAPI)',
        'layers': [
            {'material': 'ceramic_alumina', 'thickness_m': 0.015},
            {'material': 'kevlar',          'thickness_m': 0.010},
            {'material': 'steel_mild',      'thickness_m': 0.003},
        ],
    },
    'tank_composite': {
        'name': 'Modern Tank Composite (Chobham-style)',
        'layers': [
            {'material': 'steel_mild',      'thickness_m': 0.050},
            {'gap': True,                   'thickness_m': 0.020},
            {'material': 'ceramic_alumina', 'thickness_m': 0.040},
            {'material': 'steel_mild',      'thickness_m': 0.030},
            {'material': 'rubber',          'thickness_m': 0.010},
            {'material': 'steel_mild',      'thickness_m': 0.020},
        ],
    },
    'reactive_armor': {
        'name': 'ERA Reactive Armor',
        'layers': [
            {'material': 'steel_mild',  'thickness_m': 0.006},
            {'material': 'plastic_abs', 'thickness_m': 0.010},  # explosive layer proxy
            {'material': 'steel_mild',  'thickness_m': 0.006},
            {'material': 'steel_mild',  'thickness_m': 0.060},  # main hull
        ],
    },
    'bunker': {
        'name': 'Concrete Bunker Wall',
        'layers': [
            {'material': 'concrete',   'thickness_m': 1.000},
            {'material': 'steel_mild', 'thickness_m': 0.020},
        ],
    },

    # ── NAVAL / MARITIME ─────────────────────────────────────────────────────
    'wwii_battleship': {
        'name': 'WWII Battleship Belt (Iowa class)',
        'layers': [
            {'material': 'steel_mild', 'thickness_m': 0.307},
            {'material': 'steel_mild', 'thickness_m': 0.019},
        ],
    },

    # ── AEROSPACE ────────────────────────────────────────────────────────────
    'aircraft_fuselage': {
        'name': 'Aircraft Fuselage (aluminum)',
        'layers': [
            {'material': 'aluminum',     'thickness_m': 0.003},
            {'material': 'carbon_fiber', 'thickness_m': 0.005},
        ],
    },
    'spacecraft_hull': {
        'name': 'Deep-Space Hull (titanium + Whipple shield)',
        'layers': [
            {'material': 'aluminum',   'thickness_m': 0.003},  # bumper shield
            {'gap': True,              'thickness_m': 0.100},  # Whipple gap
            {'material': 'titanium',   'thickness_m': 0.010},
            {'material': 'kevlar',     'thickness_m': 0.005},
        ],
    },

    # ── FUTURISTIC / THEORETICAL ─────────────────────────────────────────────
    'sci_fi_light': {
        'name': 'Sci-Fi Light Combat Suit',
        'layers': [
            {'material': 'titanium',     'thickness_m': 0.005},
            {'material': 'carbon_fiber', 'thickness_m': 0.020},
            {'material': 'ceramic_alumina', 'thickness_m': 0.010},
        ],
    },
    'sci_fi_heavy': {
        'name': 'Sci-Fi Heavy Battle-Armor',
        'layers': [
            {'material': 'tungsten',        'thickness_m': 0.030},
            {'material': 'ceramic_alumina', 'thickness_m': 0.050},
            {'gap': True,                   'thickness_m': 0.050},
            {'material': 'depleted_uranium','thickness_m': 0.025},
            {'material': 'carbon_fiber',    'thickness_m': 0.020},
            {'material': 'titanium',        'thickness_m': 0.010},
        ],
    },
    'plasma_shielded': {
        'name': 'Plasma-Cooled Shell (ablative outer)',
        'layers': [
            {'material': 'carbon_fiber',    'thickness_m': 0.030},  # ablative
            {'material': 'ceramic_alumina', 'thickness_m': 0.040},
            {'material': 'tungsten',        'thickness_m': 0.020},
            {'material': 'titanium',        'thickness_m': 0.015},
        ],
    },
    'exotic_sigma': {
        'name': 'Sigma-Field Hardened Composite',
        'layers': [
            {'material': 'tungsten',        'thickness_m': 0.050},
            {'material': 'depleted_uranium','thickness_m': 0.050},
            {'material': 'tungsten',        'thickness_m': 0.050},
            {'material': 'ceramic_alumina', 'thickness_m': 0.030},
        ],
    },
}


# ── Weapon presets ────────────────────────────────────────────────────────────
C_LIGHT = 299_792_458.0

WEAPON_PRESETS = {

    # ── DEFAULT ──────────────────────────────────────────────────────────────
    'default': {
        'name': 'Flint Arrowhead',
        'weapon': {
            'type': 'kinetic',
            'mass_kg': 0.0025,
            'velocity_ms': 55.0,
            'radius_m': 0.003,
            'material': 'stone_proxy',  # closest: granite
        },
        '_mat_override': 'granite',
    },

    # ── HISTORICAL ───────────────────────────────────────────────────────────
    'throwing_spear': {
        'name': 'Throwing Spear (bronze)',
        'weapon': {'type':'kinetic','mass_kg':0.8,'velocity_ms':35,'radius_m':0.012,'material':'iron'},
    },
    'crossbow_bolt': {
        'name': 'Crossbow Bolt (steel tip)',
        'weapon': {'type':'kinetic','mass_kg':0.075,'velocity_ms':80,'radius_m':0.006,'material':'steel_mild'},
    },
    'cannonball': {
        'name': 'Cast Iron Cannonball (1800s)',
        'weapon': {'type':'kinetic','mass_kg':5.5,'velocity_ms':400,'radius_m':0.060,'material':'iron'},
    },

    # ── MODERN FIREARMS ──────────────────────────────────────────────────────
    'pistol_9mm': {
        'name': '9mm Pistol Round',
        'weapon': {'type':'kinetic','mass_kg':0.008,'velocity_ms':370,'radius_m':0.0045,'material':'lead'},
    },
    'ak47': {
        'name': 'AK-47 (7.62x39mm)',
        'weapon': {'type':'kinetic','mass_kg':0.0079,'velocity_ms':715,'radius_m':0.0038,'material':'steel_mild'},
    },
    'm2_browning': {
        'name': '.50 Cal M2 Browning',
        'weapon': {'type':'kinetic','mass_kg':0.046,'velocity_ms':895,'radius_m':0.0063,'material':'steel_mild'},
    },
    'armor_piercing_30mm': {
        'name': '30mm APDS (autocannon)',
        'weapon': {'type':'kinetic','mass_kg':0.35,'velocity_ms':1100,'radius_m':0.015,'material':'tungsten'},
    },

    # ── ANTI-ARMOR ───────────────────────────────────────────────────────────
    'rpg7': {
        'name': 'RPG-7 (shaped charge)',
        'weapon': {'type':'explosive','tnt_kg':0.35,'standoff_m':0,'shaped_charge':True},
    },
    'atgm': {
        'name': 'Anti-Tank Missile (Tandem HEAT)',
        'weapon': {'type':'explosive','tnt_kg':4.5,'standoff_m':0,'shaped_charge':True},
    },
    'sabot_apfsds': {
        'name': 'Tank APFSDS (depleted uranium)',
        'weapon': {'type':'kinetic','mass_kg':4.5,'velocity_ms':1700,'radius_m':0.020,'material':'depleted_uranium'},
    },

    # ── ENERGY WEAPONS ───────────────────────────────────────────────────────
    'laser_designator': {
        'name': 'Military Laser Designator (1kW)',
        'weapon': {'type':'laser','power_w':1e3,'wavelength_nm':1064,'pulse_duration_s':1.0,'spot_radius_m':0.01},
    },
    'hel_100kw': {
        'name': 'High-Energy Laser (100 kW, HELIOS-class)',
        'weapon': {'type':'laser','power_w':1e5,'wavelength_nm':1064,'pulse_duration_s':3.0,'spot_radius_m':0.008},
    },
    'hel_megawatt': {
        'name': 'Megawatt Laser (near-future)',
        'weapon': {'type':'laser','power_w':1e6,'wavelength_nm':532,'pulse_duration_s':5.0,'spot_radius_m':0.005},
    },

    # ── FUTURISTIC ───────────────────────────────────────────────────────────
    'railgun_hypersonic': {
        'name': 'Railgun (Mach 8, tungsten slug)',
        'weapon': {'type':'kinetic','mass_kg':3.2,'velocity_ms':2700,'radius_m':0.025,'material':'tungsten'},
    },
    'railgun_relativistic': {
        'name': 'Relativistic Railgun (0.1c)',
        'weapon': {'type':'kinetic','mass_kg':1.0,'velocity_ms':C_LIGHT*0.1,'radius_m':0.020,'material':'tungsten'},
    },
    'plasma_cannon': {
        'name': 'Plasma Cannon (sci-fi, 1keV)',
        'weapon': {'type':'plasma','temperature_ev':1000,'pressure_pa':1e8,'duration_s':0.2,'contact_radius_m':0.05},
    },
    'plasma_torch_industrial': {
        'name': 'Industrial Plasma Torch (100eV)',
        'weapon': {'type':'plasma','temperature_ev':100,'pressure_pa':1e5,'duration_s':2.0,'contact_radius_m':0.03},
    },
    'proton_cannon': {
        'name': 'Proton Beam Cannon (1 GeV)',
        'weapon': {'type':'particle','energy_mev':1000,'current_a':1e-4,'duration_s':1.0,'particle':'proton','beam_radius_m':0.005},
    },
    'neutron_bomb': {
        'name': 'Enhanced Neutron Pulse (200 MeV)',
        'weapon': {'type':'particle','energy_mev':200,'current_a':1e-3,'duration_s':0.5,'particle':'neutron','beam_radius_m':0.050},
    },
    'gamma_laser': {
        'name': 'Gamma-Ray Laser (graser)',
        'weapon': {'type':'laser','power_w':1e9,'wavelength_nm':0.001,'pulse_duration_s':0.001,'spot_radius_m':0.001},
    },
    'tidal_disruptor': {
        'name': 'Tidal Disruptor (white-dwarf gradient)',
        'weapon': {'type':'gravitational','tidal_gradient_ms2':1e7,'sigma_spike':0.0},
    },
    'sigma_spike_weapon': {
        'name': 'Sigma-Field Disruptor (sigma-combat exclusive)',
        'weapon': {'type':'gravitational','tidal_gradient_ms2':0,'sigma_spike':1.6},
    },
    'city_buster': {
        'name': 'Thermonuclear (1 Mt)',
        'weapon': {'type':'explosive','tnt_kg':1e9,'standoff_m':500},
    },

    # ── SCI-FI FAVOURITES ────────────────────────────────────────────────────
    'phaser_stun': {
        'name': 'Phaser (stun setting, low-power laser)',
        'weapon': {'type':'laser','power_w':500,'wavelength_nm':600,'pulse_duration_s':0.05,'spot_radius_m':0.003},
    },
    'photon_torpedo': {
        'name': 'Photon Torpedo (antimatter proxy: 10 Mt)',
        'weapon': {'type':'explosive','tnt_kg':1e10,'standoff_m':50},
    },
}


def get_armor(key: str = 'default') -> list:
    return list(ARMOR_PRESETS.get(key, ARMOR_PRESETS['default'])['layers'])


def get_weapon(key: str = 'default') -> dict:
    p = WEAPON_PRESETS.get(key, WEAPON_PRESETS['default'])
    w = dict(p['weapon'])
    # Patch material override for stone (flint arrow)
    if '_mat_override' in p:
        w['material'] = p['_mat_override']
    return w


def armor_menu() -> list:
    """Returns list of (key, display_name) tuples for UI menus."""
    return [(k, v['name']) for k, v in ARMOR_PRESETS.items()]


def weapon_menu() -> list:
    return [(k, v['name']) for k, v in WEAPON_PRESETS.items()]
