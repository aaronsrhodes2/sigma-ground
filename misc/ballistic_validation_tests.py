"""
ballistic_validation_tests.py
Authoritative real-world ballistic penetration data for physics simulator validation.

All data is drawn from publicly available, authoritative sources:
  - NIJ Standard-0101.06 (NIST/OJP publication 223054)
  - NATO STANAG 4569 / AEP-55 Volume 1
  - Wikipedia articles citing Jane's, Bird & Livingston (2001), and manufacturer specs
  - GlobalSecurity.org
  - Wikipedia articles on specific weapon systems with traceable citations
  - Published academic/military technical data

Penetration figures marked ESTIMATED are derived from open-source calculations
(e.g. Lanz-Odermatt formula) or secondary analyses, not direct declassified test data.
Penetration figures for modern composite/Chobham armor are RHA-equivalent estimates
and are broadly consistent with open literature; exact values remain classified.

Units throughout:
  mass_kg      -- projectile or warhead mass in kilograms
  velocity_ms  -- impact velocity at test distance, metres per second
  radius_m     -- projectile body radius in metres
  thickness_m  -- armor / target thickness in metres
  penetration_m -- RHA-equivalent or concrete penetration depth in metres

'expected_breach': True  = test threat is expected to defeat the stated armor
'expected_breach': False = test threat should be stopped
"""

# ---------------------------------------------------------------------------
# CATEGORY 1 — Small arms vs. body armor  (NIJ 0101.06 and STANAG 4569)
# ---------------------------------------------------------------------------
# Source for NIJ levels: NIJ Standard-0101.06, Table 1.
#   Published by NIST/OJP as document 223054.
#   URL: https://www.ojp.gov/pdffiles1/nij/223054.pdf
#   All test velocities are at the muzzle (chrono screen distance < 2 m).
#   Tolerance on every velocity: ±9.1 m/s (±30 ft/s).
#   Pass criterion: ZERO penetrations in 6 shots per panel, two panels.
#
# Source for STANAG 4569: NATO AEP-55 Vol.1.
#   Public summary: https://en.wikipedia.org/wiki/STANAG_4569
#   All STANAG tests at 30 m stand-off; tolerance ±20 m/s.

VALIDATION_TESTS = [

    # -----------------------------------------------------------------------
    # NIJ Level IIA — Soft body armor (light pistol threats)
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_IIA_9mm',
        'label': '9 mm FMJ RN vs NIJ Level IIA soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '9x19mm Parabellum',
            'projectile': 'FMJ Round Nose',
            'mass_kg': 0.008,          # 8.0 g / 124 gr
            'velocity_ms': 373.0,      # NIJ test velocity (±9.1 m/s)
            'radius_m': 0.00455,       # 9.1 mm / 2 diameter = 4.55 mm radius
            'material': 'lead_fmj',
        },
        'armor': [
            {'material': 'UHMWPE_or_Kevlar_soft', 'thickness_m': 0.008},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1 (OJP pub 223054)',
        'notes': 'Level IIA minimum; tested new and conditioned. '
                 'Represents everyday-carry threat for law enforcement.',
    },
    {
        'id': 'NIJ_IIA_40SW',
        'label': '.40 S&W FMJ vs NIJ Level IIA soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '.40 S&W',
            'projectile': 'FMJ',
            'mass_kg': 0.01170,        # 11.7 g / 180 gr
            'velocity_ms': 352.0,      # NIJ test velocity (±9.1 m/s)
            'radius_m': 0.005,         # 10.2 mm / 2
            'material': 'lead_fmj',
        },
        'armor': [
            {'material': 'UHMWPE_or_Kevlar_soft', 'thickness_m': 0.008},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'Second test threat for Level IIA certification.',
    },

    # -----------------------------------------------------------------------
    # NIJ Level II — Soft body armor (standard pistol threats)
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_II_9mm',
        'label': '9 mm FMJ RN vs NIJ Level II soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '9x19mm Parabellum',
            'projectile': 'FMJ Round Nose',
            'mass_kg': 0.008,
            'velocity_ms': 398.0,
            'radius_m': 0.00455,
            'material': 'lead_fmj',
        },
        'armor': [
            {'material': 'Kevlar_soft', 'thickness_m': 0.010},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'Higher velocity than IIA; represents +P pistol loads.',
    },
    {
        'id': 'NIJ_II_357mag',
        'label': '.357 Magnum JSP vs NIJ Level II soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '.357 Magnum',
            'projectile': 'Jacketed Soft Point',
            'mass_kg': 0.01020,        # 10.2 g / 158 gr
            'velocity_ms': 408.0,      # NIJ table value (some sources give 436 m/s;
                                        # 408 m/s is the 0101.06 conditioned-test value)
            'radius_m': 0.00455,
            'material': 'lead_jsp',
        },
        'armor': [
            {'material': 'Kevlar_soft', 'thickness_m': 0.010},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'JSP expands; FMJ at same velocity would be more penetrating. '
                 'Velocity in some printings listed as 436 m/s unconditioned.',
    },

    # -----------------------------------------------------------------------
    # NIJ Level IIIA — Soft armor (magnum pistol / submachine-gun threats)
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_IIIA_357SIG',
        'label': '.357 SIG FMJ FN vs NIJ Level IIIA soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '.357 SIG',
            'projectile': 'FMJ Flat Nose',
            'mass_kg': 0.00810,        # 8.1 g / 125 gr
            'velocity_ms': 448.0,
            'radius_m': 0.00455,
            'material': 'lead_fmj',
        },
        'armor': [
            {'material': 'Kevlar_soft', 'thickness_m': 0.012},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'Highest NIJ pistol threat stopped by soft armor.',
    },
    {
        'id': 'NIJ_IIIA_44mag',
        'label': '.44 Magnum SJHP vs NIJ Level IIIA soft armor',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '.44 Magnum',
            'projectile': 'Semi-Jacketed Hollow Point',
            'mass_kg': 0.01500,        # 15.0 g / 240 gr (some sources 15.6 g)
            'velocity_ms': 430.0,
            'radius_m': 0.00545,       # .429 cal = 10.9 mm diameter
            'material': 'lead_sjhp',
        },
        'armor': [
            {'material': 'Kevlar_soft', 'thickness_m': 0.012},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'Second threat for Level IIIA. Hollow-point expands; '
                 'flat-nose FMJ at similar energy is the harder test.',
    },

    # -----------------------------------------------------------------------
    # NIJ Level III — Hard plate (rifle threats)
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_III_762M80',
        'label': '7.62×51 NATO M80 FMJ vs NIJ Level III hard plate',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '7.62x51mm NATO',
            'projectile': 'M80 FMJ Steel Jacket',
            'mass_kg': 0.00960,        # 9.6 g / 147 gr
            'velocity_ms': 847.0,
            'radius_m': 0.00381,       # 7.62 mm / 2
            'material': 'steel_jacket_lead_core',
        },
        'armor': [
            {'material': 'steel_or_ceramic_hard_plate', 'thickness_m': 0.010},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'FMJ ball (non-AP). Level III does NOT certify against '
                 'steel-core M855A1 or M80A1.',
    },

    # -----------------------------------------------------------------------
    # NIJ Level IV — Hard plate (armor-piercing rifle threat)
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_IV_30cal_AP',
        'label': '.30-06 M2 AP vs NIJ Level IV hard plate',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '.30-06 Springfield',
            'projectile': 'M2 AP (tungsten carbide core)',
            'mass_kg': 0.01080,        # 10.8 g / 166 gr
            'velocity_ms': 878.0,
            'radius_m': 0.00381,
            'material': 'tungsten_carbide_core',
        },
        'armor': [
            {'material': 'ceramic_boron_carbide_or_alumina', 'thickness_m': 0.015},
            {'material': 'UHMWPE_backing', 'thickness_m': 0.008},
        ],
        'expected_breach': False,
        'source': 'NIJ Standard-0101.06, Table 1',
        'notes': 'Hardest NIJ soft-armor / plate standard. Single-hit test only.',
    },

    # -----------------------------------------------------------------------
    # STANAG 4569 vehicle armor levels
    # Source: NATO AEP-55 Vol.1; public summary Wikipedia / Plasan
    # Tests at 30 m stand-off, velocity tolerance ±20 m/s
    # -----------------------------------------------------------------------
    {
        'id': 'STANAG_L1_762NATO',
        'label': '7.62×51 NATO M80 ball vs STANAG 4569 Level 1 vehicle armor',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '7.62x51mm NATO',
            'projectile': 'M80 Ball',
            'mass_kg': 0.00960,
            'velocity_ms': 833.0,      # at 30 m
            'radius_m': 0.00381,
            'material': 'steel_jacket_lead_core',
        },
        'armor': [
            {'material': 'STANAG_Level1_steel_composite', 'thickness_m': 0.006},
        ],
        'expected_breach': False,
        'source': 'NATO STANAG 4569 / AEP-55 Vol.1 Edition C',
        'notes': 'Baseline vehicle armor for light logistic vehicles. '
                 '5.56×45 SS109 at 900 m/s is a co-test threat.',
    },
    {
        'id': 'STANAG_L2_762x39_API',
        'label': '7.62×39 API BZ vs STANAG 4569 Level 2 vehicle armor',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '7.62x39mm',
            'projectile': 'API BZ (mild-steel core)',
            'mass_kg': 0.00780,        # ~7.8 g — API BZ nominal
            'velocity_ms': 695.0,
            'radius_m': 0.00381,
            'material': 'mild_steel_core',
        },
        'armor': [
            {'material': 'STANAG_Level2_RHA_equivalent', 'thickness_m': 0.008},
        ],
        'expected_breach': False,
        'source': 'NATO STANAG 4569 / AEP-55 Vol.1',
        'notes': 'AKM threat at 30 m. Used for APCs and light patrol vehicles.',
    },
    {
        'id': 'STANAG_L3_762x51_AP_WC',
        'label': '7.62×51 AP WC-core vs STANAG 4569 Level 3 vehicle armor',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '7.62x51mm',
            'projectile': 'AP (tungsten carbide core)',
            'mass_kg': 0.00960,
            'velocity_ms': 930.0,
            'radius_m': 0.00381,
            'material': 'tungsten_carbide_core',
        },
        'armor': [
            {'material': 'STANAG_Level3_RHA_equivalent', 'thickness_m': 0.010},
        ],
        'expected_breach': False,
        'source': 'NATO STANAG 4569 / AEP-55 Vol.1',
        'notes': 'Represents dedicated anti-material rifle threat.',
    },
    {
        'id': 'STANAG_L4_14p5mm_API',
        'label': '14.5×114 mm API B32 vs STANAG 4569 Level 4 vehicle armor',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '14.5x114mm',
            'projectile': 'API B32',
            'mass_kg': 0.06400,        # ~64 g nominal for 14.5mm B32
            'velocity_ms': 911.0,      # at 200 m (STANAG 4569 standoff)
            'radius_m': 0.00725,
            'material': 'steel_AP_core',
        },
        'armor': [
            {'material': 'STANAG_Level4_steel_composite', 'thickness_m': 0.020},
        ],
        'expected_breach': False,
        'source': 'NATO STANAG 4569 / AEP-55 Vol.1',
        'notes': 'ZPU / KPV HMG threat at 200 m. '
                 'Relevant for Bradley/Warrior class IFVs.',
    },
    {
        'id': 'STANAG_L5_25mm_APDS',
        'label': '25×137 mm APDS-T M791 vs STANAG 4569 Level 5 vehicle armor',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '25x137mm',
            'projectile': 'APDS-T M791',
            'mass_kg': 0.1520,         # ~152 g — 25mm APDS-T nominal
            'velocity_ms': 1258.0,     # at 500 m
            'radius_m': 0.01275,
            'material': 'tungsten_carbide_core',
        },
        'armor': [
            {'material': 'STANAG_Level5_composite', 'thickness_m': 0.040},
        ],
        'expected_breach': False,
        'source': 'NATO STANAG 4569 / AEP-55 Vol.1',
        'notes': 'Bushmaster / Oerlikon KBA cannon threat at 500 m.',
    },

    # -----------------------------------------------------------------------
    # CATEGORY 2 — HEAT anti-armor (shaped charge)
    # -----------------------------------------------------------------------
    # RPG-7 warhead data: Wikipedia article on RPG-7 (citing Jane's).
    # PG-7V penetration figure: 260 mm RHA (most conservative credible value;
    #   some sources give 330 mm — marked in notes).
    # -----------------------------------------------------------------------
    {
        'id': 'RPG7_PG7V_vs_RHA',
        'label': 'RPG-7 PG-7V HEAT vs RHA (bare steel)',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'PG-7V',
            'mass_kg': 2.25,           # complete round
            'warhead_diameter_m': 0.085,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.260},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/RPG-7 citing Jane\'s Infantry Weapons; '
                  'GlobalSecurity.org RPG-7 specs',
        'notes': 'Nominal: 260 mm RHA at 0°. Some sources cite 330 mm; '
                 '260 mm is the lowest well-cited figure. '
                 'Defeats early Cold War armor (T-55 hull: ~200 mm equiv).',
    },
    {
        'id': 'RPG7_PG7V_vs_RHA_fail',
        'label': 'RPG-7 PG-7V HEAT vs 320 mm RHA (Leopard 1 turret equiv)',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'PG-7V',
            'mass_kg': 2.25,
            'warhead_diameter_m': 0.085,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.320},
        ],
        'expected_breach': False,
        'source': 'Wikipedia/RPG-7; penetration ceiling ~260–330 mm',
        'notes': 'At the upper RHA estimate (260 mm) PG-7V fails against '
                 '320 mm target. Tests the lower bound of uncertainty.',
    },
    {
        'id': 'RPG7_PG7VL_vs_RHA',
        'label': 'RPG-7 PG-7VL HEAT vs 500 mm RHA',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'PG-7VL',
            'mass_kg': 2.60,
            'warhead_diameter_m': 0.093,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.500},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/RPG-7 (citing Jane\'s); '
                  'nominal penetration >500 mm, some sources 600 mm',
        'notes': 'Introduced 1977. Defeats T-72 with no ERA.',
    },
    {
        'id': 'RPG7_PG7VR_vs_ERA_plus_RHA',
        'label': 'RPG-7 PG-7VR tandem HEAT vs ERA + 750 mm RHA equivalent',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge_tandem',
            'designation': 'PG-7VR',
            'mass_kg': 4.50,
            'warhead_diameter_m': 0.105,  # main charge
            'precursor_diameter_m': 0.064,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_tandem',
        },
        'armor': [
            {'material': 'ERA_Kontakt5_equivalent', 'thickness_m': 0.050},
            {'material': 'RHA_steel', 'thickness_m': 0.700},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/RPG-7; GlobalSecurity.org RPG-7 specs; '
                  'Jane\'s Land Based Air Defence',
        'notes': '750 mm RHA behind ERA, or 900 mm bare RHA. '
                 'Concrete penetration: >1.5 m RC, >2 m brick (Wikipedia). '
                 'Defeats T-90 frontal arc with ERA. ESTIMATED.',
    },
    {
        'id': 'RPG7_PG7VR_vs_concrete',
        'label': 'RPG-7 PG-7VR tandem HEAT vs 1.5 m reinforced concrete',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge_tandem',
            'designation': 'PG-7VR',
            'mass_kg': 4.50,
            'warhead_diameter_m': 0.105,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_tandem',
        },
        'armor': [
            {'material': 'reinforced_concrete', 'thickness_m': 1.500},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/RPG-7 warhead table',
        'notes': 'Cited as >1.5 m RC penetration in gunrf.ru PG-7VR data sheet '
                 'and Wikipedia. Used for bunker-busting role.',
    },

    # -----------------------------------------------------------------------
    # ATGMs
    # -----------------------------------------------------------------------
    {
        'id': 'ATGM_Konkurs_9M113_vs_RHA',
        'label': '9M113 Konkurs HEAT vs 600 mm RHA',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': '9M113 Konkurs',
            'mass_kg': 2.70,           # warhead 9N131
            'warhead_diameter_m': 0.135,
            'flight_velocity_ms': 208.0,  # nominal cruise speed
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.600},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/9M113 Konkurs',
        'notes': '600 mm bare RHA. Konkurs-M (9M113M) tandem: '
                 '750–800 mm behind ERA.',
    },
    {
        'id': 'ATGM_Kornet_9M133_vs_RHA_behind_ERA',
        'label': '9M133 Kornet HEAT vs ERA + 1000 mm RHA equivalent',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge_tandem',
            'designation': '9M133 Kornet',
            'mass_kg': 4.60,           # 9N178 warhead mass
            'warhead_diameter_m': 0.152,
            'flight_velocity_ms': 240.0,
            'material': 'HEAT_tandem',
        },
        'armor': [
            {'material': 'ERA_Kontakt5_equivalent', 'thickness_m': 0.050},
            {'material': 'RHA_steel', 'thickness_m': 0.900},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/9M133 Kornet — 1000–1200 mm behind ERA',
        'notes': 'Kornet-1 standard: 1000+ mm behind ERA. '
                 'Kornet-M (9M133M-2): 1100–1300 mm behind ERA. '
                 'Decisively defeats T-90 with Kontakt-5.',
    },
    {
        'id': 'ATGM_TOW2_BGM71D_vs_RHA',
        'label': 'BGM-71D TOW-2 HEAT vs 900 mm RHA',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'BGM-71D TOW-2',
            'mass_kg': 5.90,           # warhead only
            'warhead_diameter_m': 0.152,
            'flight_velocity_ms': 278.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.900},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/BGM-71 TOW — TOW-2 warhead 900 mm RHA',
        'notes': 'TOW-2A (BGM-71E) rated at same penetration behind ERA. '
                 'TOW-2B (BGM-71F) is an EFP top-attack variant — '
                 'no RHA figure; defeats T-72/T-80 roof armor.',
    },
    {
        'id': 'ATGM_TOW_original_BGM71A_vs_RHA',
        'label': 'BGM-71A original TOW HEAT vs 430 mm RHA',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'BGM-71A TOW',
            'mass_kg': 3.90,
            'warhead_diameter_m': 0.152,
            'flight_velocity_ms': 278.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.430},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/BGM-71 TOW — 430 mm ESTIMATED value',
        'notes': 'Wikipedia marks this as an estimated figure. '
                 'Vietnam era; would defeat T-55 (200 mm glacis equiv) '
                 'but marginal against T-64/T-72.',
    },

    # -----------------------------------------------------------------------
    # HEAT vs modern composite armor — expected outcome tests
    # -----------------------------------------------------------------------
    {
        'id': 'PG7VL_vs_M1A2_frontal_fail',
        'label': 'RPG-7 PG-7VL HEAT vs M1A2 Abrams frontal turret (fail)',
        'category': 'heat_antitank',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'PG-7VL',
            'mass_kg': 2.60,
            'warhead_diameter_m': 0.093,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_single_stage',
        },
        'armor': [
            {'material': 'Chobham_composite_M1A2_turret_front',
             'thickness_m': 0.800,    # physical depth ~800 mm
             'rha_equivalent_heat_m': 1.320},  # ~1320 mm RHA equiv vs HEAT
        ],
        'expected_breach': False,
        'source': 'Open literature M1A2 armor estimates (HEAT equiv ~1320 mm); '
                  'PG-7VL nominal 500–600 mm HEAT penetration',
        'notes': 'ESTIMATED — exact M1A2 armor classified. '
                 '500 mm << 1320 mm RHAe; clear failure expected.',
    },

    # -----------------------------------------------------------------------
    # CATEGORY 3 — Kinetic energy penetrators (tank guns)
    # -----------------------------------------------------------------------
    # M829A3 data: Wikipedia/M829 citing Lanz-Odermatt calculator result.
    # 3BM42/60 data: Wikipedia/125mm smoothbore ammunition.
    # 17-pounder data: Bird & Livingston (2001) via Wikipedia.
    # -----------------------------------------------------------------------
    {
        'id': 'KE_M829A3_vs_RHA_2km',
        'label': 'M829A3 APFSDS-T vs 700 mm RHA at 2 km',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': 'M829A3',
            'gun': '120 mm M256 smoothbore (M1A2)',
            'total_mass_kg': 22.3,
            'penetrator_mass_kg': 10.0,   # DU rod + steel tip; ESTIMATED
            'muzzle_velocity_ms': 1555.0,
            'penetrator_diameter_m': 0.025,
            'penetrator_material': 'depleted_uranium',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.700},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/M829 — 680–720 mm RHA at 2 km '
                  '(Lanz-Odermatt calculation); '
                  'M829A3 product sheet Northrop/Arcon Partners',
        'notes': 'ESTIMATED from Lanz-Odermatt formula. Classified actual data. '
                 'Special DU+steel tip defeats Kontakt-5 ERA. '
                 'Penetrator length ~800 mm; diameter 25 mm.',
    },
    {
        'id': 'KE_M829A3_vs_T90_frontal',
        'label': 'M829A3 APFSDS vs T-90 frontal hull (expected breach)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': 'M829A3',
            'gun': '120 mm M256 smoothbore',
            'total_mass_kg': 22.3,
            'penetrator_mass_kg': 10.0,
            'muzzle_velocity_ms': 1555.0,
            'penetrator_diameter_m': 0.025,
            'penetrator_material': 'depleted_uranium',
        },
        'armor': [
            {'material': 'T90_composite_glacis',
             'rha_equivalent_ke_m': 0.600,   # ~600 mm RHAe vs KE (open sources)
             'thickness_m': 0.520},
        ],
        'expected_breach': True,
        'source': 'Open literature T-90 armor estimates (~600 mm RHAe vs KE); '
                  'M829A3 ~700 mm at 2 km',
        'notes': 'ESTIMATED. M829A3 nominal 700 mm > T-90 ~600 mm RHAe. '
                 'Outcome sensitive to ERA and obliquity.',
    },
    {
        'id': 'KE_3BM42_vs_RHA_2km',
        'label': '3BM42 Mango APFSDS vs 520 mm RHA at 2 km',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': '3BM42 Mango',
            'gun': '125 mm 2A46 smoothbore (T-72/T-80/T-90)',
            'total_mass_kg': 20.4,
            'penetrator_mass_kg': 4.85,
            'muzzle_velocity_ms': 1700.0,
            'penetrator_diameter_m': 0.022,
            'penetrator_material': 'tungsten_alloy_double_rod',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.520},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/125mm smoothbore ammunition — '
                  '520 mm at 0° / 230 mm at 60° at 2 km',
        'notes': 'Introduced 1986. Primary Soviet/early-Russian MBT round. '
                 'Marginal against Leopard 2A5 or M1A2 frontal.',
    },
    {
        'id': 'KE_3BM60_vs_RHA_2km',
        'label': '3BM60 Svinets-2 APFSDS vs 800 mm RHA at 2 km',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': '3BM60 Svinets-2',
            'gun': '125 mm 2A46M-5 smoothbore (T-90MS)',
            'total_mass_kg': 22.0,
            'projectile_mass_kg': 8.4,
            'muzzle_velocity_ms': 1660.0,
            'penetrator_length_m': 0.760,
            'penetrator_diameter_m': 0.022,
            'penetrator_material': 'tungsten_alloy',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.800},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/125mm smoothbore ammunition; '
                  'Defense Update 2020 Svinets article; '
                  'ResearchGate publication "Lethality of Russian APFSDS"',
        'notes': 'ESTIMATED 800–830 mm at 0°, 300 mm at 60° at 2 km. '
                 'Figures vary across sources (600–850 mm range). '
                 'Penetrator material WHA (tungsten heavy alloy).',
    },
    {
        'id': 'KE_3BM60_vs_Leopard2A6_frontal_uncertain',
        'label': '3BM60 vs Leopard 2A6 frontal turret (uncertain outcome)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': '3BM60 Svinets-2',
            'gun': '125 mm 2A46M-5',
            'total_mass_kg': 22.0,
            'projectile_mass_kg': 8.4,
            'muzzle_velocity_ms': 1660.0,
            'penetrator_diameter_m': 0.022,
            'penetrator_material': 'tungsten_alloy',
        },
        'armor': [
            {'material': 'Leopard2A6_turret_composite',
             'rha_equivalent_ke_m': 0.800,   # ESTIMATED ~800 mm RHAe vs KE
             'thickness_m': 0.600},
        ],
        'expected_breach': None,   # genuinely uncertain; within error bars
        'source': 'Open literature estimates; both values ~800 mm',
        'notes': 'UNCERTAIN — 3BM60 nominal ~800 mm ≈ Leopard 2A6 ~800 mm RHAe. '
                 'Outcome depends on exact obliquity and ERA presence. '
                 'Marks the boundary of simulator sensitivity.',
    },

    # -----------------------------------------------------------------------
    # WWII — 17-pounder vs Panther and Tiger I
    # -----------------------------------------------------------------------
    {
        'id': 'WW2_17pdr_APDS_vs_Panther_glacis_1000m',
        'label': 'British 17-pdr APDS vs Panther glacis at 1000 m (breach)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apds',
            'designation': 'Ordnance QF 17-pounder APDS Mk I',
            'gun': '76.2 mm QF 17-pounder',
            'projectile_mass_kg': 3.50,  # penetrator (sabot discarded) ~3.5 kg
            'muzzle_velocity_ms': 1204.0,
            'radius_m': 0.02540,         # 50.8 mm sub-calibre penetrator
            'penetrator_material': 'steel',
        },
        'armor': [
            {'material': 'RHA_Panther_glacis',
             'physical_thickness_m': 0.080,
             'slope_deg': 55.0,          # from vertical
             'effective_rha_m': 0.139},  # 80 mm at 55°  ≈ 139 mm normal
        ],
        'expected_breach': True,
        'source': 'Bird & Livingston (2001) via Wikipedia/17-pounder; '
                  'Panther armor: Wikipedia/Panther_tank',
        'notes': '17-pdr APDS penetration at 1000 m: 233 mm at 0° (Bird 2001). '
                 '233 mm >> 139 mm effective — decisive breach. '
                 'APCBC at same range: 150 mm; marginal/fail.',
    },
    {
        'id': 'WW2_17pdr_APCBC_vs_Panther_glacis_1000m_fail',
        'label': 'British 17-pdr APCBC vs Panther glacis at 1000 m (fail)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apcbc',
            'designation': 'Ordnance QF 17-pounder APCBC Shot Mk VIII',
            'gun': '76.2 mm QF 17-pounder',
            'projectile_mass_kg': 7.70,  # 17 lb
            'muzzle_velocity_ms': 884.0,
            'radius_m': 0.03810,
            'penetrator_material': 'steel',
        },
        'armor': [
            {'material': 'RHA_Panther_glacis',
             'physical_thickness_m': 0.080,
             'slope_deg': 55.0,
             'effective_rha_m': 0.139},
        ],
        'expected_breach': False,
        'source': 'Bird & Livingston (2001) via Wikipedia/17-pounder — '
                  'APCBC at 1000 m: 150 mm at 0°. '
                  'Effective = 139 mm but AP shot loses more to obliquity.',
        'notes': 'APCBC 150 mm vs 139 mm nominal effective is close; '
                 'historical accounts confirm difficulty at typical combat range. '
                 'Normalisation reduces penetration at 55° slope significantly.',
    },
    {
        'id': 'WW2_17pdr_APDS_vs_Tiger_I_frontal_500m',
        'label': 'British 17-pdr APDS vs Tiger I frontal hull at 500 m (breach)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apds',
            'designation': 'Ordnance QF 17-pounder APDS Mk I',
            'gun': '76.2 mm QF 17-pounder',
            'projectile_mass_kg': 3.50,
            'muzzle_velocity_ms': 1204.0,
            'radius_m': 0.02540,
            'penetrator_material': 'steel',
        },
        'armor': [
            {'material': 'RHA_Tiger_I_hull_front',
             'physical_thickness_m': 0.100,
             'slope_deg': 9.0,           # nearly vertical
             'effective_rha_m': 0.102},  # ~100 mm at 9° ≈ 102 mm effective
        ],
        'expected_breach': True,
        'source': 'Bird & Livingston (2001) — APDS at 500 m: 256 mm at 0°. '
                  'Tiger I hull front: Wikipedia/Tiger_I — 100 mm at 9°.',
        'notes': '256 mm >> 102 mm; decisive penetration. '
                 'Soviet firing trials (tankarchives.com) confirm 100 mm '
                 'plate penetrated from >1800 m with standard AP.',
    },

    # -----------------------------------------------------------------------
    # CATEGORY 4 — Directed energy (laser)
    # -----------------------------------------------------------------------
    # Laser data is sparse in unclassified literature.
    # Welding/cutting data is authoritative; DEW test data is reported performance.
    # -----------------------------------------------------------------------
    {
        'id': 'LASER_industrial_6kW_fiber_steel_25mm',
        'label': '6 kW fiber laser cuts 25 mm carbon steel (proxy for DEW)',
        'category': 'directed_energy',
        'weapon': {
            'type': 'laser_cw',
            'power_w': 6000.0,
            'wavelength_nm': 1070.0,    # typical fiber laser (Yb)
            'beam_diameter_m': 0.0003,  # ~300 µm focused
            'mode': 'cutting_with_assist_gas',
        },
        'target': {
            'material': 'carbon_steel',
            'thickness_m': 0.025,
        },
        'expected_breach': True,
        'source': 'Artizono fiber laser cutting chart; ACCURL laser specs; '
                  'ADH Machine Tool cutting guide',
        'notes': 'Industrial CNC data: 6 kW fiber laser cuts 25 mm mild steel. '
                 'In DEW context: sustained dwell at lower irradiance; '
                 'assist gas not available in field. '
                 'Time-to-cut at 25 mm: seconds at close range on stationary target.',
    },
    {
        'id': 'LASER_industrial_3kW_fiber_steel_10mm',
        'label': '3 kW fiber laser cuts 10 mm carbon steel',
        'category': 'directed_energy',
        'weapon': {
            'type': 'laser_cw',
            'power_w': 3000.0,
            'wavelength_nm': 1070.0,
            'beam_diameter_m': 0.0003,
            'mode': 'cutting_with_assist_gas',
        },
        'target': {
            'material': 'carbon_steel',
            'thickness_m': 0.010,
        },
        'expected_breach': True,
        'source': 'Artizono / ACCURL fiber laser cutting charts',
        'notes': 'Practical threshold for 3 kW class. '
                 'Extrapolation to DEW: at 3 kW spot size 10 cm diameter '
                 '(representative DEW), irradiance ~400 W/cm² — '
                 'sufficient for ablation but not clean cut.',
    },
    {
        'id': 'LASER_HELIOS_60kW_drone_defeat',
        'label': 'HELIOS 60 kW laser defeats small UAV at <6 km',
        'category': 'directed_energy',
        'weapon': {
            'type': 'laser_high_energy',
            'designation': 'HELIOS (Lockheed Martin)',
            'platform': 'USS Preble DDG-88',
            'power_w': 60000.0,         # delivered 60 kW
            'max_power_w': 120000.0,    # upgradeable to 120 kW
            'max_range_m': 9600.0,      # ~6 miles
        },
        'target': {
            'material': 'small_UAV_aluminum_composite',
            'thickness_m': 0.002,       # typical drone skin
            'description': 'Group 1-2 fixed-wing or rotary drone',
        },
        'expected_breach': True,
        'source': 'Euronews March 2026; The War Zone USS Preble HELIOS test; '
                  'Lockheed Martin HELIOS infographic; '
                  'National Interest Apr 2026',
        'notes': 'USS Preble downed 4 drones in 2025 test. '
                 'Time-to-defeat not publicly released. '
                 'Atmospheric conditions (rain, smoke) degrade performance. '
                 'Cost per shot: cents of electricity.',
    },
    {
        'id': 'LASER_IronBeam_100kW_drone_defeat',
        'label': 'Iron Beam 100 kW laser defeats drone/rocket at <10 km',
        'category': 'directed_energy',
        'weapon': {
            'type': 'laser_high_energy',
            'designation': 'Iron Beam (Rafael/Elbit)',
            'platform': 'ground_mobile',
            'power_w': 100000.0,
            'beam_diameter_m': 0.450,   # 450 mm aperture (reported)
            'max_range_m': 10000.0,
        },
        'target': {
            'material': 'drone_or_rocket_body',
            'thickness_m': 0.002,
            'description': 'UAV, mortar, short-range rocket',
        },
        'expected_breach': True,
        'source': 'Tom\'s Hardware / Jerusalem Post Dec 2025 deployment report; '
                  'Interesting Engineering 2026; TheDefenseWatch Iron Beam specs',
        'notes': 'Operationally deployed Israel 2025 (first combat-ready DEW). '
                 'Destroyed rockets, mortars, UAVs in trials (exact timing NR). '
                 'Lite Beam variant: 10 kW, 2 km range.',
    },

    # -----------------------------------------------------------------------
    # CATEGORY 5 — Explosive blast vs structures
    # -----------------------------------------------------------------------
    # Engineering rules of thumb from FM 90-10 App.E (GlobalSecurity.org);
    # RPG-7 concrete data from Wikipedia/RPG-7 warhead table.
    # -----------------------------------------------------------------------
    {
        'id': 'EXP_C4_4p5kg_vs_RC_600mm',
        'label': '4.5 kg C-4 contact charge vs 600 mm reinforced concrete (breach)',
        'category': 'blast_vs_structure',
        'weapon': {
            'type': 'contact_explosive',
            'material': 'C4_RDX_composite',
            'mass_kg': 4.50,
            'tnt_equivalent_kg': 6.075,  # C-4 TNT equiv factor ~1.35
            'placement': 'contact_surface',
        },
        'target': {
            'material': 'reinforced_concrete',
            'thickness_m': 0.610,       # 2 feet (standard military cite)
            'description': 'Typical wall or bunker up to 61 cm thick',
        },
        'expected_breach': True,
        'source': 'FM 90-10 Appendix E Demolitions (GlobalSecurity.org) — '
                  '"tie 4.5 kg C-4 on pole against target, waist height"',
        'notes': 'Army field manual rule of thumb; not a precise formula. '
                 'For thicker walls, the Remennikov-Mentus-Uy breaching model '
                 '(SAGE Journals 2015) provides a more rigorous calculation.',
    },
    {
        'id': 'EXP_C4_20lb_vs_bunker',
        'label': '20 lb (9.1 kg) C-4 satchel vs reinforced concrete bunker',
        'category': 'blast_vs_structure',
        'weapon': {
            'type': 'contact_explosive',
            'material': 'C4_RDX_composite',
            'mass_kg': 9.10,
            'tnt_equivalent_kg': 12.285,
            'placement': 'satchel_charge_contact',
        },
        'target': {
            'material': 'reinforced_concrete_bunker',
            'thickness_m': 0.610,
            'description': 'Heavy reinforced concrete fortification',
        },
        'expected_breach': True,
        'source': 'FM 90-10 Appendix E (GlobalSecurity.org) — '
                  '"20 lb C-4 can breach reinforced concrete bunkers"',
        'notes': 'Large satchel charge. Practical military demolition datum.',
    },
    {
        'id': 'RPG7_PG7V_vs_masonry_wall',
        'label': 'RPG-7 PG-7V HEAT vs 350 mm brick/masonry wall (breach)',
        'category': 'blast_vs_structure',
        'weapon': {
            'type': 'heat_shaped_charge',
            'designation': 'PG-7V',
            'mass_kg': 2.25,
            'warhead_diameter_m': 0.085,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_single_stage',
        },
        'target': {
            'material': 'brick_masonry',
            'thickness_m': 0.350,
        },
        'expected_breach': True,
        'source': 'PG-7VR data sheet gives >2 m brick for tandem; '
                  'PG-7V (smaller) defeats standard wall construction',
        'notes': 'PG-7VR penetrates >2 m brick (Wikipedia/gunrf.ru). '
                 'PG-7V would breach standard 35 cm masonry wall. ESTIMATED.',
    },
    {
        'id': 'RPG7_PG7VR_vs_brick_2m',
        'label': 'RPG-7 PG-7VR tandem HEAT vs 2 m brick wall (breach)',
        'category': 'blast_vs_structure',
        'weapon': {
            'type': 'heat_shaped_charge_tandem',
            'designation': 'PG-7VR',
            'mass_kg': 4.50,
            'warhead_diameter_m': 0.105,
            'flight_velocity_ms': 300.0,
            'material': 'HEAT_tandem',
        },
        'target': {
            'material': 'brick_masonry',
            'thickness_m': 2.000,
        },
        'expected_breach': True,
        'source': 'gunrf.ru PG-7VR data sheet; Wikipedia/RPG-7 warhead table',
        'notes': 'Documented at >2 m brick. Useful for breaching HESCO/adobe.',
    },

    # -----------------------------------------------------------------------
    # Additional cross-category boundary / edge tests
    # -----------------------------------------------------------------------
    {
        'id': 'NIJ_IIIA_vs_M855_rifle_fail',
        'label': '5.56×45 M855 rifle round vs NIJ Level IIIA soft armor (breach)',
        'category': 'small_arms_vs_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '5.56x45mm NATO',
            'projectile': 'M855 SS109 steel-core FMJ',
            'mass_kg': 0.00400,        # 4.0 g / 62 gr
            'velocity_ms': 900.0,      # standard M16 muzzle velocity
            'radius_m': 0.00284,       # 5.7 mm diam / 2
            'material': 'steel_core_lead_jacket',
        },
        'armor': [
            {'material': 'Kevlar_soft_IIIA', 'thickness_m': 0.012},
        ],
        'expected_breach': True,
        'source': 'NIJ 0101.06 explicitly states Level IIIA does NOT protect '
                  'against rifle rounds. M855 at 900 m/s exceeds Level IIIA rating.',
        'notes': 'Critical boundary case: IIIA armor stops .44 Mag handgun '
                 'but is defeated by intermediate rifle calibers. '
                 'Level III hard plate required for rifle protection.',
    },
    {
        'id': 'KE_M829_original_vs_RHA_2km',
        'label': 'M829 (original 1985) APFSDS vs 540 mm RHA at 2 km',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': 'M829',
            'gun': '120 mm M256 smoothbore (M1)',
            'total_mass_kg': 18.6,
            'penetrator_diameter_m': 0.027,
            'muzzle_velocity_ms': 1670.0,
            'penetrator_material': 'depleted_uranium',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.540},
        ],
        'expected_breach': True,
        'source': 'Wikipedia/M829 — "540 mm RHA at up to 2000 m"',
        'notes': 'First DU penetrator for M1. Replaced by M829A1 after Gulf War.',
    },
    {
        'id': 'KE_3BM42_vs_M1A2_hull_fail',
        'label': '3BM42 Mango APFSDS vs M1A2 hull glacis (fail)',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apfsds',
            'designation': '3BM42 Mango',
            'gun': '125 mm 2A46',
            'total_mass_kg': 20.4,
            'penetrator_mass_kg': 4.85,
            'muzzle_velocity_ms': 1700.0,
            'penetrator_diameter_m': 0.022,
            'penetrator_material': 'tungsten_alloy',
        },
        'armor': [
            {'material': 'M1A2_hull_glacis_composite',
             'rha_equivalent_ke_m': 0.560,   # ~560 mm RHAe vs KE (open literature)
             'thickness_m': 0.480},
        ],
        'expected_breach': False,
        'source': 'Open literature M1A2 hull estimates (~560 mm RHAe vs KE); '
                  '3BM42: 520 mm at 2 km',
        'notes': 'ESTIMATED. 520 mm < 560 mm RHAe; expected failure at standoff. '
                 'At closer range (e.g. 500 m), muzzle velocity gives '
                 '~560 mm which approaches the boundary.',
    },
    {
        'id': 'WW2_17pdr_APDS_vs_RHA_at_2000m',
        'label': 'British 17-pdr APDS penetration at 2000 m — 194 mm RHA',
        'category': 'kinetic_penetrator',
        'weapon': {
            'type': 'apds',
            'designation': 'Ordnance QF 17-pounder APDS Mk I',
            'gun': '76.2 mm QF 17-pounder',
            'projectile_mass_kg': 3.50,
            'muzzle_velocity_ms': 1204.0,
            'radius_m': 0.02540,
            'penetrator_material': 'steel',
        },
        'armor': [
            {'material': 'RHA_steel', 'thickness_m': 0.194},
        ],
        'expected_breach': True,
        'source': 'Bird & Livingston (2001) p. 60 via Wikipedia/17-pounder — '
                  '194 mm at 2000 m at 0° obliquity',
        'notes': 'Long-range potential of 17-pdr APDS. '
                 'Tiger II (Königstiger) turret front 185 mm — '
                 'theoretically penetrable at 2 km (though barely). '
                 'Production APDS accuracy at 2 km was poor in practice.',
    },
    {
        'id': 'STANAG_L4_vs_AK_fail',
        'label': '7.62×39 FMJ AKM ball vs STANAG 4569 Level 4 armor (fail)',
        'category': 'stanag_vehicle_armor',
        'weapon': {
            'type': 'kinetic',
            'caliber': '7.62x39mm',
            'projectile': 'FMJ PS ball (mild steel core)',
            'mass_kg': 0.00800,
            'velocity_ms': 715.0,      # AKM muzzle velocity
            'radius_m': 0.00381,
            'material': 'mild_steel_core',
        },
        'armor': [
            {'material': 'STANAG_Level4_steel_composite', 'thickness_m': 0.020},
        ],
        'expected_breach': False,
        'source': 'STANAG 4569: Level 4 defeats 14.5 mm API; '
                  '7.62×39 FMJ is a Level 1-2 threat',
        'notes': 'Confirms hierarchy: Level 4 armor overkill for AKM ball.',
    },
]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
VALIDATION_METADATA = {
    'version': '1.0.0',
    'count': len(VALIDATION_TESTS),
    'categories': {
        'small_arms_vs_armor': 'NIJ 0101.06 and STANAG 4569 personal/vehicle armor',
        'stanag_vehicle_armor': 'NATO light-vehicle ballistic protection levels',
        'heat_antitank': 'Shaped charge HEAT warheads vs RHA and composite armor',
        'kinetic_penetrator': 'APFSDS / APDS / APCBC kinetic energy rounds',
        'directed_energy': 'Laser weapon and industrial cutting proxy data',
        'blast_vs_structure': 'Contact explosive and shaped charge vs concrete/masonry',
    },
    'primary_sources': [
        'NIJ Standard-0101.06 (OJP pub 223054) — https://www.ojp.gov/pdffiles1/nij/223054.pdf',
        'NATO STANAG 4569 / AEP-55 Vol.1 — https://en.wikipedia.org/wiki/STANAG_4569',
        'Wikipedia/RPG-7 citing Jane\'s Infantry Weapons',
        'Wikipedia/M829 citing Lanz-Odermatt and Northrop product sheets',
        'Wikipedia/125mm smoothbore ammunition',
        'Wikipedia/9M133 Kornet, 9M113 Konkurs, BGM-71 TOW',
        'Wikipedia/Ordnance QF 17-pounder citing Bird & Livingston (2001)',
        'GlobalSecurity.org RPG-7 specs — https://www.globalsecurity.org/military/world/russia/rpg-7-specs.htm',
        'FM 90-10 Appendix E Demolitions — https://www.globalsecurity.org/military/library/policy/army/fm/90-10/90-10ape.htm',
        'HELIOS: Lockheed Martin; The War Zone USS Preble test reports',
        'Iron Beam: Jerusalem Post / Tom\'s Hardware operational deployment Dec 2025',
        'Artizono / ACCURL fiber laser cutting charts (industrial proxy for DEW)',
    ],
    'uncertainty_flags': [
        'ESTIMATED — derived from open-source calculation, not declassified test',
        'UNCERTAIN — outcome within model error bars; use as sensitivity check',
        'Modern composite armor RHAe values are open-literature estimates; exact values classified',
        'Laser DEW time-to-defeat figures not released publicly; industrial cutting used as proxy',
    ],
}


if __name__ == '__main__':
    by_cat = {}
    for t in VALIDATION_TESTS:
        c = t['category']
        by_cat.setdefault(c, []).append(t)
    print(f"Total tests: {len(VALIDATION_TESTS)}")
    for cat, tests in by_cat.items():
        breach_count = sum(1 for t in tests if t['expected_breach'] is True)
        hold_count   = sum(1 for t in tests if t['expected_breach'] is False)
        unk_count    = sum(1 for t in tests if t['expected_breach'] is None)
        print(f"  {cat}: {len(tests)} tests "
              f"({breach_count} breach / {hold_count} hold / {unk_count} uncertain)")
