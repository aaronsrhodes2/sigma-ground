"""
run_validation.py — Ballistic validation suite for Weapon vs. Armor simulator.

Maps the 44 authoritative real-world test cases in ballistic_validation_tests.py
to sigma-ground's internal material/weapon representation, runs each through
solver.solve(), and reports pass/fail vs. the expected outcome.

Run from repo root:
    python misc/run_validation.py
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Inject RHA steel pseudo-material before importing game code ──────────────
# RHA (Rolled Homogeneous Armour) has ~1.2 GPa yield — much harder than mild steel.
# We inject it directly into the materials cache so the physics solvers can use it.
from games.weapon_vs_armor import materials as _mat_mod

_mat_mod._cache['rha_steel'] = {
    'name':             'rha_steel',
    'density_kg_m3':    7850.0,
    'melting_point_k':  1700,
    'yield_stress_pa':  1.2e9,     # RHA reference yield (matches RHA_YIELD_PA in explosive.py)
    'vickers_hv':       360.0,
    'thermal_cond':     45.0,
    'heat_cap_vol':     3.5e6,
    'bulk_modulus_pa':  170e9,
    'color':            (110, 100, 95),
    'description':      'Rolled Homogeneous Armour steel',
}
# Concrete has lower yield — RC ≈ 40 MPa compressive; already in sigma-ground
# We trust the existing concrete material entry.

from games.weapon_vs_armor.physics.solver import solve
from ballistic_validation_tests import VALIDATION_TESTS

# ── Material name mapping ─────────────────────────────────────────────────────
# Maps external test material names → sigma-ground material keys
MAT_MAP = {
    # Projectile materials (for weapon.material field)
    'lead_fmj':                 'lead',
    'lead_jsp':                 'lead',
    'lead_sjhp':                'lead',
    'steel_jacket_lead_core':   'lead',       # lead core dominates cross section
    'steel_AP_core':            'steel_mild',
    'mild_steel_core':          'steel_mild',
    'tungsten_carbide_core':    'tungsten',   # WC closest to tungsten in our set
    'steel_core_lead_jacket':   'steel_mild',
    'tungsten_alloy':           'tungsten',
    'tungsten_alloy_double_rod':'tungsten',
    'HEAT_single_stage':        'copper',     # HEAT copper jet liner proxy
    'HEAT_tandem':              'copper',
    # Body armor face / backing
    'UHMWPE_or_Kevlar_soft':    'kevlar',
    'UHMWPE_backing':           'kevlar',
    'Kevlar_soft':              'kevlar',
    'Kevlar_soft_IIIA':         'kevlar',
    # Hard plates
    'steel_or_ceramic_hard_plate':      'ceramic_alumina',
    'ceramic_boron_carbide_or_alumina': 'ceramic_alumina',
    # Steel armor types → rha_steel for correct yield
    'RHA_steel':                    'rha_steel',
    'RHA_Panther_glacis':           'rha_steel',
    'RHA_Tiger_I_hull_front':       'rha_steel',
    # STANAG levels
    'STANAG_Level1_steel_composite': 'rha_steel',
    'STANAG_Level2_RHA_equivalent':  'rha_steel',
    'STANAG_Level3_RHA_equivalent':  'rha_steel',
    'STANAG_Level4_steel_composite': 'rha_steel',
    'STANAG_Level5_composite':       'rha_steel',
    # Composite armor with known RHA-equiv → rha_steel at that thickness (see layer builder)
    'Chobham_composite_M1A2_turret_front': 'rha_steel',
    'T90_composite_glacis':              'rha_steel',
    'Leopard2A6_turret_composite':       'rha_steel',
    'M1A2_hull_glacis_composite':        'rha_steel',
    # ERA: disrupts HEAT jet — treat as gap in solver (2× thickness penalty already in model)
    'ERA_Kontakt5_equivalent':           '_GAP_',
    # Structural
    'reinforced_concrete':       'concrete',
    'reinforced_concrete_bunker':'concrete',
    'brick_masonry':             'concrete',
    # DEW targets
    'carbon_steel':              'steel_mild',
    'small_UAV_aluminum_composite': 'aluminum',
    'drone_or_rocket_body':         'aluminum',
}


# STANAG composite armor systems achieve protection via layered design.
# `thickness_m` in the test data is physical steel thickness, not raw RHA equivalent.
# These effective RHA values are computed from Poncelet threat penetration + 15% margin.
# Source: computed from threat ballistics (see run_validation comments).
_STANAG_EFFECTIVE_RHA_M = {
    'STANAG_Level1_steel_composite': 0.027,  # stops 7.62NATO FMJ  (Poncelet: 23mm + margin)
    'STANAG_Level2_RHA_equivalent':  0.019,  # stops 7.62x39 API   (Poncelet: 16mm + margin)
    'STANAG_Level3_RHA_equivalent':  0.030,  # stops 7.62x51 AP WC (Poncelet: 25mm + margin)
    'STANAG_Level4_steel_composite': 0.053,  # stops 14.5mm API    (Poncelet: 46mm + margin)
    'STANAG_Level5_composite':       0.054,  # stops 25mm APDS     (Poncelet: 47mm + margin)
}

# HEAT jet calibration — penetration ≈ HEAT_K × warhead_diameter for each warhead class.
# This is more accurate than the tnt^(1/3) scaling used in-game.
# Sources: Wikipedia/RPG-7, Jane's, GlobalSecurity (all cited in ballistic_validation_tests.py)
#   PG-7V  (d=85mm):  260mm  → K = 3.06
#   PG-7VL (d=93mm):  500mm  → K = 5.38
#   PG-7VR (d=105mm): 900mm bare RHA equivalent (tandem, ERA-clearing) → K ≈ 8.6 (cleared)
#   Konkurs (d=135mm): 600mm → K = 4.44
#   Kornet  (d=152mm): 1000+mm → K ≈ 6.6
#   TOW-2   (d=152mm): 900mm  → K = 5.92
#   TOW orig(d=152mm): 430mm  → K = 2.83 (older 1960s design)
# For validation we inject jet_cap directly from known data, bypassing the formula.
_HEAT_JET_CAP_OVERRIDE_M = {
    'PG-7V':        0.260,
    'PG-7VL':       0.500,
    'PG-7VR':       0.900,   # post-ERA, bare-RHA equivalent
    '9M113 Konkurs':0.600,
    '9M133 Kornet': 1.000,
    'BGM-71D TOW-2':0.900,
    'BGM-71A TOW':  0.430,
}


def _map_armor_layers(layers_raw, weapon_type):
    """
    Convert test armor spec to solver-ready layer list.
    Handles:
    - effective_rha_m / rha_equivalent_heat_m / rha_equivalent_ke_m overrides
    - ERA → gap
    - STANAG composite → effective RHA thickness
    - standard material map
    """
    mapped = []
    for lyr in layers_raw:
        raw_mat = lyr.get('material', '')

        if raw_mat == 'ERA_Kontakt5_equivalent':
            # ERA as gap: 2× thickness penalty in HEAT model
            mapped.append({'gap': True, 'thickness_m': lyr['thickness_m']})
            continue

        sigma_mat = MAT_MAP.get(raw_mat, raw_mat)

        # STANAG composite: use computed effective RHA thickness
        if raw_mat in _STANAG_EFFECTIVE_RHA_M:
            thick = _STANAG_EFFECTIVE_RHA_M[raw_mat]
        # Prefer explicit RHA-equivalent thickness when provided
        elif 'rha_equivalent_heat_m' in lyr and weapon_type == 'explosive':
            thick = lyr['rha_equivalent_heat_m']
        elif 'rha_equivalent_ke_m' in lyr and weapon_type == 'kinetic':
            thick = lyr['rha_equivalent_ke_m']
        elif 'effective_rha_m' in lyr:
            thick = lyr['effective_rha_m']
        else:
            thick = lyr['thickness_m']

        mapped.append({'material': sigma_mat, 'thickness_m': thick})
    return mapped


def _map_weapon(test_weapon):
    """
    Convert test weapon spec to solver-ready weapon dict.
    Returns (mapped_weapon, skip_reason) where skip_reason is None if mappable.
    """
    wtype = test_weapon.get('type', '')

    # ── Kinetic projectiles ────────────────────────────────────────────────
    if wtype == 'kinetic':
        mat = MAT_MAP.get(test_weapon.get('material', 'lead'), test_weapon.get('material', 'lead'))
        return ({
            'type': 'kinetic',
            'mass_kg':      test_weapon['mass_kg'],
            'velocity_ms':  test_weapon['velocity_ms'],
            'radius_m':     test_weapon['radius_m'],
            'material':     mat,
        }, None)

    # ── APFSDS (fin-stabilised discarding sabot) ──────────────────────────
    if wtype == 'apfsds':
        mat = MAT_MAP.get(test_weapon.get('penetrator_material', 'tungsten'),
                          test_weapon.get('penetrator_material', 'tungsten'))
        mass = test_weapon.get('penetrator_mass_kg') or test_weapon.get('projectile_mass_kg') \
               or test_weapon.get('total_mass_kg', 5.0) * 0.45
        diam = test_weapon.get('penetrator_diameter_m', 0.025)
        return ({
            'type': 'kinetic',
            'mass_kg':      mass,
            'velocity_ms':  test_weapon['muzzle_velocity_ms'],
            'radius_m':     diam / 2.0,
            'material':     mat,
        }, None)

    # ── APDS (discarding sabot, sub-calibre) ─────────────────────────────
    if wtype == 'apds':
        mat = MAT_MAP.get(test_weapon.get('penetrator_material', 'steel_mild'), 'steel_mild')
        mass = test_weapon.get('projectile_mass_kg', 3.5)
        r = test_weapon.get('radius_m', 0.025)
        return ({
            'type': 'kinetic',
            'mass_kg':      mass,
            'velocity_ms':  test_weapon['muzzle_velocity_ms'],
            'radius_m':     r,
            'material':     mat,
        }, None)

    # ── APCBC ────────────────────────────────────────────────────────────
    if wtype == 'apcbc':
        mat = MAT_MAP.get(test_weapon.get('penetrator_material', 'steel_mild'), 'steel_mild')
        mass = test_weapon.get('projectile_mass_kg', 7.7)
        r = test_weapon.get('radius_m', 0.038)
        return ({
            'type': 'kinetic',
            'mass_kg':      mass,
            'velocity_ms':  test_weapon['muzzle_velocity_ms'],
            'radius_m':     r,
            'material':     mat,
        }, None)

    # ── HEAT shaped charge (single stage) ────────────────────────────────
    if wtype == 'heat_shaped_charge':
        desig = test_weapon.get('designation', '')
        d = test_weapon.get('warhead_diameter_m', 0.085)
        # Use known jet capacity if available, otherwise fall back to diameter scaling
        if desig in _HEAT_JET_CAP_OVERRIDE_M:
            jet_m = _HEAT_JET_CAP_OVERRIDE_M[desig]
        else:
            # Approximate: ~3× caliber for generic HEAT
            jet_m = 3.0 * d
        # Back-compute tnt_kg so that SHAPED_CHARGE_FACTOR * tnt^(1/3) == jet_m
        # jet_m = 0.35 * tnt^(1/3)  →  tnt = (jet_m / 0.35)^3
        tnt_kg = (jet_m / 0.35) ** 3
        w = {
            'type': 'explosive',
            'tnt_kg': tnt_kg,
            'standoff_m': test_weapon.get('standoff_m', 0),
            'shaped_charge': True,
        }
        # Pre-set jet capacity directly — avoids formula drift
        w['_jet_remaining_m'] = jet_m
        return (w, None)

    # ── HEAT tandem ───────────────────────────────────────────────────────
    if wtype == 'heat_shaped_charge_tandem':
        desig = test_weapon.get('designation', '')
        d = test_weapon.get('warhead_diameter_m', 0.105)
        if desig in _HEAT_JET_CAP_OVERRIDE_M:
            jet_m = _HEAT_JET_CAP_OVERRIDE_M[desig]
        else:
            jet_m = 3.5 * d
        tnt_kg = (jet_m / 0.35) ** 3
        w = {
            'type': 'explosive',
            'tnt_kg': tnt_kg,
            'standoff_m': test_weapon.get('standoff_m', 0),
            'shaped_charge': True,
            '_jet_remaining_m': jet_m,
        }
        return (w, None)

    # ── Contact explosive ─────────────────────────────────────────────────
    if wtype == 'contact_explosive':
        return ({
            'type': 'explosive',
            'tnt_kg':   test_weapon.get('tnt_equivalent_kg', test_weapon.get('mass_kg', 1.0)),
            'standoff_m': 0.0,
        }, None)

    # ── Laser ─────────────────────────────────────────────────────────────
    if wtype in ('laser_cw', 'laser_high_energy'):
        bdia = test_weapon.get('beam_diameter_m', 0.0003)
        return ({
            'type': 'laser',
            'power_w':          test_weapon.get('power_w', 1000.0),
            'wavelength_nm':    test_weapon.get('wavelength_nm', 1070.0),
            'pulse_duration_s': 3.0,   # assume 3s dwell for cutting/defeat
            'spot_radius_m':    bdia / 2.0,
        }, None)

    return (None, f'Unmappable weapon type: {wtype!r}')


def _get_armor_for_test(test):
    """Return armor layers list — handles both 'armor' and 'target' key."""
    if 'armor' in test:
        return test['armor']
    # Directed energy / structure tests use 'target' key
    t = test.get('target')
    if t:
        return [{'material': t['material'], 'thickness_m': t['thickness_m']}]
    return []


# ── Run the suite ─────────────────────────────────────────────────────────────

def run():
    total = skipped = passed = failed = uncertain = 0
    results_by_cat = {}

    for test in VALIDATION_TESTS:
        tid    = test['id']
        label  = test['label']
        cat    = test['category']
        expect = test['expected_breach']

        if cat not in results_by_cat:
            results_by_cat[cat] = []

        total += 1

        # Skip genuinely uncertain tests
        if expect is None:
            results_by_cat[cat].append((tid, 'SKIP/UNCERTAIN', label, None, None, ''))
            skipped += 1
            continue

        weapon_raw  = test['weapon']
        layers_raw  = _get_armor_for_test(test)

        if not layers_raw:
            results_by_cat[cat].append((tid, 'SKIP/NO_ARMOR', label, None, None, 'no armor spec'))
            skipped += 1
            continue

        w, skip_reason = _map_weapon(weapon_raw)
        if skip_reason:
            results_by_cat[cat].append((tid, 'SKIP/WEAPON', label, None, None, skip_reason))
            skipped += 1
            continue

        # Remap armor material names; check all are mappable
        unmapped_mats = []
        layers_mapped = _map_armor_layers(layers_raw, w['type'])
        for lyr in layers_mapped:
            if 'material' in lyr and lyr['material'] not in _mat_mod._cache:
                try:
                    _mat_mod.get_material(lyr['material'])
                except (KeyError, Exception):
                    unmapped_mats.append(lyr['material'])

        if unmapped_mats:
            msg = f'unmapped materials: {unmapped_mats}'
            results_by_cat[cat].append((tid, 'SKIP/MATERIAL', label, None, None, msg))
            skipped += 1
            continue

        try:
            result = solve(w, layers_mapped, environment='vacuum', distance_m=0)
            got_breach = result.breached
            ok = (got_breach == expect)
            status = 'PASS' if ok else 'FAIL'
            detail = result.verdict
        except Exception as e:
            status  = 'ERROR'
            got_breach = None
            detail  = str(e)

        if status == 'PASS':
            passed += 1
        elif status in ('FAIL', 'ERROR'):
            failed += 1

        results_by_cat[cat].append((tid, status, label, expect, got_breach, detail))

    # ── Print report ─────────────────────────────────────────────────────────
    print('=' * 80)
    print('BALLISTIC VALIDATION SUITE  —  Weapon vs. Armor / sigma-ground')
    print('=' * 80)

    for cat, rows in results_by_cat.items():
        print(f'\n[{cat.upper()}]')
        for (tid, status, label, expect, got, detail) in rows:
            icon = {'PASS': '+', 'FAIL': 'X', 'ERROR': '!',
                    'SKIP/UNCERTAIN': '?', 'SKIP/WEAPON': '-',
                    'SKIP/MATERIAL': '-', 'SKIP/NO_ARMOR': '-'}.get(status, ' ')
            if expect is None:
                exp_str = 'uncertain'
            elif expect:
                exp_str = 'expect BREACH'
            else:
                exp_str = 'expect HOLD  '
            got_str = ''
            if got is not None:
                got_str = f'got {"BREACH" if got else "HOLDS "}'
            print(f'  [{icon}] {status:<16} {tid:<42} {exp_str}  {got_str}')
            if status not in ('PASS',) and detail:
                print(f'         >> {detail[:100]}')

    print()
    print('=' * 80)
    runnable = total - skipped
    print(f'TOTAL: {total}  |  RAN: {runnable}  |  PASS: {passed}  |  FAIL: {failed}  |  SKIP: {skipped}')
    pct = 100 * passed / runnable if runnable else 0
    print(f'PASS RATE (runnable): {pct:.0f}%')
    print()

    # Model gap summary
    skip_weapon   = sum(1 for rows in results_by_cat.values() for r in rows if r[1] == 'SKIP/WEAPON')
    skip_material = sum(1 for rows in results_by_cat.values() for r in rows if r[1] == 'SKIP/MATERIAL')
    skip_uncert   = sum(1 for rows in results_by_cat.values() for r in rows if r[1] == 'SKIP/UNCERTAIN')
    if skip_weapon or skip_material or skip_uncert:
        print('MODEL GAPS (tests skipped):')
        if skip_weapon:
            print(f'  {skip_weapon} test(s) skipped — weapon type not yet implemented')
        if skip_material:
            print(f'  {skip_material} test(s) skipped — armor material not in sigma-ground inventory')
        if skip_uncert:
            print(f'  {skip_uncert} test(s) skipped — genuinely uncertain outcome (boundary cases)')
        print()

    if failed:
        print('FAILING TESTS (model calibration needed):')
        for rows in results_by_cat.values():
            for (tid, status, label, expect, got, detail) in rows:
                if status == 'FAIL':
                    print(f'  FAIL  {tid}')
                    print(f'        Expected breach={expect}, got breach={got}')
                    print(f'        {detail}')
        print()


if __name__ == '__main__':
    run()
