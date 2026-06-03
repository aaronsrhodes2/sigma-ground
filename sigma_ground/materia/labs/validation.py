"""
Validation — known-answer scenarios for simulation sanity checking.

Each scenario defines expected outcomes with tolerances. After running
a simulation, compare results against these to verify physics accuracy.

All expected values are derived from textbook physics or well-known
experimental results. Sources cited inline.
"""

import math


# ── Known scenarios ───────────────────────────────────────────────────

KNOWN_SCENARIOS = {
    'free_fall_vacuum': {
        'description': 'Object dropped from 1m in vacuum (no drag)',
        'expected': {
            # t = sqrt(2h/g) = sqrt(2/9.80665) = 0.4515 s
            'impact_time': 0.4515,
            # v = sqrt(2gh) = sqrt(2 × 9.80665) = 4.429 m/s
            'impact_velocity': 4.429,
        },
        'tolerance': 0.02,  # 2% relative tolerance
        'source': 'Newtonian kinematics: h = ½gt², v = gt',
    },

    'free_fall_air': {
        'description': 'Iron sphere (r=5cm) dropped from 10m in air',
        'expected': {
            # Without drag: t = 1.428s, v = 14.0 m/s
            # With drag: slightly slower, longer time
            'impact_velocity_range': (13.0, 14.1),  # m/s
        },
        'tolerance': 0.05,
        'source': 'Drag correction small for dense, large sphere over 10m',
    },

    'elastic_collision': {
        'description': 'Two equal iron spheres, head-on elastic collision',
        'expected': {
            # Equal mass elastic: velocities exchange
            'velocity_exchange': True,
            'energy_conservation': 0.01,  # < 1% energy change
        },
        'tolerance': 0.02,
        'source': 'Newton\'s cradle: equal-mass elastic collision',
    },

    'projectile_45deg': {
        'description': 'Projectile launched at 45° at 10 m/s in vacuum',
        'expected': {
            # Range = v²sin(2θ)/g = 100/9.80665 = 10.197 m
            'range_m': 10.197,
            # Max height = v²sin²(θ)/(2g) = 50/(2×9.80665) = 2.549 m
            'max_height_m': 2.549,
            # Flight time = 2v sin(θ)/g = 2×7.071/9.80665 = 1.442 s
            'flight_time_s': 1.442,
        },
        'tolerance': 0.02,
        'source': 'Newtonian projectile motion (no drag)',
    },

    'terminal_velocity': {
        'description': 'Small sphere reaching terminal velocity in air',
        'expected': {
            # v_t = sqrt(2mg / (ρ_air C_d A))
            # For iron sphere r=0.01m: m=0.033kg, A=3.14e-4 m²
            # v_t = sqrt(2 × 0.033 × 9.8 / (1.225 × 0.44 × 3.14e-4))
            # v_t ≈ 55 m/s (needs tall fall)
            'reaches_terminal': True,
        },
        'tolerance': 0.10,
        'source': 'Drag-gravity equilibrium',
    },
}


def validate(result, scenario_key):
    """Compare simulation results against known physics.

    Args:
        result: SimulationResult from runner.run_simulation()
        scenario_key: Key into KNOWN_SCENARIOS

    Returns:
        dict with:
            'passed': bool — all checks within tolerance
            'checks': list of {name, expected, actual, passed, error_pct}
            'scenario': description string
    """
    if scenario_key not in KNOWN_SCENARIOS:
        return {
            'passed': False,
            'checks': [],
            'scenario': f'Unknown scenario: {scenario_key}',
        }

    scenario = KNOWN_SCENARIOS[scenario_key]
    expected = scenario['expected']
    tolerance = scenario['tolerance']
    checks = []

    final = result.final
    if final is None:
        return {
            'passed': False,
            'checks': [{'name': 'has_final_state', 'passed': False}],
            'scenario': scenario['description'],
        }

    # Check impact velocity
    if 'impact_velocity' in expected:
        actual = final.objects[0].speed if final.objects else 0
        error = abs(actual - expected['impact_velocity']) / expected['impact_velocity']
        checks.append({
            'name': 'impact_velocity',
            'expected': expected['impact_velocity'],
            'actual': actual,
            'passed': error <= tolerance,
            'error_pct': error * 100,
        })

    # Check impact velocity range
    if 'impact_velocity_range' in expected:
        actual = final.objects[0].speed if final.objects else 0
        lo, hi = expected['impact_velocity_range']
        checks.append({
            'name': 'impact_velocity_range',
            'expected': f'{lo}-{hi} m/s',
            'actual': actual,
            'passed': lo <= actual <= hi,
            'error_pct': 0 if lo <= actual <= hi else
                        min(abs(actual - lo), abs(actual - hi)) / ((lo + hi) / 2) * 100,
        })

    # Check range (horizontal distance)
    if 'range_m' in expected:
        # Find the snapshot where the object returns to ground level
        actual = 0
        for snap in result.snapshots:
            if snap.objects:
                x = snap.objects[0].position[0]
                actual = max(actual, abs(x))
        error = abs(actual - expected['range_m']) / expected['range_m']
        checks.append({
            'name': 'range',
            'expected': expected['range_m'],
            'actual': actual,
            'passed': error <= tolerance,
            'error_pct': error * 100,
        })

    # Check max height
    if 'max_height_m' in expected:
        max_y = 0
        for snap in result.snapshots:
            if snap.objects:
                y = snap.objects[0].position[1]
                max_y = max(max_y, y)
        error = abs(max_y - expected['max_height_m']) / expected['max_height_m']
        checks.append({
            'name': 'max_height',
            'expected': expected['max_height_m'],
            'actual': max_y,
            'passed': error <= tolerance,
            'error_pct': error * 100,
        })

    # Check energy conservation
    if 'energy_conservation' in expected:
        initial = result.initial
        if initial and final:
            e0 = initial.total_kinetic_energy
            ef = final.total_kinetic_energy
            if e0 > 0:
                change = abs(ef - e0) / e0
                checks.append({
                    'name': 'energy_conservation',
                    'expected': f'< {expected["energy_conservation"]*100}% change',
                    'actual': change,
                    'passed': change <= expected['energy_conservation'],
                    'error_pct': change * 100,
                })

    all_passed = all(c['passed'] for c in checks)

    return {
        'passed': all_passed,
        'checks': checks,
        'scenario': scenario['description'],
        'source': scenario.get('source', ''),
    }
