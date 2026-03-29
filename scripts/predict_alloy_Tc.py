#!/usr/bin/env python3
"""
Batch alloy T_c predictions — blind predictions + literature validation.

Runs all mixing models for every alloy in the ALLOYS_PREDICTABLE database,
compares against known measured values from literature, and outputs
structured JSON for visualization.

Output: docs/alloy_predictions.json

Usage:
    cd /path/to/sigma-ground
    python scripts/predict_alloy_Tc.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigma_ground.field.interface.alloys import (
    predict_all,
    composition_sweep,
    ALLOYS_PREDICTABLE,
)

# ── Literature validation data ────────────────────────────────────
# Measured T_c values from published sources.
# Sources: Roberts NBS Tech Note 983 (1978), Matthias et al.,
#          Testardi (1975), ASM International, review literature.
#
# Format: {alloy_key: {"T_c_measured_K": float, "source": str, "notes": str}}
# Only entries with reliable literature data are included.

MEASURED = {
    'NbTi_wire': {
        'T_c_measured_K': 9.5,
        'source': 'Roberts NBS 983 (1978); commercial wire spec',
        'notes': 'Nb-47at%Ti. Range 9.2-9.8K depending on processing.',
    },
    'NbTi_80_20': {
        'T_c_measured_K': 9.8,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Nb-20at%Ti. T_c peaks near Nb-rich compositions.',
    },
    'NbZr_75_25': {
        'T_c_measured_K': 10.8,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Nb-25at%Zr. Early SC wire material before NbTi.',
    },
    'NbTa_50_50': {
        'T_c_measured_K': 6.0,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Nb-50at%Ta. T_c suppressed by Ta dilution.',
    },
    'PbSn_eutectic': {
        'T_c_measured_K': 7.05,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Pb-37at%Sn eutectic solder.',
    },
    'PbIn_60_40': {
        'T_c_measured_K': 6.6,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Pb-40at%In.',
    },
    'PbTl_80_20': {
        'T_c_measured_K': 5.8,
        'source': 'Roberts NBS 983 (1978)',
        'notes': 'Pb-20at%Tl.',
    },
    'MoRe_50_50': {
        'T_c_measured_K': 12.4,
        'source': 'Testardi et al. (1975); Geballe (1965)',
        'notes': 'Mo-50at%Re. Anomalously high — σ-phase enhancement.',
    },
    'MoRe_60_40': {
        'T_c_measured_K': 10.6,
        'source': 'Geballe et al. (1965)',
        'notes': 'Mo-40at%Re.',
    },
    'CuZn_brass': {
        'T_c_measured_K': 0.0,
        'source': 'Well established — brass is non-superconducting',
        'notes': 'Cu-30at%Zn. Neither element is SC; alloy is not.',
    },
    'CuSn_bronze': {
        'T_c_measured_K': 0.0,
        'source': 'Well established — bronze is non-superconducting',
        'notes': 'Cu-12at%Sn. Cu dominates; tin fraction too small.',
    },
    'Nb3Sn_stoich': {
        'T_c_measured_K': 18.3,
        'source': 'Matthias et al. (1954); NIST',
        'notes': 'A15 compound. Crystal structure enhances λ beyond solid-solution prediction.',
    },
}


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'docs'
    output_dir.mkdir(exist_ok=True)

    # Run all predictions
    print("Running alloy T_c predictions...")
    predictions = predict_all()

    # Add measured data where available
    for name, result in predictions.items():
        if name in MEASURED:
            result['measured'] = MEASURED[name]
        else:
            result['measured'] = None

    # Composition sweep for NbTi system
    print("Running NbTi composition sweep...")
    nbti_sweep = composition_sweep('niobium', 'titanium', steps=41)

    # Summary statistics
    validated = {k: v for k, v in predictions.items() if v['measured'] is not None}
    sc_validated = {k: v for k, v in validated.items()
                    if v['measured']['T_c_measured_K'] > 0}
    non_sc_validated = {k: v for k, v in validated.items()
                        if v['measured']['T_c_measured_K'] == 0}

    # Compute errors for SC alloys
    errors = []
    for k, v in sc_validated.items():
        measured = v['measured']['T_c_measured_K']
        predicted = v['summary']['T_c_mean_K']
        error = predicted - measured
        pct_error = (error / measured * 100) if measured > 0 else 0
        errors.append({
            'alloy': k,
            'measured': measured,
            'predicted': predicted,
            'error_K': round(error, 2),
            'pct_error': round(pct_error, 1),
        })

    if errors:
        abs_errors = [abs(e['error_K']) for e in errors]
        mae = sum(abs_errors) / len(abs_errors)
        rmse = (sum(e ** 2 for e in abs_errors) / len(abs_errors)) ** 0.5
    else:
        mae = rmse = 0

    output = {
        'predictions': predictions,
        'nbti_sweep': nbti_sweep,
        'measured_data': MEASURED,
        'validation': {
            'total_predictions': len(predictions),
            'validated_against_literature': len(validated),
            'unvalidated_blind': len(predictions) - len(validated),
            'sc_validated': len(sc_validated),
            'non_sc_validated': len(non_sc_validated),
            'errors': sorted(errors, key=lambda e: abs(e['pct_error'])),
            'mae_K': round(mae, 2),
            'rmse_K': round(rmse, 2),
        },
    }

    output_path = output_dir / 'alloy_predictions.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"ALLOY T_c PREDICTIONS")
    print(f"{'='*60}")
    print(f"Total predictions:         {len(predictions)}")
    print(f"Validated (literature):     {len(validated)}")
    print(f"Blind (no literature):      {len(predictions) - len(validated)}")
    print(f"MAE:                        {mae:.2f} K")
    print(f"RMSE:                       {rmse:.2f} K")
    print()

    print(f"{'Alloy':<20} {'Predicted':>9} {'Measured':>9} {'Error':>8} {'Pct':>7}")
    print('-' * 55)
    for e in sorted(errors, key=lambda x: -abs(x['pct_error'])):
        print(f"{e['alloy']:<20} {e['predicted']:9.2f} {e['measured']:9.2f} "
              f"{e['error_K']:+8.2f} {e['pct_error']:+6.1f}%")

    print(f"\n{'='*60}")
    print("BLIND PREDICTIONS (no literature data):")
    print(f"{'='*60}")
    blind = {k: v for k, v in predictions.items() if v['measured'] is None}
    for name, result in sorted(blind.items(),
                                key=lambda x: -x[1]['summary']['T_c_mean_K']):
        s = result['summary']
        print(f"  {name:<20} T_c = {s['T_c_mean_K']:.2f} K "
              f"(range {s['T_c_min_K']:.2f}-{s['T_c_max_K']:.2f}, "
              f"confidence: {s['confidence']})")

    print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()
