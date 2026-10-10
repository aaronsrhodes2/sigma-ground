"""Append the verdict block (pre-registered criteria applied as fixed, design corrections, engine defects) to u11_results.json."""
import json, os
D = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11"
R = json.load(open(os.path.join(D, 'u11_results.json')))
runs = R['runs']
cpu_s = sum(v['wall_s'] for v in runs.values())


def f(name):
    return runs[name]['final']


R['summary'] = dict(
    outcome='MIXED',
    compute_seconds_all_runs=cpu_s,
    pre_registered_SPH_verdict=dict(
        criterion='|H_fit/H-1|<=0.10 AND rms<=0.05 AND KE<1e-4 of peak (all three)',
        design_verbatim_A='FAIL on all three legs (H_fit/H = %.0f, KE never decreases: dt = cfl_timestep = %.3g, total time %.3g H/c_s)' % (
            f('A_literal')['H_fit_over_H'], runs['A_literal']['dt_first'], runs['A_literal']['total_time']),
        design_geometry_cs_cfl_B='FAIL: no column forms; lateral std %.1f H at t = %.1f, all parcels on the floor, no fit possible' % (
            f('B_literal_cs')['lateral_std'], runs['B_literal_cs']['total_time']),
        corrected_line_h0p1_C1='H leg PASS at final (%.3f), rms PASS (%.3f), KE FAIL (%.1e); at the 2000-step budget H FAIL (%.3f)' % (
            f('C1_line_N128_h0.1')['H_fit_over_H'], f('C1_line_N128_h0.1')['rms'], f('C1_line_N128_h0.1')['ke_over_peak'],
            runs['C1_line_N128_h0.1']['at_step_budget']['H_fit_over_H']),
        corrected_line_h0p2_C2='H leg PASS (%.4f), rms PASS (%.4f), KE FAIL (%.1e); at the 2000-step budget H %.3f' % (
            f('C2_line_N128_h0.2')['H_fit_over_H'], f('C2_line_N128_h0.2')['rms'], f('C2_line_N128_h0.2')['ke_over_peak'],
            runs['C2_line_N128_h0.2']['at_step_budget']['H_fit_over_H']),
        every_run_strict_PASS=any(v['final']['PASS'] for v in runs.values()),
    ),
    controls=R['controls'],
    post_hoc_diagnostics=dict(
        ground_contact_KE_share={k: v['final'].get('ground_contact_KE_share') for k, v in runs.items() if v['final'].get('ground_contact_KE_share') is not None},
        free_atmosphere_KE_min_over_peak={k: v['min_ke_free_over_peak'] for k, v in runs.items() if v['min_ke_free_over_peak'] is not None},
        second_half_H_fit_over_H={k: v['second_half_H_fit_over_H'] for k, v in runs.items()},
        cumulative_mass_H_fit_over_H={k: v['final'].get('cummass_H_fit_over_H') for k, v in runs.items()},
        merged_parcels={k: [v['final'].get('largest_merged_clump'), v['final'].get('parcels_in_merged_clumps')] for k, v in runs.items()},
        E2_alpha5='killed at step ~2100 as invalid: dt = 0.3 h / c_s omits the viscous signal speed (~(1 + 1.2 alpha) c_s), so alpha = 5 ran ~7x over the viscous CFL limit and heated the column (H_fit/H 1.8-2.9); no result drawn from it',
    ),
    engine_defects=[
        dict(file='sigma_ground/dynamics/continuum.py', line=936,
             defect="cfl_timestep uses each parcel's material sound speed (p.sound_speed) even when scene.eos is 'isothermal' or 'ideal'; continuum_step itself uses scene.sound_speed_m_s for those EOSs (line 741)",
             evidence='design run A: water_ice material c = 4354 m/s gives dt = 6.89e-6 for a c_s = 1 gas (0.3 h / c_s = 0.03, 4354x larger); 2000 steps cover 0.014 H/c_s and the gas has not left free fall'),
        dict(file='sigma_ground/dynamics/collision.py', line=286,
             defect='continuum_step resolves SPH ground contact with the rigid-body resolve_sphere_plane: velocity impulse with restitution 0.5 plus a position correction y -> y + 0.8 (r - y - 0.001) that pulls every penetrating parcel toward one height; resting fluid on the floor never comes to rest, and in a planar column the corrected parcels become bitwise coincident',
             evidence='no run reaches KE < 1e-4 of peak (floors 7e-4 to 4.5e-3); where per-parcel KE was saved, 99.9-99.99 % of the residual KE sits in parcels within 2 r of the floor once the column is quiet (E0 = C2 replica, C8, 3-D D), 98 % in the h/H 0.1 replica E1 and 20 % in C7, which was still oscillating; the free atmosphere of E0 falls to 4e-9 of peak;C1/C2: 19 and 33 parcels at bitwise-identical y = 0.01874 / 0.01842; 3-D: 79 of 200 parcels in the floor layer'),
        dict(file='sigma_ground/dynamics/fluid/kernel.py', line=141,
             defect='grad_W returns (0,0,0) for r < L_PLANCK, so coincident parcels exert no pressure force on each other and never separate; together with the cubic spline pairing instability at high neighbour number (Dehnen & Aly 2012, arXiv:1204.2471) parcels in the dense lower column collapse into near-coincident pairs and clumps',
             evidence='C1 final: clumps at y = 0.1393 with gaps 3e-17; C2: gaps 1e-11 to 1e-10 between paired parcels; up to 49 of 256 parcels in merged clumps (C3)'),
    ],
    design_corrections=[
        'cfl_timestep is not the isothermal CFL: the design\'s "2000 steps at cfl_timestep (80 s)" covers 0.014 H/c_s (dt = 6.9e-6); corrected runs use 0.3 h / max(c_s, v_max), the engine formula with the isothermal c_s',
        'the 1 x 4 x 1 H box has no side walls: the gas spreads sideways (lateral std 25.9 H by t = 30) and never forms a column; corrected by a planar column (all parcels on x = z = 0, laterally confined by symmetry, unmodified engine) and by a 3-D lateral-periodic image tiling',
        'N = 128 in a 1 x 4 x 1 box at h = 0.1 H gives a mean of 0.94 neighbours: the SPH density is the self-term m W(0); 3-D at h/H 0.1-0.2 needs N ~ 1e3-1e4 per H^2 of footprint (9x that with periodic images), beyond pure-Python cost',
        'the predicted resolution trend is reversed: at fixed N a larger h means more neighbours, so h/H = 0.2 is the clean case (1.002) and h/H = 0.1 the marginal one (oscillating +/-4 %, pairing); h/H = 0.05 fails (1.27, 2 neighbours at the window top)',
        'the KE < 1e-4 settle criterion cannot be met with a GroundPlane floor: the parcels resting on it chatter; measured on the free atmosphere only (post hoc) the h/H = 0.2 column reaches 4e-9 of peak',
        'my scaling control (fixed absolute h, g in 0.5/1/2) confounds g with h/H: slope -1.17 fails because the g = 0.5 point is under-resolved, not because H deviates from c_s^2/g (its cumulative-mass H is 0.97 H)',
        'design statement "w ~ 3e-12" is 3P/(rho c^2) = 2.8e-12; P/(rho c^2) itself is 9.4e-13 (Mentat)',
    ],
)
json.dump(R, open(os.path.join(D, 'u11_results.json'), 'w'), indent=1)
print(json.dumps(R['summary'], indent=1, default=str)[:6000])
