"""Append the v2 verdict block (original pre-registered criteria applied unchanged, corrected engine defects, design
corrections and the repair log) to u11_results_v2.json. Numbers are read from the collected results, not typed in."""
import json
D = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11"
R = json.load(open(D + '/u11_results_v2.json'))
T, C, S, DT, K, F = R['runs'], R['controls'], R['seed_summary'], R['dt_scaling'], R['kernel'], R['formula']
f = lambda n: T[n]['final']
fv2 = F['v2']

R['summary'] = dict(
    outcome='MIXED',
    pre_registered_SPH_verdict=dict(
        criterion='prereg.json, unchanged: final state |H_fit/H-1|<=0.10 AND rms<=0.05 AND KE<1e-4 of peak',
        any_pre_registered_run_PASS=any(f(n)['PASS'] for n in T if not n.startswith('R')),
        design_verbatim_A='FAIL all three legs: H_fit/H = %.1f, KE/peak %.2f (dt = cfl_timestep = %.3g, t_end = %.3g H/c_s)' % (
            f('A_literal')['H_fit_over_H'], f('A_literal')['ke_over_peak'], T['A_literal']['dt_first'], T['A_literal']['total_time']),
        design_geometry_B='FAIL: no column, lateral std %.2f H at t = %.1f, mean neighbours at step 1 %.3f, no fit' % (
            f('B_literal_cs')['lateral_std'], T['B_literal_cs']['total_time'], T['B_literal_cs']['mean_nnb_step1']),
        line_h0p2_seeds='H and rms PASS, KE FAIL in all %d seeds: H_fit/H %.5f-%.5f, KE/peak %.2e-%.2e' % (
            S['n'], S['min'], S['max'], min(v['ke_over_peak'] for v in R['seeds'].values()), max(v['ke_over_peak'] for v in R['seeds'].values())),
        line_h0p1_C1='H PASS (%.4f), rms PASS (%.4f), KE FAIL (%.2e); at 2000 steps H FAIL (%.4f)' % (
            f('C1_line_N128_h0.1')['H_fit_over_H'], f('C1_line_N128_h0.1')['rms'], f('C1_line_N128_h0.1')['ke_over_peak'],
            T['C1_line_N128_h0.1']['at_step_budget']['H_fit_over_H']),
        three_D_D='FAIL: H_fit/H %.4f (second half %.4f +- %.4f), rms %.3f, KE/peak %.2e' % (
            f('D_periodic3d_h0.4')['H_fit_over_H'], T['D_periodic3d_h0.4']['second_half_H_fit_over_H']['mean'],
            T['D_periodic3d_h0.4']['second_half_H_fit_over_H']['std'], f('D_periodic3d_h0.4')['rms'], f('D_periodic3d_h0.4')['ke_over_peak']),
        post_hoc_dt0p25_R3='formal three-leg PASS under the prereg stop rule (step %d, t %.1f, KE/peak %.3e, H_fit/H %.4f) - but the stop caught an oscillation node; continued to t %.0f the total KE hovers across the line (final %.3e, min %.3e). Post hoc, not a pre-registered run.' % (
            DT['0.25']['steps'], DT['0.25']['t_end'], DT['0.25']['ke_over_peak'], DT['0.25']['H_fit_over_H'],
            DT['0.25_long']['t_end'], DT['0.25_long']['ke_over_peak'], DT['0.25_long']['total_KE']['min']),
    ),
    controls=dict(
        i_scaling='FAIL (slope %.3f); confounded design, see design_corrections' % C['i_scaling']['slope'],
        ii_dimensional='PASS %.6f [BY CONSTRUCTION]' % C['ii_dimensional']['ratio_C6_over_C1'],
        iii_initial_condition='PASS on final snapshots (dH %+.4f, %+.4f) but low power: second-half std up to %.3f; second-half means %.3f, %.3f, %.3f' % (
            C['iii_initial_condition']['dH_final']['C7_line_expH0.5'], C['iii_initial_condition']['dH_final']['C8_line_expH2'],
            max(v['std'] for v in C['iii_initial_condition']['second_half_means'].values()),
            *[v['mean'] for v in C['iii_initial_condition']['second_half_means'].values()]),
        iv_hydrostatic='FAIL overall against its pre-registered rms |ratio-1| <= 0.05; passes in %s' % ', '.join(C['iv_hydrostatic']['passing']),
    ),
    engine_defects=[
        dict(file='sigma_ground/dynamics/continuum.py', line=936,
             line_note='936 at HEAD 0fa3d15 (952 in the current working copy, after another session added docstrings only at 04:36)',
             defect="cfl_timestep takes c_max from p.sound_speed (the parcel material's sound speed) even for eos='isothermal' or 'ideal'; continuum_step uses scene.sound_speed_m_s or sqrt(gamma R T/M) for those (HEAD line 741, now 757). Over-conservative: dt too small, stable but slow; unsafe only when the material c is below the EOS c_s.",
             evidence='run A: dt = %.4g vs 0.3 h/c_s = 0.03, a factor %.0f in the scaled units (c_s = 1); for a physical air column at 288.15 K the factor would be %.2f (water_ice %.2f m/s vs Mentat air %.2f m/s)' % (
                 T['A_literal']['dt_first'], fv2['cfl_defect_size']['mismatch_scaled_cs1'], fv2['cfl_defect_size']['mismatch_physical_air'],
                 fv2['cfl_defect_size']['material_sound_speed_water_ice'], fv2['cfl_defect_size']['air_sound_speed_288K'])),
        dict(file='sigma_ground/dynamics/collision.py', line=288,
             line_note='position correction in resolve_sphere_plane, called by continuum_step (continuum.py HEAD line 875, now 891)',
             defect='SPH floor contact uses the rigid-body resolve_sphere_plane: restitution impulse (e = 0.5) plus an 80 % position correction toward y = r - 0.001. Fluid resting on the floor merges into one bitwise-coincident clump that bounces every step in a restitution limit cycle v_out = e N g dt / ((1+e) N_f); its KE scales as dt^2 / N_f and sets the KE floor of every quiet run.',
             evidence='C2: clump vy %.6f vs predicted %.6f; floor KE/peak %.4e vs %.4e. dt x0.5: %.4e vs %.4e. dt x0.25 (to t 292): %.4e vs %.4e. All 5 h/H 0.2 seeds: |vy/pred - 1| <= 1.3e-4. Bitwise-coincident floor clumps of 10-36 parcels in every line run; 79/200 parcels in the floor layer of the 3-D run.' % (
                 DT['1.0']['limit_cycle']['vy_measured_mean'], DT['1.0']['limit_cycle']['vy_predicted'],
                 DT['1.0']['limit_cycle']['floor_KE_over_peak_measured'], DT['1.0']['limit_cycle']['floor_KE_over_peak_predicted'],
                 DT['0.5']['limit_cycle']['floor_KE_over_peak_measured'], DT['0.5']['limit_cycle']['floor_KE_over_peak_predicted'],
                 DT['0.25_long']['limit_cycle']['floor_KE_over_peak_measured'], DT['0.25_long']['limit_cycle']['floor_KE_over_peak_predicted'])),
        dict(file='sigma_ground/dynamics/fluid/kernel.py', line=151,
             line_note='grad_W, cubic-spline branch dW/dq = -2q + 1.5q^2 (the L_PLANCK guard at line 141 is correct and is not the cause)',
             defect='Observed artefact, mechanism not established: parcels above the floor collapse into near-coincident clusters (gaps 0 to ~2e-7 H) that grow with time. The M4 repulsion falls for q < 2/3 and vanishes as q -> 0, so captured parcels feel almost no pressure force and the viscosity removes their relative velocity [SPECULATIVE mechanism]. It is NOT the Dehnen & Aly linear pairing instability: the line chain uses the kernel profile W(|y|), whose 1-D Fourier transform (sin(kh/2)/(kh/2))^4 is >= 0 (Mentat). D&A applies to the 3-D run only (slab transform negative for kh in (2 pi, 8.99) and (4 pi, 15.45)). Exactly coincident parcels on a line feel identical forces and cannot separate under any central kernel; there the guard only avoids 0/0. Kernel choice is census A7 (Captain\'s call); a Wendland kernel is not shown to cure this, since its gradient also vanishes at r = 0.',
             evidence='above-floor cluster parcels at step 2000 -> end: C1 %d -> %d, C2 %d -> %d, C3 %d -> %d; all clusters sit below 0.5 H (none in the fit window); total clustered fraction (floor + above) %.0f %% (C2), %.0f-%.0f %% over the five h/H 0.2 seeds, %.0f-%.0f %% at h/H 0.1' % (
                 R['clump_growth_above_floor']['C1_line_N128_h0.1']['at_2000'], R['clump_growth_above_floor']['C1_line_N128_h0.1']['final'],
                 R['clump_growth_above_floor']['C2_line_N128_h0.2']['at_2000'], R['clump_growth_above_floor']['C2_line_N128_h0.2']['final'],
                 R['clump_growth_above_floor']['C3_line_N256_h0.1']['at_2000'], R['clump_growth_above_floor']['C3_line_N256_h0.1']['final'],
                 100 * R['clump_census']['C2_line_N128_h0.2']['fraction'],
                 100 * min(R['clump_census'][v['name']]['fraction'] for v in R['seeds'].values()),
                 100 * max(R['clump_census'][v['name']]['fraction'] for v in R['seeds'].values()),
                 100 * min(R['clump_census'][n]['fraction'] for n in ('C1_line_N128_h0.1', 'C3_line_N256_h0.1', 'C7_line_expH0.5', 'C8_line_expH2')),
                 100 * max(R['clump_census'][n]['fraction'] for n in ('C1_line_N128_h0.1', 'C3_line_N256_h0.1', 'C7_line_expH0.5', 'C8_line_expH2')))),
    ],
    hydrostatic_normalisation=dict(
        predicted_ratio_closed_form='(sinh(a/2)/(a/2))^4, a = h/H (Mentat integrate_expr; closure to the closed form = 0)',
        predicted={k: v['closed_form'] for k, v in K['hydrostatic_ratio_prediction'].items()},
        measured_h0p2_seed_means=S['hydrostatic_means'], C2_minus_pred=C['iv_hydrostatic']['per_run']['C2_line_N128_h0.2']['mean_minus_pred'],
        C5_minus_pred=C['iv_hydrostatic']['per_run']['C5_line_g2']['mean_minus_pred']),
    formula=dict(crossing_5pct_m=fv2['crossing_5pct']['z_m'], v1_interpolated_m=fv2['crossing_5pct']['v1_linear_interpolation_m'],
                 w_288K=fv2['sigma_column']['w_288K'], tolman_over_scale_height=fv2['sigma_column']['tolman_over_scale_height'],
                 w_with_20C_density=fv2['sigma_column']['w_with_20C_density'], identity=fv2['sigma_column']['identity_w_minus_gH_over_c2']),
)
json.dump(R, open(D + '/u11_results_v2.json', 'w'), indent=1, default=float)
print(json.dumps(R['summary'], indent=1, default=str))
