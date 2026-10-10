"""U11 v2 collector: applies the ORIGINAL pre-registered criteria (prereg.json, unchanged) to the v2 runs, adds the
diagnostics the skeptic review asked for, checks the post-hoc predictions of prereg_v2.json, writes
u11_results_v2.json and u11_figure_v2.png.

Inputs: v2/run_*_v2.json (u11_sph_v2.py: C1..C8 re-runs, seeds 4/5/7/11, dt-scaling R3), the v1 runs that were not
re-run (A_literal, B_literal_cs, D_periodic3d_h0.4; reproduced bitwise by the reproduce skeptic), formula_results.json
(v1), formula_v2.json and kernel_v2.json (Mentat).
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11"
V2 = D + '/v2'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#8b5cf6']
MINUS = '\u2212'
E_REST = 0.5            # GroundPlane default restitution 0.50 = ContinuumParcel restitution 0.5 (min of the two)
R_PARCEL = 0.02         # parcel radius in every run (driver constant)
CLUMP_TOL = 1e-6        # near-coincident: gap < 1e-6 h (skeptic census definition)

form = json.load(open(D + '/formula_results.json'))
form2 = json.load(open(D + '/formula_v2.json'))
kern = json.load(open(D + '/kernel_v2.json'))
K_line_h2 = form['K_line_times_h2']
PRED = {float(k): v['closed_form'] for k, v in kern['hydrostatic_ratio_prediction'].items()}

runs = {}
for p in sorted(glob.glob(V2 + '/run_*_v2.json')):
    runs[os.path.basename(p)[4:-8]] = json.load(open(p))
for name in ('A_literal', 'B_literal_cs', 'D_periodic3d_h0.4'):
    runs[name] = json.load(open(f'{D}/run_{name}.json'))


# ---------------- R1: bitwise check of the v2 re-runs against v1 ----------------
def leaves_of(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves_of(v, f'{prefix}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves_of(v, f'{prefix}[{i}]')
    else:
        yield prefix, obj


bitwise = {}
for name in ('C1_line_N128_h0.1', 'C2_line_N128_h0.2', 'C3_line_N256_h0.1', 'C4_line_g0.5', 'C5_line_g2',
             'C6_line_cs1.4142_g2', 'C7_line_expH0.5', 'C8_line_expH2'):
    if name not in runs:
        continue
    v1 = json.load(open(f'{D}/run_{name}.json'))
    v2 = runs[name]
    a = dict(leaves_of({k: v1[k] for k in ('H', 'parcel_mass', 'ke_peak', 'dt_first', 'dt_last', 'total_time', 'steps',
                                          'history', 'at_step_budget', 'final')}))
    b = dict(leaves_of({k: v2[k] for k in ('H', 'parcel_mass', 'ke_peak', 'dt_first', 'dt_last', 'total_time', 'steps',
                                          'history', 'at_step_budget', 'final')}))
    shared = [k for k in a if k in b and '/wall_s' not in k]
    diff = [k for k in shared if not (a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]))]
    bitwise[name] = dict(shared_leaves=len(shared), differing=len(diff), first_diffs=diff[:5],
                         new_leaves_in_v2=len([k for k in b if k not in a]))


# ---------------- per-run diagnostics ----------------
def clump_census(ys, h):
    """near-coincident clusters (consecutive gaps < CLUMP_TOL h) along the sorted heights"""
    gaps = np.diff(ys)
    out, i = [], 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and gaps[j] < CLUMP_TOL * h:
            j += 1
        if j > i:
            out.append(dict(size=j - i + 1, y=float(ys[i]), max_gap=float(gaps[i:j].max()),
                            min_gap=float(gaps[i:j].min()), bitwise_equal=bool(ys[i] == ys[j])))
        i = j + 1
    return out


def first_and_stay_below(hist, key, thr=1e-4):
    vals = [(hh['t'], hh[key]) for hh in hist if key in hh]
    if not vals:
        return None
    first = next((t for t, v in vals if v < thr), None)
    stay = None
    for k in range(len(vals)):
        if all(v < thr for _, v in vals[k:]):
            stay = vals[k][0]
            break
    return dict(first_below_t=first, stays_below_from_t=stay, min=min(v for _, v in vals))


def verdict(r, state='final'):
    s = r['final'] if state == 'final' else r.get('at_step_budget')
    if s is None:
        return None
    f = s['fit']
    hf = f.get('H_fit_over_H', float('nan'))
    ok_H = bool(np.isfinite(hf) and abs(hf - 1) <= 0.10)
    ok_rms = bool(np.isfinite(f.get('rms', float('nan'))) and f['rms'] <= 0.05)
    ok_ke = bool(s['ke_over_peak'] < 1e-4)
    out = dict(step=s['step'], t=s['t'], H_fit_over_H=hf, rms=f.get('rms'), n_fit=f.get('n_fit'),
               ke_over_peak=s['ke_over_peak'], pass_H=ok_H, pass_rms=ok_rms, pass_KE=ok_ke,
               PASS=ok_H and ok_rms and ok_ke)
    if state != 'final':
        return out
    H, h, N, m = r['H'], r['args']['h'], r['args']['N'], r['parcel_mass']
    y = np.array(s['y']); o = np.argsort(y, kind='stable'); ys = y[o]
    sel = (ys > 0.8 * H) & (ys < 2.8 * H)
    if 'counted_fit' in s:
        out['counted_density_diagnostic_H_fit_over_H'] = s['counted_fit'].get('H_fit_over_H')
    if 'lateral_std' in s:
        out['lateral_std'] = s['lateral_std']
    nab = np.arange(len(ys))[::-1] + 0.5
    if sel.sum() >= 3:
        out['cummass_H_fit_over_H'] = float(-1.0 / np.polyfit(ys[sel], np.log(nab[sel]), 1)[0] / H)
    nnb = np.array(s['nnb'])[o]
    if sel.sum():
        out['nnb_window_bottom_top'] = [int(nnb[sel][0]), int(nnb[sel][-1])]
    # hydrostatic balance (pre-registered control iv) and the kernel-MGF prediction (Mentat, kernel_v2.json)
    if 'hydrostatic' in s and r['args']['mode'] == 'line':
        ratio = np.array(s['hydrostatic']['ratio_all'])[sel]
        pred = PRED.get(round(h / H, 6))
        out['hydrostatic'] = dict(ratio_mean=float(ratio.mean()), rms_dev_from_1=float(np.sqrt(np.mean((ratio - 1) ** 2))),
                                  pass_prereg=bool(np.sqrt(np.mean((ratio - 1) ** 2)) <= 0.05),
                                  predicted_mgf=pred,
                                  mean_minus_pred=None if pred is None else float(ratio.mean() - pred),
                                  rms_dev_from_pred=None if pred is None else float(np.sqrt(np.mean((ratio / pred - 1) ** 2))))
    elif 'hydrostatic' in s:
        out['hydrostatic'] = dict(ratio_mean=s['hydrostatic']['ratio_mean'], rms_dev_from_1=s['hydrostatic']['ratio_rms_dev'],
                                  pass_prereg=bool(s['hydrostatic']['ratio_rms_dev'] <= 0.05))
    if r['args']['mode'] == 'line':
        cl = clump_census(ys, h)
        floor = [c for c in cl if c['y'] < 2 * R_PARCEL]
        above = [c for c in cl if c['y'] >= 2 * R_PARCEL]
        out['clumps'] = dict(parcels_in_clusters=int(sum(c['size'] for c in cl)), fraction=float(sum(c['size'] for c in cl) / N),
                             floor_cluster=floor[0] if floor else None,
                             above_floor=[dict(size=c['size'], y_over_H=c['y'] / H, max_gap=c['max_gap'], min_gap=c['min_gap']) for c in above],
                             above_floor_parcels=int(sum(c['size'] for c in above)),
                             above_floor_in_window=int(sum(c['size'] for c in above if 0.8 * H < c['y'] < 2.8 * H)),
                             bitwise_identical_at_floor=int(np.sum(ys == ys[0])) if ys[0] < 2 * R_PARCEL else 0)
        # floor restitution limit cycle: v_out = e N g dt / ((1+e) N_f)
        if floor:
            Nf = floor[0]['size']
            idx = o[:Nf]
            vy = np.array(s['vy'])[idx]
            dt = r['dt_last']
            v_pred = E_REST * N * r['args']['g'] * dt / ((1 + E_REST) * Nf)
            lc = dict(N_f=Nf, dt=dt, vy_measured_mean=float(vy.mean()), vy_measured_spread=float(vy.max() - vy.min()),
                      vy_predicted=v_pred, vy_rel_dev=float(vy.mean() / v_pred - 1),
                      floor_KE_over_peak_predicted=0.5 * Nf * m * v_pred ** 2 / r['ke_peak'])
            if 'ke_parcels' in s:
                kp = np.array(s['ke_parcels'])
                lc['floor_KE_over_peak_measured'] = float(kp[idx].sum() / r['ke_peak'])
                lc['floor_KE_share_of_total'] = float(kp[idx].sum() / max(kp.sum(), 1e-300))
            out['floor_limit_cycle'] = lc
    if 'ke_parcels' in s:
        kp = np.array(s['ke_parcels'])
        out['ground_contact_KE_share'] = float(kp[y < 2 * R_PARCEL].sum() / max(kp.sum(), 1e-300))
        out['free_atmosphere_KE_over_peak_final'] = float(kp[y >= 2 * R_PARCEL].sum() / r['ke_peak'])
    if 'x' in s:
        pts = np.round(np.c_[np.array(s['x']), y, np.array(s['z'])], 10)
        u, cn = np.unique(pts, axis=0, return_counts=True)
        out['parcels_in_merged_clumps_3d'] = int(cn[cn > 1].sum())
        out['floor_layer_parcels'] = int(np.sum(y < 2 * R_PARCEL))
    return out


def second_half(r):
    hs = [hh['fit']['H_fit'] / r['H'] for hh in r['history'] if hh['step'] > r['steps'] / 2 and np.isfinite(hh['fit']['H_fit'])]
    return dict(mean=float(np.mean(hs)), std=float(np.std(hs)), n=len(hs)) if hs else dict(mean=float('nan'), std=float('nan'), n=0)


table = {}
for name, r in runs.items():
    a = r['args']
    table[name] = dict(source='v2/run_%s_v2.json' % name if name not in ('A_literal', 'B_literal_cs', 'D_periodic3d_h0.4') else f'run_{name}.json (v1)',
                       mode=a['mode'], N=a['N'], h=a['h'], g=a['g'], cs=a['cs'], seed=a['seed'], cfl=a.get('cfl', 0.3),
                       H=r['H'], h_over_H=a['h'] / r['H'], init=a['init'], dt_first=r['dt_first'], dt_last=r['dt_last'],
                       total_time=r['total_time'], steps=r['steps'], wall_s=r['wall_s'], ke_peak=r['ke_peak'],
                       cfl_timestep_engine=r['cfl_timestep_engine'], material_sound_speed=r['material_sound_speed'],
                       first_settle=r['first_settle'], second_half_H_fit_over_H=second_half(r),
                       free_KE=first_and_stay_below(r['history'], 'ke_free_over_peak'),
                       mean_nnb_step1=r['history'][0]['mean_nnb'],
                       at_step_budget=verdict(r, 'budget'), final=verdict(r, 'final'))

# ---------------- controls (pre-registered, unchanged) ----------------
ctrl = {}
def Hf(name):
    return table[name]['final']['H_fit_over_H'] * table[name]['H']
gs = [(0.5, 'C4_line_g0.5'), (1.0, 'C1_line_N128_h0.1'), (2.0, 'C5_line_g2')]
slope = float(np.polyfit(np.log([g for g, _ in gs]), np.log([Hf(n) for _, n in gs]), 1)[0])
ctrl['i_scaling'] = dict(slope=slope, PASS=bool(abs(slope + 1) <= 0.10 and all(abs(table[n]['final']['H_fit_over_H'] - 1) <= 0.10 for _, n in gs)),
                         note='confounds g with h/H at fixed absolute h (design correction); C4 is under-resolved, its cumulative-mass H is %.3f H' % table['C4_line_g0.5']['final']['cummass_H_fit_over_H'])
rr = Hf('C6_line_cs1.4142_g2') / Hf('C1_line_N128_h0.1')
ctrl['ii_dimensional'] = dict(ratio_C6_over_C1=rr, PASS=bool(abs(rr - 1) <= 0.02), tag='[BY CONSTRUCTION]')
dH = {n: Hf(n) - Hf('C1_line_N128_h0.1') for n in ('C7_line_expH0.5', 'C8_line_expH2')}
ctrl['iii_initial_condition'] = dict(dH_final=dH, PASS=bool(all(abs(v) <= 0.05 for v in dH.values())),
                                     second_half_means={n: table[n]['second_half_H_fit_over_H'] for n in ('C1_line_N128_h0.1', 'C7_line_expH0.5', 'C8_line_expH2')},
                                     low_power_note='the snapshots compared oscillate by more than the 0.05 tolerance (second-half std up to %.3f)' % max(
                                         table[n]['second_half_H_fit_over_H']['std'] for n in ('C1_line_N128_h0.1', 'C7_line_expH0.5', 'C8_line_expH2')))
hyd = {n: table[n]['final']['hydrostatic'] for n in table if table[n]['mode'] == 'line' and 'hydrostatic' in table[n]['final']}
ctrl['iv_hydrostatic'] = dict(per_run={n: dict(mean=v['ratio_mean'], rms_dev_from_1=v['rms_dev_from_1'], PASS=v['pass_prereg'],
                                               predicted=v['predicted_mgf'], mean_minus_pred=v['mean_minus_pred']) for n, v in hyd.items()},
                              PASS_overall=bool(all(v['pass_prereg'] for v in hyd.values())),
                              passing=[n for n, v in hyd.items() if v['pass_prereg']])

# ---------------- seeds (h/H 0.2, line, N 128) ----------------
seed_names = ['C2_line_N128_h0.2'] + sorted([n for n in table if n.startswith('R2_seed')], key=lambda s: int(s.split('_')[1][4:]))
seeds = {table[n]['seed']: dict(name=n, H_fit_over_H=table[n]['final']['H_fit_over_H'], rms=table[n]['final']['rms'],
                                ke_over_peak=table[n]['final']['ke_over_peak'], second_half=table[n]['second_half_H_fit_over_H'],
                                hydrostatic_mean=table[n]['final']['hydrostatic']['ratio_mean'],
                                hydrostatic_mean_minus_pred=table[n]['final']['hydrostatic']['mean_minus_pred'],
                                clump_fraction=table[n]['final']['clumps']['fraction'],
                                floor_KE_share=table[n]['final'].get('ground_contact_KE_share'),
                                prereg_v2_pass=bool(0.995 <= table[n]['final']['H_fit_over_H'] <= 1.010 and table[n]['final']['rms'] <= 0.005))
         for n in seed_names}
hv = np.array([v['H_fit_over_H'] for v in seeds.values()])
seed_summary = dict(n=len(hv), min=float(hv.min()), max=float(hv.max()), mean=float(hv.mean()), std=float(hv.std(ddof=1)),
                    hydrostatic_means=[v['hydrostatic_mean'] for v in seeds.values()],
                    hydrostatic_mean_of_means=float(np.mean([v['hydrostatic_mean'] for v in seeds.values()])),
                    predicted=PRED[0.2])

# ---------------- R3: dt scaling of the floor limit cycle ----------------
dtscale = {}
for key, f_, n in (('1.0', 1.0, 'C2_line_N128_h0.2'), ('0.5', 0.5, 'R3_dt0.5_line_N128_h0.2'), ('0.25', 0.25, 'R3_dt0.25_line_N128_h0.2'),
                   ('0.25_long', 0.25, 'R3c_dt0.25_long_line_N128_h0.2')):
    if n not in table:
        continue
    v = table[n]['final']
    lc = v.get('floor_limit_cycle')
    d = dict(name=n, dt_factor=f_, dt_last=table[n]['dt_last'], steps=table[n]['steps'], t_end=table[n]['total_time'],
             first_settle=table[n]['first_settle'], H_fit_over_H=v['H_fit_over_H'], rms=v['rms'], ke_over_peak=v['ke_over_peak'],
             prereg_three_legs=dict(H=v['pass_H'], rms=v['pass_rms'], KE=v['pass_KE']), free_KE=table[n]['free_KE'])
    if lc:
        d.update(limit_cycle=lc, model_b_prediction=1.578e-3 * f_ ** 2 * 33 / lc['N_f'])
        d['pred_a_pass'] = bool(abs(lc['vy_rel_dev']) <= 0.05)
        if 'floor_KE_over_peak_measured' in lc:
            d['pred_b_pass'] = bool(abs(lc['floor_KE_over_peak_measured'] / d['model_b_prediction'] - 1) <= 0.25)
    d['pred_c_pass'] = bool(abs(v['H_fit_over_H'] - 1) <= 0.10)
    d['total_KE'] = first_and_stay_below(runs[n]['history'], 'ke_over_peak')
    dtscale[key] = d

# ---------------- clump census across line runs ----------------
census = {n: dict(fraction=t['final']['clumps']['fraction'], parcels=t['final']['clumps']['parcels_in_clusters'],
                  floor=(t['final']['clumps']['floor_cluster'] or {}).get('size'),
                  bitwise_identical_at_floor=t['final']['clumps']['bitwise_identical_at_floor'],
                  above=[(c['size'], round(c['y_over_H'], 4)) for c in t['final']['clumps']['above_floor']],
                  above_gap_range=[min([c['min_gap'] for c in t['final']['clumps']['above_floor']] or [float('nan')]),
                                   max([c['max_gap'] for c in t['final']['clumps']['above_floor']] or [float('nan')])],
                  above_in_window=t['final']['clumps']['above_floor_in_window'])
          for n, t in table.items() if t['mode'] == 'line'}
# growth of the above-floor clusters between the 2000-step snapshot and the end (C1, C2)
growth = {}
for n in ('C1_line_N128_h0.1', 'C2_line_N128_h0.2', 'C3_line_N256_h0.1'):
    r = runs[n]
    s = r['at_step_budget']
    ys = np.sort(np.array(s['y']))
    cl = clump_census(ys, r['args']['h'])
    growth[n] = dict(at_2000=int(sum(c['size'] for c in cl if c['y'] >= 2 * R_PARCEL)),
                     final=table[n]['final']['clumps']['above_floor_parcels'])

mentat_leaves = {}
for src, L in (('formula_v1', form['mentat_leaves']), ('formula_v2', form2['mentat_leaves']), ('kernel_v2', kern['mentat_leaves'])):
    for k, v in L.items():
        mentat_leaves[f'{src}:{k}'] = v

results = dict(id='U11', version='v2 (repair round)', prereg_file=D + '/prereg.json', prereg_v2_file=D + '/prereg_v2.json',
               runs=table, controls=ctrl, seeds=seeds, seed_summary=seed_summary, dt_scaling=dtscale,
               clump_census=census, clump_growth_above_floor=growth, bitwise_v1_v2=bitwise,
               kernel=dict({k: v for k, v in kern.items() if k != 'mentat_leaves'}),
               formula=dict(verdicts=form['verdicts'], air=form['air'], isa_anchor_check=form['isa_anchor_check'], rows=form['rows'],
                            engine_polytrope=form['engine_polytrope'], sigma_column_v1=form['sigma_column'],
                            H_eq_cs2_over_g=form['H_eq_cs2_over_g'], K_line_times_h2=K_line_h2, kernel3d_norm=form['kernel3d_norm'],
                            v2={k: v for k, v in form2.items() if k != 'mentat_leaves'}),
               mentat_leaves=mentat_leaves)
json.dump(results, open(D + '/u11_results_v2.json', 'w'), indent=1, default=float)

# ---------------- figure ----------------
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.unicode_minus': True, 'font.family': 'DejaVu Sans'})
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.2), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
fmt_minus = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('-', MINUS))

# (a) final density profiles, normalised by the predicted continuum level K * MGF * (column mass / H)
ax = axs[0, 0]
zz = np.linspace(0, 4.5, 200)
ax.semilogy(zz, np.exp(-zz), color=INK, lw=1.4, label='isothermal hydrostatic e$^{' + MINUS + 'z/H}$, H = c$_s^2$/g')
series = [('C2_line_N128_h0.2', 'line chain h/H 0.2 (seed 3)', COL[1]),
          ('C1_line_N128_h0.1', 'line chain h/H 0.1', COL[0]),
          ('C4_line_g0.5', 'line chain h/H 0.05', COL[2]),
          ('D_periodic3d_h0.4', '3-D lateral-periodic h/H 0.4', COL[3])]
for name, lab, c in series:
    r = runs[name]
    y = np.array(r['final']['y']); rho = np.array(r['final']['rho'])
    if r['args']['mode'] == 'line':
        norm = (K_line_h2 / r['args']['h'] ** 2) * PRED[round(r['args']['h'] / r['H'], 6)] * r['args']['N'] * r['parcel_mass'] / r['H']
    else:
        norm = r['args']['N'] * r['parcel_mass'] / (r['L'] ** 2 * r['H'])
    v = table[name]['final']
    ax.semilogy(y / r['H'], rho / norm, 'o', ms=2.8, color=c, alpha=0.8,
                label=f"{lab}: H$_{{fit}}$/H = {v['H_fit_over_H']:.3f}")
ax.axvspan(0.8, 2.8, color=GRID, alpha=0.5, lw=0)
ax.text(1.8, 1.9, 'fit window 0.8–2.8 H', ha='center', color=INK2, fontsize=8)
ax.set_xlim(0, 4.5); ax.set_ylim(5e-3, 4)
ax.set_xlabel('height z / H'); ax.set_ylabel('ρ$_{SPH}$ / (K · MGF(h/H) · N m / H)')
ax.set_title('(a) final column; line chains scaled by K·(sinh(h/2H)/(h/2H))$^4$ (Mentat)', loc='left', fontsize=9.5)
ax.legend(frameon=False, fontsize=7.2, loc='lower left')

# (b) H_fit/H for every run
ax = axs[0, 1]
order = [('A_literal', 'design verbatim'), ('B_literal_cs', 'design geometry, c$_s$ CFL'),
         ('C4_line_g0.5', 'line h/H 0.05'), ('C1_line_N128_h0.1', 'line h/H 0.1'),
         ('C6_line_cs1.4142_g2', 'line h/H 0.1, c$_s$ √2, g 2'), ('C3_line_N256_h0.1', 'line h/H 0.1, N 256'),
         ('C7_line_expH0.5', 'line h/H 0.1, start 0.5H'), ('C8_line_expH2', 'line h/H 0.1, start 2H'),
         ('C2_line_N128_h0.2', 'line h/H 0.2, seed 3'), ('C5_line_g2', 'line h/H 0.2, g 2 (rescaled C2)')] + \
        [(n, f"line h/H 0.2, seed {table[n]['seed']}") for n in seed_names[1:]] + \
        [('R3_dt0.5_line_N128_h0.2', 'line h/H 0.2, dt ×0.5'), ('R3_dt0.25_line_N128_h0.2', 'dt ×0.25, stopped at t 129'),
         ('R3c_dt0.25_long_line_N128_h0.2', 'dt ×0.25, run to t 292'),
         ('D_periodic3d_h0.4', '3-D periodic h/H 0.4')]
ax.axhspan(0.9, 1.1, color=GRID, alpha=0.6, lw=0, label='pass band |H$_{fit}$/H ' + MINUS + ' 1| ≤ 0.10')
ax.axhline(1.0, color=INK, lw=1)
xs, labs = [], []
i = 0
for name, lab in order:
    if name not in table:
        continue
    v = table[name]['final']; val = v['H_fit_over_H']; th = table[name]['second_half_H_fit_over_H']
    if np.isfinite(val) and 0 < val < 2:
        ax.plot(i, val, 'o', color=COL[0], ms=6, zorder=3)
    else:
        ax.plot(i, 1.9, 'x', color=COL[1], ms=8, mew=2, zorder=3)
        ax.annotate('no fit' if not np.isfinite(val) else f'{val:.0f}'.replace('-', MINUS), (i, 1.9),
                    textcoords='offset points', xytext=(0, -14), ha='center', fontsize=6.8, color=INK2)
    if np.isfinite(th['mean']) and 0 < th['mean'] < 2:
        ax.errorbar(i + 0.25, th['mean'], yerr=th['std'], fmt='s', color=COL[2], ms=4, capsize=2, zorder=3)
    xs.append(i); labs.append(lab); i += 1
ax.plot([], [], 'o', color=COL[0], label='final state (pre-registered reading)')
ax.errorbar([], [], yerr=[], fmt='s', color=COL[2], label='mean ± std over the 2nd half (post hoc)')
ax.plot([], [], 'x', color=COL[1], label='design as written (off scale)')
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=45, ha='right', fontsize=6.8)
ax.set_ylim(0, 2.0); ax.set_ylabel('H$_{fit}$ / (c$_s^2$/g), ln ρ$_{SPH}$ over 0.8–2.8 H')
ax.set_title('(b) fitted scale height, every run', loc='left', fontsize=9.5)
ax.legend(frameon=False, fontsize=7.2, loc='lower right')

# (c) the settle leg: total KE vs the floor limit-cycle prediction for three time steps
ax = axs[1, 0]
for (key, f_, n), c in zip((('1.0', 1.0, 'C2_line_N128_h0.2'), ('0.5', 0.5, 'R3_dt0.5_line_N128_h0.2'),
                            ('0.25_long', 0.25, 'R3c_dt0.25_long_line_N128_h0.2')), COL[:3]):
    if n not in runs:
        continue
    hs = runs[n]['history']
    t = np.array([hh['t'] for hh in hs]); k = np.array([max(hh['ke_over_peak'], 1e-12) for hh in hs])
    ax.loglog(np.maximum(t, 1e-2), k, '-', color=c, lw=1.2, label=f'total KE, dt = {f_:g} × 0.3 h/c$_s$')
    lc = dtscale[key].get('limit_cycle')
    if lc:
        ax.axhline(lc['floor_KE_over_peak_predicted'], color=c, lw=1.1, ls=':')
        ax.text(1.6e-2, lc['floor_KE_over_peak_predicted'] * (1.25 if f_ > 0.3 else 0.55),
                f"floor limit cycle, dt × {f_:g}: {lc['floor_KE_over_peak_predicted']:.2e}".replace('e-0', 'e' + MINUS), fontsize=7, color=c)
if 'R3_dt0.25_line_N128_h0.2' in runs:
    r3 = runs['R3_dt0.25_line_N128_h0.2']
    ax.plot(r3['final']['t'], r3['final']['ke_over_peak'], 'v', color=COL[2], ms=8, mec=INK, zorder=4,
            label='dt × 0.25 run stopped by the pre-registered rule (oscillation node)')
hs = runs['C2_line_N128_h0.2']['history']
ax.loglog([max(hh['t'], 1e-2) for hh in hs], [max(hh['ke_free_over_peak'], 1e-12) for hh in hs], '-', color=COL[3], lw=1.2,
          label='dt × 1: KE above the floor layer (post hoc)')
ax.plot([], [], ':', color=INK2, label='floor-clump limit cycle ½N$_f$m v$^2$, v = eNg dt/((1+e)N$_f$)')
ax.axhline(1e-4, color=INK, lw=1, ls='--')
ax.text(2.0, 1.25e-4, 'pre-registered settle line: 10$^{' + MINUS + '4}$ of peak', fontsize=7.5, color=INK2)
ax.set_xlabel('time (H / c$_s$)'); ax.set_ylabel('kinetic energy / its peak')
ax.set_title('(c) the KE leg is set by the floor bounce, which scales as dt$^2$', loc='left', fontsize=9.5)
ax.set_ylim(1e-9, 3); ax.set_xlim(1e-2, 800)
ax.legend(frameon=False, fontsize=7.2, loc='lower left')

# (d) formula side vs ISA 1976
ax = axs[1, 1]
z = np.array([r_['z_m'] for r_ in form['rows']]) / 1000.0
ax.axhspan(-5, 5, color=GRID, alpha=0.6, lw=0, label='± 5 %')
ax.plot(z, [100 * r_['p_rel_err'] for r_ in form['rows']], 'o-', color=COL[0], label='pressure_at_altitude(z, 288.15 K), isothermal')
ax.plot(z, [100 * r_['T_rel_err'] for r_ in form['rows']], 's-', color=COL[1],
        label=f"temperature_at_altitude(z), dry adiabatic {form['air']['dry_lapse_engine_K_per_km']:.2f} K/km")
ax.plot(z, [100 * r_['rel_err'] for r_ in form['engine_polytrope']], '^-', color=COL[2], label='engine M$_{air}$ and R with the ISA 6.5 K/km lapse')
ax.axhline(0, color=INK, lw=1)
cr = form2['crossing_5pct']['z_m']
ax.axvline(cr / 1000, color=INK2, lw=0.8, ls=':')
ax.text(cr / 1000 + 0.15, 12, f'isothermal pressure\nleaves ±5 % at {cr/1000:.2f} km', fontsize=8, color=INK2)
ax.set_xlabel('geometric altitude (km)'); ax.set_ylabel('relative error vs US Standard Atmosphere 1976 (%)')
ax.set_title('(d) atmosphere.py vs ISA 1976 [LIT]', loc='left', fontsize=9.5)
ax.legend(frameon=False, fontsize=7.2, loc='lower left')
ax.yaxis.set_major_formatter(fmt_minus)

title = ('U11: a laterally confined SPH line chain relaxes to the barometric law H = c$_s^2$/g '
         f"(H$_{{fit}}$/H {seed_summary['min']:.4f}–{seed_summary['max']:.4f} over {seed_summary['n']} seeds at h/H 0.2), "
         'but no run passes the pre-registered three-leg criterion' + chr(10) +
         'the 3-D design never forms a column, the only 3-D run reads H$_{fit}$/H = '
         f"{table['D_periodic3d_h0.4']['final']['H_fit_over_H']:.2f}, and the KE leg fails because the floor clump bounces with KE ∝ dt$^2$" + chr(10) +
         f"atmosphere.py: isothermal pressure within 5 % of ISA 1976 only below {cr/1000:.2f} km; dry-adiabatic T(z) misses the 6.5 K/km lapse")
fig.suptitle(title, fontsize=9.6, color=INK, x=0.01, ha='left')
fig.subplots_adjust(left=0.06, right=0.985, top=0.885, bottom=0.115, hspace=0.62, wspace=0.17)
fig.savefig(D + '/u11_figure_v2.png', dpi=150, facecolor=SURF)

brief = dict(bitwise=bitwise, controls=ctrl, seed_summary=seed_summary, seeds=seeds, dt_scaling=dtscale, census=census, growth=growth,
             finals={k: {kk: v['final'].get(kk) for kk in ('H_fit_over_H', 'rms', 'ke_over_peak', 'cummass_H_fit_over_H', 'ground_contact_KE_share',
                                                          'free_atmosphere_KE_over_peak_final', 'hydrostatic', 'floor_limit_cycle')}
                     for k, v in table.items()},
             budget={k: (v['at_step_budget'] or {}).get('H_fit_over_H') for k, v in table.items()},
             second_half={k: v['second_half_H_fit_over_H'] for k, v in table.items()},
             free_KE={k: v['free_KE'] for k, v in table.items()},
             nnb1={k: v['mean_nnb_step1'] for k, v in table.items()})
print(json.dumps(brief, indent=1, default=str))
