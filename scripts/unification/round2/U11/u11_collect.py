"""Collect U11 runs, apply the pre-registered criteria, write u11_results.json and the figure."""
import json, math, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11"
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#8b5cf6']
MINUS = '\u2212'


def load(name):
    p = os.path.join(D, f'run_{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


form = json.load(open(os.path.join(D, 'formula_results.json')))
K_line_h2 = form['K_line_times_h2']
runs = {os.path.basename(p)[4:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(D, 'run_*.json')))}


def verdict(r, state='final'):
    s = r[state] if state == 'final' else r.get('at_step_budget')
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
    if 'counted_fit' in s:
        out['counted_H_fit_over_H'] = s['counted_fit'].get('H_fit_over_H')
    if 'hydrostatic' in s:
        out['hydrostatic_ratio_mean'] = s['hydrostatic']['ratio_mean']
        out['hydrostatic_ratio_rms_dev'] = s['hydrostatic']['ratio_rms_dev']
    if 'lateral_std' in s:
        out['lateral_std'] = s['lateral_std']
    # post-hoc diagnostics (not pre-registered criteria)
    H = r['H']
    y = np.array(s['y']); o = np.argsort(y); ys = y[o]
    nab = np.arange(len(ys))[::-1] + 0.5
    sel = (ys > 0.8 * H) & (ys < 2.8 * H)
    if sel.sum() >= 3:
        out['cummass_H_fit_over_H'] = float(-1.0 / np.polyfit(ys[sel], np.log(nab[sel]), 1)[0] / H)
    if 'x' in s:   # 3-D: a merged clump means coincident in all three coordinates
        pts = np.round(np.c_[np.array(s['x']), y, np.array(s['z'])], 10)
        u, cn = np.unique(pts, axis=0, return_counts=True)
    else:
        u, cn = np.unique(np.round(ys, 10), return_counts=True)
    out['largest_merged_clump'] = int(cn.max())
    out['parcels_in_merged_clumps'] = int(cn[cn > 1].sum())
    nnb = np.array(s['nnb'])[o]
    if sel.sum():
        out['nnb_window_bottom_top'] = [int(nnb[sel][0]), int(nnb[sel][-1])]
    if 'ke_parcels' in s:
        kp = np.array(s['ke_parcels'])[o]
        out['ground_contact_KE_share'] = float(kp[ys < 0.04].sum() / max(kp.sum(), 1e-300))
    return out


table = {}
for name, r in runs.items():
    table[name] = dict(mode=r['args']['mode'], N=r['args']['N'], h=r['args']['h'], g=r['args']['g'], cs=r['args']['cs'],
                       H=r['H'], h_over_H=r['args']['h'] / r['H'], init=r['args']['init'], dt_first=r['dt_first'],
                       dt_last=r['dt_last'], total_time=r['total_time'], steps=r['steps'], wall_s=r['wall_s'],
                       cfl_timestep_engine=r['cfl_timestep_engine'], material_sound_speed=r['material_sound_speed'],
                       first_settle=r['first_settle'], alpha=r['args'].get('alpha', 1.0),
                       second_half_H_fit_over_H=(lambda hs: dict(mean=float(np.mean(hs)), std=float(np.std(hs)), n=len(hs)))(
                           [hh['fit']['H_fit'] / r['H'] for hh in r['history'] if hh['step'] > r['steps'] / 2 and np.isfinite(hh['fit']['H_fit'])] or [float('nan')]),
                       min_ke_free_over_peak=(min(hh['ke_free_over_peak'] for hh in r['history'] if 'ke_free_over_peak' in hh)
                                              if any('ke_free_over_peak' in hh for hh in r['history']) else None),
                       at_step_budget=verdict(r, 'budget'), final=verdict(r, 'final'))

# controls
ctrl = {}
def hf(name):
    r = table.get(name)
    return None if r is None else r['final']['H_fit_over_H'] * r['H']
gs = [(0.5, 'C4_line_g0.5'), (1.0, 'C1_line_N128_h0.1'), (2.0, 'C5_line_g2')]
if all(hf(n) is not None for _, n in gs):
    lg = np.log([g for g, _ in gs]); lh = np.log([hf(n) for _, n in gs])
    slope = float(np.polyfit(lg, lh, 1)[0])
    ctrl['scaling_slope_dlnH_dlng'] = slope
    ctrl['scaling_pass'] = bool(abs(slope + 1) <= 0.10 and all(abs(table[n]['final']['H_fit_over_H'] - 1) <= 0.10 for _, n in gs))
if hf('C6_line_cs1.4142_g2') and hf('C1_line_N128_h0.1'):
    rr = hf('C6_line_cs1.4142_g2') / hf('C1_line_N128_h0.1')
    ctrl['dimensional_ratio_C6_over_C1'] = rr
    ctrl['dimensional_pass'] = bool(abs(rr - 1) <= 0.02)
ics = [n for n in ('C7_line_expH0.5', 'C8_line_expH2') if hf(n) is not None]
if ics and hf('C1_line_N128_h0.1'):
    d = {n: hf(n) - hf('C1_line_N128_h0.1') for n in ics}
    ctrl['ic_independence_dH'] = d
    ctrl['ic_independence_pass'] = bool(all(abs(v) <= 0.05 for v in d.values()))
hyd = {n: table[n]['final'].get('hydrostatic_ratio_rms_dev') for n in table if table[n]['mode'] == 'line'}
ctrl['hydrostatic_rms_dev'] = hyd
ctrl['hydrostatic_pass'] = bool(all(v is not None and v <= 0.05 for v in hyd.values()))

periodic = load('D_periodic3d_h0.4')
chk = json.load(open(os.path.join(D, 'check_periodic_images.json'))) if os.path.exists(os.path.join(D, 'check_periodic_images.json')) else None

results = dict(id='U11', prereg_file=os.path.join(D, 'prereg.json'), runs=table, controls=ctrl,
               periodic_image_check=chk, formula=dict(verdicts=form['verdicts'], air=form['air'],
               isa_anchor_check=form['isa_anchor_check'], rows=form['rows'], engine_polytrope=form['engine_polytrope'],
               sigma_column=form['sigma_column'], H_eq_cs2_over_g=form['H_eq_cs2_over_g'],
               K_line_times_h2=K_line_h2, kernel3d_norm=form['kernel3d_norm']),
               mentat_leaves=form['mentat_leaves'])
json.dump(results, open(os.path.join(D, 'u11_results.json'), 'w'), indent=1)

# ---------------- figure ----------------
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.unicode_minus': True, 'font.family': 'DejaVu Sans'})
fig, axs = plt.subplots(2, 2, figsize=(13, 9.6), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
fmt_minus = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'.replace('-', MINUS))

# (a) settled columns, absolute normalisation (no fitted intercept)
ax = axs[0, 0]
zz = np.linspace(0, 4.5, 200)
ax.semilogy(zz, np.exp(-zz), color=INK, lw=1.4, label='hydrostatic isothermal e$^{' + MINUS + 'z/H}$, H = c$_s^2$/g')
series = [('C1_line_N128_h0.1', 'line column N 128, h/H 0.1', COL[0]),
          ('C2_line_N128_h0.2', 'line column N 128, h/H 0.2', COL[1]),
          ('C4_line_g0.5', 'line column N 128, h/H 0.05', COL[2]),
          ('D_periodic3d_h0.4', '3-D periodic N 200, h/H 0.4', COL[3])]
for name, lab, c in series:
    r = runs.get(name)
    if r is None:
        continue
    y = np.array(r['final']['y']); rho = np.array(r['final']['rho'])
    if r['args']['mode'] == 'line':
        norm = (K_line_h2 / r['args']['h'] ** 2) * r['args']['N'] * r['parcel_mass'] / r['H']
    else:
        norm = r['args']['N'] * r['parcel_mass'] / (r['L'] ** 2 * r['H'])
    v = table[name]['final']
    ax.semilogy(y / r['H'], rho / norm, 'o', ms=2.8, color=c, alpha=0.8,
                label=f"{lab}: H$_{{fit}}$/H = {v['H_fit_over_H']:.3f}")
ax.axvspan(0.8, 2.8, color=GRID, alpha=0.5, lw=0)
ax.text(1.8, 1.7, 'fit window 0.8–2.8 H', ha='center', color=INK2, fontsize=8)
ax.set_xlim(0, 4.5); ax.set_ylim(5e-3, 3)
ax.set_xlabel('height z / H'); ax.set_ylabel('SPH density / (column mass per area / H)')
ax.set_title('(a) final state of the column, absolute normalisation', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.3, loc='lower left')

# (b) H_fit/H: final state (pre-registered) and second-half time average (post hoc)
ax = axs[0, 1]
order = [('A_literal', 'design verbatim'), ('B_literal_cs', 'design geometry, c$_s$ CFL'),
         ('C4_line_g0.5', 'line h/H 0.05 (g 0.5)'), ('C1_line_N128_h0.1', 'line h/H 0.1'),
         ('C6_line_cs1.4142_g2', 'line h/H 0.1, c$_s$ √2, g 2'), ('C3_line_N256_h0.1', 'line h/H 0.1, N 256'),
         ('C7_line_expH0.5', 'line h/H 0.1, start H$_0$ 0.5H'), ('C8_line_expH2', 'line h/H 0.1, start H$_0$ 2H'),
         ('E2_line_N128_h0.1_alpha5', 'line h/H 0.1, α 5'),
         ('C2_line_N128_h0.2', 'line h/H 0.2'), ('C5_line_g2', 'line h/H 0.2 (g 2)'),
         ('D_periodic3d_h0.4', '3-D periodic h/H 0.4')]
ax.axhspan(0.9, 1.1, color=GRID, alpha=0.6, lw=0, label='pass band |H$_{fit}$/H ' + MINUS + ' 1| ≤ 0.10')
ax.axhline(1.0, color=INK, lw=1)
xs, labs = [], []
i = 0
for name, lab in order:
    if name not in table:
        continue
    v = table[name]['final']; val = v['H_fit_over_H']
    th = table[name]['second_half_H_fit_over_H']
    if np.isfinite(val) and 0 < val < 2:
        ax.plot(i, val, 'o', color=COL[0], ms=6, zorder=3)
    else:
        ax.plot(i, 1.9, 'x', color=COL[1], ms=8, mew=2, zorder=3)
        ax.annotate('no fit' if not np.isfinite(val) else f'{val:.0f}'.replace('-', MINUS), (i, 1.9),
                    textcoords='offset points', xytext=(5, -3), fontsize=7, color=INK2)
    if np.isfinite(th['mean']) and 0 < th['mean'] < 2:
        ax.errorbar(i + 0.22, th['mean'], yerr=th['std'], fmt='s', color=COL[2], ms=4, capsize=2, zorder=3)
    xs.append(i); labs.append(lab); i += 1
ax.plot([], [], 'o', color=COL[0], label='final state (pre-registered reading)')
ax.errorbar([], [], yerr=[], fmt='s', color=COL[2], label='mean ± std over the 2nd half of the run (post hoc)')
ax.plot([], [], 'x', color=COL[1], label='design as written (off scale)')
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=40, ha='right', fontsize=7)
ax.set_ylim(0, 2.0); ax.set_ylabel('H$_{fit}$ / (c$_s^2$/g)  from ln ρ$_{SPH}$ over 0.8–2.8 H')
ax.set_title('(b) fitted scale height, every run', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.3, loc='lower right')

# (c) settling
ax = axs[1, 0]
for c, name, key, lab in [(COL[1], 'A_literal', 'ke_over_peak', 'design verbatim: dt = cfl_timestep = 6.9e' + MINUS + '6 H/c$_s$, free fall'),
                          (COL[3], 'B_literal_cs', 'ke_over_peak', 'design geometry, c$_s$ CFL: parcels slide off sideways'),
                          (COL[0], 'C2_line_N128_h0.2', 'ke_over_peak', 'line h/H 0.2: total KE'),
                          (COL[2], 'E0_line_N128_h0.2_kefree', 'ke_free_over_peak', 'same run (replica E0): KE without the ground-contact clump')]:
    r = runs.get(name)
    if r is None:
        continue
    hs = [hh for hh in r['history'] if key in hh]
    t = np.array([hh['t'] for hh in hs]); k = np.array([max(hh[key], 1e-12) for hh in hs])
    ax.loglog(np.maximum(t, 1e-4), k, '-', color=c, lw=1.5, label=lab)
ax.axhline(1e-4, color=INK, lw=1, ls='--')
ax.text(1.2e-4, 4.5e-5, 'settled: KE < 10$^{' + MINUS + '4}$ of peak', fontsize=8, color=INK2)
ax.set_xlabel('time (H / c$_s$)'); ax.set_ylabel('kinetic energy / its peak')
ax.set_title('(c) the settle criterion', loc='left', fontsize=10)
ax.set_ylim(1e-6, 3)
ax.legend(frameon=False, fontsize=7.3, loc='lower left', bbox_to_anchor=(0.0, 0.3))

# (d) formula side vs ISA 1976
ax = axs[1, 1]
z = np.array([r_['z_m'] for r_ in form['rows']]) / 1000.0
ax.axhspan(-5, 5, color=GRID, alpha=0.6, lw=0, label='± 5 %')
ax.plot(z, [100 * r_['p_rel_err'] for r_ in form['rows']], 'o-', color=COL[0],
        label='pressure_at_altitude(z, 288.15 K), isothermal')
ax.plot(z, [100 * r_['T_rel_err'] for r_ in form['rows']], 's-', color=COL[1],
        label=f"temperature_at_altitude(z), dry adiabatic {form['air']['dry_lapse_engine_K_per_km']:.2f} K/km")
ax.plot(z, [100 * r_['rel_err'] for r_ in form['engine_polytrope']], '^-', color=COL[2],
        label='engine M$_{air}$ and R with the ISA 6.5 K/km lapse')
ax.axhline(0, color=INK, lw=1)
cr = form['verdicts']['pressure_5pct_crossing_altitude_m']
if cr:
    ax.axvline(cr / 1000, color=INK2, lw=0.8, ls=':')
    ax.text(cr / 1000 + 0.15, 12, f'isothermal pressure\nleaves ±5 % at {cr/1000:.1f} km', fontsize=8, color=INK2)
ax.set_xlabel('geometric altitude (km)'); ax.set_ylabel('relative error vs US Standard Atmosphere 1976 (%)')
ax.set_title('(d) atmosphere.py vs ISA 1976 [LIT]', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.3, loc='lower left')
ax.yaxis.set_major_formatter(fmt_minus)

c2 = table.get('C2_line_N128_h0.2', {}).get('final', {})
d3 = table.get('D_periodic3d_h0.4', {}).get('final', {})
title = ('U11: in a confined planar column the unmodified SPH engine holds the barometric law H = c$_s^2$/g '
         f"(H$_{{fit}}$/H = {c2.get('H_fit_over_H', float('nan')):.3f} at h/H 0.2)" + chr(10) +
         'but not as designed: the 3-D design never forms a column, h/H 0.05 and 3-D at h/H 0.4 read high, and no run meets the KE settle criterion' + chr(10) +
         f"atmosphere.py: isothermal pressure within 5 % of ISA 1976 only below {(cr or 0)/1000:.1f} km; dry-adiabatic T(z) misses the 6.5 K/km lapse")
fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(D, 'u11_figure.png'), dpi=150, facecolor=SURF)
print(json.dumps(dict(controls=ctrl, finals={k: v['final'] for k, v in table.items()},
                      second_half={k: v['second_half_H_fit_over_H'] for k, v in table.items()},
                      budget={k: v['at_step_budget'] for k, v in table.items()}), indent=1, default=str))
