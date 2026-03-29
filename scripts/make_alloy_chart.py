#!/usr/bin/env python3
"""
Generate the alloy T_c prediction chart.

Reads predictions from docs/alloy_predictions.json and builds a
self-contained HTML visualization with:
  1. Scatter: predicted vs measured T_c
  2. Bar: all predictions sorted by T_c
  3. NbTi composition sweep
  4. Data table with full details

Output: docs/alloy_Tc_predictions.html  (self-contained, no external deps)

Usage:
    cd /path/to/sigma-ground
    python scripts/make_alloy_chart.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_data():
    data_path = Path(__file__).resolve().parent.parent / 'docs' / 'alloy_predictions.json'
    with open(data_path, 'r') as f:
        return json.load(f)


def build_html(data):
    predictions = data['predictions']
    validation = data['validation']
    nbti_sweep = data['nbti_sweep']
    measured = data['measured_data']

    # Build validated scatter data
    scatter_data = []
    for e in validation['errors']:
        scatter_data.append({
            'x': e['measured'], 'y': e['predicted'],
            'label': e['alloy'], 'pct': e['pct_error'],
        })

    # Build bar data (all predictions sorted by T_c)
    bar_data = []
    for name in sorted(predictions.keys(),
                       key=lambda k: -predictions[k]['summary']['T_c_mean_K']):
        p = predictions[name]
        entry = {
            'name': name,
            'T_c_mean': p['summary']['T_c_mean_K'],
            'T_c_min': p['summary']['T_c_min_K'],
            'T_c_max': p['summary']['T_c_max_K'],
            'has_measured': name in measured,
            'T_c_measured': measured[name]['T_c_measured_K'] if name in measured else None,
        }
        bar_data.append(entry)

    scatter_json = json.dumps(scatter_data)
    bar_json = json.dumps(bar_data)
    sweep_json = json.dumps(nbti_sweep)

    scatter_max = 22
    for s in scatter_data:
        scatter_max = max(scatter_max, s['x'] * 1.15, s['y'] * 1.15)

    # Table rows
    table_rows = ''
    for name in sorted(predictions.keys(),
                       key=lambda k: -predictions[k]['summary']['T_c_mean_K']):
        p = predictions[name]
        s = p['summary']
        m = measured.get(name)
        meas_str = f"{m['T_c_measured_K']:.1f}" if m else '&mdash;'
        err_str = ''
        if m and m['T_c_measured_K'] > 0:
            err = s['T_c_mean_K'] - m['T_c_measured_K']
            pct = err / m['T_c_measured_K'] * 100
            err_str = f'<span style="color:{"#f85149" if abs(pct) > 50 else "#d29922" if abs(pct) > 25 else "#3fb950"}">{pct:+.0f}%</span>'
        elif m and m['T_c_measured_K'] == 0 and s['T_c_mean_K'] == 0:
            err_str = '<span style="color:#3fb950">correct</span>'

        # Composition string
        comp = p['composition']
        comp_str = ', '.join(f'{k}:{v:.0%}' for k, v in comp.items())

        lin = p['models']['linear']
        dos = p['models']['dos_weighted']

        table_rows += f'''<tr>
            <td style="text-align:left">{name}</td>
            <td style="text-align:left;font-size:0.75em;color:#8b949e">{comp_str}</td>
            <td>{lin['T_c_predicted_K']:.2f}</td>
            <td>{dos['T_c_predicted_K']:.2f}</td>
            <td><strong>{s['T_c_mean_K']:.2f}</strong></td>
            <td>{meas_str}</td>
            <td>{err_str}</td>
            <td>{s['confidence']}</td>
        </tr>\n'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alloy T_c Predictions — sigma-ground</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9;
    padding: 20px;
  }}
  h1 {{ color: #58a6ff; text-align: center; margin-bottom: 8px; font-size: 1.6em; }}
  h2 {{ color: #79c0ff; margin: 24px 0 12px; font-size: 1.2em; }}
  .subtitle {{ text-align: center; color: #8b949e; margin-bottom: 24px; font-size: 0.9em; }}
  .chart-container {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 20px; margin-bottom: 24px; overflow-x: auto;
  }}
  canvas {{ display: block; margin: 0 auto; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;
    margin: 12px 0; font-size: 0.85em;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{
    width: 12px; height: 12px; border-radius: 50%; display: inline-block;
  }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 16px 0;
  }}
  .stat-card {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px; text-align: center;
  }}
  .stat-value {{ color: #58a6ff; font-size: 1.4em; font-weight: bold; }}
  .stat-label {{ color: #8b949e; font-size: 0.8em; margin-top: 4px; }}
  .note {{ color: #8b949e; font-size: 0.8em; font-style: italic; margin-top: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; font-size: 0.82em; }}
  th {{ color: #58a6ff; text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2128; }}
  .warning {{ color: #d29922; font-size: 0.75em; }}
  .nav-bar {{
    display: flex; gap: 4px; justify-content: center; margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .nav-bar a {{
    color: #8b949e; text-decoration: none; padding: 6px 14px;
    border: 1px solid #30363d; border-radius: 6px; font-size: 0.82em;
    transition: all 0.15s;
  }}
  .nav-bar a:hover {{ color: #c9d1d9; border-color: #58a6ff; background: #161b22; }}
  .nav-bar a.active {{ color: #58a6ff; border-color: #58a6ff; background: #161b22; }}
</style>
</head>
<body>

<nav class="nav-bar">
  <a href="mcmillan_validation.html">McMillan Validation</a>
  <a href="beyond_mcmillan.html">Beyond McMillan (Z&rarr;T<sub>c</sub>)</a>
  <a href="alloy_Tc_predictions.html" class="active">Alloy Predictions</a>
  <a href="dependency_chart.html">Dependency Chart</a>
</nav>

<h1>Alloy Superconductivity Predictions</h1>
<p class="subtitle">
  Blind T<sub>c</sub> predictions from atomic-fraction mixing rules + McMillan formula
  &nbsp;|&nbsp; sigma-ground alloys module
</p>

<div class="stats">
  <div class="stat-card">
    <div class="stat-value">{validation['total_predictions']}</div>
    <div class="stat-label">Total Predictions</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{validation['validated_against_literature']}</div>
    <div class="stat-label">Validated</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{validation['unvalidated_blind']}</div>
    <div class="stat-label">Blind (No Data)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{validation['mae_K']:.1f} K</div>
    <div class="stat-label">Mean Abs Error</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{validation['rmse_K']:.1f} K</div>
    <div class="stat-label">RMSE</div>
  </div>
</div>

<h2>1. Predicted vs Measured T<sub>c</sub></h2>
<div class="chart-container">
  <canvas id="scatter" width="600" height="500"></canvas>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span> Validated alloy</div>
    <div class="legend-item"><span class="legend-dot" style="background:#f85149"></span> MoRe (&#963;-phase anomaly)</div>
    <div class="legend-item" style="color:#8b949e">— ideal (predicted = measured)</div>
  </div>
</div>

<h2>2. All Predictions (sorted by T<sub>c</sub>)</h2>
<div class="chart-container">
  <canvas id="bar" width="900" height="400"></canvas>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span> Predicted (mean)</div>
    <div class="legend-item"><span class="legend-dot" style="background:#3fb950"></span> Measured</div>
    <div class="legend-item" style="color:#30363d">| model spread</div>
  </div>
</div>

<h2>3. Nb-Ti Composition Sweep</h2>
<div class="chart-container">
  <canvas id="sweep" width="700" height="350"></canvas>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span> Linear model</div>
    <div class="legend-item"><span class="legend-dot" style="background:#d29922"></span> DOS-weighted</div>
    <div class="legend-item"><span class="legend-dot" style="background:#3fb950"></span> Measured (NbTi wire)</div>
  </div>
  <p class="note">Pure Nb (x=0) &rarr; Pure Ti (x=1). Measured NbTi wire at x&approx;0.47.</p>
</div>

<h2>4. Full Results Table</h2>
<div class="chart-container">
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Alloy</th>
        <th style="text-align:left">Composition</th>
        <th>T<sub>c</sub> Linear</th>
        <th>T<sub>c</sub> DOS</th>
        <th>T<sub>c</sub> Mean</th>
        <th>Measured</th>
        <th>Error</th>
        <th>Confidence</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  <p class="note">
    All temperatures in Kelvin. Models assume disordered solid solution.
    MoRe forms an ordered &#963;-phase that dramatically enhances &#955;<sub>ep</sub> &mdash;
    this is the expected failure mode of the solid-solution approximation.
  </p>
</div>

<p class="note" style="text-align:center; margin-top:24px;">
  Generated by sigma-ground &mdash; zero external dependencies
  &nbsp;|&nbsp; McMillan, Phys. Rev. 167, 331 (1968)
</p>

<script>
const SCATTER = {scatter_json};
const BAR = {bar_json};
const SWEEP = {sweep_json};

// ── Scatter Plot ──
(function() {{
  const c = document.getElementById('scatter');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:60, r:20, t:20, b:50}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  const maxV = {scatter_max:.1f};

  function tx(v) {{ return pad.l + (v/maxV)*pw; }}
  function ty(v) {{ return pad.t + ph - (v/maxV)*ph; }}

  // Axes
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();

  // Grid + labels
  ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let v = 0; v <= maxV; v += 5) {{
    ctx.fillText(v.toFixed(0), tx(v), H-pad.b+16);
    ctx.fillText(v.toFixed(0), pad.l-12, ty(v)+4);
    ctx.strokeStyle = '#21262d';
    ctx.beginPath(); ctx.moveTo(tx(v), pad.t); ctx.lineTo(tx(v), H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, ty(v)); ctx.lineTo(W-pad.r, ty(v)); ctx.stroke();
  }}
  ctx.fillText('Measured T_c (K)', W/2, H-4);
  ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Predicted T_c (K)', 0, 0); ctx.restore();

  // Ideal line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(tx(0), ty(0)); ctx.lineTo(tx(maxV), ty(maxV)); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  SCATTER.forEach(p => {{
    const isMoRe = p.label.startsWith('MoRe');
    ctx.fillStyle = isMoRe ? '#f85149' : '#58a6ff';
    ctx.beginPath(); ctx.arc(tx(p.x), ty(p.y), 6, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#c9d1d9'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(p.label.replace(/_/g,' '), tx(p.x)+9, ty(p.y)+4);
  }});
}})();

// ── Bar Chart ──
(function() {{
  const c = document.getElementById('bar');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:50, r:20, t:20, b:100}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  const n = BAR.length;
  const bw = Math.min(30, pw/n - 4);
  const maxTc = Math.max(...BAR.map(b => Math.max(b.T_c_max, b.T_c_measured||0))) * 1.15;

  function ty(v) {{ return pad.t + ph - (v/maxTc)*ph; }}

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let v = 0; v <= maxTc; v += 5) {{
    const y = ty(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(0)+' K', pad.l-6, y+4);
  }}

  BAR.forEach((b, i) => {{
    const x = pad.l + (i+0.5)*(pw/n) - bw/2;
    const ym = ty(b.T_c_mean);
    const ymin = ty(b.T_c_min);
    const ymax = ty(b.T_c_max);

    // Spread bar
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x+bw/2, ymin); ctx.lineTo(x+bw/2, ymax); ctx.stroke();

    // Predicted bar
    ctx.fillStyle = b.has_measured ? '#58a6ff' : '#1f6feb80';
    ctx.fillRect(x, ym, bw, ty(0)-ym);

    // Measured dot
    if (b.T_c_measured !== null && b.T_c_measured > 0) {{
      ctx.fillStyle = '#3fb950';
      ctx.beginPath(); ctx.arc(x+bw/2, ty(b.T_c_measured), 4, 0, Math.PI*2); ctx.fill();
    }}

    // Label
    ctx.fillStyle = '#8b949e'; ctx.font = '9px sans-serif';
    ctx.save(); ctx.translate(x+bw/2, H-pad.b+6); ctx.rotate(Math.PI/4);
    ctx.textAlign = 'left';
    ctx.fillText(b.name.replace(/_/g,' '), 0, 0);
    ctx.restore();
  }});
}})();

// ── NbTi Sweep ──
(function() {{
  const c = document.getElementById('sweep');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:60, r:20, t:20, b:50}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  const maxTc = Math.max(...SWEEP.map(s => Math.max(s.T_c_linear_K, s.T_c_dos_K))) * 1.15;

  function tx(x) {{ return pad.l + x*pw; }}
  function ty(v) {{ return pad.t + ph - (v/maxTc)*ph; }}

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let x = 0; x <= 1; x += 0.2) {{
    ctx.fillText((x*100).toFixed(0)+'%', tx(x), H-pad.b+16);
    ctx.strokeStyle = '#21262d';
    ctx.beginPath(); ctx.moveTo(tx(x), pad.t); ctx.lineTo(tx(x), H-pad.b); ctx.stroke();
  }}
  ctx.textAlign = 'right';
  for (let v = 0; v <= maxTc; v += 5) {{
    ctx.fillText(v.toFixed(0)+' K', pad.l-8, ty(v)+4);
    ctx.beginPath(); ctx.moveTo(pad.l, ty(v)); ctx.lineTo(W-pad.r, ty(v)); ctx.stroke();
  }}
  ctx.textAlign = 'center';
  ctx.fillText('Ti fraction (at%)', W/2, H-4);
  ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Predicted T_c (K)', 0, 0); ctx.restore();

  // Linear curve
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2;
  ctx.beginPath();
  SWEEP.forEach((s,i) => {{
    const fn = i === 0 ? 'moveTo' : 'lineTo';
    ctx[fn](tx(s.x_B), ty(s.T_c_linear_K));
  }});
  ctx.stroke();

  // DOS curve
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 2;
  ctx.beginPath();
  SWEEP.forEach((s,i) => {{
    const fn = i === 0 ? 'moveTo' : 'lineTo';
    ctx[fn](tx(s.x_B), ty(s.T_c_dos_K));
  }});
  ctx.stroke();

  // Measured point (NbTi wire at x=0.47, T_c=9.5K)
  ctx.fillStyle = '#3fb950';
  ctx.beginPath(); ctx.arc(tx(0.47), ty(9.5), 6, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Measured 9.5K', tx(0.47)+10, ty(9.5)+4);
}})();
</script>

</body>
</html>"""


def main():
    data = load_data()
    html = build_html(data)

    output_path = Path(__file__).resolve().parent.parent / 'docs' / 'alloy_Tc_predictions.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Chart written to {output_path}")


if __name__ == '__main__':
    main()
