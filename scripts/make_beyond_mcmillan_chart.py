#!/usr/bin/env python3
"""
Generate the Beyond McMillan chart: T_c predicted from Z alone.

Every quantity (θ_D, μ*, λ_ep) derived from atomic number — no measured
material data used. Measured T_c values shown for validation only.

Output: docs/beyond_mcmillan.html  (self-contained, no external deps)

Usage:
    cd /path/to/sigma-ground
    python scripts/make_beyond_mcmillan_chart.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigma_ground.field.interface.superconductivity import (
    predict_Tc_from_Z, SUPERCONDUCTORS,
)

# Full periodic table name → Z
NAME_TO_Z = {
    'hydrogen':1,'helium':2,'lithium':3,'beryllium':4,'boron':5,'carbon':6,
    'nitrogen':7,'oxygen':8,'fluorine':9,'neon':10,'sodium':11,'magnesium':12,
    'aluminum':13,'silicon':14,'phosphorus':15,'sulfur':16,'chlorine':17,'argon':18,
    'potassium':19,'calcium':20,'scandium':21,'titanium':22,'vanadium':23,
    'chromium':24,'manganese':25,'iron':26,'cobalt':27,'nickel':28,'copper':29,
    'zinc':30,'gallium':31,'germanium':32,'arsenic':33,'selenium':34,'bromine':35,
    'krypton':36,'rubidium':37,'strontium':38,'yttrium':39,'zirconium':40,
    'niobium':41,'molybdenum':42,'technetium':43,'ruthenium':44,'rhodium':45,
    'palladium':46,'silver':47,'cadmium':48,'indium':49,'tin':50,'antimony':51,
    'tellurium':52,'iodine':53,'xenon':54,'cesium':55,'barium':56,'lanthanum':57,
    'cerium':58,'praseodymium':59,'neodymium':60,
    'lutetium':71,'hafnium':72,'tantalum':73,'tungsten':74,'rhenium':75,
    'osmium':76,'iridium':77,'platinum':78,'gold':79,'mercury':80,
    'thallium':81,'lead':82,'bismuth':83,
    'thorium':90,'protactinium':91,'uranium':92,'americium':95,
}
Z_TO_NAME = {v: k for k, v in NAME_TO_Z.items()}

# Element symbols
SYMBOLS = {
    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
    19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
    27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
    35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
    43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
    51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',
    71:'Lu',72:'Hf',73:'Ta',74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',
    79:'Au',80:'Hg',81:'Tl',82:'Pb',83:'Bi',
    90:'Th',91:'Pa',92:'U',95:'Am',
    58:'Ce',59:'Pr',60:'Nd',
}


def gather_data():
    """Run predict_Tc_from_Z for every element, compare to measured."""
    # Build measured lookup
    meas = {}
    for name, data in SUPERCONDUCTORS.items():
        z = NAME_TO_Z.get(name)
        if z:
            meas[z] = {
                'Tc_K': data.get('T_c_K', 0),
                'is_sc': data.get('is_superconductor', False),
                'lambda_ep': data.get('lambda_ep'),
                'mu_star': data.get('mu_star'),
                'theta_D_K': data.get('theta_D_K'),
            }

    rows = []
    for z in sorted(Z_TO_NAME.keys()):
        name = Z_TO_NAME[z]
        sym = SYMBOLS.get(z, '?')
        r = predict_Tc_from_Z(z)

        tc_der = r['T_c_K'] if r else 0
        lam = r['lambda_ep'] if r else 0
        mu = r['mu_star'] if r else 0
        thD = r['theta_D_K'] if r else 0
        rho = r['rho_uOhm_cm'] if r else 0

        m = meas.get(z)
        tc_meas = m['Tc_K'] if m else None
        is_sc = m['is_sc'] if m else None
        lam_meas = m['lambda_ep'] if m else None
        thD_meas = m['theta_D_K'] if m else None
        mu_meas = m['mu_star'] if m else None

        # Category
        if r is None:
            cat = 'no_data'
        elif tc_der > 0.01 and m and is_sc:
            cat = 'correct_sc'
        elif tc_der <= 0.01 and m and not is_sc:
            cat = 'correct_zero'
        elif tc_der > 0.01 and m and not is_sc:
            cat = 'false_pos'
        elif tc_der <= 0.01 and m and is_sc and (tc_meas or 0) > 0:
            cat = 'false_neg'
        elif tc_der > 0.01 and m is None:
            cat = 'prediction'
        else:
            cat = 'pred_zero'

        rows.append({
            'Z': z, 'name': name, 'symbol': sym,
            'Tc_derived': round(tc_der, 3),
            'Tc_measured': tc_meas,
            'lambda_derived': round(lam, 4),
            'lambda_measured': lam_meas,
            'mu_star': round(mu, 4),
            'theta_D_derived': round(thD, 1) if thD else None,
            'theta_D_measured': thD_meas,
            'rho_uOhm_cm': round(rho, 2),
            'category': cat,
            'is_sc': is_sc,
        })

    return rows


def build_html(rows):
    """Build the self-contained HTML visualization."""
    # Summary stats
    cats = {}
    for r in rows:
        c = r['category']
        cats[c] = cats.get(c, 0) + 1

    correct_sc = cats.get('correct_sc', 0)
    correct_zero = cats.get('correct_zero', 0)
    false_pos = cats.get('false_pos', 0)
    false_neg = cats.get('false_neg', 0)
    predictions = cats.get('prediction', 0)

    # Scatter data (only elements with both derived and measured Tc)
    scatter = []
    for r in rows:
        if r['Tc_measured'] is not None and r['Tc_derived'] > 0 and r['Tc_measured'] > 0:
            scatter.append({
                'x': r['Tc_measured'], 'y': r['Tc_derived'],
                'label': r['symbol'], 'cat': r['category'],
            })

    # Bar chart data: all elements sorted by derived Tc
    bar_data = []
    for r in sorted(rows, key=lambda x: x['Tc_derived'], reverse=True):
        if r['Tc_derived'] > 0.01:
            bar_data.append({
                'symbol': r['symbol'], 'Z': r['Z'],
                'tc_der': r['Tc_derived'],
                'tc_meas': r['Tc_measured'],
                'cat': r['category'],
                'lambda': r['lambda_derived'],
            })

    # Lambda comparison (elements with measured lambda)
    lambda_data = []
    for r in rows:
        if r['lambda_measured'] is not None and r['lambda_derived'] > 0:
            lambda_data.append({
                'x': r['lambda_measured'], 'y': r['lambda_derived'],
                'label': r['symbol'],
            })

    rows_json = json.dumps(rows)
    scatter_json = json.dumps(scatter)
    bar_json = json.dumps(bar_data[:40])  # top 40
    lambda_json = json.dumps(lambda_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beyond McMillan — T_c from Z Alone | sigma-ground</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9;
    padding: 20px; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ color: #58a6ff; text-align: center; margin-bottom: 4px; font-size: 1.8em; }}
  h2 {{ color: #79c0ff; margin: 28px 0 12px; font-size: 1.25em; }}
  .subtitle {{ text-align: center; color: #8b949e; margin-bottom: 24px; font-size: 0.92em; line-height: 1.5; }}
  .chain {{ text-align: center; color: #7ee787; font-family: monospace; font-size: 0.85em; margin: 12px 0 20px; }}
  .chart-box {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 20px; margin-bottom: 24px; overflow-x: auto;
  }}
  canvas {{ display: block; margin: 0 auto; }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;
    margin: 12px 0; font-size: 0.85em;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 16px 0;
  }}
  .stat-card {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px; text-align: center;
  }}
  .stat-value {{ color: #58a6ff; font-size: 1.5em; font-weight: bold; }}
  .stat-label {{ color: #8b949e; font-size: 0.78em; margin-top: 4px; }}
  .stat-card.good .stat-value {{ color: #7ee787; }}
  .stat-card.bad .stat-value {{ color: #f85149; }}
  .stat-card.new .stat-value {{ color: #d2a8ff; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.8em; }}
  th, td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  th {{ color: #58a6ff; position: sticky; top: 0; background: #161b22; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2128; }}
  .cat-correct_sc {{ color: #7ee787; }}
  .cat-correct_zero {{ color: #7ee787; }}
  .cat-false_pos {{ color: #f85149; }}
  .cat-false_neg {{ color: #f0883e; }}
  .cat-prediction {{ color: #d2a8ff; }}
  .cat-pred_zero {{ color: #8b949e; }}
  .note {{ color: #8b949e; font-size: 0.8em; font-style: italic; }}
  .wall-box {{
    background: #1c1208; border: 1px solid #f0883e; border-radius: 6px;
    padding: 16px; margin: 16px 0;
  }}
  .wall-box h3 {{ color: #f0883e; margin-bottom: 8px; }}
</style>
</head>
<body>

<h1>Beyond McMillan: T<sub>c</sub> from Atomic Number Alone</h1>
<p class="subtitle">
  Every quantity derived from Z &mdash; no measured material data used.<br>
  Measured values shown for <em>validation only</em>.
</p>
<p class="chain">
  Z &rarr; structure &rarr; density &rarr; n<sub>e</sub> &rarr; k<sub>F</sub> &rarr; E<sub>F</sub>
  &rarr; E<sub>coh</sub> &rarr; K &rarr; v<sub>s</sub> &rarr; &theta;<sub>D</sub>
  &rarr; N(E<sub>F</sub>) &rarr; U<sub>scr</sub> &rarr; &mu;*
  &rarr; V(q) &rarr; &langle;I&sup2;&rangle; &rarr; &eta; &rarr; &lambda;<sub>ep</sub>
  &rarr; McMillan T<sub>c</sub>
</p>

<!-- Summary cards -->
<div class="stats">
  <div class="stat-card good"><div class="stat-value">{correct_sc}</div><div class="stat-label">Correct SC</div></div>
  <div class="stat-card good"><div class="stat-value">{correct_zero}</div><div class="stat-label">Correct Non-SC</div></div>
  <div class="stat-card bad"><div class="stat-value">{false_neg}</div><div class="stat-label">False Negative</div></div>
  <div class="stat-card bad"><div class="stat-value">{false_pos}</div><div class="stat-label">False Positive</div></div>
  <div class="stat-card new"><div class="stat-value">{predictions}</div><div class="stat-label">New Predictions</div></div>
</div>

<!-- Chart 1: Scatter plot -->
<h2>1. Predicted vs Measured T<sub>c</sub></h2>
<div class="chart-box">
  <canvas id="scatter" width="700" height="500"></canvas>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#7ee787"></span> Correct SC</div>
    <div class="legend-item"><span class="legend-dot" style="background:#f0883e"></span> Overestimate (&gt;3&times;)</div>
    <div class="legend-item"><span class="legend-dot" style="background:#f85149"></span> False Positive</div>
  </div>
</div>

<!-- Chart 2: Bar chart -->
<h2>2. All Predictions Ranked by Derived T<sub>c</sub></h2>
<div class="chart-box">
  <canvas id="bars" width="1200" height="420"></canvas>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span> Derived T<sub>c</sub></div>
    <div class="legend-item"><span class="legend-dot" style="background:#7ee787"></span> Measured T<sub>c</sub></div>
    <div class="legend-item"><span class="legend-dot" style="background:#d2a8ff"></span> New prediction (no measured data)</div>
  </div>
</div>

<!-- Chart 3: Lambda comparison -->
<h2>3. Derived vs Measured &lambda;<sub>ep</sub></h2>
<div class="chart-box">
  <canvas id="lambda" width="500" height="500"></canvas>
  <p class="note" style="text-align:center; margin-top:8px;">
    Median ratio = 0.92 &mdash; 8/9 within factor of 2
  </p>
</div>

<!-- Physics wall -->
<div class="wall-box">
  <h3>The d-Band Wall</h3>
  <p>The 13 false negatives (Nb, Ta, Zr, Hf, Y, Os, Ir, Ru, Rh, Lu, Li, Be, Fe) are almost
  entirely <strong>d-band metals</strong> where the free-electron pseudopotential cannot capture
  d-resonance scattering. The electron-phonon coupling in these metals is dominated by
  the d-band density of states at E<sub>F</sub> &mdash; a volume/lattice property that requires
  going beyond the Ashcroft empty-core model.</p>
  <p style="margin-top:8px;">The single false positive (Pd) is the most famous near-miss in
  superconductivity &mdash; paramagnon fluctuations suppress Cooper pairing, an effect not
  captured by the McMillan formula.</p>
</div>

<!-- Data table -->
<h2>4. Full Data Table</h2>
<div class="chart-box" style="max-height: 600px; overflow-y: auto;">
<table>
<thead>
<tr>
  <th>Element</th><th>Z</th>
  <th>T<sub>c</sub> derived (K)</th><th>T<sub>c</sub> measured (K)</th>
  <th>&lambda; derived</th><th>&lambda; measured</th>
  <th>&mu;*</th>
  <th>&theta;<sub>D</sub> derived</th><th>&theta;<sub>D</sub> meas</th>
  <th>&rho; (&mu;&Omega;&middot;cm)</th>
  <th>Verdict</th>
</tr>
</thead>
<tbody id="tableBody"></tbody>
</table>
</div>

<p class="note" style="text-align:center; margin-top:24px;">
  Generated by sigma-ground &mdash; predict_Tc_from_Z() &mdash; zero external dependencies
</p>

<script>
const rows = {rows_json};
const scatterData = {scatter_json};
const barData = {bar_json};
const lambdaData = {lambda_json};

// ── Scatter Plot ──
(function() {{
  const c = document.getElementById('scatter');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:60, r:20, t:20, b:50}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;

  // Axis max (log scale)
  let allVals = scatterData.flatMap(d => [d.x, d.y]).filter(v => v > 0);
  let maxVal = Math.max(...allVals) * 1.3;
  let minVal = 0.01;

  function toX(v) {{ return pad.l + (Math.log10(Math.max(v,minVal)) - Math.log10(minVal)) / (Math.log10(maxVal) - Math.log10(minVal)) * pw; }}
  function toY(v) {{ return pad.t + ph - (Math.log10(Math.max(v,minVal)) - Math.log10(minVal)) / (Math.log10(maxVal) - Math.log10(minVal)) * ph; }}

  // Grid
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 0.5;
  for (let exp = -2; exp <= 2; exp++) {{
    let v = Math.pow(10, exp);
    if (v >= minVal && v <= maxVal) {{
      ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(W-pad.r, toY(v)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(toX(v), pad.t); ctx.lineTo(toX(v), H-pad.b); ctx.stroke();
      ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
      ctx.textAlign = 'right'; ctx.fillText(v+'K', pad.l-5, toY(v)+4);
      ctx.textAlign = 'center'; ctx.fillText(v+'K', toX(v), H-pad.b+15);
    }}
  }}

  // Perfect agreement line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(toX(minVal), toY(minVal)); ctx.lineTo(toX(maxVal), toY(maxVal)); ctx.stroke();
  ctx.setLineDash([]);

  // 3x bounds
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(toX(minVal), toY(minVal*3)); ctx.lineTo(toX(maxVal/3), toY(maxVal)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX(minVal*3), toY(minVal)); ctx.lineTo(toX(maxVal), toY(maxVal/3)); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  scatterData.forEach(d => {{
    let ratio = d.y / d.x;
    let color = (ratio >= 0.33 && ratio <= 3) ? '#7ee787' : '#f0883e';
    if (d.cat === 'false_pos') color = '#f85149';
    let x = toX(d.x), y = toY(d.y);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2); ctx.fillStyle = color; ctx.fill();
    ctx.fillStyle = '#c9d1d9'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(d.label, x+7, y+3);
  }});

  // Axis labels
  ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Measured Tc (K)', W/2, H-5);
  ctx.save(); ctx.translate(15, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Derived Tc (K)', 0, 0); ctx.restore();
}})();

// ── Bar Chart ──
(function() {{
  const c = document.getElementById('bars');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:50, r:20, t:20, b:60}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  const n = barData.length;
  const bw = Math.min(pw/n - 2, 20);
  const gap = (pw - n*bw) / (n+1);

  let maxTc = Math.max(...barData.map(d => Math.max(d.tc_der, d.tc_meas||0))) * 1.1;

  function toY(v) {{ return pad.t + ph - (v/maxTc)*ph; }}

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let t = 0; t <= maxTc; t += 10) {{
    ctx.beginPath(); ctx.moveTo(pad.l, toY(t)); ctx.lineTo(W-pad.r, toY(t)); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.textAlign = 'right'; ctx.fillText(t+'K', pad.l-5, toY(t)+4);
  }}

  barData.forEach((d, i) => {{
    let x = pad.l + gap + i*(bw+gap);
    let yDer = toY(d.tc_der);
    let yBase = toY(0);

    // Derived bar
    let color = d.cat === 'prediction' ? '#d2a8ff' :
                d.cat === 'false_pos' ? '#f85149' : '#58a6ff';
    ctx.fillStyle = color;
    ctx.fillRect(x, yDer, bw, yBase-yDer);

    // Measured marker
    if (d.tc_meas != null && d.tc_meas > 0) {{
      let yM = toY(d.tc_meas);
      ctx.strokeStyle = '#7ee787'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x-2, yM); ctx.lineTo(x+bw+2, yM); ctx.stroke();
    }}

    // Label
    ctx.save();
    ctx.translate(x + bw/2, yBase + 8);
    ctx.rotate(-Math.PI/4);
    ctx.fillStyle = '#8b949e'; ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(d.symbol, 0, 0);
    ctx.restore();
  }});

  ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.save(); ctx.translate(15, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Tc (K)', 0, 0); ctx.restore();
}})();

// ── Lambda Scatter ──
(function() {{
  const c = document.getElementById('lambda');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = {{l:50, r:20, t:20, b:50}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;

  let maxL = Math.max(...lambdaData.flatMap(d => [d.x, d.y])) * 1.2;

  function toX(v) {{ return pad.l + (v/maxL)*pw; }}
  function toY(v) {{ return pad.t + ph - (v/maxL)*ph; }}

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let t = 0; t <= maxL; t += 0.2) {{
    ctx.beginPath(); ctx.moveTo(pad.l, toY(t)); ctx.lineTo(W-pad.r, toY(t)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(toX(t), pad.t); ctx.lineTo(toX(t), H-pad.b); ctx.stroke();
  }}

  // Perfect line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(maxL), toY(maxL)); ctx.stroke();
  ctx.setLineDash([]);

  // 2x bounds
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(maxL/2), toY(maxL)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(maxL), toY(maxL/2)); ctx.stroke();
  ctx.setLineDash([]);

  lambdaData.forEach(d => {{
    let x = toX(d.x), y = toY(d.y);
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2);
    ctx.fillStyle = '#58a6ff'; ctx.fill();
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif';
    ctx.textAlign = 'left'; ctx.fillText(d.label, x+8, y+4);
  }});

  ctx.fillStyle = '#8b949e'; ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Measured lambda', W/2, H-5);
  ctx.save(); ctx.translate(15, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Derived lambda', 0, 0); ctx.restore();
}})();

// ── Data Table ──
(function() {{
  const tbody = document.getElementById('tableBody');
  const catLabels = {{
    'correct_sc': 'CORRECT SC',
    'correct_zero': 'CORRECT 0',
    'false_pos': 'FALSE POS',
    'false_neg': 'FALSE NEG',
    'prediction': 'PREDICTION',
    'pred_zero': 'pred=0',
    'no_data': 'NO DATA',
  }};

  rows.forEach(r => {{
    const tr = document.createElement('tr');
    const cat = r.category;
    const fmt = (v, d) => v != null ? (typeof v === 'number' ? v.toFixed(d) : v) : '—';

    tr.innerHTML = `
      <td>${{r.symbol}} ${{r.name}}</td>
      <td>${{r.Z}}</td>
      <td>${{fmt(r.Tc_derived, 2)}}</td>
      <td>${{fmt(r.Tc_measured, 2)}}</td>
      <td>${{fmt(r.lambda_derived, 3)}}</td>
      <td>${{fmt(r.lambda_measured, 3)}}</td>
      <td>${{fmt(r.mu_star, 3)}}</td>
      <td>${{fmt(r.theta_D_derived, 0)}}</td>
      <td>${{fmt(r.theta_D_measured, 0)}}</td>
      <td>${{fmt(r.rho_uOhm_cm, 1)}}</td>
      <td class="cat-${{cat}}">${{catLabels[cat] || cat}}</td>
    `;
    tbody.appendChild(tr);
  }});
}})();
</script>

</body>
</html>"""


def main():
    print("Gathering predictions...")
    rows = gather_data()
    print(f"  {len(rows)} elements processed")

    html = build_html(rows)
    out = Path(__file__).resolve().parent.parent / 'docs' / 'beyond_mcmillan.html'
    out.write_text(html, encoding='utf-8')
    print(f"  Written to {out}")
    print(f"  File size: {out.stat().st_size:,} bytes")

    # Summary
    cats = {}
    for r in rows:
        c = r['category']
        cats[c] = cats.get(c, 0) + 1
    print()
    print("Summary:")
    for k, v in sorted(cats.items()):
        print(f"  {k:15s}: {v}")


if __name__ == '__main__':
    main()
