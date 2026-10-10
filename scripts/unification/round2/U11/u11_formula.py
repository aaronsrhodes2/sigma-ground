"""U11 formula side + Mentat leaves.

1. H = c_s^2/g from Mentat (sympy.dsolve of dP/dz = -g P / c_s^2) and the residual check (differentiate + simplify).
2. K = integral of the 3-D cubic-spline kernel along a line (Mentat integrate_expr), used by the 1-D line column's
   hydrostatic-weight check: rho_sph = K * (linear density).
3. atmosphere.py (repo engine = Mentat) vs the US Standard Atmosphere 1976 [LIT]:
   - ISA troposphere from its defining constants [LIT, verified] evaluated by Mentat simplify_expr at geometric
     z = 0..11 km (geopotential H = r0 z / (r0 + z));
   - the formula anchored to tabulated rows [LIT, PDAS Table 1]: z = 5 km 255.676 K / 5.4048e4 Pa, 10 km 223.252 K / 2.6500e4 Pa.
   - pressure_at_altitude(z, 288.15), temperature_at_altitude(z), dry_adiabatic_lapse_rate() vs ISA.
4. sigma column: w = P/(rho c^2) at sea level and Tolman-Ehrenfest g dz / c^2 over one scale height and over 11 km.
"""
import sys, json, math
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.field.interface import atmosphere as atm

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11/formula_results.json"
leaves = {}


def M(tool, args, key):
    v, s = mentat(tool, args)
    leaves[key] = dict(tool=tool, args=args, value=v, source=s)
    return v


def ev(expr, key):
    """Mentat as the calculator: sympy.simplify of a numeric expression -> float."""
    v = M('simplify_expr', {'expression': f'N({expr}, 17)'}, key)
    return float(v)


# 1. the scale height, symbolically
M('solve_ode', {'equation': 'Derivative(y(x),x) + g*y(x)/c_s**2'}, 'ode_hydrostatic_isothermal')
dP = M('differentiate_expr', {'expression': 'rho0*c_s**2*exp(-g*z/c_s**2)', 'variable': 'z'}, 'dPdz')
M('simplify_expr', {'expression': f'({dP}) + g*rho0*exp(-g*z/c_s**2)'}, 'hydrostatic_residual')
M('simplify_expr', {'expression': 'c_s**2/g'}, 'H_symbolic')

# 2. line integral of the 3-D cubic spline (h = 1): K h^2 = (3/(2 pi)) * 2 * [int_0^1 w + int_1^2 w]
i1 = M('integrate_expr', {'expression': '2/3 - q**2 + q**3/2', 'variable': 'q', 'lower': '0', 'upper': '1'}, 'kernel_int_inner')
i2 = M('integrate_expr', {'expression': '(2-q)**3/6', 'variable': 'q', 'lower': '1', 'upper': '2'}, 'kernel_int_outer')
K_line_h2 = ev(f'(3/(2*pi))*2*(({i1}) + ({i2}))', 'K_line_times_h2')
# 3-D normalisation check of the same kernel: 4 pi (3/(2 pi)) int q^2 w dq = 1
j1 = M('integrate_expr', {'expression': 'q**2*(2/3 - q**2 + q**3/2)', 'variable': 'q', 'lower': '0', 'upper': '1'}, 'kernel3d_inner')
j2 = M('integrate_expr', {'expression': 'q**2*(2-q)**3/6', 'variable': 'q', 'lower': '1', 'upper': '2'}, 'kernel3d_outer')
norm3d = ev(f'4*pi*(3/(2*pi))*(({j1}) + ({j2}))', 'kernel3d_norm')

# 3. constants
R = M('lookup_constant', {'name': 'molar gas constant'}, 'R')
g0 = M('lookup_constant', {'name': 'standard acceleration of gravity'}, 'g_standard')
p_atm = M('lookup_constant', {'name': 'standard atmosphere'}, 'atm')
c = M('lookup_constant', {'name': 'c'}, 'c')
rho_air = M('density', {'material': 'air'}, 'rho_air_20C')
cs_air = M('speed_of_sound_in_ideal_gas', {'temperature_k': 288.15}, 'cs_air_288K')

# ISA 1976 defining constants [LIT]: Wikipedia "U.S. Standard Atmosphere" (layer 0: 288.15 K, 101325 Pa,
# -6.5 K/km; 11 km base 216.65 K, 22632.1 Pa; M0 = 28.9644 g/mol, R* = 8.31432 J/(mol K), g = 9.80665);
# r0 = 6356.766 km (NASA-TM-X-74335 via the PDAS atmos76 code and US patent 20140249753, verified by web search).
ISA = dict(T0=288.15, p0=101325.0, L=0.0065, M0=0.0289644, Rstar=8.31432, g0=9.80665, r0=6356766.0)
expo = ev(f"{ISA['g0']}*{ISA['M0']}/({ISA['Rstar']}*{ISA['L']})", 'ISA_exponent')


def isa(zgeo):
    Hgp = ev(f"{ISA['r0']}*{zgeo}/({ISA['r0']}+{zgeo})", f'ISA_Hgp_{zgeo:.0f}')
    T = ev(f"{ISA['T0']} - {ISA['L']}*{Hgp}", f'ISA_T_{zgeo:.0f}')
    p = ev(f"{ISA['p0']}*({T}/{ISA['T0']})**({expo})", f'ISA_p_{zgeo:.0f}')
    return Hgp, T, p


# tabulated anchors [LIT] PDAS Table 1 (geometric Z): 5 km 255.676 K 5.4048E+04 Pa; 10 km 223.252 K 2.6500E+04 Pa
anchors = {5000.0: (255.676, 5.4048e4), 10000.0: (223.252, 2.6500e4)}

zs = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0]
T_surf = 288.15
H_air = atm.scale_height(T_surf)
M_air = atm.mean_molar_mass_kg_mol()
Gamma_d = atm.dry_adiabatic_lapse_rate(T_surf)
rows = []
for z in zs:
    Hgp, T_isa, p_isa = isa(z)
    p_iso = atm.pressure_at_altitude(z, T_surf)
    T_dry = atm.temperature_at_altitude(z, T_surf)
    rows.append(dict(z_m=z, H_geopotential_m=Hgp, T_ISA=T_isa, p_ISA=p_isa,
                     p_isothermal_engine=p_iso, p_rel_err=(p_iso - p_isa) / p_isa,
                     T_dry_adiabatic_engine=T_dry, T_rel_err=(T_dry - T_isa) / T_isa))
anchor_check = {str(int(z)): dict(T_tab=v[0], p_tab=v[1],
                                  T_formula=[r for r in rows if r['z_m'] == z][0]['T_ISA'],
                                  p_formula=[r for r in rows if r['z_m'] == z][0]['p_ISA'])
                for z, v in anchors.items()}
for k, d in anchor_check.items():
    d['p_rel'] = (d['p_formula'] - d['p_tab']) / d['p_tab']
    d['T_rel'] = (d['T_formula'] - d['T_tab']) / d['T_tab']

# decomposition: the engine's own M_air and R with the ISA lapse rate (polytropic hydrostatic integral) -> how
# much of the isothermal formula's error is model scope (T(z)) vs the engine's constants
expo_engine = ev(f"{g0}*{M_air}/({R}*{ISA['L']})", 'engine_polytrope_exponent')
poly = []
for r in rows:
    p_poly = ev(f"{p_atm}*(({ISA['T0']} - {ISA['L']}*{r['H_geopotential_m']})/{ISA['T0']})**({expo_engine})",
                f"engine_poly_p_{r['z_m']:.0f}")
    poly.append(dict(z_m=r['z_m'], p_engine_polytrope=p_poly, rel_err=(p_poly - r['p_ISA']) / r['p_ISA']))

# pass/fail (pre-registered)
p_ok_5km = all(abs(r['p_rel_err']) <= 0.05 for r in rows if r['z_m'] <= 5000.0)
p_fail_above_8 = all(abs(r['p_rel_err']) > 0.05 for r in rows if r['z_m'] > 8000.0)
cross = None
for r0_, r1_ in zip(rows[:-1], rows[1:]):
    if abs(r0_['p_rel_err']) <= 0.05 < abs(r1_['p_rel_err']):
        f = (0.05 - r0_['p_rel_err']) / (r1_['p_rel_err'] - r0_['p_rel_err'])
        cross = r0_['z_m'] + f * (r1_['z_m'] - r0_['z_m'])
T_ok = all(abs(r['T_rel_err']) <= 0.05 for r in rows)
lapse_ok = abs(Gamma_d * 1000.0 - 6.5) / 6.5 <= 0.10

# scale height cross-check: atmosphere.scale_height vs R T/(M g) by Mentat with the engine's M
H_mentat = ev(f"{R}*{T_surf}/({M_air}*{g0})", 'H_air_RT_over_Mg')
H_isa_M = ev(f"{ISA['Rstar']}*{T_surf}/({ISA['M0']}*{ISA['g0']})", 'H_air_ISA_constants')

# 4. sigma / Tolman column
w_sea = ev(f"{p_atm}/({rho_air}*{c}**2)", 'w_sea_level')
three_w = ev(f"3*{p_atm}/({rho_air}*{c}**2)", 'three_w_sea_level')
tolman_H = ev(f"{g0}*{H_air}/{c}**2", 'tolman_over_scale_height')
tolman_11 = ev(f"{g0}*11000/{c}**2", 'tolman_over_11km')
w_cs = ev(f"{cs_air}**2/{c}**2", 'cs2_over_c2_air')

res = dict(
    id='U11-formula',
    H_eq_cs2_over_g=dict(ode=leaves['ode_hydrostatic_isothermal']['value'], residual=leaves['hydrostatic_residual']['value']),
    K_line_times_h2=K_line_h2, kernel3d_norm=norm3d,
    air=dict(scale_height_engine_m=H_air, scale_height_mentat_m=H_mentat, scale_height_ISA_constants_m=H_isa_M,
             mean_molar_mass_engine=M_air, M0_ISA=ISA['M0'], dry_lapse_engine_K_per_km=Gamma_d * 1000.0,
             ISA_lapse_K_per_km=6.5, ISA_exponent=expo, engine_exponent=expo_engine),
    isa_constants_LIT=ISA, isa_anchor_check=anchor_check, rows=rows, engine_polytrope=poly,
    verdicts=dict(pressure_within_5pct_to_5km=p_ok_5km, pressure_beyond_5pct_above_8km=p_fail_above_8,
                  pressure_5pct_crossing_altitude_m=cross, T_dry_within_5pct_to_11km=T_ok,
                  lapse_within_10pct_of_6p5=lapse_ok),
    sigma_column=dict(w_sea_level=w_sea, three_w=three_w, tolman_over_scale_height=tolman_H,
                      tolman_over_11km=tolman_11, cs2_over_c2=w_cs),
    mentat_leaves=leaves)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('H_eq_cs2_over_g', 'K_line_times_h2', 'kernel3d_norm', 'air', 'isa_anchor_check', 'verdicts', 'sigma_column')}, indent=1))
for r in rows:
    print(f"{r['z_m']:7.0f}  ISA p={r['p_ISA']:9.1f} T={r['T_ISA']:7.2f} | iso p={r['p_isothermal_engine']:9.1f} ({100*r['p_rel_err']:+6.2f}%)  dryT={r['T_dry_adiabatic_engine']:7.2f} ({100*r['T_rel_err']:+6.2f}%)")
