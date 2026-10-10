"""U11 v2 formula-side additions (skeptic review), every leaf through Mentat:
1. exact 5 % crossing of atmosphere.pressure_at_altitude(z, 288.15 K) against ISA 1976 [LIT] by bisection; the ISA
   value and the relative error at each probe are evaluated by Mentat simplify_expr (v1 interpolated linearly
   between the 5 and 6 km rows: 5797.7 m).
2. sigma column: for an isothermal ideal gas w = P/(rho c^2) and the Tolman shift g H / c^2 are the same quantity
   R T / (M c^2) (symbolic identity by Mentat); w re-evaluated with the 288.15 K ideal-gas density instead of the
   20 C density('air') that v1 used.
3. size of engine defect 1 for a physical air column: the material (water_ice) sound speed that cfl_timestep uses
   against the isothermal/adiabatic air sound speed at 288.15 K.
"""
import sys, json
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.field.interface import atmosphere as atm

U = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11"
leaves = {}


def M(tool, args, key):
    v, s = mentat(tool, args)
    leaves[key] = dict(tool=tool, args=args, value=v, source=s)
    return v


def ev(expr, key):
    return float(M('simplify_expr', {'expression': f'N({expr}, 17)'}, key))


R = M('lookup_constant', {'name': 'molar gas constant'}, 'R')
g0 = M('lookup_constant', {'name': 'standard acceleration of gravity'}, 'g_standard')
p_atm = M('lookup_constant', {'name': 'standard atmosphere'}, 'atm')
c = M('lookup_constant', {'name': 'c'}, 'c')
rho_air20 = M('density', {'material': 'air'}, 'rho_air_20C')
cs_air = M('speed_of_sound_in_ideal_gas', {'temperature_k': 288.15}, 'cs_air_288K')

# ISA 1976 [LIT] (same constants as v1; anchored there to PDAS Table 1 rows at 5 and 10 km to 5e-6)
ISA = dict(T0=288.15, p0=101325.0, L=0.0065, M0=0.0289644, Rstar=8.31432, g0=9.80665, r0=6356766.0)
expo = ev(f"{ISA['g0']}*{ISA['M0']}/({ISA['Rstar']}*{ISA['L']})", 'ISA_exponent')
T_surf = 288.15


def relerr(z, tag):
    p_iso = atm.pressure_at_altitude(z, T_surf)
    Hgp = f"({ISA['r0']}*{z}/({ISA['r0']}+{z}))"
    p_isa = ev(f"{ISA['p0']}*(({ISA['T0']} - {ISA['L']}*{Hgp})/{ISA['T0']})**({expo})", f'ISA_p_{tag}')
    return ev(f"{p_iso}/{p_isa} - 1", f'relerr_{tag}'), p_iso, p_isa


lo, hi = 5000.0, 6000.0
r_lo = relerr(lo, 'lo')[0]
r_hi = relerr(hi, 'hi')[0]
assert r_lo < 0.05 < r_hi
probes = []
for i in range(16):
    mid = 0.5 * (lo + hi)
    r, p_iso, p_isa = relerr(mid, f'bisect_{i}')
    probes.append(dict(z_m=mid, rel_err=r, p_engine=p_iso, p_ISA=p_isa))
    if r > 0.05:
        hi = mid
    else:
        lo = mid
crossing = 0.5 * (lo + hi)
check = relerr(crossing, 'crossing')

# sigma column
M_air = atm.mean_molar_mass_kg_mol()
H_air = atm.scale_height(T_surf)
identity = M('simplify_expr', {'expression': '(P/(P*M/(R*T)))/c**2 - g*(R*T/(M*g))/c**2'}, 'w_equals_gH_over_c2')
rho_288 = ev(f"{p_atm}*{M_air}/({R}*{T_surf})", 'rho_air_ideal_288K')
w_288 = ev(f"{p_atm}/({rho_288}*{c}**2)", 'w_288K')
w_20C = ev(f"{p_atm}/({rho_air20}*{c}**2)", 'w_with_20C_density')
RT_Mc2 = ev(f"{R}*{T_surf}/({M_air}*{c}**2)", 'RT_over_Mc2')
tolman_H = ev(f"{g0}*{H_air}/{c}**2", 'tolman_over_scale_height')
dens_ratio = ev(f"{rho_288}/{rho_air20}", 'rho288_over_rho20C')
w_ratio = ev(f"{w_20C}/{w_288}", 'w20C_over_w288')

# cfl_timestep mismatch for a physical air column
runA = json.load(open(U + '/run_A_literal.json'))
c_ice = runA['material_sound_speed']          # engine: water_ice parcel sound speed used by cfl_timestep
mismatch_air = ev(f"{c_ice}/{cs_air}", 'cfl_mismatch_air_288K')
mismatch_scaled = ev(f"{c_ice}/1", 'cfl_mismatch_scaled_units')

res = dict(
    crossing_5pct=dict(z_m=crossing, rel_err_at_crossing=check[0], bracket=[lo, hi], probes=probes,
                       v1_linear_interpolation_m=5797.7),
    sigma_column=dict(identity_w_minus_gH_over_c2=identity, rho_air_ideal_288K=rho_288, rho_air_20C=rho_air20,
                      w_288K=w_288, w_with_20C_density=w_20C, RT_over_Mc2=RT_Mc2,
                      tolman_over_scale_height=tolman_H, rho288_over_rho20C=dens_ratio, w20C_over_w288=w_ratio,
                      scale_height_engine_m=H_air, M_air_engine=M_air),
    cfl_defect_size=dict(material_sound_speed_water_ice=c_ice, air_sound_speed_288K=cs_air,
                         mismatch_physical_air=mismatch_air, mismatch_scaled_cs1=mismatch_scaled,
                         direction='over-conservative: dt is too SMALL by c_material/c_s (stable, slow); unsafe only when the material c is below the EOS c_s'),
    mentat_leaves=leaves)
json.dump(res, open(U + '/formula_v2.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'mentat_leaves'}, indent=1, default=str)[:4000])
