"""Pilot P6 — the σ-bridge: the repo's field equation □σ = −ξR with a perfect fluid inserted.
Trace (−,+,+,+): T = −ρc² + 3P ⇒ R = (8πG/c⁴)(ρc² − 3P) ⇒ □σ = −(8πGξ/c⁴)(ρc² − 3P).
Static weak field: ∇²σ = −(8πGξ/c²)(ρ − 3P/c²) ⇒ σ = −(2ξ/c²)Φ + σ_P, i.e. the SAME elliptic operator as the
Newtonian potential (P1) with source (ρ − 3P/c²); point mass: σ = 2ξGM/(rc²) = ξ r_s/r.
Leaf quantities from Mentat (PLATINUM §21); ξ from inventory.core.sigma (the only place it survives after e66615b).
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from mentat_client import mentat
from sigma_ground.field.constants import G, C
from sigma_ground.inventory.core.sigma import XI, sigma_from_potential

earth, src_e = mentat('solar_system_body', {'body_name': 'earth'})
sun, src_s = mentat('solar_system_body', {'body_name': 'sun'})
rho_w, src_w = mentat('density', {'material': 'water'})
rho_fe, src_fe = mentat('density', {'material': 'iron'})
P_air, src_air = mentat('ideal_gas_pressure', {'n_moles': 1.0, 'temperature_k': 288.15, 'volume_m3': 0.02364})
rho_air = 0.02897 / 0.02364   # kg/m³ from the same mole: M_air·n/V (M_air 0.02897 kg/mol, CODATA/NIST)
rho_crit, src_crit = mentat('critical_density', {})

def pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d: return d[k]
    raise KeyError(keys)
M_E, R_E = pick(earth, 'mass_kg', 'mass'), pick(earth, 'radius_m', 'mean_radius_m', 'radius')
M_S, R_S = pick(sun, 'mass_kg', 'mass'), pick(sun, 'radius_m', 'mean_radius_m', 'radius')
rs_E, src_rs = mentat('schwarzschild_radius', {'mass_kg': M_E})
rs_S, _ = mentat('schwarzschild_radius', {'mass_kg': M_S})

states = [  # (label, rho kg/m³, P Pa, provenance)
    ('air, 288.15 K, 1 atm', rho_air, P_air, src_air),
    ('water, 1 atm', rho_w, 101325.0, src_w),
    ('iron, 1 atm', rho_fe, 101325.0, src_fe),
    ('Sun core', 1.5e5, 2.5e16, '[LIT] Bahcall et al. 2005 standard solar model (not a Mentat tool)'),
    ('white dwarf centre', 1e9, 1e22, '[LIT] Chandrasekhar-mass WD, order of magnitude (not a Mentat tool)'),
    ('neutron star centre', 5e17, 1e34, '[LIT] ~3 n₀, stiff EOS, order of magnitude (not a Mentat tool)'),
    ('radiation', 1.0, C * C / 3.0, 'P = ρc²/3 exactly (trace-free)'),
    ('cosmic critical density, dust', rho_crit, 0.0, src_crit),
]
rows = []
for label, rho, P, src in states:
    w = 3 * P / (rho * C * C)
    rows.append(dict(state=label, rho_kg_m3=rho, P_Pa=P, three_P_over_rho_c2=w, source_sign='zero' if abs(1 - w) < 1e-12 else ('attractive (ρ-like)' if w < 1 else 'repulsive (stiff)'),
                     box_sigma_source=-(8 * math.pi * G * XI / C ** 4) * (rho * C * C - 3 * P), provenance=src))
sigma_fe = XI * rs_E / R_E;  sigma_inv = sigma_from_potential(R_E, M_E)
out = dict(xi=XI, xi_source='sigma_ground.inventory.core.sigma (Planck 2018 baryon fraction; excised from field/ in e66615b)',
           earth=dict(M_kg=M_E, R_m=R_E, r_s_m=rs_E, source=src_e, r_s_source=src_rs),
           sigma_earth_surface=dict(field_equation_xi_rs_over_r=sigma_fe, inventory_sigma_from_potential=sigma_inv,
                                    inventory_docstring_quote=7e-10, ratio_field_over_inventory=sigma_fe / sigma_inv),
           sigma_sun_surface_field_eq=XI * rs_S / R_S, sigma_at_horizon_field_eq=XI, sigma_at_horizon_inventory=sigma_from_potential(rs_S, M_S),
           states=rows)
print(json.dumps(out, indent=1))
