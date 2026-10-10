"""Close the two residuals that sympy.simplify left unsimplified in kernel_v2.json (MGF vs (sinh(a/2)/(a/2))^4 and the
line FT vs (sin(k/2)/(k/2))^4) by rewriting in exponentials and expanding (Mentat simplify_expr); appends to kernel_v2.json."""
import sys, json
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
P = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11/kernel_v2.json"
k = json.load(open(P))
out = {}
for key, resid in (('mgf_closed_form_closure', k['mgf_minus_closed_form_symbolic']), ('ft1_identity_closure', k['FT1_minus_sinc4_symbolic'])):
    expr = f'expand(({resid}).rewrite(exp))'
    v, s = mentat('simplify_expr', {'expression': expr})
    out[key] = v
    k['mentat_leaves'][key] = dict(tool='simplify_expr', args={'expression': expr}, value=v, source=s)
k['identity_closures'] = out
json.dump(k, open(P, 'w'), indent=1)
print(out)
