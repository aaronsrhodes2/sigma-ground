"""U11 v2: kernel-side derivations asked for by the skeptic review, every number from Mentat (sympy via the
skippytel-mentat container). The kernel is the engine's M4 cubic spline (sigma_ground/dynamics/fluid/kernel.py):
W(r) = (3/(2 pi h^3)) f(q), f = 2/3 - q^2 + q^3/2 (q < 1), (2 - q)^3/6 (1 <= q < 2), q = r/h.

1. Line normalisation K h^2 = 2 (3/(2 pi)) int_0^2 f dq = 3/(2 pi)      (re-derived; v1 used the same value)
2. Eigenfunction identity: for the symmetric SPH pressure force with P = c_s^2 rho_SPH, an exponential linear
   density lambda ~ exp(-y/H) gives a uniform acceleration c_s^2/H for ANY h, because
       int e^{a u} f'(|u|) sgn(u) du + a int e^{a u} f(|u|) du = 0        (integration by parts, f(2) = 0)
   so the continuum line chain has H_eq = c_s^2/g exactly: no smoothing bias in H.
3. Hydrostatic-ratio prediction: c_s^2 rho_SPH / (g K M_above) equals the kernel's normalised moment generating
   function at a = h/H, int e^{a u} f(|u|) du / int f(|u|) du, and the closed form (sinh(a/2)/(a/2))^4 (M4 = four
   boxes of width h convolved). Second order: 1 + a^2/6.
4. Fourier transforms: line chain (kernel profile f(|u|) along the line): 2 int_0^2 f cos(k q) dq = (sin(k/2)/(k/2))^4
   >= 0 for all k (pairing-stable by the Dehnen & Aly 2012 criterion); a laterally uniform 3-D slab uses the
   plane-projected kernel, whose 1-D transform is the 3-D transform along k_y:
   FT3(k) = 4 pi (3/(2 pi)) int_0^2 f q^2 sin(k q)/(k q) dq, negative in bands (scanned here by Mentat).
"""
import sys, json, re
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11/kernel_v2.json"
leaves = {}
F_IN, F_OUT = '(2/3 - q**2 + q**3/2)', '((2 - q)**3/6)'
DF_IN, DF_OUT = '(-2*q + 3*q**2/2)', '(-(2 - q)**2/2)'


def M(tool, args, key):
    v, s = mentat(tool, args)
    leaves[key] = dict(tool=tool, args=args, value=v, source=s)
    return v


def ev(expr, key):
    return float(M('simplify_expr', {'expression': f'N({expr}, 17)'}, key))


def branch(pw):
    """first (a != 0 / k != 0) branch of a sympy Piecewise string; text handling only"""
    s = str(pw)
    if not s.startswith('Piecewise(('):
        return s
    depth, i0 = 0, len('Piecewise((')
    for i in range(i0, len(s)):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            return s[i0:i]
    raise ValueError(s)


def integ(expr, lo, hi, key):
    return branch(M('integrate_expr', {'expression': expr, 'variable': 'q', 'lower': lo, 'upper': hi}, key))


res = {}
# 1. line normalisation
half = f"({integ(F_IN, '0', '1', 'int_f_inner')}) + ({integ(F_OUT, '1', '2', 'int_f_outer')})"
res['int_f_line_half'] = M('simplify_expr', {'expression': half}, 'int_f_half')
res['K_h2_symbolic'] = M('simplify_expr', {'expression': f'2*(3/(2*pi))*({half})'}, 'K_h2')
res['K_h2_minus_3_over_2pi'] = M('simplify_expr', {'expression': f'2*(3/(2*pi))*({half}) - 3/(2*pi)'}, 'K_h2_check')
res['K_h2'] = ev('3/(2*pi)', 'K_h2_numeric')

# 2./3. moment generating function (symbolic in a) and the eigenfunction identity
C = f"({integ(F_IN + '*cosh(a*q)', '0', '1', 'mgf_cosh_inner')}) + ({integ(F_OUT + '*cosh(a*q)', '1', '2', 'mgf_cosh_outer')})"
S = f"({integ(DF_IN + '*sinh(a*q)', '0', '1', 'dW_sinh_inner')}) + ({integ(DF_OUT + '*sinh(a*q)', '1', '2', 'dW_sinh_outer')})"
# int_{-2}^{2} e^{au} f(|u|) du = 2 C ; int e^{au} f'(|u|) sgn(u) du = 2 S
MGF = f"(2*({C}))/(2*({half}))"
res['eigen_identity_symbolic'] = M('simplify_expr', {'expression': f'2*({S}) + a*2*({C})'}, 'eigen_identity')
res['mgf_minus_closed_form_symbolic'] = M('simplify_expr', {'expression': f'({MGF}) - (sinh(a/2)/(a/2))**4'}, 'mgf_closed_form')
res['closed_form_series'] = M('series_expansion', {'expression': '(sinh(a/2)/(a/2))**4', 'variable': 'a', 'point': '0', 'order': 6}, 'mgf_series')
res['hydrostatic_ratio_prediction'] = {}
for a in ('0.05', '0.1', '0.2', '0.4'):
    integral = ev(f'({MGF}).subs(a, {a})', f'mgf_integral_{a}')
    closed = ev(f'(sinh({a}/2)/({a}/2))**4', f'mgf_closed_{a}')
    second = ev(f'1 + {a}**2/6', f'mgf_second_order_{a}')
    ident = ev(f'(2*({S}) + a*2*({C})).subs(a, {a})', f'eigen_identity_{a}')
    res['hydrostatic_ratio_prediction'][a] = dict(integral=integral, closed_form=closed, second_order=second,
                                                  eigen_identity_residual=ident)

# 4. Fourier transforms
FT1 = f"2*(({integ(F_IN + '*cos(k*q)', '0', '1', 'ft1_inner')}) + ({integ(F_OUT + '*cos(k*q)', '1', '2', 'ft1_outer')}))/(2*({half}))"
res['FT1_minus_sinc4_symbolic'] = M('simplify_expr', {'expression': f'({FT1}) - (sin(k/2)/(k/2))**4'}, 'ft1_identity')
FT3 = f"4*pi*(3/(2*pi))*(({integ(F_IN + '*q*sin(k*q)', '0', '1', 'ft3_inner')}) + ({integ(F_OUT + '*q*sin(k*q)', '1', '2', 'ft3_outer')}))/k"
grid = [round(0.5 * i, 2) for i in range(1, 33)]          # kh = 0.5 .. 16
res['FT_scan'] = []
for kk in grid:
    f1 = ev(f'({FT1}).subs(k, {kk})', f'ft1_{kk}')
    f3 = ev(f'({FT3}).subs(k, {kk})', f'ft3_{kk}')
    res['FT_scan'].append(dict(kh=kk, FT_line=f1, FT_slab=f3))
# bracket the negative bands of FT3 by bisection, each point by Mentat
def ft3(kk, tag):
    return ev(f'({FT3}).subs(k, {kk})', f'ft3_bisect_{tag}')
bands = []
sc = res['FT_scan']
for p, q in zip(sc[:-1], sc[1:]):
    if (p['FT_slab'] > 0) != (q['FT_slab'] > 0):
        lo, hi = p['kh'], q['kh']
        flo = p['FT_slab']
        for it in range(12):
            mid = 0.5 * (lo + hi)
            fm = ft3(mid, f'{p["kh"]}_{it}')
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        bands.append(dict(sign_change_kh=0.5 * (lo + hi), from_positive=p['FT_slab'] > 0))
res['FT3_sign_changes'] = bands
res['FT3_at_7'] = ft3(7, 'k7')
res['FT3_at_8'] = ft3(8, 'k8')
res['FT_line_min_over_scan'] = min(x['FT_line'] for x in sc)
res['mentat_leaves'] = leaves
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('mentat_leaves', 'FT_scan')}, indent=1))
print('FT scan', [(x['kh'], round(x['FT_line'], 6), round(x['FT_slab'], 6)) for x in sc])
