"""The Minor Planet Center's orbit files (MPCORB.DAT, DAILY.DAT, NEA*.txt): the fixed-column record, the packed designations and epochs, and the state at epoch.

Format (MPC "Export Format for Minor-Planet Orbits", 1-based columns): 1-7 packed designation; 9-13 H; 15-19 G; 21-25 packed epoch; 27-35 mean anomaly M (deg);
38-46 argument of perihelion (deg); 49-57 longitude of ascending node (deg); 60-68 inclination (deg); 71-79 eccentricity; 81-91 mean daily motion (deg/day); 93-103 semimajor
axis (au); 106 uncertainty parameter U; 108-116 reference; 118-122 number of observations; 124-126 number of oppositions; 128-136 arc (years "1845-2026" or days "12 days");
138-141 rms residual (arcsec); 143-145 coarse perturbers; 147-149 precise perturbers; 151-160 computer; 162-165 hex flags; 167-194 readable designation; 195-202 date of last
observation (YYYYMMDD). Angles are J2000.0 ecliptic. The elements are MEASURED input (the MPC's fit); nothing here is derived physics.

Packed forms (MPC documentation "Packed Provisional and Permanent Designations"):
  numbered    : 5 columns; digits, or a letter A-Z, a-z as the ten-thousands digit (A0029 = 100029), or '~' + four base-62 characters + 620000.
  provisional : 7 columns; century letter (I 18, J 19, K 20) + 2-digit year + half-month letter + cycle (digit or letter a-z/A-Z for 10..61 followed by a digit) + order letter.
  epoch       : 5 columns; century letter + 2-digit year + month (1-9, A-C) + day (1-9, A-V), 00:00 TT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_CENTURY = {"I": 1800, "J": 1900, "K": 2000, "L": 2100}


def _b62(ch: str) -> int:
    return _B62.index(ch)


def unpack_number(p: str) -> int:
    p = p.strip()
    if p[0] == "~":
        n = 0
        for ch in p[1:]:
            n = n * 62 + _b62(ch)
        return 620000 + n
    if p[0].isdigit():
        return int(p)
    return _b62(p[0]) * 10000 + int(p[1:])


def pack_number(n: int) -> str:
    """The 5-column packed form of a permanent number: 00433, A0029 (100029), ~0J43 (693287)."""
    if n < 100000:
        return f"{n:05d}"
    if n < 620000:
        return _B62[n // 10000] + f"{n % 10000:04d}"
    x = n - 620000; out = ""
    for _ in range(4):
        out = _B62[x % 62] + out; x //= 62
    return "~" + out


def unpack_provisional(p: str) -> str:
    """K25MM4T -> '2025 MT224'; survey forms (PLS2040 -> '2040 P-L', T1S3138 -> '3138 T-1') and the extended '_' form are passed through unpacked where possible."""
    p = p.strip()
    if p[:3] in ("PLS", "T1S", "T2S", "T3S"):
        return f"{p[3:]} {p[0]}-{p[1]}" if p[0] == "P" else f"{p[3:]} T-{p[1]}"
    if p[0] in _CENTURY and len(p) == 7:
        year = _CENTURY[p[0]] + int(p[1:3])
        cyc = _b62(p[4]) * 10 + int(p[5])
        return f"{year} {p[3]}{p[6]}{cyc if cyc else ''}"
    return p


def unpack_designation(p: str) -> tuple[int | None, str]:
    """(number or None, readable designation) from the 7-column packed field: a numbered body's packed number sits in columns 1-5."""
    p = p.rstrip()
    if len(p) <= 5 or p[5:7] == "  ":
        n = unpack_number(p[:5]); return n, str(n)
    return None, unpack_provisional(p)


def unpack_epoch_jd(p: str) -> float:
    """Packed epoch 'K2669' -> JD at 00:00 TT of 2026-06-09 (the MPC's epochs are TT)."""
    year = _CENTURY[p[0]] + int(p[1:3])
    month = _b62(p[3]); day = _b62(p[4])
    a = (14 - month) // 12; y = year + 4800 - a; m = month + 12 * a - 3                                          # Fliegel-Van Flandern, proleptic Gregorian
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5


@dataclass(frozen=True)
class Orbit:
    packed: str
    number: int | None
    designation: str
    h_mag: float | None
    g_slope: float | None
    epoch_jd_tt: float
    ma_deg: float
    w_deg: float
    om_deg: float
    i_deg: float
    e: float
    n_deg_day: float
    a_au: float
    u_param: str
    reference: str
    n_obs: int | None
    n_opp: int | None
    arc: str
    rms_arcsec: float | None
    flags_hex: str
    readable: str
    last_obs_yyyymmdd: str

    @property
    def q_au(self) -> float:
        return self.a_au * (1.0 - self.e)

    @property
    def orbit_class(self) -> str:
        """nea (q < 1.3 au), mba, outer (a > 5.5 au); the step groups of the batch engine."""
        if self.q_au < 1.3:
            return "nea"
        return "outer" if self.a_au > 5.5 else "mba"


def _f(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def _i(s: str) -> int | None:
    s = s.strip()
    return int(s) if s.isdigit() else None


def parse_line(ln: str) -> Orbit | None:
    """One MPCORB record, or None for a header, blank or malformed line."""
    if len(ln) < 103 or not ln[0].strip() or ln[0] in "#-":
        return None
    try:
        number, desig = unpack_designation(ln[0:7])
        return Orbit(packed=ln[0:7].rstrip(), number=number, designation=desig, h_mag=_f(ln[8:13]), g_slope=_f(ln[14:19]), epoch_jd_tt=unpack_epoch_jd(ln[20:25]),
                     ma_deg=float(ln[26:35]), w_deg=float(ln[37:46]), om_deg=float(ln[48:57]), i_deg=float(ln[59:68]), e=float(ln[70:79]), n_deg_day=float(ln[80:91]),
                     a_au=float(ln[92:103]), u_param=ln[105:106].strip(), reference=ln[107:116].strip(), n_obs=_i(ln[117:122]), n_opp=_i(ln[123:126]), arc=ln[127:136].strip(),
                     rms_arcsec=_f(ln[137:141]), flags_hex=ln[161:165].strip(), readable=ln[166:194].strip(), last_obs_yyyymmdd=ln[194:202].strip())
    except (ValueError, IndexError):
        return None


def parse_lines(lines) -> list[Orbit]:
    """Records of an MPCORB-format file; the header (everything before the first record line, ended by a line of dashes in MPCORB.DAT) is skipped."""
    out = []
    started = False
    for ln in lines:
        if not started:
            if ln.startswith("----"):
                started = True; continue
            o = parse_line(ln)                                               # DAILY.DAT and the NEA files have no header
            if o is None:
                continue
            started = True; out.append(o); continue
        o = parse_line(ln)
        if o is not None:
            out.append(o)
    return out


def state_at_epoch(o: Orbit, kernel, gm_sun_km3_s2: float) -> np.ndarray:
    """Barycentric ICRF state (km, km/day) at the record's epoch: the Kepler state on the Sun's barycentric motion (perturber_paths.barycentric_state_at_epoch)."""
    from sigma_ground.field.interface.asteroid_perturbers import Perturber
    from sigma_ground.field.interface.perturber_paths import barycentric_state_at_epoch
    p = Perturber(o.designation, 0.0, 0.0, o.epoch_jd_tt, o.a_au, o.e, o.i_deg, o.om_deg, o.w_deg, o.ma_deg)
    return barycentric_state_at_epoch(p, kernel, gm_sun_km3_s2)[1]


def states_at_epoch(orbits: list[Orbit], kernel, gm_sun_km3_s2: float) -> np.ndarray:
    """(N, 6) states; the Sun's state is looked up once per distinct epoch."""
    from sigma_ground.field.interface.asteroid_perturbers import Perturber, kepler_state_ecliptic
    from sigma_ground.field.interface.fast_ephemeris import _ECL_TO_EQ
    DAY_S = 86400.0
    sun = {}
    out = np.empty((len(orbits), 6))
    for k, o in enumerate(orbits):
        if o.epoch_jd_tt not in sun:
            sp, sv = kernel[0, 10].compute_and_differentiate(o.epoch_jd_tt)
            sun[o.epoch_jd_tt] = (np.asarray(sp, dtype=float), np.asarray(sv, dtype=float))
        r_ecl, v_ecl = kepler_state_ecliptic(Perturber(o.designation, 0.0, 0.0, o.epoch_jd_tt, o.a_au, o.e, o.i_deg, o.om_deg, o.w_deg, o.ma_deg), gm_sun_km3_s2)
        sp, sv = sun[o.epoch_jd_tt]
        out[k, :3] = _ECL_TO_EQ @ r_ecl + sp
        out[k, 3:] = (_ECL_TO_EQ @ v_ecl) * DAY_S + sv
    return out
