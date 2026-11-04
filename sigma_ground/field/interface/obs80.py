"""The MPC 80-column optical observation record (MPC1992 / "obs80"), read with its designation kept: the whole-sky reader (stage 2, phase 1).

Columns (1-based): 1-5 packed number; 6-12 packed provisional designation; 13 discovery asterisk; 14 note 1; 15 note 2 (C CCD, B CMOS, S/s satellite, R/r radar, X/x
replaced, V/v roving, M micrometer, P photographic, E encoder, T meridian, A/N/n/e other, ' ' legacy); 16-32 date "YYYY MM DD.ddddd" (UTC); 33-44 RA "HH MM SS.ddd"; 45-56 Dec
"sDD MM SS.dd"; 66-70 magnitude; 71 band; 72-77 reference; 78-80 observatory code. A second line (note 2 S, R, V) carries the satellite or roving observer's position and is
not an astrometric line. The reader keeps only optical CCD/CMOS lines (note 2 C or B), as the scoreboard's parser does (scripts/eros_sky_residuals.py: parse_obs80), and
returns the designation packed exactly as the MPC wrote it so that the orbit files join on it.
"""
from __future__ import annotations

import math

KEEP_NOTE2 = frozenset("CB")


def parse_line(ln: str):
    """(packed_number, packed_provisional, note2, year, month, day_frac, ra_rad, dec_rad, mag or None, band, station) or None for a non-optical or malformed line."""
    if len(ln) < 80 or ln[14] not in KEEP_NOTE2:
        return None
    try:
        y, m, d = int(ln[15:19]), int(ln[20:22]), float(ln[23:32])
        ra = (int(ln[32:34]) + int(ln[35:37]) / 60.0 + float(ln[38:44]) / 3600.0) * 15.0
        sgn = -1.0 if ln[44] == "-" else 1.0
        dec = sgn * (int(ln[45:47]) + int(ln[48:50]) / 60.0 + float(ln[51:56] or 0.0) / 3600.0)
    except ValueError:
        return None
    mag_s = ln[65:70].strip()
    try:
        mag = float(mag_s) if mag_s else None
    except ValueError:
        mag = None
    return (ln[0:5].rstrip(), ln[5:12].rstrip(), ln[14], y, m, d, math.radians(ra), math.radians(dec), mag, ln[70:71].strip(), ln[77:80])


def packed_key(num: str, prov: str) -> str:
    """The join key with the orbit files' 7-column packed designation: the packed number when numbered, else the packed provisional designation."""
    return num if num else prov


COLUMNS = ["packed", "packed_number", "packed_prov", "note2", "year", "month", "day", "ra_rad", "dec_rad", "mag", "band", "station"]


def rows_to_columns(rows) -> dict[str, list]:
    cols = {c: [] for c in COLUMNS}
    for r in rows:
        cols["packed"].append(packed_key(r[0], r[1])); cols["packed_number"].append(r[0]); cols["packed_prov"].append(r[1]); cols["note2"].append(r[2])
        cols["year"].append(r[3]); cols["month"].append(r[4]); cols["day"].append(r[5]); cols["ra_rad"].append(r[6]); cols["dec_rad"].append(r[7])
        cols["mag"].append(r[8]); cols["band"].append(r[9]); cols["station"].append(r[10])
    return cols
