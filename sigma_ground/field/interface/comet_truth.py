"""Parser for JPL Horizons comet responses (API, format=json, OBJ_DATA=YES, EPHEM_TYPE=VECTORS, CSV, KM-D, ICRF, centre 500@0).

Returns the orbital solution's identity and non-gravitational model, and the barycentric ICRF states (km, km/day) at the requested epochs.
Anything the parser cannot find raises ValueError: a silent default for a force parameter would corrupt the test it feeds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"


def _grab(text: str, key: str, required: bool = True):
    m = re.search(rf"\b{key}=\s*{_NUM}", text)
    if m is None:
        if required:
            raise ValueError(f"Horizons response has no {key}=")
        return None
    return float(m.group(1))


def parse_horizons_comet(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    text = d["result"]
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("Horizons response has no ephemeris block (an ambiguous or unknown target?)")
    head = text.split("$$SOE")[0]
    m_name = re.search(r"Target body name:\s*(.+?)\s*\{source:\s*([^}]+)\}", head)
    m_soln = re.search(r"Soln\.date:\s*(\S+)\s+# obs:\s*(\d+)\s*\(([^)]*)\)", head)
    ng = head[head.index("non-gravitational force model"):]
    states, jds = [], []
    for ln in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        cells = [c.strip() for c in ln.split(",") if c.strip()]
        jds.append(float(cells[0]))
        states.append([float(x) for x in cells[2:8]])
    return {"target": m_name.group(1) if m_name else None, "solution_source": m_name.group(2) if m_name else None,
            "solution_date": m_soln.group(1) if m_soln else None, "n_obs": int(m_soln.group(2)) if m_soln else None, "obs_arc": m_soln.group(3) if m_soln else None,
            "A": (_grab(ng, "A1"), _grab(ng, "A2"), _grab(ng, "A3")), "DT": _grab(ng, "DT"), "AMRAT": _grab(ng, "AMRAT"),
            "law": {"alpha": _grab(ng, "ALN"), "k": _grab(ng, "NK"), "m": _grab(ng, "NM"), "n": _grab(ng, "NN"), "r0": _grab(ng, "R0")},
            "jd": np.array(jds), "state_km_kmday": np.array(states)}
