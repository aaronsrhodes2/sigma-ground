"""Serialise a Python value as a Lua table literal (for the generated spec files). Lua 5.1 safe, deterministic."""
from __future__ import annotations

import math
import re

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _str(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif o < 32 or o == 127:
            out.append("\\%03d" % o)          # Lua 5.1 has no \u escapes
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def lua(obj, indent: int = 0) -> str:
    pad = "  " * indent
    if obj is None:
        return "nil"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite number {obj!r} in a spec")
        return repr(obj)
    if isinstance(obj, str):
        return _str(obj)
    if isinstance(obj, (list, tuple)):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            return "{" + ", ".join(lua(x) for x in obj) + "}"
        inner = ",\n".join(pad + "  " + lua(x, indent + 1) for x in obj)
        return "{\n" + inner + ",\n" + pad + "}" if obj else "{}"
    if isinstance(obj, dict):
        items = []
        for k in sorted(obj):
            if obj[k] is None:
                continue
            key = k if _IDENT.match(k) else "[" + _str(k) + "]"
            items.append(pad + "  " + key + " = " + lua(obj[k], indent + 1))
        return "{\n" + ",\n".join(items) + ",\n" + pad + "}" if items else "{}"
    raise TypeError(f"cannot serialise {type(obj).__name__}")
