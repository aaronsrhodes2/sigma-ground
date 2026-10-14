"""One parser for every expression a Mentat math tool takes.

Why this exists (audit 2026-10-13, misc/MENTAT_AUDIT_2026-10-13.md, finding F1): the tools called
``sympy.sympify(text, convert_xor=True)``, which reads ordinary mathematical writing wrongly and
silently: ``x²`` became a variable named ``x2`` (derivative 0), ``√x`` and ``πx`` likewise, ``e^x`` took
``e`` for an unknown (derivative ``e**x*log(e)``), ``2x``, ``sin x``, ``(x+1)(x-1)`` and ``|x|`` failed,
and ``infinity`` was a symbol. A calculator that answers a different question than the one asked, with
full confidence, is worse than one that refuses.

The reading rules (the ones Wolfram|Alpha and a textbook use):

* ``^`` and ``**`` are powers; superscript digits are powers (``x²``, ``x⁻¹``); ``√`` takes the number,
  name or parenthesis after it (``√2x`` is ``√2 · x``); ``|…|`` is the absolute value; ``n!`` is a factorial.
* Multiplication may be written by juxtaposition: ``2x``, ``3x^2``, ``(x+1)(x-1)``, ``x(x+1)``, ``2π x``.
  ``xy`` is ``x·y`` only when ``x`` and ``y`` are both declared variables; any other word is one symbol.
* A function may take its argument without parentheses: ``sin x``, ``sin 2x`` (= ``sin(2x)``), ``ln x``;
  ``sin^2 x`` is ``sin(x)^2``; ``sinx`` is ``sin(x)`` when ``x`` is a declared variable.
* Constants: ``π``/``pi``, ``e`` (Euler's number), ``i``/``I`` (imaginary unit), ``∞``/``inf``/``infinity``
  (``oo``). **A declared variable wins**: ask for the derivative with respect to ``E`` and ``E`` is a symbol.
  An undeclared ``E``/``I`` stays Euler's number / the imaginary unit and the reading is disclosed.
* Any other bare name is a symbol (a constant of the problem). The ones that are not declared variables are
  disclosed ("treated a as a constant"), as Wolfram does.
* A name followed by ``(`` must be a known function; a multi-letter unknown (``foo(x)``) is refused rather
  than invented, and a single unknown letter before ``(`` multiplies (``a(x+1)`` is ``a·(x+1)``).

Safety: the text is tokenised before anything is evaluated. Strings, attribute access, Python keywords,
double underscores, unknown calls, runaway exponents and power towers are refused; evaluation runs with no
builtins and a whitelist of sympy functions. This is the only door through which Mentat evaluates user math.

``parse`` returns a :class:`Parsed` with the sympy object, ``interpreted_as`` (how the input was read, shown
in the tool result) and ``notes`` (the disclosures above). A refusal is a :class:`MathParseError`.
"""
from __future__ import annotations

import functools
import keyword
import re
import tokenize
import unicodedata
from dataclasses import dataclass
from io import StringIO

MAX_CHARS = 4000
MAX_NUMBER_DIGITS = 400
_HUGE = re.compile(r"(\*\*)\s*\(?\s*\d{4,}"                       # x**10000
                   r"|factorial\s*\(\s*\d{4,}|\d{4,}\s*!")           # 1000!
_TOWER = re.compile(r"\d+(?:\s*\*\*\s*\(?\s*\d+\s*\)*)+")           # a chain of integer literals joined by ** (9**9**9)


def _tower_too_big(s: str) -> bool:
    """True if a chain of integer literals joined by ** would be an astronomically large number (9**9**9 is 3.7e8 digits).
    Evaluated right to left against a cap, so 2**3**2 and x**2**3 pass."""
    for m in _TOWER.finditer(s):
        nums = [int(n) for n in re.findall(r"\d+", m.group())]
        if len(nums) < 3:
            continue
        value = nums[-1]
        for base in reversed(nums[:-1]):
            if base > 1 and value * (base.bit_length()) > 40000:
                return True
            value = base ** value if base > 1 else base
    return False

_DENY_FUNCTIONS = {"Function", "WildFunction", "UndefinedFunction", "AppliedUndef", "Lambda"}
_ALIASES = {"ln": "log", "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arcsinh": "asinh", "arccosh": "acosh",
            "arctanh": "atanh", "arccot": "acot", "abs": "Abs", "sgn": "sign", "ceil": "ceiling", "min": "Min", "max": "Max"}
_SUPERSCRIPT = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                "⁻": "-", "⁺": "+"}
_SUBSCRIPT = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
_OPERATORS = {"−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "≤": "<=", "≥": ">=",
              "（": "(", "）": ")"}
_ALLOWED_OPS = {"+", "-", "*", "/", "**", "(", ")", ",", "[", "]", "<", ">", "<=", ">=", "!", "//"}


_GREEK_NAMES = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda_", "mu", "nu", "xi",
                "omicron", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"}
_GREEK_NAMES |= {g.capitalize() for g in _GREEK_NAMES}


class MathParseError(ValueError):
    """A reading Mentat refuses to guess. ``hint`` says what to write instead."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class Parsed:
    expr: object                     # a sympy expression (or a python list of them, for matrices / vectors)
    interpreted_as: str              # how the input was read, e.g. "3*x**2 + 2*x"
    notes: str                       # disclosures: "read e as Euler's number", "treated a as a constant"
    free: tuple                      # names of the free symbols that were not declared variables


@functools.lru_cache(maxsize=1)
def _namespace():
    """(functions, constants) the parser accepts, built once from sympy."""
    import sympy as sp
    funcs = {}
    for name in dir(sp):
        obj = getattr(sp, name)
        if isinstance(obj, type) and issubclass(obj, sp.Function) and name not in _DENY_FUNCTIONS and not name.startswith("_"):
            funcs[name] = obj
    funcs.update({"sqrt": sp.sqrt, "root": sp.root, "Min": sp.Min, "Max": sp.Max, "Abs": sp.Abs, "binomial": sp.binomial,
                  "Derivative": None})                        # Derivative only where a tool passes it in `functions`
    funcs.pop("Derivative")
    for alias, target in _ALIASES.items():
        if target and target in funcs:
            funcs[alias] = funcs[target]
    constants = {"pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I, "i": sp.I, "oo": sp.oo, "EulerGamma": sp.EulerGamma}
    return funcs, constants


# ── normalisation: what the text says → plain python-ish math ──────────────────────────────────────

def _greek(match) -> str:
    ch = match.group()
    if ch in ("π", "Π"):
        return " pi " if ch == "π" else " Pi "
    nm = unicodedata.name(ch, "").split()
    if len(nm) >= 4 and nm[0] == "GREEK":
        word = nm[-1].lower()
        word = "lambda_" if word == "lambda" else word
        return " " + (word if "SMALL" in nm else word.capitalize()) + " "
    return ch


def _declared_name(v) -> str:
    """A declared variable written in Greek ('θ') is the same symbol as the 'theta' the text normalises to."""
    return re.sub(r"[Α-Ωα-ω]", lambda m: _greek(m).strip(), str(v))


def _bars_to_abs(s: str) -> str:
    """|x| → Abs(x). A bar opens when nothing (or an operator, an open bracket, a comma) precedes it, else it closes."""
    if "|" not in s:
        return s
    out, depth, prev = [], 0, ""
    for ch in s:
        if ch == "|":
            opens = depth == 0 or prev in "+-*/^(,[<>="
            if opens:
                out.append("Abs(")
                depth += 1
            else:
                if depth == 0:
                    raise MathParseError("unbalanced absolute-value bars |…|", "write abs(x) or Abs(x)")
                out.append(")")
                depth -= 1
            prev = "(" if opens else ")"
            continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
    if depth:
        raise MathParseError("unbalanced absolute-value bars |…|", "write abs(x) or Abs(x)")
    return "".join(out)


def _normalise(text: str) -> str:
    if not isinstance(text, str):
        raise MathParseError(f"expected text, got {type(text).__name__}")
    s = text.strip()
    if not s:
        raise MathParseError("empty expression")
    if len(s) > MAX_CHARS:
        raise MathParseError(f"expression longer than {MAX_CHARS} characters")
    for a, b in _OPERATORS.items():
        s = s.replace(a, b)
    s = s.replace("∞", " oo ")
    s = re.sub(r"[Α-Ωα-ω]", _greek, s)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "**(" + "".join(_SUPERSCRIPT[c] for c in m.group()) + ")", s)
    s = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "_" + "".join(_SUBSCRIPT[c] for c in m.group()), s)
    s = re.sub(r"√\s*(\(|[0-9]+(?:\.[0-9]+)?|[A-Za-z_][A-Za-z_0-9]*)",
               lambda m: " sqrt" + ("(" if m.group(1) == "(" else "(" + m.group(1) + ")"), s)
    if "√" in s:
        raise MathParseError("a √ must be followed by a number, a name or a parenthesis", "write sqrt(x)")
    s = _bars_to_abs(s)
    s = re.sub(r"(?<![A-Za-z_0-9])(infinity|inf)(?![A-Za-z_0-9])", " oo ", s, flags=re.I)
    s = s.replace("^", "**")
    return s


# ── names: constants, declared variables, functions, symbols ───────────────────────────────────────

def _splits_into(word: str, declared: set) -> bool:
    return len(word) > 1 and all(c in declared for c in word)


def _rewrite(s: str, declared: set, extra_functions: dict):
    """Tokenise `s`, refuse what must never be evaluated, decide what each name is.
    Returns (text, local_dict, global_additions, disclosures)."""
    import sympy as sp
    funcs, consts = _namespace()
    local: dict = {}
    extra_globals: dict = {}
    said: list = []
    try:
        toks = list(tokenize.generate_tokens(StringIO(s).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        raise MathParseError(f"could not read '{s}': {e}") from e
    out = []
    fpos: set = set()                  # indices in `out` of function names written without brackets ("sin x")
    n = len(toks)
    for idx, t in enumerate(toks):
        ttype, tok = t.type, t.string
        if ttype in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
            continue
        if ttype == tokenize.STRING:
            raise MathParseError("quotes are not allowed in a math expression", "remove the quotes")
        if ttype == tokenize.OP and tok not in _ALLOWED_OPS:
            hint = "use solve_equation for an equation (a = b)" if tok in ("=", "==") else ""
            raise MathParseError(f"'{tok}' is not allowed in a math expression", hint)
        if ttype == tokenize.NUMBER:
            if len(tok) > MAX_NUMBER_DIGITS:
                raise MathParseError("number too long")
            out.append((ttype, tok))
            continue
        if ttype != tokenize.NAME:
            out.append((ttype, tok))
            continue
        # a NAME
        if keyword.iskeyword(tok) or "__" in tok:
            raise MathParseError(f"'{tok}' is not valid in a math expression")
        nxt = toks[idx + 1].string if idx + 1 < n else ""
        prev = toks[idx - 1].string if idx > 0 else ""
        if prev == ".":
            raise MathParseError("attribute access is not allowed in a math expression")
        is_call = nxt == "("
        if is_call:
            if tok in extra_functions:
                extra_globals[tok] = extra_functions[tok]
            elif tok in declared and len(tok) == 1:
                local[tok] = sp.Symbol(tok)                          # x(x+1): multiplication
            elif tok in funcs:
                pass
            elif re.fullmatch(r"log_?(\d+)", tok):
                base = int(re.fullmatch(r"log_?(\d+)", tok).group(1))
                extra_globals[tok] = (lambda b: (lambda a: sp.log(a, b)))(base)
            elif tok in consts and tok in declared:
                local[tok] = sp.Symbol(tok)
            elif len(tok) == 1 or _splits_into(tok, declared):
                for ch in tok:
                    local[ch] = sp.Symbol(ch)
                if len(tok) > 1:
                    out.append((tokenize.OP, "("))
                    for k, ch in enumerate(tok):
                        out.extend([(tokenize.NAME, ch)] + ([(tokenize.OP, "*")] if k < len(tok) - 1 else []))
                    out.append((tokenize.OP, ")"))
                    continue
            else:
                raise MathParseError(f"unknown function '{tok}(…)'",
                                     "known functions include sin, cos, tan, exp, log, ln, sqrt, abs, asin, acos, atan, sinh, cosh, tanh, "
                                     "gamma, erf, factorial, binomial; to multiply write " + tok + "*(…)")
            out.append((ttype, tok))
            continue
        # not a call: a declared variable, a constant, a function name used bare ("sin x"), a product of declared letters, or a symbol
        if tok in declared:
            local[tok] = sp.Symbol(tok)
            out.append((ttype, tok))
        elif tok in extra_functions:
            extra_globals[tok] = extra_functions[tok]
            out.append((ttype, tok))
        elif tok in consts:
            local[tok] = consts[tok]
            if tok in ("e", "E"):
                said.append(f"read '{tok}' as Euler's number (declare it as the variable to use it as a symbol)")
            elif tok in ("i", "I"):
                said.append(f"read '{tok}' as the imaginary unit (declare it as the variable to use it as a symbol)")
            out.append((ttype, tok))
        elif tok in funcs:
            follows = toks[idx + 1] if idx + 1 < n else None
            applies = follows is not None and (follows.type in (tokenize.NAME, tokenize.NUMBER) or follows.string in ("(", "**"))
            if applies:
                fpos.add(len(out))                                   # "sin x", "sin^2 x": the argument is wrapped below
                out.append((ttype, tok))
            elif tok in _GREEK_NAMES:
                local[tok] = sp.Symbol(tok)                          # "gamma*x": the Lorentz factor, not the gamma function
                out.append((ttype, tok))
            else:
                raise MathParseError(f"'{tok}' needs an argument", f"write {tok}(x)")
        elif _splits_into(tok, declared):
            for k, ch in enumerate(tok):
                local[ch] = sp.Symbol(ch)
                out.append((tokenize.NAME, ch))
                if k < len(tok) - 1:
                    out.append((tokenize.OP, "*"))
        else:
            cut = next((f for f in sorted(funcs, key=len, reverse=True)
                        if tok.startswith(f) and len(tok) > len(f) and (_splits_into(tok[len(f):], declared) or tok[len(f):] in declared)), None)
            if cut:                                                  # "sinx" with x declared
                out.append((tokenize.NAME, cut))
                rest = tok[len(cut):]
                for k, ch in enumerate(rest if rest not in declared else [rest]):
                    local[ch] = sp.Symbol(ch)
                    out.append((tokenize.NAME, ch))
                    if k < len(rest) - 1 and rest not in declared:
                        out.append((tokenize.OP, "*"))
            elif 2 <= len(tok) <= 3 and tok.isascii() and tok.isalpha() and tok not in _GREEK_NAMES:
                letters, k = [], 0                                   # "ax" is a·x, "xy" is x·y (as a textbook writes ax²+bx+c)
                while k < len(tok):
                    letters.append("pi" if tok[k:k + 2] == "pi" else tok[k])
                    k += len(letters[-1])
                for k, ch in enumerate(letters):
                    if ch in consts and ch not in declared:          # ue^u is u·e^u: e is Euler's number, pi is pi
                        local[ch] = consts[ch]
                        if ch in ("e", "i"):
                            said.append(f"read '{ch}' as " + ("Euler's number" if ch == "e" else "the imaginary unit"))
                    else:
                        local[ch] = sp.Symbol(ch)
                    out.append((tokenize.NAME, ch))
                    if k < len(letters) - 1:
                        out.append((tokenize.OP, "*"))
                said.append(f"read '{tok}' as " + "·".join(letters))
            else:
                local[tok] = sp.Symbol(tok)
                out.append((ttype, tok))
    return tokenize.untokenize(_wrap_bare_functions(out, fpos)), local, extra_globals, said


# ── functions written without brackets ──────────────────────────────────────────────────────────────

def _wrap_bare_functions(out: list, fpos: set) -> list:
    """`sin x cos x` is sin(x)·cos(x), `sin 2x` is sin(2x), `sin^2 x` is sin(x)^2, `sin cos x` is sin(cos(x)):
    a bare function takes the run of factors after it, up to the next function name or any operator but ** and !."""
    OP, NAME, NUMBER = tokenize.OP, tokenize.NAME, tokenize.NUMBER
    n = len(out)

    def group(i):                       # balanced ( … ) or [ … ] starting at out[i] -> (processed tokens, next index)
        depth, j = 0, i
        while j < n:
            if out[j][1] in ("(", "["):
                depth += 1
            elif out[j][1] in (")", "]"):
                depth -= 1
            j += 1
            if depth == 0:
                break
        return [out[i]] + walk(i + 1, j - 1) + [out[j - 1]], j

    def operand(i):                     # one power operand: [sign] number | name | group
        toks, j = [], i
        while j < n and out[j][1] in ("-", "+"):
            toks.append(out[j])
            j += 1
        if j < n and out[j][1] in ("(", "["):
            g, j = group(j)
            toks += g
        elif j < n and out[j][0] in (NAME, NUMBER):
            toks.append(out[j])
            j += 1
        return toks, j

    def apply(i):                       # a bare function at out[i] -> (tokens of f(arg), next index)
        toks, j = [out[i]], i + 1
        if j < n and out[j][1] == "**":
            toks.append(out[j])
            op, j = operand(j + 1)
            toks += op
        if j < n and out[j][1] == "(":
            return toks, j              # a normal call (sin(x), sin**2(x)): the parser handles it
        arg, j = argument(j)
        if not arg:
            raise MathParseError(f"'{out[i][1]}' needs an argument", f"write {out[i][1]}(x)")
        return toks + [(OP, "(")] + arg + [(OP, ")")], j

    def argument(i):
        run, first = [], True
        while i < n:
            t, s = out[i]
            if i in fpos:
                if not first:
                    break               # the next function name ends the argument: sin x cos x
                sub, i = apply(i)
                run += sub
                first = False
            elif t in (NAME, NUMBER):
                run.append(out[i])
                i += 1
                first = False
            elif s in ("(", "["):
                g, i = group(i)
                run += g
                first = False
            elif s == "**" and run:
                run.append(out[i])
                op, i = operand(i + 1)
                run += op
            elif s == "!" and run:
                run.append(out[i])
                i += 1
            else:
                break
        return run, i

    def walk(lo, hi):
        res, i = [], lo
        while i < hi:
            if i in fpos:
                toks, i = apply(i)
                res += toks
            elif out[i][1] in ("(", "["):
                g, i = group(i)
                res += g
            else:
                res.append(out[i])
                i += 1
        return res

    return walk(0, n)


# ── the parser ─────────────────────────────────────────────────────────────────────────────────────

def parse(text: str, variables=(), *, functions: dict | None = None) -> Parsed:
    """Read `text` as mathematics. ``variables``: names the caller declares as variables (they win over
    constants such as E, I, e, i). ``functions``: extra callable names a tool allows (an ODE's ``y``)."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (convert_xor, function_exponentiation, implicit_application, implicit_multiplication,
                                            parse_expr, standard_transformations)
    declared = {_declared_name(v) for v in variables if v}
    s = _normalise(text)
    if _HUGE.search(s) or _tower_too_big(s):
        raise MathParseError("exponent or factorial too large (runaway computation refused)",
                             "use an exponent below 1000 and no power towers (9^9^9)")
    funcs, _consts = _namespace()
    rewritten, local, extra_globals, said = _rewrite(s, declared, functions or {})
    global_dict = {"__builtins__": {}, **funcs, **extra_globals, "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float,
                   "Rational": sp.Rational, "factorial": sp.factorial}
    transforms = standard_transformations + (implicit_multiplication, implicit_application, function_exponentiation, convert_xor)
    try:
        obj = parse_expr(rewritten, local_dict=dict(local), global_dict=global_dict, transformations=transforms)
    except MathParseError:
        raise
    except Exception as e:                                           # SyntaxError, TypeError, TokenError, sympy errors
        raise MathParseError(f"could not read '{text}' as a mathematical expression ({type(e).__name__}: {str(e)[:120]})",
                             "check the brackets and operators; write products with * if unsure") from e
    free = ()
    if isinstance(obj, (list, tuple)):
        shown = str(sp.Matrix(obj).tolist()) if obj and isinstance(obj[0], (list, tuple)) else str([str(x) for x in obj])
    else:
        shown = sp.sstr(obj)
        names = sorted(str(sym) for sym in getattr(obj, "free_symbols", ()))
        free = tuple(nm for nm in names if nm not in declared)
        if declared and free:
            said.append("treated " + ", ".join(free) + (" as a constant" if len(free) == 1 else " as constants"))
    return Parsed(expr=obj, interpreted_as=shown, notes="; ".join(said), free=free)


def parse_point(text: str, variables=()) -> Parsed:
    """A limit point or an integration bound: a number, an expression, or ±infinity (``oo``, ``∞``, ``infinity``)."""
    return parse(text, variables)


def infer_variable(expr, preferred=("x", "t", "y", "z", "n", "s", "u", "r", "k")):
    """The variable to work with when the caller did not name one. -> (name, disclosure).
    One free symbol: that one. None: ``x`` (the expression is a constant). Several: the first of ``preferred`` present,
    else the first alphabetically, and the choice is disclosed. ``expr`` is a sympy object, a list of them, or names."""
    if isinstance(expr, (list, tuple, set, frozenset)):
        syms = sorted({str(s) for e in expr for s in (getattr(e, "free_symbols", None) or ([e] if isinstance(e, str) else ()))})
    else:
        syms = sorted(str(s) for s in getattr(expr, "free_symbols", ()))
    if len(syms) == 1:
        return syms[0], f"variable not given: used {syms[0]}"
    if not syms:
        return "x", "variable not given and the expression has none: used x"
    choice = next((p for p in preferred if p in syms), syms[0])
    others = ", ".join(s for s in syms if s != choice)
    return choice, f"variable not given: used {choice} (treated {others} as constant{'s' if len(syms) > 2 else ''})"


def resolve_variable(variable, expression_text: str):
    """(name, disclosure): the caller's variable, or one inferred from the expression."""
    if variable:
        return str(variable), ""
    return infer_variable(parse(expression_text).expr)
