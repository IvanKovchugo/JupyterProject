"""Marginal analysis engine: symbolic derivatives (SymPy) plus exact/numeric optimization.

Everything the Streamlit app shows is computed here so it can be tested without a UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

Q = sp.Symbol("Q", real=True)

_FUNCS = {"sqrt": sp.sqrt, "ln": sp.log, "log": sp.log, "exp": sp.exp, "e": sp.E, "pi": sp.pi}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
# parse_expr evaluates Python, so only digits, operators, brackets and whitelisted names get through.
_SAFE = re.compile(r"^[0-9A-Za-z\s+\-*/^().]*$")
_MAX_LEN = 200


class InputError(ValueError):
    """A user-facing message about an expression that cannot be used."""


# ---------------------------------------------------------------- parsing

def _clean(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"[−–—]", "-", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # decimal comma
    return s.replace("×", "*").replace("·", "*")


def parse(text: str, what: str = "the expression", variable: str | None = "Q") -> tuple[sp.Expr, sp.Symbol]:
    """Parse user text into an exact SymPy expression.

    variable="Q" accepts Q or q as quantity. variable=None accepts any one single-letter variable
    (defaulting to x) and returns the symbol it found.
    """
    s = _clean(text)
    if not s:
        raise InputError(f"Enter {what}.")
    if len(s) > _MAX_LEN:
        raise InputError(f"{what.capitalize()} is too long (max {_MAX_LEN} characters).")
    if not _SAFE.match(s):
        raise InputError(f"{what.capitalize()} has characters I can't read. Use numbers, letters, + - * / ^ and brackets.")

    names = set(re.findall(r"[A-Za-z]+", s))
    unknown = names - set(_FUNCS)
    if variable == "Q":
        bad = unknown - {"Q", "q"}
        if bad:
            raise InputError(f'{what.capitalize()} uses "{sorted(bad)[0]}". Write quantity as Q, e.g. 0.5Q or Q^2.')
        sym = Q
        local = {"Q": Q, "q": Q}
    else:
        if any(len(n) > 1 for n in unknown):
            raise InputError(f'"{sorted(n for n in unknown if len(n) > 1)[0]}" is not a known function. '
                             "Use a single-letter variable and sqrt, ln, exp.")
        if len(unknown) > 1:
            raise InputError(f"Use one variable only. Found: {', '.join(sorted(unknown))}.")
        name = next(iter(unknown), "x")
        sym = sp.Symbol(name, real=True)
        local = {name: sym}

    try:
        expr = parse_expr(s, local_dict={**_FUNCS, **local}, transformations=_TRANSFORMS)
    except Exception:
        raise InputError(f"Couldn't read {what}. Check for a missing operator or bracket.") from None
    if not isinstance(expr, sp.Expr):
        raise InputError(f"Couldn't read {what}.")
    expr = sp.nsimplify(expr, rational=True)  # 0.5 -> 1/2 so answers come out exact
    extra = expr.free_symbols - {sym}
    if extra:
        raise InputError(f"{what.capitalize()} has an unexpected symbol: {extra.pop()}.")
    return expr, sym


# ---------------------------------------------------------------- numerics

def numeric(expr: sp.Expr, var: sp.Symbol = Q):
    """Vectorised float evaluator; undefined points become nan."""
    fn = sp.lambdify(var, expr, modules=["numpy"])

    def g(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            try:
                y = np.asarray(fn(x))
            except (ValueError, TypeError, ZeroDivisionError):
                return np.full(x.shape, np.nan)
        if np.iscomplexobj(y):
            y = np.where(np.abs(y.imag) < 1e-12, y.real, np.nan)
        return np.broadcast_to(y.astype(float), x.shape).copy()

    return g


def _fval(expr: sp.Expr, var: sp.Symbol, x) -> float:
    try:
        v = complex(sp.N(expr.subs(var, x)))
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    return v.real if abs(v.imag) < 1e-12 else float("nan")


def _symbolic_ok(expr: sp.Expr, var: sp.Symbol) -> bool:
    try:
        return expr.is_polynomial(var) and sp.degree(expr, var) <= 4
    except sp.PolynomialError:
        return False


def real_roots(expr: sp.Expr, var: sp.Symbol, lo, hi) -> list[sp.Expr]:
    """Roots of expr in [lo, hi]: exact for polynomials up to degree 4, numeric otherwise."""
    expr = sp.simplify(expr)
    if expr.is_number:
        return []
    lo, hi = sp.nsimplify(lo), sp.nsimplify(hi)
    num_expr = sp.numer(sp.together(expr))
    if _symbolic_ok(num_expr, var):
        sols = sp.solveset(sp.Eq(num_expr, 0), var, sp.Interval(lo, hi))
        if isinstance(sols, sp.FiniteSet):
            good = [s for s in sols if s.is_real and abs(_fval(expr, var, s)) < 1e-9 * max(1, abs(float(s)))]
            return sorted(good, key=float)
    g = numeric(expr, var)
    xs = np.linspace(float(lo), float(hi), 6001)
    ys = g(xs)
    out: list[float] = []
    for i in range(len(xs) - 1):
        a, b, ya, yb = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if not (np.isfinite(ya) and np.isfinite(yb)):
            continue
        if ya == 0:
            out.append(a)
        elif ya * yb < 0:
            for _ in range(200):
                m = (a + b) / 2
                ym = float(g(np.array([m]))[0])
                if ym == 0 or b - a < 1e-14 * max(1.0, abs(m)):
                    break
                if ya * ym < 0:
                    b = m
                else:
                    a, ya = m, ym
            out.append((a + b) / 2)
    if np.isfinite(ys[-1]) and ys[-1] == 0:
        out.append(xs[-1])
    res: list[sp.Expr] = []
    for x in out:
        r = round(x, 9)
        x = r if abs(x - r) < 1e-10 else x
        if not res or abs(x - float(res[-1])) > 1e-7 * max(1, abs(x)):
            res.append(sp.nsimplify(x, rational=True) if x == round(x, 6) else sp.Float(x, 15))
    return res


def first_positive_root(expr: sp.Expr, var: sp.Symbol = Q):
    for limit in (10, 100, 1_000, 10_000, 100_000, 1_000_000):
        r = [x for x in real_roots(expr, var, 0, limit) if float(x) > 1e-9]
        if r:
            return r[0]
    return None


@dataclass
class Extremum:
    x: sp.Expr
    y: sp.Expr
    where: str  # "interior", "lower" or "upper"


def argmax(f: sp.Expr, df: sp.Expr, lo, hi, var: sp.Symbol = Q) -> Extremum | None:
    best = None
    for x in [sp.nsimplify(lo), sp.nsimplify(hi), *real_roots(df, var, lo, hi)]:
        y = _fval(f, var, x)
        if np.isfinite(y) and (best is None or y > best[1] + 1e-12):
            best = (x, y)
    if best is None:
        return None
    x = best[0]
    where = "lower" if abs(float(x) - float(lo)) < 1e-9 else "upper" if abs(float(x) - float(hi)) < 1e-9 else "interior"
    return Extremum(x, sp.simplify(f.subs(var, x)), where)


# ---------------------------------------------------------------- formatting

def num(x, d: int = 4) -> str:
    """Plain decimal for tables and metrics."""
    if x is None:
        return "—"
    v = x if isinstance(x, (int, float)) else _fval(sp.sympify(x), Q, 0)
    if not np.isfinite(v):
        return "∞" if v == float("inf") else "—"
    r = round(v, d)
    if r == 0:
        r = 0.0
    s = f"{r:,.{d}f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def tex(x) -> str:
    """LaTeX for a value: exact form plus decimal when the exact form isn't a short decimal."""
    x = sp.nsimplify(x) if not isinstance(x, sp.Basic) else x
    if x.is_Integer:
        return sp.latex(x)
    d = num(x)
    if x.is_Rational:
        dec = sp.Rational(str(float(x)))
        return d.replace(",", "{,}") if dec == x and len(d) <= 12 else rf"{sp.latex(x)} \approx {d.replace(',', '{,}')}"
    if isinstance(x, sp.Float):
        return d.replace(",", "{,}")
    return rf"{sp.latex(x)} \approx {d.replace(',', '{,}')}"


def ltx(expr: sp.Expr) -> str:
    e = sp.expand(expr) if expr.is_polynomial(*expr.free_symbols) else sp.simplify(expr)
    return sp.latex(e, order="lex" if e.is_polynomial(*e.free_symbols) else None)


# ---------------------------------------------------------------- the firm

@dataclass
class Scenario:
    title: str
    rule: str
    opt: Extremum | None


@dataclass
class Result:
    mode: str
    P: sp.Expr
    R: sp.Expr
    C: sp.Expr
    MR: sp.Expr
    MC: sp.Expr
    profit: sp.Expr
    dprofit: sp.Expr
    d2profit: sp.Expr
    q_max: sp.Expr
    capacity_set: bool
    opt: Extremum | None
    rev: Extremum | None
    break_even: list
    unbounded: bool
    rev_unbounded: bool
    share: sp.Expr
    steps: list[str] = field(default_factory=list)
    franchise: list[Scenario] = field(default_factory=list)
    linear_break_even: dict | None = None

    def price_at(self, x):
        return sp.simplify(self.P.subs(Q, x)) if self.mode != "revenue" else sp.simplify(sp.cancel(self.R / Q).subs(Q, x))


def analyze(mode: str, demand_text: str, cost_text: str, share_pct: float = 0, capacity: float | None = None) -> Result:
    if mode not in ("demand", "revenue", "price"):
        raise InputError("Unknown mode.")
    if mode == "demand":
        P, _ = parse(demand_text, "the demand curve")
        R = sp.expand(P * Q)
    elif mode == "revenue":
        R, _ = parse(demand_text, "the revenue function")
        P = sp.cancel(R / Q)
    else:
        P, _ = parse(demand_text, "the price")
        if Q in P.free_symbols:
            raise InputError('A fixed price can\'t depend on Q. Pick "Demand curve P(Q)" instead.')
        R = P * Q
    C, _ = parse(cost_text, "the cost function")

    share = sp.nsimplify(min(max(float(share_pct or 0), 0), 99), rational=True) / 100
    MR, MC = sp.diff(R, Q), sp.diff(C, Q)
    profit = sp.expand(R - C) if (R - C).is_polynomial(Q) else R - C
    dprofit = sp.diff(profit, Q)
    d2profit = sp.diff(dprofit, Q)

    cap_set = capacity is not None and capacity > 0
    if cap_set:
        q_max = sp.nsimplify(capacity, rational=True)
    else:
        end = first_positive_root(P if mode == "demand" else R) if mode != "price" else None
        if end is not None:
            q_max = end
        else:
            crit = first_positive_root(dprofit)
            if crit is not None:
                q_max = 2 * crit
            else:
                be = first_positive_root(profit)
                q_max = max(3 * be, sp.Integer(10)) if be is not None else sp.Integer(100)

    opt = argmax(profit, dprofit, 0, q_max)
    rev = argmax(R, MR, 0, q_max)
    be = real_roots(profit, Q, 0, q_max)
    unbounded = bool(opt and opt.where == "upper" and _fval(dprofit, Q, q_max) > 1e-12)
    rev_unbounded = bool(rev and rev.where == "upper" and _fval(MR, Q, q_max) > 1e-12)

    res = Result(mode, P, R, C, MR, MC, profit, dprofit, d2profit, q_max, cap_set, opt, rev, be,
                 unbounded, rev_unbounded, share)

    if mode == "price" and sp.diff(C, Q, 2) == 0 and MC.is_number:
        F, c = C.subs(Q, 0), MC
        res.linear_break_even = {"F": F, "c": c, "P": P, "Q": F / (P - c) if P > c else None}

    if share > 0:
        keep = 1 - share
        G = keep * R - C
        res.franchise = [
            Scenario("Franchisor sets it", r"max $sR$ → $MR = 0$", argmax(R, MR, 0, q_max)),
            Scenario("Franchisee sets it", rf"max $(1-s)R - C$ → ${sp.latex(keep)}\,MR = MC$", argmax(G, sp.diff(G, Q), 0, q_max)),
            Scenario("Joint / profit sharing", r"max $R - C$ → $MR = MC$", opt),
        ]

    res.steps = _steps(res)
    return res


def _steps(r: Result) -> list[str]:
    st: list[str] = []
    if r.mode == "demand":
        st.append(rf"Revenue is price times quantity: $R(Q) = P(Q)\cdot Q = {ltx(r.R)}$")
    elif r.mode == "revenue":
        st.append(rf"Revenue is given: $R(Q) = {ltx(r.R)}$, so the demand curve is $P = R/Q = {ltx(r.P)}$")
    else:
        st.append(rf"The firm is a price taker at $P = {sp.latex(r.P)}$, so $R(Q) = {ltx(r.R)}$")
    st.append(rf"Marginal revenue: $MR = \dfrac{{dR}}{{dQ}} = {ltx(r.MR)}$" + (" (equal to the price)" if r.mode == "price" else ""))
    st.append(rf"Marginal cost from $C(Q) = {ltx(r.C)}$: $MC = \dfrac{{dC}}{{dQ}} = {ltx(r.MC)}$")
    st.append(rf"Profit $\pi(Q) = R - C = {ltx(r.profit)}$, marginal profit $\pi'(Q) = MR - MC = {ltx(r.dprofit)}$")

    o = r.opt
    if r.unbounded:
        tail = (rf"At capacity $Q = {tex(o.x)}$: $P = {tex(r.price_at(o.x))}$, $\pi = {tex(o.y)}$."
                if r.capacity_set else "Enter a capacity to get a number.")
        st.append(rf"$\pi'(Q) = {ltx(r.dprofit)}$ is never zero: **MR stays above MC**, so every extra unit adds profit and "
                  rf"the MR = MC rule has no solution. Modified rule: keep producing while $P > MC$, up to capacity. {tail}")
    elif o is not None:
        if o.where == "interior":
            d2 = sp.simplify(r.d2profit.subs(Q, o.x))
            st.append(rf"Set $\pi'(Q) = 0$ (same as $MR = MC$): $Q^* = {tex(o.x)}$. "
                      rf"Check: $MR(Q^*) = {tex(r.MR.subs(Q, o.x))}$, $MC(Q^*) = {tex(r.MC.subs(Q, o.x))}$.")
            verdict = "< 0$, so this is a **maximum**" if d2 < 0 else r"\ge 0$, check the endpoints"
            st.append(rf"Second-order condition: $\pi''(Q) = {ltx(r.d2profit)}$; at $Q^*$ it equals ${tex(d2)} {verdict}.")
        else:
            st.append("No interior maximum: profit peaks at " +
                      ("$Q = 0$ (shut down)." if o.where == "lower" else rf"the upper end $Q = {tex(o.x)}$."))
        loss = " Profit is negative, so this output only minimizes the loss." if o.y < 0 else ""
        st.append(rf"At the optimum: $P^* = {tex(r.price_at(o.x))}$, $R = {tex(r.R.subs(Q, o.x))}$, "
                  rf"$C = {tex(r.C.subs(Q, o.x))}$, $\pi^* = {tex(o.y)}$.{loss}")

    if r.mode != "price" and r.rev is not None:
        if r.rev_unbounded:
            st.append("Revenue keeps rising up to the upper limit, so there is no interior revenue maximum.")
        else:
            qr = r.rev.x
            st.append(rf"Revenue maximum: set $MR = {ltx(r.MR)} = 0$ → $Q = {tex(qr)}$, $P = {tex(r.price_at(qr))}$, "
                      rf"$R = {tex(r.rev.y)}$.")
            if o is not None and not r.unbounded:
                if qr > o.x:
                    st.append(rf"Revenue-max output ${tex(qr)}$ is **greater** than profit-max output ${tex(o.x)}$. "
                              rf"At $Q = {tex(qr)}$ the last unit adds nothing to revenue ($MR = 0$) but still costs "
                              rf"$MC = {tex(r.MC.subs(Q, qr))}$. A profit-maximizer stops earlier, where $MR = MC > 0$. "
                              rf"Profit at the revenue maximum is only ${tex(r.profit.subs(Q, qr))}$.")
                elif qr == o.x:
                    st.append("Revenue-max and profit-max output coincide, which happens when MC = 0 at the optimum.")

    if r.break_even:
        pts = r", \; ".join(tex(x) for x in r.break_even)
        st.append(rf"Break-even ($\pi = 0$) at $Q = {pts}$" +
                  (": the firm makes money between these outputs." if len(r.break_even) == 2 else "."))
    else:
        st.append(rf"$\pi(Q)$ never crosses zero on $[0, {tex(r.q_max)}]$.")

    lb = r.linear_break_even
    if lb:
        val = rf"= \dfrac{{{sp.latex(lb['F'])}}}{{{sp.latex(lb['P'])} - {sp.latex(lb['c'])}}} = {tex(lb['Q'])}" if lb["Q"] is not None else ""
        st.append(rf"General break-even, fixed price and linear cost $C = F + cQ$: $\pi = PQ - F - cQ = 0$ → "
                  rf"$Q_{{BE}} = \dfrac{{F}}{{P - c}} {val}$. Each unit earns a margin of $P - c$; "
                  "you need enough units to cover the fixed cost $F$.")
    return st


# ---------------------------------------------------------------- any function

@dataclass
class FreeResult:
    var: sp.Symbol
    f: sp.Expr
    d1: sp.Expr
    d2: sp.Expr
    critical: list  # (x, f(x), f''(x), kind)
    gmax: tuple | None
    gmin: tuple | None


def optimize_any(text: str, lo: float, hi: float) -> FreeResult:
    if not hi > lo:
        raise InputError('"To" must be larger than "From".')
    f, v = parse(text, "the function", variable=None)
    d1, d2 = sp.diff(f, v), sp.diff(f, v, 2)
    crit = []
    for x in real_roots(d1, v, lo, hi):
        s2 = _fval(d2, v, x)
        kind = "local max" if s2 < -1e-10 else "local min" if s2 > 1e-10 else "flat / inflection (f'' = 0)"
        crit.append((x, sp.simplify(f.subs(v, x)), s2, kind))
    pts = [(x, _fval(f, v, x)) for x in [sp.nsimplify(lo), sp.nsimplify(hi), *[c[0] for c in crit]]]
    pts = [p for p in pts if np.isfinite(p[1])]
    gmax = max(pts, key=lambda p: p[1]) if pts else None
    gmin = min(pts, key=lambda p: p[1]) if pts else None
    return FreeResult(v, f, d1, d2, crit, gmax, gmin)
