"""Elasticity and supply-and-demand engine.

Point elasticities of any (multi-variable) demand function, arc elasticity, markup pricing,
and market equilibrium with surplus, taxes and price controls. Shares parsing safety rules
and formatting with marginal_engine.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from marginal_engine import _FUNCS, _MAX_LEN, _SAFE, InputError, _clean, _fval, ltx, num, real_roots, tex

# No symbol splitting, so multi-letter names like Ps, Pc or Inc stay one variable.
_TRANSFORMS = standard_transformations + (implicit_multiplication, implicit_application, convert_xor)
_INCOME = {"y", "i", "m", "inc", "income"}
_ADVERTISING = {"a", "adv", "ad"}
P = sp.Symbol("P", real=True)


def parse_vars(text: str, what: str = "the function", allowed: set[str] | None = None):
    """Parse an expression in any number of named variables. Returns (expr, {name: Symbol})."""
    s = _clean(text)
    if not s:
        raise InputError(f"Enter {what}.")
    if len(s) > _MAX_LEN:
        raise InputError(f"{what.capitalize()} is too long (max {_MAX_LEN} characters).")
    if not _SAFE.match(s):
        raise InputError(f"{what.capitalize()} has characters I can't read. Use numbers, letters, + - * / ^ and brackets.")
    names = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", s)) - set(_FUNCS)
    bad = [n for n in names if keyword.iskeyword(n)]
    if bad:
        raise InputError(f'"{bad[0]}" can\'t be used as a variable name.')
    if allowed is not None and names - allowed:
        raise InputError(f'{what.capitalize()} uses "{sorted(names - allowed)[0]}". '
                         f"Use only {', '.join(sorted(allowed))} and plug numbers in for anything else.")
    syms = {n: sp.Symbol(n, real=True) for n in names}
    try:
        expr = parse_expr(s, local_dict={**_FUNCS, **syms}, transformations=_TRANSFORMS)
    except Exception:
        raise InputError(f"Couldn't read {what}. Check for a missing operator or bracket.") from None
    if not isinstance(expr, sp.Expr):
        raise InputError(f"Couldn't read {what}.")
    return sp.nsimplify(expr, rational=True), syms


def role(name: str, own: str) -> str:
    low = name.lower()
    if name == own:
        return "own price"
    if low in _INCOME:
        return "income"
    if name.startswith("P"):
        return "price of a related good"
    if low in _ADVERTISING:
        return "advertising"
    return "other"


def interpret(rl: str, e: float, name: str) -> str:
    a = abs(e)
    if rl == "own price":
        if abs(a - 1) < 1e-9:
            return "**Unit elastic**: revenue does not change with a small price change; revenue is at its maximum."
        if a > 1:
            return (f"**Elastic** (|E| > 1): a 1% price rise cuts quantity by {num(a, 3)}%, "
                    "so raising price lowers revenue and cutting price raises it.")
        return (f"**Inelastic** (|E| < 1): a 1% price rise cuts quantity by only {num(a, 3)}%, "
                "so raising price raises revenue.")
    if rl == "income":
        if e < 0:
            return f"**Inferior good** (E < 0): a 1% income rise lowers quantity by {num(a, 3)}%."
        if e > 1:
            return f"**Normal good, luxury** (E > 1): a 1% income rise raises quantity by {num(e, 3)}%."
        return f"**Normal good, necessity** (0 < E < 1): a 1% income rise raises quantity by {num(e, 3)}%."
    if rl == "price of a related good":
        if abs(e) < 1e-12:
            return "**Unrelated goods** (E = 0)."
        kind = "Substitutes" if e > 0 else "Complements"
        sign = "raises" if e > 0 else "lowers"
        return f"**{kind}** (E {'>' if e > 0 else '<'} 0): a 1% rise in {name} {sign} quantity by {num(a, 3)}%."
    sign = "raises" if e > 0 else "lowers"
    return f"A 1% increase in {name} {sign} quantity by {num(a, 3)}%."


# ---------------------------------------------------------------- point elasticity

@dataclass
class Row:
    name: str
    role: str
    deriv: sp.Expr
    deriv_value: sp.Expr
    value: sp.Expr
    elasticity: sp.Expr
    text: str


@dataclass
class PointResult:
    expr: sp.Expr
    q0: sp.Expr
    rows: list[Row]
    steps: list[str] = field(default_factory=list)
    revenue_max: tuple | None = None  # (P, Q, R) holding other variables fixed
    own: str = "P"

    def predict(self, pct: dict[str, float]) -> tuple[float, list[str]]:
        """%ΔQ ≈ Σ E_i · %Δx_i, plus the steps."""
        total, parts = 0.0, []
        for r in self.rows:
            d = float(pct.get(r.name, 0) or 0)
            if d == 0:
                continue
            e = float(r.elasticity)
            total += e * d
            parts.append(rf"{num(e, 4)} \times ({num(d, 2)}\%)")
        return total, parts


def point_elasticity(demand_text: str, values: dict[str, float], own: str = "P") -> PointResult:
    expr, syms = parse_vars(demand_text, "the demand function")
    if not syms:
        raise InputError("The demand function needs at least one variable, e.g. P.")
    missing = [n for n in sorted(syms) if values.get(n) is None]
    if missing:
        raise InputError(f"Give a value for {', '.join(missing)}.")
    at = {syms[n]: sp.nsimplify(values[n], rational=True) for n in syms}
    q0 = sp.simplify(expr.subs(at))
    if not q0.is_real or q0 <= 0:
        raise InputError(f"Quantity at these values is {num(q0)}; elasticity needs a positive quantity.")

    order = sorted(syms, key=lambda n: (n != own, role(n, own) != "income", n))
    rows, steps = [], [rf"Quantity at the given values: $Q = {ltx(expr)} = {tex(q0)}$"]
    for n in order:
        v = syms[n]
        d = sp.diff(expr, v)
        dv = sp.simplify(d.subs(at))
        e = sp.simplify(dv * at[v] / q0)
        rl = role(n, own)
        rows.append(Row(n, rl, d, dv, at[v], e, interpret(rl, float(e), n)))
        steps.append(rf"$E_{{{n}}} = \dfrac{{\partial Q}}{{\partial {n}}}\cdot\dfrac{{{n}}}{{Q}} = "
                     rf"({tex(dv)})\cdot\dfrac{{{tex(at[v])}}}{{{tex(q0)}}} = {tex(e)}$ ({rl})")

    res = PointResult(expr, q0, rows, steps, own=own)
    if own in syms:
        p = syms[own]
        q_of_p = expr.subs({syms[n]: at[syms[n]] for n in syms if n != own})
        R = p * q_of_p
        cands = [x for x in real_roots(sp.diff(R, p), p, 0, max(10 * float(at[p]), 1000)) if float(x) > 0]
        best = max(cands, key=lambda x: _fval(R, p, x), default=None)
        if best is not None and _fval(q_of_p, p, best) > 0:
            res.revenue_max = (best, sp.simplify(q_of_p.subs(p, best)), sp.simplify(R.subs(p, best)))
            steps.append(rf"Revenue $R = {own}\cdot Q = {ltx(R)}$ is largest where $dR/d{own} = 0$, i.e. where "
                         rf"$E_{{{own}}} = -1$: ${own} = {tex(best)}$, $Q = {tex(res.revenue_max[1])}$, "
                         rf"$R = {tex(res.revenue_max[2])}$.")
    return res


def inverse_point_elasticity(inverse_text: str, price: float | None = None, quantity: float | None = None) -> PointResult:
    """Own-price elasticity from an inverse demand P(Q): E = (1 / P'(Q)) · P / Q."""
    pq, syms = parse_vars(inverse_text, "the inverse demand curve", allowed={"Q", "q"})
    Q = syms.get("Q") or syms.get("q") or sp.Symbol("Q", real=True)
    if quantity is not None:
        q0 = sp.nsimplify(quantity, rational=True)
    elif price is not None:
        p_given = sp.nsimplify(price, rational=True)
        roots = [x for x in real_roots(pq - p_given, Q, 0, 1e6) if float(x) > 0]
        if not roots:
            raise InputError(f"No positive quantity gives P = {num(price)} on this demand curve.")
        q0 = roots[0]
    else:
        raise InputError("Give either the price or the quantity.")
    p0 = sp.simplify(pq.subs(Q, q0))
    slope = sp.simplify(sp.diff(pq, Q).subs(Q, q0))
    if slope == 0 or q0 <= 0:
        raise InputError("Elasticity is undefined here (flat demand or zero quantity).")
    e = sp.simplify(p0 / (q0 * slope))
    steps = [
        rf"At $Q = {tex(q0)}$: $P = {ltx(pq)} = {tex(p0)}$",
        rf"Slope $\dfrac{{dP}}{{dQ}} = {ltx(sp.diff(pq, Q))} = {tex(slope)}$, so $\dfrac{{dQ}}{{dP}} = \dfrac{{1}}{{{tex(slope)}}}$",
        rf"$E = \dfrac{{dQ}}{{dP}}\cdot\dfrac{{P}}{{Q}} = \dfrac{{1}}{{{tex(slope)}}}\cdot\dfrac{{{tex(p0)}}}{{{tex(q0)}}} = {tex(e)}$",
    ]
    row = Row("P", "own price", 1 / sp.diff(pq, Q), 1 / slope, p0, e, interpret("own price", float(e), "P"))
    res = PointResult(pq, q0, [row], steps)
    R = pq * Q
    cands = [x for x in real_roots(sp.diff(R, Q), Q, 0, 1e6) if float(x) > 0]
    best = max(cands, key=lambda x: _fval(R, Q, x), default=None)
    if best is not None:
        res.revenue_max = (sp.simplify(pq.subs(Q, best)), best, sp.simplify(R.subs(Q, best)))
        steps.append(rf"Revenue peaks where $MR = 0$ (unit elastic): $Q = {tex(best)}$, "
                     rf"$P = {tex(res.revenue_max[0])}$, $R = {tex(res.revenue_max[2])}$.")
    return res


# ---------------------------------------------------------------- arc elasticity

@dataclass
class ArcResult:
    midpoint: float
    simple: float
    revenue1: float
    revenue2: float
    steps: list[str]
    text: str


def arc_elasticity(x1: float, q1: float, x2: float, q2: float, label: str = "P", kind: str = "own price") -> ArcResult:
    if x1 == x2:
        raise InputError(f"The two {label} values must differ.")
    if q1 + q2 == 0 or x1 + x2 == 0:
        raise InputError("Averages can't be zero.")
    dq, dx = q2 - q1, x2 - x1
    mq, mx = (q1 + q2) / 2, (x1 + x2) / 2
    e = (dq / mq) / (dx / mx)
    simple = (dq / q1) / (dx / x1) if q1 and x1 else float("nan")
    r1, r2 = x1 * q1, x2 * q2
    steps = [
        rf"$\%\Delta Q = \dfrac{{Q_2 - Q_1}}{{(Q_1 + Q_2)/2}} = \dfrac{{{num(dq)}}}{{{num(mq)}}} = {num(dq / mq * 100, 2)}\%$",
        rf"$\%\Delta {label} = \dfrac{{{label}_2 - {label}_1}}{{({label}_1 + {label}_2)/2}} = \dfrac{{{num(dx)}}}{{{num(mx)}}} = {num(dx / mx * 100, 2)}\%$",
        rf"Arc elasticity $E = \dfrac{{\%\Delta Q}}{{\%\Delta {label}}} = {num(e)}$",
        rf"For comparison, using the starting point as the base: $E = \dfrac{{{num(dq)}/{num(q1)}}}{{{num(dx)}/{num(x1)}}} = {num(simple)}$. "
        "The midpoint version gives the same answer whichever direction you move, which is why it's preferred.",
    ]
    if kind == "own price":
        steps.append(rf"Revenue: ${num(x1)}\times{num(q1)} = {num(r1)}$ → ${num(x2)}\times{num(q2)} = {num(r2)}$ "
                     f"({'up' if r2 > r1 else 'down' if r2 < r1 else 'unchanged'} {num(abs(r2 - r1))}).")
    return ArcResult(e, simple, r1, r2, steps, interpret(kind, e, label))


# ---------------------------------------------------------------- markup pricing

@dataclass
class MarkupResult:
    price: float | None
    elasticity: float | None
    lerner: float | None
    markup_on_cost: float | None
    steps: list[str]


def markup(elasticity: float | None = None, mc: float | None = None, price: float | None = None) -> MarkupResult:
    """Optimal price from elasticity (P = MC·E/(1+E)) or the elasticity implied by a price."""
    if mc is None or mc < 0:
        raise InputError("Enter a marginal cost of 0 or more.")
    if elasticity is not None:
        e = float(elasticity)
        if e >= -1:
            raise InputError("At |E| ≤ 1 demand is inelastic: raising price always adds profit, so there is no "
                             "finite optimum. A profit-maximizing firm always prices where |E| > 1.")
        p = mc * e / (1 + e)
        lerner = -1 / e
        steps = [
            r"Profit is maximized where $MR = MC$, and $MR = P\left(1 + \dfrac{1}{E}\right)$.",
            rf"So $P = \dfrac{{MC}}{{1 + 1/E}} = MC\cdot\dfrac{{E}}{{1 + E}} = {num(mc)}\cdot\dfrac{{{num(e)}}}{{{num(1 + e)}}} = {num(p)}$",
            rf"Lerner index (markup as a share of price): $\dfrac{{P - MC}}{{P}} = -\dfrac{{1}}{{E}} = {num(lerner * 100, 2)}\%$",
            rf"Markup on cost: $\dfrac{{P - MC}}{{MC}} = {num((p / mc - 1) * 100, 2) if mc else '—'}\%$. "
            "The more elastic demand is, the smaller the markup.",
        ]
        return MarkupResult(p, e, lerner, (p / mc - 1) if mc else None, steps)
    if price is None:
        raise InputError("Enter either the elasticity or the current price.")
    if price <= mc:
        raise InputError("Price must be above marginal cost to back out an elasticity.")
    lerner = (price - mc) / price
    e = -1 / lerner
    steps = [
        rf"Lerner index: $\dfrac{{P - MC}}{{P}} = \dfrac{{{num(price)} - {num(mc)}}}{{{num(price)}}} = {num(lerner * 100, 2)}\%$",
        rf"If this price is profit-maximizing, $-\dfrac{{1}}{{E}} = {num(lerner, 4)}$, so $E = {num(e)}$.",
    ]
    return MarkupResult(price, e, lerner, price / mc - 1 if mc else None, steps)


# ---------------------------------------------------------------- market

@dataclass
class MarketResult:
    qd: sp.Expr
    qs: sp.Expr
    p_eq: sp.Expr
    q_eq: sp.Expr
    ed: sp.Expr
    es: sp.Expr
    choke: sp.Expr | None
    p_min: sp.Expr
    cs: sp.Expr | None
    ps: sp.Expr
    steps: list[str] = field(default_factory=list)
    tax: dict | None = None
    control: dict | None = None


def _integral(f: sp.Expr, a, b) -> sp.Expr:
    try:
        v = sp.integrate(f, (P, a, b))
        if v.has(sp.Integral):
            raise ValueError
        return sp.simplify(v)
    except Exception:
        return sp.Float(sp.Integral(f, (P, a, b)).evalf(15), 15)


def _first_root(expr, lo=0):
    for hi in (100, 10_000, 1_000_000):
        r = [x for x in real_roots(expr, P, lo, hi) if float(x) >= lo]
        if r:
            return r[0]
    return None


def market(qd_text: str, qs_text: str, tax: float = 0.0, control: float | None = None) -> MarketResult:
    qd, _ = parse_vars(qd_text, "demand Qd(P)", allowed={"P"})
    qs, _ = parse_vars(qs_text, "supply Qs(P)", allowed={"P"})
    qd, qs = qd.subs(sp.Symbol("P", real=True), P), qs.subs(sp.Symbol("P", real=True), P)

    eqs = [x for x in (real_roots(qd - qs, P, 0, 1e6) or []) if _fval(qd, P, x) > 0]
    if not eqs:
        raise InputError("Demand and supply don't cross at a positive price and quantity. Check the signs: "
                         "demand should fall with P and supply should rise.")
    pe = eqs[0]
    qe = sp.simplify(qd.subs(P, pe))
    ed = sp.simplify(sp.diff(qd, P).subs(P, pe) * pe / qe)
    es = sp.simplify(sp.diff(qs, P).subs(P, pe) * pe / qe)
    choke = _first_root(qd, 0)
    choke = choke if choke is not None and choke > pe else None
    p_min = _first_root(qs, 0) if _fval(qs, P, 0) < 0 else sp.Integer(0)
    p_min = p_min if p_min is not None else sp.Integer(0)
    cs = _integral(qd, pe, choke) if choke is not None else None
    ps = _integral(qs, p_min, pe)

    st = [
        rf"Equilibrium: $Q_d = Q_s$ → ${ltx(qd)} = {ltx(qs)}$ → $P^* = {tex(pe)}$, $Q^* = {tex(qe)}$",
        rf"Price elasticity of demand at equilibrium: $E_d = Q_d'(P)\cdot\dfrac{{P^*}}{{Q^*}} = {tex(ed)}$"
        + (" (elastic)" if abs(float(ed)) > 1 else " (inelastic)" if abs(float(ed)) < 1 else " (unit elastic)"),
        rf"Price elasticity of supply at equilibrium: $E_s = Q_s'(P)\cdot\dfrac{{P^*}}{{Q^*}} = {tex(es)}$",
        (rf"Consumer surplus = area under demand above $P^*$: $\int_{{{tex(pe)}}}^{{{tex(choke)}}} Q_d\,dP = {tex(cs)}$ "
         rf"(demand reaches zero at $P = {tex(choke)}$)") if cs is not None else
        "Consumer surplus is unbounded here because demand never reaches zero.",
        rf"Producer surplus = area above supply below $P^*$: $\int_{{{tex(p_min)}}}^{{{tex(pe)}}} Q_s\,dP = {tex(ps)}$",
    ]
    res = MarketResult(qd, qs, pe, qe, ed, es, choke, p_min, cs, ps, st)

    if tax:
        t = sp.nsimplify(tax, rational=True)
        qs_t = qs.subs(P, P - t)  # sellers keep P - t
        r = [x for x in real_roots(qd - qs_t, P, 0, 1e6) if _fval(qd, P, x) > 0]
        if not r:
            raise InputError("With this tax the market shuts down (no positive quantity is traded).")
        pb = r[0]
        psell = pb - t
        qt = sp.simplify(qd.subs(P, pb))
        rev = t * qt
        cs1 = _integral(qd, pb, choke) if choke is not None else None
        ps1 = _integral(qs, p_min, psell) if psell > p_min else sp.Integer(0)
        dwl = (cs + ps - cs1 - ps1 - rev) if cs is not None else None
        share_c = (pb - pe) / t
        approx = es / (es - ed)
        word = "tax" if t > 0 else "subsidy"
        res.tax = dict(t=t, pb=pb, ps=psell, q=qt, revenue=rev, cs=cs1, ps_surplus=ps1, dwl=dwl,
                       consumer_share=share_c, approx_share=approx)
        st += [
            rf"Per-unit {word} $t = {tex(abs(t))}$ on sellers: sellers keep $P - t$, so solve $Q_d(P) = Q_s(P - t)$ → "
            rf"buyers pay $P_b = {tex(pb)}$, sellers get $P_s = {tex(psell)}$, $Q = {tex(qt)}$",
            rf"Burden: buyers {'pay' if t > 0 else 'save'} ${tex(abs(pb - pe))}$ more per unit "
            rf"({num(float(share_c) * 100, 2)}%), sellers ${tex(abs(pe - psell))}$ ({num((1 - float(share_c)) * 100, 2)}%). "
            rf"Check with elasticities: buyers' share $\approx \dfrac{{E_s}}{{E_s - E_d}} = {tex(approx)}$. "
            "The less elastic side bears more of the tax.",
            rf"{'Tax revenue' if t > 0 else 'Subsidy cost'}: $t\cdot Q = {tex(abs(rev))}$"
            + (rf"; deadweight loss $= (CS_0 + PS_0) - (CS_1 + PS_1 \pm \text{{revenue}}) = {tex(dwl)}$" if dwl is not None else ""),
        ]

    if control is not None:
        pc = sp.nsimplify(control, rational=True)
        d_, s_ = sp.simplify(qd.subs(P, pc)), sp.simplify(qs.subs(P, pc))
        if pc < pe:
            kind, gap = "ceiling", d_ - s_
            msg = (rf"A price ceiling at $P = {tex(pc)}$ is **binding** (below $P^* = {tex(pe)}$): "
                   rf"$Q_d = {tex(d_)}$, $Q_s = {tex(s_)}$, **shortage** of ${tex(gap)}$. Only ${tex(s_)}$ units are sold.")
        elif pc > pe:
            kind, gap = "floor", s_ - d_
            msg = (rf"A price floor at $P = {tex(pc)}$ is **binding** (above $P^* = {tex(pe)}$): "
                   rf"$Q_d = {tex(d_)}$, $Q_s = {tex(s_)}$, **surplus** of ${tex(gap)}$. Only ${tex(d_)}$ units are sold.")
        else:
            kind, gap, msg = "none", 0, "The controlled price equals the equilibrium price, so it changes nothing."
        res.control = dict(price=pc, kind=kind, qd=d_, qs=s_, gap=gap)
        st.append(msg + " (A ceiling above, or a floor below, the equilibrium price would not bind.)")
    return res
