"""Elasticity and supply-and-demand calculators (Streamlit page)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from elasticity_engine import P, arc_elasticity, inverse_point_elasticity, markup, market, parse_vars, point_elasticity
from marginal_engine import InputError, num, numeric, tex

st.set_page_config(page_title="Elasticity & Markets", page_icon="📉", layout="wide")
DARK, BLUE, GREEN, GREY = "#0b3d2c", "#3f5f8a", "#2fb574", "#8a949b"

st.title("Elasticity & Markets")
st.caption("Point and arc elasticity, markup pricing, and supply-and-demand equilibrium with surplus, taxes "
           "and price controls. Type any function; every step is written out.")


def steps_md(steps):
    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))


def curve_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quantity Q",
                      yaxis_title="Price P", legend=dict(orientation="h", y=1.08))
    return fig


t_point, t_arc, t_markup, t_market = st.tabs(["Point elasticity", "Arc elasticity", "Markup pricing", "Supply & demand"])

# ---------------------------------------------------------------- point elasticity
with t_point:
    examples = {
        "Several variables: Q = 400 − 2P + 0.5Ps + 0.1Y": ("400 - 2P + 0.5Ps + 0.1Y", {"P": 100, "Ps": 200, "Y": 1000}),
        "Linear: Q = 100 − 2P": ("100 - 2P", {"P": 10}),
        "Constant elasticity: Q = 1000·P^−1.5·Y^0.8": ("1000 P^(-1.5) Y^0.8", {"P": 4, "Y": 50}),
    }
    kind = st.radio("What does the problem give you?", ["Demand Q = f(P, …)", "Inverse demand P(Q)"], horizontal=True)
    if kind.startswith("Demand"):
        ex = st.selectbox("Start from an example (or type your own below)", list(examples))
        text = st.text_input("Q =", examples[ex][0], key=f"pd_{ex}",
                             help="Name variables freely: P (own price), Ps / Pc (other goods' prices), "
                                  "Y or I (income), A (advertising).")
        try:
            _, syms = parse_vars(text, "the demand function")
        except InputError as e:
            st.error(str(e))
            st.stop()
        names = sorted(syms, key=lambda n: (n != "P", n))
        own = st.selectbox("Which variable is the good's own price?", names, index=0) if names else "P"
        st.markdown("**Values at the point you're evaluating**")
        cols = st.columns(max(len(names), 1))
        vals = {n: cols[i].number_input(n, value=float(examples[ex][1].get(n, 1.0)), key=f"v_{ex}_{n}")
                for i, n in enumerate(names)}
        try:
            r = point_elasticity(text, vals, own)
        except InputError as e:
            st.error(str(e))
            st.stop()
        mcols = st.columns(len(r.rows) + 1)
        mcols[0].metric("Quantity Q", num(r.q0))
        for c, row in zip(mcols[1:], r.rows):
            c.metric(f"E ({row.name}, {row.role})", num(row.elasticity))
        for row in r.rows:
            st.markdown(f"- **{row.name}**: {row.text}")
        with st.expander("Worked solution", expanded=True):
            steps_md(r.steps)

        st.markdown("**Predict a change.** Enter % changes; quantity moves by %ΔQ ≈ Σ E × %Δx.")
        pc = st.columns(len(r.rows))
        pct = {row.name: pc[i].number_input(f"% change in {row.name}", value=0.0, step=1.0, key=f"pct_{ex}_{row.name}")
               for i, row in enumerate(r.rows)}
        total, parts = r.predict(pct)
        if parts:
            st.markdown(rf"$\%\Delta Q \approx {' + '.join(parts)} = {num(total, 4)}\%$, "
                        f"so Q goes from {num(r.q0)} to about **{num(float(r.q0) * (1 + total / 100))}**. "
                        "(This is a linear approximation, most accurate for small changes.)")

        if own in syms:
            p0 = float(vals[own])
            others = {syms[n]: vals[n] for n in syms if n != own}
            q_of_p = numeric(r.expr.subs(others), syms[own])
            hi = max(2.2 * p0, float(r.revenue_max[0]) * 1.4 if r.revenue_max else 0)
            ps = np.linspace(max(hi / 400, 1e-6), hi, 400)
            qs = q_of_p(ps)
            ok = np.isfinite(qs) & (qs >= 0)
            fig = go.Figure(go.Scatter(x=qs[ok], y=ps[ok], name="Demand", line=dict(color=DARK, width=2.5)))
            fig.add_trace(go.Scatter(x=[float(r.q0)], y=[p0], mode="markers+text", text=[f"E = {num(r.rows[0].elasticity, 3)}"],
                                     textposition="top right", marker=dict(size=11, color=GREEN), showlegend=False))
            if r.revenue_max:
                fig.add_trace(go.Scatter(x=[float(r.revenue_max[1])], y=[float(r.revenue_max[0])], mode="markers+text",
                                         text=["unit elastic (max revenue)"], textposition="top right",
                                         marker=dict(size=10, color=BLUE), showlegend=False))
            st.plotly_chart(curve_layout(fig), width="stretch")
    else:
        text = st.text_input("P(Q) =", "120 - 0.5Q", key="inv")
        c1, c2 = st.columns(2)
        by = c1.radio("Evaluate at", ["a price", "a quantity"], horizontal=True)
        x = c2.number_input("Value", value=110.0 if by == "a price" else 20.0)
        try:
            r = inverse_point_elasticity(text, price=x if by == "a price" else None, quantity=x if by == "a quantity" else None)
        except InputError as e:
            st.error(str(e))
            st.stop()
        m = st.columns(3)
        m[0].metric("Quantity Q", num(r.q0))
        m[1].metric("Price P", num(r.rows[0].value))
        m[2].metric("Elasticity E", num(r.rows[0].elasticity))
        st.markdown(r.rows[0].text)
        steps_md(r.steps)

# ---------------------------------------------------------------- arc elasticity
with t_arc:
    st.markdown("Elasticity between two observed points, using the midpoint (arc) formula.")
    what = st.selectbox("What changed?", ["Own price (P)", "Income (Y)", "Price of another good (Ps)"])
    label, kind = {"Own price (P)": ("P", "own price"), "Income (Y)": ("Y", "income"),
                   "Price of another good (Ps)": ("Ps", "price of a related good")}[what]
    c = st.columns(4)
    x1 = c[0].number_input(f"{label}₁", value=10.0)
    q1 = c[1].number_input("Q₁", value=100.0)
    x2 = c[2].number_input(f"{label}₂", value=12.0)
    q2 = c[3].number_input("Q₂", value=80.0)
    try:
        a = arc_elasticity(x1, q1, x2, q2, label, kind)
    except InputError as e:
        st.error(str(e))
    else:
        st.metric("Arc elasticity", num(a.midpoint))
        st.markdown(a.text)
        steps_md(a.steps)

# ---------------------------------------------------------------- markup
with t_markup:
    st.markdown("A profit-maximizing firm sets **P = MC · E / (1 + E)**. Or, given a price, back out the elasticity it implies.")
    mode = st.radio("I know", ["the elasticity", "the current price"], horizontal=True)
    c = st.columns(2)
    mc = c[0].number_input("Marginal cost MC", value=10.0, min_value=0.0)
    try:
        if mode == "the elasticity":
            e = c[1].number_input("Price elasticity E (negative)", value=-2.0, max_value=0.0, step=0.1)
            r = markup(elasticity=e, mc=mc)
            st.metric("Profit-maximizing price", num(r.price))
        else:
            p = c[1].number_input("Current price P", value=20.0, min_value=0.0)
            r = markup(mc=mc, price=p)
            st.metric("Implied elasticity", num(r.elasticity))
    except InputError as e:
        st.error(str(e))
    else:
        steps_md(r.steps)

# ---------------------------------------------------------------- market
with t_market:
    st.markdown("Demand and supply as functions of price P only (plug in numbers for income and other prices).")
    c = st.columns(2)
    qd_text = c[0].text_input("Demand  Qd(P) =", "100 - 2P")
    qs_text = c[1].text_input("Supply  Qs(P) =", "3P - 50")
    c = st.columns(2)
    tax = c[0].number_input("Per-unit tax on sellers (negative = subsidy)", value=0.0, step=1.0)
    use_ctrl = c[1].checkbox("Add a price ceiling / floor")
    ctrl = c[1].number_input("Controlled price", value=25.0) if use_ctrl else None
    try:
        m = market(qd_text, qs_text, tax=tax, control=ctrl)
    except InputError as e:
        st.error(str(e))
        st.stop()
    k = st.columns(4)
    k[0].metric("Equilibrium price P*", num(m.p_eq))
    k[1].metric("Equilibrium quantity Q*", num(m.q_eq))
    k[2].metric("Consumer surplus", num(m.cs) if m.cs is not None else "∞")
    k[3].metric("Producer surplus", num(m.ps))
    if m.tax:
        t = m.tax
        k = st.columns(4)
        k[0].metric("Buyers pay", num(t["pb"]), delta=round(float(t["pb"] - m.p_eq), 4), delta_color="inverse")
        k[1].metric("Sellers keep", num(t["ps"]), delta=round(float(t["ps"] - m.p_eq), 4))
        k[2].metric("Tax revenue" if tax > 0 else "Subsidy cost", num(abs(t["revenue"])))
        k[3].metric("Deadweight loss", num(t["dwl"]) if t["dwl"] is not None else "—")
    steps_md(m.steps)

    top = float(m.choke) if m.choke is not None else 2.5 * float(m.p_eq)
    ps = np.linspace(0, top * 1.05, 400)
    qd_n, qs_n = numeric(m.qd, P)(ps), numeric(m.qs, P)(ps)
    okd, oks = np.isfinite(qd_n) & (qd_n >= 0), np.isfinite(qs_n) & (qs_n >= 0)
    fig = go.Figure([go.Scatter(x=qd_n[okd], y=ps[okd], name="Demand", line=dict(color=DARK, width=2.5)),
                     go.Scatter(x=qs_n[oks], y=ps[oks], name="Supply", line=dict(color=BLUE, width=2.5))])
    # surplus areas
    band = ps[(ps >= float(m.p_eq)) & okd]
    if m.cs is not None and len(band):
        qb = numeric(m.qd, P)(band)
        fig.add_trace(go.Scatter(x=np.r_[0, qb, 0], y=np.r_[band[0], band, band[-1]], fill="toself", mode="none",
                                 fillcolor="rgba(47,181,116,.18)", name="Consumer surplus"))
    band = ps[(ps <= float(m.p_eq)) & oks]
    if len(band):
        qb = numeric(m.qs, P)(band)
        fig.add_trace(go.Scatter(x=np.r_[0, qb, 0], y=np.r_[band[0], band, band[-1]], fill="toself", mode="none",
                                 fillcolor="rgba(63,95,138,.16)", name="Producer surplus"))
    fig.add_trace(go.Scatter(x=[float(m.q_eq)], y=[float(m.p_eq)], mode="markers+text", text=["equilibrium"],
                             textposition="middle right", marker=dict(size=11, color=GREEN), showlegend=False))
    if m.tax:
        fig.add_trace(go.Scatter(x=numeric(m.qs.subs(P, P - m.tax["t"]), P)(ps), y=ps, name="Supply + tax",
                                 line=dict(color=BLUE, dash="dash")))
        fig.add_hline(y=float(m.tax["pb"]), line_dash="dot", line_color=GREY, annotation_text="buyers pay")
        fig.add_hline(y=float(m.tax["ps"]), line_dash="dot", line_color=GREY, annotation_text="sellers keep")
    if m.control:
        fig.add_hline(y=float(m.control["price"]), line_color="#c0392b", annotation_text=f"price {m.control['kind']}")
    st.plotly_chart(curve_layout(fig, 420), width="stretch")
