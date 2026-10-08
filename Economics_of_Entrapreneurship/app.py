"""Marginal Analysis Workbench: Streamlit front end for marginal_engine.

Run locally:  streamlit run Economics_of_Entrapreneurship/app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from marginal_engine import Q, InputError, analyze, ltx, num, numeric, optimize_any, tex

st.set_page_config(page_title="Marginal Analysis Workbench", page_icon="📈", layout="wide")

BLUE, ORANGE, GREEN, PURPLE = "#104226", "#ec6725", "#4fae68", "#6a7682"  # Sol-Millennium palette
MODES = {"demand": "Demand curve P(Q)", "revenue": "Revenue R(Q)", "price": "Fixed market price P"}
INPUT_LABEL = {"demand": "Inverse demand  P(Q) =", "revenue": "Total revenue  R(Q) =", "price": "Market price  P ="}

PRESETS = {
    "Problem 2": dict(mode="revenue", rev="170Q - 20Q^2", cost="100 + 38Q", share=0.0,
                      note="Problem 2 gives only revenue. The cost C = 100 + 38Q is the microchip example from the "
                           "textbook chapter; change it if your notes use a different one."),
    "Problem 4a": dict(mode="demand", rev="120 - 0.5Q", cost="420 + 60Q + Q^2", share=0.0),
    "Problem 4b": dict(mode="price", rev="120", cost="420 + 60Q + Q^2", share=0.0),
    "Problem 5a": dict(mode="price", rev="120", cost="420 + 60Q", share=0.0,
                       note="Fixed price and linear cost: profit rises forever, so set a capacity to see the corner "
                            "solution. The general break-even formula (5b) is the last step of the solution."),
    "Problem 14": dict(mode="demand", rev="3 - Q/800", cost="0.8Q", share=20.0,
                       note="Burger Queen: 20% of revenue goes to BQ and each Slopper costs the franchisee $0.80. "
                            "The Franchise tab answers parts a–d."),
}

ss = st.session_state
if "mode" not in ss:
    p = PRESETS["Problem 4a"]
    ss.update(mode=p["mode"], rev=p["rev"], cost=p["cost"], share=p["share"], cap=0.0, note="")


def load_preset(name):
    p = PRESETS[name]
    ss.update(mode=p["mode"], rev=p["rev"], cost=p["cost"], share=p["share"], cap=0.0, note=p.get("note", ""))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Load a problem")
    for name in PRESETS:
        st.button(name, on_click=load_preset, args=(name,), width="stretch")
    st.divider()
    st.markdown(
        "**How to type functions**\n\n"
        "- Quantity is `Q` (or `q`)\n"
        "- `0.5Q`, `Q^2`, `Q/800`, `sqrt(Q)`, `ln(Q)`, `e^(0.1Q)`\n"
        "- Decimal commas work: `0,5Q`\n\n"
        "Derivatives and roots are exact (SymPy) where possible, numeric otherwise."
    )

# ---------------------------------------------------------------- inputs
st.title("Marginal Analysis Workbench")
st.caption("Economics of Entrepreneurship · MR = MC, revenue maximization, break-even and franchise problems, "
           "with every step written out.")

left, right = st.columns([1, 2], gap="large")
with left:
    st.radio("What does the problem give you?", list(MODES), format_func=MODES.get, key="mode")
    st.text_input(INPUT_LABEL[ss.mode], key="rev")
    st.text_input("Total cost  C(Q) =", key="cost")
    c1, c2 = st.columns(2)
    c1.number_input("Revenue share to franchisor (%)", 0.0, 99.0, step=5.0, key="share",
                    help="For franchise problems: the franchisor takes this % of revenue.")
    c2.number_input("Capacity (max Q, 0 = auto)", 0.0, step=10.0, key="cap")
    if ss.note:
        st.info(ss.note)

try:
    r = analyze(ss.mode, ss.rev, ss.cost, ss.share, ss.cap or None)
except InputError as e:
    right.error(str(e))
    st.stop()

o = r.opt
inf = r.unbounded and not r.capacity_set

with right:
    m = st.columns(4)
    m[0].metric("Profit-max Q*", "∞" if inf else num(o.x),
                help="Where MR = MC" if not r.unbounded else "No interior maximum")
    m[1].metric("Price P*", num(r.price_at(o.x)))
    m[2].metric("Max profit π*", "∞" if inf else num(o.y))
    if r.mode == "price":
        m[3].metric("Break-even Q", ", ".join(num(x) for x in r.break_even) or "none")
    else:
        m[3].metric("Revenue-max Q", "∞" if r.rev_unbounded else num(r.rev.x), help="Where MR = 0")
    if r.unbounded:
        st.warning("MR is always above MC, so profit keeps rising with output. "
                   + ("The optimum is at capacity." if r.capacity_set else "Set a capacity to get a finite answer."))

    # ---- charts
    xs = np.linspace(0, float(r.q_max), 400)
    def line(expr, name, color, dash=None):
        return go.Scatter(x=xs, y=numeric(expr)(xs), name=name, line=dict(color=color, width=2.5, dash=dash))

    def point(x, y, text, color):
        return go.Scatter(x=[float(x)], y=[float(y)], mode="markers+text", text=[text], textposition="top center",
                          marker=dict(size=10, color=color, line=dict(width=2, color="white")), showlegend=False)

    def layout(fig):
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Q",
                          legend=dict(orientation="h", y=1.08), hovermode="x unified")
        return fig

    t1, t2 = st.tabs(["Revenue, cost, profit", "MR and MC"])
    with t1:
        fig = go.Figure([line(r.R, "Revenue R", BLUE), line(r.C, "Cost C", ORANGE), line(r.profit, "Profit π", GREEN)])
        if not inf:
            fig.add_trace(point(o.x, o.y, f"π* at Q={num(o.x, 2)}", GREEN))
            fig.add_vline(x=float(o.x), line_dash="dot", line_color="gray")
        if r.mode != "price" and not r.rev_unbounded:
            fig.add_trace(point(r.rev.x, r.rev.y, f"max R at Q={num(r.rev.x, 2)}", BLUE))
        for x in r.break_even:
            fig.add_trace(point(x, r.R.subs(Q, x), f"BE {num(x, 2)}", ORANGE))
        st.plotly_chart(layout(fig), width="stretch")
    with t2:
        traces = [line(r.MR, "MR", BLUE), line(r.MC, "MC", ORANGE)]
        if r.mode != "price":
            traces.append(line(r.P, "Demand P", BLUE, "dash"))
        if r.share > 0:
            traces.append(line((1 - r.share) * r.MR, f"Franchisee MR = {num(1 - r.share, 2)}·MR", PURPLE))
        fig = go.Figure(traces)
        if o.where == "interior":
            fig.add_trace(point(o.x, r.MC.subs(Q, o.x), f"MR = MC at Q={num(o.x, 2)}", ORANGE))
            fig.add_vline(x=float(o.x), line_dash="dot", line_color="gray")
        st.plotly_chart(layout(fig), width="stretch")

# ---------------------------------------------------------------- detail tabs
tabs = ["Worked solution"] + (["Franchise"] if r.share > 0 else []) + ["Any-function optimizer"]
tab = dict(zip(tabs, st.tabs(tabs)))

with tab["Worked solution"]:
    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(r.steps, 1)))

if "Franchise" in tab:
    with tab["Franchise"]:
        keep = 1 - r.share
        st.markdown(f"The franchisor takes **s = {num(r.share * 100, 2)}%** of revenue; the franchisee pays all of C(Q). "
                    "Each party prefers a different output.")
        rows = {
            "Quantity Q": lambda x: x,
            "Price P": r.price_at,
            "Revenue R": lambda x: r.R.subs(Q, x),
            "Franchisee costs C": lambda x: r.C.subs(Q, x),
            f"Franchisor receives ({num(r.share * 100, 2)}% of R)": lambda x: r.share * r.R.subs(Q, x),
            "Franchisee net profit": lambda x: keep * r.R.subs(Q, x) - r.C.subs(Q, x),
            "Total profit (both parties)": lambda x: r.profit.subs(Q, x),
        }
        head = "| | " + " | ".join(f"**{s.title}**<br>{s.rule}" for s in r.franchise) + " |"
        table = [head, "|---|" + "---:|" * len(r.franchise)]
        for label, g in rows.items():
            table.append(f"| {label} | " + " | ".join(num(g(s.opt.x)) if s.opt else "—" for s in r.franchise) + " |")
        st.markdown("\n".join(table), unsafe_allow_html=True)
        fr, fe, _ = r.franchise
        st.markdown(
            f"- **Franchisor sets it:** it wants maximum revenue, so it pushes output to MR = 0 and ignores the "
            f"franchisee's costs.\n"
            f"- **Franchisee sets it:** it keeps only {num(keep * 100, 2)}¢ of each extra revenue dollar, so it stops "
            f"where {num(keep, 2)}·MR = MC: lower output, higher price. Total profit "
            f"{num(r.profit.subs(Q, fe.opt.x))} vs {num(r.profit.subs(Q, fr.opt.x))}.\n"
            f"- **Profit sharing:** if the franchisor gets any share α of R − C, both sides maximize R − C, so both "
            f"want MR = MC. The split does not change Q or P, and total profit is the highest possible, {num(o.y)}.\n"
            "- **Why profit sharing is rare:** profit is easy to manipulate and costly to audit (the franchisee can "
            "inflate costs or pay itself a large salary), while revenue is simple to verify."
        )

with tab["Any-function optimizer"]:
    st.caption("For anything that isn't a revenue/cost setup: utility, average cost, production functions. "
               "Use any single-letter variable.")
    a, b, c = st.columns([3, 1, 1])
    ftext = a.text_input("f =", "x^3 - 6x^2 + 9x + 2", key="ff")
    lo = b.number_input("From", value=0.0, key="ffa")
    hi = c.number_input("To", value=5.0, key="ffb")
    try:
        fr_ = optimize_any(ftext, lo, hi)
    except InputError as e:
        st.error(str(e))
    else:
        v = fr_.var
        st.markdown(rf"$f'({v}) = {ltx(fr_.d1)}$  \n$f''({v}) = {ltx(fr_.d2)}$")
        if fr_.critical:
            st.markdown(f"| Critical point {v} | f({v}) | f''({v}) | Type |\n|---:|---:|---:|---|\n" + "\n".join(
                f"| ${tex(x)}$ | ${tex(y)}$ | {num(s2)} | {k} |" for x, y, s2, k in fr_.critical))
        else:
            st.markdown(f"No point with f' = 0 between {num(lo)} and {num(hi)}.")
        if fr_.gmax:
            st.markdown(f"On [{num(lo)}, {num(hi)}]: **global max** f = {num(fr_.gmax[1])} at {v} = {num(fr_.gmax[0])}; "
                        f"**global min** f = {num(fr_.gmin[1])} at {v} = {num(fr_.gmin[0])}.")
        xs2 = np.linspace(lo, hi, 400)
        fig = go.Figure([go.Scatter(x=xs2, y=numeric(fr_.f, v)(xs2), name="f", line=dict(color=PURPLE, width=2.5)),
                         go.Scatter(x=xs2, y=numeric(fr_.d1, v)(xs2), name="f′", line=dict(color=BLUE, dash="dash"))])
        for x, y, _, k in fr_.critical:
            fig.add_trace(go.Scatter(x=[float(x)], y=[float(y)], mode="markers+text", text=[k],
                                     textposition="top center", marker=dict(size=10, color=PURPLE), showlegend=False))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=str(v))
        st.plotly_chart(fig, width="stretch")
