# Economics of Entrepreneurship

## Marginal Analysis Workbench

**Use it online:** https://ivankovchugo.github.io/JupyterProject/ (JavaScript version, instant, works on phones).
The Python/Streamlit version in `Economics_of_Entrapreneurship/app.py` gives exact symbolic answers (e.g. `20 − 2√30`).

A calculator for marginal analysis problems. Enter a demand curve (or revenue function, or fixed market price) and a cost function. The workbench:

- takes the derivatives symbolically (MR, MC, π′, π″)
- solves MR = MC and checks the second-order condition
- finds revenue-maximizing output and break-even points, with exact answers where possible (e.g. `20 − 2√30`)
- explains corner solutions (fixed price, linear cost) and gives the break-even formula Q = F / (P − c)
- compares who sets output in franchise revenue-sharing problems (franchisor, franchisee, profit sharing)
- optimizes any single-variable function (critical points, local/global max and min)
- writes out every step and charts R, C, π and MR vs MC

The class problems (2, 4a, 4b, 5a, 14) are one click away in the sidebar.

### Type functions like this

`120 - 0.5Q`, `420 + 60Q + Q^2`, `3 - Q/800`, `sqrt(Q)`, `ln(Q)`, `e^(0.1Q)`. Decimal commas (`0,5Q`) work too.

### Run it locally

```bash
pip install -r requirements.txt
streamlit run Economics_of_Entrapreneurship/app.py
```

### Tests

```bash
cd Economics_of_Entrapreneurship && pytest -q
```
