"""Elasticity engine checks with textbook-style numbers worked by hand."""
import pytest

from elasticity_engine import arc_elasticity, inverse_point_elasticity, markup, market, parse_vars, point_elasticity
from marginal_engine import InputError


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def test_multivariable_point_elasticities():
    r = point_elasticity("400 - 2P + 0.5Ps + 0.1Y", {"P": 100, "Ps": 200, "Y": 1000})
    assert r.q0 == 400
    e = {row.name: row for row in r.rows}
    assert close(e["P"].elasticity, -0.5) and "Inelastic" in e["P"].text
    assert close(e["Ps"].elasticity, 0.25) and "Substitutes" in e["Ps"].text
    assert close(e["Y"].elasticity, 0.25) and "necessity" in e["Y"].text
    assert [row.name for row in r.rows][0] == "P"
    pct, _ = r.predict({"P": 10, "Y": 4})
    assert close(pct, -0.5 * 10 + 0.25 * 4)


def test_revenue_max_on_linear_demand():
    r = point_elasticity("100 - 2P", {"P": 10})
    assert r.revenue_max[0] == 25 and r.revenue_max[2] == 1250


def test_constant_elasticity_demand():
    r = point_elasticity("1000 P^(-1.5) Y^0.8", {"P": 4, "Y": 50})
    e = {row.name: row.elasticity for row in r.rows}
    assert close(e["P"], -1.5) and close(e["Y"], 0.8)


def test_inverse_demand_elasticity():
    r = inverse_point_elasticity("120 - 0.5Q", price=110)
    assert r.q0 == 20 and r.rows[0].elasticity == -11
    assert r.revenue_max[1] == 120


def test_arc_midpoint():
    a = arc_elasticity(10, 100, 12, 80)
    assert close(a.midpoint, (-20 / 90) / (2 / 11))
    assert a.revenue2 < a.revenue1 and "Elastic" in a.text


def test_markup_both_directions():
    assert close(markup(elasticity=-2, mc=10).price, 20)
    assert close(markup(mc=10, price=20).elasticity, -2)
    with pytest.raises(InputError):
        markup(elasticity=-0.5, mc=10)


def test_market_equilibrium_surplus_and_tax():
    m = market("100 - 2P", "3P - 50", tax=5)
    assert m.p_eq == 30 and m.q_eq == 40
    assert m.cs == 400 and close(m.ps, 800 / 3)
    t = m.tax
    assert t["pb"] == 33 and t["ps"] == 28 and t["q"] == 34 and t["revenue"] == 170
    assert t["dwl"] == 15 and close(t["consumer_share"], 0.6) and close(t["approx_share"], 0.6)


def test_price_ceiling_shortage():
    m = market("100 - 2P", "3P - 50", control=25)
    assert m.control["kind"] == "ceiling" and m.control["gap"] == 25


def test_market_rejects_extra_variables():
    with pytest.raises(InputError):
        market("100 - 2P + Y", "3P")


@pytest.mark.parametrize("text", ["__import__('os')", "lambda: 1", "P.__class__", "import os"])
def test_rejects_unsafe_input(text):
    with pytest.raises(InputError):
        parse_vars(text)
