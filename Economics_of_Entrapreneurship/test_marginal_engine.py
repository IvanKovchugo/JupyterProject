"""Checks the engine against the class problem set (Marginal Analysis - Case Studies)."""
import pytest
import sympy as sp

from marginal_engine import Q, InputError, analyze, optimize_any, parse


def close(a, b):
    return abs(float(a) - float(b)) < 1e-6


def test_problem_2_revenue_max_exceeds_profit_max():
    r = analyze("revenue", "170Q - 20Q^2", "100 + 38Q")
    assert r.rev.x == sp.Rational(17, 4)
    assert close(r.opt.x, 3.3) and close(r.opt.y, 117.8)
    assert r.rev.x > r.opt.x


def test_problem_4a():
    r = analyze("demand", "120 - .5Q", "420 + 60Q + Q^2")
    assert r.opt.x == 20 and r.price_at(r.opt.x) == 110 and r.opt.y == 180


def test_problem_4b_price_taker():
    r = analyze("price", "120", "420 + 60Q + Q^2")
    assert r.opt.x == 30 and r.opt.y == 480


def test_problem_5a_break_even_and_no_interior_max():
    r = analyze("price", "120", "420 + 60Q")
    assert r.break_even == [7]
    assert r.unbounded
    assert r.linear_break_even["Q"] == 7
    capped = analyze("price", "120", "420 + 60Q", capacity=50)
    assert capped.opt.x == 50 and capped.opt.y == 60 * 50 - 420


def test_problem_14_franchise():
    r = analyze("demand", "3 - Q/800", "0.8Q", share_pct=20)
    franchisor, franchisee, joint = r.franchise
    assert franchisor.opt.x == 1200 and franchisee.opt.x == 800 and joint.opt.x == 880
    assert [r.profit.subs(Q, s.opt.x) for s in r.franchise] == [840, 960, 968]


def test_free_optimizer():
    fr = optimize_any("x^3 - 6x^2 + 9x + 2", 0, 5)
    assert [(c[0], c[3]) for c in fr.critical] == [(1, "local max"), (3, "local min")]
    assert fr.gmax[0] == 5


def test_non_polynomial_uses_numeric_roots():
    r = analyze("demand", "100/sqrt(Q)", "10Q")
    assert close(r.opt.x, 25)


@pytest.mark.parametrize("text", ["__import__('os')", "Q.__class__", "open('x')", "Q; 1", "lambda: 1"])
def test_rejects_unsafe_input(text):
    with pytest.raises(InputError):
        parse(text)


def test_decimal_comma_and_lowercase_q():
    expr, _ = parse("0,5q^2")
    assert expr == sp.Rational(1, 2) * sp.Symbol("Q", real=True) ** 2
