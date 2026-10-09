"""Headless UI test: every preset button loads and renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

PRESETS = ["Problem 2", "Problem 4a", "Problem 4b", "Problem 5a", "Problem 14"]


@pytest.mark.parametrize("name", PRESETS)
def test_preset_renders(name):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    next(b for b in at.sidebar.button if b.label == name).click().run()
    assert not at.exception and not at.error
    assert at.metric[0].label in ("Profit-max Q*", "Joint optimum Q* (part c)")


def test_problem_14_franchise_table():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    next(b for b in at.sidebar.button if b.label == "Problem 14").click().run()
    assert [m.value for m in at.metric][:3] == ["880", "1.9", "968"]
    text = " ".join(md.value for md in at.markdown)
    assert "| Total profit (both parties) | 840 | 960 | 968 |" in text


def test_bad_input_shows_error():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.text_input(key="cost").input("420 + 60x").run()
    assert at.error and "Write quantity as Q" in at.error[0].value


ELASTICITY_PAGE = "pages/1_Elasticity_and_markets.py"


def test_elasticity_page_defaults_render():
    at = AppTest.from_file(ELASTICITY_PAGE, default_timeout=60).run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels["Quantity Q"] == "400"
    assert labels["Equilibrium price P*"] == "30" and labels["Consumer surplus"] == "400"
    assert labels["Profit-maximizing price"] == "20"


def test_elasticity_page_tax_and_inverse_mode():
    at = AppTest.from_file(ELASTICITY_PAGE, default_timeout=60).run()
    next(n for n in at.number_input if n.label.startswith("Per-unit tax")).set_value(5.0).run()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Buyers pay"] == "33" and labels["Deadweight loss"] == "15"
    at.radio[0].set_value("Inverse demand P(Q)").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Elasticity E"] == "−11"
