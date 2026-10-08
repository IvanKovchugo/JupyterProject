"""Headless UI test: every preset button loads and renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

PRESETS = ["Problem 2", "Problem 4a", "Problem 4b", "Problem 5a", "Problem 14"]


@pytest.mark.parametrize("name", PRESETS)
def test_preset_renders(name):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    next(b for b in at.sidebar.button if b.label == name).click().run()
    assert not at.exception and not at.error
    assert at.metric[0].label == "Profit-max Q*"


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
