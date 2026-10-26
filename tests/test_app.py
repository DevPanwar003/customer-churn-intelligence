from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("page", ["Overview", "Data quality & explorer", "SQL analysis", "Model evaluation", "Retention playbook"])
def test_dashboard_pages_render(page):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error


def test_empty_filter_is_handled():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No customers match" in message.value for message in app.warning)


def test_evaluation_ignores_exploration_filters():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.multiselect[0].set_value([]).run()
    app.sidebar.radio[0].set_value("Model evaluation").run()
    assert not app.exception
    assert app.metric[1].value == "450"
    for name in ["Random Forest", "Majority baseline"]:
        app.selectbox[0].set_value(name).run()
        assert not app.exception


def test_new_evaluation_sections_render():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Model evaluation").run()
    assert not app.exception
    headings = [heading.value for heading in app.subheader]
    assert "Precision–Recall comparison · test holdout" in headings
    assert "Threshold trade-offs · training-only validation" in headings
    assert any("No threshold is selected" in message.value for message in app.info)


@pytest.mark.parametrize("dimension", ["Contract type", "Tenure", "Monthly charges", "Customer segment"])
def test_sql_dimensions_render(dimension):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("SQL analysis").run()
    app.selectbox[0].set_value(dimension).run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 1
