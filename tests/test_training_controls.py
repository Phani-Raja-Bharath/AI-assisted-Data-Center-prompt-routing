"""Train-all button and retraining on a selected weather period: both warn and wait for Confirm."""
from datetime import date

import pandas as pd
from streamlit.testing.v1 import AppTest

import ai_assisted_datacenter_allocation as app
from model_store import DEFAULT_TRAINING_PERIOD, training_config
from energy_model import COOLING_TYPES, FEATURE_NAMES


def _app(monkeypatch, tmp_path):
    weather = {"temperature": 20., "humidity": 50., "wind_speed": 5., "pressure": 1013.25,
               "solar_radiation": 200., "success": True, "source": "test fixture"}
    monkeypatch.setattr(app, "cached_weather", lambda *a, **k: dict(weather))
    monkeypatch.setattr(app, "fetch_weather_data", lambda *a, **k: dict(weather))
    monkeypatch.setattr(app, "st_figure_downloads", lambda *a, **k: None)
    monkeypatch.setattr(app, "ARTIFACT_DIR", tmp_path)
    monkeypatch.delenv("MODEL_DATABASE_URL", raising=False)
    monkeypatch.setattr(app, "get_database_url", lambda *a, **k: None)
    original_train = app.AIModelSuite.train_all
    calls = []

    def train(suite, mode, **kwargs):
        calls.append((mode, tuple(suite.training_period)))
        return original_train(suite, mode, use_real_weather=False, n_samples=120)

    monkeypatch.setattr(app.AIModelSuite, "train_all", train)
    at = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    at.run(timeout=30)
    assert not at.exception
    at.selectbox(key="ui_radio_3521").set_value("Multiple Linear Regression (MLR) only").run()
    for checkbox in at.checkbox:
        if checkbox.label == "Enable Monte Carlo Analysis":
            checkbox.uncheck()
    return at, calls


def test_train_all_warns_and_waits_for_confirmation(monkeypatch, tmp_path):
    at, calls = _app(monkeypatch, tmp_path)
    at.button(key="train_all_btn").click().run(timeout=30)
    assert not at.exception and not calls
    assert any("15 minutes" in warning.value for warning in at.warning)
    at.button(key="train_all_cancel_btn").click().run(timeout=30)
    assert not calls and not at.session_state["confirm_train_all"]
    at.button(key="train_all_btn").click().run(timeout=30)
    at.button(key="train_all_confirm_btn").click().run(timeout=60)
    assert not at.exception
    assert [mode for mode, _ in calls] == ["all"]
    assert at.selectbox(key="ui_radio_3521").value == "Train All & Compare (Recommended)"


def test_retrain_on_selected_period_after_confirmation(monkeypatch, tmp_path):
    at, calls = _app(monkeypatch, tmp_path)
    at.segmented_control(key="dashboard_view").set_value("Models").run(timeout=60)
    calls.clear()
    period = (date(2023, 1, 1), date(2023, 6, 30))
    at.date_input(key="training_period_input").set_value(period).run(timeout=30)
    at.button(key="retrain_period_btn").click().run(timeout=30)
    assert not at.exception and not calls  # warning only
    assert any("2023-01-01" in warning.value and "2023-06-30" in warning.value for warning in at.warning)
    at.button(key="confirm_period_btn").click().run(timeout=60)
    assert not at.exception
    assert calls == [("mlr", ("2023-01-01", "2023-06-30"))]
    assert tuple(at.session_state["training_period"]) == ("2023-01-01", "2023-06-30")


def test_training_period_is_part_of_the_model_identity():
    default = training_config("mlr", "fp", FEATURE_NAMES, COOLING_TYPES)
    other = training_config("mlr", "fp", FEATURE_NAMES, COOLING_TYPES, period=("2023-01-01", "2023-06-30"))
    assert default["period"] == list(DEFAULT_TRAINING_PERIOD) and default != other


def test_historical_download_uses_the_selected_dates(monkeypatch):
    requested = []

    def fake(**kwargs):
        requested.append((kwargs.get("start_date"), kwargs.get("end_date")))
        return pd.DataFrame()

    monkeypatch.setattr(app, "collect_historical_training_data", fake)
    suite = app.AIModelSuite()
    suite.training_period = ("2022-03-01", "2022-08-31")
    suite.generate_training_data()  # empty download falls back to synthetic data
    assert requested == [("2022-03-01", "2022-08-31")]
