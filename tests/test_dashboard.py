import ai_assisted_datacenter_allocation as app
from streamlit.testing.v1 import AppTest


def test_dashboard_reuses_models_and_retrains_only_on_request(monkeypatch, tmp_path):
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
        calls.append(mode)
        return original_train(suite, mode, use_real_weather=False, n_samples=120)

    monkeypatch.setattr(app.AIModelSuite, "train_all", train)
    # Drive the real main UI; replace only network, export, and training data sources.
    at = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    at.run(timeout=30)
    assert not at.exception
    at.selectbox(key="ui_radio_3521").set_value("Multiple Linear Regression (MLR) only").run()
    for checkbox in at.checkbox:
        if checkbox.label == "Enable Monte Carlo Analysis":
            checkbox.uncheck()
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert not at.exception
    assert calls == ["mlr"]
    revision = next(iter(at.session_state["model_bundles"].values()))["revision"]
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    at.run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    previous_key = at.session_state["sim_key"]
    capacity = next(s for s in at.slider if s.label == "Max share per site (%)")
    capacity.set_value(50).run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    assert at.session_state["sim_key"] == previous_key
    assert at.session_state["sim_inputs_changed"]
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert at.session_state["sim_key"] != previous_key
    previous_receipt = at.session_state["sim_receipt"]
    # Diagnostics and explicit retraining now live in the Models workspace.
    at.segmented_control(key="dashboard_view").set_value("Models").run(timeout=30)
    at.button(key="retrain_models_btn").click().run(timeout=30)
    assert not at.exception and calls == ["mlr", "mlr"]
    assert next(iter(at.session_state["model_bundles"].values()))["revision"] != revision
    assert at.session_state["sim_receipt"] == previous_receipt
    assert at.session_state["sim_inputs_changed"]
    # A new browser session must load the saved bundle before training.
    fresh = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    fresh.run(timeout=30)
    fresh.selectbox(key="ui_radio_3521").set_value("Multiple Linear Regression (MLR) only").run()
    next(c for c in fresh.checkbox if c.label == "Enable Monte Carlo Analysis").uncheck()
    fresh.button(key="run_sim_btn").click().run(timeout=30)
    assert not fresh.exception and calls == ["mlr", "mlr"]


def test_monte_carlo_passes_constraints_and_counts_unserved(monkeypatch):
    calls = []
    original = app.run_simulation

    def capture(*args, **kwargs):
        calls.append((args[-3:], kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "run_simulation", capture)
    dc = {"a": {"lat": 0, "lon": 0, "region": "default", "default_cooling": "free_air"}}
    weather = {"a": {"temperature": 20, "humidity": 50, "wind_speed": 5}}
    result = app.run_monte_carlo(dc, weather, (0, 0), 100, {"a": "free_air"}, 2, None,
                                 n_runs=2, latency_threshold=1, max_dc_capacity_mw=0.000001)
    assert all(args == (1, 0.000001, 1.0) for args, _ in calls)
    assert result["Multi-Objective"]["raw_data"]["served_requests"].sum() == 0
    assert result["Multi-Objective"]["raw_data"]["unserved_requests"].min() > 0
