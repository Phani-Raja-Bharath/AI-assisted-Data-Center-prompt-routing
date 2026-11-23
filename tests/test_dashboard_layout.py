from streamlit.testing.v1 import AppTest
from unittest.mock import Mock
import ai_assisted_datacenter_allocation as app


def test_dashboard_views_reuse_models_and_defer_experiments(monkeypatch, tmp_path):
    weather = {"temperature": 20., "humidity": 50., "wind_speed": 5., "pressure": 1013.25,
               "solar_radiation": 200., "success": True, "source": "test fixture"}
    cached = Mock(side_effect=lambda *a, **k: dict(weather))
    monkeypatch.setattr(app, "cached_weather", cached)
    monkeypatch.setattr(app, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(app, "get_database_url", lambda *a, **k: None)
    train_calls, monte_carlo_calls = [], []
    original_train, original_monte_carlo = app.AIModelSuite.train_all, app.run_monte_carlo

    def train(suite, mode):
        train_calls.append(mode)
        return original_train(suite, mode, use_real_weather=False, n_samples=120)

    def monte_carlo(*args, **kwargs):
        monte_carlo_calls.append(True)
        return original_monte_carlo(*args, **kwargs)

    monkeypatch.setattr(app.AIModelSuite, "train_all", train)
    monkeypatch.setattr(app, "run_monte_carlo", monte_carlo)
    at = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    at.run(timeout=30)
    assert not at.exception
    assert not at.sidebar.selectbox and not at.sidebar.slider
    assert at.selectbox(key="city_selector").label == "User request origin"
    assert at.segmented_control(key="dashboard_view").options == ["Overview", "Routing", "Models", "Experiments"]
    assert not at.main.metric
    assert not any("SECTION" in item.value for item in at.markdown)
    at.selectbox(key="ui_radio_3521").set_value("Multiple Linear Regression (MLR) only").run()
    at.slider(key="ui_slider_3490").set_value(20)
    at.segmented_control(key="dashboard_view").set_value("Routing").run(timeout=30)
    assert not at.exception and train_calls == ["mlr"]
    # The receipt line names the prediction model; the training time is in the model details panel.
    assert any("Prediction model" in item.value and "New simulation completed" in item.value for item in at.caption)
    assert any("Saved model trained at" in item.value for item in at.markdown)
    assert any("Synthetic training data: no observed calendar date range applies." == item.value for item in at.caption)
    first_receipt = at.session_state["sim_receipt"]
    assert first_receipt["trigger"] == "Initial simulation"
    assert first_receipt["compute_duration_seconds"] >= 0
    assert first_receipt["strategies"] == list(at.session_state["sim_results"])
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert not at.exception and train_calls == ["mlr"] and not monte_carlo_calls
    manual_receipt = at.session_state["sim_receipt"]
    assert manual_receipt["run_id"] != first_receipt["run_id"]
    assert manual_receipt["trigger"] == "Run simulation button"
    assert manual_receipt["inputs_sha256"] == first_receipt["inputs_sha256"]
    previous = at.session_state["sim_key"]
    weather["temperature"] = 21.
    at.button(key="refresh_weather_btn").click().run(timeout=30)
    assert not at.exception and train_calls == ["mlr"] and not monte_carlo_calls
    assert cached.clear.call_count == 3
    assert at.session_state["sim_key"] != previous
    weather_receipt = at.session_state["sim_receipt"]
    assert weather_receipt["run_id"] != manual_receipt["run_id"]
    assert weather_receipt["trigger"] == "Manual weather refresh"
    assert all(value["temperature"] == 21. for value in weather_receipt["inputs"]["weather_data"].values())
    assert all(value["temperature"] == 20. for value in manual_receipt["inputs"]["weather_data"].values())
    key = at.session_state["sim_key"]
    for view in ["Routing", "Models", "Experiments", "Overview"]:
        at.segmented_control(key="dashboard_view").set_value(view).run(timeout=30)
        assert not at.exception
        assert at.session_state["sim_key"] == key
        assert at.session_state["sim_receipt"] == weather_receipt
        assert not at.session_state["sim_executed_this_render"]
        assert train_calls == ["mlr"] and not monte_carlo_calls
    assert not any(widget.key == "dashboard_strategy" for widget in at.selectbox)
    comparison = next(frame.value for frame in at.dataframe if "Unserved Requests" in frame.value.columns)
    assert comparison["Strategy"].tolist() == list(at.session_state["sim_results"])
    # Cooling is edited in the weather table; AppTest cannot edit data_editor cells, so set the
    # resulting selection directly and verify the simulation and retention behaviour.
    at.session_state["cooling_selections"] = {**at.session_state["cooling_selections"], "Phoenix, AZ": "liquid_cooling"}
    at.run(timeout=30)
    assert not at.exception and train_calls == ["mlr"]
    assert at.session_state["sim_receipt"]["run_id"] != weather_receipt["run_id"]
    assert at.session_state["sim_receipt"]["inputs"]["cooling_selections"]["Phoenix, AZ"] == "liquid_cooling"
    at.multiselect(key="ui_multiselect_3238").set_value(["Phoenix, AZ", "San Francisco, CA", "Dublin, Ireland"]).run(timeout=30)
    assert not at.exception and train_calls == ["mlr"]
    assert at.session_state["cooling_selections"]["Phoenix, AZ"] == "liquid_cooling"
    assert at.session_state["cooling_selections"]["Dublin, Ireland"] == "air_economizer"
    editor = next(frame.value for frame in at.dataframe if "Cooling technology" in frame.value.columns)
    assert editor.set_index("Datacenter")["Cooling technology"].to_dict() == at.session_state["cooling_selections"]
    assert "Stockholm, Sweden" not in at.session_state["cooling_selections"]
    assert set(at.session_state["sim_receipt"]["inputs"]["cooling_selections"]) == {"Phoenix, AZ", "San Francisco, CA", "Dublin, Ireland"}
    at.segmented_control(key="dashboard_view").set_value("Experiments").run(timeout=30)
    at.button(key="run_monte_carlo_btn").click().run(timeout=30)
    assert not at.exception and monte_carlo_calls == [True]
    at.segmented_control(key="dashboard_view").set_value("Overview").run()
    at.segmented_control(key="dashboard_view").set_value("Experiments").run(timeout=30)
    assert not at.exception and monte_carlo_calls == [True]
    at.selectbox(key="experiment_tool").set_value("Temperature sensitivity").run()
    at.button(key="run_temp_sensitivity").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["temp_sensitivity"] is not None
    at.selectbox(key="experiment_tool").set_value("What-if scenario").run()
    at.multiselect(key="scenario_datacenters").set_value(["Phoenix, AZ"]).run()
    at.button(key="run_scenario_btn").click().run(timeout=30)
    assert not at.exception and train_calls == ["mlr"]
    assert at.session_state["scenario_results"] is not None


def test_failed_simulation_does_not_issue_a_new_receipt(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import dashboard_ui

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    old_receipt = {"run_id": "previous-success", "inputs": {}}
    old_results = {"previous": True}
    state = State(cooling_selections={}, sim_receipt=old_receipt, sim_results=old_results)
    monkeypatch.setattr(dashboard_ui, "st", SimpleNamespace(session_state=state))
    backend = SimpleNamespace(run_simulation=Mock(side_effect=RuntimeError("calculation failed")),
                              get_carbon_data=lambda region: {})
    settings = SimpleNamespace(selected=[], num_requests=1000, energy_multiplier=1.,
                               latency_threshold=100, max_dc_capacity_mw=100, max_site_share=1.0,
                               user_location=(0, 0), run=True)
    with pytest.raises(RuntimeError, match="calculation failed"):
        dashboard_ui.simulation_results(backend, settings, {}, {}, None, {"revision": "test"})
    assert state.sim_receipt is old_receipt and state.sim_results is old_results
    assert not state.sim_executed_this_render
