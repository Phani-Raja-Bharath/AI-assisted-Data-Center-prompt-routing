from unittest.mock import Mock

import pytest
import requests
from streamlit.testing.v1 import AppTest

import ai_assisted_datacenter_allocation as app
import carbon_model


@pytest.fixture
def carbon_feed(monkeypatch):
    carbon_model._argentina_carbon_data.cache_clear()
    response = Mock(text="Entity,Code,Year,Carbon intensity\n"
                         "Argentina,ARG,2024,350\nBrazil,BRA,2025,80\n"
                         "Argentina,ARG,2025,346\nArgentina,ARG,2023,400\n")
    fetch = Mock(return_value=response)
    monkeypatch.setattr(carbon_model.requests, "get", fetch)
    yield fetch
    carbon_model._argentina_carbon_data.cache_clear()


def test_argentina_carbon_fetch_converts_units_and_reuses_daily_cache(carbon_feed):
    data = carbon_model.get_carbon_data("argentina")
    assert data["base_kg_per_kwh"] == pytest.approx(0.346)
    assert data["year"] == 2025 and data["fetched"] and not data["fallback"]
    assert carbon_model.get_carbon_intensity("argentina", hour=18) == pytest.approx(0.346)
    assert carbon_model.get_carbon_data("argentina") == data
    assert carbon_feed.call_count == 1
    assert carbon_feed.call_args.kwargs["params"] == {"country": "~ARG"}


@pytest.mark.parametrize("failure", ["network", "invalid"])
def test_argentina_carbon_fallback_is_explicit(carbon_feed, failure):
    if failure == "network":
        carbon_feed.side_effect = requests.Timeout("offline")
    else:
        carbon_feed.return_value.text = "Entity,Code,Year,Carbon intensity\nArgentina,ARG,2025,nan\n"
    data = carbon_model.get_carbon_data("argentina")
    assert data["fallback"] and not data["fetched"]
    assert data["year"] == 2025 and data["base_kg_per_kwh"] == pytest.approx(0.346)
    assert "fallback" in data["source"]


def test_neuquen_location_weather_map_routing_and_carbon(monkeypatch, tmp_path, carbon_feed):
    weather_calls = []

    def weather(lat, lon, name, *args, **kwargs):
        weather_calls.append((lat, lon, name))
        return {"temperature": 21.8, "humidity": 36., "wind_speed": 5.25, "pressure": 978.,
                "solar_radiation": 906., "success": True, "source": "test fixture"}

    monkeypatch.setattr(app, "cached_weather", weather)
    monkeypatch.setattr(app, "fetch_weather_data", weather)
    monkeypatch.setattr(app, "st_figure_downloads", lambda *a, **k: None)
    monkeypatch.setattr(app, "ARTIFACT_DIR", tmp_path)
    monkeypatch.delenv("MODEL_DATABASE_URL", raising=False)
    monkeypatch.setattr(app, "get_database_url", lambda *a, **k: None)
    original_train = app.AIModelSuite.train_all
    monkeypatch.setattr(app.AIModelSuite, "train_all",
                        lambda suite, mode: original_train(suite, mode, use_real_weather=False, n_samples=120))
    at = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    at.run(timeout=30)
    at.selectbox(key="city_selector").select("Neuquén").run(timeout=30)
    assert at.session_state["user_lat"] == -38.95
    assert at.session_state["user_lon"] == -68.067
    at.multiselect[0].set_value(["Neuquén, Argentina", "Phoenix, AZ"]).run(timeout=30)
    assert not at.exception
    assert (-38.95, -68.067, "Neuquén, Argentina") in weather_calls
    at.selectbox(key="ui_radio_3521").set_value("Multiple Linear Regression (MLR) only").run()
    next(c for c in at.checkbox if c.label == "Enable Monte Carlo Analysis").uncheck()
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert not at.exception
    results = at.session_state["sim_results"]
    # Carbon is frozen once per run at each site's local solar hour and recorded in the receipt.
    frozen = at.session_state["sim_receipt"]["inputs"]["carbon"]["sites"]["Neuquén, Argentina"]
    lon = app.EXTENDED_DC_LOCATIONS["Neuquén, Argentina"]["lon"]
    assert frozen["intensity_kg_per_kwh"] == pytest.approx(
        carbon_model.get_carbon_intensity("argentina", frozen["local_solar_hour"]))
    for strategy in results.values():
        dc = strategy["datacenters"]["Neuquén, Argentina"]
        assert dc["distance_km"] == pytest.approx(0)
        assert dc["carbon_kg"] == pytest.approx(dc["energy_wh"] / 1000 * frozen["intensity_kg_per_kwh"])
    sources = next(frame.value for frame in at.dataframe if "Source URL" in frame.value.columns)
    neuquen = sources.loc[sources["Datacenter"] == "Neuquén, Argentina"].iloc[0]
    assert neuquen["Source"] == "OWID / Ember annual grid data"
    assert neuquen["Baseline (kg CO2e/kWh)"] == pytest.approx(0.346)
    locations = {"Neuquén, Argentina": app.EXTENDED_DC_LOCATIONS["Neuquén, Argentina"],
                 "Phoenix, AZ": app.DEFAULT_DATACENTERS["Phoenix, AZ"]}
    fig = app.create_geographic_map(locations, app.CITY_DATABASE["Neuquén"],
                                     {"Neuquén, Argentina": 500, "Phoenix, AZ": 500})
    assert any(trace.mode == "lines" and -38.95 in trace.lat for trace in fig.data)
    chart = app.create_carbon_intensity_curves(locations)
    curves = [trace for trace in chart.data if trace.mode == "lines"]
    assert [trace.name for trace in curves] == list(locations)
    assert len(curves[0].y) == 24
    # One "now" marker per site, at that site's own local hour (not the server clock).
    assert sum(1 for trace in chart.data if trace.mode == "markers+text") == len(locations)
