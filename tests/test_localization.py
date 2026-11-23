import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

import ai_assisted_datacenter_allocation as app
from localization import tr, localized_dataframe, localized_figure
from provenance import provenance_rows, reference_markdown, executive_results_markdown


def test_technical_spanish_preserves_units_identifiers_html_and_citations():
    assert tr("Latency Threshold (ms)", True) == "Umbral de latencia (ms)"
    assert tr("Energy per Request (Wh)", True) == "Energía por solicitud (Wh)"
    assert tr("PUE = 1.10; ΔT = 0.0012 × Q/A; MLR / ANN / R²", True) == "PUE = 1.10; ΔT = 0.0012 × Q/A; MLR / ANN / R²"
    html = '<div class="Source">Temperature (°C): <strong>21.8</strong></div>'
    assert tr(html, True) == '<div class="Source">Temperatura (°C): <strong>21.8</strong></div>'
    cited = '<div class="reference-box">Source: Data centers cooling: A critical review</div>'
    assert "Data centers cooling: A critical review" in tr(cited, True)
    assert tr("`Energy` https://example.com/Source", True) == "`Energy` https://example.com/Source"
    assert "<style>" in tr("<style>Source Serif 4</style>", True)
    assert tr("Energy", False) == "Energy"


def test_provenance_declares_assumptions_and_verified_context():
    english = str(provenance_rows(False))
    spanish = str(provenance_rows(True))
    assert "not measurements" in english and "Static constants" in english
    assert "supuestos" in spanish and "no son mediciones" in spanish
    assert "0.346" in english and "2025" in spanish
    for language in (False, True):
        refs = reference_markdown(language)
        assert "10.1016/j.ijrefrig.2024.02.007" in refs
        assert "Data centers cooling: A critical review" in refs
        assert "historical-forecast-api" in refs
    result = {"Random": {"totals": {"served_requests": 10, "offered_requests": 10,
                                    "energy_wh": 3, "carbon_g": 1, "peak_ΔT": .001}}}
    summary = executive_results_markdown(result, {"p_value": .42}, True)
    assert "0.42" in summary and "p < 0.001" not in summary


def test_localized_charts_and_tables_do_not_mutate_analysis(monkeypatch):
    monkeypatch.setattr("localization.is_spanish", lambda: True)
    original = pd.DataFrame({"Energy (Wh)": [0.346], "Strategy": ["Energy-Only"]})
    translated = localized_dataframe(original)
    assert translated.columns.tolist() == ["Energía (Wh)", "Estrategia"]
    assert translated.iloc[0, 0] == original.iloc[0, 0]
    assert original.iloc[0, 1] == "Energy-Only"
    figure = go.Figure(go.Bar(name="Energy-Only", x=["Energy"], y=[0.346]))
    figure.update_layout(title="Carbon Intensity", xaxis_title="Energy (Wh)")
    displayed = localized_figure(figure)
    assert displayed.layout.title.text == "Intensidad de carbono"
    assert displayed.layout.xaxis.title.text == "Energía (Wh)"
    assert displayed.data[0].name == "Solo energía"
    assert displayed.data[0].y == figure.data[0].y
    assert figure.data[0].name == "Energy-Only"


def test_missing_weather_values_are_not_reported_as_api_measurements(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(app.requests_cache, "CachedSession", Mock())
    monkeypatch.setattr(app, "retry", Mock())
    client = Mock()
    response = Mock()
    response.Current.return_value = None
    client.weather_api.return_value = [response]
    monkeypatch.setattr(app.openmeteo_requests, "Client", Mock(return_value=client))
    result = app.fetch_weather_data(-38.95, -68.067, "Neuquén", "moderate")
    assert result["success"] is False
    assert result["classification"] == "Assumed climate baseline"
    assert result["observed_at"] is None


def test_language_toggle_keeps_experiment_models_and_result_cache(monkeypatch, tmp_path):
    weather = {"temperature": 20., "humidity": 50., "wind_speed": 5., "pressure": 1013.25,
               "solar_radiation": 200., "success": True, "source": "test fixture"}
    monkeypatch.setattr(app, "cached_weather", lambda *a, **k: dict(weather))
    monkeypatch.setattr(app, "fetch_weather_data", lambda *a, **k: dict(weather))
    monkeypatch.setattr(app, "st_figure_downloads", lambda *a, **k: None)
    monkeypatch.setattr(app, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(app, "get_database_url", lambda *a, **k: None)
    original = app.AIModelSuite.train_all
    calls = []

    def train(suite, mode):
        calls.append(mode)
        return original(suite, mode, use_real_weather=False, n_samples=120)

    monkeypatch.setattr(app.AIModelSuite, "train_all", train)
    at = AppTest.from_string("import ai_assisted_datacenter_allocation as app\napp.main()")
    at.run(timeout=30)
    assert "not requests per second" in at.slider(key="ui_slider_3454").proto.help
    at.radio[0].set_value("Multiple Linear Regression (MLR) only").run()
    next(s for s in at.slider if s.label == "Total Requests").set_value(2300)
    next(c for c in at.checkbox if c.label == "Enable Monte Carlo Analysis").uncheck()
    at.button(key="run_sim_btn").click().run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    key = at.session_state["sim_key"]
    results = at.session_state["sim_results"]
    revision = next(iter(at.session_state["model_bundles"].values()))["revision"]
    receipt = at.session_state["sim_receipt"]
    at.toggle(key="spanish_ui").set_value(True).run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    assert at.session_state["sim_key"] == key
    assert at.session_state["sim_results"] == results
    assert at.session_state["sim_receipt"] == receipt
    assert not at.session_state["sim_executed_this_render"]
    assert next(iter(at.session_state["model_bundles"].values()))["revision"] == revision
    assert at.radio[0].value == "Multiple Linear Regression (MLR) only"
    assert next(s for s in at.slider if s.label == "Número total de solicitudes").value == 2300
    assert "no solicitudes por segundo" in at.slider(key="ui_slider_3454").proto.help
    assert at.button(key="run_sim_btn").label == "Ejecutar simulación"
    assert at.title[0].value == "AI-Assisted Datacenter Prompt Routing for ΔT-AR Mitigation"
    assert any("Modelos disponibles: sesión" in item.value for item in at.caption)
    provenance = next(frame.value for frame in at.dataframe if "Clasificación" in frame.value.columns)
    assert any("Supuestos" in str(value) for value in provenance["Clasificación"])
    # Original scientific data is not translated or mutated.
    assert set(results) == {"Random", "Energy-Only", "Heat-Only", "ΔT-AR", "Multi-Objective"}
    at.toggle(key="spanish_ui").set_value(False).run(timeout=30)
    assert not at.exception and calls == ["mlr"]
    assert at.button(key="run_sim_btn").label == "Run simulation"
    assert at.session_state["sim_key"] == key
