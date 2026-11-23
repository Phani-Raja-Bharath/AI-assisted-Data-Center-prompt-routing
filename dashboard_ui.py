"""Focused dashboard views over the existing research/simulation backend."""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from localization import tr, is_spanish, localized_formatter, localized_dataframe, localized_figure
from provenance import provenance_rows, reference_markdown, recommendation_markdown, executive_results_markdown
from model_store import ModelStore, training_config, bundle_from_suite, restore_suite, get_or_train_bundle
from temperature_display import fahrenheit, temperature_text, dual_temperature_axes
from weather_strip import render_weather_strip

PALETTE = ["#176b62", "#c17b45", "#6f7b9b"]
VIEWS = ["Overview", "Routing", "Models", "Experiments"]


def display_plot(figure, key, height=360):
    """Keep paper-sized source figures intact; use compact dashboard typography."""
    if figure is None:
        return
    figure = go.Figure(localized_figure(figure))
    dual_temperature_axes(figure)
    figure.update_layout(height=height, title=None, paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=35, r=20, t=25, b=45),
                         font=dict(family="Aptos, Calibri, sans-serif", size=13, color="#263e43"),
                         legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.85)"))
    figure.update_xaxes(title_font=dict(size=12, family="Aptos"), tickfont=dict(size=11, family="Aptos"), gridcolor="#e8edf4")
    figure.update_yaxes(title_font=dict(size=12, family="Aptos"), tickfont=dict(size=11, family="Aptos"), gridcolor="#e8edf4")
    for annotation in figure.layout.annotations or []:
        annotation.font.size = 12
        annotation.font.family = "Aptos"
    for trace in figure.data:
        if hasattr(trace, "textfont"):
            trace.textfont.size = 11
            trace.textfont.family = "Aptos"
    st.plotly_chart(figure, key=key, width="stretch", config={
        "displaylogo": False, "responsive": True,
        "toImageButtonOptions": {"format": "png", "scale": 3, "width": 1600, "height": 900},
    })


def experiment_sidebar(a):
    with st.sidebar, st.container(key="experiment_sidebar"):
        st.header(tr("Experiment"))
        st.caption(tr("Configure once. Compare across views."))
        st.session_state.setdefault("user_lat", 28.5383)
        st.session_state.setdefault("user_lon", -81.3792)
        st.session_state.setdefault("user_label", "Orlando")
        st.session_state.setdefault("user_lat_input", st.session_state.user_lat)
        st.session_state.setdefault("user_lon_input", st.session_state.user_lon)

        def change_city():
            city = st.session_state.city_selector
            lat, lon = a.CITY_DATABASE[city]
            st.session_state.update(user_lat=lat, user_lon=lon, user_label=city,
                                    user_lat_input=lat, user_lon_input=lon)

        cities = sorted(a.CITY_DATABASE)
        st.selectbox(tr("User request origin"), cities, index=cities.index(st.session_state.user_label)
                     if st.session_state.user_label in cities else 0,
                     key="city_selector", on_change=change_city,
                     help=tr("Sets the request origin used to calculate distance and modeled latency. This is not a datacenter location."))
        with st.expander(tr("Coordinates")):
            lat = st.number_input(tr("Latitude"), key="user_lat_input", format="%.4f",
                                  help=tr("Request-origin latitude in decimal degrees: north is positive, south is negative."))
            lon = st.number_input(tr("Longitude"), key="user_lon_input", format="%.4f",
                                  help=tr("Request-origin longitude in decimal degrees: east is positive, west is negative."))
        st.session_state.user_lat, st.session_state.user_lon = lat, lon

        catalog = {**a.DEFAULT_DATACENTERS, **a.EXTENDED_DC_LOCATIONS}
        defaults = st.session_state.get("selected_dcs", list(a.DEFAULT_DATACENTERS))
        selected = st.multiselect(tr("Choose 1–3 datacenters"), list(catalog), default=defaults,
                                  max_selections=3, key="ui_multiselect_3238",
                                  help=tr("Choose the candidate sites among which requests can be allocated. Each site has its own weather, cooling selection, and regional carbon baseline."))
        st.session_state.selected_dcs = selected
        requests = st.slider(tr("Total Requests"), 100, 10000, 1000, 100, key="ui_slider_3454",
                             help=tr("Number of AI requests offered in this simulated batch, not requests per second. Constraints may leave some requests unserved; compare served counts alongside energy and emissions."))

        with st.expander(tr("Constraints and workload")):
            latency = st.slider(tr("Latency Threshold (ms)"), 50, 200, 100, 10, key="ui_slider_3463",
                                help=tr("Maximum modeled latency enforced by Multi-Objective routing. Other strategies are comparison baselines and may exceed it. Latency is calculated, not measured."))
            capacity = st.slider(tr("Max Power Capacity per Datacenter (MW)"), 10, 200, 100, 10, key="ui_slider_3471",
                                 help=tr("Scenario power-capacity limit per site. The allocator converts it into a request budget using a one-hour energy-equivalent assumption; this is not a measured facility capacity."))
            multiplier = st.slider(tr("Energy Multiplier"), .5, 2., 1., .1, key="ui_slider_3478",
                                   help=tr("Scales energy per request: 1.0 uses the model output, 2.0 doubles it, and 0.5 halves it. This scenario assumption also affects calculated emissions, heat, and capacity."))
        with st.expander(tr("Models and analysis settings")):
            choice = st.radio(tr("Select AI Models to Train"), [
                "Train All & Compare (Recommended)", "Multiple Linear Regression (MLR) only",
                "Artificial Neural Network (ANN) only",
                "Bayesian Optimization only" if a.BAYES_AVAILABLE else "Bayesian (Not Available)"],
                key="ui_radio_3521", format_func=localized_formatter(),
                help=tr("Selects the saved model configuration to reuse. Training occurs only if that configuration has no compatible saved bundle, or after explicit retraining. Train All compares MLR, ANN, and Bayesian-optimized ANN when available."))
            monte_carlo = st.checkbox(tr("Enable Monte Carlo Analysis"), value=True, key="ui_checkbox_3487",
                                     help=tr("Enables the Monte Carlo action in Experiments. Checking this box does not start analysis or retrain a model."))
            runs = st.slider(tr("Monte Carlo Runs"), 20, 500, 100, 20, key="ui_slider_3490",
                             help=tr("Number of sampled simulation trials with varied weather and demand. More trials take longer; intervals describe simulated variability, not field-measurement uncertainty."))
        run = st.button(tr("Run simulation"), type="primary", width="stretch", key="run_sim_btn",
                        help=tr("Recalculates routing for the current settings using compatible saved models. The Random baseline can change because its allocation is sampled."))
        st.caption(tr("Models are reused. Analyses run on demand."))
    return SimpleNamespace(catalog=catalog, selected=selected, user_location=(lat, lon),
                           num_requests=requests, latency_threshold=latency,
                           max_dc_capacity_mw=capacity, energy_multiplier=multiplier,
                           model_choice=choice, enable_monte_carlo=monte_carlo, n_runs=runs, run=run)


def selected_conditions(a, settings):
    datacenters, weather = {}, {}
    for name in settings.selected:
        info = dict(settings.catalog[name])
        current = a.cached_weather(info["lat"], info["lon"], name, info.get("climate", "moderate"))
        climate = a.classify_climate(current["temperature"], current["humidity"], current.get("wind_speed", 5))
        info.update(climate_detail=climate["climate_detail"], emoji=climate["emoji"])
        info["default_cooling"] = climate["recommended_cooling"]
        datacenters[name], weather[name] = info, current
    selections = st.session_state.get("cooling_selections", {})
    st.session_state.cooling_selections = {
        name: selections.get(name, info["default_cooling"]) for name, info in datacenters.items()}
    return datacenters, weather


@st.fragment(run_every="5m")
def weather_panel(a, settings, displayed_weather):
    """Poll current inputs; update the full dashboard only after a new fetch."""
    controls, info = st.columns([1, 1], vertical_alignment="center")
    with controls:
        refresh = st.button(tr("Refresh weather"), key="refresh_weather_btn",
                            help=tr("Fetch current conditions for the selected datacenters now. Reuses saved models; the provider's valid-at time may remain unchanged."))
    with info:
        with st.popover(tr("Weather information"), icon=":material/info:"):
            st.write(tr("Automatic weather check every 5 minutes while this page is open. Refresh weather bypasses the app cache. Provider values may stay unchanged between 15-minute model time steps; assumed fallbacks are labeled."))
    if refresh:
        for name in settings.selected:
            info = settings.catalog[name]
            a.cached_weather.clear(info["lat"], info["lon"], name, info.get("climate", "moderate"))
        st.session_state.weather_refresh_trigger = "Manual weather refresh"
        st.rerun()
    _, latest = selected_conditions(a, settings)
    if latest != displayed_weather:
        # Keep routing, weather cards, and exported inputs on the same snapshot.
        st.session_state.weather_refresh_trigger = "Automatic weather refresh"
        st.rerun()
    render_weather_strip(latest, PALETTE)


def ready_models(a, choice, retrain):
    mode = ("all" if "All" in choice else "mlr" if "MLR" in choice else
            "ann" if "ANN" in choice else "bayesian" if a.BAYES_AVAILABLE else "mlr")
    source = (inspect.getsource(a.AIModelSuite) + inspect.getsource(a.classify_climate)
              + inspect.getsource(a.collect_historical_training_data)
              + Path(a.energy_model.__file__).read_text(encoding="utf-8")
              + json.dumps({"default": a.DEFAULT_DATACENTERS, "extended": a.EXTENDED_DC_LOCATIONS}, sort_keys=True))
    config = training_config(mode, hashlib.sha256(source.encode()).hexdigest(), a.FEATURE_NAMES, a.COOLING_TYPES)
    config["bayesian_available"] = a.BAYES_AVAILABLE
    try:
        secrets = st.secrets.to_dict()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        secrets = {}
    store = ModelStore(a.ARTIFACT_DIR, a.get_database_url(secrets))
    st.session_state.setdefault("model_bundles", {})

    def train():
        suite = a.AIModelSuite()
        with st.spinner(tr("Training selected models (only needed once per configuration)...")):
            suite.train_all(mode)
        return bundle_from_suite(suite, config)

    bundle, origin, notices = get_or_train_bundle(config, store, st.session_state.model_bundles,
                                                  train, force=retrain)
    for notice in notices:
        st.warning(tr(notice))
    return restore_suite(a.AIModelSuite(), bundle), bundle, origin


def simulation_results(a, settings, datacenters, weather, models, bundle):
    inputs = {"selected_dc_names": settings.selected, "num_requests": settings.num_requests,
              "cooling_selections": dict(st.session_state.cooling_selections),
              "energy_multiplier": settings.energy_multiplier,
              "latency_threshold": settings.latency_threshold,
              "max_dc_capacity_mw": settings.max_dc_capacity_mw,
              "user_location": settings.user_location, "weather_data": weather,
              "datacenters": datacenters, "model_revision": bundle["revision"],
              "carbon_hour": datetime.now().hour,
              "carbon_baselines": {name: a.get_carbon_data(info.get("region", "default"))
                                   for name, info in datacenters.items()}}
    key = hashlib.md5(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
    weather_trigger = st.session_state.pop("weather_refresh_trigger", None)
    st.session_state.sim_executed_this_render = False
    if (settings.run or st.session_state.get("force_sim_rerun") or
            st.session_state.get("sim_key") != key or st.session_state.get("sim_results") is None):
        previous = st.session_state.get("sim_receipt", {}).get("inputs", {})
        trigger = ("Run simulation button" if settings.run else weather_trigger or
                   ("Initial simulation" if not previous else
                    "Prediction model changed" if previous.get("model_revision") != bundle["revision"] else
                    "Weather inputs updated" if previous.get("weather_data") != weather else
                    "Carbon profile hour changed" if previous.get("carbon_hour") != inputs["carbon_hour"] else
                    "Simulation settings changed"))
        snapshot = deepcopy(inputs)
        started_at = datetime.now(timezone.utc).isoformat()
        clock_start = perf_counter()
        calculated = a.run_simulation(
            datacenters, weather, settings.user_location, settings.num_requests,
            st.session_state.cooling_selections, settings.energy_multiplier, models,
            settings.latency_threshold, settings.max_dc_capacity_mw)
        elapsed = perf_counter() - clock_start
        completed_at = datetime.now(timezone.utc).isoformat()
        receipt = {"run_id": uuid4().hex, "started_at_utc": started_at,
                   "completed_at_utc": completed_at, "compute_duration_seconds": elapsed,
                   "trigger": trigger, "prediction_model": bundle.get("best_model_name"),
                   "model_revision": bundle["revision"], "strategies": list(calculated),
                   "inputs_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest(),
                   "results_sha256": hashlib.sha256(json.dumps(calculated, sort_keys=True, default=str).encode()).hexdigest(),
                   "inputs": snapshot,
                   "strategy_results": deepcopy({name: {"totals": data["totals"], "distribution": data["distribution"]}
                                                for name, data in calculated.items()})}
        st.session_state.sim_results = calculated
        st.session_state.sim_receipt = receipt
        st.session_state.sim_executed_this_render = True
        st.session_state.sim_key = key
        st.session_state.force_sim_rerun = False
        st.session_state.temp_sensitivity = None
        st.session_state.scenario_results = None
        st.session_state.pop("mc_cache", None)
    return st.session_state.sim_results


def execution_receipt():
    receipt = st.session_state.get("sim_receipt")
    if not receipt:
        return
    fresh = st.session_state.sim_executed_this_render
    completed = datetime.fromisoformat(receipt["completed_at_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
    with st.container(key="execution_receipt"):
        status, details = st.columns([3, 1], vertical_alignment="center")
        with status:
            label = tr("New simulation completed") if fresh else tr("Reusing completed simulation")
            st.markdown(f"**{label}** · `{receipt['run_id'][:12]}` · `{completed}` · "
                        f"{receipt['compute_duration_seconds']:.3f} s")
        with details:
            with st.popover(tr("Execution receipt"), icon=":material/fact_check:"):
                st.markdown(f"**{tr('Run ID')}:** `{receipt['run_id']}`")
                st.write(f"{tr('Trigger')}: {tr(receipt['trigger'])}")
                st.write(f"{tr('Completed at')}: {completed}")
                st.caption(tr("Duration measures routing calculations only; weather retrieval and model loading or training are excluded."))
                st.caption(tr("A new run ID is created only after all strategy calculations complete successfully. Navigation and language changes reuse the original receipt."))
                rows = []
                for name, current in receipt["inputs"]["weather_data"].items():
                    rows.append({"Datacenter": name, "Temperature (°C)": current["temperature"],
                                 "Temperature (°F)": fahrenheit(current["temperature"]),
                                 "Humidity (%)": current["humidity"], "Wind (m/s)": current["wind_speed"],
                                 "Pressure (hPa)": current.get("pressure"),
                                 "Solar radiation (W/m²)": current.get("solar_radiation"),
                                 "Weather valid at (UTC)": current.get("observed_at") or "Not supplied",
                                 "Last fetched": current.get("retrieved_at") or "Not supplied",
                                 "Weather source": current.get("source", "Not supplied")})
                st.markdown(f"**{tr('Weather snapshot used for this run')}**")
                st.dataframe(localized_dataframe(pd.DataFrame(rows)), hide_index=True, width="stretch")
                st.download_button(tr("Download execution receipt (JSON)"),
                                   json.dumps(receipt, indent=2, ensure_ascii=False, default=str),
                                   f"simulation_{receipt['run_id']}.json", "application/json", key="download_sim_receipt")


def metrics_table(results):
    return pd.DataFrame([{
        "Strategy": strategy, "Energy (Wh)": data["totals"]["energy_wh"],
        "Carbon (g)": data["totals"]["carbon_g"], "Latency (ms)": data["totals"]["avg_latency_ms"],
        "Peak ΔT-AR (°C)": data["totals"]["peak_ΔT"],
        "Peak ΔT-AR (°F)": fahrenheit(data["totals"]["peak_ΔT"], difference=True),
        "Heat CV": data["totals"]["heat_cv"],
        "Served Requests": data["totals"]["served_requests"],
        "Unserved Requests": data["totals"]["unserved_requests"]} for strategy, data in results.items()])


def conditions_table(a, datacenters, weather):
    rows = []
    for name, info in datacenters.items():
        data = a.get_carbon_data(info.get("region", "default"))
        current = weather[name]
        rows.append({"Datacenter": name, "Temperature (°C)": current["temperature"],
                     "Temperature (°F)": fahrenheit(current["temperature"]),
                     "Humidity (%)": current["humidity"], "Wind (m/s)": current["wind_speed"],
                     "Weather source": current.get("source", "Not supplied"),
                     "Weather valid at (UTC)": current.get("observed_at") or "Not supplied",
                     "Electricity grid region": ({"uk": "United Kingdom", "default": "Default regional baseline"}.get(data["scope"], data["scope"].replace("_", " ").title())
                        if data["scope"] == info.get("region", "default") else data["scope"]),
                     "Source": data["source"], "Year": data["year"],
                     "Baseline (kg CO2e/kWh)": data["base_kg_per_kwh"],
                     "Source URL": data["source_url"],
                     "Classification": "Published annual fallback" if data["fallback"] else
                        "Fetched annual average" if data["fetched"] else "Static scenario assumption"})
    return pd.DataFrame(rows)


def routing_map(datacenters, user_location, distribution):
    figure = go.Figure()
    served = sum(distribution.values())
    for index, (name, info) in enumerate(datacenters.items()):
        color = PALETTE[index % len(PALETTE)]
        count = distribution.get(name, 0)
        share = count / served if served else 0
        active = count > 0 or not distribution
        figure.add_trace(go.Scattergeo(lat=[user_location[0], info["lat"]],
                         lon=[user_location[1], info["lon"]], mode="lines",
                         line=dict(color=color if active else "#aab7ad", width=1.5 + share * 5,
                                   dash="solid" if active else "dot"), opacity=.8 if active else .35, showlegend=False,
                         hoverinfo="skip"))
        label = name.split(",")[0]
        if distribution:
            label += f"<br>{count:,} · {share:.1%}"
        figure.add_trace(go.Scattergeo(lat=[info["lat"]], lon=[info["lon"]],
                         mode="markers+text", text=[label],
                         textposition="top left" if info["lon"] < user_location[1] else "top right",
                         marker=dict(size=11 + share * 14, color=color if active else "#aab7ad",
                                     line=dict(color="#ffffff", width=2)), name=name,
                         hovertext=[f"{name}<br>{tr('Allocated requests')}: {count:,}<br>"
                                    f"{tr('Share of served requests')}: {share:.1%}" if distribution else name],
                         hoverinfo="text"))
    figure.add_trace(go.Scattergeo(lat=[user_location[0]], lon=[user_location[1]],
                     mode="markers+text", text=[tr("Request origin")], textposition="bottom right",
                     name=tr("Request origin"), hoverinfo="text", hovertext=[tr("Request origin")],
                     marker=dict(size=15, symbol="star", color="#193b40", line=dict(color="white", width=1.5))))
    figure.update_geos(projection_type="natural earth", fitbounds="locations", showland=True,
                       landcolor="#dce4d8", showocean=True, oceancolor="#edf0e8",
                       showcountries=True, countrycolor="#c2cebf", coastlinecolor="#b0c1b2",
                       lataxis=dict(showgrid=True, gridcolor="#dfe5da", dtick=30),
                       lonaxis=dict(showgrid=True, gridcolor="#dfe5da", dtick=30),
                       showframe=False, bgcolor="rgba(0,0,0,0)")
    figure.update_layout(showlegend=False)
    return figure


def overview(a, settings, datacenters, weather, results, model_origin=None):
    if results:
        st.caption(tr(f"Models ready: {model_origin}. Simulation runs reuse these models."))
        st.subheader(tr("All routing strategies"))
        definitions = {
            "Energy (Wh)": "Energy: saved-model prediction, aggregated by allocated requests.",
            "Carbon (g)": "Emissions: deterministic energy × modeled grid intensity calculation.",
            "Latency (ms)": "Latency: deterministic distance and queue calculation; not measured network latency.",
            "Peak ΔT-AR (°C)": "Temperature rise: uncalibrated thermal estimate; not measured urban warming.",
        }
        st.dataframe(localized_dataframe(metrics_table(results)), width="stretch", hide_index=True,
                     column_config={tr(label): st.column_config.NumberColumn(help=tr(note),
                         format="%.6f" if "ΔT" in label else "%.1f") for label, note in definitions.items()})
        st.caption(tr("Compare served and unserved requests before comparing energy, emissions, or thermal impact."))
        st.caption(tr("Calculated simulation results. Fixed inputs, saved model, and carbon hour give repeatable values for non-random strategies; Random routing is sampled."))
    else:
        with st.container(key="results_strip"):
            for column, label in zip(st.columns(4), ["Energy (Wh)", "Carbon (g)", "Latency (ms)", "Peak ΔT-AR (°C)"]):
                column.metric(tr(label), "—")
        st.caption(tr("Ready to compare. Run a simulation from the sidebar."))

    map_column, traffic_column = st.columns([2.1, 1], gap="medium")
    with map_column, st.container(key="routing_atlas"):
        st.subheader(tr("Request origin and datacenters"))
        display_plot(routing_map(datacenters, settings.user_location, {}), "overview_map", 380)
        st.caption(tr("Shared candidate locations for all strategies. Star: request origin. Connections are schematic; strategy allocations appear in the adjacent chart."))
    with traffic_column, st.container(key="allocation_panel"):
        st.subheader(tr("Traffic Distribution"))
        if results:
            names = list(datacenters)
            chart = go.Figure()
            for index, name in enumerate(names):
                counts = [data["distribution"].get(name, 0) for data in results.values()]
                chart.add_trace(go.Bar(x=counts, y=list(results), name=name.split(",")[0],
                               orientation="h", marker_color=PALETTE[index % len(PALETTE)],
                               text=[f"{count:,}" if count else "" for count in counts], textposition="inside",
                               hovertemplate="%{y}<br>%{fullData.name}: %{x:,}<extra></extra>"))
            chart.update_layout(barmode="stack", xaxis_title="Requests", yaxis=dict(autorange="reversed"),
                                legend=dict(orientation="h", y=-.3, x=0))
            display_plot(chart, "overview_traffic", 350)
            recommendation = a.generate_final_recommendation(results, settings.latency_threshold)
            st.markdown(f"**{tr('Recommended:')} {tr(recommendation['recommended_strategy'])}**")
            st.caption(tr("Selected by served demand, then modeled peak thermal impact."))
        else:
            st.info(tr("Run the experiment to see request allocation and recommendations."))
            for index, (name, current) in enumerate(weather.items()):
                st.markdown(f"**{name.split(',')[0]}** · {temperature_text(current['temperature'])}")
    with st.expander(tr("Datacenter conditions and sources")):
        st.dataframe(localized_dataframe(conditions_table(a, datacenters, weather)), width="stretch", hide_index=True)
        st.caption(tr("Reported predictions and comparisons are analysis outputs, not datacenter field measurements."))


def routing_view(a, settings, datacenters, weather, results):
    st.subheader(tr("Routing comparison"))
    st.caption(tr("Compare service coverage alongside energy, emissions, and thermal impact."))
    st.dataframe(localized_dataframe(metrics_table(results)), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader(tr("Multi-Metric Comparison"))
        display_plot(a.create_metrics_comparison_chart(results), "routing_metrics", 390)
    with right, st.container(border=True):
        st.subheader(tr("Heat Distribution by Strategy"))
        display_plot(a.create_heat_distribution_chart(results), "routing_heat", 390)
    with st.expander(tr("Cooling Technology Configuration"), expanded=True):
        for column, (name, info) in zip(st.columns(len(datacenters)), datacenters.items()):
            with column:
                cooling = st.session_state.cooling_selections[name]
                selected = st.selectbox(name, list(a.COOLING_SYSTEMS),
                    index=list(a.COOLING_SYSTEMS).index(cooling), key=f"cooling_select_{name}",
                    format_func=localized_formatter(lambda value: a.COOLING_SYSTEMS[value]["name"]),
                    help=tr("Chooses the site's cooling technology. PUE settings are fixed scenario assumptions. Changing cooling recalculates routing using saved models."))
                if selected != cooling:
                    st.session_state.cooling_selections[name] = selected
                    st.session_state.force_sim_rerun = True
                    st.rerun()
        st.caption(tr("Exact PUE settings are scenario assumptions; the cited review provides cooling context."))
    with st.expander(tr("Carbon Intensity Analysis")):
        display_plot(a.create_carbon_intensity_curves(datacenters), "routing_carbon", 330)
        st.dataframe(localized_dataframe(conditions_table(a, datacenters, weather)), width="stretch", hide_index=True)
    with st.expander(tr("Cooling Technology Analysis")):
        st.caption(tr("The cooling matrix is a qualitative scenario assumption, not a measured ranking extracted from the cited review."))
        display_plot(a.create_cooling_effectiveness_heatmap(), "routing_cooling", 330)
        display_plot(a.create_pue_comparison_chart(), "routing_pue", 300)


def models_view(a, suite, bundle):
    best_name, _ = suite.get_best_model()
    st.caption(tr(f"Trained: {bundle['trained_at']} | Samples: {suite.training_samples:,}"))
    st.caption(tr("Historical API weather; formula-generated energy targets") if suite.training_source == "real" else
               tr("Synthetic weather and formula-generated energy targets"))
    st.caption(tr("Validation scores compare model predictions with formula-generated targets, not measured hardware energy."))
    metrics = [{"Model": name, "Validation R²": value.get("val_r2"), "Test R²": value["r2"],
                "Test MAE": value["mae"], "Test RMSE": value["rmse"]}
               for name, value in suite.metrics.items() if isinstance(value, dict) and "r2" in value]
    st.dataframe(localized_dataframe(pd.DataFrame(metrics)), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader(tr("Model Performance Comparison"))
        display_plot(a.create_model_comparison_chart(suite.metrics), "models_performance", 360)
    with right, st.container(border=True):
        st.subheader(tr("Predicted vs formula-generated energy"))
        display_plot(a.create_prediction_scatter(suite.metrics, best_name), "models_predictions", 360)
    with st.expander(tr("Training diagnostics")):
        for name, function in [("learning", a.create_learning_curve_chart), ("bayes", a.create_bayesopt_convergence_chart)]:
            display_plot(function(suite.metrics), f"models_{name}", 330)
        if suite.metrics.get("feature_importance"):
            display_plot(a.create_feature_importance_chart(suite.metrics["feature_importance"], best_name),
                         "models_importance", 330)


def experiments_view(a, settings, datacenters, weather, suite, results):
    tool = st.selectbox(tr("Analysis"), ["Monte Carlo", "Temperature sensitivity", "What-if scenario"],
                        key="experiment_tool", format_func=localized_formatter(),
                        help=tr("Choose a sampled uncertainty analysis, a temperature sweep, or an alternative datacenter configuration. Each runs only when its action button is pressed."))
    key = st.session_state.sim_key
    if tool == "Monte Carlo":
        st.caption(tr("Monte Carlo analysis runs only when requested."))
        st.caption(tr("Monte Carlo samples weather and demand. Intervals describe simulated variability, not measurement uncertainty; new analyses may differ."))
        if st.button(tr("Run Monte Carlo analysis"), key="run_monte_carlo_btn", disabled=not settings.enable_monte_carlo):
            with st.spinner(tr("Running Monte Carlo simulations...")):
                data = a.run_monte_carlo(datacenters, weather, settings.user_location, settings.num_requests,
                    st.session_state.cooling_selections, settings.energy_multiplier, suite,
                    settings.n_runs, settings.latency_threshold, settings.max_dc_capacity_mw)
            st.session_state.mc_cache = {"key": (key, settings.n_runs), "results": data}
        cache = st.session_state.get("mc_cache", {})
        if cache.get("key") != (key, settings.n_runs):
            st.info(tr("Choose the run count in the sidebar, then start the analysis."))
            return
        data = cache["results"]
        left, right = st.columns(2)
        with left:
            display_plot(a.create_monte_carlo_mean_ci_chart(data), "mc_mean", 390)
        with right:
            display_plot(a.create_monte_carlo_boxplots(data), "mc_box", 390)
        stats = a.calculate_statistical_significance(data)
        st.markdown(executive_results_markdown(results, stats, is_spanish()))
        with st.expander(tr("Distribution and convergence")):
            display_plot(a.create_monte_carlo_hist_overlay(data, metric="peak_ΔT"), "mc_distribution", 330)
            display_plot(a.create_monte_carlo_running_mean_chart(data, metric="peak_ΔT"), "mc_convergence", 330)
    elif tool == "Temperature sensitivity":
        if st.button(tr("🌡️ Run Temperature Sensitivity Analysis"), key="run_temp_sensitivity"):
            temperatures = list(range(0, 46, 3))
            values = {strategy: [] for strategy in results}
            with st.spinner(tr("Running temperature sensitivity analysis...")):
                for temperature in temperatures:
                    modified = {name: {**current, "temperature": temperature} for name, current in weather.items()}
                    evaluated = a.run_simulation(datacenters, modified, settings.user_location,
                        settings.num_requests, st.session_state.cooling_selections, settings.energy_multiplier,
                        suite, settings.latency_threshold, settings.max_dc_capacity_mw)
                    for strategy, result in evaluated.items():
                        values[strategy].append(result["totals"]["peak_ΔT"])
            st.session_state.temp_sensitivity = {"x": temperatures, "y": values}
        if st.session_state.get("temp_sensitivity"):
            display_plot(a.create_sensitivity_temperature_chart(st.session_state.temp_sensitivity), "temperature_analysis", 430)
    else:
        names = st.multiselect(tr("Select datacenters for alternative scenario:"), list(settings.catalog),
                               max_selections=3, key="scenario_datacenters",
                               help=tr("Choose up to three alternative sites to compare with the current configuration using the same demand, constraints, and saved prediction model."))
        if st.button(tr("🔄 Run Scenario Comparison"), disabled=not names, key="run_scenario_btn"):
            alternatives, alternative_weather, cooling = {}, {}, {}
            for name in names:
                info = dict(settings.catalog[name])
                current = a.cached_weather(info["lat"], info["lon"], name, info.get("climate", "moderate"))
                climate = a.classify_climate(current["temperature"], current["humidity"])
                info["default_cooling"] = info.get("default_cooling") or climate["recommended_cooling"]
                alternatives[name], alternative_weather[name] = info, current
                cooling[name] = info["default_cooling"]
            evaluated = a.run_simulation(alternatives, alternative_weather, settings.user_location,
                settings.num_requests, cooling, settings.energy_multiplier, suite,
                settings.latency_threshold, settings.max_dc_capacity_mw)
            st.session_state.scenario_results = {"key": key, "names": names, "alt_results": evaluated}
        scenario = st.session_state.get("scenario_results")
        if scenario and scenario["key"] == key and scenario["names"] == names:
            display_plot(a.create_scenario_comparison_chart(results["Multi-Objective"],
                scenario["alt_results"]["Multi-Objective"], "Current Configuration", "Alternative Configuration"),
                "scenario_comparison", 420)


def exports(a, results, recommendation):
    with st.expander(tr("Export results")):
        data = pd.DataFrame([{"Strategy": name, **value["totals"]} for name, value in results.items()])
        data["peak_temperature_rise_f"] = data["peak_ΔT"] * 1.8
        st.download_button(tr("Download Results CSV"), data.to_csv(index=False),
                           "routing_results.csv", "text/csv", key="download_results_csv")
        st.download_button(tr("Download Summary Report"), executive_results_markdown(results, spanish=is_spanish()),
                           "routing_summary.txt", "text/plain", key="download_summary")
        if a.FPDF_AVAILABLE and st.button(tr("Prepare PDF report"), key="prepare_pdf"):
            st.session_state.dashboard_pdf = (st.session_state.sim_key, bytes(a.create_pdf_report(results, recommendation)))
        document = st.session_state.get("dashboard_pdf")
        if document and document[0] == st.session_state.sim_key:
            st.download_button(tr("Download PDF Report"), document[1], "routing_report.pdf",
                               "application/pdf", key="download_pdf")


def render_dashboard(namespace):
    a = SimpleNamespace(**namespace)
    st.html("""<style>
    .st-key-dashboard_header { border-top: 4px solid #176b62; padding-top: 1.1rem; }
    .st-key-dashboard_header h1 { letter-spacing: -.035em; line-height: 1.08; }
    .st-key-dashboard_header h1, .st-key-results_strip { font-variant-numeric: tabular-nums; }
    .st-key-results_strip { background: #193b40; color: #f5f3ed;
        padding: 1.35rem 1.5rem; border-radius: 6px; margin: .4rem 0 .2rem; }
    .st-key-results_strip p { color: #d0dfd7; }
    .st-key-routing_atlas { background: #edf0e8; border-radius: 6px;
        padding: 1.2rem 1.3rem .25rem; }
    .st-key-allocation_panel { border-top: 2px solid #c17b45; padding: 1.1rem .7rem 0; }
    .st-key-routing_atlas h3, .st-key-allocation_panel h3 { padding-top: 0; }
    .st-key-experiment_sidebar h2 { letter-spacing: -.025em; }
    .st-key-experiment_sidebar { border-top: 4px solid #193b40; padding-top: 1rem; }
    @media(max-width: 760px) {
        .st-key-results_strip { padding: 1rem; }
        .st-key-dashboard_header h1 { font-size: 1.85rem; }
    }
    </style>""")
    with st.container(key="dashboard_header"):
        title, tools = st.columns([3, 1])
        with title:
            st.title("AI-Assisted Datacenter Prompt Routing for ΔT-AR Mitigation")
            st.markdown("Demonstrating the usage of AI to mitigate localized heat concentration effect caused by datacenters")
            st.caption("Phani Raja Bharath Balijepalli | University of Central Florida")
        with tools:
            st.toggle("Español", key="spanish_ui", help=tr("Switch between English and technical Spanish."))
            with st.popover(tr("Sources and methodology")):
                st.dataframe(pd.DataFrame(provenance_rows(is_spanish())), hide_index=True, width="stretch")
                st.markdown(reference_markdown(is_spanish()))
    weather_slot = st.container(key="header_weather")
    view = st.segmented_control(tr("Workspace"), VIEWS, default="Overview", key="dashboard_view",
                                format_func=localized_formatter(), label_visibility="collapsed", width="stretch") or "Overview"
    settings = experiment_sidebar(a)
    if not settings.selected:
        st.info(tr("Choose 1–3 datacenters"))
        return
    datacenters, weather = selected_conditions(a, settings)
    with weather_slot:
        weather_panel(a, settings, weather)
    if settings.run or view != "Overview":
        st.session_state.simulation_run = True
    retrain = False
    if view == "Models":
        st.subheader(tr("Saved prediction models"))
        retrain = st.button(tr("Retrain selected models"), key="retrain_models_btn",
                            help=tr("Replace the saved models for this training configuration."))
        if retrain:
            st.session_state.simulation_run = True
    if not st.session_state.get("simulation_run"):
        if view == "Overview":
            overview(a, settings, datacenters, weather, None)
        else:
            st.info(tr("Run the experiment to populate this view."))
        return
    suite, bundle, origin = ready_models(a, settings.model_choice, retrain)
    trained = datetime.fromisoformat(bundle["trained_at"]).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"{tr('Saved model trained at')}: {trained} · {tr('Prediction model')}: {suite.best_model_name}")
    with st.expander(tr("Training data and model details"), expanded=view == "Models"):
        st.markdown(f"**{tr('Available models')}:** {', '.join(suite.models)} · "
                    f"**{tr('Training samples')}:** {suite.training_samples:,}")
        years = bundle.get("training_config", {}).get("years")
        if suite.training_source == "real" and years:
            st.markdown(f"**{tr('Requested historical weather window')}:** {years[0]}-01-01 — {years[-1]}-12-31")
            st.caption(tr("This is the configured API request window. Exact returned dates and per-location coverage were not recorded in this saved artifact."))
        else:
            st.caption(tr("Synthetic training data: no observed calendar date range applies."))
        st.caption(tr("Energy targets were generated by formulas with added training noise; historical weather does not make them measured hardware energy."))
    if view != "Overview":
        st.caption(tr(f"Models ready: {origin}. Simulation runs reuse these models."))
    results = simulation_results(a, settings, datacenters, weather, suite, bundle)
    execution_receipt()
    for name, value in results.items():
        if value["totals"]["unserved_requests"]:
            st.warning(tr(f"{name}: {value['totals']['unserved_requests']:,} of {settings.num_requests:,} "
                          "requests could not be served within the capacity/latency constraints. Compare service rates alongside energy and heat metrics."))
    if view == "Overview":
        overview(a, settings, datacenters, weather, results, origin)
    elif view == "Routing":
        routing_view(a, settings, datacenters, weather, results)
    elif view == "Models":
        models_view(a, suite, bundle)
    else:
        experiments_view(a, settings, datacenters, weather, suite, results)
    recommendation = a.generate_final_recommendation(results, settings.latency_threshold)
    exports(a, results, recommendation)
    st.caption(tr("Simulation outputs · Sources and assumptions available above"))
