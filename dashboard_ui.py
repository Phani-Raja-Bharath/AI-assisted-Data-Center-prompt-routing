"""Focused dashboard views over the existing research/simulation backend."""
from datetime import date, datetime, timedelta, timezone
from copy import deepcopy
import hashlib
import inspect
import base64
from functools import lru_cache
import json
import math
import random
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
from model_store import (ModelStore, training_config, bundle_from_suite, restore_suite, get_or_train_bundle,
                         DEFAULT_TRAINING_PERIOD)
from temperature_display import fahrenheit, temperature_text, dual_temperature_axes
from weather_strip import render_weather_table, weather_sources
from analysis_guidance import analysis_context, guidance, GUIDANCE

VIEWS = ["Overview", "Routing", "Models", "Experiments"]

# Colors Plotly needs directly. These read well in both light and dark mode: the map always
# uses a light basemap, and chart text/grid colors are left to Streamlit's own Plotly theme,
# which follows the active appearance in the browser. Page and panel colors live in CSS
# (light-dark()) in render_dashboard, so they also switch instantly with the theme.
UI_COLORS = {"series": ["#0f766e", "#d97706", "#4f46e5"], "text": "#1f3a40", "muted": "#9aa8ab",
             "marker_edge": "#ffffff", "origin": "#111827", "map_style": "carto-voyager-nolabels",
             "legend_bg": "rgba(255,255,255,0.9)", "best": "rgba(15, 118, 110, 0.18)"}


def ui_colors():
    return UI_COLORS


@lru_cache(maxsize=1)
def network_backdrop():
    """Faint datacenter network as an SVG data URI: links and nodes warm from teal to red
    across the page, with soft heat glows at the busiest nodes. Deterministic (fixed seed)."""
    rng = random.Random(3238)
    width, height = 1600, 1000
    nodes = [(rng.uniform(20, width - 20), rng.uniform(20, height - 20)) for _ in range(64)]
    links = set()
    for i, (x, y) in enumerate(nodes):
        nearest = sorted(range(len(nodes)), key=lambda j: (nodes[j][0] - x) ** 2 + (nodes[j][1] - y) ** 2)[1:4]
        links.update(tuple(sorted((i, j))) for j in nearest)
    degree = [sum(i in link for link in links) for i in range(len(nodes))]
    hubs = sorted(range(len(nodes)), key=lambda i: (-degree[i], nodes[i][0]))[:7]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid slice">',
        '<defs><linearGradient id="h" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1600" y2="1000">'
        '<stop offset="0" stop-color="#0f9488"/><stop offset=".55" stop-color="#e08a1e"/>'
        '<stop offset="1" stop-color="#d33a2c"/></linearGradient>'
        '<radialGradient id="g"><stop offset="0" stop-color="#e8702a" stop-opacity=".22"/>'
        '<stop offset="1" stop-color="#e8702a" stop-opacity="0"/></radialGradient></defs>',
    ]
    parts += [f'<circle cx="{nodes[i][0]:.0f}" cy="{nodes[i][1]:.0f}" r="{60 + 8 * degree[i]}" fill="url(#g)"/>'
              for i in hubs if nodes[i][0] > width * .35]
    parts.append('<g stroke="url(#h)" stroke-width="1" stroke-opacity=".26" fill="none">')
    parts += [f'<line x1="{nodes[a][0]:.0f}" y1="{nodes[a][1]:.0f}" x2="{nodes[b][0]:.0f}" y2="{nodes[b][1]:.0f}"/>'
              for a, b in sorted(links)]
    parts.append('</g><g fill="url(#h)" fill-opacity=".45">')
    parts += [f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{2 + .6 * degree[i]:.1f}"/>' for i, (x, y) in enumerate(nodes)]
    parts.append('</g></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode("".join(parts).encode()).decode()


AXIS_COLOR_KEYS = ("gridcolor", "linecolor", "zerolinecolor", "tickcolor")


def _aptos_fonts(node, tile_map=False):
    """Normalize nested Plotly fonts, including templates and hover labels.

    Tile-map text uses server-provided glyphs instead of browser web fonts.
    """
    if isinstance(node, list):
        for item in node:
            _aptos_fonts(item, tile_map)
    elif isinstance(node, dict):
        tile_map = tile_map or node.get("type") in ("scattermap", "scattermapbox")
        for key, value in node.items():
            if tile_map and key == "textfont":
                continue
            if key.endswith("font") and isinstance(value, dict):
                value["family"] = "Aptos, sans-serif"
            _aptos_fonts(value, tile_map)


def _theme_neutral(node, in_trace=False, trace_type=None):
    """Strip hard-coded paper styling (black Times text, white legend boxes, fixed grid colors)
    so Streamlit's Plotly theme can color the chart for light or dark mode.

    Heatmap cell labels keep their color: they sit on bright cells and need dark text in
    both modes. Fonts switch to the dashboard face."""
    if isinstance(node, list):
        for item in node:
            _theme_neutral(item, in_trace, trace_type)
        return
    if not isinstance(node, dict):
        return
    for key, value in list(node.items()):
        if isinstance(value, dict):
            if key.endswith("font"):
                if not (in_trace and trace_type == "heatmap" and key == "textfont"):
                    value.pop("color", None)
                if "family" in value:
                    value["family"] = "Aptos, sans-serif"
            elif key == "legend":
                value.pop("bgcolor", None)
                value.pop("bordercolor", None)
            elif key == "colorbar":  # only colorbars the figure defines; never create new ones
                value.setdefault("tickfont", {})["size"] = 11
                title = value.get("title")
                if isinstance(title, dict):
                    title.setdefault("font", {})["size"] = 12
            elif key.startswith(("xaxis", "yaxis")):
                for axis_key in AXIS_COLOR_KEYS:
                    value.pop(axis_key, None)
        _theme_neutral(value, in_trace, trace_type)


def display_plot(figure, key, height=360):
    """Show a figure in the dashboard: compact typography, and colors from the active theme.
    The source figures keep their sizing/colors for downloads; only this copy is restyled."""
    if figure is None:
        return
    figure = go.Figure(localized_figure(figure))
    dual_temperature_axes(figure)
    is_map = any(trace.type in ("scattergeo", "scattermap") for trace in figure.data)
    if not is_map:  # maps sit on a light basemap and keep their own dark label colors
        spec = figure.to_plotly_json()
        _theme_neutral(spec.get("layout", {}))
        for trace in spec.get("data", []):
            _theme_neutral(trace, in_trace=True, trace_type=trace.get("type"))
        layout = spec.get("layout", {})
        layout.pop("template", None)
        # Paper figures fix a pixel width/height; the dashboard sizes charts to their panel.
        layout.pop("width", None)
        layout.pop("height", None)
        figure = go.Figure(spec)
    # Text, grid and legend colors are deliberately unset: Streamlit's Plotly theme fills them in
    # for whichever appearance (light/dark) is active in the browser.
    figure.update_layout(height=height, title=None, paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=0, r=0, t=0, b=0) if is_map else dict(l=35, r=20, t=25, b=45),
                         font=dict(family="Aptos, sans-serif", size=13),
                         legend=dict(font=dict(size=11)))
    # automargin grows the margins to fit tick labels and axis titles, so they never overlap.
    figure.update_xaxes(title_font=dict(size=12, family="Aptos"), tickfont=dict(size=11, family="Aptos"),
                        automargin=True, title_standoff=8)
    figure.update_yaxes(title_font=dict(size=12, family="Aptos"), tickfont=dict(size=11, family="Aptos"),
                        automargin=True, title_standoff=8)
    for annotation in figure.layout.annotations or []:
        annotation.font.size = 12
        annotation.font.family = "Aptos"
    # A legend placed inside the plot area covers data and tick labels at dashboard size;
    # move it to a single row just above the plot.
    legend = figure.layout.legend
    if not is_map and legend.x is not None and legend.y is not None and 0 <= legend.x <= 1 and 0 <= legend.y <= 1:
        figure.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"))
    for trace in figure.data:
        # Tile-map labels render with the map's own glyph fonts; Aptos would make them disappear.
        if hasattr(trace, "textfont") and trace.type != "scattermap":
            trace.textfont.size = 11
            trace.textfont.family = "Aptos"
        # Unnamed bars would hover as "trace 1"; show the category and value instead.
        if trace.type == "bar" and not trace.name and not trace.hovertemplate:
            trace.hovertemplate = ("%{y}: %{x:.4g}" if trace.orientation == "h" else "%{x}: %{y:.4g}") + "<extra></extra>"
    spec = figure.to_plotly_json()
    _aptos_fonts(spec)
    figure = go.Figure(spec)
    figure.update_layout(hoverlabel_font_family="Aptos, sans-serif")
    st.plotly_chart(figure, key=key, width="stretch", config={
        "displaylogo": False, "responsive": True,
        "toImageButtonOptions": {"format": "png", "scale": 3, "width": 1600, "height": 900},
    })


TRAIN_ALL_CHOICE = "Train All & Compare (Recommended)"


def _confirm_train_all():
    # Runs in a callback, so it may set the model selector before that widget is drawn again.
    st.session_state.update(confirm_train_all=False, train_all_now=True, ui_radio_3521=TRAIN_ALL_CHOICE)


def train_all_confirmation():
    """Warning shown before training all models; training starts only on Confirm."""
    with st.container(key="train_all_confirm"):
        st.warning(tr("Retrains all three models. Takes about 15 minutes; the app is unavailable meanwhile."), icon=":material/warning:")
        with st.container(horizontal=True, gap="small"):
            st.button(tr("Confirm: train all models"), type="primary", key="train_all_confirm_btn",
                      icon=":material/model_training:", on_click=_confirm_train_all)
            st.button(tr("Cancel"), key="train_all_cancel_btn",
                      on_click=lambda: st.session_state.update(confirm_train_all=False))


EARLIEST_TRAINING_DATE = date(2021, 1, 1)  # start of the Open-Meteo historical forecast archive used here
MIN_TRAINING_DAYS = 30


def _training_minutes(start, end, choice):
    """Rough training time, scaled from the measured 4-year, all-models run (~15 min)."""
    years = max((end - start).days + 1, 1) / 365.25
    per_year = {"all": 3.7, "ann": 1.2, "bayesian": 2.5, "mlr": 0.3}
    mode = ("all" if "All" in choice else "mlr" if "MLR" in choice else "ann" if "ANN" in choice else "bayesian")
    return max(1, round(per_year[mode] * years))


def period_retrain_panel(settings):
    """Pick a historical weather period and retrain on it; training starts only after confirming."""
    start_default, end_default = (date.fromisoformat(value) for value in active_training_period())
    latest = date.today() - timedelta(days=2)  # archive lags by about a day
    with st.container(key="period_retrain", border=True):
        st.markdown(f"**{tr('Retrain with a new data period')}**")
        st.caption(tr("Active training period: {start} to {end}. Simulations use models trained on this period.").format(
            start=start_default.isoformat(), end=end_default.isoformat()))
        chosen = st.date_input(tr("Training weather period"), value=(start_default, min(end_default, latest)),
                               min_value=EARLIEST_TRAINING_DATE, max_value=latest, key="training_period_input",
                               help=tr("Weather used for training. Longer periods take longer."))
        valid = isinstance(chosen, (tuple, list)) and len(chosen) == 2
        if valid and (chosen[1] - chosen[0]).days + 1 < MIN_TRAINING_DAYS:
            st.caption(tr("Choose at least {days} days so the models see enough weather variation.").format(days=MIN_TRAINING_DAYS))
            valid = False
        with st.container(horizontal=True, gap="small"):
            if st.button(tr("Retrain with this period"), key="retrain_period_btn", icon=":material/date_range:",
                         disabled=not valid):
                st.session_state.confirm_period = (chosen[0].isoformat(), chosen[1].isoformat())
            if active_training_period() != tuple(DEFAULT_TRAINING_PERIOD):
                st.button(tr("Use default period (2021 to 2024)"), key="default_period_btn",
                          on_click=lambda: st.session_state.update(training_period=DEFAULT_TRAINING_PERIOD))
        pending = st.session_state.get("confirm_period")
        if pending:
            start, end = (date.fromisoformat(value) for value in pending)
            st.warning(tr("Retrains {model} on {start} to {end}. Takes about {minutes} min; the app is unavailable meanwhile.").format(
                model=tr(settings.model_choice), start=pending[0], end=pending[1],
                minutes=_training_minutes(start, end, settings.model_choice)), icon=":material/warning:")
            with st.container(horizontal=True, gap="small"):
                st.button(tr("Confirm: retrain on this period"), type="primary", key="confirm_period_btn",
                          on_click=lambda: st.session_state.update(training_period=pending, confirm_period=None,
                                                                   retrain_period_now=True))
                st.button(tr("Cancel"), key="cancel_period_btn",
                          on_click=lambda: st.session_state.update(confirm_period=None))


def experiment_controls(a):
    with st.container(key="experiment_panel"):
        st.markdown(f"**{tr('Simulation settings')}**")
        origin, sites, workload, action = st.columns([1.2, 2.45, 1.3, 1.55], vertical_alignment="bottom", gap="medium")
        # Second row: every other setting visible as a regular input, styled like the top row.
        (lat_col, lon_col, latency_col, capacity_col, multiplier_col,
         model_col, runs_col, mc_col) = st.columns([.9, .9, 1.1, 1.1, 1.1, 1.55, 1.05, 1.5],
                                                   vertical_alignment="bottom", gap="small")
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
        with origin:
            st.selectbox(tr("User request origin"), cities, index=cities.index(st.session_state.user_label)
                     if st.session_state.user_label in cities else 0,
                     key="city_selector", on_change=change_city,
                     help=tr("Sets the request origin used to calculate distance and modeled latency. This is not a datacenter location."))
        with lat_col:
            lat = st.number_input(tr("Latitude"), key="user_lat_input", format="%.4f",
                                  min_value=-90.0, max_value=90.0,
                                  help=tr("Request-origin latitude in decimal degrees: north is positive, south is negative."))
        with lon_col:
            lon = st.number_input(tr("Longitude"), key="user_lon_input", format="%.4f",
                                  min_value=-180.0, max_value=180.0,
                                  help=tr("Request-origin longitude in decimal degrees: east is positive, west is negative."))
        st.session_state.user_lat, st.session_state.user_lon = lat, lon

        catalog = {**a.DEFAULT_DATACENTERS, **a.EXTENDED_DC_LOCATIONS}
        defaults = st.session_state.get("selected_dcs", list(a.DEFAULT_DATACENTERS))
        with sites:
            selected = st.multiselect(tr("Choose 1–3 datacenters"), list(catalog), default=defaults,
                                  max_selections=3, key="ui_multiselect_3238",
                                  help=tr("Candidate sites for routing."))
        st.session_state.selected_dcs = selected
        with workload:
            requests = st.slider(tr("Total Requests"), 100, 10000, 1000, 100, key="ui_slider_3454",
                             help=tr("Requests in one simulated hour, not requests per second."))
        with latency_col:
            latency = st.slider(tr("Latency limit (ms)"), 50, 200, 100, 10, key="ui_slider_3463",
                                help=tr("Applies to every strategy. Requests over the limit are unserved."))
        with capacity_col:
            share = st.slider(tr("Max share per site (%)"), 10, 100, 100, 5, key="ui_slider_site_share",
                              help=tr("Most of the requests one site can take. 100% means no limit."))
        with multiplier_col:
            multiplier = st.slider(tr("Energy multiplier"), .5, 2., 1., .1, key="ui_slider_3478",
                                   help=tr("Scales energy per request (1.0 = model output)."))
        model_names = {"Train All & Compare (Recommended)": "All models (recommended)",
                       "Multiple Linear Regression (MLR) only": "MLR only",
                       "Artificial Neural Network (ANN) only": "ANN only",
                       "Bayesian Optimization only": "Bayesian ANN only",
                       "Bayesian (Not Available)": "Bayesian (not available)"}
        with model_col:
            choice = st.selectbox(tr("Prediction model"), [
                "Train All & Compare (Recommended)", "Multiple Linear Regression (MLR) only",
                "Artificial Neural Network (ANN) only",
                "Bayesian Optimization only" if a.BAYES_AVAILABLE else "Bayesian (Not Available)"],
                key="ui_radio_3521", format_func=localized_formatter(lambda option: model_names.get(option, option)),
                help=tr("Model used for predictions. Trains only if no saved model matches."))
        with runs_col:
            runs = st.slider(tr("Monte Carlo runs"), 20, 500, 100, 20, key="ui_slider_3490",
                             help=tr("Number of sampled trials. More runs take longer."))
        with mc_col:
            monte_carlo = st.checkbox(tr("Enable Monte Carlo Analysis"), value=True, key="ui_checkbox_3487",
                                     help=tr("Enables the Monte Carlo action in Experiments. Checking this box does not start analysis or retrain a model."))
        # Both actions share one slot: a compact Run simulation and Train all models beside it.
        actions = action.container(horizontal=True, horizontal_alignment="right", gap="small")
        with actions:
            run = st.button(tr("Run simulation"), type="primary", width="content", key="run_sim_btn",
                        icon=":material/play_arrow:",
                        help=tr("Recalculate routing with the current settings."))
        with actions:
            # Only asks for confirmation; nothing trains until the warning below is confirmed.
            st.button(tr("Train all models"), width="content", key="train_all_btn", icon=":material/model_training:",
                      on_click=lambda: st.session_state.update(confirm_train_all=True),
                      help=tr("Retrain MLR, ANN and Bayesian ANN."))
        if st.session_state.get("confirm_train_all"):
            train_all_confirmation()
    return SimpleNamespace(catalog=catalog, selected=selected, user_location=(lat, lon),
                           num_requests=requests, latency_threshold=latency,
                           max_dc_capacity_mw=100, max_site_share=share / 100, energy_multiplier=multiplier,
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
    with st.container(key="weather_heading", horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"**{tr('Datacenter conditions')}**", width="stretch")
        refresh = st.button(tr("Refresh weather"), key="refresh_weather_btn", width="content",
                            type="tertiary", icon=":material/refresh:",
                            help=tr("Fetch current weather now."))
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
    indicators = {}
    for name, current in latest.items():
        climate = a.classify_climate(current["temperature"], current["humidity"], current.get("wind_speed", 5))
        cooling = a.COOLING_SYSTEMS[st.session_state.cooling_selections[name]]
        detail_labels = {"hot_dry": "Hot / dry", "hot_humid": "Hot / humid",
                         "hot_moderate_humidity": "Hot / moderate humidity"}
        indicators[name] = {"category": climate["climate"], "emoji": climate["emoji"],
                            "label": detail_labels.get(climate["climate_detail"], climate["climate"].title()),
                            "cooling_name": cooling["name"], "cooling_icon": cooling["icon"], "pue": cooling["pue"]}
    st.session_state.dashboard_weather = deepcopy(latest)
    updated = dict(st.session_state.cooling_selections)
    for column, name in zip(st.columns(len(settings.selected)), settings.selected):
        with column:
            updated[name] = st.selectbox(f"{tr('Cooling technology')} · {name}", list(a.COOLING_SYSTEMS),
                index=list(a.COOLING_SYSTEMS).index(updated[name]), key=f"cooling_select_{name}",
                format_func=localized_formatter(lambda code: f"{a.COOLING_SYSTEMS[code]['icon']} {a.COOLING_SYSTEMS[code]['name']}"),
                help=tr("Press Run simulation to apply."))
    if updated != st.session_state.cooling_selections:
        st.session_state.cooling_selections = updated
        st.rerun()
    with st.expander(tr("Weather readings"), expanded=not bool(st.session_state.get("sim_results"))):
        render_weather_table(latest, indicators, updated)


def active_training_period():
    """Weather period of the models simulations use: the default, or one confirmed on the Models tab."""
    return tuple(st.session_state.get("training_period", DEFAULT_TRAINING_PERIOD))


def ready_models(a, choice, retrain, period=None):
    mode = ("all" if "All" in choice else "mlr" if "MLR" in choice else
            "ann" if "ANN" in choice else "bayesian" if a.BAYES_AVAILABLE else "mlr")
    source = (inspect.getsource(a.AIModelSuite) + inspect.getsource(a.classify_climate)
              + inspect.getsource(a.collect_historical_training_data)
              + Path(a.energy_model.__file__).read_text(encoding="utf-8")
              + json.dumps({"default": a.DEFAULT_DATACENTERS, "extended": a.EXTENDED_DC_LOCATIONS}, sort_keys=True))
    period = tuple(period or active_training_period())
    config = training_config(mode, hashlib.sha256(source.encode()).hexdigest(), a.FEATURE_NAMES, a.COOLING_TYPES,
                             period=period)
    config["bayesian_available"] = a.BAYES_AVAILABLE
    try:
        secrets = st.secrets.to_dict()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        secrets = {}
    store = ModelStore(a.ARTIFACT_DIR, a.get_database_url(secrets))
    st.session_state.setdefault("model_bundles", {})

    def train():
        suite = a.AIModelSuite()
        suite.training_period = period
        with st.spinner(tr("Training selected models (only needed once per configuration)...")):
            suite.train_all(mode)
        return bundle_from_suite(suite, config)

    bundle, origin, notices = get_or_train_bundle(config, store, st.session_state.model_bundles,
                                                  train, force=retrain)
    for notice in notices:
        st.warning(tr(notice))
    return restore_suite(a.AIModelSuite(), bundle), bundle, origin


def frozen_carbon(a, datacenters):
    """Carbon snapshot for this run. Reuses the last recorded snapshot while the hour-level carbon
    pattern is unchanged, so navigation does not re-time results; otherwise records a new one."""
    previous = st.session_state.get("sim_receipt", {}).get("inputs", {}).get("carbon")
    snapshot = a.carbon_snapshot(datacenters)
    if previous and previous.get("sites") == snapshot["sites"]:
        return previous
    return snapshot


def carbon_inputs(snapshot=None):
    """Keyword arguments that make a simulation use the recorded carbon snapshot."""
    snapshot = snapshot or st.session_state.get("sim_receipt", {}).get("inputs", {}).get("carbon")
    if not snapshot:
        return {}
    return {"carbon_intensities": {name: site["intensity_kg_per_kwh"] for name, site in snapshot["sites"].items()},
            "carbon_time": datetime.fromisoformat(snapshot["at_utc"])}


def simulation_results(a, settings, datacenters, weather, models, bundle):
    inputs = {"selected_dc_names": settings.selected, "num_requests": settings.num_requests,
              "cooling_selections": dict(st.session_state.cooling_selections),
              "energy_multiplier": settings.energy_multiplier,
              "latency_threshold": settings.latency_threshold,
              "max_dc_capacity_mw": settings.max_dc_capacity_mw, "max_site_share": settings.max_site_share,
              "user_location": settings.user_location, "weather_data": weather,
              "datacenters": datacenters, "model_revision": bundle["revision"],
              "model_choice": getattr(settings, "model_choice", None),
              "origin_name": st.session_state.get("user_label"),
              # Carbon inputs frozen once: reruns and later analyses reuse these exact values.
              "carbon": frozen_carbon(a, datacenters),
              "carbon_baselines": {name: a.get_carbon_data(info.get("region", "default"))
                                   for name, info in datacenters.items()}}
    key = hashlib.md5(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
    st.session_state.pop("weather_refresh_trigger", None)
    st.session_state.sim_executed_this_render = False
    st.session_state.sim_inputs_changed = (st.session_state.get("sim_key") != key
                                           or st.session_state.get("simulation_pending_model", False))
    if settings.run:
        trigger = "Run simulation button"
        snapshot = deepcopy(inputs)
        started_at = datetime.now(timezone.utc).isoformat()
        clock_start = perf_counter()
        calculated = a.run_simulation(
            datacenters, weather, settings.user_location, settings.num_requests,
            st.session_state.cooling_selections, settings.energy_multiplier, models,
            settings.latency_threshold, settings.max_dc_capacity_mw,
                max_site_share=settings.max_site_share, **carbon_inputs(inputs["carbon"]))
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
        st.session_state.sim_model_bundle = bundle
        st.session_state.sim_receipt = receipt
        st.session_state.sim_executed_this_render = True
        st.session_state.sim_key = key
        st.session_state.sim_inputs_changed = False
        st.session_state.simulation_pending_model = False
        st.session_state.force_sim_rerun = False
        st.session_state.temp_sensitivity = None
        st.session_state.scenario_results = None
        st.session_state.pop("mc_cache", None)
    return st.session_state.get("sim_results")


def execution_receipt(model_note=""):
    """One compact, right-aligned status line (model, run, timing) with the full receipt in a popover."""
    receipt = st.session_state.get("sim_receipt")
    if not receipt:
        return
    fresh = st.session_state.sim_executed_this_render
    completed = datetime.fromisoformat(receipt["completed_at_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
    with st.container(key="execution_receipt", horizontal=True, horizontal_alignment="right",
                      vertical_alignment="center", gap="small"):
        label = tr("New simulation completed") if fresh else tr("Reusing completed simulation")
        st.caption((f"{model_note} · " if model_note else "") +
                   f"**{label}** · `{receipt['run_id'][:12]}` · {completed} · "
                   f"{receipt['compute_duration_seconds']:.3f} s", width="content")
        with st.popover(tr("Execution receipt"), icon=":material/fact_check:", type="tertiary", width="content"):
            st.markdown(f"**{tr('Run ID')}:** `{receipt['run_id']}`")
            st.write(f"{tr('Trigger')}: {tr(receipt['trigger'])}")
            st.write(f"{tr('Completed at')}: {completed}")
            st.caption(tr("Duration measures routing calculations only; weather retrieval and model loading or training are excluded."))
            st.caption(tr("Navigation and language changes reuse this receipt."))
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


METRIC_NOTES = {
    "Energy (Wh)": "Energy: saved-model prediction, aggregated by allocated requests.",
    "Carbon (g)": "Emissions: deterministic energy × modeled grid intensity calculation.",
    "Latency (ms)": "Latency: deterministic distance and queue calculation; not measured network latency.",
    "Peak ΔT-AR (°C)": "Temperature rise: uncalibrated thermal estimate; not measured urban warming.",
}
METRIC_FORMATS = {"Energy (Wh)": "localized", "Carbon (g)": "localized", "Latency (ms)": "localized",
                  "Peak ΔT-AR (°C)": "scientific", "Peak ΔT-AR (°F)": "scientific", "Heat CV": "%.3f",
                  "Served Requests": "localized", "Unserved Requests": "localized"}
LOWER_IS_BETTER = ["Energy (Wh)", "Carbon (g)", "Latency (ms)", "Peak ΔT-AR (°C)", "Peak ΔT-AR (°F)",
                   "Heat CV", "Unserved Requests"]
def show_metrics_table(results):
    """Strategy comparison with the best value in each column highlighted (ties on every row are not)."""
    best_cell = f"font-weight: 600; background-color: {ui_colors()['best']}"
    frame = metrics_table(results)
    frame[["Energy (Wh)", "Carbon (g)", "Latency (ms)"]] = frame[["Energy (Wh)", "Carbon (g)", "Latency (ms)"]].round(1)
    frame = localized_dataframe(frame)

    def best(column, lowest):
        target = column.min() if lowest else column.max()
        if column.nunique() <= 1:
            return [""] * len(column)
        return [best_cell if value == target else "" for value in column]

    styler = frame.style
    for label in LOWER_IS_BETTER + ["Served Requests"]:
        if tr(label) in frame.columns:
            styler = styler.apply(best, subset=[tr(label)], lowest=label != "Served Requests")
    st.dataframe(styler, width="stretch", hide_index=True, column_config={
        tr(label): st.column_config.NumberColumn(help=tr(METRIC_NOTES[label]) if label in METRIC_NOTES else None,
                                                 format=fmt)
        for label, fmt in METRIC_FORMATS.items()})


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


def _label_positions(points):
    """Place each label above its marker, or below when a nearby marker sits higher."""
    positions = []
    for lat, lon in points:
        crowded = any(abs(lat - o_lat) < 8 and abs(lon - o_lon) < 15 and o_lat > lat
                      for o_lat, o_lon in points if (o_lat, o_lon) != (lat, lon))
        positions.append("bottom center" if crowded else "top center")
    return positions


def _fit_view(points, width=820, height=400, padding=.22):
    """Center and zoom for a web-mercator tile map that shows every point with some margin."""
    lats = [max(-80, min(80, p[0])) for p in points]
    lons = [p[1] for p in points]
    mercator = [math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) for lat in lats]
    lon_span = max(max(lons) - min(lons), 4) * (1 + 2 * padding)
    y_span = max(max(mercator) - min(mercator), .08) * (1 + 2 * padding)
    zoom = min(math.log2(width * 360 / (512 * lon_span)), math.log2(height * 2 * math.pi / (512 * y_span)))
    center_y = (max(mercator) + min(mercator)) / 2
    center_lat = math.degrees(2 * math.atan(math.exp(center_y)) - math.pi / 2)
    return {"lat": center_lat, "lon": (max(lons) + min(lons)) / 2}, max(.4, min(zoom, 9))


def _great_circle(start, end, spacing_deg=1.1):
    """Evenly spaced points along the great-circle route between two (lat, lon) pairs
    (roughly one every ``spacing_deg`` of arc), longitudes unwrapped."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*start, *end))
    vectors = [(math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
               for la, lo in ((lat1, lon1), (lat2, lon2))]
    angle = math.acos(max(-1, min(1, sum(a * b for a, b in zip(*vectors)))))
    if angle < 1e-9:
        return [start[0], end[0]], [start[1], end[1]]
    steps = max(8, round(math.degrees(angle) / spacing_deg))
    lats, lons = [], []
    for step in range(steps + 1):
        t = step / steps
        w1, w2 = math.sin((1 - t) * angle) / math.sin(angle), math.sin(t * angle) / math.sin(angle)
        x, y, z = (w1 * a + w2 * b for a, b in zip(*vectors))
        lat, lon = math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))
        if lons:  # keep the path continuous across the antimeridian
            lon += 360 * round((lons[-1] - lon) / 360)
        lats.append(lat)
        lons.append(lon)
    return lats, lons


ENERGY_SCALE = [[0, "#fde68a"], [.5, "#f59e0b"], [1, "#dc2626"]]


def routing_map(datacenters, user_location, distribution, origin_name=None, energy=None):
    """Tile map of the datacenters and request origin; markers sized by allocation share.

    ``energy`` maps datacenter name to energy per request (Wh); each site then gets a soft
    halo colored from low (pale yellow) to high (red), with a colorbar.
    """
    colors = ui_colors()
    figure = go.Figure()
    served = sum(distribution.values())
    names = list(datacenters)
    points = [(datacenters[n]["lat"], datacenters[n]["lon"]) for n in names] + [tuple(user_location)]
    positions = _label_positions(points)
    # Dotted routes from the request origin to each datacenter, drawn first so markers sit on
    # top. Tile-map lines cannot be dashed, so each route is a row of small evenly spaced dots.
    # Dot size follows the allocation share after a run; sites with no traffic fade to grey.
    route_points = []
    for index, name in enumerate(names):
        info = datacenters[name]
        lats, lons = _great_circle(tuple(user_location), (info["lat"], info["lon"]))
        route_points += list(zip(lats, lons))
        count = distribution.get(name, 0)
        share = count / served if served else 0
        active = count > 0 or not distribution
        figure.add_trace(go.Scattermap(
            lat=lats[1:-1], lon=lons[1:-1], mode="markers", hoverinfo="skip", showlegend=False,
            opacity=.8 if active else .4,
            marker=dict(size=(2.5 + share * 2.5) if distribution else 3,
                        color=colors["series"][index % len(colors["series"])] if active else colors["muted"])))
    energized = [n for n in names if energy and energy.get(n) is not None]
    if energized and energy is not None:
        values = [energy[n] for n in energized]
        low, high = min(values), max(values)
        figure.add_trace(go.Scattermap(
            lat=[datacenters[n]["lat"] for n in energized], lon=[datacenters[n]["lon"] for n in energized],
            mode="markers", hoverinfo="text", name=tr("Energy per request"),
            hovertext=[f"{n}<br>{tr('Energy per request')}: {v:.3g} Wh" for n, v in zip(energized, values)],
            marker=dict(size=38, opacity=.5, color=values, colorscale=ENERGY_SCALE,
                        cmin=low, cmax=high if high > low else low + 1e-9,
                        colorbar=dict(title=dict(text=tr("Wh / request"), side="top",
                                                 font=dict(size=11, color=colors["text"])),
                                      orientation="h", x=.02, xanchor="left", y=.04, yanchor="bottom",
                                      len=.28, thickness=8, outlinewidth=0,
                                      tickfont=dict(size=10, color=colors["text"]),
                                      nticks=3, tickformat=".3~g", bgcolor=colors["legend_bg"]))))
    for index, name in enumerate(names):
        info = datacenters[name]
        count = distribution.get(name, 0)
        share = count / served if served else 0
        active = count > 0 or not distribution
        size = 13 + share * 14
        label = name.split(",")[0] + (f" · {share:.0%}" if distribution else "")
        # A slightly larger ring underneath gives each marker an outline (tile-map markers have none).
        figure.add_trace(go.Scattermap(lat=[info["lat"]], lon=[info["lon"]], mode="markers", hoverinfo="skip",
                         marker=dict(size=size + 4, color=colors["marker_edge"]), showlegend=False))
        figure.add_trace(go.Scattermap(lat=[info["lat"]], lon=[info["lon"]],
                         mode="markers+text", text=[label], textposition=positions[index],
                         textfont=dict(color=colors["text"], size=13),
                         marker=dict(size=size, color=colors["series"][index % len(colors["series"])] if active else colors["muted"]),
                         name=name, hoverinfo="text",
                         hovertext=[f"{name}<br>{tr('Allocated requests')}: {count:,}<br>"
                                    f"{tr('Share of served requests')}: {share:.1%}" if distribution else name]))
    origin_label = origin_name or tr("Request origin")
    figure.add_trace(go.Scattermap(lat=[user_location[0]], lon=[user_location[1]], mode="markers", hoverinfo="skip",
                     marker=dict(size=18, color=colors["marker_edge"]), showlegend=False))
    figure.add_trace(go.Scattermap(lat=[user_location[0]], lon=[user_location[1]],
                     mode="markers+text", text=[origin_label], textposition=positions[-1],
                     textfont=dict(color=colors["origin"], size=13), name=origin_label, hoverinfo="text",
                     hovertext=[f"{tr('Request origin')}: {origin_label}"],
                     marker=dict(size=13, color=colors["origin"])))
    center, zoom = _fit_view(points + route_points)
    figure.update_layout(map=dict(style=colors["map_style"], center=center, zoom=zoom),
                         showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return figure


SUMMARY_METRICS = [("Energy (Wh)", "energy_wh", "{:,.1f}"), ("Carbon (g)", "carbon_g", "{:,.1f}"),
                   ("Latency (ms)", "avg_latency_ms", "{:,.1f}"), ("Peak ΔT-AR (°C)", "peak_ΔT", "{:.3g}")]
BASELINE_STRATEGY = "Energy-Only"


SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci_number(value, digits=3):
    """Scientific notation for display: 7.68 × 10⁻⁵ (``digits`` significant figures).
    Values between 0.01 and 10,000 stay as plain decimals."""
    if not value:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -2 <= exponent < 4:
        return f"{value:,.{max(0, digits - 1 - exponent)}f}"
    mantissa = value / 10 ** exponent
    if round(abs(mantissa), digits - 1) >= 10:  # e.g. 9.999 rounds up to the next power
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.{digits - 1}f} × 10{str(exponent).translate(SUPERSCRIPTS)}"


METRIC_LABELS = {"energy_wh": "energy", "carbon_g": "carbon", "avg_latency_ms": "latency", "peak_ΔT": "peak ΔT-AR"}


def recommendation_reason(results, recommendation):
    """Plain-language 'why', built from the scores generate_final_recommendation computed."""
    recommended = recommendation["recommended_strategy"]
    scores, gaps = recommendation["scores"], recommendation["gaps"]
    candidates = sorted(recommendation["candidates"], key=lambda name: scores[name])
    served = recommendation["served_requests"]
    offered = results[recommended]["totals"].get("offered_requests", served)
    served_text = (tr("all {n} requests").format(n=f"{served:,}") if served == offered
                   else tr("{n} of {total} requests").format(n=f"{served:,}", total=f"{offered:,}"))
    lines = [f"{tr('Why:')} " + tr("{name} has the best balance of energy, carbon, latency and peak ΔT-AR among {count} strategies serving {served} (average gap {gap}).").format(
        count=len(candidates), served=served_text, name=f"**{tr(recommended)}**",
        gap=f"{scores[recommended]:.1%}")]
    behind = [f"{tr(METRIC_LABELS[m])} +{value:.1%}" for m, value in gaps[recommended].items() if value > 0.0005]
    lines.append(tr("Gaps to the best: {list}.").format(list=", ".join(behind)) if behind
                 else tr("It is the best candidate on every metric."))
    runner_up = next((name for name in candidates if name != recommended), None)
    if runner_up:
        margin = scores[runner_up] - scores[recommended]
        lines.append(tr("Next: {name} (average gap {gap}).").format(name=f"**{tr(runner_up)}**",
                                                                     gap=f"{scores[runner_up]:.1%}")
                     + (" " + tr("The two are within one percentage point, effectively tied.") if margin < 0.01 else ""))
    sites = results[recommended].get("datacenters", {})
    if sites:
        hottest = max(sites, key=lambda name: sites[name].get("ΔT_contribution", 0))
        lines.append(tr("Its peak heating is at {site}, which carries {share} of the traffic.").format(
            site=hottest, share=f"{sites[hottest].get('percentage', 0):.0f}%"))
    fewer = [name for name, why in recommendation["excluded"].items() if why == "fewer requests served"]
    if fewer:
        lines.append(tr("Not compared because they served fewer requests:") + " " + ", ".join(
            f"{tr(name)} ({results[name]['totals'].get('unserved_requests', 0):,} {tr('unserved')})" for name in fewer) + ".")
    lines.append(tr("Random is a reference baseline and is never recommended."))
    return lines


def results_summary(results, recommendation):
    """Headline numbers for the recommended strategy, compared with the Energy-Only baseline."""
    recommended = recommendation["recommended_strategy"]
    best = results[recommended]["totals"]
    base = results.get(BASELINE_STRATEGY, {}).get("totals")
    with st.container(key="results_summary", border=True):
        st.markdown(f"**{tr('Recommended:')} {tr(recommended)}** · "
                    f"{tr('Best balance of energy, carbon, latency and peak ΔT-AR.')}")
        # Where the recommended strategy sends the requests, largest share first.
        allocation = {name: count for name, count in results[recommended]["distribution"].items() if count}
        served = sum(allocation.values())
        if served:
            sites = " · ".join(f"**{name}** {count / served:.0%} ({count:,})"
                               for name, count in sorted(allocation.items(), key=lambda item: -item[1]))
            unserved = best.get("unserved_requests", 0)
            st.markdown(f":material/location_on: {tr('Where:')} {sites}"
                        + (f" · {tr('Unserved Requests')}: {unserved:,}" if unserved else ""))
            st.caption(tr("The split depends on the strategy, weather, cooling and limits."))
        st.markdown(":material/lightbulb: " + " ".join(recommendation_reason(results, recommendation)))
        if recommended == BASELINE_STRATEGY:
            st.caption(tr("Energy-Only is also the comparison baseline, so no change percentages are shown."))
        offered = best.get("offered_requests", best.get("served_requests", 0))
        st.caption(tr("Totals for {name} over one simulated hour ({n} requests). Percentages compare with Energy-Only; lower is better.").format(
            name=tr(recommended), n=f"{offered:,}"))
        for column, (label, field, fmt) in zip(st.columns(4), SUMMARY_METRICS):
            delta = None
            if base and recommended != BASELINE_STRATEGY and base[field]:
                change = (best[field] - base[field]) / base[field] * 100
                delta = f"{change:+.1f}% {tr('vs Energy-Only')}"
            value = sci_number(best[field]) if field == "peak_ΔT" else fmt.format(best[field])
            column.metric(tr(label), value, delta=delta, delta_color="inverse",
                          help=tr(METRIC_NOTES[label]) if label in METRIC_NOTES else None)


STRATEGY_GUIDE = [
    ("Random", ":material/shuffle:", "Splits requests at random.",
     "Reference baseline for comparison; never recommended."),
    ("Energy-Only", ":material/bolt:", "Favors the sites with the lowest predicted energy per request.",
     "Directs more requests to lower-energy sites while respecting the shared limits."),
    ("Heat-Only", ":material/local_fire_department:", "Favors the site with the lowest temperature rise per request.",
     "Directs more requests to sites with less modeled warming per request; concentrating traffic can still increase peak heating."),
    ("ΔT-AR", ":material/thermostat:", "Splits requests so every site warms equally.",
     "Minimizes the peak modeled temperature rise (ΔT-AR) across sites."),
    ("Multi-Objective", ":material/balance:", "Scores each site on energy, latency, carbon and local heat.",
     "Equal weights; favors the site with the best combined score."),
]


def strategy_guide():
    """Short explanation of the five routing strategies, shown at the top of the Overview."""
    with st.container(key="strategy_guide"):
        st.markdown(f"**{tr('How the routing strategies work')}**")
        for column, (name, icon, what, detail) in zip(st.columns(len(STRATEGY_GUIDE), gap="small"), STRATEGY_GUIDE):
            with column:
                st.markdown(f"{icon} **{tr(name)}**")
                st.markdown(tr(what))
                st.caption(tr(detail))
        st.caption(tr("All strategies share the same weather, model and limits. Requests that can't be served are shown as unserved."))


def overview(a, settings, datacenters, weather, results, model_origin=None):
    strategy_guide()
    recommended = None
    if results:
        recommendation = a.generate_final_recommendation(results, settings.latency_threshold)
        if recommendation.get("feasible", True):
            recommended = recommendation["recommended_strategy"]
            results_summary(results, recommendation)
        else:
            st.warning(tr("No strategy can serve any request.") + " " +
                       tr("Binding limit: {limit}.").format(limit=tr(recommendation.get("binding_constraint", ""))) + " " +
                       tr("Raise the latency limit, power capacity or share per site, choose closer datacenters, "
                          "or move the request origin."), icon=":material/block:")
        st.subheader(tr("All routing strategies"))
        analysis_context("routing")
        show_metrics_table(results)
        st.caption(tr("Compare served and unserved requests before comparing energy, emissions, or thermal impact.") + " " +
                   tr("Random routing is sampled; the other strategies repeat exactly for the same inputs."))
    else:
        st.caption(tr("Ready to compare. Use Run simulation above."))

    map_column, traffic_column = st.columns([2.1, 1], gap="medium")
    with map_column, st.container(key="routing_atlas"):
        st.subheader(tr("Request origin and datacenters"))
        city = getattr(settings, "origin_name", st.session_state.get("user_label", ""))
        origin_name = city if a.CITY_DATABASE.get(city) == settings.user_location else f"{settings.user_location[0]:.4f}, {settings.user_location[1]:.4f}"
        distribution = results[recommended]["distribution"] if recommended else {}
        per_site = results[recommended].get("datacenters", {}) if recommended else {}
        energy = {name: site["energy_per_req"] for name, site in per_site.items() if site.get("requests")}
        display_plot(routing_map(datacenters, settings.user_location, distribution, origin_name, energy),
                     "overview_map", 420)
        if recommended:
            st.caption(tr("Dark marker: request origin. Size and % show each site's share.") + " " +
                       tr("Halo color: energy per request (pale = lower, red = higher)."))
        else:
            st.caption(tr("Dark marker: request origin. Lines: candidate routes."))
    with traffic_column, st.container(key="allocation_panel"):
        st.subheader(tr("Traffic Distribution"))
        if results:
            names = list(datacenters)
            chart = go.Figure()
            for index, name in enumerate(names):
                counts = [data["distribution"].get(name, 0) for data in results.values()]
                chart.add_trace(go.Bar(x=counts, y=list(results), name=name.split(",")[0],
                               orientation="h", marker_color=ui_colors()["series"][index % 3],
                               text=[f"{count:,}" if count else "" for count in counts], textposition="inside",
                               hovertemplate="%{y}<br>%{fullData.name}: %{x:,}<extra></extra>"))
            chart.update_layout(barmode="stack", xaxis_title="Requests", yaxis=dict(autorange="reversed"),
                                legend=dict(orientation="h", y=-.3, x=0))
            display_plot(chart, "overview_traffic", 350)
        else:
            st.info(tr("Run the experiment to see request allocation and recommendations."))
            for index, (name, current) in enumerate(weather.items()):
                st.markdown(f"**{name.split(',')[0]}** · {temperature_text(current['temperature'])}")
    with st.expander(tr("Datacenter conditions and sources")):
        st.dataframe(localized_dataframe(conditions_table(a, datacenters, weather)), width="stretch", hide_index=True)
        st.caption(tr("Reported predictions and comparisons are analysis outputs, not datacenter field measurements."))


def routing_view(a, settings, datacenters, weather, results):
    st.subheader(tr("Routing comparison"))
    analysis_context("routing")
    st.caption(tr("Compare service coverage alongside energy, emissions, and thermal impact."))
    show_metrics_table(results)
    left, right = st.columns(2)
    with left, st.container(key="chart_card_metrics"):
        st.subheader(tr("Multi-Metric Comparison"))
        display_plot(a.create_metrics_comparison_chart(results), "routing_metrics", 390)
    with right, st.container(key="chart_card_heat"):
        st.subheader(tr("Heat Distribution by Strategy"))
        display_plot(a.create_heat_distribution_chart(results), "routing_heat", 390)
    with st.expander(tr("Carbon Intensity Analysis")):
        analysis_context("carbon")
        display_plot(a.create_carbon_intensity_curves(datacenters), "routing_carbon", 330)
        st.dataframe(localized_dataframe(conditions_table(a, datacenters, weather)), width="stretch", hide_index=True)
    with st.expander(tr("Cooling Technology Analysis")):
        analysis_context("cooling")
        st.caption(tr("The cooling matrix is a qualitative scenario assumption, not a measured ranking extracted from the cited review."))
        display_plot(a.create_cooling_effectiveness_heatmap(), "routing_cooling", 330)
        display_plot(a.create_pue_comparison_chart(), "routing_pue", 300)


def models_view(a, suite, bundle):
    best_name, _ = suite.get_best_model()
    with st.container(key="models_note", border=True):
        st.markdown(f":material/model_training: **{tr('What the models do')}**")
        st.markdown(tr("They predict energy per request (Wh) at each site from its weather and cooling. "
                       "Routing, carbon and ΔT-AR all build on this prediction; the app uses {best}.").format(
            best=f"**{best_name}**"))
        st.caption((tr("Trained on real historical weather, but the energy labels come from the app's energy formula. "
                       "The models are a surrogate of that formula, not measurements of real hardware.")
                    if suite.training_source == "real" else
                    tr("Trained on synthetic weather with energy labels from the app's energy formula. "
                       "The models are a surrogate of that formula, not measurements of real hardware."))
                   + f" {tr('Trained')}: {bundle['trained_at'][:16].replace('T', ' ')} UTC · "
                     f"{suite.training_samples:,} {tr('samples')}.")
    analysis_context("models")
    metrics = [{"Model": name, "Validation R²": value.get("val_r2"), "Test R²": value["r2"],
                "Test MAE": value["mae"], "Test RMSE": value["rmse"]}
               for name, value in suite.metrics.items() if isinstance(value, dict) and "r2" in value]
    st.dataframe(localized_dataframe(pd.DataFrame(metrics)), width="stretch", hide_index=True)
    # Cooling comparisons are validated separately: changing a site's cooling relies on the model
    # ordering technologies correctly under the same weather, which overall R² does not measure.
    checks = suite.metrics.get("cooling_validation")
    st.markdown(f"**{tr('Cooling comparison check')}**")
    if checks:
        st.dataframe(localized_dataframe(pd.DataFrame([
            {"Model": name, "Same order as formula": check["order_agreement"],
             "Difference error (Wh/request)": check["delta_mae_wh"],
             "Worst difference error (Wh/request)": check["delta_max_error_wh"],
             "Weather samples": check["weather_samples"]} for name, check in checks.items()])),
            width="stretch", hide_index=True, column_config={
                tr("Same order as formula"): st.column_config.ProgressColumn(
                    tr("Same order as formula"), format="percent", min_value=0, max_value=1,
                    help=tr("Share of cooling pairs ranked the same way as the formula.")),
                tr("Difference error (Wh/request)"): st.column_config.NumberColumn(format="%.4f"),
                tr("Worst difference error (Wh/request)"): st.column_config.NumberColumn(format="%.4f")})
        st.caption(tr("How often each model ranks cooling options the same way as the reference formula."))
    else:
        st.warning(tr("This saved model predates the cooling comparison check. Retrain the selected models to validate cooling comparisons."))
    left, right = st.columns(2)
    with left, st.container(key="chart_card_performance"):
        st.subheader(tr("Model Performance Comparison"))
        display_plot(a.create_model_comparison_chart(suite.metrics), "models_performance", 360)
    with right, st.container(key="chart_card_predictions"):
        st.subheader(tr("Predicted vs formula-generated energy"))
        display_plot(a.create_prediction_scatter(suite.metrics, best_name), "models_predictions", 360)
        total = len(suite.metrics.get(best_name, {}).get("y_test", []))
        if total > 4000:
            st.caption(tr("Showing {shown} of {total} points; R² uses all.").format(
                shown="4,000", total=f"{total:,}"))
    with st.expander(tr("Training diagnostics")):
        for name, function in [("learning", a.create_learning_curve_chart), ("bayes", a.create_bayesopt_convergence_chart)]:
            display_plot(function(suite.metrics), f"models_{name}", 330)
        if suite.metrics.get("feature_importance"):
            display_plot(a.create_feature_importance_chart(suite.metrics["feature_importance"], best_name),
                         "models_importance", 330)


def experiments_view(a, settings, datacenters, weather, suite, results):
    cooling_selections = st.session_state.sim_receipt["inputs"]["cooling_selections"]
    tool = st.selectbox(tr("Analysis"), ["Monte Carlo", "Temperature sensitivity", "What-if scenario"],
                        key="experiment_tool", format_func=localized_formatter(),
                        help=tr("Each analysis runs only when you press its button."))
    key = st.session_state.sim_key
    if tool == "Monte Carlo":
        analysis_context("monte_carlo")
        st.caption(tr("Monte Carlo analysis runs only when requested."))
        st.caption(tr("Intervals show simulated variability, not measurement uncertainty."))
        if st.button(tr("Run Monte Carlo analysis"), key="run_monte_carlo_btn", disabled=not settings.enable_monte_carlo):
            with st.spinner(tr("Running Monte Carlo simulations...")):
                data = a.run_monte_carlo(datacenters, weather, settings.user_location, settings.num_requests,
                    cooling_selections, settings.energy_multiplier, suite,
                    settings.n_runs, settings.latency_threshold, settings.max_dc_capacity_mw,
                max_site_share=settings.max_site_share, **carbon_inputs())
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
        analysis_context("statistics")
        st.markdown(executive_results_markdown(results, stats, is_spanish()))
        with st.expander(tr("Distribution and convergence")):
            analysis_context("convergence")
            display_plot(a.create_monte_carlo_hist_overlay(data, metric="peak_ΔT"), "mc_distribution", 330)
            display_plot(a.create_monte_carlo_running_mean_chart(data, metric="peak_ΔT"), "mc_convergence", 330)
    elif tool == "Temperature sensitivity":
        analysis_context("temperature")
        if st.button(tr("🌡️ Run Temperature Sensitivity Analysis"), key="run_temp_sensitivity"):
            temperatures = list(range(0, 46, 3))
            values = {strategy: [] for strategy in results}
            with st.spinner(tr("Running temperature sensitivity analysis...")):
                for temperature in temperatures:
                    modified = {name: {**current, "temperature": temperature} for name, current in weather.items()}
                    evaluated = a.run_simulation(datacenters, modified, settings.user_location,
                        settings.num_requests, cooling_selections, settings.energy_multiplier,
                        suite, settings.latency_threshold, settings.max_dc_capacity_mw,
                max_site_share=settings.max_site_share, **carbon_inputs())
                    for strategy, result in evaluated.items():
                        values[strategy].append(result["totals"]["peak_ΔT"])
            st.session_state.temp_sensitivity = {"x": temperatures, "y": values}
        if st.session_state.get("temp_sensitivity"):
            display_plot(a.create_sensitivity_temperature_chart(st.session_state.temp_sensitivity), "temperature_analysis", 430)
    else:
        analysis_context("scenario")
        names = st.multiselect(tr("Select datacenters for alternative scenario:"), list(settings.catalog),
                               max_selections=3, key="scenario_datacenters",
                               help=tr("Compare up to three alternative sites under the same settings."))
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
                settings.latency_threshold, settings.max_dc_capacity_mw,
                max_site_share=settings.max_site_share, **carbon_inputs())
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


HELP_SETTING_KEYS = ("city_selector", "user_lat_input", "user_lon_input", "ui_multiselect_3238",
                     "ui_slider_3454", "ui_slider_3463", "ui_slider_site_share", "ui_slider_3478",
                     "ui_radio_3521", "ui_checkbox_3487", "ui_slider_3490", "dashboard_view",
                     "experiment_tool", "scenario_datacenters")


def open_help():
    st.session_state.help_return_settings = {key: st.session_state[key] for key in HELP_SETTING_KEYS if key in st.session_state}
    st.session_state.show_help = True


def close_help():
    st.session_state.update(st.session_state.get("help_return_settings", {}))
    st.session_state.show_help = False


def help_page():
    st.button(tr("Back to dashboard"), key="back_to_dashboard", on_click=close_help, icon=":material/arrow_back:")
    st.subheader(tr("Help and methodology"))
    start, sources, weather, analyses = st.tabs([tr("Getting started"), tr("Sources and methodology"),
                                               tr("Weather information"), tr("Analysis guide")])
    with start:
        st.write(tr("Pick an origin, up to three datacenters and a request count. Set cooling per site in the conditions panel."))
        st.write(tr("Run simulation compares all strategies using the saved models. The execution receipt records the inputs."))
        st.write(tr("Overview: summary. Routing: detailed comparison. Models: validation and retraining. Experiments: extra analyses."))
    with sources:
        st.dataframe(pd.DataFrame(provenance_rows(is_spanish())), hide_index=True, width="stretch")
        st.markdown(reference_markdown(is_spanish()))
    with weather:
        current = st.session_state.get("dashboard_weather", {})
        if current:
            st.dataframe(localized_dataframe(weather_sources(current)), hide_index=True, width="stretch")
        st.write(tr("Weather is checked every 5 minutes. Fallback values are labeled."))
        st.write(tr("Results update only when you press Run simulation."))
        st.write(tr("Icons show the temperature category: cold below 10 °C [50 °F], cool 10 to 20 °C, moderate 20 to 28 °C, hot 28 °C [82.4 °F] and above."))
        st.caption(tr("PUE values are fixed assumptions, not measured efficiency."))
    with analyses:
        for key in GUIDANCE:
            title, purpose, method, limits = guidance(key)
            with st.expander(title):
                st.markdown(f"**{tr('Purpose')}**\n\n{purpose}\n\n**{tr('Method')}**\n\n{method}\n\n**{tr('Interpretation and limits')}**\n\n{limits}")


def render_dashboard(namespace):
    a = SimpleNamespace(**namespace)
    # Streamlit sets `color-scheme` on the app container to the active appearance, so
    # light-dark() picks the right color instantly when the user switches themes.
    st.html("""<style>
    /* Page: theme background color (from config.toml) under a faint datacenter network that
       warms from teal to red, softened by a veil so it stays in the background. */
    .stApp {
        background-image:
            linear-gradient(light-dark(rgba(242, 246, 247, .35), rgba(11, 20, 23, .45)),
                            light-dark(rgba(242, 246, 247, .35), rgba(11, 20, 23, .45))),
            url("__NETWORK__");
        background-size: 100% 100%, cover; background-position: center; background-attachment: fixed; }
    [data-testid="stHeader"] { background: transparent; }

    /* Header: title, subtitle and author with an even, readable rhythm. */
    .st-key-dashboard_header { padding: .4rem 0 .9rem; }
    .st-key-dashboard_header h1 { letter-spacing: -.03em; line-height: 1.15; padding: 0 0 .75rem; }
    .st-key-dashboard_header h1, .st-key-weather_matrix { font-variant-numeric: tabular-nums; }
    .st-key-project_heading { gap: .7rem; }
    .st-key-project_heading p { line-height: 1.6; }
    .st-key-project_heading [data-testid="stCaptionContainer"] { padding-top: .15rem; }

    /* Cards: one surface, one border, one padding everywhere. */
    .st-key-experiment_panel, .st-key-header_weather, .st-key-routing_atlas,
    .st-key-allocation_panel, .st-key-results_summary, .st-key-strategy_guide, .st-key-models_note,
    .st-key-period_retrain {
        background: light-dark(#ffffff, #121e22);
        border: 1px solid light-dark(#dbe5e8, #22343a); border-radius: 10px;
        padding: 1rem 1.15rem; box-shadow: 0 1px 2px rgba(0, 0, 0, .06); }
    .st-key-dashboard_view { margin: .35rem 0 0; }
    [class*="st-key-chart_card"] { background: light-dark(#ffffff, #121e22);
        border: 1px solid light-dark(#dbe5e8, #22343a); border-radius: 10px; padding: 1rem 1.15rem; }
    [data-testid="stExpander"] details { background: light-dark(#ffffff, #121e22); }
    /* Cards side by side share one height: each card grows to fill its column, and columns in
       a row already stretch to the tallest one. */
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(
        > [class*="st-key-chart_card"], > .st-key-routing_atlas, > .st-key-allocation_panel) { flex: 1 1 auto; }
    /* Coordinates are typed, not stepped: hide the +/- buttons so the full value fits. */
    .st-key-user_lat_input [data-testid^="stNumberInputStep"],
    .st-key-user_lon_input [data-testid^="stNumberInputStep"] { display: none; }
    .st-key-header_weather:has(.st-key-conditions_expander) { padding: .2rem .4rem; }
    .st-key-conditions_expander details { border: none; }
    .st-key-execution_receipt { margin: -.25rem 0 -.35rem; }
    /* Selected workspace tab: filled with the primary teal instead of only outlined. */
    .st-key-dashboard_view button[role="radio"][aria-checked="true"] {
        background: #0f766e; border-color: #0f766e; }
    .st-key-dashboard_view button[role="radio"][aria-checked="true"],
    .st-key-dashboard_view button[role="radio"][aria-checked="true"] * { color: #ffffff; }
    .st-key-dashboard_view button[role="radio"][aria-checked="true"]:hover {
        background: #0b5f58; border-color: #0b5f58; }
    .st-key-routing_atlas .js-plotly-plot { border-radius: 8px; overflow: hidden; }
    .st-key-routing_atlas h3, .st-key-allocation_panel h3 { padding: 0 0 .5rem; }
    @media(max-width: 760px) {
        .st-key-experiment_panel, .st-key-header_weather, .st-key-routing_atlas,
        .st-key-allocation_panel, .st-key-results_summary { padding: .75rem; }
        .st-key-dashboard_header h1 { font-size: 1.85rem; }
    }
    </style>""".replace("__NETWORK__", network_backdrop()))
    with st.container(key="dashboard_header"):
        title, tools = st.columns([4.5, 1.2], vertical_alignment="top")
        with title, st.container(key="project_heading", gap="small"):
            st.title("AI-Assisted Datacenter Prompt Routing for ΔT-AR Mitigation")
            st.markdown("Demonstrating the usage of AI to mitigate localized heat concentration effect caused by datacenters")
            st.caption("Phani Raja Bharath Balijepalli | Urban Digital Twin Lab | University of Central Florida")
        with tools, st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small"):
            st.toggle("Español", key="spanish_ui", help=tr("Switch between English and technical Spanish."))
            st.button(tr("Help"), key="open_help", on_click=open_help, icon=":material/help:")
    if st.session_state.get("show_help"):
        help_page()
        return
    controls_slot = st.container(key="horizontal_controls")
    weather_slot = st.container(key="header_weather")
    view = st.segmented_control(tr("Workspace"), VIEWS, default="Overview", key="dashboard_view",
                                format_func=localized_formatter(), label_visibility="collapsed", width="content") or "Overview"
    with controls_slot:
        settings = experiment_controls(a)
    if not settings.selected:
        st.info(tr("Choose 1–3 datacenters"))
        return
    datacenters, weather = selected_conditions(a, settings)
    with weather_slot:
        weather_panel(a, settings, weather)
    retrain = False
    if view == "Models":
        st.subheader(tr("Saved prediction models"))
        retrain = st.button(tr("Retrain selected models"), key="retrain_models_btn",
                            help=tr("Replace the saved models for this training configuration."))
    if view == "Models":
        period_retrain_panel(settings)
    retrain = retrain or st.session_state.pop("retrain_period_now", False)
    retrained = None
    train_all = st.session_state.pop("train_all_now", False)
    if retrain or train_all:
        started = perf_counter()
        retrained = ready_models(a, TRAIN_ALL_CHOICE if train_all else settings.model_choice, True)
        st.session_state.simulation_pending_model = True
        if train_all:
            st.success(tr("All models retrained and saved in {minutes} min. Press Run simulation to apply them to routing results.").format(
                minutes=f"{(perf_counter() - started) / 60:.1f}"), icon=":material/check_circle:")
        else:
            st.info(tr("Retrained models are saved. Press Run simulation to apply them to routing results."))
    if not settings.run and st.session_state.get("sim_results") is None:
        if retrained:
            models_view(a, retrained[0], retrained[1])
            return
        if view == "Overview":
            overview(a, settings, datacenters, weather, None)
        else:
            st.info(tr("Run the experiment to populate this view."))
        return
    if settings.run:
        suite, bundle, origin = ready_models(a, settings.model_choice, False)
        st.session_state.sim_model_origin = origin
    else:
        bundle = st.session_state.get("sim_model_bundle")
        if bundle is None:
            revision = st.session_state.get("sim_receipt", {}).get("model_revision")
            bundle = next((item for item in st.session_state.get("model_bundles", {}).values()
                           if item.get("revision") == revision), None)
        if bundle is None:
            st.info(tr("Run the experiment to populate this view."))
            return
        st.session_state.sim_model_bundle = bundle
        suite = restore_suite(a.AIModelSuite(), bundle)
        origin = "session"
    if retrained:
        suite, bundle, origin = retrained
    trained = datetime.fromisoformat(bundle["trained_at"]).strftime("%Y-%m-%d %H:%M UTC")

    def model_details():
        with st.expander(tr("Training data and model details"), expanded=view == "Models"):
            st.markdown(f"**{tr('Available models')}:** {', '.join(suite.models)} · "
                        f"**{tr('Training samples')}:** {suite.training_samples:,} · "
                        f"**{tr('Saved model trained at')}:** {trained}")
            st.caption(tr(f"Models ready: {origin}. Simulation runs reuse these models."))
            period = bundle.get("training_config", {}).get("period")
            if suite.training_source == "real" and period:
                st.markdown(f"**{tr('Requested historical weather window')}:** {period[0]} {'a' if is_spanish() else 'to'} {period[1]}")
                st.caption(tr("Requested download window; actual coverage per site was not recorded."))
            else:
                st.caption(tr("Synthetic training data: no observed calendar date range applies."))
            st.caption(tr("Energy targets come from formulas, not measured hardware."))

    # Keep the first screen for results: model details sit at the top only on the Models tab.
    if view == "Models":
        model_details()
    results = simulation_results(a, settings, datacenters, weather, suite, bundle)
    receipt = st.session_state.sim_receipt
    execution_receipt(f"{tr('Prediction model')}: {receipt['prediction_model']}")
    if st.session_state.sim_inputs_changed:
        st.warning(tr("Inputs changed. Press Run simulation to update the results."),
                   icon=":material/pending_actions:")
    # Results, maps, comparisons, and analyses share the committed run snapshot.
    applied = receipt["inputs"]
    applied_settings = SimpleNamespace(**vars(settings))
    for field in ("num_requests", "energy_multiplier", "latency_threshold", "max_dc_capacity_mw", "max_site_share", "user_location"):
        setattr(applied_settings, field, applied[field])
    applied_settings.selected = applied["selected_dc_names"]
    applied_settings.origin_name = applied.get("origin_name", "")
    settings = applied_settings
    datacenters, weather = deepcopy(applied["datacenters"]), deepcopy(applied["weather_data"])
    applied_suite = restore_suite(a.AIModelSuite(), st.session_state.sim_model_bundle)
    shortfalls = [f"{tr(name)} ({value['totals']['unserved_requests']:,})" for name, value in results.items()
                  if value["totals"]["unserved_requests"]]
    if shortfalls:
        st.warning(f"{tr('Unserved Requests')} ({tr('of')} {settings.num_requests:,}): " + ", ".join(shortfalls) + ". " +
                   tr("Compare service rates alongside energy and heat metrics."), icon=":material/warning:")
    if view == "Overview":
        overview(a, settings, datacenters, weather, results, origin)
    elif view == "Routing":
        routing_view(a, settings, datacenters, weather, results)
    elif view == "Models":
        models_view(a, suite, bundle)
    else:
        experiments_view(a, settings, datacenters, weather, applied_suite, results)
    if view != "Models":
        model_details()
    recommendation = a.generate_final_recommendation(results, settings.latency_threshold)
    exports(a, results, recommendation)
    st.caption(tr("Simulation outputs · Sources and assumptions available above"))
