import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from scipy import stats
import joblib, os
import plotly.io as pio
from io import BytesIO
import hashlib, json
import logging
import openmeteo_requests
import requests_cache
from retry_requests import retry
from typing import Any
from streamlit.errors import StreamlitSecretNotFoundError
from energy_model import (BASE_ENERGY_WH, ENERGY_PER_PROMPT_BASE, COOLING_SYSTEMS,
                          COOLING_TYPES, FEATURE_NAMES, MODEL_INPUT_NAMES, cooling_index, encode_features,
                          calculate_energy_per_request, calculate_ΔT_contribution,
                          surrogate_energy_target)
from routing_core import allocate_requests, latency_request_capacity
from model_store import (ModelStore, training_config, config_key, bundle_from_suite,
                         restore_suite, get_or_train_bundle)
from network_model import haversine_distance, calculate_latency, site_utilization
from carbon_model import CARBON_INTENSITY_BASE, get_carbon_intensity, get_carbon_data, local_solar_hour, carbon_snapshot
from routing_strategies import (latency_caps_for, energy_map, route_random, route_energy_only, route_heat_only,
                                route_ΔT_aware, route_multi_objective)
import inspect
import energy_model
from app_config import get_database_url
from localization import tr, is_spanish, localized_dataframe, localized_figure, localized_formatter
from provenance import provenance_rows, reference_markdown, executive_results_markdown, recommendation_markdown
import warnings
warnings.filterwarnings('ignore')

# Configure logging for API call tracking
logging.basicConfig(
    filename='datacenter_api_calls.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try to import Bayesian Optimization
try:
    from bayes_opt import BayesianOptimization
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False

# Try to import FPDF for PDF export
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "ai_models_bundle.joblib"

PLOTLY_CONFIG = {
    "responsive": True,          # Make Plotly react to container resize
    "displaylogo": False,        # Hide Plotly logo
    "displayModeBar": True,      # Show/hide toolbar; set to "hover" if you prefer
    "scrollZoom": True,          # Allow mousewheel zoom
    "doubleClick": "reset",      # double click resets axes
    "toImageButtonOptions": {    # export settings
        "format": "png",
        "scale": 4,              # higher = sharper export (increased for PDF quality)
        "width": 1920,
        "height": 1080,
    },
    # Optional: remove noisy buttons
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d",
        "autoScale2d", "toggleSpikelines"
    ],
}

PLOTLY_CONFIG_CLEAN = {**PLOTLY_CONFIG, "displayModeBar": False}
PLOTLY_CONFIG_DEFAULT = {**PLOTLY_CONFIG, "displayModeBar": True}

FIG_EXPORT_PRESETS = {
    # High-quality paper-friendly sizes (pixels) with higher scale for better clarity
    "Paper: single-column (3.5in @300dpi)": {"width": 1050, "height": 650, "scale": 3},
    "Paper: double-column (7.2in @300dpi)": {"width": 2160, "height": 1200, "scale": 3},
    "Paper: portrait figure (3.5in wide, tall)": {"width": 1050, "height": 1200, "scale": 3},

    # High-resolution exports for presentations and posters
    "Presentation (1920x1080)": {"width": 1920, "height": 1080, "scale": 3},
    "High-res Export (4K)": {"width": 3840, "height": 2160, "scale": 2},

    # Full-page exports with enhanced quality
    "A4 Portrait @300dpi": {"width": 2480, "height": 3508, "scale": 2},
    "A4 Landscape @300dpi": {"width": 3508, "height": 2480, "scale": 2},
}


# ============================================================================
# FIGURE EXPORT HELPERS (publication-quality downloads)
# ============================================================================

def _fig_to_image_bytes(fig, fmt="png", scale=4, width=None, height=None):
    """Convert a Plotly figure to image bytes with high quality (requires kaleido)."""
    try:
        return pio.to_image(fig, format=fmt, scale=scale, width=width, height=height)
    except Exception:
        return None

def st_figure_downloads(fig, base_name, preset=None, width=None, height=None, scale=None):
    default_cfg = FIG_EXPORT_PRESETS["Paper: single-column (3.5in @300dpi)"]
    # Ensure the key is always a string and not a conditional expression
    if "fig_export_cfg" in st.session_state:
        export_cfg = st.session_state.get("fig_export_cfg", FIG_EXPORT_PRESETS.get("Paper: single-column (3.5in @300dpi)"))
    else:
        export_cfg = FIG_EXPORT_PRESETS.get("Paper: single-column (3.5in @300dpi)")
    cfg = FIG_EXPORT_PRESETS.get(preset if isinstance(preset, str) and preset else "Paper: single-column (3.5in @300dpi)", export_cfg)
    
    if cfg is None:
        cfg = default_cfg

    w = width or cfg["width"]
    h = height or cfg["height"]
    sc = scale or cfg.get("scale", 1)

    png = _fig_to_image_bytes(fig, fmt="png", scale=sc, width=w, height=h)
    if png is None:
        st.info(tr("Figure export needs 'kaleido'. Install: pip install -U kaleido"))
        return

    st.download_button(tr('⬇️ Download PNG (paper-ready)'), png, file_name=f'{base_name}.png', mime='image/png')

    pdf = _fig_to_image_bytes(fig, fmt="pdf", scale=1, width=w, height=h)
    if pdf is not None:
        st.download_button(tr('⬇️ Download PDF (vector)'), pdf, file_name=f'{base_name}.pdf', mime='application/pdf')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI-Assisted Datacenter Prompt Routing for ΔT-AR Mitigation",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CSS STYLING
# ============================================================================


# ============================================================================
# CONSTANTS - Research-Backed Values
# ============================================================================

# Base energy per AI prompt (Stern, 2025 - WSJ)
api_fire_count = 0

# Default Datacenters
DEFAULT_DATACENTERS: dict[str, dict[str, Any]] = {
    "Phoenix, AZ": {
        "lat": 33.4484, "lon": -112.0740,
        "country": "USA", "region": "arizona",
        "climate": "hot", "climate_detail": "hot_dry",
        "default_cooling": "evaporative",
        "description": "Desert climate with high cooling demand",
        "emoji": "🔥"
    },
    "San Francisco, CA": {
        "lat": 37.7749, "lon": -122.4194,
        "country": "USA", "region": "california",
        "climate": "moderate", "climate_detail": "moderate",
        "default_cooling": "air_economizer",
        "description": "Mild coastal climate with fog cooling",
        "emoji": "🌤️"
    },
    "Stockholm, Sweden": {
        "lat": 59.3293, "lon": 18.0686,
        "country": "Sweden", "region": "sweden",
        "climate": "cold", "climate_detail": "cold",
        "default_cooling": "free_air",
        "description": "Nordic climate enabling year-round free cooling",
        "emoji": "❄️"
    }
}

# Extended DC locations for suggestions
EXTENDED_DC_LOCATIONS: dict[str, dict[str, Any]] = {
    "Neuquén, Argentina": {"lat": -38.9500, "lon": -68.0670, "country": "Argentina", "region": "argentina", "climate": "moderate"},
    "Dublin, Ireland": {"lat": 53.3498, "lon": -6.2603, "region": "ireland", "climate": "cold"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "region": "uk", "climate": "moderate"},
    "Frankfurt, Germany": {"lat": 50.1109, "lon": 8.6821, "region": "germany", "climate": "moderate"},
    "Amsterdam, Netherlands": {"lat": 52.3676, "lon": 4.9041, "region": "netherlands", "climate": "moderate"},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "region": "singapore", "climate": "hot"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "region": "japan", "climate": "moderate"},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "region": "australia", "climate": "moderate"},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "region": "india", "climate": "hot"},
    "São Paulo, Brazil": {"lat": -23.5505, "lon": -46.6333, "region": "brazil", "climate": "moderate"},
    "Toronto, Canada": {"lat": 43.6532, "lon": -79.3832, "region": "canada", "climate": "cold"},
    "Seattle, WA": {"lat": 47.6062, "lon": -122.3321, "region": "washington", "climate": "moderate"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "region": "texas", "climate": "hot"},
    "Chicago, IL": {"lat": 41.8781, "lon": -87.6298, "region": "illinois", "climate": "moderate"},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "region": "florida", "climate": "hot"},
    "Denver, CO": {"lat": 39.7392, "lon": -104.9903, "region": "colorado", "climate": "moderate"},
    "Helsinki, Finland": {"lat": 60.1699, "lon": 24.9384, "region": "finland", "climate": "cold"},
    "Oslo, Norway": {"lat": 59.9139, "lon": 10.7522, "region": "norway", "climate": "cold"},
    "Reykjavik, Iceland": {"lat": 64.1466, "lon": -21.9426, "region": "iceland", "climate": "cold"},
}

# Carbon intensity by region (kg CO2/kWh)
# ============================================================================
# SOURCES:
# - US States: EPA eGRID 2023 (https://www.epa.gov/egrid)
# - California: CAISO/CARB 2024 (https://www.caiso.com/todaysoutlook/Pages/emissions.html)
# - Europe: EEA 2024 (https://www.eea.europa.eu/en/analysis/indicators/greenhouse-gas-emission-intensity-of-1)
# - Nordic: Ember/Statista 2024 (https://ember-climate.org/)
# - Global: IEA World Energy Outlook 2024 (https://www.iea.org/)
#
# METHODOLOGY:
# - Values represent annual average grid emission intensity
# - Unit: kg CO2 per kWh of electricity consumed
# - Includes transmission losses where applicable
# ============================================================================

# City database for user location search
CITY_DATABASE = {
    "Neuquén": (-38.9500, -68.0670),
    "Orlando": (28.5383, -81.3792),
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "San Francisco": (37.7749, -122.4194),
    "Seattle": (47.6062, -122.3321),
    "Miami": (25.7617, -80.1918),
    "Boston": (42.3601, -71.0589),
    "Denver": (39.7392, -104.9903),
    "Atlanta": (33.7490, -84.3880),
    "Dallas": (32.7767, -96.7970),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198),
    "Sydney": (-33.8688, 151.2093),
    "Mumbai": (19.0760, 72.8777),
    "Dubai": (25.2048, 55.2708),
    "Toronto": (43.6532, -79.3832),
    "São Paulo": (-23.5505, -46.6333),
    "Mexico City": (19.4326, -99.1332),
    "Stockholm": (59.3293, 18.0686),
    "Amsterdam": (52.3676, 4.9041),
    "Dublin": (53.3498, -6.2603),
    "Helsinki": (60.1699, 24.9384),
    "Oslo": (59.9139, 10.7522),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_pdf_report(results, final_rec):
    """
    Generate PDF report using FPDF.
    """
    if not FPDF_AVAILABLE:
        return None

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 15)
            self.cell(0, 10, 'AI-Assisted Datacenter Routing Report', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C', new_x="RIGHT", new_y="TOP")

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def pdf_safe_text(text):
        if not isinstance(text, str):
            return text
        return text.encode("latin-1", errors="replace").decode("latin-1")
    
    # Title Info
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, pdf_safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), border=0, align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Executive Summary
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, '1. Executive Summary', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    
    feasible = final_rec.get('feasible', True) and final_rec.get('recommended_strategy')
    headline = (f"Recommended Strategy: {final_rec['recommended_strategy']}" if feasible
                else "No feasible strategy: no requests could be served.")
    summary_text = (
        f"{headline}\n\n"
        f"Reasoning:\n" + 
        "\n".join([f"- {r.replace('✅', '[OK]').replace('⚠️', '[WARN]').replace('❌', '[X]').replace('🌡️', '').replace('🔥', '').replace('⏱️', '').replace('⚡', '').replace('Δ', 'Delta')}" for r in final_rec['reasoning']])
    )
    pdf.multi_cell(0, 7, pdf_safe_text(summary_text))
    pdf.ln(10)

    # Metrics Comparison
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, '2. Metrics Comparison', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    
    # Table Header
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(40, 10, 'Strategy', border=1, align='C', fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(35, 10, 'Energy (Wh)', border=1, align='C', fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(35, 10, 'Carbon (g)', border=1, align='C', fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(35, 10, 'Latency (ms)', border=1, align='C', fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(35, 10, 'Peak Delta-T AR (C)', border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
    
    # Table Rows
    for strategy, data in results.items():
        totals = data['totals']
        pdf.cell(40, 10, pdf_safe_text(strategy), 1)
        pdf.cell(35, 10, f"{totals['energy_wh']:.1f}", 1)
        pdf.cell(35, 10, f"{totals['carbon_g']:.1f}", 1)
        latency = totals['avg_latency_ms']
        pdf.cell(35, 10, f"{latency:.1f}" if latency == latency else "n/a", 1)  # NaN: nothing served
        pdf.cell(35, 10, f"{totals['peak_ΔT']:.3e}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)
    
    # Conclusion
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, '3. Conclusion', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    conclusion_text = (
        f"The {final_rec['recommended_strategy']} routing strategy is recommended for this scenario: among "
        f"the strategies serving the most requests within the limits, it is closest to the best value on "
        f"energy, carbon, latency and peak Delta-T AR together (equal weights). This is a simulation result, "
        f"not a measured outcome." if feasible else
        "No routing strategy could serve requests in this scenario, so no strategy is recommended. "
        + "; ".join(final_rec.get('reasoning', []))
    )
    pdf.multi_cell(0, 7, pdf_safe_text(conclusion_text))

    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1", errors="replace")
    return bytes(pdf_output)








# One single baseline table (match Table 1)
CLIMATE_BASELINES = {
    "cold":     {"temperature": 5.0,  "humidity": 70.0, "wind_speed": 6.0, "pressure": 1013.25, "solar_radiation": 150.0},
    "moderate": {"temperature": 18.0, "humidity": 55.0, "wind_speed": 4.0, "pressure": 1013.25, "solar_radiation": 200.0},
    "hot":      {"temperature": 35.0, "humidity": 30.0, "wind_speed": 3.0, "pressure": 1013.25, "solar_radiation": 350.0},
}

api_fire_count = 0  # prevent NameError

import requests
import requests_cache
from typing import Any, cast

def fetch_weather_data(lat: float, lon: float, location_name: str = "Location", climate_hint: str | None = None, *, force_refresh: bool = False) -> dict[str, Any]:
    global api_fire_count

    lat = float(lat)
    lon = float(lon)

    # ========== 1) TRY ACTUAL API CALL FIRST ==========
    try:
        api_fire_count += 1
        logging.info(
            f"API call #{api_fire_count} - Fetching weather for {location_name} at ({lat:.4f}, {lon:.4f})"
        )

        cache_session = requests.Session() if force_refresh else requests_cache.CachedSession(".cache", expire_after=300)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)

        # openmeteo_requests.Client type stubs expect niquests.Session
        session = cast(Any, retry_session)
        openmeteo = openmeteo_requests.Client(session=session)  # type: ignore[arg-type]

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "surface_pressure",
                "shortwave_radiation",
            ],
            "timezone": "auto",
            "wind_speed_unit": "ms",
        }

        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        current = response.Current()
        if current is None:
            raise ValueError("The weather API returned no current conditions.")

        def _safe_current_value(current_data, idx: int, default: float) -> float:
            var = current_data.Variables(idx) if current_data else None
            if var is None or not np.isfinite(var.Value()):
                # Never label a default value as an API-sourced estimate.
                raise ValueError(f"The weather API omitted current variable {idx}.")
            return float(var.Value())

        return {
            "temperature": _safe_current_value(current, 0, 20.0),
            "humidity": _safe_current_value(current, 1, 50.0),
            "wind_speed": _safe_current_value(current, 2, 5.0),
            "pressure": _safe_current_value(current, 3, 1013.25),
            "solar_radiation": _safe_current_value(current, 4, 200.0),
            "source": "Current weather estimate (Open-Meteo API)",
            "observed_at": datetime.fromtimestamp(current.Time(), tz=timezone.utc).isoformat() if current else None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_url": "https://open-meteo.com/en/docs",
            "classification": "API-fetched model estimate",
            "success": True,
        }

    except Exception as e:
        logging.warning(f"API call failed for {location_name}: {str(e)}")

    # ========== 2) FALLBACK ==========
    climate_key = str(climate_hint).lower() if climate_hint else "moderate"
    if climate_key.startswith("hot"):
        climate_key = "hot"
    elif climate_key.startswith("cold"):
        climate_key = "cold"
    elif climate_key.startswith("mod"):
        climate_key = "moderate"

    base = CLIMATE_BASELINES.get(climate_key, CLIMATE_BASELINES["moderate"])
    return {
        **base,
        "source": f"Estimated baseline ({climate_key}, API unavailable)",
        "observed_at": None,
        "source_url": None,
        "classification": "Assumed climate baseline",
        "success": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

def classify_climate(temperature, humidity, wind_speed=None):
    """
    Improved datacenter climate classifier using temperature, humidity, and wind.
    Returns:
        climate, climate_detail, recommended_cooling, emoji
    """

    # ----- PRIMARY CLIMATE BANDS -----
    if temperature < 10:
        climate = "cold"
        climate_detail = "very_cold"
        recommended_cooling = "free_air"

    elif 10 <= temperature < 20:
        climate = "cool"
        climate_detail = "cool_mild"
        # Windy cool regions do better with air economizers
        if wind_speed is not None and wind_speed >= 8:
            recommended_cooling = "air_economizer"
        else:
            recommended_cooling = "free_air"

    elif 20 <= temperature < 28:
        climate = "moderate"
        climate_detail = "temperate"

        # In this band, humidity decides efficiency:
        if humidity < 60:
            # Dry moderate → best for economizers
            recommended_cooling = "air_economizer"
        else:
            # Humid moderate → free air isn't enough
            recommended_cooling = "mechanical_chiller"

    else:
        # ----- HOT REGIONS -----
        climate = "hot"

        if humidity < 40:
            climate_detail = "hot_dry"
            recommended_cooling = "evaporative"

        elif 40 <= humidity < 70:
            climate_detail = "hot_moderate_humidity"
            # Economizers work at night, evaporative in day → pick evaporative as general case
            recommended_cooling = "evaporative"

        else:
            climate_detail = "hot_humid"
            # Very humid → evaporative fails → liquid cooling
            recommended_cooling = "liquid_cooling"

    # ----- EMOJI -----
    emoji_map = {
        "cold": "❄️",
        "cool": "🧊",
        "moderate": "🌤️",
        "hot": "🔥"
    }

    return {
        "climate": climate,
        "climate_detail": climate_detail,
        "recommended_cooling": recommended_cooling,
        "emoji": emoji_map.get(climate, "🌍")
    }

def collect_historical_training_data(start_year=2021, end_year=2024, progress_cb=None,
                                     start_date=None, end_date=None):
    """
    Fetch 4 years of hourly weather for all datacenters
    using Open-Meteo Historical Forecast API.
    
    Returns a SINGLE combined DataFrame:
    columns = [location, datetime, temperature, humidity, wind_speed, pressure, solar_radiation, cooling_type]
    """
    # Setup Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = cast(Any, retry_session))

    all_records = []

    # An explicit date range (the selectable training period) takes precedence over whole years.
    start_date = str(start_date) if start_date else f"{start_year}-01-01"
    end_date = str(end_date) if end_date else f"{end_year}-12-31"

    logging.info(f"📥 Fetching hourly historical weather from {start_date} to {end_date}")

    # Combine datacenters
    datacenters = {}
    for name, info in DEFAULT_DATACENTERS.items():
        datacenters[name] = info
    for name, info in EXTENDED_DC_LOCATIONS.items():
        datacenters[name] = info

    for dc_name, dc_info in datacenters.items():
        # Use correct keys for latitude/longitude
        lat = dc_info.get("lat", dc_info.get("latitude"))
        lon = dc_info.get("lon", dc_info.get("longitude"))

        if lat is None or lon is None:
            logging.info(f"⚠️ Skipping {dc_name}: Missing coordinates")
            continue

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
            "timezone": "auto",
            "wind_speed_unit": "ms"
        }

        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            
            # Check if hourly data is available
            if hourly is None:
                logging.info(f"  ✗ {dc_name}: No hourly data returned")
                continue

            # Process hourly data (order matches API request)
            temp_var = hourly.Variables(0)
            humidity_var = hourly.Variables(1)
            wind_var = hourly.Variables(2)
            pressure_var = hourly.Variables(3)
            solar_var = hourly.Variables(4)
            
            if temp_var is None or humidity_var is None or wind_var is None:
                logging.info(f"  ✗ {dc_name}: Missing expected hourly variables")
                continue
            
            temp = temp_var.ValuesAsNumpy()
            humidity = humidity_var.ValuesAsNumpy()
            wind = wind_var.ValuesAsNumpy()
            pressure = pressure_var.ValuesAsNumpy() if pressure_var else np.full(len(temp), 1013.25)
            solar = solar_var.ValuesAsNumpy() if solar_var else np.full(len(temp), 200.0)

            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )

            n = len(temp)
            logging.info(f"  ✔ {dc_name}: {n} hourly rows")

            for i in range(n):
                # Classify climate for each hour
                climate_info = classify_climate(temp[i], humidity[i], wind[i])
                cooling_type = climate_info["recommended_cooling"]

                all_records.append([
                    dc_name,
                    timestamps[i],
                    float(temp[i]),
                    float(humidity[i]),
                    float(wind[i]),
                    float(pressure[i]) if not np.isnan(pressure[i]) else 1013.25,
                    float(solar[i]) if not np.isnan(solar[i]) else 200.0,
                    cooling_type
                ])

        except Exception as e:
            logging.info(f"❌ Error fetching {dc_name}: {e}")

    df = pd.DataFrame(all_records, columns=[
        "location",
        "datetime",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "solar_radiation",
        "cooling_type"
    ])

    logging.info(f"\n✅ Finished! Total samples collected: {len(df):,}")
    return df

    









def get_optimal_cooling_for_climate(climate_detail):
    """
    Recommend optimal cooling technology based on climate.
    Based on Alkrush et al. (2024) effectiveness matrix.
    """
    recommendations = {
        'cold': ('free_air', "Free air cooling is optimal for cold climates with minimal mechanical overhead"),
        'moderate': ('air_economizer', "Air-side economizers work efficiently in moderate temperatures"),
        'hot_dry': ('evaporative', "Evaporative cooling is highly effective in hot, dry climates"),
        'hot_humid': ('liquid_cooling', "Liquid cooling recommended for hot, humid climates where evaporative is less effective")
    }
    return recommendations.get(climate_detail, ('air_economizer', "Default recommendation for mixed conditions"))

def collect_training_data(days: int = 7) -> pd.DataFrame:  # kept for compatibility
    records = []

    # 1) Default datacenters
    for dc_name, dc_info in DEFAULT_DATACENTERS.items():
        w = fetch_weather_data(dc_info["lat"], dc_info["lon"], dc_name)
        records.append({
            "datacenter": dc_name,
            "temperature": w["temperature"],
            "humidity": w["humidity"],
            "wind_speed": w["wind_speed"],
            "pressure": w.get("pressure", 1013.25),
            "solar_radiation": w.get("solar_radiation", 200.0),
            "cooling_type": dc_info.get("default_cooling", "mechanical_chiller"),
        })

    # 2) Extended datacenters
    if EXTENDED_DC_LOCATIONS:
        with st.spinner(tr(f'🌐 Loading {len(EXTENDED_DC_LOCATIONS)} extended datacenters...')):
            for dc_name, loc in EXTENDED_DC_LOCATIONS.items():
                w = fetch_weather_data(loc["lat"], loc["lon"], dc_name)
                climate_info = classify_climate(w["temperature"], w["humidity"], w.get("wind_speed", 5.0))
                cooling_key, _ = get_optimal_cooling_for_climate(climate_info["climate_detail"])

                records.append({
                    "datacenter": dc_name,
                    "temperature": w["temperature"],
                    "humidity": w["humidity"],
                    "wind_speed": w["wind_speed"],
                    "pressure": w.get("pressure", 1013.25),
                    "solar_radiation": w.get("solar_radiation", 200.0),
                    "cooling_type": cooling_key,
                })

    return pd.DataFrame(records)


def apply_capacity_limit(requests, energy_per_request_wh, max_capacity_mw, time_window_hours=1.0):
    """
    Limit requests based on DC power capacity over a time window.
    capacity (MW) -> energy capacity over window (Wh) = MW * 1e6 * hours
    """
    if requests <= 0:
        return 0

    max_capacity_wh = max_capacity_mw * 1e6 * float(time_window_hours)
    total_wh = requests * energy_per_request_wh

    if total_wh <= max_capacity_wh:
        return requests

    max_requests = int(max_capacity_wh / max(energy_per_request_wh, 1e-9))
    return max(0, max_requests)



# ============================================================================
# FEATURE ENGINEERING FOR AI MODELS
# ============================================================================

def engineer_features(df):
    """
    Take raw weather + DC info and add engineered features for ML:
    - pue (from cooling type)
    - base_capacity_mw (simple default for now)
    - urban_density (simple default)
    - carbon_intensity (simple default)
    - utilization (simulated load)
    - energy_wh (target)
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe passed to engineer_features().")

    # 1. PUE based on cooling type using existing COOLING_SYSTEMS
    def get_pue_from_cooling(cooling_type):
        cooling = COOLING_SYSTEMS.get(cooling_type)
        if cooling is None:
            # Fallback to mechanical_chiller if unknown
            cooling = COOLING_SYSTEMS["mechanical_chiller"]
        return cooling["pue"]

    df["pue"] = df["cooling_type"].apply(get_pue_from_cooling)

    # 2. Simple datacenter-level features (you can refine later per-DC)
    #    For now, use reasonable defaults; these can be made DC-specific later.
    df["base_capacity_mw"] = 50.0       # assume 50 MW DC
    df["urban_density"] = 0.7           # assume relatively dense urban siting
    df["carbon_intensity"] = 0.300      # kgCO2/kWh, generic grid mix

    # 3. Simulate utilization (0.3–0.85). In production, this would be real load.
    df["utilization"] = np.random.uniform(0.3, 0.85, len(df))

    # 4. Target: Energy per prompt (Wh), physics-informed
    df["energy_wh"] = ENERGY_PER_PROMPT_BASE * df["pue"] * (
        1 + df["utilization"] * 0.1
    )

    return df

# ============================================================================
# AI MODEL SUITE
# ============================================================================

class AIModelSuite:
    """
    Suite of AI models for energy prediction.
    Models: MLR, ANN, Bayesian-optimized ANN

    Training data:
    - Prefer real weather (Open-Meteo) from collect_training_data()
    - Fall back to synthetic physics-based data if needed
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        # Historical weather window for training; set before train_all to use another period.
        self.training_period = ("2021-01-01", "2024-12-31")
        self.metrics = {}
        self.timing = {}
        self.feature_names = list(FEATURE_NAMES)
        self.is_trained = False
        self.used_real_weather = False
        self.training_source = "unknown"   # "real" or "synthetic"
        self.training_samples = 0
        self.best_model_name = None
        self.best_model = None

    def generate_training_data(self, n_samples=5000, days=7, use_real_weather=True, progress_cb=None): #Intentional unsed 'days' param for API compatibility
        """
        Build training dataset.

        Priority:
        1) Use real weather-based records from collect_training_data()
           and compute energy via physics model.
        2) If unavailable / error / empty, fall back to synthetic data
           (uniform sampling + physics model).
        """
        data = []
        cooling_types = COOLING_TYPES
        # Every record gets a cooling technology drawn uniformly at random (fixed seed), so all
        # supported cooling choices are trained across the full range of weather. Assigning only
        # the climate-recommended technology left most weather/cooling combinations unseen.
        cooling_rng = np.random.default_rng(20240101)

# --- 1. Try real historical weather-based training data ---
        if use_real_weather:
            try:
                # 4 years of hourly data for all datacenters
                weather_df = collect_historical_training_data(start_date=self.training_period[0],
                                                              end_date=self.training_period[1])

                if weather_df is not None and not weather_df.empty:
                    for _, row in weather_df.iterrows():
                        temp = float(row.get("temperature", 20.0))
                        humidity = float(row.get("humidity", 50.0))
                        wind = float(row.get("wind_speed", 5.0))
                        pressure = float(row.get("pressure", 1013.25))
                        solar = float(row.get("solar_radiation", 200.0))

                        cooling_idx = int(cooling_rng.integers(0, len(cooling_types)))
                        cooling_str = cooling_types[cooling_idx]

                        energy = surrogate_energy_target(temp, humidity, wind, pressure, solar, cooling_str)
                        energy += np.random.normal(0, 0.003)

                        # Calculate delta_t for complete training data
                        heat_kwh = energy / 1000.0
                        # Option 1: Use default area (recommended for training data)
                        area_km2 = 1.0  # Default area for training data generation
                        delta_t = calculate_ΔT_contribution(heat_kwh, area_km2=area_km2, wind_speed=wind)

                        # Option 2: Get from row data if available
                        area_km2 = float(row.get("area_km2", 1.0))
                        delta_t = calculate_ΔT_contribution(heat_kwh, area_km2=area_km2, wind_speed=wind)
                        
                        data.append([temp, humidity, wind, pressure, solar, cooling_idx, energy, delta_t])
                                            
                    # Optional: track metadata in the suite
                    self.used_real_weather = True
                    self.training_source = "real"
                    self.training_samples = len(data)

                    logging.info(f"✅ Training on historical weather data (2021-2024): {len(data)} samples")
                else:
                    logging.info("⚠️ collect_historical_training_data() returned empty – using synthetic data.")
            except Exception as e:
                logging.info(f"❌ Historical weather training data error: {e}")
                logging.info("   Falling back to synthetic training data.")

        # --- 2. Fallback: synthetic physics-based data (original behavior) ---
        if not data:
            np.random.seed(42)
            for _ in range(n_samples):
                temp = np.random.uniform(0, 45)
                humidity = np.random.uniform(20, 95)
                wind = np.random.uniform(0.5, 15)
                pressure = np.random.uniform(980, 1040)  # hPa range
                solar = np.random.uniform(0, 1000)  # W/m² (0 at night, ~1000 peak)
                cooling_idx = np.random.randint(0, len(cooling_types))
                cooling_str = cooling_types[cooling_idx]
                energy = surrogate_energy_target(temp, humidity, wind, pressure, solar, cooling_str)
                energy += np.random.normal(0, 0.005)
                heat_kwh = energy / 1000.0
                delta_t = calculate_ΔT_contribution(heat_kwh, area_km2=1.0, wind_speed=wind)
                data.append([temp, humidity, wind, pressure, solar, cooling_idx, energy, delta_t])
                    
            # Set metadata for synthetic data
            self.used_real_weather = False
            self.training_source = "synthetic"
            self.training_samples = len(data)
            logging.info(f"✅ Training on synthetic physics-based data: {len(data)} samples")

        df = pd.DataFrame(
            data,
            columns=['temperature', 'humidity', 'wind_speed', 'pressure', 'solar_radiation', 'cooling_type', 'energy', 'delta_t']
        )
        
        # Remove any NaN values to prevent training errors
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        return df

    def train_all(self, train_selected='all', use_real_weather=True, days=7, n_samples=5000,  progress_cb=None):
        """
        Train all or selected models.

        Parameters:
        - train_selected: 'all', 'mlr', 'ann', 'bayesian'
        - use_real_weather: if True, try real Open-Meteo-based data first
        - days: number of days of historical weather to pull per datacenter
        """
        df = self.generate_training_data(
            n_samples=n_samples,
            days=days,
            use_real_weather=use_real_weather
        )

        # ========== DATA SPLITTING (70/15/15) ==========
        X = encode_features(df[['temperature', 'humidity', 'wind_speed', 'pressure', 'solar_radiation', 'cooling_type']].values)
        y = df['energy'].values

        # Split into train+val (85%) and test (15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        # Split train+val into train (70%) and validation (15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, random_state=42
        )

        # Standardize features (fit ONLY on train, transform val and test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # Verify split sizes
        logging.info(f"✓ Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        logging.info(f"✓ Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        logging.info(f"✓ Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

        results: dict[str, Any] = {}

        # ========== 1. MULTIPLE LINEAR REGRESSION ==========
        if train_selected in ['all', 'mlr']:
            mlr = LinearRegression()
            start_time = time.perf_counter()
            mlr.fit(X_train_scaled, y_train)
            train_time = time.perf_counter() - start_time
            
            # Evaluate on TEST set only
            mlr_pred = mlr.predict(X_test_scaled)
            self.models['MLR'] = mlr
            
            results['MLR'] = {
                'r2': r2_score(y_test, mlr_pred),
                'mae': mean_absolute_error(y_test, mlr_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, mlr_pred)),
                'train_time': train_time,
                'predictions': mlr_pred,
                'y_test': y_test,
                'coefficients': dict(zip(MODEL_INPUT_NAMES, mlr.coef_))
            }
            logging.info(f"[MLR] R²: {results['MLR']['r2']:.4f}")

        # ========== 2. ARTIFICIAL NEURAL NETWORK ==========
        if train_selected in ['all', 'ann']:
            ann = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,  
                validation_fraction=0.1, 
                n_iter_no_change=20
            )
            
            start_time = time.perf_counter()
            ann.fit(X_train_scaled, y_train)
            train_time = time.perf_counter() - start_time
            
            # Evaluate on TEST set only
            ann_pred = ann.predict(X_test_scaled)
            self.models['ANN'] = ann
            
            results['ANN'] = {
                'r2': r2_score(y_test, ann_pred),
                'mae': mean_absolute_error(y_test, ann_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ann_pred)),
                'train_time': train_time,
                'predictions': ann_pred,
                'y_test': y_test,
                'architecture': '64→32→16'
            }
            logging.info(f"[ANN] R²: {results['ANN']['r2']:.4f}")

        # ========== 3. BAYESIAN OPTIMIZATION ==========
        if train_selected in ['all', 'bayesian'] and BAYES_AVAILABLE:
            try:
                def ann_objective(hidden1, hidden2, alpha):
                    """Optimize using VALIDATION set (not test!)"""
                    h1, h2 = int(hidden1), int(hidden2)
                    model = MLPRegressor(
                        hidden_layer_sizes=(h1, h2),
                        alpha=alpha,
                        max_iter=300,
                        random_state=42,
                        early_stopping=False
                    )
                    model.fit(X_train_scaled, y_train)
                    
                    # USE VALIDATION SET for hyperparameter selection! ✓
                    val_pred = model.predict(X_val_scaled)
                    return r2_score(y_val, val_pred)

                optimizer = BayesianOptimization(
                    f=ann_objective,
                    pbounds={
                        'hidden1': (16, 128),
                        'hidden2': (8, 64),
                        'alpha': (0.0001, 0.1)
                    },
                    random_state=42,
                    verbose=0
                )
                optimizer.maximize(init_points=3, n_iter=7)
                # Store BO convergence history (validation R² per iteration)
                bo_history = []
                for i, r in enumerate(getattr(optimizer, "res", []), start=1):
                    row = {"iteration": i, "val_r2": float(r.get("target", float("nan")))}
                    params = r.get("params", {}) or {}
                    for k, v in params.items():
                        row[k] = float(v)
                    bo_history.append(row)

                best_val_r2 = float(getattr(optimizer, "max", {}).get("target", float("nan"))) if getattr(optimizer, "max", None) else float("nan")

                best = optimizer.max['params'] if optimizer.max else {
                    'hidden1': 64, 'hidden2': 32, 'alpha': 0.01
                }

                # Train final model with best hyperparameters
                bayes_ann = MLPRegressor(
                    hidden_layer_sizes=(int(best['hidden1']), int(best['hidden2'])),
                    alpha=best['alpha'],
                    max_iter=500,
                    random_state=42,
                    early_stopping=False
                )
                # Time training
                start_time = time.perf_counter()
                bayes_ann.fit(X_train_scaled, y_train)
                train_time = time.perf_counter() - start_time
                
                # Evaluate on TEST set only (unseen during optimization!)
                bayes_pred = bayes_ann.predict(X_test_scaled)
                self.models['Bayesian'] = bayes_ann
                
                results['Bayesian'] = {
                    'r2': r2_score(y_test, bayes_pred),
                    'mae': mean_absolute_error(y_test, bayes_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, bayes_pred)),
                    'train_time': train_time,
                    'bo_history': bo_history,
                    'bo_best_val_r2': best_val_r2,
                    'predictions': bayes_pred,
                    'y_test': y_test,
                    'best_params': best
                }
                logging.info(f"[Bayesian] R²: {results['Bayesian']['r2']:.4f}")
                logging.info(f"[Bayesian] Best params: {best}")
                
            except Exception as e:
                logging.info(f"[Bayesian] Error: {str(e)}")
                results['Bayesian'] = {'error': str(e)}

        # Feature importance (using TEST set for evaluation, not training)
        if 'MLR' in self.models:
            mlr = self.models['MLR']
            importances = dict(zip(MODEL_INPUT_NAMES, mlr.coef_))
            results.setdefault("feature_importance", {})
            results["feature_importance"]["MLR"] = importances   # from MLR
# ...
            logging.info(f"Feature importances (MLR): {importances}")        # Feature importance (using TEST set for evaluation, not training)
        # Feature importance (using TEST set for evaluation, not training)
        if 'ANN' in self.models:
            try:
                perm_importance = permutation_importance(
                    self.models['ANN'], X_test_scaled, y_test,
                    n_repeats=10, random_state=42
                )
                # Fix: Access importances_mean correctly from Bunch object
                importance_dict = {
                    name: float(imp) for name, imp in 
                    zip(MODEL_INPUT_NAMES, perm_importance['importances_mean'])
                }
                results.setdefault("feature_importance", {})["ANN"] = importance_dict
            except Exception as e:
                logging.info(f"[Feature Importance] Error: {str(e)}")

# ========== COMPUTE TRAIN/VAL/TEST METRICS FOR ALL MODELS ==========
        for model_name in ['MLR', 'ANN', 'Bayesian']:  # ← Explicit list of models only
            if model_name not in self.models:
                continue  # Skip if model wasn't trained
                
            if model_name in results and isinstance(results[model_name], dict):
                if 'error' in results[model_name]:
                    continue  # Skip models that had errors
                    
                model = self.models[model_name]
                
                # Training set metrics
                train_pred = model.predict(X_train_scaled)
                results[model_name]['train_r2'] = r2_score(y_train, train_pred)
                results[model_name]['train_mae'] = mean_absolute_error(y_train, train_pred)
                results[model_name]['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
                
                # Validation set metrics
                val_pred = model.predict(X_val_scaled)
                results[model_name]['val_r2'] = r2_score(y_val, val_pred)
                results[model_name]['val_mae'] = mean_absolute_error(y_val, val_pred)
                results[model_name]['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                
                # Keep existing test metrics (already computed)
                # results[model_name]['r2'] = test R²
                # results[model_name]['mae'] = test MAE
                # results[model_name]['rmse'] = test RMSE
                
                # Compute overfitting indicator
                r2_gap = results[model_name]['train_r2'] - results[model_name]['r2']
                if r2_gap > 0.05:
                    results[model_name]['status'] = 'Overfitting'
                elif results[model_name]['r2'] < 0.7:
                    results[model_name]['status'] = 'Underfitting'
                else:
                    results[model_name]['status'] = 'Good Fit'
                
                # For ANN/Bayesian: Extract convergence traces (if available)
                if model_name in ["ANN", "Bayesian"]:
                    # loss_curve_ exists after fit; still guard just in case
                    loss = getattr(model, "loss_curve_", None)
                    results[model_name]["loss_curve"] = list(loss) if loss is not None else []

                    # validation_scores_ exists only with early_stopping=True in sklearn,
                    # but in some setups it can exist and still be None -> must guard
                    val_scores = getattr(model, "validation_scores_", None)
                    try:
                        results[model_name]["val_score_curve"] = list(val_scores) if val_scores is not None else []
                    except TypeError:
                        results[model_name]["val_score_curve"] = []


                    if hasattr(model, 'n_iter_'):
                        results[model_name]['n_iter'] = int(model.n_iter_)
               
                logging.info(f"[{model_name}] Train R²: {results[model_name]['train_r2']:.4f}, "
                      f"Val R²: {results[model_name]['val_r2']:.4f}, "
                      f"Test R²: {results[model_name]['r2']:.4f} - "
                      f"{results[model_name]['status']}")

        # ========== RETURN RESULTS ==========
        self.is_trained = True
        # Select by validation R²; reserve test scores for reporting.
        if results:
            # Filter out error results
            valid_results = {k: v for k, v in results.items() 
                           if 'r2' in v and 'error' not in v}
            
            if valid_results:
                self.best_model_name = max(
                    valid_results.keys(), 
                    key=lambda x: valid_results[x]['val_r2']
                )
                self.best_model = self.models[self.best_model_name]
                # Separate check of cooling comparisons (what changing cooling relies on).
                results["cooling_validation"] = {name: self.validate_cooling_comparisons(name)
                                                 for name in valid_results}

        #         # Prepare ΔT target
        #         y_dt = df['delta_t'].values

        #         # same splits as energy (reuse X_train, X_val, X_test indices if possible)
        #         # simplest: redo split with same random_state (works because df order fixed)
        #         X = df[['temperature','humidity','wind_speed','pressure','solar_radiation','cooling_type']].values
        #         X_train_val, X_test, y_train_val_dt, y_test_dt = train_test_split(X, y_dt, test_size=0.15, random_state=42)
        #         X_train, X_val, y_train_dt, y_val_dt = train_test_split(X_train_val, y_train_val_dt, test_size=0.1765, random_state=42)

        #         # use the same feature scaler as energy
        #         scaler_main = self.scalers['main']
        #         X_train_scaled = scaler_main.transform(X_train)
        #         X_val_scaled   = scaler_main.transform(X_val)
        #         X_test_scaled  = scaler_main.transform(X_test)

        #         # energy predictions become an extra feature
        #         best_energy = self.models[self.best_model_name]
        #         Ehat_train = best_energy.predict(X_train_scaled).reshape(-1,1)
        #         Ehat_val   = best_energy.predict(X_val_scaled).reshape(-1,1)
        #         Ehat_test  = best_energy.predict(X_test_scaled).reshape(-1,1)

        #         Xdt_train = np.hstack([X_train_scaled, Ehat_train])
        #         Xdt_val   = np.hstack([X_val_scaled,   Ehat_val])
        #         Xdt_test  = np.hstack([X_test_scaled,  Ehat_test])

        #         # scale ΔT features separately (because Ehat is in Wh scale)
        #         scaler_dt = StandardScaler()
        #         Xdt_train_s = scaler_dt.fit_transform(Xdt_train)
        #         Xdt_val_s   = scaler_dt.transform(Xdt_val)  
        #         Xdt_test_s  = scaler_dt.transform(Xdt_test)
        #         self.scalers['dt'] = scaler_dt

        #         # clone the best model type and train for ΔT
        #         dt_model = clone(best_energy)
        #         dt_model.fit(Xdt_train_s, y_train_dt)
        #         self.models['DT'] = dt_model
                
        #         # Evaluate ΔT model
        #         dt_val_pred = dt_model.predict(Xdt_val_s)
        #         dt_test_pred = dt_model.predict(Xdt_test_s)
                
        #         results['DT'] = {
        #             'r2': r2_score(y_test_dt, dt_test_pred),
        #             'mae': mean_absolute_error(y_test_dt, dt_test_pred),
        #             'rmse': np.sqrt(mean_squared_error(y_test_dt, dt_test_pred)),
        #             'val_r2': r2_score(y_val_dt, dt_val_pred),
        #             'val_mae': mean_absolute_error(y_val_dt, dt_val_pred),
        #         }
        #         logging.info(f"[ΔT Model] Test R²: {results['DT']['r2']:.4f}, MAE: {results['DT']['mae']:.6f}°C")
                
        #         logging.info(f"\n{'='*60}")
        #         logging.info(f"✓ SELECTED BEST MODEL: {self.best_model_name}")
        #         logging.info(f"  R² = {valid_results[self.best_model_name]['r2']:.4f}")
        #         logging.info(f"  MAE = {valid_results[self.best_model_name]['mae']:.4f} Wh")
        #         logging.info(f"  RMSE = {valid_results[self.best_model_name]['rmse']:.4f} Wh")
        #         logging.info(f"{'='*60}\n")

        # Store metrics for later access
        self.metrics = results
        return results

    def validate_cooling_comparisons(self, model_name=None, n_weather=400, seed=7):
        """Check the use the dashboard makes of the model: comparing cooling technologies under
        the same weather. For sampled weather conditions, every pair of technologies is compared
        by the model and by the reference formula (noise-free training target).

        Reports how often the model orders the pair the same way as the formula, and the error in
        the predicted difference between the two technologies. Overall R² does not measure this.
        """
        name = model_name or self.best_model_name
        model, scaler = self.models[name], self.scalers['main']
        rng = np.random.default_rng(seed)
        weather = np.column_stack([rng.uniform(0, 45, n_weather), rng.uniform(20, 95, n_weather),
                                   rng.uniform(0.5, 15, n_weather), rng.uniform(980, 1040, n_weather),
                                   rng.uniform(0, 1000, n_weather)])
        predicted, reference = [], []
        for index, cooling in enumerate(COOLING_TYPES):
            rows = np.column_stack([weather, np.full(n_weather, index)])
            predicted.append(model.predict(scaler.transform(encode_features(rows))))
            reference.append([surrogate_energy_target(*w, cooling) for w in weather])
        predicted, reference = np.array(predicted), np.array(reference)
        agree, deltas = [], []
        for a in range(len(COOLING_TYPES)):
            for b in range(a + 1, len(COOLING_TYPES)):
                true_delta, model_delta = reference[a] - reference[b], predicted[a] - predicted[b]
                meaningful = np.abs(true_delta) > 1e-3  # ignore pairs the formula treats as equal
                agree.append(np.sign(true_delta[meaningful]) == np.sign(model_delta[meaningful]))
                deltas.append(np.abs(model_delta - true_delta))
        agreement = float(np.mean(np.concatenate(agree))) if agree else 1.0
        return {"model": name, "weather_samples": n_weather, "pairs": len(deltas),
                "order_agreement": agreement,
                "delta_mae_wh": float(np.mean(np.concatenate(deltas))),
                "delta_max_error_wh": float(np.max(np.concatenate(deltas)))}

    def predict(self, features, model_name=None):
        """
        Predict energy using specified or best model.
        Features: [temperature, humidity, wind_speed, pressure, solar_radiation, cooling_type_idx]
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all() first.")

        if model_name is None:
            model_name = self.get_best_model()[0]

        if model_name not in self.models:
            model_name = list(self.models.keys())[0]

        X = encode_features(np.array(features).reshape(1, -1))
        X_scaled = self.scalers['main'].transform(X)

        return max(1e-6, float(self.models[model_name].predict(X_scaled)[0]))
    
    def predict_dt(self, features):
        raise NotImplementedError("ΔT model disabled. Use physics-based ΔT (calculate_ΔT_contribution).")

    
    def get_best_model(self):
        """Return the best ENERGY model (exclude DT + feature_importance)."""
        if not self.metrics:
            return ("MLR", 0.0)

        skip = {"feature_importance", "DT", "cooling_validation"}
        best_name, best_r2 = None, -np.inf

        for name, metrics in self.metrics.items():
            if name in skip:
                continue
            if isinstance(metrics, dict) and "r2" in metrics and "error" not in metrics:
                if best_name is None or metrics.get("val_r2", metrics["r2"]) > self.metrics[best_name].get("val_r2", self.metrics[best_name]["r2"]):
                    best_r2 = metrics["r2"]
                    best_name = name

        return (best_name or "MLR", best_r2)

# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

# ============================================================================
# SIMULATION ENGINE
# ============================================================================
def validate_coordinates(lat, lon, label="Location"):
    """Reject impossible coordinates before they reach distance, latency and map calculations."""
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and -90 <= lat <= 90):
        raise ValueError(f"{label}: latitude must be between -90 and 90 degrees (got {lat}).")
    if not (math.isfinite(lon) and -180 <= lon <= 180):
        raise ValueError(f"{label}: longitude must be between -180 and 180 degrees (got {lon}).")


def run_simulation(datacenters, weather_data, user_location, num_requests,
                   cooling_selections, energy_multiplier=1.0, ai_models=None,
                   latency_threshold=100, max_dc_capacity_mw=100, max_site_share=1.0,
                   carbon_intensities=None, carbon_time=None):
    """
    Run complete simulation for all routing strategies.
    ``max_site_share`` caps the fraction of the batch any one datacenter may take.
    ``carbon_intensities`` (site -> kg CO2e/kWh) are frozen carbon inputs recorded when the
    simulation was first run; sites missing from it are evaluated at ``carbon_time`` (UTC), so a
    rerun or later analysis reproduces the same carbon results instead of following the clock.
    Returns comprehensive metrics for comparison.

    Fixes included:
    - Use true load_fraction in latency (no hardcoded 0.7)
    - Apply energy_multiplier consistently (ANN + physics)
    - Compute cooling_idx once
    - Compute ΔT once (no duplicate fallback)
    - Make Wh/kWh conversions explicit and consistent
    """
    if num_requests <= 0:
        raise ValueError("Request volume must be positive.")
    validate_coordinates(*user_location, label="Request origin")
    for dc_name, dc_info in datacenters.items():
        validate_coordinates(dc_info["lat"], dc_info["lon"], label=dc_name)
    user_lat, user_lon = user_location
    # One latency rule for every strategy: a site takes no more requests than it can serve
    # within the limit. Requests no site can serve in time are reported as unserved.
    latency_caps = latency_caps_for(datacenters, user_location, num_requests, latency_threshold)
    common = dict(max_site_share=max_site_share, latency_caps=latency_caps)
    site_energy = energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier)
    limits = {"latency": latency_caps,
              "power": {dc: int(max_dc_capacity_mw * 1e6 / e) for dc, e in site_energy.items()},
              "share": int(num_requests * max_site_share + 1e-9)}
    frozen = dict(carbon_intensities or {})
    missing = {dc: info for dc, info in datacenters.items() if dc not in frozen}
    if missing:
        snapshot = carbon_snapshot(missing, carbon_time)
        frozen.update({dc: site["intensity_kg_per_kwh"] for dc, site in snapshot["sites"].items()})

    strategies = {
        'Random': route_random(datacenters, num_requests, weather_data, cooling_selections, ai_models, max_dc_capacity_mw, energy_multiplier,
                                         **common),
        'Energy-Only': route_energy_only(datacenters, weather_data, cooling_selections,
                                         num_requests, ai_models, max_dc_capacity_mw, energy_multiplier,
                                         **common),
        'Heat-Only': route_heat_only(datacenters, weather_data, cooling_selections,
                                     num_requests, ai_models, max_dc_capacity_mw, energy_multiplier,
                                         **common),
        'ΔT-AR': route_ΔT_aware(datacenters, weather_data, cooling_selections,
                                num_requests, ai_models, max_dc_capacity_mw, energy_multiplier,
                                         **common),
        'Multi-Objective': route_multi_objective(datacenters, weather_data, cooling_selections,
                                                 user_location, num_requests, ai_models,
                                                 latency_threshold, max_dc_capacity_mw, energy_multiplier,
                                                 max_site_share=max_site_share, carbon_intensities=frozen)
    }

    results = {}

    for strategy_name, distribution in strategies.items():
        strategy_results = {
            'distribution': distribution,
            'datacenters': {},
            'limits': limits,
        }

        served_requests = sum(distribution.values())
        total_energy_wh = 0.0
        total_carbon_kg = 0.0
        weighted_latency_ms = 0.0
        total_heat_wh = 0.0
        heat_values = []
        ΔT_values = []

        for dc_name, requests in distribution.items():
            if requests == 0:
                continue

            dc_info = datacenters[dc_name]
            weather = weather_data.get(dc_name, {})
            cooling = cooling_selections.get(dc_name, dc_info.get('default_cooling', 'air_economizer'))

            # Weather defaults
            temp     = weather.get('temperature', 20)
            humidity = weather.get('humidity', 50)
            wind     = weather.get('wind_speed', 5)
            pressure = weather.get('pressure', 1013.25)
            solar    = weather.get('solar_radiation', 200.0)

            # Cooling index (compute once)
            cooling_idx = cooling_index(cooling)

            # -------------------------
            # Energy per request (Wh/request)
            # -------------------------
            if ai_models is not None and getattr(ai_models, "is_trained", False):
                energy_per_req = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
                energy_per_req = float(energy_per_req) * float(energy_multiplier)
            else:
                energy_per_req = calculate_energy_per_request(temp, humidity, cooling, energy_multiplier)

            # Total energy (Wh) and heat (kWh for ΔT + carbon)
            dc_energy_wh = float(energy_per_req) * float(requests)
            dc_heat_kwh  = dc_energy_wh / 1000.0  # 1000 Wh = 1 kWh

            # -------------------------
            # ΔT proxy (AI DT if available, else physics)
            # -------------------------
            ΔT = None
            if (ai_models is not None
                and getattr(ai_models, "is_trained", False)
                and "DT" in getattr(ai_models, "models", {})):
                try:
                    ΔT = ai_models.predict_dt([temp, humidity, wind, pressure, solar, cooling_idx])
                    ΔT = float(ΔT)
                except Exception:
                    ΔT = None

            if ΔT is None:
                ΔT = calculate_ΔT_contribution(dc_heat_kwh, area_km2=1.0, wind_speed=wind)

            # -------------------------
            # Carbon (kg) : kWh × (kg/kWh)
            # -------------------------
            region = dc_info.get('region', 'default')
            carbon_intensity = frozen[dc_name]  # recorded once for this simulation
            dc_carbon_kg = dc_heat_kwh * float(carbon_intensity)

            # -------------------------
            # Latency (ms) – use true load_fraction
            # -------------------------
            distance_km = haversine_distance(user_lat, user_lon, dc_info['lat'], dc_info['lon'])
            load_fraction = float(requests) / float(num_requests)  # share of the batch (reported)
            dc_latency_ms = calculate_latency(distance_km, load_fraction=site_utilization(requests))

            # Store DC-level metrics
            strategy_results['datacenters'][dc_name] = {
                'requests': requests,
                'percentage': (requests / num_requests) * 100,
                'energy_wh': dc_energy_wh,
                'energy_per_req': energy_per_req,
                'carbon_kg': dc_carbon_kg,
                'carbon_g': dc_carbon_kg * 1000,
                'latency_ms': dc_latency_ms,
                'heat_wh': dc_energy_wh,     # all energy becomes heat (Wh)
                'heat_kwh': dc_heat_kwh,
                'ΔT_contribution': ΔT,
                'load_fraction': load_fraction,
                'distance_km': distance_km
            }

            # Accumulate strategy totals
            total_energy_wh += dc_energy_wh
            total_carbon_kg += dc_carbon_kg
            weighted_latency_ms += dc_latency_ms * requests / max(1, served_requests)
            total_heat_wh += dc_energy_wh

            heat_values.append(dc_energy_wh)
            ΔT_values.append(ΔT)

        # Include idle sites so concentration across the selected fleet is represented.
        heat_values.extend([0.0] * (len(datacenters) - len(heat_values)))
        # Concentration metric (CV)
        if len(heat_values) > 1 and np.mean(heat_values) > 0:
            heat_cv = float(np.std(heat_values) / np.mean(heat_values))
        else:
            heat_cv = 0.0

        peak_ΔT = float(max(ΔT_values)) if ΔT_values else 0.0
        avg_ΔT  = float(np.mean(ΔT_values)) if ΔT_values else 0.0

        strategy_results['totals'] = {
            'offered_requests': num_requests,
            'served_requests': served_requests,
            'unserved_requests': num_requests - served_requests,
            'service_rate': served_requests / num_requests,
            'energy_wh': total_energy_wh,
            'carbon_kg': total_carbon_kg,
            'carbon_g': total_carbon_kg * 1000,
            'avg_latency_ms': weighted_latency_ms if served_requests else float('nan'),
            'total_heat_wh': total_heat_wh,
            'total_heat_kwh': total_heat_wh / 1000.0,
            'heat_cv': heat_cv,
            'peak_ΔT': peak_ΔT,
            'avg_ΔT': avg_ΔT
        }

        results[strategy_name] = strategy_results

    return results


def run_monte_carlo(datacenters, weather_data, user_location, num_requests,
                    cooling_selections, energy_multiplier, ai_models, n_runs=100,
                    latency_threshold=100, max_dc_capacity_mw=100, max_site_share=1.0,
                    carbon_intensities=None, carbon_time=None):
    """
    Run Monte Carlo simulation with stochastic weather variations.
    """
    all_results = {strategy: [] for strategy in ['Random', 'Energy-Only', 'Heat-Only', 'ΔT-AR', 'Multi-Objective']}
    
    for run in range(n_runs):
        # Generate stochastic weather variations
        varied_weather = {}
        for dc_name, weather in weather_data.items():
            varied_weather[dc_name] = {
                'temperature': weather['temperature'] + np.random.normal(0, 5),
                'humidity': np.clip(weather['humidity'] + np.random.normal(0, 15), 20, 95),
                'wind_speed': max(0.5, weather['wind_speed'] + np.random.normal(0, 2)),
                'pressure': weather.get('pressure', 1013.25) + np.random.normal(0, 10),
                'solar_radiation': max(0, weather.get('solar_radiation', 200.0) + np.random.normal(0, 100)),
                'source': 'Monte Carlo variation'
            }
                
        # Vary request volume
        varied_requests = max(1, int(num_requests * (1 + np.random.uniform(-0.2, 0.2))))
        
        # Run simulation
        results = run_simulation(
            datacenters, varied_weather, user_location,
            varied_requests, cooling_selections, energy_multiplier, ai_models,
            latency_threshold, max_dc_capacity_mw, max_site_share,
            carbon_intensities=carbon_intensities, carbon_time=carbon_time
        )
        
        for strategy, data in results.items():
            all_results[strategy].append(data['totals'])
    
    # Aggregate statistics
    mc_summary = {}
    for strategy, runs in all_results.items():
        df = pd.DataFrame(runs)
        mc_summary[strategy] = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'ci_lower': df.quantile(0.025).to_dict(),
            'ci_upper': df.quantile(0.975).to_dict(),
            'raw_data': df
        }
    
    return mc_summary


# ============================================================================
# VISUALIZATION FUNCTIONS - FIXED DIMENSIONS
# ============================================================================

def create_geographic_map(datacenters, user_location, distribution=None, title="Geographic Distribution"):
    """
    Create interactive world map showing user and datacenter locations.
    Map #1, #2, #3 in the dashboard.
    """
    fig = go.Figure()
    
    user_lat, user_lon = user_location
    
    # Add user location
    fig.add_trace(go.Scattergeo(
        lon=[user_lon],
        lat=[user_lat],
        mode='markers+text',
        marker=dict(size=18, color='#e94560', symbol='star',
                   line=dict(width=2, color='white')),
        text=['You'],
        textposition='top center',
        textfont=dict(size=28, color='#e94560', family='Aptos'),
        name='User Location',
        hovertemplate="<b>Your Location</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
    ))
    
    # Color mapping
    climate_colors = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}
    
    # Add datacenters
    for dc_name, dc_info in datacenters.items():
        climate = dc_info.get('climate', 'moderate')
        color = climate_colors.get(climate, '#6b7280')
        
        # Calculate size based on traffic if distribution provided
        if distribution and dc_name in distribution:
            total = sum(distribution.values())
            traffic_pct = (distribution[dc_name] / total) * 100 if total > 0 else 0
            size = 12 + traffic_pct * 0.4
            hover_text = f"<b>{dc_name}</b><br>Traffic: {traffic_pct:.1f}%<br>Requests: {distribution[dc_name]:,}"
        else:
            size = 15
            traffic_pct = 0
            hover_text = f"<b>{dc_name}</b>"
        
        fig.add_trace(go.Scattergeo(
            lon=[dc_info['lon']],
            lat=[dc_info['lat']],
            mode='markers',
            marker=dict(size=size, color=color, 
                       line=dict(width=2, color='white'),
                       opacity=0.9),
            name=dc_name,
            hovertemplate=hover_text + "<extra></extra>"
        ))
        
        # Draw connection line
        line_width = 2 if not distribution else max(1, traffic_pct * 0.08)
        fig.add_trace(go.Scattergeo(
            lon=[user_lon, dc_info['lon']],
            lat=[user_lat, dc_info['lat']],
            mode='lines',
            line=dict(width=line_width, color=color, dash='dot'),
            opacity=0.5,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#f5f5f5",
        showocean=True, oceancolor="#e8f4fc",
        showcoastlines=True, coastlinecolor="#999",
        showcountries=True, countrycolor="#ddd",
        showlakes=True, lakecolor="#e8f4fc"
    )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=48, family='Aptos', color='#000000')),
        height=900,
        autosize=True,
        margin=dict(l=0, r=0, t=120, b=0),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.95)',
                   bordercolor='#999', borderwidth=2,
                   font=dict(size=28, family='Aptos', color='#000000')),
        paper_bgcolor='white'
    )
    
    return fig


def create_traffic_distribution_chart(results, title="Traffic Distribution by Strategy"):
    """
    Create grouped bar chart showing traffic distribution.
    Graph #2, #6 in the dashboard.
    """
    strategies = list(results.keys())
    dc_names = list(results[strategies[0]]['distribution'].keys())
    
    climate_colors = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}
    
    fig = go.Figure()
    
    for dc_name in dc_names:
        # Get climate color - check both default and extended datacenters
        climate = 'moderate'
        if dc_name in DEFAULT_DATACENTERS:
            climate = DEFAULT_DATACENTERS[dc_name].get('climate', 'moderate')
        elif dc_name in EXTENDED_DC_LOCATIONS:
            climate = EXTENDED_DC_LOCATIONS[dc_name].get('climate', 'moderate')
        
        values = []
        for strategy in strategies:
            dist = results[strategy]['distribution']
            values.append(dist.get(dc_name, 0))
        
        # Shorter name for display
        short_name = dc_name.split(',')[0]
        
        fig.add_trace(go.Bar(
            name=short_name,
            x=strategies,
            y=values,
            marker_color=climate_colors.get(climate, '#6b7280'),
            text=[f"<b>{v:,}</b>" for v in values],
            textposition='outside',
            textfont=dict(size=28, color='#000000', family='Aptos'),
            cliponaxis=False,
            hovertemplate=f"<b>{short_name}</b><br>%{{x}}: %{{y:,}} requests<extra></extra>"
        ))


    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Number of Requests</b>", font=dict(size=36, family='Aptos', color='#000000')),
        barmode='group',
        bargap=0.25,
        bargroupgap=0.1,
        height=900,
        margin=dict(l=100, r=60, t=120, b=180),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Aptos', color='#000000')),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        uniformtext_minsize=28,
        uniformtext_mode='hide',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    
    return fig


def create_model_comparison_chart(model_metrics):
    """
    Create grouped bar chart for AI model performance.
    Graph #3 in the dashboard.
    """
    models = [m for m in model_metrics.keys() if m != 'feature_importance' and isinstance(model_metrics[m], dict) and 'r2' in model_metrics[m]]
    
    if not models:
        return None
    
    metrics = ['r2', 'mae', 'rmse']
    metric_names = ['R² Score', 'MAE (Wh)', 'RMSE (Wh)']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=metric_names,
                        horizontal_spacing=0.12)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [model_metrics[m][metric] for m in models]
        
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color=colors[i],
            text=[f"<b>{v:.4f}</b>" for v in values],
            textposition='auto',
            textfont=dict(size=28, color='#000000', family='Aptos'),
            showlegend=False
        ), row=1, col=i+1)

    fig.update_layout(
        title=dict(text="<b>AI Model Performance Comparison</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        height=900,
        margin=dict(l=100, r=60, t=120, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Aptos', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    
    return fig


def create_prediction_scatter(model_metrics, model_name='ANN'):
    """
    Create scatter plot of predicted vs actual values.
    Graph #4 in the dashboard.
    """
    if model_name not in model_metrics or 'predictions' not in model_metrics[model_name]:
        return None
    
    y_test = model_metrics[model_name]['y_test']
    y_pred = model_metrics[model_name]['predictions']
    r2 = model_metrics[model_name]['r2']
    
    y_test, y_pred = np.asarray(y_test, dtype=float), np.asarray(y_pred, dtype=float)
    total_points = len(y_test)

    # Calculate residuals for color
    residuals = y_test - y_pred

    # Drawing every test prediction (100k+) stalls the browser. Plot a fixed random sample
    # with WebGL; R² above is still computed on the full test set.
    max_points = 4000
    if total_points > max_points:
        sample = np.sort(np.random.default_rng(0).choice(total_points, max_points, replace=False))
        shown_test, shown_pred, shown_residuals = y_test[sample], y_pred[sample], residuals[sample]
    else:
        shown_test, shown_pred, shown_residuals = y_test, y_pred, residuals
    sample_note = (f" · {len(shown_test):,} of {total_points:,} points shown"
                   if total_points > max_points else "")

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scattergl(
        x=shown_test,
        y=shown_pred,
        mode='markers',
        marker=dict(
            size=5,
            color=shown_residuals,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title='<b>Residual</b>', x=1.02),
            line=dict(width=0.3, color='white'),
            opacity=0.6
        ),
        name='Predictions',
        hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>"
    ))
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#e94560', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{model_name}: Predicted vs Actual (R² = {r2:.4f}){sample_note}</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Actual Energy (Wh)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Predicted Energy (Wh)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=700,
        width=700,
        margin=dict(l=120, r=140, t=100, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Force 1:1 aspect ratio for scatter plots
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            tickfont=dict(size=32, color='#000000', family='Aptos')
        ),
        yaxis=dict(
            constrain='domain',
            tickfont=dict(size=32, color='#000000', family='Aptos')
        )
    )

    # Update colorbar title font
    fig.update_coloraxes(colorbar=dict(title=dict(text='<b>Residual</b>', font=dict(size=32, color='#000000', family='Aptos')),
                                       tickfont=dict(size=28, color='#000000', family='Aptos')))
    
    return fig


def create_feature_importance_chart(feature_importance, model_name='MLR'):
    """
    Create horizontal bar chart for feature importance.
    Graph #5 in the dashboard.
    """
    if not feature_importance:
        return None

    # Extract feature importance for the specified model
    model_importance = feature_importance
    if isinstance(feature_importance, dict):
        # Check if this is a nested dict with model names as keys
        if model_name and model_name in feature_importance:
            model_importance = feature_importance[model_name]
        elif all(isinstance(v, dict) for v in feature_importance.values()):
            # It's nested, get first available model
            for key, value in feature_importance.items():
                if isinstance(value, dict) and any(isinstance(v, (int, float)) for v in value.values()):
                    model_importance = value
                    break

    # Ensure model_importance has numeric values
    if not model_importance or not isinstance(model_importance, dict):
        return None

    # Filter to only include numeric values, skip any nested dicts or non-numeric values
    numeric_importance = {}
    for key, value in model_importance.items():
        try:
            if isinstance(value, (int, float)):
                numeric_importance[key] = value
            elif not isinstance(value, dict):
                # Try to convert to float
                numeric_importance[key] = float(value)
        except (ValueError, TypeError):
            # Skip items that can't be converted to numbers
            continue

    if not numeric_importance:
        return None

    # Sort by importance
    sorted_features = sorted(numeric_importance.items(), key=lambda x: abs(float(x[1])), reverse=True)
    features = [f[0].replace('_', ' ').title() for f in sorted_features]
    importances = [abs(f[1]) for f in sorted_features]
    
    # Normalize to percentages
    total = sum(importances)
    percentages = [i/total * 100 for i in importances]
    
    colors = ['#e94560', '#3b82f6', '#10b981', '#f59e0b']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=percentages,
        orientation='h',
        marker_color=colors[:len(features)],
        text=[f"<b>{p:.1f}%</b>" for p in percentages],
        textposition='outside',
        textfont=dict(size=32, family='Aptos', color='#000000')
    ))

    fig.update_layout(
        title=dict(text="<b>Feature Importance for Energy Prediction</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Relative Importance (%)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=800,
        width=900,
        margin=dict(l=220, r=120, t=100, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        yaxis=dict(autorange='reversed', tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(range=[0, max(percentages) * 1.15], tickfont=dict(size=32, color='#000000', family='Aptos'))
    )
    
    return fig


def create_metrics_comparison_chart(results):
    """
    Create multi-metric comparison chart.
    Graph #7 in the dashboard.
    """
    strategies = list(results.keys())
    
    metrics = {
        'Energy (Wh)': [results[s]['totals']['energy_wh'] for s in strategies],
        'Carbon (g)': [results[s]['totals']['carbon_g'] for s in strategies],
        'Latency (ms)': [results[s]['totals']['avg_latency_ms'] for s in strategies],
        'Peak ΔT-AR (µ°C, ×10⁶)': [results[s]['totals']['peak_ΔT'] * 1e6 for s in strategies],
        'Heat CV (×100)': [results[s]['totals']['heat_cv'] * 100 for s in strategies]
    }
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#e94560', '#8b5cf6']
    
    fig = go.Figure()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            name=metric_name,
            x=strategies,
            y=values,
            marker_color=colors[i],
            opacity=0.85
        ))
    
    fig.update_layout(
        title=dict(text="<b>Performance Metrics by Strategy</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Metric Value</b>", font=dict(size=36, family='Aptos', color='#000000')),
        barmode='group',
        height=900,
        margin=dict(l=100, r=60, t=120, b=200),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Aptos', color='#000000')),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig
def create_bayesopt_convergence_chart(model_results):
    """
    Plot Bayesian Optimization convergence:
    validation R² per iteration + best-so-far curve.
    """
    try:
        bay = model_results.get("Bayesian", {})
        hist = bay.get("bo_history", None)
        if not hist:
            return None

        iters = [h["iteration"] for h in hist if "iteration" in h]
        vals = [h["val_r2"] for h in hist if "val_r2" in h]

        if not iters or not vals:
            return None

        best_so_far = []
        running = -1e9
        for v in vals:
            if v is None:
                best_so_far.append(running)
            else:
                running = max(running, float(v))
                best_so_far.append(running)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iters, y=vals, mode="lines+markers", name="Val R²"))
        fig.add_trace(go.Scatter(x=iters, y=best_so_far, mode="lines", name="Best so far"))

        fig.update_layout(
            title=dict(text="<b>Bayesian Optimization Convergence (Validation R²)</b>",
                      font=dict(size=48, family='Aptos', color='#000000')),
            xaxis_title=dict(text="<b>Iteration</b>", font=dict(size=36, family='Aptos', color='#000000')),
            yaxis_title=dict(text="<b>Validation R²</b>", font=dict(size=36, family='Aptos', color='#000000')),
            height=600,
            margin=dict(l=100, r=60, t=120, b=140),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Aptos", size=28, color='#000000'),
            xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
            yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
            legend=dict(y=-0.3, x=0.5, xanchor='center', orientation='h', font=dict(size=32, color='#000000', family='Aptos'))
        )
        return fig
    except Exception:
        return None

def create_learning_curve_chart(model_results):
    """
    ANN convergence plots.

    Row 1: Training loss (MSE proxy from sklearn MLPRegressor.loss_curve_)
    Row 2: Validation R² over epochs (only present when early_stopping=True)
    """

    series = []
    for name in ['ANN', 'Bayesian']:
        if name in model_results and isinstance(model_results[name], dict):
            loss = model_results[name].get('loss_curve', None)
            val_scores = model_results[name].get('val_score_curve', None)
            if loss is not None and len(loss) > 0:
                series.append((name, list(loss), list(val_scores) if val_scores is not None else None))

    if not series:
        return None

    has_val = any(v is not None and len(v) > 0 for _, _, v in series)
    rows = 2 if has_val else 1
    subplot_titles = ("Training loss (MSE)", "Validation R² (early stopping)") if has_val else ("Training loss (MSE)",)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.25,
        subplot_titles=subplot_titles
    )

    colors = {'ANN': '#10b981', 'Bayesian': '#f59e0b'}

    for name, loss, val_scores in series:
        epochs = list(range(1, len(loss) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs, y=loss,
                mode='lines+markers',
                name=f"{name} loss",
                line=dict(width=4, color=colors.get(name, '#6b7280'), shape='spline'),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        if has_val and val_scores is not None and len(val_scores) > 0:
            ve = list(range(1, len(val_scores) + 1))
            fig.add_trace(
                go.Scatter(
                    x=ve, y=val_scores,
                    mode='lines+markers',
                    name=f"{name} val R²",
                    line=dict(width=4, dash='dot', color=colors.get(name, '#6b7280'), shape='spline'),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )

    fig.update_yaxes(title_text="<b>Loss</b>", title_font=dict(size=36, color='#000000', family='Aptos'),
                     tickfont=dict(size=32, color='#000000', family='Aptos'), row=1, col=1)
    if has_val:
        fig.update_yaxes(title_text="<b>R²</b>", title_font=dict(size=36, color='#000000', family='Aptos'),
                        tickfont=dict(size=32, color='#000000', family='Aptos'), row=2, col=1)

    fig.update_xaxes(title_text="<b>Epoch</b>", title_font=dict(size=36, color='#000000', family='Aptos'),
                     tickfont=dict(size=32, color='#000000', family='Aptos'), row=rows, col=1)

    fig.update_layout(
        title=dict(text="<b>ANN Training Convergence</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        height=1200 if has_val else 700,
        margin=dict(l=130, r=80, t=140, b=180),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Aptos', size=28, color='#000000'),
        legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos'))
    )

    # Update subplot titles with larger fonts
    fig.update_annotations(font=dict(size=36, family='Aptos', color='#000000'))

    return fig


def create_heat_distribution_chart(results):
    """
    Create stacked bar chart showing heat distribution.
    Graph #8 in the dashboard.
    """
    strategies = list(results.keys())
    dc_names = list(results[strategies[0]]['datacenters'].keys())
    
    climate_colors = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}
    
    fig = go.Figure()
    
    for dc_name in dc_names:
        values = []
        for strategy in strategies:
            dc_data = results[strategy]['datacenters'].get(dc_name, {})
            values.append(dc_data.get('heat_wh', 0))
        
        short_name = dc_name.split(',')[0]
        climate = DEFAULT_DATACENTERS.get(dc_name, {}).get('climate', 'moderate')
        
        fig.add_trace(go.Bar(
            name=short_name,
            x=strategies,
            y=values,
            marker_color=climate_colors.get(climate, '#6b7280'),
            text=[f"<b>{v:,.0f}</b>" for v in values],
            textposition='auto',
            textfont=dict(size=28, color='#000000', family='Aptos')
        ))

    fig.update_layout(
        title=dict(text="<b>Heat Distribution by Datacenter (Wh)</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Heat Dissipation (Wh)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        barmode='stack',
        height=900,
        margin=dict(l=100, r=60, t=120, b=200),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Aptos', color='#000000')),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_monte_carlo_boxplots(mc_results):
    """
    Create box plots for Monte Carlo results.
    Graphs #9-12 in the dashboard.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Energy (Wh)", "Peak ΔT-AR (°C)", "Heat CV", "Carbon (g)"),
        vertical_spacing=0.25,
        horizontal_spacing=0.15
    )
    
    strategies = list(mc_results.keys())
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'ΔT-AR': '#10b981',
        'Heat-Only': '#9333ea',
        'Multi-Objective': '#3b82f6'
    }
    
    metrics = [
        ('energy_wh', 1, 1),
        ('peak_ΔT', 1, 2),
        ('heat_cv', 2, 1),
        ('carbon_g', 2, 2)
    ]
    
    for metric, row, col in metrics:
        for strategy in strategies:
            data = mc_results[strategy]['raw_data'][metric]
            fig.add_trace(go.Box(
                y=data,
                name=strategy,
                marker_color=colors.get(strategy, '#6b7280'),
                showlegend=(row == 1 and col == 1),
                boxmean=True
            ), row=row, col=col)
    
    fig.update_layout(
        title=dict(text="<b>Monte Carlo Results: distribution across runs</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        height=1400,
        margin=dict(l=120, r=80, t=140, b=220),
        font=dict(family='Aptos', size=28, color='#000000'),
        showlegend=True,
        legend=dict(orientation='h', y=-0.20, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Aptos', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    
    return fig

def create_monte_carlo_mean_ci_chart(mc_results):
    """
    Mean ± 95% CI of the MEAN (not the percentile interval).
    """
    metrics = [
        ("energy_wh", "Energy (Wh)"),
        ("peak_ΔT", "Peak ΔT-AR (°C)"),
        ("heat_cv", "Heat CV"),
        ("carbon_g", "Carbon (g)")
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.25,
        horizontal_spacing=0.15
    )

    strategies = list(mc_results.keys())
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'ΔT-AR': '#10b981',
        'Multi-Objective': '#3b82f6'
    }

    for i, (metric, label) in enumerate(metrics):
        r = 1 if i < 2 else 2
        c = 1 if i % 2 == 0 else 2

        means, err_plus, err_minus = [], [], []
        for s in strategies:
            data = mc_results[s]['raw_data'][metric].values
            n = len(data)
            mean = float(np.mean(data))
            sem = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci = 1.96 * sem
            means.append(mean)
            err_plus.append(ci)
            err_minus.append(ci)

        fig.add_trace(
            go.Bar(
                x=strategies,
                y=means,
                error_y=dict(type='data', array=err_plus, arrayminus=err_minus, visible=True),
                marker_color=[colors.get(s, '#6b7280') for s in strategies],
                showlegend=False
            ),
            row=r, col=c
        )

    fig.update_layout(
        title=dict(text="<b>Monte Carlo: Mean ± 95% CI of the Mean</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        height=1400,
        margin=dict(l=120, r=80, t=140, b=160),
        font=dict(family='Aptos', size=28, color='#000000'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Aptos', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Aptos'))
    return fig


def create_monte_carlo_hist_overlay(mc_results, metric="peak_ΔT", title=None, x_label=None):
    """
    Distribution overlay plot (histogram density) for a single Monte Carlo metric.
    """
    strategies = list(mc_results.keys())
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'ΔT-AR': '#10b981',
        'Multi-Objective': '#3b82f6'
    }

    fig = go.Figure()
    for s in strategies:
        data = mc_results[s]['raw_data'][metric].values
        fig.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name=s,
            opacity=0.55,
            marker_color=colors.get(s, '#6b7280')
        ))

        mu = float(np.mean(data))
        fig.add_vline(x=mu, line_width=2, line_dash="dot", line_color=colors.get(s, '#6b7280'))

    fig.update_layout(
        barmode='overlay',
        title=dict(text=title or f"Monte Carlo Distribution Overlay: {metric}",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text=x_label or metric, font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Density</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=650,
        margin=dict(l=110, r=60, t=120, b=140),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos'))
    )
    return fig


def create_monte_carlo_running_mean_chart(mc_results, metric="peak_ΔT", title=None, y_label=None):
    """
    Running mean (convergence) plot to show stability as #runs increases.
    """
    strategies = list(mc_results.keys())
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'ΔT-AR': '#10b981',
        'Multi-Objective': '#3b82f6'
    }

    fig = go.Figure()
    for s in strategies:
        data = mc_results[s]['raw_data'][metric].values
        running = pd.Series(data).expanding().mean().values
        fig.add_trace(go.Scatter(
            x=list(range(1, len(running) + 1)),
            y=running,
            mode='lines',
            name=s,
            line=dict(width=2, color=colors.get(s, '#6b7280'))
        ))

    fig.update_layout(
        title=dict(text=title or f"Monte Carlo Convergence: Running Mean of {metric}",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Monte Carlo run index</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text=y_label or metric, font=dict(size=36, family='Aptos', color='#000000')),
        height=650,
        margin=dict(l=110, r=60, t=120, b=140),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos'))
    )
    return fig



def create_sensitivity_temperature_chart(sensitivity_data):
    """
    Create line chart for temperature sensitivity.
    Graph #13 in the dashboard.
    """
    fig = go.Figure()
    
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'ΔT-AR': '#10b981',
        'Multi-Objective': '#3b82f6'
    }
    
    for strategy, values in sensitivity_data['y'].items():
        fig.add_trace(go.Scatter(
            x=sensitivity_data['x'],
            y=values,
            mode='lines+markers',
            name=strategy,
            line=dict(color=colors.get(strategy, '#6b7280'), width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=dict(text="<b>Temperature Sensitivity Analysis</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Ambient Temperature (°C)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Peak ΔT-AR (°C)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=800,
        margin=dict(l=110, r=60, t=120, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_carbon_intensity_curves(datacenters=None):
    """
    Create 24-hour carbon intensity curves.
    Graph #17 in the dashboard.
    """
    hours = list(range(24))
    
    regions = {
        'Phoenix (Arizona)': 'arizona',
        'San Francisco (California)': 'california', 
        'Stockholm (Sweden)': 'sweden'
    }
    longitudes = {'Phoenix (Arizona)': -112.07, 'San Francisco (California)': -122.42, 'Stockholm (Sweden)': 18.07}
    if datacenters:
        regions = {name: info.get('region', 'default') for name, info in datacenters.items()}
        longitudes = {name: info.get('lon', 0.0) for name, info in datacenters.items()}
    
    colors = {
        'Phoenix (Arizona)': '#dc2626',
        'San Francisco (California)': '#d97706',
        'Stockholm (Sweden)': '#2563eb'
    }
    
    fig = go.Figure()
    
    palette = px.colors.qualitative.Safe
    for index, (region_name, region_key) in enumerate(regions.items()):
        values = [get_carbon_intensity(region_key, h) for h in hours]
        color = colors.get(region_name, palette[index % len(palette)])
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=values,
            mode='lines',
            name=region_name,
            line=dict(color=color, width=3),
            fill='tozeroy',
            # Translucent fills: the curves overlap (they are not stacked), so each stays visible.
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)"
            if color.startswith('#') and len(color) == 7 else color,
        ))
        # Each site's "now" is its own local solar hour, so the marker sits on its own curve.
        now_hour = local_solar_hour(longitudes.get(region_name, 0.0))
        fig.add_trace(go.Scatter(
            x=[now_hour], y=[values[now_hour]], mode='markers+text', text=['now'],
            textposition='top center', marker=dict(size=12, color=color, line=dict(width=2, color='white')),
            showlegend=False, hovertemplate=f"{region_name}<br>Local hour now: {now_hour}:00<br>"
                                            "%{y:.3f} kg CO₂/kWh<extra></extra>",
        ))
    
    
    fig.update_layout(
        title=dict(text="<b>24-Hour Carbon Intensity Patterns</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Local hour of day at each site</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Carbon Intensity (kg CO₂/kWh)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=800,
        margin=dict(l=110, r=60, t=120, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Aptos')),
        xaxis=dict(tickmode='linear', tick0=0, dtick=4, tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_cooling_effectiveness_heatmap():
    """
    Create heatmap for cooling effectiveness by climate.
    Graph #18 in the dashboard.
    """
    cooling_types = ['Free Air', 'Air Economizer', 'Evaporative', 'Liquid', 'Mechanical']
    climates = ['Cold', 'Moderate', 'Hot/Dry', 'Hot/Humid']
    
    # Effectiveness matrix (1=optimal, 0.5=acceptable, 0=not recommended)
    effectiveness = [
        [1.0, 0.7, 0.2, 0.1],   # Free Air
        [1.0, 1.0, 0.5, 0.3],   # Air Economizer
        [0.5, 0.7, 1.0, 0.2],   # Evaporative
        [1.0, 1.0, 1.0, 1.0],   # Liquid
        [0.5, 0.5, 0.5, 0.5],   # Mechanical
    ]
    
    # Custom colorscale: red (bad) -> yellow (ok) -> green (good)
    colorscale = [[0, '#ef4444'], [0.5, '#fbbf24'], [1, '#22c55e']]
    
    fig = go.Figure(data=go.Heatmap(
        z=effectiveness,
        x=climates,
        y=cooling_types,
        colorscale=colorscale,
        text=[['✗' if v < 0.4 else '○' if v < 0.8 else '✓' for v in row] for row in effectiveness],
        texttemplate="%{text}",
        textfont=dict(size=36, color='#000000', family='Aptos'),
        hovertemplate="Cooling: %{y}<br>Climate: %{x}<br>Effectiveness: %{z:.0%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Cooling Effectiveness by Climate Type</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Climate Type</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Cooling Technology</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=800,
        margin=dict(l=240, r=80, t=120, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_pue_comparison_chart():
    """
    Create horizontal bar chart for PUE comparison.
    Graph #19 in the dashboard.
    """
    cooling_data = [(name, info['pue'], info['icon']) 
                    for name, info in COOLING_SYSTEMS.items()]
    cooling_data.sort(key=lambda x: x[1])
    
    names = [f"{d[2]} {d[0].replace('_', ' ').title()}" for d in cooling_data]
    pues = [d[1] for d in cooling_data]
    
    # Color gradient from green (low PUE) to red (high PUE)
    colors = ['#22c55e', '#84cc16', '#fbbf24', '#f97316', '#ef4444']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names,
        x=pues,
        orientation='h',
        marker_color=colors,
        text=[f"<b>PUE: {p:.2f}</b>" for p in pues],
        textposition='outside',
        textfont=dict(size=32, family='Aptos', color='#000000')
    ))

    fig.update_layout(
        title=dict(text="<b>Power Usage Effectiveness (PUE) by Cooling Technology</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>PUE Value (lower is better)</b>", font=dict(size=36, family='Aptos', color='#000000')),
        height=800,
        width=900,
        margin=dict(l=280, r=140, t=120, b=100),
        font=dict(family='Aptos', size=28, color='#000000'),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        xaxis=dict(range=[1.0, 2.2], tickfont=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_scenario_comparison_chart(scenario_a, scenario_b, scenario_a_name, scenario_b_name):
    """
    Create grouped bar chart for scenario comparison.
    Graph #16 in the dashboard.
    """
    metrics = ['energy_wh', 'carbon_g', 'avg_latency_ms', 'peak_ΔT', 'heat_cv']
    metric_labels = ['Energy (Wh)', 'Carbon (g)', 'Latency (ms)', 'Peak ΔT-AR (µ°C, ×10⁶)', 'Heat CV']
    
    values_a = [scenario_a['totals'][m] if m != 'peak_ΔT' else scenario_a['totals'][m] * 1e6 for m in metrics]
    values_b = [scenario_b['totals'][m] if m != 'peak_ΔT' else scenario_b['totals'][m] * 1e6 for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=scenario_a_name,
        x=metric_labels,
        y=values_a,
        marker_color='#3b82f6'
    ))
    
    fig.add_trace(go.Bar(
        name=scenario_b_name,
        x=metric_labels,
        y=values_b,
        marker_color='#10b981'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Scenario Comparison</b>",
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis_title=dict(text="<b>Metrics</b>", font=dict(size=36, family='Aptos', color='#000000')),
        yaxis_title=dict(text="<b>Value</b>", font=dict(size=36, family='Aptos', color='#000000')),
        barmode='group',
        height=900,
        margin=dict(l=100, r=60, t=120, b=180),
        font=dict(family='Aptos', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Aptos')),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_train_val_test_comparison(model_results):
    """Create comparison chart showing train/val/test metrics"""

    models = []
    train_r2 = []
    val_r2 = []
    test_r2 = []

    for name in ['MLR', 'ANN', 'Bayesian']:
        if name in model_results and isinstance(model_results[name], dict):
            if 'train_r2' in model_results[name]:
                models.append(name)
                train_r2.append(model_results[name]['train_r2'])
                val_r2.append(model_results[name]['val_r2'])
                test_r2.append(model_results[name]['r2'])

    if not models:
        return None

    fig = go.Figure()

    # Convert to horizontal bars
    fig.add_trace(go.Bar(
        name='Training',
        y=models,
        x=train_r2,
        orientation='h',
        marker_color='#3b82f6',
        text=[f"<b>{v:.4f}</b>" for v in train_r2],
        textposition='outside',
        textfont=dict(size=28, family='Aptos', color='#000000')
    ))

    fig.add_trace(go.Bar(
        name='Validation',
        y=models,
        x=val_r2,
        orientation='h',
        marker_color='#10b981',
        text=[f"<b>{v:.4f}</b>" for v in val_r2],
        textposition='outside',
        textfont=dict(size=28, family='Aptos', color='#000000')
    ))

    fig.add_trace(go.Bar(
        name='Test',
        y=models,
        x=test_r2,
        orientation='h',
        marker_color='#f59e0b',
        text=[f"<b>{v:.4f}</b>" for v in test_r2],
        textposition='outside',
        textfont=dict(size=28, family='Aptos', color='#000000')
    ))

    fig.update_layout(
        title=dict(text='Model Performance: Train vs Validation vs Test',
                  font=dict(size=48, family='Aptos', color='#000000')),
        xaxis=dict(title=dict(text='R² Score', font=dict(size=36, family='Aptos', color='#000000')),
                  range=[0, 1.08], tickfont=dict(size=32, color='#000000', family='Aptos')),
        yaxis=dict(title=dict(text='Model', font=dict(size=36, family='Aptos', color='#000000')),
                  tickfont=dict(size=32, color='#000000', family='Aptos')),
        barmode='group',
        height=600,
        width=1000,
        margin=dict(l=120, r=180, t=120, b=180),
        showlegend=True,
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Aptos')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Aptos', size=28, color='#000000')
    )

    return fig

# ============================================================================
# RECOMMENDATION GENERATORS (All Dynamic)
# ============================================================================

def generate_initial_recommendation(datacenters, weather_data, user_location, latency_threshold):
    """
    Generate initial DC recommendation based on current conditions.
    All calculations are DYNAMIC.
    """
    user_lat, user_lon = user_location
    
    scores = {}
    details = {}
    
    for dc_name, dc_info in datacenters.items():
        weather = weather_data.get(dc_name, {})
        
        # Calculate metrics
        distance = haversine_distance(user_lat, user_lon, dc_info['lat'], dc_info['lon'])
        latency = calculate_latency(distance, load_fraction=0.7)
        temp = weather.get('temperature', 20)
        cooling = dc_info.get('default_cooling', 'air_economizer')
        energy = calculate_energy_per_request(temp, weather.get('humidity', 50), cooling)
        region = dc_info.get('region', 'default')
        carbon = get_carbon_intensity(region, local_solar_hour(dc_info['lon']))
        
        details[dc_name] = {
            'distance': distance,
            'latency': latency,
            'temperature': temp,
            'energy': energy,
            'carbon': carbon,
            'eligible': latency <= latency_threshold
        }
        
        if latency <= latency_threshold:
            # Score: lower is better (normalize each component)
            scores[dc_name] = (
                0.35 * (energy / 0.6) +  # Energy normalized
                0.25 * (latency / latency_threshold) +  # Latency normalized
                0.25 * (carbon / 0.5) +  # Carbon normalized
                0.15 * (max(0, temp - 20) / 20)  # ΔT-AR risk normalized
            )
    
    if not scores:
        return {
            'recommended': None,
            'reasoning': ["No datacenters meet the latency threshold"],
            'details': details
        }
    
    # Find best
    best_dc = min(scores, key=lambda dc: scores[dc])
    best_details = details[best_dc]
    
    # Generate dynamic reasoning
    reasoning = []
    if best_details['temperature'] < 20:
        reasoning.append(f"Low temperature ({best_details['temperature']:.1f}°C) enables efficient cooling")
    elif best_details['temperature'] < 28:
        reasoning.append(f"Moderate temperature ({best_details['temperature']:.1f}°C) for reasonable cooling")
    
    if best_details['carbon'] < 0.2:
        reasoning.append(f"Very low carbon intensity ({best_details['carbon']:.2f} kg/kWh)")
    elif best_details['carbon'] < 0.35:
        reasoning.append(f"Low carbon intensity ({best_details['carbon']:.2f} kg/kWh)")
    
    if best_details['latency'] < latency_threshold * 0.8:
        reasoning.append(f"Well within latency threshold ({best_details['latency']:.0f}ms < {latency_threshold}ms)")
    else:
        reasoning.append(f"Meets latency threshold ({best_details['latency']:.0f}ms ≤ {latency_threshold}ms)")
    
    # Generate alternatives analysis
    alternatives = []
    for dc_name, dc_details in details.items():
        if dc_name != best_dc:
            if not dc_details['eligible']:
                alternatives.append(f"{dc_name.split(',')[0]}: Exceeds latency ({dc_details['latency']:.0f}ms > {latency_threshold}ms)")
            elif dc_details['temperature'] > 30:
                alternatives.append(f"{dc_name.split(',')[0]}: High temperature ({dc_details['temperature']:.1f}°C) increases cooling demand")
            elif dc_details['carbon'] > 0.4:
                alternatives.append(f"{dc_name.split(',')[0]}: Higher carbon intensity ({dc_details['carbon']:.2f} kg/kWh)")
    
    return {
        'recommended': best_dc,
        'reasoning': reasoning,
        'alternatives': alternatives,
        'details': details
    }


def binding_constraint(results):
    """Name the limit that left every site with zero capacity, from the limits each result records."""
    limits = next((r.get('limits') for r in results.values() if r.get('limits')), None)
    if not limits:
        return "constraints left no capacity"
    reasons = []
    if limits.get("latency") is not None and not any(limits["latency"].values()):
        reasons.append("no selected datacenter can respond within the latency limit")
    if not any(limits.get("power", {}).values()):
        reasons.append("the power capacity allows no requests")
    if limits.get("share", 1) == 0:
        reasons.append("the maximum share per site allows no requests")
    return "; ".join(reasons) or "the combined limits leave no capacity at any site"


RECOMMENDATION_METRICS = {"energy_wh": "Energy", "carbon_g": "Carbon", "avg_latency_ms": "Latency",
                          "peak_ΔT": "Peak ΔT-AR"}
RECOMMENDATION_BASELINES = ("Random",)  # sampled reference only; never recommended


def generate_final_recommendation(results, latency_threshold):
    """
    Recommend a strategy from this run's calculated results.

    1. Candidates: every strategy except the Random baseline. Latency limits are already applied
       to all strategies during allocation, so "served" means served within the limit.
    2. Keep the candidates that serve the most requests.
    3. Among those, pick the strategy closest to the best value on every metric at once:
       for each of energy, carbon, latency and peak ΔT-AR, its gap to the best candidate
       (value / best - 1); the score is the average gap with equal weights. Lower is better;
       ties go to the lower peak ΔT-AR.

    Scoring several metrics avoids a circular result: ΔT-AR minimizes peak ΔT by construction,
    so ranking on peak ΔT alone would always pick it.
    """
    strategies = list(results.keys())
    eligible = [s for s in strategies if s not in RECOMMENDATION_BASELINES] or strategies
    excluded = {s: "baseline" for s in strategies if s not in eligible}
    maximum_served = max(results[s]['totals'].get('served_requests', 0) for s in eligible)
    candidates = [s for s in eligible if results[s]['totals'].get('served_requests', 0) == maximum_served]
    excluded.update({s: "fewer requests served" for s in eligible if s not in candidates})
    if maximum_served == 0:
        constraint = binding_constraint(results)
        return {
            'recommended_strategy': None, 'feasible': False, 'binding_constraint': constraint,
            'reasoning': [f"No feasible strategy: no requests could be served ({constraint})."],
            'improvements': {}, 'scores': {}, 'gaps': {}, 'candidates': [], 'excluded': excluded,
            'served_requests': 0, 'energy_only_metrics': results.get('Energy-Only', {}).get('totals'),
            'best_metrics': None,
        }

    best_values = {m: min(results[s]['totals'][m] for s in candidates) for m in RECOMMENDATION_METRICS}
    gaps = {s: {m: (results[s]['totals'][m] / best_values[m] - 1) if best_values[m] > 0
                else (0.0 if results[s]['totals'][m] <= 0 else float('inf'))
                for m in RECOMMENDATION_METRICS} for s in candidates}
    scores = {s: sum(gaps[s].values()) / len(RECOMMENDATION_METRICS) for s in candidates}
    best_ΔT = min(candidates, key=lambda s: (scores[s], results[s]['totals']['peak_ΔT']))

    # Calculate improvements vs Energy-Only
    energy_only = results['Energy-Only']['totals']
    best = results[best_ΔT]['totals']

    latency_info = {}
    for strategy_name, data in results.items():
        avg_latency = data['totals'].get('avg_latency_ms', None)
        if avg_latency is None:
            continue

        status = "within latency SLO" if avg_latency <= latency_threshold else "violates latency SLO"

        latency_info[strategy_name] = {
            "avg_latency_ms": avg_latency,
            "status": status,
            "threshold_ms": latency_threshold,
        }

    
    improvements = {}
    
    if energy_only['peak_ΔT'] > 0:
        improvements['ΔT_reduction_pct'] = ((energy_only['peak_ΔT'] - best['peak_ΔT']) / energy_only['peak_ΔT']) * 100
    else:
        improvements['ΔT_reduction_pct'] = 0
    
    if energy_only['heat_cv'] > 0:
        improvements['cv_reduction_pct'] = ((energy_only['heat_cv'] - best['heat_cv']) / energy_only['heat_cv']) * 100
    else:
        improvements['cv_reduction_pct'] = 0
    
    if energy_only['avg_latency_ms'] > 0:
        improvements['latency_change_pct'] = ((best['avg_latency_ms'] - energy_only['avg_latency_ms']) / energy_only['avg_latency_ms']) * 100
    else:
        improvements['latency_change_pct'] = 0
    
    if energy_only['energy_wh'] > 0:
        improvements['energy_overhead_pct'] = ((best['energy_wh'] - energy_only['energy_wh']) / energy_only['energy_wh']) * 100
    else:
        improvements['energy_overhead_pct'] = 0

       
    # Generate dynamic reasoning
    reasoning = []
    if best.get('unserved_requests', 0):
        reasoning.append(f"Warning: {best['unserved_requests']:,} requests remain unserved; "
                         "lower energy and heat reflect reduced service.")

    latency_slo_met = None
    if best.get("served_requests", 1) == 0:
        reasoning.append("No requests were served, so latency compliance cannot be assessed.")
    elif latency_threshold is not None and latency_threshold > 0:
        best_latency = best.get('avg_latency_ms')
        if best_latency is not None:
            latency_slo_met = best_latency <= latency_threshold
            if latency_slo_met:
                reasoning.append(
                    f"✅ Average latency ({best_latency:.1f} ms) stays within the target of {latency_threshold} ms."
                )
            else:
                reasoning.append(
                    f"⚠️ Average latency ({best_latency:.1f} ms) exceeds the target of {latency_threshold} ms; "
                    f"this strategy trades some latency for energy/ΔT-AR benefits."
                )    
    
    if improvements['ΔT_reduction_pct'] > 40:
        reasoning.append(f"✅ {improvements['ΔT_reduction_pct']:.1f}% reduction in Peak ΔT-AR")
    elif improvements['ΔT_reduction_pct'] > 20:
        reasoning.append(f"✅ {improvements['ΔT_reduction_pct']:.1f}% reduction in Peak ΔT-AR")
    elif improvements['ΔT_reduction_pct'] > 0:
        reasoning.append(f"✅ {improvements['ΔT_reduction_pct']:.1f}% reduction in Peak ΔT-AR")
    
    if improvements['cv_reduction_pct'] > 50:
        reasoning.append(f"✅ {improvements['cv_reduction_pct']:.1f}% reduction in heat concentration")
    elif improvements['cv_reduction_pct'] > 0:
        reasoning.append(f"✅ {improvements['cv_reduction_pct']:.1f}% reduction in heat concentration")
    
    if improvements['latency_change_pct'] < 0:
        reasoning.append(f"✅ {-improvements['latency_change_pct']:.1f}% improvement in latency")
    
    if improvements['energy_overhead_pct'] < 10:
        reasoning.append(f"⚠️ Only {improvements['energy_overhead_pct']:.1f}% energy overhead (acceptable)")
    elif improvements['energy_overhead_pct'] < 20:
        reasoning.append(f"⚠️ {improvements['energy_overhead_pct']:.1f}% energy overhead (moderate)")
    else:
        reasoning.append(f"❌ {improvements['energy_overhead_pct']:.1f}% energy overhead (significant)")
    
    return {
        'recommended_strategy': best_ΔT,
        'improvements': improvements,
        'reasoning': reasoning,
        'energy_only_metrics': energy_only,
        'best_metrics': best,
        'scores': scores,
        'gaps': gaps,
        'candidates': candidates,
        'excluded': excluded,
        'served_requests': maximum_served,
        # False when no strategy can serve any request (e.g. no site meets the latency limit);
        # the named strategy is then only a placeholder and must not be presented as a choice.
        'feasible': maximum_served > 0,
    }


def calculate_statistical_significance(mc_results, reference="ΔT-AR", metric="peak_ΔT"):
    """
    Paired comparison of ``reference`` against every other strategy on ``metric``.

    Each Monte Carlo run applies the same sampled weather and demand to all strategies, so
    results are paired by run. For each comparison: Wilcoxon signed-rank test on the per-run
    differences, Holm correction across comparisons, median difference, paired effect size
    (Cohen's d_z = mean / std of differences) and the share of runs where the reference is lower.

    Simulated p-values shrink as more runs are added, so the effect sizes and win rates are the
    more meaningful numbers; none of this is field validation.
    """
    if reference not in mc_results:
        return None
    ref_values = np.asarray(mc_results[reference]['raw_data'][metric], dtype=float)
    ref_unserved = np.asarray(mc_results[reference]['raw_data'].get('unserved_requests', 0), dtype=float)
    comparisons = {}
    for name, data in mc_results.items():
        if name == reference or 'raw_data' not in data:
            continue
        other = np.asarray(data['raw_data'][metric], dtype=float)
        n = min(len(ref_values), len(other))
        diff = ref_values[:n] - other[:n]  # negative: reference is lower (better)
        if n == 0:
            continue
        # Only exactly equal results count as "no difference". ΔT values are ~1e-4 °C, so a
        # tolerance-based check (np.allclose) would silently discard real, consistent differences.
        if np.all(diff == 0):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        spread = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        # d_z is undefined when every difference is identical (zero spread): report None rather
        # than 0, which would wrongly suggest no effect.
        cohens_dz = float(np.mean(diff) / spread) if spread > 0 else None
        unserved = np.asarray(data['raw_data'].get('unserved_requests', 0), dtype=float)
        comparisons[name] = {
            'n_runs': int(n),
            'median_difference': float(np.median(diff)),
            'mean_difference': float(np.mean(diff)),
            'cohens_dz': cohens_dz,
            'reference_lower_share': float(np.mean(diff < 0)),
            'p_value': p_value,
            # A strategy that served fewer requests in some runs has less heat for that reason alone.
            'unserved_runs': int(np.sum(np.atleast_1d(unserved)[:n] > 0) + np.sum(np.atleast_1d(ref_unserved)[:n] > 0)),
        }
    if not comparisons:
        return None
    # Holm step-down correction across the comparisons.
    ordered = sorted(comparisons, key=lambda k: comparisons[k]['p_value'])
    running = 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * comparisons[name]['p_value'])
        running = max(running, adjusted)
        comparisons[name]['p_holm'] = running
    for result in comparisons.values():
        result['significant'] = result['p_holm'] < 0.05
    headline = comparisons.get('Energy-Only') or comparisons[ordered[0]]
    return {
        'reference': reference,
        'metric': metric,
        'test': 'Wilcoxon signed-rank (paired by Monte Carlo run), Holm-corrected',
        'comparisons': comparisons,
        # Backward-compatible summary fields (vs Energy-Only when available).
        'p_value': headline['p_holm'],
        'mean_difference': headline['mean_difference'],
        'cohens_d': headline['cohens_dz'],
        'significant': headline['significant'],
        'sample_size': headline['n_runs'],
    }


# ---- Cache weather to avoid re-fetching every rerun ----
@st.cache_data(ttl=300, show_spinner=False)  # Current weather: five-minute cache.
def cached_weather(lat, lon, name, climate_hint=None):
    lat = round(float(lat), 4)
    lon = round(float(lon), 4)

    # One cache layer: a cache miss or explicit clear must reach the provider.
    data = fetch_weather_data(lat, lon, name, climate_hint, force_refresh=True)

    # ✅ Important: NEVER cache None / broken payloads
    if data is None or not isinstance(data, dict):
        raise RuntimeError(f"Weather fetch failed (None/invalid) for {name} ({lat}, {lon})")

    # Accept current-only payloads from Open-Meteo; don't require hourly/daily.
    if all(key not in data for key in ("temperature", "humidity", "wind_speed")):
        raise RuntimeError(f"Weather response missing current fields for {name} ({lat}, {lon})")

    if "error" in data:
        raise RuntimeError(f"Weather API error for {name}: {data['error']}")

    return data

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Render the focused dashboard without changing research/model semantics."""
    from dashboard_ui import render_dashboard
    render_dashboard(globals())


if __name__ == "__main__":
    main()
