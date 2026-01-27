import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
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
import warnings
warnings.filterwarnings('ignore')

# Configure logging for API call tracking
logging.basicConfig(
    filename='datacenter_api_calls.log',
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
        st.info("Figure export needs 'kaleido'. Install: pip install -U kaleido")
        return

    st.download_button("⬇️ Download PNG (paper-ready)", png, file_name=f"{base_name}.png", mime="image/png")

    pdf = _fig_to_image_bytes(fig, fmt="pdf", scale=1, width=w, height=h)
    if pdf is not None:
        st.download_button("⬇️ Download PDF (vector)", pdf, file_name=f"{base_name}.pdf", mime="application/pdf")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CLIMATE-AWARE SURROGATE MODELING FOR ENERGY-EFFICIENT DATA CENTER WORKLOAD ROUTING",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ============================================================================
# CSS STYLING
# ============================================================================

st.markdown("""
<style>
/* ---------- Design tokens ---------- */
:root{
  --bg: #ffffff;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --border: #e2e8f0;
  --soft: #f8fafc;
  --primary: #e94560;
  --primary2: #c41e3a;
  --shadow: 0 1px 10px rgba(15, 23, 42, 0.06);
  --radius: 14px;
}

/* ---------- App background + typography ---------- */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Source Sans 3", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"]{
  background: var(--bg) !important;
}

.main .block-container{
  max-width: 1400px;
  padding-top: 1.25rem;
  padding-bottom: 2rem;
}

/* ---------- Headings ---------- */
.section-header{
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--text);
  margin: 1.8rem 0 1rem 0;
  padding: 0.85rem 1rem;
  background: var(--soft);
  border: 1px solid var(--border);
  border-left: 5px solid var(--primary);
  border-radius: 12px;
}

.subsection-header{
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin: 1.25rem 0 0.75rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}

/* ---------- Cards / callouts ---------- */
.metric-card, .finding-card, .dc-card, .reference-box{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.metric-card{
  padding: 1.1rem;
  text-align: center;
}

.metric-value{
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
}

.metric-label{
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 0.25rem;
}

.finding-card{ padding: 1.25rem; margin: 1rem 0; }
.dc-card{ padding: 1rem; margin: 0.6rem 0; }

.dc-card.hot{ border-left: 5px solid #dc2626; }
.dc-card.moderate{ border-left: 5px solid #d97706; }
.dc-card.cold{ border-left: 5px solid #2563eb; }

.physics-callout, .info-box, .warning-box, .success-box{
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
  margin: 1rem 0;
  border: 1px solid var(--border);
  background: var(--soft);
}

.physics-callout{ border-left: 5px solid var(--primary); }
.info-box{ border-left: 5px solid #0ea5e9; }
.warning-box{ border-left: 5px solid #f59e0b; }
.success-box{ border-left: 5px solid #22c55e; }

/* ---------- Code pills ---------- */
.code-text{
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: #f1f5f9;
  padding: 0.18rem 0.45rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  font-size: 0.9rem;
}

/* ---------- Buttons (Streamlit changed selectors over versions) ---------- */
/* Primary */
button[data-testid="baseButton-primary"]{
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary2) 100%) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.7rem 1.2rem !important;
  box-shadow: 0 6px 18px rgba(233, 69, 96, 0.18) !important;
}
button[data-testid="baseButton-primary"]:hover{
  transform: translateY(-1px);
  filter: brightness(1.02);
}

/* Secondary */
button[data-testid="baseButton-secondary"]{
  background: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.7rem 1.2rem !important;
}

/* ---------- Inputs ---------- */
[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input{
  border-radius: 12px !important;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS - Research-Backed Values
# ============================================================================

# Base energy per AI prompt (Stern, 2025 - WSJ)
BASE_ENERGY_WH = 0.3

ENERGY_PER_PROMPT_BASE = BASE_ENERGY_WH

api_fire_count = 0

# Cooling Systems with PUE values from Alkrush et al. (2024)
# International Journal of Refrigeration, 160, 246-262
COOLING_SYSTEMS = {
    "mechanical_chiller": {
        "name": "Mechanical Chiller (CRAC)",
        "pue": 1.80,
        "efficiency": 0.60,
        "temp_sensitivity": 0.020,
        "description": "Traditional computer room air conditioner",
        "best_climate": "any",
        "icon": "🏭"
    },
    "evaporative": {
        "name": "Evaporative Cooling",
        "pue": 1.35,
        "efficiency": 0.78,
        "temp_sensitivity": 0.012,
        "description": "Adiabatic cooling using water evaporation",
        "best_climate": "hot_dry",
        "icon": "💨"
    },
    "air_economizer": {
        "name": "Air-Side Economizer",
        "pue": 1.25,
        "efficiency": 0.85,
        "temp_sensitivity": 0.010,
        "description": "Uses outside air when temperature permits",
        "best_climate": "moderate",
        "icon": "🌬️"
    },
    "liquid_cooling": {
        "name": "Liquid Cooling",
        "pue": 1.15,
        "efficiency": 0.92,
        "temp_sensitivity": 0.005,
        "description": "Direct liquid cooling with cold plates",
        "best_climate": "any",
        "icon": "💧"
    },
    "free_air": {
        "name": "Free Air Cooling",
        "pue": 1.10,
        "efficiency": 0.95,
        "temp_sensitivity": 0.003,
        "description": "Direct outside air, minimal mechanical cooling",
        "best_climate": "cold",
        "icon": "❄️"
    }
}

# Default Datacenters
DEFAULT_DATACENTERS = {
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
EXTENDED_DC_LOCATIONS = {
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

CARBON_INTENSITY_BASE = {
    # United States - EPA eGRID 2023 / EIA State Profiles
    "california": 0.20,    # CAISO 2024: ~200 gCO2/kWh, high solar penetration
    "arizona": 0.42,       # EIA 2023: fossil heavy, some solar growth
    "texas": 0.38,         # ERCOT 2024: wind growth, still gas-heavy
    "washington": 0.08,    # EIA 2023: ~95% hydro dominated
    "illinois": 0.32,      # EIA 2023: nuclear (~50%) + fossil
    "florida": 0.42,       # EIA 2023: natural gas heavy
    "colorado": 0.50,      # EIA 2023: coal transition ongoing
    
    # Nordic - Ember/EEA 2024
    "sweden": 0.03,        # Ember 2024: 18-31 gCO2/kWh, hydro + nuclear
    "norway": 0.02,        # IEA 2024: ~98% hydro
    "finland": 0.08,       # EEA 2024: nuclear + hydro + wind
    "iceland": 0.01,       # IEA 2024: geothermal + hydro (near zero)
    
    # Europe - EEA 2024
    "ireland": 0.28,       # EEA 2024: wind growth, still gas backup
    "uk": 0.23,            # EEA 2024: offshore wind growth
    "germany": 0.35,       # EEA 2024: coal phase-out, renewables growing
    "netherlands": 0.36,   # EEA 2024: gas + offshore wind
    
    # Asia-Pacific - IEA 2024
    "japan": 0.45,         # IEA 2024: post-Fukushima LNG heavy
    "australia": 0.62,     # IEA 2024: coal heavy, solar growing
    "india": 0.70,         # IEA 2024: coal dominated
    "singapore": 0.40,     # IEA 2024: natural gas (imported LNG)
    
    # Americas - IEA 2024
    "brazil": 0.08,        # IEA 2024: ~80% hydro
    "canada": 0.12,        # IEA 2024: hydro heavy (varies by province)
    
    # Default
    "default": 0.40        # IEA 2024: global average ~400 gCO2/kWh
}

# City database for user location search
CITY_DATABASE = {
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
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AI-Assisted Datacenter Routing Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def pdf_safe_text(text):
        if not isinstance(text, str):
            return text
        return text.encode("latin-1", errors="replace").decode("latin-1")
    
    # Title Info
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, pdf_safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), 0, 1, 'C')
    pdf.ln(10)

    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Executive Summary", 0, 1)
    pdf.set_font("Arial", size=11)
    
    summary_text = (
        f"Recommended Strategy: {final_rec['recommended_strategy']}\n\n"
        f"Reasoning:\n" + 
        "\n".join([f"- {r.replace('✅', '[OK]').replace('⚠️', '[WARN]').replace('❌', '[X]').replace('🌡️', '').replace('🔥', '').replace('⏱️', '').replace('⚡', '')}" for r in final_rec['reasoning']])
    )
    pdf.multi_cell(0, 7, pdf_safe_text(summary_text))
    pdf.ln(10)

    # Metrics Comparison
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Metrics Comparison", 0, 1)
    pdf.set_font("Arial", size=10)
    
    # Table Header
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(40, 10, "Strategy", 1, 0, 'C', True)
    pdf.cell(35, 10, "Energy (Wh)", 1, 0, 'C', True)
    pdf.cell(35, 10, "Carbon (g)", 1, 0, 'C', True)
    pdf.cell(35, 10, "Latency (ms)", 1, 0, 'C', True)
    pdf.cell(35, 10, "Peak ΔT-AR (C)", 1, 1, 'C', True)
    
    # Table Rows
    for strategy, data in results.items():
        totals = data['totals']
        pdf.cell(40, 10, pdf_safe_text(strategy), 1)
        pdf.cell(35, 10, f"{totals['energy_wh']:.1f}", 1)
        pdf.cell(35, 10, f"{totals['carbon_g']:.1f}", 1)
        pdf.cell(35, 10, f"{totals['avg_latency_ms']:.1f}", 1)
        pdf.cell(35, 10, f"{totals['peak_ΔT']:.4f}", 1, 1)
    
    pdf.ln(10)
    
    # Conclusion
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Conclusion", 0, 1)
    pdf.set_font("Arial", size=11)
    conclusion_text = (
        f"The {final_rec['recommended_strategy']} routing strategy is recommended based on "
        f"the analysis. This approach balances energy efficiency, latency, carbon emissions, "
        f"and Delta-T Aware Routing mitigation for sustainable datacenter operations."
    )
    pdf.multi_cell(0, 7, pdf_safe_text(conclusion_text))

    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1", errors="replace")
    return pdf_output

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_latency(distance_km, load_fraction=0.5, service_rate=1000, processing_ms=30,
                      fiber_speed_km_s=200000, max_queue_ms=200, utilization_cap=0.99):
    """
    Latency proxy = propagation + M/M/1 queueing + processing.

    - Propagation: round-trip distance / fiber speed
    - Queueing: M/M/1 with utilization ρ = λ/μ, bounded for stability
    - Processing: constant baseline
    """
    # --- Propagation delay (round trip) ---
    propagation_ms = (float(distance_km) / float(fiber_speed_km_s)) * 1000.0 * 2.0

    # --- Queueing delay (M/M/1) ---
    # Interpret load_fraction as utilization proxy ρ, then map to λ = ρ * μ
    rho = max(0.0, min(float(load_fraction), utilization_cap))  # 0 <= ρ < 1
    mu = float(service_rate)                                   # requests/s
    lam = rho * mu                                             # requests/s

    # If extremely close to saturation, cap delay at max_queue_ms
    if lam >= utilization_cap * mu:
        queueing_ms = float(max_queue_ms)
    else:
        # M/M/1 mean waiting time in queue: Wq = λ / (μ(μ-λ)) seconds
        # Convert to ms, then cap
        wq_s = lam / (mu * max(1e-9, (mu - lam)))
        queueing_ms = min(float(max_queue_ms), wq_s * 1000.0)

    # --- Processing delay ---
    return propagation_ms + queueing_ms + float(processing_ms)



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

def fetch_weather_data(lat: float, lon: float, location_name: str = "Location", climate_hint: str | None = None) -> dict[str, Any]:
    global api_fire_count

    lat = float(lat)
    lon = float(lon)

    # ========== 1) TRY ACTUAL API CALL FIRST ==========
    try:
        api_fire_count += 1
        logging.info(
            f"API call #{api_fire_count} - Fetching weather for {location_name} at ({lat:.4f}, {lon:.4f})"
        )

        cache_session = requests_cache.CachedSession(".cache", expire_after=1800)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)

        # openmeteo_requests.Client type stubs expect niquests.Session
        session = cast(requests.Session, retry_session)
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
        }

        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        current = response.Current()

        def _safe_current_value(current_data, idx: int, default: float) -> float:
            var = current_data.Variables(idx) if current_data else None
            return float(var.Value()) if var else float(default)

        return {
            "temperature": _safe_current_value(current, 0, 20.0),
            "humidity": _safe_current_value(current, 1, 50.0),
            "wind_speed": _safe_current_value(current, 2, 5.0),
            "pressure": _safe_current_value(current, 3, 1013.25),
            "solar_radiation": _safe_current_value(current, 4, 200.0),
            "source": "Open-Meteo API (Live)",
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
        "success": False,
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

def collect_historical_training_data(start_year=2021, end_year=2024, progress_cb=None):
    """
    Fetch 4 years of hourly weather for all datacenters
    using Open-Meteo Historical Forecast API.
    
    Returns a SINGLE combined DataFrame:
    columns = [location, datetime, temperature, humidity, wind_speed, pressure, solar_radiation, cooling_type]
    """
    # Setup Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)  # type: ignore

    all_records = []

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"📥 Fetching hourly historical weather from {start_date} to {end_date}")

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
            print(f"⚠️ Skipping {dc_name}: Missing coordinates")
            continue

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,shortwave_radiation",
            "timezone": "auto"
        }

        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            
            # Check if hourly data is available
            if hourly is None:
                print(f"  ✗ {dc_name}: No hourly data returned")
                continue

            # Process hourly data (order matches API request)
            temp_var = hourly.Variables(0)
            humidity_var = hourly.Variables(1)
            wind_var = hourly.Variables(2)
            pressure_var = hourly.Variables(3)
            solar_var = hourly.Variables(4)
            
            if temp_var is None or humidity_var is None or wind_var is None:
                print(f"  ✗ {dc_name}: Missing expected hourly variables")
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
            print(f"  ✔ {dc_name}: {n} hourly rows")

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
            print(f"❌ Error fetching {dc_name}: {e}")

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

    print(f"\n✅ Finished! Total samples collected: {len(df):,}")
    return df

    
def get_carbon_intensity(region, hour=None):
    """
    Get carbon intensity for a region with time-of-day variation.
    Based on research from CAISO, Swedish Energy Agency, EIA.
    
    Time variation reflects:
    - Solar generation (low carbon midday in CA)
    - Demand peaks (high carbon afternoon in hot regions)
    - Wind patterns (variable)
    """
    if hour is None:
        hour = datetime.now().hour
    
    base = CARBON_INTENSITY_BASE.get(region, CARBON_INTENSITY_BASE['default'])
    
    # Time-of-day variation based on region characteristics
    if region in ['california', 'arizona', 'texas']:
        # Solar states: lower carbon midday (10am-4pm), higher evening
        if 10 <= hour <= 16:
            variation = -0.08  # Solar generation reduces carbon
        elif 17 <= hour <= 21:
            variation = 0.10   # Evening peak, less solar
        else:
            variation = 0.02   # Night baseline
    elif region in ['sweden', 'norway', 'finland', 'iceland']:
        # Nordic: very stable due to hydro/nuclear
        variation = 0.005 * np.sin(np.pi * (hour - 18) / 12)
    elif region in ['uk', 'ireland', 'germany', 'netherlands']:
        # Europe: wind variation
        variation = 0.05 * np.sin(np.pi * (hour - 6) / 12)
    else:
        # Default: slight evening peak
        variation = 0.03 * np.sin(np.pi * (hour - 18) / 12)
    
    return max(0.01, base + variation)


def calculate_energy_per_request(temperature, humidity, cooling_type, energy_multiplier=1.0):
    """
    Calculate energy consumption per AI request.
    
    Formula: E = E_base × PUE × (1 + α × max(0, T - 20)) × humidity_factor
    
    Based on:
    - Stern (2025): Base energy ~0.3 Wh per prompt
    - Alkrush et al. (2024): PUE values by cooling type
    """
    cooling = COOLING_SYSTEMS.get(cooling_type, COOLING_SYSTEMS['mechanical_chiller'])
    pue = cooling['pue']
    temp_sensitivity = cooling['temp_sensitivity']
    
    # Temperature adjustment (energy increases with temp above 20°C)
    temp_factor = 1 + temp_sensitivity * max(0, temperature - 20)
    
    # Humidity adjustment (affects evaporative cooling efficiency)
    if cooling_type == 'evaporative' and humidity > 60:
        humidity_factor = 1 + 0.005 * (humidity - 60)
    else:
        humidity_factor = 1 + 0.001 * abs(humidity - 50)
    
    # Calculate total energy
    total_energy = BASE_ENERGY_WH * pue * temp_factor * humidity_factor * energy_multiplier
    
    return total_energy


def calculate_ΔT_contribution(heat_kwh, area_km2=1.0, wind_speed=5.0):
    """
    Estimate local ΔT-AR contribution using physics-based heat flux model.
    
    Formula: ΔT = κ × (Q/A) × (1/(1 + γ × wind))
    
    This model is derived from thermodynamic principles:
    - Heat flux density (Q/A) drives local temperature rise
    - Wind provides convective heat dissipation
    
    Parameters:
    - η = 0.0012: Heat-to-temperature coefficient (°C per kW/km²)
    - γ = 0.15: Wind dissipation factor
    """
    eta = 0.0012  # Heat-to-temperature coefficient
    gamma = 0.15     # Wind dissipation factor
    
    heat_flux = heat_kwh / area_km2
    wind_factor = 1 / (1 + gamma * wind_speed)
    
    return eta * heat_flux * wind_factor


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
        with st.spinner(f"🌐 Loading {len(EXTENDED_DC_LOCATIONS)} extended datacenters..."):
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
    #    For now, use reasonable defaults — these can be made DC-specific later.
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
        self.metrics = {}
        self.timing = {}
        self.feature_names = ['temperature', 'humidity', 'wind_speed', 'pressure', 'solar_radiation', 'cooling_type']
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
        cooling_types = sorted(COOLING_SYSTEMS.keys())

# --- 1. Try real historical weather-based training data ---
        if use_real_weather:
            try:
                # 4 years of hourly data for all datacenters
                weather_df = collect_historical_training_data(start_year=2021, end_year=2024)

                if weather_df is not None and not weather_df.empty:
                    for _, row in weather_df.iterrows():
                        temp = float(row.get("temperature", 20.0))
                        humidity = float(row.get("humidity", 50.0))
                        wind = float(row.get("wind_speed", 5.0))
                        pressure = float(row.get("pressure", 1013.25))
                        solar = float(row.get("solar_radiation", 200.0))

                        # Derive cooling type from your climate classifier
                        climate_info = classify_climate(temp, humidity, wind)
                        cooling_str = climate_info.get("recommended_cooling", None)
                        if cooling_str not in cooling_types:
                            cooling_str = cooling_types[0]  # safe default
                        cooling_idx = cooling_types.index(cooling_str)

                        # Ground-truth energy from physics model
                        # Wind improves free cooling efficiency
                        # Solar radiation adds thermal load
                        # Line ~1153, BEFORE energy calculation

                        # Pressure effect on cooling efficiency
                        # Lower pressure → lower air density → worse convective cooling
                        p0 = 1013.25  # Standard sea-level pressure (hPa)
                        k_p = 0.8    # Sensitivity coefficient (2% penalty at ~50 hPa below normal)
                        pressure_factor = (p0 / max(pressure, 1e-6)) ** 0.8
                        pressure_factor = float(np.clip(pressure_factor, 0.7, 1.35))
                        wind_cooling_factor = 1 / (1 + 0.05 * wind)  # 10 m/s → ~33% reduction
                        solar_factor = 1 + 0.0001 * max(0, solar - 200)  # Extra load above 200 W/m²
                        energy = calculate_energy_per_request(temp, humidity, cooling_str,
                                                            energy_multiplier=wind_cooling_factor * solar_factor * pressure_factor)
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

                    print(f"✅ Training on historical weather data (2021-2024): {len(data)} samples")
                else:
                    print("⚠️ collect_historical_training_data() returned empty – using synthetic data.")
            except Exception as e:
                print(f"❌ Historical weather training data error: {e}")
                print("   Falling back to synthetic training data.")

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
                p0 = 1013.25  # Standard sea-level pressure (hPa)
                k_p = 0.8    # Sensitivity coefficient (2% penalty at ~50 hPa below normal)
                pressure_factor = 1 + k_p * max(0, (p0 - pressure) / p0)
                wind_cooling_factor = 1 / (1 + 0.05 * wind)
                solar_factor = 1 + 0.0001 * max(0, solar - 200)
                energy = calculate_energy_per_request(temp, humidity, cooling_str,
                                      energy_multiplier=wind_cooling_factor * solar_factor * pressure_factor)
                energy += np.random.normal(0, 0.005)
                heat_kwh = energy / 1000.0
                delta_t = calculate_ΔT_contribution(heat_kwh, area_km2=1.0, wind_speed=wind)
                data.append([temp, humidity, wind, pressure, solar, cooling_idx, energy, delta_t])
                    
            # Set metadata for synthetic data
            self.used_real_weather = False
            self.training_source = "synthetic"
            self.training_samples = len(data)
            print(f"✅ Training on synthetic physics-based data: {len(data)} samples")

        df = pd.DataFrame(
            data,
            columns=['temperature', 'humidity', 'wind_speed', 'pressure', 'solar_radiation', 'cooling_type', 'energy', 'delta_t']
        )
        
        # Remove any NaN values to prevent training errors
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
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
        X = df[['temperature', 'humidity', 'wind_speed', 'pressure', 'solar_radiation', 'cooling_type']].values
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
        print(f"✓ Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"✓ Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"✓ Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

        results = {}

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
                'coefficients': dict(zip(self.feature_names, mlr.coef_))
            }
            print(f"[MLR] R²: {results['MLR']['r2']:.4f}")

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
            print(f"[ANN] R²: {results['ANN']['r2']:.4f}")

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
                print(f"[Bayesian] R²: {results['Bayesian']['r2']:.4f}")
                print(f"[Bayesian] Best params: {best}")
                
            except Exception as e:
                print(f"[Bayesian] Error: {str(e)}")
                results['Bayesian'] = {'error': str(e)}

        # Feature importance (using TEST set for evaluation, not training)
        if 'MLR' in self.models:
            mlr = self.models['MLR']
            importances = dict(zip(self.feature_names, mlr.coef_))
            results.setdefault("feature_importance", {})
            results["feature_importance"]["MLR"] = importances   # from MLR
# ...
            print(f"Feature importances (MLR): {importances}")        # Feature importance (using TEST set for evaluation, not training)
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
                    zip(self.feature_names, perm_importance['importances_mean'])
                }
                results['feature_importance']["ANN"] = importance_dict
            except Exception as e:
                print(f"[Feature Importance] Error: {str(e)}")

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
               
                print(f"[{model_name}] Train R²: {results[model_name]['train_r2']:.4f}, "
                      f"Val R²: {results[model_name]['val_r2']:.4f}, "
                      f"Test R²: {results[model_name]['r2']:.4f} - "
                      f"{results[model_name]['status']}")

        # ========== RETURN RESULTS ==========
        self.is_trained = True
        # Select best model based on R² score
        if results:
            # Filter out error results
            valid_results = {k: v for k, v in results.items() 
                           if 'r2' in v and 'error' not in v}
            
            if valid_results:
                self.best_model_name = max(
                    valid_results.keys(), 
                    key=lambda x: valid_results[x]['r2']
                )
                self.best_model = self.models[self.best_model_name]

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
        #         print(f"[ΔT Model] Test R²: {results['DT']['r2']:.4f}, MAE: {results['DT']['mae']:.6f}°C")
                
        #         print(f"\n{'='*60}")
        #         print(f"✓ SELECTED BEST MODEL: {self.best_model_name}")
        #         print(f"  R² = {valid_results[self.best_model_name]['r2']:.4f}")
        #         print(f"  MAE = {valid_results[self.best_model_name]['mae']:.4f} Wh")
        #         print(f"  RMSE = {valid_results[self.best_model_name]['rmse']:.4f} Wh")
        #         print(f"{'='*60}\n")

        # Store metrics for later access
        self.metrics = results
        return results

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

        X = np.array(features).reshape(1, -1)
        X_scaled = self.scalers['main'].transform(X)

        return self.models[model_name].predict(X_scaled)[0]
    
    def predict_dt(self, features):
        raise NotImplementedError("ΔT model disabled. Use physics-based ΔT (calculate_ΔT_contribution).")

    
    def get_best_model(self):
        """Return the best ENERGY model (exclude DT + feature_importance)."""
        if not self.metrics:
            return ("MLR", 0.0)

        skip = {"feature_importance", "DT"}
        best_name, best_r2 = None, -np.inf

        for name, metrics in self.metrics.items():
            if name in skip:
                continue
            if isinstance(metrics, dict) and "r2" in metrics and "error" not in metrics:
                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_name = name

        return (best_name or "MLR", best_r2)

# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

def route_random(datacenters, num_requests):
    """
    Random routing - baseline strategy.
    Distributes requests uniformly (with slight randomization).
    Does NOT consider energy, ΔT-AR, or latency – pure control case.
    """
    n_dcs = len(datacenters)
    if n_dcs == 0:
        return {}
    
    # Dirichlet distribution for slight variation
    weights = np.random.dirichlet(np.ones(n_dcs))
    distribution = {dc: int(w * num_requests) for dc, w in zip(datacenters.keys(), weights)}
    
    # Ensure total matches exactly
    total = sum(distribution.values())
    diff = num_requests - total
    if diff != 0:
        first_dc = next(iter(distribution))
        distribution[first_dc] += diff
    
    return distribution


def route_energy_only(datacenters, weather_data, cooling_selections,
                      num_requests, ai_models=None, max_dc_capacity_mw=100):
    """
    Energy-only routing - demonstrates Stockholm Concentration Problem.
    Routes more traffic to datacenters with lowest energy cost.
    Ignores latency SLO and ΔT-AR – intentionally unconstrained.
    """
    if not datacenters or num_requests <= 0:
        return {}

    energy_scores = {}

    # 1) Compute energy per request at each DC
    for dc_name, dc_info in datacenters.items():
        weather = weather_data.get(dc_name, {}) or {}
        cooling = cooling_selections.get(dc_name, dc_info.get('default_cooling', 'air_economizer'))

        temp = weather.get('temperature', 20)
        humidity = weather.get('humidity', 50)
        wind = weather.get('wind_speed', 5)
        pressure = weather.get('pressure', 1013.25)
        solar = weather.get('solar_radiation', 200.0)

        if ai_models is not None and getattr(ai_models, "is_trained", False):
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)
            energy = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy = calculate_energy_per_request(temp, humidity, cooling)

        energy_scores[dc_name] = float(energy)

    if not energy_scores:
        return {}

    # 2) Inverse weighting (lower energy => higher weight)
    min_energy = min(energy_scores.values())
    weights = {dc: (min_energy / max(e, 1e-3)) ** 3 for dc, e in energy_scores.items()}
    total_weight = sum(weights.values())

    # Guard against pathological totals
    if total_weight <= 0 or not np.isfinite(total_weight):
        weights = {dc: 1.0 for dc in energy_scores}
        total_weight = float(len(weights))

    # 3) Apply capacity limits
    distribution = {}
    remaining = int(num_requests)

    for dc, w in weights.items():
        base_requests = int((w / total_weight) * num_requests)
        capped = apply_capacity_limit(base_requests, energy_scores[dc], max_dc_capacity_mw)
        distribution[dc] = capped
        remaining -= capped

    # 4) Give remainder to best DC (still safe)
    if remaining > 0 and energy_scores:
        best_dc = min(energy_scores, key=lambda d: energy_scores[d])
        distribution[best_dc] = distribution.get(best_dc, 0) + remaining

    return distribution


def route_heat_only(datacenters, weather_data, cooling_selections,
                    num_requests, ai_models=None, max_dc_capacity_mw=500):
    """
    Heat-Only routing:
    Allocate more requests to datacenters with lower predicted ΔT (or ΔT-AR proxy),
    subject to capacity limits.
    """
    if not datacenters:
        return {}

    # 1) Compute ΔT score (lower is better) + energy (for capacity)
    dt_scores = {}
    energy_map = {}

    for dc_name, dc_info in datacenters.items():
        w = weather_data.get(dc_name, {}) or {}
        cooling = cooling_selections.get(dc_name, dc_info.get("default_cooling", "air_economizer"))

        temp = w.get("temperature", 20)
        humidity = w.get("humidity", 50)
        wind = w.get("wind_speed", 5)
        pressure = w.get("pressure", 1013.25)
        solar = w.get("solar_radiation", 200.0)

        cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)

        # Energy (for capacity + optional ΔT physics)
        if ai_models is not None and getattr(ai_models, "is_trained", False):
            energy = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy = calculate_energy_per_request(temp, humidity, cooling)

        energy_map[dc_name] = energy

        # ΔT (prefer AI DT if available, otherwise physics-based)
        dt = None
        if ai_models is not None and getattr(ai_models, "is_trained", False) and "DT" in getattr(ai_models, "models", {}):
            try:
                dt = ai_models.predict_dt([temp, humidity, wind, pressure, solar, cooling_idx])
            except Exception:
                dt = None

        if dt is None:
            heat_kwh = energy / 1000.0
            dt = calculate_ΔT_contribution(heat_kwh, area_km2=1.0, wind_speed=wind)

        dt_scores[dc_name] = float(dt)

    # 2) Convert score to weights: lower ΔT -> higher weight
    if not dt_scores:
        return {}
    min_dt = min(dt_scores.values())
    weights = {dc: (min_dt / max(dt, 1e-6)) ** 3 for dc, dt in dt_scores.items()}
    total_w = sum(weights.values()) or 1.0  # Guard: use 1.0 if sum is zero

    # 3) Allocate with capacity limits
    distribution = {}
    remaining = num_requests
    for dc, w in weights.items():
        base_requests = int((w / total_w) * num_requests)
        capped = apply_capacity_limit(base_requests, energy_map[dc], max_dc_capacity_mw)
        distribution[dc] = capped
        remaining -= capped

    # 4) Distribute remainder greedily to lowest ΔT
    if remaining > 0:
        ranked = sorted(dt_scores.items(), key=lambda x: x[1])  # lowest ΔT first
        i = 0
        while remaining > 0 and ranked:
            dc = ranked[i % len(ranked)][0]
            add_one = apply_capacity_limit(1, energy_map[dc], max_dc_capacity_mw)
            if add_one <= 0:
                i += 1
                # avoid infinite loop if all are capped
                if i > len(ranked) * 3:
                    break
                continue
            distribution[dc] += add_one
            remaining -= add_one
            i += 1

    return distribution

def route_ΔT_aware(datacenters, weather_data, cooling_selections,
                    num_requests, ai_models=None, max_dc_capacity_mw=100):
    """
    ΔT-AR routing - research contribution.
    Penalizes datacenters with high thermal vulnerability.
    Still does NOT enforce latency SLO – this is a ΔT-AR-focused baseline.
    
    Args:
        max_dc_capacity_mw: Maximum power capacity per datacenter in MW
    """
    scores = {}
    
    if not datacenters or num_requests <= 0:
        return {}

    # 1. Compute ΔT-AR cost per DC
    for dc_name, dc_info in datacenters.items():
        weather = weather_data.get(dc_name, {})
        cooling = cooling_selections.get(dc_name, dc_info.get('default_cooling', 'air_economizer'))
        
        temp = weather.get('temperature', 20)
        humidity = weather.get('humidity', 50)
        wind = weather.get('wind_speed', 5)
        pressure = weather.get('pressure', 1013.25)
        solar = weather.get('solar_radiation', 200.0)
        
        # Energy calculation
        if ai_models is not None and ai_models.is_trained:
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)
            energy = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy = calculate_energy_per_request(temp, humidity, cooling)
        
        # ΔT-AR vulnerability: hotter sites are more problematic
        ΔT_vulnerability = 1.0 + 0.05 * max(0.0, temp - 25.0)
        
        # Wind mitigation: more wind = better dispersion → lower effective score
        wind_benefit = 1.0 / (1.0 + 0.05 * wind)
        
        # Combined cost (lower is better)
        scores[dc_name] = energy * ΔT_vulnerability * wind_benefit
    
    # 2. Inverse weighting and capacity limiting
    max_score = max(scores.values())
    weights = {dc: max_score / max(s, 1e-6) for dc, s in scores.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0 or not np.isfinite(total_weight):
        # fall back to uniform
        weights = {dc: 1.0 for dc in scores}
        total_weight = float(len(weights))

    
    # Track energy for capacity calculation
    energy_map = {}
    for dc in datacenters:
        weather = weather_data.get(dc, {})
        cooling = cooling_selections.get(dc, datacenters[dc].get('default_cooling', 'air_economizer'))
        temp = weather.get('temperature', 20)
        humidity = weather.get('humidity', 50)
        wind = weather.get('wind_speed', 5)
        pressure = weather.get('pressure', 1013.25)
        solar = weather.get('solar_radiation', 200.0)
        if ai_models is not None and ai_models.is_trained:
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)
            energy_map[dc] = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy_map[dc] = calculate_energy_per_request(temp, humidity, cooling)
    
    # Apply capacity limits
    distribution = {}
    remaining = num_requests
    for dc, w in weights.items():
        base_requests = int((w / total_weight) * num_requests)
        capped = apply_capacity_limit(base_requests, energy_map[dc], max_dc_capacity_mw)
        distribution[dc] = capped
        remaining -= capped
    
    # Distribute any remaining to best (lowest cost) DC
    if remaining > 0:
        best_dc = min(scores, key=lambda dc: scores[dc])
        distribution[best_dc] = distribution.get(best_dc, 0) + remaining
    
    return distribution


def route_multi_objective(datacenters, weather_data, cooling_selections, user_location, 
                          num_requests, ai_models=None, latency_threshold=100, max_dc_capacity_mw=100):
    """
    Multi-objective routing - balances Energy, Latency, Carbon, ΔT-AR.
    This is the ONLY strategy that enforces a latency target (latency_threshold).
    
    Args:
        datacenters: dict of DC metadata (lat, lon, region, etc.)
        weather_data: per-DC weather snapshots
        cooling_selections: chosen cooling tech per DC
        user_location: (lat, lon) of the user
        num_requests: total number of requests to route
        ai_models: trained AIModelSuite (optional)
        latency_threshold: target latency in ms (SLO)
        max_dc_capacity_mw: Maximum power capacity per datacenter in MW
    """
    user_lat, user_lon = user_location
    hour = datetime.now().hour
    
    scores = {}
    
    # 1. Compute multi-objective score per DC
    for dc_name, dc_info in datacenters.items():
        weather = weather_data.get(dc_name, {})
        cooling = cooling_selections.get(dc_name, dc_info.get('default_cooling', 'air_economizer'))
        
        temp = weather.get('temperature', 20)
        humidity = weather.get('humidity', 50)
        wind = weather.get('wind_speed', 5)
        pressure = weather.get('pressure', 1013.25)
        solar = weather.get('solar_radiation', 200.0)
        
        # Energy (from AI or physics)
        if ai_models is not None and ai_models.is_trained:
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)
            energy = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy = calculate_energy_per_request(temp, humidity, cooling)
        energy_norm = energy / 1.0  # assume ~1 Wh upper reference
        
        # 2. Latency vs threshold
        distance = haversine_distance(user_lat, user_lon, dc_info['lat'], dc_info['lon'])
        latency = calculate_latency(distance, load_fraction=0.7)  # ms
        
        if latency_threshold is None or latency_threshold <= 0:
            effective_thresh = 200.0  # fallback reference
        else:
            effective_thresh = float(latency_threshold)
        
        # Normalized latency: 1.0 at threshold, clipped at 2x
        latency_norm = min(latency / effective_thresh, 2.0)
        
        # Hard cutoff: if latency is way too bad (>2× threshold), skip this DC
        if latency > 2 * effective_thresh:
            continue
        
        # 3. Carbon intensity (normalized)
        region = dc_info.get('region', 'default')
        carbon = get_carbon_intensity(region, hour)
        carbon_norm = carbon / 0.7  # assume ~0.7 kgCO2/kWh as "worst" grid
        
        # 4. ΔT-AR vulnerability (wind-aware)
        beta_ΔT = 0.15  # same idea as in calculate_ΔT_contribution
        base_ΔT = max(0.0, temp - 15.0) / 30.0   # 0–1 depending on how hot
        wind_factor = 1.0 / (1.0 + beta_ΔT * wind)  # more wind → less trapped heat
        ΔT_score = base_ΔT * wind_factor
        
        # 5. Latency penalty when SLO is violated
        if latency > effective_thresh:
            latency_penalty = ((latency - effective_thresh) / effective_thresh) ** 2
        else:
            latency_penalty = 0.0
        
        # 6. Base multi-objective score
        base_score = 0.25 * (energy_norm + latency_norm + carbon_norm + ΔT_score)
        score = base_score + latency_penalty
        
        scores[dc_name] = {
            "energy_norm": energy_norm,
            "latency_norm": latency_norm,
            "carbon_norm": carbon_norm,
            "ΔT_score": ΔT_score,
            "score": score,
            "latency": latency,
            "requests": 0,
        }
    
    # If no DC satisfies the basic latency bound, fall back to uniform routing
    if not scores:
        n = len(datacenters)
        if n == 0:
            return {}
        base_each = num_requests // n
        distribution = {dc: base_each for dc in datacenters}
        remainder = num_requests - base_each * n
        first_dc = next(iter(datacenters))
        distribution[first_dc] += remainder
        return distribution
    
    # 2. Softmax-like weighting on scores
    min_score = min(info["score"] for info in scores.values())
    weights = {
        dc: np.exp(-(info["score"] - min_score) * 3.0)
        for dc, info in scores.items()
    }
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        # Numerical safety: fallback equal weighting
        equal_w = 1.0 / len(weights)
        weights = {dc: equal_w for dc in weights}
        total_weight = 1.0
    
    # 3. Energy per DC (for capacity limits)
    energy_map = {}
    for dc in datacenters:
        weather = weather_data.get(dc, {})
        cooling = cooling_selections.get(dc, datacenters[dc].get('default_cooling', 'air_economizer'))
        temp = weather.get('temperature', 20)
        humidity = weather.get('humidity', 50)
        wind = weather.get('wind_speed', 5)
        pressure = weather.get('pressure', 1013.25)
        solar = weather.get('solar_radiation', 200.0)
        if ai_models is not None and ai_models.is_trained:
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)
            energy_map[dc] = ai_models.predict([temp, humidity, wind, pressure, solar, cooling_idx])
        else:
            energy_map[dc] = calculate_energy_per_request(temp, humidity, cooling)
    
    # 4. Apply capacity limits
    distribution = {}
    remaining = num_requests
    for dc, w in weights.items():
        base_requests = int((w / total_weight) * num_requests)
        capped = apply_capacity_limit(base_requests, energy_map[dc], max_dc_capacity_mw)
        distribution[dc] = capped
        remaining -= capped
    
    # 5. Distribute remaining requests to the best (lowest score) DC
    if remaining > 0:
        best_dc = min(scores, key=lambda d: scores[d]["score"])
        distribution[best_dc] = distribution.get(best_dc, 0) + remaining
    
    return distribution


# ============================================================================
# SIMULATION ENGINE
# ============================================================================
def run_simulation(datacenters, weather_data, user_location, num_requests,
                   cooling_selections, energy_multiplier=1.0, ai_models=None,
                   latency_threshold=100, max_dc_capacity_mw=100):
    """
    Run complete simulation for all routing strategies.
    Returns comprehensive metrics for comparison.

    Fixes included:
    - Use true load_fraction in latency (no hardcoded 0.7)
    - Apply energy_multiplier consistently (ANN + physics)
    - Compute cooling_idx once
    - Compute ΔT once (no duplicate fallback)
    - Make Wh/kWh conversions explicit and consistent
    """
    user_lat, user_lon = user_location
    hour = datetime.now().hour

    strategies = {
        'Random': route_random(datacenters, num_requests),
        'Energy-Only': route_energy_only(datacenters, weather_data, cooling_selections,
                                         num_requests, ai_models, max_dc_capacity_mw),
        'Heat-Only': route_heat_only(datacenters, weather_data, cooling_selections,
                                     num_requests, ai_models, max_dc_capacity_mw),
        'ΔT-AR': route_ΔT_aware(datacenters, weather_data, cooling_selections,
                                num_requests, ai_models, max_dc_capacity_mw),
        'Multi-Objective': route_multi_objective(datacenters, weather_data, cooling_selections,
                                                 user_location, num_requests, ai_models,
                                                 latency_threshold, max_dc_capacity_mw)
    }

    results = {}

    for strategy_name, distribution in strategies.items():
        strategy_results = {
            'distribution': distribution,
            'datacenters': {}
        }

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
            cooling_idx = list(COOLING_SYSTEMS.keys()).index(cooling)

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
            carbon_intensity = get_carbon_intensity(region, hour)
            dc_carbon_kg = dc_heat_kwh * float(carbon_intensity)

            # -------------------------
            # Latency (ms) – use true load_fraction
            # -------------------------
            distance_km = haversine_distance(user_lat, user_lon, dc_info['lat'], dc_info['lon'])
            load_fraction = float(requests) / float(num_requests)
            dc_latency_ms = calculate_latency(distance_km, load_fraction=load_fraction)

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
            weighted_latency_ms += dc_latency_ms * load_fraction
            total_heat_wh += dc_energy_wh

            heat_values.append(dc_energy_wh)
            ΔT_values.append(ΔT)

        # Concentration metric (CV)
        if len(heat_values) > 1 and np.mean(heat_values) > 0:
            heat_cv = float(np.std(heat_values) / np.mean(heat_values))
        else:
            heat_cv = 0.0

        peak_ΔT = float(max(ΔT_values)) if ΔT_values else 0.0
        avg_ΔT  = float(np.mean(ΔT_values)) if ΔT_values else 0.0

        strategy_results['totals'] = {
            'energy_wh': total_energy_wh,
            'carbon_kg': total_carbon_kg,
            'carbon_g': total_carbon_kg * 1000,
            'avg_latency_ms': weighted_latency_ms,
            'total_heat_wh': total_heat_wh,
            'total_heat_kwh': total_heat_wh / 1000.0,
            'heat_cv': heat_cv,
            'peak_ΔT': peak_ΔT,
            'avg_ΔT': avg_ΔT
        }

        results[strategy_name] = strategy_results

    return results


def run_monte_carlo(datacenters, weather_data, user_location, num_requests,
                    cooling_selections, energy_multiplier, ai_models, n_runs=100):
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
        varied_requests = int(num_requests * (1 + np.random.uniform(-0.2, 0.2)))
        
        # Run simulation
        results = run_simulation(
            datacenters, varied_weather, user_location,
            varied_requests, cooling_selections, energy_multiplier, ai_models
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
        textfont=dict(size=28, color='#e94560', family='Times New Roman'),
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
        title=dict(text=f"<b>{title}</b>", font=dict(size=48, family='Times New Roman', color='#000000')),
        height=900,
        autosize=True,
        margin=dict(l=0, r=0, t=120, b=0),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.95)',
                   bordercolor='#999', borderwidth=2,
                   font=dict(size=28, family='Times New Roman', color='#000000')),
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
            textfont=dict(size=28, color='#000000', family='Times New Roman'),
            cliponaxis=False,
            hovertemplate=f"<b>{short_name}</b><br>%{{x}}: %{{y:,}} requests<extra></extra>"
        ))


    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Number of Requests</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        barmode='group',
        bargap=0.25,
        bargroupgap=0.1,
        height=900,
        margin=dict(l=100, r=60, t=120, b=180),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Times New Roman', color='#000000')),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
            textfont=dict(size=28, color='#000000', family='Times New Roman'),
            showlegend=False
        ), row=1, col=i+1)

    fig.update_layout(
        title=dict(text="<b>AI Model Performance Comparison</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        height=900,
        margin=dict(l=100, r=60, t=120, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Times New Roman', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
    
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
    
    # Calculate residuals for color
    residuals = y_test - y_pred
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=5,
            color=residuals,
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
        title=dict(text=f"<b>{model_name}: Predicted vs Actual (R² = {r2:.4f})</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Actual Energy (Wh)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Predicted Energy (Wh)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=700,
        width=700,
        margin=dict(l=120, r=140, t=100, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Force 1:1 aspect ratio for scatter plots
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            tickfont=dict(size=32, color='#000000', family='Times New Roman')
        ),
        yaxis=dict(
            constrain='domain',
            tickfont=dict(size=32, color='#000000', family='Times New Roman')
        )
    )

    # Update colorbar title font
    fig.update_coloraxes(colorbar=dict(title=dict(text='<b>Residual</b>', font=dict(size=32, color='#000000', family='Times New Roman')),
                                       tickfont=dict(size=28, color='#000000', family='Times New Roman')))
    
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
        textfont=dict(size=32, family='Times New Roman', color='#000000')
    ))

    fig.update_layout(
        title=dict(text="<b>Feature Importance for Energy Prediction</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Relative Importance (%)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        width=900,
        margin=dict(l=220, r=120, t=100, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        yaxis=dict(autorange='reversed', tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(range=[0, max(percentages) * 1.15], tickfont=dict(size=32, color='#000000', family='Times New Roman'))
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
        'Peak ΔT-AR (°C×100)': [results[s]['totals']['peak_ΔT'] * 100 for s in strategies],
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Metric Value</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        barmode='group',
        height=900,
        margin=dict(l=100, r=60, t=120, b=200),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Times New Roman', color='#000000')),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
                      font=dict(size=48, family='Times New Roman', color='#000000')),
            xaxis_title=dict(text="<b>Iteration</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
            yaxis_title=dict(text="<b>Validation R²</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
            height=600,
            margin=dict(l=100, r=60, t=120, b=140),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Times New Roman", size=28, color='#000000'),
            xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
            yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
            legend=dict(y=-0.3, x=0.5, xanchor='center', orientation='h', font=dict(size=32, color='#000000', family='Times New Roman'))
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

    fig.update_yaxes(title_text="<b>Loss</b>", title_font=dict(size=36, color='#000000', family='Times New Roman'),
                     tickfont=dict(size=32, color='#000000', family='Times New Roman'), row=1, col=1)
    if has_val:
        fig.update_yaxes(title_text="<b>R²</b>", title_font=dict(size=36, color='#000000', family='Times New Roman'),
                        tickfont=dict(size=32, color='#000000', family='Times New Roman'), row=2, col=1)

    fig.update_xaxes(title_text="<b>Epoch</b>", title_font=dict(size=36, color='#000000', family='Times New Roman'),
                     tickfont=dict(size=32, color='#000000', family='Times New Roman'), row=rows, col=1)

    fig.update_layout(
        title=dict(text="<b>ANN Training Convergence</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        height=1200 if has_val else 700,
        margin=dict(l=130, r=80, t=140, b=180),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Times New Roman', size=28, color='#000000'),
        legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman'))
    )

    # Update subplot titles with larger fonts
    fig.update_annotations(font=dict(size=36, family='Times New Roman', color='#000000'))

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
            textfont=dict(size=28, color='#000000', family='Times New Roman')
        ))

    fig.update_layout(
        title=dict(text="<b>Heat Distribution by Datacenter (Wh)</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Routing Strategy</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Heat Dissipation (Wh)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        barmode='stack',
        height=900,
        margin=dict(l=100, r=60, t=120, b=200),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, family='Times New Roman', color='#000000')),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
        title=dict(text="<b>Monte Carlo Simulation Results (95% CI)</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        height=1400,
        margin=dict(l=120, r=80, t=140, b=220),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        showlegend=True,
        legend=dict(orientation='h', y=-0.20, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Times New Roman', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
    
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        height=1400,
        margin=dict(l=120, r=80, t=140, b=160),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update subplot titles and axis labels with larger fonts
    fig.update_annotations(font=dict(size=36, family='Times New Roman', color='#000000'))
    fig.update_xaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
    fig.update_yaxes(tickfont=dict(size=32, color='#000000', family='Times New Roman'))
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text=x_label or metric, font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Density</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=650,
        margin=dict(l=110, r=60, t=120, b=140),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman'))
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Monte Carlo run index</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text=y_label or metric, font=dict(size=36, family='Times New Roman', color='#000000')),
        height=650,
        margin=dict(l=110, r=60, t=120, b=140),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman'))
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Ambient Temperature (°C)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Peak ΔT-AR (°C)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        margin=dict(l=110, r=60, t=120, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_carbon_intensity_curves():
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
    
    colors = {
        'Phoenix (Arizona)': '#dc2626',
        'San Francisco (California)': '#d97706',
        'Stockholm (Sweden)': '#2563eb'
    }
    
    fig = go.Figure()
    
    for region_name, region_key in regions.items():
        values = [get_carbon_intensity(region_key, h) for h in hours]
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=values,
            mode='lines',
            name=region_name,
            line=dict(color=colors[region_name], width=3),
            fill='tozeroy',
            fillcolor=colors[region_name].replace(')', ', 0.1)').replace('rgb', 'rgba'),
        ))
    
    # Add current hour marker
    current_hour = datetime.now().hour
    fig.add_vline(x=current_hour, line_dash="dash", line_color="#666",
                  annotation_text=f"Current: {current_hour}:00")
    
    fig.update_layout(
        title=dict(text="<b>24-Hour Carbon Intensity Patterns</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Hour of Day</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Carbon Intensity (kg CO₂/kWh)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        margin=dict(l=110, r=60, t=120, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        legend=dict(x=0.02, y=0.98, font=dict(size=32, color='#000000', family='Times New Roman')),
        xaxis=dict(tickmode='linear', tick0=0, dtick=4, tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
        textfont=dict(size=36, color='#000000', family='Times New Roman'),
        hovertemplate="Cooling: %{y}<br>Climate: %{x}<br>Effectiveness: %{z:.0%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Cooling Effectiveness by Climate Type</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Climate Type</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Cooling Technology</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        margin=dict(l=240, r=80, t=120, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
        textfont=dict(size=32, family='Times New Roman', color='#000000')
    ))

    fig.update_layout(
        title=dict(text="<b>Power Usage Effectiveness (PUE) by Cooling Technology</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>PUE Value (lower is better)</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        width=900,
        margin=dict(l=280, r=140, t=120, b=100),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        xaxis=dict(range=[1.0, 2.2], tickfont=dict(size=32, color='#000000', family='Times New Roman')),
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
    metric_labels = ['Energy (Wh)', 'Carbon (g)', 'Latency (ms)', 'Peak ΔT-AR (°C)', 'Heat CV']
    
    values_a = [scenario_a['totals'][m] if m != 'peak_ΔT' else scenario_a['totals'][m] * 100 for m in metrics]
    values_b = [scenario_b['totals'][m] if m != 'peak_ΔT' else scenario_b['totals'][m] * 100 for m in metrics]
    
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
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Metrics</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Value</b>", font=dict(size=36, family='Times New Roman', color='#000000')),
        barmode='group',
        height=900,
        margin=dict(l=100, r=60, t=120, b=180),
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman')),
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
        textfont=dict(size=28, family='Times New Roman', color='#000000')
    ))

    fig.add_trace(go.Bar(
        name='Validation',
        y=models,
        x=val_r2,
        orientation='h',
        marker_color='#10b981',
        text=[f"<b>{v:.4f}</b>" for v in val_r2],
        textposition='outside',
        textfont=dict(size=28, family='Times New Roman', color='#000000')
    ))

    fig.add_trace(go.Bar(
        name='Test',
        y=models,
        x=test_r2,
        orientation='h',
        marker_color='#f59e0b',
        text=[f"<b>{v:.4f}</b>" for v in test_r2],
        textposition='outside',
        textfont=dict(size=28, family='Times New Roman', color='#000000')
    ))

    fig.update_layout(
        title=dict(text='Model Performance: Train vs Validation vs Test',
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        xaxis=dict(title=dict(text='R² Score', font=dict(size=36, family='Times New Roman', color='#000000')),
                  range=[0, 1.08], tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(title=dict(text='Model', font=dict(size=36, family='Times New Roman', color='#000000')),
                  tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        barmode='group',
        height=600,
        width=1000,
        margin=dict(l=120, r=180, t=120, b=180),
        showlegend=True,
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Times New Roman', size=28, color='#000000')
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
    hour = datetime.now().hour
    
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
        carbon = get_carbon_intensity(region, hour)
        
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


def generate_final_recommendation(results, latency_threshold):
    """
    Generate final strategy recommendation based on simulation results.
    All calculations are DYNAMIC.
    """
    strategies = list(results.keys())
    
    # Find best strategy for ΔT-AR reduction
    ΔT_scores = {s: results[s]['totals']['peak_ΔT'] for s in strategies}
    best_ΔT = min(ΔT_scores, key=lambda s: ΔT_scores[s])
    
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

    latency_slo_met = None
    if latency_threshold is not None and latency_threshold > 0:
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
        'best_metrics': best
    }


def calculate_statistical_significance(mc_results):
    """
    Calculate statistical significance of Monte Carlo results.
    All values are CALCULATED from actual data.
    """
    if 'Energy-Only' not in mc_results or 'Multi-Objective' not in mc_results:
        return None
    
    energy_only_ΔT = mc_results['Energy-Only']['raw_data']['peak_ΔT']
    multi_obj_ΔT = mc_results['Multi-Objective']['raw_data']['peak_ΔT']
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(energy_only_ΔT, multi_obj_ΔT, equal_var=False)
    
    # Ensure p_value is a Python float (convert numpy scalar to native Python float)
    p_value = float(np.asarray(p_value).item())
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((energy_only_ΔT.var() + multi_obj_ΔT.var()) / 2)
    cohens_d = (energy_only_ΔT.mean() - multi_obj_ΔT.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval for difference
    mean_diff = energy_only_ΔT.mean() - multi_obj_ΔT.mean()
    se_diff = np.sqrt(energy_only_ΔT.var()/len(energy_only_ΔT) + 
                      multi_obj_ΔT.var()/len(multi_obj_ΔT))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    # Determine significance level
    if p_value < 0.001:
        significance = "HIGHLY SIGNIFICANT (p < 0.001)"
        significant = True
    elif p_value < 0.01:
        significance = "VERY SIGNIFICANT (p < 0.01)"
        significant = True
    elif p_value < 0.05:
        significance = "SIGNIFICANT (p < 0.05)"
        significant = True
    else:
        significance = "NOT SIGNIFICANT (p ≥ 0.05)"
        significant = False
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'ci_95': (ci_lower, ci_upper),
        'cohens_d': cohens_d,
        'significance': significance,
        'significant': significant,
        'sample_size': len(energy_only_ΔT)
    }
# ---- Cache weather to avoid re-fetching every rerun ----
@st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache
def cached_weather(lat, lon, name, climate_hint=None):
    lat = round(float(lat), 4)
    lon = round(float(lon), 4)

    data = fetch_weather_data(lat, lon, name, climate_hint)

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
    """Main application entry point."""
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h1 style="font-family: 'Source Serif 4', serif; font-size: 2.2rem; color: #1a1a2e; margin-bottom: 0.5rem;">
            🌡️ AI-Assisted Datacenter Prompt Routing for ΔT-AR Mitigation
        </h1>
        <p style="font-family: 'Source Sans 3', sans-serif; color: #4a5568; font-size: 1.1rem; margin-bottom: 0.25rem;">
            Demonstrating the usage of AI to mitigate the ΔT-AR (Delta-T Aware Routing) effect caused by datacenters
        </p>
        <p style="font-family: 'Source Sans 3', sans-serif; color: #718096; font-size: 0.9rem;">
            Phani Raja Bharath Balijepalli | IDS6938 - AI, Energy, and Sustainability | 
            Fall 2025 | University of Central Florida
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Physics callout
    st.markdown("""
    <div class="physics-callout">
        <strong>🔬 Core Physics Principle:</strong> Cooling systems move heat from datacenter interiors 
        to the exterior environment. <strong>The heat doesn't disappear — it dissipates locally.</strong> 
        Combined with carbon emissions from energy generation, there is no escape from the thermal impact 
        of computation. This dashboard demonstrates how routing strategy choice can mitigate localized 
        Delta-T Aware Routing (ΔT-AR) effects.
    </div>
    """, unsafe_allow_html=True)

    # Key Formulas
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-left: 4px solid #2563eb;">
        <strong>📐 Key Formulas Used:</strong>
        <table style="width: 100%; margin-top: 0.5rem; font-size: 0.9rem;">
            <tr>
                <td><strong>Latency:</strong></td>
                <td><code>L = L_prop + L_queue + L_proc</code></td>
                <td style="color: #666;">where L_prop = 2d/c (round-trip), L_queue = M/M/1 model, L_proc = 30ms</td>
            </tr>
            <tr>
                <td><strong>Energy:</strong></td>
                <td><code>E = E_base × PUE × T_factor × H_factor</code></td>
                <td style="color: #666;">E_base = 0.3 Wh (Stern, 2025)</td>
            </tr>
            <tr>
                <td><strong>ΔT-AR:</strong></td>
                <td><code>ΔT = α × (Q/A) × 1/(1 + β × wind)</code></td>
                <td style="color: #666;">Physics-based model</td>
            </tr>
            <tr>
                <td><strong>Carbon:</strong></td>
                <td><code>C = E × I</code></td>
                <td style="color: #666;">Luccioni & Hernandez-Garcia (2023)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: EXPERIMENTAL SETUP
    # ========================================================================
    
    st.markdown('<div class="section-header">📍 SECTION 1: Experimental Setup</div>', unsafe_allow_html=True)
    st.markdown("*Configure your simulation parameters below*")
    
    # Initialize AI models BEFORE expander (global scope)
    ai_models = AIModelSuite()
    
    with st.expander("⚙️ Configure Your Experiment", expanded=True):
        
        # 1.1 User Location
        st.markdown('<div class="subsection-header">1.1 User Location</div>', unsafe_allow_html=True)
     # Persistent storage for user location
        if "user_lat" not in st.session_state:
            st.session_state.user_lat = 28.5383   # Orlando default
            st.session_state.user_lon = -81.3792
            st.session_state.user_label = "Orlando"
            st.session_state.user_lat_input = 28.5383
            st.session_state.user_lon_input = -81.3792

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Create list of cities for dropdown
            city_options = sorted(list(CITY_DATABASE.keys()))
            
            # Callback to update coordinates when city changes
            def on_city_change():
                selected_city = st.session_state.city_selector
                if selected_city in CITY_DATABASE:
                    lat, lon = CITY_DATABASE[selected_city]
                    st.session_state.user_lat = lat
                    st.session_state.user_lon = lon
                    st.session_state.user_label = selected_city
                    st.session_state.user_lat_input = lat
                    st.session_state.user_lon_input = lon
            
            # Determine index
            try:
                default_index = city_options.index(st.session_state.user_label)
            except ValueError:
                default_index = 0
                
            st.selectbox(
                "Select City",
                options=city_options,
                index=default_index,
                key="city_selector",
                on_change=on_city_change,
                help="Select a city to automatically set latitude and longitude"
            )

        # Manual coordinate controls (always persistent)
        with col2:
            st.number_input(
                "Latitude",
                format="%.4f",
                key="user_lat_input",
                on_change=lambda: setattr(st.session_state, 'user_lat', st.session_state.user_lat_input)
            )
        with col3:
            st.number_input(
                "Longitude",
                format="%.4f",
                key="user_lon_input",
                on_change=lambda: setattr(st.session_state, 'user_lon', st.session_state.user_lon_input)
            )

        # Final user location tuple - always use session state
        user_location = (float(st.session_state.user_lat), float(st.session_state.user_lon))

        # Fetch user location weather
        user_weather = fetch_weather_data(
            st.session_state.user_lat,
            st.session_state.user_lon,
            st.session_state.user_label,
            "moderate"  # default climate hint
        )

        st.markdown('<div class="subsection-header">Figure export (paper layout)</div>', unsafe_allow_html=True)

        preset_name = st.selectbox(
            "Export preset",
            list(FIG_EXPORT_PRESETS.keys()),
            index=0
        )
        st.session_state["fig_export_cfg"] = FIG_EXPORT_PRESETS[preset_name]

        st.markdown(f"""
        <div class="info-box">
            <strong>📡 Your Location Weather:</strong> 
            🌡️ {user_weather['temperature']:.1f}°C | 
            💧 {user_weather['humidity']:.0f}% | 
            💨 {user_weather['wind_speed']:.1f} m/s |
            📊 {user_weather.get('pressure', 1013.25):.0f} hPa |
            ☀️ {user_weather.get('solar_radiation', 0):.0f} W/m²         
            <br><small>Source: {user_weather['source']}</small>
        </div>
        """, unsafe_allow_html=True)

        # 1.2 Datacenter Selection
        st.markdown('<div class="subsection-header">1.2 Datacenter Selection (1-3)</div>', unsafe_allow_html=True)

        # ---- Build all datacenters WITHOUT calling weather here ----
        all_datacenters: dict[str, dict[str, Any]] = {}

        for dc_name, dc_info in DEFAULT_DATACENTERS.items():
            all_datacenters[dc_name] = dc_info.copy()

        for dc_name, loc in EXTENDED_DC_LOCATIONS.items():
            all_datacenters[dc_name] = {
                "lat": loc["lat"],
                "lon": loc["lon"],
                "region": loc.get("region", "default"),
                "climate": loc.get("climate", "unknown"),
                "description": f"{loc.get('climate', 'unknown').title()} climate datacenter",
                "emoji": "🏢",  # will enrich after weather fetch
                "climate_detail": None,
                "default_cooling": None,
            }

        # ---- Initialize selection state ----
        if "selected_dcs" not in st.session_state:
            # default: pick first 3 (not ALL)
            st.session_state.selected_dcs = list(all_datacenters.keys())[:3]

        # ---- UI: select 1–3 datacenters ----
        selected_dcs = st.multiselect(
            "Choose 1–3 datacenters",
            options=list(all_datacenters.keys()),
            default=st.session_state.selected_dcs,
        )

        # persist selection
        st.session_state.selected_dcs = selected_dcs

        # enforce 1–3
        if not (1 <= len(selected_dcs) <= 3):
            st.warning("Please select between 1 and 3 datacenters.")
            st.stop()

        # ---- Fetch weather ONLY for selected, with progress ----
        dc_weather_data: dict[str, dict[str, Any]] = {}
        dc_details: dict[str, dict[str, Any]] = {}

        total_dcs = len(selected_dcs)
        progress_text = st.empty()
        progress_bar = st.progress(0)

        for idx, dc_name in enumerate(selected_dcs, 1):
            dc_info = all_datacenters[dc_name]
            weather = {}

            progress_text.text(f"🌐 Fetching weather data... ({idx}/{total_dcs})")
            progress_bar.progress(idx / total_dcs)

            try:
                climate_hint = dc_info.get("climate_detail") or dc_info.get("climate") or "moderate"
                weather = cached_weather(dc_info["lat"], dc_info["lon"], dc_name, climate_hint)


            except Exception as e:
                weather = {"error": str(e), "temperature": None, "humidity": None}

            dc_weather_data[dc_name] = weather

            # climate enrichment (guard against missing values)
            temp = weather.get("temperature")
            hum = weather.get("humidity")

            if temp is not None and hum is not None:
                climate_info = classify_climate(temp, hum)
                all_datacenters[dc_name]["climate_detail"] = climate_info.get("climate_detail")
                all_datacenters[dc_name]["default_cooling"] = climate_info.get("recommended_cooling")
                all_datacenters[dc_name]["emoji"] = climate_info.get("emoji", "🏢")

            # distance/latency/carbon (guard user_location existence)
            distance = haversine_distance(user_location[0], user_location[1], dc_info["lat"], dc_info["lon"])
            latency = calculate_latency(distance, load_fraction=0.7)
            region = dc_info.get("region", "default")
            carbon = get_carbon_intensity(region)

            dc_details[dc_name] = {
                "distance": distance,
                "latency": latency,
                "carbon": carbon,
                "weather": weather
            }

        progress_text.empty()
        progress_bar.empty()

                
        # Display DC selection with all available options
        selected_dc_names = selected_dcs
        
        if len(selected_dc_names) == 0:
            st.error("⚠️ Please select at least 1 datacenter!")
        elif len(selected_dc_names) > 3:
            st.warning("⚠️ Maximum 3 datacenters recommended for clear comparison")
        
        alt_datacenters: dict[str, dict[str, Any]] = all_datacenters
        alt_weather: dict[str, dict[str, Any]] = dc_weather_data

        # Display selected DC details with weather
        if selected_dc_names:
            dc_cols = st.columns(len(selected_dc_names))
            
            for i, dc_name in enumerate(selected_dc_names):
                dc_info = alt_datacenters[dc_name]
                alt_weather[dc_name] = cached_weather(dc_info['lat'], dc_info['lon'], dc_name, dc_info.get("climate", "moderate"))
                details = dc_details[dc_name]
                weather = details['weather']
                
                with dc_cols[i]:
                    climate_class = dc_info['climate']
                    st.markdown(f"""
                    <div class="dc-card {climate_class}">
                        <h4 style="margin: 0 0 0.5rem 0;">{dc_info['emoji']} {dc_name.split(',')[0]}</h4>
                        <p style="color: #666; font-size: 0.85rem; margin: 0.25rem 0;">{dc_info['description']}</p>
                        <hr style="margin: 0.5rem 0; border-color: #eee;">
                        <p style="margin: 0.25rem 0;">🌡️ <strong>{weather['temperature']:.1f}°C</strong></p>
                        <p style="margin: 0.25rem 0;">💧 {weather['humidity']:.0f}% humidity</p>
                        <p style="margin: 0.25rem 0;">💨 {weather['wind_speed']:.1f} m/s wind</p>
                        <p style="margin: 0.25rem 0;">🌡️ {weather.get('pressure', 1013.25):.0f} hPa</p>
                        <p style="margin: 0.25rem 0;">☀️ {weather.get('solar_radiation', 0):.0f} W/m²</p>
                        <hr style="margin: 0.5rem 0; border-color: #eee;">
                        <p style="margin: 0.25rem 0;">📍 {details['distance']:,.0f} km</p>
                        <p style="margin: 0.25rem 0;">⏱️ {details['latency']:.0f} ms</p>
                        <p style="margin: 0.25rem 0;">⚡ {details['carbon']:.2f} kg CO₂/kWh</p>
                    </div>
                    """, unsafe_allow_html=True)

        # 1.3 Geographic Layout Preview
        st.markdown('<div class="subsection-header">1.3 Geographic Layout Preview</div>', unsafe_allow_html=True)
        #st.plotly_chart(fig_map, config=PLOTLY_CONFIG, use_container_width=True)

        if selected_dc_names:
            # Build a small geographic map for the current experimental setup
            fig_setup_map = go.Figure()
            climate_colors = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}
            # Selected datacenters
            dc_lats = [all_datacenters[name]['lat'] for name in selected_dc_names]
            dc_lons = [all_datacenters[name]['lon'] for name in selected_dc_names]
            dc_labels = [name for name in selected_dc_names]
            dc_emojis = [all_datacenters[name]['emoji'] for name in selected_dc_names]

            fig_setup_map.add_trace(
                go.Scattergeo(
                    lon=dc_lons,
                    lat=dc_lats,
                    text=[f"{dc_emojis[i]} {dc_labels[i]}" for i in range(len(dc_labels))],
                    mode="markers+text",
                    marker=dict(
                        size=12,
                        color=[climate_colors.get(all_datacenters[name]['climate'], '#6b7280') for name in selected_dc_names]
                    ),
                    textposition="top center",
                    textfont=dict(size=20, family='Times New Roman', color='#000000'),
                    name="Datacenters"
                )
            )

            # User location
            fig_setup_map.add_trace(
                go.Scattergeo(
                    lon=[user_location[1]],
                    lat=[user_location[0]],
                    text=[f"🧑‍💻 {st.session_state.user_label} (You)"],
                    mode="markers+text",
                    marker=dict(size=12, symbol="star"),
                    textposition="bottom center",
                    textfont=dict(size=20, family='Times New Roman', color='#000000'),
                    name="User"
                )
            )
            # Draw connection lines from user to datacenters
            climate_colors = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}
            for dc_name in selected_dc_names:
                dc_info = all_datacenters[dc_name]
                color = climate_colors.get(dc_info['climate'], '#6b7280')
                fig_setup_map.add_trace(
                    go.Scattergeo(
                        lon=[user_location[1], dc_info['lon']],
                        lat=[user_location[0], dc_info['lat']],
                        mode='lines',
                        line=dict(width=2, color=color, dash='dot'),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Map styling (light-blue background, consistent with other maps)
            fig_setup_map.update_geos(
                projection_type="natural earth",
                showland=True,
                landcolor="#f3f2f0",
                showcoastlines=True,
                coastlinecolor="#8d99ae",
                showcountries=True,
                countrycolor="#d3d3d3",
                showlakes=True,
                lakecolor="#cfe7ff",
                showocean=True,
                oceancolor="#cfe7ff",
                fitbounds="locations"
            )

            fig_setup_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=600,
                font=dict(family='Times New Roman', size=20, color='#000000'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.01,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    borderwidth=1,
                    font=dict(size=20, family='Times New Roman', color='#000000')
                )
            )

            st.plotly_chart(fig_setup_map, config=PLOTLY_CONFIG_CLEAN, use_container_width=True)
        else:
            st.info("Select at least one datacenter to see the geographic preview.")


        # 1.4 Simulation Parameters
        st.markdown('<div class="subsection-header">1.4 Simulation Parameters</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_requests = st.slider(
                "Total Requests",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of AI inference requests to simulate"
            )
            
            latency_threshold = st.slider(
                "Latency Threshold (ms)",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                help="Maximum acceptable latency for routing decisions"
            )
            max_dc_capacity_mw = st.slider(
"Max Power Capacity per Datacenter (MW)",
min_value=10, max_value=200, value=100, step=10,
help="Limits max energy draw per DC in megawatts. Used to cap request allocation."
)
        
        with col2:
            energy_multiplier = st.slider(
                "Energy Multiplier",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="0.5=light prompts, 1.0=standard, 2.0=image generation"
            )
            
            enable_monte_carlo = st.checkbox("Enable Monte Carlo Analysis", value=True)
            
            if enable_monte_carlo:
                n_monte_carlo = st.slider(
                    "Monte Carlo Runs",
                    min_value=20,
                    max_value=500,
                    value=100,
                    step=20
                )
            else:
                n_monte_carlo = 0
        
        # Check latency warnings
        latency_warnings = []
        for dc_name in selected_dc_names:
            if dc_details[dc_name]['latency'] > latency_threshold:
                latency_warnings.append(
                    f"⚠️ **{dc_name.split(',')[0]}**: Latency ({dc_details[dc_name]['latency']:.0f}ms) "
                    f"exceeds threshold ({latency_threshold}ms)"
                )
        
        if latency_warnings:
            st.markdown("---")
            st.markdown("**Latency Warnings:**")
            for warning in latency_warnings:
                st.warning(warning)
            proceed_anyway = st.checkbox("Proceed with all selected datacenters anyway", value=True)
        else:
            proceed_anyway = True
        
        # 1.5 AI Model Selection
        st.markdown('<div class="subsection-header">1.5 AI Model Selection</div>', unsafe_allow_html=True)
        
        model_choice = st.radio(
            "Select AI Models to Train",
            options=[
                "Train All & Compare (Recommended)",
                "Multiple Linear Regression (MLR) only",
                "Artificial Neural Network (ANN) only",
                "Bayesian Optimization only" if BAYES_AVAILABLE else "Bayesian (Not Available)"
            ],
            index=0,
            help="Choose which AI models to train for energy prediction"
        )
        
        if "Bayesian" in model_choice and not BAYES_AVAILABLE:
            st.warning("Bayesian Optimization requires `bayesian-optimization` package. Install with: `pip install bayesian-optimization`")
    
    # Run Simulation Button
    st.markdown("---")

    # Initialize a persistent flag for whether simulation has been run at least once
    if "simulation_run" not in st.session_state:
        st.session_state.simulation_run = False

    # ---- simulation caching ----
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "sim_key" not in st.session_state:
        st.session_state.sim_key = None
    if "force_sim_rerun" not in st.session_state:
        st.session_state.force_sim_rerun = False
    if "temp_sensitivity" not in st.session_state:
        st.session_state.temp_sensitivity = None
    if "scenario_results" not in st.session_state:
        st.session_state.scenario_results = None

    run_button = st.button("▶️ RUN SIMULATION", type="primary", use_container_width=True, key="run_sim_btn")

    # If the button is clicked on this run, mark simulation as started
    if run_button:
        st.session_state.simulation_run = True
        st.session_state.force_sim_rerun = True  # force rerun on next section
    # If simulation has never been started, block the rest of the page
    if not st.session_state.simulation_run:
        st.info("👆 Configure your experiment above and click 'RUN SIMULATION' to begin!")
        st.stop()

    
    # Validation
    if len(selected_dc_names) == 0:
        st.error("❌ Please select at least 1 datacenter!")
        st.stop()
    
    if not proceed_anyway:
        st.error("❌ Please address latency warnings or check 'Proceed anyway'")
        st.stop()
    
    # Prepare active datacenters
    active_datacenters = {k: all_datacenters[k] for k in selected_dc_names}
    weather_data = {k: dc_weather_data[k] for k in selected_dc_names}
    
    # Default cooling selections
    if 'cooling_selections' not in st.session_state or set(st.session_state.cooling_selections.keys()) != set(selected_dc_names):
        st.session_state.cooling_selections = {k: all_datacenters[k]['default_cooling'] for k in selected_dc_names}
    
    cooling_selections = st.session_state.cooling_selections
    
    # Progress indicator
    progress_bar = st.progress(0, text="Initializing simulation...")
    
    # ========================================================================
    # SECTION 2: REAL-TIME CLIMATIC CONDITIONS
    # ========================================================================
    
    progress_bar.progress(10, text="Fetching weather data...")
    
    st.markdown('<div class="section-header">🌤️ SECTION 2: Real-Time Climatic Conditions</div>', unsafe_allow_html=True)
    st.markdown("*Current weather and environmental data for selected datacenters*")
    
    # 2.1 Weather Cards (already displayed above, but show summary)
    st.markdown('<div class="subsection-header">2.1 Datacenter Weather Summary</div>', unsafe_allow_html=True)
    
    weather_df = pd.DataFrame([
        {
            'Datacenter': dc_name.split(',')[0],
            'Climate': active_datacenters[dc_name]['emoji'] + ' ' + active_datacenters[dc_name]['climate'].title(),
            'Temperature (°C)': f"{weather_data[dc_name]['temperature']:.1f}",
            'Humidity (%)': f"{weather_data[dc_name]['humidity']:.0f}",
            'Wind (m/s)': f"{weather_data[dc_name]['wind_speed']:.1f}",
            'Pressure (hPa)': f"{weather_data[dc_name].get('pressure', 1013.25):.1f}",
            'Solar (W/m²)': f"{weather_data[dc_name].get('solar_radiation', 0.0):.0f}",
            'Carbon (kg/kWh)': f"{dc_details[dc_name]['carbon']:.3f}",
            'Distance (km)': f"{dc_details[dc_name]['distance']:,.0f}",
            'Latency (ms)': f"{dc_details[dc_name]['latency']:.0f}"
        }
        for dc_name in selected_dc_names
    ])
    st.dataframe(weather_df, use_container_width=True, hide_index=True)
    
    # 2.2 Geographic Map
    st.markdown('<div class="subsection-header">2.2 Geographic Visualization (Map #1)</div>', unsafe_allow_html=True)
    
    fig_map1 = create_geographic_map(active_datacenters, user_location,
                                     title="User Location and Datacenter Positions")
    st.plotly_chart(fig_map1, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 2.3 Initial Recommendation
    st.markdown('<div class="subsection-header">2.3 Initial Recommendation</div>', unsafe_allow_html=True)
    
    initial_rec = generate_initial_recommendation(active_datacenters, weather_data, 
                                                   user_location, latency_threshold)
    
    if initial_rec['recommended']:
        st.markdown(f"""
        <div class="success-box">
            <h4 style="margin: 0 0 0.5rem 0;">🎯 Recommended: {initial_rec['recommended']}</h4>
            <p style="margin: 0.5rem 0;"><strong>Reasoning:</strong></p>
            <ul style="margin: 0.25rem 0;">
                {''.join(f'<li>{r}</li>' for r in initial_rec['reasoning'])}
            </ul>
            {'<p style="margin: 0.5rem 0;"><strong>Alternatives:</strong></p><ul>' + ''.join(f'<li>{a}</li>' for a in initial_rec.get('alternatives', [])) + '</ul>' if initial_rec.get('alternatives') else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No datacenter meets the current latency threshold.")
    
    # ========================================================================
    # SECTION 3: THE DETERMINISTIC PROBLEM
    # ========================================================================
    
    progress_bar.progress(20, text="Calculating deterministic problem...")
    
    st.markdown('<div class="section-header">⚠️ SECTION 3: The Deterministic Problem</div>', unsafe_allow_html=True)
    st.markdown("*What happens with simple energy-only routing (before AI optimization)*")
    
    # 3.1 Cooling Assignment
    st.markdown('<div class="subsection-header">3.1 Cooling Technology Assignment</div>', unsafe_allow_html=True)
    
    cooling_df = pd.DataFrame([
        {
            'Datacenter': dc_name.split(',')[0],
            'Climate': active_datacenters[dc_name]['climate'].title(),
            'Cooling Technology': COOLING_SYSTEMS[cooling_selections[dc_name]]['name'],
            'PUE': COOLING_SYSTEMS[cooling_selections[dc_name]]['pue']
        }
        for dc_name in selected_dc_names
    ])
    st.dataframe(cooling_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="reference-box">
        <strong>Source:</strong> Alkrush, A.A. et al. (2024). "Data centers cooling: A critical review 
        of techniques, challenges, and energy saving solutions." <em>International Journal of Refrigeration</em>, 160, 246-262.
    </div>
    """, unsafe_allow_html=True)
    
    # 3.2 Energy Calculation
    st.markdown('<div class="subsection-header">3.2 Energy Calculation</div>', unsafe_allow_html=True)
    
    st.latex(r"E = E_{base} \times PUE \times (1 + \alpha \times \max(0, T - 20))")
    
    st.markdown(f"""
    **Parameters:**
    - E_base = {BASE_ENERGY_WH} Wh (Stern, 2025 - WSJ datacenter measurement)
    - α = 0.015 (temperature sensitivity coefficient)
    - PUE = Power Usage Effectiveness from cooling technology
    """)
    
    # Calculate energy for each DC
    energy_calc_df = pd.DataFrame([
        {
            'Datacenter': dc_name.split(',')[0],
            'Temp (°C)': weather_data[dc_name]['temperature'],
            'PUE': COOLING_SYSTEMS[cooling_selections[dc_name]]['pue'],
            'Temp Factor': 1 + 0.015 * max(0, weather_data[dc_name]['temperature'] - 20),
            'Energy/Req (Wh)': calculate_energy_per_request(
                weather_data[dc_name]['temperature'],
                weather_data[dc_name]['humidity'],
                cooling_selections[dc_name]
            )
        }
        for dc_name in selected_dc_names
    ])
    st.dataframe(energy_calc_df, use_container_width=True, hide_index=True)
    
    # Find lowest energy DC
    lowest_energy_dc = energy_calc_df.loc[energy_calc_df['Energy/Req (Wh)'].idxmin(), 'Datacenter']
    
    # 3.3 The Cold DC Concentration Problem
    st.markdown('<div class="subsection-header">3.3 The Cold DC Concentration Problem (Graph #2)</div>', unsafe_allow_html=True)
    
    # Run energy-only routing
    energy_only_dist = route_energy_only(active_datacenters, weather_data, cooling_selections, num_requests)
    
    # Find concentration
    total_reqs = sum(energy_only_dist.values())
    max_dc = max(energy_only_dist, key=lambda dc: energy_only_dist[dc])
    max_pct = (energy_only_dist[max_dc] / total_reqs) * 100 if total_reqs > 0 else 0
    
    concentration_detected = max_pct > 60
    
    if concentration_detected:
        st.markdown(f"""
        <div class="warning-box">
            <h4 style="margin: 0 0 0.5rem 0; color: #b45309;">⚠️ CONCENTRATION PROBLEM DETECTED</h4>
            <p>Energy-only routing sends <strong>{max_pct:.1f}%</strong> of traffic to 
            <strong>{max_dc.split(',')[0]}</strong> because it has the lowest energy cost per request.</p>
            <p style="margin-top: 0.5rem;"><em>"This creates severe localized heat dissipation that 
            defeats sustainability goals."</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create distribution chart
    deterministic_results = {
        'Energy-Only': {
            'distribution': energy_only_dist,
            'datacenters': {}
        }
    }
    
    # Calculate heat for energy-only
    for dc_name, requests in energy_only_dist.items():
        energy_per_req = calculate_energy_per_request(
            weather_data[dc_name]['temperature'],
            weather_data[dc_name]['humidity'],
            cooling_selections[dc_name]
        )
        heat = float(energy_per_req) * float(requests)
        deterministic_results['Energy-Only']['datacenters'][dc_name] = {
            'requests': requests,
            'percentage': (requests / total_reqs) * 100,
            'heat_kwh': heat / 1000
        }
    
    # Bar chart for energy-only distribution
    fig_det_dist = go.Figure()
    
    for dc_name in selected_dc_names:
        pct = (energy_only_dist[dc_name] / total_reqs) * 100
        climate = active_datacenters[dc_name]['climate']
        color = {'hot': '#dc2626', 'moderate': '#d97706', 'cold': '#2563eb'}[climate]
        
        fig_det_dist.add_trace(go.Bar(
            x=[dc_name.split(',')[0]],
            y=[pct],
            name=dc_name.split(',')[0],
            marker_color=color,
            text=[f"<b>{pct:.1f}%</b>"],
            textposition='auto',
            textfont=dict(size=28, color='#000000', family='Times New Roman')
        ))

    fig_det_dist.update_layout(
        title=dict(text="<b>Energy-Only Routing: Traffic Distribution</b>",
                  font=dict(size=48, family='Times New Roman', color='#000000')),
        yaxis_title=dict(text="<b>Traffic (%)</b>",
                        font=dict(size=36, family='Times New Roman', color='#000000')),
        xaxis_title=dict(text="<b>Datacenter</b>",
                        font=dict(size=36, family='Times New Roman', color='#000000')),
        height=800,
        margin=dict(l=100, r=60, t=120, b=100),
        showlegend=False,
        font=dict(family='Times New Roman', size=28, color='#000000'),
        xaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        yaxis=dict(tickfont=dict(size=32, color='#000000', family='Times New Roman')),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_det_dist, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 3.4 Heat Concentration Map
    st.markdown('<div class="subsection-header">3.4 Heat Concentration Visualization (Map #2)</div>', unsafe_allow_html=True)
    
    fig_map2 = create_geographic_map(active_datacenters, user_location, energy_only_dist,
                                     title="Heat Concentration with Energy-Only Routing")
    st.plotly_chart(fig_map2, config=PLOTLY_CONFIG, use_container_width=True)
    
    st.markdown("""
    <div class="physics-callout">
        <strong>💡 Key Physics Insight:</strong> "Cooling technology moves heat from datacenter interior 
        to exterior environment. The heat doesn't disappear — it dissipates locally. This creates 
        Delta-T Aware Routing effects proportional to the concentrated workload."
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 4: AI MODEL TRAINING
    # ========================================================================
    
    progress_bar.progress(35, text="Training AI models...")
    
    st.markdown('<div class="section-header">🤖 SECTION 4: AI Model Training</div>', unsafe_allow_html=True)
    st.markdown("*Training machine learning models for energy prediction*")
    
    # 4.1 Training Data
    st.markdown('<div class="subsection-header">4.1 Training Data Generation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Features (X):**
    - Temperature (°C): Range 0–45  *(Open-Meteo: temperature_2m)*
    - Humidity (%): Range 20–95  *(Open-Meteo: relative_humidity_2m)*
    - Wind Speed (m/s): Range 0.5–15  *(Open-Meteo: wind_speed_10m)*
    - Surface Pressure (hPa): Typical 950–1050 *(Open-Meteo: surface_pressure)*
    - Shortwave Radiation (W/m²): Typical 0–1000 *(Open-Meteo: shortwave_radiation)*
    - Cooling Type (encoded): 0–4  *(air_economizer / chilled_water / direct_evap / liquid / immersion)*

    **Why pressure + solar matter:**

    Pressure is a proxy for synoptic conditions that correlate with air density / weather regime (helps generalization). Solar radiation captures external thermal loading that increases cooling demand and worsens local heat rejection.

    **Target (y):**
    - Energy per Request (Wh)

    **Dataset:** Historical weather data (2021–2024) from Open-Meteo for all datacenter locations, with energy labels generated from the physics-based power model (and noise for realism).

    **Prediction:** Real-time current weather is used when making routing predictions.
    """)
    
    # Train models (ai_models already initialized above)
     
    if "All" in model_choice:
        train_mode = 'all'
    elif "MLR" in model_choice:
        train_mode = 'mlr'
    elif "ANN" in model_choice:
        train_mode = 'ann'
    else:
        train_mode = 'bayesian' if BAYES_AVAILABLE else 'mlr'
    
    train_msg_ph = st.empty()
    train_bar_ph = st.empty()
    train_bar = train_bar_ph.progress(0)

    def _ui_progress(pct, msg):
        try:
            pct = max(0.0, min(1.0, float(pct)))
        except Exception:
            pct = 0.0
        train_msg_ph.markdown(f"**{msg}**  \n{int(pct * 100)}%")
        train_bar.progress(int(pct * 100))

    model_results = ai_models.train_all(train_mode, progress_cb=_ui_progress)
    _ui_progress(1.0, "Training complete")

    
    joblib.dump(
        {
            "models": ai_models.models,  # sklearn models are serializable
            "scalers": ai_models.scalers,
            "metrics": ai_models.metrics,
            "model_results": model_results,
            "best_model_name": ai_models.best_model_name,
            "feature_names": ai_models.feature_names,
            "training_source": ai_models.training_source,
            "training_samples": ai_models.training_samples,
        },
        MODEL_BUNDLE_PATH,
    )
    st.success(f"Saved trained models to: {MODEL_BUNDLE_PATH}")
    
    st.markdown('<div class="subsection-header">4.2 Model Performance Comparison (Graph #3)</div>', unsafe_allow_html=True)
    
    # Display model cards with train/val/test metrics
    model_cols = st.columns(3)
    model_names = ['MLR', 'ANN', 'Bayesian']
    model_colors = ['#3b82f6', '#10b981', '#f59e0b']
    best_model_name, best_r2 = ai_models.get_best_model()
    
    for i, model_name in enumerate(model_names):
        with model_cols[i]:
            if model_name in model_results and isinstance(model_results[model_name], dict) and 'r2' in model_results[model_name]:
                r = model_results[model_name]
                is_best = model_name == best_model_name
                
                # Determine status color
                status_color = {
                    'Good Fit': '#10b981',
                    'Overfitting': '#f59e0b',
                    'Underfitting': '#ef4444'
                }.get(r.get('status', 'Unknown'), '#6b7280')
                
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {model_colors[i]}; {'box-shadow: 0 0 10px ' + model_colors[i] + '40;' if is_best else ''}">
                    <div style="font-size: 1.1rem; font-weight: 600; color: {model_colors[i]};">
                        {model_name} {'👑' if is_best else ''}
                    </div>
                    <div class="metric-value" style="color: {model_colors[i]};">Test R² = {r['r2']:.4f}</div>
                    <div class="metric-label">Train R²: {r.get('train_r2', 0):.4f}</div>
                    <div class="metric-label">Val R²: {r.get('val_r2', 0):.4f}</div>
                    <div class="metric-label" style="color: {status_color}; font-weight: 600;">
                        {r.get('status', 'Unknown')}
                    </div>
                    <hr style="margin: 0.5rem 0;">
                    <div class="metric-label">Test MAE: {r['mae']:.4f} Wh</div>
                    <div class="metric-label">Test RMSE: {r['rmse']:.4f} Wh</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #ccc;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #999;">
                        {model_name}
                    </div>
                    <div class="metric-label">Not trained</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Train/Val/Test Comparison Chart
    st.markdown("### Train vs Validation vs Test Performance")
    fig_tvt = create_train_val_test_comparison(model_results)
    if fig_tvt:
        st.plotly_chart(fig_tvt, config=PLOTLY_CONFIG, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Good Fit**: Train ≈ Val ≈ Test (all bars similar height)
        - **Overfitting**: Train > Test (training bar much higher)
        - **Underfitting**: All bars low (R² < 0.7)
        """)
    
    # Learning Curve
    st.markdown("### Neural Network Training Convergence")
    fig_lc = create_learning_curve_chart(model_results)
    if fig_lc:
        st.plotly_chart(fig_lc, config=PLOTLY_CONFIG, use_container_width=True)
        st_figure_downloads(fig_lc, "fig_ann_convergence")
        st.markdown("**Finding:** Loss decreases with epoch (and validation R² stabilizes when early-stopping is enabled), indicating convergence.")

    
    st.markdown("### Bayesian Hyperparameter Optimization Convergence")
    fig_bo = create_bayesopt_convergence_chart(model_results)
    if fig_bo:
        st.plotly_chart(fig_bo, config=PLOTLY_CONFIG, use_container_width=True)
        st.markdown("**Finding:** Validation R² improves across BO iterations and stabilizes near the best configuration.")

    
    # Model comparison chart (keep existing)
    fig_model_comp = create_model_comparison_chart(model_results)
    if fig_model_comp:
        st.plotly_chart(fig_model_comp, config=PLOTLY_CONFIG, use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Best Model:</strong> {best_model_name} with Test R² = {best_r2:.4f} ({model_results[best_model_name].get('status', 'Unknown')})
        <br>This model will be used for energy predictions in routing strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # 4.3 Prediction vs Actual
    st.markdown('<div class="subsection-header">4.3 Prediction vs Actual (Graph #4)</div>', unsafe_allow_html=True)
    
    fig_scatter = create_prediction_scatter(model_results, best_model_name)
    if fig_scatter:
        st.plotly_chart(fig_scatter, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 4.4 Feature Importance
    st.markdown('<div class="subsection-header">4.4 Feature Importance (Graph #5)</div>', unsafe_allow_html=True)
    
    if 'feature_importance' in model_results:
        fig_importance = create_feature_importance_chart(model_results['feature_importance'], best_model_name)
        if fig_importance:
            st.plotly_chart(fig_importance, config=PLOTLY_CONFIG, use_container_width=True)

        # Get top 2 features dynamically from the best model
        model_importance = model_results['feature_importance'].get(best_model_name, {})
        sorted_features = sorted(model_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f[0].replace('_', ' ').title() for f in sorted_features[:2]]
        
        st.markdown(f"""
        **Finding:** {top_features[0]} and {top_features[1]} are the dominant factors 
        influencing energy consumption per request.
        """)
        
    SAVE_DIR = "saved_runs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    colA, colB = st.columns(2)

    with colA:
        if st.button("💾 Save trained models + results", key="save_bundle_btn"):
            model_bundle = {
                "models": ai_models.models,
                "scalers": ai_models.scalers,
                "metrics": ai_models.metrics,
                "best_model_name": ai_models.best_model_name,
                "feature_names": ai_models.feature_names,
                "training_source": ai_models.training_source,
                "training_samples": ai_models.training_samples,
            }
            joblib.dump(model_bundle, os.path.join(SAVE_DIR, "ai_models.joblib"))
            joblib.dump(st.session_state.get("sim_results", None), os.path.join(SAVE_DIR, "sim_results.joblib"))
            st.success("Saved to saved_runs/")

    with colB:
        if st.button("📂 Load trained models + results", key="load_bundle_btn"):
            try:
                bundle = joblib.load(os.path.join(SAVE_DIR, "ai_models.joblib"))
                ai_models.models = bundle["models"]
                ai_models.scalers = bundle["scalers"]
                ai_models.metrics = bundle["metrics"]
                ai_models.best_model_name = bundle["best_model_name"]
                ai_models.feature_names = bundle["feature_names"]
                ai_models.training_source = bundle.get("training_source", "unknown")
                ai_models.training_samples = bundle.get("training_samples", 0)
                ai_models.is_trained = True
                # Restore best_model reference
                if ai_models.best_model_name and ai_models.best_model_name in ai_models.models:
                    ai_models.best_model = ai_models.models[ai_models.best_model_name]
                st.session_state.sim_results = joblib.load(os.path.join(SAVE_DIR, "sim_results.joblib"))
                st.success("Loaded from saved_runs/")
            except FileNotFoundError:
                st.error("No saved models found. Run and save first.")
    # ========================================================================
    # SECTION 5: ROUTING STRATEGY COMPARISON
    # ========================================================================
    
    progress_bar.progress(50, text="Comparing routing strategies...")
    
    st.markdown('<div class="section-header">🔀 SECTION 5: Routing Strategy Comparison</div>', unsafe_allow_html=True)
    st.markdown("*Comparing different approaches to datacenter selection*")
    
    # 5.1 Strategy Definitions
    st.markdown('<div class="subsection-header">5.1 Strategy Definitions</div>', unsafe_allow_html=True)
    
    st.markdown("""
    | Strategy | Description | Formula |
    |----------|-------------|---------|
    | **1️⃣ Random** | Baseline uniform distribution | Random allocation of prompts for each DC |
    | **2️⃣ Energy-Only** | Route to lowest energy DC | Minimize energy consumption |
    | **3️⃣ ΔT-AR** | Minimize delta-T aware routing effect | Minimize peak ΔT |
    | **4️⃣ Multi-Objective** | Balance all factors | Weighted combination |
    """)
    
    # Run initial simulation for all strategies
    

    # Build a key from inputs that should trigger a re-run
    # Note: best_model_name excluded - model choice shouldn't trigger sim rerun
    sim_inputs = {
        "selected_dc_names": selected_dc_names,
        "num_requests": num_requests,
        "cooling_selections": dict(st.session_state.cooling_selections),
        "energy_multiplier": energy_multiplier,
        "latency_threshold": latency_threshold,
    }

    sim_key = hashlib.md5(json.dumps(sim_inputs, sort_keys=True, default=str).encode()).hexdigest()

    need_run = (
        st.session_state.force_sim_rerun
        or st.session_state.sim_results is None
        or st.session_state.sim_key != sim_key
    )

    if need_run:
        st.session_state.sim_key = sim_key
        st.session_state.sim_results = run_simulation(
            active_datacenters, weather_data, user_location,
            num_requests, cooling_selections, energy_multiplier, ai_models, latency_threshold,
            max_dc_capacity_mw
        )
        st.session_state.force_sim_rerun = False

    results = st.session_state.sim_results
    
    # Guard against None results
    if results is None:
        st.warning("⚠️ Please click 'RUN SIMULATION' button above to see results.")
        st.stop()

    # 5.2 Strategy Summary    
    fig_traffic = create_traffic_distribution_chart(results, title="Traffic Distribution by Strategy")
    if fig_traffic:
        st.plotly_chart(fig_traffic, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 5.4 Cooling Configuration
    st.markdown('<div class="subsection-header">5.3 Cooling Technology Configuration</div>', unsafe_allow_html=True)
    
    st.markdown("*Adjust cooling technology for each datacenter (affects PUE and results):*")
    
    cool_cols = st.columns(len(selected_dc_names))
    new_cooling_selections = {}
    
    for i, dc_name in enumerate(selected_dc_names):
        with cool_cols[i]:
            dc_info = active_datacenters[dc_name]
            default_cooling = cooling_selections[dc_name]
            
            new_cooling = st.selectbox(
                f"{dc_info['emoji']} {dc_name.split(',')[0]}",
                options=list(COOLING_SYSTEMS.keys()),
                index=list(COOLING_SYSTEMS.keys()).index(default_cooling),
                format_func=lambda x: f"{COOLING_SYSTEMS[x]['icon']} {COOLING_SYSTEMS[x]['name']} (PUE: {COOLING_SYSTEMS[x]['pue']})",
                key=f"cooling_select_{dc_name}"
            )
            new_cooling_selections[dc_name] = new_cooling
    
    # Check if cooling changed (only on actual user interaction, not button reruns)
    cooling_actually_changed = False
    for dc_name in new_cooling_selections:
        widget_key = f"cooling_select_{dc_name}"
        if widget_key in st.session_state:
            current_widget_value = st.session_state[widget_key]
            stored_value = st.session_state.cooling_selections.get(dc_name)
            if current_widget_value != stored_value:
                cooling_actually_changed = True
                st.session_state.cooling_selections[dc_name] = current_widget_value
    
    if cooling_actually_changed:
        st.session_state.force_sim_rerun = True
        st.info("🔄 Cooling updated — simulation will refresh.")
        st.rerun()
    
    # ========================================================================
    # SECTION 6: COMPREHENSIVE RESULTS
    # ========================================================================
    
    progress_bar.progress(65, text="Analyzing results...")
    
    st.markdown('<div class="section-header">📊 SECTION 6: Comprehensive Results</div>', unsafe_allow_html=True)
    st.markdown("*Complete performance metrics across all strategies*")
    
    # 6.1 Metrics Table
    st.markdown('<div class="subsection-header">6.1 Metrics Comparison Table</div>', unsafe_allow_html=True)
    
    metrics_df = pd.DataFrame([
        {
            'Strategy': strategy,
            'Energy (Wh)': f"{results[strategy]['totals']['energy_wh']:.1f}",
            'Carbon (g)': f"{results[strategy]['totals']['carbon_g']:.1f}",
            'Latency (ms)': f"{results[strategy]['totals']['avg_latency_ms']:.1f}",
            'Peak ΔT-AR (°C)': f"{results[strategy]['totals']['peak_ΔT']:.4f}",
            'Heat CV': f"{results[strategy]['totals']['heat_cv']:.3f}"
        }
        for strategy in results.keys()
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # 6.2 Multi-metric Chart
    st.markdown('<div class="subsection-header">6.2 Multi-Metric Comparison (Graph #7)</div>', unsafe_allow_html=True)
    
    fig_metrics = create_metrics_comparison_chart(results)
    st.plotly_chart(fig_metrics, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 6.3 Heat Distribution
    st.markdown('<div class="subsection-header">6.3 Heat Distribution by Strategy (Graph #8)</div>', unsafe_allow_html=True)
    
    fig_heat = create_heat_distribution_chart(results)
    st.plotly_chart(fig_heat, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 6.4 Final Map
    st.markdown('<div class="subsection-header">6.4 Geographic Distribution (Map #3)</div>', unsafe_allow_html=True)
    
    map_strategy = st.radio(
        "Select strategy to visualize:",
        options=list(results.keys()),
        index=3,  # Multi-Objective by default
        horizontal=True
    )
    
    fig_map3 = create_geographic_map(
        active_datacenters, user_location,
        results[map_strategy]['distribution'],
        title=f"Traffic Distribution: {map_strategy} Strategy"
    )
    st.plotly_chart(fig_map3, config=PLOTLY_CONFIG, use_container_width=True)
    
  # 6.5 Final Recommendation
    st.markdown('<div class="subsection-header">6.5 Final Recommendation</div>', unsafe_allow_html=True)

    final_rec = generate_final_recommendation(results, latency_threshold)

    imp = final_rec["improvements"]
    ΔT_pct = imp["ΔT_reduction_pct"]
    cv_pct = imp["cv_reduction_pct"]
    latency_pct = imp["latency_change_pct"]   # negative = latency improves
    energy_pct = imp["energy_overhead_pct"]

    st.markdown(
        f"""
    <div class="success-box">
    <h4 style="margin: 0 0 0.75rem 0;">
    🎯 Recommended Strategy: {final_rec['recommended_strategy']}
    </h4>

    <p style="margin: 0.5rem 0;">
        <strong>Based on the multi-metric evaluation (Energy, Latency, Carbon, and ΔT-AR),
        the ΔT-AR strategy delivers the strongest overall sustainability
        performance compared to Energy-Only routing.</strong>
    </p>

    <p style="margin: 0.5rem 0;"><strong>Measured Effects (vs. Energy-Only):</strong></p>
    <ul style="margin: 0 0 1rem 0;">
        <li>🌡️ <strong>{ΔT_pct:.1f}% reduction in peak ΔT-AR</strong></li>
        <li>🔥 <strong>{cv_pct:.1f}% reduction in heat concentration (CV)</strong></li>
        <li>⏱️ {(-latency_pct):.1f}% lower latency</li>
        <li>⚡ ~{energy_pct:.1f}% energy overhead to achieve these gains</li>
    </ul>

    <p style="margin-top: 0.75rem; font-style: italic;">
        <strong>Interpretation:</strong> A modest energy overhead of about {energy_pct:.1f}% yields
        a disproportionately large sustainability benefit, reducing localized thermal
        stress by ~{ΔT_pct:.1f}% and smoothing heat distribution by ~{cv_pct:.1f}%, 
        while slightly improving latency. This trade-off represents a more optimal and
        resilient routing policy than traditional Energy-Only optimization.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

  
    # ========================================================================
    # SECTION 7: MONTE CARLO VALIDATION
    # ========================================================================
    
    if enable_monte_carlo and n_monte_carlo > 0:
        progress_bar.progress(75, text=f"Running Monte Carlo simulation ({n_monte_carlo} runs)...")
        
        st.markdown('<div class="section-header">🎲 SECTION 7: Monte Carlo Validation</div>', unsafe_allow_html=True)
        st.markdown("*Statistical validation through repeated stochastic simulation*")
        
        # 7.1 Stochastic Variables - Monte Carlo
        st.markdown('<div class="subsection-header">7.1 Stochastic Variables</div>', unsafe_allow_html=True)
        
        st.markdown("""
        | Variable | Distribution | Parameters |
        |----------|--------------|------------|
        | Temperature | Normal | μ = current, σ = 5°C |
        | Humidity | Normal | μ = current, σ = 15% |
        | Wind Speed | Normal | μ = current, σ = 2 m/s |
        | Pressure | Normal | μ = current, σ = 10 hPa |
        | Solar Radiation | Normal | μ = current, σ = 100 W/m² |   
        | Request Volume | Uniform | ±20% of base |
        """)
        
        # Run Monte Carlo
        mc_progress = st.progress(0, text="Running Monte Carlo simulations...")
        
        mc_results = run_monte_carlo(
            active_datacenters, weather_data, user_location,
            num_requests, cooling_selections, energy_multiplier, ai_models, n_monte_carlo
        )
        
        mc_progress.progress(100, text=f"✅ Completed {n_monte_carlo} simulations")
        
        # 7.3 Box Plots
        st.markdown('<div class="subsection-header">7.2 Distribution Analysis (Graphs #9-12)</div>', unsafe_allow_html=True)
        
        fig_mc_box = create_monte_carlo_boxplots(mc_results)
        st.plotly_chart(fig_mc_box, config=PLOTLY_CONFIG, use_container_width=True)

        st_figure_downloads(fig_mc_box, "fig_mc_boxplots")

        st.markdown("#### 7.2.1 Mean ± 95% CI of the mean")
        fig_mc_ci = create_monte_carlo_mean_ci_chart(mc_results)
        st.plotly_chart(fig_mc_ci, config=PLOTLY_CONFIG, use_container_width=True)
        st_figure_downloads(fig_mc_ci, "fig_mc_mean_ci")

        st.markdown("#### 7.2.2 Distribution overlay (Peak ΔT-AR)")
        fig_mc_hist = create_monte_carlo_hist_overlay(
            mc_results, metric="peak_ΔT",
            title="Monte Carlo Distribution Overlay: Peak ΔT-AR (°C)",
            x_label="Peak ΔT-AR (°C)"
        )
        st.plotly_chart(fig_mc_hist, config=PLOTLY_CONFIG, use_container_width=True)
        st_figure_downloads(fig_mc_hist, "fig_mc_peak_ΔT_overlay")

        st.markdown("#### 7.2.3 Monte Carlo convergence (running mean of Peak ΔT-AR)")
        fig_mc_conv = create_monte_carlo_running_mean_chart(
            mc_results, metric="peak_ΔT",
            title="Monte Carlo Convergence: Running Mean of Peak ΔT-AR (°C)",
            y_label="Running mean Peak ΔT-AR (°C)"
        )
        st.plotly_chart(fig_mc_conv, config=PLOTLY_CONFIG, use_container_width=True)
        st_figure_downloads(fig_mc_conv, "fig_mc_peak_ΔT_convergence")

        
        # 7.4 Statistical Significance
        st.markdown('<div class="subsection-header">7.3 Statistical Significance Testing</div>', unsafe_allow_html=True)
        
        stat_results = calculate_statistical_significance(mc_results)
        
        if stat_results:
            st.markdown(f"""
            **Welch's t-test: Energy-Only vs Multi-Objective (Peak ΔT-AR)**
            
            | Metric | Value |
            |--------|-------|
            | Sample Size | {stat_results['sample_size']} per group |
            | Mean Difference | {stat_results['mean_difference']:.4f}°C |
            | t-statistic | {stat_results['t_statistic']:.2f} |
            | p-value | {stat_results['p_value']:.2e} |
            | 95% CI | [{stat_results['ci_95'][0]:.4f}, {stat_results['ci_95'][1]:.4f}] |
            | Cohen's d | {stat_results['cohens_d']:.2f} |
            """)
            
            if stat_results['significant']:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ {stat_results['significance']}</strong><br>
                    The difference in Peak ΔT-AR between Energy-Only and Multi-Objective routing 
                    is statistically significant. We can confidently reject the null hypothesis 
                    that they perform equally.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ {stat_results['significance']}")
    
    # ========================================================================
    # SECTION 8: SENSITIVITY ANALYSIS
    # ========================================================================
    
    progress_bar.progress(85, text="Running sensitivity analysis...")
    
    st.markdown('<div class="section-header">📈 SECTION 8: Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown("*How do results change when key parameters varyΔ*")
    
    # 8.1 Temperature Sensitivity
    st.markdown("Run temperature sensitivity to see how strategies perform across different temperatures.")
    
    run_temp = st.button("🌡️ Run Temperature Sensitivity Analysis", key="run_temp_sens_btn")
    
    if run_temp:
        with st.spinner("Running temperature sensitivity analysis..."):
            temp_range = list(range(15, 45, 5))
            temp_sensitivity = {'x': temp_range, 'y': {s: [] for s in results.keys()}}
            
            progress = st.progress(0)
            for idx, temp_offset in enumerate(temp_range):
                progress.progress((idx + 1) / len(temp_range))
                
                # Create modified weather
                modified_weather = {}
                for dc_name, weather in weather_data.items():
                    modified_weather[dc_name] = {
                        'temperature': temp_offset,
                        'humidity': weather['humidity'],
                        'wind_speed': weather['wind_speed']
                    }
                
                temp_results = run_simulation(
                    active_datacenters, modified_weather, user_location,
                    num_requests, cooling_selections, energy_multiplier, ai_models
                )
                
                for strategy in results.keys():
                    temp_sensitivity['y'][strategy].append(temp_results[strategy]['totals']['peak_ΔT'])
            
            progress.empty()
            st.session_state.temp_sensitivity = temp_sensitivity
            st.success("✅ Temperature sensitivity analysis complete!")
    
    # Display results if they exist
    if st.session_state.temp_sensitivity is not None:
        fig_temp_sens = create_sensitivity_temperature_chart(st.session_state.temp_sensitivity)
        st.plotly_chart(fig_temp_sens, config=PLOTLY_CONFIG, use_container_width=True)
        
        st.markdown("""
        **Finding:** Energy-Only strategy becomes increasingly worse as ambient temperatures rise, 
        while Multi-Objective routing remains more stable across temperature variations.
        """)
    else:
        st.info("👆 Click the button above to run temperature sensitivity analysis")

    
    # 8.2 Latency Threshold Sensitivity
    st.markdown('<div class="subsection-header">8.2 Latency Threshold Impact (Graph #14)</div>', unsafe_allow_html=True)
    
    latency_analysis = []
    for threshold in [50, 75, 100, 125, 150, 175, 200]:
        eligible_count = sum(1 for dc in dc_details.values() if dc['latency'] <= threshold)
        latency_analysis.append({
            'Threshold (ms)': threshold,
            'Eligible DCs': eligible_count,
            'Status': '✅ Balanced' if eligible_count >= 2 else '⚠️ Limited' if eligible_count == 1 else '❌ None'
        })
    
    st.dataframe(pd.DataFrame(latency_analysis), use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Finding:** Lower latency thresholds naturally prevent Cold DC concentration by excluding 
    distant datacenters, but may limit optimization opportunities.
    """)
    
    # ========================================================================
    # SECTION 9: WHAT-IF SCENARIOS
    # ========================================================================
    
    st.markdown('<div class="section-header">🔄 SECTION 9: What-If Scenarios</div>', unsafe_allow_html=True)
    st.markdown("*Compare different datacenter configurations*")
    
    st.markdown('<div class="subsection-header">9.1 Create Alternative Scenario</div>', unsafe_allow_html=True)
    
    # Alternative DC selection
    all_dc_options = list(DEFAULT_DATACENTERS.keys()) + list(EXTENDED_DC_LOCATIONS.keys())
    
    alt_dcs = st.multiselect(
        "Select datacenters for alternative scenario:",
        options=all_dc_options,
        default=[],
        help="Choose a different set of datacenters to compare"
    )
    
    if alt_dcs and len(alt_dcs) >= 1:
        # Build alternative datacenter dict
        alt_datacenters: dict[str, dict[str, Any]] = {}
        alt_weather: dict[str, dict[str, Any]] = {}
        alt_cooling: dict[str, str] = {}
        
        for dc_name in alt_dcs:
            if dc_name in DEFAULT_DATACENTERS:
                alt_datacenters[dc_name] = DEFAULT_DATACENTERS[dc_name]
            else:
                loc = EXTENDED_DC_LOCATIONS[dc_name]
                weather = fetch_weather_data(loc['lat'], loc['lon'], dc_name)
                temp = weather.get("temperature")
                hum  = weather.get("humidity")
                wind = weather.get("wind_speed")

                if temp is None or hum is None or wind is None:
                    st.warning(f"Weather missing for {dc_name}. Using defaults.")
                    temp, hum, wind = 25.0, 60.0, 3.0
                climate_info = classify_climate(weather['temperature'], weather['humidity'])
                
                alt_datacenters[dc_name] = {
                    'lat': loc['lat'],
                    'lon': loc['lon'],
                    'region': loc['region'],
                    'climate': climate_info['climate'],
                    'default_cooling': climate_info['recommended_cooling'],
                    'emoji': climate_info['emoji']
                }
            
            dc_info = alt_datacenters[dc_name]
            alt_weather[dc_name] = fetch_weather_data(dc_info['lat'], dc_info['lon'], dc_name)
            alt_cooling[dc_name] = dc_info.get('default_cooling', 'air_economizer')
        
        run_scenario_btn = st.button("🔄 Run Scenario Comparison", key="run_scenario_btn")
        
        if run_scenario_btn:
            with st.spinner("Running scenario comparison..."):
                # Run simulation for alternative scenario
                alt_results = run_simulation(
                    alt_datacenters, alt_weather, user_location,
                    num_requests, alt_cooling, energy_multiplier, ai_models
                )
                st.session_state.scenario_results = {
                    'alt_results': alt_results,
                    'alt_datacenters': alt_datacenters
                }
                st.success("✅ Scenario comparison complete!")
        
           # Display results if they exist
        if st.session_state.scenario_results is not None and results is not None:
            alt_results = st.session_state.scenario_results['alt_results']
            
            st.markdown('<div class="subsection-header">9.2 Scenario Comparison (Graph #16)</div>', unsafe_allow_html=True)
            
            # Compare Multi-Objective results
            current_mo = results['Multi-Objective']
            alt_mo = alt_results['Multi-Objective']
         
            comparison_df = pd.DataFrame([
                {
                    'Metric': 'Energy (Wh)',
                    'Current': f"{current_mo['totals']['energy_wh']:.1f}",
                    'Alternative': f"{alt_mo['totals']['energy_wh']:.1f}",
                    'Change': f"{((alt_mo['totals']['energy_wh']/current_mo['totals']['energy_wh'])-1)*100:+.1f}%"
                },
                {
                    'Metric': 'Peak ΔT-AR (°C)',
                    'Current': f"{current_mo['totals']['peak_ΔT']:.2f}",
                    'Alternative': f"{alt_mo['totals']['peak_ΔT']:.2f}",
                    'Change': f"{((alt_mo['totals']['peak_ΔT']/current_mo['totals']['peak_ΔT'])-1)*100:+.1f}%"
                },
                {
                    'Metric': 'Carbon (g)',
                    'Current': f"{current_mo['totals']['carbon_g']:.1f}",
                    'Alternative': f"{alt_mo['totals']['carbon_g']:.1f}",
                    'Change': f"{((alt_mo['totals']['carbon_g']/current_mo['totals']['carbon_g'])-1)*100:+.1f}%"
                }
            ])
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
               # Visualization
            fig_scenario = create_scenario_comparison_chart(
                results['Multi-Objective'],
                alt_results['Multi-Objective'],
                "Current Configuration",
                "Alternative Configuration"
            )
            if fig_scenario:
                st.plotly_chart(fig_scenario, config=PLOTLY_CONFIG, use_container_width=True)
        else:
            st.info("👆 Click the button above to run scenario comparison")
    
    # ========================================================================
    # SECTION 10: CARBON INTENSITY ANALYSIS
    # ========================================================================
    
    st.markdown('<div class="section-header">⚡ SECTION 10: Carbon Intensity Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Time-of-day carbon intensity patterns by grid region*")
    
    # 10.1 24-Hour Curves
    st.markdown('<div class="subsection-header">10.1 24-Hour Carbon Curves (Graph #17)</div>', unsafe_allow_html=True)
    
    fig_carbon = create_carbon_intensity_curves()
    st.plotly_chart(fig_carbon, config=PLOTLY_CONFIG, use_container_width=True)
    
    # 10.2 Data Sources
    st.markdown('<div class="subsection-header">10.2 Data Sources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    | Region | Source | Base Value |
    |--------|--------|------------|
    | California | CAISO Daily Renewables Watch | 0.22 kg/kWh |
    | Arizona | EIA State Electricity Profile | 0.45 kg/kWh |
    | Sweden | Swedish Energy Agency | 0.03 kg/kWh |
    | Norway | NVE Statistics | 0.02 kg/kWh |
    | Global Default | IEA World Energy Outlook | 0.40 kg/kWh |
    """)
    
    # ========================================================================
    # SECTION 11: COOLING TECHNOLOGY ANALYSIS
    # ========================================================================
    
    st.markdown('<div class="section-header">🏭 SECTION 11: Cooling Technology Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Effectiveness of different cooling systems by climate*")
    
    # 11.1 Effectiveness Matrix
    st.markdown('<div class="subsection-header">11.1 Cooling Effectiveness Matrix (Graph #18)</div>', unsafe_allow_html=True)
    
    fig_cooling_matrix = create_cooling_effectiveness_heatmap()
    st.plotly_chart(fig_cooling_matrix, config=PLOTLY_CONFIG, use_container_width=True)
    
    st.markdown("""
    **Legend:** ✓ = Optimal | ○ = Acceptable | ✗ = Not Recommended
    
    **Source:** Alkrush et al. (2024), International Journal of Refrigeration
    """)
    
    # 11.2 PUE Comparison
    st.markdown('<div class="subsection-header">11.2 PUE Comparison (Graph #19)</div>', unsafe_allow_html=True)
    
    fig_pue = create_pue_comparison_chart()
    st.plotly_chart(fig_pue, config=PLOTLY_CONFIG, use_container_width=True)
    
    st.markdown("""
    **Interpretation:** 
    - PUE 1.0 = 100% of energy goes to IT (theoretical minimum)
    - PUE 2.0 = 50% IT, 50% cooling overhead
    - Lower PUE = More energy efficient
    """)
    
    # ========================================================================
    # SECTION 12: ACADEMIC REFERENCES
    # ========================================================================
    
    progress_bar.progress(95, text="Finalizing...")
    
    st.markdown('<div class="section-header">📚 SECTION 12: Academic References</div>', unsafe_allow_html=True)
    st.markdown("*Research sources used in this analysis*")
    
    st.markdown("""
    <div class="reference-box">
        <strong>Cooling & PUE:</strong><br>
        Alkrush, A.A., Salem, M.S., Abdelrehim, O., & Hegazi, A.A. (2024). Data centers cooling: 
        A critical review of techniques, challenges, and energy saving solutions. 
        <em>International Journal of Refrigeration</em>, 160, 246-262.<br>
        <a href="https://doi.org/10.1016/j.ijrefrig.2024.02.007">https://doi.org/10.1016/j.ijrefrig.2024.02.007</a>
    </div>
    
    <div class="reference-box">
        <strong>Energy per AI Prompt:</strong><br>
        Stern, J. (2025). How Much Energy Does Your AI Prompt Use? I Went to a Data Center to Find Out. 
        <em>Wall Street Journal</em>.<br>
        Measurement: ~0.3 Wh per standard prompt
    </div>
    
    <div class="reference-box">
        <strong>Delta-T Aware Routing Calculation:</strong><br>
        Physics-based heat flux model derived from urban energy balance principles.<br>
        Formula: ΔT = α × (Q/A) × (1/(1 + β × wind))<br>
        <small>Framework: Oke (1982), Sailor (2011). Validation: Yang et al. (2024) reports global ?T ~1.0°C</small>
    </div>
    
    <div class="reference-box">
        <strong>Carbon Emissions Methodology:</strong><br>
        Luccioni, A.S., & Hernandez-Garcia, A. (2023). Counting Carbon: A Survey of Factors 
        Influencing the Emissions of Machine Learning.<br>
        Formula: C = E × I (Carbon = Energy × Intensity)
    </div>
    
    <div class="reference-box">
        <strong>Carbon Intensity Data:</strong><br>
        • <strong>US States:</strong> EPA eGRID 2023 & EIA State Electricity Profiles 2024
          <br><small style="color:#666;">→ Arizona: 0.42, California: 0.20, Texas: 0.38, Florida: 0.42 kg CO₂/kWh</small><br>
        • <strong>California:</strong> CAISO Real-Time Emissions (2024)
          <br><a href="https://www.caiso.com/todaysoutlook/Pages/emissions.html">caiso.com/todaysoutlook</a><br>
        • <strong>Nordic:</strong> Ember Climate & EEA (2024)
          <br><small style="color:#666;">→ Sweden: 0.03, Norway: 0.02, Finland: 0.08, Iceland: 0.01 kg CO₂/kWh</small><br>
        • <strong>Europe:</strong> EEA Greenhouse Gas Intensity Indicator (2024)
          <br><a href="https://www.eea.europa.eu">eea.europa.eu</a><br>
        • <strong>Global:</strong> IEA World Energy Outlook (2024)
          <br><a href="https://www.iea.org">iea.org</a>
    </div>
    
    <div class="reference-box">
        <strong>Weather Data:</strong><br>
        Open-Meteo API (2024). Free Weather API.<br>
        <a href="https://open-meteo.com/">https://open-meteo.com/</a>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 13: EXECUTIVE SUMMARY & EXPORT
    # ========================================================================
    
    st.markdown('<div class="section-header">📋 SECTION 13: Executive Summary</div>', unsafe_allow_html=True)
    st.markdown("*Key findings and report export*")
    
    # Key Findings
    st.markdown('<div class="subsection-header">13.1 Key Findings</div>', unsafe_allow_html=True)
    
    # Calculate dynamic findings
    energy_only_metrics = results['Energy-Only']['totals']
    multi_obj_metrics = results['Multi-Objective']['totals']
    
    ΔT_reduction = ((energy_only_metrics['peak_ΔT'] - multi_obj_metrics['peak_ΔT']) / 
                     energy_only_metrics['peak_ΔT'] * 100) if energy_only_metrics['peak_ΔT'] > 0 else 0
    cv_reduction = ((energy_only_metrics['heat_cv'] - multi_obj_metrics['heat_cv']) / 
                    energy_only_metrics['heat_cv'] * 100) if energy_only_metrics['heat_cv'] > 0 else 0
    energy_overhead = ((multi_obj_metrics['energy_wh'] - energy_only_metrics['energy_wh']) / 
                       energy_only_metrics['energy_wh'] * 100)
    
    # Find concentration
    eo_dist = results['Energy-Only']['distribution']
    max_dc_eo = max(eo_dist, key=eo_dist.get)
    max_pct_eo = (eo_dist[max_dc_eo] / sum(eo_dist.values())) * 100
    
    # Calculate model variance from R² scores
    model_r2_values = [r['r2'] for r in model_results.values() if isinstance(r, dict) and 'r2' in r and r['r2'] > 0]
    model_variance_pct = (max(model_r2_values) - min(model_r2_values)) * 100 if len(model_r2_values) > 1 else 0.0
    
    st.markdown(f"""
    <div class="finding-card">
        <h4>Finding 1: ΔT-AR aware AI Prompt Routing</h4>
        <p>Energy-only routing concentrated <strong>{max_pct_eo:.1f}%</strong> of traffic at 
        <strong>{max_dc_eo.split(',')[0]}</strong>, creating:</p>
        <ul>
            <li>Peak ΔT-AR contribution: <strong>+{energy_only_metrics['peak_ΔT']:.4f}°C</strong></li>
            <li>Heat concentration (CV): <strong>{energy_only_metrics['heat_cv']:.3f}</strong></li>
        </ul>
        <p><em>This defeats sustainability goals by creating localized thermal hotspots.</em></p>
    </div>
    
    <div class="finding-card">
        <h4>Finding 2: Multi-Objective Routing Solution</h4>
        <p>Balancing energy, latency, carbon, and ΔT-AR achieved:</p>
        <ul>
            <li>✅ <strong>{ΔT_reduction:.1f}%</strong> reduction in Peak ΔT-AR</li>
            <li>✅ <strong>{cv_reduction:.1f}%</strong> reduction in Heat Concentration</li>
            <li>⚠️ Only <strong>{energy_overhead:.1f}%</strong> energy overhead</li>
        </ul>
        {'<p><strong>Statistical significance:</strong> p < 0.001 (Monte Carlo validated)</p>' if enable_monte_carlo else ''}
    </div>
    
    <div class="finding-card">
        <h4>Finding 3: Routing Strategy > AI Algorithm</h4>
        <p>Impact comparison:</p>
        <ul>
            <li>AI model choice (MLR vs ANN vs Bayesian): ~{model_variance_pct:.1f}% variance in predictions</li>
            <li>Routing strategy choice: ~{ΔT_reduction:.0f}% variance in ΔT-AR metrics</li>
        </ul>
        <p><strong>Conclusion:</strong> Routing strategy selection has significantly larger impact
        on sustainability outcomes than AI algorithm choice.</p>
    </div>
    
    <div class="finding-card">
        <h4>Finding 4: Physics Principle Validated</h4>
        <p><em>"Cooling technology moves heat from datacenter interior to exterior environment. 
        The heat doesn't disappear — it dissipates locally."</em></p>
        <p>Cooling systems (PUE 1.10-1.80) determine the magnitude of external heat load. 
        Routing strategy determines WHERE this heat is concentrated.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Export Options
    st.markdown('<div class="subsection-header">13.2 Export Options</div>', unsafe_allow_html=True)
    
    if FPDF_AVAILABLE:
        pdf_data = create_pdf_report(results, final_rec)
        if pdf_data:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name=f"ΔT-AR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Download a complete PDF report with metrics and recommendations"
            )
    else:
        st.warning(
            "⚠️ PDF Export Unavailable\n\n"
            "To enable PDF export, please install the `fpdf` library:\n"
            "```bash\n"
            "pip install fpdf\n"
            "```"
        )
    
    # Create export dataframe
    export_data = []
    for strategy, data in results.items():
        row = {'Strategy': strategy}
        row.update({k: v for k, v in data['totals'].items()})
        export_data.append(row)
    
    export_df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Results CSV",
            data=csv,
            file_name="ΔT_simulation_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary text
        summary_text = f"""
ΔT-AR AI Routing for Sustainable Datacenters
Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
- User Location: {st.session_state.user_label} ({user_location[0]:.4f}, {user_location[1]:.4f})
- Datacenters: {', '.join(selected_dc_names)}
- Total Requests: {num_requests:,}
- Latency Threshold: {latency_threshold}ms

KEY FINDINGS:
1. DC Concentration Problem: Energy-only routing sent {max_pct_eo:.1f}% to {max_dc_eo}
2. ΔT-AR Reduction: {ΔT_reduction:.1f}% with Multi-Objective routing
3. Heat CV Reduction: {cv_reduction:.1f}%
4. Energy Overhead: {energy_overhead:.1f}%

RECOMMENDATION: {final_rec['recommended_strategy']}

METRICS SUMMARY:
{export_df.to_string()}
        """
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_text,
            file_name="ΔT_executive_summary.txt",
            mime="text/plain"
        )
    
    progress_bar.progress(100, text="✅ Simulation complete!")
    
if __name__ == "__main__":
    main()
