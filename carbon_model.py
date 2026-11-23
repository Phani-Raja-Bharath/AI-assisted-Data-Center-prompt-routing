"""Regional grid baselines with modeled hourly variation, not live grid telemetry."""
from datetime import datetime, timezone
from functools import lru_cache
import csv
import io
import math
import time
import numpy as np
import requests

CARBON_DATA_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"
CARBON_SOURCE_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity"

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
    "argentina": 0.346,    # OWID/Ember 2025 national average; offline fallback
    
    # Default
    "default": 0.40        # IEA 2024: global average ~400 gCO2/kWh
}


@lru_cache(maxsize=8)
def _argentina_carbon_data(day):
    """Refresh the latest annual national average once per UTC day.

    Neuquén uses Argentina's national grid proxy, not city-level measurements.
    Cache failures too, so a service outage does not delay every simulation trial.
    """
    fallback = {"base_kg_per_kwh": CARBON_INTENSITY_BASE["argentina"], "year": 2025,
                "source": "OWID / Ember annual grid data (offline fallback)",
                "source_url": CARBON_SOURCE_URL, "fetched": False, "fallback": True,
                "scope": "Argentina national grid proxy for Neuquén"}
    try:
        response = requests.get(CARBON_DATA_URL, params={"country": "~ARG"}, timeout=10)
        response.raise_for_status()
        records = []
        for row in csv.DictReader(io.StringIO(response.text)):
            if row.get("Code") != "ARG":
                continue
            try:
                year = int(row["Year"])
                intensity = float(row["Carbon intensity"]) / 1000.0  # g -> kg CO2e/kWh
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(intensity) and intensity >= 0:
                records.append((year, intensity))
        if not records:
            return fallback
        year, intensity = max(records, key=lambda item: item[0])
        return {**fallback, "base_kg_per_kwh": intensity, "year": year,
                "source": "OWID / Ember annual grid data", "fetched": True, "fallback": False,
                "fetched_at": datetime.now(timezone.utc).isoformat()}
    except requests.RequestException:
        return fallback


def get_carbon_data(region):
    """Return baseline intensity and provenance for the selected grid region."""
    if region == "argentina":
        return dict(_argentina_carbon_data(int(time.time() // 86400)))
    return {"base_kg_per_kwh": CARBON_INTENSITY_BASE.get(region, CARBON_INTENSITY_BASE["default"]),
            "year": None, "source": "Configured regional baseline", "source_url": None,
            "fetched": False, "fallback": region not in CARBON_INTENSITY_BASE,
            "scope": region}


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
    
    base = get_carbon_data(region)["base_kg_per_kwh"]
    
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
