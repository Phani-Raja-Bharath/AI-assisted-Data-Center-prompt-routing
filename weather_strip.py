"""Compact, read-only weather matrix; cooling controls live above it."""
from datetime import datetime, timezone
import math

import pandas as pd
import streamlit as st

from localization import tr
from temperature_display import temperature_text


def _utc_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return None


def weather_frame(weather, indicators, selections):
    rows = []
    for name, data in weather.items():
        temp = data.get("temperature")
        temperature = temperature_text(temp) if isinstance(temp, (int, float)) and math.isfinite(temp) else "n/a"
        site = indicators[name]
        rows.append({"Datacenter": name, "Thermal category": f"{site['emoji']} {tr(site['label'])}",
                     "Temperature": temperature, "Humidity (%)": data.get("humidity"),
                     "Wind (m/s)": data.get("wind_speed"), "Pressure (hPa)": data.get("pressure"),
                     "Solar radiation (W/m²)": data.get("solar_radiation"),
                     "Cooling technology": selections[name], "PUE": site["pue"],
                     "Data source": tr("API estimate") if data.get("success") is True else tr("Assumed baseline")})
    return pd.DataFrame(rows)


def weather_sources(weather):
    rows = []
    for name, data in weather.items():
        api = data.get("success") is True
        rows.append({"Datacenter": name,
                     "Source": data.get("source") or ("API weather estimate" if api else "Assumed weather baseline"),
                     "Weather valid at": _utc_time(data.get("observed_at")) if api else None,
                     "Last fetched": _utc_time(data.get("retrieved_at")) if api else None,
                     "Last fetch attempt": _utc_time(data.get("checked_at")) if not api else None})
    return pd.DataFrame(rows)


def render_weather_table(weather, indicators, selections):
    """Read-only conditions; explicit cooling dropdowns remain visible above the readings."""
    frame = weather_frame(weather, indicators, selections).set_index("Datacenter", drop=False)
    with st.container(key="weather_matrix"):
        st.dataframe(frame.drop(columns=["Cooling technology"]), hide_index=True, row_height=38, width="stretch",
            column_config={
                "Datacenter": st.column_config.TextColumn(tr("Datacenter"), width="medium", pinned=True),
                "Thermal category": st.column_config.TextColumn(tr("Thermal category"), width="small"),
                "Temperature": st.column_config.TextColumn(tr("Temperature") + " · °C [°F]", width="medium"),
                "Humidity (%)": st.column_config.NumberColumn(tr("Humidity") + " · %", format="%.0f", width="small"),
                "Wind (m/s)": st.column_config.NumberColumn(tr("Wind") + " · m/s", format="%.1f", width="small"),
                "Pressure (hPa)": st.column_config.NumberColumn(tr("Pressure") + " · hPa", format="%.0f", width="small"),
                "Solar radiation (W/m²)": st.column_config.NumberColumn(tr("Solar") + " · W/m²", help=tr("Solar radiation"), format="%.0f", width="small"),
                "PUE": st.column_config.NumberColumn("PUE", format="%.2f", width="small",
                    help=tr("Fixed scenario efficiency assumption for the selected cooling technology.")),
                "Data source": st.column_config.TextColumn(tr("Data source"), width="small"),
            })
