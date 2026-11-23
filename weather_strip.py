"""Animated weather presentation; uses the same cached inputs as the simulation."""
from html import escape
from datetime import datetime, timezone
from math import isfinite

import streamlit as st

from localization import tr
from temperature_display import fahrenheit


STYLE = """<style>
.weather-strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));
  gap:12px; margin:4px 0 18px; font-family:Aptos,Calibri,sans-serif; }
.weather-site { border-top:2px solid var(--site-color); background:#edf0e8;
  border-radius:5px; padding:14px 16px; animation:weather-enter .55s ease-out both;
  animation-delay:var(--entry-delay); color:#193b40; }
.weather-site header { display:flex; align-items:center; gap:8px; font-size:.95rem;
  font-weight:700; margin-bottom:10px; }
.weather-site .weather-mark { width:25px; height:25px; color:var(--site-color); flex:none;
  animation:weather-breeze 2.6s ease-in-out 2; }
.weather-site .weather-reading { font-size:1.8rem; font-variant-numeric:tabular-nums; line-height:1.1; }
.weather-site .weather-reading small { font-size:.85rem; font-weight:400; color:#50676b; }
.weather-site dl { display:grid; grid-template-columns:1fr 1fr; gap:9px 12px; margin:14px 0 10px; }
.weather-site dl div { display:flex; flex-wrap:wrap; justify-content:space-between; gap:2px 6px; }
.weather-site dt { font-size:.76rem; color:#50676b; }
.weather-site dd { margin:0; font-size:.85rem; font-variant-numeric:tabular-nums; }
.weather-site .weather-source { font-size:.72rem; color:#50676b; line-height:1.5; }
.weather-site .weather-source strong { font-weight:600; }
.weather-site.is-fallback { background:#f4ede1; border-top-color:#a76d35; }
.weather-site.is-fallback .weather-source strong { color:#8a5425; }
@keyframes weather-enter { from { opacity:0; transform:translateY(9px); }
  to { opacity:1; transform:translateY(0); } }
@keyframes weather-breeze { 0%,100% { transform:translateX(0); } 50% { transform:translateX(3px); } }
@media(prefers-reduced-motion:reduce) { .weather-site,.weather-site .weather-mark { animation:none; } }
</style>"""

# A wind symbol, not a claim about sky conditions (cloud cover is not fetched).
WIND_ICON = """<svg class="weather-mark" viewBox="0 0 28 28" fill="none" aria-hidden="true">
<path d="M3 10h15c6 0 6-7 1-7-2 0-3 1-3 3M3 15h19c5 0 5 8 0 8-2 0-3-1-3-3M3 20h9"
stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>"""


def _number(data, key, precision=0):
    value = data.get(key)
    return f"{value:.{precision}f}" if isinstance(value, (float, int)) and isfinite(value) else "—"


def _utc_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return None


def weather_html(weather, palette):
    """Escape location/source strings and never present fallback data as current API data."""
    cards = []
    for index, (name, data) in enumerate(weather.items()):
        api = data.get("success") is True
        source = tr("API weather estimate") if api else tr("Assumed weather baseline")
        valid_at = _utc_time(data.get("observed_at")) if api else None
        stamp = f"{tr('Weather valid at')}: {valid_at}" if valid_at else tr("Valid-at time unavailable")
        fetched_at = _utc_time(data.get("retrieved_at")) if api else None
        fetched = f"{tr('Last fetched')}: {fetched_at}" if fetched_at else tr("No successful fetch timestamp recorded")
        if not api and data.get("checked_at"):
            fetched = f"{tr('Last fetch attempt')}: {_utc_time(data['checked_at'])}"
        temp = _number(data, "temperature", 1)
        temp_f = f"{fahrenheit(data['temperature']):.1f}" if temp != "—" else "—"
        values = [("Humidity", "humidity", 0, "%"), ("Wind", "wind_speed", 1, "m/s"),
                  ("Pressure", "pressure", 0, "hPa"), ("Solar radiation", "solar_radiation", 0, "W/m²")]
        readings = ''.join(f"<div><dt>{escape(tr(label))}</dt><dd>{_number(data, key, precision)} {unit}</dd></div>"
                           for label, key, precision, unit in values)
        cards.append(f"""<article class="weather-site {'' if api else 'is-fallback'}"
            style="--site-color:{palette[index % len(palette)]};--entry-delay:{index * 90}ms">
            <header>{WIND_ICON}<span>{escape(name)}</span></header>
            <div class="weather-reading" aria-label="{escape(tr('Temperature'))}">{temp} °C <small>[{temp_f} °F]</small></div>
            <dl>{readings}</dl>
            <div class="weather-source"><strong>{escape(source)}</strong><br>{escape(stamp)}<br>{escape(fetched)}</div>
            </article>""")
    return STYLE + f'<section class="weather-strip" aria-label="{escape(tr("Datacenter weather"))}">' + ''.join(cards) + '</section>'


def render_weather_strip(weather, palette):
    st.html(weather_html(weather, palette))
