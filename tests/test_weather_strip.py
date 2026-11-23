from weather_strip import weather_frame, weather_sources
import pandas as pd


def test_conditions_keep_units_selection_and_fallback_provenance_separate():
    weather = {
        'Phoenix': {'temperature': 20., 'humidity': 50., 'wind_speed': 3.,
                    'pressure': 1013., 'solar_radiation': 150., 'success': True,
                    'observed_at': '2026-11-23T16:00:00+00:00',
                    'retrieved_at': '2026-11-23T16:04:12+00:00'},
        'Fallback': {'temperature': 5., 'success': False,
                     'observed_at': '2026-11-23T16:00:00+00:00'},
    }
    indicators = {'Phoenix': {'emoji': '🌤️', 'label': 'Moderate', 'pue': 1.15},
                  'Fallback': {'emoji': '❄️', 'label': 'Cold', 'pue': 1.10}}
    conditions = weather_frame(weather, indicators, {'Phoenix': 'liquid_cooling', 'Fallback': 'free_air'})
    assert conditions.loc[0, 'Temperature'] == '20.0 °C [68.0 °F]'
    assert conditions.loc[0, 'Thermal category'] == '🌤️ Moderate'
    assert conditions.loc[0, 'Cooling technology'] == 'liquid_cooling'
    assert conditions.loc[1, 'Data source'] == 'Assumed baseline'
    sources = weather_sources(weather)
    assert sources.loc[0, 'Weather valid at'] == '2026-11-23 16:00:00 UTC'
    assert sources.loc[0, 'Last fetched'] == '2026-11-23 16:04:12 UTC'
    assert pd.isna(sources.loc[1, 'Weather valid at'])
    assert pd.isna(sources.loc[1, 'Last fetched'])
