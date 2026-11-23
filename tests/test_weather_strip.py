from weather_strip import weather_html


def test_weather_strip_distinguishes_api_and_assumptions_and_escapes_text():
    html = weather_html({
        '<unsafe>': {'temperature': 20., 'humidity': 50., 'wind_speed': 3.,
                     'pressure': 1013., 'solar_radiation': 150., 'success': True,
                     'observed_at': '2026-11-23T16:00:00+00:00',
                     'retrieved_at': '2026-11-23T16:04:12+00:00'},
        'Fallback site': {'temperature': 5., 'success': False,
                          'observed_at': 'do not present this as valid'},
    }, ['#176b62'])
    assert '&lt;unsafe&gt;' in html and '<unsafe>' not in html
    assert '20.0 °C' in html and '[68.0 °F]' in html
    assert 'API weather estimate' in html and 'Assumed weather baseline' in html
    assert '2026-11-23 16:00:00 UTC' in html
    assert 'Last fetched: 2026-11-23 16:04:12 UTC' in html
    assert 'do not present this as valid' not in html
    assert 'prefers-reduced-motion:reduce' in html
