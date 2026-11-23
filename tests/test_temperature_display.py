import pytest
import plotly.graph_objects as go
from temperature_display import fahrenheit, temperature_text, dual_temperature_axes


def test_absolute_temperature_and_temperature_rise_use_different_offsets():
    assert fahrenheit(0) == 32
    assert fahrenheit(100) == 212
    assert fahrenheit(0, difference=True) == 0
    assert fahrenheit(.00015, difference=True) == pytest.approx(.00027)
    assert temperature_text(20) == '20.0 °C [68.0 °F]'


def test_dual_axes_preserve_celsius_data_and_distinguish_thermal_rise():
    figure = go.Figure(go.Scatter(x=[0, 20, 40], y=[0, .001, .002]))
    figure.update_layout(xaxis_title='Ambient Temperature (°C)', yaxis_title='Peak ΔT-AR (°C)')
    dual_temperature_axes(figure)
    assert figure.data[0].x == (0, 20, 40)
    assert figure.layout.xaxis.ticktext[0] == '0 [32]'
    assert figure.layout.yaxis.ticktext[0] == '0 [0]'
