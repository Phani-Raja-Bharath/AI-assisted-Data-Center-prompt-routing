"""Display conversions only; simulation inputs and calculations stay in Celsius."""
import math


def fahrenheit(celsius, *, difference=False):
    return float(celsius) * 1.8 + (0 if difference else 32)


def temperature_text(celsius, *, difference=False, precision=1):
    return f"{celsius:.{precision}f} °C [{fahrenheit(celsius, difference=difference):.{precision}f} °F]"


def dual_temperature_axes(figure):
    """Show Celsius/Fahrenheit ticks for temperature axes, including scaled ΔT."""
    for axis_name in figure.layout:
        if not axis_name.startswith(('xaxis', 'yaxis')):
            continue
        axis = figure.layout[axis_name]
        title = axis.title.text or ''
        if '°C' not in title:
            continue
        direction = axis_name[0]
        reference = direction + axis_name[len(direction + 'axis'):]
        values = []
        for trace in figure.data:
            if (getattr(trace, direction + 'axis', None) or direction) != reference:
                continue
            data = getattr(trace, direction, None)
            if data is not None:
                values.extend(float(v) for v in data if isinstance(v, (float, int)) and math.isfinite(v))
        if not values:
            continue
        low, high = min(values), max(values)
        span = high - low
        ticks = [low + span * i / 4 for i in range(5)] if span else [low]
        scale = 100 if '×100' in title else 1
        difference = 'ΔT' in title
        axis.update(tickmode='array', tickvals=ticks,
                    ticktext=[f"{v:.3g} [{fahrenheit(v / scale, difference=difference) * scale:.3g}]" for v in ticks],
                    title_text=title.replace('°C', '°C [°F]'))
