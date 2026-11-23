"""Routing policies with shared energy, capacity, and integer-allocation rules."""
from datetime import datetime
import math
import numpy as np

from energy_model import (cooling_index, calculate_energy_per_request,
                          calculate_ΔT_contribution)
from network_model import haversine_distance, calculate_latency
from carbon_model import get_carbon_intensity
from routing_core import allocate_requests, latency_request_capacity


def energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier):
    energies = {}
    for dc, info in datacenters.items():
        weather = weather_data.get(dc, {})
        cooling = cooling_selections.get(dc, info.get('default_cooling', 'air_economizer'))
        temp, humidity = weather.get('temperature', 20), weather.get('humidity', 50)
        if ai_models is not None and ai_models.is_trained:
            features = [temp, humidity, weather.get('wind_speed', 5),
                        weather.get('pressure', 1013.25), weather.get('solar_radiation', 200),
                        cooling_index(cooling)]
            energy = ai_models.predict(features) * energy_multiplier
        else:
            energy = calculate_energy_per_request(temp, humidity, cooling, energy_multiplier)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError(f'Energy prediction must be finite and positive for {dc}.')
        energies[dc] = energy
    return energies


def _allocate(weights, energies, num_requests, max_dc_capacity_mw, latency_caps=None):
    if not math.isfinite(max_dc_capacity_mw) or max_dc_capacity_mw < 0:
        raise ValueError('Datacenter capacity must be finite and nonnegative.')
    capacities = {dc: min(num_requests, int(max_dc_capacity_mw * 1e6 / energy))
                  for dc, energy in energies.items()}
    if latency_caps is not None:
        capacities = {dc: min(cap, latency_caps.get(dc, 0)) for dc, cap in capacities.items()}
    return allocate_requests(weights, capacities, num_requests)


def route_random(datacenters, num_requests, weather_data=None, cooling_selections=None,
                 ai_models=None, max_dc_capacity_mw=100, energy_multiplier=1.0):
    if not datacenters or num_requests <= 0:
        return {}
    energies = energy_map(datacenters, weather_data or {}, cooling_selections or {},
                          ai_models, energy_multiplier)
    weights = dict(zip(datacenters, np.random.dirichlet(np.ones(len(datacenters)))))
    return _allocate(weights, energies, num_requests, max_dc_capacity_mw)


def route_energy_only(datacenters, weather_data, cooling_selections, num_requests,
                      ai_models=None, max_dc_capacity_mw=100, energy_multiplier=1.0):
    if not datacenters or num_requests <= 0:
        return {}
    energies = energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier)
    best = min(energies.values())
    weights = {dc: (best / energy) ** 3 for dc, energy in energies.items()}
    return _allocate(weights, energies, num_requests, max_dc_capacity_mw)


def route_heat_only(datacenters, weather_data, cooling_selections, num_requests,
                    ai_models=None, max_dc_capacity_mw=100, energy_multiplier=1.0):
    if not datacenters or num_requests <= 0:
        return {}
    energies = energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier)
    scores = {dc: calculate_ΔT_contribution(energy / 1000,
              wind_speed=weather_data.get(dc, {}).get('wind_speed', 5))
              for dc, energy in energies.items()}
    best = min(scores.values())
    weights = {dc: (best / max(score, 1e-15)) ** 3 for dc, score in scores.items()}
    return _allocate(weights, energies, num_requests, max_dc_capacity_mw)


def route_ΔT_aware(datacenters, weather_data, cooling_selections, num_requests,
                  ai_models=None, max_dc_capacity_mw=100, energy_multiplier=1.0):
    if not datacenters or num_requests <= 0:
        return {}
    energies = energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier)
    scores = {}
    for dc, energy in energies.items():
        weather = weather_data.get(dc, {})
        vulnerability = 1 + 0.05 * max(0, weather.get('temperature', 20) - 25)
        scores[dc] = energy * vulnerability / (1 + 0.05 * weather.get('wind_speed', 5))
    best = min(scores.values())
    weights = {dc: best / score for dc, score in scores.items()}
    return _allocate(weights, energies, num_requests, max_dc_capacity_mw)


def route_multi_objective(datacenters, weather_data, cooling_selections, user_location,
                          num_requests, ai_models=None, latency_threshold=100,
                          max_dc_capacity_mw=100, energy_multiplier=1.0):
    """Balance four costs, subject to power capacity and modeled latency bounds.

    Demand that cannot meet both constraints remains unserved.
    """
    if not datacenters or num_requests <= 0:
        return {}
    energies = energy_map(datacenters, weather_data, cooling_selections, ai_models, energy_multiplier)
    scores, latency_caps = {}, {}
    reference = latency_threshold if latency_threshold and latency_threshold > 0 else 200.0
    hour = datetime.now().hour
    for dc, info in datacenters.items():
        weather = weather_data.get(dc, {})
        distance = haversine_distance(*user_location, info['lat'], info['lon'])
        latency_caps[dc] = latency_request_capacity(distance, num_requests,
                                                   latency_threshold, calculate_latency)
        if not latency_caps[dc]:
            continue
        latency = calculate_latency(distance, load_fraction=0.7)
        carbon = get_carbon_intensity(info.get('region', 'default'), hour)
        thermal = max(0, weather.get('temperature', 20) - 15) / 30
        thermal /= 1 + 0.15 * weather.get('wind_speed', 5)
        scores[dc] = 0.25 * (energies[dc] + latency / reference + carbon / 0.7 + thermal)
    best = min(scores.values()) if scores else 0
    weights = {dc: math.exp(-(score - best) * 3) for dc, score in scores.items()}
    return _allocate(weights, energies, num_requests, max_dc_capacity_mw, latency_caps)
