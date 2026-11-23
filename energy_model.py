"""Shared energy formulas and stable model feature encoding."""
from typing import Any

BASE_ENERGY_WH = 0.3

ENERGY_PER_PROMPT_BASE = BASE_ENERGY_WH



# Cooling Systems with PUE values from Alkrush et al. (2024)
# International Journal of Refrigeration, 160, 246-262
COOLING_SYSTEMS: dict[str, dict[str, Any]] = {
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


FEATURE_NAMES = ("temperature", "humidity", "wind_speed", "pressure", "solar_radiation", "cooling_type")
COOLING_TYPES = tuple(sorted(COOLING_SYSTEMS))

def cooling_index(cooling_type):
    return COOLING_TYPES.index(cooling_type)

def calculate_energy_per_request(temperature, humidity, cooling_type, energy_multiplier=1.0):
    # """
    # Calculate energy consumption per AI request.
    
    # Formula: E = E_base × PUE × (1 + α × max(0, T - 20)) × humidity_factor
    
    # Based on:
    # - Stern (2025): Base energy ~0.3 Wh per prompt
    # - Alkrush et al. (2024): PUE values by cooling type
    # """
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
    # """
    # Estimate local ΔT-AR contribution using physics-based heat flux model.
    
    # Formula: ΔT = κ × (Q/A) × (1/(1 + γ × wind))
    
    # This model is derived from thermodynamic principles:
    # - Heat flux density (Q/A) drives local temperature rise
    # - Wind provides convective heat dissipation
    
    # Parameters:
    # - η = 0.0012: Heat-to-temperature coefficient (°C per kW/km²)
    # - γ = 0.15: Wind dissipation factor
    # """
    eta = 0.0012  # Heat-to-temperature coefficient
    gamma = 0.15     # Wind dissipation factor
    
    heat_flux = heat_kwh / area_km2
    wind_factor = 1 / (1 + gamma * wind_speed)
    
    return eta * heat_flux * wind_factor


def surrogate_energy_target(temperature, humidity, wind, pressure, solar, cooling_type):
    """Noise-free training target; identical for real and synthetic weather."""
    pressure_factor = min(1.35, max(0.7, (1013.25 / max(pressure, 1e-6)) ** 0.8))
    wind_factor = 1 / (1 + 0.05 * wind)
    solar_factor = 1 + 0.0001 * max(0, solar - 200)
    return calculate_energy_per_request(
        temperature, humidity, cooling_type,
        energy_multiplier=pressure_factor * wind_factor * solar_factor)
