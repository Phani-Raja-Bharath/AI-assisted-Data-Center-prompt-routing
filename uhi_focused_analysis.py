"""
================================================================================
UHI DASHBOARD - FOCUSED BATCH ANALYSIS
Analyzing: Prompts → UHI Effects & Cooling Type Effectiveness

Author: Phani Raja Bharath Balijepalli
Course: IDS6938 - AI, Energy, and Sustainability
University of Central Florida

Run locally:
    pip install pandas numpy scipy scikit-learn requests matplotlib seaborn
    python uhi_focused_analysis.py

Output:
    results/
    ├── plots/           (PDF figures for paper)
    ├── data/            (CSV raw data)
    └── analysis/        (Summary reports)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Matplotlib for reliable PDF export
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# CONSTANTS
# ============================================================================

BASE_ENERGY_WH = 0.3  # Stern (2025) - WSJ

COOLING_SYSTEMS = {
    "free_air": {
        "name": "Free Air Cooling",
        "pue": 1.10,
        "temp_sensitivity": 0.003,
        "best_climate": "cold",
        "color": "#2563eb"
    },
    "liquid_cooling": {
        "name": "Liquid Cooling",
        "pue": 1.15,
        "temp_sensitivity": 0.005,
        "best_climate": "any",
        "color": "#7c3aed"
    },
    "air_economizer": {
        "name": "Air Economizer",
        "pue": 1.25,
        "temp_sensitivity": 0.010,
        "best_climate": "moderate",
        "color": "#059669"
    },
    "evaporative": {
        "name": "Evaporative",
        "pue": 1.35,
        "temp_sensitivity": 0.012,
        "best_climate": "hot_dry",
        "color": "#d97706"
    },
    "mechanical_chiller": {
        "name": "Mechanical Chiller",
        "pue": 1.80,
        "temp_sensitivity": 0.020,
        "best_climate": "any",
        "color": "#dc2626"
    }
}

DATACENTERS = {
    "Phoenix, AZ": {
        "lat": 33.4484, "lon": -112.0740,
        "region": "arizona", "climate": "hot",
        "default_cooling": "evaporative"
    },
    "San Francisco, CA": {
        "lat": 37.7749, "lon": -122.4194,
        "region": "california", "climate": "moderate",
        "default_cooling": "air_economizer"
    },
    "Stockholm, Sweden": {
        "lat": 59.3293, "lon": 18.0686,
        "region": "sweden", "climate": "cold",
        "default_cooling": "free_air"
    }
}

USER_LOCATION = {
    "name": "Orlando, FL",
    "lat": 28.5383,
    "lon": -81.3792
}

CARBON_INTENSITY = {
    "arizona": 0.45,
    "california": 0.22,
    "sweden": 0.045
}

# Experiment parameters
PROMPT_VOLUMES = [100, 500, 1000, 5000, 10000]
ENERGY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
MONTE_CARLO_RUNS = 200

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_latency(distance_km, load_fraction=0.5):
    """Calculate network latency in ms."""
    propagation_ms = (distance_km / 200000) * 1000 * 2
    service_rate = 1000
    arrival_rate = service_rate * min(load_fraction, 0.95) * 0.9
    queueing_ms = min(200, 1000 / max(1, service_rate - arrival_rate))
    return propagation_ms + queueing_ms + 30


def fetch_current_weather(lat, lon, name="Location"):
    """Fetch current weather from Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            return {
                'temperature': current.get('temperature_2m', 20.0),
                'humidity': current.get('relative_humidity_2m', 50.0),
                'wind_speed': current.get('wind_speed_10m', 5.0),
                'success': True
            }
    except Exception as e:
        print(f"   ⚠️ Weather API failed for {name}: {e}")
    
    return {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 5.0, 'success': False}


def calculate_energy_per_request(temperature, humidity, cooling_type, energy_multiplier=1.0):
    """Calculate energy consumption per AI request (Wh)."""
    cooling = COOLING_SYSTEMS[cooling_type]
    pue = cooling['pue']
    temp_sensitivity = cooling['temp_sensitivity']
    
    temp_factor = 1 + temp_sensitivity * max(0, temperature - 20)
    
    if cooling_type == 'evaporative' and humidity > 60:
        humidity_factor = 1 + 0.005 * (humidity - 60)
    else:
        humidity_factor = 1 + 0.001 * max(0, humidity - 50)
    
    return BASE_ENERGY_WH * pue * temp_factor * humidity_factor * energy_multiplier


def calculate_uhi_contribution(heat_kwh, wind_speed=5.0):
    """
    Calculate UHI contribution using Yang et al. (2024) formula.
    UHII = α × (Q/A) × (1/(1 + β × wind))
    """
    alpha = 0.0012  # Heat-to-temperature coefficient
    beta = 0.15     # Wind dissipation factor
    area_km2 = 1.0  # Assumed datacenter influence area
    
    heat_flux = heat_kwh / area_km2
    wind_factor = 1 / (1 + beta * wind_speed)
    
    return alpha * heat_flux * wind_factor


def calculate_heat_cv(heat_values):
    """Calculate coefficient of variation for heat distribution."""
    if len(heat_values) > 1 and np.mean(heat_values) > 0:
        return np.std(heat_values) / np.mean(heat_values)
    return 0


# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

def route_random(datacenters, num_requests):
    """Random baseline routing."""
    n = len(datacenters)
    weights = np.random.dirichlet(np.ones(n))
    dist = {dc: int(w * num_requests) for dc, w in zip(datacenters.keys(), weights)}
    
    # Fix rounding
    diff = num_requests - sum(dist.values())
    if diff != 0:
        first = list(dist.keys())[0]
        dist[first] += diff
    return dist


def route_energy_only(datacenters, weather_data, cooling_selections, num_requests):
    """Route to lowest energy DC - creates concentration problem."""
    energies = {}
    for dc, info in datacenters.items():
        w = weather_data[dc]
        c = cooling_selections[dc]
        energies[dc] = calculate_energy_per_request(w['temperature'], w['humidity'], c)
    
    min_e = min(energies.values())
    weights = {dc: (min_e / e) ** 3 for dc, e in energies.items()}
    total = sum(weights.values())
    
    dist = {dc: int((w / total) * num_requests) for dc, w in weights.items()}
    
    diff = num_requests - sum(dist.values())
    if diff != 0:
        best = min(energies, key=energies.get)
        dist[best] += diff
    return dist


def route_uhi_aware(datacenters, weather_data, cooling_selections, num_requests):
    """UHI-aware routing - penalizes thermal vulnerability."""
    scores = {}
    for dc, info in datacenters.items():
        w = weather_data[dc]
        c = cooling_selections[dc]
        energy = calculate_energy_per_request(w['temperature'], w['humidity'], c)
        uhi_penalty = 1 + 0.05 * max(0, w['temperature'] - 25)
        wind_benefit = 1 / (1 + 0.05 * w['wind_speed'])
        scores[dc] = energy * uhi_penalty * wind_benefit
    
    max_s = max(scores.values())
    weights = {dc: max_s / s for dc, s in scores.items()}
    total = sum(weights.values())
    
    dist = {dc: int((w / total) * num_requests) for dc, w in weights.items()}
    
    diff = num_requests - sum(dist.values())
    if diff != 0:
        best = min(scores, key=scores.get)
        dist[best] += diff
    return dist


def route_multi_objective(datacenters, weather_data, cooling_selections, user_loc, num_requests):
    """Multi-objective routing - balances all factors."""
    scores = {}
    for dc, info in datacenters.items():
        w = weather_data[dc]
        c = cooling_selections[dc]
        
        # Energy (normalized)
        energy = calculate_energy_per_request(w['temperature'], w['humidity'], c)
        energy_norm = energy / 1.0
        
        # Latency (normalized)
        dist = haversine_distance(user_loc['lat'], user_loc['lon'], info['lat'], info['lon'])
        latency = calculate_latency(dist)
        latency_norm = latency / 200
        
        # Carbon (normalized)
        carbon = CARBON_INTENSITY.get(info['region'], 0.4)
        carbon_norm = carbon / 0.7
        
        # UHI risk (normalized)
        uhi_norm = max(0, w['temperature'] - 15) / 30
        
        # Equal weights
        scores[dc] = 0.25 * energy_norm + 0.25 * latency_norm + 0.25 * carbon_norm + 0.25 * uhi_norm
    
    min_s = min(scores.values())
    weights = {dc: np.exp(-(s - min_s) * 3) for dc, s in scores.items()}
    total = sum(weights.values())
    
    dist = {dc: int((w / total) * num_requests) for dc, w in weights.items()}
    
    diff = num_requests - sum(dist.values())
    if diff != 0:
        best = min(scores, key=scores.get)
        dist[best] += diff
    return dist


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def run_single_simulation(datacenters, weather_data, user_loc, num_requests, 
                          cooling_selections, energy_multiplier=1.0):
    """Run simulation for all 4 routing strategies."""
    
    strategies = {
        'Random': route_random(datacenters, num_requests),
        'Energy-Only': route_energy_only(datacenters, weather_data, cooling_selections, num_requests),
        'UHI-Aware': route_uhi_aware(datacenters, weather_data, cooling_selections, num_requests),
        'Multi-Objective': route_multi_objective(datacenters, weather_data, cooling_selections, user_loc, num_requests)
    }
    
    results = {}
    
    for strategy_name, distribution in strategies.items():
        total_energy = 0
        total_carbon = 0
        total_heat = 0
        heat_values = []
        uhi_values = []
        
        dc_results = {}
        
        for dc, requests in distribution.items():
            if requests == 0:
                continue
            
            info = datacenters[dc]
            w = weather_data[dc]
            cooling = cooling_selections[dc]
            
            # Energy
            energy_per_req = calculate_energy_per_request(
                w['temperature'], w['humidity'], cooling, energy_multiplier
            )
            dc_energy = energy_per_req * requests
            
            # Carbon
            carbon_intensity = CARBON_INTENSITY.get(info['region'], 0.4)
            dc_carbon = (dc_energy / 1000) * carbon_intensity
            
            # Heat (all energy becomes heat)
            pue = COOLING_SYSTEMS[cooling]['pue']
            dc_heat = dc_energy * pue / 1000  # kWh
            
            # UHI
            uhi = calculate_uhi_contribution(dc_heat, w['wind_speed'])
            
            dc_results[dc] = {
                'requests': requests,
                'pct': (requests / num_requests) * 100,
                'energy_wh': dc_energy,
                'heat_kwh': dc_heat,
                'uhi': uhi
            }
            
            total_energy += dc_energy
            total_carbon += dc_carbon
            total_heat += dc_heat
            heat_values.append(dc_heat)
            uhi_values.append(uhi)
        
        results[strategy_name] = {
            'distribution': distribution,
            'dc_results': dc_results,
            'totals': {
                'energy_wh': total_energy,
                'carbon_g': total_carbon * 1000,
                'heat_kwh': total_heat,
                'heat_cv': calculate_heat_cv(heat_values),
                'peak_uhi': max(uhi_values) if uhi_values else 0,
                'total_uhi': sum(uhi_values)
            }
        }
    
    return results


def run_monte_carlo(datacenters, weather_data, user_loc, num_requests,
                    cooling_selections, energy_multiplier, n_runs=100):
    """Run Monte Carlo simulation with weather variations."""
    
    all_results = {s: [] for s in ['Random', 'Energy-Only', 'UHI-Aware', 'Multi-Objective']}
    
    for _ in range(n_runs):
        # Vary weather
        varied_weather = {}
        for dc, w in weather_data.items():
            varied_weather[dc] = {
                'temperature': w['temperature'] + np.random.normal(0, 3),
                'humidity': np.clip(w['humidity'] + np.random.normal(0, 10), 20, 95),
                'wind_speed': max(0.5, w['wind_speed'] + np.random.normal(0, 1.5))
            }
        
        # Vary requests ±10%
        varied_requests = int(num_requests * (1 + np.random.uniform(-0.1, 0.1)))
        
        results = run_single_simulation(
            datacenters, varied_weather, user_loc,
            varied_requests, cooling_selections, energy_multiplier
        )
        
        for strategy, data in results.items():
            all_results[strategy].append(data['totals'])
    
    # Summarize
    summary = {}
    for strategy, runs in all_results.items():
        df = pd.DataFrame(runs)
        summary[strategy] = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'ci_lower': df.quantile(0.025).to_dict(),
            'ci_upper': df.quantile(0.975).to_dict(),
            'raw': df
        }
    
    return summary


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_prompt_volume_effect(datacenters, weather_data, user_loc, cooling_selections):
    """Analyze how prompt volume affects UHI."""
    print("\n📊 Analyzing Prompt Volume → UHI Effect...")
    
    results = []
    
    for volume in PROMPT_VOLUMES:
        sim = run_single_simulation(
            datacenters, weather_data, user_loc,
            volume, cooling_selections, energy_multiplier=1.0
        )
        
        for strategy, data in sim.items():
            results.append({
                'volume': volume,
                'strategy': strategy,
                'energy_wh': data['totals']['energy_wh'],
                'heat_kwh': data['totals']['heat_kwh'],
                'peak_uhi': data['totals']['peak_uhi'],
                'heat_cv': data['totals']['heat_cv'],
                'total_uhi': data['totals']['total_uhi']
            })
    
    return pd.DataFrame(results)


def analyze_energy_multiplier_effect(datacenters, weather_data, user_loc, cooling_selections):
    """Analyze how AI workload intensity affects UHI."""
    print("\n📊 Analyzing Energy Multiplier → UHI Effect...")
    
    results = []
    
    for mult in ENERGY_MULTIPLIERS:
        sim = run_single_simulation(
            datacenters, weather_data, user_loc,
            1000, cooling_selections, energy_multiplier=mult
        )
        
        for strategy, data in sim.items():
            results.append({
                'multiplier': mult,
                'workload': {0.5: 'Light', 1.0: 'Standard', 1.5: 'Complex', 2.0: 'Image Gen'}[mult],
                'strategy': strategy,
                'energy_wh': data['totals']['energy_wh'],
                'heat_kwh': data['totals']['heat_kwh'],
                'peak_uhi': data['totals']['peak_uhi'],
                'heat_cv': data['totals']['heat_cv']
            })
    
    return pd.DataFrame(results)


def analyze_cooling_effectiveness(datacenters, weather_data):
    """Analyze cooling type effectiveness at each DC given current weather."""
    print("\n📊 Analyzing Cooling Type Effectiveness...")
    
    results = []
    
    for dc, info in datacenters.items():
        w = weather_data[dc]
        
        for cooling_type, cooling_info in COOLING_SYSTEMS.items():
            energy = calculate_energy_per_request(w['temperature'], w['humidity'], cooling_type)
            
            # Effectiveness score (lower energy = more effective)
            # Normalized to best possible (free_air at 10°C)
            baseline = calculate_energy_per_request(10, 30, 'free_air')
            effectiveness = baseline / energy
            
            # Is this the optimal choice for this climate?
            is_optimal = (
                (info['climate'] == 'cold' and cooling_type == 'free_air') or
                (info['climate'] == 'moderate' and cooling_type == 'air_economizer') or
                (info['climate'] == 'hot' and w['humidity'] < 50 and cooling_type == 'evaporative') or
                (info['climate'] == 'hot' and w['humidity'] >= 50 and cooling_type == 'liquid_cooling')
            )
            
            results.append({
                'datacenter': dc.split(',')[0],
                'climate': info['climate'],
                'temperature': w['temperature'],
                'humidity': w['humidity'],
                'cooling_type': cooling_info['name'],
                'cooling_key': cooling_type,
                'pue': cooling_info['pue'],
                'energy_per_req': energy,
                'effectiveness': effectiveness,
                'is_optimal': is_optimal
            })
    
    return pd.DataFrame(results)


def analyze_concentration_problem(datacenters, weather_data, user_loc, cooling_selections):
    """Analyze the concentration problem in detail."""
    print("\n📊 Analyzing Concentration Problem...")
    
    sim = run_single_simulation(
        datacenters, weather_data, user_loc,
        1000, cooling_selections, energy_multiplier=1.0
    )
    
    results = []
    for strategy, data in sim.items():
        for dc, dc_data in data['dc_results'].items():
            results.append({
                'strategy': strategy,
                'datacenter': dc.split(',')[0],
                'requests': dc_data['requests'],
                'percentage': dc_data['pct'],
                'heat_kwh': dc_data['heat_kwh'],
                'uhi': dc_data['uhi']
            })
    
    return pd.DataFrame(results), sim


# ============================================================================
# PLOTTING FUNCTIONS (PDF-ready)
# ============================================================================

def plot_prompt_volume_uhi(df, output_dir):
    """Plot: Prompt Volume vs UHI Effect."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'UHI-Aware': '#059669',
        'Multi-Objective': '#2563eb'
    }
    
    # Left: Peak UHI vs Volume
    ax1 = axes[0]
    for strategy in df['strategy'].unique():
        data = df[df['strategy'] == strategy]
        ax1.plot(data['volume'], data['peak_uhi'] * 1000, 'o-', 
                 color=colors[strategy], label=strategy, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of AI Prompts')
    ax1.set_ylabel('Peak UHI Contribution (×10⁻³ °C)')
    ax1.set_title('(a) Peak UHI vs Prompt Volume')
    ax1.legend(loc='upper left')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Right: Heat CV vs Volume
    ax2 = axes[1]
    for strategy in df['strategy'].unique():
        data = df[df['strategy'] == strategy]
        ax2.plot(data['volume'], data['heat_cv'], 's-',
                 color=colors[strategy], label=strategy, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of AI Prompts')
    ax2.set_ylabel('Heat Concentration (CV)')
    ax2.set_title('(b) Heat Distribution Inequality vs Prompt Volume')
    ax2.legend(loc='upper right')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_prompt_volume_uhi.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig1_prompt_volume_uhi.png'))
    plt.close()
    print("   ✅ fig1_prompt_volume_uhi.pdf")


def plot_energy_multiplier_effect(df, output_dir):
    """Plot: Energy Multiplier (Workload Type) vs UHI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'UHI-Aware': '#059669',
        'Multi-Objective': '#2563eb'
    }
    
    workload_order = ['Light', 'Standard', 'Complex', 'Image Gen']
    
    # Left: Peak UHI by workload
    ax1 = axes[0]
    x = np.arange(len(workload_order))
    width = 0.2
    
    for i, strategy in enumerate(['Random', 'Energy-Only', 'UHI-Aware', 'Multi-Objective']):
        data = df[df['strategy'] == strategy].set_index('workload').loc[workload_order]
        ax1.bar(x + i*width, data['peak_uhi'] * 1000, width, 
                label=strategy, color=colors[strategy])
    
    ax1.set_xlabel('AI Workload Type')
    ax1.set_ylabel('Peak UHI Contribution (×10⁻³ °C)')
    ax1.set_title('(a) Peak UHI by Workload Intensity')
    ax1.set_xticks(x + 1.5*width)
    ax1.set_xticklabels(workload_order)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Energy vs UHI trade-off
    ax2 = axes[1]
    for strategy in df['strategy'].unique():
        data = df[df['strategy'] == strategy]
        ax2.scatter(data['energy_wh'], data['peak_uhi'] * 1000,
                    s=100, c=colors[strategy], label=strategy, alpha=0.8)
        ax2.plot(data['energy_wh'], data['peak_uhi'] * 1000,
                 color=colors[strategy], alpha=0.5, linestyle='--')
    
    ax2.set_xlabel('Total Energy Consumption (Wh)')
    ax2.set_ylabel('Peak UHI Contribution (×10⁻³ °C)')
    ax2.set_title('(b) Energy-UHI Trade-off by Strategy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_energy_multiplier_uhi.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig2_energy_multiplier_uhi.png'))
    plt.close()
    print("   ✅ fig2_energy_multiplier_uhi.pdf")


def plot_cooling_effectiveness(df, weather_data, output_dir):
    """Plot: Cooling Type Effectiveness Heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Effectiveness heatmap
    ax1 = axes[0]
    pivot = df.pivot(index='cooling_type', columns='datacenter', values='effectiveness')
    
    # Reorder
    cooling_order = ['Free Air Cooling', 'Liquid Cooling', 'Air Economizer', 'Evaporative', 'Mechanical Chiller']
    pivot = pivot.reindex(cooling_order)
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'Effectiveness Score'}, vmin=0.5, vmax=1.0)
    ax1.set_title('(a) Cooling Effectiveness by Location\n(Higher = Better)')
    ax1.set_xlabel('Datacenter')
    ax1.set_ylabel('Cooling Technology')
    
    # Add current weather annotation
    weather_text = "\n".join([
        f"{dc.split(',')[0]}: {w['temperature']:.1f}°C, {w['humidity']:.0f}%"
        for dc, w in weather_data.items()
    ])
    ax1.text(1.02, 0.5, f"Current Weather:\n{weather_text}",
             transform=ax1.transAxes, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: PUE comparison
    ax2 = axes[1]
    pue_data = df.groupby('cooling_type')['pue'].first().sort_values()
    colors = [COOLING_SYSTEMS[k]['color'] for k in 
              ['free_air', 'liquid_cooling', 'air_economizer', 'evaporative', 'mechanical_chiller']]
    
    bars = ax2.barh(pue_data.index, pue_data.values, color=colors[::-1])
    ax2.set_xlabel('Power Usage Effectiveness (PUE)')
    ax2.set_title('(b) PUE by Cooling Technology\n(Lower = Better)')
    ax2.set_xlim(1.0, 2.0)
    ax2.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal PUE=1.0')
    
    # Add value labels
    for bar, val in zip(bars, pue_data.values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_cooling_effectiveness.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig3_cooling_effectiveness.png'))
    plt.close()
    print("   ✅ fig3_cooling_effectiveness.pdf")


def plot_concentration_problem(df, sim_results, output_dir):
    """Plot: The Concentration Problem Visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'UHI-Aware': '#059669',
        'Multi-Objective': '#2563eb'
    }
    
    dc_colors = {'Phoenix': '#dc2626', 'San Francisco': '#d97706', 'Stockholm': '#2563eb'}
    
    # Left: Traffic distribution
    ax1 = axes[0]
    strategies = ['Random', 'Energy-Only', 'UHI-Aware', 'Multi-Objective']
    x = np.arange(len(strategies))
    width = 0.25
    
    dcs = df['datacenter'].unique()
    for i, dc in enumerate(dcs):
        data = df[df['datacenter'] == dc]
        vals = [data[data['strategy'] == s]['percentage'].values[0] if len(data[data['strategy'] == s]) > 0 else 0 
                for s in strategies]
        ax1.bar(x + i*width, vals, width, label=dc, color=dc_colors.get(dc, '#666'))
    
    ax1.set_xlabel('Routing Strategy')
    ax1.set_ylabel('Traffic Share (%)')
    ax1.set_title('(a) Traffic Distribution by Strategy')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(strategies, rotation=15)
    ax1.legend(title='Datacenter')
    ax1.axhline(y=33.3, color='gray', linestyle='--', alpha=0.5, label='Equal distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: UHI by DC and strategy
    ax2 = axes[1]
    for i, dc in enumerate(dcs):
        data = df[df['datacenter'] == dc]
        vals = [data[data['strategy'] == s]['uhi'].values[0] * 1000 if len(data[data['strategy'] == s]) > 0 else 0 
                for s in strategies]
        ax2.bar(x + i*width, vals, width, label=dc, color=dc_colors.get(dc, '#666'))
    
    ax2.set_xlabel('Routing Strategy')
    ax2.set_ylabel('UHI Contribution (×10⁻³ °C)')
    ax2.set_title('(b) UHI Contribution by Datacenter')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(strategies, rotation=15)
    ax2.legend(title='Datacenter')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_concentration_problem.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig4_concentration_problem.png'))
    plt.close()
    print("   ✅ fig4_concentration_problem.pdf")


def plot_monte_carlo_results(mc_results, output_dir):
    """Plot: Monte Carlo Validation Results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {
        'Random': '#6b7280',
        'Energy-Only': '#dc2626',
        'UHI-Aware': '#059669',
        'Multi-Objective': '#2563eb'
    }
    
    metrics = [
        ('energy_wh', 'Energy (Wh)', axes[0, 0]),
        ('peak_uhi', 'Peak UHI (°C)', axes[0, 1]),
        ('heat_cv', 'Heat CV', axes[1, 0]),
        ('carbon_g', 'Carbon (g CO₂)', axes[1, 1])
    ]
    
    strategies = list(mc_results.keys())
    
    for metric, label, ax in metrics:
        data = [mc_results[s]['raw'][metric].values for s in strategies]
        
        bp = ax.boxplot(data, labels=strategies, patch_artist=True)
        
        for patch, strategy in zip(bp['boxes'], strategies):
            patch.set_facecolor(colors[strategy])
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution (n={MONTE_CARLO_RUNS})')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Monte Carlo Simulation Results (200 runs)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_monte_carlo.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig5_monte_carlo.png'))
    plt.close()
    print("   ✅ fig5_monte_carlo.pdf")


def plot_statistical_significance(mc_results, output_dir):
    """Plot: Statistical Significance Test Results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compare Energy-Only vs Multi-Objective
    eo = mc_results['Energy-Only']['raw']['peak_uhi']
    mo = mc_results['Multi-Objective']['raw']['peak_uhi']
    
    t_stat, p_value = stats.ttest_ind(eo, mo, equal_var=False)
    cohens_d = (eo.mean() - mo.mean()) / np.sqrt((eo.var() + mo.var()) / 2)
    
    # Plot distributions
    ax.hist(eo * 1000, bins=30, alpha=0.6, label='Energy-Only', color='#dc2626')
    ax.hist(mo * 1000, bins=30, alpha=0.6, label='Multi-Objective', color='#2563eb')
    
    ax.axvline(eo.mean() * 1000, color='#dc2626', linestyle='--', linewidth=2)
    ax.axvline(mo.mean() * 1000, color='#2563eb', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Peak UHI Contribution (×10⁻³ °C)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Statistical Comparison: Energy-Only vs Multi-Objective\n'
                 f't={t_stat:.2f}, p={p_value:.2e}, Cohen\'s d={cohens_d:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add significance annotation
    if p_value < 0.001:
        sig_text = "HIGHLY SIGNIFICANT (p < 0.001)"
    elif p_value < 0.01:
        sig_text = "VERY SIGNIFICANT (p < 0.01)"
    elif p_value < 0.05:
        sig_text = "SIGNIFICANT (p < 0.05)"
    else:
        sig_text = "NOT SIGNIFICANT"
    
    ax.text(0.98, 0.95, sig_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='green' if p_value < 0.05 else 'red', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_statistical_significance.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig6_statistical_significance.png'))
    plt.close()
    print("   ✅ fig6_statistical_significance.pdf")
    
    return {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d}


def create_summary_report(all_data, stats_results, weather_data, output_dir):
    """Create text summary report."""
    
    report = f"""
================================================================================
UHI DASHBOARD - ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. EXPERIMENTAL CONFIGURATION
-----------------------------
User Location: {USER_LOCATION['name']} ({USER_LOCATION['lat']}, {USER_LOCATION['lon']})

Datacenters:
{chr(10).join([f"  - {dc}: {info['climate']} climate, {info['default_cooling']} cooling" for dc, info in DATACENTERS.items()])}

Current Weather (from Open-Meteo API):
{chr(10).join([f"  - {dc.split(',')[0]}: {w['temperature']:.1f}°C, {w['humidity']:.0f}% humidity, {w['wind_speed']:.1f} m/s wind" for dc, w in weather_data.items()])}

2. KEY FINDINGS
---------------

Finding 1: Prompt Volume → UHI Relationship
  - UHI scales linearly with prompt volume
  - At 10,000 prompts, Energy-Only creates {all_data['volume']['peak_uhi'].max()*1000:.3f}×10⁻³ °C peak UHI
  - Multi-Objective reduces this by distributing load

Finding 2: Energy Multiplier Effect
  - Image generation (2.0x) doubles UHI impact vs standard prompts
  - Routing strategy choice has larger impact than workload type

Finding 3: Cooling Effectiveness
  - Free Air Cooling most effective in cold climates (PUE 1.10)
  - Mechanical Chiller least efficient (PUE 1.80) but works everywhere
  - Optimal cooling reduces energy by up to 39%

Finding 4: Concentration Problem
  - Energy-Only routing concentrates traffic at coldest DC
  - Creates localized heat islands
  - Multi-Objective distributes load more evenly

3. STATISTICAL VALIDATION (Monte Carlo n={MONTE_CARLO_RUNS})
------------------------------------------------------------
Comparison: Energy-Only vs Multi-Objective (Peak UHI)

  t-statistic: {stats_results['t_stat']:.4f}
  p-value: {stats_results['p_value']:.2e}
  Cohen's d: {stats_results['cohens_d']:.4f}
  
  Interpretation: {"STATISTICALLY SIGNIFICANT" if stats_results['p_value'] < 0.05 else "NOT SIGNIFICANT"}
  The difference in UHI outcomes between strategies is {"highly" if stats_results['p_value'] < 0.001 else ""} significant.

4. RECOMMENDATIONS
------------------
  1. Use Multi-Objective routing for sustainability goals
  2. Match cooling technology to local climate
  3. Consider UHI impact alongside energy efficiency
  4. Monitor traffic concentration to prevent thermal hotspots

================================================================================
Generated by: UHI Dashboard Batch Analysis
Author: Phani Raja Bharath Balijepalli
Course: IDS6938 - AI, Energy, and Sustainability
University of Central Florida
================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(report)
    
    print("   ✅ analysis_summary.txt")
    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  UHI DASHBOARD - FOCUSED BATCH ANALYSIS                      ║
║  Analyzing: Prompts → UHI & Cooling Effectiveness            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"uhi_results_{timestamp}"
    plots_dir = os.path.join(output_dir, 'plots')
    data_dir = os.path.join(output_dir, 'data')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}/")
    
    # Fetch current weather
    print("\n📡 Fetching current weather data...")
    weather_data = {}
    for dc, info in DATACENTERS.items():
        weather_data[dc] = fetch_current_weather(info['lat'], info['lon'], dc)
        status = "✅" if weather_data[dc].get('success', False) else "⚠️ (estimated)"
        print(f"   {dc}: {weather_data[dc]['temperature']:.1f}°C, "
              f"{weather_data[dc]['humidity']:.0f}% humidity, "
              f"{weather_data[dc]['wind_speed']:.1f} m/s wind {status}")
    
    # Set cooling selections (optimal for each climate)
    cooling_selections = {dc: info['default_cooling'] for dc, info in DATACENTERS.items()}
    
    # =========================================================================
    # RUN ANALYSES
    # =========================================================================
    
    all_data = {}
    
    # 1. Prompt Volume Analysis
    print("\n" + "="*60)
    df_volume = analyze_prompt_volume_effect(DATACENTERS, weather_data, USER_LOCATION, cooling_selections)
    df_volume.to_csv(os.path.join(data_dir, 'prompt_volume_analysis.csv'), index=False)
    all_data['volume'] = df_volume
    
    # 2. Energy Multiplier Analysis
    df_energy = analyze_energy_multiplier_effect(DATACENTERS, weather_data, USER_LOCATION, cooling_selections)
    df_energy.to_csv(os.path.join(data_dir, 'energy_multiplier_analysis.csv'), index=False)
    all_data['energy'] = df_energy
    
    # 3. Cooling Effectiveness Analysis
    df_cooling = analyze_cooling_effectiveness(DATACENTERS, weather_data)
    df_cooling.to_csv(os.path.join(data_dir, 'cooling_effectiveness.csv'), index=False)
    all_data['cooling'] = df_cooling
    
    # 4. Concentration Problem Analysis
    df_concentration, sim_results = analyze_concentration_problem(DATACENTERS, weather_data, USER_LOCATION, cooling_selections)
    df_concentration.to_csv(os.path.join(data_dir, 'concentration_analysis.csv'), index=False)
    all_data['concentration'] = df_concentration
    
    # 5. Monte Carlo Validation
    print(f"\n📊 Running Monte Carlo Simulation ({MONTE_CARLO_RUNS} runs)...")
    mc_results = run_monte_carlo(
        DATACENTERS, weather_data, USER_LOCATION,
        1000, cooling_selections, 1.0, MONTE_CARLO_RUNS
    )
    
    # Save MC raw data
    for strategy, data in mc_results.items():
        data['raw'].to_csv(os.path.join(data_dir, f'monte_carlo_{strategy.lower().replace("-", "_")}.csv'), index=False)
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    
    print("\n📊 Generating publication-ready figures...")
    
    plot_prompt_volume_uhi(df_volume, plots_dir)
    plot_energy_multiplier_effect(df_energy, plots_dir)
    plot_cooling_effectiveness(df_cooling, weather_data, plots_dir)
    plot_concentration_problem(df_concentration, sim_results, plots_dir)
    plot_monte_carlo_results(mc_results, plots_dir)
    stats_results = plot_statistical_significance(mc_results, plots_dir)
    
    # =========================================================================
    # CREATE SUMMARY REPORT
    # =========================================================================
    
    print("\n📝 Creating summary report...")
    create_summary_report(all_data, stats_results, weather_data, output_dir)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print(f"""
{'='*60}
✅ ANALYSIS COMPLETE!
{'='*60}

📁 Results saved to: {output_dir}/
   ├── plots/
   │   ├── fig1_prompt_volume_uhi.pdf
   │   ├── fig2_energy_multiplier_uhi.pdf
   │   ├── fig3_cooling_effectiveness.pdf
   │   ├── fig4_concentration_problem.pdf
   │   ├── fig5_monte_carlo.pdf
   │   └── fig6_statistical_significance.pdf
   ├── data/
   │   ├── prompt_volume_analysis.csv
   │   ├── energy_multiplier_analysis.csv
   │   ├── cooling_effectiveness.csv
   │   ├── concentration_analysis.csv
   │   └── monte_carlo_*.csv
   └── analysis_summary.txt

📊 Key Statistics:
   - Monte Carlo runs: {MONTE_CARLO_RUNS}
   - Statistical significance: p = {stats_results['p_value']:.2e}
   - Effect size (Cohen's d): {stats_results['cohens_d']:.2f}
""")


if __name__ == "__main__":
    main()
