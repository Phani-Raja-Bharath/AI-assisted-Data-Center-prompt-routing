# UHI Dashboard - Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run ai_assisted_datacenter_allocation.py
```

## Features

### Four focused workspaces
- **Overview**: key metrics, routing map, traffic split, recommendation, and expandable source data.
- **Routing**: strategy comparison, service coverage, cooling controls, and carbon patterns.
- **Models**: training provenance, model comparison, diagnostics, and explicit retraining.
- **Experiments**: Monte Carlo, temperature sensitivity, and what-if scenarios, executed only when requested.

Configure location, datacenters, workload, and constraints in the sidebar. Use the top **Español** switch for technical Spanish. Open **Sources and methodology** to distinguish published data from model assumptions. View changes preserve models and cached results.

### Key Features
- Results are calculated from modeled inputs; constants and fallback weather are assumptions
- Real-time weather from Open-Meteo API
- Explicit PUE scenario assumptions, with Alkrush et al. (2024) cited for cooling context
- Statistical significance testing with scipy
- Adjustable parameters via sliders
- PDF/CSV export

## Research Contribution

This dashboard demonstrates the **Stockholm Concentration Problem**:
- Energy-only routing can concentrate traffic at the most energy-efficient DC
- Creates localized heat dissipation defeating sustainability goals
- Multi-objective routing compares heat, energy, carbon, and latency; reductions depend on the scenario

## Citation

Balijepalli, P.R.B. (2025). AI-Assisted Datacenter Routing for UHI Mitigation.
IDS6938 - AI, Energy, and Sustainability. University of Central Florida.

## Academic References

- Alkrush et al. (2024) - Cooling PUE values
- Stern (2025) - Energy per AI prompt
- Yang et al. (2024) - UHI formula
- Luccioni & Hernandez-Garcia (2023) - Carbon emissions

## Training and persistence

The first run trains and saves the selected models. Later runs reuse them; **Retrain selected models** forces a refresh. Old bundles require one fresh training pass after the schema update. Local artifacts survive server restarts on persistent disk. Optional Neon storage supports hosted deployments: install `requirements-neon.txt` and configure `MODEL_DATABASE_URL` in environment variables or Streamlit secrets. See README.md for setup and modeling limitations.
