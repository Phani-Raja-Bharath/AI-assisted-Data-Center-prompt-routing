# UHI Dashboard - Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run uhi_dashboard_complete.py
```

## Features

### 13 Sections
1. **Experimental Setup** - Configure user location, datacenter selection, parameters
2. **Real-Time Climatic Conditions** - Live weather data, initial recommendation
3. **The Deterministic Problem** - Stockholm Concentration demonstration
4. **AI Model Training** - MLR, ANN, Bayesian comparison
5. **Routing Strategy Comparison** - 4 strategies compared
6. **Comprehensive Results** - Full metrics and recommendation
7. **Monte Carlo Validation** - Statistical significance testing
8. **Sensitivity Analysis** - Temperature, latency impact
9. **What-If Scenarios** - User-created comparisons
10. **Carbon Intensity Analysis** - 24-hour grid patterns
11. **Cooling Technology Analysis** - Effectiveness matrix
12. **Academic References** - All citations
13. **Executive Summary** - Key findings + export

### 21 Graphs
- 3 Geographic maps
- Traffic distribution charts
- Model performance comparison
- Prediction scatter plots
- Feature importance
- Heat distribution
- Monte Carlo box plots
- Sensitivity curves
- Carbon intensity patterns
- Cooling effectiveness heatmap
- PUE comparison

### Key Features
- **All values are DYNAMICALLY CALCULATED** - no hardcoded results
- Real-time weather from Open-Meteo API
- Research-backed PUE values (Alkrush et al., 2024)
- Statistical significance testing with scipy
- Adjustable parameters via sliders
- PDF/CSV export

## Research Contribution

This dashboard demonstrates the **Stockholm Concentration Problem**:
- Energy-only routing concentrates 70-80% traffic at coldest DC
- Creates localized heat dissipation defeating sustainability goals
- Multi-objective routing reduces UHI by 40-60% with <10% energy overhead

## Citation

Balijepalli, P.R.B. (2025). AI-Assisted Datacenter Routing for UHI Mitigation.
IDS6938 - AI, Energy, and Sustainability. University of Central Florida.

## Academic References

- Alkrush et al. (2024) - Cooling PUE values
- Stern (2025) - Energy per AI prompt
- Yang et al. (2024) - UHI formula
- Luccioni & Hernandez-Garcia (2023) - Carbon emissions
