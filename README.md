
---

## Modeling Approach

### Energy Consumption per Request

Energy consumed by an AI request is modeled as:

E = E_base × PUE × (1 + α × max(0, T − 20)) × H_factor

Where:
- `E_base` ≈ 0.3 Wh per AI prompt
- `PUE` depends on cooling technology
- Temperature and humidity adjust cooling overhead

---

### Urban Heat Island (UHI) Contribution

All consumed energy is eventually dissipated as heat:

ΔT = α × (Q / A) × 1 / (1 + β × wind_speed)

This formulation captures:
- Heat flux intensity
- Wind-based heat dissipation
- Local thermal vulnerability

---

### Network Latency Model

Latency is decomposed into:
- Propagation delay
- Queueing delay (M/M/1)
- Processing delay

---

## Machine Learning Models

The system predicts energy consumption per request using:

| Model | Purpose |
|------|--------|
| Multiple Linear Regression (MLR) | Baseline and interpretability |
| Artificial Neural Network (ANN) | Nonlinear behavior capture |
| Bayesian-Optimized ANN | Performance-optimized ANN |

### Training Data
- Historical weather data (2021–2024) from Open-Meteo
- Physics-based synthetic fallback data
- Features: temperature, humidity, wind speed, cooling type
- Target: energy per AI request (Wh)

---

## Routing Strategies Evaluated

| Strategy | Description |
|--------|-------------|
| Random | Baseline control |
| Energy-Only | Minimizes energy, ignores UHI |
| UHI-Aware | Penalizes thermal vulnerability |
| Multi-Objective (Proposed) | Balances energy, latency, carbon, and UHI |

---

## Monte Carlo Validation

To ensure robustness:
- Weather parameters are perturbed stochastically
- Request volumes vary by ±20%
- 95% confidence intervals are computed
- Welch’s t-test and Cohen’s d are applied

Results show statistically significant reductions in peak UHI intensity.

---

## Key Outcomes

- 40–60% reduction in peak UHI intensity
- Over 50% reduction in spatial heat concentration
- Latency maintained within SLA constraints
- Modest energy overhead (<15%)

**Key Insight:**  
Routing strategy selection has a larger sustainability impact than the choice of AI model itself.

---

## Dashboard Features

- Real-time weather integration
- Geographic routing visualization
- Cooling technology comparison
- AI model performance analysis
- Monte Carlo result visualization
- Sensitivity analysis
- Exportable results (CSV)

---

## Tech Stack

- Python
- Streamlit
- NumPy, Pandas
- Scikit-learn
- Plotly
- Open-Meteo API

---

## How to Run

```bash
git clone https://github.com/<your-username>/ai-datacenter-uhi-routing.git
cd ai-datacenter-uhi-routing
pip install -r requirements.txt
streamlit run ai_assisted_datacenter_allocation.py
