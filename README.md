# Climate-aware AI datacenter routing

A Streamlit research dashboard comparing AI request routing by energy, latency, carbon emissions, and modeled local heat impact. The default sites are Phoenix, San Francisco, and Stockholm, with additional locations available.

The interface uses horizontal experiment controls and four workspaces: **Overview** (all strategies, routing map, traffic split), **Routing** (strategy comparison), **Models** (performance, diagnostics, explicit retraining), and **Experiments** (Monte Carlo, sensitivity, and alternative scenarios). Cooling dropdowns remain visible for each selected datacenter above the weather readings. Only **Run simulation** commits current inputs to routing results. Weather refreshes and settings changes leave the last run, map, receipt, and exports unchanged and show a pending-input warning. Analyses run on request using the last committed simulation snapshot. Sources and assumptions are available on the **Help** page. Plotly's toolbar exports figures without generating image files on every app rerun.

The **Español** switch at the top enables human-authored technical Spanish for controls, tables, figures, and analysis text. It preserves equations, units, source titles, canonical model identifiers, experiment settings, and trained-model caches. Language changes do not retrain models. The bilingual **Data provenance and modeling assumptions** panel distinguishes API-sourced weather-model estimates, annual grid averages, static scenario assumptions, predictions, and simulation results. Weather tables include source and valid-at time; carbon tables include source, year, and classification. Base energy, exact PUE settings, heat coefficients, latency parameters, and the qualitative cooling matrix are declared assumptions rather than attributed measurements. The cooling review is cited for context with its verified DOI; statistical summaries report calculated p-values rather than fixed significance claims.

Neuquén, Argentina is available in both the user-city and datacenter selectors (38.950° S, 68.067° W; [UN/IGN coordinates](https://ungegn.un.org/dashboard/cities/details?id=3058)). It uses the same Open-Meteo weather, map connections, latency calculation, and routing pipeline as other locations. Its carbon baseline fetches Argentina's latest annual [OWID/Ember grid intensity](https://ourworldindata.org/grapher/carbon-intensity-electricity) once daily, converts g CO2e/kWh to kg CO2e/kWh, and displays the source and year. If the feed is unavailable, a labeled 2025 fallback of 0.346 kg CO2e/kWh is used. This is a national grid proxy with modeled hourly variation, not live Neuquén-specific telemetry. Carbon charts and source tables follow the selected datacenters.

## Run locally

Use Python 3.12 or newer and a virtual environment created on your machine. If the copied `.venv` references a missing Python installation, create a new environment:

```powershell
python -m venv .venv-dev
.\.venv-dev\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-dev\Scripts\python.exe -m streamlit run ai_assisted_datacenter_allocation.py
```

## Model reuse

Click **RUN SIMULATION** to train the selected models once. Further simulation runs and widget interactions reuse the models in session state. Compatible bundles are automatically saved under `artifacts/models_<configuration hash>.joblib` and loaded after a browser refresh or server restart. The latest trained bundle is also written to `artifacts/ai_models_bundle.joblib` for benchmarking.

Use **Retrain selected models** to deliberately replace a bundle. **Train all models** (next to Run simulation) retrains MLR, ANN and Bayesian ANN; on the Models tab, **Retrain with a new data period** trains the selected model on hourly historical weather from a chosen date range (2021 onward, at least 30 days), which then becomes the active period for simulations (**Use default period** switches back to 2021-2024). Both show a warning with an estimated duration (all three models on 2021-2024 take about 15 minutes) and start only after **Confirm**; the dashboard cannot be used while training, and saved models are shared by every session. The training period is part of the bundle identity, so models from different periods are stored side by side. Changing model/label code, feature schema, cooling encoding, Bayesian availability, or scikit-learn version invalidates the relevant model cache. Selecting another training mode stages that choice until **Run simulation** or an explicit retraining request. Weather, workload, cooling, location, capacity, and latency changes are staged: existing results stay unchanged until **Run simulation**. Explicit retraining saves models without replacing the completed routing run. Monte Carlo uses that completed run's inputs and is recalculated only through its own analysis action.

Legacy bundles without the new schema metadata are ignored: one fresh training pass is required after this update. Only load model artifacts from a trusted application-controlled store.

## Optional Neon persistence

Local disk is sufficient on a persistent workstation. Neon provides shared Postgres persistence when hosting on an ephemeral filesystem or across application instances. For the current small model bundles, the optional backend stores serialized bundles in a `BYTEA` column and configuration metadata in `JSONB`. Larger future artifacts are better kept in object storage with their locations recorded in Postgres.

```powershell
.\.venv-dev\Scripts\python.exe -m pip install -r requirements-neon.txt
```

Set `MODEL_DATABASE_URL` to the connection string from Neon (including `sslmode=require`) in the hosting environment, or in an uncommitted `.streamlit/secrets.toml`. Copy `.streamlit/secrets.example.toml` to that path if needed, then uncomment and fill in the setting:

```toml
MODEL_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require"
```

The real secrets file, its common backup variants, `.env` variants, `.neon` context, and PostgreSQL credential files are ignored by Git; the example template remains trackable. The app reads `MODEL_DATABASE_URL` or Neon's standard `DATABASE_URL`, in this order: process environment, Streamlit secrets, `.env.local`, then `.env`. It does not export these file contents to the browser or to the process environment. On a hosting service, configure the secret in its secret manager. Ignore rules do not remove credentials from existing commits; rotate any credential that was previously committed.

The app creates `routing_model_bundles` on its first database access, so the configured role needs table-creation privileges. It checks session state, then local disk, then the database. Newly trained bundles are saved locally and, when configured, to Neon. Connection or storage failures produce a warning while preserving the trained model in the session. Existing local-only bundles are not automatically uploaded; use **Retrain selected models** after enabling Neon to publish a new bundle there. No database credentials are required for local operation.

### Neon CLI workflow and verification

`neon.ts` declares Postgres only. `neon deploy` applies that configuration and pulls its connection variables into `.env.local` (or an existing `.env`). The Streamlit server uses Psycopg 3 with the pooled `DATABASE_URL`; migrations use `DATABASE_URL_UNPOOLED`. The versioned schema is `migrations/001_model_bundles.sql`.

This checkout has workspace-local Node/npm/Neon tools in the ignored `.tools` directory. In PowerShell, activate them for the current terminal:

```powershell
. .\scripts\activate-tools.ps1
python -m pip install -r requirements-dev.txt
npm ci
npm run build
npm start
```

Use the active project's Python virtual environment for the install command above. The npm commands prefer the working `.venv-dev`, then `.venv`, then Python on PATH; `PYTHON` can override it. `npm run build` checks `neon.ts` with TypeScript, validates Python syntax/runtime imports, and runs the offline regression suite. Streamlit is interpreted, so there is no browser JavaScript bundle. The CLI manages the backend; the Streamlit frontend still runs locally or on a Python hosting service.

```powershell
# Test a schema migration on an isolated Neon branch before production.
neon branches create --parent production --name model-store-test --schema-only --no-secrets
neon env pull --branch model-store-test --service postgres --file .tools/.env.neon-test
npm run db:verify -- --env-file .tools/.env.neon-test --migrate

# After the branch test passes, migrate and test the linked production database.
npm run db:verify -- --migrate
```

The verification command checks a TLS connection, trains a small real MLR through the application, saves it in Postgres, restores it into an empty local directory, and verifies identical predictions without retraining. It deletes only its uniquely identified test record afterward. Database URLs are never printed. Test branches should be deleted after verification.

## Modeling

Energy starts at 0.3 Wh per request, adjusted by cooling PUE, temperature, and humidity. Surrogate training labels additionally include wind, pressure, solar radiation, and small synthetic noise. The model inputs are temperature, humidity, wind speed, pressure, solar radiation, and the cooling technology, which is **one-hot encoded** (one indicator column per technology) because it is a category, not an ordered number.

Training attempts hourly Open-Meteo weather for 2021-2024 and falls back to synthetic weather. Each training record is assigned a cooling technology **uniformly at random** (fixed seed), so every supported technology is trained across the full range of weather; the dashboard lets users pick any technology for any site, so the model must see those combinations. A separate **cooling comparison check** samples weather, compares every pair of technologies with the model and with the reference formula, and reports how often the model orders them the same way and the error in the predicted difference (shown on the Models tab). A high overall R² does not establish accuracy for cooling changes. **Energy labels always come from formulas, not measured datacenter consumption.** Models include multiple linear regression, a neural network, and an optional Bayesian-optimized neural network. Data is split approximately 70/15/15; model selection uses validation R-squared and test metrics are reported separately.

**A simulated batch is one hour of traffic.** Total Requests is requests per hour, so batch energy (Wh) equals average power over that hour (W). This is what the temperature-rise formula and the power-capacity limit expect, and it is why peak ΔT-AR grows with request volume.

All consumed energy becomes waste heat. The temperature-rise proxy scales with heat per assumed influence area (1 km² for every site) and decreases with wind. It does **not** depend on ambient temperature: a hot and a cold site warm equally per unit of heat, and climate enters only through wind and the energy estimate. This is a comparative proxy, not a calibrated urban climate prediction.

Carbon intensity uses regional constants (or the fetched Argentina annual average for Neuquén) and modeled hourly variation; it is not live grid telemetry. Each site is evaluated at its own **local solar hour** (UTC hour + longitude / 15), so the time-of-day pattern follows that site's sun rather than the server clock; solar time can differ from civil time by about an hour. Each site's carbon intensity is **calculated once when a simulation runs and recorded** in the execution receipt; routing, reporting, Monte Carlo and the what-if analyses reuse those frozen values, so results do not drift with the clock.

Latency combines geographic propagation, a bounded M/M/1 queue proxy, and fixed processing delay. Server utilization is the site's request rate over the hour (requests / 3,600 s) divided by its service rate (1,000 requests/s). At dashboard volumes queueing is therefore small and latency is driven mainly by distance; the queue matters only as a site approaches saturation. Server **throughput is a hard limit**: no site receives more than 99% of its service rate over the hour (about 3.56 million requests), independently of the latency limit, so an overloaded site cannot appear feasible because the displayed queue delay is capped. When a strategy serves no requests, its average latency is reported as unavailable rather than 0 ms.

## Five dashboard strategies

| Strategy | Behavior |
|---|---|
| Random | Dirichlet-weighted random baseline |
| Energy-Only | Favors lower energy per request |
| Heat-Only | Favors lower wind-adjusted temperature-rise contribution per request |
| Delta-T-Aware (Delta-T-AR) | Minimizes the peak modeled temperature rise: splits requests so every site warms equally (requests ∝ 1 / ΔT per request). An exact integer minimax allocation is used, so the result is the true optimum for whole requests within all site limits |
| Multi-Objective | Balances energy, latency, carbon, and thermal vulnerability |

All policies respect the same constraints: power capacity over the one-hour window (including the energy multiplier), the maximum share per site, and the **latency limit**. A site never receives more requests than it can serve within the limit, whichever strategy is used, and demand no site can serve in time remains **unserved** and is reported explicitly.

**Recommendation.** Random is a reference baseline and is never recommended. Among the other strategies, those serving the most requests are compared on energy, carbon, latency and peak ΔT-AR together: each metric's gap to the best candidate (value / best − 1) is averaged with equal weights, and the lowest average wins (ties go to the lower peak ΔT-AR). Peak ΔT-AR alone is not used because Delta-T-AR minimizes it by construction, which would make the result circular. The weights are a stated choice, not an empirical finding. If no strategy can serve any request, no strategy is recommended (on screen, in summaries and in the PDF) and the binding limit is named: latency, power capacity or share per site. Request-origin coordinates are validated (latitude −90 to 90, longitude −180 to 180).

Monte Carlo analysis perturbs weather and request volume by up to 20%, preserving the chosen latency, capacity and share settings. Every run applies the same sampled conditions to all strategies, so results are **paired by run**: Delta-T-AR is compared with each other strategy on peak ΔT-AR using a Wilcoxon signed-rank test with Holm correction, reported with the median difference, a paired effect size (Cohen's d_z) and the share of runs in which Delta-T-AR is lower. Simulated p-values shrink as runs are added, so read the effect sizes first. Comparisons note when a strategy left requests unserved, since serving less also lowers heat. Improvement percentages depend on the selected scenario and are not guaranteed empirical results.

## Other commands

```powershell
# Benchmark current saved models (energy formula loop vs. batch scaling + prediction)
.\.venv-dev\Scripts\python.exe Benchmarking.py --n-samples 10000

# Standalone batch study, with CSV data, PDF figures, and a text report
.\.venv-dev\Scripts\python.exe uhi_focused_analysis.py

# Regression checks
.\.venv-dev\Scripts\python.exe -m pip install pytest
.\.venv-dev\Scripts\python.exe -m pytest tests -q
```

The standalone batch study is a separate four-policy experiment with its own scenario settings; dashboard comparisons use the five policies above.

Main modules: `energy_model.py` (formulas and feature schema), `network_model.py` (distance and latency), `carbon_model.py` (modeled grid intensity), `routing_core.py` (bounded integer allocation), `routing_strategies.py` (policies), and `model_store.py` (local/Neon model persistence).
