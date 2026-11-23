# Climate-aware AI datacenter routing

A Streamlit research dashboard comparing AI request routing by energy, latency, carbon emissions, and modeled local heat impact. The default sites are Phoenix, San Francisco, and Stockholm, with additional locations available.

The interface uses a persistent experiment sidebar and four workspaces: **Overview** (KPI cards, routing map, traffic split), **Routing** (strategy comparison and cooling controls), **Models** (performance, diagnostics, explicit retraining), and **Experiments** (Monte Carlo, sensitivity, and alternative scenarios). Only the selected workspace is rendered. Monte Carlo and sensitivity computations run on request; navigation and language changes reuse compatible models and simulation results. Sources and assumptions are available from the header, and results can be exported from a collapsed panel. Plotly's toolbar exports figures without generating image files on every app rerun.

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

Use **Retrain selected models** to deliberately replace a bundle. Changing the selected training mode, model/label code, feature schema, cooling encoding, Bayesian availability, or scikit-learn version invalidates the relevant cache. Weather, request volume, cooling selections, location, capacity, and latency changes rerun the simulation without retraining. Monte Carlo results are also reused until their simulation inputs or run count change.

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

Energy starts at 0.3 Wh per request, adjusted by cooling PUE, temperature, and humidity. Surrogate training labels additionally include wind, pressure, solar radiation, and small synthetic noise. The six model inputs are temperature, humidity, wind speed, pressure, solar radiation, and a stable cooling category index.

Training attempts hourly Open-Meteo weather for 2021-2024 and falls back to synthetic weather. **Energy labels always come from formulas, not measured datacenter consumption.** Models include multiple linear regression, a neural network, and an optional Bayesian-optimized neural network. Data is split approximately 70/15/15; model selection uses validation R-squared and test metrics are reported separately.

All consumed energy becomes waste heat. The temperature-rise proxy scales with heat per assumed influence area and decreases with wind. This is a comparative proxy, not a calibrated urban climate prediction. Carbon intensity uses regional constants (or the fetched Argentina annual average for Neuquén) and modeled hourly variation; it is not live grid telemetry. Latency combines geographic propagation, a bounded M/M/1 queue proxy, and fixed processing delay. Utilization is approximated by a site's share of offered requests.

## Five dashboard strategies

| Strategy | Behavior |
|---|---|
| Random | Dirichlet-weighted random baseline |
| Energy-Only | Favors lower energy per request |
| Heat-Only | Favors lower wind-adjusted temperature-rise contribution per request |
| Delta-T-Aware (Delta-T-AR) | Penalizes energy use and thermal vulnerability |
| Multi-Objective | Balances energy, latency, carbon, and thermal vulnerability |

All policies respect power capacity over an assumed one-hour window, including the energy multiplier. Multi-Objective also limits each site's allocation so its modeled latency meets the selected threshold. Infeasible demand remains **unserved** and is reported explicitly. Recommendations prefer policies serving the full workload; if none do, they prefer the highest service count before comparing heat. Compare service rates alongside energy and heat, especially in deliberately constrained scenarios.

Monte Carlo analysis perturbs weather and request volume by up to 20%, preserving the chosen latency and capacity settings. Distribution plots, mean confidence intervals, Welch's t-test, and Cohen's d describe simulation outcomes. Improvement percentages depend on the selected scenario and are not guaranteed empirical results.

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
