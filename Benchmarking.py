from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import joblib
from energy_model import COOLING_TYPES, FEATURE_NAMES, surrogate_energy_target
from model_store import SCHEMA_VERSION, validate_bundle


def _load_bundle(bundle_path: Path) -> Dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at: {bundle_path}. "
            f"Run the Streamlit app once to train + save models."
        )
    bundle = joblib.load(bundle_path)
    config = bundle.get("training_config", {})
    if (config.get("schema_version") != SCHEMA_VERSION
            or config.get("features") != list(FEATURE_NAMES)
            or config.get("cooling_types") != list(COOLING_TYPES)):
        raise ValueError("Legacy/incompatible bundle. Run the dashboard once to train the current schema.")
    validate_bundle(bundle, config)

    # Basic sanity checks
    for key in ("models", "scalers", "best_model_name"):
        if key not in bundle:
            raise KeyError(f"Bundle missing required key: '{key}'")

    if "main" not in bundle["scalers"]:
        raise KeyError("Bundle scalers missing 'main' StandardScaler.")

    best_name = bundle["best_model_name"]
    if best_name not in bundle["models"]:
        raise KeyError(
            f"best_model_name='{best_name}' not found in bundle['models'] keys: "
            f"{list(bundle['models'].keys())}"
        )

    return bundle


def benchmark_inference(bundle: Dict[str, Any], n_samples: int = 10_000, seed: int = 42) -> Dict[str, Any]:
    """Compare formula energy evaluation with end-to-end batch model inference.

    The baseline is a Python loop; this is not a vectorized physics benchmark.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)

    # Generate test samples (same ranges as your Streamlit benchmark)
    test_temps = rng.uniform(5, 40, n_samples)
    test_humidity = rng.uniform(30, 90, n_samples)
    test_wind = rng.uniform(0, 15, n_samples)
    test_pressure = rng.uniform(980, 1040, n_samples)
    test_solar = rng.uniform(0, 1000, n_samples)
    test_cooling = rng.integers(0, len(COOLING_TYPES), n_samples)

    X_test = np.column_stack([test_temps, test_humidity, test_wind,
                              test_pressure, test_solar, test_cooling])

    scaler_main = bundle["scalers"]["main"]

    # 1) Baseline physics formulas (loop, like your app)
    t0 = time.perf_counter()
    baseline_results = []
    for i in range(n_samples):
        baseline_results.append(surrogate_energy_target(
            test_temps[i], test_humidity[i], test_wind[i], test_pressure[i],
            test_solar[i], COOLING_TYPES[test_cooling[i]]))

    baseline_time = time.perf_counter() - t0

    # 2) Surrogate prediction
    best_name = bundle["best_model_name"]
    model = bundle["models"][best_name]

    # small warm-up (helps stabilize first-call overhead)
    _ = model.predict(scaler_main.transform(X_test[:32]))

    t1 = time.perf_counter()
    predictions = model.predict(scaler_main.transform(X_test))
    surrogate_time = time.perf_counter() - t1

    # Derived stats
    baseline_per_sample_ms = (baseline_time / n_samples) * 1000.0
    surrogate_per_sample_ms = (surrogate_time / n_samples) * 1000.0

    baseline_throughput = n_samples / baseline_time if baseline_time > 0 else float("inf")
    surrogate_throughput = n_samples / surrogate_time if surrogate_time > 0 else float("inf")

    speedup = (baseline_time / surrogate_time) if surrogate_time > 0 else float("inf")

    return {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "baseline_total_sec": float(baseline_time),
        "baseline_per_sample_ms": float(baseline_per_sample_ms),
        "baseline_throughput_per_sec": float(baseline_throughput),
        "surrogate_total_sec": float(surrogate_time),
        "surrogate_per_sample_ms": float(surrogate_per_sample_ms),
        "surrogate_throughput_per_sec": float(surrogate_throughput),
        "speedup_factor": float(speedup),
        "model_used": str(best_name),
        "mae_against_formula_wh": float(np.mean(np.abs(predictions - baseline_results))),
        "baseline_method": "Python loop: energy target only",
        "surrogate_method": "Batch scaling and energy prediction",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle",
        type=str,
        default=str((Path(__file__).parent / "artifacts" / "ai_models_bundle.joblib").resolve()),
        help="Path to ai_models_bundle.joblib saved by the Streamlit app",
    )
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write JSON results",
    )
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    bundle = _load_bundle(bundle_path)

    results = benchmark_inference(bundle, n_samples=args.n_samples, seed=args.seed)

    print("\n=== Inference benchmark (standalone) ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
