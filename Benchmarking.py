from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import joblib


# Keep PUE values consistent with your main app (Alkrush et al., 2024)
COOLING_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "mechanical_chiller": {"pue": 1.80},
    "evaporative": {"pue": 1.35},
    "air_economizer": {"pue": 1.25},
    "liquid_cooling": {"pue": 1.15},
    "free_air": {"pue": 1.10},
}


def _load_bundle(bundle_path: Path) -> Dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at: {bundle_path}. "
            f"Run the Streamlit app once to train + save models."
        )
    bundle = joblib.load(bundle_path)

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
    """Return timing dict similar to AIModelSuite.benchmark_inference()."""

    np.random.seed(seed)

    # Generate test samples (same ranges as your Streamlit benchmark)
    test_temps = np.random.uniform(5, 40, n_samples)
    test_humidity = np.random.uniform(30, 90, n_samples)
    test_wind = np.random.uniform(0, 15, n_samples)
    test_cooling = np.random.randint(0, len(COOLING_SYSTEMS), n_samples)

    X_test = np.column_stack([test_temps, test_humidity, test_wind, test_cooling])

    scaler_main = bundle["scalers"]["main"]
    X_test_scaled = scaler_main.transform(X_test)

    # 1) Baseline physics formulas (loop, like your app)
    t0 = time.perf_counter()
    cooling_types = list(COOLING_SYSTEMS.keys())
    Ebase = 0.3  # Wh
    alpha = 0.015
    alpha_temp = 0.0012
    beta = 0.15

    baseline_results = []
    for i in range(n_samples):
        pue = COOLING_SYSTEMS[cooling_types[test_cooling[i]]]["pue"]
        T = test_temps[i]
        E = Ebase * pue * (1 + alpha * max(0.0, T - 20.0))

        Q = E / 1000.0  # kWh
        A = 1.0
        W = test_wind[i]
        delta_T = alpha_temp * (Q / A) * (1.0 / (1.0 + beta * W))

        baseline_results.append((E, delta_T))

    baseline_time = time.perf_counter() - t0

    # 2) Surrogate prediction
    best_name = bundle["best_model_name"]
    model = bundle["models"][best_name]

    # small warm-up (helps stabilize first-call overhead)
    _ = model.predict(X_test_scaled[:32])

    t1 = time.perf_counter()
    _ = model.predict(X_test_scaled)
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
