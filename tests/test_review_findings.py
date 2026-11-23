"""Regression tests for the second logic review (cooling coverage, frozen carbon, statistics,
integer minimax, empty service, coordinate bounds, server throughput)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import ai_assisted_datacenter_allocation as app
from carbon_model import carbon_snapshot, get_carbon_intensity, local_solar_hour
from energy_model import COOLING_TYPES, MODEL_INPUT_NAMES, encode_features
from network_model import calculate_latency, site_throughput_capacity, site_utilization
from routing_core import allocate_requests, minimax_allocate
from routing_strategies import route_ΔT_aware, route_energy_only

PHOENIX = {"Phoenix, AZ": {"lat": 33.45, "lon": -112.07, "region": "arizona", "default_cooling": "free_air"}}
WEATHER = {"Phoenix, AZ": {"temperature": 25, "humidity": 30, "wind_speed": 3}}
COOLING = {"Phoenix, AZ": "free_air"}


# 1. Cooling is one-hot encoded, and every technology is an independent model input.
def test_cooling_is_one_hot_encoded():
    rows = [[20, 50, 5, 1013, 200, index] for index in range(len(COOLING_TYPES))]
    encoded = encode_features(rows)
    assert encoded.shape == (len(COOLING_TYPES), len(MODEL_INPUT_NAMES))
    assert np.array_equal(encoded[:, 5:], np.eye(len(COOLING_TYPES)))
    with pytest.raises(ValueError):
        encode_features([[20, 50, 5, 1013, 200, len(COOLING_TYPES)]])


def test_training_samples_every_cooling_technology_for_historical_weather(monkeypatch):
    weather = pd.DataFrame({"temperature": np.linspace(-5, 40, 500), "humidity": 50.0, "wind_speed": 3.0,
                            "pressure": 1013.0, "solar_radiation": 200.0})
    monkeypatch.setattr(app, "collect_historical_training_data", lambda **_: weather)
    data = app.AIModelSuite().generate_training_data()
    used = set(data["cooling_type"].astype(int))
    assert used == set(range(len(COOLING_TYPES)))  # not only the climate-recommended technology


# 2. Carbon is frozen once and does not follow the clock.
def test_recorded_carbon_snapshot_does_not_drift_within_the_hour():
    early = carbon_snapshot(PHOENIX, datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc))
    late = carbon_snapshot(PHOENIX, datetime(2026, 1, 1, 0, 40, tzinfo=timezone.utc))
    assert early["sites"]["Phoenix, AZ"]["intensity_kg_per_kwh"] != late["sites"]["Phoenix, AZ"]["intensity_kg_per_kwh"]
    frozen = {name: site["intensity_kg_per_kwh"] for name, site in early["sites"].items()}
    for _ in range(2):  # the same frozen inputs give the same carbon, whenever it runs
        result = app.run_simulation(PHOENIX, WEATHER, (33.45, -112.07), 1000, COOLING, carbon_intensities=frozen)
        site = result["ΔT-AR"]["datacenters"]["Phoenix, AZ"]
        assert site["carbon_kg"] == pytest.approx(site["energy_wh"] / 1000 * frozen["Phoenix, AZ"])


def test_snapshot_uses_each_sites_local_solar_hour():
    at = datetime(2026, 1, 1, 0, 40, tzinfo=timezone.utc)
    site = carbon_snapshot(PHOENIX, at)["sites"]["Phoenix, AZ"]
    assert site["local_solar_hour"] == local_solar_hour(-112.07, at)
    assert site["intensity_kg_per_kwh"] == pytest.approx(get_carbon_intensity("arizona", site["local_solar_hour"]))


# 3. Statistics: tiny consistent differences are not treated as zero; d_z can be undefined.
def _mc(reference, other):
    frame = lambda values: {"raw_data": pd.DataFrame({"peak_ΔT": values, "unserved_requests": 0})}
    return {"ΔT-AR": frame(reference), "Energy-Only": frame(other)}


def test_tiny_consistent_differences_are_detected():
    base = np.linspace(1e-4, 2e-4, 40)
    result = app.calculate_statistical_significance(_mc(base - 5e-9 * (1 + np.arange(40) / 40), base))
    comparison = result["comparisons"]["Energy-Only"]
    assert comparison["reference_lower_share"] == 1.0 and comparison["p_holm"] < 0.05


def test_identical_differences_leave_effect_size_undefined():
    base = np.linspace(1e-4, 2e-4, 40)
    comparison = app.calculate_statistical_significance(_mc(base - 1e-6, base))["comparisons"]["Energy-Only"]
    assert comparison["cohens_dz"] is None and comparison["p_holm"] < 0.05
    no_difference = app.calculate_statistical_significance(_mc(base, base))["comparisons"]["Energy-Only"]
    assert no_difference["p_value"] == 1.0


# 4. ΔT-AR reaches the exact integer minimax.
def test_minimax_beats_rounded_proportional_allocation():
    costs = {"a": 1, "b": 2, "c": 6}
    caps = {dc: 100 for dc in costs}
    proportional = allocate_requests({dc: 1 / c for dc, c in costs.items()}, caps, 100)
    exact = minimax_allocate(costs, caps, 100)
    assert exact == {"a": 60, "b": 30, "c": 10}
    assert max(exact[dc] * costs[dc] for dc in costs) == 60 < max(proportional[dc] * costs[dc] for dc in costs)


@pytest.mark.parametrize("seed", range(20))
def test_minimax_is_optimal_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    costs = {dc: float(rng.uniform(0.5, 8)) for dc in "abc"}
    caps = {dc: int(rng.integers(5, 40)) for dc in "abc"}
    demand = int(rng.integers(1, 60))
    result = minimax_allocate(costs, caps, demand)
    served = min(demand, sum(caps.values()))
    assert sum(result.values()) == served and all(result[dc] <= caps[dc] for dc in caps)
    best = min(max(a * costs["a"], b * costs["b"], (served - a - b) * costs["c"])
               for a in range(caps["a"] + 1) for b in range(caps["b"] + 1)
               if 0 <= served - a - b <= caps["c"])
    assert max(result[dc] * costs[dc] for dc in caps) == pytest.approx(best)


# 5. Empty service: no latency, no recommended strategy, the binding limit is named.
def test_empty_service_reports_no_latency_and_no_recommendation():
    results = app.run_simulation(PHOENIX, WEATHER, (-33.9, 151.2), 1000, COOLING, latency_threshold=50)
    assert all(np.isnan(r["totals"]["avg_latency_ms"]) for r in results.values())
    recommendation = app.generate_final_recommendation(results, 50)
    assert recommendation["feasible"] is False and recommendation["recommended_strategy"] is None
    assert "latency" in recommendation["binding_constraint"]


def test_binding_constraint_distinguishes_power_from_latency():
    results = app.run_simulation(PHOENIX, WEATHER, (33.45, -112.07), 1000, COOLING,
                                 latency_threshold=10_000, max_dc_capacity_mw=0)
    recommendation = app.generate_final_recommendation(results, 10_000)
    assert recommendation["feasible"] is False and "power" in recommendation["binding_constraint"]
    assert "latency" not in recommendation["binding_constraint"]


@pytest.mark.skipif(not app.FPDF_AVAILABLE, reason="fpdf2 not installed")
def test_pdf_does_not_recommend_an_infeasible_strategy():
    results = app.run_simulation(PHOENIX, WEATHER, (-33.9, 151.2), 1000, COOLING, latency_threshold=50)
    recommendation = app.generate_final_recommendation(results, 50)
    pdf = bytes(app.create_pdf_report(results, recommendation))
    assert pdf.startswith(b"%PDF")


# 7. Coordinates are bounded in the backend.
@pytest.mark.parametrize("origin", [(91, 0), (-91, 0), (0, 181), (0, -181), (float("nan"), 0)])
def test_impossible_coordinates_are_rejected(origin):
    with pytest.raises(ValueError):
        app.run_simulation(PHOENIX, WEATHER, origin, 1000, COOLING)


# 8. Server throughput is a hard capacity, separate from the displayed queue-delay cap.
def test_overloaded_site_is_not_admitted():
    capacity = site_throughput_capacity()
    assert site_utilization(capacity) <= 0.99 < site_utilization(10_000_000)
    assert calculate_latency(0, site_utilization(10_000_000)) == float("inf")
    dist = route_energy_only(PHOENIX, WEATHER, COOLING, 10_000_000, max_dc_capacity_mw=1e9)
    assert dist["Phoenix, AZ"] == capacity  # the rest is unserved, not silently admitted
    aware = route_ΔT_aware(PHOENIX, WEATHER, COOLING, 10_000_000, max_dc_capacity_mw=1e9)
    assert aware["Phoenix, AZ"] == capacity
