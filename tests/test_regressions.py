from unittest.mock import Mock
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from energy_model import COOLING_TYPES, FEATURE_NAMES, cooling_index
from model_store import (ModelStore, training_config, get_or_train_bundle,
                         validate_bundle, config_key)
from routing_core import allocate_requests
from routing_strategies import (route_random, route_energy_only, route_heat_only,
                                route_ΔT_aware, route_multi_objective, energy_map)
from network_model import haversine_distance, calculate_latency
from Benchmarking import benchmark_inference, _load_bundle


def config(mode="mlr", fingerprint="test"):
    return training_config(mode, fingerprint, FEATURE_NAMES, COOLING_TYPES)


def make_bundle(cfg):
    x = np.random.default_rng(42).normal(size=(30, 6))
    scaler = StandardScaler().fit(x)
    model = LinearRegression().fit(scaler.transform(x), x[:, 0] * 0.01 + 0.4)
    return {"training_config": cfg, "models": {"MLR": model}, "scalers": {"main": scaler},
            "metrics": {"MLR": {"r2": 1.0}}, "best_model_name": "MLR",
            "feature_names": list(FEATURE_NAMES)}


def test_training_reused_across_reruns_and_restarts(tmp_path):
    cfg = config()
    train = Mock(side_effect=lambda: make_bundle(cfg))
    store, memory = ModelStore(tmp_path), {}
    first, source, notices = get_or_train_bundle(cfg, store, memory, train)
    assert source == "new training" and not notices
    second, source, _ = get_or_train_bundle(cfg, store, memory, train)
    assert second is first and source == "session"
    third, source, _ = get_or_train_bundle(cfg, ModelStore(tmp_path), {}, train)
    assert source == "saved bundle"
    np.testing.assert_allclose(first["models"]["MLR"].coef_, third["models"]["MLR"].coef_)
    assert train.call_count == 1
    get_or_train_bundle(cfg, store, memory, train, force=True)
    assert train.call_count == 2


@pytest.mark.parametrize("changed", [config("ann"), config(fingerprint="new physics")])
def test_training_configuration_invalidates_cache(tmp_path, changed):
    store = ModelStore(tmp_path)
    store.save(make_bundle(config()), config())
    assert store.load(changed) is None


def test_corrupt_artifact_retrains_once(tmp_path):
    cfg = config()
    (tmp_path / f"models_{config_key(cfg)}.joblib").write_bytes(b"broken")
    train = Mock(return_value=make_bundle(cfg))
    get_or_train_bundle(cfg, ModelStore(tmp_path), {}, train)
    assert train.call_count == 1
    assert ModelStore(tmp_path).load(cfg) is not None


def test_storage_failure_preserves_session_model(tmp_path):
    cfg, memory = config(), {}
    store = Mock()
    store.load.side_effect = ConnectionError("offline")
    store.save.side_effect = ConnectionError("offline")
    train = Mock(return_value=make_bundle(cfg))
    _, _, notices = get_or_train_bundle(cfg, store, memory, train)
    assert len(notices) == 2
    _, source, _ = get_or_train_bundle(cfg, store, memory, train)
    assert source == "session" and train.call_count == 1


def test_neon_store_roundtrip_and_parameterized_sql(tmp_path):
    cfg, bundle = config(), make_bundle(config())
    conn = Mock()
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=False)
    store = ModelStore(tmp_path / "first", "postgresql://test")
    store._connect = Mock(return_value=conn)
    store.save(bundle, cfg)
    insert = [c for c in conn.execute.call_args_list if "INSERT" in c.args[0]][0]
    assert "%s" in insert.args[0] and insert.args[1][0] == config_key(cfg)
    assert json.loads(insert.args[1][1]) == cfg
    conn.execute.return_value.fetchone.return_value = (insert.args[1][2],)
    restored = ModelStore(tmp_path / "second", "postgresql://test")
    restored._connect = Mock(return_value=conn)
    loaded = restored.load(cfg)
    np.testing.assert_allclose(loaded["models"]["MLR"].coef_, bundle["models"]["MLR"].coef_)


def test_allocator_redistributes_overflow_and_reports_infeasible_demand():
    assert allocate_requests({"a": 100, "b": 1}, {"a": 2, "b": 100}, 10) == {"a": 2, "b": 8}
    assert allocate_requests({"a": 100, "b": 1}, {"a": 2, "b": 3}, 10) == {"a": 2, "b": 3}


DC = {"near": {"lat": 28.5, "lon": -81.3, "default_cooling": "free_air", "region": "florida"},
      "far": {"lat": 59.3, "lon": 18.1, "default_cooling": "evaporative", "region": "sweden"}}
WEATHER = {"near": {"temperature": 35, "humidity": 30, "wind_speed": 2},
           "far": {"temperature": 5, "humidity": 70, "wind_speed": 8}}
COOLING = {name: info["default_cooling"] for name, info in DC.items()}


@pytest.mark.parametrize("policy", [route_random, route_energy_only, route_heat_only, route_ΔT_aware,
                                    route_multi_objective])
def test_all_policies_respect_total_capacity_with_energy_multiplier(policy):
    kwargs: dict = dict(datacenters=DC, weather_data=WEATHER, cooling_selections=COOLING,
                  num_requests=100, max_dc_capacity_mw=0.00001, energy_multiplier=2)
    if policy is route_multi_objective:
        kwargs["user_location"] = (28.5, -81.3)
    dist = policy(**kwargs)
    energies = energy_map(DC, WEATHER, COOLING, None, 2)
    assert sum(dist.values()) < 100
    for dc, count in dist.items():
        assert count >= 0 and count * energies[dc] <= 10


@pytest.mark.parametrize("policy", [route_random, route_energy_only, route_heat_only, route_ΔT_aware,
                                    route_multi_objective])
@pytest.mark.parametrize("share", [0.3, 0.5, 1.0])
def test_all_policies_respect_max_site_share(policy, share):
    kwargs: dict = dict(datacenters=DC, weather_data=WEATHER, cooling_selections=COOLING,
                        num_requests=1000, max_site_share=share)
    if policy is route_multi_objective:
        kwargs.update(user_location=(28.5, -81.3), latency_threshold=10_000)
    dist = policy(**kwargs)
    assert all(count <= 1000 * share for count in dist.values())
    # Two sites can absorb at most 2 × share of the batch; the rest stays unserved.
    assert sum(dist.values()) == min(1000, int(2 * 1000 * share))


def test_max_site_share_must_be_a_fraction():
    with pytest.raises(ValueError):
        route_energy_only(DC, WEATHER, COOLING, 100, max_site_share=0)
    with pytest.raises(ValueError):
        route_energy_only(DC, WEATHER, COOLING, 100, max_site_share=1.5)


def test_multi_objective_enforces_latency_at_allocated_load():
    dist = route_multi_objective(DC, WEATHER, COOLING, (28.5, -81.3), 1000, latency_threshold=50)
    assert dist["far"] == 0 and 0 < dist["near"] < 1000
    assert calculate_latency(0, dist["near"] / 1000) <= 50
    assert calculate_latency(0, (dist["near"] + 1) / 1000) > 50
    assert sum(route_multi_objective(DC, WEATHER, COOLING, (28.5, -81.3), 1000,
                                    latency_threshold=1).values()) == 0


@pytest.mark.parametrize("seed", range(12))
def test_ΔT_aware_minimizes_peak_ΔT_among_fully_served_policies(seed):
    """ΔT-AR must reach the lowest peak ΔT-AR (as the simulation computes it) of all
    strategies that serve every request, and match the equal-warming optimum."""
    from energy_model import calculate_ΔT_contribution
    rng = np.random.default_rng(seed)
    sites = {f"dc{i}": {"lat": float(rng.uniform(-50, 60)), "lon": float(rng.uniform(-150, 150)),
                        "default_cooling": str(rng.choice(COOLING_TYPES)), "region": "default"}
             for i in range(3)}
    weather = {dc: {"temperature": float(rng.uniform(-5, 42)), "humidity": float(rng.uniform(10, 95)),
                    "wind_speed": float(rng.uniform(0, 12))} for dc in sites}
    cooling = {dc: info["default_cooling"] for dc, info in sites.items()}
    requests = 1000
    energies = energy_map(sites, weather, cooling, None, 1.0)
    per_request = {dc: calculate_ΔT_contribution(energies[dc] / 1000, wind_speed=weather[dc]["wind_speed"])
                   for dc in sites}

    def peak(dist):
        return max(count * per_request[dc] for dc, count in dist.items())

    common = dict(datacenters=sites, weather_data=weather, cooling_selections=cooling, num_requests=requests)
    policies = {"Random": route_random(**common), "Energy-Only": route_energy_only(**common),
                "Heat-Only": route_heat_only(**common),
                "Multi-Objective": route_multi_objective(**common, user_location=(0, 0), latency_threshold=10_000)}
    aware = route_ΔT_aware(**common)
    assert sum(aware.values()) == requests
    tolerance = max(per_request.values())  # integer rounding moves at most ~one request per site
    for name, dist in policies.items():
        if sum(dist.values()) == requests:
            assert peak(aware) <= peak(dist) + tolerance, name
    optimum = requests / sum(1 / value for value in per_request.values())
    assert peak(aware) <= optimum + tolerance


def test_cooling_encoding_used_by_prediction():
    model = Mock(is_trained=True)
    model.predict.return_value = 0.3
    energy_map(DC, WEATHER, COOLING, model, 1)
    assert [c.args[0][-1] for c in model.predict.call_args_list] == [cooling_index(COOLING[dc]) for dc in DC]


def test_benchmark_matches_current_schema_and_does_not_touch_global_rng(tmp_path):
    bundle = make_bundle(config())
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    loaded = _load_bundle(path)
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    result = benchmark_inference(loaded, n_samples=40)
    assert np.random.random() == expected
    assert result["n_samples"] == 40 and result["mae_against_formula_wh"] >= 0
    with pytest.raises(ValueError, match="positive"):
        benchmark_inference(loaded, n_samples=0)
    loaded.pop("training_config")
    joblib.dump(loaded, path)
    with pytest.raises(ValueError, match="Legacy"):
        _load_bundle(path)
