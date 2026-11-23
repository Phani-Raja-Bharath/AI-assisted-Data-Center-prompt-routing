"""Distance and latency proxies used by routing and reporting."""
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_latency(distance_km, load_fraction=0.5, service_rate=1000, processing_ms=30,
                      fiber_speed_km_s=200000, max_queue_ms=200, utilization_cap=0.99):
    """
    Latency proxy = propagation + M/M/1 queueing + processing.

    - Propagation: round-trip distance / fiber speed
    - Queueing: M/M/1 with utilization ρ = λ/μ, bounded for stability
    - Processing: constant baseline
    """
    # --- Propagation delay (round trip) ---
    propagation_ms = (float(distance_km) / float(fiber_speed_km_s)) * 1000.0 * 2.0

    # --- Queueing delay (M/M/1) ---
    # Interpret load_fraction as utilization proxy ρ, then map to λ = ρ * μ
    rho = max(0.0, min(float(load_fraction), utilization_cap))  # 0 <= ρ < 1
    mu = float(service_rate)                                   # requests/s
    lam = rho * mu                                             # requests/s

    # If extremely close to saturation, cap delay at max_queue_ms
    if lam >= utilization_cap * mu:
        queueing_ms = float(max_queue_ms)
    else:
        # M/M/1 mean waiting time in queue: Wq = λ / (μ(μ-λ)) seconds
        # Convert to ms, then cap
        wq_s = lam / (mu * max(1e-9, (mu - lam)))
        queueing_ms = min(float(max_queue_ms), wq_s * 1000.0)

    # --- Processing delay ---
    return propagation_ms + queueing_ms + float(processing_ms)
