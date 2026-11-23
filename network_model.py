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

# A simulated batch is one hour of traffic. Energy per batch (Wh) is then average power over
# that hour (W), which is what the ΔT-AR heat-flux formula and the MW capacity limit expect,
# and server load is the batch's arrival rate (requests / 3600 s) against the service rate.
BATCH_SECONDS = 3600
SERVICE_RATE = 1000  # requests per second one site can process (M/M/1 service rate μ)
MAX_UTILIZATION = 0.99  # sites are never loaded beyond this fraction of their service rate


def site_throughput_capacity(service_rate=SERVICE_RATE, batch_seconds=BATCH_SECONDS,
                             max_utilization=MAX_UTILIZATION):
    """Most requests one site can process in the one-hour batch. Enforced as a hard capacity,
    separately from the latency limit: the queue-delay formula is capped for display and would
    otherwise make an overloaded site look feasible."""
    return int(max_utilization * service_rate * batch_seconds)


def site_utilization(requests, service_rate=SERVICE_RATE, batch_seconds=BATCH_SECONDS):
    """Server utilization ρ = λ / μ for ``requests`` arriving over one simulated hour."""
    return (float(requests) / batch_seconds) / service_rate


def calculate_latency(distance_km, load_fraction=0.5, service_rate=SERVICE_RATE, processing_ms=30,
                      fiber_speed_km_s=200000, max_queue_ms=200, utilization_cap=0.99):
    """
    Latency proxy = propagation + M/M/1 queueing + processing.

    - Propagation: round-trip distance / fiber speed
    - Queueing: M/M/1 with utilization ρ = λ/μ, bounded for stability. ``load_fraction`` is that
      utilization; callers derive it from the site's request rate with ``site_utilization``, not
      from the site's share of the batch.
    - Processing: constant baseline
    """
    if float(load_fraction) >= 1.0:
        return float("inf")  # demand at or above the service rate: no steady-state latency
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
