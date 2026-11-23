"""Capacity-safe integer allocation shared by all routing strategies."""
import math


def allocate_requests(weights, capacities, num_requests):
    """Distribute as much demand as feasible, never exceeding a site's cap."""
    allocation = {dc: 0 for dc in capacities}
    remaining = max(0, int(num_requests))
    while remaining:
        active = {dc: float(weights.get(dc, 0)) for dc, cap in capacities.items()
                  if cap > allocation[dc] and math.isfinite(weights.get(dc, 0))
                  and weights.get(dc, 0) > 0}
        if not active:
            break
        total = sum(active.values())
        shares = {dc: remaining * weight / total for dc, weight in active.items()}
        assigned = 0
        for dc, share in shares.items():
            count = min(capacities[dc] - allocation[dc], int(share))
            allocation[dc] += count
            assigned += count
        remaining -= assigned
        if not assigned:
            # Largest remainders resolve integer rounding without favoring input order.
            for dc in sorted(active, key=lambda dc: shares[dc], reverse=True):
                if remaining == 0:
                    break
                allocation[dc] += 1
                remaining -= 1
    return allocation


def latency_request_capacity(distance_km, num_requests, threshold_ms, latency_fn):
    """Find the largest integer allocation whose modeled latency meets the SLO."""
    if threshold_ms is None or threshold_ms <= 0:
        return num_requests
    low, high = 0, num_requests
    while low < high:
        middle = (low + high + 1) // 2
        if latency_fn(distance_km, load_fraction=middle / num_requests) <= threshold_ms:
            low = middle
        else:
            high = middle - 1
    return low
