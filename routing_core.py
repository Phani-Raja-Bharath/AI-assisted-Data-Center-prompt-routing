"""Capacity-safe integer allocation shared by all routing strategies."""
import math

from network_model import site_utilization, site_throughput_capacity


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


def minimax_allocate(costs, capacities, num_requests):
    """Integer allocation minimizing max(requests_i × cost_i) within each site's capacity.

    Exact for whole requests: a binary search finds the highest common level T at which
    floor(T / cost_i) (capped) still fits the demand, then the few remaining requests go one at a
    time to the site whose resulting load is lowest. Demand beyond total capacity stays unserved.
    """
    import heapq
    sites = [dc for dc in capacities if capacities[dc] > 0 and math.isfinite(costs.get(dc, math.inf))
             and costs[dc] > 0]
    allocation = {dc: 0 for dc in capacities}
    demand = max(0, int(num_requests))
    if not sites or demand == 0:
        return allocation
    total_capacity = sum(capacities[dc] for dc in sites)
    if total_capacity <= demand:
        allocation.update({dc: capacities[dc] for dc in sites})
        return allocation

    def fill(level):
        return {dc: min(capacities[dc], int(math.floor(level / costs[dc] + 1e-12))) for dc in sites}

    low, high = 0.0, max(capacities[dc] * costs[dc] for dc in sites)
    for _ in range(200):  # converges far below one request's cost
        middle = (low + high) / 2
        if sum(fill(middle).values()) <= demand:
            low = middle
        else:
            high = middle
    allocation.update(fill(low))
    remaining = demand - sum(allocation.values())
    heap = [((allocation[dc] + 1) * costs[dc], dc) for dc in sites if allocation[dc] < capacities[dc]]
    heapq.heapify(heap)
    while remaining > 0 and heap:
        _, dc = heapq.heappop(heap)
        allocation[dc] += 1
        remaining -= 1
        if allocation[dc] < capacities[dc]:
            heapq.heappush(heap, ((allocation[dc] + 1) * costs[dc], dc))
    return allocation


def latency_request_capacity(distance_km, num_requests, threshold_ms, latency_fn):
    """Find the largest integer allocation whose modeled latency meets the SLO.

    Server load comes from the site's own request rate over the one-hour batch
    (``site_utilization``), not from its share of the batch."""
    if threshold_ms is None or threshold_ms <= 0:
        return num_requests
    low, high = 0, min(num_requests, site_throughput_capacity())
    while low < high:
        middle = (low + high + 1) // 2
        if latency_fn(distance_km, load_fraction=site_utilization(middle)) <= threshold_ms:
            low = middle
        else:
            high = middle - 1
    return low
