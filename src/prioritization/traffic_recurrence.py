"""Traffic and recurrence proxies (Phase 4 Task 9).

TrafficProxy: no live traffic API is wired up (blueprint Section 5.3 explicitly allows
falling back to "road-type + time-of-day heuristics" when one isn't available), so this
combines the road-type weight with a simple rush-hour time-of-day multiplier.

RecurrenceFactor: counts how many independent reports fall within a GPS radius of each
other (blueprint: "number of independent citizen reports at same GPS cluster within X
meters") using real haversine great-circle distance, then saturates the count to [0, 1].
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0

# Multiplier applied to the road-type baseline by hour-of-day (0-23), modeling typical
# weekday commute patterns: morning (7-10) and evening (17-20) rush hours score highest,
# late night (0-5) lowest. A heuristic, not measured data -- see module docstring.
HOURLY_TRAFFIC_MULTIPLIER = {
    **{h: 0.2 for h in range(0, 6)},
    **{h: 0.6 for h in (6,)},
    **{h: 1.0 for h in (7, 8, 9)},
    **{h: 0.7 for h in (10, 11, 12, 13, 14, 15)},
    **{h: 0.85 for h in (16,)},
    **{h: 1.0 for h in (17, 18, 19)},
    **{h: 0.6 for h in (20,)},
    **{h: 0.4 for h in (21, 22, 23)},
}


def traffic_proxy(road_type_weight: float, hour_of_day: int) -> float:
    """Returns a value in [0, 1]. road_type_weight is expected in [0, 1] (from
    road_type.road_type_weight); hour_of_day in [0, 23]."""
    if not 0 <= hour_of_day <= 23:
        raise ValueError("hour_of_day must be in [0, 23]")
    multiplier = HOURLY_TRAFFIC_MULTIPLIER[hour_of_day]
    return float(max(0.0, min(1.0, road_type_weight * multiplier)))


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return float(2 * EARTH_RADIUS_M * math.asin(math.sqrt(a)))


def cluster_recurrence_counts(reports: list[tuple[float, float]], radius_m: float = 25.0) -> list[int]:
    """For each report (lat, lon), counts how many reports (including itself) fall
    within radius_m -- an O(n^2) pairwise scan, which is fine at citizen-report volumes
    per region; swap for a spatial index (e.g. a KD-tree or PostGIS ST_DWithin query,
    per the blueprint's PostGIS backend) if a region's report volume grows large."""
    counts = []
    for i, (lat_i, lon_i) in enumerate(reports):
        count = sum(
            1
            for lat_j, lon_j in reports
            if haversine_distance_m(lat_i, lon_i, lat_j, lon_j) <= radius_m
        )
        counts.append(count)
    return counts


def recurrence_factor(count: int, saturation_count: int = 5) -> float:
    """Returns a value in [0, 1]: 0 at count<=1 (a single, non-recurring report), 1.0 at
    count>=saturation_count. saturation_count=5 is a placeholder default -- like the
    fusion normalization bounds in severity/fusion.py, real calibration against actual
    citizen-report volume needs live usage data this project doesn't have yet."""
    if saturation_count <= 1:
        raise ValueError("saturation_count must be > 1")
    return float(max(0.0, min(1.0, (count - 1) / (saturation_count - 1))))
