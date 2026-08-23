"""Road-type weight lookup from OpenStreetMap (Phase 4 Task 8): query the OSM Overpass
API for the nearest tagged road at a GPS coordinate and map its `highway` tag to a
weight (blueprint Section 5.3: highway=primary/secondary/residential -> 1.0/0.7/0.4).
"""

from __future__ import annotations

from functools import lru_cache

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass API rejects requests with the default python-requests User-Agent (406 Not
# Acceptable) -- their fair-use policy expects a real, identifying client string.
REQUEST_HEADERS = {"User-Agent": "PRISM-pothole-severity-research/0.1 (NMIMS capstone project)"}

# blueprint gives primary/secondary/residential -> 1.0/0.7/0.4 explicitly; the rest are
# grouped into the same three tiers by how OSM's highway classification hierarchy
# (https://wiki.openstreetmap.org/wiki/Key:highway) maps onto road importance/traffic.
HIGHWAY_TAG_WEIGHTS = {
    "motorway": 1.0,
    "motorway_link": 1.0,
    "trunk": 1.0,
    "trunk_link": 1.0,
    "primary": 1.0,
    "primary_link": 1.0,
    "secondary": 0.7,
    "secondary_link": 0.7,
    "tertiary": 0.7,
    "tertiary_link": 0.7,
    "residential": 0.4,
    "unclassified": 0.4,
    "living_street": 0.4,
    "service": 0.4,
}
DEFAULT_WEIGHT = 0.4  # unmapped/unknown highway tags (footway, cycleway, no tag found, API unreachable, ...)


def highway_tag_to_weight(highway_tag: str | None) -> float:
    if highway_tag is None:
        return DEFAULT_WEIGHT
    return HIGHWAY_TAG_WEIGHTS.get(highway_tag, DEFAULT_WEIGHT)


def query_nearest_highway_tag(lat: float, lon: float, radius_m: int = 50, timeout: float = 15.0) -> str | None:
    """Real Overpass API call -- returns the `highway` tag of the nearest tagged road
    within radius_m of (lat, lon), or None if none found or the API is unreachable."""
    query = f"[out:json][timeout:{int(timeout)}];way(around:{radius_m},{lat},{lon})[highway];out tags 10;"
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout, headers=REQUEST_HEADERS)
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except (requests.RequestException, ValueError):
        return None

    for element in elements:
        tag = element.get("tags", {}).get("highway")
        if tag in HIGHWAY_TAG_WEIGHTS:
            return tag
    # no recognized highway tag in range -- return whatever the first result was (if
    # any), so callers can still see "we found something, just an unmapped type"
    if elements:
        return elements[0].get("tags", {}).get("highway")
    return None


@lru_cache(maxsize=2048)
def road_type_weight(lat: float, lon: float, radius_m: int = 50) -> float:
    """Cached (repeated reports at/near the same intersection shouldn't re-query OSM
    every time). Falls back to DEFAULT_WEIGHT when nothing is found or the API call
    fails -- never raises, since this feeds an interactive dashboard, not a batch job
    where a hard failure would be preferable to a silent default."""
    tag = query_nearest_highway_tag(lat, lon, radius_m)
    return highway_tag_to_weight(tag)
