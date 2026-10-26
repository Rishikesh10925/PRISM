import time
from functools import lru_cache

import pytest
import requests

import road_type
from road_type import (
    highway_tag_to_weight,
    query_nearest_highway_tag,
    road_type_weight,
    road_type_weight_with_source,
)


@lru_cache(maxsize=1)
def _overpass_reachable() -> bool:
    try:
        road_type.fetch_highway_elements(40.7580, -73.9855, radius_m=50, timeout=10)
    except (requests.RequestException, ValueError):
        return False
    return True


# live-API tests: run when the public Overpass instance answers, skip (not fail) when
# the machine is offline or Overpass is down/rate-limiting -- deselect entirely with
# `-m "not network"`.
def requires_overpass(test_fn):
    skip = pytest.mark.skipif("not _overpass_reachable()", reason="Overpass API unreachable")
    return pytest.mark.network(skip(test_fn))


def _query_with_retries(lat, lon, radius_m=50, timeout=20, attempts=3, backoff_s=5):
    """Overpass's public instance is fair-use rate-limited and occasionally returns an
    empty result under back-to-back test-suite load rather than a clear error -- retry
    with backoff instead of treating a transient empty response as this module's bug."""
    tag = None
    for attempt in range(attempts):
        tag = query_nearest_highway_tag(lat, lon, radius_m=radius_m, timeout=timeout)
        if tag is not None:
            return tag
        if attempt < attempts - 1:
            time.sleep(backoff_s)
    return tag


def test_highway_tag_to_weight_known_tags():
    assert highway_tag_to_weight("primary") == 1.0
    assert highway_tag_to_weight("trunk_link") == 1.0
    assert highway_tag_to_weight("secondary") == 0.7
    assert highway_tag_to_weight("tertiary") == 0.7
    assert highway_tag_to_weight("residential") == 0.4
    assert highway_tag_to_weight("living_street") == 0.4


def test_highway_tag_to_weight_unmapped_and_none_fall_back_to_default():
    assert highway_tag_to_weight("footway") == 0.4
    assert highway_tag_to_weight(None) == 0.4


@requires_overpass
def test_query_nearest_highway_tag_real_overpass_call():
    # 7th Avenue / West 45th St, Manhattan -- real, stable OSM-tagged secondary roads
    tag = _query_with_retries(40.7580, -73.9855)

    assert tag is not None
    assert highway_tag_to_weight(tag) in (1.0, 0.7, 0.4)


@requires_overpass
def test_road_type_weight_real_lookup_is_cached():
    w1 = road_type_weight(40.7580, -73.9855, radius_m=50)
    w2 = road_type_weight(40.7580, -73.9855, radius_m=50)  # should hit lru_cache, not re-query

    assert w1 == w2
    assert w1 in (1.0, 0.7, 0.4)


def test_query_nearest_highway_tag_handles_unreachable_host_gracefully():
    # a URL that will fail fast (invalid port) -- proves the function returns None
    # rather than raising when the network call fails
    import road_type

    original = road_type.OVERPASS_URL
    road_type.OVERPASS_URL = "http://127.0.0.1:1/interpreter"
    try:
        assert query_nearest_highway_tag(0.0, 0.0, timeout=2.0) is None
    finally:
        road_type.OVERPASS_URL = original


@pytest.fixture
def fake_overpass(monkeypatch):
    """Replaces the raw Overpass call; set .result to a list of elements or an exception."""
    road_type._source_cache.clear()

    class Fake:
        result: object = []

        def __call__(self, lat, lon, radius_m=50, timeout=15.0):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    fake = Fake()
    monkeypatch.setattr(road_type, "fetch_highway_elements", fake)
    yield fake
    road_type._source_cache.clear()


def test_road_type_weight_with_source_osm(fake_overpass):
    fake_overpass.result = [{"tags": {"highway": "secondary"}}]
    assert road_type_weight_with_source(1.0, 2.0) == (0.7, road_type.SOURCE_OSM)


def test_road_type_weight_with_source_no_road(fake_overpass):
    fake_overpass.result = []
    assert road_type_weight_with_source(1.0, 2.0) == (0.4, road_type.SOURCE_DEFAULT_NO_ROAD)


def test_road_type_weight_with_source_api_error_is_not_cached(fake_overpass):
    fake_overpass.result = requests.ConnectionError("down")
    assert road_type_weight_with_source(1.0, 2.0) == (0.4, road_type.SOURCE_DEFAULT_API_ERROR)

    fake_overpass.result = [{"tags": {"highway": "primary"}}]  # API recovers
    assert road_type_weight_with_source(1.0, 2.0) == (1.0, road_type.SOURCE_OSM)
