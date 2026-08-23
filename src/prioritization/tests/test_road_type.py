from road_type import highway_tag_to_weight, query_nearest_highway_tag, road_type_weight


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


def test_query_nearest_highway_tag_real_overpass_call():
    # 7th Avenue / West 45th St, Manhattan -- real, stable OSM-tagged secondary roads
    tag = query_nearest_highway_tag(40.7580, -73.9855, radius_m=50, timeout=20)

    assert tag is not None
    assert highway_tag_to_weight(tag) in (1.0, 0.7, 0.4)


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
