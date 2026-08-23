import pytest

from traffic_recurrence import (
    cluster_recurrence_counts,
    haversine_distance_m,
    recurrence_factor,
    traffic_proxy,
)


def test_traffic_proxy_rush_hour_higher_than_late_night():
    rush = traffic_proxy(road_type_weight=1.0, hour_of_day=8)
    late_night = traffic_proxy(road_type_weight=1.0, hour_of_day=3)
    assert rush > late_night


def test_traffic_proxy_scales_with_road_type_weight():
    primary = traffic_proxy(road_type_weight=1.0, hour_of_day=8)
    residential = traffic_proxy(road_type_weight=0.4, hour_of_day=8)
    assert primary > residential
    assert primary == pytest.approx(1.0)  # 1.0 road weight * 1.0 rush-hour multiplier


def test_traffic_proxy_rejects_invalid_hour():
    with pytest.raises(ValueError):
        traffic_proxy(0.5, 24)
    with pytest.raises(ValueError):
        traffic_proxy(0.5, -1)


def test_haversine_distance_known_landmarks():
    # NYC City Hall to the Brooklyn Bridge's Manhattan end: roughly 500-900m apart
    d = haversine_distance_m(40.7128, -74.0060, 40.7061, -73.9969)
    assert 500 <= d <= 1500


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance_m(12.34, 56.78, 12.34, 56.78) == pytest.approx(0.0, abs=1e-6)


def test_cluster_recurrence_counts_groups_nearby_reports():
    reports = [
        (17.4239, 78.4738),  # cluster A, report 1
        (17.42391, 78.47381),  # cluster A, report 2 (a few metres away)
        (17.5000, 78.5500),  # isolated, far away
    ]

    counts = cluster_recurrence_counts(reports, radius_m=25.0)

    assert counts[0] == 2  # sees itself + the other cluster-A report
    assert counts[1] == 2
    assert counts[2] == 1  # only sees itself


def test_recurrence_factor_saturates():
    assert recurrence_factor(1, saturation_count=5) == 0.0
    assert recurrence_factor(5, saturation_count=5) == 1.0
    assert recurrence_factor(10, saturation_count=5) == 1.0
    assert recurrence_factor(3, saturation_count=5) == pytest.approx(0.5)


def test_recurrence_factor_rejects_bad_saturation_count():
    with pytest.raises(ValueError):
        recurrence_factor(3, saturation_count=1)
