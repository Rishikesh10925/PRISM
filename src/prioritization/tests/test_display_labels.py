from formula import PRIORITY_CATEGORIES, priority_category
from road_type import road_importance_label
from traffic_recurrence import recurrence_level_label, traffic_level_label


def test_priority_category_covers_full_range():
    assert PRIORITY_CATEGORIES == ["Less Important", "Moderate", "Important", "Very Important"]
    assert priority_category(0) == "Less Important"
    assert priority_category(24.9) == "Less Important"
    assert priority_category(25) == "Moderate"
    assert priority_category(49.9) == "Moderate"
    assert priority_category(50) == "Important"
    assert priority_category(74.9) == "Important"
    assert priority_category(75) == "Very Important"
    assert priority_category(100) == "Very Important"


def test_road_importance_label_covers_real_osm_weight_values():
    # the three weight values road_type_weight() actually produces today (1.0/0.7/0.4)
    assert road_importance_label(1.0) == "High"
    assert road_importance_label(0.7) == "Medium"
    assert road_importance_label(0.4) == "Low"
    assert road_importance_label(0.1) == "Very Low"


def test_traffic_level_label_thresholds():
    assert traffic_level_label(0.1) == "Low"
    assert traffic_level_label(0.5) == "Medium"
    assert traffic_level_label(0.9) == "High"


def test_recurrence_level_label_thresholds():
    assert recurrence_level_label(1) == "None"
    assert recurrence_level_label(2) == "Few"
    assert recurrence_level_label(3) == "Few"
    assert recurrence_level_label(4) == "Several"
    assert recurrence_level_label(6) == "Several"
    assert recurrence_level_label(7) == "Many"
    assert recurrence_level_label(20) == "Many"
