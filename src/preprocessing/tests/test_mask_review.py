import csv

from build_mask_review import select_items
from summarize_mask_review import summarize, wilson_interval


def _row(image, rating, missed="false"):
    return {"review_id": "R0", "image": image, "rating": rating, "missed_pothole": missed, "false_mask": "false", "artifact": "false", "notes": ""}


def test_summary_separates_sam_from_human_controls():
    rows = [
        _row("kaggle_a_1.jpg", "good"),
        _row("kaggle_a_2.jpg", "acceptable", missed="true"),
        _row("kaggle_a_3.jpg", "bad"),
        _row("kaggle_a_4.jpg", ""),
        _row("pothole600_testing_0001.png", "good"),
    ]
    s = summarize(rows)
    sam = s["sam_pseudo_label"]
    assert sam["num_rated"] == 3
    assert sam["counts"] == {"good": 1, "acceptable": 1, "bad": 1}
    assert abs(sam["usable_share"] - 2 / 3) < 1e-9
    assert sam["flag_counts"]["missed_pothole"] == 1
    assert s["human_drawn_control"]["num_rated"] == 1
    assert s["num_unrated"] == 1


def test_wilson_interval_bounds():
    low, high = wilson_interval(45, 50)
    assert 0.78 < low < 0.9 < high < 0.97
    assert wilson_interval(0, 10)[0] == 0.0


def test_select_items_includes_all_sam_and_requested_controls(tmp_path):
    checklist = tmp_path / "review_checklist.csv"
    with open(checklist, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "image"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"source": "kaggle_x", "image": f"kaggle_x_{i}.jpg"})
        for i in range(10):
            writer.writerow({"source": "pothole600_training", "image": f"pothole600_training_{i}.png"})

    items = select_items(checklist, num_controls=3, seed=0)
    assert sum(r["source"] == "kaggle_x" for r in items) == 5
    assert sum(r["source"].startswith("pothole600") for r in items) == 3
    assert items == select_items(checklist, num_controls=3, seed=0)  # deterministic
