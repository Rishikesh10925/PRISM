from pathlib import Path

import write_results_doc as wrd


def test_build_report_handles_missing_and_present_csvs(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()

    monkeypatch.setattr(wrd, "EVAL_DIR", eval_dir)

    # no CSVs yet -> report should say "not yet evaluated", not crash
    report = wrd.build_report()
    assert "not yet evaluated" in report

    (eval_dir / "detection_metrics_test_summary.csv").write_text(
        "box_map50,box_map50_95,box_precision_mean,box_recall_mean,mask_map50,mask_map50_95\n"
        "0.812,0.501,0.77,0.80,0.79,0.44\n",
        encoding="utf-8",
    )
    (eval_dir / "detection_metrics_test.csv").write_text(
        "class,box_precision,box_recall,box_map50,box_map50_95,mask_map50,mask_map50_95\n"
        "pothole,0.77,0.80,0.812,0.501,0.79,0.44\n",
        encoding="utf-8",
    )

    report = wrd.build_report()
    assert "0.812" in report
    assert "pothole" in report
    assert "not yet evaluated" in report  # Mask R-CNN still missing
