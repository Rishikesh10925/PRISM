import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from convert_coco import convert_coco_file
from convert_supervisely import convert_supervisely_dir
from convert_voc import convert_voc_dir
from convert_yolo_box import convert_yolo_box_dir
from write_yolo_seg import write_yolo_seg


VOC_XML = """<annotation>
  <filename>img1.jpg</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object>
    <name>D40</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>40</ymax></bndbox>
  </object>
  <object>
    <name>D99_unknown</name>
    <bndbox><xmin>60</xmin><ymin>10</ymin><xmax>80</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>"""


def test_convert_voc(tmp_path: Path):
    images_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    images_dir.mkdir()
    ann_dir.mkdir()
    (ann_dir / "img1.xml").write_text(VOC_XML, encoding="utf-8")

    results = convert_voc_dir(images_dir, ann_dir, source="rdd2022")

    assert len(results) == 1
    ann = results[0]
    assert ann.width == 200 and ann.height == 100
    # D40 -> pothole kept, D99_unknown dropped (unmapped)
    assert len(ann.instances) == 1
    inst = ann.instances[0]
    assert inst.class_name == "pothole"
    assert inst.needs_mask is True
    assert inst.polygon == [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]


def test_convert_coco_with_segmentation_and_bbox_fallback(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "pothole"}],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
                "bbox": [10, 10, 20, 20],
            },
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": [],
                "bbox": [50, 50, 10, 10],
            },
        ],
    }
    json_path = tmp_path / "coco.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    results = convert_coco_file(json_path, tmp_path / "images", source="roboflow_pothole")

    assert len(results) == 1
    instances = results[0].instances
    assert len(instances) == 2
    assert instances[0].needs_mask is False
    assert instances[0].polygon == [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    assert instances[1].needs_mask is True  # fell back to bbox rectangle
    assert instances[1].polygon == [(50.0, 50.0), (60.0, 50.0), (60.0, 60.0), (50.0, 60.0)]


def _encode_bitmap(mask: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", mask)
    assert ok
    return base64.b64encode(zlib.compress(png.tobytes())).decode("ascii")


def test_convert_supervisely_bitmap_and_polygon(tmp_path: Path):
    images_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    images_dir.mkdir()
    ann_dir.mkdir()

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255  # a 10x10 filled square

    data = {
        "size": {"width": 100, "height": 100},
        "objects": [
            {
                "classTitle": "pothole",
                "geometryType": "bitmap",
                "bitmap": {"data": _encode_bitmap(mask), "origin": [30, 40]},
            },
            {
                "classTitle": "road",
                "geometryType": "polygon",
                "points": {"exterior": [[0, 0], [50, 0], [50, 50], [0, 50]]},
            },
            {
                "classTitle": "shallow",  # unmapped -> dropped
                "geometryType": "polygon",
                "points": {"exterior": [[0, 0], [5, 0], [5, 5]]},
            },
        ],
    }
    (ann_dir / "img1.jpg.json").write_text(json.dumps(data), encoding="utf-8")

    results = convert_supervisely_dir(images_dir, ann_dir, source="indian_roads")

    assert len(results) == 1
    instances = results[0].instances
    class_names = sorted(i.class_name for i in instances)
    assert class_names == ["pothole", "road_surface"]

    pothole_inst = next(i for i in instances if i.class_name == "pothole")
    xs = [p[0] for p in pothole_inst.polygon]
    ys = [p[1] for p in pothole_inst.polygon]
    # bitmap origin (30, 40) offset + the 10x10 square drawn at [5:15, 5:15]
    assert min(xs) == pytest.approx(35, abs=1)
    assert min(ys) == pytest.approx(45, abs=1)


def test_convert_yolo_box(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    Image.new("RGB", (100, 100), color="white").save(images_dir / "a.jpg")
    # class 0 = pothole, box centered at (0.5, 0.5), 20% width/height
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    results = convert_yolo_box_dir(images_dir, labels_dir, source="roboflow_pothole", source_classes=["pothole"])

    assert len(results) == 1
    ann = results[0]
    assert ann.width == 100 and ann.height == 100
    assert len(ann.instances) == 1
    assert ann.instances[0].needs_mask is True
    assert ann.instances[0].polygon == [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]


def test_write_yolo_seg_skips_unmasked_instances(tmp_path: Path):
    from schema import ImageAnnotation, Instance

    ann = ImageAnnotation(
        image_path="img.jpg",
        width=100,
        height=100,
        source="test",
        instances=[
            Instance(class_name="pothole", polygon=[(10, 10), (30, 10), (30, 30), (10, 30)], needs_mask=False),
            Instance(class_name="pothole", polygon=[(1, 1), (2, 1), (2, 2), (1, 2)], needs_mask=True),
        ],
    )

    out_dir = tmp_path / "out"
    write_yolo_seg([ann], out_dir)

    label_text = (out_dir / "labels" / "img.txt").read_text(encoding="utf-8")
    lines = [l for l in label_text.splitlines() if l.strip()]
    assert len(lines) == 1  # the needs_mask instance was skipped
    assert lines[0].startswith("0 ")  # pothole is class id 0
