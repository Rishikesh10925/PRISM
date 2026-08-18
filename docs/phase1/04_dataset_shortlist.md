# Phase 1 / Task 4 — Dataset Shortlist

| # | Dataset | Size | Format | Classes | License (as stated by source) | Link |
|---|---|---|---|---|---|---|
| D1 | RDD2022 (Road Damage Dataset, CRDDC'2022) | 47,420 images, 55,000+ damage instances, 6 countries (Japan, India, Czech Republic, Norway, USA, China) | Images + PASCAL VOC XML bounding boxes | longitudinal crack, transverse crack, alligator crack, pothole | CC BY-SA 4.0 (per figshare record) | [figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) / [arXiv paper](https://arxiv.org/abs/2209.08538) |
| D2 | Annotated Potholes Image Dataset (Kaggle, Chitholian) | ~665 images | Images + PASCAL VOC XML bounding boxes | pothole | Kaggle-hosted; original author's undergraduate thesis dataset — no explicit OSS license stated on the listing, needs direct confirmation before redistribution (see Task 5) | [Kaggle](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) |
| D3 | Potholes-Detection-YOLOv8 (Kaggle) | not stated on listing page (moderate size, pre-split for YOLOv8) | Images + YOLO txt bounding boxes | pothole | CC0: Public Domain | [Kaggle](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) |
| D4 | Potholes and Roads Instance Segmentation (Roboflow Universe) | 2,592 images | Images + YOLO-seg / COCO JSON (multi-format export) | pothole, road | CC BY 4.0 (Roboflow Universe default license) | [Roboflow](https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation) |
| D5 | Pothole Object Detection Dataset (Roboflow public dataset) | 665 images (mirrors D2 source) | Images + multiple export formats incl. YOLOv8 | pothole | ODbL v1.0 (confirmed on-page, see [05_dataset_verification.md](05_dataset_verification.md)) | [Roboflow](https://public.roboflow.com/object-detection/pothole) |
| D6 | Pothole-600 | 600 collections (67 stereo pairs, 3 resolution subsets: 1028x1730 / 1030x1720 / 1028x1710) | RGB image + transformed disparity map + pixel-level binary ground-truth mask | pothole (binary mask) + per-pixel disparity (depth proxy) | Research/academic use per project site; explicit license terms need direct confirmation (see Task 5) | [project site](https://sites.google.com/view/pothole-600/dataset) |
| D7 | Indian Roads (semantic segmentation) | 3,227 images, 8,129 labeled objects, ~979 MB | Images + Supervisely-format pixel-level segmentation masks (also mirrored on Kaggle) | road, footpath, shallow, pothole | GNU GPL 2.0 | [Dataset Ninja](https://datasetninja.com/indian-roads-semantic-segmentation) / [Kaggle mirror](https://www.kaggle.com/datasets/mitangshu11/indian-roads-dataset) |

## Note on CNRDD

The blueprint lists "CNRDD/Pothole-600" together as if interchangeable. On verification, these are **not** the
same dataset: CNRDD (referenced in P03-adjacent literature) is a Chinese road-damage dataset labeled to the
JTG5210-2018 civil-engineering standard, with no confirmed public download link or depth/stereo annotations
found during this search — unlike Pothole-600, which is confirmed to have real stereo-derived depth
(disparity) ground truth. **Decision: drop CNRDD from the shortlist** (no accessible source found) and keep
Pothole-600 as the sole depth-annotated dataset. This is exactly the kind of correction Task 5 (verify
access and licensing) is meant to catch before it becomes a false claim in the paper's Dataset section.

## Selection rationale

- **D1 (RDD2022)** is the backbone multi-class, multi-country dataset — largest, most cited, and the only
  source giving crack classes needed for the Contribution 3 open-vocabulary comparison.
- **D2/D3 or D4/D5** (Kaggle + Roboflow pothole-only sets) supply additional pothole-only volume and
  geographic/camera diversity to merge with RDD2022's pothole subset; D3/D5 (CC0/Public Domain) are
  preferred over D2 wherever overlapping content exists, since their license is unambiguous.
- **D6 (Pothole-600)** is kept specifically for its stereo depth ground truth, to validate the monocular
  MiDaS depth-proxy module against real disparity on a subset — not primarily for training volume.
- **D7 (Indian Roads)** supplies India-specific geography (relevant to the NMIMS Hyderabad context and to
  road-surface segmentation for the area-ratio severity cue, since it separately labels "road" and
  "footpath").

Next: [05_dataset_verification.md](05_dataset_verification.md) confirms each of D1-D7 is actually reachable and re-checks license terms before finalizing.
