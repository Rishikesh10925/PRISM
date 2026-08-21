"""Weather/capture-condition augmentation pipeline for the detector.

Road images are captured in highly variable real-world conditions (rain, sun glare on
wet asphalt, motion blur from a moving vehicle/phone). Ultralytics' built-in augment
config covers geometric transforms (flip/scale/mosaic) and light color jitter (HSV) but
has no rain/glare/motion-blur simulation, so this is a standalone Albumentations
pipeline applied as a pre-processing step before YOLO's own augmentations run.

Usage: build_train_augmentations() returns a Compose that takes/returns a dict with
"image" (HWC uint8 array) — polygons aren't touched since these are all pixel-only
photometric/blur transforms (no geometric warp), so segmentation labels stay valid
unchanged.
"""

from __future__ import annotations

import albumentations as A


def build_train_augmentations(p_rain: float = 0.15, p_glare: float = 0.15, p_blur: float = 0.25) -> A.Compose:
    """Each effect is independently sampled (not mutually exclusive) so combinations
    like rain + motion blur, which do occur in real dashcam footage, can appear."""
    return A.Compose(
        [
            A.RandomRain(
                slant_range=(-10, 10), drop_length=12, drop_width=1, blur_value=3, brightness_coefficient=0.85, p=p_rain
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.6), src_radius=120, num_flare_circles_range=(3, 6), p=p_glare
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(5, 15), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                ],
                p=p_blur,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ]
    )


def build_eval_augmentations() -> A.Compose:
    """No augmentation at eval time — identity pipeline, kept for API symmetry so
    training/eval code paths look the same regardless of which is active."""
    return A.Compose([])
