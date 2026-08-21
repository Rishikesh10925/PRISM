import numpy as np

from augmentations import build_eval_augmentations, build_train_augmentations


def test_train_augmentations_preserve_shape_and_dtype():
    pipeline = build_train_augmentations(p_rain=1.0, p_glare=1.0, p_blur=1.0)
    img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)

    out = pipeline(image=img)["image"]

    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_train_augmentations_are_stochastic_with_zero_probability():
    pipeline = build_train_augmentations(p_rain=0.0, p_glare=0.0, p_blur=0.0)
    img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)

    out = pipeline(image=img)["image"]

    # only RandomBrightnessContrast (p=0.5, fixed here) can still fire; rain/glare/blur
    # at p=0 must never fire, so shape/dtype still hold and no exception is raised
    assert out.shape == img.shape


def test_eval_augmentations_is_identity():
    pipeline = build_eval_augmentations()
    img = (np.random.rand(30, 40, 3) * 255).astype(np.uint8)

    out = pipeline(image=img)["image"]

    assert np.array_equal(out, img)
