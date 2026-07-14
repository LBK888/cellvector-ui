from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from scipy import ndimage
from skimage.transform import resize

from .models import (
    AugmentationError,
    AugmentationPreviewRecord,
    AugmentationProfile,
    ResolvedTransform,
    TransformSpec,
)


_IMAGE_ONLY = frozenset(
    {
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "contrast",
        "gamma",
        "low_resolution",
        "block_occlusion",
    }
)


def _sample_range(rng: np.random.Generator, value: object) -> float:
    low, high = value  # type: ignore[misc]
    if low == high:
        return float(low)
    return float(rng.uniform(float(low), float(high)))


def _centered_affine(
    array: np.ndarray,
    *,
    y_factor: float,
    x_factor: float,
    order: int,
) -> np.ndarray:
    matrix = np.diag((1.0 / y_factor, 1.0 / x_factor))
    center = (np.asarray(array.shape, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center
    return ndimage.affine_transform(
        array,
        matrix,
        offset=offset,
        output_shape=array.shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _resize(array: np.ndarray, shape: tuple[int, int], *, order: int) -> np.ndarray:
    return resize(
        array,
        shape,
        order=order,
        mode="constant",
        cval=0.0,
        clip=False,
        preserve_range=True,
        anti_aliasing=False,
    )


def _restore_image_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(np.clip(array, limits.min, limits.max)).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return (array >= 0.5).astype(dtype)
    return array.astype(dtype, copy=False)


def _validate_arrays(image: np.ndarray, labels: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or not isinstance(labels, np.ndarray):
        raise AugmentationError(
            "AUGMENTATION_ARRAY_INVALID", "image and labels must be NumPy arrays"
        )
    if image.ndim != 2 or labels.ndim != 2 or 0 in image.shape or 0 in labels.shape:
        raise AugmentationError(
            "AUGMENTATION_SHAPE_INVALID", "image and labels must be nonempty 2D arrays"
        )
    if image.shape != labels.shape:
        raise AugmentationError(
            "AUGMENTATION_SHAPE_MISMATCH", "image and label shapes must match"
        )
    if not (
        np.issubdtype(image.dtype, np.number)
        and not np.issubdtype(image.dtype, np.complexfloating)
    ):
        raise AugmentationError(
            "AUGMENTATION_IMAGE_DTYPE_INVALID", "image dtype must be real numeric"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise AugmentationError(
            "AUGMENTATION_LABEL_DTYPE_INVALID", "label dtype must contain discrete integers"
        )
    if not np.isfinite(image).all():
        raise AugmentationError(
            "AUGMENTATION_IMAGE_NONFINITE", "image contains non-finite values"
        )


def _apply_transform(
    image: np.ndarray,
    labels: np.ndarray,
    transform: TransformSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    name = transform.name
    parameters = transform.parameters
    sampled: dict[str, object] = {}

    if name == "horizontal_flip":
        sampled["axis"] = 1
        return image[:, ::-1].copy(), labels[:, ::-1].copy(), sampled
    if name == "vertical_flip":
        sampled["axis"] = 0
        return image[::-1, :].copy(), labels[::-1, :].copy(), sampled
    if name == "rotation":
        degrees = _sample_range(rng, parameters["degrees"])
        sampled["degrees"] = degrees
        return (
            ndimage.rotate(
                image,
                degrees,
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ),
            ndimage.rotate(
                labels,
                degrees,
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(labels.dtype, copy=False),
            sampled,
        )
    if name == "scale":
        factor = _sample_range(rng, parameters["factor"])
        sampled["factor"] = factor
        return (
            _centered_affine(image, y_factor=factor, x_factor=factor, order=1),
            _centered_affine(labels, y_factor=factor, x_factor=factor, order=0).astype(
                labels.dtype, copy=False
            ),
            sampled,
        )
    if name == "stretch":
        x_factor = _sample_range(rng, parameters["x_factor"])
        y_factor = _sample_range(rng, parameters["y_factor"])
        sampled.update({"x_factor": x_factor, "y_factor": y_factor})
        return (
            _centered_affine(
                image, y_factor=y_factor, x_factor=x_factor, order=1
            ),
            _centered_affine(
                labels, y_factor=y_factor, x_factor=x_factor, order=0
            ).astype(labels.dtype, copy=False),
            sampled,
        )
    if name == "crop":
        height, width = image.shape
        fraction = _sample_range(rng, parameters["fraction"])
        crop_height = max(1, min(height, int(round(height * fraction))))
        crop_width = max(1, min(width, int(round(width * fraction))))
        top = int(rng.integers(0, height - crop_height + 1))
        left = int(rng.integers(0, width - crop_width + 1))
        sampled.update(
            {
                "fraction": fraction,
                "top": top,
                "left": left,
                "height": crop_height,
                "width": crop_width,
                "output_shape": (height, width),
            }
        )
        slices = np.s_[top : top + crop_height, left : left + crop_width]
        return (
            _resize(image[slices], image.shape, order=1),
            _resize(labels[slices], labels.shape, order=0).astype(
                labels.dtype, copy=False
            ),
            sampled,
        )
    if name == "gaussian_noise":
        sigma = _sample_range(rng, parameters["sigma"])
        sampled["sigma"] = sigma
        noise = rng.normal(0.0, sigma, size=image.shape)
        return image + noise, labels, sampled
    if name == "gaussian_blur":
        sigma = _sample_range(rng, parameters["sigma"])
        sampled["sigma"] = sigma
        return ndimage.gaussian_filter(image, sigma=sigma), labels, sampled
    if name == "brightness":
        factor = _sample_range(rng, parameters["factor"])
        sampled["factor"] = factor
        return image * factor, labels, sampled
    if name == "contrast":
        factor = _sample_range(rng, parameters["factor"])
        sampled["factor"] = factor
        mean = float(np.mean(image))
        return (image - mean) * factor + mean, labels, sampled
    if name == "gamma":
        gamma = _sample_range(rng, parameters["gamma"])
        sampled["gamma"] = gamma
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if maximum == minimum:
            return image.copy(), labels, sampled
        normalized = (image - minimum) / (maximum - minimum)
        return np.power(normalized, gamma) * (maximum - minimum) + minimum, labels, sampled
    if name == "low_resolution":
        scale = _sample_range(rng, parameters["scale"])
        down_shape = (
            max(1, int(round(image.shape[0] * scale))),
            max(1, int(round(image.shape[1] * scale))),
        )
        sampled.update({"scale": scale, "downsampled_shape": down_shape})
        downsampled = _resize(image, down_shape, order=1)
        return _resize(downsampled, image.shape, order=1), labels, sampled
    if name == "block_occlusion":
        block_height = int(parameters["height"])
        block_width = int(parameters["width"])
        count = int(parameters["count"])
        fill = float(parameters["fill"])
        if block_height > image.shape[0] or block_width > image.shape[1]:
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                "block_occlusion dimensions exceed the preview image",
            )
        result = image.copy()
        rectangles: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            top = int(rng.integers(0, image.shape[0] - block_height + 1))
            left = int(rng.integers(0, image.shape[1] - block_width + 1))
            result[top : top + block_height, left : left + block_width] = fill
            rectangles.append((top, left, block_height, block_width))
        sampled.update({"fill": fill, "rectangles": tuple(rectangles)})
        return result, labels, sampled

    raise AugmentationError(
        "AUGMENTATION_UNSUPPORTED", f"preview does not implement transform {name!r}"
    )


def preview_augmentation(
    image: np.ndarray,
    labels: np.ndarray,
    profile: AugmentationProfile,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, AugmentationPreviewRecord]:
    """Apply a reproducible 2D grayscale preview with synchronized geometry."""
    _validate_arrays(image, labels)
    if type(seed) is not int or seed < 0:
        raise AugmentationError(
            "AUGMENTATION_SEED_INVALID", "preview seed must be a non-negative integer"
        )
    if not isinstance(profile, AugmentationProfile):
        raise AugmentationError(
            "AUGMENTATION_PROFILE_INVALID", "profile is not an AugmentationProfile"
        )

    rng = np.random.default_rng(seed)
    image_dtype = image.dtype
    label_dtype = labels.dtype
    result_image = image.astype(np.float64, copy=True)
    result_labels = labels.copy()
    decisions: list[ResolvedTransform] = []

    for transform in profile.transforms:
        applied = profile.enabled and bool(rng.random() < transform.probability)
        if not applied:
            decisions.append(
                ResolvedTransform(
                    name=transform.name, applied=False, sampled_parameters={}
                )
            )
            continue
        result_image, result_labels, sampled = _apply_transform(
            result_image, result_labels, transform, rng
        )
        if transform.name in _IMAGE_ONLY and result_labels.dtype != label_dtype:
            raise AugmentationError(
                "AUGMENTATION_LABEL_MISMATCH",
                f"image-only transform {transform.name} changed label dtype",
            )
        decisions.append(
            ResolvedTransform(
                name=transform.name,
                applied=True,
                sampled_parameters=sampled,
            )
        )

    if result_image.shape != result_labels.shape:
        raise AugmentationError(
            "AUGMENTATION_SHAPE_MISMATCH", "preview transform desynchronized arrays"
        )
    if not np.isfinite(result_image).all():
        raise AugmentationError(
            "AUGMENTATION_IMAGE_NONFINITE", "preview produced non-finite image values"
        )
    result_image = _restore_image_dtype(result_image, image_dtype)
    result_labels = result_labels.astype(label_dtype, copy=False)
    record = AugmentationPreviewRecord(
        profile_hash=profile.identity_hash(), seed=seed, transforms=tuple(decisions)
    )
    return result_image, result_labels, record
