from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from uuid import UUID, uuid4

from .models import AugmentationProfile, TransformSpec


NNUNET_DEFAULT_IMPLEMENTATION = "nnunetv2.standard"
CELLVECTOR_IMPLEMENTATION = "cellvector.batchgeneratorsv2"
CELLVECTOR_IMPLEMENTATION_VERSION = "1.0.0"


def nnunet_default_profile() -> AugmentationProfile:
    """Describe delegation to nnU-Net's installed standard augmentation path."""
    try:
        installed_version = version("nnunetv2")
    except PackageNotFoundError:
        installed_version = "not-installed"
    return AugmentationProfile(
        profile_id=UUID("83c2021c-8664-4bd0-b51e-cf251658a07f"),
        name="nnunet_default",
        implementation=NNUNET_DEFAULT_IMPLEMENTATION,
        implementation_version=installed_version,
        seed_policy="training_seed",
        transforms=(),
    )


def conservative_profile() -> AugmentationProfile:
    """Return a conservative grayscale profile with explicit executable settings."""
    transforms = (
        TransformSpec(
            name="rotation", probability=0.15, parameters={"degrees": (-10.0, 10.0)}
        ),
        TransformSpec(name="horizontal_flip", probability=0.5),
        TransformSpec(name="vertical_flip", probability=0.5),
        TransformSpec(
            name="scale", probability=0.15, parameters={"factor": (0.9, 1.1)}
        ),
        TransformSpec(
            name="stretch",
            probability=0.1,
            parameters={"x_factor": (0.9, 1.1), "y_factor": (0.9, 1.1)},
        ),
        TransformSpec(name="crop", probability=0.15),
        TransformSpec(
            name="gaussian_noise",
            probability=0.08,
            parameters={"sigma": (0.0, 0.05)},
        ),
        TransformSpec(
            name="gaussian_blur",
            probability=0.1,
            parameters={"sigma": (0.5, 0.8)},
        ),
        TransformSpec(
            name="brightness",
            probability=0.1,
            parameters={"factor": (0.9, 1.1)},
        ),
        TransformSpec(
            name="contrast",
            probability=0.1,
            parameters={"factor": (0.9, 1.1)},
        ),
        TransformSpec(
            name="gamma", probability=0.1, parameters={"gamma": (0.9, 1.1)}
        ),
        TransformSpec(
            name="low_resolution",
            probability=0.08,
            parameters={"scale": (0.75, 1.0)},
        ),
        TransformSpec(
            name="block_occlusion",
            probability=0.05,
            parameters={"height": 8, "width": 8, "count": 1, "fill": 0.0},
        ),
    )
    return AugmentationProfile(
        profile_id=UUID("740fe3b7-43ea-41c9-8e2a-bbff1a93d83d"),
        name="conservative",
        implementation=CELLVECTOR_IMPLEMENTATION,
        implementation_version=CELLVECTOR_IMPLEMENTATION_VERSION,
        seed_policy="training_seed",
        transforms=transforms,
    )


def custom_profile(
    transforms: Iterable[TransformSpec],
    *,
    name: str = "custom",
    seed_policy: str = "training_seed",
) -> AugmentationProfile:
    """Build a validated custom profile without accepting executable import names."""
    return AugmentationProfile(
        profile_id=uuid4(),
        name=name,
        implementation=CELLVECTOR_IMPLEMENTATION,
        implementation_version=CELLVECTOR_IMPLEMENTATION_VERSION,
        seed_policy=seed_policy,
        transforms=tuple(transforms),
    )
