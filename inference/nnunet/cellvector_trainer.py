from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.blank_rectangle import BlankRectangleTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform,
)
from pydantic import ValidationError

# batchgenerators 0.25.1 still imports this SciPy compatibility namespace while
# nnU-Net 2.8.1 is imported. Keep the suppression local and message-specific so
# the project's warnings-as-errors gate remains effective for all other warnings.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Please import `.*` from the `scipy\.ndimage` namespace.*",
        category=DeprecationWarning,
    )
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from cellvector.augmentation import (
    AugmentationError,
    AugmentationProfile,
    CELLVECTOR_IMPLEMENTATION,
    CELLVECTOR_IMPLEMENTATION_VERSION,
    NNUNET_DEFAULT_IMPLEMENTATION,
    TransformSpec,
    nnunet_default_profile,
)


PROFILE_ENVIRONMENT_VARIABLE = "CELLVECTOR_AUGMENTATION_PROFILE"
HASH_ENVIRONMENT_VARIABLE = "CELLVECTOR_AUGMENTATION_SHA256"
PROVENANCE_FILENAME = "cellvector-augmentation-profile.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _VerifiedArtifact:
    profile: AugmentationProfile
    path: Path | None
    file_sha256: str | None


@dataclass(frozen=True)
class _ProbabilisticRange:
    probability: float
    value_range: tuple[float, float]

    def sample(self, identity: float) -> float:
        if np.random.uniform() >= self.probability:
            return identity
        low, high = self.value_range
        return float(low if low == high else np.random.uniform(low, high))


@dataclass(frozen=True)
class _ProbabilisticStretch:
    probability: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def sample(self) -> tuple[float, float]:
        if np.random.uniform() >= self.probability:
            return (1.0, 1.0)
        x_low, x_high = self.x_range
        y_low, y_high = self.y_range
        x_factor = float(
            x_low if x_low == x_high else np.random.uniform(x_low, x_high)
        )
        y_factor = float(
            y_low if y_low == y_high else np.random.uniform(y_low, y_high)
        )
        return (y_factor, x_factor)


@dataclass(frozen=True)
class _ProbabilisticCrop:
    probability: float
    fraction_range: tuple[float, float]

    def sample(self) -> float | None:
        if np.random.uniform() >= self.probability:
            return None
        low, high = self.fraction_range
        return float(low if low == high else np.random.uniform(low, high))


class _CombinedRotationSampler:
    """Combine independently sampled rotation requests into one grid angle."""

    def __init__(self, requests: Sequence[_ProbabilisticRange]) -> None:
        self.requests = tuple(requests)
        self._sampled_angle = 0.0

    def __call__(self, *, dim: int | None = None, **_: Any) -> float:
        if dim in (None, 0):
            self._sampled_angle = sum(request.sample(0.0) for request in self.requests)
        # SpatialTransform's 2D matrix consumes angles[-1]. Returning zero for
        # dim 0 avoids applying the 2D angle twice if the runtime changes.
        return self._sampled_angle if dim in (None, 1) else 0.0


class _CombinedScalingSampler:
    """Combine independent isotropic scale and per-axis stretch decisions."""

    def __init__(
        self,
        scale_requests: Sequence[_ProbabilisticRange],
        stretch_requests: Sequence[_ProbabilisticStretch],
    ) -> None:
        self.scale_requests = tuple(scale_requests)
        self.stretch_requests = tuple(stretch_requests)
        self._sampled_factors = (1.0, 1.0)

    def __call__(self, *, dim: int | None = None, **_: Any) -> float:
        if dim in (None, 0):
            isotropic = _bounded_scale_product(
                request.sample(1.0) for request in self.scale_requests
            )
            sampled_stretches = [request.sample() for request in self.stretch_requests]
            y_factor = _bounded_scale_product(item[0] for item in sampled_stretches)
            x_factor = _bounded_scale_product(item[1] for item in sampled_stretches)
            self._sampled_factors = (
                _bounded_scale_product((isotropic, y_factor)),
                _bounded_scale_product((isotropic, x_factor)),
            )
        if dim is None:
            return self._sampled_factors[0]
        return self._sampled_factors[dim]


def _bounded_scale_product(values) -> float:
    """Multiply runtime scale factors without NumPy overflow/warning paths."""
    result = 1.0
    for value in values:
        factor = float(value)
        if not math.isfinite(factor):
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                "fused scaling received a non-finite runtime factor",
            )
        result *= factor
        if not math.isfinite(result) or not 0.25 <= result <= 4.0:
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                "fused scaling product is outside the operational range [0.25, 4]",
            )
    return result


class _FusedSpatialTransform(SpatialTransform):
    """One mandatory patch extraction with independent optional decisions."""

    def __init__(self, *args, crop_requests: Sequence[_ProbabilisticCrop], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.crop_requests = tuple(crop_requests)

    def get_parameters(self, **data_dict) -> dict:
        sampled_fractions = [request.sample() for request in self.crop_requests]
        applied_fractions = [item for item in sampled_fractions if item is not None]
        apply_random_crop = bool(applied_fractions)
        configured_random_crop = self.random_crop
        configured_border = self.patch_center_dist_from_border
        try:
            # Installed SpatialTransform performs either random displacement or
            # deterministic center extraction while always emitting patch_size.
            self.random_crop = apply_random_crop
            if applied_fractions:
                fraction = min(applied_fractions)
                shape = data_dict["image"].shape[1:]
                # Map the validated crop fraction onto the displacement range
                # available between initial_patch_size and patch_size.
                displacement_ratio = (1.0 - fraction) / 0.5
                self.patch_center_dist_from_border = tuple(
                    (dimension / 2.0)
                    - max(0.0, (dimension - patch) / 2.0) * displacement_ratio
                    for dimension, patch in zip(shape, self.patch_size)
                )
            return super().get_parameters(**data_dict)
        finally:
            self.random_crop = configured_random_crop
            self.patch_center_dist_from_border = configured_border


class _AxisFlipTransform(BasicTransform):
    """A batchgeneratorsv2 transform that always flips one spatial axis."""

    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def _apply_to_image(self, image: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.flip(image, (self.axis + 1,))

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return torch.flip(segmentation, (self.axis + 1,))

    def _apply_to_regr_target(
        self, regression_target: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return torch.flip(regression_target, (self.axis + 1,))


def _runtime_error(code: str, message: str) -> RuntimeError:
    return RuntimeError(f"{code}: {message}")


def _load_verified_artifact() -> _VerifiedArtifact:
    configured_path = os.environ.get(PROFILE_ENVIRONMENT_VARIABLE)
    configured_hash = os.environ.get(HASH_ENVIRONMENT_VARIABLE)
    if not configured_path:
        if configured_hash:
            raise _runtime_error(
                "AUGMENTATION_PROFILE_MISSING",
                f"{PROFILE_ENVIRONMENT_VARIABLE} is required when a hash is supplied",
            )
        return _VerifiedArtifact(nnunet_default_profile(), None, None)

    candidate = Path(configured_path).expanduser()
    if not candidate.exists():
        raise _runtime_error(
            "AUGMENTATION_PROFILE_MISSING", f"profile file does not exist: {candidate}"
        )
    if candidate.is_symlink() or candidate.suffix.lower() != ".json":
        raise _runtime_error(
            "AUGMENTATION_PROFILE_PATH_INVALID",
            "profile must be a non-symlink JSON file",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_PATH_INVALID", "profile path cannot be resolved"
        ) from error
    if not resolved.is_file():
        raise _runtime_error(
            "AUGMENTATION_PROFILE_PATH_INVALID", "profile path is not a regular file"
        )
    if not configured_hash:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_SHA256_MISSING",
            f"{HASH_ENVIRONMENT_VARIABLE} is required",
        )

    try:
        file_bytes = resolved.read_bytes()
    except FileNotFoundError as error:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_MISSING",
            "profile disappeared before exact-file verification",
        ) from error
    except OSError as error:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_UNREADABLE",
            "profile could not be read for exact-file verification",
        ) from error
    actual_hash = sha256(file_bytes).hexdigest()
    if _SHA256_PATTERN.fullmatch(configured_hash) is None or configured_hash != actual_hash:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_HASH_MISMATCH",
            "configured SHA-256 does not match the exact profile file bytes",
        )
    try:
        payload = json.loads(file_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("profile JSON root is not an object")
        profile = AugmentationProfile.model_validate(payload)
    except AugmentationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValidationError) as error:
        raise _runtime_error(
            "AUGMENTATION_PROFILE_INVALID", "profile JSON failed immutable schema validation"
        ) from error

    if (
        profile.implementation != CELLVECTOR_IMPLEMENTATION
        or profile.implementation_version != CELLVECTOR_IMPLEMENTATION_VERSION
        or profile.seed_policy != "training_seed"
    ):
        raise _runtime_error(
            "AUGMENTATION_UNSUPPORTED",
            "verified custom trainer profiles must use the supported CellVector implementation",
        )
    return _VerifiedArtifact(profile, resolved, actual_hash)


def load_verified_augmentation_profile() -> AugmentationProfile:
    """Load an immutable local profile only after exact-file SHA-256 verification."""
    return _load_verified_artifact().profile


def _construct_fused_spatial_transform(
    profile: AugmentationProfile, patch_size: tuple[int, ...]
) -> SpatialTransform:
    optional_specs = profile.transforms if profile.enabled else ()
    rotation_requests: list[_ProbabilisticRange] = []
    scale_requests: list[_ProbabilisticRange] = []
    stretch_requests: list[_ProbabilisticStretch] = []
    crop_requests: list[_ProbabilisticCrop] = []
    for spec in optional_specs:
        if spec.name == "rotation":
            degrees = spec.parameters["degrees"]
            rotation_requests.append(
                _ProbabilisticRange(
                    spec.probability,
                    tuple(float(np.deg2rad(item)) for item in degrees),  # type: ignore[arg-type]
                )
            )
        elif spec.name == "scale":
            scale_requests.append(
                _ProbabilisticRange(
                    spec.probability, spec.parameters["factor"]  # type: ignore[arg-type]
                )
            )
        elif spec.name == "stretch":
            stretch_requests.append(
                _ProbabilisticStretch(
                    spec.probability,
                    spec.parameters["x_factor"],  # type: ignore[arg-type]
                    spec.parameters["y_factor"],  # type: ignore[arg-type]
                )
            )
        elif spec.name == "crop":
            crop_requests.append(
                _ProbabilisticCrop(
                    spec.probability,
                    spec.parameters["fraction"],  # type: ignore[arg-type]
                )
            )

    border = tuple(max(0, dimension // 2) for dimension in patch_size)
    return _FusedSpatialTransform(
        patch_size,
        patch_center_dist_from_border=border,
        random_crop=False,
        p_elastic_deform=0,
        # Both branches always run so an identity grid still performs the
        # mandatory initial_patch_size -> patch_size extraction.
        p_rotation=1,
        rotation=_CombinedRotationSampler(rotation_requests),
        p_scaling=1,
        scaling=_CombinedScalingSampler(scale_requests, stretch_requests),
        p_synchronize_scaling_across_axes=-1,
        bg_style_seg_sampling=False,
        mode_seg="nearest" if profile.label_interpolation == "nearest" else "nearest",
        border_mode_seg="zeros",
        padding_value_seg=-1,
        mode_image="bilinear" if profile.image_interpolation == "linear" else "bilinear",
        padding_mode_image="zeros",
        padding_value_image=0,
        crop_requests=crop_requests,
    )


def _construct_transform(
    spec: TransformSpec, patch_size: tuple[int, ...], ignore_axes: tuple[int, ...] | None
) -> BasicTransform:
    parameters = spec.parameters
    if spec.name == "horizontal_flip":
        return _AxisFlipTransform(1)
    if spec.name == "vertical_flip":
        return _AxisFlipTransform(0)
    if spec.name in {"rotation", "scale", "stretch", "crop"}:
        raise AugmentationError(
            "AUGMENTATION_UNSUPPORTED",
            f"{spec.name} must be translated through the fused spatial transform",
        )
    if spec.name == "gaussian_noise":
        low, high = parameters["sigma"]  # type: ignore[misc]
        return GaussianNoiseTransform(
            noise_variance=(float(low) ** 2, float(high) ** 2),
            p_per_channel=1,
            synchronize_channels=True,
        )
    if spec.name == "gaussian_blur":
        return GaussianBlurTransform(
            blur_sigma=parameters["sigma"],
            synchronize_channels=True,
            synchronize_axes=False,
            p_per_channel=1,
            benchmark=True,
        )
    if spec.name == "brightness":
        return MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast(parameters["factor"]),  # type: ignore[arg-type]
            synchronize_channels=True,
            p_per_channel=1,
        )
    if spec.name == "contrast":
        return ContrastTransform(
            contrast_range=BGContrast(parameters["factor"]),  # type: ignore[arg-type]
            preserve_range=True,
            synchronize_channels=True,
            p_per_channel=1,
        )
    if spec.name == "gamma":
        return GammaTransform(
            gamma=BGContrast(parameters["gamma"]),  # type: ignore[arg-type]
            p_invert_image=0,
            synchronize_channels=True,
            p_per_channel=1,
            p_retain_stats=1,
        )
    if spec.name == "low_resolution":
        return SimulateLowResolutionTransform(
            scale=parameters["scale"],
            synchronize_channels=True,
            synchronize_axes=True,
            ignore_axes=ignore_axes,
            allowed_channels=None,
            p_per_channel=1,
        )
    if spec.name == "block_occlusion":
        if (
            int(parameters["height"]) > patch_size[0]
            or int(parameters["width"]) > patch_size[1]
        ):
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                "block_occlusion dimensions exceed the configured training patch",
            )
        return BlankRectangleTransform(
            rectangle_size=(int(parameters["height"]), int(parameters["width"])),
            rectangle_value=float(parameters["fill"]),
            num_rectangles=int(parameters["count"]),
            force_square=False,
            p_per_channel=1,
        )
    raise AugmentationError(
        "AUGMENTATION_UNSUPPORTED",
        f"batchgeneratorsv2 translation is absent for {spec.name!r}",
    )


def _construct_controlled_pipeline(
    profile: AugmentationProfile,
    *,
    patch_size: Sequence[int],
    deep_supervision_scales: Sequence[Sequence[float]] | None,
    do_dummy_2d_data_aug: bool,
    use_mask_for_norm: Sequence[bool] | None,
    is_cascaded: bool,
    foreground_labels: Sequence[int] | None,
    regions: Sequence[Sequence[int] | int] | None,
    ignore_label: int | None,
) -> tuple[BasicTransform, list[dict[str, object]]]:
    transforms: list[BasicTransform] = []
    resolved: list[dict[str, object]] = []
    if do_dummy_2d_data_aug:
        transforms.append(Convert3DTo2DTransform())
        spatial_patch_size = tuple(int(item) for item in patch_size[1:])
        ignore_axes: tuple[int, ...] | None = (0,)
    else:
        spatial_patch_size = tuple(int(item) for item in patch_size)
        ignore_axes = None
    if len(spatial_patch_size) != 2:
        raise AugmentationError(
            "AUGMENTATION_UNSUPPORTED",
            "CellVector controlled augmentation supports the native 2D dataset contract",
        )

    # This is the single mandatory spatial operation. It always converts the
    # loader's initial patch to the configured patch and fuses all optional
    # rotation/scale/stretch/crop decisions into its shared image/seg grid.
    fused_spatial = _construct_fused_spatial_transform(profile, spatial_patch_size)
    transforms.append(fused_spatial)

    for spec in profile.transforms:
        if not profile.enabled:
            resolved.append(
                {
                    "name": spec.name,
                    "probability": spec.probability,
                    "parameters": dict(spec.parameters),
                    "enabled": False,
                    "batchgenerators_transform": None,
                }
            )
            continue
        if spec.name in {"rotation", "scale", "stretch", "crop"}:
            concrete_name = type(fused_spatial).__name__
        else:
            concrete = _construct_transform(spec, spatial_patch_size, ignore_axes)
            transforms.append(
                RandomTransform(concrete, apply_probability=spec.probability)
            )
            concrete_name = type(concrete).__name__
        resolved.append(
            {
                "name": spec.name,
                "probability": spec.probability,
                "parameters": dict(spec.parameters),
                "enabled": True,
                "batchgenerators_transform": concrete_name,
            }
        )

    if do_dummy_2d_data_aug:
        transforms.append(Convert2DTo3DTransform())
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(
            MaskImageTransform(
                apply_to_channels=[
                    index for index, enabled in enumerate(use_mask_for_norm) if enabled
                ],
                channel_idx_in_seg=0,
                set_outside_to=0,
            )
        )
    transforms.append(RemoveLabelTansform(-1, 0))
    if is_cascaded:
        if foreground_labels is None:
            raise _runtime_error(
                "AUGMENTATION_PROFILE_INVALID",
                "foreground labels are required for cascaded augmentation",
            )
        labels = list(foreground_labels)
        transforms.append(
            MoveSegAsOneHotToDataTransform(
                source_channel_idx=1,
                all_labels=labels,
                remove_channel_from_source=True,
            )
        )
        transforms.append(
            RandomTransform(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(labels), 0)),
                    strel_size=(1, 8),
                    p_per_label=0.5,
                ),
                apply_probability=0.4,
            )
        )
        transforms.append(
            RandomTransform(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(labels), 0)),
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                    p_per_label=0.5,
                ),
                apply_probability=0.2,
            )
        )
    if regions is not None:
        region_values = list(regions)
        if ignore_label is not None:
            region_values.append(ignore_label)
        transforms.append(
            ConvertSegmentationToRegionsTransform(
                regions=region_values, channel_in_seg=0
            )
        )
    if deep_supervision_scales is not None:
        transforms.append(
            DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
        )
    return ComposeTransforms(transforms), resolved


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _stable_runtime_value(value: object, seen: set[int] | None = None) -> object:
    """Describe runtime transform configuration without unstable repr addresses."""
    if seen is None:
        seen = set()
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_runtime_value(item, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_runtime_value(item, seen) for item in value]

    identity = id(value)
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if identity in seen:
        return {"type": type_name, "recursive_reference": True}
    seen.add(identity)
    try:
        attributes = getattr(value, "__dict__", None)
        if attributes is None:
            return {"type": type_name}
        config = {
            key: _stable_runtime_value(item, seen)
            for key, item in sorted(attributes.items())
            if not key.startswith("_")
        }
        return {"type": type_name, "config": config}
    finally:
        seen.remove(identity)


def _persist_provenance(
    trainer: "CellVectorNnUNetTrainer",
    artifact: _VerifiedArtifact,
    pipeline: BasicTransform,
    resolved_transforms: Sequence[Mapping[str, object]],
) -> None:
    _atomic_write_json(
        Path(trainer.output_folder) / PROVENANCE_FILENAME,
        {
            "schema_version": "1.0.0",
            "profile": artifact.profile.model_dump(mode="json"),
            "profile_identity_sha256": artifact.profile.identity_hash(),
            "profile_file_sha256": artifact.file_sha256,
            "profile_source_path": str(artifact.path) if artifact.path is not None else None,
            "nnunetv2_version": version("nnunetv2"),
            "batchgeneratorsv2_version": version("batchgeneratorsv2"),
            "resolved_transforms": list(resolved_transforms),
            "resolved_pipeline": _stable_runtime_value(pipeline),
        },
    )


class CellVectorNnUNetTrainer(nnUNetTrainer):
    """Narrow nnU-Net v2 trainer with a verified CellVector augmentation boundary."""

    def get_training_transforms(
        self,
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ) -> BasicTransform:
        artifact = _load_verified_artifact()
        if artifact.profile.implementation == NNUNET_DEFAULT_IMPLEMENTATION:
            pipeline = nnUNetTrainer.get_training_transforms(
                patch_size,
                rotation_for_DA,
                deep_supervision_scales,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm=use_mask_for_norm,
                is_cascaded=is_cascaded,
                foreground_labels=foreground_labels,
                regions=regions,
                ignore_label=ignore_label,
            )
            _persist_provenance(self, artifact, pipeline, ())
            return pipeline

        pipeline, resolved = _construct_controlled_pipeline(
            artifact.profile,
            patch_size=patch_size,
            deep_supervision_scales=deep_supervision_scales,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        _persist_provenance(self, artifact, pipeline, resolved)
        return pipeline


__all__ = ["CellVectorNnUNetTrainer", "load_verified_augmentation_profile"]
