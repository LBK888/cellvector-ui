from __future__ import annotations

from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


SUPPORTED_TRANSFORM_NAMES = frozenset(
    {
        "rotation",
        "horizontal_flip",
        "vertical_flip",
        "scale",
        "stretch",
        "crop",
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "contrast",
        "gamma",
        "low_resolution",
        "block_occlusion",
    }
)


class AugmentationError(RuntimeError):
    """An augmentation failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class AugmentationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a fully revalidated copy (Pydantic's default skips validation)."""
        del deep
        payload = self.model_dump(mode="python", round_trip=True)
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _numeric_range(
    transform_name: str,
    parameter_name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = False,
) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_INVALID",
            f"{transform_name}.{parameter_name} must be a two-number range",
        )
    low, high = value
    if not _is_number(low) or not _is_number(high):
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_INVALID",
            f"{transform_name}.{parameter_name} must contain finite numbers",
        )
    result = (float(low), float(high))
    if result[0] > result[1]:
        raise AugmentationError(
            "AUGMENTATION_RANGE_INVALID",
            f"{transform_name}.{parameter_name} range is inverted",
        )
    if minimum is not None:
        invalid_low = result[0] < minimum if minimum_inclusive else result[0] <= minimum
        if invalid_low:
            operator = ">=" if minimum_inclusive else ">"
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                f"{transform_name}.{parameter_name} values must be {operator} {minimum}",
            )
    if maximum is not None and result[1] > maximum:
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_INVALID",
            f"{transform_name}.{parameter_name} values must be <= {maximum}",
        )
    return result


_PARAMETER_KEYS: dict[str, frozenset[str]] = {
    "rotation": frozenset({"degrees"}),
    "horizontal_flip": frozenset(),
    "vertical_flip": frozenset(),
    "scale": frozenset({"factor"}),
    "stretch": frozenset({"x_factor", "y_factor"}),
    "crop": frozenset({"fraction"}),
    "gaussian_noise": frozenset({"sigma"}),
    "gaussian_blur": frozenset({"sigma"}),
    "brightness": frozenset({"factor"}),
    "contrast": frozenset({"factor"}),
    "gamma": frozenset({"gamma"}),
    "low_resolution": frozenset({"scale"}),
    "block_occlusion": frozenset({"height", "width", "count", "fill"}),
}

_REQUIRED_PARAMETER_KEYS = {
    name: keys for name, keys in _PARAMETER_KEYS.items() if name != "crop"
}
_REQUIRED_PARAMETER_KEYS["crop"] = frozenset()


def _validated_parameters(name: str, value: object) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_INVALID", f"{name}.parameters must be a mapping"
        )
    supplied = set(value)
    allowed = _PARAMETER_KEYS[name]
    unknown = supplied - allowed
    if unknown:
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_UNSUPPORTED",
            f"{name} does not support parameters: {', '.join(sorted(unknown))}",
        )

    normalized: dict[str, object] = {}
    # Operational bounds are part of schema 1.0.0. They cover the finite,
    # numerically stable ranges shared by the NumPy preview and the installed
    # batchgeneratorsv2 transforms.
    range_rules = {
        "rotation": ("degrees", -180.0, 180.0, True),
        "scale": ("factor", 0.5, 2.0, True),
        "crop": ("fraction", 0.5, 1.0, True),
        "gaussian_noise": ("sigma", 0.0, 1.0, True),
        "gaussian_blur": ("sigma", 0.1, 5.0, True),
        "brightness": ("factor", 0.25, 4.0, True),
        "contrast": ("factor", 0.25, 4.0, True),
        "gamma": ("gamma", 0.25, 4.0, True),
        "low_resolution": ("scale", 0.25, 1.0, True),
    }
    if name in range_rules and range_rules[name][0] in value:
        key, minimum, maximum, inclusive = range_rules[name]
        normalized[key] = _numeric_range(
            name,
            key,
            value[key],
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=inclusive,
        )
    elif name == "stretch":
        for key in ("x_factor", "y_factor"):
            if key in value:
                normalized[key] = _numeric_range(
                    name,
                    key,
                    value[key],
                    minimum=0.5,
                    maximum=2.0,
                    minimum_inclusive=True,
                )
    elif name == "block_occlusion":
        for key in ("height", "width", "count"):
            if key in value:
                item = value[key]
                maximum = 32 if key == "count" else 1024
                if type(item) is not int or item <= 0 or item > maximum:
                    raise AugmentationError(
                        "AUGMENTATION_PARAMETER_INVALID",
                        f"{name}.{key} must be an integer within [1, {maximum}]",
                    )
                normalized[key] = item
        if "fill" in value:
            fill = value["fill"]
            if not _is_number(fill) or abs(float(fill)) > 1_000_000.0:
                raise AugmentationError(
                    "AUGMENTATION_PARAMETER_INVALID",
                    f"{name}.fill must be finite and within [-1000000, 1000000]",
                )
            normalized["fill"] = float(fill)

        if {"height", "width", "count"}.issubset(normalized):
            requested_area = (
                int(normalized["height"])
                * int(normalized["width"])
                * int(normalized["count"])
            )
            if requested_area > 1_048_576:
                raise AugmentationError(
                    "AUGMENTATION_PARAMETER_INVALID",
                    "block_occlusion total requested area exceeds 1048576 pixels",
                )

    if name == "crop" and "fraction" not in normalized:
        normalized["fraction"] = (0.75, 0.95)

    missing = _REQUIRED_PARAMETER_KEYS[name] - supplied
    if missing:
        raise AugmentationError(
            "AUGMENTATION_PARAMETER_MISSING",
            f"{name} requires parameters: {', '.join(sorted(missing))}",
        )
    return MappingProxyType(normalized)


class TransformSpec(AugmentationModel):
    name: str
    probability: float
    parameters: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        if type(value) is not str or value not in SUPPORTED_TRANSFORM_NAMES:
            raise AugmentationError(
                "AUGMENTATION_UNSUPPORTED", f"unsupported transform: {value!r}"
            )
        return value

    @field_validator("probability", mode="before")
    @classmethod
    def _validate_probability(cls, value: object) -> float:
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            raise AugmentationError(
                "AUGMENTATION_PROBABILITY_INVALID",
                "transform probability must be finite and within [0, 1]",
            )
        return float(value)

    @field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls, value: object, info) -> MappingProxyType:
        name = info.data.get("name")
        if name not in SUPPORTED_TRANSFORM_NAMES:
            raise AugmentationError(
                "AUGMENTATION_UNSUPPORTED", f"unsupported transform: {name!r}"
            )
        return _validated_parameters(name, value)

    @field_validator("parameters", mode="after")
    @classmethod
    def _freeze_parameters(cls, value: Mapping[str, object]) -> MappingProxyType:
        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)


class AugmentationProfile(AugmentationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    profile_id: UUID
    name: str
    implementation: str
    implementation_version: str
    seed_policy: Literal["training_seed", "explicit_preview"]
    enabled: bool = True
    image_interpolation: Literal["linear"] = "linear"
    label_interpolation: Literal["nearest"] = "nearest"
    transforms: tuple[TransformSpec, ...]

    @field_validator("profile_id", mode="before")
    @classmethod
    def _parse_profile_id(cls, value: object) -> UUID:
        if isinstance(value, UUID):
            return value
        if type(value) is str:
            try:
                return UUID(value)
            except ValueError as error:
                raise AugmentationError(
                    "AUGMENTATION_PROFILE_INVALID", "profile_id is not a UUID"
                ) from error
        raise AugmentationError(
            "AUGMENTATION_PROFILE_INVALID", "profile_id is not a UUID"
        )

    @field_validator("name", "implementation", "implementation_version", mode="before")
    @classmethod
    def _nonblank_text(cls, value: object, info) -> str:
        if type(value) is not str or not value.strip():
            raise AugmentationError(
                "AUGMENTATION_PROFILE_INVALID", f"{info.field_name} must be nonblank"
            )
        return value.strip()

    @field_validator("transforms", mode="before")
    @classmethod
    def _tuple_transforms(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (tuple, list)):
            raise AugmentationError(
                "AUGMENTATION_PROFILE_INVALID", "transforms must be an ordered sequence"
            )
        return tuple(value)

    @field_validator("enabled", mode="before")
    @classmethod
    def _strict_enabled(cls, value: object) -> bool:
        if type(value) is not bool:
            raise AugmentationError(
                "AUGMENTATION_PROFILE_INVALID", "enabled must be a strict boolean"
            )
        return value

    @field_validator("image_interpolation", mode="before")
    @classmethod
    def _supported_image_interpolation(cls, value: object) -> str:
        if value != "linear":
            raise AugmentationError(
                "AUGMENTATION_UNSUPPORTED",
                "only linear grayscale image interpolation is supported",
            )
        return "linear"

    @field_validator("label_interpolation", mode="before")
    @classmethod
    def _supported_label_interpolation(cls, value: object) -> str:
        if value != "nearest":
            raise AugmentationError(
                "AUGMENTATION_UNSUPPORTED",
                "label interpolation must be nearest",
            )
        return "nearest"

    @model_validator(mode="after")
    def _unique_transform_controls(self) -> Self:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for transform in self.transforms:
            if transform.name in seen:
                duplicates.add(transform.name)
            seen.add(transform.name)
        if duplicates:
            raise AugmentationError(
                "AUGMENTATION_TRANSFORM_DUPLICATE",
                "profile contains duplicate controls: "
                + ", ".join(sorted(duplicates)),
            )
        return self

    def canonical_json(self, *, exclude_display_identity: bool = False) -> str:
        exclude = {"profile_id", "name"} if exclude_display_identity else None
        payload = self.model_dump(mode="json", exclude=exclude)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def identity_hash(self) -> str:
        canonical = self.canonical_json(exclude_display_identity=True)
        return sha256(canonical.encode("utf-8")).hexdigest()


def _freeze_resolved(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_resolved(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_resolved(item) for item in value)
    return value


def _thaw_resolved(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_resolved(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_resolved(item) for item in value]
    return value


class ResolvedTransform(AugmentationModel):
    name: str
    applied: bool
    sampled_parameters: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("sampled_parameters", mode="before")
    @classmethod
    def _immutable_parameters(cls, value: object) -> MappingProxyType:
        if not isinstance(value, Mapping):
            raise AugmentationError(
                "AUGMENTATION_PARAMETER_INVALID",
                "resolved transform parameters must be a mapping",
            )
        return _freeze_resolved(value)  # type: ignore[return-value]

    @field_validator("sampled_parameters", mode="after")
    @classmethod
    def _freeze_sampled_parameters(
        cls, value: Mapping[str, object]
    ) -> MappingProxyType:
        return _freeze_resolved(value)  # type: ignore[return-value]

    @field_serializer("sampled_parameters")
    def _serialize_sampled(self, value: Mapping[str, object]) -> dict[str, object]:
        return _thaw_resolved(value)  # type: ignore[return-value]


class AugmentationPreviewRecord(AugmentationModel):
    profile_hash: str
    seed: int
    transforms: tuple[ResolvedTransform, ...]
