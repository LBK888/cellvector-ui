from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DatasetError(RuntimeError):
    """A dataset failure with an API-stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    FROZEN_TEST = "frozen_test"


class SampleSourceKind(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class DatasetModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        payload = self.model_dump()
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


class DatasetSample(DatasetModel):
    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    annotation_id: UUID
    revision_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_uri: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    coordinate_unit: Literal["pixel"] = "pixel"
    pixel_size_um: tuple[float, float] | None = None
    specimen_id: str | None = None
    stack_group_id: str | None = None
    group_key: str = Field(min_length=1)
    split: SplitName | None = None
    reviewed: bool
    qc_errors: tuple[str, ...] = ()
    source_kind: SampleSourceKind = Field(
        default=SampleSourceKind.REAL,
        exclude_if=lambda value: value is SampleSourceKind.REAL,
    )


class DatasetSnapshot(DatasetModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    samples: tuple[DatasetSample, ...]
    seed: int
    created_at: datetime
    created_by: str = Field(min_length=1)
    output_path: str | None = None
    software_smoke_test: bool = False
    label_policy_version: str = "exclusive-v1"

    def identity_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "output_path"},
        )
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()
