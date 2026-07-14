from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cellvector.backends.contracts import Sha256


_DEFAULT_AUGMENTATION_PROFILE_HASH = sha256(
    b"cellvector:augmentation-profile:default"
).hexdigest()


class TrainingProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
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


class DatasetPurpose(StrEnum):
    EXPLORATORY = "exploratory"
    COMPARATIVE = "comparative"


class FoldPolicy(TrainingProfileModel):
    mode: Literal["auto", "explicit"] = "auto"
    requested_folds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _mode_matches_requested_folds(self) -> FoldPolicy:
        if self.mode == "auto" and self.requested_folds is not None:
            raise ValueError("auto fold policy cannot request an explicit fold count")
        if self.mode == "explicit" and self.requested_folds is None:
            raise ValueError("explicit fold policy requires requested_folds")
        return self


class TrainingDatasetProfile(TrainingProfileModel):
    purpose: DatasetPurpose
    fold_policy: FoldPolicy
    augmentation_profile_hash: Sha256
    allow_synthetic_training: bool = False

    @classmethod
    def exploratory(
        cls,
        *,
        requested_folds: int | None = None,
        augmentation_profile_hash: Sha256 = _DEFAULT_AUGMENTATION_PROFILE_HASH,
        allow_synthetic_training: bool = False,
    ) -> TrainingDatasetProfile:
        return cls(
            purpose=DatasetPurpose.EXPLORATORY,
            fold_policy=FoldPolicy(
                mode="explicit" if requested_folds is not None else "auto",
                requested_folds=requested_folds,
            ),
            augmentation_profile_hash=augmentation_profile_hash,
            allow_synthetic_training=allow_synthetic_training,
        )

    @classmethod
    def comparative(
        cls,
        *,
        requested_folds: int | None = None,
        augmentation_profile_hash: Sha256 = _DEFAULT_AUGMENTATION_PROFILE_HASH,
        allow_synthetic_training: bool = False,
    ) -> TrainingDatasetProfile:
        return cls(
            purpose=DatasetPurpose.COMPARATIVE,
            fold_policy=FoldPolicy(
                mode="explicit" if requested_folds is not None else "auto",
                requested_folds=requested_folds,
            ),
            augmentation_profile_hash=augmentation_profile_hash,
            allow_synthetic_training=allow_synthetic_training,
        )
