from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InferenceProfile(str, Enum):
    PREVIEW = "preview"
    STANDARD = "standard"
    ACCURACY = "accuracy"


class RunState(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRecord(ContractModel):
    argv: tuple[str, ...]
    state: RunState
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_code: str | None = None
    artifact_sha256: dict[str, str] = Field(default_factory=dict)


class ExperimentSpec(ContractModel):
    dataset_id: int = Field(ge=1, le=999)
    architecture: Literal["plainconv", "resenc_l"]
    configuration: Literal["2d"] = "2d"
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)

    @classmethod
    def plainconv(cls, dataset_id: int) -> "ExperimentSpec":
        return cls(dataset_id=dataset_id, architecture="plainconv")

    @classmethod
    def resenc_l(cls, dataset_id: int) -> "ExperimentSpec":
        return cls(dataset_id=dataset_id, architecture="resenc_l")

    @property
    def planner(self) -> str | None:
        return "nnUNetPlannerResEncL" if self.architecture == "resenc_l" else None

    @property
    def plans(self) -> str | None:
        return "nnUNetResEncUNetLPlans" if self.architecture == "resenc_l" else None
