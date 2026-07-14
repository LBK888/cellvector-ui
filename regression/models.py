from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RegressionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateKind(StrEnum):
    SOFTWARE = "software"
    SCIENTIFIC = "scientific"


class Comparison(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EQUAL = "equal"


class EvaluationState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUABLE = "not_evaluable"


class RegressionGate(RegressionModel):
    name: str = Field(min_length=1)
    kind: GateKind
    metric: str = Field(min_length=1)
    comparison: Comparison
    threshold: float


class RegressionPolicy(RegressionModel):
    schema_version: str = "1.0.0"
    policy_id: UUID
    name: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: UUID
    gates: tuple[RegressionGate, ...]


class RegressionEvidence(RegressionModel):
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_id: UUID
    model_id: UUID
    artifact_sha256: dict[str, Sha256] = Field(min_length=1)
    frozen_test: bool
    software_smoke_test: bool


class GateResult(RegressionModel):
    name: str
    state: EvaluationState
    actual: float | None = None
    threshold: float
    reason_code: str | None = None


class RegressionEvaluation(RegressionModel):
    schema_version: str = "1.0.0"
    evaluation_id: UUID
    policy_id: UUID
    state: EvaluationState
    gates: tuple[GateResult, ...]
    created_at: datetime
