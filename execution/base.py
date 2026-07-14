"""Execution contracts shared by local, remote, CLI, and UI callers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cellvector.domain.models import AnnotationDocument, PredictionProposal


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendHealth(ContractModel):
    status: Literal["available", "offline"]
    detail: str | None = None
    server_instance_id: str | None = None


class PredictRequest(ContractModel):
    image_path: Path
    method: Literal["fiji_reconstruction", "hessian_vector"] = "hessian_vector"
    frame_index: int = Field(default=0, ge=0)


class PredictResult(ContractModel):
    status: Literal["succeeded"] = "succeeded"
    document: AnnotationDocument
    proposal: PredictionProposal
    warnings: list[str] = Field(default_factory=list)


class ExecutionBackend(Protocol):
    def health(self) -> BackendHealth:
        raise NotImplementedError

    def predict(self, request: PredictRequest) -> PredictResult:
        raise NotImplementedError

