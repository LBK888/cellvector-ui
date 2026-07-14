from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isclose
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueueState(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class QueueOrigin(StrEnum):
    MANUAL = "manual"
    AGREEMENT = "agreement"
    REGRESSION = "regression"
    MODEL_UNCERTAINTY = "model_uncertainty"


class QueuePriority(QueueModel):
    components: dict[str, float]
    total: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def components_are_normalized(self):
        if any(value < 0 or value > 1 for value in self.components.values()):
            raise ValueError("priority components must be in [0, 1]")
        return self


class QueuePriorityPolicy(QueueModel):
    schema_version: str = "1.0.0"
    weights: dict[str, float]

    @model_validator(mode="after")
    def weights_are_normalized(self):
        if not self.weights or any(value < 0 or value > 1 for value in self.weights.values()):
            raise ValueError("priority weights must be in [0, 1]")
        if not isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("priority weights must sum to 1.0")
        return self

    @classmethod
    def manual_only(cls) -> "QueuePriorityPolicy":
        return cls(weights={"manual": 1.0})

    def score(self, components: dict[str, float]) -> QueuePriority:
        total = round(
            sum(self.weights.get(name, 0.0) * value for name, value in components.items()),
            12,
        )
        return QueuePriority(components=dict(sorted(components.items())), total=total)


class QueueAuditEvent(QueueModel):
    previous_state: QueueState | None
    new_state: QueueState
    actor: str
    note: str = ""
    timestamp: datetime
    result_revision_id: UUID | None = None


class ReviewQueueItem(QueueModel):
    schema_version: str = "1.0.0"
    item_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0)
    origin: QueueOrigin
    priority: QueuePriority
    reasons: tuple[str, ...]
    note: str = ""
    annotation_id: UUID | None = None
    revision_id: UUID | None = None
    proposal_id: UUID | None = None
    model_id: UUID | None = None
    snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state: QueueState = QueueState.QUEUED
    claimed_by: str | None = None
    result_revision_id: UUID | None = None
    created_at: datetime
    audit: tuple[QueueAuditEvent, ...]
