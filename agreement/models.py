from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AgreementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgreementPair(AgreementModel):
    left_annotator: str
    right_annotator: str
    left_revision_id: UUID
    right_revision_id: UUID
    metrics: dict[str, float]
    feature_counts: dict[str, int]


class AgreementReport(AgreementModel):
    schema_version: str = "1.0.0"
    report_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int
    width_px: int
    height_px: int
    raster_policy: str = "exclusive-v1"
    pairs: tuple[AgreementPair, ...]
    aggregate: dict[str, float]
    software_fixture: bool = False
    created_at: datetime
