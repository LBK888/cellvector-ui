"""Versioned annotation and prediction contracts.

All geometry is stored in source-image pixel coordinates. Physical pixel size is
optional metadata and never changes the stored coordinate system.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float

    @model_validator(mode="after")
    def coordinates_are_finite(self) -> Point:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("point coordinates must be finite")
        return self


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    AMBIGUOUS = "ambiguous"


ProvenanceMethod = Literal[
    "manual",
    "classical_fiji_reconstruction",
    "classical_hessian",
    "ai_nnunet",
    "derived",
]


class Provenance(StrictModel):
    method: ProvenanceMethod
    implementation_version: str | None = None
    model_version: str | None = None
    source_proposal_id: UUID | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class SourceImage(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    coordinate_unit: Literal["pixel"] = "pixel"
    pixel_size_um: tuple[PositiveFloat, PositiveFloat] | None = None


class CellFeature(StrictModel):
    id: UUID
    contour: list[Point] = Field(min_length=2)
    partial: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    provenance: Provenance

    @model_validator(mode="after")
    def complete_cells_are_closed(self) -> CellFeature:
        if not self.partial:
            if len(self.contour) < 4 or self.contour[0] != self.contour[-1]:
                raise ValueError("a complete cell contour must be a closed ring")
        return self


class BoundarySegment(StrictModel):
    id: UUID
    points: list[Point] = Field(min_length=2)
    left_cell_id: UUID | None = None
    right_cell_id: UUID | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    provenance: Provenance

    @model_validator(mode="after")
    def boundary_has_valid_sides(self) -> BoundarySegment:
        if self.left_cell_id is not None and self.left_cell_id == self.right_cell_id:
            raise ValueError("left and right cell identifiers must differ")
        _require_two_distinct_points(self.points)
        return self


class MicroridgeFeature(StrictModel):
    id: UUID
    points: list[Point] = Field(min_length=2)
    cell_id: UUID | None = None
    closed: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    provenance: Provenance

    @model_validator(mode="after")
    def microridge_has_distinct_points(self) -> MicroridgeFeature:
        _require_two_distinct_points(self.points)
        if self.closed and self.points[0] != self.points[-1]:
            raise ValueError("a closed microridge must repeat its first point")
        return self


class PredictionProposal(StrictModel):
    id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Provenance
    cells: list[CellFeature] = Field(default_factory=list)
    boundaries: list[BoundarySegment] = Field(default_factory=list)
    microridges: list[MicroridgeFeature] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReviewRecord(StrictModel):
    author: str = Field(min_length=1)
    timestamp: datetime
    edit_summary: str = Field(min_length=1)


class QCRecord(StrictModel):
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnnotationDocument(StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    annotation_id: UUID
    revision_id: UUID
    parent_revision_id: UUID | None = None
    source: SourceImage
    cells: list[CellFeature] = Field(default_factory=list)
    boundaries: list[BoundarySegment] = Field(default_factory=list)
    microridges: list[MicroridgeFeature] = Field(default_factory=list)
    proposals: list[PredictionProposal] = Field(default_factory=list)
    review: ReviewRecord
    qc: QCRecord = Field(default_factory=QCRecord)


def _require_two_distinct_points(points: list[Point]) -> None:
    if len({(point.x, point.y) for point in points}) < 2:
        raise ValueError("a path requires at least two distinct points")
