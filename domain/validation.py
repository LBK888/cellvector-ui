"""Annotation validation that reports stable, UI-safe error codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from shapely import LineString, Point as ShapelyPoint, Polygon

from .geometry import GeometryError, derive_cell_polygon
from .models import AnnotationDocument, CellFeature, Point


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Literal["error", "warning"]
    feature_id: UUID | None = None


@dataclass
class QCReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_codes(self) -> list[str]:
        return [issue.code for issue in self.errors]

    @property
    def warning_codes(self) -> list[str]:
        return [issue.code for issue in self.warnings]

    def add_error(self, code: str, message: str, feature_id: UUID | None = None) -> None:
        self.errors.append(ValidationIssue(code, message, "error", feature_id))

    def add_warning(self, code: str, message: str, feature_id: UUID | None = None) -> None:
        self.warnings.append(ValidationIssue(code, message, "warning", feature_id))


class QCError(ValueError):
    def __init__(self, report: QCReport) -> None:
        self.report = report
        super().__init__(", ".join(report.error_codes))


def validate_document(document: AnnotationDocument) -> QCReport:
    report = QCReport()
    _check_unique_ids(document, report)

    cell_polygons: dict[UUID, Polygon] = {}
    for cell in document.cells:
        _check_points_in_bounds(cell.contour, document, report, cell.id)
        if not cell.partial:
            polygon = Polygon([(point.x, point.y) for point in cell.contour])
            if not polygon.is_valid:
                report.add_error(
                    "CELL_SELF_INTERSECTION",
                    "Complete cell contour is self-intersecting or otherwise invalid.",
                    cell.id,
                )
            elif polygon.area <= 0:
                report.add_error("CELL_ZERO_AREA", "Complete cell contour has zero area.", cell.id)
            else:
                cell_polygons[cell.id] = polygon

    _validate_shared_boundaries(document, report, cell_polygons)

    for boundary in document.boundaries:
        _check_points_in_bounds(boundary.points, document, report, boundary.id)

    for ridge in document.microridges:
        _check_points_in_bounds(ridge.points, document, report, ridge.id)
        if LineString([(point.x, point.y) for point in ridge.points]).length < 1.0:
            report.add_warning("MICRORIDGE_TOO_SHORT", "Microridge is shorter than one pixel.", ridge.id)
        if ridge.cell_id is not None and ridge.cell_id in cell_polygons:
            polygon = cell_polygons[ridge.cell_id]
            if not all(polygon.covers(ShapelyPoint(point.x, point.y)) for point in ridge.points):
                report.add_error(
                    "MICRORIDGE_OUTSIDE_CELL",
                    "Microridge contains points outside its assigned cell.",
                    ridge.id,
                )

    return report


def _check_unique_ids(document: AnnotationDocument, report: QCReport) -> None:
    identifiers = [
        *(feature.id for feature in document.cells),
        *(feature.id for feature in document.boundaries),
        *(feature.id for feature in document.microridges),
        *(feature.id for feature in document.proposals),
    ]
    if len(identifiers) != len(set(identifiers)):
        report.add_error("DUPLICATE_FEATURE_ID", "Feature identifiers must be unique.")


def _check_points_in_bounds(
    points: list[Point],
    document: AnnotationDocument,
    report: QCReport,
    feature_id: UUID,
) -> None:
    if any(
        point.x < 0
        or point.y < 0
        or point.x > document.source.width_px - 1
        or point.y > document.source.height_px - 1
        for point in points
    ):
        report.add_error(
            "COORDINATE_OUT_OF_BOUNDS",
            "Feature coordinate lies outside the source image.",
            feature_id,
        )


def _validate_shared_boundaries(
    document: AnnotationDocument,
    report: QCReport,
    cell_polygons: dict[UUID, Polygon],
) -> None:
    referenced_ids = {
        cell_id
        for segment in document.boundaries
        for cell_id in (segment.left_cell_id, segment.right_cell_id)
        if cell_id is not None
    }
    declared = {cell.id: cell for cell in document.cells}
    for cell_id in referenced_ids:
        if cell_id not in declared:
            report.add_error(
                "BOUNDARY_UNKNOWN_CELL",
                "Boundary refers to an undeclared cell.",
                cell_id,
            )
            continue
        if declared[cell_id].partial:
            continue
        try:
            cell_polygons[cell_id] = derive_cell_polygon(cell_id, document.boundaries)
        except GeometryError as error:
            report.add_error(error.code, "Shared boundary graph does not form one valid cell.", cell_id)

