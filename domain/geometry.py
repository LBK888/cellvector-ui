"""Topology helpers for cell contours and shared membrane boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from shapely import LineString, Polygon, get_parts
from shapely.ops import polygonize, unary_union

from .models import BoundarySegment, CellFeature, Point, Provenance


class GeometryError(ValueError):
    """Raised when boundary linework cannot form the requested geometry."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def derive_cell_polygon(
    cell_id: UUID,
    segments: Sequence[BoundarySegment],
) -> Polygon:
    """Derive one cell polygon from all shared segments touching ``cell_id``."""

    linework = [
        LineString([(point.x, point.y) for point in segment.points])
        for segment in segments
        if cell_id in (segment.left_cell_id, segment.right_cell_id)
    ]
    if not linework:
        raise GeometryError("CELL_BOUNDARY_MISSING")
    polygons = [polygon for polygon in polygonize(unary_union(linework)) if polygon.area > 0]
    if len(polygons) != 1:
        raise GeometryError("CELL_BOUNDARY_NOT_SINGLE_LOOP")
    polygon = polygons[0]
    if not polygon.is_valid:
        raise GeometryError("CELL_BOUNDARY_INVALID")
    return polygon


def contour_polygon(points: Sequence[tuple[float, float]]) -> Polygon:
    """Create a Shapely polygon without repairing invalid annotation input."""

    return Polygon(points)


def build_shared_boundaries(
    cells: Sequence[CellFeature],
    *,
    tolerance: float = 1e-7,
) -> list[BoundarySegment]:
    """Node complete cell contours and attach each unique edge to its cells."""

    complete = [cell for cell in cells if not cell.partial]
    polygons = {
        cell.id: Polygon([(point.x, point.y) for point in cell.contour])
        for cell in complete
    }
    if any(not polygon.is_valid or polygon.area <= 0 for polygon in polygons.values()):
        raise GeometryError("CELL_BOUNDARY_INVALID")
    if not polygons:
        return []

    noded = unary_union([polygon.boundary for polygon in polygons.values()])
    boundaries: list[BoundarySegment] = []
    for line in get_parts(noded):
        if line.geom_type != "LineString" or line.length <= tolerance:
            continue
        touching = sorted(
            (
                cell_id
                for cell_id, polygon in polygons.items()
                if polygon.boundary.intersection(line).length
                >= line.length - tolerance
            ),
            key=str,
        )
        if not touching:
            continue
        if len(touching) > 2:
            raise GeometryError("BOUNDARY_MORE_THAN_TWO_CELLS")
        coordinates = list(line.coords)
        boundaries.append(
            BoundarySegment(
                id=uuid4(),
                points=[Point(x=x, y=y) for x, y in coordinates],
                left_cell_id=touching[0],
                right_cell_id=touching[1] if len(touching) == 2 else None,
                provenance=Provenance(
                    method="derived",
                    parameters={"source": "cell_contours"},
                ),
            )
        )
    return boundaries
