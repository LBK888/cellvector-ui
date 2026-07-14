"""Derived membrane and microridge raster masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.draw import line
from skimage.morphology import dilation, disk

from cellvector.domain.models import AnnotationDocument, Point


def rasterize_annotation(
    document: AnnotationDocument,
    *,
    line_width_px: int = 1,
) -> NDArray[np.uint8]:
    """Return ``[membrane, microridge]`` binary channels in source pixels."""

    height = document.source.height_px
    width = document.source.width_px
    result = np.zeros((2, height, width), dtype=np.uint8)

    for cell in document.cells:
        _draw_polyline(result[0], cell.contour)
    for boundary in document.boundaries:
        _draw_polyline(result[0], boundary.points)
    for ridge in document.microridges:
        _draw_polyline(result[1], ridge.points)

    if line_width_px > 1:
        radius = max(1, line_width_px // 2)
        footprint = disk(radius)
        result[0] = dilation(result[0], footprint).astype(np.uint8)
        result[1] = dilation(result[1], footprint).astype(np.uint8)
    return result


def _draw_polyline(canvas: NDArray[np.uint8], points: list[Point]) -> None:
    for first, second in zip(points, points[1:]):
        rows, columns = line(
            int(round(first.y)),
            int(round(first.x)),
            int(round(second.y)),
            int(round(second.x)),
        )
        valid = (
            (rows >= 0)
            & (rows < canvas.shape[0])
            & (columns >= 0)
            & (columns < canvas.shape[1])
        )
        canvas[rows[valid], columns[valid]] = 1
