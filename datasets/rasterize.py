from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.draw import line, polygon
from skimage.morphology import dilation, disk

from cellvector.domain.models import AnnotationDocument, Point

from .models import DatasetError


def _draw_polyline(
    shape: tuple[int, int],
    points: list[Point],
    width_px: int,
) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    for start, end in zip(points, points[1:]):
        rows, columns = line(
            round(start.y), round(start.x), round(end.y), round(end.x)
        )
        valid = (
            (rows >= 0)
            & (rows < shape[0])
            & (columns >= 0)
            & (columns < shape[1])
        )
        mask[rows[valid], columns[valid]] = True
    radius = max(0, (width_px - 1) // 2)
    return dilation(mask, disk(radius)) if radius else mask


def rasterize_nnunet_labels(
    document: AnnotationDocument,
    membrane_width_px: int = 1,
    microridge_width_px: int = 1,
    *,
    allow_empty: bool = False,
) -> NDArray[np.uint8]:
    """Derive the mutually exclusive baseline labels from vector truth."""

    if membrane_width_px < 1 or microridge_width_px < 1:
        raise DatasetError("INVALID_LABEL_POLICY", "line widths must be positive")
    shape = (document.source.height_px, document.source.width_px)
    paths = [cell.contour for cell in document.cells]
    paths.extend(boundary.points for boundary in document.boundaries)
    paths.extend(ridge.points for ridge in document.microridges)
    for points in paths:
        if any(
            point.x < 0
            or point.y < 0
            or point.x >= shape[1]
            or point.y >= shape[0]
            for point in points
        ):
            raise DatasetError(
                "GEOMETRY_OUT_OF_BOUNDS",
                "annotation geometry lies outside the source image",
            )
    labels = np.zeros(shape, dtype=np.uint8)
    for cell in document.cells:
        if len(cell.contour) < 3:
            continue
        rows = np.asarray([point.y for point in cell.contour], dtype=float)
        columns = np.asarray([point.x for point in cell.contour], dtype=float)
        rr, cc = polygon(rows, columns, shape=shape)
        labels[rr, cc] = 1
    for boundary in document.boundaries:
        labels[_draw_polyline(shape, boundary.points, membrane_width_px)] = 2
    for ridge in document.microridges:
        labels[_draw_polyline(shape, ridge.points, microridge_width_px)] = 3
    if not allow_empty and not np.any(labels):
        raise DatasetError("EMPTY_LABELS", "annotation produced no training labels")
    return labels
