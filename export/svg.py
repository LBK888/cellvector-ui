"""SVG export from the annotation source of truth."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from cellvector.domain.models import AnnotationDocument, Point


def export_svg(document: AnnotationDocument, path: str | Path) -> Path:
    output = Path(path)
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {document.source.width_px} {document.source.height_px}",
            "data-coordinate-unit": "pixel",
        },
    )
    metadata = ET.SubElement(root, "metadata")
    metadata.text = json.dumps(
        {
            "annotation_id": str(document.annotation_id),
            "revision_id": str(document.revision_id),
            "source_sha256": document.source.sha256,
            "frame_index": document.source.frame_index,
            "coordinate_unit": document.source.coordinate_unit,
            "pixel_size_um": document.source.pixel_size_um,
        },
        sort_keys=True,
    )

    cells = ET.SubElement(root, "g", {"id": "cells"})
    for cell in document.cells:
        ET.SubElement(
            cells,
            "path",
            {
                "id": str(cell.id),
                "d": _path_data(cell.contour, close=not cell.partial),
                "fill": "none",
                "stroke": "#00bcd4",
                "data-partial": str(cell.partial).lower(),
                "data-provenance": cell.provenance.method,
            },
        )

    boundaries = ET.SubElement(root, "g", {"id": "boundaries"})
    for boundary in document.boundaries:
        ET.SubElement(
            boundaries,
            "path",
            {
                "id": str(boundary.id),
                "d": _path_data(boundary.points),
                "fill": "none",
                "stroke": "#00bcd4",
                "data-left-cell-id": str(boundary.left_cell_id or ""),
                "data-right-cell-id": str(boundary.right_cell_id or ""),
                "data-provenance": boundary.provenance.method,
            },
        )

    ridges = ET.SubElement(root, "g", {"id": "microridges"})
    for ridge in document.microridges:
        ET.SubElement(
            ridges,
            "path",
            {
                "id": str(ridge.id),
                "d": _path_data(ridge.points, close=ridge.closed),
                "fill": "none",
                "stroke": "#ff2d95",
                "data-cell-id": str(ridge.cell_id or ""),
                "data-provenance": ridge.provenance.method,
                "data-source-proposal-id": str(ridge.provenance.source_proposal_id or ""),
            },
        )

    ET.indent(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def _path_data(points: list[Point], *, close: bool = False) -> str:
    commands = [f"M {points[0].x:g} {points[0].y:g}"]
    commands.extend(f"L {point.x:g} {point.y:g}" for point in points[1:])
    if close and points[-1] != points[0]:
        commands.append("Z")
    return " ".join(commands)

