from __future__ import annotations

from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours, label as connected_components
from skimage.morphology import skeletonize

from cellvector.domain.models import (
    AnnotationDocument,
    BoundarySegment,
    CellFeature,
    MicroridgeFeature,
    Point,
    PredictionProposal,
    Provenance,
    ReviewStatus,
)
from cellvector.vectorize.skeleton import trace_skeleton
from cellvector.datasets.models import DatasetError

from .contracts import InferenceProfile
from .registry import ModelRecord, ModelState


def proposal_from_label_map(
    document: AnnotationDocument,
    labels: NDArray[np.uint8],
    *,
    model: ModelRecord,
    profile: InferenceProfile,
) -> PredictionProposal:
    """Convert an nnU-Net class map without mutating the final annotation."""

    if profile == InferenceProfile.ACCURACY and model.state != ModelState.PROMOTED:
        raise DatasetError(
            "MODEL_NOT_PROMOTED",
            "the accuracy profile requires an explicitly promoted model",
        )
    expected_shape = (document.source.height_px, document.source.width_px)
    if labels.shape != expected_shape:
        raise ValueError(f"prediction shape {labels.shape} differs from {expected_shape}")
    unknown = set(np.unique(labels)).difference({0, 1, 2, 3})
    if unknown:
        raise ValueError(f"prediction contains unknown labels: {sorted(unknown)}")
    provenance = Provenance(
        method="ai_nnunet",
        implementation_version="wp2-baseline-v1",
        model_version=str(model.model_id),
        parameters={
            "snapshot_hash": model.snapshot_hash,
            "checkpoint_sha256": model.checkpoint_sha256,
            "profile": profile.value,
            "software_smoke_test": model.software_smoke_test,
        },
    )

    cells: list[CellFeature] = []
    cell_components = connected_components(np.isin(labels, (1, 3)), connectivity=1)
    for component_id in range(1, int(cell_components.max()) + 1):
        component = cell_components == component_id
        contours = find_contours(component.astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        points = [Point(x=float(column), y=float(row)) for row, column in contour]
        touches_edge = bool(
            component[0].any()
            or component[-1].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_edge and points[0] != points[-1]:
            points.append(points[0])
        if len(points) < 2:
            continue
        cells.append(
            CellFeature(
                id=uuid4(),
                contour=points,
                partial=touches_edge,
                review_status=ReviewStatus.UNREVIEWED,
                provenance=provenance,
            )
        )

    boundaries = [
        BoundarySegment(
            id=uuid4(),
            points=path,
            review_status=ReviewStatus.UNREVIEWED,
            provenance=provenance,
        )
        for path in trace_skeleton(skeletonize(labels == 2))
    ]
    microridges = [
        MicroridgeFeature(
            id=uuid4(),
            points=path,
            closed=path[0] == path[-1],
            review_status=ReviewStatus.UNREVIEWED,
            provenance=provenance,
        )
        for path in trace_skeleton(skeletonize(labels == 3))
    ]
    return PredictionProposal(
        id=uuid4(),
        source_sha256=document.source.sha256,
        provenance=provenance,
        cells=cells,
        boundaries=boundaries,
        microridges=microridges,
        warnings=(
            ["software smoke-test model; not biologically validated"]
            if model.software_smoke_test
            else []
        ),
    )
