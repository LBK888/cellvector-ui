from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from uuid import uuid4

import numpy as np

from cellvector.datasets.lineage import WP3Error
from cellvector.datasets.rasterize import rasterize_nnunet_labels
from cellvector.domain.models import AnnotationDocument
from cellvector.inference.nnunet.benchmark import benchmark_case

from .models import AgreementPair, AgreementReport


def _source_key(document: AnnotationDocument) -> tuple[object, ...]:
    source = document.source
    return (
        document.schema_version,
        source.sha256,
        source.frame_index,
        source.width_px,
        source.height_px,
    )


def compare_annotations(
    documents: list[AnnotationDocument],
    *,
    software_fixture: bool = False,
) -> AgreementReport:
    if len(documents) < 2:
        raise ValueError("at least two annotations are required")
    first_key = _source_key(documents[0])
    if any(_source_key(document) != first_key for document in documents[1:]):
        raise WP3Error(
            "AGREEMENT_SOURCE_MISMATCH",
            "annotations must identify the same source frame and dimensions",
        )
    ordered = sorted(
        documents,
        key=lambda document: (document.review.author, str(document.revision_id)),
    )
    labeled_documents = [
        (document, rasterize_nnunet_labels(document, allow_empty=True))
        for document in ordered
    ]
    pairs: list[AgreementPair] = []
    for (left, left_labels), (right, right_labels) in combinations(
        labeled_documents, 2
    ):
        metrics = benchmark_case(left_labels, right_labels)
        pairs.append(
            AgreementPair(
                left_annotator=left.review.author,
                right_annotator=right.review.author,
                left_revision_id=left.revision_id,
                right_revision_id=right.revision_id,
                metrics=metrics,
                feature_counts={
                    "left_cells": len(left.cells),
                    "right_cells": len(right.cells),
                    "left_boundaries": len(left.boundaries),
                    "right_boundaries": len(right.boundaries),
                    "left_microridges": len(left.microridges),
                    "right_microridges": len(right.microridges),
                },
            )
        )
    metric_names = pairs[0].metrics.keys()
    aggregate = {
        name: float(np.mean([pair.metrics[name] for pair in pairs]))
        for name in metric_names
    }
    source = ordered[0].source
    return AgreementReport(
        report_id=uuid4(),
        source_sha256=source.sha256,
        frame_index=source.frame_index,
        width_px=source.width_px,
        height_px=source.height_px,
        pairs=tuple(pairs),
        aggregate=aggregate,
        software_fixture=software_fixture,
        created_at=datetime.now(timezone.utc),
    )
