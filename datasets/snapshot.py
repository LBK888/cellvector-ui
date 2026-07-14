from __future__ import annotations

from cellvector.domain.models import AnnotationDocument
from cellvector.domain.validation import validate_document

from .models import DatasetError, DatasetSample


def sample_from_document(
    document: AnnotationDocument,
    *,
    reviewed: bool,
    specimen_id: str | None = None,
    stack_group_id: str | None = None,
    sample_id: str | None = None,
) -> DatasetSample:
    """Create a training candidate tied to one exact annotation revision."""

    if not reviewed:
        raise DatasetError(
            "DATASET_NOT_REVIEWED",
            f"revision {document.revision_id} is not marked review-complete",
        )
    validation = validate_document(document)
    qc_errors = tuple(dict.fromkeys([*document.qc.errors, *validation.error_codes]))
    if qc_errors:
        raise DatasetError(
            "DATASET_QC_FAILED",
            f"revision {document.revision_id} has blocking QC errors: {', '.join(qc_errors)}",
        )
    specimen = specimen_id.strip() if specimen_id else None
    stack = stack_group_id.strip() if stack_group_id else None
    if specimen:
        group_key = f"specimen:{specimen}"
    elif stack:
        group_key = f"stack:{stack}"
    else:
        group_key = f"source:{document.source.sha256}"
    case_id = sample_id or (
        f"cv_{document.source.sha256[:12]}_{document.source.frame_index:04d}"
    )
    return DatasetSample(
        sample_id=case_id,
        annotation_id=document.annotation_id,
        revision_id=document.revision_id,
        source_sha256=document.source.sha256,
        image_uri=document.source.image_uri,
        frame_index=document.source.frame_index,
        width_px=document.source.width_px,
        height_px=document.source.height_px,
        dtype=document.source.dtype,
        pixel_size_um=document.source.pixel_size_um,
        specimen_id=specimen,
        stack_group_id=stack,
        group_key=group_key,
        reviewed=True,
        qc_errors=qc_errors,
    )
