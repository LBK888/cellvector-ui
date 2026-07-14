"""In-process classical execution backend."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from cellvector.classical.fiji import FijiReconstructionPipeline
from cellvector.classical.hessian import HessianVectorPipeline
from cellvector.domain.models import AnnotationDocument, ReviewRecord
from cellvector.io.images import import_frames

from .base import BackendHealth, PredictRequest, PredictResult


class LocalExecutionBackend:
    def health(self) -> BackendHealth:
        return BackendHealth(status="available", detail="in-process classical analysis")

    def predict(self, request: PredictRequest) -> PredictResult:
        frames = import_frames(request.image_path)
        if request.frame_index >= len(frames):
            raise IndexError(
                f"frame index {request.frame_index} is outside a {len(frames)}-frame source"
            )
        frame = frames[request.frame_index]
        pipeline = (
            FijiReconstructionPipeline()
            if request.method == "fiji_reconstruction"
            else HessianVectorPipeline()
        )
        proposal = pipeline.analyze(frame)
        document = AnnotationDocument(
            annotation_id=uuid4(),
            revision_id=uuid4(),
            source=frame.source,
            proposals=[proposal],
            review=ReviewRecord(
                author="cellvector",
                timestamp=datetime.now(timezone.utc),
                edit_summary=f"generated {request.method} proposal",
            ),
        )
        return PredictResult(
            document=document,
            proposal=proposal,
            warnings=proposal.warnings,
        )

