"""Headless controller used by the napari UI and tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import UUID, uuid4

from numpy.typing import NDArray
import numpy as np

from cellvector.application.review import accept_proposal_features
from cellvector.application.wp3 import WP3Application
from cellvector.domain.models import (
    AnnotationDocument,
    CellFeature,
    MicroridgeFeature,
    Point,
    PredictionProposal,
    Provenance,
    ReviewRecord,
    ReviewStatus,
)
from cellvector.domain.geometry import build_shared_boundaries
from cellvector.domain.revisions import create_revision
from cellvector.domain.validation import QCError, validate_document
from cellvector.execution.base import ExecutionBackend, PredictRequest
from cellvector.export.svg import export_svg
from cellvector.inference.nnunet.adapter import proposal_from_label_map
from cellvector.inference.nnunet.contracts import InferenceProfile
from cellvector.inference.nnunet.registry import ModelRecord
from cellvector.io.images import ImportedFrame, import_frames
from cellvector.review_queue.models import QueueState, ReviewQueueItem


class CellVectorController:
    def __init__(
        self,
        local: ExecutionBackend,
        remote: ExecutionBackend,
    ) -> None:
        self.local = local
        self.remote = remote
        self.remote_health = remote.health()
        self.remote_status = self.remote_health.status
        self.frames: list[ImportedFrame] = []
        self.frame: ImportedFrame | None = None
        self.document: AnnotationDocument | None = None
        self.proposal: PredictionProposal | None = None
        self.classical_proposal: PredictionProposal | None = None
        self.ai_proposal: PredictionProposal | None = None
        self.wp3: WP3Application | None = None

    def configure_wp3(
        self,
        dataset_registry_path: str | Path,
        queue_path: str | Path,
    ) -> None:
        """Attach persistent WP3 registries without requiring an AI service."""

        self.wp3 = WP3Application(dataset_registry_path, queue_path)

    def list_review_queue(
        self,
        state: QueueState | None = None,
    ) -> list[ReviewQueueItem]:
        return self._require_wp3().list_queue(state=state)

    def claim_review_item(self, item_id: UUID, *, actor: str) -> ReviewQueueItem:
        return self._require_wp3().claim(item_id, actor=actor)

    def release_review_item(
        self,
        item_id: UUID,
        *,
        actor: str,
        note: str = "",
    ) -> ReviewQueueItem:
        return self._require_wp3().release(item_id, actor=actor, note=note)

    def resolve_review_item(
        self,
        item_id: UUID,
        *,
        actor: str,
        note: str = "",
    ) -> ReviewQueueItem:
        _, document = self._require_open_document()
        wp3 = self._require_wp3()
        item = wp3.get_queue_item(item_id)
        if (
            item.source_sha256 != document.source.sha256
            or item.frame_index != document.source.frame_index
        ):
            raise ValueError("review item does not match the open source frame")
        return wp3.resolve(
            item_id,
            actor=actor,
            result_revision_id=document.revision_id,
            note=note,
        )

    def dismiss_review_item(
        self,
        item_id: UUID,
        *,
        actor: str,
        note: str = "",
    ) -> ReviewQueueItem:
        return self._require_wp3().dismiss(item_id, actor=actor, note=note)

    def _require_wp3(self) -> WP3Application:
        if self.wp3 is None:
            raise RuntimeError("configure WP3 registries first")
        return self.wp3

    def open_image(
        self,
        path: str | Path,
        frame_index: int = 0,
    ) -> AnnotationDocument:
        self.frames = import_frames(path)
        return self.select_frame(frame_index)

    def select_frame(self, frame_index: int) -> AnnotationDocument:
        if not self.frames:
            raise RuntimeError("open an image before selecting a frame")
        if not 0 <= frame_index < len(self.frames):
            raise IndexError(f"frame index {frame_index} is outside {len(self.frames)} frames")
        self.frame = self.frames[frame_index]
        self.proposal = None
        self.classical_proposal = None
        self.ai_proposal = None
        self.document = AnnotationDocument(
            annotation_id=uuid4(),
            revision_id=uuid4(),
            source=self.frame.source,
            review=ReviewRecord(
                author="cellvector",
                timestamp=datetime.now(timezone.utc),
                edit_summary=f"opened source frame {frame_index}",
            ),
        )
        return self.document

    def open_annotation(self, path: str | Path) -> AnnotationDocument:
        """Reopen a saved revision after verifying its immutable source image."""

        annotation_path = Path(path)
        document = AnnotationDocument.model_validate_json(
            annotation_path.read_text(encoding="utf-8")
        )
        frames = import_frames(Path(document.source.image_uri))
        if document.source.frame_index >= len(frames):
            raise ValueError("annotation frame index is outside the source stack")
        frame = frames[document.source.frame_index]
        if frame.source.sha256 != document.source.sha256:
            raise ValueError("annotation source checksum does not match the current image file")
        if (
            frame.source.width_px != document.source.width_px
            or frame.source.height_px != document.source.height_px
        ):
            raise ValueError("annotation source dimensions do not match the current image file")
        report = validate_document(document)
        if report.errors:
            raise QCError(report)
        self.frames = frames
        self.frame = frame
        self.document = document
        self.proposal = document.proposals[-1] if document.proposals else None
        return document

    def run_classical(self, method: str) -> PredictionProposal:
        frame, document = self._require_open_document()
        result = self.local.predict(
            PredictRequest(
                image_path=Path(frame.source.image_uri),
                method=method,
                frame_index=frame.source.frame_index,
            )
        )
        if result.document.source.sha256 != document.source.sha256:
            raise ValueError("analysis result source checksum differs from the open image")
        self.proposal = result.proposal
        self.classical_proposal = result.proposal
        self.document = document.model_copy(
            update={"proposals": [*document.proposals, self.proposal]}
        )
        return self.proposal

    def add_ai_prediction(
        self,
        labels: NDArray[np.uint8],
        *,
        model: ModelRecord,
        profile: InferenceProfile,
    ) -> PredictionProposal:
        """Attach an AI result as an isolated proposal, never as final geometry."""

        _, document = self._require_open_document()
        proposal = proposal_from_label_map(
            document,
            labels,
            model=model,
            profile=profile,
        )
        self.ai_proposal = proposal
        self.proposal = proposal
        self.document = document.model_copy(
            update={"proposals": [*document.proposals, proposal]}
        )
        return proposal

    def accept_selected(
        self,
        feature_ids: Sequence[UUID],
        *,
        author: str,
    ) -> AnnotationDocument:
        _, document = self._require_open_document()
        if self.proposal is None:
            raise RuntimeError("run an analysis before accepting proposal features")
        self.document = accept_proposal_features(
            document,
            self.proposal.id,
            feature_ids,
            author=author,
        )
        return self.document

    def replace_microridge_points(
        self,
        feature_id: UUID,
        points: list[Point],
        *,
        author: str,
    ) -> AnnotationDocument:
        _, document = self._require_open_document()
        replacement_found = False
        updated: list[MicroridgeFeature] = []
        for ridge in document.microridges:
            if ridge.id != feature_id:
                updated.append(ridge)
                continue
            replacement_found = True
            updated.append(
                ridge.model_copy(
                    update={
                        "points": points,
                        "closed": points[0] == points[-1],
                        "review_status": ReviewStatus.MODIFIED,
                        "provenance": Provenance(
                            method="manual",
                            source_proposal_id=ridge.provenance.source_proposal_id,
                            implementation_version=ridge.provenance.implementation_version,
                            model_version=ridge.provenance.model_version,
                            parameters=ridge.provenance.parameters,
                        ),
                    }
                )
            )
        if not replacement_found:
            raise KeyError(f"microridge not found: {feature_id}")
        self.document = create_revision(
            document,
            author=author,
            summary=f"edited microridge {feature_id}",
            microridges=updated,
        )
        return self.document

    def replace_all_microridges(
        self,
        paths: Sequence[Sequence[Point]],
        *,
        author: str,
    ) -> AnnotationDocument:
        """Commit the complete editable microridge layer as one new revision."""

        _, document = self._require_open_document()
        updated: list[MicroridgeFeature] = []
        for index, path in enumerate(paths):
            points = list(path)
            if index < len(document.microridges):
                previous = document.microridges[index]
                feature_id = previous.id
                cell_id = previous.cell_id
                confidence = previous.confidence
                source_proposal_id = previous.provenance.source_proposal_id
            else:
                feature_id = uuid4()
                cell_id = None
                confidence = None
                source_proposal_id = None
            updated.append(
                MicroridgeFeature(
                    id=feature_id,
                    points=points,
                    cell_id=cell_id,
                    closed=points[0] == points[-1],
                    confidence=confidence,
                    review_status=ReviewStatus.MODIFIED,
                    provenance=Provenance(
                        method="manual",
                        source_proposal_id=source_proposal_id,
                    ),
                )
            )
        self.document = create_revision(
            document,
            author=author,
            summary=(
                "committed editable microridge layer: "
                f"{len(updated)} path(s), previously {len(document.microridges)}"
            ),
            microridges=updated,
        )
        return self.document

    def replace_all_cells(
        self,
        contours: Sequence[Sequence[Point]],
        *,
        author: str,
    ) -> AnnotationDocument:
        """Commit complete cell polygons and derive one shared boundary graph."""

        _, document = self._require_open_document()
        cells: list[CellFeature] = []
        for index, contour in enumerate(contours):
            points = list(contour)
            if points[0] != points[-1]:
                points.append(points[0])
            feature_id = (
                document.cells[index].id
                if index < len(document.cells)
                else uuid4()
            )
            cells.append(
                CellFeature(
                    id=feature_id,
                    contour=points,
                    partial=False,
                    review_status=ReviewStatus.MODIFIED,
                    provenance=Provenance(method="manual"),
                )
            )
        boundaries = build_shared_boundaries(cells)
        self.document = create_revision(
            document,
            author=author,
            summary=(
                f"committed {len(cells)} complete cell polygon(s) and "
                f"derived {len(boundaries)} shared boundary segment(s)"
            ),
            cells=cells,
            boundaries=boundaries,
        )
        return self.document

    def save_json(self, path: str | Path) -> Path:
        _, document = self._require_open_document()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return output

    def export_svg(self, path: str | Path) -> Path:
        _, document = self._require_open_document()
        return export_svg(document, path)

    def _require_open_document(self) -> tuple[ImportedFrame, AnnotationDocument]:
        if self.frame is None or self.document is None:
            raise RuntimeError("open an image first")
        return self.frame, self.document
