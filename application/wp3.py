from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
from uuid import UUID

from cellvector.agreement.metrics import compare_annotations
from cellvector.datasets.diff import SnapshotDiff, compare_snapshots
from cellvector.datasets.lineage import DatasetRegistry, SnapshotRecord
from cellvector.datasets.models import DatasetSnapshot
from cellvector.domain.models import AnnotationDocument
from cellvector.regression.evaluate import evaluate_policy
from cellvector.regression.models import (
    RegressionEvidence,
    RegressionEvaluation,
    RegressionPolicy,
)
from cellvector.review_queue.models import QueueOrigin, QueuePriorityPolicy, ReviewQueueItem
from cellvector.review_queue.registry import ReviewQueueRegistry


class WP3Application:
    def __init__(self, dataset_registry_path: str | Path, queue_path: str | Path) -> None:
        self.datasets = DatasetRegistry(dataset_registry_path)
        self.queue = ReviewQueueRegistry(queue_path)

    def register_snapshot(self, snapshot: DatasetSnapshot, manifest_path: str | Path, **kwargs) -> SnapshotRecord:
        return self.datasets.register(snapshot, manifest_path, **kwargs)

    @staticmethod
    def diff(base: DatasetSnapshot, target: DatasetSnapshot) -> SnapshotDiff:
        return compare_snapshots(base, target)

    def compare_and_enqueue(self, documents: Sequence[AnnotationDocument]):
        report = compare_annotations(list(documents))
        disagreement = 1.0 - report.aggregate["microridge_dice"]
        priority = QueuePriorityPolicy(weights={"agreement": 1.0}).score(
            {"agreement": disagreement}
        )
        first = documents[0]
        item = self.queue.add(
            source_sha256=first.source.sha256,
            frame_index=first.source.frame_index,
            origin=QueueOrigin.AGREEMENT,
            priority=priority,
            reasons=("inter-annotator-agreement",),
            revision_id=first.revision_id,
            note=f"agreement report {report.report_id}",
        )
        return report, item

    def list_queue(self, **kwargs):
        return self.queue.list(**kwargs)

    def get_queue_item(self, item_id: UUID) -> ReviewQueueItem:
        return self.queue.get(item_id)

    def claim(self, item_id: UUID, *, actor: str) -> ReviewQueueItem:
        return self.queue.claim(item_id, actor=actor)

    def release(self, item_id: UUID, *, actor: str, note: str = "") -> ReviewQueueItem:
        return self.queue.release(item_id, actor=actor, note=note)

    def resolve(
        self,
        item_id: UUID,
        *,
        actor: str,
        result_revision_id: UUID,
        note: str = "",
    ) -> ReviewQueueItem:
        return self.queue.resolve(
            item_id,
            actor=actor,
            result_revision_id=result_revision_id,
            note=note,
        )

    def dismiss(self, item_id: UUID, *, actor: str, note: str = "") -> ReviewQueueItem:
        return self.queue.dismiss(item_id, actor=actor, note=note)

    @staticmethod
    def evaluate(
        policy: RegressionPolicy,
        evidence: RegressionEvidence,
        metrics: Mapping[str, float],
    ) -> RegressionEvaluation:
        return evaluate_policy(policy, evidence, metrics)
