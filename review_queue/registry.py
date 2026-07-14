from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import UUID, uuid4

from cellvector.datasets.lineage import WP3Error

from .models import (
    QueueAuditEvent,
    QueueOrigin,
    QueuePriority,
    QueueState,
    ReviewQueueItem,
)


class ReviewQueueRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items = self._load()

    def add(
        self,
        *,
        source_sha256: str,
        frame_index: int,
        origin: QueueOrigin,
        priority: QueuePriority,
        reasons: tuple[str, ...],
        note: str = "",
        annotation_id: UUID | None = None,
        revision_id: UUID | None = None,
        proposal_id: UUID | None = None,
        model_id: UUID | None = None,
        snapshot_hash: str | None = None,
    ) -> ReviewQueueItem:
        identity = (source_sha256, frame_index, origin, model_id, proposal_id)
        if any(
            (item.source_sha256, item.frame_index, item.origin, item.model_id, item.proposal_id)
            == identity
            and item.state in {QueueState.QUEUED, QueueState.CLAIMED}
            for item in self._items.values()
        ):
            raise WP3Error(
                "QUEUE_DUPLICATE_OPEN_ITEM", "an equivalent queue item is already open"
            )
        now = datetime.now(timezone.utc)
        item = ReviewQueueItem(
            item_id=uuid4(),
            source_sha256=source_sha256,
            frame_index=frame_index,
            origin=origin,
            priority=priority,
            reasons=reasons,
            note=note,
            annotation_id=annotation_id,
            revision_id=revision_id,
            proposal_id=proposal_id,
            model_id=model_id,
            snapshot_hash=snapshot_hash,
            created_at=now,
            audit=(
                QueueAuditEvent(
                    previous_state=None,
                    new_state=QueueState.QUEUED,
                    actor="system",
                    timestamp=now,
                ),
            ),
        )
        self._items[item.item_id] = item
        self._save()
        return item

    def list(self, *, state: QueueState | None = None) -> list[ReviewQueueItem]:
        values = [item for item in self._items.values() if state is None or item.state == state]
        return sorted(values, key=lambda item: (-item.priority.total, item.created_at, str(item.item_id)))

    def get(self, item_id: UUID) -> ReviewQueueItem:
        try:
            return self._items[item_id]
        except KeyError as error:
            raise WP3Error("QUEUE_ITEM_NOT_FOUND", str(item_id)) from error

    def claim(self, item_id: UUID, *, actor: str) -> ReviewQueueItem:
        return self._transition(item_id, QueueState.CLAIMED, actor=actor)

    def release(self, item_id: UUID, *, actor: str, note: str = "") -> ReviewQueueItem:
        return self._transition(item_id, QueueState.QUEUED, actor=actor, note=note)

    def resolve(
        self,
        item_id: UUID,
        *,
        actor: str,
        result_revision_id: UUID | None,
        note: str = "",
    ) -> ReviewQueueItem:
        if result_revision_id is None:
            raise WP3Error(
                "QUEUE_RESULT_REVISION_REQUIRED", "resolution requires an exact revision"
            )
        return self._transition(
            item_id,
            QueueState.RESOLVED,
            actor=actor,
            note=note,
            result_revision_id=result_revision_id,
        )

    def dismiss(self, item_id: UUID, *, actor: str, note: str = "") -> ReviewQueueItem:
        return self._transition(item_id, QueueState.DISMISSED, actor=actor, note=note)

    def _transition(
        self,
        item_id: UUID,
        new_state: QueueState,
        *,
        actor: str,
        note: str = "",
        result_revision_id: UUID | None = None,
    ) -> ReviewQueueItem:
        actor = actor.strip()
        if not actor:
            raise WP3Error("QUEUE_ACTOR_REQUIRED", "queue transition requires an actor")
        item = self.get(item_id)
        allowed = {
            QueueState.QUEUED: {QueueState.CLAIMED, QueueState.DISMISSED},
            QueueState.CLAIMED: {
                QueueState.QUEUED,
                QueueState.RESOLVED,
                QueueState.DISMISSED,
            },
        }
        if new_state not in allowed.get(item.state, set()):
            raise WP3Error(
                "QUEUE_INVALID_TRANSITION", f"{item.state.value} -> {new_state.value}"
            )
        event = QueueAuditEvent(
            previous_state=item.state,
            new_state=new_state,
            actor=actor,
            note=note,
            timestamp=datetime.now(timezone.utc),
            result_revision_id=result_revision_id,
        )
        updated = item.model_copy(
            update={
                "state": new_state,
                "claimed_by": actor if new_state == QueueState.CLAIMED else None,
                "result_revision_id": result_revision_id or item.result_revision_id,
                "audit": (*item.audit, event),
            }
        )
        self._items[item_id] = updated
        self._save()
        return updated

    def _load(self) -> dict[UUID, ReviewQueueItem]:
        if not self.path.is_file():
            return {}
        values = json.loads(self.path.read_text(encoding="utf-8"))
        items = [ReviewQueueItem.model_validate(value) for value in values]
        return {item.item_id: item for item in items}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in self.list()],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)
