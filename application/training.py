"""UI/CLI-independent model lifecycle use cases."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
from uuid import UUID

from cellvector.datasets.models import DatasetSnapshot
from cellvector.inference.nnunet.registry import ModelRecord, ModelRegistry


class TrainingApplication:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    @classmethod
    def at_path(cls, path: str | Path) -> "TrainingApplication":
        return cls(ModelRegistry(path))

    def register_draft(
        self,
        *,
        model_id: UUID,
        snapshot: DatasetSnapshot,
        checkpoint_sha256: str,
        artifact_sha256: Mapping[str, str],
        folds_completed: Sequence[int],
        benchmark_id: UUID | None = None,
    ) -> ModelRecord:
        """Register a completed run as a draft tied to an immutable snapshot."""

        record = ModelRecord(
            model_id=model_id,
            snapshot_hash=snapshot.identity_hash(),
            checkpoint_sha256=checkpoint_sha256,
            artifact_sha256=dict(artifact_sha256),
            folds_completed=tuple(folds_completed),
            benchmark_id=benchmark_id,
            software_smoke_test=snapshot.software_smoke_test,
        )
        return self.registry.register(record)

    def mark_candidate(self, model_id: UUID) -> ModelRecord:
        return self.registry.mark_candidate(model_id)

    def promote(self, model_id: UUID, *, actor: str, reason: str) -> ModelRecord:
        return self.registry.promote(model_id, actor=actor, reason=reason)

    def list_models(self) -> list[ModelRecord]:
        return self.registry.list()

