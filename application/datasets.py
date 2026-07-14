from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence
from uuid import UUID

from cellvector.backends import BackendCapabilities
from cellvector.datasets.models import DatasetSnapshot, SplitName
from cellvector.datasets.preflight import (
    DatasetPreflightReport,
    preflight_dataset as _preflight_dataset,
)
from cellvector.datasets.snapshot import sample_from_document
from cellvector.datasets.splitting import assign_group_splits
from cellvector.datasets.training_profile import TrainingDatasetProfile
from cellvector.domain.models import AnnotationDocument


@dataclass(frozen=True)
class DatasetEntry:
    document: AnnotationDocument
    reviewed: bool
    specimen_id: str | None = None
    stack_group_id: str | None = None
    sample_id: str | None = None


def preflight_dataset(
    snapshot: DatasetSnapshot,
    documents: Mapping[UUID, AnnotationDocument],
    profile: TrainingDatasetProfile,
    capabilities: BackendCapabilities,
) -> DatasetPreflightReport:
    """Run dataset preflight through the application boundary."""

    return _preflight_dataset(snapshot, documents, profile, capabilities)


def create_dataset_snapshot(
    entries: Sequence[DatasetEntry],
    *,
    seed: int,
    created_by: str,
    frozen_ledger: Mapping[str, str] | None = None,
    software_smoke_test: bool = False,
    output_path: str | None = None,
) -> DatasetSnapshot:
    samples = [
        sample_from_document(
            entry.document,
            reviewed=entry.reviewed,
            specimen_id=entry.specimen_id,
            stack_group_id=entry.stack_group_id,
            sample_id=entry.sample_id,
        )
        for entry in entries
    ]
    if software_smoke_test:
        assigned = [
            sample.model_copy(update={"split": SplitName.TRAIN}) for sample in samples
        ]
    else:
        assigned = assign_group_splits(
            samples,
            seed=seed,
            frozen_ledger=frozen_ledger,
        )
    return DatasetSnapshot(
        samples=tuple(assigned),
        seed=seed,
        created_at=datetime.now(timezone.utc),
        created_by=created_by,
        output_path=output_path,
        software_smoke_test=software_smoke_test,
    )
