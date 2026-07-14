from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import Field

from cellvector.datasets.models import DatasetError

from .contracts import ContractModel


class ModelState(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    RETIRED = "retired"


class ModelRecord(ContractModel):
    model_id: UUID
    state: ModelState = ModelState.DRAFT
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: dict[str, str]
    folds_completed: tuple[int, ...]
    benchmark_id: UUID | None = None
    software_smoke_test: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    promotion_actor: str | None = None
    promotion_reason: str | None = None

    def to_model_artifact(self):
        """Return the generic in-memory view without changing legacy storage."""

        from cellvector.models.registry import model_artifact_from_legacy_record

        return model_artifact_from_legacy_record(self)


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._load()

    def register(self, record: ModelRecord) -> ModelRecord:
        if record.model_id in self._records:
            raise DatasetError("MODEL_ALREADY_EXISTS", str(record.model_id))
        self._records[record.model_id] = record
        self._save()
        return record

    def list(self) -> list[ModelRecord]:
        return sorted(self._records.values(), key=lambda item: str(item.model_id))

    def get(self, model_id: UUID) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as error:
            raise DatasetError("MODEL_MISSING", str(model_id)) from error

    def mark_candidate(self, model_id: UUID) -> ModelRecord:
        record = self.get(model_id)
        if record.software_smoke_test:
            raise DatasetError(
                "SOFTWARE_SMOKE_MODEL",
                "software fixtures cannot become candidate models",
            )
        if (
            set(record.folds_completed) != {0, 1, 2, 3, 4}
            or record.benchmark_id is None
            or not record.artifact_sha256
        ):
            raise DatasetError(
                "MODEL_ARTIFACTS_INCOMPLETE",
                "five folds, a benchmark, and verified artifacts are required",
            )
        updated = record.model_copy(update={"state": ModelState.CANDIDATE})
        self._records[model_id] = updated
        self._save()
        return updated

    def promote(self, model_id: UUID, *, actor: str, reason: str) -> ModelRecord:
        record = self.get(model_id)
        if record.state != ModelState.CANDIDATE:
            raise DatasetError("MODEL_NOT_CANDIDATE", str(model_id))
        if not actor.strip() or not reason.strip():
            raise DatasetError(
                "PROMOTION_REQUIRES_HUMAN_REVIEW",
                "actor and reason are required",
            )
        updated = record.model_copy(
            update={
                "state": ModelState.PROMOTED,
                "promotion_actor": actor.strip(),
                "promotion_reason": reason.strip(),
            }
        )
        self._records[model_id] = updated
        self._save()
        return updated

    def retire(self, model_id: UUID) -> ModelRecord:
        record = self.get(model_id)
        updated = record.model_copy(update={"state": ModelState.RETIRED})
        self._records[model_id] = updated
        self._save()
        return updated

    def _load(self) -> dict[UUID, ModelRecord]:
        if not self.path.is_file():
            return {}
        import json

        records = [
            ModelRecord.model_validate_json(json.dumps(item))
            for item in json.loads(self.path.read_text(encoding="utf-8"))
        ]
        return {record.model_id: record for record in records}

    def _save(self) -> None:
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in self.list()],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)
