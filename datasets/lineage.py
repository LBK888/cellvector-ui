from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import DatasetSnapshot


class WP3Error(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class SnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0.0"
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actor: str = Field(min_length=1)
    created_at: datetime
    software_smoke_test: bool
    release_note: str | None = None


class DatasetRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._load()

    def register(
        self,
        snapshot: DatasetSnapshot,
        manifest_path: str | Path,
        *,
        actor: str,
        parent_hash: str | None = None,
        release_note: str | None = None,
    ) -> SnapshotRecord:
        manifest = Path(manifest_path).resolve()
        try:
            stored = DatasetSnapshot.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise WP3Error(
                "SNAPSHOT_MANIFEST_MISMATCH", "manifest is not a valid snapshot"
            ) from error
        if stored.identity_hash() != snapshot.identity_hash() or stored != snapshot:
            raise WP3Error(
                "SNAPSHOT_MANIFEST_MISMATCH",
                "manifest does not contain the supplied immutable snapshot",
            )
        if parent_hash is not None and parent_hash not in self._records:
            raise WP3Error("SNAPSHOT_PARENT_MISSING", parent_hash)
        record = SnapshotRecord(
            snapshot_hash=snapshot.identity_hash(),
            parent_hash=parent_hash,
            manifest_path=str(manifest),
            manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
            actor=actor.strip(),
            created_at=datetime.now(timezone.utc),
            software_smoke_test=snapshot.software_smoke_test,
            release_note=release_note,
        )
        existing = self._records.get(record.snapshot_hash)
        if existing is not None:
            comparable = {"created_at", "manifest_path", "manifest_sha256"}
            if existing.model_dump(exclude=comparable) == record.model_dump(exclude=comparable):
                return existing
            raise WP3Error(
                "SNAPSHOT_ALREADY_REGISTERED",
                "snapshot hash already has conflicting registry metadata",
            )
        self._records[record.snapshot_hash] = record
        self._save()
        return record

    def get(self, snapshot_hash: str) -> SnapshotRecord:
        try:
            return self._records[snapshot_hash]
        except KeyError as error:
            raise WP3Error("SNAPSHOT_NOT_FOUND", snapshot_hash) from error

    def list(self) -> list[SnapshotRecord]:
        return sorted(self._records.values(), key=lambda item: item.created_at)

    def _load(self) -> dict[str, SnapshotRecord]:
        if not self.path.is_file():
            return {}
        values = json.loads(self.path.read_text(encoding="utf-8"))
        records = [SnapshotRecord.model_validate(item) for item in values]
        return {item.snapshot_hash: item for item in records}

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

