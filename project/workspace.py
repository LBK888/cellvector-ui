"""Filesystem lifecycle for versioned CellVector project workspaces."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError

from cellvector.io.images import ImportedFrame, import_frames

from .models import FrameRecord, ProjectManifest, SourceRecord


_MANIFEST_NAME = "project.json"
_DEFAULT_REGISTRIES = {
    "sources": "sources/source-index.json",
    "annotations": "annotations",
    "datasets": "datasets",
    "augmentation": "augmentation",
    "jobs": "jobs",
    "models": "models",
    "predictions": "predictions",
    "review_queue": "review-queue",
    "measurements": "measurements",
    "exports": "exports",
}


class ProjectError(ValueError):
    """A stable project-workspace failure with a machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class ProjectWorkspace:
    def __init__(
        self,
        root: Path,
        manifest: ProjectManifest,
        sources: tuple[SourceRecord, ...],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._sources = {source.source_id: source for source in sources}

    @classmethod
    def create(cls, root: str | Path, *, name: str) -> "ProjectWorkspace":
        root_path = Path(root).resolve()
        manifest_path = root_path / _MANIFEST_NAME
        if manifest_path.exists():
            raise ProjectError("PROJECT_ALREADY_EXISTS", str(root_path))
        source_index_path = root_path / _DEFAULT_REGISTRIES["sources"]
        if source_index_path.exists():
            raise ProjectError(
                "PROJECT_ROOT_NOT_EMPTY",
                f"managed source index already exists: {source_index_path}",
            )

        root_path.mkdir(parents=True, exist_ok=True)
        for relative in _DEFAULT_REGISTRIES.values():
            target = root_path / relative
            directory = target.parent if target.suffix else target
            directory.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        manifest = ProjectManifest(
            project_id=uuid4(),
            name=name,
            created_at=now,
            updated_at=now,
            registries=dict(_DEFAULT_REGISTRIES),
        )
        workspace = cls(root_path, manifest, ())
        workspace._save_sources()
        workspace._save_manifest()
        return workspace

    @classmethod
    def open(cls, root: str | Path) -> "ProjectWorkspace":
        root_path = Path(root).resolve()
        manifest_path = root_path / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise ProjectError("PROJECT_NOT_FOUND", str(root_path))

        try:
            manifest = ProjectManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ProjectError("PROJECT_MANIFEST_INVALID", str(error)) from error

        source_index_path = root_path / manifest.registries["sources"]
        if not source_index_path.is_file():
            raise ProjectError("SOURCE_INDEX_NOT_FOUND", str(source_index_path))
        try:
            payload = json.loads(source_index_path.read_text(encoding="utf-8"))
            sources = tuple(SourceRecord.model_validate(value) for value in payload)
        except (OSError, TypeError, ValidationError, ValueError) as error:
            raise ProjectError("SOURCE_INDEX_INVALID", str(error)) from error
        return cls(root_path, manifest, sources)

    def import_source(
        self,
        path: str | Path,
        *,
        specimen_id: str | None = None,
        stack_group_id: str | None = None,
    ) -> SourceRecord:
        source_path, imported = self._import_frames(path)
        checksum = imported[0].source.sha256
        duplicate = next(
            (source for source in self._sources.values() if source.sha256 == checksum),
            None,
        )
        if duplicate is not None:
            return duplicate

        source = SourceRecord(
            source_id=uuid4(),
            path=str(source_path),
            sha256=checksum,
            frames=self._frame_records(imported),
            specimen_id=specimen_id,
            stack_group_id=stack_group_id,
        )
        self._sources[source.source_id] = source
        self._persist_change()
        return source

    def relocate_source(self, source_id: UUID, path: str | Path) -> SourceRecord:
        try:
            current = self._sources[source_id]
        except KeyError as error:
            raise ProjectError("SOURCE_NOT_FOUND", str(source_id)) from error

        source_path, imported = self._import_frames(path)
        if imported[0].source.sha256 != current.sha256:
            raise ProjectError(
                "SOURCE_CHECKSUM_MISMATCH",
                f"expected {current.sha256}, received {imported[0].source.sha256}",
            )
        if self._frame_records(imported) != current.frames:
            raise ProjectError(
                "SOURCE_FRAME_LAYOUT_MISMATCH",
                "the selected file does not match the recorded frame layout",
            )
        relocated = current.model_copy(update={"path": str(source_path)})
        self._sources[source_id] = relocated
        self._persist_change()
        return relocated

    def list_sources(self) -> tuple[SourceRecord, ...]:
        return tuple(sorted(self._sources.values(), key=lambda source: str(source.source_id)))

    @property
    def _source_index_path(self) -> Path:
        return self.root / self.manifest.registries["sources"]

    def _import_frames(self, path: str | Path) -> tuple[Path, list[ImportedFrame]]:
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise ProjectError("SOURCE_PATH_NOT_FOUND", str(source_path))
        imported = import_frames(source_path)
        if not imported:
            raise ProjectError("SOURCE_HAS_NO_FRAMES", str(source_path))
        return source_path, imported

    @staticmethod
    def _frame_records(imported: list[ImportedFrame]) -> tuple[FrameRecord, ...]:
        return tuple(
            FrameRecord(
                frame_index=frame.source.frame_index,
                width_px=frame.source.width_px,
                height_px=frame.source.height_px,
                dtype=frame.source.dtype,
                pixel_size_um=frame.source.pixel_size_um,
            )
            for frame in imported
        )

    def _persist_change(self) -> None:
        self._save_sources()
        self.manifest = self.manifest.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        self._save_manifest()

    def _save_manifest(self) -> None:
        self._atomic_write_json(
            self.root / _MANIFEST_NAME,
            self.manifest.model_dump(mode="json"),
        )

    def _save_sources(self) -> None:
        self._atomic_write_json(
            self._source_index_path,
            [source.model_dump(mode="json") for source in self.list_sources()],
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
