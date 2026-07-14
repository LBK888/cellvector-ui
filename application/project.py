"""UI- and AI-independent project-workspace application facade."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from cellvector.project import ProjectWorkspace, SourceRecord


class ProjectApplication:
    def __init__(self, workspace: ProjectWorkspace) -> None:
        self.workspace = workspace

    @classmethod
    def create(cls, root: str | Path, *, name: str) -> "ProjectApplication":
        return cls(ProjectWorkspace.create(root, name=name))

    @classmethod
    def open(cls, root: str | Path) -> "ProjectApplication":
        return cls(ProjectWorkspace.open(root))

    def import_source(self, path: str | Path, **grouping: str | None) -> SourceRecord:
        return self.workspace.import_source(path, **grouping)

    def relocate_source(self, source_id: UUID, path: str | Path) -> SourceRecord:
        return self.workspace.relocate_source(source_id, path)

    def list_sources(self) -> tuple[SourceRecord, ...]:
        return self.workspace.list_sources()
