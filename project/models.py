"""Strict, versioned contracts for a CellVector project workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


class ProjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameRecord(ProjectModel):
    frame_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: str
    pixel_size_um: tuple[PositiveFloat, PositiveFloat] | None = None


class SourceRecord(ProjectModel):
    source_id: UUID
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frames: tuple[FrameRecord, ...]
    specimen_id: str | None = None
    stack_group_id: str | None = None


class ProjectManifest(ProjectModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: UUID
    name: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    default_backend_id: str = "nnunet_v2"
    default_worker_profile: str = "local"
    registries: dict[str, str]

    @field_validator("registries")
    @classmethod
    def registry_paths_are_relative(cls, registries: dict[str, str]) -> dict[str, str]:
        if "sources" not in registries:
            raise ValueError("project manifest requires a sources registry")
        credential_words = {
            "auth",
            "authentication",
            "authorization",
            "credential",
            "credentials",
            "passwd",
            "password",
            "secret",
            "token",
        }
        credential_pairs = {("api", "key"), ("access", "key")}
        compact_credential_names = {
            "accesskey",
            "accesssecret",
            "accesstoken",
            "apikey",
            "apisecret",
            "apitoken",
            "authkey",
            "authsecret",
            "authtoken",
            "clientkey",
            "clientsecret",
            "clienttoken",
        }
        for name, value in registries.items():
            normalized_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
            normalized_name = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized_name
            ).lower()
            name_parts = tuple(
                part for part in re.split(r"[^a-z0-9]+", normalized_name) if part
            )
            if (
                credential_words.intersection(name_parts)
                or any(
                    pair in credential_pairs for pair in zip(name_parts, name_parts[1:])
                )
                or "".join(name_parts) in compact_credential_names
            ):
                raise ValueError(f"credential-bearing registry key is forbidden: {name}")
            normalized = PurePosixPath(value.replace("\\", "/"))
            if (
                not name
                or not value
                or normalized.is_absolute()
                or PureWindowsPath(value).is_absolute()
                or ".." in normalized.parts
            ):
                raise ValueError("registry paths must be project-relative")
        return registries
