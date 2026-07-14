from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from cellvector.backends.contracts import (
    ModelArtifact,
    ModelStatus,
    ProvenanceField,
    ProvenanceStatus,
)


SCHEMA_VERSION = "1.0.0"
_LEGACY_STATES = {
    "draft": ModelStatus.TRAINED,
    "candidate": ModelStatus.TRAINED,
    "promoted": ModelStatus.SELECTED,
    "retired": ModelStatus.RETIRED,
}


class ModelRegistryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def model_artifact_from_legacy_record(record: Any) -> ModelArtifact:
    if hasattr(record, "model_dump"):
        raw = record.model_dump(mode="json")
    elif isinstance(record, Mapping):
        raw = dict(record)
    else:
        raise TypeError("legacy model record must be a mapping or Pydantic model")

    state = str(raw["state"])
    try:
        status = _LEGACY_STATES[state]
    except KeyError as error:
        raise ModelRegistryError("MODEL_STATE_UNSUPPORTED", state) from error

    incomplete: list[ProvenanceField] = []

    def optional(field: ProvenanceField) -> Any:
        value = raw.get(field.value)
        if value is None:
            incomplete.append(field)
        return value

    def optional_text(field: ProvenanceField) -> str | None:
        value = raw.get(field.value)
        if value is None:
            incomplete.append(field)
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field.value} must be a string when present")
        if not value.strip():
            incomplete.append(field)
            return None
        return value

    backend_version = optional_text(ProvenanceField.BACKEND_VERSION)
    architecture = optional_text(ProvenanceField.ARCHITECTURE)
    if "configuration" in raw and raw["configuration"] is not None:
        configuration = raw["configuration"]
    else:
        configuration = {}
        incomplete.append(ProvenanceField.CONFIGURATION)
    dataset_artifact_hash = optional(ProvenanceField.DATASET_ARTIFACT_HASH)
    augmentation_profile_hash = optional(ProvenanceField.AUGMENTATION_PROFILE_HASH)
    input_contract = optional_text(ProvenanceField.INPUT_CONTRACT)
    training_job_id = optional(ProvenanceField.TRAINING_JOB_ID)
    created_at = optional(ProvenanceField.CREATED_AT)
    updated_at = optional(ProvenanceField.UPDATED_AT)

    model_id = raw["model_id"]
    checkpoint_sha256 = raw["checkpoint_sha256"]
    artifact_sha256 = raw.get("artifact_sha256", {})
    matching_checkpoint_paths = [
        path for path, checksum in artifact_sha256.items() if checksum == checkpoint_sha256
    ]
    if len(matching_checkpoint_paths) == 1:
        checkpoint_path = matching_checkpoint_paths[0]
    else:
        checkpoint_path = None
        incomplete.append(ProvenanceField.CHECKPOINT_PATH)

    selection_actor = raw.get("selection_actor", raw.get("promotion_actor"))
    selection_reason = raw.get("selection_reason", raw.get("promotion_reason"))
    selected_at = raw.get("selected_at", raw.get("promotion_at"))
    if status is ModelStatus.SELECTED:
        if selection_actor is None or not str(selection_actor).strip():
            selection_actor = None
            incomplete.append(ProvenanceField.SELECTION_ACTOR)
        if selection_reason is None or not str(selection_reason).strip():
            selection_reason = None
            incomplete.append(ProvenanceField.SELECTION_REASON)
        if selected_at is None:
            incomplete.append(ProvenanceField.SELECTED_AT)

    provenance_status = (
        ProvenanceStatus.LEGACY_INCOMPLETE
        if incomplete
        else ProvenanceStatus.VERIFIED
    )
    payload = {
        "model_id": model_id,
        "status": status.value,
        "backend_id": raw.get("backend_id", "nnunet_v2"),
        "backend_version": backend_version,
        "architecture": architecture,
        "configuration": configuration,
        "snapshot_hash": raw["snapshot_hash"],
        "dataset_artifact_hash": dataset_artifact_hash,
        "augmentation_profile_hash": augmentation_profile_hash,
        "input_contract": input_contract,
        "label_contract": raw.get(
            "label_contract", "cellvector.annotation/1.0.0"
        ),
        "folds_completed": raw.get("folds_completed", []),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "artifact_sha256": artifact_sha256,
        "training_job_id": training_job_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "software_smoke_test": raw.get("software_smoke_test", False),
        "metrics": raw.get("metrics"),
        "benchmark_id": raw.get("benchmark_id"),
        "parent_model_id": raw.get("parent_model_id"),
        "selection_actor": selection_actor,
        "selection_reason": selection_reason,
        "selected_at": selected_at,
        "provenance_status": provenance_status.value,
        "incomplete_provenance_fields": [field.value for field in incomplete],
    }
    return ModelArtifact.model_validate_json(json.dumps(payload))


class GenericModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._load()

    def register(self, record: ModelArtifact) -> ModelArtifact:
        if record.model_id in self._records:
            raise ModelRegistryError("MODEL_ALREADY_EXISTS", str(record.model_id))
        self._records[record.model_id] = record
        self._save()
        return record

    def list(self) -> list[ModelArtifact]:
        return sorted(self._records.values(), key=lambda item: str(item.model_id))

    def get(self, model_id: UUID) -> ModelArtifact:
        try:
            return self._records[model_id]
        except KeyError as error:
            raise ModelRegistryError("MODEL_MISSING", str(model_id)) from error

    def _load(self) -> dict[UUID, ModelArtifact]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelRegistryError("MODEL_REGISTRY_INVALID", str(self.path)) from error

        if isinstance(payload, list):
            raw_records = payload
            legacy = True
        elif isinstance(payload, dict):
            schema_version = payload.get("schema_version")
            if schema_version != SCHEMA_VERSION:
                raise ModelRegistryError(
                    "MODEL_REGISTRY_SCHEMA_UNSUPPORTED", str(schema_version)
                )
            raw_records = payload.get("models")
            if not isinstance(raw_records, list):
                error = TypeError("models must be a list")
                raise ModelRegistryError(
                    "MODEL_REGISTRY_INVALID", "models must be a list"
                ) from error
            legacy = False
        else:
            error = TypeError("registry root must be a list or object")
            raise ModelRegistryError("MODEL_REGISTRY_INVALID", str(self.path)) from error

        records: dict[UUID, ModelArtifact] = {}
        kind = "legacy" if legacy else "schema 1.0.0"
        for index, item in enumerate(raw_records):
            try:
                record = (
                    model_artifact_from_legacy_record(item)
                    if legacy
                    else ModelArtifact.model_validate_json(json.dumps(item))
                )
            except ModelRegistryError as error:
                if error.code == "MODEL_STATE_UNSUPPORTED":
                    raise
                raise ModelRegistryError(
                    "MODEL_REGISTRY_INVALID", f"{kind} entry {index}"
                ) from error
            except Exception as error:
                raise ModelRegistryError(
                    "MODEL_REGISTRY_INVALID", f"{kind} entry {index}"
                ) from error
            if record.model_id in records:
                raise ModelRegistryError(
                    "MODEL_REGISTRY_DUPLICATE_ID", str(record.model_id)
                )
            records[record.model_id] = record
        return records

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "models": [
                        record.model_dump(mode="json") for record in self.list()
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)
