from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Annotated, Any, Mapping, Protocol, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from cellvector.annotations.models import AnnotationDocument
    from cellvector.datasets.models import DatasetSnapshot
    from cellvector.datasets.preflight import DatasetPreflightReport
    from cellvector.datasets.training_profile import TrainingDatasetProfile
    from cellvector.inference.proposals import PredictionProposal
    from cellvector.training.augmentation import AugmentationProfile
    from cellvector.worker.models import WorkerJob


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ALLOWED_COMMAND_ENVIRONMENT_KEYS = frozenset(
    {
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
    }
)


class FrozenDict(dict):
    """A JSON-serializable dict that rejects every normal mutation operation."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("validated contract mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def copy(self) -> FrozenDict:
        return FrozenDict(self)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _artifact_relative_path(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("artifact path must be non-empty and contain no NUL bytes")
    for path_type in (PurePosixPath, PureWindowsPath):
        path = path_type(value)
        if path.is_absolute() or path.anchor or path.drive or ".." in path.parts:
            raise ValueError(f"artifact path must be relative: {value}")
    return value


def _artifact_checksum_mapping(value: dict[str, Sha256]) -> FrozenDict:
    for path in value:
        _artifact_relative_path(path)
    return FrozenDict(value)


class BackendModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> BackendModel:
        del deep
        payload = self.model_dump()
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


class ModelStatus(StrEnum):
    TRAINED = "trained"
    SELECTED = "selected"
    RETIRED = "retired"


class ProvenanceStatus(StrEnum):
    VERIFIED = "verified"
    LEGACY_INCOMPLETE = "legacy_incomplete"


class ProvenanceField(StrEnum):
    BACKEND_VERSION = "backend_version"
    ARCHITECTURE = "architecture"
    CONFIGURATION = "configuration"
    DATASET_ARTIFACT_HASH = "dataset_artifact_hash"
    AUGMENTATION_PROFILE_HASH = "augmentation_profile_hash"
    INPUT_CONTRACT = "input_contract"
    TRAINING_JOB_ID = "training_job_id"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    CHECKPOINT_PATH = "checkpoint_path"
    SELECTION_ACTOR = "selection_actor"
    SELECTION_REASON = "selection_reason"
    SELECTED_AT = "selected_at"


class BackendCapabilities(BackendModel):
    backend_id: str
    backend_version: str
    input_dimensions: tuple[int, ...]
    channel_counts: tuple[int, ...]
    label_contract: str
    architectures: tuple[str, ...]
    supports_resume: bool
    supports_probabilities: bool
    owns_augmentation: bool

    @field_validator("backend_id", "backend_version", "label_contract")
    @classmethod
    def _nonempty_identifier(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("capability identifiers must be non-empty and trimmed")
        return value

    @field_validator("input_dimensions", "channel_counts")
    @classmethod
    def _positive_unique_counts(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item <= 0 for item in value) or len(set(value)) != len(value):
            raise ValueError("capability counts must be non-empty, positive, and unique")
        return value

    @field_validator("architectures")
    @classmethod
    def _unique_architectures(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or any(not item.strip() or item != item.strip() for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("architectures must be non-empty, trimmed, and unique")
        return value


class ExpectedArtifact(BackendModel):
    path: str
    required: bool = True
    expected_sha256: Sha256 | None = None

    _validate_path = field_validator("path")(_artifact_relative_path)


class CommandSpec(BackendModel):
    operation: str
    argv: tuple[str, ...]
    workdir: str
    environment: dict[str, str] = Field(default_factory=dict)
    expected_artifacts: tuple[ExpectedArtifact, ...] = ()
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("environment")
    @classmethod
    def _safe_environment(cls, value: dict[str, str]) -> FrozenDict:
        for key, item in value.items():
            if key not in ALLOWED_COMMAND_ENVIRONMENT_KEYS:
                raise ValueError(f"environment key is not allowed: {key}")
            if "\x00" in key or "\x00" in item:
                raise ValueError("environment keys and values must not contain NUL")
        return FrozenDict(value)

    @field_serializer("environment")
    def _serialize_environment(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class PredictionCase(BackendModel):
    case_id: str
    source_sha256: Sha256
    image_uri: str
    frame_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class TrainingRequest(BackendModel):
    request_id: UUID
    backend_id: str
    architecture: str
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    snapshot_hash: Sha256
    dataset_artifact_hash: Sha256
    augmentation_profile_hash: Sha256
    folds: tuple[int, ...]
    software_smoke_test: bool = False
    parent_model_id: UUID | None = None

    @field_validator("configuration")
    @classmethod
    def _freeze_configuration(cls, value: dict[str, JsonValue]) -> FrozenDict:
        return _deep_freeze(value)

    @field_serializer("configuration")
    def _serialize_configuration(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


class PredictionRequest(BackendModel):
    request_id: UUID
    backend_id: str
    model_id: UUID
    source_cases: tuple[PredictionCase, ...]
    save_probabilities: bool = False


class DatasetArtifact(BackendModel):
    artifact_id: UUID
    backend_id: str
    snapshot_hash: Sha256
    root: str
    artifact_sha256: dict[str, Sha256]
    label_contract: str

    _validate_root = field_validator("root")(_artifact_relative_path)
    _validate_artifact_paths = field_validator("artifact_sha256")(
        _artifact_checksum_mapping
    )

    @field_serializer("artifact_sha256")
    def _serialize_artifact_sha256(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class PredictionCaseArtifact(BackendModel):
    case_id: str
    source_sha256: Sha256
    frame_index: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    label_path: str
    label_sha256: Sha256
    probability_path: str | None = None
    probability_sha256: Sha256 | None = None

    _validate_paths = field_validator("label_path", "probability_path")(
        lambda value: _artifact_relative_path(value) if value is not None else value
    )

    @model_validator(mode="after")
    def _validate_probability_pair(self) -> PredictionCaseArtifact:
        if (self.probability_path is None) != (self.probability_sha256 is None):
            raise ValueError("probability_path and probability_sha256 must be provided together")
        return self


class PredictionArtifact(BackendModel):
    artifact_id: UUID
    backend_id: str
    model_id: UUID
    job_id: UUID
    cases: tuple[PredictionCaseArtifact, ...]
    artifact_sha256: dict[str, Sha256]

    _validate_artifact_paths = field_validator("artifact_sha256")(
        _artifact_checksum_mapping
    )

    @model_validator(mode="after")
    def _validate_case_artifacts(self) -> PredictionArtifact:
        referenced_paths: list[str] = []
        for case in self.cases:
            pairs = [(case.label_path, case.label_sha256)]
            if case.probability_path is not None:
                pairs.append((case.probability_path, case.probability_sha256))
            for path, checksum in pairs:
                referenced_paths.append(path)
                if self.artifact_sha256.get(path) != checksum:
                    raise ValueError(f"prediction artifact checksum mismatch: {path}")
        if len(referenced_paths) != len(set(referenced_paths)):
            raise ValueError("prediction cases must not reuse artifact paths")
        return self

    @field_serializer("artifact_sha256")
    def _serialize_artifact_sha256(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class ModelArtifact(BackendModel):
    model_id: UUID
    status: ModelStatus
    backend_id: str
    backend_version: str | None
    architecture: str | None
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    snapshot_hash: Sha256
    dataset_artifact_hash: Sha256 | None
    augmentation_profile_hash: Sha256 | None
    input_contract: str | None
    label_contract: str
    folds_completed: tuple[int, ...]
    checkpoint_path: str | None = None
    checkpoint_sha256: Sha256
    artifact_sha256: dict[str, Sha256]
    training_job_id: UUID | None
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    software_smoke_test: bool = False
    metrics: dict[str, JsonValue] | None = None
    benchmark_id: UUID | None = None
    parent_model_id: UUID | None = None
    selection_actor: str | None = None
    selection_reason: str | None = None
    selected_at: datetime | None = None
    provenance_status: ProvenanceStatus = ProvenanceStatus.VERIFIED
    incomplete_provenance_fields: tuple[ProvenanceField, ...] = ()

    _validate_artifact_paths = field_validator("artifact_sha256")(
        _artifact_checksum_mapping
    )
    _validate_checkpoint_path = field_validator("checkpoint_path")(
        lambda value: _artifact_relative_path(value) if value is not None else value
    )

    @field_validator("configuration")
    @classmethod
    def _freeze_configuration(cls, value: dict[str, JsonValue]) -> FrozenDict:
        return _deep_freeze(value)

    @field_validator("metrics")
    @classmethod
    def _freeze_metrics(
        cls, value: dict[str, JsonValue] | None
    ) -> FrozenDict | None:
        return _deep_freeze(value) if value is not None else None

    @model_validator(mode="after")
    def _validate_checkpoint_artifact(self) -> ModelArtifact:
        if (
            self.checkpoint_path is not None
            and self.artifact_sha256.get(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_path checksum must match checkpoint_sha256")
        if len(self.incomplete_provenance_fields) != len(
            set(self.incomplete_provenance_fields)
        ):
            raise ValueError("incomplete provenance fields must be unique")

        identity_values = {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "architecture": self.architecture,
            "input_contract": self.input_contract,
            "label_contract": self.label_contract,
        }
        invalid_identity = [
            name
            for name, value in identity_values.items()
            if value is not None and (not value.strip() or value != value.strip())
        ]
        if invalid_identity:
            raise ValueError(
                "provenance identity fields must be nonblank and trimmed: "
                + ", ".join(invalid_identity)
            )
        for name, value in (
            ("selection_actor", self.selection_actor),
            ("selection_reason", self.selection_reason),
        ):
            if value is not None and value and value != value.strip():
                raise ValueError(f"{name} must be trimmed")

        required = {
            ProvenanceField.BACKEND_VERSION: self.backend_version,
            ProvenanceField.ARCHITECTURE: self.architecture,
            ProvenanceField.DATASET_ARTIFACT_HASH: self.dataset_artifact_hash,
            ProvenanceField.AUGMENTATION_PROFILE_HASH: self.augmentation_profile_hash,
            ProvenanceField.INPUT_CONTRACT: self.input_contract,
            ProvenanceField.TRAINING_JOB_ID: self.training_job_id,
            ProvenanceField.CREATED_AT: self.created_at,
            ProvenanceField.UPDATED_AT: self.updated_at,
            ProvenanceField.CHECKPOINT_PATH: self.checkpoint_path,
        }
        if self.provenance_status is ProvenanceStatus.VERIFIED:
            if self.incomplete_provenance_fields:
                raise ValueError("verified provenance cannot declare incomplete fields")
            missing = [field.value for field, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "verified provenance is missing required fields: " + ", ".join(missing)
                )
            if self.status is ModelStatus.SELECTED:
                if (
                    self.selection_actor is None
                    or not self.selection_actor.strip()
                    or self.selection_reason is None
                    or not self.selection_reason.strip()
                    or self.selected_at is None
                ):
                    raise ValueError(
                        "selected verified models require actor, reason, and selected_at"
                    )
        else:
            if not self.incomplete_provenance_fields:
                raise ValueError("legacy incomplete provenance must name missing fields")
            declared = set(self.incomplete_provenance_fields)
            expected = {field for field, value in required.items() if value is None}
            if ProvenanceField.CONFIGURATION in declared:
                if self.configuration:
                    raise ValueError(
                        "configuration cannot be populated when declared incomplete"
                    )
                expected.add(ProvenanceField.CONFIGURATION)
            if self.status is ModelStatus.SELECTED:
                if self.selection_actor is None or not self.selection_actor.strip():
                    expected.add(ProvenanceField.SELECTION_ACTOR)
                if self.selection_reason is None or not self.selection_reason.strip():
                    expected.add(ProvenanceField.SELECTION_REASON)
                if self.selected_at is None:
                    expected.add(ProvenanceField.SELECTED_AT)
            if declared != expected:
                omitted = sorted(field.value for field in expected - declared)
                false_missing = sorted(field.value for field in declared - expected)
                raise ValueError(
                    "legacy incomplete provenance must exactly match absent fields; "
                    f"omitted={omitted}, falsely_declared={false_missing}"
                )
        return self

    @property
    def selection_eligible(self) -> bool:
        return (
            self.provenance_status is ProvenanceStatus.VERIFIED
            and not self.incomplete_provenance_fields
            and not self.software_smoke_test
            and self.status is not ModelStatus.RETIRED
        )

    @field_serializer("configuration", "metrics")
    def _serialize_json_mapping(self, value: Mapping[str, Any] | None) -> Any:
        return _deep_thaw(value) if value is not None else None

    @field_serializer("artifact_sha256")
    def _serialize_artifact_sha256(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


@runtime_checkable
class DatasetAdapter(Protocol):
    def preflight(
        self,
        snapshot: DatasetSnapshot,
        documents: Mapping[UUID, AnnotationDocument],
        profile: TrainingDatasetProfile,
    ) -> DatasetPreflightReport: ...

    def export(
        self,
        snapshot: DatasetSnapshot,
        documents: Mapping[UUID, AnnotationDocument],
        destination: Path,
        profile: TrainingDatasetProfile,
    ) -> DatasetArtifact: ...


@runtime_checkable
class TrainingBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...

    def plan_training(
        self,
        request: TrainingRequest,
        dataset: DatasetArtifact,
        augmentation: AugmentationProfile,
    ) -> tuple[CommandSpec, ...]: ...

    def collect_training(
        self,
        job: WorkerJob,
        request: TrainingRequest,
        dataset: DatasetArtifact,
    ) -> ModelArtifact: ...


@runtime_checkable
class InferenceBackend(Protocol):
    def plan_prediction(
        self,
        request: PredictionRequest,
        model: ModelArtifact,
        workdir: Path,
    ) -> tuple[CommandSpec, ...]: ...

    def collect_prediction(
        self,
        job: WorkerJob,
        request: PredictionRequest,
        model: ModelArtifact,
    ) -> PredictionArtifact: ...


@runtime_checkable
class PredictionAdapter(Protocol):
    def to_proposal(
        self,
        document: AnnotationDocument,
        prediction: PredictionCaseArtifact,
        model: ModelArtifact,
        profile: str,
    ) -> PredictionProposal: ...
