"""Bounded in-memory job manager for the reference worker."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from cellvector.domain.models import AnnotationDocument
from cellvector.execution.base import PredictRequest, PredictResult
from cellvector.execution.local import LocalExecutionBackend
from cellvector.inference.nnunet.commands import build_plan_command, build_train_command
from cellvector.inference.nnunet.contracts import ExperimentSpec


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequestManifest(ApiModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_image_uri: str
    method: Literal["fiji_reconstruction", "hessian_vector"]
    frame_index: int = Field(ge=0)


class TrainRequestManifest(ApiModel):
    dataset_id: int = Field(ge=1, le=999)
    architecture: Literal["plainconv", "resenc_l"]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_smoke_test: bool = False


class PromotionRequest(ApiModel):
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class JobState(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class JobAccepted(ApiModel):
    job_id: UUID
    status: JobState
    server_instance_id: UUID


class JobStatus(ApiModel):
    job_id: UUID
    status: JobState
    server_instance_id: UUID
    error_code: str | None = None
    error_message: str | None = None


class JobArtifact(ApiModel):
    sha256: str
    predict_result: dict


class TrainingJobArtifact(ApiModel):
    sha256: str
    job_type: Literal["train_contract"] = "train_contract"
    commands: list[list[str]]
    snapshot_hash: str
    software_smoke_test: bool


@dataclass
class _JobRecord:
    job_id: UUID
    manifest: PredictRequestManifest
    image_path: Path
    status: JobState = JobState.QUEUED
    result: PredictResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    future: Future[None] | None = None


@dataclass
class _TrainingRecord:
    job_id: UUID
    manifest: TrainRequestManifest
    status: JobState = JobState.QUEUED
    artifact: TrainingJobArtifact | None = None
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    future: Future[None] | None = None


class ReferenceTrainingJobManager:
    """Validate train contracts and create reproducible command artifacts."""

    def __init__(self, server_instance_id: UUID, *, max_workers: int = 1) -> None:
        self.server_instance_id = server_instance_id
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cellvector-train")
        self._records: dict[UUID, _TrainingRecord] = {}
        self._idempotency: dict[str, tuple[UUID, str]] = {}
        self._lock = Lock()

    def submit(self, manifest: TrainRequestManifest, idempotency_key: str) -> JobAccepted:
        if not idempotency_key:
            raise ValueError("idempotency key is required")
        signature = sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()
        with self._lock:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                job_id, old_signature = existing
                if signature != old_signature:
                    raise ValueError("IDEMPOTENCY_CONFLICT")
                return JobAccepted(job_id=job_id, status=self._records[job_id].status, server_instance_id=self.server_instance_id)
            job_id = uuid4()
            record = _TrainingRecord(job_id=job_id, manifest=manifest)
            self._records[job_id] = record
            self._idempotency[idempotency_key] = (job_id, signature)
            record.future = self._executor.submit(self._run, job_id)
            return JobAccepted(job_id=job_id, status=record.status, server_instance_id=self.server_instance_id)

    def status(self, job_id: UUID) -> JobStatus:
        with self._lock:
            record = self._get(job_id)
            return self._status(record)

    def cancel(self, job_id: UUID) -> JobStatus:
        with self._lock:
            record = self._get(job_id)
            if record.status in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                return self._status(record)
            record.cancel_requested = True
            if record.future is not None and record.future.cancel():
                record.status = JobState.CANCELLED
            else:
                record.status = JobState.CANCELLING
            return self._status(record)

    def artifact(self, job_id: UUID) -> TrainingJobArtifact:
        with self._lock:
            record = self._get(job_id)
            if record.status != JobState.SUCCEEDED or record.artifact is None:
                raise ValueError("ARTIFACT_NOT_READY")
            return record.artifact

    def _run(self, job_id: UUID) -> None:
        try:
            with self._lock:
                record = self._get(job_id)
                record.status = JobState.VALIDATING
                manifest = record.manifest
                if record.cancel_requested:
                    record.status = JobState.CANCELLED
                    return
            spec = (
                ExperimentSpec.plainconv(manifest.dataset_id)
                if manifest.architecture == "plainconv"
                else ExperimentSpec.resenc_l(manifest.dataset_id)
            )
            commands = [build_plan_command(spec)] + [
                build_train_command(spec, fold=fold) for fold in spec.folds
            ]
            payload = {
                "job_type": "train_contract",
                "commands": commands,
                "snapshot_hash": manifest.snapshot_hash,
                "software_smoke_test": manifest.software_smoke_test,
            }
            checksum = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            artifact = TrainingJobArtifact(sha256=checksum, **payload)
            with self._lock:
                record = self._get(job_id)
                if record.cancel_requested:
                    record.status = JobState.CANCELLED
                else:
                    record.artifact = artifact
                    record.status = JobState.SUCCEEDED
        except Exception as error:
            with self._lock:
                record = self._get(job_id)
                record.status = JobState.FAILED
                record.error_code = "INTERNAL_ERROR"
                record.error_message = str(error)

    def _get(self, job_id: UUID) -> _TrainingRecord:
        try:
            return self._records[job_id]
        except KeyError as error:
            raise KeyError(job_id) from error

    def _status(self, record: _TrainingRecord) -> JobStatus:
        return JobStatus(
            job_id=record.job_id,
            status=record.status,
            server_instance_id=self.server_instance_id,
            error_code=record.error_code,
            error_message=record.error_message,
        )


class ReferenceJobManager:
    def __init__(self, storage_dir: Path, *, max_workers: int = 2) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.server_instance_id = uuid4()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cellvector-worker",
        )
        self._records: dict[UUID, _JobRecord] = {}
        self._idempotency: dict[str, tuple[UUID, str]] = {}
        self._lock = Lock()

    def submit(
        self,
        manifest: PredictRequestManifest,
        image_path: Path,
        idempotency_key: str,
    ) -> JobAccepted:
        if not idempotency_key:
            raise ValueError("idempotency key is required")
        signature = sha256(
            manifest.model_dump_json().encode("utf-8")
        ).hexdigest()
        with self._lock:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                existing_job_id, existing_signature = existing
                if existing_signature != signature:
                    raise ValueError("IDEMPOTENCY_CONFLICT")
                record = self._records[existing_job_id]
                return JobAccepted(
                    job_id=record.job_id,
                    status=record.status,
                    server_instance_id=self.server_instance_id,
                )
            job_id = uuid4()
            record = _JobRecord(job_id=job_id, manifest=manifest, image_path=image_path)
            self._records[job_id] = record
            self._idempotency[idempotency_key] = (job_id, signature)
            record.future = self._executor.submit(self._run, job_id)
            return JobAccepted(
                job_id=job_id,
                status=record.status,
                server_instance_id=self.server_instance_id,
            )

    def status(self, job_id: UUID) -> JobStatus:
        with self._lock:
            record = self._get(job_id)
            return JobStatus(
                job_id=record.job_id,
                status=record.status,
                server_instance_id=self.server_instance_id,
                error_code=record.error_code,
                error_message=record.error_message,
            )

    def cancel(self, job_id: UUID) -> JobStatus:
        with self._lock:
            record = self._get(job_id)
            if record.status in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                return self.status_unlocked(record)
            record.cancel_requested = True
            if record.future is not None and record.future.cancel():
                record.status = JobState.CANCELLED
            else:
                record.status = JobState.CANCELLING
            return self.status_unlocked(record)

    def artifact(self, job_id: UUID) -> JobArtifact:
        with self._lock:
            record = self._get(job_id)
            if record.status != JobState.SUCCEEDED or record.result is None:
                raise ValueError("ARTIFACT_NOT_READY")
            payload = record.result.model_dump(mode="json")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return JobArtifact(
            sha256=sha256(serialized).hexdigest(),
            predict_result=payload,
        )

    def status_unlocked(self, record: _JobRecord) -> JobStatus:
        return JobStatus(
            job_id=record.job_id,
            status=record.status,
            server_instance_id=self.server_instance_id,
            error_code=record.error_code,
            error_message=record.error_message,
        )

    def _run(self, job_id: UUID) -> None:
        try:
            self._set_state(job_id, JobState.VALIDATING)
            record = self._record(job_id)
            if record.cancel_requested:
                self._set_state(job_id, JobState.CANCELLED)
                return
            actual = sha256(record.image_path.read_bytes()).hexdigest()
            if actual != record.manifest.source_sha256:
                self._fail(job_id, "CHECKSUM_MISMATCH", "Stored upload checksum changed.")
                return
            self._set_state(job_id, JobState.RUNNING)
            result = LocalExecutionBackend().predict(
                PredictRequest(
                    image_path=record.image_path,
                    method=record.manifest.method,
                    frame_index=record.manifest.frame_index,
                )
            )
            source = result.document.source.model_copy(
                update={"image_uri": record.manifest.original_image_uri}
            )
            document = AnnotationDocument.model_validate(
                {**result.document.model_dump(), "source": source}
            )
            result = result.model_copy(update={"document": document})
            with self._lock:
                current = self._get(job_id)
                if current.cancel_requested:
                    current.status = JobState.CANCELLED
                else:
                    current.result = result
                    current.status = JobState.SUCCEEDED
        except Exception as error:
            self._fail(job_id, "INTERNAL_ERROR", str(error))

    def _set_state(self, job_id: UUID, state: JobState) -> None:
        with self._lock:
            self._get(job_id).status = state

    def _fail(self, job_id: UUID, code: str, message: str) -> None:
        with self._lock:
            record = self._get(job_id)
            record.status = JobState.FAILED
            record.error_code = code
            record.error_message = message

    def _record(self, job_id: UUID) -> _JobRecord:
        with self._lock:
            return self._get(job_id)

    def _get(self, job_id: UUID) -> _JobRecord:
        try:
            return self._records[job_id]
        except KeyError as error:
            raise KeyError("JOB_NOT_FOUND") from error
