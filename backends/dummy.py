from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .contracts import (
    BackendCapabilities,
    CommandSpec,
    DatasetArtifact,
    ExpectedArtifact,
    ModelArtifact,
    ModelStatus,
    PredictionArtifact,
    PredictionCaseArtifact,
    PredictionRequest,
    TrainingRequest,
)
from .registry import BackendError


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CHECKPOINT_SHA256 = sha256(b"cellvector-dummy-checkpoint-v1").hexdigest()


class DummyBackend:
    backend_id = "dummy"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            backend_version="1",
            input_dimensions=(2,),
            channel_counts=(1,),
            label_contract="cellvector.annotation/1.0.0",
            architectures=("identity",),
            supports_resume=False,
            supports_probabilities=True,
            owns_augmentation=False,
        )

    def plan_training(
        self,
        request: TrainingRequest,
        dataset: DatasetArtifact,
        augmentation: Any,
    ) -> tuple[CommandSpec, ...]:
        self._validate_training(request, dataset)
        return (
            CommandSpec(
                operation="dummy.identity.train",
                argv=("cellvector-dummy", "train", str(request.request_id)),
                workdir=".",
                expected_artifacts=(
                    ExpectedArtifact(path="model/checkpoint.json", required=True),
                ),
            ),
        )

    def collect_training(
        self,
        job: Any,
        request: TrainingRequest,
        dataset: DatasetArtifact,
    ) -> ModelArtifact:
        self._validate_training(request, dataset)
        model_id = uuid5(NAMESPACE_URL, f"cellvector:dummy:model:{request.request_id}")
        return ModelArtifact(
            model_id=model_id,
            status=ModelStatus.TRAINED,
            backend_id=self.backend_id,
            backend_version=self.capabilities().backend_version,
            architecture=request.architecture,
            configuration=request.configuration,
            snapshot_hash=request.snapshot_hash,
            dataset_artifact_hash=request.dataset_artifact_hash,
            augmentation_profile_hash=request.augmentation_profile_hash,
            input_contract="image/2d-single-channel",
            label_contract=dataset.label_contract,
            folds_completed=request.folds,
            checkpoint_path="model/checkpoint.json",
            checkpoint_sha256=_CHECKPOINT_SHA256,
            artifact_sha256={"model/checkpoint.json": _CHECKPOINT_SHA256},
            training_job_id=job.job_id,
            created_at=_EPOCH,
            updated_at=_EPOCH,
            software_smoke_test=request.software_smoke_test,
            parent_model_id=request.parent_model_id,
        )

    def plan_prediction(
        self,
        request: PredictionRequest,
        model: ModelArtifact,
        workdir: Path,
    ) -> tuple[CommandSpec, ...]:
        self._validate_prediction(request, model)
        return (
            CommandSpec(
                operation="dummy.identity.predict",
                argv=("cellvector-dummy", "predict", str(request.request_id)),
                workdir=str(workdir),
                expected_artifacts=(
                    ExpectedArtifact(path="predictions/manifest.json", required=True),
                ),
            ),
        )

    def collect_prediction(
        self,
        job: Any,
        request: PredictionRequest,
        model: ModelArtifact,
    ) -> PredictionArtifact:
        self._validate_prediction(request, model)
        cases: list[PredictionCaseArtifact] = []
        checksums: dict[str, str] = {}
        for source in request.source_cases:
            stem = uuid5(
                NAMESPACE_URL,
                f"cellvector:dummy:prediction:{request.request_id}:{source.case_id}",
            ).hex
            label_path = f"predictions/{stem}.label.json"
            label_sha = sha256(
                f"{source.source_sha256}:{source.frame_index}".encode("ascii")
            ).hexdigest()
            probability_path = None
            probability_sha = None
            if request.save_probabilities:
                probability_path = f"predictions/{stem}.probability.json"
                probability_sha = sha256(f"probability:{label_sha}".encode("ascii")).hexdigest()
                checksums[probability_path] = probability_sha
            checksums[label_path] = label_sha
            cases.append(
                PredictionCaseArtifact(
                    case_id=source.case_id,
                    source_sha256=source.source_sha256,
                    frame_index=source.frame_index,
                    width_px=source.width_px,
                    height_px=source.height_px,
                    label_path=label_path,
                    label_sha256=label_sha,
                    probability_path=probability_path,
                    probability_sha256=probability_sha,
                )
            )
        return PredictionArtifact(
            artifact_id=uuid5(
                NAMESPACE_URL, f"cellvector:dummy:prediction:{request.request_id}"
            ),
            backend_id=self.backend_id,
            model_id=model.model_id,
            job_id=job.job_id,
            cases=tuple(cases),
            artifact_sha256=checksums,
        )

    def _validate_training(
        self,
        request: TrainingRequest,
        dataset: DatasetArtifact,
    ) -> None:
        if request.backend_id != self.backend_id or dataset.backend_id != self.backend_id:
            raise BackendError("BACKEND_ID_MISMATCH", self.backend_id)
        if request.architecture not in self.capabilities().architectures:
            raise BackendError(
                "BACKEND_ARCHITECTURE_UNSUPPORTED",
                f"{self.backend_id}:{request.architecture}",
            )

    def _validate_prediction(
        self,
        request: PredictionRequest,
        model: ModelArtifact,
    ) -> None:
        if request.backend_id != self.backend_id or model.backend_id != self.backend_id:
            raise BackendError("BACKEND_ID_MISMATCH", self.backend_id)
        if request.model_id != model.model_id:
            raise BackendError("MODEL_ID_MISMATCH", str(request.model_id))
