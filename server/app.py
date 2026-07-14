"""FastAPI transport for optional LAN execution."""

from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from cellvector import __version__
from cellvector.datasets.models import DatasetError
from cellvector.inference.nnunet.registry import ModelRegistry

from .jobs import (
    PredictRequestManifest,
    PromotionRequest,
    ReferenceJobManager,
    ReferenceTrainingJobManager,
    TrainRequestManifest,
)


def create_app(
    *,
    token: str | None = None,
    storage_dir: str | Path | None = None,
) -> FastAPI:
    api_token = token or os.environ.get("CELLVECTOR_API_TOKEN")
    if not api_token:
        raise RuntimeError("CELLVECTOR_API_TOKEN is required")
    root = Path(storage_dir or Path(tempfile.gettempdir()) / "cellvector-worker")
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    manager = ReferenceJobManager(root / "jobs")
    training_manager = ReferenceTrainingJobManager(manager.server_instance_id)
    model_registry = ModelRegistry(root / "models.json")

    app = FastAPI(title="CellVector Worker", version=__version__)

    def require_token(request: Request) -> None:
        if request.headers.get("Authorization") != f"Bearer {api_token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_FAILED", "message": "Invalid bearer token."},
            )

    def job_or_404(operation):
        try:
            return operation()
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "JOB_NOT_FOUND", "message": "Unknown job identifier."},
            ) from error

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "server_instance_id": str(manager.server_instance_id)}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, object]:
        return {
            "api_version": "1",
            "cellvector_version": __version__,
            "annotation_schema_versions": ["1.0.0"],
            "execution_modes": [
                "classical_reference",
                "nnunet_train_contract",
            ],
            "formats": ["png", "tif", "tiff"],
            "server_instance_id": str(manager.server_instance_id),
        }

    @app.get("/v1/models")
    def models() -> list[object]:
        return model_registry.list()

    @app.post("/v1/models/{model_id}/promote")
    def promote_model(model_id: UUID, promotion: PromotionRequest, request: Request):
        require_token(request)
        try:
            return model_registry.promote(
                model_id,
                actor=promotion.actor,
                reason=promotion.reason,
            )
        except DatasetError as error:
            status_code = 404 if error.code == "MODEL_MISSING" else 409
            raise HTTPException(
                status_code=status_code,
                detail={"code": error.code, "message": error.message},
            ) from error

    @app.post("/v1/jobs/predict", status_code=202)
    async def submit_predict(
        request: Request,
        image: UploadFile,
        manifest: str = Form(...),
        idempotency_key: str = Form(...),
    ):
        require_token(request)
        try:
            parsed = PredictRequestManifest.model_validate_json(manifest)
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_REQUEST", "message": str(error)},
            ) from error
        content = await image.read()
        actual = sha256(content).hexdigest()
        if actual != parsed.source_sha256:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "CHECKSUM_MISMATCH",
                    "message": "Uploaded bytes do not match the manifest checksum.",
                },
            )
        suffix = Path(image.filename or ".bin").suffix.lower()
        target = uploads / f"{actual}{suffix}"
        # The upload path is content-addressed. Rewriting it for an idempotent
        # retry can briefly truncate the file while the original background
        # job is reading it, especially on Windows. Once present, identical
        # bytes do not need to be written again.
        if not target.exists():
            target.write_bytes(content)
        try:
            return manager.submit(parsed, target, idempotency_key)
        except ValueError as error:
            if str(error) == "IDEMPOTENCY_CONFLICT":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "IDEMPOTENCY_CONFLICT",
                        "message": "Idempotency key was already used for a different request.",
                    },
                ) from error
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_REQUEST", "message": str(error)},
            ) from error

    @app.post("/v1/jobs/train", status_code=202)
    def submit_train(
        manifest: TrainRequestManifest,
        request: Request,
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ):
        require_token(request)
        try:
            return training_manager.submit(manifest, idempotency_key)
        except ValueError as error:
            if str(error) == "IDEMPOTENCY_CONFLICT":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "IDEMPOTENCY_CONFLICT",
                        "message": "Idempotency key was already used for a different request.",
                    },
                ) from error
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_REQUEST", "message": str(error)},
            ) from error

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: UUID, request: Request):
        require_token(request)
        try:
            return manager.status(job_id)
        except KeyError:
            return job_or_404(lambda: training_manager.status(job_id))

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: UUID, request: Request):
        require_token(request)
        try:
            return manager.cancel(job_id)
        except KeyError:
            return job_or_404(lambda: training_manager.cancel(job_id))

    @app.get("/v1/jobs/{job_id}/artifacts")
    def get_artifacts(job_id: UUID, request: Request):
        require_token(request)
        try:
            return manager.artifact(job_id)
        except KeyError as error:
            try:
                return training_manager.artifact(job_id)
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "JOB_NOT_FOUND", "message": "Unknown job identifier."},
                ) from error
            except ValueError as artifact_error:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "ARTIFACT_NOT_READY",
                        "message": str(artifact_error),
                    },
                ) from artifact_error
        except ValueError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "ARTIFACT_NOT_READY", "message": str(error)},
            ) from error

    return app
