"""HTTP client adapter for a CellVector FastAPI worker."""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx

from cellvector.inference.nnunet.contracts import ExperimentSpec

from .base import BackendHealth, PredictRequest, PredictResult


class RemoteExecutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RemoteExecutionBackend:
    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        timeout_seconds: float = 2.0,
        poll_interval_seconds: float = 0.1,
        job_timeout_seconds: float = 300.0,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.job_timeout_seconds = job_timeout_seconds
        self._uses_injected_client = client is not None
        self.client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def health(self) -> BackendHealth:
        try:
            response = self.client.get(
                f"{self.base_url}/health",
                **self._timeout_kwargs(self.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
            return BackendHealth(
                status="available",
                detail="FastAPI worker reachable",
                server_instance_id=payload.get("server_instance_id"),
            )
        except Exception as error:
            return BackendHealth(status="offline", detail=str(error))

    def predict(self, request: PredictRequest) -> PredictResult:
        image_path = Path(request.image_path).resolve()
        content = image_path.read_bytes()
        digest = sha256(content).hexdigest()
        manifest = {
            "source_sha256": digest,
            "original_image_uri": str(image_path),
            "method": request.method,
            "frame_index": request.frame_index,
        }
        accepted = self.client.post(
            f"{self.base_url}/v1/jobs/predict",
            headers=self._headers,
            files={"image": (image_path.name, content, "application/octet-stream")},
            data={
                "manifest": json.dumps(manifest),
                "idempotency_key": str(uuid4()),
            },
            **self._timeout_kwargs(max(self.timeout_seconds, 10.0)),
        )
        self._raise_api_error(accepted)
        job_id = accepted.json()["job_id"]
        deadline = time.monotonic() + self.job_timeout_seconds
        while time.monotonic() < deadline:
            status_response = self.client.get(
                f"{self.base_url}/v1/jobs/{job_id}",
                headers=self._headers,
                **self._timeout_kwargs(self.timeout_seconds),
            )
            self._raise_api_error(status_response)
            status_payload = status_response.json()
            state = status_payload["status"]
            if state == "succeeded":
                break
            if state in {"failed", "cancelled"}:
                raise RemoteExecutionError(
                    status_payload.get("error_code") or "JOB_CANCELLED",
                    status_payload.get("error_message") or state,
                )
            time.sleep(self.poll_interval_seconds)
        else:
            raise RemoteExecutionError("SERVER_UNAVAILABLE", "Remote job timed out.")

        artifact_response = self.client.get(
            f"{self.base_url}/v1/jobs/{job_id}/artifacts",
            headers=self._headers,
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(artifact_response)
        artifact = artifact_response.json()
        serialized = json.dumps(
            artifact["predict_result"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if sha256(serialized).hexdigest() != artifact["sha256"]:
            raise RemoteExecutionError(
                "CHECKSUM_MISMATCH",
                "Downloaded result checksum does not match the artifact manifest.",
            )
        return PredictResult.model_validate(artifact["predict_result"])

    def train_contract(
        self,
        spec: ExperimentSpec,
        *,
        snapshot_hash: str,
        software_smoke_test: bool,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Submit a remote nnU-Net train contract and verify its artifact."""

        accepted = self.submit_train(
            spec,
            snapshot_hash=snapshot_hash,
            software_smoke_test=software_smoke_test,
            idempotency_key=idempotency_key,
        )
        job_id = accepted["job_id"]
        deadline = time.monotonic() + self.job_timeout_seconds
        while time.monotonic() < deadline:
            status_payload = self.job_status(job_id)
            if status_payload["status"] == "succeeded":
                break
            if status_payload["status"] in {"failed", "cancelled"}:
                raise RemoteExecutionError(
                    status_payload.get("error_code") or "JOB_CANCELLED",
                    status_payload.get("error_message") or status_payload["status"],
                )
            time.sleep(self.poll_interval_seconds)
        else:
            raise RemoteExecutionError("SERVER_UNAVAILABLE", "Remote job timed out.")
        artifact_response = self.client.get(
            f"{self.base_url}/v1/jobs/{job_id}/artifacts",
            headers=self._headers,
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(artifact_response)
        artifact = artifact_response.json()
        checksum_payload = {
            key: artifact[key]
            for key in (
                "job_type",
                "commands",
                "snapshot_hash",
                "software_smoke_test",
            )
        }
        serialized = json.dumps(
            checksum_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        if sha256(serialized).hexdigest() != artifact["sha256"]:
            raise RemoteExecutionError(
                "ARTIFACT_CHECKSUM_MISMATCH",
                "Downloaded training contract checksum is invalid.",
            )
        return artifact

    def submit_train(
        self,
        spec: ExperimentSpec,
        *,
        snapshot_hash: str,
        software_smoke_test: bool,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Submit without polling so UI clients can reconnect by job ID."""

        response = self.client.post(
            f"{self.base_url}/v1/jobs/train",
            headers={
                **self._headers,
                "Idempotency-Key": idempotency_key,
            },
            json={
                "dataset_id": spec.dataset_id,
                "architecture": spec.architecture,
                "snapshot_hash": snapshot_hash,
                "software_smoke_test": software_smoke_test,
            },
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(response)
        return response.json()

    def job_status(self, job_id: str | UUID) -> dict[str, Any]:
        response = self.client.get(
            f"{self.base_url}/v1/jobs/{job_id}",
            headers=self._headers,
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(response)
        return response.json()

    def cancel_job(self, job_id: str | UUID) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/v1/jobs/{job_id}/cancel",
            headers=self._headers,
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(response)
        return response.json()

    def list_models(self) -> list[dict[str, Any]]:
        response = self.client.get(
            f"{self.base_url}/v1/models",
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(response)
        return response.json()

    def promote_model(
        self,
        model_id: str | UUID,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/v1/models/{model_id}/promote",
            headers=self._headers,
            json={"actor": actor, "reason": reason},
            **self._timeout_kwargs(self.timeout_seconds),
        )
        self._raise_api_error(response)
        return response.json()

    def _timeout_kwargs(self, value: float) -> dict[str, float]:
        return {} if self._uses_injected_client else {"timeout": value}

    @staticmethod
    def _raise_api_error(response: Any) -> None:
        if 200 <= response.status_code < 300:
            return
        try:
            detail = response.json().get("detail", {})
        except Exception:
            detail = {}
        raise RemoteExecutionError(
            detail.get("code", "INTERNAL_ERROR"),
            detail.get("message", response.text),
        )
