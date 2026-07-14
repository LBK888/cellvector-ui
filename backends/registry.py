from __future__ import annotations

from typing import Any

from .contracts import BackendCapabilities, PredictionRequest, TrainingRequest


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Any] = {}

    def register(self, backend: Any) -> Any:
        capabilities = self._capabilities(backend)
        declared_id = getattr(backend, "backend_id", capabilities.backend_id)
        if declared_id != capabilities.backend_id:
            raise BackendError(
                "BACKEND_CAPABILITY_ID_MISMATCH",
                f"{declared_id}:{capabilities.backend_id}",
            )
        if capabilities.backend_id in self._backends:
            raise BackendError("BACKEND_ALREADY_REGISTERED", capabilities.backend_id)
        self._backends[capabilities.backend_id] = backend
        return backend

    def get(self, backend_id: str) -> Any:
        try:
            backend = self._backends[backend_id]
        except KeyError as error:
            raise BackendError("BACKEND_NOT_REGISTERED", backend_id) from error
        capabilities = self._capabilities(backend)
        if capabilities.backend_id != backend_id:
            raise BackendError(
                "BACKEND_CAPABILITY_ID_MISMATCH",
                f"{backend_id}:{capabilities.backend_id}",
            )
        return backend

    def validate_training_request(self, request: TrainingRequest) -> Any:
        backend = self.get(request.backend_id)
        capabilities = self._capabilities(backend)
        if request.architecture not in capabilities.architectures:
            raise BackendError(
                "BACKEND_ARCHITECTURE_UNSUPPORTED",
                f"{request.backend_id}:{request.architecture}",
            )
        return backend

    def validate_prediction_request(self, request: PredictionRequest) -> Any:
        backend = self.get(request.backend_id)
        capabilities = self._capabilities(backend)
        if request.save_probabilities and not capabilities.supports_probabilities:
            raise BackendError(
                "BACKEND_PROBABILITIES_UNSUPPORTED",
                request.backend_id,
            )
        return backend

    @staticmethod
    def _capabilities(backend: Any) -> BackendCapabilities:
        capabilities = backend.capabilities()
        if not isinstance(capabilities, BackendCapabilities):
            raise BackendError(
                "BACKEND_CAPABILITIES_INVALID",
                type(capabilities).__name__,
            )
        return capabilities
