"""Public analysis use case."""

from cellvector.execution.base import ExecutionBackend, PredictRequest, PredictResult


def analyze(request: PredictRequest, backend: ExecutionBackend) -> PredictResult:
    return backend.predict(request)

