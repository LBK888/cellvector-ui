"""Local and remote execution adapters."""

from .base import ExecutionBackend, PredictRequest, PredictResult
from .local import LocalExecutionBackend

__all__ = [
    "ExecutionBackend",
    "LocalExecutionBackend",
    "PredictRequest",
    "PredictResult",
]

