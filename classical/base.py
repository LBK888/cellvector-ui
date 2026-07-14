"""Shared classical-pipeline normalization and proposal conversion."""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from cellvector import __version__
from cellvector.domain.models import MicroridgeFeature, PredictionProposal, Provenance
from cellvector.io.images import ImportedFrame
from cellvector.vectorize.skeleton import trace_skeleton


def normalize_percentile(array: NDArray[np.generic]) -> tuple[NDArray[np.float64], list[str]]:
    values = np.asarray(array, dtype=np.float64)
    low, high = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= np.finfo(float).eps:
        return np.zeros_like(values), ["LOW_DYNAMIC_RANGE"]
    return np.clip((values - low) / (high - low), 0.0, 1.0), []


def proposal_from_skeleton(
    frame: ImportedFrame,
    skeleton: NDArray[np.bool_],
    *,
    method: str,
    parameters: Mapping[str, Any],
    warnings: list[str] | None = None,
    min_path_px: float = 0.0,
) -> PredictionProposal:
    provenance = Provenance(
        method=method,
        implementation_version=__version__,
        parameters=dict(parameters),
    )
    features: list[MicroridgeFeature] = []
    for path in trace_skeleton(skeleton):
        length = sum(
            hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(path, path[1:])
        )
        if length < min_path_px:
            continue
        features.append(
            MicroridgeFeature(
                id=uuid4(),
                points=path,
                closed=path[0] == path[-1],
                provenance=provenance,
            )
        )
    return PredictionProposal(
        id=uuid4(),
        source_sha256=frame.source.sha256,
        provenance=provenance,
        microridges=features,
        warnings=list(warnings or []),
    )


def empty_proposal(
    frame: ImportedFrame,
    *,
    method: str,
    parameters: Mapping[str, Any],
    warnings: list[str],
) -> PredictionProposal:
    return proposal_from_skeleton(
        frame,
        np.zeros(frame.array.shape, dtype=bool),
        method=method,
        parameters=parameters,
        warnings=warnings,
    )

