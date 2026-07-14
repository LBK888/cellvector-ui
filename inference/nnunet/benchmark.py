from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from skimage.morphology import dilation, disk, skeletonize

from .contracts import ContractModel


class BenchmarkCaseResult(ContractModel):
    case_id: str = Field(min_length=1)
    metrics: dict[str, float]


class BenchmarkReport(ContractModel):
    report_id: UUID
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: UUID
    software_smoke_test: bool = False
    cases: tuple[BenchmarkCaseResult, ...]
    aggregate: dict[str, float]
    created_at: datetime


def _dice(truth: NDArray[np.bool_], prediction: NDArray[np.bool_]) -> float:
    denominator = int(truth.sum() + prediction.sum())
    return 1.0 if denominator == 0 else 2.0 * int((truth & prediction).sum()) / denominator


def _iou(truth: NDArray[np.bool_], prediction: NDArray[np.bool_]) -> float:
    union = int((truth | prediction).sum())
    return 1.0 if union == 0 else int((truth & prediction).sum()) / union


def _precision_recall(
    truth: NDArray[np.bool_], prediction: NDArray[np.bool_]
) -> tuple[float, float]:
    true_positive = int((truth & prediction).sum())
    predicted = int(prediction.sum())
    actual = int(truth.sum())
    precision = 1.0 if predicted == 0 and actual == 0 else (
        0.0 if predicted == 0 else true_positive / predicted
    )
    recall = 1.0 if actual == 0 and predicted == 0 else (
        0.0 if actual == 0 else true_positive / actual
    )
    return precision, recall


def _boundary_f1(
    truth: NDArray[np.bool_], prediction: NDArray[np.bool_], tolerance: int = 1
) -> float:
    if not truth.any() and not prediction.any():
        return 1.0
    footprint = disk(tolerance)
    matched_prediction = int((prediction & dilation(truth, footprint)).sum())
    matched_truth = int((truth & dilation(prediction, footprint)).sum())
    precision = matched_prediction / int(prediction.sum()) if prediction.any() else 0.0
    recall = matched_truth / int(truth.sum()) if truth.any() else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def benchmark_case(
    truth: NDArray[np.uint8], prediction: NDArray[np.uint8]
) -> dict[str, float]:
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    region_truth, region_prediction = truth == 1, prediction == 1
    membrane_truth, membrane_prediction = truth == 2, prediction == 2
    ridge_truth, ridge_prediction = truth == 3, prediction == 3
    ridge_precision, ridge_recall = _precision_recall(ridge_truth, ridge_prediction)
    truth_length = int(skeletonize(ridge_truth).sum())
    predicted_length = int(skeletonize(ridge_prediction).sum())
    length_error = (
        (0.0 if predicted_length == 0 else 1.0)
        if truth_length == 0
        else abs(predicted_length - truth_length) / truth_length
    )
    return {
        "cell_region_dice": _dice(region_truth, region_prediction),
        "cell_region_iou": _iou(region_truth, region_prediction),
        "cell_membrane_dice": _dice(membrane_truth, membrane_prediction),
        "cell_membrane_boundary_f1": _boundary_f1(
            membrane_truth, membrane_prediction
        ),
        "microridge_dice": _dice(ridge_truth, ridge_prediction),
        "microridge_precision": ridge_precision,
        "microridge_recall": ridge_recall,
        "microridge_skeleton_length_error": length_error,
    }


def benchmark_dataset(
    cases: Mapping[
        str,
        tuple[NDArray[np.uint8], NDArray[np.uint8]],
    ],
    *,
    snapshot_hash: str,
    model_id: UUID,
    software_smoke_test: bool,
) -> BenchmarkReport:
    """Create an immutable per-case and mean-metric benchmark report."""

    if not cases:
        raise ValueError("at least one benchmark case is required")
    results = tuple(
        BenchmarkCaseResult(
            case_id=case_id,
            metrics=benchmark_case(truth, prediction),
        )
        for case_id, (truth, prediction) in sorted(cases.items())
    )
    metric_names = results[0].metrics.keys()
    aggregate = {
        name: float(np.mean([result.metrics[name] for result in results]))
        for name in metric_names
    }
    return BenchmarkReport(
        report_id=uuid4(),
        snapshot_hash=snapshot_hash,
        model_id=model_id,
        software_smoke_test=software_smoke_test,
        cases=results,
        aggregate=aggregate,
        created_at=datetime.now(timezone.utc),
    )
