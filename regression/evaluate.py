from __future__ import annotations

from datetime import datetime, timezone
from math import isclose
from typing import Mapping
from uuid import uuid4

from .models import (
    Comparison,
    EvaluationState,
    GateKind,
    GateResult,
    RegressionEvidence,
    RegressionEvaluation,
    RegressionGate,
    RegressionPolicy,
)


def _passes(gate: RegressionGate, actual: float) -> bool:
    if gate.comparison == Comparison.MINIMUM:
        return actual >= gate.threshold
    if gate.comparison == Comparison.MAXIMUM:
        return actual <= gate.threshold
    return isclose(actual, gate.threshold, abs_tol=1e-12)


def evaluate_policy(
    policy: RegressionPolicy,
    evidence: RegressionEvidence,
    metrics: Mapping[str, float],
) -> RegressionEvaluation:
    lineage_matches = (
        policy.snapshot_hash == evidence.snapshot_hash
        and policy.model_id == evidence.model_id
    )
    results: list[GateResult] = []
    for gate in policy.gates:
        if not lineage_matches:
            results.append(
                GateResult(
                    name=gate.name,
                    state=EvaluationState.NOT_EVALUABLE,
                    threshold=gate.threshold,
                    reason_code="REGRESSION_LINEAGE_MISMATCH",
                )
            )
            continue
        if gate.kind == GateKind.SCIENTIFIC and (
            evidence.software_smoke_test or not evidence.frozen_test
        ):
            results.append(
                GateResult(
                    name=gate.name,
                    state=EvaluationState.NOT_EVALUABLE,
                    threshold=gate.threshold,
                    reason_code="SCIENTIFIC_GATE_REQUIRES_REAL_DATA",
                )
            )
            continue
        actual = metrics.get(gate.metric)
        if actual is None:
            results.append(
                GateResult(
                    name=gate.name,
                    state=EvaluationState.NOT_EVALUABLE,
                    threshold=gate.threshold,
                    reason_code="REGRESSION_EVIDENCE_MISSING",
                )
            )
            continue
        results.append(
            GateResult(
                name=gate.name,
                state=(EvaluationState.PASSED if _passes(gate, actual) else EvaluationState.FAILED),
                actual=float(actual),
                threshold=gate.threshold,
            )
        )
    states = {result.state for result in results}
    if EvaluationState.NOT_EVALUABLE in states or not results:
        overall = EvaluationState.NOT_EVALUABLE
    elif EvaluationState.FAILED in states:
        overall = EvaluationState.FAILED
    else:
        overall = EvaluationState.PASSED
    return RegressionEvaluation(
        evaluation_id=uuid4(),
        policy_id=policy.policy_id,
        state=overall,
        gates=tuple(results),
        created_at=datetime.now(timezone.utc),
    )
