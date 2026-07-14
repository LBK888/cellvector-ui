from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cellvector.datasets.lineage import WP3Error


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ExtensionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopologyKind(StrEnum):
    SOFT_CLDICE = "soft_cldice"
    THIN_LINE_SAMPLING = "thin_line_sampling"
    MULTI_HEAD = "multi_head"


class ActivationState(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class TopologyExperimentManifest(ExtensionModel):
    schema_version: str = "1.0.0"
    experiment_id: UUID
    kind: TopologyKind
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: UUID
    state: ActivationState = ActivationState.DISABLED
    baseline_benchmark_ids: tuple[UUID, ...] = ()
    observed_metric: str | None = None
    regression_failed: bool = False
    software_smoke_evidence: bool = True
    actor: str | None = None
    rationale: str | None = None


def validate_topology_activation(
    manifest: TopologyExperimentManifest,
) -> TopologyExperimentManifest:
    if manifest.state == ActivationState.DISABLED:
        return manifest
    if (
        not manifest.baseline_benchmark_ids
        or not manifest.observed_metric
        or not manifest.regression_failed
        or manifest.software_smoke_evidence
        or not (manifest.actor or "").strip()
        or not (manifest.rationale or "").strip()
    ):
        raise WP3Error(
            "TOPOLOGY_ACTIVATION_REQUIRES_EVIDENCE",
            "enabled topology experiments require real failing baseline evidence and human authorization",
        )
    return manifest


class SyntheticValidationState(StrEnum):
    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    FAILED = "failed"


class SyntheticAllowedUse(StrEnum):
    SOFTWARE_FIXTURE = "software_fixture"
    BIOLOGICAL_TRAINING = "biological_training"
    FROZEN_TEST = "frozen_test"


class SyntheticAdapterManifest(ExtensionModel):
    schema_version: str = "1.0.0"
    adapter_id: UUID
    generator_name: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    seed_policy: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    source_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optical_validation: SyntheticValidationState = SyntheticValidationState.NOT_VALIDATED
    validator: str | None = None
    evidence_sha256: dict[str, Sha256] = Field(default_factory=dict)
    allowed_use: SyntheticAllowedUse = SyntheticAllowedUse.SOFTWARE_FIXTURE


def validate_synthetic_use(
    manifest: SyntheticAdapterManifest,
    requested_use: SyntheticAllowedUse,
) -> bool:
    if requested_use == SyntheticAllowedUse.FROZEN_TEST:
        raise WP3Error(
            "SYNTHETIC_FROZEN_TEST_FORBIDDEN",
            "synthetic artifacts can never enter frozen-test",
        )
    if requested_use == SyntheticAllowedUse.BIOLOGICAL_TRAINING and (
        manifest.allowed_use != SyntheticAllowedUse.BIOLOGICAL_TRAINING
        or manifest.optical_validation != SyntheticValidationState.PASSED
        or not (manifest.validator or "").strip()
        or not manifest.evidence_sha256
    ):
        raise WP3Error(
            "SYNTHETIC_NOT_OPTICALLY_VALIDATED",
            "biological training requires passed optical validation evidence",
        )
    return True
