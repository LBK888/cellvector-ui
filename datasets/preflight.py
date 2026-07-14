from __future__ import annotations

from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
import re
from typing import Any, Mapping, Self
from urllib.parse import unquote, urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cellvector.backends.contracts import BackendCapabilities, Sha256
from cellvector.domain.models import AnnotationDocument
from cellvector.domain.validation import validate_document
from cellvector.io.images import import_frames

from .models import DatasetSample, DatasetSnapshot, SampleSourceKind, SplitName
from .training_profile import DatasetPurpose, TrainingDatasetProfile


_MACHINE_QC_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class PreflightModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        payload = self.model_dump()
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class PreflightIssue(PreflightModel):
    severity: IssueSeverity
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    message: str = Field(min_length=1)
    sample_id: str | None = None
    revision_id: UUID | None = None
    stage: str = "dataset_preflight"
    recommended_action: str | None = None


class DatasetSizeSummary(PreflightModel):
    sample_count: int = Field(ge=0)
    width_min: int | None = Field(default=None, gt=0)
    width_max: int | None = Field(default=None, gt=0)
    height_min: int | None = Field(default=None, gt=0)
    height_max: int | None = Field(default=None, gt=0)
    area_min: int | None = Field(default=None, gt=0)
    area_max: int | None = Field(default=None, gt=0)
    aspect_ratio_min: float | None = Field(default=None, gt=0)
    aspect_ratio_max: float | None = Field(default=None, gt=0)
    width_distribution: tuple[tuple[int, int], ...] = ()
    height_distribution: tuple[tuple[int, int], ...] = ()
    area_distribution: tuple[tuple[int, int], ...] = ()
    aspect_ratio_distribution: tuple[tuple[float, int], ...] = ()
    smallest_sample_ids: tuple[str, ...] = ()
    largest_sample_ids: tuple[str, ...] = ()


class DtypeCount(PreflightModel):
    dtype: str = Field(min_length=1)
    count: int = Field(ge=1)


class ClassSummary(PreflightModel):
    cell_count: int = Field(ge=0)
    boundary_count: int = Field(ge=0)
    microridge_count: int = Field(ge=0)
    empty_sample_count: int = Field(ge=0)


class ScaleSummary(PreflightModel):
    available_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    distinct_pixel_sizes_um: tuple[tuple[float, float], ...] = ()


class SplitSourceCount(PreflightModel):
    split: SplitName
    real_count: int = Field(ge=0)
    synthetic_count: int = Field(ge=0)


class PaddingEstimate(PreflightModel):
    sample_id: str
    padding_fraction: float = Field(ge=0, le=1)


class ResourceEstimateSummary(PreflightModel):
    batch_size_range: tuple[int, int] | None = None
    vram_bytes_range: tuple[int, int] | None = None


class DatasetPreflightReport(PreflightModel):
    snapshot_hash: Sha256
    purpose: DatasetPurpose
    recommended_folds: int = Field(ge=0)
    requested_folds: int | None = Field(default=None, ge=1)
    size_summary: DatasetSizeSummary
    dtype_distribution: tuple[DtypeCount, ...]
    class_summary: ClassSummary
    scale_summary: ScaleSummary
    source_counts: tuple[SplitSourceCount, ...]
    planned_patch_size: tuple[int, int] | None = None
    padding_estimates: tuple[PaddingEstimate, ...] | None = None
    resource_estimate: ResourceEstimateSummary | None = None
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def blocking_errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.WARNING
        )


class _Issues:
    def __init__(self) -> None:
        self._items: dict[
            tuple[IssueSeverity, str, str | None, UUID | None, str | None],
            PreflightIssue,
        ] = {}

    def add(
        self,
        severity: IssueSeverity,
        code: str,
        message: str,
        sample_id: str | None = None,
        recommended_action: str | None = None,
        revision_id: UUID | None = None,
        dedupe_token: str | None = None,
    ) -> None:
        key = (severity, code, sample_id, revision_id, dedupe_token)
        self._items.setdefault(
            key,
            PreflightIssue(
                severity=severity,
                code=code,
                message=message,
                sample_id=sample_id,
                revision_id=revision_id,
                recommended_action=recommended_action,
            ),
        )

    def error(
        self,
        code: str,
        message: str,
        sample_id: str | None = None,
        recommended_action: str | None = None,
        revision_id: UUID | None = None,
        dedupe_token: str | None = None,
    ) -> None:
        self.add(
            IssueSeverity.ERROR,
            code,
            message,
            sample_id,
            recommended_action,
            revision_id,
            dedupe_token,
        )

    def warning(
        self,
        code: str,
        message: str,
        sample_id: str | None = None,
        recommended_action: str | None = None,
        revision_id: UUID | None = None,
        dedupe_token: str | None = None,
    ) -> None:
        self.add(
            IssueSeverity.WARNING,
            code,
            message,
            sample_id,
            recommended_action,
            revision_id,
            dedupe_token,
        )

    def sorted(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            sorted(
                self._items.values(),
                key=lambda issue: (
                    issue.severity.value,
                    issue.code,
                    issue.sample_id or "",
                    str(issue.revision_id or ""),
                ),
            )
        )


def preflight_dataset(
    snapshot: DatasetSnapshot,
    documents: Mapping[UUID, AnnotationDocument],
    profile: TrainingDatasetProfile,
    capabilities: BackendCapabilities,
) -> DatasetPreflightReport:
    """Validate one exact dataset selection without changing native image data."""

    issues = _Issues()
    samples = tuple(
        sorted(
            snapshot.samples,
            key=lambda sample: (sample.sample_id, str(sample.revision_id)),
        )
    )
    _validate_backend(capabilities, issues)
    _validate_split_and_source_policy(samples, profile, issues)
    for sample_id, count in sorted(Counter(sample.sample_id for sample in samples).items()):
        if count > 1:
            issues.error(
                "DUPLICATE_SAMPLE_ID",
                f"Sample identifier {sample_id} occurs {count} times.",
                sample_id,
                "Assign a unique case identifier to every selected revision.",
            )

    class_counts = Counter[str]()
    empty_sample_count = 0
    for sample in samples:
        _validate_source(sample, issues)
        document = documents.get(sample.revision_id)
        if document is None or not sample.reviewed:
            issues.error(
                "MISSING_REVIEWED_REVISION",
                "The exact reviewed annotation revision is unavailable.",
                sample.sample_id,
                "Review the annotation and recreate the dataset snapshot.",
                revision_id=sample.revision_id,
            )
            continue
        _validate_document_identity(sample, document, issues)
        validation = validate_document(document)
        for validation_issue in validation.errors:
            issues.error(
                validation_issue.code,
                validation_issue.message,
                sample.sample_id,
                revision_id=sample.revision_id,
            )
        for validation_issue in validation.warnings:
            issues.warning(
                validation_issue.code,
                validation_issue.message,
                sample.sample_id,
                revision_id=sample.revision_id,
            )
        for code in document.qc.errors:
            _add_stored_qc_issue(
                issues,
                IssueSeverity.ERROR,
                code,
                "document error",
                sample,
            )
        for code in document.qc.warnings:
            _add_stored_qc_issue(
                issues,
                IssueSeverity.WARNING,
                code,
                "document warning",
                sample,
            )
        for code in sample.qc_errors:
            _add_stored_qc_issue(
                issues,
                IssueSeverity.ERROR,
                code,
                "snapshot error",
                sample,
            )
        expected_snapshot_qc = set(document.qc.errors).union(validation.error_codes)
        if set(sample.qc_errors) != expected_snapshot_qc:
            issues.error(
                "STORED_QC_MISMATCH",
                "Snapshot QC errors disagree with the current exact document and validation result.",
                sample.sample_id,
                "Recreate the snapshot from the reviewed document.",
                revision_id=sample.revision_id,
            )
        class_counts["cell"] += len(document.cells)
        class_counts["boundary"] += len(document.boundaries)
        class_counts["microridge"] += len(document.microridges)
        if not (document.cells or document.boundaries or document.microridges):
            empty_sample_count += 1
            issues.warning(
                "EMPTY_LABELS",
                "The reviewed annotation contains no training labels.",
                sample.sample_id,
                "Confirm that an empty annotation is scientifically intended.",
                revision_id=sample.revision_id,
            )

    missing_classes = tuple(
        name
        for name in ("cell", "boundary", "microridge")
        if class_counts[name] == 0
    )
    if missing_classes:
        issues.warning(
            "MISSING_REQUIRED_CLASS",
            "No reviewed examples are present for: " + ", ".join(missing_classes),
            recommended_action="Add representative reviewed labels or record the limitation.",
        )

    eligible_fold_groups = _eligible_fold_groups(samples)
    recommended_folds = _fold_recommendation(eligible_fold_groups, capabilities)
    requested_folds = profile.fold_policy.requested_folds
    if requested_folds is not None and requested_folds > recommended_folds:
        issues.error(
            "IMPOSSIBLE_FOLD_CONFIGURATION",
            f"Requested {requested_folds} folds but only {recommended_folds} independent non-test groups are available.",
            recommended_action="Reduce folds or add independent groups.",
        )
    effective_folds = requested_folds or recommended_folds
    if recommended_folds and (recommended_folds < 5 or effective_folds < recommended_folds):
        issues.warning(
            "REDUCED_FOLDS",
            f"Only {effective_folds} fold(s) will be used; evaluation power is limited.",
        )

    size_summary = _size_summary(samples)
    dtype_counts = Counter(sample.dtype for sample in samples)
    distinct_scales = tuple(
        sorted({sample.pixel_size_um for sample in samples if sample.pixel_size_um})
    )
    missing_scale = sum(sample.pixel_size_um is None for sample in samples)
    available_scale = len(samples) - missing_scale
    if missing_scale:
        issues.warning(
            "MISSING_SCALE",
            f"Physical pixel scale is missing for {missing_scale} sample(s).",
        )
    if len(distinct_scales) > 1 or (missing_scale and available_scale):
        issues.warning(
            "INCONSISTENT_SCALE",
            "Physical scale is mixed or inconsistent across the dataset.",
        )

    source_counts = tuple(
        SplitSourceCount(
            split=split,
            real_count=sum(
                sample.split is split
                and sample.source_kind is SampleSourceKind.REAL
                for sample in samples
            ),
            synthetic_count=sum(
                sample.split is split
                and sample.source_kind is SampleSourceKind.SYNTHETIC
                for sample in samples
            ),
        )
        for split in SplitName
    )

    return DatasetPreflightReport(
        snapshot_hash=snapshot.identity_hash(),
        purpose=profile.purpose,
        recommended_folds=recommended_folds,
        requested_folds=requested_folds,
        size_summary=size_summary,
        dtype_distribution=tuple(
            DtypeCount(dtype=dtype, count=count)
            for dtype, count in sorted(dtype_counts.items())
        ),
        class_summary=ClassSummary(
            cell_count=class_counts["cell"],
            boundary_count=class_counts["boundary"],
            microridge_count=class_counts["microridge"],
            empty_sample_count=empty_sample_count,
        ),
        scale_summary=ScaleSummary(
            available_count=available_scale,
            missing_count=missing_scale,
            distinct_pixel_sizes_um=distinct_scales,
        ),
        source_counts=source_counts,
        planned_patch_size=None,
        padding_estimates=None,
        resource_estimate=None,
        issues=issues.sorted(),
    )


def _validate_backend(capabilities: BackendCapabilities, issues: _Issues) -> None:
    if 2 not in capabilities.input_dimensions:
        issues.error(
            "BACKEND_DIMENSION_UNSUPPORTED",
            "The backend does not declare support for native 2D frames.",
        )
    if 1 not in capabilities.channel_counts:
        issues.error(
            "BACKEND_CHANNEL_UNSUPPORTED",
            "The backend does not declare support for single-channel input.",
        )


def _add_stored_qc_issue(
    issues: _Issues,
    severity: IssueSeverity,
    raw_code: str,
    origin: str,
    sample: DatasetSample,
) -> None:
    if _MACHINE_QC_CODE.fullmatch(raw_code):
        code = raw_code
        message = f"Stored {origin} records blocking or advisory QC state: {raw_code}."
        dedupe_token = None
    else:
        code = "MALFORMED_STORED_QC_CODE"
        message = f"Stored {origin} contains malformed legacy QC text: {raw_code!r}."
        dedupe_token = raw_code
    issues.add(
        severity,
        code,
        message,
        sample.sample_id,
        revision_id=sample.revision_id,
        dedupe_token=dedupe_token,
    )
def _validate_split_and_source_policy(
    samples: tuple[DatasetSample, ...],
    profile: TrainingDatasetProfile,
    issues: _Issues,
) -> None:
    groups_by_split: dict[SplitName, set[str]] = {
        split: set() for split in SplitName
    }
    splits_by_group: dict[str, set[SplitName]] = defaultdict(set)
    revision_groups: dict[UUID, set[str]] = defaultdict(set)
    source_groups: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.split is None:
            issues.error(
                "UNASSIGNED_SPLIT",
                "Every dataset sample must have an explicit split assignment.",
                sample.sample_id,
                revision_id=sample.revision_id,
            )
        else:
            groups_by_split[sample.split].add(sample.group_key)
            splits_by_group[sample.group_key].add(sample.split)
        revision_groups[sample.revision_id].add(sample.group_key)
        source_groups[sample.source_sha256].add(sample.group_key)
        if sample.source_kind is SampleSourceKind.SYNTHETIC:
            if sample.split is not SplitName.TRAIN:
                issues.error(
                    "SYNTHETIC_NONTRAIN_SPLIT",
                    "Synthetic samples are prohibited from validation and frozen-test splits.",
                    sample.sample_id,
                    revision_id=sample.revision_id,
                )
            elif not profile.allow_synthetic_training:
                issues.error(
                    "SYNTHETIC_TRAINING_NOT_ALLOWED",
                    "Synthetic training requires explicit profile permission.",
                    sample.sample_id,
                    revision_id=sample.revision_id,
                )

    for group_key, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            issues.error(
                "GROUP_SPLIT_LEAKAGE",
                f"Independent group {group_key} occurs in multiple splits.",
            )
        if SplitName.FROZEN_TEST in splits and len(splits) > 1:
            issues.error(
                "FROZEN_GROUP_LEAKAGE",
                f"Frozen-test group {group_key} also occurs outside frozen-test.",
            )
    if any(len(groups) > 1 for groups in revision_groups.values()) or any(
        len(groups) > 1 for groups in source_groups.values()
    ):
        issues.error(
            "GROUP_SPLIT_LEAKAGE",
            "A source or annotation revision occurs in multiple independent groups.",
        )

    train_groups = groups_by_split[SplitName.TRAIN]
    validation_groups = groups_by_split[SplitName.VALIDATION]
    frozen_groups = groups_by_split[SplitName.FROZEN_TEST]
    if not train_groups:
        issues.error("NO_TRAIN_SPLIT", "The dataset has no training group.")

    independent_validation = validation_groups.difference(train_groups, frozen_groups)
    independent_frozen = frozen_groups.difference(train_groups, validation_groups)
    if profile.purpose is DatasetPurpose.COMPARATIVE:
        if not independent_validation:
            issues.error(
                "NO_INDEPENDENT_VALIDATION",
                "Comparative datasets require an explicit independent validation group.",
            )
        if not independent_frozen:
            issues.error(
                "NO_FROZEN_TEST",
                "Comparative datasets require an explicit independent frozen-test group.",
            )
    else:
        if not independent_validation:
            issues.warning(
                "NO_INDEPENDENT_VALIDATION",
                "No independent validation group is present.",
            )
        if not independent_frozen:
            issues.warning(
                "NO_FROZEN_TEST",
                "No frozen-test group is present; results are exploratory only.",
            )


def _validate_source(sample: DatasetSample, issues: _Issues) -> None:
    try:
        path = _source_path(sample.image_uri)
    except ValueError as error:
        issues.error(
            "SOURCE_MISSING",
            str(error),
            sample.sample_id,
            revision_id=sample.revision_id,
        )
        return
    if not path.is_file():
        issues.error(
            "SOURCE_MISSING",
            f"Source image does not exist: {sample.image_uri}",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
        return
    try:
        frames = import_frames(path)
    except Exception as error:
        issues.error(
            "SOURCE_INVALID_FRAME",
            f"Source image cannot be read as native 2D single-channel data: {error}",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
        return
    if sample.frame_index >= len(frames):
        issues.error(
            "SOURCE_FRAME_MISSING",
            f"Frame {sample.frame_index} is absent from the source image.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
        return
    actual = frames[sample.frame_index].source
    if actual.sha256 != sample.source_sha256:
        issues.error(
            "SOURCE_CHECKSUM_MISMATCH",
            "Source file checksum differs from the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    if (actual.width_px, actual.height_px) != (sample.width_px, sample.height_px):
        issues.error(
            "SOURCE_DIMENSION_MISMATCH",
            "Source frame dimensions differ from the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    if actual.dtype != sample.dtype:
        issues.error(
            "SOURCE_DTYPE_MISMATCH",
            "Source frame dtype differs from the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )


def _validate_document_identity(
    sample: DatasetSample,
    document: AnnotationDocument,
    issues: _Issues,
) -> None:
    if document.revision_id != sample.revision_id:
        issues.error(
            "REVISION_MISMATCH",
            "Document revision does not match the selected snapshot revision.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    if document.annotation_id != sample.annotation_id:
        issues.error(
            "DATASET_ARTIFACT_MISMATCH",
            "Document annotation identity does not match the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    source = document.source
    if source.sha256 != sample.source_sha256 or source.frame_index != sample.frame_index:
        issues.error(
            "DOCUMENT_SOURCE_MISMATCH",
            "Document source identity does not match the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    if (source.width_px, source.height_px) != (sample.width_px, sample.height_px):
        issues.error(
            "DOCUMENT_SOURCE_DIMENSION_MISMATCH",
            "Document source dimensions do not match the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )
    if (
        source.dtype != sample.dtype
        or source.coordinate_unit != sample.coordinate_unit
        or source.pixel_size_um != sample.pixel_size_um
    ):
        issues.error(
            "DOCUMENT_SOURCE_METADATA_MISMATCH",
            "Document source metadata does not match the dataset snapshot.",
            sample.sample_id,
            revision_id=sample.revision_id,
        )


def _source_path(image_uri: str) -> Path:
    if len(image_uri) >= 2 and image_uri[1] == ":":
        return Path(image_uri)
    parsed = urlparse(image_uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"Source URI is not a local file: {image_uri}")
    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
            path_text = path_text[1:]
        return Path(path_text)
    return Path(image_uri)


def _eligible_fold_groups(samples: tuple[DatasetSample, ...]) -> frozenset[str]:
    ineligible_groups = {
        sample.group_key
        for sample in samples
        if sample.split is None or sample.split is SplitName.FROZEN_TEST
    }
    return frozenset(
        sample.group_key
        for sample in samples
        if sample.source_kind is SampleSourceKind.REAL
        and sample.split in {SplitName.TRAIN, SplitName.VALIDATION}
        and sample.group_key not in ineligible_groups
    )


def _fold_recommendation(
    eligible_groups: frozenset[str], capabilities: BackendCapabilities
) -> int:
    limit = 5
    declared_max = (
        getattr(capabilities, "max_folds")
        if "max_folds" in type(capabilities).model_fields
        else None
    )
    if isinstance(declared_max, int) and declared_max > 0:
        limit = min(limit, declared_max)
    return min(limit, len(eligible_groups))


def _size_summary(samples: tuple[DatasetSample, ...]) -> DatasetSizeSummary:
    if not samples:
        return DatasetSizeSummary(sample_count=0)
    widths = [sample.width_px for sample in samples]
    heights = [sample.height_px for sample in samples]
    area_records = [
        (sample.sample_id, sample.width_px * sample.height_px) for sample in samples
    ]
    areas = [area for _, area in area_records]
    aspects = [sample.width_px / sample.height_px for sample in samples]
    smallest = min(areas)
    largest = max(areas)
    return DatasetSizeSummary(
        sample_count=len(samples),
        width_min=min(widths),
        width_max=max(widths),
        height_min=min(heights),
        height_max=max(heights),
        area_min=smallest,
        area_max=largest,
        aspect_ratio_min=min(aspects),
        aspect_ratio_max=max(aspects),
        width_distribution=tuple(sorted(Counter(widths).items())),
        height_distribution=tuple(sorted(Counter(heights).items())),
        area_distribution=tuple(sorted(Counter(areas).items())),
        aspect_ratio_distribution=tuple(sorted(Counter(aspects).items())),
        smallest_sample_ids=tuple(
            sorted(sample_id for sample_id, area in area_records if area == smallest)
        ),
        largest_sample_ids=tuple(
            sorted(sample_id for sample_id, area in area_records if area == largest)
        ),
    )
