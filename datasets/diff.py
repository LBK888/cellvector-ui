from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import DatasetSnapshot, SplitName


class DiffIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str


class SnapshotDiff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_hash: str
    target_hash: str
    added_sample_ids: tuple[str, ...] = ()
    removed_sample_ids: tuple[str, ...] = ()
    revision_changed_sample_ids: tuple[str, ...] = ()
    group_changed_sample_ids: tuple[str, ...] = ()
    split_changed_sample_ids: tuple[str, ...] = ()
    source_changed_sample_ids: tuple[str, ...] = ()
    label_policy_changed: bool = False
    software_smoke_changed: bool = False
    blocking_issues: tuple[DiffIssue, ...] = ()


def compare_snapshots(base: DatasetSnapshot, target: DatasetSnapshot) -> SnapshotDiff:
    before = {sample.sample_id: sample for sample in base.samples}
    after = {sample.sample_id: sample for sample in target.samples}
    common = sorted(before.keys() & after.keys())
    frozen_groups = {
        sample.group_key
        for sample in base.samples
        if sample.split == SplitName.FROZEN_TEST
    }
    target_frozen_groups = {
        sample.group_key
        for sample in target.samples
        if sample.split == SplitName.FROZEN_TEST
    }
    moved_groups = sorted(frozen_groups - target_frozen_groups)
    issues = tuple(
        DiffIssue(
            code="SNAPSHOT_FROZEN_TEST_VIOLATION",
            message=f"frozen-test group moved or removed in target snapshot: {group}",
        )
        for group in moved_groups
    )
    changed = lambda attribute: tuple(
        sample_id
        for sample_id in common
        if getattr(before[sample_id], attribute) != getattr(after[sample_id], attribute)
    )
    return SnapshotDiff(
        base_hash=base.identity_hash(),
        target_hash=target.identity_hash(),
        added_sample_ids=tuple(sorted(after.keys() - before.keys())),
        removed_sample_ids=tuple(sorted(before.keys() - after.keys())),
        revision_changed_sample_ids=changed("revision_id"),
        group_changed_sample_ids=changed("group_key"),
        split_changed_sample_ids=changed("split"),
        source_changed_sample_ids=changed("source_sha256"),
        label_policy_changed=base.label_policy_version != target.label_policy_version,
        software_smoke_changed=base.software_smoke_test != target.software_smoke_test,
        blocking_issues=issues,
    )
