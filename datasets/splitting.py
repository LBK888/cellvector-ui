from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

from .models import DatasetError, DatasetSample, SplitName


def assign_group_splits(
    samples: Sequence[DatasetSample],
    *,
    seed: int,
    frozen_ledger: Mapping[str, str] | None = None,
    forced_train: Iterable[str] = (),
) -> list[DatasetSample]:
    """Assign indivisible groups deterministically to three dataset splits."""

    ledger = dict(frozen_ledger or {})
    train_groups = set(forced_train)
    conflict = train_groups.intersection(ledger)
    if conflict:
        raise DatasetError(
            "FROZEN_TEST_VIOLATION",
            f"frozen groups cannot enter training: {sorted(conflict)}",
        )

    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    revision_owner: dict[object, str] = {}
    source_owner: dict[str, str] = {}
    for sample in samples:
        previous = revision_owner.setdefault(sample.revision_id, sample.group_key)
        if previous != sample.group_key:
            raise DatasetError(
                "GROUP_SPLIT_LEAKAGE",
                f"revision {sample.revision_id} occurs in multiple groups",
            )
        previous_source_group = source_owner.setdefault(
            sample.source_sha256,
            sample.group_key,
        )
        if previous_source_group != sample.group_key:
            raise DatasetError(
                "GROUP_SPLIT_LEAKAGE",
                f"source {sample.source_sha256} occurs in multiple groups",
            )
        grouped[sample.group_key].append(sample)

    groups = set(grouped)
    if len(groups) < 3:
        raise DatasetError(
            "INSUFFICIENT_SPLIT_GROUPS",
            "at least three independent groups are required",
        )
    unknown_forced = train_groups.difference(groups)
    if unknown_forced:
        raise DatasetError(
            "GROUP_SPLIT_LEAKAGE",
            f"forced groups are absent from the dataset: {sorted(unknown_forced)}",
        )

    frozen_groups = groups.intersection(ledger)
    undecided = groups.difference(frozen_groups, train_groups)
    ranked = sorted(
        undecided,
        key=lambda group: sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    target_test = max(1, round(len(groups) * 0.15))
    target_validation = max(1, round(len(groups) * 0.15))
    extra_test_count = max(0, target_test - len(frozen_groups))
    test_groups = frozen_groups.union(ranked[:extra_test_count])
    remaining = ranked[extra_test_count:]
    validation_groups = set(remaining[:target_validation])
    resulting_train = groups.difference(test_groups, validation_groups)
    if not resulting_train or not validation_groups or not test_groups:
        raise DatasetError(
            "INSUFFICIENT_SPLIT_GROUPS",
            "group constraints cannot populate train, validation, and frozen-test",
        )

    split_by_group = {
        **{group: SplitName.TRAIN for group in resulting_train},
        **{group: SplitName.VALIDATION for group in validation_groups},
        **{group: SplitName.FROZEN_TEST for group in test_groups},
    }
    return [
        sample.model_copy(update={"split": split_by_group[sample.group_key]})
        for sample in samples
    ]
