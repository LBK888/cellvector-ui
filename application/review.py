"""Human review operations over immutable prediction proposals."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from cellvector.domain.models import (
    AnnotationDocument,
    MicroridgeFeature,
    Provenance,
    ReviewStatus,
)
from cellvector.domain.revisions import create_revision


def accept_proposal_features(
    document: AnnotationDocument,
    proposal_id: UUID,
    feature_ids: Sequence[UUID],
    *,
    author: str,
) -> AnnotationDocument:
    """Copy selected proposal ridges into a new human-reviewed revision."""

    proposal = next(
        (candidate for candidate in document.proposals if candidate.id == proposal_id),
        None,
    )
    if proposal is None:
        raise KeyError(f"proposal not found: {proposal_id}")
    requested = set(feature_ids)
    available = {feature.id for feature in proposal.microridges}
    missing = requested - available
    if missing:
        raise KeyError(f"proposal features not found: {sorted(str(item) for item in missing)}")

    selected: list[MicroridgeFeature] = []
    for feature in proposal.microridges:
        if feature.id not in requested:
            continue
        selected.append(
            feature.model_copy(
                update={
                    "review_status": ReviewStatus.ACCEPTED,
                    "provenance": Provenance(
                        method="manual",
                        source_proposal_id=proposal.id,
                        implementation_version=feature.provenance.implementation_version,
                        model_version=feature.provenance.model_version,
                        parameters=feature.provenance.parameters,
                    ),
                }
            )
        )

    return create_revision(
        document,
        author=author,
        summary=f"accepted {len(selected)} feature(s) from proposal {proposal.id}",
        microridges=[*document.microridges, *selected],
    )

