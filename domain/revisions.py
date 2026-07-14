"""Full-snapshot immutable annotation revisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import AnnotationDocument, ReviewRecord
from .validation import QCError, validate_document


def create_revision(
    document: AnnotationDocument,
    *,
    author: str,
    summary: str,
    **changes: Any,
) -> AnnotationDocument:
    """Create and validate a new full snapshot without mutating ``document``."""

    payload = document.model_dump()
    payload.update(changes)
    payload["parent_revision_id"] = document.revision_id
    payload["revision_id"] = uuid4()
    payload["review"] = ReviewRecord(
        author=author,
        timestamp=datetime.now(timezone.utc),
        edit_summary=summary,
    )
    revised = AnnotationDocument.model_validate(payload)
    report = validate_document(revised)
    if report.errors:
        raise QCError(report)
    return revised
