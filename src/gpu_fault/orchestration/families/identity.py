"""Deterministic record ids for the ingestion families (F-B7 / P2-49H).

An incident or workflow minted as ``incident-<uuid4>`` cannot be rebuilt: the
duplicate fast paths now repair a dangling link by running the builder again,
and with random ids that second run would leave a second incident and a second
workflow for one fault, neither knowing about the other. The id is therefore a
function of what the record is about -- the family that owns it and the event
identity it was built from -- the way ``failure_containment_ids`` already does
for attempt failures.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def derived_record_id(prefix: str, family: str, *parts: str) -> str:
    """``<prefix>-<digest>`` where the digest names the family and the event.

    ``parts`` are JSON-encoded so two different part lists cannot spell the
    same string, and the family is part of the digest so two families that
    both build from one ``event_id`` still get distinct records.
    """

    digest = hashlib.sha256(
        json.dumps(
            [family, *parts],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"{prefix}-{digest}"


def note_stale_event_link(
    store: Any, *, event_id: str, incident_id: str, pointer: str | None
) -> None:
    """Record that an ``incident_by_event`` link resolved to a missing workflow.

    The ingestion fast paths in ``coordinator.ingest`` and
    ``escalation.escalate`` read the workflow behind ``workflow_request_id``
    unguarded; a dangling pointer made every re-post of the event -- the retry
    path -- raise out of the handler (control-plane review 2026-09-08, C-04;
    the remainder of F-B7). They now fall through to the build path, which is
    the same idempotent repair the store's ``_duplicate_event_records`` does,
    and count it on the store's ``stale_event_link_repairs`` so the existing
    ``gpu_fault_ingest_stale_event_link_repairs_total`` gauge sees it.
    """

    LOGGER.warning(
        "incident_by_event link for %s points at incident %s whose workflow %s is "
        "missing; rebuilding the event instead of failing the re-post",
        event_id,
        incident_id,
        pointer,
    )
    store.stale_event_link_repairs = getattr(store, "stale_event_link_repairs", 0) + 1
