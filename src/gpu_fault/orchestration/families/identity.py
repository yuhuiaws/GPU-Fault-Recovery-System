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
