"""Posting the same fault event twice: the ingest contract a client relies on.

``POST /v1/gpu-events/xid`` and ``/sxid`` deduplicate by ``event_id`` in the
handler, whether or not the post carries ``Idempotency-Key`` (which only adds
the queue's replay of the stored response in queued processor mode): the
event row is saved if absent, the incident is found through its event link,
the marker is upserted under ``marker-<event_id>`` and every notification has
an event-derived deduplication key. So a client that cannot tell whether its
first post landed -- the perf harness after a closed connection -- may post the
identical body again and gets one incident, one workflow, one marker and one
notification, answered with the same incident and marker. A repeat that comes
after the incident's recovery retired its marker leaves the marker retired.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from gpu_fault.app import create_app
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.store import SqliteStore
from tests._builders import build_context
from tests.orchestration._cross_fault_support import (
    observation,
    sxid_payload,
    xid_payload,
)


@pytest.fixture(params=["memory", "sqlite"])
def context(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    """An application context with one attempt; SQLite dedupes the event row
    with ``INSERT OR IGNORE`` as PostgreSQL does with ``ON CONFLICT``."""

    if request.param == "memory":
        subject = build_context()
    else:
        subject = build_context(store=SqliteStore(str(tmp_path / "ingest.db")))
    subject.store.save_attempt_observation(observation())
    try:
        yield subject
    finally:
        close = getattr(subject.store, "close", None)
        if close is not None:
            close()


ROUTES = {
    "xid": ("/v1/gpu-events/xid", lambda event_id: xid_payload(48, event_id)),
    "sxid": ("/v1/gpu-events/sxid", sxid_payload),
}


def post_twice(
    context: Any,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str],
    between: Any = None,
) -> list[dict[str, Any]]:
    """The identical body twice through one application; ``between`` runs in
    the gap (the incident's recovery, say)."""

    async def run() -> list[dict[str, Any]]:
        transport = httpx.ASGITransport(app=create_app(context))
        answers = []
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as api:
            for attempt in range(2):
                if attempt and between is not None:
                    between(answers[0])
                response = await api.post(path, json=body, headers=headers)
                assert response.status_code == 200, response.text
                answers.append(response.json())
        return answers

    return asyncio.run(run())


@pytest.mark.parametrize("header", [False, True], ids=["no-header", "idempotency-key"])
@pytest.mark.parametrize("kind", sorted(ROUTES))
def test_a_repeated_fault_post_creates_one_incident_workflow_marker_and_notice(
    context: Any, kind: str, header: bool
) -> None:
    path, payload = ROUTES[kind]
    event_id = f"repeat-{kind}"
    headers = {"Idempotency-Key": event_id} if header else {}

    first, repeat = post_twice(context, path, payload(event_id), headers)

    assert first["incident_id"] and first["workflow_request_id"]
    assert repeat["incident_id"] == first["incident_id"]
    assert repeat["workflow_request_id"] == first["workflow_request_id"]
    assert repeat["marker"]["marker_id"] == first["marker"]["marker_id"]
    assert first["duplicate"] is False
    if kind == "xid":
        assert repeat["duplicate"] is True, "the stored XID decision says so"
    incident = context.store.get_incident_by_event(event_id)
    assert incident is not None and incident.incident_id == first["incident_id"]
    assert [item.request_id for item in context.store.list_workflows(limit=10)] == [
        first["workflow_request_id"]
    ], "one workflow"
    markers = context.store.list_markers_for_incident(first["incident_id"])
    assert [marker.marker_id for marker in markers] == [first["marker"]["marker_id"]]
    notices = [
        item.notification_id
        for item in context.store.list_notifications()
        if item.incident_id == first["incident_id"]
    ]
    assert len(notices) == len(set(notices)) >= 1, "each notice exists once"
    assert (
        repeat["investigatory_notification_id"]
        == (first["investigatory_notification_id"])
    )


@pytest.mark.parametrize("kind", sorted(ROUTES))
def test_a_repeat_after_recovery_leaves_the_retired_marker_retired(
    context: Any, kind: str
) -> None:
    path, payload = ROUTES[kind]
    retired: list[Any] = []

    def recover(first: dict[str, Any]) -> None:
        [marker] = context.store.list_markers_for_incident(first["incident_id"])
        if marker.active:
            assert retire_markers_for_incident(
                context.store,
                first["incident_id"],
                reason="unit recovery",
                retired_by="unit-test",
            ), "the recovery retires the live marker"
        else:
            # This SXID's marker is informational and never active; retire it
            # the way ``retire_markers_for_incident`` writes an active one.
            context.store.add_marker(
                marker.model_copy(
                    update={
                        "retired_at": marker.observed_at,
                        "retired_reason": "unit recovery",
                        "retired_by": "unit-test",
                    }
                )
            )
        [marker] = context.store.list_markers_for_incident(first["incident_id"])
        retired.append(marker)

    first, repeat = post_twice(
        context, path, payload(f"late-{kind}"), {}, between=recover
    )

    assert repeat["incident_id"] == first["incident_id"]
    assert context.store.list_markers_for_incident(first["incident_id"]) == retired, (
        "the repeat did not revive the recovered fault's marker"
    )
    assert retired[0].active is False and retired[0].retired_by == "unit-test"
    assert (kind == "xid") is first["marker"]["active"], (
        "the XID 48 marker is the one a revival would have re-armed"
    )
