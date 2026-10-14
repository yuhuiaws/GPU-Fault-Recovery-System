"""A registry write that would change a recorded identity fails fast (batch 91).

``/v1/installation-resources/sync`` saved its snapshot row by row and let the
store's ``ValueError`` become a 500. The processor books a 500 as retryable, so
every deploy and ``config`` sync of a conflicting snapshot was replayed for the
300-s retry horizon, and every replay rewrote the rows ahead of the conflict
(live 2026-10-04..13, "installation resource identity cannot change"). The
conflict is now a 409 that names each refused key and field, the snapshot is
one transaction, and the processor completes the 409 at once.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from gpu_fault.processor import ProcessorCoordinator, ProcessorRequestStatus
from gpu_fault.processor.replay_completion import finalize_replay_response
from tests._builders import asgi_client, build_store, processor_request

TOKEN = "installation-registry-conflict-" + "x" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": TOKEN}
SYNC = "/v1/installation-resources/sync"


def _row(key: str, resource_id: str, **updates: object) -> InstallationResource:
    return InstallationResource(
        site_id="site-a",
        resource_key=key,
        resource_type="security_group",
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
    ).model_copy(update=updates)


def _snapshot(*rows: InstallationResource) -> dict[str, object]:
    return InstallationResourceSnapshot(
        site_id="site-a", resources=list(rows)
    ).model_dump(mode="json")


def _context() -> ApplicationContext:
    context = ApplicationContext(execution_token=TOKEN)
    context.regional_mode = True
    return context


def _requests(context: ApplicationContext, *calls: tuple[str, str, object]) -> list:
    async def scenario() -> list:
        async with asgi_client(context) as client:
            return [
                await client.request(method, path, json=body, headers=HEADERS)
                for method, path, body in calls
            ]

    return asyncio.run(scenario())


def test_a_sync_identity_conflict_is_a_409_that_writes_nothing() -> None:
    context = _context()
    first, second = _row("aws/a", "sg-a"), _row("aws/b", "sg-b")
    pending = first.model_copy(
        update={"status": InstallationResourceStatus.DELETE_PENDING}
    )
    created, refused = _requests(
        context,
        ("POST", SYNC, _snapshot(first, second)),
        (
            "POST",
            SYNC,
            _snapshot(
                pending,
                _row("aws/b", "sg-replaced", dependencies=["aws/a"]),
                _row("aws/c", "sg-c"),
            ),
        ),
    )
    assert created.status_code == 200, created.text
    assert refused.status_code == 409, "a deterministic conflict must not be a 5xx"
    assert refused.json() == {
        "detail": (
            "installation resource identity cannot change: "
            "aws/b (resource_id, dependencies)"
        ),
        "reason": "installation resource identity cannot change",
        "conflicts": [
            {"resource_key": "aws/b", "fields": ["resource_id", "dependencies"]}
        ],
    }
    assert context.store.list_installation_resources("site-a") == [first, second], (
        "the refused sync rewrote the row ahead of its conflict or added one"
    )


def test_a_single_row_update_that_changes_an_identity_is_a_409() -> None:
    context = _context()
    original = _row("aws/a", "sg-a")
    created, refused = _requests(
        context,
        ("POST", SYNC, _snapshot(original)),
        (
            "PUT",
            "/v1/installation-resources/site-a/aws/a",
            _row("aws/a", "sg-other").model_dump(mode="json"),
        ),
    )
    assert created.status_code == 200, created.text
    assert refused.status_code == 409
    assert refused.json()["conflicts"] == [
        {"resource_key": "aws/a", "fields": ["resource_id"]}
    ]
    assert context.store.list_installation_resources("site-a") == [original]


def _claimed_sync(store, body: bytes):
    request = store.enqueue_processor_request(
        processor_request(SYNC, body=body, cluster_id=None)
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=120),
        limit=1,
    )[0]
    assert claimed.request_id == request.request_id
    return claimed


def _finalize(store, claimed, status: int, body: bytes) -> None:
    finalize_replay_response(
        ProcessorCoordinator(
            store,
            owner_id="pod-a:1",
            internal_token="processor-token",
            active_consumers=True,
        ),
        claimed,
        status=status,
        content_type="application/json",
        retry_partition=None,
        body=body,
        started=time.monotonic(),
    )


def test_the_processor_completes_the_conflict_at_once() -> None:
    context = _context()
    conflicting = _snapshot(_row("aws/a", "sg-replaced"))
    _, refused = _requests(
        context,
        ("POST", SYNC, _snapshot(_row("aws/a", "sg-a"))),
        ("POST", SYNC, conflicting),
    )
    store = build_store()
    claimed = _claimed_sync(store, json.dumps(conflicting).encode())

    _finalize(store, claimed, refused.status_code, refused.content)

    current = store.get_processor_request(claimed.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED, (
        "a 409 must be the request's final answer, not a retry"
    )
    assert current.retry_count == 0
    assert current.response_status == 409
    assert json.loads(current.response_body())["conflicts"] == [
        {"resource_key": "aws/a", "fields": ["resource_id"]}
    ], "the waiting caller must receive the named conflict"


def test_a_500_is_still_replayed_as_retryable() -> None:
    """The contrast: what every conflicting sync got before -- a reschedule."""

    store = build_store()
    claimed = _claimed_sync(store, b"{}")

    _finalize(store, claimed, 500, b'{"detail":"Internal Server Error"}')

    current = store.get_processor_request(claimed.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
