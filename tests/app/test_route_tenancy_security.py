"""Cross-tenant authorization guards on global-id and path route reads.

Two data-plane read routes address records by an id that is not scoped by the
caller's cluster in the request path, so the handler must compare the record's
owning cluster to the authenticated cluster before returning it:

* ``GET /v1/training-progress/{cluster_id}/{attempt_id}`` (M-1) takes the
  cluster in the path but historically trusted it, so a cluster-A token could
  read cluster-B's training progress -- the same gap the ``latest_gpu_metrics``
  route already closes.
* ``GET /v1/gpu-events/xid/{event_id}/correlation`` (M-2) looks a correlation
  up by a *global* event id with no cluster in the path at all, so without a
  check a cluster-A token could read the correlation and policy decision of any
  cluster's XID event.

These proxy the live regional auth-boundary cases: they prove denial semantics
against the real application, authorization registry and store so the boundary
regresses in CI rather than in the next acceptance window.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.policy import XidEvent
from gpu_fault.policy.models import XidCorrelationRecord
from gpu_fault.training_health import TrainingProgressHeartbeat
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import NOW, TOKEN_A, TOKEN_B, registration

EXECUTION_TOKEN = "e" * 32

HEADERS_A = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
HEADERS_B = {
    "Authorization": f"Bearer {TOKEN_B}",
    "X-GPU-Fault-Cluster-ID": "cluster-b",
}


def regional_context():
    context = build_context()
    context.regional_mode = True
    context.execution_token = EXECUTION_TOKEN
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


def test_m1_training_progress_read_denies_a_foreign_cluster() -> None:
    context = regional_context()
    context.store.observe_training_progress(
        TrainingProgressHeartbeat(
            cluster_id="cluster-b",
            attempt_id="attempt-b",
            rank=0,
            observed_at=NOW,
            step=7,
        )
    )
    path = "/v1/training-progress/cluster-b/attempt-b"

    async def scenario() -> None:
        async with asgi_client(context) as client:
            spoofed = await client.get(path, headers=HEADERS_A)
            owner = await client.get(path, headers=HEADERS_B)

            assert spoofed.status_code == 403, spoofed.text
            assert spoofed.json() == {
                "detail": (
                    "authenticated cluster cannot read training progress "
                    "for another cluster"
                )
            }

            assert owner.status_code == 200, owner.text
            body = owner.json()
            assert [item["attempt_id"] for item in body] == ["attempt-b"]

    asyncio.run(scenario())


def test_m2_xid_correlation_read_denies_a_foreign_cluster() -> None:
    context = regional_context()
    context.store.save_xid_event_if_absent(
        XidEvent(
            event_id="xid-b",
            cluster_id="cluster-b",
            node_id="node-b",
            observed_at=NOW,
            xid=79,
        )
    )
    context.store.save_xid_correlation_if_absent(
        XidCorrelationRecord(event_id="xid-b", deadline=NOW + timedelta(minutes=5))
    )
    path = "/v1/gpu-events/xid/xid-b/correlation"

    async def scenario() -> None:
        async with asgi_client(context) as client:
            spoofed = await client.get(path, headers=HEADERS_A)
            owner = await client.get(path, headers=HEADERS_B)

            assert spoofed.status_code == 403, spoofed.text
            assert spoofed.json() == {
                "detail": (
                    "authenticated cluster cannot read an XID correlation "
                    "from another cluster"
                )
            }

            assert owner.status_code == 200, owner.text
            assert owner.json()["correlation"]["event_id"] == "xid-b"

    asyncio.run(scenario())
