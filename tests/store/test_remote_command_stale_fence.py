"""A remote command whose workflow moved to a later generation is told to stop,
settles FAILED when its executor reports, and is swept if it never does.

Control-plane review 2026-09-08, D-9 (direction B). A workflow replaced in
place keeps its ``request_id`` and takes ``fencing_token + 1``; a command the
data plane had already leased under the old token was then stranded:

* ``complete_remote_command`` refused the result with ``WorkflowLeaseError``
  (a 409 on the wire), so the row stayed LEASED for ever;
* ``renew_remote_command_lease`` never compared the fence, so the executor's
  ``CommandLeaseWatch`` saw no cancellation and the node-side action ran to
  the end while the new generation minted a second command on the same node;
* no sweep touched a LEASED row, so nothing ever closed it.

Now the renewal carries a cancellation request (the lease is still extended so
the executor can report), the completion writes FAILED with
``status_source="stale-fence"``, and ``expire_stale_fenced_remote_commands``
fails a LEASED command whose lease lapsed under a stale fence.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import SqliteStore, WorkflowLeaseError
from gpu_fault.store.shared.remote_commands import STALE_FENCE_STATUS_SOURCE
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

CLUSTER = "cluster-a"
EXECUTOR = "regional-executor-a"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "stale-fence.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _leased_command(
    store,
    *,
    command_id: str = "remote-old",
    request_id: str = "workflow-a",
    token: int = 3,
):
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        cluster_id=CLUSTER,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        fencing_token=token,
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        WorkflowStatus.RUNNING,
        fencing_token=token,
        official_steps=[
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])
        ],
    )
    store.save_incident_and_workflow(incident, workflow)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id,
            cluster_id=CLUSTER,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=token,
            idempotency_key=f"{request_id}/0/RESET_GPU",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=incident,
        )
    )
    (leased,) = store.claim_remote_commands(
        CLUSTER, EXECUTOR, limit=1, lease_seconds=600
    )
    return workflow, leased


def _replace_in_place(store, workflow):
    """The same row at the next generation: what REPLACE_IN_PLACE writes."""

    current = store.get_workflow(workflow.request_id)
    store.save_workflow(
        copy_model(current, fencing_token=current.fencing_token + 1), expected=current
    )


def test_a_result_for_a_stale_fence_settles_the_command_failed(store):
    workflow, leased = _leased_command(store)
    _replace_in_place(store, workflow)

    settled = store.complete_remote_command(
        CLUSTER,
        leased.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={"reset": "done"},
        ),
    )

    assert settled.status is RemoteCommandStatus.FAILED
    assert settled.status_source == STALE_FENCE_STATUS_SOURCE
    assert settled.lease_owner is None
    assert settled.last_lease_owner == EXECUTOR
    assert settled.result_details["post_stale_fence_status"] == "SUCCEEDED"
    assert settled.result_details["reset"] == "done"
    assert "generation" in (settled.error or "")
    stored = store.get_remote_command(leased.command_id)
    assert stored.status is RemoteCommandStatus.FAILED
    # Idempotent: a repeated report returns the settled row.
    again = store.complete_remote_command(
        CLUSTER,
        leased.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token, status=RemoteCommandStatus.SUCCEEDED
        ),
    )
    assert again.status is RemoteCommandStatus.FAILED


def test_a_matching_fence_still_rejects_a_stale_lease_token(store):
    _, leased = _leased_command(store)

    with pytest.raises(WorkflowLeaseError):
        store.complete_remote_command(
            CLUSTER,
            leased.command_id,
            RemoteCommandResult(
                lease_token="not-the-lease", status=RemoteCommandStatus.SUCCEEDED
            ),
        )


def test_a_renewal_under_a_stale_fence_asks_the_executor_to_stop(store):
    workflow, leased = _leased_command(store)
    _replace_in_place(store, workflow)

    renewed = store.renew_remote_command_lease(
        CLUSTER, leased.command_id, EXECUTOR, leased.lease_token, lease_seconds=600
    )

    assert renewed.status is RemoteCommandStatus.LEASED
    assert renewed.cancellation_requested_at is not None
    assert "generation" in (renewed.cancellation_reason or "")
    # The lease is extended so the executor can still report what it did.
    assert renewed.lease_expires_at is not None
    assert renewed.lease_expires_at > leased.lease_expires_at - timedelta(seconds=1)
    stored = store.get_remote_command(leased.command_id)
    assert stored.cancellation_requested_at == renewed.cancellation_requested_at
    # No other executor is ever handed it.
    assert (
        store.claim_remote_commands(CLUSTER, "executor-b", limit=5, lease_seconds=60)
        == []
    )


def test_a_renewal_under_a_matching_fence_carries_no_cancellation(store):
    _, leased = _leased_command(store)

    renewed = store.renew_remote_command_lease(
        CLUSTER, leased.command_id, EXECUTOR, leased.lease_token, lease_seconds=600
    )

    assert renewed.cancellation_requested_at is None


def _expire_lease(store, command_id: str, *, ago: timedelta):
    current = store.get_remote_command(command_id)
    expired = current.model_copy(
        update={"lease_expires_at": datetime.now(timezone.utc) - ago}
    )
    if hasattr(store, "_remote_commands"):
        store._remote_commands[command_id] = expired
    else:
        store._put("remote_command", command_id, expired)


def test_the_sweep_fails_a_leased_command_whose_lease_lapsed_under_a_stale_fence(store):
    workflow, leased = _leased_command(store)
    _replace_in_place(store, workflow)
    _expire_lease(store, leased.command_id, ago=timedelta(minutes=20))

    swept = store.expire_stale_fenced_remote_commands(
        lease_expired_before=datetime.now(timezone.utc) - timedelta(minutes=10),
        limit=10,
    )

    assert swept == 1
    stored = store.get_remote_command(leased.command_id)
    assert stored.status is RemoteCommandStatus.FAILED
    assert stored.status_source == STALE_FENCE_STATUS_SOURCE
    assert stored.lease_owner is None
    assert stored.last_lease_owner == EXECUTOR
    assert stored.result_details["stale_fence_swept"] is True
    # Nothing left to sweep.
    assert (
        store.expire_stale_fenced_remote_commands(
            lease_expired_before=datetime.now(timezone.utc), limit=10
        )
        == 0
    )


def test_the_sweep_leaves_a_fresh_lease_and_a_matching_fence_alone(store):
    workflow, leased = _leased_command(store)
    _replace_in_place(store, workflow)
    # Stale fence, but the lease is still live: the executor is still reporting
    # (and its renewal already carries the cancellation).
    assert (
        store.expire_stale_fenced_remote_commands(
            lease_expired_before=datetime.now(timezone.utc) - timedelta(minutes=10),
            limit=10,
        )
        == 0
    )
    assert (
        store.get_remote_command(leased.command_id).status is RemoteCommandStatus.LEASED
    )

    # Lapsed lease, matching fence: that is the claim path's business (it is
    # re-leased by the next executor), not the sweep's.
    _, other_leased = _leased_command(
        store, command_id="remote-current", request_id="workflow-b"
    )
    _expire_lease(store, other_leased.command_id, ago=timedelta(minutes=20))
    assert (
        store.expire_stale_fenced_remote_commands(
            lease_expired_before=datetime.now(timezone.utc) - timedelta(minutes=10),
            limit=10,
        )
        == 0
    )
    assert (
        store.get_remote_command(other_leased.command_id).status
        is RemoteCommandStatus.LEASED
    )
