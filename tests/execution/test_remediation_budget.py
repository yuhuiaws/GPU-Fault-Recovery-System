from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    budget_work_begun,
    remediation_budget_claims,
)
from gpu_fault.failure_domains import FAILURE_DOMAIN_MAP_ENV
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore, RemediationBudgetError, SqliteStore
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
JOB = "default/pytorchjob/job-7"


def _workflow(
    store,
    suffix: str,
    *,
    node_id: str = "node-a",
    job_id: str | None = None,
    attempt_id: str = "attempt-1",
):
    incident = fault_incident(
        f"incident-{suffix}",
        f"event-{suffix}",
        node_ids=[node_id],
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
        job_id=job_id,
        attempt_id=attempt_id if job_id else None,
    )
    workflow = workflow_request(
        f"workflow-{suffix}",
        incident.incident_id,
        WorkflowStatus.PENDING,
        fencing_token=3,
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[node_id])
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    incident = incident.model_copy(update={"workflow_request_id": workflow.request_id})
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


@pytest.mark.parametrize("durable", [False, True])
def test_remediation_budget_is_durable_and_released_when_the_holder_ends(
    tmp_path, durable: bool
) -> None:
    """A lapsed lease releases nothing (a reboot outlives its executor's lease);
    the holder's terminal write does."""

    store = SqliteStore(str(tmp_path / "budget.db")) if durable else InMemoryStore()
    _, first = _workflow(store, "first", node_id="node-a")
    _, second = _workflow(store, "second", node_id="node-b")
    # Two workflows stopping one job on disjoint nodes share its token.
    scope = f"job:cluster-a:{JOB}"
    claims = {scope: 1}

    holder = store.claim_workflow(
        first.request_id,
        "executor-a",
        first.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=claims,
    )
    with pytest.raises(RemediationBudgetError, match=f"scope={scope}") as raised:
        store.claim_workflow(
            second.request_id,
            "executor-b",
            second.fencing_token,
            now=NOW,
            lease_duration=timedelta(seconds=30),
            remediation_budget_claims=claims,
        )

    # The scope travels structurally, not by parsing the message: the
    # waiting gauges read it back off the workflow (S1).
    assert raised.value.scope == scope
    blocked = store.get_workflow(second.request_id)
    assert blocked.remediation_budget_wait_count == 1
    assert scope in (blocked.remediation_budget_last_blocked_reason or "")
    assert blocked.remediation_budget_last_blocked_scope == scope

    with pytest.raises(RemediationBudgetError, match=f"scope={scope}"):
        store.claim_workflow(
            second.request_id,
            "executor-b",
            second.fencing_token,
            now=NOW + timedelta(seconds=31),
            lease_duration=timedelta(seconds=30),
            remediation_budget_claims=claims,
        )
    store.save_workflow_if_leased(
        holder.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        "executor-a",
        holder.execution_epoch,
        now=NOW + timedelta(seconds=1),
    )
    claimed = store.claim_workflow(
        second.request_id,
        "executor-b",
        second.fencing_token,
        now=NOW + timedelta(seconds=31),
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=claims,
    )
    assert claimed.remediation_budget_claims == sorted(claims)
    assert claimed.remediation_budget_last_blocked_reason is None
    assert claimed.remediation_budget_last_blocked_scope is None


def test_workflow_rows_persisted_without_a_blocked_scope_still_load() -> None:
    """Rows written before the scope field existed carry only the reason."""

    payload = json.loads(
        workflow_request(
            "workflow-legacy",
            "incident-legacy",
            WorkflowStatus.PENDING,
            remediation_budget_last_blocked_reason="scope=cluster:cluster-a",
        ).model_dump_json()
    )
    del payload["remediation_budget_last_blocked_scope"]

    loaded = WorkflowRequest.model_validate(payload)

    assert loaded.remediation_budget_last_blocked_reason == "scope=cluster:cluster-a"
    assert loaded.remediation_budget_last_blocked_scope is None


def test_remediation_budget_claims_cover_all_required_scopes() -> None:
    store = InMemoryStore()
    incident, workflow = _workflow(store, "scopes")
    step = workflow.official_steps[0].model_copy(
        update={
            "parameters": {
                "fabric_partition": "fabric-a",
                # Nothing in the control plane writes this key; the limiter
                # must not read it, or it reads as protection it never was.
                "availability_zone": "us-west-2a",
            }
        }
    )
    workflow = workflow.model_copy(update={"official_steps": [step]})

    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, incident, workflow.official_steps
    )

    # The node, and the fabric partition as the one domain: no label group, no
    # workload slot, and no region, cluster or resource-class count -- the
    # cluster's storm breaker bounds a correlated fault, not a fixed number.
    assert claims == {"node:cluster-a:node-a": 1, "domain:cluster-a:fabric-a": 1}


def _job_step(operation, nodes, **values):
    return workflow_step(operation, node_ids=nodes, workload_ids=[JOB], **values)


def test_one_job_workflow_over_several_nodes_claims_each_node_and_no_domain() -> None:
    """A job is one workflow: its nodes are repaired in parallel inside it, and
    the per-node scopes are what keep any other workflow off them."""

    store = InMemoryStore()
    incident, workflow = _workflow(store, "job", job_id="job-7")
    workflow = workflow.model_copy(
        update={
            "official_steps": [
                _job_step(WorkflowOperation.STOP_WORKLOADS, ["a", "b"]),
                _job_step(WorkflowOperation.RESET_GPU, ["a"]),
                _job_step(WorkflowOperation.RESTART_NODE, ["b"]),
                _job_step(
                    WorkflowOperation.RESTART_WORKLOAD,
                    ["a", "b"],
                    parameters={"job_id": "job-7"},
                ),
            ]
        }
    )

    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, incident, workflow.official_steps
    )

    assert claims["node:cluster-a:a"] == claims["node:cluster-a:b"] == 1
    assert not [scope for scope in claims if scope.startswith("domain:")], claims
    # Its STOP and RESTART act on the whole job: the job's token, once.
    assert sorted(scope for scope in claims if scope.startswith("job:")) == [
        f"job:cluster-a:{JOB}",
        "job:cluster-a:job-id/job-7",
    ]


def test_only_whole_job_steps_take_the_job_token() -> None:
    store = InMemoryStore()
    incident, workflow = _workflow(store, "local", job_id="job-7")
    steps = [
        _job_step(WorkflowOperation.RESET_GPU, ["a"]),
        _job_step(WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN, ["a"]),
    ]

    claims = remediation_budget_claims(
        RemediationBudgetPolicy(),
        workflow.model_copy(update={"official_steps": steps}),
        incident,
        steps,
    )

    assert not [scope for scope in claims if scope.startswith("job:")], claims


def test_the_policy_reads_no_node_label_failure_domain_map(tmp_path) -> None:
    """The control-worker still mounts the published label map; the limiter
    does not read it, so not even a missing file reaches the policy."""

    policy = RemediationBudgetPolicy.from_mapping(
        {FAILURE_DOMAIN_MAP_ENV: str(tmp_path / "absent.json")}
    )

    assert policy == RemediationBudgetPolicy()


def _admit(store, workflow, incident, executor: str):
    return store.claim_workflow(
        workflow.request_id,
        executor,
        workflow.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=remediation_budget_claims(
            RemediationBudgetPolicy(), workflow, incident, workflow.official_steps
        ),
    )


def _store(tmp_path, durable: bool):
    return SqliteStore(str(tmp_path / "budget.db")) if durable else InMemoryStore()


@pytest.mark.parametrize("durable", [False, True])
def test_two_jobs_on_two_nodes_of_one_node_group_are_repaired_together(
    tmp_path, durable: bool
) -> None:
    """node-a and node-b share an instance group / EKS node group: under the
    label domains the second job's reboot waited for the first one's."""

    store = _store(tmp_path, durable)
    first = _workflow(store, "job-1", node_id="node-a", job_id="job-1")
    second = _workflow(store, "job-2", node_id="node-b", job_id="job-2")

    admitted = [
        _admit(store, workflow, incident, f"executor-{index}")
        for index, (incident, workflow) in enumerate((first, second))
    ]

    assert [item.status for item in admitted] == [WorkflowStatus.RUNNING] * 2
    occupancy = store.remediation_budget_occupancy()
    assert occupancy["node:cluster-a:node-a"] == occupancy["node:cluster-a:node-b"] == 1
    assert not [scope for scope in occupancy if scope.startswith("cluster:")], occupancy


@pytest.mark.parametrize("durable", [False, True])
def test_two_idle_nodes_are_repaired_together(tmp_path, durable: bool) -> None:
    store = _store(tmp_path, durable)
    first = _workflow(store, "idle-a", node_id="node-a")
    second = _workflow(store, "idle-b", node_id="node-b")

    for index, (incident, workflow) in enumerate((first, second)):
        assert (
            _admit(store, workflow, incident, f"executor-{index}").status
            is WorkflowStatus.RUNNING
        )


@pytest.mark.parametrize("durable", [False, True])
def test_a_second_workflow_on_a_node_waits_for_the_first(
    tmp_path, durable: bool
) -> None:
    """A job's later attempt, a successor or a node workflow: one node, one
    workflow at a time. The job's own STOP lists every node of the job, so a
    later workflow of the same job on any of them waits too."""

    store = _store(tmp_path, durable)
    incident, workflow = _workflow(store, "job-first", node_id="node-a", job_id="job-7")
    workflow = workflow.model_copy(
        update={
            "official_steps": [
                workflow_step(
                    WorkflowOperation.STOP_WORKLOADS, node_ids=["node-a", "node-b"]
                ),
                workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"]),
            ]
        }
    )
    store.save_workflow(workflow)
    later_incident, later = _workflow(
        store, "job-later", node_id="node-b", job_id="job-7", attempt_id="attempt-2"
    )
    holder = _admit(store, workflow, incident, "executor-a")

    with pytest.raises(RemediationBudgetError) as raised:
        _admit(store, later, later_incident, "executor-b")

    assert raised.value.scope == "node:cluster-a:node-b"
    waiting = store.get_workflow(later.request_id)
    assert waiting.status is WorkflowStatus.PENDING
    assert waiting.remediation_budget_last_blocked_scope == raised.value.scope

    store.save_workflow_if_leased(
        holder.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        "executor-a",
        holder.execution_epoch,
        now=NOW + timedelta(seconds=1),
    )
    assert (
        _admit(store, waiting, later_incident, "executor-b").status
        is WorkflowStatus.RUNNING
    )


@pytest.mark.parametrize("durable", [False, True])
def test_two_workflows_stopping_one_job_on_disjoint_nodes_are_serialized(
    tmp_path, durable: bool
) -> None:
    """A STOP compiled for the faulted node alone still stops every rank (an
    event no grouping attached, a health finding): the per-node scopes of
    node-a and node-b do not meet, the job's token does."""

    store = _store(tmp_path, durable)
    pairs = []
    for name, node_id in (("stop-a", "node-a"), ("stop-b", "node-b")):
        incident, workflow = _workflow(store, name, node_id=node_id, job_id="job-7")
        workflow = workflow.model_copy(
            update={
                "official_steps": [
                    _job_step(WorkflowOperation.STOP_WORKLOADS, [node_id]),
                    _job_step(WorkflowOperation.RESTART_NODE, [node_id]),
                ]
            }
        )
        store.save_workflow(workflow)
        pairs.append((incident, workflow))
    _admit(store, pairs[0][1], pairs[0][0], "executor-a")

    with pytest.raises(RemediationBudgetError) as raised:
        _admit(store, pairs[1][1], pairs[1][0], "executor-b")

    assert raised.value.scope == f"job:cluster-a:{JOB}"


@pytest.mark.parametrize("durable", [False, True])
def test_a_shared_fabric_partition_still_serializes_its_workflows(
    tmp_path, durable: bool
) -> None:
    """The physical NVSwitch partition stays a domain of its own."""

    store = _store(tmp_path, durable)
    pairs = []
    for name, node_id in (("fabric-a", "node-a"), ("fabric-b", "node-b")):
        incident, workflow = _workflow(store, name, node_id=node_id)
        step = workflow.official_steps[0].model_copy(
            update={"parameters": {"fabric_partition": "partition-1"}}
        )
        workflow = workflow.model_copy(update={"official_steps": [step]})
        store.save_workflow(workflow)
        pairs.append((incident, workflow))
    _admit(store, pairs[0][1], pairs[0][0], "executor-a")

    with pytest.raises(RemediationBudgetError) as raised:
        _admit(store, pairs[1][1], pairs[1][0], "executor-b")

    assert raised.value.scope == "domain:cluster-a:partition-1"


@pytest.mark.parametrize("durable", [False, True])
def test_a_label_domain_an_older_release_holds_blocks_nobody_and_drains(
    tmp_path, durable: bool
) -> None:
    """Upgrade: a RUNNING row admitted on its node group's domain keeps that
    string until it ends; no claim of this release asks for it, so another
    node of the group is admitted, and the row's own hold is never refused."""

    store = _store(tmp_path, durable)
    incident, old = _workflow(store, "old", node_id="node-a", job_id="job-1")
    legacy = ["cluster:cluster-a", "domain:cluster-a:gpu-group", "region"]
    old = old.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "completed_step_indexes": [0],
            "completed_operations": [WorkflowOperation.RESTART_NODE],
            "official_steps": [
                *old.official_steps,
                workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"]),
            ],
            "remediation_budget_claims": legacy,
            "remediation_budget_limits": {
                "cluster:cluster-a": 5,
                "domain:cluster-a:gpu-group": 1,
                "region": 20,
            },
        }
    )
    store.save_workflow(old)
    other_incident, other = _workflow(store, "new", node_id="node-b", job_id="job-2")

    assert _admit(store, other, other_incident, "executor-b").status is (
        WorkflowStatus.RUNNING
    )
    held = store.claim_workflow(
        old.request_id,
        "executor-a",
        old.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=remediation_budget_claims(
            RemediationBudgetPolicy(), old, incident, old.official_steps
        ),
        remediation_budget_hold=budget_work_begun(old, old.official_steps),
    )

    assert "domain:cluster-a:gpu-group" in held.remediation_budget_claims
    assert "node:cluster-a:node-a" in held.remediation_budget_claims
    occupancy = store.remediation_budget_occupancy()
    assert occupancy["domain:cluster-a:gpu-group"] == 1
    assert occupancy["node:cluster-a:node-b"] == 1


@pytest.mark.parametrize("durable", [False, True])
def test_the_fixed_limits_an_older_release_holds_block_nobody(
    tmp_path, durable: bool
) -> None:
    """Upgrade: six RUNNING rows an older release admitted each list
    ``region``, ``cluster:`` and ``class:`` -- over that release's 5 per
    cluster and 2 per resource class. No claim of this release asks for those
    scopes, so a seventh node's reboot is admitted; the strings stay on the
    rows until they end."""

    store = _store(tmp_path, durable)
    legacy = ["class:cluster-a:NODE_LIFECYCLE_MUTATION", "cluster:cluster-a", "region"]
    for index in range(6):
        _, old = _workflow(store, f"old-{index}", node_id=f"node-{index}")
        store.save_workflow(
            old.model_copy(
                update={
                    "status": WorkflowStatus.RUNNING,
                    "remediation_budget_claims": [
                        *legacy,
                        f"node:cluster-a:node-{index}",
                    ],
                    "remediation_budget_limits": {
                        "class:cluster-a:NODE_LIFECYCLE_MUTATION": 2,
                        "cluster:cluster-a": 5,
                        "region": 20,
                        f"node:cluster-a:node-{index}": 1,
                    },
                }
            )
        )
    incident, fresh = _workflow(store, "fresh", node_id="node-9")

    admitted = _admit(store, fresh, incident, "executor-b")

    assert admitted.status is WorkflowStatus.RUNNING
    assert admitted.remediation_budget_claims == ["node:cluster-a:node-9"]
    occupancy = store.remediation_budget_occupancy()
    assert occupancy["cluster:cluster-a"] == occupancy["region"] == 6


def test_restore_compensation_never_waits_for_a_remediation_budget() -> None:
    store = InMemoryStore()
    incident, workflow = _workflow(store, "restore")
    restore = workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, node_ids=["node-a"])
    workflow = workflow.model_copy(
        update={
            "official_steps": [restore],
            "pending_failure_step_index": 0,
            "pending_failure_error": "reset failed",
        }
    )

    # Compensation is a hold (never refused) and asks for nothing new.
    assert budget_work_begun(workflow, workflow.official_steps) is True
    assert (
        remediation_budget_claims(
            RemediationBudgetPolicy(), workflow, incident, workflow.official_steps
        )
        == {}
    )


@pytest.mark.parametrize(
    "operations",
    [
        (WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.CHECK_MECHANICALS),
        (
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.CHECK_MECHANICALS,
        ),
    ],
)
def test_an_operator_acknowledgement_wait_holds_no_remediation_budget(
    operations: tuple[WorkflowOperation, ...],
) -> None:
    """The CHECK_MECHANICALS plans the planner builds have no budgeted step, so
    the wait for an operator (up to the 24-hour acknowledgement floor) keeps no
    slot (design 9.5); a reset or reboot ordered afterwards is its own workflow.
    """

    store = InMemoryStore()
    incident, workflow = _workflow(store, "inspection")
    steps = [workflow_step(operation, node_ids=["node-a"]) for operation in operations]
    waiting = workflow.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "official_steps": steps,
            "completed_step_indexes": [0],
        }
    )

    assert budget_work_begun(waiting, steps) is False
    assert (
        remediation_budget_claims(RemediationBudgetPolicy(), waiting, incident, steps)
        == {}
    )
