from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.memory.store import InMemoryStore
from gpu_fault.store.shared.errors import NotFoundError
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire

NOW = 2_000_000_000
STAMP = datetime.fromtimestamp(NOW, timezone.utc)


def plan(**changes: Any) -> wire.Plan:
    value = {
        "schema_version": 1,
        "run_id": "destr008-run",
        "cluster_id": "cluster-a",
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "event_id": "destr008-event",
        "release_id": "release-a",
        "fault_node": "fault-a",
        "spare_node": "spare-a",
        "runtime_profile_version": "profile-a",
        "workload_ids": ["training/job-a"],
        "probe_sha256": wire.source_sha256(),
        "created_at": NOW,
        "deadline_at": NOW + 100,
        "fence": {
            "policy": "policy-a",
            "policy_uid": "policy-uid-a",
            "binding": "binding-a",
            "binding_uid": "binding-uid-a",
            "marker": "a" * 64,
            "node": "spare-a",
            "node_uid": "spare-uid-a",
        },
        **changes,
    }
    return wire.Plan.model_validate(value)


class FakeApi:
    """Atomic JSON Patch over a single owned ConfigMap, never a cluster client."""

    def __init__(
        self,
        bound: wire.Plan,
        *,
        namespace: str = "cpu",
        name: str = "watchdog",
        uid: str = "cm-uid",
    ) -> None:
        self.namespace = namespace
        self.name = name
        self.uid = uid
        self.value = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(
                namespace=namespace, name=name, uid=uid, resource_version="1"
            ),
            data=wire.initial_data(bound),
        )
        self.read_errors: list[Exception] = []
        self.patch_errors: list[Exception] = []
        self.patches: list[list[dict[str, Any]]] = []
        self.after_patch: Callable[[], None] | None = None

    def read_namespaced_config_map(
        self, name: str, namespace: str, **kwargs: Any
    ) -> Any:
        assert (name, namespace) == (self.name, self.namespace), (
            "read escaped the owned ConfigMap"
        )
        assert kwargs == {"_request_timeout": (3, 5)}, "API reads need bounded timeouts"
        if self.read_errors:
            raise self.read_errors.pop(0)
        return copy.deepcopy(self.value)

    def patch_namespaced_config_map(
        self, name: str, namespace: str, body: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        assert (name, namespace) == (self.name, self.namespace), (
            "patch escaped the owned ConfigMap"
        )
        assert kwargs == {"_request_timeout": (3, 5)}, (
            "API patches need bounded timeouts"
        )
        if self.patch_errors:
            raise self.patch_errors.pop(0)
        self.patches.append(copy.deepcopy(body))
        document = {
            "metadata": {
                "uid": self.value.metadata.uid,
                "resourceVersion": self.value.metadata.resource_version,
            },
            "data": copy.deepcopy(self.value.data),
        }
        for operation in body:
            section, name = operation["path"].lstrip("/").split("/")
            if operation["op"] == "test":
                if document[section].get(name) != operation["value"]:
                    raise ApiException(status=409)
            else:
                assert operation["op"] == "replace" and section == "data", (
                    "only owned data may change"
                )
                document[section][name] = operation["value"]
        self.value.data = document["data"]
        self.value.metadata.resource_version = str(
            int(self.value.metadata.resource_version) + 1
        )
        if self.after_patch is not None:
            callback, self.after_patch = self.after_patch, None
            callback()
        return copy.deepcopy(self.value)


class MemoryCPU(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def setup() -> tuple[
    wire.Plan, FakeApi, probe.KubernetesControlMap, MemoryCPU, probe.Watchdog
]:
    bound = plan()
    api = FakeApi(bound)
    port = probe.KubernetesControlMap(
        api,
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256=bound.probe_sha256,
    )
    store = MemoryCPU()
    watchdog = probe.Watchdog(port, store, sleep=lambda _: None)
    return bound, api, port, store, watchdog


def parent_write(port: probe.KubernetesControlMap, control: wire.Control) -> None:
    current = port.read()
    if current.status is None:
        raise AssertionError("parent must observe ARMED before claiming")
    port.write(current, control, current.status)


def claim(
    bound: wire.Plan, port: probe.KubernetesControlMap, *, now: int = NOW + 1
) -> None:
    parent_write(
        port,
        wire.claim_submission(bound, port.read().control, claim_id="claim-a", now=now),
    )


def acknowledge(
    bound: wire.Plan,
    port: probe.KubernetesControlMap,
    *,
    now: int = NOW + 2,
    incident_id: str = "incident-a",
    workflow_id: str = "workflow-a",
) -> None:
    parent_write(
        port,
        wire.acknowledge_submission(
            bound,
            port.read().control,
            now=now,
            acknowledgement=wire.Acknowledgement(
                claim_id="claim-a",
                event_id=bound.event_id,
                completed_at=now,
                incident_id=incident_id,
                workflow_request_id=workflow_id,
            ),
        ),
    )


def seed(
    store: InMemoryStore,
    bound: wire.Plan,
    *,
    status: WorkflowStatus = WorkflowStatus.PENDING,
    workflow_id: str = "workflow-a",
    incident_id: str = "incident-a",
    event_id: str | None = None,
    predecessor: str | None = None,
    failure_handled: bool = True,
) -> tuple[FaultIncident, WorkflowRequest]:
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=event_id or bound.event_id,
        event_type="NODE_HEALTH",
        cluster_id=bound.cluster_id,
        job_id=bound.job_id,
        attempt_id=bound.attempt_id,
        node_ids=[bound.fault_node],
        policy_version="610",
        policy_source="SITE_SYNTHETIC_REPLACEMENT_TEST",
        workflow_request_id=workflow_id,
        created_at=STAMP,
        updated_at=STAMP,
    )
    workflow = WorkflowRequest(
        request_id=workflow_id,
        incident_id=incident_id,
        runtime_profile_version=bound.runtime_profile_version,
        predecessor_workflow_id=predecessor,
        status=status,
        fencing_token=1,
        failure_handled_at=STAMP
        if failure_handled and status is WorkflowStatus.FAILED
        else None,
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.REPLACE_NODE,
                execution_owner="gpu-fault-hyperpod-adapter",
                node_ids=[bound.fault_node],
                workload_ids=bound.workload_ids,
                parameters={
                    "cluster_id": bound.cluster_id,
                    "job_id": bound.job_id,
                    "source_attempt_id": bound.attempt_id,
                    "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
                },
            )
        ],
        created_at=STAMP,
        updated_at=STAMP,
    )
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


def command(
    store: InMemoryStore,
    incident: FaultIncident,
    workflow: WorkflowRequest,
    /,
    *,
    command_id: str = "command-a",
    status: RemoteCommandStatus = RemoteCommandStatus.PENDING,
    **changes: Any,
) -> RemoteActionCommand:
    values = {
        "command_id": command_id,
        "cluster_id": incident.cluster_id,
        "workflow_request_id": workflow.request_id,
        "incident_id": incident.incident_id,
        "step_index": 0,
        "fencing_token": workflow.fencing_token,
        "idempotency_key": command_id,
        "step": workflow.official_steps[0],
        "workflow": workflow,
        "incident": incident,
        "status": status,
        "created_at": STAMP,
        "updated_at": STAMP,
        **changes,
    }
    if status is RemoteCommandStatus.LEASED:
        values = {
            **values,
            "lease_owner": "local-executor",
            "lease_token": "local-test-lease",
            "lease_expires_at": datetime.fromtimestamp(NOW + 1000, timezone.utc),
            **changes,
        }
    result = RemoteActionCommand.model_validate(values)
    store.ensure_remote_command(result)
    return result


def terminalize(store: InMemoryStore, workflow_id: str = "workflow-a") -> None:
    current = store.get_workflow(workflow_id)
    store.save_workflow(
        current.model_copy(
            update={
                "status": WorkflowStatus.SUPERSEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        expected=current,
    )


def armed_submission(
    *, status: WorkflowStatus = WorkflowStatus.PENDING
) -> tuple[Any, ...]:
    bound, api, port, store, watchdog = setup()
    armed = watchdog.tick(NOW)
    assert armed.state == "ARMED", "submission requires a persisted ARMED receipt"
    claim(bound, port)
    incident, workflow = seed(store, bound, status=status)
    acknowledge(bound, port)
    return bound, api, port, store, watchdog, incident, workflow


def test_deadline_revokes_before_any_store_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, port, store, watchdog = setup()
    read = store.list_job_recovery_workflow_incidents
    calls: list[bool] = []

    def guarded_read(
        *args: Any, **kwargs: Any
    ) -> list[tuple[FaultIncident, WorkflowRequest]]:
        calls.append(port.read().control.revocation is not None)
        assert kwargs["include_terminal"] is True
        return read(*args, **kwargs)

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", guarded_read)
    monkeypatch.setattr(
        store, "list_workflows", lambda **_: pytest.fail("global scan is forbidden")
    )
    assert watchdog.tick(NOW).state == "ARMED"
    assert calls == [], "arming must not scan or mutate recovery records"
    first = watchdog.tick(bound.deadline_at)
    assert first.state == "REVOKED" and first.producer_revoked
    assert calls and all(calls), "durable revocation must precede every Store read"
    closed = watchdog.tick(bound.deadline_at + wire.QUIET_SECONDS)
    assert closed.state == "QUIESCENT" and closed.root is None
    assert closed.commands_active == closed.workflows_active == 0
    assert closed.source_complete and not closed.fence_release_authorized
    assert closed.fence == bound.fence and closed.run_id == bound.run_id
    assert port.read().status == closed, "stdout is not the receipt authority"
    with pytest.raises(wire.ProbeError, match="SUBMISSION_REVOKED"):
        wire.claim_submission(
            bound, port.read().control, claim_id="late", now=bound.deadline_at + 6
        )


def test_pending_commands_cancel_but_workflow_is_never_force_terminalized() -> None:
    bound, _, port, store, watchdog, incident, workflow = armed_submission()
    pending = command(store, incident, workflow)
    result = watchdog.tick(bound.deadline_at)
    withdrawn = store.get_workflow(workflow.request_id)
    assert result.state == "REVOKED" and result.workflows_active == 1
    assert withdrawn.status is WorkflowStatus.PENDING, (
        "probe may withdraw, not terminalize"
    )
    assert withdrawn.workload_withdrawn_at is not None
    assert withdrawn.merge_revision == workflow.merge_revision + 1
    assert withdrawn.events[-1].code == "WORKLOAD_WITHDRAWN"
    assert withdrawn.events[-1].details["plan_sha256"] == wire.digest(bound)
    assert (
        store.get_remote_command(pending.command_id).status
        is RemoteCommandStatus.FAILED
    )
    watchdog.tick(bound.deadline_at + 1)
    assert (
        store.get_workflow(workflow.request_id).merge_revision
        == withdrawn.merge_revision
    )
    assert store.get_workflow(workflow.request_id).events == withdrawn.events
    terminalize(store)
    assert watchdog.tick(bound.deadline_at + 2).state == "REVOKED"
    closed = watchdog.tick(bound.deadline_at + 7)
    assert closed.state == "QUIESCENT" and port.read().status == closed
    with pytest.raises(NotFoundError):
        store.get_event_by_attempt(bound.cluster_id, bound.attempt_id)


@pytest.mark.parametrize("expired", [False, True])
def test_still_leased_command_blocks_quiescence_even_after_timeout(
    expired: bool,
) -> None:
    bound, _, port, store, watchdog, incident, workflow = armed_submission()
    remote = command(
        store,
        incident,
        workflow,
        status=RemoteCommandStatus.LEASED,
        lease_expires_at=datetime.fromtimestamp(
            NOW + (1 if expired else 1000), timezone.utc
        ),
    )
    first = watchdog.tick(bound.deadline_at)
    terminalize(store)
    failed = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS)
    actual = store.get_remote_command(remote.command_id)
    assert first.commands_active == failed.commands_active == 1
    assert failed.state == "FAILED" and failed.monitoring
    assert failed.error_code == "DRAIN_UNRESOLVED"
    assert (
        actual.status is RemoteCommandStatus.LEASED
        and actual.cancellation_requested_at is not None
    )
    assert (
        actual.lease_owner == remote.lease_owner
        and actual.lease_token == remote.lease_token
    )
    assert actual.lease_expires_at == remote.lease_expires_at
    assert (
        port.read().control.revocation is not None
        and not failed.fence_release_authorized
    )


def test_actual_executor_completion_after_cancellation_can_drain() -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(store, incident, workflow, status=RemoteCommandStatus.LEASED)
    watchdog.tick(bound.deadline_at)
    completed = store.complete_remote_command(
        bound.cluster_id,
        remote.command_id,
        RemoteCommandResult(
            lease_token="local-test-lease",
            status=RemoteCommandStatus.FAILED,
            error="local physical operation finished",
            status_source="local-executor",
        ),
    )
    assert completed.status_source == "completed-after-cancellation"
    terminalize(store)
    watchdog.tick(bound.deadline_at + 1)
    assert watchdog.tick(bound.deadline_at + 6).state == "QUIESCENT"


def test_parent_loss_before_ack_is_not_idle_and_late_event_is_withdrawn() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    first = watchdog.tick(bound.deadline_at)
    assert first.state == "FAILED" and first.error_code == "SOURCE_UNRESOLVED"
    assert first.root is None and first.source_complete is False and first.monitoring
    incident, workflow = seed(store, bound)
    late = command(store, incident, workflow)
    resumed = probe.Watchdog(port, store, sleep=lambda _: None)
    result = resumed.tick(bound.deadline_at + 1)
    assert result.root == wire.Root(
        incident_id=incident.incident_id, workflow_request_id=workflow.request_id
    )
    assert result.state == "FAILED" and result.source_complete is False
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is not None
    assert (
        store.get_remote_command(late.command_id).status is RemoteCommandStatus.FAILED
    )
    acknowledge(bound, port, now=bound.deadline_at + 2)
    terminalize(store)
    resumed.tick(bound.deadline_at + 2)
    assert resumed.tick(bound.deadline_at + 7).state == "QUIESCENT"


def test_command_arriving_after_first_drain_scan_is_cancelled() -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    assert watchdog.tick(bound.deadline_at).state == "REVOKED"
    late = command(store, incident, workflow, command_id="late-command")
    result = watchdog.tick(bound.deadline_at + 5)
    assert result.state == "REVOKED", "a new actual record must reset the quiet proof"
    assert (
        store.get_remote_command(late.command_id).status is RemoteCommandStatus.FAILED
    )
    assert watchdog.tick(bound.deadline_at + 10).state == "QUIESCENT"


@pytest.mark.parametrize("submitted", [False, True])
def test_exact_parent_close_can_finish_before_deadline(submitted: bool) -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    if submitted:
        claim(bound, port)
        seed(store, bound, status=WorkflowStatus.FAILED)
        acknowledge(bound, port)
    parent_write(port, wire.request_close(bound, port.read().control, now=NOW + 3))
    assert watchdog.tick(NOW + 3).state == "REVOKED"
    closed = watchdog.tick(NOW + 8)
    assert closed.state == "QUIESCENT" and closed.observed_at < bound.deadline_at
    assert closed.revocation is not None and closed.revocation.reason == "PARENT_CLOSE"
    before = port.read()
    assert probe.Watchdog(port, store).tick(bound.deadline_at) == closed
    assert port.read() == before, "a resumed Job must not reset a completed tombstone"
