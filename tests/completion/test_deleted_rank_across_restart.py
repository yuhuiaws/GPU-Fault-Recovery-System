"""A rank whose Pod was asked to go is a stop across a watcher restart too (batch 157).

A rank that exits non-zero after its Pod's deletion was requested -- a graceful
``kubectl delete``, an eviction through the Eviction API (a drain, a node
consolidation) -- is not a training failure: the Completion Watcher reads
``metadata.deletionTimestamp`` into ``deletion_requested``, and neither the
reading nor the watcher core counts that exit. The persisted active record kept
the exit but dropped the deletion request, so a watcher restarted after such a
reading restored the rank as an unexcused non-zero exit. The watcher core then
held a failure no event ever reported, and the attempt -- still running on its
other rank and on the job controller's replacement Pod -- ended TIMED_OUT at the
cleanup deadline. The control plane answers that failure terminal by opening
the attempt's containment (FREEZE_EVIDENCE, STOP_WORKLOADS) and planning a
RESTART_WORKLOAD: a healthy job was stopped. The record now keeps the request
(``deleting``, written only while it is true).

The Pods have the shapes ``test_exited_rank_gpu_uuids`` gives a managed two-rank
PyTorchJob on plain EKS and on HyperPod, and everything the watchers post is
handed, in order, to the control plane's own observation, failure and terminal
handlers.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault import completion_attempt_state
from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.workload_observations import ingest_workload_observation
from gpu_fault.completion_attempt_store import (
    PODS_KEY,
    attempt_observation_payload,
    attempt_state_record,
)
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    Environment,
    TerminalEvent,
)
from gpu_fault.watcher import AttemptObservation, FailureDetectedEvent, WorkloadPhase
from tests.completion._support import Clock, FakeCoreApi, FakeSink
from tests.completion.test_exited_rank_gpu_uuids import (
    ATTEMPT,
    CLUSTER,
    FAILURE_PATH,
    OBSERVATION_PATH,
    TERMINAL_PATH,
    Exec,
    StopNotYetReadable,
    control_plane,
    gpus,
    posted,
    rank_gpus,
    training_pod,
    uid,
    watcher,
)

ACTIVE_KEY = "active-attempts.json"
PASSIVE_EVENT = f"{CLUSTER}/{ATTEMPT}/TrainingAttemptFailureDetected"
#: The job controller's replacement for rank 1 (worker-0): a new Pod UID.
REPLACEMENT = "uid-worker-0-replacement"
#: Two nodes the attempt runs on and a third one a replacement can land on.
PROVIDERS = [
    pytest.param(
        Environment.EKS,
        (
            "ip-10-0-45-43.us-west-2.compute.internal",
            "ip-10-0-43-5.us-west-2.compute.internal",
            "ip-10-0-47-9.us-west-2.compute.internal",
        ),
        id="plain-eks",
    ),
    pytest.param(
        Environment.HYPERPOD_EKS,
        (
            "hyperpod-i-00000000000000001",
            "hyperpod-i-00000000000000002",
            "hyperpod-i-0000000000000000",
        ),
        id="hyperpod",
    ),
]
#: ``POD_FIELDS`` as the previous release wrote and read it.
EARLIER_POD_FIELDS: tuple[tuple[str, str], ...] = (
    ("uid", "pod_uid"),
    ("name", "pod_name"),
    ("container", "container_name"),
    ("role", "role"),
    ("rank", "rank"),
    ("critical", "critical"),
    ("node", "node_id"),
    ("instance", "instance_id"),
    ("gpus", "gpu_count"),
    ("uuids", "gpu_uuids"),
    ("cid", "container_id"),
    ("cgroup", "cgroup_path"),
    ("pid", "host_pid"),
    ("terminated", "terminated"),
    ("exit_code", "exit_code"),
    ("finished_at", "finished_at"),
)
EARLIER_SPEC_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "environment",
    "job_id",
    "attempt_id",
    "workload_phase",
    "observed_at",
    "started_at",
    "expected_critical_ranks",
    "workload_ids",
    "cleanup_timeout_seconds",
    "checkpoint_manifest_ref",
    "termination_initiator_incident_id",
    "runtime_profile_version",
    "restart_budget",
)


def earlier_release_record(payload: dict[str, Any]) -> dict[str, Any]:
    """``attempt_state_record`` exactly as the previous release ran it."""

    record: dict[str, Any] = {
        field: payload[field]
        for field in EARLIER_SPEC_FIELDS
        if payload.get(field) is not None
    }
    pods = []
    for container in payload.get("containers") or []:
        pod = {
            key: container[field]
            for key, field in EARLIER_POD_FIELDS
            if container.get(field) is not None
        }
        pods.append(pod)
    record[PODS_KEY] = pods
    return record


def earlier_release_payload(record: dict[str, Any]) -> dict[str, Any]:
    """``attempt_observation_payload`` exactly as the previous release ran it:
    it keeps the Pod keys it knows and nothing else."""

    if PODS_KEY not in record:
        return record
    payload = {key: value for key, value in record.items() if key != PODS_KEY}
    containers = []
    for pod in record.get(PODS_KEY) or []:
        if not isinstance(pod, dict):
            continue
        container = {field: pod[key] for key, field in EARLIER_POD_FIELDS if key in pod}
        containers.append(container)
    payload["containers"] = containers
    return payload


def deleted(pod: dict[str, Any], at: datetime) -> dict[str, Any]:
    """``pod`` once the API server was asked to remove it (a graceful delete)."""

    pod["metadata"]["deletionTimestamp"] = at.isoformat()
    pod["metadata"]["deletionGracePeriodSeconds"] = 30
    return pod


def evicted(pod: dict[str, Any], at: datetime) -> dict[str, Any]:
    """``pod`` evicted through the Eviction API (``kubectl drain``, a node
    consolidation): the DisruptionTarget condition, and the graceful delete an
    eviction is."""

    pod["status"]["conditions"] = [
        {
            "type": "DisruptionTarget",
            "status": "True",
            "reason": "EvictionByEvictionAPI",
            "message": "Eviction API: evicting",
            "lastTransitionTime": at.isoformat(),
        }
    ]
    return deleted(pod, at)


def pending(pod: dict[str, Any]) -> dict[str, Any]:
    """``pod`` created and not started yet: no container status at all."""

    pod["status"] = {"phase": "Pending"}
    return pod


def outbox_watcher(
    core: FakeCoreApi,
    delivered: FakeSink,
    environment: Environment,
    clock: Clock,
    *,
    stopper: Any = None,
) -> KubernetesCompletionController:
    """One watcher process: a fresh outbox over the same ConfigMaps."""

    return watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        environment,
        clock,
        Exec(),
        stopper=stopper,
    )


def persisted_pods(core: FakeCoreApi) -> dict[int, dict[str, Any]]:
    """The persisted record's Pods by rank: the Pod UID and the verdict fields."""

    [record] = json.loads(core.config_map_data[ACTIVE_KEY]).values()
    return {
        pod["rank"]: {
            key: pod[key]
            for key in ("uid", "terminated", "exit_code", "deleting")
            if key in pod
        }
        for pod in record[PODS_KEY]
    }


def control_plane_reads(
    delivered: FakeSink,
) -> tuple[ApplicationContext, list[CompletionDecision]]:
    """Every POST the watchers made, handed in order to the control plane's own
    handlers; the terminal decisions it took come back with its context."""

    context, completion = control_plane()
    decisions = []
    for path, payload in delivered.posts:
        if path == OBSERVATION_PATH:
            ingest_workload_observation(
                context, AttemptObservation.model_validate(payload)
            )
        elif path == FAILURE_PATH:
            completion.handle_failure_detected(
                FailureDetectedEvent.model_validate(payload)
            )
        elif path == TERMINAL_PATH:
            decisions.append(
                completion.handle_terminal(TerminalEvent.model_validate(payload))
            )
    return context, decisions


def assert_nothing_stopped(context: ApplicationContext) -> None:
    """No containment incident for the attempt, and no workflow at all -- so no
    FREEZE_EVIDENCE, no STOP_WORKLOADS and no RESTART_WORKLOAD."""

    incident = context.store.get_incident_by_event(PASSIVE_EVENT)
    assert incident is None, (
        f"a containment was opened for a deletion: {incident and incident.reasons}"
    )
    workflows = {
        workflow.request_id: [step.operation.value for step in workflow.official_steps]
        for workflow in context.store.list_workflows()
    }
    assert workflows == {}, f"the control plane acted on a healthy job: {workflows}"


class DeletedPyTorchJob:
    """The PyTorchJob after ``kubectl delete``: every read is a 404."""

    def __init__(self) -> None:
        self.reads = 0

    def get_namespaced_custom_object(self, *_args: Any, **_kwargs: Any) -> Any:
        self.reads += 1
        error = RuntimeError("pytorchjobs.kubeflow.org not found")
        error.status = 404
        raise error


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_deleted_rank_is_still_a_stop_after_a_restart(
    environment: Environment, nodes: tuple[str, str, str]
) -> None:
    """``kubectl delete pod`` of worker-0 while the job trains: torchrun takes the
    SIGTERM and exits 1, the first watcher reads that exit with the deletion and
    persists it, and the watcher restarts before it reads anything else. The
    PyTorchJob operator's replacement (new UID) runs. The restarted watcher
    restores the rank as deleted: no failure event, RUNNING through twice the
    cleanup timeout -- where it used to end TIMED_OUT -- and SUCCEEDED once both
    ranks finish, which the control plane takes as a success: no containment,
    nothing stopped, no restart planned."""

    clock = Clock()
    delivered = FakeSink()
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = outbox_watcher(core, delivered, environment, clock)
    first.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [
        training_pod(0, nodes[0]),
        deleted(training_pod(1, nodes[1], exit_code=1), clock()),
    ]
    first.run_once()
    after_deletion = persisted_pods(core)

    clock.value += timedelta(seconds=20)
    core.pods = [
        training_pod(0, nodes[0]),
        training_pod(1, nodes[1], pod_uid=REPLACEMENT),
    ]
    restarted = outbox_watcher(core, delivered, environment, clock)
    for _ in range(13):
        restarted.run_once()
        clock.value += timedelta(seconds=5)
    running = posted(delivered, OBSERVATION_PATH)
    core.pods = [
        training_pod(0, nodes[0], exit_code=0),
        training_pod(1, nodes[1], pod_uid=REPLACEMENT, exit_code=0),
    ]
    restarted.run_once()

    assert posted(delivered, FAILURE_PATH) == [], "a deletion was reported as a failure"
    phases = [reading["workload_phase"] for reading in running]
    assert set(phases) == {"RUNNING"}, f"the healthy attempt was ended: {phases}"
    [terminal] = posted(delivered, TERMINAL_PATH)
    assert terminal["terminal_status"] == "SUCCEEDED", terminal
    assert rank_gpus(terminal["allocation"]) == {
        0: gpus(uid(0)),
        1: gpus(REPLACEMENT),
    }, terminal["allocation"]
    context, decisions = control_plane_reads(delivered)
    assert [(item.status, item.reason) for item in decisions] == [
        (DecisionStatus.NO_ACTION, "training attempt succeeded")
    ], decisions
    assert_nothing_stopped(context)
    assert after_deletion[1] == {
        "uid": uid(1),
        "terminated": True,
        "exit_code": 1,
        "deleting": True,
    }, f"the record must keep the deletion request with the exit: {after_deletion}"
    assert json.loads(core.config_map_data[ACTIVE_KEY]) == {}, (
        "the terminal reading must drop the record"
    )


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_an_evicted_rank_is_still_a_stop_after_a_restart(
    environment: Environment, nodes: tuple[str, str, str]
) -> None:
    """A drain evicts worker-0 through the Eviction API: the DisruptionTarget
    condition, the graceful delete, exit 143 on the SIGTERM -- and the watcher
    restarts right after reading that. The restarted watcher still lists the
    evicted Pod (terminated, being deleted), then the operator's replacement on
    another node, Pending and then Running. The attempt keeps running past twice
    the cleanup timeout: no failure, no terminal, and the control plane reads it
    RUNNING on the surviving node and on the replacement's, each rank with its
    own GPUs, and stops nothing."""

    clock = Clock()
    delivered = FakeSink()
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = outbox_watcher(core, delivered, environment, clock)
    first.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [
        training_pod(0, nodes[0]),
        evicted(training_pod(1, nodes[1], exit_code=143), clock()),
    ]
    first.run_once()
    after_eviction = persisted_pods(core)

    clock.value += timedelta(seconds=5)
    restarted = outbox_watcher(core, delivered, environment, clock)
    restarted.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [
        training_pod(0, nodes[0]),
        pending(training_pod(1, nodes[2], pod_uid=REPLACEMENT)),
    ]
    restarted.run_once()
    core.pods = [
        training_pod(0, nodes[0]),
        training_pod(1, nodes[2], pod_uid=REPLACEMENT),
    ]
    for _ in range(12):
        clock.value += timedelta(seconds=5)
        restarted.run_once()

    assert posted(delivered, FAILURE_PATH) == [], (
        "an eviction was reported as a failure"
    )
    assert posted(delivered, TERMINAL_PATH) == [], "the running attempt was ended"
    readings = posted(delivered, OBSERVATION_PATH)
    phases = [reading["workload_phase"] for reading in readings]
    assert set(phases) == {"RUNNING"}, f"the healthy attempt was ended: {phases}"
    evicted_reading = readings[2]["containers"]
    assert [
        (item["pod_uid"], item["deletion_requested"], sorted(item["gpu_uuids"]))
        for item in evicted_reading
        if item["rank"] == 1
    ] == [(uid(1), True, gpus(uid(1)))], evicted_reading
    assert {
        (item["rank"], item["pod_uid"], item["node_id"]): sorted(item["gpu_uuids"])
        for item in readings[-1]["containers"]
    } == {
        (0, uid(0), nodes[0]): gpus(uid(0)),
        (1, REPLACEMENT, nodes[2]): gpus(REPLACEMENT),
    }, readings[-1]["containers"]
    context, decisions = control_plane_reads(delivered)
    assert decisions == [], decisions
    assert_nothing_stopped(context)
    [state] = context.store.list_attempt_observation_states(CLUSTER)
    assert state.observation.workload_phase is WorkloadPhase.RUNNING, state
    for node, gpu in ((nodes[0], gpus(uid(0))[0]), (nodes[2], gpus(REPLACEMENT)[0])):
        owner = context.topology.resolve(CLUSTER, node, clock(), target_gpu_uuids={gpu})
        assert (owner.workload_state, owner.attempt_ids) == ("ACTIVE", [ATTEMPT]), (
            f"{node} must still read as the running attempt's: {owner}"
        )
    assert after_eviction[1].get("deleting") is True, after_eviction


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_job_deleted_across_a_restart_ends_as_the_users_stop(
    environment: Environment, nodes: tuple[str, str, str]
) -> None:
    """``kubectl delete pytorchjob`` just before the watcher restarts: both Pods
    get their graceful delete, the master exits 1 first, and the first watcher
    reads that -- worker-0 still running, being deleted -- before it stops. When
    the restarted watcher lists, the Pods and the PyTorchJob are gone. The
    restored attempt is a stop, at once because the PyTorchJob reads 404, and
    the control plane takes it as the user's: no containment, nothing stopped or
    restarted. It used to restore the master as failed and, past the cleanup
    deadline, end TIMED_OUT: a containment and a restart of the job the user had
    just deleted."""

    clock = Clock()
    delivered = FakeSink()
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = outbox_watcher(core, delivered, environment, clock)
    first.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [
        deleted(training_pod(0, nodes[0], exit_code=1), clock()),
        deleted(training_pod(1, nodes[1]), clock()),
    ]
    first.run_once()
    after_deletion = persisted_pods(core)

    clock.value += timedelta(seconds=60)
    core.pods = []
    workload = DeletedPyTorchJob()
    restarted = outbox_watcher(
        core,
        delivered,
        environment,
        clock,
        stopper=SimpleNamespace(custom=workload, batch=None),
    )
    restarted.run_once()

    assert workload.reads == 1, "the restored attempt reads its workload object once"
    assert posted(delivered, FAILURE_PATH) == [], "a deletion was reported as a failure"
    [terminal] = posted(delivered, TERMINAL_PATH)
    assert (
        terminal["terminal_status"],
        terminal["termination_initiator_incident_id"],
    ) == ("STOPPED", None), terminal
    assert rank_gpus(terminal["allocation"]) == {0: gpus(uid(0)), 1: gpus(uid(1))}, (
        terminal["allocation"]
    )
    context, decisions = control_plane_reads(delivered)
    assert [(item.status, item.reason) for item in decisions] == [
        (
            DecisionStatus.NO_ACTION,
            "controller-initiated or user stop; no automatic restart",
        )
    ], decisions
    assert_nothing_stopped(context)
    assert after_deletion == {
        0: {"uid": uid(0), "terminated": True, "exit_code": 1, "deleting": True},
        1: {"uid": uid(1), "terminated": False, "deleting": True},
    }, after_deletion
    assert json.loads(core.config_map_data[ACTIVE_KEY]) == {}, (
        "the terminal reading must drop the record"
    )


def test_a_restored_attempt_whose_pods_are_gone_still_says_they_were_deleted() -> None:
    """What reaches the control plane after a restart is what reached it before.
    A restored attempt whose Pods are gone, and whose workload object does not
    answer the first read, republishes its record for the missing grace; that
    copy used to say the Pods were not being deleted at all."""

    clock = Clock()
    delivered = FakeSink()
    nodes = ("node-a", "node-b")
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = outbox_watcher(core, delivered, Environment.EKS, clock)
    first.run_once()
    core.pods = [
        deleted(training_pod(0, nodes[0], exit_code=1), clock()),
        deleted(training_pod(1, nodes[1]), clock()),
    ]
    first.run_once()
    before = posted(delivered, OBSERVATION_PATH)[-1]

    core.pods = []
    restarted = outbox_watcher(
        core,
        delivered,
        Environment.EKS,
        clock,
        stopper=SimpleNamespace(custom=StopNotYetReadable(), batch=None),
    )
    restarted.run_once()

    after = posted(delivered, OBSERVATION_PATH)[-1]
    assert [item["deletion_requested"] for item in before["containers"]] == [
        True,
        True,
    ], before["containers"]
    assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True), (
        "a restored attempt with no Pods left may only republish what it "
        f"persisted:\nbefore={before}\nafter ={after}"
    )
    assert posted(delivered, TERMINAL_PATH) == [], "the grace has not ended yet"


def test_the_deletion_request_costs_no_byte_until_true_and_is_never_inferred() -> None:
    """The record of a Pod nobody is deleting is byte for byte the one an earlier
    release wrote -- so its digest is too, and an upgrade rewrites nothing --
    while a Pod being deleted, running or exited, carries ``deleting: true``.
    Only a literal ``true`` restores a deletion: a record without the key (every
    record an earlier release wrote, an exit included) or with any other value
    restores none."""

    clock = Clock()
    delivered = FakeSink()
    core = FakeCoreApi([training_pod(0, "node-a"), training_pod(1, "node-b")])
    subject = outbox_watcher(core, delivered, Environment.EKS, clock)
    subject.run_once()
    running = posted(delivered, OBSERVATION_PATH)[-1]
    core.pods = [
        deleted(training_pod(0, "node-a"), clock()),
        deleted(training_pod(1, "node-b", exit_code=143), clock()),
    ]
    subject.run_once()
    deleting = posted(delivered, OBSERVATION_PATH)[-1]

    assert json.dumps(attempt_state_record(running), sort_keys=True) == json.dumps(
        earlier_release_record(running), sort_keys=True
    ), "a Pod nobody is deleting must cost no byte"
    record = attempt_state_record(deleting)
    assert [
        (pod.get("terminated"), pod.get("deleting")) for pod in record[PODS_KEY]
    ] == [(False, True), (True, True)], record[PODS_KEY]
    restored = AttemptObservation.model_validate(attempt_observation_payload(record))
    assert [item.deletion_requested for item in restored.containers] == [True, True], (
        restored.containers
    )

    for written in (None, False, "true", 1, "yes"):
        pods = [dict(pod) for pod in record[PODS_KEY]]
        for pod in pods:
            pod.pop("deleting")
            if written is not None:
                pod["deleting"] = written
        restored = AttemptObservation.model_validate(
            attempt_observation_payload({**record, PODS_KEY: pods})
        )
        assert [item.deletion_requested for item in restored.containers] == [
            False,
            False,
        ], f"deleting={written!r} must not restore a deletion"


def test_an_earlier_watcher_restores_a_record_this_one_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility, new record -> earlier watcher (a rollback). The
    earlier release keeps only the Pod keys it knows: it restores the attempt
    with nothing skipped and reads the deletion request as absent -- exactly
    its own reading -- and its first write leaves a record it would have
    written itself."""

    clock = Clock()
    delivered = FakeSink()
    nodes = ("node-a", "node-b")
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = outbox_watcher(core, delivered, Environment.HYPERPOD_EKS, clock)
    first.run_once()
    core.pods = [
        training_pod(0, nodes[0]),
        deleted(training_pod(1, nodes[1], exit_code=143), clock()),
    ]
    first.run_once()
    assert persisted_pods(core)[1].get("deleting") is True, persisted_pods(core)

    monkeypatch.setattr(
        completion_attempt_state, "attempt_observation_payload", earlier_release_payload
    )
    monkeypatch.setattr(
        completion_attempt_state, "attempt_state_record", earlier_release_record
    )
    core.pods = []
    earlier = outbox_watcher(
        core,
        delivered,
        Environment.HYPERPOD_EKS,
        clock,
        stopper=SimpleNamespace(custom=StopNotYetReadable(), batch=None),
    )
    assert earlier.restore_skipped_total == 0, "the earlier release skipped the record"
    earlier.run_once()

    republished = posted(delivered, OBSERVATION_PATH)[-1]
    assert republished["workload_phase"] == "RUNNING", republished
    assert [
        (item["rank"], item["deletion_requested"]) for item in republished["containers"]
    ] == [(0, False), (1, False)], republished["containers"]
    assert persisted_pods(core) == {
        0: {"uid": uid(0), "terminated": False},
        1: {"uid": uid(1), "terminated": True, "exit_code": 143},
    }, "the earlier release rewrote the record in its own format"


def test_this_watcher_restores_an_earlier_record_and_rewrites_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility, earlier record -> this watcher (the upgrade). The
    record an earlier release left for a running attempt is restored with
    nothing skipped, the first pass publishes the reading the earlier process
    published and writes nothing back: the record is already what this release
    writes for a Pod nobody is deleting."""

    clock = Clock()
    delivered = FakeSink()
    nodes = ("node-a", "node-b")
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    with monkeypatch.context() as earlier:
        earlier.setattr(
            completion_attempt_state, "attempt_state_record", earlier_release_record
        )
        outbox_watcher(core, delivered, Environment.EKS, clock).run_once()
    document = core.config_map_data[ACTIVE_KEY]
    version = core.config_map_version
    before = posted(delivered, OBSERVATION_PATH)[-1]

    upgraded = outbox_watcher(core, delivered, Environment.EKS, clock)
    assert upgraded.restore_skipped_total == 0, "the upgrade skipped the record"
    upgraded.run_once()

    after = posted(delivered, OBSERVATION_PATH)[-1]
    assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True), (
        f"the upgrade changed the published reading:\nbefore={before}\nafter ={after}"
    )
    assert (core.config_map_data[ACTIVE_KEY], core.config_map_version) == (
        document,
        version,
    ), "the upgrade rewrote a record it reads exactly as the earlier release did"
