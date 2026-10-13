"""The watcher ends the missing-Pod grace for an attempt our own stop suspended.

STOP_WORKLOADS writes ``gpu-fault.io/termination-initiator-incident-id``, its
step key and ``suspend`` on the workload object in one patch and only then
deletes the Pods, so the first watcher pass that finds no Pod of the attempt can
read all three. A
suspended object recreates nothing; re-posting the stopped attempt as RUNNING
for the whole 300-s grace kept it bound to its node next to the successor that
RESTART_WORKLOAD started there -- two fresh owners per node, a critical
``GpuFaultExclusiveNodeOwnershipInvariantViolation`` after every product-driven
restart, and every fault event on that node attributed to no job (round 8,
live). Everything that is not provably our own suspend keeps the grace (F2).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.workload_observations import ingest_workload_observation
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_observation import (
    OPERATION_ANNOTATION,
    TERMINATION_INCIDENT_ANNOTATION,
    workload_objects_initiator,
    workload_objects_stopped_by_us,
)
from gpu_fault.models import TerminalEvent, WorkflowOperation, WorkflowStepStatus
from gpu_fault.orchestration.families.evidence import EvidenceOperationService
from gpu_fault.policy import XidEvent
from gpu_fault.remote_step_batching import step_idempotency_key
from gpu_fault.watcher import AttemptObservation
from tests._builders import workflow_step, workflow_step_execution
from tests.completion._support import Clock, FakeCoreApi, FakeSink, FakeStopper, pod
from tests.execution._cov95_runtime_restart import RestartHarness

SOURCE = "train-a1"
SUCCESSOR = "train-a1-r-48464154"
STOP_INCIDENT = "inc-stop"
STOP_OPERATION = "wf-stop/2/STOP_WORKLOADS"
RESTART_OPERATION = "wf-stop/5/RESTART_WORKLOAD"
OBSERVATIONS = "/v1/workload-observations"
TERMINAL = "/v1/attempts/terminal"
EARLY_STOP_LOG = "our own stop"


def workload_object(
    kind: str,
    *,
    suspended: bool = True,
    initiator: str | None = STOP_INCIDENT,
    operation: str | None = STOP_OPERATION,
) -> dict[str, Any]:
    """The object STOP_WORKLOADS leaves behind: suspended, annotated, stamped
    with the stop's step key, and with ``status.failed`` from the Pods it
    deleted (which must not be read as an outcome, C1)."""

    annotations = {TERMINATION_INCIDENT_ANNOTATION: initiator} if initiator else {}
    if operation:
        annotations[OPERATION_ANNOTATION] = operation
    spec: dict[str, Any] = (
        {"runPolicy": {"suspend": suspended}}
        if kind == "pytorchjob"
        else {"suspend": suspended}
    )
    return {
        "metadata": {"name": "trainer", "annotations": annotations},
        "spec": spec,
        "status": {"failed": 1},
    }


class WorkloadObjects:
    """``BatchV1Api`` and ``CustomObjectsApi`` in one: serves objects by
    ``namespace/kind/name`` and answers anything else with its HTTP status."""

    _PLURALS = {"pytorchjobs": "pytorchjob", "jobsets": "jobset"}

    def __init__(
        self,
        objects: dict[str, dict[str, Any]] | None = None,
        errors: dict[str, int] | None = None,
    ) -> None:
        self.objects = dict(objects or {})
        self.errors = dict(errors or {})
        self.reads: list[str] = []

    def _read(self, workload_id: str, timeout: Any) -> dict[str, Any]:
        assert timeout is not None, "the workload read must stay bounded"
        self.reads.append(workload_id)
        if workload_id in self.objects:
            return self.objects[workload_id]
        error = RuntimeError(f"{workload_id}: {self.errors.get(workload_id, 404)}")
        error.status = self.errors.get(workload_id, 404)  # type: ignore[attr-defined]
        raise error

    def read_namespaced_job(self, name, namespace, **kwargs):
        return self._read(f"{namespace}/job/{name}", kwargs.get("_request_timeout"))

    def get_namespaced_custom_object(
        self, group, version, namespace, plural, name, **kwargs
    ):
        kind = self._PLURALS[plural]
        return self._read(f"{namespace}/{kind}/{name}", kwargs.get("_request_timeout"))


def watcher(
    apis: WorkloadObjects,
    core: FakeCoreApi,
    sink: Any,
    clock: Clock,
    *,
    grace: float = 300,
) -> KubernetesCompletionController:
    stopper = FakeStopper()
    stopper.batch = apis
    stopper.custom = apis
    return KubernetesCompletionController(
        core,
        sink,
        cluster_id="hp-cluster",
        cleanup_timeout_seconds=30,
        attempt_missing_grace_seconds=grace,
        now=clock,
        workload_stopper=stopper,
        publish_observations=True,
    )


def posted(sink: Any, path: str) -> list[dict[str, Any]]:
    return [payload for item, payload in sink.posts if item == path]


def stop_pods(apis: WorkloadObjects, workload_ids: list[str], *, passes: int = 1):
    """Watch one running attempt, then let STOP_WORKLOADS take its Pods and
    run ``passes`` more passes 5 s apart: far inside the 300-s grace."""

    clock = Clock()
    core = FakeCoreApi([pod(0, workload_ids=workload_ids)])
    sink = FakeSink()
    subject = watcher(apis, core, sink, clock)
    subject.run_once()
    core.pods = []
    for _ in range(passes):
        clock.value += timedelta(seconds=5)
        subject.run_once()
    return subject, core, sink, clock


@pytest.mark.parametrize(
    "workload_id",
    ["default/job/trainer", "default/pytorchjob/trainer", "default/jobset/trainer"],
)
def test_attempt_our_stop_suspended_is_stopped_on_the_first_missing_pass(
    workload_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    kind = workload_id.split("/")[1]
    apis = WorkloadObjects({workload_id: workload_object(kind)})
    caplog.set_level(logging.INFO, logger="gpu_fault.completion_observation")

    _subject, _core, sink, _clock = stop_pods(apis, [workload_id], passes=3)

    observations = posted(sink, OBSERVATIONS)
    stopped = [item for item in observations if item["workload_phase"] == "STOPPED"]
    assert observations[0]["workload_phase"] == "RUNNING", observations[0]
    # The first missing pass already publishes the tombstone, initiator and all.
    assert observations[1]["workload_phase"] == "STOPPED", observations[1]
    assert observations[1]["containers"] == []
    assert observations[1]["termination_initiator_incident_id"] == STOP_INCIDENT
    assert len(stopped) == len(observations) - 1, (
        f"a later pass re-posted the stopped attempt as live: {observations}"
    )
    terminals = posted(sink, TERMINAL)
    assert [item["terminal_status"] for item in terminals] == ["STOPPED"]
    assert terminals[0]["termination_initiator_incident_id"] == STOP_INCIDENT
    assert apis.reads == [workload_id], (
        f"the early stop must reuse the one read of the first pass: {apis.reads}"
    )
    early = [
        record
        for record in caplog.records
        if EARLY_STOP_LOG in record.getMessage() and record.levelno == logging.INFO
    ]
    assert len(early) == 1, f"expected the early stop logged once: {early}"
    assert STOP_INCIDENT in early[0].getMessage(), early[0].getMessage()


def test_every_workload_object_suspended_by_us_stops_a_multi_object_attempt() -> None:
    ids = ["default/pytorchjob/trainer", "default/job/side"]
    apis = WorkloadObjects(
        {ids[0]: workload_object("pytorchjob"), ids[1]: workload_object("job")}
    )

    _subject, _core, sink, _clock = stop_pods(apis, ids)

    terminals = posted(sink, TERMINAL)
    assert [item["terminal_status"] for item in terminals] == ["STOPPED"]
    assert terminals[0]["termination_initiator_incident_id"] == STOP_INCIDENT


KEEP_GRACE_CASES = {
    # Our annotation without a suspend: whatever owns the object may still
    # recreate the Pods.
    "not-suspended": (
        {"default/pytorchjob/trainer": workload_object("pytorchjob", suspended=False)},
        {},
        ["default/pytorchjob/trainer"],
    ),
    # A user or Kueue suspend comes back by itself (F2).
    "suspended-without-our-annotation": (
        {"default/job/trainer": workload_object("job", initiator=None)},
        {},
        ["default/job/trainer"],
    ),
    # So does a suspend of a restarted object: RESTART_WORKLOAD copies our
    # initiator annotation into the successor but stamps its own step key.
    "suspended-after-our-restart": (
        {"default/job/trainer": workload_object("job", operation=RESTART_OPERATION)},
        {},
        ["default/job/trainer"],
    ),
    "annotated-without-our-stop-stamp": (
        {"default/jobset/trainer": workload_object("jobset", operation=None)},
        {},
        ["default/jobset/trainer"],
    ),
    "suspend-in-the-wrong-place": (
        {
            "default/pytorchjob/trainer": {
                **workload_object("pytorchjob"),
                "spec": {"runPolicy": {}, "suspend": "true"},
            }
        },
        {},
        ["default/pytorchjob/trainer"],
    ),
    "unreadable": ({}, {"default/jobset/trainer": 504}, ["default/jobset/trainer"]),
    "rbac-refusal": ({}, {"default/job/trainer": 403}, ["default/job/trainer"]),
    "partial-one-unreadable": (
        {"default/pytorchjob/trainer": workload_object("pytorchjob")},
        {"default/job/side": 504},
        ["default/pytorchjob/trainer", "default/job/side"],
    ),
    "partial-one-gone": (
        {"default/pytorchjob/trainer": workload_object("pytorchjob")},
        {"default/job/side": 404},
        ["default/pytorchjob/trainer", "default/job/side"],
    ),
    "partial-one-not-ours": (
        {
            "default/pytorchjob/trainer": workload_object("pytorchjob"),
            "default/job/side": workload_object("job", initiator=None),
        },
        {},
        ["default/pytorchjob/trainer", "default/job/side"],
    ),
}


@pytest.mark.parametrize("case", sorted(KEEP_GRACE_CASES))
def test_anything_but_our_own_suspend_keeps_the_missing_grace(case: str) -> None:
    objects, errors, ids = KEEP_GRACE_CASES[case]
    apis = WorkloadObjects(objects, errors)

    _subject, _core, sink, _clock = stop_pods(apis, ids, passes=3)

    observations = posted(sink, OBSERVATIONS)
    assert [item["workload_phase"] for item in observations] == ["RUNNING"] * 4, (
        f"{case}: the grace did not hold the attempt: {observations}"
    )
    assert observations[-1]["containers"], "the re-post lost its containers"
    assert posted(sink, TERMINAL) == []
    assert len(apis.reads) == len(ids), (
        f"{case}: later grace passes must not read again: {apis.reads}"
    )


def test_attempt_without_workload_ids_keeps_the_grace() -> None:
    apis = WorkloadObjects()

    _subject, _core, sink, _clock = stop_pods(apis, [], passes=2)

    assert posted(sink, OBSERVATIONS)[-1]["workload_phase"] == "RUNNING"
    assert posted(sink, TERMINAL) == []
    assert apis.reads == []


def test_an_attempt_suspended_by_someone_else_still_ends_after_the_grace() -> None:
    """The kept grace still ends the way it always did: one fresh read at the
    end, and no initiator when the object does not name one."""

    apis = WorkloadObjects(
        {"default/job/trainer": workload_object("job", initiator=None)}
    )
    subject, _core, sink, clock = stop_pods(apis, ["default/job/trainer"])

    clock.value += timedelta(seconds=301)
    subject.run_once()

    terminals = posted(sink, TERMINAL)
    assert [item["terminal_status"] for item in terminals] == ["STOPPED"]
    assert terminals[0].get("termination_initiator_incident_id") is None


def test_unsuspended_object_brings_the_early_stopped_attempt_back() -> None:
    """Un-suspending the old object recreates Pods under the same attempt id:
    the early tombstone is still only a missing-Pod tombstone, and live Pods
    disprove it (F2)."""

    workload_id = "default/pytorchjob/trainer"
    apis = WorkloadObjects({workload_id: workload_object("pytorchjob")})
    subject, core, sink, clock = stop_pods(apis, [workload_id])
    assert [item["terminal_status"] for item in posted(sink, TERMINAL)] == ["STOPPED"]

    apis.objects[workload_id] = workload_object("pytorchjob", suspended=False)
    core.pods = [pod(0, workload_ids=[workload_id])]
    clock.value += timedelta(seconds=5)
    subject.run_once()

    assert subject.resumed_attempts_total == 1, "the tombstone was not withdrawn"
    resumed = posted(sink, OBSERVATIONS)[-1]
    assert resumed["workload_phase"] == "RUNNING", resumed
    assert [item["node_id"] for item in resumed["containers"]] == ["node-0"]

    core.pods = [pod(0, exit_code=1, workload_ids=[workload_id])]
    clock.value += timedelta(seconds=5)
    subject.run_once()

    failures = posted(sink, "/v1/attempts/failure-detected")
    assert [item["exit_code"] for item in failures] == [1], failures
    assert [item["terminal_status"] for item in posted(sink, TERMINAL)] == [
        "STOPPED",
        "FAILED",
    ]


class ControlPlaneSink:
    """The watcher's sink wired straight into one control-plane context, the
    way the ingress routes do it."""

    def __init__(self, context: ApplicationContext) -> None:
        self.context = context
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        if path == OBSERVATIONS:
            ingest_workload_observation(
                self.context, AttemptObservation.model_validate(payload)
            )
        elif path == TERMINAL:
            decision = self.context.completion.handle_terminal(
                TerminalEvent.model_validate(payload)
            )
            return dict(decision.model_dump(mode="json"))
        return {"accepted": True}


def control_plane_pod(attempt_id: str, workload_id: str) -> dict[str, Any]:
    item = pod(0, attempt_id=attempt_id, workload_ids=[workload_id])
    metadata = item["metadata"]
    metadata["annotations"]["gpu-fault.io/runtime-profile-version"] = "simulated-v1"
    if attempt_id != SOURCE:
        metadata["name"] = f"{attempt_id}-worker-0"
        metadata["uid"] = f"{attempt_id}-pod-0"
    return item


def test_restart_leaves_one_owner_per_node_in_the_control_plane_evidence() -> None:
    """STOP_WORKLOADS -> RESTART_WORKLOAD on one node, end to end through the
    evidence the ownership gauge and per-event attribution read."""

    context = ApplicationContext()
    sink = ControlPlaneSink(context)
    evidence = EvidenceOperationService(context.store, lambda *_args: None)
    source_id = "default/pytorchjob/trainer"
    successor_id = "default/pytorchjob/trainer-r"
    apis = WorkloadObjects()
    clock = Clock()
    core = FakeCoreApi([control_plane_pod(SOURCE, source_id)])
    subject = watcher(apis, core, sink, clock)
    subject.run_once()
    node = ("hp-cluster", "node-0")
    assert evidence.ownership_metric_snapshot(now=clock())["current"][node] == 1

    # STOP_WORKLOADS: annotate + suspend the object, then delete its Pods.
    apis.objects[source_id] = workload_object("pytorchjob")
    core.pods = []
    clock.value += timedelta(seconds=15)
    subject.run_once()
    # RESTART_WORKLOAD: the successor attempt starts on the same node.
    apis.objects[successor_id] = workload_object(
        "pytorchjob", suspended=False, operation=RESTART_OPERATION
    )
    core.pods = [control_plane_pod(SUCCESSOR, successor_id)]
    clock.value += timedelta(seconds=60)
    subject.run_once()

    stored = {
        item.attempt_id: item
        for item in context.store.list_attempt_observations("hp-cluster")
    }
    assert stored[SOURCE].workload_phase.value == "STOPPED", stored[SOURCE]
    assert stored[SOURCE].termination_initiator_incident_id == STOP_INCIDENT
    assert stored[SUCCESSOR].workload_phase.value == "RUNNING", stored[SUCCESSOR]
    snapshot = evidence.ownership_metric_snapshot(now=clock())
    assert snapshot["current"][node] == 1, snapshot
    event = XidEvent(
        cluster_id="hp-cluster",
        node_id="node-0",
        observed_at=clock(),
        ingested_at=clock(),
        xid=79,
        gpu_uuid="GPU-0",
        runtime_profile_version="simulated-v1",
    )
    owner = evidence.attempt_observation(event)
    assert owner is not None, "the event on the restarted node was attributed to no job"
    assert (owner.job_id, owner.attempt_id) == ("train", SUCCESSOR)
    assert evidence.ambiguous_attempt_ownership_total() == 0
    terminals = [payload for path, payload in sink.posts if path == TERMINAL]
    assert [(item["attempt_id"], item["terminal_status"]) for item in terminals] == [
        (SOURCE, "STOPPED")
    ]


def merge_patch(source: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """A JSON merge patch applied the way the API server applies it."""

    result = deepcopy(source)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_patch(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def suspend_spec(kind: str, body: dict[str, Any], suspended: bool) -> None:
    spec = body.setdefault("spec", {})
    target = spec.setdefault("runPolicy", {}) if kind == "pytorchjob" else spec
    target["suspend"] = suspended


@pytest.mark.parametrize(
    ("kind", "terminal"),
    [
        ("job", True),
        ("pytorchjob", True),
        ("pytorchjob", False),
        ("jobset", True),
        ("jobset", False),
    ],
)
def test_the_adapters_stop_reads_as_ours_and_a_suspended_successor_does_not(
    kind: str, terminal: bool
) -> None:
    """The writer the watcher trusts, through the real Kubernetes adapter:
    STOP_WORKLOADS (dispatched under the executor's step key) leaves an object
    that ends the grace; RESTART_WORKLOAD -- a new object, or the old one
    resumed in place -- carries our initiator annotation over, so a later
    user or Kueue suspend of the successor must still keep the grace (F2)."""

    harness = RestartHarness(kind, terminal=terminal)
    restart = harness.context
    [workload_id] = restart.step.workload_ids
    stop_step = workflow_step(
        WorkflowOperation.STOP_WORKLOADS,
        "gpu-fault-kubernetes-adapter",
        workload_ids=[workload_id],
    )
    workflow = restart.workflow.model_copy(
        update={"official_steps": [stop_step, restart.step]}
    )
    incident = restart.incident.incident_id
    kubelet = harness.adapter.core
    harness.adapter.core = SimpleNamespace(
        list_namespaced_pod=lambda **_kwargs: SimpleNamespace(items=[])
    )
    harness.context = replace(
        restart,
        step=stop_step,
        workflow=workflow,
        step_index=0,
        idempotency_key=step_idempotency_key(workflow, 0, stop_step),
    )
    suspend_spec(kind, harness.api.source, False)

    stopped = harness.execute()

    assert stopped.status is WorkflowStepStatus.SUCCEEDED, stopped
    [stop_patch] = harness.api.patches
    harness.api.source = merge_patch(harness.api.source, stop_patch)
    assert workload_objects_stopped_by_us([harness.api.source], [workload_id]) == (
        incident
    ), f"the adapter's own stop must end the grace: {stop_patch}"

    restart_key = f"{workflow.request_id}/1/RESTART_WORKLOAD"
    authorization = restart.request.restart_authorization
    assert authorization is not None, restart.request
    harness.adapter.core = kubelet
    harness.api.patches.clear()
    harness.context = replace(
        restart,
        workflow=workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.STOP_WORKLOADS,
                        WorkflowStepStatus.SUCCEEDED,
                    )
                ]
            }
        ),
        step_index=1,
        idempotency_key=restart_key,
        request=restart.request.model_copy(
            update={
                "restart_authorization": authorization.model_copy(
                    update={"reservation_id": restart_key}
                )
            }
        ),
    )

    restarted = harness.execute()

    assert restarted.status is WorkflowStepStatus.SUCCEEDED, restarted
    if harness.api.created:
        [successor] = harness.api.created.values()
    else:
        [restart_patch] = harness.api.patches
        successor = merge_patch(harness.api.source, restart_patch)
    suspend_spec(kind, successor, True)
    assert workload_objects_initiator([successor]) == incident, (
        "the successor no longer inherits the initiator annotation; "
        "revisit the stop-stamp check"
    )
    assert workload_objects_stopped_by_us([successor], [workload_id]) is None, (
        "a suspended successor read as our standing stop"
    )
