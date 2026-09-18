"""Real STOP mutations must not be mistaken for unrelated workload drift."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_EXECUTION_EPOCH,
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_STEP_INDEX,
    ANNOTATION_TERMINATION_INCIDENT,
    ANNOTATION_WORKFLOW,
)
from gpu_fault.adapters.kubernetes.stop_ownership import stop_ownership_scope
from gpu_fault.models import WorkflowStepStatus
from scripts.e2e.regional import late_ownership_resources as resources
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    NodeIdentity,
    Participant,
    StopReceipt,
    WorkloadIdentity,
)
from tests.regional._late_ownership_runtime import runtime
from tests.regional._late_ownership_support import evidence, scope
from tests.regional.test_late_ownership_resources import Api

Transition = tuple[Any, resources.OwnedMutation, StopReceipt]


@pytest.fixture
def transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Transition:
    state, adapter, validator, context = cast(Any, runtime)()
    original = deepcopy(state.job)
    data = scope(getattr(request, "param", "ownership-drift")).model_dump()
    data.update(
        cluster_id=context.incident.cluster_id,
        workflow_id=context.workflow.request_id,
        incident_id=context.incident.incident_id,
        fencing_token=context.workflow.fencing_token,
        execution_epoch=context.workflow.execution_epoch,
        nodes=tuple(
            NodeIdentity(
                name=name,
                uid=value["metadata"]["uid"],
                boot_id=value["status"]["nodeInfo"]["bootID"],
            )
            for name, value in state.nodes.items()
        ),
        workload=WorkloadIdentity(
            namespace=state.job["metadata"]["namespace"],
            name=state.job["metadata"]["name"],
            uid=state.job["metadata"]["uid"],
            owner_uid=state.job["metadata"]["uid"],
            attempt_id=context.incident.attempt_id,
        ),
        participants=tuple(
            Participant(
                pod_uid=pod["metadata"]["uid"],
                owner_uid=state.job["metadata"]["uid"],
                node_uid=state.nodes[pod["spec"]["nodeName"]]["metadata"]["uid"],
            )
            for pod in state.pods
        ),
    )
    binding = AcceptanceScope.model_validate(data)
    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert not state.pods and state.job["spec"]["runPolicy"]["suspend"] is True
    assert ("patch-workload", "training", "job") in state.calls
    api = cast(Any, Api)(binding)
    api.source = deepcopy(state.job)
    api.objects = {("pytorchjob", binding.workload.name): deepcopy(state.job)}
    monkeypatch.setattr(
        resources,
        "read_resource",
        lambda regional, kind, name: regional.read(kind, name),
    )
    monkeypatch.setattr(
        resources, "delete_resource", lambda regional, value: regional.delete(value)
    )
    mutation = resources.OwnedMutation(api, binding, original, "approved-image")
    mutation.journal_path = tmp_path / "mutation.json"
    stop_data = evidence().stop.model_dump()
    stop_data.update(
        scope_sha256=binding.digest(),
        executor_uid=binding.executor_uid,
        stop_command_id=context.idempotency_key,
        workload=binding.workload,
        participants=binding.participants,
        absent_pod_uids=tuple(item.pod_uid for item in binding.participants),
        empty_client_node_uids=tuple(item.uid for item in binding.nodes),
    )
    stop = StopReceipt.model_validate(stop_data)
    return api, mutation, stop


@pytest.mark.parametrize(
    ("transition", "method"),
    [("ownership-drift", "change_owner"), ("late-sibling", "late_sibling")],
    indirect=["transition"],
)
def test_actual_stop_transition_permits_only_owned_mutation_and_cleanup(
    transition: Transition, method: str
) -> None:
    api, mutation, stop = transition
    original = deepcopy(mutation.source)
    stopped = deepcopy(api.source)
    assert original["spec"]["runPolicy"]["suspend"] is False
    assert (
        stopped["metadata"]["annotations"][ANNOTATION_OPERATION] == stop.stop_command_id
    )
    with pytest.raises(BoundaryDenied, match="declared state changed"):
        mutation.source_object()
    mutation.acknowledge_stop(stop)
    assert mutation.source_object() == stopped
    assert mutation.source == original, (
        "a readback cannot replace approved source intent"
    )
    assert mutation.journal_path is not None, "the STOP receipt needs a durable journal"
    saved = json.loads(mutation.journal_path.read_text())
    assert saved["stop_receipt"] == stop.model_dump(mode="json")
    getattr(mutation, method)()
    mutation.cleanup()
    assert mutation.source_object() == stopped
    assert list(api.objects) == [("pytorchjob", mutation.scope.workload.name)]
    assert mutation.source == original


@pytest.mark.parametrize(
    "annotation",
    [
        ANNOTATION_INCIDENT,
        ANNOTATION_WORKFLOW,
        ANNOTATION_FENCING,
        ANNOTATION_EXECUTION_EPOCH,
        ANNOTATION_STEP_INDEX,
        ANNOTATION_OPERATION,
        ANNOTATION_TERMINATION_INCIDENT,
        "foreign-controller-field",
    ],
)
def test_unapproved_annotations_cannot_be_rebased_as_a_successful_stop(
    transition: Transition, annotation: str
) -> None:
    api, mutation, stop = transition
    api.objects[("pytorchjob", mutation.scope.workload.name)]["metadata"][
        "annotations"
    ][annotation] = "foreign"
    with pytest.raises(BoundaryDenied, match="declared state changed"):
        mutation.acknowledge_stop(stop)
    assert mutation.stop_receipt is None and not api.calls


@pytest.mark.parametrize(
    "defect", ["active", "spec", "owner", "deleting", "uid", "gone"]
)
def test_stop_acknowledgement_does_not_authorize_other_source_drift(
    transition: Transition, defect: str
) -> None:
    api, mutation, stop = transition
    current = api.objects[("pytorchjob", mutation.scope.workload.name)]
    if defect == "active":
        current["spec"]["runPolicy"]["suspend"] = False
    elif defect == "spec":
        current["spec"]["unapproved"] = True
    elif defect == "owner":
        current["metadata"]["ownerReferences"] = [{"uid": "foreign"}]
    elif defect == "deleting":
        current["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    elif defect == "uid":
        current["metadata"]["uid"] = "replacement"
    else:
        api.objects.clear()
    with pytest.raises(BoundaryDenied):
        mutation.acknowledge_stop(stop)
    assert mutation.stop_receipt is None and not api.calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope_sha256", "d" * 64),
        ("executor_uid", "replacement"),
        ("participants", ()),
        ("stop_command_id", "workflow-local/1/STOP_WORKLOADS"),
    ],
)
def test_stop_receipt_must_bind_the_original_first_step_before_any_read(
    transition: Transition, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    api, mutation, stop = transition

    def forbidden_read(*args: object) -> None:
        raise AssertionError("an unbound STOP receipt reached Kubernetes")

    monkeypatch.setattr(resources, "read_resource", forbidden_read)
    with pytest.raises(BoundaryDenied):
        mutation.acknowledge_stop(stop.model_copy(update={field: value}))
    assert mutation.stop_receipt is None and not api.calls


@pytest.mark.parametrize("state", ["repeat", "created", "started"])
def test_stop_transition_cannot_be_rearmed_after_forward_progress(
    transition: Transition, state: str
) -> None:
    api, mutation, stop = transition
    if state == "repeat":
        mutation.acknowledge_stop(stop)
    elif state == "created":
        mutation.resources.append({"name": "existing"})
    else:
        mutation.mutation_started = True
    with pytest.raises(BoundaryDenied, match="late, repeated or unbound"):
        mutation.acknowledge_stop(stop)
    assert not api.calls, "a repeated STOP transition must not mutate Kubernetes"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("apiVersion", "foreign/v1"),
        ("kind", "Job"),
        ("name", "foreign"),
        ("namespace", "foreign"),
        ("uid", "foreign"),
        ("annotations", None),
        ("spec", None),
        ("runPolicy", None),
    ],
)
def test_invalid_original_intent_cannot_authorize_a_stop_projection(
    transition: Transition, field: str, value: object
) -> None:
    api, mutation, stop = transition
    if field in {"apiVersion", "kind", "spec"}:
        mutation.source[field] = value
    elif field == "runPolicy":
        mutation.source["spec"][field] = value
    else:
        mutation.source["metadata"][field] = value
    with pytest.raises(BoundaryDenied, match="STOP source"):
        mutation.acknowledge_stop(stop)
    assert mutation.stop_receipt is None and not api.calls


def test_stop_projection_keeps_status_and_resource_version_as_server_fields(
    transition: Transition,
) -> None:
    api, mutation, stop = transition
    current = api.objects[("pytorchjob", mutation.scope.workload.name)]
    current["metadata"]["resourceVersion"] = "newer"
    current["metadata"]["generation"] = 2
    current["status"] = {"conditions": [{"type": "Suspended", "status": "True"}]}
    mutation.acknowledge_stop(stop)
    assert mutation.source_object() == current


@pytest.mark.parametrize("field", ["annotations", "spec"])
def test_later_drift_cannot_be_hidden_by_the_acknowledged_stop(
    transition: Transition, field: str
) -> None:
    api, mutation, stop = transition
    mutation.acknowledge_stop(stop)
    mutation.change_owner()
    current = api.objects[("pytorchjob", mutation.scope.workload.name)]
    if field == "annotations":
        current["metadata"]["annotations"][ANNOTATION_INCIDENT] = "foreign"
    else:
        current["spec"]["runPolicy"]["suspend"] = False
    before = deepcopy(api.objects)
    calls = len(api.calls)
    with pytest.raises(BoundaryDenied, match="declared state changed"):
        mutation.cleanup()
    assert api.objects == before and len(api.calls) == calls
