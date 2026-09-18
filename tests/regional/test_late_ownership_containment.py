from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.adapters.kubernetes import stop_ownership as guard
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    RestartAuthorization,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.restart_containment import RestartContainmentProof
from tests.regional._late_ownership_runtime import (
    NODES,
    WORKLOAD,
    runtime,
    stopped_runtime,
)


def receipt(context):
    return guard.StopOwnershipReceipt.model_validate_json(
        json.dumps(context.workflow.step_executions[0].details[guard.STOP_RECEIPT_KEY])
    )


def with_receipt(context, value):
    execution = context.workflow.step_executions[0].model_copy(
        update={"details": {guard.STOP_RECEIPT_KEY: value.model_dump(mode="json")}}
    )
    return replace(
        context,
        workflow=context.workflow.model_copy(update={"step_executions": [execution]}),
    )


def refused(validator, context, reason):
    with guard.stop_ownership_scope(validator):
        result = guard.node_submission_ownership_guard(context)
    assert result is not None and result.status is WorkflowStepStatus.FAILED
    assert result.details["reason"] == reason
    assert (
        result.details["safety_rejection"]
        and result.details["manual_confirmation_required"]
    )
    return result


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {},
        {"uid": "uid", "name": "pod"},
        {"uid": 1, "name": "pod", "resourceVersion": "1"},
        {
            "uid": "uid",
            "name": "pod",
            "resourceVersion": "1",
            "deletionTimestamp": "now",
        },
    ],
)
def test_missing_or_terminating_identity_is_not_an_ownership_receipt(metadata):
    state, _adapter, validator, context = runtime()
    state.job["metadata"] = metadata
    with pytest.raises(guard.StopOwnershipError, match="IDENTITY_UNKNOWN"):
        validator.capture(
            context, [("training", "pytorchjob", "job", WORKLOAD, state.job)]
        )


def test_deleting_pod_identity_can_be_observed_but_duplicate_owners_are_ambiguous():
    state, _adapter, validator, context = runtime()
    state.pods[0]["metadata"]["deletionTimestamp"] = "now"
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    captured = validator.capture(context, prepared)
    assert state.pods[0]["metadata"]["uid"] in {pod.uid for pod in captured.pods}
    state.job["metadata"]["ownerReferences"] = {}
    with pytest.raises(guard.StopOwnershipError, match="IDENTITY_UNKNOWN"):
        validator.capture(context, prepared)
    owner = {
        "apiVersion": "v1",
        "kind": "Job",
        "name": "job",
        "uid": "uid",
        "controller": True,
    }
    state.job["metadata"]["ownerReferences"] = [owner, owner]
    with pytest.raises(guard.StopOwnershipError, match="IDENTITY_UNKNOWN"):
        validator.capture(context, prepared)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("namespace", "other", "STOP_OWNERSHIP_IDENTITY_UNKNOWN"),
        ("name", "other", "STOP_OWNERSHIP_IDENTITY_UNKNOWN"),
        ("labels", {}, "STOP_OWNERSHIP_IDENTITY_UNKNOWN"),
        (
            "labels",
            {"gpu-fault.io/managed": "false", "gpu-fault.io/attempt-id": "attempt"},
            "STOP_OWNERSHIP_IDENTITY_UNKNOWN",
        ),
        ("annotations", {}, "STOP_OWNERSHIP_DRIFT"),
    ],
)
def test_workload_metadata_drift_blocks_the_new_action(field, value, reason):
    state, _adapter, validator, context = stopped_runtime()
    state.job["metadata"][field] = value
    refused(validator, context, reason)


def test_stop_scope_workload_kind_and_namespace_are_not_inferred():
    state, _adapter, validator, context = runtime()
    with pytest.raises(guard.StopOwnershipError, match="SCOPE_MISMATCH"):
        validator.capture(
            context,
            [("foreign", "pytorchjob", "job", "foreign/pytorchjob/job", state.job)],
        )
    with pytest.raises(guard.StopOwnershipError, match="SCOPE_MISMATCH"):
        validator.capture(
            context, [("training", "pytorchjob", "job", "wrong-id", state.job)]
        )
    with pytest.raises(ValueError):
        guard.KubernetesStopOwnershipValidator(
            cluster_id="",
            allowed_namespaces=frozenset(),
            core_api=None,
            read_workload=lambda *a: None,
            serialize=lambda value: value,
            workload_active=lambda *a: False,
        )


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("unknown-state", "STOP_PARTICIPANTS_UNKNOWN"),
        ("active-state", "STOP_PARTICIPANTS_ACTIVE"),
        ("unsuspended", "STOP_PARTICIPANTS_ACTIVE"),
        ("wrong-node-name", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("unknown-node-boot", "STOP_OWNERSHIP_UNVERIFIABLE"),
    ],
)
def test_fresh_node_and_workload_state_are_required(defect, reason):
    state, _adapter, validator, context = stopped_runtime()
    if defect == "unknown-state":
        validator.workload_active = lambda *a: None
    elif defect == "active-state":
        validator.workload_active = lambda *a: True
    elif defect == "unsuspended":
        state.job["spec"]["runPolicy"]["suspend"] = False
    elif defect == "wrong-node-name":
        state.nodes["node-b"]["metadata"]["name"] = "other"
    else:
        state.nodes["node-b"]["status"]["nodeInfo"]["bootID"] = ""
    refused(validator, context, reason)


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("missing-items", "STOP_PARTICIPANTS_UNKNOWN"),
        ("paginated", "STOP_PARTICIPANTS_UNKNOWN"),
        ("two-candidates", "STOP_PARTICIPANTS_UNKNOWN"),
        ("label-removed", "STOP_OWNERSHIP_DRIFT"),
        ("no-owner", "STOP_OWNERSHIP_DRIFT"),
        ("noncontroller", "STOP_OWNERSHIP_DRIFT"),
        ("wrong-owner-kind", "STOP_OWNERSHIP_DRIFT"),
        ("wrong-owner-name", "STOP_OWNERSHIP_DRIFT"),
        ("foreign-owner", "STOP_OWNERSHIP_DRIFT"),
        ("foreign-namespace", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("duplicate-pod", "STOP_PARTICIPANTS_UNKNOWN"),
    ],
)
def test_source_pod_inventory_cannot_hide_unknown_or_changed_participants(
    defect, reason
):
    state, _adapter, validator, context = stopped_runtime()
    saved = receipt(context)
    pods = [state.pod("node-a", saved.pods[0].uid)]
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    context = replace(context, step=context.workflow.official_steps[0], step_index=0)
    context = replace(
        context, workflow=context.workflow.model_copy(update={"step_executions": []})
    )
    response = {"items": pods, "metadata": {}}
    if defect == "missing-items":
        response["items"] = None
    elif defect == "paginated":
        response["metadata"]["continue"] = "next-page"
    elif defect == "two-candidates":
        prepared *= 2
    elif defect == "label-removed":
        pods[0]["metadata"]["labels"] = {}
    elif defect == "no-owner":
        pods[0]["metadata"]["ownerReferences"] = []
    elif defect == "noncontroller":
        pods[0]["metadata"]["ownerReferences"][0]["controller"] = False
    elif defect == "wrong-owner-kind":
        pods[0]["metadata"]["ownerReferences"][0]["kind"] = "Job"
    elif defect == "wrong-owner-name":
        pods[0]["metadata"]["ownerReferences"][0]["name"] = "other"
    elif defect == "foreign-owner":
        pods[0]["metadata"]["ownerReferences"][0]["uid"] = "other"
    elif defect == "foreign-namespace":
        pods[0]["metadata"]["namespace"] = "other"
    else:
        pods.append(deepcopy(pods[0]))
    state.list_namespaced_pod = lambda *a: response
    with pytest.raises(guard.StopOwnershipError, match=reason):
        validator.capture(context, prepared)


def test_removed_owner_and_label_still_match_the_original_pod_uid():
    state, _adapter, validator, context = stopped_runtime()
    saved = receipt(context)
    pod = state.pod("node-a", saved.pods[0].uid)
    pod["metadata"].update(labels={}, ownerReferences=[])
    state.pods = [pod]
    refused(validator, context, "STOP_OWNERSHIP_DRIFT")


@pytest.mark.parametrize("defect", ["none", "child-uid", "parent-owner", "not-job"])
def test_jobset_child_job_ownership_is_verified_transitively(defect):
    state, _adapter, validator, context = stopped_runtime()
    metadata = deepcopy(state.job["metadata"])
    metadata.update(name="set", uid="set-uid", ownerReferences=[])
    root = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": metadata,
    }
    context = replace(
        context,
        step=context.workflow.official_steps[0].model_copy(
            update={"workload_ids": ["training/jobset/set"]}
        ),
        step_index=0,
    )
    context = replace(
        context, workflow=context.workflow.model_copy(update={"step_executions": []})
    )
    prepared = [("training", "jobset", "set", "training/jobset/set", root)]
    pod = state.pod("node-a", "jobset-pod")
    pod["metadata"]["ownerReferences"] = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": "child",
            "uid": "child-uid",
            "controller": True,
        }
    ]
    child = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "uid": "child-uid",
            "name": "child",
            "namespace": "training",
            "resourceVersion": "1",
            "ownerReferences": [
                {
                    "apiVersion": "jobset.x-k8s.io/v1alpha2",
                    "kind": "JobSet",
                    "name": "set",
                    "uid": "set-uid",
                    "controller": True,
                }
            ],
        },
    }
    if defect == "child-uid":
        child["metadata"]["uid"] = "other"
    elif defect == "parent-owner":
        child["metadata"]["ownerReferences"][0]["controller"] = False
    elif defect == "not-job":
        pod["metadata"]["ownerReferences"][0]["kind"] = "Deployment"
    state.pods = [pod]
    reads = []
    validator.read_workload = lambda *args: reads.append(args) or child
    if defect == "none":
        observed = validator.capture(context, prepared)
        assert observed.pods[0].uid == "jobset-pod"
        assert reads == [("training", "job", "child")]
    else:
        with pytest.raises(guard.StopOwnershipError, match="OWNERSHIP_DRIFT"):
            validator.capture(context, prepared)


def idle_context(context):
    step = context.step.model_copy(
        update={"workload_ids": [], "node_ids": ["node-a", "node-b"]}
    )
    return replace(
        context,
        step=step,
        workflow=context.workflow.model_copy(
            update={
                "official_steps": [step],
                "step_executions": [],
                "completed_step_indexes": [],
            }
        ),
    )


@pytest.mark.parametrize(
    "resource,amount,phase,container,active",
    [
        ("nvidia.com/gpu", "1", "Running", "containers", True),
        ("nvidia.com/mig-1g.10gb", 1, "Pending", "initContainers", True),
        ("nvidia.com/gpu", 1, "Unknown", "ephemeralContainers", True),
        ("nvidia.com/gpu", 0, "Running", "containers", False),
        ("nvidia.com/gpu", "1", "Succeeded", "containers", False),
        ("nvidia.com/gpu", "1", "Failed", "containers", False),
        ("cpu", "100m", "Running", "containers", False),
    ],
)
def test_idle_action_reads_both_nodes_and_every_gpu_container_family(
    resource, amount, phase, container, active
):
    state, _adapter, validator, context = stopped_runtime()
    pod = state.pod("node-b", "unrelated-gpu-client")
    pod["status"]["phase"] = phase
    pod["spec"]["containers"] = [{"name": "main", "resources": {}}]
    pod["spec"][container] = [
        {
            "name": "resource-holder",
            "resources": {"requests": {resource: amount}, "limits": {resource: amount}},
        }
    ]
    state.pods = [pod]
    current = idle_context(context)
    state.calls.clear()
    with guard.stop_ownership_scope(validator):
        result = guard.node_submission_ownership_guard(current)
    assert (result is not None) is active
    assert [call for call in state.calls if call[0] == "list-node-pods"] == [
        ("list-node-pods", "node-a"),
        ("list-node-pods", "node-b"),
    ]
    if active:
        assert result.details["reason"] == "STOP_PARTICIPANTS_ACTIVE"


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("items", "STOP_PARTICIPANTS_UNKNOWN"),
        ("pagination", "STOP_PARTICIPANTS_UNKNOWN"),
        ("placement", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("containers", "STOP_PARTICIPANTS_UNKNOWN"),
        ("requests", "STOP_PARTICIPANTS_UNKNOWN"),
        ("amount-bool", "STOP_PARTICIPANTS_UNKNOWN"),
        ("amount-float", "STOP_PARTICIPANTS_UNKNOWN"),
        ("amount-negative", "STOP_PARTICIPANTS_UNKNOWN"),
        ("amount-decimal", "STOP_PARTICIPANTS_UNKNOWN"),
        ("io", "STOP_OWNERSHIP_UNVERIFIABLE"),
    ],
)
def test_unknown_idle_gpu_inventory_fails_closed(defect, reason):
    state, _adapter, validator, context = stopped_runtime()
    pod = state.pod("node-a", "unrelated")
    response = {"metadata": {}, "items": [pod]}
    if defect == "items":
        response["items"] = None
    elif defect == "pagination":
        response["metadata"]["continue"] = "not-drained"
    elif defect == "placement":
        pod["spec"]["nodeName"] = "foreign"
    elif defect == "containers":
        pod["spec"]["containers"] = []
    elif defect == "requests":
        pod["spec"]["containers"][0]["resources"]["requests"] = ["not-a-map"]
    elif defect.startswith("amount"):
        pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = {
            "amount-bool": True,
            "amount-float": 1.0,
            "amount-negative": "-1",
            "amount-decimal": "1.0",
        }[defect]
    if defect == "io":

        def failed(**kwargs):
            raise OSError("local fake API")

        state.list_pod_for_all_namespaces = failed
    else:
        state.list_pod_for_all_namespaces = lambda **kw: response
    refused(validator, idle_context(context), reason)


@pytest.mark.parametrize(
    "defect",
    ["cluster", "incident", "workflow-fence", "request-fence", "nodes", "duplicate"],
)
def test_idle_scope_is_not_inferred_or_widened(defect):
    state, _adapter, validator, context = stopped_runtime()
    context = idle_context(context)
    if defect == "cluster":
        context = replace(
            context,
            incident=context.incident.model_copy(update={"cluster_id": "other"}),
        )
    elif defect == "incident":
        context = replace(
            context,
            workflow=context.workflow.model_copy(update={"incident_id": "other"}),
        )
    elif defect == "workflow-fence":
        context = replace(
            context, workflow=context.workflow.model_copy(update={"fencing_token": 2})
        )
    elif defect == "request-fence":
        context = replace(
            context,
            request=context.request.model_copy(update={"expected_fencing_token": 2}),
        )
    else:
        context = replace(
            context,
            step=context.step.model_copy(
                update={"node_ids": [] if defect == "nodes" else ["node-a", "node-a"]}
            ),
        )
    state.calls.clear()
    refused(validator, context, "STOP_OWNERSHIP_SCOPE_MISMATCH")
    assert state.calls == []


@pytest.mark.parametrize(
    "defect", ["cluster", "incident", "fence", "request", "attempt", "empty", "scope"]
)
def test_stop_capture_refuses_bad_scope_before_workload_mutation(defect):
    state, _adapter, validator, context = runtime()
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    reason = "STOP_OWNERSHIP_SCOPE_MISMATCH"
    if defect == "cluster":
        context = replace(
            context,
            incident=context.incident.model_copy(update={"cluster_id": "other"}),
        )
    elif defect == "incident":
        context = replace(
            context,
            workflow=context.workflow.model_copy(update={"incident_id": "other"}),
        )
    elif defect == "fence":
        context = replace(
            context, workflow=context.workflow.model_copy(update={"fencing_token": 2})
        )
    elif defect == "request":
        context = replace(
            context,
            request=context.request.model_copy(update={"expected_fencing_token": 2}),
        )
    elif defect == "attempt":
        context = replace(
            context,
            incident=context.incident.model_copy(update={"attempt_id": "other"}),
        )
        reason = "STOP_OWNERSHIP_DRIFT"
    elif defect == "empty":
        prepared = []
        reason = "STOP_OWNERSHIP_IDENTITY_UNKNOWN"
    else:
        context = replace(
            context,
            step=context.step.model_copy(
                update={"workload_ids": ["training/job/other"]}
            ),
        )
        reason = "STOP_OWNERSHIP_IDENTITY_UNKNOWN"
    with guard.stop_ownership_scope(validator):
        result = guard.capture_stop_ownership(context, prepared)
    assert (
        isinstance(result, WorkflowStepOutcome) and result.details["reason"] == reason
    )
    assert not any("patch" in call[0] or "delete" in call[0] for call in state.calls), (
        "invalid STOP scope must not mutate workloads"
    )


def test_waiting_stop_reuses_exact_receipt_and_rejects_replaced_source():
    state, _adapter, validator, context = runtime()
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    captured = validator.capture(context, prepared)
    assert len(captured.digest()) == 64
    execution = WorkflowStepExecution(
        step_index=0,
        operation=WorkflowOperation.STOP_WORKLOADS,
        status=WorkflowStepStatus.WAITING,
        phase="official",
        details={guard.STOP_RECEIPT_KEY: captured.model_dump(mode="json")},
    )
    context = replace(
        context,
        workflow=context.workflow.model_copy(update={"step_executions": [execution]}),
    )
    assert validator.capture(context, prepared) == captured
    state.job["metadata"]["uid"] = "replacement"
    with pytest.raises(guard.StopOwnershipError, match="DRIFT"):
        validator.capture(context, prepared)


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("no-stop", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("incomplete", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("no-execution", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("no-receipt", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("malformed", "STOP_OWNERSHIP_UNVERIFIABLE"),
        ("uncontained", "STOP_PARTICIPANTS_CHANGED"),
        ("no-time", "STOP_PARTICIPANTS_CHANGED"),
        ("future-time", "STOP_PARTICIPANTS_CHANGED"),
        ("naive-time", "STOP_PARTICIPANTS_CHANGED"),
        ("no-workloads", "STOP_PARTICIPANTS_CHANGED"),
        ("no-nodes", "STOP_PARTICIPANTS_CHANGED"),
        ("foreign-step-node", "STOP_PARTICIPANTS_CHANGED"),
        ("foreign-step-workload", "STOP_PARTICIPANTS_CHANGED"),
        ("stop-scope", "STOP_PARTICIPANTS_CHANGED"),
        ("wrong-epoch", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("wrong-workflow", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("wrong-phase", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("wrong-step", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("wrong-attempt", "STOP_OWNERSHIP_DRIFT"),
    ],
)
def test_receipt_binding_is_mandatory_for_every_new_hardware_action(defect, reason):
    _state, _adapter, validator, context = stopped_runtime()
    saved = receipt(context)
    updates = {
        "uncontained": {"contained": False},
        "no-time": {"completed_at": None},
        "future-time": {"completed_at": datetime.now(timezone.utc) + timedelta(days=1)},
        "naive-time": {"completed_at": datetime.now()},
        "no-workloads": {"workloads": ()},
        "no-nodes": {"nodes": ()},
        "wrong-epoch": {"execution_epoch": 50},
        "wrong-workflow": {"workflow_id": "foreign"},
        "wrong-phase": {"phase": "safety"},
        "wrong-step": {"stop_step_index": 5},
    }
    if defect in updates:
        context = with_receipt(context, saved.model_copy(update=updates[defect]))
    elif defect == "wrong-attempt":
        context = replace(
            context,
            incident=context.incident.model_copy(update={"attempt_id": "other"}),
        )
    elif defect == "foreign-step-node":
        context = replace(
            context, step=context.step.model_copy(update={"node_ids": ["foreign"]})
        )
    elif defect == "foreign-step-workload":
        context = replace(
            context,
            step=context.step.model_copy(
                update={"workload_ids": ["training/job/foreign"]}
            ),
        )
    elif defect == "stop-scope":
        steps = list(context.workflow.official_steps)
        steps[0] = steps[0].model_copy(
            update={"workload_ids": ["training/job/foreign"]}
        )
        context = replace(
            context,
            workflow=context.workflow.model_copy(update={"official_steps": steps}),
        )
    elif defect == "no-stop":
        context = replace(
            context,
            workflow=context.workflow.model_copy(
                update={"official_steps": [context.step]}
            ),
        )
    elif defect == "incomplete":
        context = replace(
            context,
            workflow=context.workflow.model_copy(update={"completed_step_indexes": []}),
        )
    elif defect == "no-execution":
        context = replace(
            context,
            workflow=context.workflow.model_copy(update={"step_executions": []}),
        )
    else:
        execution = context.workflow.step_executions[0].model_copy(
            update={
                "details": {}
                if defect == "no-receipt"
                else {guard.STOP_RECEIPT_KEY: []}
            }
        )
        context = replace(
            context,
            workflow=context.workflow.model_copy(
                update={"step_executions": [execution]}
            ),
        )
    refused(validator, context, reason)


def test_inherited_stop_requires_explicit_predecessor_fence_and_fresh_ownership():
    state, _adapter, validator, context = stopped_runtime()
    original = context.workflow
    execution = original.step_executions[0].model_copy(
        update={
            "details": {
                **original.step_executions[0].details,
                "preemption_reuse": True,
                "inherited_from_workflow_id": original.request_id,
            }
        }
    )
    context = replace(
        context,
        workflow=original.model_copy(
            update={
                "request_id": "successor",
                "predecessor_workflow_id": original.request_id,
                "inherited_step_indexes": [0],
                "fencing_token": 2,
                "step_executions": [execution],
            }
        ),
        incident=context.incident.model_copy(update={"fencing_token": 2}),
        request=context.request.model_copy(update={"expected_fencing_token": 2}),
    )
    with guard.stop_ownership_scope(validator):
        assert guard.node_submission_ownership_guard(context) is None
    state.job["metadata"]["uid"] = "late-replacement"
    refused(validator, context, "STOP_OWNERSHIP_DRIFT")


def test_safety_phase_stop_is_captured_and_verified_with_its_own_phase():
    state, adapter, validator, context = runtime()
    workflow = context.workflow.model_copy(
        update={"safety_only": True, "safety_steps": context.workflow.official_steps}
    )
    context = replace(context, workflow=workflow)
    with guard.stop_ownership_scope(validator):
        outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    saved = guard.StopOwnershipReceipt.model_validate_json(
        json.dumps(outcome.details[guard.STOP_RECEIPT_KEY])
    )
    assert saved.phase == "safety"
    execution = WorkflowStepExecution(
        step_index=0,
        operation=WorkflowOperation.STOP_WORKLOADS,
        status=WorkflowStepStatus.SUCCEEDED,
        phase="safety",
        details=outcome.details,
    )
    context = replace(
        context,
        step=workflow.safety_steps[1],
        step_index=1,
        workflow=workflow.model_copy(
            update={"step_executions": [execution], "completed_step_indexes": [0]}
        ),
    )
    with guard.stop_ownership_scope(validator):
        assert guard.node_submission_ownership_guard(context) is None


@pytest.mark.parametrize("defect", ["none", "owner", "late-sibling"])
def test_software_restart_preserves_source_checks_after_node_recovery(defect):
    state, _adapter, validator, context = stopped_runtime()
    for node in state.nodes.values():
        node["metadata"]["uid"] += "-replaced"
        node["status"]["nodeInfo"]["bootID"] = "new-boot"
    context = replace(
        context,
        step=context.step.model_copy(
            update={
                "operation": WorkflowOperation.RESTART_WORKLOAD,
                "node_ids": ["healthy-spare"],
            }
        ),
    )
    if defect == "owner":
        state.job["metadata"]["uid"] = "replacement"
    elif defect == "late-sibling":
        state.pods = [state.pod("node-b", "late")]
    with guard.stop_ownership_scope(validator):
        outcome = guard.node_submission_ownership_guard(context)
    if defect == "none":
        assert outcome is None, (
            "the existing signed restart authorization still validates the destination"
        )
    else:
        assert outcome is not None and outcome.details["manual_confirmation_required"]


def test_optional_legacy_scope_does_not_erase_required_regional_validation():
    state, _adapter, validator, context = runtime()
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    assert guard.capture_stop_ownership(context, prepared) is None
    outcome = WorkflowStepOutcome.succeeded()
    assert guard.finish_stop_ownership(context, None, outcome) is outcome
    captured = validator.capture(context, prepared)
    assert (
        guard.finish_stop_ownership(context, captured, outcome).details["reason"]
        == "STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE"
    )
    with guard.stop_ownership_scope(None):
        assert guard.capture_stop_ownership(context, prepared).details[
            "manual_confirmation_required"
        ]
        assert (
            guard.node_submission_ownership_guard(context).details["reason"]
            == "STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE"
        )


@pytest.mark.parametrize(
    "error",
    [
        guard.StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN"),
        OSError("private diagnostic"),
    ],
)
def test_capture_and_finish_exceptions_are_safety_refusals(error):
    state, _adapter, validator, context = runtime()
    prepared = [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    captured = validator.capture(context, prepared)

    def fail(*args):
        raise error

    validator.capture = fail
    validator.finish = fail
    with guard.stop_ownership_scope(validator):
        for result in (
            guard.capture_stop_ownership(context, prepared),
            guard.finish_stop_ownership(
                context, captured, WorkflowStepOutcome.succeeded()
            ),
        ):
            assert (
                isinstance(result, WorkflowStepOutcome)
                and result.details["manual_confirmation_required"]
            )
            assert "private diagnostic" not in str(result)


@pytest.mark.parametrize(
    "status", [WorkflowStepStatus.WAITING, WorkflowStepStatus.FAILED]
)
def test_incomplete_stop_retains_its_receipt_for_retry(status):
    state, _adapter, validator, context = runtime()
    captured = validator.capture(
        context, [("training", "pytorchjob", "job", WORKLOAD, state.job)]
    )
    outcome = WorkflowStepOutcome(status=status, details={"retained": True})
    result = validator.finish(context, captured, outcome)
    assert result.status is status and result.details["retained"] is True
    assert result.details[guard.STOP_RECEIPT_KEY]["contained"] is False


def test_stop_does_not_complete_while_any_original_participant_is_active():
    state, _adapter, validator, context = stopped_runtime()
    saved = receipt(context)
    state.pods = [state.pod("node-a", saved.pods[0].uid)]
    outcome = validator.finish(context, saved, WorkflowStepOutcome.succeeded())
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["reason"] == "STOP_PARTICIPANTS_ACTIVE"


@pytest.mark.parametrize(
    "defect",
    [
        "workload-api",
        "workload-kind",
        "workload-namespace",
        "pod-owner-api",
        "pod-owner-kind",
        "pod-owner-name",
    ],
)
def test_invalid_initial_gvk_or_owner_reference_cannot_authorize_even_stop(defect):
    state, adapter, validator, context = runtime()
    if defect == "workload-api":
        state.job["apiVersion"] = "foreign.example/v1"
    elif defect == "workload-kind":
        state.job["kind"] = "Job"
    elif defect == "workload-namespace":
        state.job["metadata"]["namespace"] = "foreign"
    else:
        key, value = {
            "pod-owner-api": ("apiVersion", "foreign.example/v1"),
            "pod-owner-kind": ("kind", "pytorchjob"),
            "pod-owner-name": ("name", "foreign"),
        }[defect]
        state.pods[0]["metadata"]["ownerReferences"][0][key] = value
    with guard.stop_ownership_scope(validator):
        result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    assert result.details["manual_confirmation_required"] is True
    assert not any("patch" in call[0] or "delete" in call[0] for call in state.calls), (
        "invalid workload identity must not authorize STOP mutations"
    )
    assert state.job["spec"]["runPolicy"]["suspend"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "child-reference-api",
        "child-document-api",
        "child-document-kind",
        "child-namespace",
        "child-name",
        "root-api",
        "root-kind",
        "root-name",
        "root-controller-count",
    ],
)
def test_initial_jobset_chain_requires_exact_child_and_root_gvk_namespace_and_name(
    defect,
):
    state, _adapter, validator, context = runtime()
    root = deepcopy(state.job)
    root.update(apiVersion="jobset.x-k8s.io/v1alpha2", kind="JobSet")
    root["metadata"].update(name="set", uid="set-uid")
    context = replace(
        context,
        step=context.step.model_copy(update={"workload_ids": ["training/jobset/set"]}),
    )
    pod = state.pod("node-a", "owned-pod")
    pod["metadata"]["ownerReferences"] = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": "child",
            "uid": "child-uid",
            "controller": True,
        }
    ]
    child = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "child",
            "namespace": "training",
            "uid": "child-uid",
            "resourceVersion": "1",
            "ownerReferences": [
                {
                    "apiVersion": root["apiVersion"],
                    "kind": "JobSet",
                    "name": "set",
                    "uid": "set-uid",
                    "controller": True,
                }
            ],
        },
    }
    if defect == "child-reference-api":
        pod["metadata"]["ownerReferences"][0]["apiVersion"] = "foreign.example/v1"
    elif defect == "child-document-api":
        child["apiVersion"] = "foreign.example/v1"
    elif defect == "child-document-kind":
        child["kind"] = "Deployment"
    elif defect == "child-namespace":
        child["metadata"]["namespace"] = "foreign"
    elif defect == "child-name":
        child["metadata"]["name"] = "foreign"
    elif defect == "root-controller-count":
        child["metadata"]["ownerReferences"].append(
            {
                "apiVersion": root["apiVersion"],
                "kind": "JobSet",
                "name": "other",
                "uid": "other",
                "controller": True,
            }
        )
    else:
        key, value = {
            "root-api": ("apiVersion", "foreign.example/v1"),
            "root-kind": ("kind", "Job"),
            "root-name": ("name", "foreign"),
        }[defect]
        child["metadata"]["ownerReferences"][0][key] = value
    state.pods = [pod]
    validator.read_workload = lambda *args: child
    with pytest.raises(guard.StopOwnershipError):
        validator.capture(
            context, [("training", "jobset", "set", "training/jobset/set", root)]
        )
    assert not any("patch" in call[0] or "delete" in call[0] for call in state.calls), (
        "an invalid JobSet ownership chain must not mutate workloads"
    )


def test_passive_stop_of_a_failed_pytorchjob_is_contained_despite_frozen_counts():
    # training-operator stops reconciling a Failed PyTorchJob: replicaStatuses
    # keeps the counts of the moment it failed however many Pods are deleted
    # and whatever ``suspend`` says (live, v1-855e096). The counts carry no
    # information; the terminal condition and Pod absence do. COLLECT-021's
    # passive STOP waited on STOP_PARTICIPANTS_ACTIVE until the runner gave up.
    state, adapter, validator, context = runtime()
    frozen = {
        "conditions": [{"type": "Failed", "status": "True"}],
        "replicaStatuses": {"Master": {"failed": 1}, "Worker": {"active": 2}},
    }
    state.job["status"] = deepcopy(frozen)
    reconciled = state.patch_namespaced_custom_object

    def ignored_by_the_operator(*args, **kwargs):
        reconciled(*args, **kwargs)
        state.job["status"] = deepcopy(frozen)

    state.patch_namespaced_custom_object = ignored_by_the_operator
    with guard.stop_ownership_scope(validator):
        outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, (
        f"a terminal PyTorchJob whose Pods are gone is contained: {outcome}"
    )
    saved = guard.StopOwnershipReceipt.model_validate_json(
        json.dumps(outcome.details[guard.STOP_RECEIPT_KEY])
    )
    assert saved.contained and saved.completed_at is not None, (
        "the receipt must record the containment"
    )
    assert state.job["spec"]["runPolicy"]["suspend"] is True, (
        "the suspend patch still goes out so a resume cannot revive the attempt"
    )


def passive_restart_context(*, with_proof=True):
    """The one-step passive recovery workflow behind a contained STOP."""
    state, _adapter, validator, context = stopped_runtime()
    containment = receipt(context)
    restart = WorkflowStepSpec(
        operation=WorkflowOperation.RESTART_WORKLOAD,
        execution_owner="gpu-fault-kubernetes-adapter",
        node_ids=NODES,
        workload_ids=[WORKLOAD],
        parameters={
            "cluster_id": "cluster-local",
            "job_id": "job",
            "source_attempt_id": "attempt",
            "source_gpu_count": 2,
            "restart_budget": 1,
            "requires_incident_state": "RECOVERED",
            "incident_id": "incident-local",
            "incident_node_ids": NODES,
        },
    )
    recovery = WorkflowRequest(
        request_id="workflow-recovery",
        incident_id="incident-recovery",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        execution_epoch=1,
        official_action="RESTART_WORKLOAD",
        official_steps=[restart],
        predecessor_workflow_id=context.workflow.request_id,
    )
    incident = context.incident.model_copy(
        update={
            "incident_id": "incident-recovery",
            "event_id": "event-recovery",
            "event_type": "TRAINING_ATTEMPT_TERMINAL",
            "workflow_request_id": "workflow-recovery",
            "policy_version": "passive-recovery-v1",
        }
    )
    proof = RestartContainmentProof(
        workflow_id=context.workflow.request_id,
        incident_id="incident-local",
        receipt=containment.model_dump(mode="json"),
    )
    authorization = RestartAuthorization(
        cluster_id="cluster-local",
        job_id="job",
        source_attempt_id="attempt",
        source_gpu_count=2,
        restart_budget=1,
        restart_count=1,
        reservation_id="workflow-recovery/0/RESTART_WORKLOAD",
        containment=proof if with_proof else None,
    )
    return (
        state,
        validator,
        WorkflowStepContext(
            workflow=recovery,
            incident=incident,
            step=restart,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=1, restart_authorization=authorization
            ),
            idempotency_key="workflow-recovery/0/RESTART_WORKLOAD",
        ),
    )


def with_proof(context, proof):
    authorization = context.request.restart_authorization.model_copy(
        update={"containment": proof}
    )
    return replace(
        context,
        request=context.request.model_copy(
            update={"restart_authorization": authorization}
        ),
    )


def test_passive_restart_binds_the_predecessor_containment_receipt():
    state, validator, context = passive_restart_context()
    with guard.stop_ownership_scope(validator):
        assert guard.node_submission_ownership_guard(context) is None, (
            "the signed containment receipt stands in for the STOP this "
            "workflow never had"
        )
    state.job["metadata"]["uid"] = "replacement"
    refused(validator, context, "STOP_OWNERSHIP_DRIFT")


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("no-authorization", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("no-proof", "STOP_OWNERSHIP_RECEIPT_MISSING"),
        ("foreign-predecessor", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("no-predecessor", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("foreign-incident", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("no-premise", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("foreign-cluster", "STOP_OWNERSHIP_SCOPE_MISMATCH"),
        ("wrong-attempt", "STOP_OWNERSHIP_DRIFT"),
        ("uncontained", "STOP_PARTICIPANTS_CHANGED"),
        ("foreign-workload", "STOP_PARTICIPANTS_CHANGED"),
        ("malformed", "STOP_OWNERSHIP_UNVERIFIABLE"),
        ("late-sibling", "STOP_PARTICIPANTS_CHANGED"),
    ],
)
def test_passive_restart_containment_proof_defects_are_refused(defect, reason):
    state, validator, context = passive_restart_context()
    proof = context.request.restart_authorization.containment
    if defect == "no-authorization":
        context = replace(
            context,
            request=context.request.model_copy(update={"restart_authorization": None}),
        )
    elif defect == "no-proof":
        context = with_proof(context, None)
    elif defect == "foreign-predecessor":
        context = with_proof(context, proof.model_copy(update={"workflow_id": "other"}))
    elif defect == "no-predecessor":
        context = replace(
            context,
            workflow=context.workflow.model_copy(
                update={"predecessor_workflow_id": None}
            ),
        )
    elif defect == "foreign-incident":
        context = with_proof(context, proof.model_copy(update={"incident_id": "other"}))
    elif defect == "no-premise":
        parameters = {
            key: value
            for key, value in context.step.parameters.items()
            if key != "requires_incident_state"
        }
        context = replace(
            context, step=context.step.model_copy(update={"parameters": parameters})
        )
    elif defect == "foreign-cluster":
        context = with_proof(
            context,
            proof.model_copy(
                update={"receipt": {**proof.receipt, "cluster_id": "other"}}
            ),
        )
    elif defect == "wrong-attempt":
        context = replace(
            context,
            incident=context.incident.model_copy(update={"attempt_id": "other"}),
        )
    elif defect == "uncontained":
        context = with_proof(
            context,
            proof.model_copy(update={"receipt": {**proof.receipt, "contained": False}}),
        )
    elif defect == "foreign-workload":
        context = replace(
            context,
            step=context.step.model_copy(
                update={"workload_ids": ["training/job/foreign"]}
            ),
        )
    elif defect == "malformed":
        context = with_proof(context, proof.model_copy(update={"receipt": {"v": 1}}))
    elif defect == "late-sibling":
        state.pods = [state.pod("node-b", "late")]
    refused(validator, context, reason)


def test_a_workflow_with_its_own_stop_step_ignores_the_authorization_proof():
    _state, _adapter, validator, context = stopped_runtime()
    proof = RestartContainmentProof(
        workflow_id=context.workflow.request_id,
        incident_id="incident-local",
        receipt=receipt(context).model_dump(mode="json"),
    )
    authorization = RestartAuthorization(
        cluster_id="cluster-local",
        job_id="job",
        source_attempt_id="attempt",
        source_gpu_count=2,
        restart_budget=1,
        restart_count=1,
        reservation_id="workflow-local/1/RESET_GPU",
        containment=proof,
    )
    context = replace(
        context,
        request=context.request.model_copy(
            update={"restart_authorization": authorization}
        ),
        workflow=context.workflow.model_copy(update={"step_executions": []}),
    )
    refused(validator, context, "STOP_OWNERSHIP_RECEIPT_MISSING")


@pytest.mark.parametrize(
    "operation",
    [
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
    ],
)
def test_device_plugin_restarts_run_beside_live_training_without_a_stop(operation):
    # A device-plugin restart deletes the plugin Pod so kubelet re-registers the
    # node's devices; the containers already holding them keep them. COLLECT-017
    # segment C restarts the EFA plugin under a live 24-GPU training on purpose
    # and its workflow must not contain STOP_WORKLOADS, so the STOP-ownership
    # guard has no receipt to demand here (2026-09-18: it demanded one anyway
    # and quarantined a healthy node).
    state, _adapter, validator, context = runtime()
    restart = WorkflowStepSpec(
        operation=operation,
        execution_owner="gpu-fault-kubernetes-adapter",
        node_ids=[NODES[0]],
        workload_ids=[WORKLOAD],
        parameters={"expected_count": 16},
    )
    freeze = WorkflowStepSpec(
        operation=WorkflowOperation.FREEZE_EVIDENCE,
        execution_owner="gpu-fault-control-plane",
        node_ids=[NODES[0]],
    )
    workflow = context.workflow.model_copy(
        update={
            "official_action": operation.value,
            "official_steps": [freeze, restart],
            "completed_step_indexes": [0],
        }
    )
    context = replace(
        context,
        workflow=workflow,
        step=restart,
        step_index=1,
        idempotency_key=f"workflow-local/1/{operation.value}",
    )
    assert state.pods and all(
        pod["status"]["phase"] == "Running" for pod in state.pods
    ), "the training Pods still hold their GPUs while the plugin restarts"
    with guard.stop_ownership_scope(validator):
        assert guard.node_submission_ownership_guard(context) is None, (
            "re-registering devices disturbs no device holder; no STOP receipt is due"
        )
    # Behind a STOP the restart still binds to the receipt: a replaced source
    # workload is caught before the adapter acts (the dispatch suite pins the
    # late-sibling case).
    _state, _adapter, stopped_validator, stopped = stopped_runtime()
    plugin_after_stop = replace(
        stopped, step=stopped.step.model_copy(update={"operation": operation})
    )
    with guard.stop_ownership_scope(stopped_validator):
        assert guard.node_submission_ownership_guard(plugin_after_stop) is None, (
            "an intact receipt admits the restart behind its STOP"
        )
    _state.job["metadata"]["uid"] = "replaced"
    refused(stopped_validator, plugin_after_stop, "STOP_OWNERSHIP_DRIFT")
    # The same shape with a real runtime mutation is still refused.
    reset = replace(
        context,
        step=restart.model_copy(update={"operation": WorkflowOperation.RESET_GPU}),
        workflow=workflow.model_copy(
            update={
                "official_steps": [
                    freeze,
                    restart.model_copy(
                        update={"operation": WorkflowOperation.RESET_GPU}
                    ),
                ]
            }
        ),
    )
    refused(validator, reset, "STOP_OWNERSHIP_RECEIPT_MISSING")
