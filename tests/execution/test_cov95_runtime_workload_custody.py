from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_EXECUTION_EPOCH,
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_STEP_INDEX,
    ANNOTATION_WORKFLOW,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import fault_incident, workflow_request
from tests.execution._cov95_runtime_restart import RestartHarness


def stopping_workload():
    h = RestartHarness("pytorchjob", terminal=False)
    step = h.context.step.model_copy(
        update={"operation": WorkflowOperation.STOP_WORKLOADS}
    )
    h.context = replace(
        h.context,
        step=step,
        workflow=h.context.workflow.model_copy(update={"official_steps": [step]}),
        idempotency_key="unit-workload-stop",
    )
    h.adapter.core = SimpleNamespace(
        list_namespaced_pod=lambda **_kwargs: SimpleNamespace(items=[])
    )
    return h


@pytest.mark.parametrize("checkpoint", [None, "s3://unit/checkpoint-manifest"])
def test_checkpoint_requires_explicit_evidence_and_never_reads_or_mutates_kubernetes(
    checkpoint,
):
    h = stopping_workload()
    step = h.context.step.model_copy(
        update={
            "operation": WorkflowOperation.CHECKPOINT_WORKLOADS,
            "parameters": {"checkpoint_manifest_ref": checkpoint},
        }
    )
    h.context = replace(h.context, step=step)
    result = h.execute()
    assert result.status is (
        WorkflowStepStatus.SUCCEEDED if checkpoint else WorkflowStepStatus.FAILED
    ), result
    if checkpoint:
        assert result.details == {"checkpoint_manifest_ref": checkpoint}, result
    else:
        assert "checkpoint manifest evidence is missing" in (result.error or ""), result
    assert h.api.reads == [] and h.api.patches == [] and h.api.created == {}, (
        "a checkpoint receipt must not execute a workload operation"
    )


def test_empty_workload_scope_is_refused_before_api_reads():
    h = stopping_workload()
    h.context = replace(
        h.context, step=h.context.step.model_copy(update={"workload_ids": []})
    )
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    assert result.error == "workload operation requires workload_ids", result
    assert h.api.reads == [] and h.api.patches == [], (
        "an empty scope must never fall back to a namespace-wide mutation"
    )


@pytest.mark.parametrize(
    "prior",
    [
        "local-missing",
        "local-unlinked",
        "local-terminal",
        "remote-open",
        "remote-terminal",
        "remote-unavailable",
        "no-owner-reader",
    ],
)
def test_workload_takeover_requires_positive_terminal_evidence_from_the_owner(prior):
    h = stopping_workload()
    h.api.source["metadata"]["annotations"][ANNOTATION_INCIDENT] = "prior-incident"
    reads = []
    if prior.startswith("remote") or prior == "no-owner-reader":
        h.adapter.store = None

        def owner_is_terminal(incident_id):
            reads.append(incident_id)
            if prior == "remote-unavailable":
                raise OSError("ownership read unavailable")
            return prior == "remote-terminal"

        h.adapter.ownership_provider = (
            None
            if prior == "no-owner-reader"
            else SimpleNamespace(incident_workflow_is_terminal=owner_is_terminal)
        )
    elif prior != "local-missing":
        incident = fault_incident(
            "prior-incident",
            "prior-event",
            workflow_request_id="prior-workflow" if prior == "local-terminal" else None,
        )
        h.store.save_incident(incident)
        if prior == "local-terminal":
            h.store.save_workflow(
                workflow_request(
                    "prior-workflow", incident.incident_id, WorkflowStatus.SUCCEEDED
                )
            )
    allowed = prior in {"local-terminal", "remote-terminal"}
    if allowed:
        result = h.execute()
        assert result.status is WorkflowStepStatus.SUCCEEDED, result
        assert len(h.api.patches) == 1, h.api.patches
        assert h.api.patches[0]["metadata"]["annotations"][ANNOTATION_INCIDENT] == (
            h.context.incident.incident_id
        ), h.api.patches[0]
    else:
        with pytest.raises(ValueError, match="controlled by another incident"):
            h.execute()
        assert h.api.patches == [] and h.api.created == {}, (
            "unknown or live custody must block every workload write",
            prior,
        )
    assert reads == (["prior-incident"] if prior.startswith("remote") else []), reads


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("invalid-fence", "invalid gpu-fault fencing"),
        ("newer-fence", "newer workflow generation"),
        ("newer-epoch", "newer execution epoch"),
        ("later-step", "advanced past this step"),
    ],
)
def test_workload_mutation_rejects_invalid_or_newer_durable_fences(field, reason):
    h = stopping_workload()
    annotations = h.api.source["metadata"]["annotations"]
    annotations.update(
        {
            ANNOTATION_INCIDENT: h.context.incident.incident_id,
            ANNOTATION_FENCING: str(h.context.workflow.fencing_token),
            ANNOTATION_EXECUTION_EPOCH: str(h.context.workflow.execution_epoch),
            ANNOTATION_WORKFLOW: h.context.workflow.request_id,
            ANNOTATION_STEP_INDEX: str(h.context.step_index),
        }
    )
    if field == "invalid-fence":
        annotations[ANNOTATION_FENCING] = "unreadable"
    elif field == "newer-fence":
        annotations[ANNOTATION_FENCING] = str(h.context.workflow.fencing_token + 1)
    elif field == "newer-epoch":
        annotations[ANNOTATION_EXECUTION_EPOCH] = str(
            h.context.workflow.execution_epoch + 1
        )
    else:
        annotations[ANNOTATION_STEP_INDEX] = str(h.context.step_index + 1)
    with pytest.raises(ValueError, match=reason):
        h.execute()
    assert h.api.patches == [] and h.api.created == {}, (
        "fencing rejection must precede the workload mutation",
        annotations,
    )


def test_completed_workload_mutation_replay_does_not_patch_again():
    h = stopping_workload()
    first = h.execute()
    assert first.status is WorkflowStepStatus.SUCCEEDED, first
    [patch] = h.api.patches
    h.api.source["metadata"]["annotations"].update(patch["metadata"]["annotations"])
    assert patch["metadata"]["annotations"][ANNOTATION_OPERATION] == (
        h.context.idempotency_key
    ), patch
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    assert h.api.patches == [patch], "an acknowledged operation must not write twice"
