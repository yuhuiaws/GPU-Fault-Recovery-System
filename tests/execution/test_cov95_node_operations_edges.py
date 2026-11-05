"""``adapters/kubernetes/node_operations``: what the node steps do when the node
does not cooperate.

A device plugin restart under a ``gpu-fault.io/quiesce`` NoExecute taint, a
mechanical inspection nobody can be mailed about, an isolation whose node is
deleted between the pre-check and the patch (or whose re-read fails after it),
an API error that is not the node's absence, an annotation that is not a
fencing token, and the receipts and ownership reports the steps read. The
Kubernetes API is the fake core of ``tests/execution/_support.py`` (and the
device plugin harness of ``tests/execution/_cov95_runtime_plugins.py``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    QUARANTINE_TAINT,
    QUIESCE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.adapters.kubernetes.node_operations import (
    IsolationOwnership,
    isolation_ownership,
    isolation_refusal,
    prior_node_receipt,
)
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step_execution
from tests.execution._cov95_runtime_plugins import PluginHarness
from tests.execution._support import FakeCoreApi, UnusedApi, workflow_state

OWNER = "gpu-fault-kubernetes-adapter"
NODE_OPERATIONS_LOGGER = "gpu_fault.adapters.kubernetes.node_operations"
OP = WorkflowOperation


class ApiStatusError(RuntimeError):
    """What the Kubernetes client raises: an error carrying an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FlakyCore(FakeCoreApi):
    """The fake core whose ``read_number``-th read or every patch fails."""

    def __init__(
        self,
        *,
        read_error: Exception | None = None,
        read_number: int = 0,
        patch_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.read_error = read_error
        self.read_number = read_number
        self.patch_error = patch_error
        self.reads = 0
        self.patches: list[dict[str, Any]] = []

    def read_node(self, node_id: str) -> Any:
        self.reads += 1
        if self.read_error is not None and self.reads == self.read_number:
            raise self.read_error
        return super().read_node(node_id)

    def patch_node(self, node_id: str, body: dict[str, Any]) -> None:
        if self.patch_error is not None:
            raise self.patch_error
        self.patches.append(body)
        super().patch_node(node_id, body)


def _adapter(core: Any) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )


def _context(
    operation: WorkflowOperation, *, executions: tuple[Any, ...] = ()
) -> WorkflowStepContext:
    incident, workflow = workflow_state(InMemoryStore(), [operation])
    step = workflow.official_steps[0].model_copy(
        update={"execution_owner": OWNER, "node_ids": ["node-a"]}
    )
    workflow = workflow.model_copy(
        update={"official_steps": [step], "step_executions": list(executions)}
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=f"workflow-active/0/{operation.value}",
    )


def _isolated(core: FakeCoreApi, incident_id: str = "incident-active") -> None:
    core.node["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: incident_id,
        ANNOTATION_FENCING: "3",
        ANNOTATION_PREVIOUS_UNSCHEDULABLE: "false",
    }
    core.node["spec"]["unschedulable"] = True
    core.node["spec"]["taints"].append(
        {
            "key": QUARANTINE_TAINT,
            "value": quarantine_taint_value(incident_id),
            "effect": "NoSchedule",
        }
    )


# ------------------------------------------------------- device plugin restart


def test_a_device_plugin_restart_waits_out_a_live_quiesce_taint() -> None:
    harness = PluginHarness()
    held = quarantine_taint_value("incident-other")
    harness.node["spec"]["taints"] = [
        {"key": QUIESCE_TAINT, "value": held, "effect": "NoExecute"}
    ]

    result = harness.execute()

    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details["node_results"]["node-a"] == {
        "quiesce_taint_present": True,
        "quiesce_taint_values": [held],
        "waiting_for_quiesce_taint_removal": True,
    }
    assert (harness.core.patches, harness.core.deleted, harness.core.queries) == (
        [],
        [],
        [],
    ), "kubelet would evict the replacement: nothing is touched under the taint"


def test_an_orphaned_quiesce_taint_is_stripped_before_the_restart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = PluginHarness()
    own = quarantine_taint_value(harness.context.incident.incident_id)
    provider = {
        "key": "sagemaker.amazonaws.com/node-health-status",
        "value": "Schedulable",
        "effect": "NoSchedule",
    }
    harness.node["spec"]["taints"] = [
        provider,
        {"key": QUIESCE_TAINT, "value": own, "effect": "NoExecute"},
    ]
    caplog.set_level(logging.WARNING, logger=NODE_OPERATIONS_LOGGER)

    result = harness.execute()

    assert harness.node["spec"]["taints"] == [provider], "only the orphan goes"
    _node, strip = harness.core.patches[0]
    assert (strip["metadata"]["annotations"], strip["spec"]) == (
        {},
        {"taints": [provider]},
    ), "the strip is a taint-only patch"
    assert result.status is WorkflowStepStatus.WAITING, result
    assert harness.core.deleted == [("kube-system", "plugin-old")], (
        "the restart goes on once the taint no workflow holds is gone"
    )
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage()
        == (
            f"stripped orphaned quiesce taint(s) ['{own}'] from node-a before the "
            "device plugin restart of workflow-active"
        )
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------- mechanical check


def test_a_mechanical_inspection_without_a_mailbox_still_names_the_command() -> None:
    adapter = _adapter(FakeCoreApi())
    assert adapter.notification_sink is None, "no store, no sink"

    result = adapter.execute(_context(OP.CHECK_MECHANICALS))

    assert result.status is WorkflowStepStatus.WAITING, result
    expected = "incident-active:3"
    assert result.details["pending_nodes"] == ["node-a"]
    assert result.details["notification_id"] is None, "nothing could be queued"
    assert result.details["required_annotation_value"] == expected
    assert result.details["acknowledgement_commands"] == [
        f"kubectl annotate node node-a {ANNOTATION_MECHANICAL_INSPECTION_COMPLETE}"
        f"='{expected}' --overwrite"
    ]


# ------------------------------------------------------------------ isolation


def test_an_isolation_whose_re_read_fails_records_the_error_as_its_after() -> None:
    core = FlakyCore(read_error=ApiStatusError(500, "apiserver timeout"), read_number=3)

    result = _adapter(core).execute(_context(OP.MARK_UNSCHEDULABLE))

    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    baseline = result.details["node_baselines"]["node-a"]
    assert baseline["after"] == {"error": "ApiStatusError: apiserver timeout"}
    assert baseline["before"]["unschedulable"] is False, baseline
    assert core.node["metadata"]["annotations"][ANNOTATION_INCIDENT] == (
        "incident-active"
    ), "the patch itself landed"


def test_an_isolation_whose_node_is_deleted_before_the_patch_is_absent() -> None:
    core = FlakyCore(patch_error=ApiStatusError(404, "nodes node-a not found"))

    result = _adapter(core).execute(_context(OP.MARK_UNSCHEDULABLE))

    assert result.status is WorkflowStepStatus.FAILED, result
    assert result.error == "node node-a is absent; isolation cannot be observed"
    assert result.details == {
        "safety_rejection": True,
        "node_id": "node-a",
        "absent": True,
        "isolated_nodes": [],
    }


@pytest.mark.parametrize(
    "operation",
    [OP.MARK_UNSCHEDULABLE, OP.RESTORE_SCHEDULING],
    ids=["isolate", "restore"],
)
def test_an_api_error_that_is_not_an_absence_propagates(
    operation: WorkflowOperation,
) -> None:
    core = FlakyCore(patch_error=ApiStatusError(500, "etcdserver: leader changed"))
    if operation is OP.RESTORE_SCHEDULING:
        _isolated(core)

    with pytest.raises(ApiStatusError, match="leader changed"):
        _adapter(core).execute(_context(operation))

    assert core.patches == [], "nothing was written"


def test_a_fencing_annotation_that_is_not_a_number_refuses_the_isolation() -> None:
    core = FlakyCore()
    core.node["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: "incident-active",
        ANNOTATION_FENCING: "three",
    }

    with pytest.raises(ValueError, match="node has invalid gpu-fault fencing token"):
        _adapter(core).execute(_context(OP.MARK_UNSCHEDULABLE))

    assert (core.reads, core.patches) == (1, []), "refused at the pre-check"


def test_the_pre_check_lets_an_unexpected_read_error_through() -> None:
    core = FlakyCore(read_error=ApiStatusError(403, "forbidden"), read_number=1)

    with pytest.raises(ApiStatusError, match="forbidden"):
        isolation_refusal(core, lambda node, context: {}, _context(OP.QUARANTINE))


# ------------------------------------------------------- receipts, ownership


def test_the_prior_receipt_is_read_from_this_steps_own_record_only() -> None:
    other_step = workflow_step_execution(
        1,
        OP.RESTART_GPU_DEVICE_PLUGIN,
        phase="official",
        details={"node_results": {"node-a": {"deleted_pod_uid": "uid-other"}}},
    )
    own = workflow_step_execution(
        0,
        OP.RESTART_GPU_DEVICE_PLUGIN,
        WorkflowStepStatus.WAITING,
        phase="official",
        details={"node_results": {"node-a": {"deleted_pod_uid": "uid-own"}}},
    )
    context = _context(OP.RESTART_GPU_DEVICE_PLUGIN, executions=(own, other_step))

    assert prior_node_receipt(context, "node-a") == {"deleted_pod_uid": "uid-own"}
    assert prior_node_receipt(context, "node-b") == {}, "no receipt for the node"


def test_an_ownership_report_without_a_successor_asks_by_incident_alone() -> None:
    class Provider:
        def __init__(self) -> None:
            self.asked: list[tuple[str, dict[str, Any]]] = []

        def incident_ownership(self, incident_id: str, **kwargs: Any) -> Any:
            self.asked.append((incident_id, kwargs))
            return SimpleNamespace(
                known=True,
                terminal=True,
                quarantine_hold=True,
                terminal_quarantine=False,
                workflow_request_id="workflow-owner",
                escalation_lineage=None,
            )

    provider = Provider()

    verdict = isolation_ownership(None, provider, "incident-owner")

    assert provider.asked == [("incident-owner", {})], "no successor lineage asked"
    assert verdict == IsolationOwnership(True, True, False, "workflow-owner", None)
