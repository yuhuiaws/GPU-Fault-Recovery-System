"""The restore an operator approves is the restore the Pod writes (batch 155).

``submit-remediation --disposition restore`` judges each node on the admin side.
A node whose only isolation by the incident is its eks-managed quiesce taint
(``quiesce_taint_only``: another incident's cordon keeps it out of scheduling)
is released but not validated, and the plan builds its preview with that node
in ``release_only_node_ids``. The in-Pod ``restore`` mode built the workflow
without the list, so the created restore validated the node anyway: its
VALIDATE_GPU waits on the exporter the very taint evicts and fails at the step
cap before the RESTORE_SCHEDULING that would strip the taint, and on a mixed
restore the validated node's steps named the GPU the release-only node's
inventory places there (batch 149's rule, applied to the preview only). The
payload now carries the plan's release-only nodes and the Pod hands them to
the same builder, so the steps written are the steps planned; an image whose
builder predates the keyword refuses such a restore instead of writing another
one. The restore still never restarts the job a STOP left suspended.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    QUIESCE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.admin import submit_remediation as module
from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration import validated_restore
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY
from tests._builders import (
    build_store,
    contained_stop_receipt,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.admin.test_admin_submit_remediation import _site

RESET = "inc-xid46-reset"
OWNER = "inc-support-after-workflow-c1aef7fb"
WORKFLOW = "wf-reset"
NODE_A = "node-a"
NODE_B = "node-b"
INVENTORY = {NODE_A: ["GPU-a"], NODE_B: ["GPU-b"]}
SUSPENDED = (
    "workload training/pytorchjob/llm stays suspended (stopped by workflow wf-reset)"
)


def _node(name: str, *, owner: str, quiesce: str | None) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "annotations": {ANNOTATION_INCIDENT: owner}},
        "spec": {
            "unschedulable": True,
            "taints": [
                {
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(owner),
                    "effect": "NoSchedule",
                },
                *(
                    [{"key": QUIESCE_TAINT, "value": quiesce, "effect": "NoExecute"}]
                    if quiesce is not None
                    else []
                ),
            ],
        },
    }


def _isolated(name: str) -> dict[str, Any]:
    """The incident's own cordon, quarantine taint and quiesce taint."""

    return _node(name, owner=RESET, quiesce=quarantine_taint_value(RESET))


def _quiesce_only(name: str) -> dict[str, Any]:
    """Another incident's cordon over this incident's orphaned quiesce taint."""

    return _node(name, owner=OWNER, quiesce=quarantine_taint_value(RESET))


def _store(nodes: list[str]):
    store = build_store()
    incident = fault_incident(
        RESET,
        "event-xid46",
        cluster_id="gpu-a",
        node_ids=nodes,
        gpu_uuids=[gpu for node in nodes for gpu in INVENTORY[node]],
        state=IncidentState.QUARANTINED,
        workflow_request_id=WORKFLOW,
        fencing_token=1,
    )
    receipt = contained_stop_receipt(
        workflow_id=WORKFLOW,
        workloads=[
            {
                "workload_id": "training/pytorchjob/llm",
                "namespace": "training",
                "kind": "pytorchjob",
                "name": "llm",
                "uid": "llm-uid",
                "attempt_id": "attempt-1",
                "owners": [],
            }
        ],
    )
    failed = workflow_request(
        WORKFLOW,
        RESET,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=nodes),
            workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=nodes),
            workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=nodes),
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=nodes),
            workflow_step(WorkflowOperation.RESTART_WORKLOAD, node_ids=nodes),
        ],
        step_executions=[
            workflow_step_execution(
                1,
                WorkflowOperation.STOP_WORKLOADS,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={
                    STOP_OWNERSHIP_RECEIPT_KEY: receipt,
                    "workloads_suspended": True,
                },
            )
        ],
    )
    store.save_incident_and_workflow(incident, failed)
    return store


class Pod:
    """``kubectl exec`` of the remediation script on ``store``, in-process."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        store,
        inventory: dict[str, list[str] | None],
    ) -> None:
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        self.inventory = inventory
        self.payloads: list[dict[str, Any]] = []
        context = ApplicationContext(store=store)
        monkeypatch.setattr(context.dispatcher, "wake", lambda: None)
        monkeypatch.setattr(
            ApplicationContext, "from_environment", classmethod(lambda cls: context)
        )
        monkeypatch.setattr(store, "get_gpu_inventory_snapshot", self._snapshot)

    def _snapshot(self, cluster_id: str, node_id: str):
        gpus = self.inventory.get(node_id)
        if gpus is None:
            return None
        return SimpleNamespace(devices=[SimpleNamespace(gpu_uuid=gpu) for gpu in gpus])

    def __call__(
        self, site: Any, payload: dict[str, Any], *, script: str
    ) -> dict[str, Any]:
        self.payloads.append(json.loads(json.dumps(payload)))
        self.monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        self.capsys.readouterr()
        exec(compile(script, "<submit-remediation>", "exec"), {})
        return json.loads(self.capsys.readouterr().out.strip().splitlines()[-1])


def _submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pod: Pod,
    live_nodes: dict[str, dict[str, Any]],
    *,
    plan_only: bool = False,
) -> module.SubmissionResult:
    clock = [datetime.now(timezone.utc)]

    def now() -> datetime:
        clock[0] += timedelta(seconds=1)
        return clock[0]

    monkeypatch.setattr(module, "run_control_plane_script", pod)
    monkeypatch.setattr(module, "cluster_nodes", lambda site, cluster_id: live_nodes)
    return module.submit_remediation(
        module.SubmitRemediationRequest(
            site=_site(tmp_path),
            incident_id=RESET,
            disposition="restore",
            reference="CHG-155",
            plan_only=plan_only,
        ),
        now=now,
    )


def _steps(steps: Any) -> list[tuple[str, str, list[str], list[str]]]:
    return [
        (
            str(getattr(step["operation"], "value", step["operation"])),
            str(step["execution_owner"]),
            list(step["node_ids"]),
            list(step["gpu_uuids"]),
        )
        for step in steps
    ]


VALIDATION = "gpu-fault-validation-adapter"
KUBERNETES = "gpu-fault-kubernetes-adapter"
SHAPES = {
    "quiesce-only": (
        {NODE_A: _quiesce_only(NODE_A)},
        [("RESTORE_SCHEDULING", KUBERNETES, [NODE_A], [])],
    ),
    "mixed": (
        {NODE_A: _isolated(NODE_A), NODE_B: _quiesce_only(NODE_B)},
        [
            ("VALIDATE_GPU", VALIDATION, [NODE_A], ["GPU-a"]),
            ("VALIDATE_HOST", VALIDATION, [NODE_A], ["GPU-a"]),
            ("VALIDATE_FABRIC", VALIDATION, [NODE_A], ["GPU-a"]),
            ("RESTORE_SCHEDULING", KUBERNETES, [NODE_A, NODE_B], ["GPU-a"]),
        ],
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_created_restore_is_the_one_the_plan_showed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shape: str,
) -> None:
    live_nodes, expected = SHAPES[shape]
    store = _store(sorted(live_nodes))
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))

    preview = _submit(tmp_path, monkeypatch, pod, live_nodes, plan_only=True)
    result = _submit(tmp_path, monkeypatch, pod, live_nodes)

    assert _steps(preview.plan.steps) == expected, "the plan --plan printed"
    assert result.submission is not None, result.message
    created = store.get_workflow(result.submission["workflow"]["request_id"])
    assert _steps(step.model_dump() for step in created.official_steps) == expected, (
        "the Pod wrote another workflow than the plan the operator approved"
    )
    assert pod.payloads[-1]["release_only_node_ids"] == [
        node
        for node, item in sorted(live_nodes.items())
        if item["metadata"]["annotations"][ANNOTATION_INCIDENT] == OWNER
    ]
    assert (
        result.plan.as_dict()["release_only_node_ids"]
        == (pod.payloads[-1]["release_only_node_ids"])
    )
    assert "differ" not in result.message, result.message
    # The restore never restarts the job the failed chain left suspended: it is
    # named on the plan and on the result, and no step restarts it.
    assert WorkflowOperation.RESTART_WORKLOAD not in {
        step.operation for step in created.official_steps
    }
    assert any(SUSPENDED in line for line in preview.plan.warnings), (
        preview.plan.warnings
    )
    assert SUSPENDED in result.message, result.message
    incident = store.get_incident(RESET)
    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.workflow_request_id == created.request_id


def test_a_fully_isolated_restore_still_validates_every_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No release-only node: the payload's list is empty and nothing changes."""

    live_nodes = {NODE_A: _isolated(NODE_A), NODE_B: _isolated(NODE_B)}
    store = _store([NODE_A, NODE_B])
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))

    result = _submit(tmp_path, monkeypatch, pod, live_nodes)

    assert pod.payloads[-1]["release_only_node_ids"] == []
    assert result.submission is not None, result.message
    created = store.get_workflow(result.submission["workflow"]["request_id"])
    assert _steps(step.model_dump() for step in created.official_steps) == _steps(
        result.plan.steps
    )
    assert [
        (step.operation.value, step.node_ids, step.gpu_uuids)
        for step in created.official_steps
        if step.operation is WorkflowOperation.VALIDATE_GPU
    ] == [("VALIDATE_GPU", [NODE_A], ["GPU-a"]), ("VALIDATE_GPU", [NODE_B], ["GPU-b"])]


def _older_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployed image whose builder predates ``release_only_node_ids``."""

    real = validated_restore.build_validated_restore_workflow

    def build_validated_restore_workflow(
        incident,
        *,
        operator,
        reference,
        now,
        node_ids=None,
        runtime_profile_version=None,
        reason=None,
        node_gpu_uuids=None,
    ):
        return real(
            incident,
            operator=operator,
            reference=reference,
            now=now,
            node_ids=node_ids,
            runtime_profile_version=runtime_profile_version,
            reason=reason,
            node_gpu_uuids=node_gpu_uuids,
        )

    monkeypatch.setattr(
        validated_restore,
        "build_validated_restore_workflow",
        build_validated_restore_workflow,
    )


def test_an_older_image_refuses_a_release_only_restore_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Such an image would validate the quiesce-only node first (and its
    RESTORE_SCHEDULING cannot strip a quiesce taint under another incident's
    cordon): it refuses rather than write a workflow the plan never showed."""

    live_nodes = {NODE_A: _isolated(NODE_A), NODE_B: _quiesce_only(NODE_B)}
    store = _store([NODE_A, NODE_B])
    before = store.get_incident(RESET)
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))
    _older_builder(monkeypatch)

    with pytest.raises(SystemExit, match="cannot release node-b unvalidated"):
        _submit(tmp_path, monkeypatch, pod, live_nodes)

    assert store.get_incident(RESET) == before, "a refusal writes nothing"
    assert [item.request_id for item in store.list_workflows(limit=100)] == [WORKFLOW]


def test_an_older_image_still_writes_a_restore_without_release_only_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live_nodes = {NODE_A: _isolated(NODE_A)}
    store = _store([NODE_A])
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))
    _older_builder(monkeypatch)

    result = _submit(tmp_path, monkeypatch, pod, live_nodes)

    assert result.submission is not None, result.message
    created = store.get_workflow(result.submission["workflow"]["request_id"])
    assert _steps(step.model_dump() for step in created.official_steps) == _steps(
        result.plan.steps
    )


def test_the_pod_refuses_release_only_nodes_it_does_not_restore(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _store([NODE_A, NODE_B])
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))
    payload = {
        "mode": "restore",
        "incident_id": RESET,
        "expected": {
            "fencing_token": 1,
            "workflow_request_id": WORKFLOW,
            "state": "QUARANTINED",
            "node_ids": [NODE_A, NODE_B],
        },
        "node_ids": [NODE_A],
        "release_only_node_ids": [NODE_B],
        "operator": "ops@example",
        "reference": "CHG-155",
        "runtime_profile_version": "eks-v1",
    }

    with pytest.raises(SystemExit, match="release-only node node-b is not restored"):
        pod(None, payload, script=module.REMEDIATION_SCRIPT)

    assert store.get_incident(RESET).workflow_request_id == WORKFLOW


def test_the_result_names_steps_that_differ_from_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Pod re-reads the inventories when it writes: one that changed after
    the inspection (node-b's became unreadable) builds other steps than the
    plan showed, and the result says so instead of repeating the plan."""

    live_nodes = {NODE_A: _isolated(NODE_A), NODE_B: _isolated(NODE_B)}
    store = _store([NODE_A, NODE_B])
    pod = Pod(monkeypatch, capsys, store, dict(INVENTORY))
    inspect = module.inspect_incident

    def inspect_then_lose_inventory(site, incident_id, *, disposition):
        inspected = inspect(site, incident_id, disposition=disposition)
        pod.inventory[NODE_B] = None
        return inspected

    monkeypatch.setattr(module, "inspect_incident", inspect_then_lose_inventory)

    result = _submit(tmp_path, monkeypatch, pod, live_nodes)

    assert result.submission is not None, result.message
    assert "the created workflow's steps differ from the plan" in result.message
    assert "VALIDATE_GPU on node-a, node-b" in result.message, result.message
