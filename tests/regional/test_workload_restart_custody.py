from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.managed_workload_fixture import WorkloadOwnershipError
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._workload_restart_support import (
    apply_restart_metadata,
    restart_state,
)
from tests.regional.test_managed_workload_fixture import harness


def pod(parent: dict[str, Any], *, uid: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "training",
            "uid": uid,
            "namespace": parent["metadata"]["namespace"],
            "labels": deepcopy(parent["metadata"]["labels"]),
            "ownerReferences": [
                {
                    "apiVersion": parent["apiVersion"],
                    "kind": parent["kind"],
                    "name": parent["metadata"]["name"],
                    "uid": parent["metadata"]["uid"],
                    "controller": True,
                }
            ],
        },
        "spec": {"nodeName": "node-a", "containers": [{"name": "trainer"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"name": "trainer", "ready": True}],
        },
    }


@pytest.mark.parametrize("retry_parent", [False, True])
def test_receipted_restart_adopts_new_pods_and_cleans_after_orphaning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retry_parent: bool
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    original = api.objects[(fixture.resource, fixture.name)]
    api.add(pod(original, uid="old-pod"))
    assert fixture.pods()[0]["uid"] == "old-pod"
    state = restart_state(
        fixture, retry_parent_name=fixture.name + "-retry" if retry_parent else None
    )
    target = deepcopy(original)
    if retry_parent:
        target["metadata"].update(name=fixture.name + "-retry", uid="retry-parent")
    apply_restart_metadata(target, state)
    api.add(target)
    api.add(pod(target, uid="new-pod"))
    fixture.authorize_restart(state)
    fixture.authorize_restart(deepcopy(state))
    monkeypatch.setattr(
        fixture,
        "heartbeat_logs",
        lambda pods: {item["name"]: "HEARTBEAT all_reduce=1.0" for item in pods},
    )
    result = fixture.wait_restarted({"old-pod"}, timeout_seconds=5)
    assert [item["uid"] for item in result["pods"]] == ["new-pod"]
    assert result["workload"]["name"] == target["metadata"]["name"]
    assert result["workload"]["uid"] == target["metadata"]["uid"]
    assert "old-pod" in fixture.retired_pod_uids
    fixture.delete()
    assert api.objects == {}
    assert api.deletes[-1][2]["preconditions"]["uid"] == "new-pod"
    assert all(
        options["propagationPolicy"] == "Orphan"
        for kind, _name, options in api.deletes
        if kind != "pod"
    ), (
        'test_receipted_restart_adopts_new_pods_and_cleans_after_orphaning: expected all( options["propagationPolicy"] == "Orphan" for kind, _name...'
    )


def test_a_later_product_stop_on_the_adopted_controller_does_not_break_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The restart binding is proven once, when the replacement is adopted.

    A later product mutation of the same controller (COLLECT-016 segment B:
    a second RESTART_APP whose STOP_WORKLOADS re-annotates the PyTorchJob with
    its own workflow/incident/operation) is legitimate; cleanup must identify
    the controller by its already-adopted UID, not re-demand segment A's
    annotations (live 2026-09-17: every A/B run failed at cleanup this way).
    """
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    original = api.objects[(fixture.resource, fixture.name)]
    api.add(pod(original, uid="old-pod"))
    fixture.pods()
    state = restart_state(fixture)
    apply_restart_metadata(original, state)
    api.add(pod(original, uid="new-pod"))
    fixture.authorize_restart(state)
    monkeypatch.setattr(
        fixture,
        "heartbeat_logs",
        lambda pods: {item["name"]: "HEARTBEAT all_reduce=1.0" for item in pods},
    )
    fixture.wait_restarted({"old-pod"}, timeout_seconds=5)
    # Segment B's STOP_WORKLOADS re-annotates the same controller.
    original["metadata"]["annotations"].update(
        {
            "gpu-fault.io/workflow-id": "workflow-b",
            "gpu-fault.io/incident-id": "incident-b",
            "gpu-fault.io/operation-id": "workflow-b/1/STOP_WORKLOADS",
            "gpu-fault.io/workflow-step-index": "1",
            "gpu-fault.io/termination-initiator-incident-id": "incident-b",
        }
    )
    fixture.delete()
    assert api.objects == {}, "cleanup must remove the adopted controller and Pods"
    # A controller never adopted under this binding is still held to it.
    (tmp_path / "second").mkdir()
    fixture2, api2, rendered2 = harness(tmp_path / "second", monkeypatch)
    fixture2.submit_rendered(rendered2)
    parent2 = api2.objects[(fixture2.resource, fixture2.name)]
    api2.add(pod(parent2, uid="old-pod"))
    fixture2.pods()
    state2 = restart_state(fixture2)
    apply_restart_metadata(parent2, state2)
    parent2["metadata"]["annotations"]["gpu-fault.io/operation-id"] = (
        "workflow-b/1/STOP"
    )
    api2.add(pod(parent2, uid="new-pod"))
    fixture2.authorize_restart(state2)
    with pytest.raises(WorkloadOwnershipError, match="does not bind"):
        fixture2.wait_restarted({"old-pod"}, timeout_seconds=1, poll_seconds=1)


@pytest.mark.parametrize(
    "defect",
    [
        "workflow-failed",
        "foreign-incident",
        "missing-execution",
        "missing-command",
        "failed-command",
        "wrong-command",
        "wrong-scope",
        "stale-fence",
        "future-plan",
        "wrong-target",
        "missing-authorization",
        "wrong-reservation",
        "boolean-budget",
        "contradictory-result",
        "foreign-controller",
        "multiple-controllers",
        "wrong-source-uid",
        "wrong-source-attempt",
        "missing-parent-uid",
    ],
)
def test_restart_authority_rejects_incomplete_or_foreign_product_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    state = restart_state(fixture)
    workflow = state["workflow"]
    execution = workflow["step_executions"][0]
    command = state["commands"][0]
    if defect == "workflow-failed":
        workflow["status"] = "FAILED"
    elif defect == "foreign-incident":
        state["incident"]["job_id"] = "another-job"
    elif defect == "missing-execution":
        workflow["step_executions"] = []
    elif defect == "missing-command":
        state["commands"] = []
    elif defect == "failed-command":
        command["status"] = "FAILED"
    elif defect == "wrong-command":
        command["workflow_request_id"] = "another-workflow"
    elif defect == "wrong-scope":
        command["step"]["parameters"]["source_attempt_id"] = "another-attempt"
    elif defect == "stale-fence":
        command["fencing_token"] = 3
    elif defect == "future-plan":
        execution["step_index"] = 1
    elif defect == "wrong-target":
        command["result_details"]["restart_attempt_id"] = "another-target"
    elif defect == "missing-authorization":
        command.pop("restart_authorization")
    elif defect == "wrong-reservation":
        command["restart_authorization"]["reservation_id"] = "another-reservation"
    elif defect == "boolean-budget":
        command["restart_authorization"]["restart_budget"] = True
    elif defect == "contradictory-result":
        command["result_details"]["suspended"] = True
    elif defect in {"foreign-controller", "multiple-controllers"}:
        targets = (
            ["another-namespace/job/foreign"]
            if defect == "foreign-controller"
            else ["gpu-fault-system/job/retry-a", "gpu-fault-system/job/retry-b"]
        )
        execution["details"]["restarted_workload_ids"] = targets
        command["result_details"]["restarted_workload_ids"] = targets
    elif defect in {"wrong-source-uid", "wrong-source-attempt"}:
        source = workflow["step_executions"][1]["details"]["stop_ownership_receipt_v1"][
            "workloads"
        ][0]
        source["uid" if defect == "wrong-source-uid" else "attempt_id"] = "foreign"
    else:
        fixture.owned_uids.clear()
    before = deepcopy(fixture.owned_uids)
    with pytest.raises(WorkloadOwnershipError):
        fixture.authorize_restart(state)
    assert fixture.restart_custody is None
    assert fixture.owned_uids == before
    assert api.deletes == []


@pytest.mark.parametrize(
    "defect",
    [
        "unapproved",
        "foreign-owner",
        "foreign-attempt",
        "foreign-parent",
        "parent-replaced",
        "wrong-operation",
        "wrong-epoch",
        "missing-controller",
        "controller-read-failed",
        "multiple-owners",
        "reappeared",
    ],
)
def test_replacement_checks_real_parent_chain_and_fails_identity_drift_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    parent = api.objects[(fixture.resource, fixture.name)]
    api.add(pod(parent, uid="old-pod"))
    fixture.pods()
    state = restart_state(fixture)
    apply_restart_metadata(parent, state)
    replacement = api.add(pod(parent, uid="new-pod"))
    if defect != "unapproved":
        fixture.authorize_restart(state)
    if defect == "foreign-owner":
        replacement["metadata"]["labels"]["gpu-fault.io/acceptance-owner"] = "foreign"
    elif defect == "foreign-attempt":
        replacement["metadata"]["labels"]["gpu-fault.io/attempt-id"] = "foreign"
    elif defect == "foreign-parent":
        replacement["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif defect == "parent-replaced":
        parent["metadata"]["uid"] = "foreign"
        replacement["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif defect == "wrong-operation":
        parent["metadata"]["annotations"]["gpu-fault.io/operation-id"] = "foreign"
    elif defect == "wrong-epoch":
        parent["metadata"]["annotations"]["gpu-fault.io/execution-epoch"] = "1"
    elif defect == "missing-controller":
        replacement["metadata"]["ownerReferences"] = []
    elif defect == "controller-read-failed":
        api.objects.pop((fixture.resource, fixture.name))
    elif defect == "multiple-owners":
        replacement["metadata"]["ownerReferences"] *= 2
    elif defect == "reappeared":
        fixture.pods()
        replacement["metadata"]["uid"] = "old-pod"
    before = deepcopy(fixture.owned_uids)
    with pytest.raises(WorkloadOwnershipError):
        fixture.wait_restarted({"old-pod"}, timeout_seconds=1, poll_seconds=1)
    assert fixture.owned_uids == before
    assert api.deletes == []


def test_cleanup_rejects_content_changes_after_parent_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    parent = api.objects[(fixture.resource, fixture.name)]
    api.add(pod(parent, uid="old-pod"))
    fixture.pods()
    state = restart_state(fixture)
    apply_restart_metadata(parent, state)
    replacement = api.add(pod(parent, uid="new-pod"))
    fixture.authorize_restart(state)
    fixture.pods()
    original = api.run

    def run(command: list[str], **kwargs: Any) -> Any:
        result = original(command, **kwargs)
        if api.deletes and api.deletes[-1][0] == fixture.resource:
            replacement["spec"]["nodeName"] = "foreign-node"
        return result

    monkeypatch.setattr(fixture.regional, "run", run)
    with pytest.raises(RegionalFixtureError, match="cleanup ownership changed"):
        fixture.delete()
    assert [kind for kind, _name, _options in api.deletes] == [fixture.resource]


def test_other_stopped_workloads_do_not_replace_this_fixtures_uid_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, _api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    state = restart_state(fixture)
    state["workflow"]["step_executions"][1]["details"]["stop_ownership_receipt_v1"][
        "workloads"
    ].append(
        {
            "workload_id": "gpu-fault-system/job/another-workload",
            "uid": "another-uid",
            "attempt_id": "another-attempt",
        }
    )
    fixture.authorize_restart(state)
    assert fixture.restart_custody is not None
    assert (
        fixture.restart_custody.source_uid
        == fixture.owned_uids[(fixture.resource, fixture.name)]
    )


def test_restart_controller_walk_has_a_fixed_depth_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    state = restart_state(fixture)
    fixture.authorize_restart(state)
    parent = api.objects[(fixture.resource, fixture.name)]
    apply_restart_metadata(parent, state)
    for index in range(9):
        child = deepcopy(parent)
        child["metadata"].update(name=f"intermediate-{index}", uid=f"uid-{index}")
        child["metadata"]["ownerReferences"] = [
            {
                "apiVersion": parent["apiVersion"],
                "kind": parent["kind"],
                "name": parent["metadata"]["name"],
                "uid": parent["metadata"]["uid"],
                "controller": True,
            }
        ]
        parent = api.add(child)
    child_pod = api.add(pod(parent, uid="new-pod"))
    before = deepcopy(fixture.owned_uids)
    with pytest.raises(WorkloadOwnershipError, match="exceeds its bound"):
        fixture.adopt(child_pod)
    assert fixture.owned_uids == before
