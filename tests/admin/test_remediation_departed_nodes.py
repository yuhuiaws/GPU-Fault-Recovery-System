"""``submit-remediation`` for nodes the platform took away, and suspended jobs.

Batch 139. ``restore`` skips an incident node whose departure is proven and
validates and releases the nodes the cluster still has; ``confirm-node-action``
settles a departed node's parked node actions ``node_departed``; both read the
departure proof through ``admin/remediation_departures.py``. The restore plan
and its result name every job the incident's records suspended and nothing
restarted -- the restore never restarts it (user decision G7) -- and the Pod
re-checks a ``node_departed`` confirmation against the Agent record it reads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import remediation_departures
from gpu_fault.admin import submit_remediation as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.departed_nodes import node_evidence
from gpu_fault.execution.node_action_uncertainty import (
    has_unresolved_node_action,
    operator_confirmation,
)
from gpu_fault.models import WorkflowStatus
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY
from gpu_fault.suspended_workloads import suspended_workload_line, suspended_workloads
from tests._builders import contained_stop_receipt
from tests.admin.test_admin_confirm_node_action import (
    INCIDENT,
    NODE,
    SIBLING,
    WORKFLOW,
    _agent_record,
    _inspect_payload,
    _node_json,
    _parked_store,
    _pod,
)
from tests.admin.test_admin_submit_remediation import RestoreHarness, _site

WORKLOAD = "training/pytorchjob/llm"


def _departure(node_id: str, *, gone: bool = True) -> dict[str, Any]:
    return node_evidence(
        node_id,
        departed={
            "node_id": node_id,
            "exists": False,
            "provider": "hyperpod",
            "instance_id": "i-0b139",
            "absent_from_provider": gone,
            "blockers": [] if gone else ["HyperPod still lists instance i-0b139"],
        },
        target={"hyperpod_cluster_name": "hp-cluster"},
        aws_region="us-west-2",
    )


def _stop(status: str = "SUCCEEDED", **details: Any) -> dict[str, Any]:
    receipt = contained_stop_receipt(
        workflow_id="wf-quarantine",
        workload_id=WORKLOAD,
        workloads=[
            {
                "workload_id": WORKLOAD,
                "namespace": "training",
                "kind": "pytorchjob",
                "name": "llm",
                "uid": "llm-uid",
                "attempt_id": "attempt-1",
                "owners": [],
            }
        ],
    )
    return {
        "step_index": 1,
        "operation": "STOP_WORKLOADS",
        "status": status,
        "phase": "official",
        "details": {STOP_OWNERSHIP_RECEIPT_KEY: receipt, **details},
    }


def _restart(status: str = "SUCCEEDED") -> dict[str, Any]:
    return {
        "step_index": 2,
        "operation": "RESTART_WORKLOAD",
        "status": status,
        "phase": "official",
        "details": {},
    }


def _record(request_id: str, *executions: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "official_steps": [
            {"operation": "MARK_UNSCHEDULABLE", "node_ids": ["node-a"]},
            {"operation": "STOP_WORKLOADS", "workload_ids": [WORKLOAD]},
            {"operation": "RESTART_WORKLOAD", "workload_ids": [WORKLOAD]},
        ],
        "step_executions": list(executions),
    }


# ----------------------------------------------------------- suspended jobs


def test_a_stop_that_landed_without_a_restart_leaves_its_job_suspended() -> None:
    [item] = suspended_workloads([_record("wf-a", _stop())])

    assert item == {
        "workload_id": WORKLOAD,
        "namespace": "training",
        "kind": "pytorchjob",
        "name": "llm",
        "stopped_by_workflow_id": "wf-a",
    }
    assert suspended_workload_line(item) == (
        "workload training/pytorchjob/llm stays suspended (stopped by workflow "
        "wf-a): the restore does not restart it; its owner must restart it"
    )


def test_only_a_suspend_that_landed_and_no_later_restart_counts() -> None:
    """A FAILED STOP counts only with ``workloads_suspended`` (the post-suspend
    refusal, batch 139); a restart later in the same workflow or in a newer one
    answers the suspend, a restart in an older workflow does not."""

    refused = _record("wf-refused", _stop("FAILED"))
    landed = _record("wf-landed", _stop("FAILED", workloads_suspended=True))
    restarted = _record("wf-restarted", _stop(), _restart())
    later = _record("wf-later", _restart())

    assert suspended_workloads([refused]) == []
    assert [
        item["stopped_by_workflow_id"] for item in suspended_workloads([landed])
    ] == ["wf-landed"]
    assert suspended_workloads([restarted]) == []
    assert suspended_workloads([later, landed]) == [], "the newer restart answers it"
    assert [
        item["stopped_by_workflow_id"] for item in suspended_workloads([landed, later])
    ] == ["wf-landed"], "an older restart does not answer a newer suspend"


def test_the_restore_plan_and_its_result_name_the_suspended_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection["workflow"].update(_record("wf-quarantine", _stop()))
    line = (
        "workload training/pytorchjob/llm stays suspended (stopped by workflow "
        "wf-quarantine): the restore does not restart it; its owner must restart it"
    )

    planned = harness.submit(plan_only=True, reference="CHG-R2")

    plan = planned.as_dict()["plan"]
    assert plan["suspended_workloads"] == [
        {
            "workload_id": WORKLOAD,
            "namespace": "training",
            "kind": "pytorchjob",
            "name": "llm",
            "stopped_by_workflow_id": "wf-quarantine",
        }
    ]
    assert line in plan["warnings"]
    assert line in planned.message
    assert "RESTART_WORKLOAD" not in plan["next_operations"], (
        "the restore never restarts the job"
    )

    applied = harness.submit(reference="CHG-R2")

    assert line in applied.message
    assert harness.submissions[0]["plan"]["suspended_workloads"][0]["name"] == "llm"


def test_a_restore_with_nothing_suspended_says_nothing_about_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection["workflow"].update(_record("wf-quarantine", _stop(), _restart()))

    result = harness.submit(plan_only=True, reference="CHG-R2")

    assert result.as_dict()["plan"]["suspended_workloads"] == []
    assert "stays suspended" not in result.message


# --------------------------------------------------------- departed nodes


def test_the_restore_reads_the_departure_of_a_missing_node_and_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection["incident"]["node_ids"] = ["node-a", "node-b"]
    harness.inspection["agents"] = {"node-a": None, "node-b": None}
    read: list[tuple[str, list[str]]] = []

    def departures(site, cluster_id, node_ids, *, agents):
        read.append((cluster_id, list(node_ids)))
        assert agents == {"node-a": None, "node-b": None}
        return {node: _departure(node) for node in node_ids}

    monkeypatch.setattr(module, "node_departures", departures)

    result = harness.submit(reference="CHG-R3")

    assert read == [("gpu-a", ["node-b"])], "only the node the cluster lost"
    plan = harness.submissions[0]["plan"]
    assert plan["restore_node_ids"] == ["node-a"]
    decisions = {item["node_id"]: item for item in plan["node_decisions"]}
    assert decisions["node-b"]["decision"] == "skip"
    assert decisions["node-b"]["departed"] is True
    assert "node-a" in result.message


def test_the_restore_refuses_a_missing_node_whose_departure_is_unproven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection["incident"]["node_ids"] = ["node-a", "node-b"]
    harness.inspection["agents"] = {"node-a": None, "node-b": None}
    monkeypatch.setattr(
        module,
        "node_departures",
        lambda site, cluster_id, node_ids, *, agents: {
            node: _departure(node, gone=False) for node in node_ids
        },
    )

    with pytest.raises(BootstrapError, match="missing from cluster gpu-a: node-b"):
        harness.submit(reference="CHG-R3")
    assert harness.submissions == []


def test_a_restore_of_present_nodes_reads_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "node_departures",
        lambda *a, **k: pytest.fail("no node is missing; nothing to prove"),
    )

    harness.submit(plan_only=True, reference="CHG-R4")


def test_node_departures_reads_the_provider_for_the_recorded_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str | None]] = []

    def provider(site, cluster_id, node_id, inventories, read_at, *, instance_id):
        calls.append((node_id, instance_id))
        read_at["ec2"] = "2026-10-18T12:00:00+00:00"
        return {
            "node_id": node_id,
            "exists": False,
            "provider": "eks-managed",
            "instance_id": instance_id,
            "ec2_state": "terminated",
            "absent_from_provider": True,
            "restored": True,
            "blockers": [],
        }

    monkeypatch.setattr(remediation_departures, "_departed_node_evidence", provider)
    monkeypatch.setattr(
        remediation_departures,
        "cluster_target",
        lambda site, cluster_id: {"cluster_id": cluster_id, "provider": "eks-managed"},
    )
    site = _site(tmp_path)

    proofs = remediation_departures.node_departures(
        site,
        "gpu-a",
        ["ip-10-0-1-1", "ip-10-0-1-2"],
        agents={
            "ip-10-0-1-1": {"node_instance_id": "i-0aaa1111"},
            "ip-10-0-1-2": {"node_instance_id": "node-uid-not-an-instance"},
        },
    )

    assert calls == [("ip-10-0-1-1", "i-0aaa1111"), ("ip-10-0-1-2", None)]
    proof = proofs["ip-10-0-1-1"]
    assert proof["absent_from_kubernetes"] is True
    assert proof["absent_from_provider"] is True
    assert proof["sources"]["ec2"]["state"] == "terminated"
    assert set(proof["read_at"]) == {"kubernetes", "ec2"}


def test_a_failed_provider_read_is_a_blocker_not_a_departure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def provider(*args, **kwargs):
        raise BootstrapError("hyperpod list-cluster-nodes was throttled")

    monkeypatch.setattr(remediation_departures, "_departed_node_evidence", provider)
    monkeypatch.setattr(
        remediation_departures,
        "cluster_target",
        lambda site, cluster_id: {"cluster_id": cluster_id},
    )

    [proof] = remediation_departures.node_departures(
        _site(tmp_path), "gpu-a", ["hyperpod-i-0b139"], agents={}
    ).values()

    assert proof["absent_from_provider"] is False
    assert any("throttled" in item for item in proof["blockers"]), proof["blockers"]


# --------------------------------------------------- confirm-node-action


def _departed_agent(store) -> None:
    now = datetime.now(timezone.utc)
    _agent_record(
        store,
        last_seen_at=now - timedelta(minutes=25),
        lease_expires_at=now - timedelta(minutes=23),
    )


def test_confirm_node_action_settles_a_departed_node_end_to_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The admin plan judges the departure proof, the Pod re-checks that the
    Agent is not alive and writes ``node_departed``; a fresh heartbeat in
    between refuses the write."""

    store, workflow = _parked_store()
    _departed_agent(store)
    run = _pod(monkeypatch, capsys, store)
    inspected = run(_inspect_payload())

    plan = module.build_confirm_node_action_plan(
        inspected,
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-DEPARTED",
        live_nodes={SIBLING: _node_json(SIBLING)},
        now=datetime.now(timezone.utc),
        departure=_departure(NODE),
    )
    [confirmation] = plan.confirmations
    assert confirmation["settled_by"] == "node_departed"
    payload = module.confirmation_payload(
        plan, operator="ops@example", reference="CHG-DEPARTED"
    )
    assert payload["node_evidence"]["exists"] is False
    assert payload["node_evidence"]["departure"]["absent_from_provider"] is True

    # An Agent that heartbeats again is not departed: the Pod refuses.
    _agent_record(store)
    with pytest.raises(SystemExit, match="lease has not expired"):
        run(payload)
    assert has_unresolved_node_action(store.get_workflow(WORKFLOW)), (
        "the refused write left the record unresolved"
    )

    _departed_agent(store)
    written = run(payload)

    assert written["no_op"] is False
    stored = store.get_workflow(WORKFLOW)
    assert not has_unresolved_node_action(stored), "the Pod wrote node_departed"
    assert stored.status is WorkflowStatus.BLOCKED
    execution = next(item for item in stored.step_executions if item.step_index == 1)
    assert operator_confirmation(execution.details)["settled_by"] == "node_departed"


def test_the_pod_refuses_a_departure_the_payload_does_not_prove(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store, _workflow = _parked_store()
    _departed_agent(store)
    run = _pod(monkeypatch, capsys, store)
    plan = module.build_confirm_node_action_plan(
        run(_inspect_payload()),
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-DEPARTED",
        live_nodes={SIBLING: _node_json(SIBLING)},
        now=datetime.now(timezone.utc),
        departure=_departure(NODE),
    )
    payload = module.confirmation_payload(
        plan, operator="ops@example", reference="CHG-DEPARTED"
    )
    forged = {
        **payload,
        "node_evidence": {
            **payload["node_evidence"],
            "departure": {
                **payload["node_evidence"]["departure"],
                "absent_from_provider": False,
            },
        },
    }

    with pytest.raises(SystemExit, match="does not show it gone"):
        run(forged)
    assert has_unresolved_node_action(store.get_workflow(WORKFLOW)), (
        "a forged departure writes nothing"
    )


def test_the_cli_reads_the_departure_of_a_node_the_cluster_no_longer_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.admin.test_admin_confirm_node_action import ConfirmHarness

    harness = ConfirmHarness(tmp_path, monkeypatch)
    harness.nodes = {SIBLING: _node_json(SIBLING)}
    now = harness.clock
    harness.inspection["agents"][NODE].update(
        last_seen_at=(now - timedelta(minutes=30)).isoformat(),
        lease_expires_at=(now - timedelta(minutes=28)).isoformat(),
    )
    read: list[list[str]] = []

    def departures(site, cluster_id, node_ids, *, agents):
        read.append(list(node_ids))
        return {node: _departure(node) for node in node_ids}

    monkeypatch.setattr(module, "node_departures", departures)

    result = harness.submit(plan_only=True, reference=None)

    assert read == [[NODE]]
    [step] = result.as_dict()["plan"]["confirmations"]
    assert step["settled_by"] == "node_departed"
    assert "(node departed)" in result.message


def test_the_cli_reads_no_provider_for_a_node_the_cluster_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.admin.test_admin_confirm_node_action import ConfirmHarness

    harness = ConfirmHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module, "node_departures", lambda *a, **k: pytest.fail("node is present")
    )

    harness.submit(plan_only=True, reference=None)
