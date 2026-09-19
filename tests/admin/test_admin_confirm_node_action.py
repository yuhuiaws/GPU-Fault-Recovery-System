"""``gpu-fault-admin submit-remediation --disposition confirm-node-action``.

The operator's lever for a workflow parked BLOCKED / NEEDS_OPERATOR on a node
action whose outcome the executor never saw (GF-REGIONAL-DESTR-014,
unknown-reboot). The admin side reads the node through the GPU kubeconfig,
the Pod reads the fleet record and the workflow, both compute the same verdict
(``execution.node_action_confirmation``), and the Pod writes the confirmation
with a compare-and-set. These tests pin the plan the CLI derives, the evidence
it demands, the refusals, the no-op rerun, the in-Pod write, and that the
CHECK_MECHANICALS deadline refusal no longer speaks for a parked record.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import submit_remediation as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.execution.node_action_uncertainty import (
    has_unresolved_node_action,
    operator_confirmation,
)
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.admin.conftest import TEST_OPERATOR_ARN
from tests.admin.test_admin_submit_remediation import _site

NOW = datetime(2026, 9, 19, 11, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=4)
INCIDENT = "inc-kernel-log-kmsg-xid-79"
WORKFLOW = "workflow-2a1858b1"
NODE = "hyperpod-node-0246"
SIBLING = "hyperpod-i-096b"
OLD_BOOT = "8f0c5f2e-old"
NEW_BOOT = "3b7e9a11-new"
DISPOSITION = "confirm-node-action"
RESTART = WorkflowOperation.RESTART_NODE


def _workflow_model(**overrides) -> WorkflowRequest:
    values = {
        "status": WorkflowStatus.BLOCKED,
        "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        "fencing_token": 4,
        "execution_epoch": 3,
        "merge_revision": 2,
        "dag_enabled": True,
        "official_steps": [
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[NODE]),
            workflow_step(RESTART, node_ids=[NODE]),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[SIBLING]),
            workflow_step(RESTART, node_ids=[SIBLING]),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[SIBLING]),
        ],
        "completed_step_indexes": [0, 2, 3, 4],
        "completed_operations": [
            WorkflowOperation.MARK_UNSCHEDULABLE,
            RESTART,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
        "step_executions": [
            workflow_step_execution(
                1,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-0246",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "cmd-0246",
                },
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=10),
            ),
            workflow_step_execution(
                3, RESTART, WorkflowStepStatus.SUCCEEDED, phase="official"
            ),
        ],
        "blocked_reasons": ["node branch escalation exhausted"],
        "execution_deadline": NOW - timedelta(hours=3),
        "lifetime_deadline_at": NOW - timedelta(hours=2),
        "created_at": STARTED,
        "updated_at": NOW - timedelta(hours=2),
    }
    values.update(overrides)
    return workflow_request(WORKFLOW, INCIDENT, **values)


def _incident_model(**overrides) -> FaultIncident:
    values = {
        "cluster_id": "gpu-a",
        "node_ids": [NODE, SIBLING],
        "gpu_uuids": ["GPU-0246-0"],
        "state": IncidentState.QUARANTINED,
        "workflow_request_id": WORKFLOW,
        "fencing_token": 4,
        "created_at": STARTED,
        "updated_at": NOW - timedelta(hours=2),
    }
    values.update(overrides)
    return fault_incident(INCIDENT, "event-79", **values)


def _dump(model) -> dict:
    return json.loads(model.model_dump_json())


def _agent_dump(node_id: str = NODE, **overrides) -> dict:
    values = {
        "cluster_id": "gpu-a",
        "node_id": node_id,
        "endpoint": "https://10.0.0.7:9099",
        "agent_version": "2026.9.18",
        "artifact_sha256": "a" * 64,
        "policy_version": "610",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "b" * 64,
        "allowed_operations": ["RESTART_NODE"],
        "boot_id": NEW_BOOT,
        "agent_incarnation_id": "inc-new",
        "retired_incarnation_ids": ["inc-old"],
        "first_seen_at": STARTED.isoformat(),
        "last_seen_at": (NOW - timedelta(seconds=15)).isoformat(),
        "lease_expires_at": (NOW + timedelta(seconds=45)).isoformat(),
        "generation": 7,
        "lifecycle_state": "ACTIVE",
    }
    values.update(overrides)
    return values


def _command(**overrides) -> dict:
    values = {
        "command_id": "cmd-0246",
        "step_index": 1,
        "batched_step_indexes": [],
        "status": "FAILED",
        "status_source": "executor-execution-timeout-outcome-unknown",
        "fencing_token": 4,
        "result_details": {"agent_baselines": {NODE: {"boot_id": OLD_BOOT}}},
    }
    values.update(overrides)
    return values


def _inspection(**overrides) -> dict:
    workflow = _dump(_workflow_model())
    value = {
        "incident": _dump(_incident_model()),
        "workflow": workflow,
        "open_workflows": [],
        "blocked_workflows": [workflow],
        "agents": {NODE: _agent_dump(), SIBLING: _agent_dump(SIBLING)},
        "remote_commands": {WORKFLOW: [_command()]},
        "node_gpu_uuids": {NODE: ["GPU-0246-0"], SIBLING: ["GPU-096B-0"]},
        "active_observations": [],
        "existing_markers": [],
        "acknowledgement_timeout_seconds": 86400,
    }
    value.update(overrides)
    return value


def _node_json(name: str = NODE, **overrides) -> dict:
    value = {
        "metadata": {"name": name, "uid": f"uid-{name}", "resourceVersion": "91"},
        "spec": {"unschedulable": False},
        "status": {
            "nodeInfo": {"bootID": NEW_BOOT},
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "lastTransitionTime": (NOW - timedelta(minutes=50)).isoformat(),
                }
            ],
        },
    }
    for key, item in overrides.items():
        value[key] = item
    return value


def _nodes(**node_overrides) -> dict:
    return {NODE: _node_json(**node_overrides), SIBLING: _node_json(SIBLING)}


def _plan(inspection: dict | None = None, nodes: dict | None = None, **kwargs):
    values = {
        "incident_id": INCIDENT,
        "node_id": NODE,
        "operator": TEST_OPERATOR_ARN,
        "reference": "CHG-2026-0919-04",
        "live_nodes": _nodes() if nodes is None else nodes,
        "now": NOW,
    }
    values.update(kwargs)
    return module.build_confirm_node_action_plan(inspection or _inspection(), **values)


# --------------------------------------------------------------------- plan


def test_the_plan_derives_the_confirmation_from_the_record_and_the_node() -> None:
    plan = _plan()

    assert plan.disposition == DISPOSITION
    assert plan.incident_id == INCIDENT and plan.cluster_id == "gpu-a"
    assert plan.node_id == NODE
    assert plan.workflow_request_id == WORKFLOW
    assert (plan.fencing_token, plan.execution_epoch, plan.merge_revision) == (4, 3, 2)
    assert plan.workflow_status == "BLOCKED" and plan.blocked_kind == "NEEDS_OPERATOR"
    assert len(plan.confirmations) == 1
    confirmation = plan.confirmations[0]
    assert (
        confirmation["step_index"] == 1 and confirmation["operation"] == "RESTART_NODE"
    )
    assert confirmation["previous_boot_id"] == OLD_BOOT
    assert confirmation["previous_boot_id_source"] == (
        "remote_command.cmd-0246.agent_baselines"
    )
    assert confirmation["observed_boot_id"] == NEW_BOOT
    assert confirmation["agent_generation"] == 7
    assert confirmation["actor"] == TEST_OPERATOR_ARN
    assert confirmation["reference"] == "CHG-2026-0919-04"
    rendered = plan.as_dict()
    assert rendered["steps"] == [
        {
            "step_index": 1,
            "operation": "RESTART_NODE",
            "phase": "official",
            "remote_command_id": "cmd-0246",
            "previous_boot_id": OLD_BOOT,
            "previous_boot_id_source": "remote_command.cmd-0246.agent_baselines",
            "observed_boot_id": NEW_BOOT,
        }
    ]
    assert rendered["node_evidence"]["boot_id"] == NEW_BOOT
    assert rendered["node_evidence"]["ready"] is True
    assert rendered["agent_evidence"]["lifecycle_state"] == "ACTIVE"
    assert rendered["already_confirmed"] == []
    assert rendered["next_operations"] == []


def test_a_parked_record_past_every_deadline_is_accepted() -> None:
    inspection = _inspection()
    inspection["workflow"]["execution_deadline"] = (NOW - timedelta(days=4)).isoformat()
    inspection["workflow"]["lifetime_deadline_at"] = (
        NOW - timedelta(days=4)
    ).isoformat()
    inspection["blocked_workflows"] = [inspection["workflow"]]

    assert _plan(inspection).confirmations, "deadlines do not gate a confirmation"


def test_the_check_mechanicals_deadline_refusal_no_longer_speaks_for_a_parked_record() -> (
    None
):
    """``_deadline_passed`` said "the workflow will fail on its own"; a BLOCKED
    record never does. It is refused as not waiting, with the parked-record
    levers named, while a live investigation past its deadline keeps the old
    refusal."""

    parked = _inspection()
    with pytest.raises(BootstrapError) as refused:
        module.build_remediation_plan(
            parked,
            incident_id=INCIDENT,
            disposition="inspected",
            profile_version="hyperpod-v1",
            now=NOW,
        )
    message = str(refused.value)
    assert "will fail on its own" not in message
    assert "BLOCKED" in message and DISPOSITION in message, message

    live = _inspection()
    live["workflow"]["status"] = "RUNNING"
    live["workflow"]["blocked_kind"] = None
    with pytest.raises(BootstrapError, match="will fail on its own"):
        module.build_remediation_plan(
            live,
            incident_id=INCIDENT,
            disposition="inspected",
            profile_version="hyperpod-v1",
            now=NOW,
        )


def test_the_plan_refuses_a_node_outside_the_incident() -> None:
    with pytest.raises(BootstrapError, match="is not named by incident"):
        _plan(
            node_id="hyperpod-i-else",
            nodes={**_nodes(), "hyperpod-i-else": _node_json("hyperpod-i-else")},
        )


def test_the_plan_refuses_when_no_parked_record_is_unresolved_on_the_node() -> None:
    with pytest.raises(BootstrapError) as refused:
        _plan(node_id=SIBLING)

    message = str(refused.value)
    assert f"has an unresolved node action on node {SIBLING}" in message, message
    assert f"step 1 RESTART_NODE on {NODE}" in message, (
        "the refusal lists what is unresolved so a wrong node id is obvious"
    )


def test_the_plan_names_the_missing_evidence() -> None:
    with pytest.raises(BootstrapError, match="still runs boot 8f0c5f2e-old"):
        _plan(
            nodes=_nodes(
                status={
                    "nodeInfo": {"bootID": OLD_BOOT},
                    "conditions": [{"type": "Ready", "status": "True"}],
                }
            )
        )

    draining = _inspection()
    draining["agents"][NODE]["lifecycle_state"] = "DRAINING"
    with pytest.raises(BootstrapError, match="lifecycle state is DRAINING"):
        _plan(draining)

    with pytest.raises(BootstrapError, match="is not Ready"):
        _plan(
            nodes=_nodes(
                status={
                    "nodeInfo": {"bootID": NEW_BOOT},
                    "conditions": [{"type": "Ready", "status": "False"}],
                }
            )
        )

    with pytest.raises(BootstrapError, match="is not in the cluster"):
        _plan(nodes={SIBLING: _node_json(SIBLING)})


def test_the_plan_refuses_an_image_that_returns_no_fleet_evidence() -> None:
    inspection = _inspection()
    del inspection["agents"]
    del inspection["remote_commands"]

    with pytest.raises(BootstrapError, match="control plane image predates"):
        _plan(inspection)


def test_the_plan_refuses_two_parked_records_unresolved_on_the_same_node() -> None:
    inspection = _inspection()
    twin = json.loads(json.dumps(inspection["workflow"]))
    twin["request_id"] = "workflow-twin"
    inspection["blocked_workflows"] = [inspection["workflow"], twin]
    inspection["remote_commands"]["workflow-twin"] = []

    with pytest.raises(BootstrapError, match="more than one BLOCKED workflow"):
        _plan(inspection)


def test_the_plan_finds_the_parked_record_behind_a_later_pointer() -> None:
    """The incident may already point at a restore that failed; the parked
    record is still found among the incident's BLOCKED workflows."""

    inspection = _inspection()
    later = json.loads(json.dumps(inspection["workflow"]))
    later["request_id"] = "workflow-validated-restore-failed"
    later["status"] = "FAILED"
    later["blocked_kind"] = None
    later["step_executions"] = []
    inspection["incident"]["workflow_request_id"] = later["request_id"]
    inspection["workflow"] = later

    plan = _plan(inspection)

    assert plan.workflow_request_id == WORKFLOW


def test_a_stale_generation_is_refused() -> None:
    inspection = _inspection()
    inspection["incident"]["fencing_token"] = 5

    with pytest.raises(BootstrapError, match="stale generation"):
        _plan(inspection)


def test_an_already_confirmed_node_yields_a_plan_with_nothing_to_write() -> None:
    inspection = _inspection()
    execution = inspection["workflow"]["step_executions"][0]
    execution["details"] = {
        "outcome_unknown": False,
        "manual_confirmation_required": False,
        "operator_confirmed": {
            "actor": "ops",
            "reference": "CHG-EARLIER",
            "confirmed_at": (NOW - timedelta(hours=1)).isoformat(),
            "node_id": NODE,
            "operation": "RESTART_NODE",
        },
    }
    inspection["blocked_workflows"] = [inspection["workflow"]]

    plan = _plan(inspection)

    assert plan.confirmations == ()
    assert [item["reference"] for item in plan.already_confirmed] == ["CHG-EARLIER"]


# ------------------------------------------------------------------ request


def test_the_request_requires_a_node_and_a_reference(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with pytest.raises(BootstrapError, match="requires --node"):
        module.SubmitRemediationRequest(
            site=site, incident_id=INCIDENT, disposition=DISPOSITION, reference="CHG"
        )
    with pytest.raises(BootstrapError, match="requires --reference"):
        module.SubmitRemediationRequest(
            site=site, incident_id=INCIDENT, disposition=DISPOSITION, node_id=NODE
        )
    planned = module.SubmitRemediationRequest(
        site=site,
        incident_id=INCIDENT,
        disposition=DISPOSITION,
        node_id=NODE,
        plan_only=True,
    )
    assert planned.node_id == NODE, "a plan may be printed before the ticket exists"
    with pytest.raises(BootstrapError, match="--node applies to"):
        module.SubmitRemediationRequest(
            site=site, incident_id=INCIDENT, disposition="inspected", node_id=NODE
        )


# ------------------------------------------------------------------ submit


class ConfirmHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = _site(tmp_path)
        self.inspection = _inspection()
        self.nodes = _nodes()
        self.submissions: list[dict] = []
        self.clock = NOW

        def inspect(site, incident_id, *, disposition):
            assert disposition == DISPOSITION, disposition
            return json.loads(json.dumps(self.inspection))

        def submit(site, plan, *, operator, reference):
            self.submissions.append(
                {"plan": plan.as_dict(), "operator": operator, "reference": reference}
            )
            workflow = json.loads(json.dumps(self.inspection["workflow"]))
            workflow["step_executions"][0]["details"] = {
                "outcome_unknown": False,
                "manual_confirmation_required": False,
                "operator_confirmed": plan.confirmations[0],
            }
            return {
                "no_op": False,
                "workflow": workflow,
                "confirmations": list(plan.confirmations),
            }

        monkeypatch.setattr(module, "inspect_incident", inspect)
        monkeypatch.setattr(
            module, "cluster_nodes", lambda site, cluster_id: self.nodes
        )
        monkeypatch.setattr(module, "submit_node_action_confirmation", submit)
        monkeypatch.setattr(
            module,
            "run_control_plane_script",
            lambda *a, **k: pytest.fail("unreachable"),
        )

    def submit(self, **overrides) -> module.SubmissionResult:
        values = {
            "site": self.site,
            "incident_id": INCIDENT,
            "disposition": DISPOSITION,
            "node_id": NODE,
            "reference": "CHG-2026-0919-04",
        }
        values.update(overrides)
        return module.submit_remediation(
            module.SubmitRemediationRequest(**values), now=self.now
        )

    def now(self) -> datetime:
        self.clock += timedelta(seconds=1)
        return self.clock

    def evidence(self) -> list[Path]:
        return sorted(module.evidence_directory(self.site, INCIDENT).glob("*.json"))


def test_plan_only_prints_the_confirmation_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ConfirmHarness(tmp_path, monkeypatch)

    result = harness.submit(plan_only=True, reference=None)

    assert result.message.startswith("plan only"), result.message
    assert harness.submissions == []
    assert harness.evidence() == []
    assert result.as_dict()["plan"]["steps"][0]["observed_boot_id"] == NEW_BOOT


def test_the_submission_binds_the_record_and_carries_the_node_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ConfirmHarness(tmp_path, monkeypatch)

    result = harness.submit()

    assert result.no_op is False
    assert len(harness.submissions) == 1
    submitted = harness.submissions[0]
    assert submitted["operator"] == TEST_OPERATOR_ARN
    assert submitted["reference"] == "CHG-2026-0919-04"
    assert submitted["plan"]["workflow_request_id"] == WORKFLOW
    assert "RESTART_NODE" in result.message and NODE in result.message
    assert WORKFLOW in result.message
    [evidence] = harness.evidence()
    assert evidence.name.startswith("confirm-node-action-"), evidence.name
    recorded = json.loads(evidence.read_text(encoding="utf-8"))
    assert recorded["status"] == "SUBMITTED"
    assert recorded["operator"] == TEST_OPERATOR_ARN
    assert recorded["fencing_token"] == 4
    assert recorded["plan"]["node_id"] == NODE
    assert recorded["submission"]["confirmations"][0]["observed_boot_id"] == NEW_BOOT


def test_a_rerun_on_a_confirmed_node_is_a_no_op_that_names_the_earlier_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ConfirmHarness(tmp_path, monkeypatch)
    execution = harness.inspection["workflow"]["step_executions"][0]
    execution["details"] = {
        "outcome_unknown": False,
        "manual_confirmation_required": False,
        "operator_confirmed": {
            "actor": "ops",
            "reference": "CHG-EARLIER",
            "confirmed_at": (NOW - timedelta(hours=1)).isoformat(),
            "node_id": NODE,
            "operation": "RESTART_NODE",
        },
    }
    harness.inspection["blocked_workflows"] = [harness.inspection["workflow"]]

    result = harness.submit()

    assert result.no_op is True
    assert "CHG-EARLIER" in result.message
    assert harness.submissions == []
    [evidence] = harness.evidence()
    assert json.loads(evidence.read_text(encoding="utf-8"))["status"] == "NO_OP"


def test_a_refusal_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ConfirmHarness(tmp_path, monkeypatch)
    harness.inspection["agents"][NODE]["boot_id"] = OLD_BOOT
    harness.nodes[NODE]["status"]["nodeInfo"]["bootID"] = OLD_BOOT

    with pytest.raises(BootstrapError, match="still runs boot"):
        harness.submit()

    assert harness.submissions == []
    assert harness.evidence() == []


def test_submit_node_action_confirmation_sends_the_bound_record_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    payloads: list[dict] = []
    monkeypatch.setattr(
        module,
        "run_control_plane_script",
        lambda site, payload, *, script: payloads.append(payload) or {"no_op": False},
    )
    plan = _plan()

    module.submit_node_action_confirmation(
        site, plan, operator=TEST_OPERATOR_ARN, reference="CHG-2026-0919-04"
    )

    [payload] = payloads
    assert payload["mode"] == DISPOSITION
    assert payload["incident_id"] == INCIDENT
    assert payload["workflow_request_id"] == WORKFLOW
    assert payload["node_id"] == NODE
    assert payload["expected"] == {
        "fencing_token": 4,
        "workflow_request_id": WORKFLOW,
        "workflow_status": "BLOCKED",
        "execution_epoch": 3,
        "merge_revision": 2,
        "node_ids": sorted([NODE, SIBLING]),
    }
    assert payload["node_evidence"]["boot_id"] == NEW_BOOT
    assert payload["node_evidence"]["uid"] == f"uid-{NODE}"
    assert payload["operator"] == TEST_OPERATOR_ARN
    assert payload["reference"] == "CHG-2026-0919-04"
    [pair] = payload["confirmations"]
    assert pair["execution"] == {
        "phase": "official",
        "step_index": 1,
        "operation": "RESTART_NODE",
        "status": "FAILED",
        "adapter_operation_id": "remote/cmd-0246",
        "details": {
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "remote_command_id": "cmd-0246",
        },
    }, "the Pod binds its write to the step record exactly as inspected"
    assert pair["confirmation"] == plan.confirmations[0]


# ------------------------------------------------------------------ in-Pod


def _agent_record(store, node_id: str = NODE, **overrides) -> AgentRecord:
    now = datetime.now(timezone.utc)
    values = {
        "cluster_id": "gpu-a",
        "node_id": node_id,
        "endpoint": "https://10.0.0.7:9099",
        "agent_version": "2026.9.18",
        "artifact_sha256": "a" * 64,
        "policy_version": "610",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "b" * 64,
        "allowed_operations": [RESTART],
        "boot_id": NEW_BOOT,
        "agent_incarnation_id": "inc-new",
        "retired_incarnation_ids": ["inc-old"],
        "first_seen_at": now - timedelta(hours=5),
        "last_seen_at": now - timedelta(seconds=10),
        "lease_expires_at": now + timedelta(seconds=50),
        "generation": 7,
    }
    values.update(overrides)
    record = AgentRecord(**values)
    store.save_agent(record)
    return record


def _pod(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], store):
    """Run ``REMEDIATION_SCRIPT`` the way ``kubectl exec`` would, on ``store``."""

    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store)
    monkeypatch.setattr(context.dispatcher, "wake", lambda: None)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )

    def run(payload: dict) -> dict:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(compile(module.REMEDIATION_SCRIPT, "<submit-remediation>", "exec"), {})
        return json.loads(capsys.readouterr().out)

    return run


def _parked_store(*, agent_boot_id: str = NEW_BOOT):
    store = build_store()
    started = datetime.now(timezone.utc) - timedelta(hours=4)
    workflow = _workflow_model(
        step_executions=[
            workflow_step_execution(
                1,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-0246",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "cmd-0246",
                    "agent_baselines": {NODE: {"boot_id": OLD_BOOT}},
                },
                started_at=started,
                updated_at=started + timedelta(minutes=10),
            ),
            workflow_step_execution(
                3, RESTART, WorkflowStepStatus.SUCCEEDED, phase="official"
            ),
        ]
    )
    store.save_incident(_incident_model())
    store.save_workflow(workflow)
    _agent_record(store, boot_id=agent_boot_id)
    _agent_record(store, SIBLING)
    return store, workflow


def _inspect_payload() -> dict:
    return {
        "mode": "inspect",
        "incident_id": INCIDENT,
        "action_incident_id": f"inc-operator-{INCIDENT}-{DISPOSITION}",
    }


def test_the_in_pod_mode_writes_the_admin_verdict_with_a_compare_and_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole lever against the deployed image: the script's own inspection
    feeds the checkout's plan builder, the plan's payload goes back to the
    script, which re-checks the live facts it can read and writes."""

    store, workflow = _parked_store()
    run = _pod(monkeypatch, capsys, store)

    inspected = run(_inspect_payload())
    assert [item["request_id"] for item in inspected["blocked_workflows"]] == [WORKFLOW]
    assert inspected["agents"][NODE]["boot_id"] == NEW_BOOT
    assert inspected["remote_commands"] == {WORKFLOW: []}
    assert set(inspected["node_gpu_uuids"]) == {NODE, SIBLING}
    plan = module.build_confirm_node_action_plan(
        inspected,
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-1",
        live_nodes=_nodes(),
        now=datetime.now(timezone.utc),
    )
    payload = module.confirmation_payload(
        plan, operator="ops@example", reference="CHG-1"
    )

    first = run(payload)

    assert first["no_op"] is False
    assert first["confirmations"][0]["previous_boot_id"] == OLD_BOOT
    assert (
        first["confirmations"][0]["previous_boot_id_source"] == "step.agent_baselines"
    )
    stored = store.get_workflow(WORKFLOW)
    assert not has_unresolved_node_action(stored), "the Pod wrote the confirmation"
    assert stored.status is WorkflowStatus.BLOCKED
    assert stored.events[-1].code == WorkflowEventCode.NODE_ACTION_CONFIRMED.value
    assert stored.events[-1].actor == "ops@example"
    execution = next(item for item in stored.step_executions if item.step_index == 1)
    assert execution.details["outcome_unknown"] is False
    assert execution.details["manual_confirmation_required"] is False
    assert operator_confirmation(execution.details)["reference"] == "CHG-1"
    assert stored.merge_revision == workflow.merge_revision, (
        "a compare-and-set save keeps the revision the operator approved"
    )

    with pytest.raises(SystemExit, match="changed since inspection"):
        run(payload)
    assert store.get_workflow(WORKFLOW) == stored, "a stale payload writes nothing"

    replanned = module.build_confirm_node_action_plan(
        run(_inspect_payload()),
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-2",
        live_nodes=_nodes(),
        now=datetime.now(timezone.utc),
    )
    assert replanned.confirmations == ()
    assert [item["reference"] for item in replanned.already_confirmed] == ["CHG-1"]


def test_the_pod_write_matches_the_products_own_confirmation() -> None:
    """The script duplicates ``apply_node_action_confirmation`` with deployed
    primitives; this pins the two copies to the same record."""

    import io as _io

    from gpu_fault.app import ApplicationContext
    from gpu_fault.execution.node_action_confirmation import (
        AgentEvidence,
        KubernetesNodeEvidence,
        apply_node_action_confirmation,
        confirm_node_action,
    )

    store, workflow = _parked_store()
    inspected_agent = store.get_agent("gpu-a", NODE)
    now = datetime.now(timezone.utc)
    verdict = confirm_node_action(
        workflow=workflow,
        incident=store.get_incident(INCIDENT),
        node_id=NODE,
        node=KubernetesNodeEvidence.from_node(NODE, _node_json()),
        agent=AgentEvidence.from_record(NODE, inspected_agent),
        remote_commands=[],
        actor="ops@example",
        reference="CHG-1",
        now=now,
    )
    product = apply_node_action_confirmation(workflow, verdict, now=now)
    plan = module.build_confirm_node_action_plan(
        {
            "incident": json.loads(store.get_incident(INCIDENT).model_dump_json()),
            "workflow": json.loads(workflow.model_dump_json()),
            "blocked_workflows": [json.loads(workflow.model_dump_json())],
            "agents": {NODE: json.loads(inspected_agent.model_dump_json())},
            "remote_commands": {WORKFLOW: []},
        },
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-1",
        live_nodes=_nodes(),
        now=now,
    )
    payload = module.confirmation_payload(
        plan, operator="ops@example", reference="CHG-1"
    )
    context = ApplicationContext(store=store)
    stdout = _io.StringIO()
    real_stdin, real_stdout = sys.stdin, sys.stdout
    from_environment = ApplicationContext.from_environment
    try:
        ApplicationContext.from_environment = classmethod(lambda cls: context)  # type: ignore[method-assign]
        sys.stdin, sys.stdout = _io.StringIO(json.dumps(payload)), stdout
        exec(compile(module.REMEDIATION_SCRIPT, "<submit-remediation>", "exec"), {})
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
        ApplicationContext.from_environment = from_environment  # type: ignore[method-assign]

    pod = store.get_workflow(WORKFLOW)
    pod_execution = next(item for item in pod.step_executions if item.step_index == 1)
    product_execution = next(
        item for item in product.step_executions if item.step_index == 1
    )
    assert pod_execution.details == product_execution.details
    assert pod_execution.status is product_execution.status
    assert (pod.events[-1].kind, pod.events[-1].code, pod.events[-1].actor) == (
        product.events[-1].kind,
        product.events[-1].code,
        product.events[-1].actor,
    )
    assert pod.events[-1].step_index == product.events[-1].step_index
    assert pod.events[-1].details == product.events[-1].details
    assert pod.status is product.status


def test_the_in_pod_mode_refuses_drift_stale_records_and_unproven_facts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store, _workflow = _parked_store()
    run = _pod(monkeypatch, capsys, store)
    plan = module.build_confirm_node_action_plan(
        run(_inspect_payload()),
        incident_id=INCIDENT,
        node_id=NODE,
        operator="ops@example",
        reference="CHG-1",
        live_nodes=_nodes(),
        now=datetime.now(timezone.utc),
    )
    payload = module.confirmation_payload(
        plan, operator="ops@example", reference="CHG-1"
    )

    with pytest.raises(SystemExit, match="incident moved since inspection"):
        run({**payload, "expected": {**payload["expected"], "execution_epoch": 9}})

    with pytest.raises(SystemExit, match="reference differs"):
        run({**payload, "reference": "CHG-OTHER"})

    # The fleet record moved under the plan: the node is back on the old boot.
    _agent_record(store, boot_id=OLD_BOOT)
    with pytest.raises(SystemExit, match="kubelet reports boot"):
        run(payload)
    _agent_record(store, boot_id=NEW_BOOT)

    # The plan cites a pre-reboot boot equal to the running one: no reboot.
    forged = json.loads(json.dumps(payload))
    forged["confirmations"][0]["confirmation"]["previous_boot_id"] = NEW_BOOT
    with pytest.raises(SystemExit, match="still runs boot"):
        run(forged)

    assert has_unresolved_node_action(store.get_workflow(WORKFLOW)), (
        "every refusal wrote nothing"
    )


# --------------------------------------------------------------------- CLI


def test_the_cli_accepts_the_disposition_with_its_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    captured: list[module.SubmitRemediationRequest] = []
    plan = _plan()
    monkeypatch.setattr(
        module,
        "submit_remediation",
        lambda request: captured.append(request) or module.SubmissionResult(plan=plan),
    )
    arguments = cli.parser().parse_args(
        [
            "submit-remediation",
            "--state-dir",
            str(tmp_path),
            "--incident-id",
            INCIDENT,
            "--disposition",
            DISPOSITION,
            "--node",
            NODE,
            "--reference",
            "CHG-2026-0919-04",
            "--plan",
        ]
    )

    assert module.run_submit_remediation_command(arguments, site=site) == 0
    request = captured[0]
    assert (request.disposition, request.node_id, request.plan_only) == (
        DISPOSITION,
        NODE,
        True,
    )


def test_the_cli_help_names_the_disposition(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["submit-remediation", "--help"])

    help_text = capsys.readouterr().out
    assert DISPOSITION in help_text
    assert "--node" in help_text
    assert "outcome" in help_text.lower()
