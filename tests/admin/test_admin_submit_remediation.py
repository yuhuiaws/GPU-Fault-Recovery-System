from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.admin import cli
from gpu_fault.admin import submit_remediation as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.models import NodeMarker, TerminalEvent
from tests.admin.conftest import TEST_OPERATOR_ARN

NOW = datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)
INCIDENT = "inc-nvlink-74"
GPU = "GPU-1111"


def _site(tmp_path: Path) -> SimpleNamespace:
    (tmp_path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "namespace": "gpu-fault-system",
            "runtime_profile": {"version": "hyperpod-v1"},
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
    )


def _inspection(**overrides) -> dict:
    value = {
        "incident": {
            "incident_id": INCIDENT,
            "cluster_id": "gpu-a",
            "node_ids": ["node-a"],
            "gpu_uuids": [GPU],
            "fencing_token": 7,
            "workflow_request_id": "wf-check",
            "state": "ACTION_PENDING",
        },
        "workflow": {
            "request_id": "wf-check",
            "incident_id": INCIDENT,
            "fencing_token": 7,
            "status": "RUNNING",
            "official_steps": [
                {"operation": "CHECK_MECHANICALS", "node_ids": ["node-a"]},
                {"operation": "RESTORE_SCHEDULING", "node_ids": ["node-a"]},
                {"operation": "RESTART_WORKLOAD", "node_ids": ["node-a"]},
            ],
            "completed_step_indexes": [],
            "step_executions": [
                {
                    "step_index": 0,
                    "operation": "CHECK_MECHANICALS",
                    "status": "WAITING",
                    "phase": "official",
                    "details": {
                        "pending_nodes": ["node-a"],
                        "required_annotation_value": f"{INCIDENT}:7",
                    },
                    "updated_at": "2026-09-08T05:00:00+00:00",
                }
            ],
            "execution_deadline": (NOW + timedelta(hours=20)).isoformat(),
        },
        "active_observations": [],
        "existing_markers": [],
        "acknowledgement_timeout_seconds": 86400,
    }
    value.update(overrides)
    return value


def _nodes(annotations: dict | None = None) -> dict:
    return {
        "node-a": {
            "metadata": {"name": "node-a", "annotations": dict(annotations or {})},
            "spec": {},
        }
    }


def _observation() -> dict:
    return {
        "first_observed_at": "2026-09-08T04:00:00+00:00",
        "observation": {
            "cluster_id": "gpu-a",
            "environment": "hyperpod-eks",
            "job_id": "job-7",
            "attempt_id": "attempt-7-2",
            "workload_phase": "RUNNING",
            "workload_ids": ["default/train-7"],
            "runtime_profile_version": "hyperpod-v1",
            "restart_budget": 2,
            "containers": [
                {
                    "node_id": "node-a",
                    "gpu_uuids": [GPU, "GPU-2222"],
                    "gpu_count": 2,
                    "instance_id": "i-a",
                },
                {"node_id": "node-b", "gpu_uuids": ["GPU-3333"], "gpu_count": 1},
            ],
        },
    }


def _build(
    disposition: str, inspection: dict | None = None, **kwargs
) -> module.RemediationPlan:
    values = {
        "incident_id": INCIDENT,
        "disposition": disposition,
        "profile_version": "hyperpod-v1",
        "now": NOW,
    }
    values.update(kwargs)
    return module.build_remediation_plan(inspection or _inspection(), **values)


def test_inspected_plan_writes_only_the_acknowledgement_derived_from_the_record() -> (
    None
):
    plan = _build("inspected", reference="CHG-1")

    assert plan.action is None and plan.marker is None and plan.terminal is None
    assert plan.acknowledgement_value == f"{INCIDENT}:7"
    assert plan.pending_nodes == ("node-a",)
    assert plan.next_operations == ("RESTORE_SCHEDULING", "RESTART_WORKLOAD")
    assert plan.acknowledgement_timeout_seconds == 86400


def test_hardware_plan_builds_marker_and_terminal_from_the_record_for_an_idle_node() -> (
    None
):
    plan = _build("reset-gpu", reference="CHG-2")

    marker = NodeMarker.model_validate(plan.marker)
    terminal = TerminalEvent.model_validate(plan.terminal)
    assert marker.marker_id == f"marker-operator-{INCIDENT}-reset-gpu"
    assert marker.incident_id == f"inc-operator-{INCIDENT}-reset-gpu"
    assert marker.trusted is True and marker.source == "operator-change"
    assert marker.cluster_id == "gpu-a"
    assert marker.scope.node_ids == ["node-a"] and marker.scope.gpu_uuids == [GPU]
    assert marker.recommended_action == "RESET_GPU"
    assert "CHG-2" in str(marker.raw_reason)
    assert terminal.cluster_id == "gpu-a"
    assert terminal.attempt_id == f"operator-{INCIDENT}-reset-gpu"
    assert terminal.job_id == f"operator-{INCIDENT}"
    assert terminal.workload_ids == []
    assert [item.node_id for item in terminal.allocation] == ["node-a"]
    assert terminal.allocation[0].gpu_uuids == [GPU]
    assert terminal.runtime_profile_version == "hyperpod-v1"
    assert plan.workload_source == "idle-node"
    assert plan.action_incident_id == marker.incident_id


def test_hardware_plan_takes_job_attempt_and_allocation_from_the_active_observation() -> (
    None
):
    plan = _build("reboot-node", _inspection(active_observations=[_observation()]))

    terminal = TerminalEvent.model_validate(plan.terminal)
    assert terminal.job_id == "job-7" and terminal.attempt_id == "attempt-7-2"
    assert terminal.workload_ids == ["default/train-7"]
    assert terminal.restart_budget == 2
    assert [item.node_id for item in terminal.allocation] == ["node-a", "node-b"]
    assert terminal.allocation[0].gpu_uuids == [GPU, "GPU-2222"]
    assert terminal.allocation[0].instance_id == "i-a"
    assert plan.workload_source == "attempt-observation"
    assert any("stops it" in item for item in plan.warnings), (
        "an active attempt is called out as being stopped"
    )


def test_two_active_attempts_on_the_node_are_refused() -> None:
    second = _observation()
    second["observation"]["attempt_id"] = "attempt-9"
    with pytest.raises(BootstrapError, match="more than one active attempt"):
        _build("quarantine", _inspection(active_observations=[_observation(), second]))


def test_plan_fails_closed_on_not_waiting_node_change_and_stale_generation() -> None:
    inspection = _inspection()
    inspection["workflow"]["step_executions"][0]["status"] = "SUCCEEDED"
    with pytest.raises(BootstrapError, match="not waiting on CHECK_MECHANICALS"):
        _build("inspected", inspection)

    inspection = _inspection()
    inspection["workflow"]["official_steps"][0]["node_ids"] = ["node-z"]
    with pytest.raises(BootstrapError, match="node set changed"):
        _build("inspected", inspection)

    inspection = _inspection()
    inspection["workflow"]["fencing_token"] = 6
    with pytest.raises(BootstrapError, match="stale generation"):
        _build("inspected", inspection)

    inspection = _inspection()
    inspection["workflow"]["step_executions"][0]["details"][
        "required_annotation_value"
    ] = f"{INCIDENT}:6"
    with pytest.raises(BootstrapError, match="generation moved"):
        _build("inspected", inspection)

    inspection = _inspection()
    inspection["incident"]["workflow_request_id"] = "wf-other"
    with pytest.raises(BootstrapError, match="points at workflow"):
        _build("inspected", inspection)

    inspection = _inspection()
    inspection["workflow"]["execution_deadline"] = (
        NOW - timedelta(minutes=1)
    ).isoformat()
    with pytest.raises(BootstrapError, match="has passed"):
        _build("inspected", inspection)


def test_reset_gpu_requires_the_incident_to_name_gpus_and_live_nodes_must_match() -> (
    None
):
    inspection = _inspection()
    inspection["incident"]["gpu_uuids"] = []
    with pytest.raises(BootstrapError, match="GPU UUIDs"):
        _build("reset-gpu", inspection)

    with pytest.raises(BootstrapError, match="missing from cluster"):
        _build("inspected", live_nodes={})

    with pytest.raises(BootstrapError, match="isolated by incident inc-other"):
        _build(
            "inspected", live_nodes=_nodes({"gpu-fault.io/incident-id": "inc-other"})
        )


def test_unknown_disposition_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="unknown disposition"):
        module.SubmitRemediationRequest(
            site=_site(tmp_path), incident_id=INCIDENT, disposition="repaired"
        )


def test_control_plane_script_compiles() -> None:
    compile(module.REMEDIATION_SCRIPT, "submit-remediation", "exec")


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = _site(tmp_path)
        self.inspection = _inspection()
        self.nodes = _nodes()
        self.annotations: list[list[str]] = []
        self.submissions: list[dict] = []
        self.submit_result: dict = {}
        self.clock = NOW

        def inspect(site, incident_id, *, disposition):
            return json.loads(json.dumps(self.inspection))

        def run(args, **kwargs):
            self.annotations.append(list(args))
            node = args[args.index("node") + 1]
            key, value = args[args.index("node") + 2].split("=", 1)
            self.nodes[node]["metadata"]["annotations"][key] = value
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def submit(site, plan):
            self.submissions.append(
                {
                    "expected": {
                        "fencing_token": plan.fencing_token,
                        "workflow_request_id": plan.workflow_request_id,
                        "node_ids": sorted(plan.node_ids),
                    },
                    "marker": plan.marker,
                    "terminal": plan.terminal,
                }
            )
            return json.loads(json.dumps(self.submit_result))

        monkeypatch.setattr(module, "inspect_incident", inspect)
        monkeypatch.setattr(
            module, "cluster_nodes", lambda site, cluster_id: self.nodes
        )
        monkeypatch.setattr(
            module, "gpu_kubectl_command", lambda site, target: ["kubectl"]
        )
        monkeypatch.setattr(module.subprocess, "run", run)
        monkeypatch.setattr(module, "submit_operator_action", submit)

    def decision(
        self, *, duplicate: bool = False, matched: list[str] | None = None
    ) -> dict:
        marker_id = f"marker-operator-{INCIDENT}-reset-gpu"
        return {
            "duplicate": duplicate,
            "decision": {
                "cluster_id": "gpu-a",
                "attempt_id": f"operator-{INCIDENT}-reset-gpu",
                "status": "PLANNED",
                "reason": "operator marker",
                "matched_marker_ids": [marker_id] if matched is None else matched,
                "recovery_plan_id": "plan-op",
            },
            "plan": {"plan_id": "plan-op", "workflow_request_id": "wf-op"},
            "workflow": {
                "request_id": "wf-op",
                "fencing_token": 1,
                "status": "PENDING",
                "official_steps": [
                    {"operation": "MARK_UNSCHEDULABLE"},
                    {"operation": "RESET_GPU"},
                    {"operation": "RESTORE_SCHEDULING"},
                ],
            },
        }

    def submit(self, disposition: str, **overrides) -> module.SubmissionResult:
        values = {
            "site": self.site,
            "incident_id": INCIDENT,
            "disposition": disposition,
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


def test_inspected_annotates_the_node_and_is_a_no_op_the_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    first = harness.submit("inspected")

    assert harness.annotations == [
        [
            "kubectl",
            "annotate",
            "node",
            "node-a",
            f"{ANNOTATION_MECHANICAL_INSPECTION_COMPLETE}={INCIDENT}:7",
            "--overwrite",
        ]
    ]
    assert first.no_op is False
    assert "RESTORE_SCHEDULING -> RESTART_WORKLOAD" in first.message
    assert first.submission is None, "no marker or terminal for an acknowledgement"

    second = harness.submit("inspected")

    assert len(harness.annotations) == 1, "an acknowledged node is not annotated again"
    assert second.no_op is True
    assert "already acknowledged" in second.message
    assert len(harness.evidence()) == 2


def test_hardware_disposition_acknowledges_then_submits_marker_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.submit_result = harness.decision()

    result = harness.submit("reset-gpu", reference="CHG-9")

    assert len(harness.annotations) == 1
    submission = harness.submissions[0]
    assert submission["expected"] == {
        "fencing_token": 7,
        "workflow_request_id": "wf-check",
        "node_ids": ["node-a"],
    }
    assert submission["marker"]["recommended_action"] == "RESET_GPU"
    assert submission["terminal"]["allocation"] == [
        {"node_id": "node-a", "gpu_uuids": [GPU]}
    ]
    assert result.no_op is False
    assert "wf-op" in result.message
    assert "MARK_UNSCHEDULABLE -> RESET_GPU -> RESTORE_SCHEDULING" in result.message
    evidence = json.loads(harness.evidence()[0].read_text(encoding="utf-8"))
    assert evidence["status"] == "SUBMITTED"
    assert evidence["fencing_token"] == 7
    assert (
        evidence["plan"]["marker"]["marker_id"]
        == f"marker-operator-{INCIDENT}-reset-gpu"
    )


def test_resubmitting_a_hardware_disposition_is_a_no_op_even_after_the_check_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.submit_result = harness.decision()
    harness.submit("reset-gpu")
    # The acknowledgement let the investigation workflow move on.
    harness.inspection["workflow"]["step_executions"][0]["status"] = "SUCCEEDED"
    harness.inspection["workflow"]["status"] = "SUCCEEDED"
    harness.submit_result = harness.decision(duplicate=True)

    result = harness.submit("reset-gpu")

    assert result.no_op is True
    assert result.plan.resubmission is True
    assert "already submitted" in result.message
    assert len(harness.annotations) == 1, "the acknowledgement is not rewritten"
    assert len(harness.submissions) == 2, "the duplicate check is the control plane's"


def test_a_decision_that_did_not_match_the_operator_marker_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.submit_result = harness.decision(matched=["marker-collector-xid"])

    with pytest.raises(BootstrapError, match="did not match marker"):
        harness.submit("quarantine")

    harness.submit_result = harness.decision(duplicate=True, matched=[])
    with pytest.raises(BootstrapError, match="do not fabricate a new attempt id"):
        harness.submit("quarantine")


def test_plan_only_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    result = harness.submit("reboot-node", plan_only=True)

    assert harness.annotations == [] and harness.submissions == []
    assert result.plan.action == "REBOOT_NODE"
    assert "nothing was written" in result.message
    assert harness.evidence() == []


def test_a_stale_generation_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.inspection["incident"]["fencing_token"] = 8

    with pytest.raises(BootstrapError, match="stale generation"):
        harness.submit("reset-gpu")
    assert harness.annotations == [] and harness.submissions == []


def test_run_submit_remediation_command_builds_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    captured: list[module.SubmitRemediationRequest] = []
    plan = _build("inspected")
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
            "quarantine",
            "--reference",
            "CHG-3",
            "--plan",
        ]
    )

    assert module.run_submit_remediation_command(arguments, site=site) == 0
    request = captured[0]
    assert (request.incident_id, request.disposition, request.reference) == (
        INCIDENT,
        "quarantine",
        "CHG-3",
    )
    assert request.plan_only is True


def test_cli_rejects_a_disposition_outside_the_vocabulary_and_dispatches_valid_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [
                "submit-remediation",
                "--state-dir",
                str(tmp_path),
                "--incident-id",
                INCIDENT,
                "--disposition",
                "repaired",
            ]
        )
    seen: list[tuple] = []
    monkeypatch.setattr(cli, "load_site", lambda path, *, repository_root: site)
    monkeypatch.setattr(
        cli,
        "run_submit_remediation_command",
        lambda arguments, *, site: seen.append((arguments.disposition, site)) or 0,
    )
    arguments = cli.parser().parse_args(
        [
            "submit-remediation",
            "--state-dir",
            str(tmp_path),
            "--incident-id",
            INCIDENT,
            "--disposition",
            "inspected",
        ]
    )

    assert cli.run(arguments) == 0
    assert seen == [("inspected", site)]


# --------------------------------------------------------------------------
# --disposition restore: the validated restore of a QUARANTINED incident.
# --------------------------------------------------------------------------

QUARANTINED = "inc-quarantined-9"
RESTORE_STEPS = [
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
]


def _quarantined_inspection(**overrides) -> dict:
    value = {
        "incident": {
            "incident_id": QUARANTINED,
            "event_id": "event-9",
            "event_type": "XID",
            "cluster_id": "gpu-a",
            "node_ids": ["node-a"],
            "gpu_uuids": [GPU],
            "policy_version": "610",
            "policy_source": "NVIDIA",
            "fencing_token": 3,
            "workflow_request_id": "wf-quarantine",
            "state": "QUARANTINED",
            "reasons": ["quarantined after repeated resets"],
        },
        "workflow": {
            "request_id": "wf-quarantine",
            "incident_id": QUARANTINED,
            "fencing_token": 3,
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": "MARK_UNSCHEDULABLE", "node_ids": ["node-a"]},
                {"operation": "QUARANTINE", "node_ids": ["node-a"]},
            ],
        },
        "open_workflows": [],
        "active_observations": [],
        "existing_markers": [],
        "acknowledgement_timeout_seconds": 86400,
    }
    value.update(overrides)
    return value


def _quarantined_nodes(
    *,
    taint_value: str | None = quarantine_taint_value(QUARANTINED),
    annotations: dict | None = None,
    name: str = "node-a",
) -> dict:
    taints = (
        [{"key": QUARANTINE_TAINT, "value": taint_value, "effect": "NoSchedule"}]
        if taint_value is not None
        else []
    )
    return {
        name: {
            "metadata": {"name": name, "annotations": dict(annotations or {})},
            "spec": {"unschedulable": True, "taints": taints},
        }
    }


def _restore_plan(inspection: dict | None = None, nodes: dict | None = None):
    return module.build_restore_plan(
        inspection or _quarantined_inspection(),
        incident_id=QUARANTINED,
        operator=TEST_OPERATOR_ARN,
        reference="CHG-R1",
        live_nodes=_quarantined_nodes() if nodes is None else nodes,
        now=NOW,
    )


def test_restore_plan_is_the_product_workflow_under_the_same_incident_and_token() -> (
    None
):
    plan = _restore_plan()

    assert plan.disposition == "restore"
    assert plan.incident_id == QUARANTINED and plan.cluster_id == "gpu-a"
    assert plan.fencing_token == 3 and plan.workflow_request_id == "wf-quarantine"
    assert plan.state == "QUARANTINED"
    assert plan.existing_restore_workflow_id is None
    assert [step["operation"] for step in plan.steps] == RESTORE_STEPS
    assert [step["execution_owner"] for step in plan.steps] == [
        "gpu-fault-validation-adapter",
        "gpu-fault-validation-adapter",
        "gpu-fault-validation-adapter",
        "gpu-fault-kubernetes-adapter",
    ]
    assert all(step["node_ids"] == ["node-a"] for step in plan.steps), plan.steps
    assert all(step["gpu_uuids"] == [GPU] for step in plan.steps), (
        "the incident's GPU scope is carried on every step"
    )
    assert plan.reason == f"operator restore requested by {TEST_OPERATOR_ARN} (CHG-R1)"
    assert plan.as_dict()["next_operations"] == RESTORE_STEPS


def test_restore_is_refused_unless_the_incident_is_quarantined() -> None:
    inspection = _quarantined_inspection()
    inspection["incident"]["state"] = "ESCALATED"

    with pytest.raises(BootstrapError, match="is ESCALATED, not QUARANTINED"):
        _restore_plan(inspection)


def test_restore_is_refused_while_a_workflow_of_the_incident_is_open() -> None:
    inspection = _quarantined_inspection()
    inspection["workflow"]["status"] = "RUNNING"
    with pytest.raises(
        BootstrapError, match="open workflow wf-quarantine \\(RUNNING\\)"
    ):
        _restore_plan(inspection)

    inspection = _quarantined_inspection(
        open_workflows=[{"request_id": "wf-reset-9", "status": "SAFETY_PENDING"}]
    )
    with pytest.raises(BootstrapError, match="wf-reset-9 \\(SAFETY_PENDING\\)"):
        _restore_plan(inspection)


def test_restore_is_refused_on_a_stale_generation_or_a_missing_workflow() -> None:
    inspection = _quarantined_inspection()
    inspection["workflow"]["fencing_token"] = 2
    with pytest.raises(BootstrapError, match="stale generation"):
        _restore_plan(inspection)

    inspection = _quarantined_inspection(workflow=None)
    with pytest.raises(BootstrapError, match="has no workflow"):
        _restore_plan(inspection)

    inspection = _quarantined_inspection()
    inspection["incident"]["workflow_request_id"] = "wf-elsewhere"
    with pytest.raises(BootstrapError, match="points at workflow wf-elsewhere"):
        _restore_plan(inspection)


def test_restore_needs_every_node_present_and_quarantined_by_this_incident() -> None:
    with pytest.raises(BootstrapError, match="missing from cluster gpu-a: node-a"):
        _restore_plan(nodes={})

    with pytest.raises(BootstrapError, match=f"carries no {QUARANTINE_TAINT} taint"):
        _restore_plan(nodes=_quarantined_nodes(taint_value=None))

    with pytest.raises(
        BootstrapError, match="isolated by incident inc-other, not inc-quarantined-9"
    ):
        _restore_plan(
            nodes=_quarantined_nodes(
                taint_value=quarantine_taint_value("inc-other"),
                annotations={"gpu-fault.io/incident-id": "inc-other"},
            )
        )

    with pytest.raises(
        BootstrapError, match="another incident \\(taint incident-deadbeef\\)"
    ):
        _restore_plan(nodes=_quarantined_nodes(taint_value="incident-deadbeef"))

    with pytest.raises(BootstrapError, match="isolated by incident inc-other"):
        _restore_plan(
            nodes=_quarantined_nodes(
                annotations={"gpu-fault.io/incident-id": "inc-other"}
            )
        )


def test_restore_accepts_the_raw_incident_id_taint_value_of_an_older_release() -> None:
    plan = _restore_plan(nodes=_quarantined_nodes(taint_value=QUARANTINED))

    assert plan.existing_restore_workflow_id is None


def test_restore_rerun_while_its_workflow_is_open_is_a_no_op_naming_it() -> None:
    inspection = _quarantined_inspection()
    inspection["incident"]["workflow_request_id"] = "workflow-validated-restore-abc"
    inspection["workflow"] = {
        "request_id": "workflow-validated-restore-abc",
        "incident_id": QUARANTINED,
        "fencing_token": 3,
        "status": "PENDING",
        "official_steps": [],
    }

    plan = _restore_plan(inspection)

    assert plan.existing_restore_workflow_id == "workflow-validated-restore-abc"


class RestoreHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = _site(tmp_path)
        self.inspection = _quarantined_inspection()
        self.nodes = _quarantined_nodes()
        self.submissions: list[dict] = []
        self.submit_result: dict = {
            "no_op": False,
            "workflow": {
                "request_id": "workflow-validated-restore-new",
                "fencing_token": 3,
                "status": "PENDING",
            },
            "incident": {"incident_id": QUARANTINED, "state": "ACTION_PENDING"},
        }
        self.clock = NOW

        def inspect(site, incident_id, *, disposition):
            assert disposition == "restore", disposition
            return json.loads(json.dumps(self.inspection))

        def submit(site, plan, *, operator, reference):
            self.submissions.append(
                {"plan": plan.as_dict(), "operator": operator, "reference": reference}
            )
            return json.loads(json.dumps(self.submit_result))

        monkeypatch.setattr(module, "inspect_incident", inspect)
        monkeypatch.setattr(
            module, "cluster_nodes", lambda site, cluster_id: self.nodes
        )
        monkeypatch.setattr(module, "submit_restore", submit)
        monkeypatch.setattr(
            module,
            "run_control_plane_script",
            lambda *a, **k: pytest.fail("unreachable"),
        )

    def submit(self, **overrides) -> module.SubmissionResult:
        values = {
            "site": self.site,
            "incident_id": QUARANTINED,
            "disposition": "restore",
        }
        values.update(overrides)
        return module.submit_remediation(
            module.SubmitRemediationRequest(**values), now=self.now
        )

    def now(self) -> datetime:
        self.clock += timedelta(seconds=1)
        return self.clock

    def evidence(self) -> list[Path]:
        return sorted(module.evidence_directory(self.site, QUARANTINED).glob("*.json"))


def test_restore_plan_only_prints_the_workflow_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)

    result = harness.submit(plan_only=True, reference="CHG-R2")

    assert harness.submissions == []
    assert harness.evidence() == []
    assert result.no_op is False
    assert "nothing was written" in result.message
    assert " -> ".join(RESTORE_STEPS) in result.message
    assert result.as_dict()["plan"]["steps"][3]["execution_owner"] == (
        "gpu-fault-kubernetes-adapter"
    )


def test_restore_creates_the_workflow_with_the_operator_and_records_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)

    result = harness.submit(reference="CHG-R3")

    [submission] = harness.submissions
    assert submission["operator"] == TEST_OPERATOR_ARN
    assert submission["reference"] == "CHG-R3"
    assert submission["plan"]["fencing_token"] == 3
    assert submission["plan"]["workflow_request_id"] == "wf-quarantine"
    assert result.no_op is False
    assert "workflow-validated-restore-new" in result.message
    assert "ACTION_PENDING" in result.message
    [path] = harness.evidence()
    assert path.name.startswith("restore-"), path.name
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] == "SUBMITTED"
    assert evidence["operator"] == TEST_OPERATOR_ARN
    assert evidence["fencing_token"] == 3
    assert evidence["plan"]["disposition"] == "restore"
    assert [step["operation"] for step in evidence["plan"]["steps"]] == RESTORE_STEPS
    assert evidence["submission"]["workflow"]["request_id"] == (
        "workflow-validated-restore-new"
    )


def test_restore_rerun_is_a_no_op_that_names_the_open_restore_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.inspection["incident"]["workflow_request_id"] = (
        "workflow-validated-restore-abc"
    )
    harness.inspection["workflow"] = {
        "request_id": "workflow-validated-restore-abc",
        "incident_id": QUARANTINED,
        "fencing_token": 3,
        "status": "RUNNING",
        "official_steps": [],
    }

    result = harness.submit()

    assert harness.submissions == [], "nothing is created twice"
    assert result.no_op is True
    assert "workflow-validated-restore-abc" in result.message
    [path] = harness.evidence()
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "NO_OP"


def test_restore_refusals_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.nodes = _quarantined_nodes(
        taint_value=quarantine_taint_value("inc-other"),
        annotations={"gpu-fault.io/incident-id": "inc-other"},
    )

    with pytest.raises(BootstrapError, match="isolated by incident inc-other"):
        harness.submit()

    assert harness.submissions == [] and harness.evidence() == []


def test_the_in_pod_restore_mode_creates_the_pair_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REMEDIATION_SCRIPT mode ``restore`` against an in-memory control plane:
    one transaction writes the incident and the workflow, the dispatcher is
    woken, and a rerun answers ``no_op`` with the open workflow."""

    import io
    import sys

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import IncidentState, WorkflowStatus
    from tests._builders import build_store, copy_model
    from tests.orchestration._incident_closure_support import _escalated_reset

    store = build_store()
    incident, _ = _escalated_reset(store, incident_id="inc-q", node_ids=("node-a",))
    store.save_incident(
        copy_model(incident, state=IncidentState.QUARANTINED), expected=incident
    )
    context = ApplicationContext(store=store)
    woken: list[bool] = []
    monkeypatch.setattr(context.dispatcher, "wake", lambda: woken.append(True))
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )

    def run(payload: dict) -> dict:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(compile(module.REMEDIATION_SCRIPT, "<submit-remediation>", "exec"), {})
        return json.loads(capsys.readouterr().out)

    payload = {
        "mode": "restore",
        "incident_id": "inc-q",
        "expected": {
            "fencing_token": incident.fencing_token,
            "workflow_request_id": incident.workflow_request_id,
            "state": "QUARANTINED",
            "node_ids": ["node-a"],
        },
        "operator": "ops@example",
        "reference": "CHG-1",
        "runtime_profile_version": "hyperpod-v1",
    }
    first = run(payload)

    assert first["no_op"] is False
    workflow = store.get_workflow(first["workflow"]["request_id"])
    assert workflow.request_id.startswith("workflow-validated-restore-"), (
        workflow.request_id
    )
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.fencing_token == incident.fencing_token
    assert workflow.runtime_profile_version == "hyperpod-v1"
    assert [step.operation.value for step in workflow.official_steps] == RESTORE_STEPS
    stored = store.get_incident("inc-q")
    assert stored.state is IncidentState.ACTION_PENDING
    assert stored.workflow_request_id == workflow.request_id
    assert stored.reasons[-1] == "operator restore requested by ops@example (CHG-1)"
    assert woken == [True]

    with pytest.raises(SystemExit, match="incident moved since inspection"):
        run(payload)

    again = run(
        {
            **payload,
            "expected": {
                "fencing_token": incident.fencing_token,
                "workflow_request_id": workflow.request_id,
                "state": "ACTION_PENDING",
                "node_ids": ["node-a"],
            },
        }
    )
    assert again["no_op"] is True
    assert again["workflow"]["request_id"] == workflow.request_id
    assert woken == [True], "a no-op does not wake the dispatcher again"


def test_cli_accepts_the_restore_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    captured: list[module.SubmitRemediationRequest] = []
    plan = _restore_plan()
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
            QUARANTINED,
            "--disposition",
            "restore",
            "--reference",
            "CHG-R4",
            "--plan",
        ]
    )

    assert module.run_submit_remediation_command(arguments, site=site) == 0
    assert (captured[0].disposition, captured[0].plan_only) == ("restore", True)
