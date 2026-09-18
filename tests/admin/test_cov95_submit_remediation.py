from __future__ import annotations

import json

import pytest

from gpu_fault.admin import submit_remediation as remediation
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.test_admin_submit_remediation import (
    INCIDENT,
    NOW,
    QUARANTINED,
    Harness,
    RestoreHarness,
)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "kind",
    [
        "missing-incident",
        "different-incident",
        "missing-workflow",
        "empty-nodes",
        "wrong-step",
    ],
)
def test_remediation_rejects_unbound_incident_before_any_write(harness, kind):
    if kind == "missing-incident":
        harness.inspection["incident"] = None
    elif kind == "different-incident":
        harness.inspection["incident"]["incident_id"] = "different"
    elif kind == "missing-workflow":
        harness.inspection["workflow"] = None
    elif kind == "empty-nodes":
        harness.inspection["incident"]["node_ids"] = []
    else:
        harness.inspection["workflow"]["official_steps"][0]["operation"] = "OTHER"
    with pytest.raises(
        BootstrapError,
        match="not returned|different incident|no workflow|no nodes|not CHECK_MECHANICALS",
    ):
        harness.submit("inspected")
    assert harness.annotations == []
    assert harness.submissions == []
    assert harness.evidence() == []


def test_planner_ignores_observations_not_owning_incident_nodes(harness):
    harness.inspection["active_observations"] = [
        {
            "observation": {
                "attempt_id": "unrelated",
                "containers": [{"node_id": "node-other"}],
            }
        }
    ]
    plan = remediation.build_remediation_plan(
        harness.inspection,
        incident_id=INCIDENT,
        disposition="reset-gpu",
        profile_version="hyperpod-v1",
        live_nodes=harness.nodes,
        now=NOW,
    )
    assert plan.workload_source == "idle-node"
    assert plan.terminal["attempt_id"] == remediation.operator_attempt_id(
        INCIDENT, "reset-gpu"
    )


def test_active_allocation_deduplicates_gpu_ids_and_ignores_missing_nodes(harness):
    harness.inspection["active_observations"] = [
        {
            "observation": {
                "attempt_id": "attempt-example",
                "job_id": "job-example",
                "containers": [
                    {
                        "node_id": "node-a",
                        "gpu_uuids": ["GPU-a"],
                        "gpu_count": 1,
                        "instance_id": "i-example",
                    },
                    {
                        "node_id": "node-a",
                        "gpu_uuids": ["GPU-a", "GPU-b"],
                        "gpu_count": 1,
                        "instance_id": "i-example",
                    },
                    {"gpu_uuids": ["GPU-unbound"], "gpu_count": 1},
                ],
            }
        }
    ]
    plan = remediation.build_remediation_plan(
        harness.inspection,
        incident_id=INCIDENT,
        disposition="reset-gpu",
        profile_version="hyperpod-v1",
        live_nodes=harness.nodes,
        now=NOW,
    )
    assert plan.terminal["allocation"] == [
        {
            "node_id": "node-a",
            "gpu_uuids": ["GPU-a", "GPU-b"],
            "gpu_count": 2,
            "instance_id": "i-example",
        }
    ]
    assert plan.workload_source == "attempt-observation"


def test_public_planner_default_clock_does_not_need_a_transport(harness):
    harness.inspection["workflow"].pop("execution_deadline")
    plan = remediation.build_remediation_plan(
        harness.inspection,
        incident_id=INCIDENT,
        disposition="inspected",
        profile_version="hyperpod-v1",
        live_nodes=harness.nodes,
    )
    assert plan.incident_id == INCIDENT
    assert harness.annotations == harness.submissions == []


def test_inspection_without_followup_steps_reports_completion_only(harness):
    harness.inspection["workflow"]["official_steps"] = harness.inspection["workflow"][
        "official_steps"
    ][:1]
    result = harness.submit("inspected")
    assert result.message == "workflow wf-check completes CHECK_MECHANICALS"
    assert len(harness.annotations) == 1
    assert harness.submissions == []


def test_matched_decision_without_workflow_is_reported_explicitly(harness):
    response = harness.decision()
    response.pop("workflow")
    response["decision"].update(status="BLOCKED", reason="example refusal")
    harness.submit_result = response
    result = harness.submit("reset-gpu")
    assert (
        result.message == "decision BLOCKED (example refusal); no workflow was compiled"
    )
    assert len(harness.submissions) == 1


def test_remediation_cannot_annotate_cluster_outside_managed_site(harness):
    harness.site.release_config["clusters"] = []
    with pytest.raises(BootstrapError, match="not in the managed site"):
        harness.submit("inspected")
    assert harness.annotations == harness.submissions == []


def test_previous_submission_skips_non_submitted_evidence(harness):
    root = remediation.evidence_directory(harness.site, INCIDENT)
    root.mkdir(parents=True)
    assert remediation.previous_submission(harness.site, INCIDENT, "inspected") is None
    (root / "inspected-001.json").write_text('{"status":"SUBMITTED","example":1}')
    (root / "inspected-002.json").write_text('{"status":"NO_OP"}')
    (root / "inspected-003.json").write_text("[]")
    assert remediation.previous_submission(harness.site, INCIDENT, "inspected") == {
        "status": "SUBMITTED",
        "example": 1,
    }


@pytest.mark.parametrize("kind", ["missing", "different", "empty-nodes"])
def test_restore_rejects_missing_incident_binding_before_submission(
    tmp_path, monkeypatch, kind
):
    harness = RestoreHarness(tmp_path, monkeypatch)
    if kind == "missing":
        harness.inspection["incident"] = None
    elif kind == "different":
        harness.inspection["incident"]["incident_id"] = "different"
    else:
        harness.inspection["incident"]["node_ids"] = []
    with pytest.raises(
        BootstrapError, match="not returned|different incident|no nodes"
    ):
        harness.submit()
    assert harness.submissions == []
    assert harness.evidence() == []


def test_restore_tolerates_unrelated_taints_but_requires_owned_quarantine(
    tmp_path, monkeypatch
):
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.nodes["node-a"]["spec"]["taints"].insert(
        0, {"key": "example/unrelated", "value": "other"}
    )
    plan = remediation.build_restore_plan(
        harness.inspection,
        incident_id=QUARANTINED,
        operator="example-operator",
        reference="CHG-EXAMPLE",
        live_nodes=harness.nodes,
        now=NOW,
    )
    assert plan.incident_id == QUARANTINED
    assert plan.existing_restore_workflow_id is None
    assert harness.submissions == []


def test_restore_uses_idempotent_result_returned_by_control_plane(
    tmp_path, monkeypatch
):
    harness = RestoreHarness(tmp_path, monkeypatch)
    harness.submit_result["no_op"] = True
    result = harness.submit()
    assert result.no_op is True
    assert "already open" in result.message
    assert len(harness.submissions) == 1
    assert json.loads(harness.evidence()[0].read_text())["status"] == "NO_OP"


def test_public_control_plane_payloads_are_bound_to_the_plan(harness, monkeypatch):
    calls = []

    def command(site, payload, *, script):
        calls.append((site, payload, script))
        return {"example": "response"}

    plan = remediation.build_remediation_plan(
        harness.inspection,
        incident_id=INCIDENT,
        disposition="reset-gpu",
        profile_version="hyperpod-v1",
        live_nodes=harness.nodes,
        now=NOW,
    )
    # The existing harness replaces these entrypoints; isolate their real implementations.
    import sys
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("cov95_remediation_payloads", remediation.__file__)
    assert spec is not None and spec.loader is not None, (
        "could not load isolated remediation module"
    )
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_control_plane_script", command)
    assert module.inspect_incident(harness.site, INCIDENT, disposition="reset-gpu") == {
        "example": "response"
    }
    assert module.submit_operator_action(harness.site, plan) == {"example": "response"}
    assert calls[-1][1]["expected"] == {
        "fencing_token": plan.fencing_token,
        "workflow_request_id": plan.workflow_request_id,
        "node_ids": list(plan.node_ids),
    }
    assert calls[-1][1]["marker"] == plan.marker
