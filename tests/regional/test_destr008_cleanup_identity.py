from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.models import WorkflowRequest, WorkflowStatus
from scripts.e2e.regional import destr008_cleanup_identity as identity
from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture
from tests.regional._cov95_destr_warm import WarmHarness

ROOT = "incident-root"
OWNER = "inc-support-after-workflow-root"


def history() -> tuple[Any, dict[str, Any], dict[str, Any], list[str]]:
    root: dict[str, Any] = {
        "incident_id": ROOT,
        "cluster_id": "cluster-a",
        "node_ids": ["node-a"],
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "event_id": "original-synthetic-event",
    }
    records = {
        ROOT: root,
        OWNER: {
            **copy.deepcopy(root),
            "incident_id": OWNER,
            "event_id": "support-after-workflow-root",
        },
    }
    workflows = {
        "workflow-root": {
            "request_id": "workflow-root",
            "runtime_profile_version": "profile-v1",
            "incident_id": ROOT,
            "status": "FAILED",
        }
    }
    calls: list[str] = []

    def read(key: str) -> Any:
        calls.append(key)
        return copy.deepcopy(records[key])

    def workflow(key: str, *, timeout_seconds: int) -> Any:
        assert timeout_seconds == 1, "ancestry reads must not wait for active work"
        calls.append(key)
        return copy.deepcopy(workflows[key])

    return (
        SimpleNamespace(incident_by_id=read, wait_workflow_id=workflow),
        records,
        workflows,
        calls,
    )


def require(warm: Any, owner: str = OWNER) -> None:
    identity.require_cleanup_family(
        warm,
        incident_id=ROOT,
        owner=owner,
        cluster_id="cluster-a",
        node="node-a",
        profile_version="profile-v1",
    )


@pytest.mark.parametrize("owner", ["", ROOT, OWNER])
def test_root_and_real_escalation_share_the_original_identity(owner: str) -> None:
    warm, _, _, calls = history()
    require(warm, owner)
    assert calls[0] == ROOT
    assert ("workflow-root" in calls) is (owner == OWNER)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("incident_id", "other"),
        ("cluster_id", "other"),
        ("node_ids", ["node-a", "node-b"]),
        ("job_id", ""),
        ("job_id", None),
        ("attempt_id", ""),
        ("attempt_id", False),
    ],
)
def test_unbound_root_never_authorizes_a_cleanup(field: str, value: Any) -> None:
    warm, records, _, calls = history()
    records[ROOT][field] = value
    with pytest.raises(RegionalFixtureError, match="root incident"):
        require(warm)
    assert calls == [ROOT]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("incident_id", "other"),
        ("cluster_id", "other"),
        ("node_ids", ["node-b"]),
        ("job_id", "other"),
        ("attempt_id", "other"),
        ("event_id", None),
        ("event_id", "arbitrary"),
    ],
)
def test_similar_quarantine_owner_is_not_an_escalation(field: str, value: Any) -> None:
    warm, records, _, calls = history()
    records[OWNER][field] = value
    with pytest.raises(RegionalFixtureError, match="another run"):
        require(warm)
    assert "workflow-root" not in calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "other"),
        ("runtime_profile_version", "other"),
        ("status", "RUNNING"),
        ("status", "SUCCEEDED"),
        ("incident_id", ""),
        ("incident_id", None),
    ],
)
def test_parent_workflow_must_be_the_actual_failed_source(
    field: str, value: Any
) -> None:
    warm, _, workflows, _ = history()
    workflows["workflow-root"][field] = value
    with pytest.raises(RegionalFixtureError, match="source"):
        require(warm)


def test_unregistered_escalation_prefix_does_not_confer_ownership() -> None:
    warm, records, _, _ = history()
    unknown = "inc-other-after-workflow-root"
    records[unknown] = {
        **records[OWNER],
        "incident_id": unknown,
        "event_id": unknown.removeprefix("inc-"),
    }
    with pytest.raises(RegionalFixtureError, match="not an escalation"):
        require(warm, unknown)


def test_actual_workflow_contract_binds_cluster_through_its_incident() -> None:
    warm, _, workflows, _ = history()
    workflow = WorkflowRequest(
        request_id="workflow-root",
        incident_id=ROOT,
        runtime_profile_version="profile-v1",
        fencing_token=1,
        status=WorkflowStatus.FAILED,
        official_steps=[],
    )
    workflows["workflow-root"] = workflow.model_dump(mode="json")
    require(warm)
    assert "cluster_id" not in workflows["workflow-root"], (
        "workflow identity must use its actual incident relationship"
    )


@pytest.mark.parametrize("cycle", [True, False])
def test_ancestry_never_walks_an_unbounded_or_cyclic_history(cycle: bool) -> None:
    warm, records, workflows, calls = history()
    for index in range(10):
        owner = f"inc-support-after-workflow-{index}"
        records[owner] = {
            **records[ROOT],
            "incident_id": owner,
            "event_id": owner.removeprefix("inc-"),
        }
        workflows[f"workflow-{index}"] = {
            **workflows["workflow-root"],
            "request_id": f"workflow-{index}",
            "incident_id": f"inc-support-after-workflow-{0 if cycle else index + 1}",
        }
    with pytest.raises(RegionalFixtureError, match="cyclic|bound"):
        require(warm, "inc-support-after-workflow-0")
    assert len(calls) <= 17, "the drill must not scan unrelated history"


@pytest.mark.parametrize("target", ["root", "owner", "workflow"])
def test_malformed_api_response_is_not_cleanup_authority(target: str) -> None:
    warm, records, workflows, _ = history()
    if target == "workflow":
        workflows["workflow-root"] = []
    else:
        records[ROOT if target == "root" else OWNER] = []
    with pytest.raises(RegionalFixtureError):
        require(warm)


def test_runner_refuses_foreign_owner_before_any_cleanup_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(case, tmp_path, monkeypatch)
    h.nodes["node-a"]["annotations"]["gpu-fault.io/incident-id"] = "foreign-incident"
    report = case.restore_fault_node(
        cast(WarmSpareLiveFixture, h.warm),
        settings=h.settings,
        incident_id="original-incident",
        profile_version="profile-v1",
    )
    assert report["errors"] and "ownership" in report["errors"][0], report
    names = {name for name, _ in h.calls}
    assert not names & {"spares.release", "agent.reactivate", "restore.create"}, h.calls
