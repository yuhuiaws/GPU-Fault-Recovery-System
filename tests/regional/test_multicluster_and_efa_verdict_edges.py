"""E2E-002 and COLLECT-017 verdict edges that the live runs never exercised.

Cross-cluster notification and command scoping, the zero-budget denial
notification binding, the injection skew guard, and the EFA plugin case's
admission refusals (no bound function, too few nodes, a node that is not
Ready or already owned).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collect017_efa_plugin as c017
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e002
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional import _cov95_collect_training as training_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401

training_case = training_support.training_case

T0 = datetime(2030, 6, 1, 12, 0, tzinfo=timezone.utc)
FAR = datetime(2099, 1, 1, tzinfo=timezone.utc)


class ScriptedClock:
    """``datetime`` stand-in whose ``now`` hands out scripted instants in order."""

    def __init__(self, *instants: datetime) -> None:
        self.remaining = list(instants)
        self.lock = threading.Lock()

    def now(self, tz: Any = None) -> datetime:
        with self.lock:
            return self.remaining.pop(0)


def notification(
    cluster: str, notification_id: str, incident_id: str
) -> dict[str, Any]:
    return {
        "notification": {
            "notification_id": notification_id,
            "incident_id": incident_id,
            "cluster_name": cluster,
        },
        "result": {"status": "SENT", "provider_message_id": "m-" + notification_id},
    }


def test_notification_errors_catch_foreign_incident_scope_and_shared_notifications() -> (
    None
):
    states = [
        {
            "incident": {"incident_id": "inc-a", "cluster_id": "cluster-b"},
            "notifications": [notification("cluster-a", "shared", "inc-a")],
        },
        {
            "incident": {"incident_id": "inc-b", "cluster_id": "cluster-b"},
            "notifications": [notification("cluster-b", "shared", "inc-b")],
        },
    ]
    registrations = [
        {"cluster_id": "cluster-a", "hyperpod_cluster_name": "hp-a"},
        {"cluster_id": "cluster-b", "hyperpod_cluster_name": "hp-b"},
    ]
    errors = e2e002.notification_errors(
        states, ["cluster-a", "cluster-b"], registrations
    )
    assert "cluster-a: incident is not scoped to the cluster" in errors
    assert "a notification appears under both clusters" in errors
    assert "cluster-b: incident is not scoped to the cluster" not in errors


def test_command_scope_errors_name_a_command_that_crossed_clusters() -> None:
    state = {
        "incident": {"incident_id": "inc-a", "cluster_id": "cluster-a"},
        "workflow": {"request_id": "wf-a", "incident_id": "inc-a"},
        "commands": [
            {
                "command_id": "cmd-1",
                "cluster_id": "cluster-b",
                "workflow_request_id": "wf-a",
                "incident_id": "inc-a",
                "last_lease_owner": "executor-a",
            }
        ],
    }
    errors = e2e002.command_scope_errors(
        [state], ["cluster-a"], {"cluster-a": ["executor-a"]}
    )
    assert errors == ["remote command crossed cluster scope"]


def test_budget_denial_requires_the_named_notification_to_be_delivered() -> None:
    state = {
        "workflow": {
            "status": "FAILED",
            "step_executions": [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "FAILED",
                    "details": {
                        "reason": "RESTART_BUDGET_EXHAUSTED",
                        "restart_budget": 0,
                        "restart_count": 0,
                        "notification_id": "n-budget",
                    },
                }
            ],
        },
        "restart_budget": {"budget": 0},
        "commands": [],
        "notifications": [],
    }
    errors = e2e002.budget_denial_errors(
        state,
        cluster_id="cluster-a",
        job_id="job",
        pods=[{"phase": "Failed", "uid": "pod-1"}],
        source_uids={"pod-1"},
    )
    assert errors == ["A budget-denial notification is not causally delivered"]


def test_focused_tests_reuse_the_plan_result_instead_of_running_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("a reusable plan result must not rerun pytest")

    monkeypatch.setattr(e2e002, "RegionalLiveFixture", SimpleNamespace(run=forbidden))
    monkeypatch.setattr(
        e2e002,
        "reusable_focused_tests",
        lambda path: {"passed": True, "returncode": 0}
        if path.name == "plan.json"
        else None,
    )
    assert e2e002.focused_tests(tmp_path, reuse=True) == {
        "passed": True,
        "returncode": 0,
        "focused_tests_reused": True,
    }
    assert not (tmp_path / "focused-tests.log").exists(), "reuse writes no log"


def fixtures_and_payloads() -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    posted: list[str] = []
    fixtures = [
        SimpleNamespace(
            post_xid_event=lambda payload: posted.append(payload["record_id"])
            or {"record_id": payload["record_id"]}
        )
        for _ in range(2)
    ]
    payloads = [
        {"node_id": "node-a", "record_id": "rec-a"},
        {"node_id": "node-b", "record_id": "rec-b"},
    ]
    return fixtures, payloads, posted


def test_injection_after_the_window_closes_is_a_recorded_receipt_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        e2e002, "datetime", ScriptedClock(T0, T0 + timedelta(minutes=10))
    )
    fixtures, payloads, posted = fixtures_and_payloads()
    contexts: list[dict[str, Any] | None] = [None, None]
    result: dict[str, Any] = {"errors": []}
    injections, starts = e2e002.inject_cluster_faults(
        fixtures,
        payloads,
        contexts,
        maintenance_window_end=T0 + timedelta(minutes=1),
        result=result,
    )
    assert len(posted) == 1, "the late injection must never reach its cluster"
    assert len(starts) == 1
    assert [item for item in injections if "error_type" in item] == [
        {"error_type": "RegionalFixtureError"}
    ]
    assert len(result["errors"]) == 1
    assert result["errors"][0].endswith("injection receipt failed"), result
    assert sum(context is None for context in contexts) == 1


def test_injections_more_than_a_minute_apart_fail_the_concurrency_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        e2e002, "datetime", ScriptedClock(T0, T0 + timedelta(seconds=61))
    )
    fixtures, payloads, posted = fixtures_and_payloads()
    contexts: list[dict[str, Any] | None] = [None, None]
    result: dict[str, Any] = {"errors": []}
    injections, starts = e2e002.inject_cluster_faults(
        fixtures, payloads, contexts, maintenance_window_end=FAR, result=result
    )
    assert sorted(posted) == ["rec-a", "rec-b"]
    assert result["errors"] == ["concurrent injections exceeded 60 seconds"]
    assert sorted(starts.values()) == [T0, T0 + timedelta(seconds=61)]
    assert [item["record_id"] for item in injections] == ["rec-a", "rec-b"]


def c017_settings(tmp_path: Path) -> c017.Settings:
    return c017.Settings(
        regional=None,  # type: ignore[arg-type]
        node="node-a",
        site_file=tmp_path / "site.yaml",
        host_probe_image="img@sha256:" + "a" * 64,
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_efa_unbind_verdict_names_a_remediation_step_that_did_not_succeed() -> None:
    bundle = {
        "workflow": {
            "status": "SUCCEEDED",
            "official_action": "REMEDIATE_EFA_DRIVER",
            "official_steps": [
                {"operation": name} for name in c017.EFA_REMEDIATION_STEPS
            ],
            "step_executions": [
                {
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "status": "FAILED",
                    "adapter_operation_id": "remote/cmd-1",
                }
            ],
        },
        "incident": {
            "effective_action": "REMEDIATE_EFA_DRIVER",
            "reasons": ["driver is not bound"],
            "state": "RECOVERED",
        },
        "remote_commands": [],
    }
    inventory = {
        "discovered_count": 2,
        "active_count": 2,
        "devices": [
            {"pci_bdf": "0000:3a:00.0", "driver": "efa"},
            {"pci_bdf": "0000:3b:00.0", "driver": "efa"},
        ],
    }
    unbound = {
        "discovered_count": 1,
        "active_count": 1,
        "devices": [{"pci_bdf": "0000:3b:00.0", "driver": "efa"}],
    }
    errors = c017.efa_unbind_errors(
        bundle,
        node="node-a",
        bdf="0000:3a:00.0",
        baseline=inventory,
        unbound=unbound,
        recovered=inventory,
        restore={"already_bound": True, "timer_fired": False, "bound": True},
    )
    assert "REMEDIATE_EFA_DRIVER status 'FAILED' != SUCCEEDED" in errors
    assert "REMEDIATE_EFA_DRIVER was never executed" not in errors


def test_efa_unbind_refuses_a_node_without_a_bound_function(tmp_path: Path) -> None:
    collector = SimpleNamespace(
        snapshot=lambda: {
            "efa_inventory": {
                "discovered_count": 1,
                "devices": [{"pci_bdf": "0000:3a:00.0", "driver": "vfio-pci"}],
            }
        }
    )
    with pytest.raises(RegionalFixtureError, match="no bound EFA BDF"):
        c017.run_efa_unbind(c017_settings(tmp_path), SimpleNamespace(), collector, 1)


def test_training_plugin_case_requires_three_schedulable_gpu_nodes(
    tmp_path: Path,
) -> None:
    regional = SimpleNamespace(
        gpu_nodes=lambda: [
            {"name": "node-a", "ready": "True", "unschedulable": False},
            {"name": "node-b", "ready": "True", "unschedulable": False},
            {"name": "node-c", "ready": "False", "unschedulable": False},
        ]
    )
    with pytest.raises(RegionalFixtureError, match="three GPU nodes are required"):
        c017.run_training_plugin(c017_settings(tmp_path), regional, tmp_path, 1)


@pytest.mark.parametrize("training_case", [c017], indirect=True)
@pytest.mark.parametrize(
    ("problem", "fragment"),
    [
        ("not-ready", "target node is not Ready and schedulable"),
        ("owned-node", "target node has pre-existing workflow ownership"),
    ],
)
def test_preflight_rejects_a_node_that_is_not_ready_or_already_owned(
    training_case: Any, problem: str, fragment: str
) -> None:
    harness = training_case
    harness.problem = problem
    preflight = harness.module.read_only_preflight(harness.settings, harness.root)
    assert preflight["errors"] == [fragment]
    assert harness.collectors == [], "a refused preflight allocates nothing"


def test_focused_tests_run_pytest_and_keep_the_log_when_not_reusing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_keywords: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="1 failed", stderr="")

    monkeypatch.setattr(e2e002, "RegionalLiveFixture", SimpleNamespace(run=run))
    result = e2e002.focused_tests(tmp_path)
    assert result["passed"] is False
    assert result["returncode"] == 1
    assert commands == [result["command"]]
    assert (tmp_path / "focused-tests.log").read_text() == "1 failed"
