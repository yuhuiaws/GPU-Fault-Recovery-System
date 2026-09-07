"""DESTR-014's live wiring: which fixture answers which read, and with what.

Every test here pins a defect the 2026-09-07 review found in the runner rather
than in the pure verdicts: a follow-up snapshot sent to a probe without
``event_id``, a preflight fed placeholders so one gate could never pass, a
cleanup that restored through an incident that no longer owned the node, a
CloudTrail lookup taken before CloudTrail had caught up, and two execs per
executor Pod reading the same environment twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

FAULT = "node-b"
SIBLING = "node-c"


def _settings(tmp_path: Path) -> destr014.Settings:
    for name in ("cpu.kubeconfig", "gpu.kubeconfig", "site.yaml", "job.yaml"):
        (tmp_path / name).write_text("apiVersion: v1\n", encoding="utf-8")
    return destr014.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        site_file=tmp_path / "site.yaml",
        manifest=tmp_path / "job.yaml",
        hyperpod_cluster="hp",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        fault_node=FAULT,
        fault_pci_bdf="0000:53:00",
        fault_device="/dev/nvidia0",
        sibling_node=SIBLING,
        sibling_pci_bdf="0000:64:00",
        job_id="destr014-job",
        attempt_id="destr014-job-a001",
        verify_max_attempts=6,
        managed_recovery_timeout_seconds=300,
        variant="sibling-exhausted",
        predecessor_path=tmp_path / "predecessor.json",
    )


# --------------------------------------------------------------------------- #
# 1a: the follow-up snapshot is keyed by event id, which only the warm probe has
# --------------------------------------------------------------------------- #
class _RegionalWithoutEventId:
    """The regional probe: reads by node and marker, has no ``event_id``."""

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if "event_id" in kwargs:
            raise TypeError("store_snapshot() got an unexpected keyword 'event_id'")
        return {}


class _WarmRecorder:
    def __init__(self) -> None:
        self.event_ids: list[str] = []

    def store_snapshot(self, *, event_id: str = "", **_: Any) -> dict[str, Any]:
        self.event_ids.append(event_id)
        return {
            "incident": {"incident_id": "inc-support", "node_ids": [SIBLING]},
            "workflow": {"official_steps": [{"operation": "ESCALATE_SUPPORT"}]},
        }


def _run(tmp_path: Path, **overrides: Any) -> Any:
    @dataclass
    class _Run:
        settings: destr014.Settings = field(default_factory=lambda: _settings(tmp_path))
        case_dir: Path = tmp_path
        regional: Any = None
        warm: Any = None
        incident_id: str = ""
        started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    run = _Run()
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def test_the_follow_up_snapshot_is_read_through_the_warm_spare_fixture(
    tmp_path: Path,
) -> None:
    warm = _WarmRecorder()
    run = _run(tmp_path, regional=_RegionalWithoutEventId(), warm=warm)
    state = {
        "workflow": {"request_id": "wf-dag", "status": "FAILED"},
        "incident": {"incident_id": "inc-dag", "state": "QUARANTINED"},
    }

    errors, workflow, incident = destr014.control_plane_errors(run, state)

    assert warm.event_ids == ["support-after-wf-dag"], warm.event_ids
    assert run.incident_id == "inc-dag"
    assert not any("follow-up is missing" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# 1f: one exec per executor Pod
# --------------------------------------------------------------------------- #
class _ExecCounter:
    def __init__(self) -> None:
        self.execs: list[str] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        assert plane == "gpu"
        return [{"name": "exec-0"}, {"name": "exec-1"}]

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        if arguments[0] == "exec":
            self.execs.append(arguments[1])
            script = arguments[-1]
            for name in destr014.EXECUTOR_ENV_KEYS.values():
                assert name in script, name
            return json.dumps(
                {
                    "spare_failover": "true",
                    "remote_state": "true",
                    "allow_replace": "false",
                    "allow_reboot": "true",
                    "max_rungs": "3",
                    "poll": "2.5",
                    "budget_region": "20",
                    "budget_cluster": "5",
                    "budget_node": None,
                    "budget_resource_class": "2",
                }
            )
        assert arguments[:3] == ("get", "configmap", "gpu-fault-regional-release-state")
        return json.dumps(
            {
                "data": {
                    "state.json": json.dumps(
                        {
                            "job_workflow_lifetime_seconds": 7200,
                            "multi_node_aggregation_window_seconds": 8,
                        }
                    )
                }
            }
        )


def test_executor_survey_reads_every_replica_exactly_once() -> None:
    regional = _ExecCounter()

    survey = destr014.executor_survey(regional)  # type: ignore[arg-type]

    assert regional.execs == ["exec-0", "exec-1"], regional.execs
    assert [item["pod"] for item in survey["executor_env"]] == ["exec-0", "exec-1"]
    assert survey["executor_env"][0] == {
        "pod": "exec-0",
        "spare_failover": "true",
        "remote_state": "true",
        "allow_replace": "false",
        "allow_reboot": "true",
    }
    assert survey["control_env"] == {
        "max_rungs": 3,
        "poll_interval_seconds": 2.5,
        "job_lifetime_seconds": 7200,
        "aggregation_window_seconds": 8,
    }
    assert survey["budget_env"] == {
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "20",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": "5",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": None,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": "2",
    }
    assert destr014.budget_limits(survey["budget_env"])["node"] == 1


# --------------------------------------------------------------------------- #
# 1b: the preflight is fed live inputs, not placeholders
# --------------------------------------------------------------------------- #
class _FakeRegional:
    def __init__(self) -> None:
        self.probe_nodes: list[str] = []

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return {"profile": {"profile_version": "v9"}, "release_id": "rel-1"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {"name": node, "uid": f"uid-{node}", "boot_id": f"boot-{node}"}

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"pods": 3}

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "rel-1", "cluster_id": "cluster-a"}

    def executor_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        self.probe_nodes.append(arguments[0])
        return {"configured_cluster_name": "hp", "positive": {"safe_to_submit": True}}

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return []


class _FakeWarm:
    def __init__(self, regional: Any, cluster: str) -> None:
        self.open_calls: list[dict[str, Any]] = []
        self.claims_read = 0

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {"name": node, "labels": {}, "annotations": {}, "taints": []}

    def store_snapshot(self, **_: Any) -> dict[str, Any]:
        return {"agents": [{"node_id": FAULT}, {"node_id": SIBLING}]}

    def provider_inventory(self) -> dict[str, Any]:
        return {"count": 4, "sha256": "abc", "nodes": []}

    def spare_nodes(self) -> list[str]:
        return []

    def cluster_recovery(self) -> dict[str, Any]:
        return {"status": "InService", "node_recovery": "None"}

    def open_workflows(self, *, job_id: str, nodes: list[str]) -> list[dict[str, Any]]:
        self.open_calls.append({"job_id": job_id, "nodes": list(nodes)})
        return [{"request_id": "wf-open", "status": "RUNNING"}]

    def active_budget_claims(self) -> list[dict[str, Any]]:
        self.claims_read += 1
        return [{"request_id": "wf-x", "claims": ["region", "cluster:cluster-a"]}]


def test_read_only_preflight_wires_budget_open_workflows_probe_and_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    regional = _FakeRegional()
    built: list[_FakeWarm] = []

    def build_warm(regional_arg: Any, cluster: str) -> _FakeWarm:
        warm = _FakeWarm(regional_arg, cluster)
        built.append(warm)
        return warm

    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> list[str]:
        captured.update(kwargs)
        return ["captured"]

    identity_calls: list[dict[str, Any]] = []

    def predecessor(path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        identity_calls.append({"path": path, "case_id": case_id, **identity})
        return {"valid": True}

    monkeypatch.setattr(destr014, "RegionalLiveFixture", lambda _s: regional)
    monkeypatch.setattr(destr014, "WarmSpareLiveFixture", build_warm)
    monkeypatch.setattr(destr014, "preflight_errors", capture)
    monkeypatch.setattr(destr014, "predecessor_evidence", predecessor)
    monkeypatch.setattr(
        destr014,
        "executor_survey",
        lambda _r: {
            "executor_env": [{"pod": "exec-0", "allow_reboot": "true"}],
            "control_env": {"max_rungs": 2, "poll_interval_seconds": 5.0},
            "budget_env": {"GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "20"},
        },
    )
    monkeypatch.setattr(
        destr014,
        "reboot_preflight_errors",
        lambda probe, cluster: [f"probe:{probe['configured_cluster_name']}:{cluster}"],
    )
    ran: list[Path] = []

    def run_tests(case_dir: Path) -> dict[str, Any]:
        ran.append(case_dir)
        return {"passed": True, "returncode": 0}

    monkeypatch.setattr(destr014, "focused_tests", run_tests)

    result = destr014.read_only_preflight(settings, tmp_path)

    # Nothing is a placeholder any more: each gate reads its live input.
    assert captured["reboot_probe_errors"] == ["probe:hp:hp"]
    assert regional.probe_nodes == [FAULT]
    assert captured["open_workflows"] == [
        {"request_id": "wf-open", "status": "RUNNING"}
    ]
    assert built[0].open_calls == [
        {"job_id": "destr014-job", "nodes": [FAULT, SIBLING]}
    ]
    assert built[0].claims_read == 1
    budget = captured["budget"]
    assert budget["readable"] is True
    assert budget["scopes"]["region"] == {"limit": 20, "active": 1}
    assert budget["scopes"]["cluster:cluster-a"] == {"limit": 5, "active": 1}
    assert budget["scopes"][f"node:cluster-a:{FAULT}"] == {"limit": 1, "active": 0}
    assert captured["tests"] == {"passed": True, "returncode": 0}
    assert ran == [tmp_path]
    assert result["errors"] == ["captured"]
    # The predecessor is bound to this release and cluster.
    assert identity_calls == [
        {
            "path": settings.predecessor_path,
            "case_id": destr014.PREDECESSOR_CASE_ID,
            "release_id": "rel-1",
            "cluster_id": "cluster-a",
        }
    ]
    # And the plan records what it read, so --execute can reuse the tests.
    assert result["remediation_budget"] == budget
    assert result["open_workflows"] == captured["open_workflows"]
    assert result["focused_tests"] == {"passed": True, "returncode": 0}


def test_read_only_preflight_reuses_a_recorded_focused_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(destr014, "RegionalLiveFixture", lambda _s: _FakeRegional())
    monkeypatch.setattr(destr014, "WarmSpareLiveFixture", _FakeWarm)
    monkeypatch.setattr(destr014, "preflight_errors", lambda **kwargs: [])
    monkeypatch.setattr(
        destr014, "predecessor_evidence", lambda *a, **k: {"valid": True}
    )
    monkeypatch.setattr(
        destr014,
        "executor_survey",
        lambda _r: {"executor_env": [], "control_env": {}, "budget_env": {}},
    )
    monkeypatch.setattr(destr014, "reboot_preflight_errors", lambda probe, cluster: [])
    monkeypatch.setattr(
        destr014,
        "focused_tests",
        lambda case_dir: pytest.fail("focused tests must not re-run when reusable"),
    )

    result = destr014.read_only_preflight(
        settings, tmp_path, reusable_tests={"passed": True, "returncode": 0}
    )

    assert result["focused_tests"] == {"passed": True, "returncode": 0, "reused": True}


def test_an_unreadable_budget_limit_fails_closed_with_the_reason() -> None:
    class _Warm:
        def active_budget_claims(self) -> list[dict[str, Any]]:
            return []

    budget = destr014.remediation_budget(
        _Warm(),  # type: ignore[arg-type]
        cluster_id="cluster-a",
        nodes=[FAULT, SIBLING],
        budget_env={"GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "lots"},
    )

    assert budget["readable"] is False
    assert "not an integer" in budget["error"], budget
    assert destr014.budget_headroom_errors(budget), budget


def test_plan_details_record_the_focused_tests_with_a_source_digest(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    preflight = {
        "release_id": "rel-1",
        "fault_node": {"uid": "u-b", "boot_id": "b-b"},
        "sibling_node": {"uid": "u-c", "boot_id": "b-c"},
        "store": {"profile": {"profile_version": "v9"}},
        "provider_inventory": {"sha256": "abc"},
        "predecessor": {"valid": True},
        "focused_tests": {"passed": True, "returncode": 0, "command": ["pytest"]},
    }

    details = destr014.plan_details(settings, preflight)

    assert details["focused_tests"] == preflight["focused_tests"]
    assert len(details["focused_tests_source_digest"]) == 64


# --------------------------------------------------------------------------- #
# 1d: cleanup restores through the incident that owns the node now
# --------------------------------------------------------------------------- #
class _RestoreRecorder:
    def __init__(self, owner: str | None) -> None:
        self.owner = owner
        self.waited: list[str] = []
        self.restored: list[str] = []

    def node_snapshot(self, node: str) -> dict[str, Any]:
        assert node == SIBLING
        return {"name": node, "annotations": {"gpu-fault.io/incident-id": self.owner}}

    def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
        self.waited.append(incident_id)
        return {"incident_id": incident_id}

    def create_restore_workflow(
        self, *, incident_id: str, node: str, profile_version: str, reason: str
    ) -> dict[str, Any]:
        self.restored.append(incident_id)
        return {"workflow_request_id": f"restore-{incident_id}"}

    def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
        return {"request_id": workflow_id, "status": "SUCCEEDED"}


def test_sibling_restore_goes_through_the_successor_then_the_original(
    tmp_path: Path,
) -> None:
    # After the branch exhausts, the escalation engine's support incident owns
    # the sibling's quarantine. Restoring with the DAG's incident alone left the
    # node cordoned under the successor and reported success.
    warm = _RestoreRecorder("inc-support")

    result = destr014.restore_sibling(
        warm,  # type: ignore[arg-type]
        _settings(tmp_path),
        "inc-dag",
        "v9",
    )

    assert warm.waited == ["inc-support", "inc-dag"], warm.waited
    assert warm.restored == ["inc-support", "inc-dag"], warm.restored
    assert result["successor_incident"] == "inc-support"
    assert result["restored_through"] == ["inc-support", "inc-dag"]


def test_sibling_restore_uses_the_original_once_when_it_still_owns_the_node(
    tmp_path: Path,
) -> None:
    warm = _RestoreRecorder("inc-dag")

    result = destr014.restore_sibling(
        warm,  # type: ignore[arg-type]
        _settings(tmp_path),
        "inc-dag",
        "v9",
    )

    assert warm.restored == ["inc-dag"], warm.restored
    assert result["successor_incident"] is None


def test_sibling_restore_fails_loudly_when_the_owner_restore_fails(
    tmp_path: Path,
) -> None:
    class _Failing(_RestoreRecorder):
        def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
            return {"request_id": workflow_id, "status": "FAILED"}

    with pytest.raises(destr014.RegionalFixtureError, match="inc-support"):
        destr014.restore_sibling(
            _Failing("inc-support"),  # type: ignore[arg-type]
            _settings(tmp_path),
            "inc-dag",
            "v9",
        )


# --------------------------------------------------------------------------- #
# 1e: the two reboots are polled for; the negative claim is labelled
# --------------------------------------------------------------------------- #
class _CloudTrail:
    def __init__(self, *, provisional: bool) -> None:
        self.waits: list[dict[str, Any]] = []
        self.lookups = 0
        self._provisional = provisional

    def wait_provider_events(
        self, started_at: datetime, *, event_names: set[str], expected_count: int
    ) -> list[dict[str, str]]:
        self.waits.append(
            {"event_names": event_names, "expected_count": expected_count}
        )
        return [{"event_name": "BatchRebootClusterNodes"}] * expected_count

    def provider_events(
        self, started_at: datetime, ended_at: datetime
    ) -> list[dict[str, str]]:
        self.lookups += 1
        return [
            {"event_name": "BatchRebootClusterNodes"},
            {"event_name": "BatchRebootClusterNodes"},
        ]

    def provider_events_provisional(self, ended_at: datetime) -> bool:
        return self._provisional


@pytest.mark.parametrize("provisional", [True, False])
def test_provider_evidence_polls_for_two_reboots_then_reads_the_window(
    provisional: bool,
) -> None:
    regional = _CloudTrail(provisional=provisional)

    evidence = destr014.provider_evidence(
        regional,  # type: ignore[arg-type]
        datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
    )

    assert regional.waits == [
        {"event_names": set(destr014.REBOOT_EVENTS), "expected_count": 2}
    ]
    assert regional.lookups == 1
    assert evidence["reboots_seen_by_poll"] == 2
    assert evidence["provisional"] is provisional
    assert destr014.cloudtrail_errors(evidence["events"], FAULT, SIBLING) == []
