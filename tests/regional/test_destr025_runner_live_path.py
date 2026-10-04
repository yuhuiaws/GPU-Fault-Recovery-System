"""GF-REGIONAL-DESTR-025 runner: the live execution path against fakes.

``execute_case`` is driven end to end with the regional, warm-spare and host
probe fixtures replaced by recorders, the executor env window by a stub that
writes (or refuses to write) its record, and the clock by a fake. Each test
pins the verdict document the runner writes and the cleanup it owes for one
shape of run: the pass, a preflight or plan refusal, a maintenance window
that closed, a host that was never idle, an injected XID that opened no
record, a record that never reached a terminal state, and a node left
isolated that only the validated restore workflow may free.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import executor_env_window
from scripts.e2e.regional import run_destr016_preempting_reboot as preempting
from scripts.e2e.regional import run_destr025_single_node_reset_escalation as destr025
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings
from tests.regional.test_destr025_verdicts import (
    INCIDENT,
    NODE,
    REQUEST,
    pass_commands,
    pass_incident,
    pass_workflow,
)

EXECUTOR = executor_env_window.DEPLOYMENT
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-executor-role"
INVENTORY = [{"pci_bdf": "0000:59:00.0", "index": 0, "uuid": "GPU-a"}]
BARRIER_ROW = {
    "command_id": "cmd-3",
    "operation": "VERIFY_NO_GPU_CLIENTS",
    "state": "SUCCEEDED",
}


def identity(generation: int, template: str) -> dict[str, Any]:
    """A ready runtime whose executor Deployment sits at ``generation``."""

    value = ready_runtime()
    for plane in value["deployments"].values():
        for name, item in plane.items():
            item["template_sha256"] = f"t-{name}"
    executor = value["deployments"]["gpu"][EXECUTOR]
    executor.update(
        generation=generation, observed_generation=generation, template_sha256=template
    )
    return value


def node_snapshot(**changes: Any) -> dict[str, Any]:
    return {
        "uid": "node-uid",
        "boot_id": "boot-before",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
        **changes,
    }


def host_snapshot(boot_id: str, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "boot_id": boot_id,
        "kmsg_writable": True,
        "compute_clients": [],
        "quiesce_states": [],
        "gpu_inventory": deepcopy(INVENTORY),
        "ledger": ledger,
        "services": {"nvidia-fabricmanager.service": {"ActiveState": "active"}},
    }


def store_state() -> dict[str, Any]:
    return {
        "release_id": "rel-1",
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": [
                "QUIESCE_GPU_SERVICES",
                "VERIFY_NO_GPU_CLIENTS",
                "RESTORE_GPU_SERVICES",
            ],
        },
        "profile": {
            "profile_version": "v7",
            "capabilities": [
                {
                    "capability": "gpuReset",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                },
                {
                    "capability": "nodeReboot",
                    "mode": "OWN",
                    "owner": "gpu-fault-hyperpod-adapter",
                    "adapter": "regional-cluster-executor",
                },
            ],
            "warnings": [],
        },
        "queue": {"depth": 0, "fault_backlog_depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "event": None,
    }


def running_state() -> dict[str, Any]:
    workflow = pass_workflow()
    workflow["status"] = "RUNNING"
    return {"workflow": workflow, "incident": {**pass_incident(), "state": "OPEN"}}


def terminal_state() -> dict[str, Any]:
    return {
        "workflow": pass_workflow(),
        "incident": pass_incident(),
        "submission": {"submission_id": "sub-1"},
    }


class FakeProbe:
    def __init__(self, harness: LiveHarness, settings: Any) -> None:
        self.harness = harness
        self.label = "inject" if settings.case_id.endswith("-inject") else "holder"
        self.host_script = f"/run/{self.label}-probe.py"
        self.settings = settings

    def create(self) -> None:
        self.harness.calls.append(f"{self.label}.create")

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        command = arguments[0]
        self.harness.calls.append(f"{self.label}.{command}")
        if command == "snapshot":
            return deepcopy(self.harness.host_snapshots.pop(0))
        if command == "holder-status":
            return deepcopy(self.harness.holder_status)
        return {"command": command, "arguments": list(arguments[1:])}

    def cleanup(self) -> dict[str, bool]:
        self.harness.calls.append(f"{self.label}.cleanup")
        return dict(self.harness.residuals[self.label])


class FakeRegional:
    def __init__(self, harness: LiveHarness) -> None:
        self.harness = harness
        self.cpu_python_scripts: list[str] = []

    def store_snapshot(self, **arguments: Any) -> dict[str, Any]:
        if "marker" not in arguments:
            return deepcopy(self.harness.store)
        self.harness.calls.append("store.observe")
        self.harness.clock.sleep(self.harness.observe_advance)
        observations = self.harness.observations
        return deepcopy(
            observations.pop(0) if len(observations) > 1 else observations[0]
        )

    def node_snapshot(self, node: str) -> dict[str, Any]:
        snapshots = self.harness.node_snapshots
        return deepcopy(snapshots.pop(0) if len(snapshots) > 1 else snapshots[0])

    def runtime_identity(self) -> dict[str, Any]:
        identities = self.harness.identities
        return deepcopy(identities.pop(0) if len(identities) > 1 else identities[0])

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.harness.business_workloads)

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return []

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        blasts = self.harness.cpu_blasts
        return deepcopy(blasts.pop(0) if len(blasts) > 1 else blasts[0])

    def cpu_python(self, script: str, *arguments: str) -> dict[str, Any]:
        self.cpu_python_scripts.append(script)
        if script == destr025.SUCCESSORS:
            return {"successors": deepcopy(self.harness.successors)}
        if script == preempting.COMMANDS_BY_WORKFLOW:
            return {"commands": deepcopy(self.harness.commands)}
        raise AssertionError(f"unexpected CPU python script: {script[:40]!r}")

    def wait_for_workflow(self, **arguments: Any) -> dict[str, Any]:
        self.harness.calls.append("store.wait_for_workflow")
        return deepcopy(self.harness.opened_state)

    def wait_node_ready(self, node: str, **arguments: Any) -> dict[str, Any]:
        self.harness.calls.append("node.wait_ready")
        return node_snapshot(boot_id="boot-after")

    def provider_events(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [
            {
                "event_name": "BatchRebootClusterNodes",
                "session_issuer_role_name": "gpu-fault-executor-role",
            }
        ]


class FakeWarm:
    def __init__(self, harness: LiveHarness) -> None:
        self.harness = harness

    def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
        self.harness.calls.append(f"warm.incident_idle:{incident_id}")
        return {"incident_id": incident_id, "idle": True}

    def wait_agent_active(self, node: str) -> dict[str, Any]:
        self.harness.calls.append("warm.agent_active")
        return {"node": node}

    def create_restore_workflow(self, **arguments: Any) -> dict[str, Any]:
        self.harness.calls.append("warm.create_restore")
        self.harness.restore_requests.append(dict(arguments))
        return {"workflow_request_id": "workflow-restore-1"}

    def wait_workflow_id(self, request_id: str) -> dict[str, Any]:
        return {"request_id": request_id, "status": self.harness.restore_status}


class LiveHarness:
    """Every knob one ``execute_case`` run reads, with the pass shape as default."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = destr025.Settings(
            regional=regional_settings(tmp_path),
            node=NODE,
            host_probe_image="example.test/probe",
            hyperpod_cluster="fake-hyperpod",
            executor_role_arn=ROLE_ARN,
            pci_bdf="",
            device="",
            verify_max_attempts=6,
            predecessor_path=tmp_path / "predecessor.json",
        )
        self.run_dir = tmp_path / "run-20260912T120000Z"
        self.case_dir = self.run_dir / "cases" / destr025.CASE_ID
        self.clock = Clock()
        self.calls: list[str] = []
        self.store = store_state()
        self.business_workloads: list[dict[str, Any]] = []
        self.node_snapshots = [node_snapshot(), node_snapshot(boot_id="boot-after")]
        self.identities = [
            identity(1, "t-base"),
            identity(2, "t-open"),
            identity(3, "t-base"),
        ]
        self.cpu_blasts: list[dict[str, Any]] = [{"cpu": ["api-a"]}]
        self.host_snapshots = [
            host_snapshot("boot-before", []),
            host_snapshot("boot-after", [BARRIER_ROW]),
        ]
        self.holder_status = {
            "matched_row": {"command_id": "cmd-3"},
            "hold_started_at": "2026-09-12T12:03:10+00:00",
            "holder_error": None,
            "unit_state": {"ActiveState": "inactive"},
        }
        self.opened_state = running_state()
        self.observations = [terminal_state()]
        self.observe_advance = 0.0
        self.successors: list[dict[str, Any]] = []
        self.commands = pass_commands()
        self.residuals = {"holder": {"pod": False}, "inject": {"pod": False}}
        self.reachability = {
            "exec_allowed": True,
            "assume_disarmed": False,
            "reason": "node Ready",
            "wait_error": "",
        }
        self.restore_status = "SUCCEEDED"
        self.restore_requests: list[dict[str, Any]] = []
        self.refuse_window_open = False
        self.window_reports: list[str] = []
        self.regional = FakeRegional(self)
        self.warm = FakeWarm(self)
        monkeypatch.setattr(destr025, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(destr025, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(destr025, "HostProbeFixture", lambda s: FakeProbe(self, s))
        monkeypatch.setattr(destr025, "focused_tests", lambda _d: {"passed": True})
        monkeypatch.setattr(
            destr025, "predecessor_evidence", lambda *_a: {"valid": True}
        )
        monkeypatch.setattr(
            destr025,
            "control_env",
            lambda _r: {
                "node_lifetime_seconds": 3600,
                "poll_interval_seconds": 5.0,
                "max_rungs": 2,
            },
        )
        monkeypatch.setattr(
            destr025,
            "executor_env_snapshot",
            lambda _r: [{"pod": "x", "allow_reboot": "true", "allow_replace": "false"}],
        )
        monkeypatch.setattr(
            destr025,
            "budget_headroom",
            lambda *_a: {
                "readable": True,
                "scopes": {"node": {"active": 0, "limit": 1}},
            },
        )
        monkeypatch.setattr(destr025, "node_open_incidents", lambda *_a: [])
        monkeypatch.setattr(destr025, "host_reachability", self.reach)
        monkeypatch.setattr(
            destr025,
            "env_window",
            SimpleNamespace(
                Settings=executor_env_window.Settings,
                DEPLOYMENT=EXECUTOR,
                survey=lambda _r: {"replicas": 1},
                open_window=self.open_window,
                close_window=self.close_window,
            ),
        )
        monkeypatch.setattr(destr025, "time", self.clock)
        monkeypatch.setattr(destr025, "datetime", self.clock)

    def reach(self, regional: Any, **arguments: Any) -> dict[str, Any]:
        self.calls.append("reachability")
        return dict(self.reachability)

    def open_window(self, settings: Any, *_arguments: Any) -> dict[str, Any]:
        self.calls.append("window.open")
        if self.refuse_window_open:
            raise RegionalFixtureError("another executor env window is in effect")
        settings.baseline.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
        return {"opened": True}

    def close_window(self, settings: Any, *_arguments: Any) -> dict[str, Any]:
        self.calls.append("window.close")
        return {"closed": settings.baseline.is_file()}

    def write_plan(self, *, digest: str | None = None) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        preflight = {
            "release_id": self.store["release_id"],
            "node": self.node_snapshots[0],
            "store": self.store,
            "runtime_identity": self.identities[0],
        }
        current = destr025.identity_digest(destr025.plan_identity(preflight, node=NODE))
        (self.case_dir / "plan.json").write_text(
            json.dumps({"details": {"preflight_identity_digest": digest or current}}),
            encoding="utf-8",
        )

    def execute(
        self, *, window_end: datetime | None = None
    ) -> tuple[int, dict[str, Any]]:
        end = window_end if window_end is not None else NOW + timedelta(hours=2)
        code = destr025.execute_case(self.settings, self.run_dir, 1, end)
        document = json.loads(
            (self.case_dir / f"{destr025.CASE_ID}.json").read_text(encoding="utf-8")
        )
        return code, document

    def evidence(self, name: str) -> dict[str, Any]:
        value = json.loads((self.case_dir / name).read_text(encoding="utf-8"))
        assert isinstance(value, dict), f"{name} is not a JSON object"
        return value


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LiveHarness:
    value = LiveHarness(tmp_path, monkeypatch)
    value.write_plan()
    return value


def test_the_pass_shape_records_the_ladder_and_cleans_up_in_order(
    harness: LiveHarness,
) -> None:
    code, result = harness.execute()
    assert code == 0, result
    assert result["verdict"] == "PASS", result
    assert result["errors"] == [], result["errors"]
    assert result["workflow_request_id"] == REQUEST, result
    assert result["incident_id"] == INCIDENT, result
    assert result["marker"].startswith("destr025-"), result["marker"]
    assert result["submission"] == {"submission_id": "sub-1"}, result
    assert set(result["components"]) == {
        "preflight_identity",
        "workflow",
        "incident",
        "hosts",
    }, result["components"]
    cleanup = result["cleanup"]
    assert cleanup["errors"] == [], cleanup
    assert cleanup["holder_disarm"]["disarmed"] == "by-probe", cleanup
    assert cleanup["restore_isolated_node"] == {"isolated": False}, cleanup
    assert cleanup["env_window_close"] == {"closed": True}, cleanup
    assert cleanup["runtime_identity"] == identity(3, "t-base"), cleanup
    assert harness.calls.index("window.open") < harness.calls.index("holder.create"), (
        "the executor env window opens before any probe is created"
    )
    assert harness.calls.index("holder.disarm-holder") < harness.calls.index(
        "window.close"
    ), "the holder is disarmed before the env window closes"
    assert "holder.arm-holder" in harness.calls, harness.calls
    assert "inject.write-xid46" in harness.calls, harness.calls
    assert harness.evidence("successors.json") == {"successors": []}, (
        "the successor query result is recorded"
    )
    assert harness.evidence("step-timeline.json")["transitions"], (
        "the observed step transitions are recorded"
    )
    assert harness.evidence("node-after-boot.json")["boot_id"] == "boot-after", (
        "the first Ready snapshot after the reboot is recorded"
    )


def test_a_failed_preflight_refuses_before_anything_is_armed(
    harness: LiveHarness,
) -> None:
    harness.business_workloads = [{"name": "trainer"}]
    with pytest.raises(RegionalFixtureError, match="preflight failed: .*trainer"):
        harness.execute()
    assert harness.calls == [], "nothing runs after a refused preflight"
    assert harness.evidence("preflight.json")["errors"], (
        "the refusing preflight is still recorded"
    )


def test_a_drifted_plan_identity_refuses_execution(harness: LiveHarness) -> None:
    harness.write_plan(digest="0" * 64)
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        harness.execute()
    assert harness.calls == [], "a drifted plan arms nothing"


def test_a_closed_maintenance_window_is_recorded_and_owes_no_cleanup(
    harness: LiveHarness,
) -> None:
    harness.identities = [identity(1, "t-base")]
    code, result = harness.execute(window_end=NOW - timedelta(seconds=1))
    assert code == 1, result
    assert result["verdict"] == "FAIL", result
    assert result["error"] == (
        "RegionalFixtureError: maintenance window ended before case setup"
    ), result
    cleanup = result["cleanup"]
    assert cleanup["errors"] == [], cleanup
    assert "holder_disarm" not in cleanup, "an unarmed holder is not disarmed"
    assert "incident_idle" not in cleanup, "no incident was opened"
    assert "env_window_close" not in cleanup, "a window never opened is not closed"
    assert cleanup["runtime_identity"] == identity(1, "t-base"), cleanup
    assert harness.calls == ["holder.cleanup", "inject.cleanup"], harness.calls


def test_a_refused_env_window_open_leaves_nothing_to_close(
    harness: LiveHarness,
) -> None:
    harness.refuse_window_open = True
    # The skipped close still counts as a close in the cleanup's generation
    # arithmetic, so the identity read after cleanup must sit two writes on.
    harness.identities = [identity(1, "t-base"), identity(3, "t-base")]
    code, result = harness.execute()
    assert code == 1, result
    assert result["error"] == (
        "RegionalFixtureError: another executor env window is in effect"
    ), result
    cleanup = result["cleanup"]
    assert cleanup["errors"] == [], cleanup
    assert cleanup["env_window_close"] == {"skipped": "no window record was written"}, (
        cleanup
    )
    assert "window.close" not in harness.calls, "a record-less open is never closed"


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (
            {"compute_clients": [{"pid": 7}]},
            "the node already has NVIDIA compute clients",
        ),
        ({"kmsg_writable": False}, "/dev/kmsg is not writable from the host probe"),
        (
            {"quiesce_states": ["0000:59:00.0"]},
            "the node has a pre-existing GPU quiesce state",
        ),
    ],
)
def test_a_host_that_is_not_idle_refuses_the_injection(
    harness: LiveHarness, defect: dict[str, Any], message: str
) -> None:
    harness.host_snapshots[0].update(defect)
    harness.residuals["holder"] = {"pod": True}
    harness.identities = [identity(1, "t-base"), identity(3, "t-base")]
    code, result = harness.execute()
    assert code == 1, result
    assert result["error"] == f"RegionalFixtureError: {message}", result
    assert "holder.arm-holder" not in harness.calls, "the holder is never armed"
    assert "inject.write-xid46" not in harness.calls, "no XID is written"
    cleanup = result["cleanup"]
    assert "holder_disarm" not in cleanup, cleanup
    assert cleanup["inject_probe_cleanup"] == {"pod": False}, cleanup
    assert cleanup["errors"] == [
        "holder_probe_cleanup: RegionalFixtureError: residual resources remain: "
        "{'pod': True}"
    ], cleanup
    assert cleanup["env_window_close"] == {"closed": True}, cleanup
    assert harness.evidence("host-baseline.json")["boot_id"] == "boot-before", (
        "the refusing baseline is recorded"
    )


def test_an_injection_that_opens_no_record_assumes_a_rebooted_holder_gone(
    harness: LiveHarness,
) -> None:
    harness.opened_state = {}
    harness.identities = [identity(1, "t-base"), identity(3, "t-base")]
    harness.reachability = {
        "exec_allowed": False,
        "assume_disarmed": True,
        "reason": "the node boot id changed; transient units died with the boot",
        "wait_error": "",
    }
    code, result = harness.execute()
    assert code == 1, result
    assert result["error"] == (
        "RegionalFixtureError: the injected XID opened no workflow record"
    ), result
    cleanup = result["cleanup"]
    assert cleanup["errors"] == [], cleanup
    assert cleanup["holder_disarm"]["disarmed"] == "assumed", cleanup
    assert cleanup["holder_disarm"]["reason"].startswith("the node boot id changed"), (
        cleanup
    )
    assert "holder.disarm-holder" not in harness.calls, (
        "nothing is exec'd into a node that never answered"
    )
    assert "incident_idle" not in cleanup, "no incident id means no incident wait"


def test_a_record_that_never_settles_fails_and_restores_through_the_workflow(
    harness: LiveHarness,
) -> None:
    harness.observations = [running_state()]
    harness.observe_advance = destr025.OBSERVATION_BUDGET_SECONDS + 100
    harness.cpu_blasts = [{"cpu": ["api-a"]}, {"cpu": ["api-a", "api-b"]}]
    harness.node_snapshots = [
        node_snapshot(),
        node_snapshot(boot_id="boot-after"),
        node_snapshot(
            boot_id="boot-after",
            unschedulable=True,
            ownership_annotations={"gpu-fault.io/incident-id": "inc-other"},
        ),
    ]
    harness.reachability = {
        "exec_allowed": False,
        "assume_disarmed": False,
        "reason": "the node is not Ready and the holder may still be alive",
        "wait_error": "TimeoutError: node never Ready",
    }
    harness.restore_status = "FAILED"
    code, result = harness.execute()
    assert code == 1, result
    assert result["verdict"] == "FAIL", result
    assert "the workflow is not SUCCEEDED: RUNNING" in result["errors"], result
    assert "control-plane EKS state differs from baseline" in result["errors"], result
    assert harness.calls.count("store.observe") == 1, (
        "the observation budget ran out after one read"
    )
    cleanup = result["cleanup"]
    assert cleanup["errors"] == [
        "holder_disarm: RegionalFixtureError: the node never answered again, so the "
        "holder could not be disarmed: the node is not Ready and the holder may "
        "still be alive (TimeoutError: node never Ready)",
        f"restore_isolated_node: RegionalFixtureError: {NODE} restore workflow did "
        "not succeed: FAILED",
    ], cleanup
    assert f"warm.incident_idle:{INCIDENT}" in harness.calls, harness.calls
    assert "warm.incident_idle:inc-other" in harness.calls, (
        "the owning incident of the isolated node is waited out too"
    )
    assert harness.restore_requests == [
        {
            "incident_id": "inc-other",
            "node": NODE,
            "profile_version": "v7",
            "reason": "DESTR-025 validated cleanup",
        }
    ], harness.restore_requests


def test_an_isolated_node_owned_by_the_case_is_restored_and_identity_drift_fails(
    harness: LiveHarness,
) -> None:
    harness.node_snapshots = [
        node_snapshot(),
        node_snapshot(boot_id="boot-after"),
        node_snapshot(
            boot_id="boot-after",
            taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
        ),
    ]
    harness.identities = [
        identity(1, "t-base"),
        identity(2, "t-open"),
        identity(9, "t-x"),
    ]
    code, result = harness.execute()
    assert code == 1, result
    assert result["errors"] == [], "the case itself judged clean"
    cleanup = result["cleanup"]
    assert cleanup["restore_isolated_node"] == {
        "isolated": True,
        "owner": INCIDENT,
        "restore": "SUCCEEDED",
    }, cleanup
    assert harness.calls.count(f"warm.incident_idle:{INCIDENT}") == 1, (
        "an incident the case owns is waited out once, not twice"
    )
    assert len(cleanup["errors"]) == 1, cleanup
    assert cleanup["errors"][0].startswith(
        "runtime_identity: RegionalFixtureError: after DESTR-025 cleanup runtime "
        "identity drifted: "
    ), cleanup
    assert f"{EXECUTOR} differs from the pre-window baseline" in cleanup["errors"][0], (
        cleanup
    )
