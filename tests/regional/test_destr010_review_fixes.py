"""DESTR-010 runner contracts, driven through the runner instead of its source.

Each test here replaces a ``inspect.getsource`` substring check from
``test_destr010_runs_on_the_shared_fixture``: the shared fixture settings,
the focused pytest's working directory, the shared workflow wait, the
single-attempt replay, the terminal-only raw-evidence scan and the evidence
identity the case result is bound to.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    settings_from_arguments,
)

ROOT = Path(__file__).resolve().parents[2]
PROBE_IMAGE = "registry.example/probe@sha256:" + "a" * 64
NODE = "node-a"
MARKER_WORKFLOW = "wf-45"
IDEMPOTENCY_KEY = "idem-45"
AGENT_GENERATION = 3
NODE_ACTION_COMMAND_ID = f"{IDEMPOTENCY_KEY}/{NODE}/agent-{AGENT_GENERATION}"


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def _node_state() -> dict[str, Any]:
    return {
        "uid": "uid-a",
        "ready": "True",
        "boot_id": "boot-1",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }


def _store_state() -> dict[str, Any]:
    """A terminal, contract-shaped store read for the marker's XID 45 workflow."""

    return {
        "release_id": "release-a",
        "agent": {"generation": AGENT_GENERATION},
        "event": {"xid": 45, "evidence_ref": "kmsg://node-a/boot-1/7"},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "request_id": MARKER_WORKFLOW,
            "status": "SUCCEEDED",
            "official_action": "RESTART_FM",
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "RESTART_FABRIC_MANAGER",
                    "execution_owner": "gpu-fault-node-agent",
                },
            ],
            "completed_operations": ["FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER"],
        },
        "commands": [
            {
                "command_id": "cmd-1",
                "status": "SUCCEEDED",
                "idempotency_key": IDEMPOTENCY_KEY,
                "lease_token_sha256": "a" * 64,
                "lease_token_length": 27,
                "result_details": {},
            }
        ],
        "notifications": [
            {
                "notification": {
                    "notification_id": "n-a",
                    "category": "ACTION_COMPLETED",
                    "subject": "[GPU Fabric Manager restarted]",
                },
                "result": {"status": "SENT", "provider_message_id": "m-1"},
            }
        ],
    }


class _Calls:
    """Every collaborator call execute_case makes, in order."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, Any]]] = []

    def record(self, name: str, **kwargs: Any) -> None:
        self.entries.append((name, kwargs))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for entry, kwargs in self.entries if entry == name]

    def names(self) -> list[str]:
        return [entry for entry, _kwargs in self.entries]


def _destr010_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, calls: _Calls
) -> tuple[destr010.Settings, Path]:
    regional = _regional(tmp_path)
    fabric_probe_script = destr010.FABRIC_PROBE
    host_baseline = {
        "kmsg_writable": True,
        "compute_clients": [],
        "fabric_manager": {
            "ActiveState": "active",
            "MainPID": "100",
            "InvocationID": "inv-1",
        },
        "gpu_pci_bdf": "0000:01:00.0",
        "ledger": [],
        "gpu_fault_timers": [],
        "journal": {"started_count": 0},
    }
    host_after = {
        **host_baseline,
        "fabric_manager": {
            "ActiveState": "active",
            "MainPID": "200",
            "InvocationID": "inv-2",
        },
        "ledger": [{"command_id": NODE_ACTION_COMMAND_ID}],
        "journal": {"started_count": 1},
    }

    class FakeHost:
        def __init__(self, _settings: Any) -> None:
            pass

        def create(self) -> None:
            calls.record("host:create")

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            calls.record("host:" + arguments[0], arguments=arguments)
            if arguments[0] == "snapshot":
                return dict(host_baseline if len(arguments) == 1 else host_after)
            return {"command": arguments[0]}

        def cleanup(self) -> dict[str, bool]:
            calls.record("host:cleanup")
            return {}

    class FakeRegional:
        def __init__(self, live_settings: RegionalLiveSettings) -> None:
            self.settings = live_settings

        def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
            calls.record("regional:wait_for_workflow", **kwargs)
            return _store_state()

        def cpu_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
            calls.record(
                "regional:cpu_python", script=script, arguments=arguments, **kwargs
            )
            if script == fabric_probe_script:
                include_evidence = arguments[3]
                return {
                    "companion_window_seconds": 60,
                    "recent_xid_events": [],
                    "active_workflow_incidents": [],
                    "evidence": (
                        [{"record_id": "evidence-a"}] if include_evidence else []
                    ),
                }
            raise AssertionError(f"unexpected cpu_python script: {script[:40]}")

        def executor_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
            calls.record(
                "regional:executor_python", script=script, arguments=arguments, **kwargs
            )
            return {"status": "SUCCEEDED"}

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            calls.record("regional:store_snapshot", **kwargs)
            return _store_state()

        def node_snapshot(self, node: str) -> dict[str, Any]:
            calls.record("regional:node_snapshot", node=node)
            return _node_state()

        def business_workloads(self, node: str) -> list[dict[str, str]]:
            calls.record("regional:business_workloads", node=node)
            return []

        def provider_events(self, *_args: Any) -> list[dict[str, str]]:
            calls.record("regional:provider_events")
            return []

        @staticmethod
        def provider_events_provisional(_ended_at: datetime) -> bool:
            return True

    preflight = {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "node": _node_state(),
        "business_workloads": [],
        "store": {
            "agent": {"generation": AGENT_GENERATION},
            "profile": {"profile_version": "profile-a"},
        },
        "fabric": {"companion_window_seconds": 60},
        "focused_tests": {"passed": True, "focused_tests_reused": True},
    }

    def read_only_preflight(
        _settings: Any, _case_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        calls.record("read_only_preflight", **kwargs)
        return preflight

    monkeypatch.setattr(destr010, "read_only_preflight", read_only_preflight)
    monkeypatch.setattr(destr010, "RegionalLiveFixture", FakeRegional)
    monkeypatch.setattr(destr010, "HostProbeFixture", FakeHost)
    settings = destr010.Settings(
        regional=regional.settings, node=NODE, host_probe_image=PROBE_IMAGE
    )
    run_dir = tmp_path / "run-DESTR010"
    case_dir = run_dir / "cases" / destr010.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps(
            {"details": {"preflight_identity": destr010.preflight_identity(preflight)}}
        ),
        encoding="utf-8",
    )
    return settings, run_dir


def _execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Calls, int, dict[str, Any]]:
    calls = _Calls()
    settings, run_dir = _destr010_fakes(tmp_path, monkeypatch, calls=calls)
    exit_code = destr010.execute_case(
        settings, run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    evidence = json.loads(
        (run_dir / "cases" / destr010.CASE_ID / f"{destr010.CASE_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    return calls, exit_code, evidence


# --- configure(): the shared fixture settings -----------------------------------


def test_configure_builds_the_shared_regional_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "GPU_FAULT_TARGET_NODE",
        "GPU_FAULT_HOST_PROBE_IMAGE",
        "GPU_FAULT_CLUSTER_ID",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    regional = _regional(tmp_path)
    arguments = destr010.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--cpu-kubeconfig",
            str(regional.settings.cpu_kubeconfig),
            "--gpu-kubeconfig",
            str(regional.settings.gpu_kubeconfig),
            "--gpu-context",
            "gpu-context",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--node",
            NODE,
            "--host-probe-image",
            PROBE_IMAGE,
        ]
    )

    settings = destr010.configure(arguments)

    assert settings.regional == settings_from_arguments(arguments)
    assert settings.regional == regional.settings
    assert settings.node == NODE
    assert settings.host_probe_image == PROBE_IMAGE
    # Without kubeconfigs the shared settings parser refuses first: configure
    # has no fixture-building path of its own.
    monkeypatch.delenv("GPU_FAULT_CONTROL_KUBECONFIG", raising=False)
    monkeypatch.delenv("CPU_KUBECONFIG", raising=False)
    with pytest.raises(RegionalFixtureError, match="CPU kubeconfig is required"):
        destr010.configure(destr010.parser().parse_args(["--run-dir", str(tmp_path)]))


# --- focused_tests(): runs from the repo root ------------------------------------


def test_focused_tests_run_pytest_from_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        runs.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(destr010.RegionalLiveFixture, "run", staticmethod(run))

    result = destr010.focused_tests(tmp_path)

    assert len(runs) == 1, runs
    assert runs[0]["cwd"] == ROOT, "the focused pytest paths are repo-relative"
    assert runs[0]["check"] is False
    assert runs[0]["command"][1:4] == ["-m", "pytest", "-q"]
    assert all(
        item.startswith(("tests/node_agent/", "tests/policy/"))
        for item in runs[0]["command"][4:]
    ), runs[0]["command"]
    assert result == {"passed": True, "returncode": 0, "command": runs[0]["command"]}
    assert (tmp_path / "focused-tests.log").read_text(encoding="utf-8") == "ok\n"


def test_focused_tests_reuse_the_plan_result_without_running_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = {"passed": True, "returncode": 0, "command": ["pytest"]}
    monkeypatch.setattr(destr010, "reusable_focused_tests", lambda _path: recorded)
    monkeypatch.setattr(
        destr010.RegionalLiveFixture,
        "run",
        staticmethod(lambda *_a, **_k: pytest.fail("focused pytest must not re-run")),
    )

    result = destr010.focused_tests(tmp_path, reuse=True)

    assert result == {**recorded, "focused_tests_reused": True}


# --- execute_case(): the shared wait, the terminal scan, the bound evidence -----


def test_execute_case_waits_on_the_shared_fixture_and_scans_evidence_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, exit_code, evidence = _execute(tmp_path, monkeypatch)

    assert exit_code == 0, evidence
    assert evidence["verdict"] == "PASS", evidence
    waits = calls.named("regional:wait_for_workflow")
    assert len(waits) == 1, "the workflow wait is the shared fixture's, once"
    assert waits[0]["node"] == NODE
    assert waits[0]["marker"] == evidence["marker"]
    assert waits[0]["timeout_seconds"] == 60 + 300
    assert (
        waits[0]["case_dir"] == tmp_path / "run-DESTR010" / "cases" / destr010.CASE_ID
    )
    # The raw NVIDIA evidence scan runs once, at the terminal read, after the
    # wait returned; the wait loop itself never pages through evidence.
    probes = [
        item
        for item in calls.named("regional:cpu_python")
        if item["script"] == destr010.FABRIC_PROBE
    ]
    assert [item["arguments"] for item in probes] == [
        ("cluster-a", NODE, evidence["marker"], "1")
    ]
    names = calls.names()
    assert names.index("regional:wait_for_workflow") < names.index(
        "regional:cpu_python"
    ), names
    assert "include_evidence" in destr010.FABRIC_PROBE
    # The post-replay store read takes the single queue sample and narrows the
    # commands read to the marker's workflow.
    snapshots = calls.named("regional:store_snapshot")
    assert len(snapshots) == 1, snapshots
    assert snapshots[0]["queue_attempts"] == 1
    assert snapshots[0]["workflow_request_ids"] == [MARKER_WORKFLOW]
    assert snapshots[0]["marker"] == evidence["marker"]


def test_execute_case_binds_the_evidence_to_the_preflight_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _exit_code, evidence = _execute(tmp_path, monkeypatch)

    assert calls.named("read_only_preflight") == [{"reuse_focused_tests": True}]
    assert evidence["release_id"] == "release-a"
    assert evidence["cluster_id"] == "cluster-a"
    assert evidence["focused_tests_reused"] is True
    assert evidence["provider_events_provisional"] is True
    assert evidence["provider_events"] == []
    assert evidence["node_action_command_id"] == NODE_ACTION_COMMAND_ID
    assert evidence["workflow"]["request_id"] == MARKER_WORKFLOW


def test_execute_case_replays_the_command_once_after_the_first_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _exit_code, evidence = _execute(tmp_path, monkeypatch)

    replays = calls.named("regional:executor_python")
    assert len(replays) == 1, replays
    assert replays[0]["attempts"] == 1, "a replay no-op is the fact under test"
    assert json.loads(replays[0]["arguments"][0]) == destr010.replayable_command(
        _store_state()["commands"][0]
    )
    names = calls.names()
    assert names.index("host:snapshot") < names.index("regional:executor_python")
    assert evidence["replay"] == {"status": "SUCCEEDED"}


def test_replay_command_runs_the_adapter_with_a_single_attempt() -> None:
    issued: list[dict[str, Any]] = []

    class Regional:
        def executor_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
            issued.append({"script": script, "arguments": arguments, **kwargs})
            return {"status": "SUCCEEDED"}

    command = {
        "command_id": "c",
        "lease_token_sha256": "a" * 64,
        "lease_token_length": 27,
        "result_details": {},
    }

    result = destr010.replay_command(Regional(), command)  # type: ignore[arg-type]

    assert result == {"status": "SUCCEEDED"}
    assert len(issued) == 1, issued
    assert issued[0]["attempts"] == 1
    assert issued[0]["timeout"] == 180
    assert json.loads(issued[0]["arguments"][0]) == {
        "command_id": "c",
        "result_details": {},
        "lease_token": None,
    }
