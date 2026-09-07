"""Behavioural DESTR-002 runner contracts.

These drive ``execute_case`` and ``wait_for_submission`` with spies and assert
the calls the runner makes and the evidence it writes, replacing the
source-text checks that only proved the runner spelled itself a certain way.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

PROBE_IMAGE = "registry.example/probe@sha256:" + "a" * 64
NODE = "node-a"
EXECUTOR_ROLE_ARN = "arn:aws:iam::000000000000:role/executor-a"
SUBMISSION_KEY = "key-a"
WORKFLOW_ID = "wf-2"


def _regional_settings(tmp_path: Path) -> RegionalLiveSettings:
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
    ).settings


def _settings(tmp_path: Path) -> destr002.Settings:
    return destr002.Settings(
        regional=_regional_settings(tmp_path),
        node=NODE,
        host_probe_image=PROBE_IMAGE,
        hyperpod_cluster="hp-a",
        executor_role_arn=EXECUTOR_ROLE_ARN,
        predecessor_path=tmp_path / "predecessor.json",
    )


def _reboot_command() -> dict[str, Any]:
    return {
        "command_id": "cmd-1",
        "step": {"operation": "RESTART_NODE", "node_ids": [NODE]},
        "result_details": {
            "submission_idempotency_key": SUBMISSION_KEY,
            "executor_id": "executor-a/7",
        },
    }


def _submitted_state() -> dict[str, Any]:
    return {
        "workflow": {"request_id": WORKFLOW_ID, "status": "RUNNING"},
        "submission": {
            "state": "SUBMITTED",
            "idempotency_key": SUBMISSION_KEY,
            "action": "REBOOT",
            "requested_node_identifiers": [NODE],
        },
        "commands": [_reboot_command()],
    }


def _final_state() -> dict[str, Any]:
    return {
        "event": {"xid": 79},
        "decision": {"action": destr002.EXPECTED_ACTION},
        "workflow": {
            "request_id": WORKFLOW_ID,
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": name}
                for name in (
                    "MARK_UNSCHEDULABLE",
                    "RESTART_NODE",
                    "VALIDATE_GPU",
                    "VALIDATE_HOST",
                    "VALIDATE_FABRIC",
                    "RESTORE_SCHEDULING",
                )
            ],
        },
        "submission": {
            "state": "SUBMITTED",
            "action": "REBOOT",
            "requested_node_identifiers": [NODE],
        },
        "agent": {
            "artifact_sha256": "artifact-a",
            "boot_id": "agent-boot-2",
            "lifecycle_state": "ACTIVE",
        },
    }


class SpyRegional:
    """A regional fixture that records every call ``execute_case`` makes."""

    def __init__(self, _settings: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.forbidden_event = False
        self.deleted_pod: str | None = None

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))

    def kwargs_of(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self._record("store_snapshot", **kwargs)
        return _submitted_state()

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        self._record("ready_pods", plane=plane, app=app)
        pods = [
            {"name": "executor-a", "uid": "uid-a", "node": "gpu-1"},
            {"name": "executor-b", "uid": "uid-b", "node": "gpu-2"},
        ]
        if self.deleted_pod is None:
            return pods
        return [item for item in pods if item["name"] != self.deleted_pod] + [
            {"name": "executor-c", "uid": "uid-c", "node": "gpu-1"}
        ]

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        self._record("kubectl", plane=plane, arguments=arguments, **kwargs)
        if arguments[:2] == ("delete", "pod"):
            self.deleted_pod = arguments[2]
        return ""

    def executor_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
        self._record("executor_python", script=script, arguments=arguments, **kwargs)
        return {"replay_mode": "store-level", "duplicate": True, "checks": {}}

    def wait_node_ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self._record("wait_node_ready", node=node, **kwargs)
        return {
            "uid": "uid-node",
            "boot_id": "boot-2",
            "ready": "True",
            "gpu_allocatable": "8",
        }

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self._record("wait_for_workflow", **kwargs)
        return _final_state()

    def wait_provider_events(
        self, started: datetime, **kwargs: Any
    ) -> list[dict[str, str]]:
        self._record("wait_provider_events", started=started, **kwargs)
        return [
            {
                "event_name": "RebootClusterNodes",
                "session_issuer_role_name": "executor-a",
            }
        ]

    def provider_events(
        self, started: datetime, ended: datetime
    ) -> list[dict[str, str]]:
        self._record("provider_events", started=started, ended=ended)
        events = [{"event_name": "RebootClusterNodes"}]
        if self.forbidden_event:
            events.append({"event_name": "BatchReplaceClusterNodes"})
        return events

    def provider_events_provisional(self, ended: datetime) -> bool:
        self._record("provider_events_provisional", ended=ended)
        return True

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self._record("node_snapshot", node=node)
        return {
            "uid": "uid-node",
            "boot_id": "boot-2",
            "ready": "True",
            "unschedulable": False,
            "ownership_annotations": {},
            "gpu_allocatable": "8",
        }

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        self._record("cpu_blast_snapshot")
        return {}


class FakeHost:
    def __init__(self, _settings: Any) -> None:
        self.commands: list[tuple[str, ...]] = []

    def create(self) -> None:
        self.commands.append(("create",))

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.commands.append(arguments)
        if arguments[0] == "snapshot":
            return {
                "compute_clients": [],
                "kmsg_writable": True,
                "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
            }
        return {"command": arguments[0]}

    def cleanup(self) -> dict[str, bool]:
        self.commands.append(("cleanup",))
        return {}


def _preflight(*, focused_tests_reused: bool) -> dict[str, Any]:
    return {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "node": {
            "uid": "uid-node",
            "boot_id": "boot-1",
            "ready": "True",
            "gpu_allocatable": "8",
        },
        "store": {
            "agent": {
                "artifact_sha256": "artifact-a",
                "boot_id": "agent-boot-1",
                "generation": 3,
            },
            "profile": {"profile_version": "profile-a"},
        },
        "provider_preflight": {"positive": {"targets": [{"instance_id": "i-0123"}]}},
        "cpu_blast": {},
        "focused_tests": {"passed": True, "focused_tests_reused": focused_tests_reused},
        "predecessor": {"valid": True},
    }


def _drive_execute_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    focused_tests_reused: bool = True,
    forbidden_event: bool = False,
) -> tuple[int, dict[str, Any], SpyRegional, list[dict[str, Any]]]:
    preflight_calls: list[dict[str, Any]] = []
    regionals: list[SpyRegional] = []

    def read_only_preflight(
        _settings: Any, _case_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        preflight_calls.append(kwargs)
        return _preflight(focused_tests_reused=focused_tests_reused)

    def regional_factory(settings: Any) -> SpyRegional:
        regional = SpyRegional(settings)
        regional.forbidden_event = forbidden_event
        regionals.append(regional)
        return regional

    monkeypatch.setattr(destr002, "read_only_preflight", read_only_preflight)
    monkeypatch.setattr(destr002, "verify_plan_identity", lambda *_a, **_k: None)
    monkeypatch.setattr(destr002, "RegionalLiveFixture", regional_factory)
    monkeypatch.setattr(destr002, "HostProbeFixture", FakeHost)
    run_dir = tmp_path / "run-a"

    exit_code = destr002.execute_case(
        _settings(tmp_path), run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    evidence = json.loads(
        (run_dir / "cases" / destr002.CASE_ID / f"{destr002.CASE_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(regionals) == 1, "execute_case builds exactly one regional fixture"
    return exit_code, evidence, regionals[0], preflight_calls


def test_destr002_counts_the_reboot_with_the_visibility_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, evidence, regional, _ = _drive_execute_case(tmp_path, monkeypatch)

    assert exit_code == 0, evidence
    assert evidence["verdict"] == "PASS", evidence
    # The positive claim polls CloudTrail for exactly one reboot event rather
    # than reading whatever happens to be visible.
    waits = regional.kwargs_of("wait_provider_events")
    assert len(waits) == 1, regional.calls
    assert waits[0]["event_names"] == destr002.REBOOT_EVENTS
    assert waits[0]["expected_count"] == 1
    # The negative claim (no replace/delete) is provisional inside the
    # delivery window and is recorded as such.
    assert evidence["provider_events_provisional"] is True
    assert evidence["provider_events"] == [{"event_name": "RebootClusterNodes"}]


def test_destr002_forbidden_provider_event_fails_and_is_not_provisional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, evidence, _, _ = _drive_execute_case(
        tmp_path, monkeypatch, forbidden_event=True
    )

    assert exit_code == 1, evidence
    assert "CloudTrail contains replace/delete mutation" in evidence["errors"]
    assert evidence["provider_events_provisional"] is False


def test_destr002_replays_the_durable_record_from_the_replacement_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, evidence, regional, _ = _drive_execute_case(tmp_path, monkeypatch)

    replays = regional.kwargs_of("executor_python")
    assert len(replays) == 1, regional.calls
    assert replays[0]["script"] == destr002.STORE_REPLAY_PROBE
    # The probe receives the command as the executor recorded it, key included,
    # and never a recomputed one.
    assert json.loads(replays[0]["arguments"][0]) == _reboot_command()
    assert evidence["duplicate_replay"]["duplicate"] is True
    # The replay runs after the submitting Pod was deleted.
    deletes = [
        item["arguments"]
        for item in regional.kwargs_of("kubectl")
        if item["arguments"][:2] == ("delete", "pod")
    ]
    assert deletes == [("delete", "pod", "executor-a", "--wait=false")]
    order = [name for name, _ in regional.calls]
    assert order.index("kubectl") < order.index("executor_python")


def test_destr002_workflow_wait_is_scoped_to_the_submitted_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, evidence, regional, _ = _drive_execute_case(tmp_path, monkeypatch)

    waits = regional.kwargs_of("wait_for_workflow")
    assert len(waits) == 1, regional.calls
    assert waits[0]["workflow_request_ids"] == [WORKFLOW_ID]
    assert waits[0]["node"] == NODE
    assert evidence["workflow_request_id"] == WORKFLOW_ID


def test_destr002_evidence_is_bound_to_the_preflight_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, evidence, _, preflight_calls = _drive_execute_case(tmp_path, monkeypatch)

    # The execute preflight reuses the plan's focused tests instead of
    # re-running them, and the evidence says so.
    assert preflight_calls == [{"reuse_focused_tests": True}]
    assert evidence["focused_tests_reused"] is True
    # The evidence carries the release/cluster the preflight was read against.
    assert evidence["release_id"] == "release-a"
    assert evidence["cluster_id"] == "cluster-a"


def test_destr002_evidence_reports_when_focused_tests_were_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, evidence, _, _ = _drive_execute_case(
        tmp_path, monkeypatch, focused_tests_reused=False
    )

    assert evidence["focused_tests_reused"] is False


def test_destr002_submission_wait_takes_the_single_queue_sample(tmp_path: Path) -> None:
    regional = SpyRegional(None)
    observed_after = datetime.now(timezone.utc)

    submitted = destr002.wait_for_submission(
        regional,
        _settings(tmp_path),
        marker="destr002-m",
        observed_after=observed_after,
        case_dir=tmp_path,
        timeout_seconds=30,
    )

    # A wait loop takes the cheap queue read; the drained-backlog gate is for
    # preflights.
    polls = regional.kwargs_of("store_snapshot")
    assert polls == [
        {
            "node": NODE,
            "marker": "destr002-m",
            "observed_after": observed_after,
            "hyperpod_cluster": "hp-a",
            "queue_attempts": 1,
        }
    ]
    assert submitted["submission"]["state"] == "SUBMITTED"
    timeline = json.loads(
        (tmp_path / "submission-timeline.json").read_text(encoding="utf-8")
    )
    assert [entry["submission_state"] for entry in timeline["entries"]] == ["SUBMITTED"]
