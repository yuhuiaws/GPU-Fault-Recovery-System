"""Contracts of the shared regional live fixture every ``run_*.py`` depends on.

Each test here pins a defect the 2026-09-07 regional e2e review found in the
layer under the runners: retries of non-idempotent scripts, predecessor
evidence that was not bound to a release, lease tokens in evidence, CloudTrail
read too early, and a reboot declared on an empty boot ID.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.store import NotFoundError
from scripts.e2e.regional import regional_live_fixture as live_fixture_module
from scripts.e2e.regional.regional_live_fixture import (
    STORE_PROBE,
    RegionalCommandTimeout,
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    waiting_step_executions,
)

CASE_ID = "GF-REGIONAL-DESTR-010"


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


# -- item 1: non-idempotent scripts are not retried ---------------------------


def test_pod_python_honours_a_single_attempt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    calls: list[int] = []

    def failing(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise RegionalFixtureError("exec failed")

    monkeypatch.setattr(regional, "ready_pod", lambda *_a, **_k: "pod-a")
    monkeypatch.setattr(regional, "kubectl", failing)
    monkeypatch.setattr(live_fixture_module.time, "sleep", lambda _s: None)

    with pytest.raises(RegionalFixtureError, match="Pod probe failed"):
        regional.pod_python("cpu", "gpu-fault-api-ha", "print()", attempts=1)
    assert calls == [1], "attempts=1 must mean exactly one exec"

    calls.clear()
    with pytest.raises(RegionalFixtureError):
        regional.pod_python("cpu", "gpu-fault-api-ha", "print()")
    assert len(calls) == 3, "the default budget is still three attempts"


def test_pod_python_never_retries_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out exec may have run to completion; running it again is a
    second mutation, not a retry."""

    regional = _regional(tmp_path)
    calls: list[int] = []

    def timing_out(*_args: Any, **_kwargs: Any) -> str:
        calls.append(1)
        raise RegionalCommandTimeout(["kubectl", "exec"], 180)

    monkeypatch.setattr(regional, "ready_pod", lambda *_a, **_k: "pod-a")
    monkeypatch.setattr(regional, "kubectl", timing_out)
    monkeypatch.setattr(live_fixture_module.time, "sleep", lambda _s: None)

    with pytest.raises(RegionalFixtureError, match="not retried"):
        regional.pod_python("cpu", "gpu-fault-api-ha", "print()", attempts=3)

    assert calls == [1], calls


def test_run_converts_a_subprocess_timeout_into_a_fixture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang(command: list[str], **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(command, 7)

    monkeypatch.setattr(live_fixture_module.subprocess, "run", hang)

    with pytest.raises(RegionalCommandTimeout, match="timed out after 7s") as raised:
        RegionalLiveFixture.run(["kubectl", "get", "pod"], timeout=7)

    assert isinstance(raised.value, RegionalFixtureError), (
        "a command timeout must still be catchable as the fixture error"
    )
    assert raised.value.command == ["kubectl", "get", "pod"]


def test_post_xid_event_runs_with_a_single_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    seen: dict[str, Any] = {}

    def executor_python(script: str, *arguments: str, **kwargs: Any) -> dict[str, Any]:
        seen["kwargs"] = kwargs
        seen["payload"] = json.loads(arguments[0])
        return {"accepted": {}, "receipt": None}

    monkeypatch.setattr(regional, "executor_python", executor_python)

    regional.post_xid_event({"cluster_id": "cluster-a", "node_id": "node-a"})

    assert seen["kwargs"]["attempts"] == 1, seen
    assert seen["payload"]["node_id"] == "node-a"


def test_python_wrappers_pass_attempts_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    seen: list[dict[str, Any]] = []

    def pod_python(
        plane: str, app: str, script: str, *arguments: str, **kwargs: Any
    ) -> dict[str, Any]:
        seen.append({"plane": plane, "app": app, **kwargs})
        return {}

    monkeypatch.setattr(regional, "pod_python", pod_python)

    regional.cpu_python("print()", attempts=1)
    regional.executor_python("print()", timeout=30, attempts=2)

    assert seen == [
        {"plane": "cpu", "app": "gpu-fault-api-ha", "timeout": 180, "attempts": 1},
        {
            "plane": "gpu",
            "app": "gpu-fault-cluster-executor",
            "timeout": 30,
            "attempts": 2,
        },
    ], seen


# -- item 2: predecessor evidence is bound to release and cluster -------------


def _write_pass(path: Path, **extra: Any) -> None:
    path.write_text(
        json.dumps({"case_id": CASE_ID, "verdict": "PASS", **extra}), encoding="utf-8"
    )


def test_predecessor_evidence_requires_the_same_release_when_asked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predecessor.json"
    _write_pass(path, release_id="release-old", cluster_id="cluster-a")

    result = predecessor_evidence(
        path, CASE_ID, release_id="release-new", cluster_id="cluster-a"
    )

    assert result["evidence_valid"] is False, result
    assert result["valid"] is False, result
    assert result["execution_allowed"] is False, result
    assert result["error"] == "release_id mismatch", result
    assert result["evidence_release_id"] == "release-old", result
    assert result["expected_release_id"] == "release-new", result


def test_predecessor_evidence_without_an_identity_is_refused_when_asked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predecessor.json"
    _write_pass(path)

    result = predecessor_evidence(
        path, CASE_ID, release_id="release-a", cluster_id="cluster-a"
    )

    assert result["evidence_valid"] is False, result
    assert result["error"] == (
        "release_id missing from predecessor evidence; "
        "cluster_id missing from predecessor evidence"
    ), result


def test_predecessor_evidence_bound_to_the_same_deployment_passes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predecessor.json"
    _write_pass(path, release_id="release-a", cluster_id="cluster-a")

    result = predecessor_evidence(
        path, CASE_ID, release_id="release-a", cluster_id="cluster-a"
    )

    assert result["valid"] is True, result
    assert result["error"] is None, result
    # A cluster mismatch alone is enough to refuse.
    other = predecessor_evidence(
        path, CASE_ID, release_id="release-a", cluster_id="cluster-b"
    )
    assert other["error"] == "cluster_id mismatch", other


def test_predecessor_evidence_is_unchanged_when_no_identity_is_requested(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predecessor.json"
    _write_pass(path)

    result = predecessor_evidence(path, CASE_ID)

    assert result["valid"] is True, result
    assert result["expected_release_id"] is None, result


def test_evidence_identity_names_the_release_and_the_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    monkeypatch.setattr(regional, "release_id", lambda: "release-a")

    assert regional.evidence_identity() == {
        "release_id": "release-a",
        "cluster_id": "cluster-a",
    }


# -- items 3 and 8: the store probe redacts lease tokens and filters commands --


class _Record:
    def __init__(self, **values: Any) -> None:
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        assert mode == "json"
        return dict(self._values)


class _FakeStore:
    def __init__(self) -> None:
        self.command_filters: list[Any] = []
        self.workflow = _Record(request_id="workflow-a", status="RUNNING")
        self.commands = [
            _Record(
                command_id="command-a",
                workflow_request_id="workflow-a",
                status="LEASED",
                lease_token="sensitive-lease-token-value",
            ),
            _Record(
                command_id="command-b",
                workflow_request_id="workflow-a",
                status="PENDING",
                lease_token=None,
            ),
        ]

    def processor_fault_backlog_depth(self) -> int:
        # The probe gates on the fault tier of the processor queue, not its depth.
        return 0

    def list_xid_events(self, _cluster: str, _node: str, *, observed_after: Any):
        return [_Record(event_id="event-a", raw_message="marker-a", xid=79)]

    def get_xid_policy_decision(self, _event_id: str):
        raise NotFoundError("no decision")

    def get_incident_by_event(self, _event_id: str):
        return _Record(incident_id="incident-a", workflow_request_id="workflow-a")

    def get_workflow(self, _request_id: str):
        return self.workflow

    def list_remote_commands(self, *, workflow_request_ids: Any = None):
        self.command_filters.append(
            None if workflow_request_ids is None else list(workflow_request_ids)
        )
        return self.commands

    def list_notifications(self, **_kwargs: Any):
        return []

    def list_attempt_observations(self, _cluster: str):
        return []

    def get_agent(self, _cluster: str, _node: str):
        raise NotFoundError("no agent")

    def processor_queue_stats(self) -> dict[str, Any]:
        return {"depth": 0, "oldest_age_seconds": None}

    def remote_command_stats(self) -> dict[str, Any]:
        return {"pending": 1}


def _run_store_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> tuple[dict[str, Any], _FakeStore]:
    from gpu_fault.app import ApplicationContext

    store = _FakeStore()
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(store=store)),
    )
    monkeypatch.setattr(sys, "argv", ["store-probe", *argv])
    exec(compile(STORE_PROBE, "<store-probe>", "exec"), {"__name__": "__probe__"})
    output = capsys.readouterr().out.strip().splitlines()[-1]
    return json.loads(output), store


_EIGHT_ARGUMENTS = ["cluster-a", "node-a", "marker-a", "", "", "", "", "1"]


def test_store_probe_never_emits_the_raw_lease_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload, _store = _run_store_probe(monkeypatch, capsys, _EIGHT_ARGUMENTS)

    assert "sensitive-lease-token-value" not in json.dumps(payload)
    leased, pending = payload["commands"]
    assert "lease_token" not in leased, leased
    assert len(leased["lease_token_sha256"]) == 64, leased
    assert leased["lease_token_length"] == len("sensitive-lease-token-value")
    assert pending["lease_token_sha256"] is None, pending
    assert pending["lease_token_length"] is None, pending


def test_store_probe_filters_commands_by_the_resolved_workflow_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload, store = _run_store_probe(monkeypatch, capsys, _EIGHT_ARGUMENTS)

    # Eight arguments is the pre-filter call shape: same result as the old
    # unfiltered read, but the store does the narrowing.
    assert store.command_filters == [["workflow-a"]], store.command_filters
    assert [item["command_id"] for item in payload["commands"]] == [
        "command-a",
        "command-b",
    ]


def test_store_probe_passes_an_explicit_workflow_filter_to_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _payload, store = _run_store_probe(
        monkeypatch, capsys, [*_EIGHT_ARGUMENTS, "workflow-x,workflow-y"]
    )

    assert store.command_filters == [["workflow-x", "workflow-y"]]


def test_store_snapshot_only_appends_the_filter_argument_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    seen: list[tuple[str, ...]] = []

    def cpu_python(_script: str, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        seen.append(arguments)
        return {"release_id": "release-a"}

    monkeypatch.setattr(regional, "cpu_python", cpu_python)

    regional.store_snapshot(node="node-a", queue_attempts=1)
    regional.store_snapshot(node="node-a", workflow_request_ids=["w-a", "w-b"])

    assert len(seen[0]) == 8, seen[0]
    assert len(seen[1]) == 9 and seen[1][-1] == "w-a,w-b", seen[1]
    with pytest.raises(ValueError, match="no comma"):
        regional.store_snapshot(workflow_request_ids=["a,b"])


def test_wait_for_workflow_forwards_the_filter_and_a_cheap_queue_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    seen: list[dict[str, Any]] = []

    def store_snapshot(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {
            "workflow": {"status": "SUCCEEDED", "step_executions": []},
            "commands": [],
        }

    monkeypatch.setattr(regional, "store_snapshot", store_snapshot)

    regional.wait_for_workflow(
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=5,
        workflow_request_ids=["workflow-a"],
    )

    assert seen[0]["queue_attempts"] == 1, seen
    assert seen[0]["workflow_request_ids"] == ["workflow-a"], seen


def test_wait_for_workflow_keeps_the_snapshot_call_shape_without_a_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    seen: list[dict[str, Any]] = []

    def store_snapshot(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"workflow": {"status": "FAILED"}, "commands": []}

    monkeypatch.setattr(regional, "store_snapshot", store_snapshot)

    regional.wait_for_workflow(
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=5,
    )

    assert "workflow_request_ids" not in seen[0], seen


# -- item 7: CloudTrail is eventually consistent ------------------------------


def test_wait_provider_events_polls_until_the_expected_count_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    reboot = {"event_name": "BatchRebootClusterNodes", "event_time": "t"}
    replace = {"event_name": "BatchReplaceClusterNodes", "event_time": "t"}
    lookups = iter([[], [replace], [replace, reboot]])
    windows: list[datetime] = []
    sleeps: list[int] = []

    def provider_events(_started: datetime, ended: datetime) -> list[dict[str, str]]:
        windows.append(ended)
        return next(lookups)

    monkeypatch.setattr(regional, "provider_events", provider_events)
    monkeypatch.setattr(live_fixture_module.time, "sleep", sleeps.append)

    events = regional.wait_provider_events(
        datetime.now(timezone.utc) - timedelta(minutes=5),
        event_names={"BatchRebootClusterNodes"},
        expected_count=1,
        poll_seconds=30,
    )

    assert events == [reboot], events
    assert len(windows) == 3 and sleeps == [30, 30], (windows, sleeps)
    # Each poll re-reads up to "now" so late-delivered events land in the window.
    assert windows[0] <= windows[1] <= windows[2]


def test_wait_provider_events_returns_what_it_saw_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    lookups: list[int] = []
    clock = {"now": 0.0}

    def monotonic() -> float:
        # Every read advances the clock by 500s, so the 900s budget is spent
        # after the second lookup whatever the poll interval.
        clock["now"] += 500.0
        return clock["now"]

    def provider_events(_started: datetime, _ended: datetime) -> list[dict[str, str]]:
        lookups.append(1)
        return []

    monkeypatch.setattr(regional, "provider_events", provider_events)
    monkeypatch.setattr(live_fixture_module.time, "monotonic", monotonic)
    monkeypatch.setattr(live_fixture_module.time, "sleep", lambda _s: None)

    events = regional.wait_provider_events(
        datetime.now(timezone.utc),
        event_names={"BatchRebootClusterNodes"},
        expected_count=1,
        timeout_seconds=900,
    )

    assert events == []
    assert 1 <= len(lookups) <= 3, lookups


def test_provider_events_provisional_inside_the_visibility_window() -> None:
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

    assert RegionalLiveFixture.provider_events_provisional(
        now - timedelta(minutes=14), now=now
    ), "14 minutes ago is still inside the provider visibility window"
    assert not RegionalLiveFixture.provider_events_provisional(
        now - timedelta(minutes=16), now=now
    ), "16 minutes ago is past the provider visibility window"


# -- item 10: a reboot needs an observed boot ID ------------------------------


def test_wait_node_ready_does_not_call_an_empty_boot_id_a_reboot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {"name": "node-a", "ready": "True", "boot_id": None},
            {"name": "node-a", "ready": "True", "boot_id": ""},
            {"name": "node-a", "ready": "True", "boot_id": "boot-old"},
            {"name": "node-a", "ready": "True", "boot_id": "boot-new"},
        ]
    )
    monkeypatch.setattr(regional, "node_snapshot", lambda _node: next(snapshots))
    monkeypatch.setattr(live_fixture_module.time, "sleep", lambda _s: None)

    result = regional.wait_node_ready(
        "node-a", timeout_seconds=60, expected_boot_id="boot-old"
    )

    assert result["boot_id"] == "boot-new", result


# -- item 11: the GPU extraction the two workload readers share ---------------


def test_pod_gpu_count_takes_the_largest_request_or_limit() -> None:
    item = {
        "spec": {
            "containers": [
                {"resources": {"requests": {"nvidia.com/gpu": "2"}}},
                {"resources": {"limits": {"nvidia.com/gpu": 4}}},
                {"resources": {"limits": {"nvidia.com/gpu": "not-a-number"}}},
                {},
            ]
        }
    }

    assert RegionalLiveFixture.pod_gpu_count(item) == 4
    assert RegionalLiveFixture.pod_gpu_count({"spec": {}}) == 0


def test_waiting_projection_keeps_timestamps_only_when_the_record_has_them() -> None:
    """DESTR-017 measures the per-step waiting cap from ``started_at`` on the
    WAITING record; a record without timestamps projects exactly as before."""

    bare = {"step_index": 3, "operation": "VERIFY_NO_GPU_CLIENTS", "status": "WAITING"}
    stamped = {**bare, "started_at": "2026-09-06T10:02:00+00:00", "updated_at": None}

    [projected_bare, projected_stamped] = waiting_step_executions(
        {"workflow": None, "step_executions": [bare, stamped]}
    )

    assert "started_at" not in projected_bare and "updated_at" not in projected_bare
    assert projected_bare["error"] is None
    assert projected_stamped["started_at"] == "2026-09-06T10:02:00+00:00"
    assert "updated_at" not in projected_stamped, projected_stamped


def test_store_snapshot_widens_observed_after_for_a_marker_tagged_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kmsg line lands a little before the wall clock that timed its injection.

    COLLECT-021 read its XID at 10:51:08.9 against an injected_at of 10:51:09.x
    and an exact cut dropped the event the control plane had already decided.
    The marker identifies the line; the time bound only limits the scan.
    """

    regional = _regional(tmp_path)
    seen: list[tuple[str, ...]] = []

    def cpu_python(_script: str, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        seen.append(arguments)
        return {"release_id": "release-a"}

    monkeypatch.setattr(regional, "cpu_python", cpu_python)
    injected_at = datetime(2026, 9, 10, 10, 51, 9, 500000, tzinfo=timezone.utc)

    regional.store_snapshot(node="node-a", marker="c021-1", observed_after=injected_at)
    regional.store_snapshot(node="node-a", observed_after=injected_at)

    widened = datetime.fromisoformat(seen[0][3])
    assert widened == injected_at - timedelta(
        seconds=live_fixture_module.KMSG_CLOCK_SKEW_SECONDS
    ), "a marker-tagged read tolerates the kernel clock trailing the wall clock"
    assert datetime.fromisoformat(seen[1][3]) == injected_at, (
        "without a marker the bound is the only filter and stays exact"
    )
