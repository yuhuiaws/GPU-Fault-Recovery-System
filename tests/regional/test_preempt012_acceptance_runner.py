"""GF-REGIONAL-PREEMPT-012 runner and host probe: what holds without a node.

The case's live half is a real quiesce on a GPU host; its control half is the
``CONTROL_AUDIT`` script the runner sends into the API Pod. The script only
needs a store, so it is executed here against a SqliteStore exactly as the Pod
would run it, and the runner's verdict functions are judged on its output. The
probe's cleanup ordering and SIGTERM handling are exercised with the systemd
and quiesce calls replaced.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_preempt012_acceptance as runner
from scripts.e2e.regional.probes import preempt012_node_probe as probe

ROOT = Path(__file__).resolve().parents[2]
CONTEXT_ENVIRONMENT = {
    "GPU_FAULT_EXECUTOR_MODE": "active",
    "GPU_FAULT_ALLOW_SINGLE_CLUSTER": "true",
    "GPU_FAULT_ALLOWED_OPERATIONS": "FREEZE_EVIDENCE",
    "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
    "GPU_FAULT_EXECUTION_TOKEN": "0" * 32,
    "AWS_DEFAULT_REGION": "us-west-2",
}


def run_control_audit(tmp_path: Path, stamp: str) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_STORE_URL": f"sqlite:///{tmp_path / 'store.db'}",
        **CONTEXT_ENVIRONMENT,
    }
    return subprocess.run(
        [sys.executable, "-c", runner.CONTROL_AUDIT, "cluster-a", "node-a", stamp],
        env=environment,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def cycle_around(control: dict[str, Any]) -> dict[str, Any]:
    started = datetime.fromisoformat(control["executed_at"])
    completed = datetime.fromisoformat(control["completed_at"])
    return {
        "status": "COMPLETED",
        "quiesced_at": (started - timedelta(seconds=5)).isoformat(),
        "restored_at": (completed + timedelta(seconds=5)).isoformat(),
    }


def host_snapshot(**overrides: Any) -> dict[str, Any]:
    return {
        "services": {"kubelet": "active", "nvidia-dcgm": "active"},
        "quiesce_state_files": [],
        "gpu_count": 8,
        **overrides,
    }


def test_control_audit_supersedes_both_boundaries_against_a_real_store(
    tmp_path: Path,
) -> None:
    completed = run_control_audit(tmp_path, "stamp1")

    assert completed.returncode == 0, completed.stderr
    control = json.loads(completed.stdout.splitlines()[-1])
    print(json.dumps(control["clean"], sort_keys=True))
    checks = runner.evaluate_checks(
        control=control,
        cycle=cycle_around(control),
        baseline=host_snapshot(),
        final_host=host_snapshot(),
        final_nodes=[{"ready": "True", "unschedulable": False, "taints": []}],
        provider=[],
        cpu_blast_unchanged=True,
    )
    assert checks == {name: True for name in checks}, {
        name: control for name, value in checks.items() if not value
    }
    clean = control["clean"]
    assert clean["successor_adapter_calls"] == ["RESTART_NODE"]
    assert clean["successor_inherited_step_indexes"] == [0, 1]
    assert clean["successor_inherited_from"] == [clean["predecessor_id"]]
    assert clean["preempted_by"] == clean["successor_id"]
    assert control["dirty"]["boundary"].startswith("QUIESCE done"), (
        "the dirty group must name the PREEMPT-008 boundary it exercised"
    )
    assert control["remote_commands_for_audit_workflows"] == 0
    assert control["residual_objects"] == []
    assert control["completed_at"] >= control["executed_at"]


def test_control_audit_refuses_ids_that_already_exist(tmp_path: Path) -> None:
    from gpu_fault.models import FaultIncident
    from gpu_fault.store import SqliteStore

    store = SqliteStore(str(tmp_path / "store.db"))
    store.save_incident(
        FaultIncident(
            incident_id="incident-clean-stamp2",
            event_id="event-leftover",
            event_type="PREEMPT012_AUDIT",
            cluster_id="cluster-a",
            node_ids=["node-a"],
            policy_version="preempt012/v1",
            policy_source="ACCEPTANCE",
            fencing_token=1,
        )
    )

    completed = run_control_audit(tmp_path, "stamp2")

    assert completed.returncode == 1
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["error"] == "audit ids already exist"
    assert result["existing"] == [{"kind": "incident", "key": "incident-clean-stamp2"}]


def test_overlap_check_needs_the_whole_audit_inside_the_quiesce_window() -> None:
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    control = {
        "executed_at": now.isoformat(),
        "completed_at": (now + timedelta(seconds=20)).isoformat(),
        "clean": {
            "status": "SUPERSEDED",
            "new_adapter_calls": [],
            "predecessor_id": "pred",
            "preempted_by": "succ",
            "successor_id": "succ",
            "successor_adapter_calls": ["RESTART_NODE"],
            "successor_inherited_step_indexes": [0, 1],
            "successor_completed_operations": ["MARK_UNSCHEDULABLE", "STOP_WORKLOADS"],
            "successor_inherited_from": ["pred"],
        },
        "dirty": {
            "status": "SUPERSEDED",
            "handoff_after_claim": "pred",
            "predecessor_id": "pred",
        },
        "physical_operations_called": [],
        "remote_commands_for_audit_workflows": 0,
        "residual_objects": [],
    }

    def overlap(quiesced: datetime, restored: datetime) -> bool:
        return runner.evaluate_checks(
            control=control,
            cycle={
                "quiesced_at": quiesced.isoformat(),
                "restored_at": restored.isoformat(),
            },
            baseline=host_snapshot(),
            final_host=host_snapshot(),
            final_nodes=[],
            provider=[],
            cpu_blast_unchanged=True,
        )["control_audit_overlapped_real_quiesce"]

    assert overlap(now - timedelta(seconds=1), now + timedelta(seconds=30)), (
        "an audit fully inside the quiesce window must count as overlapped"
    )
    # Quiesced after the audit started: the runner used to accept this with a
    # fixed sleep standing in for the QUIESCED wait.
    assert not overlap(now + timedelta(seconds=1), now + timedelta(seconds=30)), (
        "an audit that started before QUIESCED must not count as overlapped"
    )
    # Restored before the audit finished: the dirty boundary ran on a live node.
    assert not overlap(now - timedelta(seconds=1), now + timedelta(seconds=10)), (
        "an audit that outlived the restore must not count as overlapped"
    )


def test_wait_for_cycle_returns_at_the_requested_status_or_a_terminal_one() -> None:
    class Host:
        def __init__(self, statuses: list[str]) -> None:
            self.statuses = iter(statuses)
            self.reads = 0

        def execute(self, *arguments: str) -> dict[str, Any]:
            self.reads += 1
            return {"status": next(self.statuses)}

    quiesced = Host(["PENDING", "RUNNING", "QUIESCED", "COMPLETED"])
    cycle = runner.wait_for_cycle(
        quiesced, "r", until=frozenset({"QUIESCED"}), timeout=5, poll_seconds=0
    )
    assert cycle["status"] == "QUIESCED" and quiesced.reads == 3

    failed_early = Host(["RUNNING", "FAILED"])
    cycle = runner.wait_for_cycle(
        failed_early, "r", until=frozenset({"QUIESCED"}), timeout=5, poll_seconds=0
    )
    assert cycle["status"] == "FAILED", "a terminal status must end the wait"

    interrupted = Host(["QUIESCED", "INTERRUPTED"])
    cycle = runner.wait_for_cycle(
        interrupted,
        "r",
        until=runner.CYCLE_TERMINAL_STATUSES,
        timeout=5,
        poll_seconds=0,
    )
    assert cycle["status"] == "INTERRUPTED"


def test_cleanup_checks_need_no_ownership_annotation_and_a_dead_timer() -> None:
    good = runner.cleanup_checks(
        host_cleanup={"timer_active_state": "inactive"},
        node_state={"ownership_annotations": {}},
    )
    assert good == {"ownership_annotations_removed": True, "cycle_timer_inactive": True}
    bad = runner.cleanup_checks(
        host_cleanup={"timer_active_state": "active", "restore_error": "x"},
        node_state={"ownership_annotations": {"gpu-fault.io/owner": "w"}},
    )
    assert bad == {
        "ownership_annotations_removed": False,
        "cycle_timer_inactive": False,
    }
    assert runner.cleanup_checks(host_cleanup=None, node_state=None) == {
        "ownership_annotations_removed": False,
        "cycle_timer_inactive": False,
    }


def test_limitations_name_the_dirty_boundary_shape_and_destr016() -> None:
    joined = "\n".join(runner.LIMITATIONS)
    assert "QUIESCE done, reset not submitted" in joined
    assert "PREEMPT-008" in joined
    assert "DESTR-016" in joined


# --------------------------------------------------------------------------- #
# the host probe
# --------------------------------------------------------------------------- #
class FakeManager:
    calls: list[str] = []
    restore_raises: BaseException | None = None

    def __init__(self, **kwargs: Any) -> None:
        pass

    def quiesce(self, **kwargs: Any) -> dict[str, Any]:
        FakeManager.calls.append("quiesce")
        return {"ok": True}

    def restore(self, **kwargs: Any) -> dict[str, Any]:
        FakeManager.calls.append("restore")
        if FakeManager.restore_raises is not None:
            raise FakeManager.restore_raises
        return {"restored": True}


@pytest.fixture
def probe_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "acceptance")
    monkeypatch.setattr(probe, "QUIESCE_ROOT", tmp_path / "quiesce")
    (tmp_path / "quiesce").mkdir()
    FakeManager.calls = []
    FakeManager.restore_raises = None
    monkeypatch.setattr(probe, "GpuServiceQuiesceManager", FakeManager)
    commands: list[str] = []

    def fake_run(command: list[str], *, check: bool = True) -> Any:
        commands.append(" ".join(command))
        FakeManager.calls.append(" ".join(command[:2]))
        stdout = "ActiveState=inactive\n" if "show" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(probe, "run", fake_run)
    return commands


def test_cleanup_stops_the_timer_before_restoring_and_survives_a_restore_error(
    probe_roots: list[str],
) -> None:
    evidence = probe.evidence_path("r1")
    probe.write_json(evidence, {"status": "QUIESCED"})
    FakeManager.restore_raises = SystemExit("restore blew up")

    result = probe.cleanup(Namespace(run_id="r1"))

    stop_index = FakeManager.calls.index("systemctl stop")
    restore_index = FakeManager.calls.index("restore")
    assert stop_index < restore_index, (
        "the timer must be stopped before the restore, or it fires afterwards"
    )
    assert result["restore"] is None
    assert result["restore_error"] == "SystemExit: restore blew up"
    assert result["timer_active_state"] == "inactive"
    assert result["evidence_exists"] is False and not evidence.exists()


def test_cycle_installs_a_sigterm_handler_and_restores_when_interrupted(
    probe_roots: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted_sleep(seconds: float) -> None:
        raise probe.ProbeInterrupted("signal 15")

    monkeypatch.setattr(probe.time, "sleep", interrupted_sleep)
    try:
        probe.cycle(Namespace(run_id="r2", hold_seconds=45, failsafe_seconds=180))
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), (
            "systemctl stop must reach the cycle as an exception, not a kill"
        )
        with pytest.raises(probe.ProbeInterrupted):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)

    evidence = json.loads(probe.evidence_path("r2").read_text(encoding="utf-8"))
    manager_calls = [
        call for call in FakeManager.calls if call in {"quiesce", "restore"}
    ]
    assert manager_calls == ["quiesce", "restore"]
    assert evidence["status"] == "INTERRUPTED"
    assert evidence["interrupted"] == "signal 15"
    assert evidence["restore"] == {"restored": True}
    assert "quiesced_at" in evidence and "restored_at" in evidence


def test_cycle_records_a_restore_error_of_any_type(
    probe_roots: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    FakeManager.restore_raises = KeyboardInterrupt()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        probe.cycle(Namespace(run_id="r3", hold_seconds=0, failsafe_seconds=180))
    finally:
        signal.signal(signal.SIGTERM, previous)

    evidence = json.loads(probe.evidence_path("r3").read_text(encoding="utf-8"))
    assert evidence["status"] == "FAILED"
    assert evidence["restore_error"].startswith("KeyboardInterrupt"), (
        "a non-Exception restore failure must still be recorded"
    )


def test_plan_parser_stays_plan_only_by_default() -> None:
    parsed = runner.parser().parse_args(["--run-dir", "/tmp/run"])
    assert parsed.execute is False
