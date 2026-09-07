"""HA-009: phase-budget TTL, refresh watchdog, rollout predicate reuse, skipped roles."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009


def test_registration_ttl_and_probe_deadline_cover_the_worst_path() -> None:
    total = ha009.total_budget_seconds()
    assert total == sum(ha009.PHASE_BUDGETS.values()) + ha009.BUDGET_MARGIN_SECONDS
    assert total > 60 * 60, (
        "the old 45-minute TTL was shorter than the ~60-minute worst path"
    )
    assert ha009.REFRESH_WATCHDOG_SECONDS == (
        ha009.PHASE_BUDGETS["managed_rotation"]
        + ha009.PHASE_BUDGETS["first_refresh_job"]
        + ha009.BUDGET_MARGIN_SECONDS
    )
    assert ha009.REFRESH_WATCHDOG_SECONDS < total


def test_refresh_watchdog_runs_detached_and_is_disarmed_by_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ha009, "CRONJOB", "gpu-fault-aurora-credential-refresh")
    monkeypatch.setattr(
        ha009.BASE._registry, "CONTROL_KUBECONFIG", "/secure/cpu.kubeconfig"
    )
    monkeypatch.setattr(ha009.BASE._registry, "CONTROL_NAMESPACE", "gpu-fault-system")

    process = ha009.start_refresh_watchdog(tmp_path, "job-w", delay_seconds=600)
    try:
        assert os.getpgid(process.pid) == process.pid, "watchdog must own its session"
        record = (tmp_path / "refresh-watchdog.json").read_text()
        assert '"job": "job-w"' in record
        assert process.poll() is None
    finally:
        outcome = ha009.stop_refresh_watchdog(process)
    assert outcome["disarmed"] is True
    assert process.poll() is not None
    assert ha009.stop_refresh_watchdog(None) == {"armed": False}


def test_refresh_job_command_creates_from_the_cronjob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ha009, "CRONJOB", "cron-x")
    monkeypatch.setattr(ha009.BASE._registry, "CONTROL_KUBECONFIG", "/k")
    monkeypatch.setattr(ha009.BASE._registry, "CONTROL_NAMESPACE", "ns")
    command = ha009.refresh_job_command("job-1")
    assert command[:5] == ["kubectl", "--kubeconfig", "/k", "-n", "ns"]
    assert command[5:] == ["create", "job", "job-1", "--from=cronjob/cron-x"]


def _deployment(
    generation: int, uids: list[str], replicas: int = 3, *, complete: bool = True
) -> dict:
    return {
        "generation": generation,
        "observed_generation": generation,
        "replicas": replicas,
        "ready": replicas,
        "updated": replicas if complete else max(0, replicas - 1),
        "available": replicas,
        "pods": [
            (f"p-{uid}", {"uid": uid, "ready": True, "restarts": 0}) for uid in uids
        ],
    }


def test_deployments_rolled_requires_updated_replicas_and_new_uids() -> None:
    before = {
        "gpu-fault-api-ha": _deployment(1, ["a1", "a2", "a3"]),
        "gpu-fault-control-worker": _deployment(1, ["w1", "w2", "w3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    rolled = {
        "gpu-fault-api-ha": _deployment(2, ["b1", "b2", "b3"]),
        "gpu-fault-control-worker": _deployment(2, ["x1", "x2", "x3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    assert ha009.deployments_rolled(before, rolled) is True, (
        "a replicas=0 role must not block completion"
    )
    half = {**rolled, "gpu-fault-control-worker": _deployment(2, ["x1", "x2", "w3"])}
    assert ha009.deployments_rolled(before, half) is False, "an old UID still present"
    not_updated = {
        **rolled,
        "gpu-fault-api-ha": _deployment(2, ["b1", "b2", "b3"], complete=False),
    }
    assert ha009.deployments_rolled(before, not_updated) is False, (
        "ready == replicas with updatedReplicas short is mid-rollout"
    )
    assert ha009.role_status(before) == {
        "gpu-fault-api-ha": "ROLLED",
        "gpu-fault-control-worker": "ROLLED",
        "gpu-fault-telemetry-spool-worker": "SKIPPED_NOT_ENABLED",
    }


def test_rotation_errors_reuse_ha005_continuity_and_skip_disabled_roles() -> None:
    before = {
        "gpu-fault-api-ha": _deployment(1, ["a1", "a2", "a3"]),
        "gpu-fault-control-worker": _deployment(1, ["w1", "w2", "w3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    after = {
        "gpu-fault-api-ha": _deployment(2, ["b1", "b2", "b3"]),
        "gpu-fault-control-worker": _deployment(2, ["x1", "x2", "x3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    probe = {
        "counters": {
            "event_attempts": 10,
            "event_accepted": 9,
            "event_failures": 1,
            "event_buffered": 1,
        },
        "outbox": {"records": 0, "replayable": 0},
        "error_types": {},
        "accepted_request_ids": ["r1"],
    }
    receipts = {
        "requests": [
            {"request_id": "r1", "status": "COMPLETED", "response_status": 200}
        ],
        "missing": [],
    }
    runtime = {
        "command": {"status": "SUCCEEDED"},
        "notification": {
            "notification": {"count": 1},
            "notification_delivery": {"count": 1},
            "notification_result": {"count": 1, "status": "SKIPPED"},
        },
    }
    common = dict(
        versions_after={"stages": {"AWSCURRENT": "v2", "AWSPREVIOUS": "v1"}},
        current_before="v1",
        digest_before="d1",
        digest_after="d2",
        first_job={"logs": ["rotated=True restarted=True"]},
        second_job={"logs": ["rotated=False restarted=False"]},
        deployments_before=before,
        deployments_after=after,
        receipts=receipts,
        runtime=runtime,
        digest_before_noop="d2",
        digest_after_noop="d2",
        before_noop=after,
        after_noop=after,
    )
    assert ha009.rotation_errors(final_probe=probe, **common) == []

    broken_probe = {**probe, "error_types": {"http-500": 1}}
    errors = ha009.rotation_errors(final_probe=broken_probe, **common)
    assert errors == ha005.continuity_errors(
        broken_probe, receipts, accepted_ids=["r1"]
    ), "HA-009 must report exactly what HA-005's shared evaluation reports"


def test_stop_refresh_watchdog_reports_a_fired_watchdog(tmp_path: Path) -> None:
    process = subprocess.Popen(
        ["/bin/sh", "-c", "exit 0"], start_new_session=True, text=True
    )
    process.wait(timeout=10)
    outcome = ha009.stop_refresh_watchdog(process)
    assert outcome == {"armed": True, "fired": True, "returncode": 0}


def test_stop_process_group_kills_sleep_child_too(tmp_path: Path) -> None:
    process = subprocess.Popen(
        ["/bin/bash", "-c", "sleep 600; exit 0"], start_new_session=True, text=True
    )
    try:
        time.sleep(0.2)
        pgid = os.getpgid(process.pid)
        outcome = ha009.stop_refresh_watchdog(process)
        assert outcome["disarmed"] is True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the watchdog's sleep survived the process-group kill")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
