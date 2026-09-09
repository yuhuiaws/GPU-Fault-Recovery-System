"""HA-009: phase-budget TTL, refresh watchdog, steady-state predicate, idle-window variant.

CP-3 (路 A): a rotation is a Secret write; running Pods reload the mounted file
and nothing restarts. The case therefore asserts the opposite of what it used
to: generation, Pod UIDs and restartCount stay put, the projected file catches
up with the Secret, and after the pool's ``max_idle`` has recycled its
connections every Pod still answers ``/healthz`` with no authentication failure
in its log (H1-5).
"""

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
    # Path A phases: the Secret propagates to the mounts, then the case waits
    # out the pool's max_idle and observes; no consumer rollout any more.
    assert "consumer_rollout" not in ha009.PHASE_BUDGETS
    assert ha009.PHASE_BUDGETS["secret_propagation"] >= 120, "kubelet sync period"
    assert ha009.PHASE_BUDGETS["idle_window"] > ha009.DEFAULT_POOL_MAX_IDLE_SECONDS
    assert ha009.PHASE_BUDGETS["post_idle_observation"] >= 60
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
    # deployments_rolled stays for the refresher's --restart-deployments
    # compatibility mode; the catalog status for path A is STEADY.
    assert ha009.role_status(before) == {
        "gpu-fault-api-ha": "STEADY",
        "gpu-fault-control-worker": "STEADY",
        "gpu-fault-telemetry-spool-worker": "SKIPPED_NOT_ENABLED",
    }


def test_deployments_steady_requires_same_generation_uids_and_restarts() -> None:
    before = {
        "gpu-fault-api-ha": _deployment(1, ["a1", "a2", "a3"]),
        "gpu-fault-control-worker": _deployment(1, ["w1", "w2", "w3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    assert ha009.deployments_steady(before, before) == []

    rolled = {**before, "gpu-fault-api-ha": _deployment(2, ["b1", "b2", "b3"])}
    assert any(
        "generation" in item for item in ha009.deployments_steady(before, rolled)
    ), "a generation bump must be reported as a rollout"

    replaced = {
        **before,
        "gpu-fault-control-worker": _deployment(1, ["w1", "w2", "w9"]),
    }
    assert any("Pod" in item for item in ha009.deployments_steady(before, replaced)), (
        "a replaced Pod uid must be reported"
    )

    restarted = {**before, "gpu-fault-api-ha": _deployment(1, ["a1", "a2", "a3"])}
    restarted["gpu-fault-api-ha"]["pods"][0][1]["restarts"] = 1
    assert any(
        "restart" in item for item in ha009.deployments_steady(before, restarted)
    ), "a container restart must be reported"
    assert ha009.role_status(before) == {
        "gpu-fault-api-ha": "STEADY",
        "gpu-fault-control-worker": "STEADY",
        "gpu-fault-telemetry-spool-worker": "SKIPPED_NOT_ENABLED",
    }


METRICS_TEXT = """# HELP gpu_fault_postgres_pool_size x
# TYPE gpu_fault_postgres_pool_size gauge
gpu_fault_postgres_pool_size 2
gpu_fault_postgres_pool_connections_errors_total 3
gpu_fault_postgres_pool_checkout_wait_seconds_count 44
gpu_fault_aurora_credential_refresh_last_success_age_seconds 71.5
"""


def test_pool_metric_samples_are_parsed_from_the_exposition_text() -> None:
    values = ha009.parse_pool_metrics(METRICS_TEXT)
    assert values == {
        "gpu_fault_postgres_pool_size": 2.0,
        "gpu_fault_postgres_pool_connections_errors_total": 3.0,
        "gpu_fault_aurora_credential_refresh_last_success_age_seconds": 71.5,
    }


def test_idle_wait_is_max_idle_plus_margin_and_never_past_the_budget() -> None:
    assert ha009.idle_wait_seconds(300, budget=480) == 360
    assert ha009.idle_wait_seconds(900, budget=480) == 480


def _observation(
    pods: list[str], *, healthz: int = 200, errors: float = 3.0, auth_failures: int = 0
) -> dict:
    return {
        "started_at": "2026-09-08T10:00:00+00:00",
        "finished_at": "2026-09-08T10:02:00+00:00",
        "samples": {
            pod: [
                {
                    "healthz_status": healthz,
                    "metrics": {
                        "gpu_fault_postgres_pool_size": 2.0,
                        "gpu_fault_postgres_pool_connections_errors_total": errors,
                    },
                }
                for _ in range(3)
            ]
            for pod in pods
        },
        "auth_failures_in_logs": {pod: auth_failures for pod in pods},
    }


def test_rotation_errors_reuse_ha005_continuity_and_skip_disabled_roles() -> None:
    before = {
        "gpu-fault-api-ha": _deployment(1, ["a1", "a2", "a3"]),
        "gpu-fault-control-worker": _deployment(1, ["w1", "w2", "w3"]),
        "gpu-fault-telemetry-spool-worker": _deployment(1, [], replicas=0),
    }
    # Path A: nothing rolls. The snapshot after the case equals the baseline.
    after = before
    pods = ["p-a1", "p-a2", "p-a3", "p-w1", "p-w2", "p-w3"]
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
        first_job={"logs": ["rotated=True restarted=False"]},
        second_job={"logs": ["rotated=False restarted=False"]},
        deployments_before=before,
        deployments_after=after,
        propagation={"digest": "d2", "pods": {pod: "d2" for pod in pods}},
        idle_observation=_observation(pods),
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

    # The refresher must not have rolled anything (the old PASS shape).
    rolled = {
        **common,
        "deployments_after": {
            **before,
            "gpu-fault-api-ha": _deployment(2, ["b1", "b2", "b3"]),
        },
        "first_job": {"logs": ["rotated=True restarted=True"]},
    }
    errors = ha009.rotation_errors(final_probe=probe, **rolled)
    assert any("generation" in item for item in errors), (
        "the api-ha rollout must surface as a generation error"
    )
    assert any("restarted=False" in item for item in errors), (
        "the first Job log must show the restart happened"
    )

    # The projected file must have caught up in every Pod.
    stale = {
        **common,
        "propagation": {
            "digest": "d2",
            "pods": {**{p: "d2" for p in pods}, "p-w3": "d1"},
        },
    }
    assert any(
        "p-w3" in item for item in ha009.rotation_errors(final_probe=probe, **stale)
    ), "the lagging Pod must be named in the propagation error"

    # H1-5: after max_idle every Pod still serves and no reconnect was refused.
    sick = {**common, "idle_observation": _observation(pods, healthz=503)}
    assert any(
        "healthz" in item for item in ha009.rotation_errors(final_probe=probe, **sick)
    ), "a 503 healthz after max_idle must fail the case"
    refused = {**common, "idle_observation": _observation(pods, auth_failures=2)}
    assert any(
        "authentication" in item
        for item in ha009.rotation_errors(final_probe=probe, **refused)
    ), "refused reconnects must fail the case"
    unrendered = {**common, "idle_observation": _observation(pods)}
    for samples in unrendered["idle_observation"]["samples"].values():
        for sample in samples:
            sample["metrics"].pop("gpu_fault_postgres_pool_connections_errors_total")
    assert any(
        "connections_errors_total" in item
        for item in ha009.rotation_errors(final_probe=probe, **unrendered)
    ), "a missing pool error counter must fail the case"


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
