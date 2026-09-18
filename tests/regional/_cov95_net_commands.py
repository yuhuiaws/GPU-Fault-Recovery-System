"""Synthetic command-runner transport with observable ownership and timestamps."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net002_command_recovery as net002
from scripts.e2e.regional import run_net003_result_retry as net003
from scripts.e2e.regional import run_net006_lease_loss_withheld_result as net006
from tests.regional._cov95_collect_net import Clock


def command_observations(
    module: Any, run_id: str, now: datetime, executor_id: str
) -> tuple[Any, ...]:
    seed = {
        "command_id": f"remote-{run_id}",
        "workflow_id": f"workflow-{run_id}",
        "incident_id": f"incident-{run_id}",
        "event_id": f"event-{run_id}",
        "registered_agents": [],
        "notification_id": f"notification-{run_id}",
        "deduplication_key": f"{run_id}/result-retry",
    }
    state = {
        "claimed_total": 1 if module is net003 else 2,
        "reported_failures": 1 if module is net002 else 0,
        "unexpected_failures": 0,
        "lease_renewal_failures": 0 if module is net003 else 3,
        "lease_lost_total": 1,
        "results_withheld_total": 1,
        "cancellations_observed_total": 0,
        "barrier_unavailable_holds_total": 0,
        "transport_retries_total": 1,
        "last_successful_claim_at": now.isoformat(),
        "executor_id": executor_id,
    }
    ready = {
        "owner": net006.verdicts.OWNER if module is net006 else "gpu-fault-net-test",
        "executor_id": executor_id,
        "cluster_id": "perf-cap-000",
        "block_rollback_seconds": 150 if module is net006 else 100,
        "drop_rollback_seconds": 10,
        "http_timeout_seconds": 15
        if module is net006
        else 180
        if module is net002
        else 30,
        "action_hold_seconds": 100,
        "lease_seconds": 60,
        "lease_failure_limit": 3,
        "proxy_mode": "refuse-while-blocked",
        "action_requires_network_block": True,
        "result_submission_gate": True,
        "result_connection_reset": True,
        "response_loss_mode": "forward-then-reset",
        "response_quiet_seconds": 2,
        "replay_delay_seconds": 5,
        "terminal_result_replays": 1,
        "result_retry_owner": "product-executor",
    }
    leased = {
        "command_id": seed["command_id"],
        "status": "LEASED",
        "lease_expires_at": (now - timedelta(seconds=1)).isoformat(),
        "last_lease_owner": executor_id,
        "status_source": None,
    }
    final = {
        "command_id": seed["command_id"],
        "status": "SUCCEEDED",
        "updated_at": now.isoformat(),
        "lease_expires_at": None,
        "last_lease_owner": executor_id,
        "result_details": {
            "cached": module is not net003,
            "notification_id": seed["notification_id"],
        },
    }
    notifications = {
        "objects": {
            "notification": {"count": 1},
            "notification_delivery": {"count": 1},
        },
        "dedup_link_count": 1,
    }
    states = {
        "/state/action-gate-observed.json": {"observed_at_epoch": 1000.0},
        "/state/action-returned.json": {"observed_at_epoch": 1100.0, "cached": False},
        "/state/lease-guard-observed.json": {"reason": "lease renewal failed 3 times"},
        "/state/claim-state.json": {
            "executor_id": executor_id,
            "counters": deepcopy(state),
        },
        "/state/ledger.json": {"physical_count": 1, "keys": ["workflow/0"]},
        "/state/result-submit-waiting.json": {"command_id": seed["command_id"]},
        "/state/result-submit-released.json": {"command_id": seed["command_id"]},
        "/state/result-submit-started.json": {"command_id": seed["command_id"]},
        "/state/result-interrupted.json": {
            "first_post_succeeded": False,
            "exception": "ConnectionResetError",
        },
        "/state/result-replays.json": {
            "count": 1,
            "retry_owner": "product-executor",
            "replay_sent_at_epoch": now.timestamp() + 5,
            "responses": [{"status": "SUCCEEDED", "updated_at": now.isoformat()}],
        },
        "/state/drop-observed.json": {
            "connection_reset": True,
            "request_forwarded": True,
            "upstream_response_bytes": 1024,
        },
        "/state/rollback.json": {"automatic": True},
    }
    if module is net002:
        ready.update(
            first_result_submission_receipt=True,
            result_gate_closes_caller_transport_pool=True,
        )
        released_at = now.timestamp() + net002.BLOCK_SECONDS
        states["/state/result-submit-waiting.json"]["observed_at_epoch"] = (
            now.timestamp() + 5
        )
        states["/state/result-submit-released.json"]["observed_at_epoch"] = released_at
        states["/state/first-result-submission.json"] = {
            "command_id": seed["command_id"],
            "submission_index": 1,
            "status_code": 409,
            "stale_lease_reason": net002.STALE_LEASE_DETAIL,
            "gate_released_at_epoch": released_at,
            "submitted_at_epoch": released_at + 0.1,
            "observed_at_epoch": released_at + 0.2,
            "caller_transport_pool_closed": True,
        }
    if module is net003:
        seed.update(cluster_id="perf-cap-000", idempotency_key="workflow/0")
        ready["lease_seconds"] = net003.LEASE_SECONDS
        identity = {
            "command_id": seed["command_id"],
            "workflow_request_id": seed["workflow_id"],
            "incident_id": seed["incident_id"],
            "cluster_id": seed["cluster_id"],
            "idempotency_key": seed["idempotency_key"],
        }
        leased.update(
            **identity,
            lease_expires_at=(
                now + timedelta(seconds=net003.LEASE_SECONDS)
            ).isoformat(),
        )
        final.update(identity)
        states["/state/action-gate-observed.json"]["idempotency_key"] = seed[
            "idempotency_key"
        ]
        states["/state/action-gate-observed.json"]["observed_at_epoch"] = (
            now.timestamp()
        )
        states["/state/result-interrupted.json"].update(
            command_id=seed["command_id"], status_code=None
        )
        states["/state/result-replays.json"]["command_id"] = seed["command_id"]
        states["/state/result-replays.json"]["responses"][0]["command_id"] = seed[
            "command_id"
        ]

    return seed, state, ready, leased, final, notifications, states


@pytest.fixture(params=(net002, net003, net006), ids=("net002", "net003", "net006"))
def command_runner(request: Any, monkeypatch: Any, tmp_path: Path) -> Any:
    module = request.param
    transport = module.seeded if module is net006 else module.fixture
    clock = Clock()
    if module is not net003:
        monkeypatch.setattr(module, "time", clock)
    run_id = module.CASE_ID.lower() + "-fixture"
    now = datetime.now(timezone.utc)
    if module is net002:
        started = clock.now

        class Net002DateTime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:
                return now + timedelta(seconds=clock.now - started)

        monkeypatch.setattr(module, "datetime", Net002DateTime)
    executor_id = "net006-test-executor" if module is net006 else "net-test-executor"
    seed, state, ready, leased, final, notifications, states = command_observations(
        module, run_id, now, executor_id
    )
    harness = SimpleNamespace(
        module=module,
        transport=transport,
        clock=clock,
        run_id=run_id,
        root=tmp_path,
        seed=seed,
        ready=ready,
        leased=leased,
        expired=deepcopy(leased),
        final=final,
        state=state,
        states=states,
        notifications=notifications,
        final_notifications=deepcopy(notifications),
        final_notification_result={"count": 1, "status": "SKIPPED"},
        calls=[],
        fail_at=None,
        rollback=False,
        phase="Running",
        cleanup_failure=False,
        cleanup_state=None,
        notification_reads=0,
        deadline=now + timedelta(hours=1),
        snapshot_reads=0,
    )

    def note(name: str, *args: Any) -> None:
        harness.calls.append((name, args))
        if harness.fail_at == name:
            raise RuntimeError(f"{name} fixture failure")

    def create(probe: Any, case_dir: Path, *, run_id: str) -> dict[str, Any]:
        note("probe", probe)
        if module is net003:
            assert probe.environment["NOTIFICATION_ID"] == f"notification-{run_id}"
        return deepcopy(ready)

    def seed_command(run_id: str, **kwargs: Any) -> dict[str, Any]:
        note("seed", run_id, kwargs)
        return deepcopy(seed)

    def snapshot(command_id: str) -> dict[str, Any]:
        note("command-snapshot", command_id)
        harness.snapshot_reads += 1
        if harness.snapshot_reads > 1:
            if module is net003:
                return deepcopy(final)
            if module is net002:
                return deepcopy(harness.expired)
        return deepcopy(leased)

    def wait_command(command_id: str, expected: Any, timeout: int) -> dict[str, Any]:
        note("wait-command", command_id, timeout)
        if callable(expected):
            assert expected({"status": "FAILED"}) is True
            assert expected({"status": "RUNNING"}) is False
            assert expected(final) is True
        else:
            assert expected == "SUCCEEDED"
        return deepcopy(final)

    def wait_state(probe: Any, predicate: Any, timeout: int) -> dict[str, Any]:
        note("wait-state", timeout)
        assert not predicate({}), "empty executor state must never satisfy a poll"
        assert predicate(state), "the intended executor state must satisfy its poll"
        return deepcopy(state)

    def read(probe: Any, path: str) -> dict[str, Any]:
        note("read-state", path)
        return deepcopy(states[path])

    def cpu_python(script: str, *args: str) -> dict[str, Any]:
        if len(args) == 2:
            note("notification", *args)
            harness.notification_reads += 1
            result = deepcopy(notifications)
            if harness.notification_reads > 1:
                result = deepcopy(harness.final_notifications)
                result["objects"]["notification_result"] = deepcopy(
                    harness.final_notification_result
                )
            return result
        if args[0] == run_id:
            return seed_command(run_id)
        note("purge", *args)
        return {"remaining": [], "remaining_links": 0}

    def cleanup(
        probe: Any,
        case_dir: Path,
        run_id: str,
        result: Any,
        current_seed: Any,
        **kwargs: Any,
    ) -> None:
        harness.calls.append(("cleanup", (current_seed,)))
        harness.cleanup_state = deepcopy(kwargs["state"])
        if "purge" in kwargs and current_seed and "notification_id" in current_seed:
            kwargs["purge"](current_seed)
        if harness.cleanup_failure:
            result["verdict"] = "FAIL"
            result["cleanup_error"] = "fixture residual"

    monkeypatch.setattr(transport, "require_environment", lambda: note("environment"))
    monkeypatch.setattr(transport, "run_identity", lambda *a: run_id)
    monkeypatch.setattr(transport, "preflight_residuals", lambda *a: note("preflight"))
    monkeypatch.setattr(
        transport, "register_synthetic_cluster", lambda *a: note("registry")
    )
    monkeypatch.setattr(transport, "create_probe_pod", create)
    monkeypatch.setattr(transport, "seed_command", seed_command)
    monkeypatch.setattr(transport, "command_snapshot", snapshot)
    monkeypatch.setattr(transport, "wait_command", wait_command)
    monkeypatch.setattr(transport, "wait_executor_state", wait_state)
    monkeypatch.setattr(transport, "wait_file", lambda *a: note("wait-file", *a))
    monkeypatch.setattr(transport, "read_state", read)
    monkeypatch.setattr(transport, "touch", lambda *a: note("block", *a))
    monkeypatch.setattr(transport, "remove", lambda *a: note("unblock", *a))
    monkeypatch.setattr(transport, "file_present", lambda *a: harness.rollback)
    monkeypatch.setattr(transport, "pod_phase", lambda *a: harness.phase)
    monkeypatch.setattr(transport, "cleanup", cleanup)
    monkeypatch.setattr(transport, "cpu_python", cpu_python)
    logs = (
        net006.verdicts.LOST_LOG + "\n" + net006.verdicts.WITHHELD_LOG
        if module is net006
        else net003.LOST_RESPONSE_LOG
        if module is net003
        else net002.STALE_LEASE_409 + "\n" + net002.STALE_LEASE_DETAIL
    )
    harness.logs = logs
    monkeypatch.setattr(transport, "pod_logs", lambda *a: harness.logs)
    if module is not net006:
        monkeypatch.setattr(
            transport,
            "evidence_identity",
            lambda cluster: note("identity", cluster)
            or {"release_id": "release-a", "cluster_id": cluster},
        )
    return harness


def run_command(harness: Any, *, predecessor: bool = True) -> int:
    kwargs = (
        {}
        if harness.module is net006
        else {"predecessor": {"valid": predecessor}, "cluster_id": "cluster-a"}
    )
    return harness.module.run_case(harness.root, 1, harness.deadline, **kwargs)
