#!/usr/bin/env python3
"""GF-REGIONAL-NET-003: the control plane commits a result the client never sees.

One seeded ``FREEZE_EVIDENCE`` command and one drill notification for the
synthetic cluster, one probe executor Pod. The probe's loopback proxy forwards
the first result post to the control plane, waits for the control plane's
answer, discards it and resets the client connection. The control plane has
committed the command as SUCCEEDED; the client has a transport error. The
client retries the same terminal result once, and the control plane must
answer idempotently: the command is not reclaimed, the ledger shows one
physical execution, and the drill notification is not duplicated.

Plan-only by default; ``--execute`` needs the exact confirmation and a PASS
from the formal predecessor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import net_command_fixture as fixture  # noqa: E402
from scripts.e2e.regional import seeded_command_fixture as seeded  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
)
from scripts.perf.regional_capacity_registry import (  # noqa: E402
    STORE_DSN_SNIPPET,
)

CASE_ID = "GF-REGIONAL-NET-003"
CONFIRMATION = "NET003_RESULT_CONNECTION_RESET"
RUN_PREFIX = "net003-"
SCRIPT = Path(__file__).with_name("probes") / "net003_executor.py"
CONFIGMAP = "gpu-fault-net003-script"
POD = "gpu-fault-net003-executor"
OWNER = "gpu-fault-net-test"
OPERATION = "FREEZE_EVIDENCE"
NODE_IDS = ["net003-synthetic-node"]
DROP_ROLLBACK_SECONDS = 10
HTTP_TIMEOUT_SECONDS = 30
LEASE_SECONDS = 60
ACTION_SECONDS = 5
RESPONSE_QUIET_SECONDS = 2.0
REPLAY_DELAY_SECONDS = 5.0
TERMINAL_RESULT_REPLAYS = 1
RESPONSE_LOSS_MODE = "forward-then-reset"
POD_DEADLINE_SECONDS = 600
STALE_LEASE_409 = "rejected request (409)"
LOST_RESPONSE_LOG = "net003 result submission lost its response"

CaseError = fixture.NetCommandError
write_json = fixture.write_json


def probe_definition() -> fixture.NetCommandProbe:
    return fixture.NetCommandProbe(
        case_id=CASE_ID,
        run_prefix=RUN_PREFIX,
        pod=POD,
        configmap=CONFIGMAP,
        owner=OWNER,
        script=SCRIPT,
        environment={
            "DROP_ROLLBACK_SECONDS": str(DROP_ROLLBACK_SECONDS),
            "HTTP_TIMEOUT_SECONDS": str(HTTP_TIMEOUT_SECONDS),
            "LEASE_SECONDS": str(LEASE_SECONDS),
            "RESPONSE_QUIET_SECONDS": str(RESPONSE_QUIET_SECONDS),
            "REPLAY_DELAY_SECONDS": str(REPLAY_DELAY_SECONDS),
        },
        pod_deadline_seconds=POD_DEADLINE_SECONDS,
    )


def renewal_interval_seconds(lease_seconds: int) -> float:
    """``ClusterActionExecutor._renew_lease``'s cadence for one lease length."""

    return min(30.0, lease_seconds / 3)


def timing_errors(
    *,
    lease_seconds: int = LEASE_SECONDS,
    quiet_seconds: float = RESPONSE_QUIET_SECONDS,
    replay_delay_seconds: float = REPLAY_DELAY_SECONDS,
    http_timeout_seconds: float = HTTP_TIMEOUT_SECONDS,
) -> list[str]:
    """Refuse a parameter set that could muddle the verdict.

    The whole result exchange -- the action, the withheld response, the
    replay delay -- has to finish before the executor's first lease renewal,
    because a renewal sent against the now-terminal command is refused with
    409 and would count as a renewal failure the case has no business
    producing. That is why the lease is not shortened to 20s here: a 20s
    lease renews every 6.7s and cannot fit the exchange.
    """

    errors: list[str] = []
    interval = renewal_interval_seconds(lease_seconds)
    exchange = ACTION_SECONDS + quiet_seconds + replay_delay_seconds
    if exchange + 5 >= interval:
        errors.append(
            f"result exchange takes {exchange:.1f}s (+5s margin), not inside the "
            f"{interval:.1f}s renewal interval of a {lease_seconds}s lease; a "
            "renewal against the terminal command would be refused"
        )
    if quiet_seconds + replay_delay_seconds >= http_timeout_seconds:
        errors.append(
            f"quiet {quiet_seconds}s plus replay delay {replay_delay_seconds}s "
            f"is not below the HTTP timeout {http_timeout_seconds}s"
        )
    if replay_delay_seconds < 2:
        errors.append(
            f"replay delay {replay_delay_seconds}s is too short to attribute the "
            "commit to the first post against clock skew"
        )
    return errors


def stop_conditions() -> list[str]:
    return [
        "formal predecessor evidence is not PASS",
        "any preflight or rollout failure",
        "test pod readiness failure",
        "the first result post is not forwarded and its response not withheld",
        "the control plane did not commit the first result before the reset",
        "the client retry of the terminal result is not answered idempotently",
        "the command is reclaimed or a lease renewal is refused",
        "simulated physical execution count differs from one",
        "terminal result replay creates a second notification",
        "executor run loop exits",
        "any cleanup or postflight residual check fails",
    ]


def rollback_contract() -> dict[str, Any]:
    return {
        "drop_marker_auto_release_seconds": DROP_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
        "synthetic_registry_expires_minutes": 30,
        "runner_finally_deletes_test_resources": True,
        "runner_finally_purges_synthetic_state": True,
        "runner_finally_restores_registry": True,
    }


def preflight_metadata(
    attempt: int, maintenance_window_end: datetime
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if now >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    fixture.require_environment()
    timing = timing_errors()
    if timing:
        raise CaseError("; ".join(timing))
    return {
        "observed_at": now.isoformat(),
        "attempt": attempt,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "region": fixture.AWS_REGION,
        "control_namespace": fixture.CONTROL_NAMESPACE,
        "dataplane_context": fixture.DATAPLANE_CONTEXT,
        "dataplane_namespace": fixture.NAMESPACE,
        "cluster_id": fixture.SYNTHETIC_CLUSTER_ID,
        "node_ids": NODE_IDS,
        "operation": OPERATION,
        "destructive": False,
        "network_scope": (
            "one test-pod result connection reset on loopback proxy, after the "
            "control plane answered"
        ),
        "timing": {
            "lease_seconds": LEASE_SECONDS,
            "response_quiet_seconds": RESPONSE_QUIET_SECONDS,
            "replay_delay_seconds": REPLAY_DELAY_SECONDS,
            "renewal_interval_seconds": renewal_interval_seconds(LEASE_SECONDS),
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        },
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


# --------------------------------------------------------------------------- #
# Store scripts specific to NET-003 (a notification rides along with the seed)
# --------------------------------------------------------------------------- #
_SEED_COMMAND = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand

(
    run_id,
    cluster_id,
    owner,
    notification_id,
    deduplication_key,
    raw_node_ids,
    raw_lease_seconds,
) = sys.argv[1:]
node_ids = [item for item in raw_node_ids.split(",") if item]
lease_seconds = int(raw_lease_seconds)
incident_id = f"incident-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
command_id = f"remote-{run_id}"
step = WorkflowStepSpec(
    operation=WorkflowOperation.FREEZE_EVIDENCE,
    execution_owner=owner,
    node_ids=node_ids,
)
incident = FaultIncident(
    incident_id=incident_id,
    event_id=f"event-{run_id}",
    event_type="NET_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=node_ids,
    policy_version="net-acceptance/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id=workflow_id,
    fencing_token=1,
    drill_id=run_id,
)
# Leased to the probe's seed identity exactly like the shared seed
# (seeded_command_fixture, commit 87c2697): the deployed dispatcher wakes on
# the workflow save and claims an unleased PENDING row within the second,
# drives its step against a node no cluster carries, fails it, and the orphan
# sweep then cancels the command before the probe can claim it.
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.PENDING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step],
    execution_owner_id=f"{owner}-seed",
    execution_epoch=1,
    execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds),
)
notification = AdvisoryNotification(
    notification_id=notification_id,
    deduplication_key=deduplication_key,
    cluster_name=cluster_id,
    incident_id=incident_id,
    subject="NET-003 result retry drill",
    body_text="Synthetic non-destructive result retry acceptance notification.",
    support_case_draft="No provider action required; acceptance drill only.",
    drill_id=run_id,
    category="NET_ACCEPTANCE",
    not_before=datetime.now(timezone.utc) + timedelta(minutes=10),
)
command = RemoteActionCommand(
    command_id=command_id,
    cluster_id=cluster_id,
    workflow_request_id=workflow_id,
    incident_id=incident_id,
    step_index=0,
    fencing_token=1,
    idempotency_key=f"{workflow_id}/0/FREEZE_EVIDENCE",
    step=step,
    workflow=workflow,
    incident=incident,
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
store.save_notification_if_absent(notification)
store.ensure_remote_command(command)
agents = [item.node_id for item in store.list_agents(cluster_id)]
print(json.dumps({
    "incident_id": incident_id,
    "event_id": incident.event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "cluster_id": cluster_id,
    "notification_id": notification_id,
    "deduplication_key": deduplication_key,
    "node_ids": node_ids,
    "registered_agents": agents,
}, sort_keys=True))
"""


def seed_command(run_id: str) -> dict[str, Any]:
    return fixture.cpu_python(
        _SEED_COMMAND,
        run_id,
        fixture.SYNTHETIC_CLUSTER_ID,
        OWNER,
        f"notification-{run_id}",
        f"{run_id}/result-retry",
        ",".join(NODE_IDS),
        str(seeded.SEED_LEASE_SECONDS),
    )


_NOTIFICATION_SNAPSHOT = (
    STORE_DSN_SNIPPET
    + r"""
import json
import sys
import psycopg

notification_id, deduplication_key = sys.argv[1:]
objects = {}
with psycopg.connect(store_dsn()) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT kind, payload->>'status'
            FROM gpu_fault_objects
            WHERE key=%s
              AND kind IN (
                  'notification',
                  'notification_delivery',
                  'notification_result'
              )
            ORDER BY kind
            ''',
            (notification_id,),
        )
        for kind, status in cursor.fetchall():
            objects[kind] = {
                "count": objects.get(kind, {}).get("count", 0) + 1,
                "status": status,
            }
        cursor.execute(
            '''
            SELECT count(*)
            FROM gpu_fault_links
            WHERE kind='notification_dedup'
              AND key=%s
              AND value=%s
            ''',
            (deduplication_key, notification_id),
        )
        link_count = int(cursor.fetchone()[0])
print(json.dumps({"objects": objects, "dedup_link_count": link_count}, sort_keys=True))
"""
)


def notification_snapshot(
    notification_id: str, deduplication_key: str
) -> dict[str, Any]:
    return fixture.cpu_python(
        _NOTIFICATION_SNAPSHOT, notification_id, deduplication_key
    )


_PURGE_SEED = (
    STORE_DSN_SNIPPET
    + r"""
import json
import sys
import psycopg

command_id, workflow_id, incident_id, notification_id, dedup_key, event_id = (
    sys.argv[1:]
)
deleted = {}
with psycopg.connect(store_dsn(), autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links WHERE kind='incident_by_event' "
        "AND key=%s AND value=%s",
        (event_id, incident_id),
    )
    deleted[f"incident_by_event/{event_id}"] = cursor.rowcount
    cursor.execute(
        "DELETE FROM gpu_fault_links WHERE kind='notification_dedup' "
        "AND key=%s AND value=%s",
        (dedup_key, notification_id),
    )
    deleted[f"notification_dedup/{dedup_key}"] = cursor.rowcount
    items = (
        ("remote_command", command_id),
        ("workflow", workflow_id),
        ("incident", incident_id),
        ("notification_result", notification_id),
        ("notification_delivery", notification_id),
        ("notification", notification_id),
    )
    for kind, key in items:
        cursor.execute(
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s", (kind, key)
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        "SELECT kind, key FROM gpu_fault_objects WHERE (kind, key) IN ("
        "('remote_command', %s), ('workflow', %s), ('incident', %s), "
        "('notification_result', %s), ('notification_delivery', %s), "
        "('notification', %s)) ORDER BY kind, key",
        (
            command_id,
            workflow_id,
            incident_id,
            notification_id,
            notification_id,
            notification_id,
        ),
    )
    remaining = [{"kind": kind, "key": key} for kind, key in cursor.fetchall()]
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_links WHERE "
        "(kind='notification_dedup' AND key=%s AND value=%s) "
        "OR (kind='incident_by_event' AND key=%s AND value=%s)",
        (dedup_key, notification_id, event_id, incident_id),
    )
    remaining_links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining": remaining,
    "remaining_links": remaining_links,
}, sort_keys=True))
"""
)


def purge_seed(seed: dict[str, Any]) -> dict[str, Any]:
    result = fixture.cpu_python(
        _PURGE_SEED,
        str(seed["command_id"]),
        str(seed["workflow_id"]),
        str(seed["incident_id"]),
        str(seed["notification_id"]),
        str(seed["deduplication_key"]),
        str(seed["event_id"]),
    )
    if result.get("remaining") or result.get("remaining_links"):
        raise CaseError(f"seed cleanup left residual state: {result}")
    return result


# --------------------------------------------------------------------------- #
# The case
# --------------------------------------------------------------------------- #
def _run_net003_case(
    probe: fixture.NetCommandProbe,
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict[str, Any],
) -> dict[str, Any]:
    notification_id = f"notification-{run_id}"
    deduplication_key = f"{run_id}/result-retry"
    preflight = preflight_metadata(attempt, maintenance_window_end)
    write_json(case_dir / "preflight.json", preflight)
    fixture.preflight_residuals(probe, case_dir)
    fixture.register_synthetic_cluster(case_dir, run_id)
    probe_with_notification = fixture.NetCommandProbe(
        case_id=probe.case_id,
        run_prefix=probe.run_prefix,
        pod=probe.pod,
        configmap=probe.configmap,
        owner=probe.owner,
        script=probe.script,
        environment={**probe.environment, "NOTIFICATION_ID": notification_id},
        pod_deadline_seconds=probe.pod_deadline_seconds,
    )
    ready = fixture.create_probe_pod(probe_with_notification, case_dir)
    seed = seed_command(run_id)
    state["seed"] = seed
    write_json(case_dir / "seed.json", seed)
    if (
        seed.get("notification_id") != notification_id
        or seed.get("deduplication_key") != deduplication_key
    ):
        raise CaseError("seed notification identity is inconsistent")
    if seed.get("registered_agents"):
        raise CaseError(
            "the synthetic cluster has registered Node Agents; the hard stop "
            f"does not hold: {seed['registered_agents']}"
        )
    notification_baseline = notification_snapshot(notification_id, deduplication_key)
    write_json(case_dir / "notification-baseline.json", notification_baseline)
    fixture.wait_file(probe, "/state/action-started", 120)
    leased = fixture.command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "leased-command.json", leased)
    if leased.get("status") != "LEASED" or not leased.get("lease_expires_at"):
        raise CaseError(f"command was not actively leased before injection: {leased}")
    fixture.wait_file(probe, "/state/result-submit-started.json", 30)
    result_submit_started = fixture.read_state(
        probe, "/state/result-submit-started.json"
    )
    write_json(case_dir / "result-submit-started.json", result_submit_started)
    fixture.wait_file(probe, "/state/drop-observed.json", 60)
    drop_observed = fixture.read_state(probe, "/state/drop-observed.json")
    write_json(case_dir / "drop-observed.json", drop_observed)
    # The proxy wrote drop-observed only after the control plane answered, so
    # this snapshot shows what the control plane committed before the client
    # saw anything -- whether it lands before or after the client's replay.
    committed = fixture.command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "committed-command.json", committed)
    fixture.wait_file(probe, "/state/result-interrupted.json", 30)
    result_interrupted = fixture.read_state(probe, "/state/result-interrupted.json")
    write_json(case_dir / "result-interrupted.json", result_interrupted)
    fixture.wait_file(
        probe, "/state/result-replays.json", int(REPLAY_DELAY_SECONDS) + 60
    )
    result_replays = fixture.read_state(probe, "/state/result-replays.json")
    write_json(case_dir / "result-replays.json", result_replays)
    final = fixture.wait_command(str(seed["command_id"]), "SUCCEEDED", 60)
    write_json(case_dir / "final-command.json", final)
    notification_final = notification_snapshot(notification_id, deduplication_key)
    write_json(case_dir / "notification-final.json", notification_final)
    executor_state = fixture.wait_executor_state(
        probe, lambda item: bool(item.get("last_successful_claim_at")), 30
    )
    write_json(case_dir / "executor-state.json", executor_state)
    ledger = fixture.read_state(probe, "/state/ledger.json")
    write_json(case_dir / "ledger.json", ledger)
    logs = fixture.pod_logs(probe)
    (case_dir / "executor.log").write_text(logs, encoding="utf-8")
    (case_dir / "executor.log").chmod(0o600)
    rollback_triggered = fixture.file_present(probe, "/state/rollback.json")
    errors = net003_errors(
        ready=ready,
        leased=leased,
        committed=committed,
        final=final,
        result_interrupted=result_interrupted,
        result_replays=result_replays,
        executor_state=executor_state,
        ledger=ledger,
        logs=logs,
        rollback_triggered=rollback_triggered,
        drop_observed=drop_observed,
        notification_id=notification_id,
        notification_baseline=notification_baseline,
        notification_final=notification_final,
    )
    phase = fixture.pod_phase(probe)
    if phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "http_timeout_seconds": ready.get("http_timeout_seconds"),
        "network_interruption": {
            "type": "single-result-connection-reset-after-response",
            "result_submit_started": result_submit_started,
            "drop_observed": drop_observed,
            "result_interrupted": result_interrupted,
            "watchdog_rollback_triggered": rollback_triggered,
        },
        "leased_command": leased,
        "committed_command": committed,
        "command": final,
        "result_replays": result_replays,
        "executor_state": executor_state,
        "ledger": ledger,
        "notification_baseline": notification_baseline,
        "notification_final": notification_final,
        "pod_phase": phase,
    }


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def net003_errors(
    *,
    ready: dict[str, Any],
    leased: dict[str, Any],
    committed: dict[str, Any],
    final: dict[str, Any],
    result_interrupted: dict[str, Any],
    result_replays: dict[str, Any],
    executor_state: dict[str, Any],
    ledger: dict[str, Any],
    logs: str,
    rollback_triggered: bool,
    drop_observed: dict[str, Any],
    notification_id: str,
    notification_baseline: dict[str, Any],
    notification_final: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if ledger.get("physical_count") != 1:
        errors.append("physical action count is not one")
    if len(ledger.get("keys") or []) != 1:
        errors.append("idempotency ledger does not contain exactly one key")
    if int(executor_state.get("claimed_total") or 0) != 1:
        errors.append(
            "the committed command was claimed again; the control plane did not "
            "hold the first result"
        )
    if int(executor_state.get("reported_failures") or 0) != 0:
        errors.append("a result post was rejected; the replay was not idempotent")
    if int(executor_state.get("unexpected_failures") or 0):
        errors.append("executor recorded an unexpected failure")
    if int(executor_state.get("lease_renewal_failures") or 0) != 0:
        errors.append("a lease renewal was refused during the result exchange")
    if ready.get("drop_rollback_seconds") != DROP_ROLLBACK_SECONDS:
        errors.append("connection-drop rollback timer is not configured")
    if ready.get("lease_seconds") != LEASE_SECONDS:
        errors.append("probe executor lease length is not the approved value")
    if ready.get("result_connection_reset") is not True:
        errors.append("result connection reset is not configured")
    if ready.get("response_loss_mode") != RESPONSE_LOSS_MODE:
        errors.append("probe proxy does not forward the request before the reset")
    if ready.get("terminal_result_replays") != TERMINAL_RESULT_REPLAYS:
        errors.append("terminal result replay count is not configured")
    errors.extend(
        timing_errors(
            lease_seconds=int(ready.get("lease_seconds") or 0),
            quiet_seconds=float(ready.get("response_quiet_seconds") or 0),
            replay_delay_seconds=float(ready.get("replay_delay_seconds") or 0),
            http_timeout_seconds=float(ready.get("http_timeout_seconds") or 0),
        )
    )
    if drop_observed.get("connection_reset") is not True:
        errors.append("proxy did not record a connection reset")
    if drop_observed.get("request_forwarded") is not True:
        errors.append("proxy reset the client before forwarding the result post")
    if int(drop_observed.get("upstream_response_bytes") or 0) <= 0:
        errors.append("proxy reset the client before the control plane answered")
    if result_interrupted.get("first_post_succeeded") is not False or not (
        result_interrupted.get("exception")
    ):
        errors.append("the client did not see a transport error on the first post")
    if rollback_triggered:
        errors.append("connection-drop marker required watchdog rollback")
    if committed.get("status") != "SUCCEEDED":
        errors.append(
            "the control plane did not commit the first result before the reset: "
            f"{committed.get('status')}"
        )
    if committed.get("lease_expires_at") is not None:
        errors.append("the committed command still carries a lease")
    if leased.get("status") != "LEASED":
        errors.append(f"command was not leased before the result post: {leased}")
    replay_sent_at = result_replays.get("replay_sent_at_epoch")
    updated_at = parse_time(final.get("updated_at"))
    if not isinstance(replay_sent_at, (int, float)) or updated_at is None:
        errors.append("replay time or command update time is missing")
    else:
        lead = float(replay_sent_at) - updated_at.timestamp()
        if lead < REPLAY_DELAY_SECONDS / 2:
            errors.append(
                f"the command was updated {lead:.1f}s before the replay was sent; "
                "the commit is not attributable to the first post"
            )
    responses = result_replays.get("responses") or []
    if result_replays.get("count") != TERMINAL_RESULT_REPLAYS or len(responses) != (
        TERMINAL_RESULT_REPLAYS
    ):
        errors.append("terminal result was not replayed exactly once")
    for response in responses:
        if response.get("status") != "SUCCEEDED":
            errors.append(f"replay was not answered SUCCEEDED: {response}")
        replay_updated = parse_time(response.get("updated_at"))
        if updated_at is not None and replay_updated != updated_at:
            errors.append("the replay rewrote the committed command")
    if LOST_RESPONSE_LOG not in logs:
        errors.append("executor log has no lost-response transport interruption")
    if STALE_LEASE_409 in logs:
        errors.append("NET-003 unexpectedly followed the stale-result 409 path")
    details = final.get("result_details") or {}
    if details.get("cached") is not False:
        errors.append("the committed result is not the first physical execution")
    if details.get("notification_id") != notification_id:
        errors.append("final result lost the notification identity")
    baseline_objects = notification_baseline.get("objects") or {}
    final_objects = notification_final.get("objects") or {}
    if (baseline_objects.get("notification") or {}).get("count") != 1:
        errors.append("notification baseline does not contain exactly one object")
    if (baseline_objects.get("notification_delivery") or {}).get("count") != 1:
        errors.append("notification baseline has no single delivery row")
    if (baseline_objects.get("notification_result") or {}).get("count", 0) != 0:
        errors.append("notification was sent before result completion")
    if notification_baseline.get("dedup_link_count") != 1:
        errors.append("notification baseline dedup link count is not one")
    for kind in ("notification", "notification_delivery", "notification_result"):
        if (final_objects.get(kind) or {}).get("count") != 1:
            errors.append(f"final {kind} count is not one")
    if (final_objects.get("notification_result") or {}).get("status") != "SKIPPED":
        errors.append("drill notification was not safely suppressed")
    if notification_final.get("dedup_link_count") != 1:
        errors.append("terminal replay created a second dedup link")
    return errors


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    *,
    predecessor: dict[str, Any],
    cluster_id: str | None,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = fixture.run_identity(run_dir, attempt, "net003")
    probe = probe_definition()
    result: dict[str, Any] = {"case_id": CASE_ID, "verdict": "FAIL"}
    state: dict[str, Any] = {"seed": {}}
    try:
        fixture.require_predecessor(predecessor)
        result = _run_net003_case(
            probe, case_dir, run_id, attempt, maintenance_window_end, state
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        fixture.cleanup(
            probe, case_dir, run_id, result, state["seed"], purge=purge_seed
        )
    result["predecessor"] = predecessor
    try:
        result.update(fixture.evidence_identity(cluster_id))
    except Exception as exc:  # noqa: BLE001 - unbound evidence is not a PASS
        result["identity_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def plan_details(predecessor: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "predecessor": predecessor,
        "synthetic_cluster_id": fixture.SYNTHETIC_CLUSTER_ID,
        "seeded_operation": OPERATION,
        "seeded_node_ids": NODE_IDS,
        "network_scope": (
            "one test-Pod result connection reset, after the control plane answered"
        ),
        "response_loss_mode": RESPONSE_LOSS_MODE,
        "drop_marker_rollback_seconds": DROP_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
        "terminal_result_replays": TERMINAL_RESULT_REPLAYS,
        "timing": {
            "lease_seconds": LEASE_SECONDS,
            "response_quiet_seconds": RESPONSE_QUIET_SECONDS,
            "replay_delay_seconds": REPLAY_DELAY_SECONDS,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        },
        "mutations": [
            "temporary synthetic registry entry",
            "controlled CPU registry rollouts",
            "temporary GPU probe Pod and ConfigMap",
        ],
        "hard_stop": (
            "the seeded command belongs to a synthetic cluster with no Node "
            "Agents and names a node id no cluster carries"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded NET-003 acceptance: the control plane commits a "
            "result whose response the client lost, and the client's retry is "
            "answered idempotently."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--cluster-id",
        default=os.getenv("GPU_FAULT_CLUSTER_ID", ""),
        help="the site's GPU cluster id recorded in the evidence for binding",
    )
    return value


def main() -> int:
    return fixture.run_main(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser=parser,
        plan_details=plan_details,
        run_case=run_case,
    )


if __name__ == "__main__":
    raise SystemExit(main())
