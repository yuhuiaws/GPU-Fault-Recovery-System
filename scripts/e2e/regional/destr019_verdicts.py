"""Pure verdict functions and constants of GF-REGIONAL-DESTR-019.

The case proves ARCH-C3/C4/C5 on a real node: after a Node Agent restart on
an idle node the Agent comes back on the *same* generation (same boot, same
incarnation), ``/healthz`` reports the ledger writable with fresh heartbeat
and zeroed counters, and the next real command -- the Fabric Manager restart
DESTR-010 already proved -- leaves a full audit trail: three structured
journald lines (accepted/started/completed) carrying the nine identifier
fields and nothing else, one ledger row keyed ``(command_id, attempt=1)`` with
every audit column, and the counters moving by exactly one. A scratch ledger
built from the pre-C4 schema is migrated in place by the deployed wheel.

What the case deliberately does *not* do is restart the Agent while a
maintenance command is in flight. ``NodeActionLedger._interrupt_in_progress``
marks such a row INTERRUPTED and ``step_execution._fold_result`` fails the
step closed with ``manual_confirmation_required``; the escalation classifier
hands such a failure to an operator (no rung is climbed on an unknown outcome),
and there is no on-node stop for it.
See the spec's 局限 for the reasoning.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from gpu_fault.node_agent.ledger import LEDGER_SCHEMA_VERSION
from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog

CASE_ID = "GF-REGIONAL-DESTR-019"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-010"
CONFIRMATION = "DESTR019_RESTART_NODE_AGENT_LEDGER_AUDIT"
EXPECTED_XID = 45
COMMAND_OPERATION = "RESTART_FABRIC_MANAGER"
EXPECTED_STEPS = ("FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER")
EXPECTED_OWNERS = ("gpu-fault-control-plane", "gpu-fault-node-agent")
FORBIDDEN_OPERATIONS = frozenset(
    {
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
        "QUIESCE_GPU_SERVICES",
        "RESTORE_GPU_SERVICES",
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
    }
)
# ARCH-C3: the identifier fields every command line carries, and the keys that
# must never appear (parameters, UUID lists, signatures, secrets).
LOG_IDENTIFIER_FIELDS = (
    "command_id",
    "incident_id",
    "workflow_request_id",
    "operation",
    "node_id",
    "fencing_token",
    "agent_generation",
    "gpu_count",
    "attempt",
)
LOG_FORBIDDEN_FIELDS = ("parameters", "gpu_uuids", "signature", "secret", "token")
SUCCESS_PHASES = ("accepted", "started", "completed")
# ARCH-C4: the audit columns of one ledger row and the schema version.
AUDIT_COLUMNS = (
    "attempt",
    "state",
    "operation",
    "started_at",
    "completed_at",
    "incident_id",
    "workflow_request_id",
    "fencing_token",
    "gpu_uuids",
    "parameters_digest",
    "signature_digest",
    "exit_code",
    # v3: the agent incarnation that took the attempt. Nullable -- NULL on rows
    # written before the column existed and on attempts accepted before the
    # agent's first heartbeat -- so the row check only types it.
    "agent_generation",
)
# ``LEDGER_SCHEMA_VERSION`` is the node agent's own: the verdict refuses any
# other ``user_version``, so a literal here lagged every schema bump (2 -> 3
# with ``agent_generation``) and failed the live case against a healthy node.
LEDGER_PRIMARY_KEY = ("command_id", "attempt")
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# ARCH-C5: the /healthz payload shape and the counters it reports.
COUNTER_NAMES = ("accepted", "completed", "failed", "rejected")
RESTORE_SECONDS = 180
WORKFLOW_TIMEOUT_SECONDS = 600
AGENT_HEARTBEAT_TIMEOUT_SECONDS = 180
REQUIRED_AGENT_OPERATIONS = {COMMAND_OPERATION}


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def expected_command_id(command: dict[str, Any], *, node: str, generation: int) -> str:
    """What ``_send_action`` names a single-node pinned command."""

    return f"{command.get('idempotency_key')}/{node}/agent-{generation}"


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    node: dict[str, Any],
    agent: dict[str, Any],
    profile: dict[str, Any],
    workloads: list[dict[str, str]],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    recent_event: dict[str, Any] | None,
    host: dict[str, Any],
    tests_passed: bool,
) -> list[str]:
    errors: list[str] = []
    if node.get("ready") != "True":
        errors.append("target node is not Ready")
    if node.get("unschedulable"):
        errors.append("target node is already unschedulable")
    if node.get("taints"):
        errors.append("target node has pre-existing taints")
    if node.get("ownership_annotations"):
        errors.append("target node has pre-existing workflow ownership")
    if workloads:
        errors.append("target node has non-system running Pods")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if not REQUIRED_AGENT_OPERATIONS <= set(agent.get("allowed_operations") or []):
        errors.append("target Node Agent does not allow RESTART_FABRIC_MANAGER")
    if not isinstance(agent.get("generation"), int):
        errors.append("target Node Agent has no generation")
    if profile.get("warnings"):
        errors.append("runtime profile has warnings")
    if processor_queue_backlog(queue):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if recent_event is not None:
        errors.append("target node has a recent XID event")
    if (host.get("agent") or {}).get("ActiveState") != "active":
        errors.append("the Node Agent unit is not active on the host")
    ledger = host.get("ledger") or {}
    if not ledger.get("present"):
        errors.append("the Node Agent ledger is absent on the host")
    if ledger.get("user_version") != LEDGER_SCHEMA_VERSION:
        errors.append(
            f"the deployed ledger schema version is {ledger.get('user_version')}, "
            f"not {LEDGER_SCHEMA_VERSION}; the node bundle predates ARCH-C4"
        )
    if (
        host.get("restore_timer", {})
        and (host.get("restore_timer") or {}).get("ActiveState") == "active"
    ):
        errors.append("an earlier run left the fail-safe restore timer armed")
    if not tests_passed:
        errors.append("focused regression tests failed")
    return errors


# --------------------------------------------------------------------------- #
# /healthz (ARCH-C5)
# --------------------------------------------------------------------------- #
def health_errors(
    health: dict[str, Any],
    *,
    label: str,
    expect_counters: dict[str, int] | None = None,
    heartbeat_required: bool = True,
    max_heartbeat_age_seconds: float | None = None,
) -> list[str]:
    errors: list[str] = []
    if health.get("http_status") != 200:
        errors.append(
            f"{label}: /healthz returned {health.get('http_status')}, not 200"
        )
    payload = health.get("payload") or {}
    if payload.get("status") != "ok":
        errors.append(
            f"{label}: /healthz status is {payload.get('status')!r}, not 'ok'"
        )
    ledger = payload.get("ledger") or {}
    if ledger.get("writable") is not True:
        errors.append(f"{label}: /healthz does not report the ledger writable")
    heartbeat = payload.get("heartbeat") or {}
    if heartbeat_required:
        if heartbeat.get("configured") is not True:
            errors.append(f"{label}: /healthz reports no configured heartbeat")
        if heartbeat.get("last_success_at") is None:
            errors.append(f"{label}: /healthz heartbeat has never succeeded")
        if int(heartbeat.get("consecutive_failures") or 0):
            errors.append(
                f"{label}: heartbeat consecutive_failures is "
                f"{heartbeat.get('consecutive_failures')}"
            )
        age = heartbeat.get("last_success_age_seconds")
        if max_heartbeat_age_seconds is not None and (
            not isinstance(age, (int, float)) or age > max_heartbeat_age_seconds
        ):
            errors.append(
                f"{label}: heartbeat age {age} exceeds {max_heartbeat_age_seconds}s"
            )
    counters = payload.get("counters")
    if not isinstance(counters, dict) or set(counters) != set(COUNTER_NAMES):
        errors.append(f"{label}: /healthz counters are not {list(COUNTER_NAMES)}")
    elif expect_counters is not None:
        actual = {name: int(counters.get(name) or 0) for name in COUNTER_NAMES}
        if actual != expect_counters:
            errors.append(f"{label}: counters {actual} != expected {expect_counters}")
    return errors


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    first = (before.get("payload") or {}).get("counters") or {}
    second = (after.get("payload") or {}).get("counters") or {}
    return {
        name: int(second.get(name) or 0) - int(first.get(name) or 0)
        for name in COUNTER_NAMES
    }


# --------------------------------------------------------------------------- #
# Restart
# --------------------------------------------------------------------------- #
def restart_errors(
    restart: dict[str, Any],
    *,
    baseline_boot_id: str,
) -> list[str]:
    errors: list[str] = []
    before = restart.get("before") or {}
    after = restart.get("after") or {}
    if after.get("ActiveState") != "active":
        errors.append("the Node Agent is not active after the restart")
    if not before.get("MainPID") or before.get("MainPID") == after.get("MainPID"):
        errors.append("the Node Agent MainPID did not change across the restart")
    if before.get("InvocationID") and before.get("InvocationID") == after.get(
        "InvocationID"
    ):
        errors.append("the Node Agent InvocationID did not change across the restart")
    if restart.get("boot_id") != baseline_boot_id:
        errors.append("the node boot id changed; this was not an Agent-only restart")
    if not restart.get("restore_unit"):
        errors.append("no fail-safe restore unit was armed before the restart")
    armed = parse_time(restart.get("armed_at"))
    restarted = parse_time(restart.get("restarted_at"))
    if armed is None or restarted is None or armed > restarted:
        errors.append("the fail-safe was not armed before the restart")
    return errors


def agent_record_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    restarted_at: datetime,
) -> list[str]:
    """Same boot, same incarnation, same generation, fresh heartbeat."""

    errors: list[str] = []
    if after.get("lifecycle_state") != "ACTIVE":
        errors.append(
            f"Agent lifecycle after restart is {after.get('lifecycle_state')}"
        )
    if after.get("generation") != before.get("generation"):
        errors.append(
            f"Agent generation moved from {before.get('generation')} to "
            f"{after.get('generation')}; an Agent-only restart must not "
            "retire the incarnation"
        )
    if after.get("agent_incarnation_id") != before.get("agent_incarnation_id"):
        errors.append("Agent incarnation id changed across an Agent-only restart")
    if after.get("boot_id") != before.get("boot_id"):
        errors.append("Agent boot id changed across an Agent-only restart")
    seen = parse_time(after.get("last_seen_at"))
    if seen is None or seen < restarted_at:
        errors.append("the control plane saw no heartbeat from the restarted Agent")
    return errors


# --------------------------------------------------------------------------- #
# Workflow (the DESTR-010 contract, compacted)
# --------------------------------------------------------------------------- #
def workflow_errors(state: dict[str, Any], *, node: str) -> list[str]:
    errors: list[str] = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    commands = state.get("commands") or []
    if event.get("xid") != EXPECTED_XID:
        errors.append(f"matched event is not XID {EXPECTED_XID}")
    if decision.get("official_action") != "RESTART_FM":
        errors.append("policy did not finalize XID 45 as RESTART_FM")
    steps = [item.get("operation") for item in workflow.get("official_steps") or []]
    if steps != list(EXPECTED_STEPS):
        errors.append(f"workflow steps {steps} != {list(EXPECTED_STEPS)}")
    owners = [
        item.get("execution_owner") for item in workflow.get("official_steps") or []
    ]
    if owners != list(EXPECTED_OWNERS):
        errors.append(f"workflow owners {owners} != {list(EXPECTED_OWNERS)}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow is {workflow.get('status')}, not SUCCEEDED")
    reached = {
        str(item.get("operation")) for item in workflow.get("step_executions") or []
    }
    forbidden = sorted(reached & FORBIDDEN_OPERATIONS)
    if forbidden:
        errors.append(f"workflow reached forbidden operations: {forbidden}")
    if len(commands) != 1 or commands[0].get("status") != "SUCCEEDED":
        errors.append("remote Fabric Manager command is not uniquely SUCCEEDED")
    if list(incident.get("node_ids") or []) != [node]:
        errors.append(f"incident covers {incident.get('node_ids')}, not just {node}")
    return errors


# --------------------------------------------------------------------------- #
# journald (ARCH-C3)
# --------------------------------------------------------------------------- #
def journal_errors(
    lines: list[dict[str, Any]],
    *,
    command_id: str,
    incident_id: str,
    workflow_request_id: str,
    node: str,
    fencing_token: int,
    generation: int,
) -> list[str]:
    errors: list[str] = []
    phases = [item.get("phase") for item in lines]
    if phases != list(SUCCESS_PHASES):
        errors.append(
            f"journal phases for {command_id} are {phases}, not {list(SUCCESS_PHASES)}"
        )
    expected = {
        "command_id": command_id,
        "incident_id": incident_id,
        "workflow_request_id": workflow_request_id,
        "operation": COMMAND_OPERATION,
        "node_id": node,
        "fencing_token": str(fencing_token),
        "agent_generation": str(generation),
        "attempt": "1",
    }
    for item in lines:
        fields = item.get("fields") or {}
        missing = [name for name in LOG_IDENTIFIER_FIELDS if name not in fields]
        if missing:
            errors.append(f"{item.get('phase')} line is missing fields {missing}")
        for name, value in expected.items():
            if name in fields and fields[name] != value:
                errors.append(
                    f"{item.get('phase')} line has {name}={fields[name]!r}, "
                    f"expected {value!r}"
                )
        leaked = [
            name
            for name in fields
            if name not in LOG_IDENTIFIER_FIELDS
            and any(token in name.lower() for token in LOG_FORBIDDEN_FIELDS)
        ]
        if leaked:
            errors.append(f"{item.get('phase')} line carries forbidden fields {leaked}")
        if not str(fields.get("gpu_count", "")).isdigit():
            errors.append(f"{item.get('phase')} line has no numeric gpu_count")
    completed = [item for item in lines if item.get("phase") == "completed"]
    for item in completed:
        fields = item.get("fields") or {}
        if fields.get("status") != "SUCCEEDED":
            errors.append(f"completed line reports status={fields.get('status')!r}")
        if not str(fields.get("duration_ms", "")).isdigit():
            errors.append("completed line has no numeric duration_ms")
    return errors


# --------------------------------------------------------------------------- #
# Ledger audit (ARCH-C4)
# --------------------------------------------------------------------------- #
def ledger_schema_errors(audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if audit.get("user_version") != LEDGER_SCHEMA_VERSION:
        errors.append(
            f"ledger user_version is {audit.get('user_version')}, not "
            f"{LEDGER_SCHEMA_VERSION}"
        )
    columns = set(audit.get("columns") or [])
    missing = [name for name in AUDIT_COLUMNS if name not in columns]
    if missing:
        errors.append(f"ledger results table lacks audit columns {missing}")
    if tuple(audit.get("primary_key") or ()) != LEDGER_PRIMARY_KEY:
        errors.append(
            f"ledger primary key is {audit.get('primary_key')}, not "
            f"{list(LEDGER_PRIMARY_KEY)}"
        )
    return errors


def ledger_row_errors(
    audit: dict[str, Any],
    *,
    command_id: str,
    incident_id: str,
    workflow_request_id: str,
    fencing_token: int,
    baseline_interrupted: int,
) -> list[str]:
    errors = ledger_schema_errors(audit)
    rows = [
        row for row in audit.get("rows") or [] if row.get("command_id") == command_id
    ]
    if len(rows) != 1:
        errors.append(f"expected one ledger row for {command_id}, found {len(rows)}")
        return errors
    row = rows[0]
    if row.get("attempt") != 1:
        errors.append(f"ledger attempt is {row.get('attempt')}, not 1")
    if row.get("state") != "SUCCEEDED":
        errors.append(f"ledger state is {row.get('state')}, not SUCCEEDED")
    if row.get("operation") != COMMAND_OPERATION:
        errors.append(f"ledger operation is {row.get('operation')}")
    for name, expected in (
        ("incident_id", incident_id),
        ("workflow_request_id", workflow_request_id),
        ("fencing_token", fencing_token),
    ):
        if row.get(name) != expected:
            errors.append(f"ledger {name} is {row.get(name)!r}, expected {expected!r}")
    started = parse_time(row.get("started_at"))
    completed = parse_time(row.get("completed_at"))
    if started is None or completed is None or completed < started:
        errors.append("ledger row has no ordered started_at/completed_at pair")
    for name in ("parameters_digest", "signature_digest"):
        value = row.get(name)
        if not isinstance(value, str) or HEX_DIGEST.fullmatch(value) is None:
            errors.append(f"ledger {name} is not a sha256 hex digest: {value!r}")
    if row.get("gpu_uuids_present") is not True:
        errors.append("ledger row carries no gpu_uuids column value")
    if row.get("exit_code") is not None:
        errors.append(f"a SUCCEEDED row carries exit_code {row.get('exit_code')!r}")
    generation = row.get("agent_generation")
    if generation is not None and (
        isinstance(generation, bool) or not isinstance(generation, int)
    ):
        errors.append(f"ledger agent_generation is not an integer: {generation!r}")
    if int(audit.get("interrupted_count") or 0) != baseline_interrupted:
        errors.append(
            "the ledger gained INTERRUPTED rows during the case: "
            f"{audit.get('interrupted_count')} != {baseline_interrupted}"
        )
    return errors


def migration_errors(report: dict[str, Any]) -> list[str]:
    """The scratch ledger built from the pre-C4 schema migrated in place."""

    errors: list[str] = []
    before = report.get("before") or {}
    after = report.get("after_open") or {}
    if before.get("user_version") != 0:
        errors.append(
            f"the legacy scratch ledger reports user_version {before.get('user_version')}"
        )
    if tuple(before.get("primary_key") or ()) != ("command_id",):
        errors.append("the legacy scratch ledger is not keyed by command_id alone")
    if after.get("user_version") != report.get("expected_schema_version"):
        errors.append(
            f"opened ledger reports user_version {after.get('user_version')}, not "
            f"{report.get('expected_schema_version')}"
        )
    if tuple(after.get("primary_key") or ()) != LEDGER_PRIMARY_KEY:
        errors.append(f"migrated primary key is {after.get('primary_key')}")
    missing = [
        name for name in AUDIT_COLUMNS if name not in set(after.get("columns") or [])
    ]
    if missing:
        errors.append(f"migrated ledger lacks audit columns {missing}")
    if after.get("row_count") != before.get("row_count"):
        errors.append(
            f"row count changed across migration: {before.get('row_count')} -> "
            f"{after.get('row_count')}"
        )
    if report.get("legacy_table_present"):
        errors.append("results_legacy was left behind after the migration")
    history = report.get("history_before_append") or []
    if len(history) != 1 or history[0].get("attempt") != 1:
        errors.append("a legacy command does not read back as one attempt-1 row")
    if (report.get("readable_legacy_result") or {}).get("status") != "SUCCEEDED":
        errors.append("the legacy result payload is not readable after migration")
    if (
        report.get("fencing_kept") is not True
        or report.get("stale_fencing_refused") is not True
    ):
        errors.append("the fencing table did not survive the migration intact")
    appended = report.get("history_after_append") or []
    if [item.get("attempt") for item in appended] != [1, 2]:
        errors.append(f"appending attempt 2 did not yield [1, 2]: {appended}")
    elif appended[1].get("exit_code") != 7 or appended[1].get("state") != "FAILED":
        errors.append("the appended attempt lost its state or exit code")
    if (report.get("latest_after_append") or {}).get("attempt") != 2:
        errors.append("get() does not return the latest attempt after the append")
    if report.get("final_row_count") != (before.get("row_count") or 0) + 1:
        errors.append("the appended attempt did not add exactly one row")
    if report.get("scratch_removed") is not True:
        errors.append("the scratch ledger was not removed")
    return errors


# --------------------------------------------------------------------------- #
# Node and cleanup
# --------------------------------------------------------------------------- #
def node_errors(baseline: dict[str, Any], final: dict[str, Any]) -> list[str]:
    if final != baseline:
        return ["target Kubernetes Node state differs from baseline"]
    return []


def cleanup_errors(disarm: dict[str, Any], *, baseline_timers: list[str]) -> list[str]:
    errors: list[str] = []
    if (disarm.get("restore_timer") or {}).get("ActiveState") == "active":
        errors.append("the fail-safe restore timer is still armed after cleanup")
    if (disarm.get("agent") or {}).get("ActiveState") != "active":
        errors.append("the Node Agent is not active after cleanup")
    extra = sorted(set(disarm.get("gpu_fault_timers") or []) - set(baseline_timers))
    if extra:
        errors.append(
            f"gpu-fault timers remain that the baseline did not have: {extra}"
        )
    return errors
