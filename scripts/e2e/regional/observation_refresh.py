"""The control plane's observation refresh: the other node action a reset window meets.

The control plane grades every current Agent's required collectors once a
minute (``app/collector_silence.notify_silent_collectors``). A node with an
UNKNOWN collector -- no fresh clean batch, or a collection error newer than it
-- gets one bounded fresh snapshot (``observation_health.
request_observation_refresh``): a SITE_OBSERVABILITY node-health finding of its
own incident, at most one per node and hour, whose workflow is FREEZE_EVIDENCE
-> TRIGGER_HEALTH_SNAPSHOT -> VALIDATE_HOST. A reset chain's
QUIESCE_GPU_SERVICES stops gpu-fault-metrics-collector and
gpu-fault-host-collector on purpose, so the scan can ask for that refresh while
the chain's window is open on the node.

TRIGGER_HEALTH_SNAPSHOT restarts those collectors, so it is one of the Agent's
own GPU activities (``node_agent.gpu_activity``): while another incident's
quiesce window is open the Agent refuses it at admission -- HTTP 409
GPU_MAINTENANCE_WINDOW_OPEN, retryable, no ledger row, one ``rejected`` on
``/healthz`` and one ``node action rejected`` journald line per submission --
and the control plane submits the same command again until
RESTORE_GPU_SERVICES has closed the window. Then it runs once. Live
(GF-REGIONAL-DESTR-019 on HyperPod, release 007595f77e11): 41 refusals while
the chain's window was open, then attempt 1 during the restore's settle.

What a case may accept is pinned by the product's own record ids: the event,
the incident, the workflow and so the node action's command id derive from the
cluster, the node and the hour the scan asked in (``families.identity.
derived_record_id``). A row or a refusal that is not exactly such a request for
this node, in the hours the case ran, stays foreign.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.gpu_activity import GpuMaintenanceWindowOpen
from gpu_fault.orchestration.families.identity import derived_record_id

OPERATION = "TRIGGER_HEALTH_SNAPSHOT"
# FREEZE_EVIDENCE is step 0 of the refresh workflow, the snapshot step 1.
STEP_INDEX = 1
# ``request_observation_refresh``'s event id and ``NodeHealthFamily``'s
# workflow id (``derived_record_id(prefix, family, *parts)``).
EVENT_PREFIX, EVENT_FAMILY = "observation-refresh", "collector-health"
WORKFLOW_PREFIX, WORKFLOW_FAMILY = "workflow", "node-health"
HOUR_SECONDS = 3600
# The nine identifier fields of an ARCH-C3 line; the action pool's own
# "rejected inside the action pool" line carries only some of them and comes
# with the executor's full line, which is the one ``rejected`` counts.
IDENTIFIER_FIELDS = (
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


@dataclass(frozen=True)
class RefreshRequest:
    """One refresh the control plane can have asked for: one node, one hour."""

    hour: int
    incident_id: str
    workflow_request_id: str
    command_id: str


def requests(
    *,
    cluster_id: str,
    node: str,
    generation: int,
    start: datetime,
    end: datetime,
) -> dict[str, RefreshRequest]:
    """Every refresh of ``node`` the scan can have asked for from ``start`` to
    ``end``, keyed by its TRIGGER_HEALTH_SNAPSHOT command id."""

    first = int(start.timestamp() // HOUR_SECONDS)
    last = int(end.timestamp() // HOUR_SECONDS)
    found: dict[str, RefreshRequest] = {}
    for hour in range(first, last + 1):
        event_id = derived_record_id(
            EVENT_PREFIX, EVENT_FAMILY, cluster_id, node, str(hour)
        )
        workflow = derived_record_id(WORKFLOW_PREFIX, WORKFLOW_FAMILY, event_id)
        command_id = f"{workflow}/{STEP_INDEX}/{OPERATION}/{node}/agent-{generation}"
        found[command_id] = RefreshRequest(
            hour=hour,
            incident_id=f"inc-{event_id}",
            workflow_request_id=workflow,
            command_id=command_id,
        )
    return found


def window_refusal(window_incident: str) -> str:
    """The Agent's own words for refusing the snapshot inside that window."""

    return str(
        GpuMaintenanceWindowOpen(
            WorkflowOperation(OPERATION), [{"incident_id": window_incident}]
        )
    )


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _journal_time(line: dict[str, Any]) -> datetime | None:
    value = str(line.get("realtime_timestamp") or "")
    if not value.isdigit():
        return None
    return datetime.fromtimestamp(int(value) / 1_000_000, tz=timezone.utc)


def row_errors(
    rows: Sequence[dict[str, Any]],
    request: RefreshRequest,
    *,
    generation: int,
    window: tuple[datetime, datetime] | None,
) -> list[str]:
    """The refresh's ledger: one attempt, SUCCEEDED, no GPU, outside the window.

    ``window`` is (the chain's QUIESCE start, its RESTORE start): the attempt
    either ended before the quiesce began, or was admitted once the restore was
    under way -- the restore closes the window after its services are back.
    """

    name = request.command_id
    if len(rows) != 1:
        return [f"observation refresh {name} has {len(rows)} ledger attempts, not one"]
    row = rows[0]
    errors: list[str] = []
    expected = {
        "operation": OPERATION,
        "attempt": 1,
        "state": "SUCCEEDED",
        "incident_id": request.incident_id,
        "workflow_request_id": request.workflow_request_id,
        "gpu_uuid_count": 0,
        "exit_code": None,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(
                f"observation refresh {name} has {field}={row.get(field)!r}, "
                f"not {value!r}"
            )
    agent_generation = row.get("agent_generation")
    if agent_generation is not None and agent_generation != generation:
        errors.append(
            f"observation refresh {name} ran under agent generation "
            f"{agent_generation!r}, not {generation}"
        )
    started, completed = _time(row.get("started_at")), _time(row.get("completed_at"))
    if started is None or completed is None or completed < started:
        errors.append(f"observation refresh {name} has no ordered start and end")
    elif window is None:
        errors.append(
            f"observation refresh {name} cannot be placed against the chain's window"
        )
    elif not (completed <= window[0] or started >= window[1]):
        errors.append(
            f"observation refresh {name} ran inside the chain's quiesce window "
            f"({window[0].isoformat()} .. {window[1].isoformat()})"
        )
    return errors


def refusal_errors(
    lines: Sequence[dict[str, Any]],
    ran: dict[str, dict[str, Any]],
    *,
    invocation_id: str | None,
    window_incident: str,
    window_opened: datetime | None,
    node: str,
    generation: int,
) -> tuple[int, list[str]]:
    """Every rejection the restarted Agent logged is the window's refusal of a
    refresh that then ran; returns how many ``rejected`` counted, and what is
    wrong.

    ``ran`` maps a refresh's command id to its one ledger row. A refusal names
    that command with the row's identity, attempt 0, the window's own words,
    and was logged after the window opened and before the row's attempt began.
    Lines of the Agent before the restart (another invocation) are not its
    counters'.
    """

    reason = window_refusal(window_incident)
    counted = 0
    errors: list[str] = []
    for line in lines:
        if line.get("phase") != "rejected":
            continue
        if invocation_id and line.get("invocation_id") != invocation_id:
            continue
        fields = line.get("fields") or {}
        full = all(name in fields for name in IDENTIFIER_FIELDS)
        if full:
            counted += 1
        command_id = str(fields.get("command_id") or "")
        label = (
            f"the Agent refused {fields.get('operation')} {command_id}: "
            f"{fields.get('reason')}"
        )
        row = ran.get(command_id)
        if row is None or fields.get("operation") != OPERATION:
            errors.append(f"{label} (not an observation refresh that ran)")
            continue
        if str(fields.get("reason") or "").strip() != reason:
            errors.append(f"{label} (not the chain's window refusal)")
            continue
        if not full:
            # The action pool's own line beside the executor's: nothing more
            # to judge on it.
            continue
        expected = {
            "incident_id": str(row.get("incident_id")),
            "workflow_request_id": str(row.get("workflow_request_id")),
            "node_id": node,
            "fencing_token": str(row.get("fencing_token")),
            "agent_generation": str(generation),
            "gpu_count": "0",
            "attempt": "0",
        }
        wrong = sorted(
            name for name, value in expected.items() if fields[name] != value
        )
        if wrong:
            errors.append(f"{label} (fields {wrong} differ from the refresh's)")
            continue
        logged = _journal_time(line)
        admitted = _time(row.get("started_at"))
        if (
            logged is None
            or window_opened is None
            or admitted is None
            or not window_opened <= logged <= admitted
        ):
            errors.append(
                f"{label} (logged at {logged.isoformat() if logged else None}, "
                "not between the window's opening and the refresh's attempt)"
            )
    return counted, errors
