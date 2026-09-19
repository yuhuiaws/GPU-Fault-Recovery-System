from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from gpu_fault.admin.atomic_json import write_json_atomic as _write_document

if __package__:
    from .acceptance_scope import scoped_case_evidence
else:
    from acceptance_scope import scoped_case_evidence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one case evidence document, scoped to what the run may record.

    The scoping is the part specific to acceptance evidence -- a selective run must
    not claim the formal sequence -- and it has to happen before the bytes are
    written, not on read, because the document on disk is the artifact an auditor
    reads. The writing itself is the same all-or-nothing rename the admin tooling
    uses for its state.
    """

    _write_document(path, scoped_case_evidence(value))


class EvidenceRecorder:
    def __init__(
        self,
        path: Path,
        *,
        case_id: str,
        inputs: dict[str, Any],
    ) -> None:
        self.path = path
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != 1
                or document.get("status") not in {"RUNNING", "COMPLETED", "FAILED"}
                or not isinstance(document.get("stages"), dict)
                or any(
                    not isinstance(value, dict) for value in document["stages"].values()
                )
            ):
                raise RuntimeError("evidence document is malformed")
            if document.get("case_id") != case_id:
                raise RuntimeError("evidence file belongs to another case")
            if document.get("inputs") != inputs:
                raise RuntimeError("evidence inputs differ from the existing run")
            self.document = document
            self.document["status"] = "RUNNING"
            self.document.pop("completed_at", None)
            if self.document.get("verdict") == "PASS":
                self.document["verdict"] = "NOT_RUN"
            self.document["updated_at"] = utc_now()
            write_json_atomic(self.path, self.document)
        else:
            self.document = {
                "schema_version": 1,
                "case_id": case_id,
                "status": "RUNNING",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "inputs": inputs,
                "stages": {},
            }
            write_json_atomic(self.path, self.document)

    def stage(
        self,
        name: str,
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        existing = self.document["stages"].get(name)
        if existing is not None:
            return dict(existing)
        result = operation()
        if not isinstance(result, dict):
            raise RuntimeError("evidence stage did not return an object")
        self.document["stages"][name] = result
        self.document["updated_at"] = utc_now()
        write_json_atomic(self.path, self.document)
        return result

    def note(self, key: str, value: Any) -> Any:
        """Record a top-level observation that is persisted but never replayed.

        ``stage`` records a step whose result a rerun must not recompute;
        ``note`` records a step a rerun must repeat -- converging the live state
        before a resume -- so it always overwrites and lives outside ``stages``.
        """

        self.document[key] = value
        self.document["updated_at"] = utc_now()
        write_json_atomic(self.path, self.document)
        return value

    def drop_stages(self, names: Iterable[str]) -> list[str]:
        """Forget recorded stages so a rerun re-executes them from scratch.

        A stage that failed a driver assertion left partial records taken
        against a live state a resume moves away from; dropping them makes the
        stage run again instead of replaying stale observations.
        """

        dropped = [name for name in names if name in self.document["stages"]]
        for name in dropped:
            del self.document["stages"][name]
        if dropped:
            self.document["updated_at"] = utc_now()
            write_json_atomic(self.path, self.document)
        return dropped

    def complete(self) -> dict[str, Any]:
        self.document["status"] = "COMPLETED"
        # A run resumed after a failure carried that failure's ``error`` in the
        # document; a completed run has none (BOOT-020 on 2026-09-11 finished
        # with every stage passed and the previous attempt's assertion text
        # still at the top level).
        self.document.pop("error", None)
        self.document["completed_at"] = utc_now()
        self.document["updated_at"] = utc_now()
        write_json_atomic(self.path, self.document)
        return self.document

    def fail(self, exc: BaseException) -> None:
        self.document["status"] = "FAILED"
        self.document["verdict"] = "FAIL"
        self.document.pop("completed_at", None)
        self.document["updated_at"] = utc_now()
        self.document["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(self.path, self.document)


def processor_queue_backlog(queue: dict[str, Any] | None) -> int:
    """The processor work a destructive preflight must wait out.

    The store snapshot reports two depths. ``fault_backlog_depth`` counts the
    reserved tier only -- control-plane actions and device events, the work a
    case's injected event would queue behind and a control-worker roll could
    disrupt. Total ``depth`` also counts routine telemetry (gpu-inventory,
    evidence), which is idempotent across a roll and can livelock one lane on a
    stale fencing token for ~120 s, so a gate on total depth flaps every
    preflight on a healthy cluster (DESTR-018 attempt 5, DESTR-019 attempt 2,
    2026-09-08). Gate on the fault tier when the snapshot carries it; an older
    snapshot without the reading falls back to total depth.
    """

    if not isinstance(queue, dict):
        raise ValueError("processor queue snapshot is missing or malformed")
    field = "fault_backlog_depth" if "fault_backlog_depth" in queue else "depth"
    value = queue.get(field)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError(f"processor queue {field} must be a nonnegative integer")
    return value


# What ``kubectl exec`` prints when its target stopped being a running replica
# between a ``ready_pods`` listing and the exec: a deleted Pod (NotFound), a
# Pod whose process already exited 0 on SIGTERM (phase Succeeded), or the
# kubelet's other wordings for the same moment. Every env window rolls its
# Deployment, so a survey taken during the roll meets these routinely
# (control-worker 2026-09-08 attempts 7/8, cluster-executor 2026-09-08
# DESTR-014 attempt 3).
_VANISHED_REPLICA = re.compile(
    r'^(?:error from server \(notfound\):\s*)?pods? "[^"\n]+" not found$'
    r"|^(?:error:\s*)?cannot exec into a container in a completed pod;"
    r" current phase is (?:Succeeded|Failed)$"
    r'|^unable to upgrade connection: container not found \("[^"\n]+"\)$'
    r"|^(?:error: Internal error occurred: error executing command in container: )?"
    r"container is not running$",
    re.I,
)


def replica_vanished(error: BaseException) -> bool:
    """Whether an exec failed because its Pod is no longer a running replica,
    as opposed to the read itself failing."""

    classified = getattr(error, "replica_disappeared", None)
    if type(classified) is bool:
        return classified
    prefix, separator, stderr = str(error).partition("stderr=")
    text = (stderr if separator else prefix).strip()
    if re.search(r"forbidden|unauthorized|permission denied|accessdenied", text, re.I):
        return False
    return _VANISHED_REPLICA.fullmatch(text) is not None


# --------------------------------------------------------------------------- #
# Open incidents on the target node
# --------------------------------------------------------------------------- #
# Runs in the CPU API Pod (``RegionalLiveFixture.cpu_python``): every incident
# of the cluster that names the node and has not RECOVERED. Read by state
# through the store's own index, not through the node's recent XID events: the
# incident a rerun trips over is older than any preflight's event lookback, and
# the event that opened it may already have been pruned by the correlator.
OPEN_INCIDENTS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState

cluster_id, node_id = sys.argv[1:3]
store = ApplicationContext.from_environment().store
open_states = [
    state for state in IncidentState if state is not IncidentState.RECOVERED
]
print(json.dumps({
    "open_incidents": [
        item.model_dump(mode="json")
        for item in store.list_incidents_by_state(
            cluster_id, open_states, node_ids={node_id}
        )
    ],
}, sort_keys=True, default=str))
"""

# RECOVERED is the only incident state that has let go of its node. In every
# other state the incident still owns it: the processor merges a new XID on that
# node into the open incident (node-scoped merge; F-N1 once ESCALATED) and plans
# nothing, so a drill that injects an XID and waits for a NEW workflow waits its
# whole budget and only ever finds the old, finished one.
CLOSED_INCIDENT_STATES = frozenset({"RECOVERED"})
# The operator exits ``gpu-fault-admin workflow-reconcile`` offers, by the state
# each one closes; the in-flight states have none and must run to their end.
INCIDENT_CLOSE_LEVERS = {
    "ESCALATED": "workflow-reconcile --close-incident",
    "QUARANTINED": "workflow-reconcile --close-quarantined",
}


def node_open_incidents(
    cpu_python: Callable[..., dict[str, Any]],
    cluster_id: str,
    node: str,
) -> list[dict[str, Any]]:
    """The node's open incidents, read through the fixture's CPU Pod probe path
    (``RegionalLiveFixture.cpu_python``)."""

    payload = cpu_python(OPEN_INCIDENTS_PROBE, cluster_id, node)
    return list(payload.get("open_incidents") or [])


def open_incident_errors(
    node: str,
    open_incidents: Iterable[dict[str, Any]],
) -> list[str]:
    """Refuse a node an open incident still owns, and name the way out.

    An incident whose ``node_ids`` are present and do not name ``node`` is
    another node's. One without the field came from a node-scoped read and
    counts: a missing field must never wave a real incident through.
    """

    errors: list[str] = []
    for item in open_incidents:
        state = str(item.get("state") or "")
        node_ids = item.get("node_ids")
        if state in CLOSED_INCIDENT_STATES:
            continue
        if node_ids is not None and node not in node_ids:
            continue
        lever = INCIDENT_CLOSE_LEVERS.get(state)
        way_out = (
            f"close it with {lever} before injecting"
            if lever
            else (
                "wait for its workflow to end, then close it with "
                "workflow-reconcile --close-incident (ESCALATED) or "
                "--close-quarantined (QUARANTINED) before injecting"
            )
        )
        errors.append(
            f"{node} carries an open incident {item.get('incident_id')} ({state}); "
            "a new XID on this node is merged into it instead of opening a "
            f"workflow; {way_out}"
        )
    return errors
