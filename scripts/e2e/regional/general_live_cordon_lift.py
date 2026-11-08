"""GF-GENERAL-LIVE-013 ``post_repair_isolation_lifted``: the event-driven cordon
lift (S74).

The case needs the reboot-after successor's RESTART_NODE to find the node
uncordoned when the node lifecycle adapter reads it
(``adapters/hyperpod/lifecycle._observe_isolation``), after that successor's own
isolation steps ran. What the product does in between decides how:

* The only writer of the cordon between the successor's MARK_UNSCHEDULABLE and
  that read is its QUARANTINE. ``node_operations._isolate`` writes the whole
  isolation again (``_node_isolation_patch`` always carries
  ``spec.unschedulable: true``, read-modify-write with 409 retries) and verifies
  nothing. On an isolated node that write changes nothing on the server: no
  resourceVersion, no watch event. A lift that lands before it is undone; the
  window opens at its write, unseen.
* The window closes at the lifecycle adapter's node read. RESTART_NODE is minted
  ~20 ms after QUARANTINE's record and claimed within ~140 ms; the read follows
  the lease renewal, the idle-node ownership check and the provider node
  resolution (HyperPod: ListClusterNodes plus one DescribeClusterNode per node):
  about 0.3 s to 1.5 s (R24 L72: the read preceded a lift that landed 1.47 s
  after the mint).
* Every ``kubectl`` call from the operator host costs ~1.5 s: a new process that
  mints an EKS token through the exec plugin. S72's polling hold therefore
  always landed late (``NOT_PROVEN_RACE(LIFT_LATE)``).

So this lift is event-driven over ONE pre-authenticated client (TLS verified,
the kubeconfig never written back): one field-selected watch on the case node,
re-listed on 410, and every write over the same client. The first write -- the
provocation -- lifts the cordon the moment the expected successor's
MARK_UNSCHEDULABLE lands, which turns QUARANTINE's re-isolation into a real,
visible write. The second -- the decisive one -- lifts it again the moment that
re-cordon arrives (within :data:`T_Q_SECONDS` of the provocation), tens of
milliseconds after QUARANTINE's write and well before the read. When no
re-cordon arrives the provocation landed after QUARANTINE's write and is itself
the decisive lift. Still a race, not a protocol: a miss is a clean reboot.

Safety, all in the write log the grading re-checks (:func:`overreach_errors`):

* only the exact successor ``inc-reboot-after-<source workflow id>``
  (``orchestration/escalation.py``: ``inc-{event_id}``, event
  ``<rung>-after-<request id>``); never the source incident, never another one;
* every lift is one JSON patch: ``test`` the incident annotation, the fencing
  token and ``spec.unschedulable == true``, then one ``replace`` of
  ``/spec/unschedulable``. The taint and the annotations are never written; a
  failed test (the node moved) writes nothing and waits for the next event;
* the watch delivers in resourceVersion order, so an event that arrives after
  one of the hold's writes and before that write's own echo was written before
  it (another writer, e.g. the node lifecycle controller adding its
  unschedulable taint to QUARANTINE's re-cordon) and is never decided on;
* at most :data:`MAX_LIFTS` lifts. The hold stops for good on another incident
  (the support-after hand-off, or anything foreign), a boot-id change, any
  re-cordon after the decisive lift or later than :data:`T_Q_SECONDS` (only the
  product's ``_reassert_isolation`` of a submitted reboot does that), the node's
  release or deletion, or the end of the window;
* the restore (re-cordon, the same ``test`` ops then one ``add``) is registered
  in the cleanup stack before the first lift, and runs at once when the hold
  ends with nobody known to have taken the node over.

Every run records the patch-ack latency of each write, the watch delivery of its
echo and the reaction time of the decisive lift, and reads the GPU-cluster
executor's RESTART_NODE claim log line (read-only) as after-the-fact evidence of
where the window was.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gpu_fault.adapters.common import ANNOTATION_FENCING, ANNOTATION_INCIDENT

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_fixture import VariantContext, utc_now
from scripts.e2e.regional.regional_commands import RegionalFixtureError

SCHEMA = "cordon-lift/v2"
#: How long after the provocation a re-cordon by the same successor is still its
#: QUARANTINE (that follows MARK_UNSCHEDULABLE by ~0.15-0.35 s); a reboot's
#: ``_reassert_isolation`` comes >= ~10 s after the RESTART_NODE mint.
T_Q_SECONDS = 5.0
MAX_LIFTS = 2
#: Precondition-failed lift requests (the node moved under a ``test``) before
#: the hold gives up; such a request changes nothing on the server.
MAX_PRECONDITION_FAILURES = 3
WATCH_CHUNK_SECONDS = 60
MAX_WATCH_FAILURES = 5
MAX_EVENTS = 400
HOLD_EVIDENCE = "efa-driver-cordon-hold"
CLIENT_CLEANUP = "cordon-lift-client"
RESTORE_CLEANUP = "cordon-restore"
EXECUTOR_APP = "gpu-fault-cluster-executor"
EXECUTOR_CONTAINER = "executor"
RESTART_NODE = "RESTART_NODE"
LOG_LIMIT_BYTES = 8 * 1024 * 1024
#: The claim lines are read from this long before the hold armed.
LOG_SINCE_MARGIN_SECONDS = 60
#: The hold ends this long before the variant's action window does.
ACTION_MARGIN_SECONDS = 120
PRECONDITION_CODES = frozenset({409, 422})
APPLIED = frozenset({"applied", "unknown"})
#: Stops after which nobody is known to hold the node: restore at once.
RESTORE_AT_STOP = frozenset({"window", "watch_failed", "precondition_failures"})
UNSCHEDULABLE_PATH = "/spec/unschedulable"
#: The watch's own event kinds; a read (ARM, READ, RELIST) is current state.
WATCH_EVENTS = frozenset({"ADDED", "MODIFIED"})

CLAIM_LINE = re.compile(
    r"regional cluster executor claimed command=(?P<command>\S+) cluster=\S+ "
    r"operation=(?P<operation>\S+) owner=\S+ nodes=(?P<nodes>\S*) "
    r"workflow=(?P<workflow>\S+) step=(?P<step>\S+) generation=(?P<generation>\S+)"
)
FINISH_LINE = re.compile(
    r"regional cluster executor finished command=(?P<command>\S+) "
    r"operation=(?P<operation>\S+) nodes=(?P<nodes>\S*) status=(?P<status>\S+)"
)


def _pointer(key: str) -> str:
    return "/metadata/annotations/" + key.replace("~", "~0").replace("/", "~1")


INCIDENT_PATH = _pointer(ANNOTATION_INCIDENT)
FENCING_PATH = _pointer(ANNOTATION_FENCING)


def successor_workflow_id(source_workflow_id: str) -> str:
    return f"workflow-reboot-after-{source_workflow_id}"


def successor_incident_id(source_workflow_id: str) -> str:
    """The reboot-after successor's incident: ``inc-<event id>`` with event id
    ``reboot-after-<source workflow id>`` (``escalation.emit``)."""

    return f"inc-reboot-after-{source_workflow_id}"


def handoff_incident_id(source_workflow_id: str) -> str:
    """The support-after hand-off of that successor's refused reboot."""

    return f"inc-support-after-{successor_workflow_id(source_workflow_id)}"


def lift_ops(incident: str, fencing: str) -> list[dict[str, Any]]:
    return [
        {"op": "test", "path": INCIDENT_PATH, "value": incident},
        {"op": "test", "path": FENCING_PATH, "value": fencing},
        {"op": "test", "path": UNSCHEDULABLE_PATH, "value": True},
        {"op": "replace", "path": UNSCHEDULABLE_PATH, "value": False},
    ]


def restore_ops(incident: str, fencing: str) -> list[dict[str, Any]]:
    """Re-cordon: ``add`` because a false ``unschedulable`` is not served."""

    return [
        {"op": "test", "path": INCIDENT_PATH, "value": incident},
        {"op": "test", "path": FENCING_PATH, "value": fencing},
        {"op": "add", "path": UNSCHEDULABLE_PATH, "value": True},
    ]


@dataclass(frozen=True)
class LiftPlan:
    """What one hold may touch: the node, the expected successor, the boot."""

    node: str
    source_workflow_id: str
    source_incident_id: str
    baseline_boot: str
    t_q_seconds: float = T_Q_SECONDS
    max_lifts: int = MAX_LIFTS

    @property
    def successor_incident(self) -> str:
        return successor_incident_id(self.source_workflow_id)

    @property
    def successor_workflow(self) -> str:
        return successor_workflow_id(self.source_workflow_id)

    @property
    def handoff_incident(self) -> str:
        return handoff_incident_id(self.source_workflow_id)


@dataclass(frozen=True)
class NodeState:
    resource_version: str
    incident: str | None
    fencing: str | None
    unschedulable: bool
    boot_id: str
    #: The newest ``managedFields`` time: the server's write time, to the second.
    managed_at: str | None


def node_state(document: dict[str, Any]) -> NodeState:
    metadata = document.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    times = [
        str(item["time"])
        for item in metadata.get("managedFields") or []
        if isinstance(item, dict) and item.get("time")
    ]
    return NodeState(
        resource_version=str(metadata.get("resourceVersion") or ""),
        incident=annotations.get(ANNOTATION_INCIDENT),
        fencing=annotations.get(ANNOTATION_FENCING),
        unschedulable=bool((document.get("spec") or {}).get("unschedulable")),
        boot_id=str(
            ((document.get("status") or {}).get("nodeInfo") or {}).get("bootID") or ""
        ),
        managed_at=max(times) if times else None,
    )


@dataclass(frozen=True)
class Decision:
    action: str  # "lift" | "stop" | "none"
    reason: str


class LiftDecider:
    """The trigger rules over node states; no I/O, monotonic seconds."""

    def __init__(self, plan: LiftPlan) -> None:
        self.plan = plan
        self.applied = 0
        self.failed = 0
        self.fencing: str | None = None
        self.successor_seen = False
        self.provocation_acked: float | None = None
        self.stopped: str | None = None

    def stop(self, reason: str) -> Decision:
        if self.stopped is None:
            self.stopped = reason
        return Decision("stop", reason)

    def wrote(self, status: str, acked: float) -> None:
        if status in APPLIED:
            self.applied += 1
            if self.applied == 1:
                self.provocation_acked = acked
            return
        self.failed += 1
        if self.failed >= MAX_PRECONDITION_FAILURES:
            self.stop("precondition_failures")

    def observe(self, state: NodeState, now: float) -> Decision:
        if self.stopped is not None:
            return Decision("none", "stopped")
        plan = self.plan
        if plan.baseline_boot and state.boot_id and state.boot_id != plan.baseline_boot:
            return self.stop("rebooted")
        if state.incident == plan.successor_incident:
            return self._successor(state, now)
        if not self.successor_seen:
            if state.incident in (None, "", plan.source_incident_id):
                return Decision("none", "the source incident holds the node")
            return self.stop("foreign_incident")
        if state.incident == plan.handoff_incident:
            return self.stop("handed_over")
        if not state.incident:
            return self.stop("released")
        return self.stop("foreign_incident")

    def _successor(self, state: NodeState, now: float) -> Decision:
        if not state.fencing:
            return self.stop("unfenced")
        if self.fencing is None:
            self.fencing = state.fencing
        elif state.fencing != self.fencing:
            return self.stop("foreign_generation")
        self.successor_seen = True
        if not state.unschedulable:
            return Decision("none", "the cordon is lifted")
        if self.applied == 0:
            return Decision("lift", "provocation")
        acked = self.provocation_acked
        if (
            self.applied == 1
            and min(self.plan.max_lifts, MAX_LIFTS) >= 2
            and acked is not None
            and now - acked <= self.plan.t_q_seconds
        ):
            return Decision("lift", "decisive")
        return self.stop("reasserted")


def _status(exc: BaseException) -> int | None:
    value = getattr(exc, "status", None)
    return value if isinstance(value, int) else None


def describe(exc: BaseException) -> str:
    """Type, HTTP status and reason only: never a response body or headers."""

    status = _status(exc)
    reason = str(getattr(exc, "reason", "") or "") if status is not None else ""
    text = type(exc).__name__ + (f"({status}) {reason}" if status is not None else "")
    return text[:200]


def _json(response: Any) -> dict[str, Any]:
    data = response.data
    value = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
    if not isinstance(value, dict):
        raise RegionalFixtureError("the Kubernetes API answered a non-object")
    return value


def _lines(response: Any) -> Iterator[str]:
    buffer = bytearray()
    for chunk in response.stream(amt=None, decode_content=False):
        buffer.extend(chunk if isinstance(chunk, bytes) else str(chunk).encode())
        while (cut := buffer.find(b"\n")) != -1:
            line = buffer[:cut].decode("utf-8", errors="replace").strip()
            del buffer[: cut + 1]
            if line:
                yield line
    tail = buffer.decode("utf-8", errors="replace").strip()
    if tail:
        yield tail


class NodeClient:
    """One authenticated Kubernetes client for one node: read, watch, patch."""

    def __init__(self, api_client: Any, core: Any, node: str) -> None:
        self.api_client = api_client
        self.core = core
        self.node = node

    @classmethod
    def from_kubeconfig(cls, kubeconfig: Path, context: str, node: str) -> NodeClient:
        from kubernetes import client, config

        api_client = config.new_client_from_config(
            config_file=str(kubeconfig), context=context, persist_config=False
        )
        configuration = api_client.configuration
        if not configuration.verify_ssl or not str(configuration.host).startswith(
            "https://"
        ):
            api_client.close()
            raise RegionalFixtureError("the cordon lift requires verified TLS")
        return cls(api_client, client.CoreV1Api(api_client), node)

    def read(self) -> dict[str, Any]:
        return _json(
            self.core.read_node(
                self.node, _preload_content=False, _request_timeout=(5, 15)
            )
        )

    def watch(self, resource_version: str, seconds: int) -> Iterator[dict[str, Any]]:
        response = self.core.list_node(
            field_selector=f"metadata.name={self.node}",
            resource_version=resource_version,
            allow_watch_bookmarks=True,
            timeout_seconds=seconds,
            watch=True,
            _preload_content=False,
            _request_timeout=(5, seconds + 15),
        )
        try:
            for line in _lines(response):
                event = json.loads(line)
                if isinstance(event, dict):
                    yield event
        finally:
            response.close()
            response.release_conn()

    def patch(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        # A list body is sent as application/json-patch+json.
        return _json(
            self.core.patch_node(
                self.node, ops, _preload_content=False, _request_timeout=(5, 15)
            )
        )

    def close(self) -> None:
        self.api_client.close()


ClientFactory = Callable[[Path, str, str], Any]


def new_record(plan: LiftPlan, *, armed_at: str, window: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "node": plan.node,
        "source_incident": plan.source_incident_id,
        "source_workflow": plan.source_workflow_id,
        "successor_incident": plan.successor_incident,
        "successor_workflow": plan.successor_workflow,
        "handoff_incident": plan.handoff_incident,
        "baseline_boot": plan.baseline_boot,
        "t_q_seconds": plan.t_q_seconds,
        "max_lifts": plan.max_lifts,
        "window_seconds": window,
        "armed_at": armed_at,
        "events": [],
        "events_dropped": 0,
        "writes": [],
        "lifts": [],
        "relists": 0,
        "stale_events": 0,
        "errors": [],
        "stopped": None,
        "stopped_at": None,
        "stop_seq": None,
        "decisive_seq": None,
        "latency": {"patch_ack_ms": [], "watch_delivery_ms": [], "reaction_ms": None},
    }


@dataclass
class _Echo:
    write: dict[str, Any]
    sent: float
    acked: float


class CordonLifter:
    """The hold: :meth:`run` once, :meth:`restore` any number of times."""

    def __init__(
        self,
        client: Any,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = utc_now,
    ) -> None:
        self.client = client
        self.monotonic = monotonic
        self.wall = wall
        self.plan: LiftPlan | None = None
        self.decider: LiftDecider | None = None
        self.record: dict[str, Any] = {}
        self.closed = False
        self.seq = 0
        self.last_rv = ""
        self.echoes: dict[str, _Echo] = {}
        #: The resourceVersion of the hold's last applied write until its echo
        #: (or a fresh read) arrives: events before it predate the write.
        self.awaiting: str | None = None

    # ----------------------------------------------------------------- the hold
    def run(self, plan: LiftPlan, *, window_seconds: float) -> dict[str, Any]:
        self.plan = plan
        decider = self.decider = LiftDecider(plan)
        self.record = new_record(
            plan, armed_at=self.wall().isoformat(), window=window_seconds
        )
        deadline = self.monotonic() + window_seconds
        try:
            version = self._arm()
        except Exception as exc:  # noqa: BLE001 - nothing was written yet
            raise RegionalFixtureError(
                f"the cordon lift cannot read {plan.node}: {describe(exc)}"
            ) from None
        failures = 0
        while decider.stopped is None:
            if deadline - self.monotonic() <= 0:
                decider.stop("window")
                break
            try:
                version = self._watch(version, deadline)
                failures = 0
            except Exception as exc:  # noqa: BLE001 - a broken stream is resynced
                failures += 1
                self.record["errors"].append(describe(exc))
                if failures >= MAX_WATCH_FAILURES:
                    decider.stop("watch_failed")
                    break
                try:
                    version = self._resync("RELIST")
                except Exception as again:  # noqa: BLE001 - counted above
                    self.record["errors"].append(describe(again))
        self._finish()
        if decider.stopped in RESTORE_AT_STOP:
            try:
                self.record["restore_at_stop"] = self.restore()
            except Exception as exc:  # noqa: BLE001 - the cleanup restore retries
                self.record["restore_at_stop"] = {"error": describe(exc)}
        return self.record

    def _arm(self) -> str:
        """The first read. A successor that already holds the node was not seen
        isolating it: where its chain stands is unknown (its RESTART_NODE may be
        submitted), so the hold stops without writing."""

        assert self.decider is not None, "the hold is not armed"
        document = self.client.read()
        state = node_state(document)
        if state.incident == self.decider.plan.successor_incident:
            self._event("ARM", state, self.monotonic())
            self.last_rv = state.resource_version
            self.decider.stop("armed_late")
            self._mark_stop()
            return state.resource_version
        return self._observe(document, "ARM")

    def _watch(self, version: str, deadline: float) -> str:
        assert self.decider is not None, "the hold is not armed"
        seconds = int(max(1.0, min(WATCH_CHUNK_SECONDS, deadline - self.monotonic())))
        for event in self.client.watch(version, seconds):
            kind = str(event.get("type") or "")
            obj = event.get("object") or {}
            if kind == "ERROR":
                if obj.get("code") == 410:
                    self.record["relists"] += 1
                    return self._resync("RELIST")
                raise RegionalFixtureError(f"node watch ERROR {obj.get('code')}")
            if kind == "BOOKMARK":
                meta = obj.get("metadata") or {}
                version = str(meta.get("resourceVersion") or version)
                continue
            if kind == "DELETED":
                self._event(kind, node_state(obj), self.monotonic())
                self.decider.stop("node_deleted")
                self._mark_stop()
                return version
            version = self._observe(obj, kind) or version
            if self.decider.stopped is not None or self.monotonic() >= deadline:
                return version
        # The server closed the chunk: resync, which also keeps a warm
        # connection in the pool for the next write.
        return self._resync("READ")

    def _resync(self, kind: str) -> str:
        return self._observe(self.client.read(), kind)

    def _observe(self, document: dict[str, Any], kind: str) -> str:
        assert self.decider is not None, "the hold is not armed"
        state = node_state(document)
        now = self.monotonic()
        self._event(kind, state, now)
        if self.awaiting is not None:
            if kind in WATCH_EVENTS and state.resource_version != self.awaiting:
                # Written before the hold's last write landed (another writer,
                # e.g. the node lifecycle controller's unschedulable taint):
                # the watch delivers in order, so it is stale; never decide on it.
                self.record["stale_events"] += 1
                return state.resource_version
            self.awaiting = None
        if not state.resource_version or state.resource_version == self.last_rv:
            return state.resource_version
        self.last_rv = state.resource_version
        decision = self.decider.observe(state, now)
        if decision.action == "lift":
            self._lift(state, decision.reason, event_seen=now)
        elif decision.action == "stop":
            self._mark_stop()
        return state.resource_version

    def _event(self, kind: str, state: NodeState, now: float) -> None:
        self.seq += 1
        echo = self.echoes.pop(state.resource_version, None)
        if echo is not None:
            delivery = round((now - echo.acked) * 1000, 3)
            echo.write["watch_delivery_ms"] = delivery
            echo.write["echo_after_send_ms"] = round((now - echo.sent) * 1000, 3)
            self.record["latency"]["watch_delivery_ms"].append(delivery)
        events = self.record["events"]
        if len(events) >= MAX_EVENTS:
            self.record["events_dropped"] += 1
            return
        events.append(
            {
                "seq": self.seq,
                "type": kind,
                "received_at": self.wall().isoformat(),
                "resource_version": state.resource_version,
                "incident": state.incident,
                "fencing": state.fencing,
                "unschedulable": state.unschedulable,
                "boot_id": state.boot_id,
                "managed_at": state.managed_at,
            }
        )

    def _mark_stop(self) -> None:
        if self.record.get("stopped") is None and self.decider is not None:
            self.record["stopped"] = self.decider.stopped
            self.record["stopped_at"] = self.wall().isoformat()
            self.record["stop_seq"] = self.seq

    def _write(
        self, kind: str, role: str, ops: list[dict[str, Any]], state: NodeState
    ) -> tuple[dict[str, Any], float]:
        self.seq += 1
        sent = self.monotonic()
        sent_at = self.wall().isoformat()
        status, after, error = "applied", None, None
        try:
            after = node_state(self.client.patch(ops)).resource_version
        except Exception as exc:  # noqa: BLE001 - recorded; the next event decides
            status = (
                "precondition_failed"
                if _status(exc) in PRECONDITION_CODES
                else "unknown"
            )
            error = describe(exc)
        acked = self.monotonic()
        write = {
            "seq": self.seq,
            "kind": kind,
            "role": role,
            "status": status,
            "incident": state.incident,
            "fencing": state.fencing,
            "decided_on_rv": state.resource_version,
            "ops": ops,
            "sent_at": sent_at,
            "acked_at": self.wall().isoformat(),
            "patch_ack_ms": round((acked - sent) * 1000, 3),
            "resource_version_after": after,
            "watch_delivery_ms": None,
            "echo_after_send_ms": None,
            "error": error,
        }
        self.record["writes"].append(write)
        self.record["latency"]["patch_ack_ms"].append(write["patch_ack_ms"])
        if after:
            self.echoes[after] = _Echo(write, sent, acked)
            self.awaiting = after
        return write, acked

    def _lift(self, state: NodeState, role: str, *, event_seen: float) -> None:
        assert self.plan is not None and self.decider is not None, "not armed"
        ops = lift_ops(self.plan.successor_incident, str(state.fencing))
        write, acked = self._write("lift", role, ops, state)
        write["reaction_ms"] = round((acked - event_seen) * 1000, 3)
        self.decider.wrote(write["status"], acked)
        if write["status"] in APPLIED:
            self.record["lifts"].append(write)
            self.record["decisive_seq"] = write["seq"]
            self.record["latency"]["reaction_ms"] = write["reaction_ms"]
        if self.decider.stopped is not None:
            self._mark_stop()

    def _finish(self) -> None:
        self._mark_stop()
        self.record["fencing"] = self.decider.fencing if self.decider else None
        roles: dict[str, int | None] = {"provocation": None, "decisive": None}
        for write in self.record["lifts"]:
            roles[write["role"]] = write["seq"]
        self.record["roles"] = roles

    # ----------------------------------------------------------- the restore
    def restore(self) -> dict[str, Any]:
        """Re-cordon a node this hold left lifted while the successor holds it.

        A no-op when no lift was applied, the node is gone, cordoned, or held
        by anyone else (the product moved on); raises only when the node is
        still this successor's, still lifted, and cannot be re-cordoned.
        """

        plan, decider = self.plan, self.decider
        if plan is None or decider is None or not self.record.get("lifts"):
            return {"skipped": True, "reason": "no cordon lift was applied"}
        state = self._restore_read(plan)
        if state is None:
            return {"skipped": True, "reason": "the node is absent"}
        if (
            state.incident != plan.successor_incident
            or state.fencing != decider.fencing
        ):
            return {
                "skipped": True,
                "reason": f"the node is held by {state.incident!r}, not the successor",
            }
        if state.unschedulable:
            return {"skipped": True, "reason": "the node is cordoned"}
        ops = restore_ops(plan.successor_incident, str(decider.fencing))
        write, _acked = self._write("restore", "restore", ops, state)
        if write["status"] == "applied":
            return {
                "restored": True,
                "resource_version_after": write["resource_version_after"],
            }
        again = self._restore_read(plan)
        if again is None:
            return {"skipped": True, "reason": "the node is absent"}
        if again.unschedulable:
            return {"restored": True, "verified": "the node reads cordoned"}
        if again.incident != plan.successor_incident:
            return {"skipped": True, "reason": "the node moved during the restore"}
        raise RegionalFixtureError(
            f"cordon restore of {plan.node} did not land: {write['error']}"
        )

    def _restore_read(self, plan: LiftPlan) -> NodeState | None:
        try:
            return node_state(self.client.read())
        except Exception as exc:  # noqa: BLE001 - classified below
            if _status(exc) == 404:
                return None
            raise RegionalFixtureError(
                f"cordon restore cannot read {plan.node}: {describe(exc)}"
            ) from None

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self.closed = True
            self.client.close()
        return {"closed": True}


def open_cordon_lifter(
    ctx: VariantContext, *, factory: ClientFactory | None = None
) -> CordonLifter:
    """Build the one client before any mutation (a TLS or credential failure is
    a harness failure, nothing injected yet) and register its close."""

    settings = ctx.regional.settings
    client = (factory or NodeClient.from_kubeconfig)(
        settings.gpu_kubeconfig, settings.gpu_context, ctx.settings.node
    )
    lifter = CordonLifter(client)
    ctx.cleanup.push(CLIENT_CLEANUP, lifter.close)
    return lifter


def hold_cordon_lifted(
    ctx: VariantContext, lifter: CordonLifter, plan: LiftPlan, *, window: float
) -> dict[str, Any]:
    """Arm, register the restore before the first lift, run, read the claim."""

    require_action_time(60)
    ctx.cleanup.push(RESTORE_CLEANUP, lifter.restore)
    window = max(0.0, min(window, ctx.remaining_seconds() - ACTION_MARGIN_SECONDS))
    record = lifter.run(plan, window_seconds=window)
    since = datetime.fromisoformat(record["armed_at"]) - timedelta(
        seconds=LOG_SINCE_MARGIN_SECONDS
    )
    record["executor_claims"] = executor_claims(
        ctx.regional,
        workflow_id=plan.successor_workflow,
        node=plan.node,
        since=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ctx.evidence.write(HOLD_EVIDENCE, record)
    return record


def parse_executor_lines(
    text: str, *, workflow_id: str, node: str, pod: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The successor's RESTART_NODE claim lines and every RESTART_NODE finish
    line on ``node`` of ``kubectl logs --timestamps`` output; fields only, never
    the line itself."""

    claims: list[dict[str, Any]] = []
    finishes: list[dict[str, Any]] = []
    for line in text.splitlines():
        stamp, _, message = line.partition(" ")
        claim = CLAIM_LINE.search(message)
        if (
            claim is not None
            and claim["operation"] == RESTART_NODE
            and claim["workflow"] == workflow_id
            and node in claim["nodes"].split(",")
        ):
            claims.append(
                {
                    "pod": pod,
                    "at": stamp,
                    "command": claim["command"],
                    "step": claim["step"],
                    "generation": claim["generation"],
                }
            )
            continue
        finish = FINISH_LINE.search(message)
        if (
            finish is not None
            and finish["operation"] == RESTART_NODE
            and node in finish["nodes"].split(",")
        ):
            finishes.append(
                {
                    "pod": pod,
                    "at": stamp,
                    "command": finish["command"],
                    "status": finish["status"],
                }
            )
    return claims, finishes


def executor_claims(
    regional: Any, *, workflow_id: str, node: str, since: str
) -> dict[str, Any]:
    """Read-only: the executors' RESTART_NODE claim/finish lines since ``since``.

    Evidence only; a read failure is recorded, never raised.
    """

    report: dict[str, Any] = {
        "workflow": workflow_id,
        "since": since,
        "pods": [],
        "claims": [],
        "finishes": [],
        "errors": [],
    }
    try:
        pods = [str(item["name"]) for item in regional.ready_pods("gpu", EXECUTOR_APP)]
    except Exception as exc:  # noqa: BLE001 - evidence only
        report["errors"].append(f"executor Pods: {describe(exc)}")
        return report
    finishes: list[dict[str, Any]] = []
    for pod in pods:
        try:
            text = regional.kubectl(
                "gpu",
                "logs",
                pod,
                "-c",
                EXECUTOR_CONTAINER,
                "--timestamps",
                f"--since-time={since}",
                f"--limit-bytes={LOG_LIMIT_BYTES}",
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001 - evidence only
            report["errors"].append(f"{pod}: {describe(exc)}")
            continue
        report["pods"].append(pod)
        claims, ends = parse_executor_lines(
            str(text or ""), workflow_id=workflow_id, node=node, pod=pod
        )
        report["claims"].extend(claims)
        finishes.extend(ends)
    commands = {item["command"] for item in report["claims"]}
    report["finishes"] = [item for item in finishes if item["command"] in commands]
    return report


def overreach_errors(record: dict[str, Any], *, successor: str = "") -> list[str]:
    """The hold's own write log against its rules; ``[]`` for a record that
    predates the log (S72's R24 hold wrote no ``schema``)."""

    if record.get("schema") != SCHEMA:
        return []
    errors: list[str] = []
    target = str(record.get("successor_incident") or "")
    if successor and target != successor:
        errors.append(f"the hold targeted {target!r}, not the successor {successor!r}")
    stop_seq = record.get("stop_seq")
    writes = [item for item in record.get("writes") or [] if isinstance(item, dict)]
    lifts = [
        item
        for item in writes
        if item.get("kind") == "lift" and item.get("status") in APPLIED
    ]
    cap = int(record.get("max_lifts") or MAX_LIFTS)
    if len(lifts) > min(cap, MAX_LIFTS):
        errors.append(f"{len(lifts)} cordon lifts were applied (at most {MAX_LIFTS})")
    for item in writes:
        kind, seq = item.get("kind"), item.get("seq")
        fencing = str(item.get("fencing") or "")
        shapes = {
            "lift": lift_ops(target, fencing),
            "restore": restore_ops(target, fencing),
        }
        if kind not in shapes or item.get("ops") != shapes[kind]:
            errors.append(f"write {seq} is not a cordon-only {kind!s} patch")
        if item.get("incident") != target:
            errors.append(
                f"write {seq} ({kind}) was decided under {item.get('incident')!r}"
            )
        if (
            kind == "lift"
            and isinstance(stop_seq, int)
            and isinstance(seq, int)
            and seq > stop_seq
        ):
            errors.append(f"lift {seq} was sent after the hold stopped ({stop_seq})")
    return errors


def decisive_lift(record: dict[str, Any]) -> dict[str, Any] | None:
    """The last applied lift of a v2 record (the decisive one), or ``None``."""

    lifts = [item for item in record.get("lifts") or [] if isinstance(item, dict)]
    return lifts[-1] if lifts and record.get("schema") == SCHEMA else None


def rv_position(record: dict[str, Any], resource_version: str) -> str | None:
    """Where ``resource_version`` sits in the hold's event log against the
    decisive lift: ``"before"``, ``"after"`` or ``None`` (not in the log)."""

    decisive = decisive_lift(record)
    if decisive is None or not resource_version:
        return None
    for event in record.get("events") or []:
        if str(event.get("resource_version")) == resource_version:
            seq = event.get("seq")
            if isinstance(seq, int) and isinstance(decisive.get("seq"), int):
                return "before" if seq < decisive["seq"] else "after"
    return None


__all__ = [
    "CLIENT_CLEANUP",
    "HOLD_EVIDENCE",
    "MAX_LIFTS",
    "RESTORE_CLEANUP",
    "SCHEMA",
    "T_Q_SECONDS",
    "CordonLifter",
    "Decision",
    "LiftDecider",
    "LiftPlan",
    "NodeClient",
    "NodeState",
    "decisive_lift",
    "executor_claims",
    "handoff_incident_id",
    "hold_cordon_lifted",
    "lift_ops",
    "node_state",
    "open_cordon_lifter",
    "overreach_errors",
    "parse_executor_lines",
    "restore_ops",
    "rv_position",
    "successor_incident_id",
    "successor_workflow_id",
]
