from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_action_guard import (  # noqa: E402
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.kmsg_clock import marker_observed_after  # noqa: E402
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.collector_recovery_safety import (  # noqa: E402
    require_bound_refresh,
    require_settled_recovery,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)


PROBE_SCRIPT = Path(__file__).with_name("probes") / "collector_node_probe.py"
# A superseded predecessor is settled even when the marker also names its successor.
TERMINAL_WORKFLOW_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"})
# The control plane's own isolation taint. A node the validated restore left
# with it is still quarantined whatever the annotations say.
QUARANTINE_TAINT_PREFIX = "gpu-fault.io/"
# One store read per poll is a kubectl exec into the API Pod (~2-3s); the
# processor plans in seconds, not milliseconds, so 5s loses nothing.
STORE_POLL_SECONDS = 5


def collector_setting(env: dict[str, str], key: str) -> int:
    """One integer setting out of the target node's live ``collector.env``.

    The whole point of the COLLECT group is that the judgement uses the values
    the node is really running with rather than a hardcoded 15/300, so a key
    that is not in the snapshot has to say which key and what the node did
    have. A bare ``KeyError`` from a dict lookup once reported a runner typo
    (``GPU_FAULT_DCGM_INTERVAL_SECONDS``, a name neither the installer writes
    nor the collector reads) as an unattributable case failure.
    """

    if key not in env:
        raise RegionalFixtureError(
            f"{key} is absent from the node's collector.env; it holds {sorted(env)}"
        )
    try:
        value = int(env[key])
    except (TypeError, ValueError):
        raise RegionalFixtureError(f"{key} is not a positive integer") from None
    finite_seconds(value, label=key)
    return value


def select_workflow(
    workflows: list[dict[str, Any]],
    *,
    operation: str | None = None,
    official_actions: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """The newest workflow that planned ``operation`` or decided an official action.

    "The latest workflow for the node" is the wrong key whenever the injection
    leaves a chain behind it -- a reboot's failed validation escalates into a
    replace-after workflow, a restore workflow follows a BLOCKED one -- and the
    runner then judges the wrong record. The caller names what the workflow it
    wants must contain; ``workflows`` is newest first, as every probe returns it.
    """

    for workflow in workflows:
        operations = [
            str(item.get("operation")) for item in workflow.get("official_steps") or []
        ]
        if operation is not None and operation not in operations:
            continue
        if (
            official_actions is not None
            and workflow.get("official_action") not in official_actions
        ):
            continue
        return workflow
    return None


STORE_PROBE = r"""
import json
import re
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.nvidia_logs import FabricManagerLogEvent
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, marker, observed_after_text = (sys.argv[1:] + [""])[:4]
# FM inputs are needed even by light polls to resolve deterministic SXID IDs.
# The fifth argument controls returning the full raw evidence to the caller.
scan_evidence = bool((sys.argv[1:] + [""] * 5)[4])
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
context = ApplicationContext.from_environment()
store = context.store
def marked(text):
    return re.search(r"(?:^|\s)marker=" + re.escape(marker) + r"(?=$|[\s,;])", text) is not None

records = store.list_raw_evidence(cluster_id, node_id=node_id, limit=2000)
matching = [
    item for item in records
    if item.cluster_id == cluster_id and item.node_id == node_id
    and (
        item.payload.get("record_id") == marker
        or marked(str(item.payload.get("message") or ""))
        or marked(str(item.payload.get("raw_message") or ""))
    )
]
evidence = (
    [item.model_dump(mode="json") for item in matching]
    if scan_evidence
    else []
)
events = [
    item.model_dump(mode="json")
    for item in store.list_xid_events(cluster_id, node_id)
    if item.cluster_id == cluster_id and item.node_id == node_id
    and marked(str(item.raw_message or ""))
]
# SXIDs have no event table. Reuse the deployed normalizer only for deterministic
# event IDs from stored input, then follow persisted decisions and event links.
# Do not re-enrich against today's topology and call that historical evidence.
fabric_events = []
for item in matching:
    if item.kind != EvidenceKind.FABRIC_MANAGER_LOG:
        continue
    source = FabricManagerLogEvent.model_validate(item.payload)
    if source.cluster_id != cluster_id or source.node_id != node_id:
        raise RuntimeError("fabric evidence payload identity mismatch")
    fabric_events.extend(
        event.model_dump(mode="json")
        for event in context.nvidia_logs.normalize_fabric_manager(source).sxid_events
    )
decisions = []
for item in [*events, *fabric_events]:
    try:
        decision = store.get_xid_policy_decision(item["event_id"])
    except NotFoundError:
        continue
    if decision is not None:
        decisions.append(decision.model_dump(mode="json"))
marked_event_ids = {item["event_id"] for item in [*events, *fabric_events]}
incidents = []
workflows = []
seen_incident_ids = set()
linked_incidents = []
for event_id in marked_event_ids:
    incident = store.get_incident_by_event(event_id)
    if incident is not None:
        linked_incidents.append(incident)
for item in decisions:
    incident_id = item.get("incident_id")
    if not incident_id:
        continue
    linked_incidents.append(store.get_incident(incident_id))
for incident in linked_incidents:
    if incident.incident_id in seen_incident_ids:
        continue
    if incident.cluster_id != cluster_id or node_id not in incident.node_ids:
        raise RuntimeError("marked event incident identity mismatch")
    seen_incident_ids.add(incident.incident_id)
    incidents.append(incident.model_dump(mode="json"))
for workflow in store.list_workflows(limit=500, newest_first=True):
    if workflow.incident_id in seen_incident_ids:
        workflows.append(workflow.model_dump(mode="json"))
seen_workflow_ids = {item["request_id"] for item in workflows}
for item in [*incidents, *decisions]:
    request_id = item.get("workflow_request_id")
    if not request_id or request_id in seen_workflow_ids:
        continue
    workflow = store.get_workflow(request_id)
    if workflow.incident_id not in seen_incident_ids:
        raise RuntimeError("marked decision workflow identity mismatch")
    workflows.append(workflow.model_dump(mode="json"))
    seen_workflow_ids.add(request_id)
# Every backend filters remote commands by workflow in the store; paging the
# whole table through the API Pod to filter it here was the slowest read of
# the poll.
request_ids = [item["request_id"] for item in workflows]
commands = (
    [
        item.model_dump(mode="json", exclude={"lease_token"})
        for item in store.list_remote_commands(workflow_request_ids=request_ids)
    ]
    if request_ids
    else []
)
print(json.dumps({
    "seed_marker": marker,
    "evidence": evidence,
    "evidence_scanned": scan_evidence,
    "evidence_scan_complete": len(records) < 2000,
    "events": events,
    "fabric_events": fabric_events,
    "fabric_events_reconstructed": True,
    "decisions": decisions,
    "incidents": incidents,
    "workflows": workflows,
    "commands": commands,
}, sort_keys=True, default=str))
"""


class CollectorAcceptanceFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
        case_dir: Path,
    ) -> None:
        self.regional = regional
        self.node = node
        self.host = HostProbeFixture(
            HostProbeSettings(
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=image,
                case_id=case_id,
                run_id=run_id,
                probe_script=PROBE_SCRIPT,
                state_directory=case_dir / "host-probes",
                active_deadline_seconds=3600,
            )
        )

    def create(self) -> None:
        require_action_time(180)
        self.host.create()

    def recreate(self) -> None:
        """Replace the probe Pod after the node it ran on rebooted.

        A host probe is a plain Pod pinned to one node; a RESTART_NODE takes it
        down with the node and leaves it Failed, and `kubectl exec` into a
        Failed Pod is refused. Deleting and re-applying is the only way back to
        a Pod that can still restore what the case changed on the host.
        """

        self.host.cleanup()
        self.host.create()

    def snapshot(self) -> dict[str, Any]:
        return self.host.execute("snapshot", timeout=180)

    def efa_inventory(self) -> dict[str, Any]:
        """The EFA inventory alone; a few hundred ms instead of a full snapshot."""

        inventory = self.host.execute("efa-inventory", timeout=60).get("efa_inventory")
        if not isinstance(inventory, dict):
            raise RegionalFixtureError("efa-inventory probe returned no inventory")
        return inventory

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        finite_seconds(timeout)
        if arguments[0] in {
            "throttle-gpu",
            "write-xid",
            "append-sxid",
            "restart-service",
            "set-persistence-mode",
            "override-expected-gpu-count",
            "unbind-efa",
            "kill-workload",
        }:
            require_action_time(timeout)
        return self.host.execute(*arguments, timeout=timeout)

    def store_snapshot(
        self,
        marker: str,
        *,
        observed_after: datetime | None = None,
        scan_evidence: bool = True,
    ) -> dict[str, Any]:
        bound = marker_observed_after(marker, observed_after)
        return self.regional.cpu_python(
            STORE_PROBE,
            self.regional.settings.cluster_id,
            self.node,
            marker,
            bound.isoformat() if bound is not None else "",
            "1" if scan_evidence else "",
        )

    def wait_marker(
        self,
        marker: str,
        *,
        case_dir: Path,
        minimum_evidence: int = 1,
        timeout_seconds: int = 300,
        terminal_workflow: bool = False,
        observed_after: datetime | None = None,
    ) -> dict[str, Any]:
        """Poll the store until the marker's evidence and workflow have settled.

        Each poll is the light read (events, decisions, workflows, commands);
        the raw-evidence scan runs only once the workflow condition holds, so
        a case that waits ten minutes on a WAITING step is not paging 2000
        evidence rows every few seconds while it does.
        """

        deadline = time.monotonic() + finite_seconds(timeout_seconds)
        timeline = []
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.store_snapshot(
                marker,
                observed_after=observed_after,
                scan_evidence=False,
            )
            statuses = [item.get("status") for item in last.get("workflows") or []]
            terminal = not terminal_workflow or (
                bool(statuses)
                and all(item in TERMINAL_WORKFLOW_STATUSES for item in statuses)
            )
            if terminal:
                last = self.store_snapshot(
                    marker,
                    observed_after=observed_after,
                    scan_evidence=True,
                )
                statuses = [item.get("status") for item in last.get("workflows") or []]
                terminal = not terminal_workflow or (
                    bool(statuses)
                    and all(item in TERMINAL_WORKFLOW_STATUSES for item in statuses)
                )
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "evidence_count": len(last.get("evidence") or []),
                    "evidence_scanned": bool(last.get("evidence_scanned")),
                    "decision_count": len(last.get("decisions") or []),
                    "workflow_statuses": statuses,
                }
            )
            write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
            enough = len(last.get("evidence") or []) >= minimum_evidence
            if enough and terminal:
                return last
            time.sleep(STORE_POLL_SECONDS)
        raise RegionalFixtureError(f"collector marker did not converge: {last}")

    def residual_isolation(self) -> dict[str, Any]:
        """What still marks the node as held after a restore: annotations, taints."""

        node = self.regional.node_snapshot(self.node)
        taints = [
            item
            for item in node.get("taints") or []
            if str(item.get("key") or "").startswith(QUARANTINE_TAINT_PREFIX)
        ]
        return {
            "ownership_annotations": dict(node.get("ownership_annotations") or {}),
            "taints": taints,
            "unschedulable": bool(node.get("unschedulable")),
        }

    def restore_incidents(
        self,
        state: dict[str, Any],
        *,
        profile_version: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        """Return the node through the validated restore path, and prove it.

        Every distinct incident in ``state`` is offered, newest first, so the
        one that currently owns the isolation goes first and the ones it
        correlated away are skipped once the ownership is gone. Then the node
        is read again: an ownership annotation or quarantine taint left behind
        -- by an incident this state never saw, or by a restore workflow that
        did not SUCCEED -- is a failure, not the silent ``[]`` it used to be.
        """

        marker = state.get("seed_marker")
        if isinstance(marker, str) and marker:
            current = self.store_snapshot(marker)
            require_bound_refresh(state, current, marker)
            state = current
        require_settled_recovery(state)
        restore = WarmSpareLiveFixture(self.regional, "")
        results = []
        seen = set()
        for incident in reversed(list(state.get("incidents") or [])):
            incident_id = str(incident.get("incident_id") or "")
            if not incident_id:
                raise RegionalFixtureError(
                    "validated recovery has no exact incident identity"
                )
            if incident_id in seen:
                continue
            seen.add(incident_id)
            node = self.regional.node_snapshot(self.node)
            if not node["ownership_annotations"]:
                continue
            restore.wait_incident_idle(incident_id)
            created = restore.create_restore_workflow(
                incident_id=incident_id,
                node=self.node,
                profile_version=profile_version,
                reason=reason,
            )
            result = restore.wait_workflow_id(str(created["workflow_request_id"]))
            if result.get("status") != "SUCCEEDED":
                raise RegionalFixtureError(
                    "product validated restoration did not succeed"
                )
            results.append(result)
        residual = self.residual_isolation()
        if (
            residual["ownership_annotations"]
            or residual["taints"]
            or residual["unschedulable"]
        ):
            raise RegionalFixtureError(
                f"node {self.node} is still isolated after the validated restore "
                f"({reason}): {residual}; restore workflows: {results}"
            )
        return results

    def cleanup(self) -> dict[str, bool]:
        return self.host.cleanup()
