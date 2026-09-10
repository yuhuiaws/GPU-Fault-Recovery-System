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
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.kmsg_clock import marker_observed_after  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)


PROBE_SCRIPT = Path(__file__).with_name("probes") / "collector_node_probe.py"
TERMINAL_WORKFLOW_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED"})
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
    return int(env[key])


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
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, marker, observed_after_text = (sys.argv[1:] + [""])[:4]
# A fifth argument turns the raw-evidence scan on. Every poll used to page
# 2000 evidence rows through the API Pod and json-dump each one to grep the
# marker; the wait loop needs that only once its workflow is terminal.
scan_evidence = bool((sys.argv[1:] + [""] * 5)[4])
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
store = ApplicationContext.from_environment().store
evidence = (
    [
        item.model_dump(mode="json")
        for item in store.list_raw_evidence(cluster_id, node_id=node_id, limit=2000)
        if marker in json.dumps(item.payload, sort_keys=True, default=str)
    ]
    if scan_evidence
    else []
)
events = [
    item.model_dump(mode="json")
    for item in store.list_xid_events(cluster_id, node_id)
    if marker in str(item.raw_message or "")
]
decisions = []
for item in events:
    try:
        decision = store.get_xid_policy_decision(item["event_id"])
    except Exception:
        continue
    decisions.append(decision.model_dump(mode="json"))
# A kmsg-injected XID carries the marker only in the raw kernel line. The
# incident and workflow it produces never quote that line, so matching them by
# marker text alone finds nothing (COLLECT-009 watched a workflow sit in WAITING
# for ten minutes while its own snapshot reported "workflows: []"). The event
# and decision records are already selected by marker above; a workflow belongs
# to this injection when its incident points at one of those events, or when a
# marked decision names it.
marked_event_ids = {item["event_id"] for item in events}
marked_workflow_ids = {
    item.get("workflow_request_id")
    for item in decisions
    if item.get("workflow_request_id")
}
incidents = []
workflows = []
for workflow in store.list_workflows(limit=500, newest_first=True):
    # With an injection time the caller wants nothing older than it, so the
    # incident read (one store round trip per workflow) is skipped for the
    # hundreds of rows that predate the case.
    if observed_after is not None and workflow.created_at < observed_after:
        continue
    try:
        incident = store.get_incident(workflow.incident_id)
    except Exception:
        continue
    if incident.cluster_id != cluster_id or node_id not in incident.node_ids:
        continue
    encoded = json.dumps({
        "incident": incident.model_dump(mode="json"),
        "workflow": workflow.model_dump(mode="json"),
    }, sort_keys=True, default=str)
    # A Fabric Manager SXID is not an XID event: nothing the marker can reach
    # (raw evidence aside) names its workflow, so the caller passes the moment
    # it injected and every workflow this node grew since then is its own.
    injected_since = (
        observed_after is not None and workflow.created_at >= observed_after
    )
    if (
        marker not in encoded
        and incident.event_id not in marked_event_ids
        and workflow.request_id not in marked_workflow_ids
        and not injected_since
    ):
        continue
    incidents.append(incident.model_dump(mode="json"))
    workflows.append(workflow.model_dump(mode="json"))
# A MONITOR_ONLY / NO_ACTION decision (RESTART_APP on an IDLE node) opens no
# workflow at all, so the workflow walk above never surfaces its incident. The
# marked decision still names it, so its incident is read directly here -- the
# only way a case can assert the RECOVERED incident that closed without a
# workflow. Deduped against the workflow-discovered incidents and filtered by
# cluster/node exactly as they are.
seen_incident_ids = {item["incident_id"] for item in incidents}
for item in decisions:
    incident_id = item.get("incident_id")
    if not incident_id or incident_id in seen_incident_ids:
        continue
    try:
        incident = store.get_incident(incident_id)
    except Exception:
        continue
    if incident.cluster_id != cluster_id or node_id not in incident.node_ids:
        continue
    seen_incident_ids.add(incident_id)
    incidents.append(incident.model_dump(mode="json"))
# Every backend filters remote commands by workflow in the store; paging the
# whole table through the API Pod to filter it here was the slowest read of
# the poll.
request_ids = [item["request_id"] for item in workflows]
commands = (
    [
        item.model_dump(mode="json")
        for item in store.list_remote_commands(workflow_request_ids=request_ids)
    ]
    if request_ids
    else []
)
print(json.dumps({
    "evidence": evidence,
    "evidence_scanned": scan_evidence,
    "events": events,
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
                active_deadline_seconds=3600,
            )
        )

    def create(self) -> None:
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

        deadline = time.monotonic() + timeout_seconds
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

        restore = WarmSpareLiveFixture(self.regional, "")
        results = []
        seen = set()
        for incident in reversed(list(state.get("incidents") or [])):
            incident_id = str(incident.get("incident_id") or "")
            if not incident_id or incident_id in seen:
                continue
            seen.add(incident_id)
            node = self.regional.node_snapshot(self.node)
            if not node["ownership_annotations"]:
                continue
            created = restore.create_restore_workflow(
                incident_id=incident_id,
                node=self.node,
                profile_version=profile_version,
                reason=reason,
            )
            result = restore.wait_workflow_id(str(created["workflow_request_id"]))
            results.append(result)
        residual = self.residual_isolation()
        if residual["ownership_annotations"] or residual["taints"]:
            raise RegionalFixtureError(
                f"node {self.node} is still isolated after the validated restore "
                f"({reason}): {residual}; restore workflows: {results}"
            )
        return results

    def cleanup(self) -> dict[str, bool]:
        return self.host.cleanup()
