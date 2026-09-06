from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, cast


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
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)


PROBE_SCRIPT = Path(__file__).with_name("probes") / "collector_node_probe.py"


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


STORE_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, marker, observed_after_text = (sys.argv[1:] + [""])[:4]
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
store = ApplicationContext.from_environment().store
evidence = [
    item.model_dump(mode="json")
    for item in store.list_raw_evidence(cluster_id, node_id=node_id, limit=2000)
    if marker in json.dumps(item.payload, sort_keys=True, default=str)
]
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
commands = [
    item.model_dump(mode="json")
    for item in store.list_remote_commands()
    if any(
        item.workflow_request_id == workflow.get("request_id")
        for workflow in workflows
    )
]
print(json.dumps({
    "evidence": evidence,
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
        return cast(dict[str, Any], self.host.execute("snapshot", timeout=180))

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.host.execute(*arguments, timeout=timeout),
        )

    def store_snapshot(
        self,
        marker: str,
        *,
        observed_after: datetime | None = None,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                STORE_PROBE,
                self.regional.settings.cluster_id,
                self.node,
                marker,
                observed_after.isoformat() if observed_after is not None else "",
            ),
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
        deadline = time.monotonic() + timeout_seconds
        timeline = []
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.store_snapshot(marker, observed_after=observed_after)
            statuses = [item.get("status") for item in last.get("workflows") or []]
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "evidence_count": len(last.get("evidence") or []),
                    "decision_count": len(last.get("decisions") or []),
                    "workflow_statuses": statuses,
                }
            )
            write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
            enough = len(last.get("evidence") or []) >= minimum_evidence
            terminal = (
                not terminal_workflow
                or bool(statuses)
                and all(item in {"SUCCEEDED", "FAILED", "BLOCKED"} for item in statuses)
            )
            if enough and terminal:
                return last
            time.sleep(2)
        raise RegionalFixtureError(f"collector marker did not converge: {last}")

    def restore_incidents(
        self,
        state: dict[str, Any],
        *,
        profile_version: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        restore = WarmSpareLiveFixture(self.regional, "")
        results = []
        seen = set()
        for incident in state.get("incidents") or []:
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
        return results

    def cleanup(self) -> dict[str, bool]:
        return cast(dict[str, bool], self.host.cleanup())
