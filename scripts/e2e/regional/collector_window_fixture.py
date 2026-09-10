"""Shared scaffolding of the collector-window acceptance cases.

GF-REGIONAL-COLLECT-018/019/020 and NET-008 all open a bounded change on one
idle GPU node through ``probes/collector_window_probe.py``, watch what the
control plane records about that node, and close the change again. The parts
that do not depend on the case live here: the host probe wrapper, the store
probes (collector statuses, findings and notifications since a moment,
raw evidence), the control-worker ``/metrics`` and log readers, the metric
text parser, and the plan/execute skeleton every runner shares.

Nothing here decides a verdict; each case's ``*_verdicts.py`` does that on the
documents these functions return.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

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
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "collector_window_probe.py"
CONTROL_WORKER_APP = "gpu-fault-control-worker"
API_APP = "gpu-fault-api-ha"
# The control worker's scrape port (render_control_plane_role_split.py sets
# prometheus.io/port=8081); the ingress scrapes on 8080.
WORKER_METRICS_PORT = 8081
API_METRICS_PORT = 8080
METRIC_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
LABEL_PAIR = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')

METRICS_PROBE = r"""
import json
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/metrics", timeout=15
) as response:
    print(json.dumps({"metrics": response.read().decode()}))
"""

COLLECTOR_STATUS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
print(json.dumps({"records": [
    {
        "collector": str(item.collector),
        "observed_at": item.observed_at.isoformat(),
        "last_success_at": (
            item.last_success_at.isoformat() if item.last_success_at else None
        ),
        "last_error_at": (
            item.last_error_at.isoformat() if item.last_error_at else None
        ),
        "batch_id": item.batch_id or "",
        "sample_count": item.sample_count,
        "errors": list(item.errors),
    }
    for item in store.list_collector_statuses(cluster_id, node_id)
]}, sort_keys=True, default=str))
"""

NODE_ACTIVITY_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

# Everything the control plane grew for one node since a moment: incidents and
# their workflows, the markers those incidents hold, and the notifications
# whose text names the node. A finding never quotes the kmsg marker, so time
# and node are the join, as COLLECT-011/014 already do for SXID injections.
cluster_id, node_id, since_text, kind = sys.argv[1:]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
incidents = []
workflows = []
markers = []
for workflow in store.list_workflows(limit=500, newest_first=True):
    if workflow.created_at < since:
        continue
    try:
        incident = store.get_incident(workflow.incident_id)
    except Exception:
        continue
    if incident.cluster_id != cluster_id or node_id not in incident.node_ids:
        continue
    incidents.append(incident.model_dump(mode="json"))
    workflows.append(workflow.model_dump(mode="json"))
    for marker in store.list_markers_for_incident(incident.incident_id):
        markers.append(marker.model_dump(mode="json"))
notifications = []
for notification in store.list_notifications(limit=500, newest_first=True):
    if notification.created_at < since:
        continue
    searchable = "\n".join(
        (notification.deduplication_key, notification.subject, notification.body_text)
    )
    if node_id not in searchable:
        continue
    result = store.get_notification_result(notification.notification_id)
    notifications.append({
        "notification_id": notification.notification_id,
        "incident_id": notification.incident_id,
        "category": notification.category,
        "created_at": notification.created_at.isoformat(),
        "subject": notification.subject,
        "status": result.status.value if result else None,
        "provider_message_id_present": bool(
            result is not None and result.provider_message_id
        ),
    })
evidence = []
if kind:
    from gpu_fault.telemetry import EvidenceKind
    evidence = [
        {
            "record_id": item.record_id,
            "observed_at": item.observed_at.isoformat(),
            "kind": str(item.kind),
            "payload": item.payload,
        }
        for item in store.list_raw_evidence(
            cluster_id, node_id=node_id, kind=EvidenceKind(kind), limit=2000
        )
        if item.observed_at >= since
    ]
print(json.dumps({
    "incidents": incidents,
    "workflows": workflows,
    "markers": markers,
    "notifications": notifications,
    "evidence": evidence,
}, sort_keys=True, default=str))
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_metric_samples(text: str, family: str) -> list[dict[str, Any]]:
    """Every sample of one metric family in Prometheus text exposition."""

    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_SAMPLE.match(line)
        if match is None or match.group("name") != family:
            continue
        labels = {
            pair.group("key"): pair.group("value")
            for pair in LABEL_PAIR.finditer(match.group("labels") or "")
        }
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        samples.append({"labels": labels, "value": value})
    return samples


def metric_sum(
    texts: list[str], family: str, *, where: dict[str, str] | None = None
) -> float:
    total = 0.0
    for text in texts:
        for sample in parse_metric_samples(text, family):
            if all(sample["labels"].get(k) == v for k, v in (where or {}).items()):
                total += float(sample["value"])
    return total


def metric_max(
    texts: list[str], family: str, *, where: dict[str, str] | None = None
) -> float | None:
    values = [
        float(sample["value"])
        for text in texts
        for sample in parse_metric_samples(text, family)
        if all(sample["labels"].get(k) == v for k, v in (where or {}).items())
    ]
    return max(values) if values else None


@dataclass(frozen=True)
class WindowSettings:
    regional: RegionalLiveSettings
    case_id: str
    node: str
    host_probe_image: str
    predecessor_path: Path | None
    predecessor_id: str | None
    endpoint_host: str = ""

    def environment(self) -> dict[str, str]:
        value = {
            **self.regional.environment(),
            "GPU_FAULT_WINDOW_CASE": self.case_id,
            "GPU_FAULT_WINDOW_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }
        if self.endpoint_host:
            value["GPU_FAULT_NET_ENDPOINT_HOST"] = self.endpoint_host
        return value


def add_window_arguments(parser: argparse.ArgumentParser, confirmation: str) -> None:
    add_live_arguments(parser, confirmation=confirmation)
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--node", default="")
    parser.add_argument("--host-probe-image", default="")
    parser.add_argument("--predecessor-evidence", default="")
    parser.add_argument(
        "--endpoint-host",
        default=os.getenv("GPU_FAULT_NET_ENDPOINT_HOST", ""),
        help="control-plane NLB host name; only the blackout cases need it",
    )


def configure(arguments: argparse.Namespace, case_id: str) -> WindowSettings:
    predecessor_id, path = predecessor_path(
        arguments.run_dir, case_id, arguments.predecessor_evidence
    )
    image = required(
        arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        "host probe image",
    )
    if "@sha256:" not in image:
        raise RegionalFixtureError("host probe image must use an immutable digest")
    return WindowSettings(
        regional=settings_from_arguments(arguments),
        case_id=case_id,
        node=required(arguments.node, "target node"),
        host_probe_image=image,
        predecessor_path=path,
        predecessor_id=predecessor_id,
        endpoint_host=str(arguments.endpoint_host or ""),
    )


class CollectorWindowFixture:
    """One host probe on one node plus the control-plane reads the cases share."""

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
        self.run_id = run_id
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

    def cleanup(self) -> dict[str, bool]:
        return self.host.cleanup()

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return self.host.execute(*arguments, timeout=timeout)

    def snapshot(self, marker: str | None = None) -> dict[str, Any]:
        arguments = ["snapshot"]
        if marker:
            arguments.extend(["--marker", marker])
        return self.execute(*arguments, timeout=240)

    def collector_statuses(self) -> list[dict[str, Any]]:
        value = self.regional.cpu_python(
            COLLECTOR_STATUS_PROBE, self.regional.settings.cluster_id, self.node
        )
        return cast(list[dict[str, Any]], value.get("records") or [])

    def node_activity(
        self, since: datetime, *, evidence_kind: str = ""
    ) -> dict[str, Any]:
        return self.regional.cpu_python(
            NODE_ACTIVITY_PROBE,
            self.regional.settings.cluster_id,
            self.node,
            since.isoformat(),
            evidence_kind,
        )

    def control_plane_metrics(self) -> list[str]:
        """Raw ``/metrics`` of every ready control-worker and ingress replica."""

        texts: list[str] = []
        for app, port in (
            (CONTROL_WORKER_APP, WORKER_METRICS_PORT),
            (API_APP, API_METRICS_PORT),
        ):
            for pod in self.regional.ready_pods("cpu", app):
                output = self.regional.kubectl(
                    "cpu",
                    "exec",
                    "-i",
                    str(pod["name"]),
                    "--",
                    "python3",
                    "-",
                    str(port),
                    input_text=METRICS_PROBE,
                    timeout=60,
                )
                texts.append(str(json.loads(output.splitlines()[-1])["metrics"]))
        return texts

    def control_plane_logs(self, since_seconds: int) -> str:
        """Concatenated logs of every control-worker and ingress Pod."""

        chunks = []
        for app in (CONTROL_WORKER_APP, API_APP):
            for pod in self.regional.ready_pods("cpu", app):
                chunks.append(
                    self.regional.kubectl(
                        "cpu",
                        "logs",
                        str(pod["name"]),
                        f"--since={since_seconds}s",
                        "--all-containers=true",
                        check=False,
                        timeout=120,
                    )
                )
        return "\n".join(chunks)

    def wait_until(
        self,
        accept: Callable[[], dict[str, Any] | None],
        *,
        timeout_seconds: int,
        poll_seconds: float,
        case_dir: Path,
        name: str,
    ) -> dict[str, Any] | None:
        """Poll ``accept`` until it answers, recording each poll in a timeline."""

        deadline = time.monotonic() + timeout_seconds
        timeline: list[dict[str, Any]] = []
        while True:
            value = accept()
            timeline.append(
                {"observed_at": utc_now().isoformat(), "accepted": value is not None}
            )
            write_json_atomic(case_dir / f"{name}-timeline.json", {"entries": timeline})
            if value is not None or time.monotonic() >= deadline:
                return value
            time.sleep(poll_seconds)


def open_window_or_rollback(
    fixture: CollectorWindowFixture,
    run_id: str,
    *arguments: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """``open-window``, closing the half-open window if the probe raises.

    The probe writes the drop-in, the window state and the deadman before it
    restarts the unit; when that restart fails it raises with the window still
    open and the unit down. The first COLLECT-019 run left the host collector
    restart-looping for a quarter of an hour that way -- the deadman is the
    last line, not the first. The close is best effort: the original error is
    what the case reports.
    """

    try:
        return fixture.execute(
            "open-window", "--run-id", run_id, *arguments, timeout=timeout
        )
    except Exception:
        try:
            fixture.execute("close-window", "--run-id", run_id, timeout=timeout)
        except Exception:  # noqa: BLE001 - the open error is the one to raise
            pass
        raise


def read_only_preflight(settings: WindowSettings, case_dir: Path) -> dict[str, Any]:
    """What must be true before a window opens; the same errors stop --execute."""

    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    workloads = regional.business_workloads(settings.node)
    predecessor = (
        predecessor_evidence(settings.predecessor_path, settings.predecessor_id)
        if settings.predecessor_id is not None and settings.predecessor_path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{settings.predecessor_id} predecessor evidence is not PASS")
    if node["ready"] != "True":
        errors.append(f"{node['name']} is not Ready")
    if node["ownership_annotations"]:
        errors.append(f"{node['name']} has pre-existing workflow ownership")
    if workloads:
        errors.append(f"{node['name']} has a business workload")
    if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append(f"{node['name']} Node Agent is not ACTIVE")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "store": state,
        "workloads": workloads,
        "predecessor": predecessor,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def run_window_case(
    *,
    case_id: str,
    confirmation: str,
    parser_description: str,
    plan_details: Callable[[WindowSettings, dict[str, Any]], dict[str, Any]],
    execute: Callable[
        [WindowSettings, CollectorWindowFixture, Path, int, datetime], dict[str, Any]
    ],
) -> int:
    """The plan/execute skeleton every collector-window case runs through.

    ``--plan`` (the default) writes the read-only preflight and the plan and
    exits non-zero if the preflight has errors. ``--execute`` re-runs the
    preflight, requires the case confirmation and an unexpired window, opens
    the probe Pod, runs ``execute`` and always tears the Pod down.
    """

    install_site_profile()
    parser = argparse.ArgumentParser(description=parser_description)
    add_window_arguments(parser, confirmation)
    arguments = parser.parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments, case_id)
    case_dir = arguments.run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=case_id,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=settings.environment(),
            details={**plan_details(settings, preflight), "preflight": preflight},
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    if arguments.confirm != confirmation:
        raise RegionalFixtureError(f"confirmation must be exactly {confirmation}")
    deadline = authorize_execution(
        arguments,
        case_id=case_id,
        confirmation=confirmation,
        environment=settings.environment(),
    )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    regional = RegionalLiveFixture(settings.regional)
    fixture = CollectorWindowFixture(
        regional,
        node=settings.node,
        image=settings.host_probe_image,
        case_id=case_id,
        run_id=f"{case_id.lower()}-{arguments.attempt}",
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": case_id,
        "attempt": arguments.attempt,
        "verdict": "FAIL",
        "started_at": utc_now().isoformat(),
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
    }
    try:
        fixture.create()
        result.update(execute(settings, fixture, case_dir, arguments.attempt, deadline))
        if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
            result.setdefault("errors", []).append(
                "control-plane EKS state differs from baseline"
            )
            result["verdict"] = "FAIL"
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    finally:
        try:
            residuals = fixture.cleanup()
        except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
            residuals = {"cleanup_error": True}
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
    result["ended_at"] = utc_now().isoformat()
    write_json_atomic(case_evidence_path(arguments.run_dir, case_id), result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["verdict"] == "PASS" else 1


__all__ = [
    "PROBE_SCRIPT",
    "CollectorWindowFixture",
    "WindowSettings",
    "add_window_arguments",
    "configure",
    "metric_max",
    "metric_sum",
    "parse_metric_samples",
    "read_only_preflight",
    "run_case_main",
    "run_window_case",
    "utc_now",
]
