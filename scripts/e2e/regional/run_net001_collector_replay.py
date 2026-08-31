from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)


CASE_ID = "GF-REGIONAL-NET-001"
CONFIRMATION = "NET001_COLLECTOR_OUTBOX_REPLAY"
HOST_TOOL = Path(__file__).with_name("probes") / "net001_node_probe.py"
SERVICES = (
    "gpu-fault-kernel-collector.service",
    "gpu-fault-metrics-collector.service",
    "gpu-fault-host-collector.service",
    "gpu-fault-fabric-manager-collector.service",
)


class CaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    cpu_kubeconfig: Path
    cpu_context: str
    gpu_kubeconfig: Path
    gpu_context: str
    namespace: str
    cluster_id: str
    region: str
    target_node: str
    endpoint_host: str
    host_probe_image: str
    predecessor: dict[str, Any]

    def __post_init__(self) -> None:
        for path in (self.cpu_kubeconfig, self.gpu_kubeconfig):
            if not path.is_file():
                raise ValueError(f"kubeconfig does not exist: {path}")
        required = (
            self.gpu_context,
            self.namespace,
            self.cluster_id,
            self.region,
            self.target_node,
            self.endpoint_host,
            self.host_probe_image,
        )
        if not all(value.strip() for value in required):
            raise ValueError("NET-001 settings contain an empty target identity")
        if "@sha256:" not in self.host_probe_image:
            raise ValueError("host probe image must use an immutable digest")

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "CPU_CONTEXT": self.cpu_context,
            "GPU_KUBECONFIG": str(self.gpu_kubeconfig),
            "GPU_EKS_CONTEXT": self.gpu_context,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "AWS_REGION": self.region,
            "GPU_FAULT_NET_TARGET_NODE": self.target_node,
            "GPU_FAULT_NET_ENDPOINT_HOST": self.endpoint_host,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(
    argv: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise CaseError(
            f"command failed ({completed.returncode}): {' '.join(argv)}: "
            f"{completed.stderr.strip()}"
        )
    return completed


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600)


class Runner:
    def __init__(
        self,
        run_dir: Path,
        settings: Settings,
        attempt: int,
        maintenance_window_end: datetime,
    ) -> None:
        self.run_dir = run_dir
        self.settings = settings
        self.maintenance_window_end = maintenance_window_end
        self.case_dir = run_dir / "cases" / CASE_ID
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.suffix = f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{attempt}"
        self.resource_suffix = self.suffix[-10:].replace("-", "")
        self.configmap = f"gpu-fault-net001-tool-{self.resource_suffix}"
        self.pod = f"gpu-fault-net001-host-{self.resource_suffix}"
        self.tag = f"gpu-fault-net001-{self.resource_suffix}"
        self.reconnect_tag = f"gpu-fault-net001-r-{self.resource_suffix}"
        self.drill_id = f"net001-{self.suffix}"
        self.test_ids = [
            f"net001-{self.resource_suffix}-{index}" for index in range(1, 4)
        ]
        self.wake_test_id = f"net001-wake-{self.resource_suffix}"
        self.wake_drill_id = f"net001-wake-{self.suffix}"
        self.ips: list[str] = []
        self.baseline: dict[str, Any] | None = None
        self.blocked = False
        self.reconnect_blocked = False
        self.created = False
        self.worker_pod = ""
        self.timeline: list[dict[str, Any]] = []

    def gpu(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return command(
            [
                "kubectl",
                "--kubeconfig",
                str(self.settings.gpu_kubeconfig),
                "--context",
                self.settings.gpu_context,
                *args,
            ],
            **kwargs,
        )

    def cpu(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command_line = [
            "kubectl",
            "--kubeconfig",
            str(self.settings.cpu_kubeconfig),
        ]
        if self.settings.cpu_context:
            command_line.extend(["--context", self.settings.cpu_context])
        return command([*command_line, *args], **kwargs)

    def create_resources(self) -> None:
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.configmap,
                "namespace": self.settings.namespace,
            },
            "data": {HOST_TOOL.name: HOST_TOOL.read_text(encoding="utf-8")},
        }
        self.gpu("apply", "-f", "-", input_text=json.dumps(configmap))
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod,
                "namespace": self.settings.namespace,
                "labels": {
                    "app": "gpu-fault-net001-host-tool",
                    "gpu-fault.io/acceptance-case": "NET-001",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "nodeName": self.settings.target_node,
                "hostPID": True,
                "hostNetwork": True,
                "dnsPolicy": "ClusterFirstWithHostNet",
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "tool",
                        "image": self.settings.host_probe_image,
                        "securityContext": {"privileged": True},
                        "command": ["/bin/bash", "-ceu"],
                        "args": ["trap : TERM INT; sleep 7200 & wait"],
                        "volumeMounts": [
                            {"name": "host-root", "mountPath": "/host"},
                            {
                                "name": "tool-script",
                                "mountPath": "/tool",
                                "readOnly": True,
                            },
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "host-root",
                        "hostPath": {"path": "/", "type": "Directory"},
                    },
                    {
                        "name": "tool-script",
                        "configMap": {
                            "name": self.configmap,
                            "defaultMode": 0o555,
                        },
                    },
                ],
            },
        }
        self.gpu("apply", "-f", "-", input_text=json.dumps(pod))
        self.created = True
        self.gpu(
            "-n",
            self.settings.namespace,
            "wait",
            "--for=condition=Ready",
            f"pod/{self.pod}",
            "--timeout=180s",
        )

    def host_tool(self, *args: str) -> dict[str, Any]:
        script = (
            "install -m 0700 /tool/net001_node_probe.py "
            "/host/tmp/gpu-fault-net001-node-probe.py; "
            "exec chroot /host /opt/gpu-fault/venv/bin/python "
            '/tmp/gpu-fault-net001-node-probe.py "$@"'
        )
        completed = self.gpu(
            "-n",
            self.settings.namespace,
            "exec",
            self.pod,
            "--",
            "/bin/bash",
            "-ceu",
            script,
            "net001-node-probe",
            *args,
            check=False,
        )
        output = completed.stdout.strip().splitlines()
        payload = json.loads(output[-1]) if output else {}
        if completed.returncode or "error" in payload:
            raise CaseError(f"host tool failed: {payload or completed.stderr.strip()}")
        return payload

    def active_business_workloads(self) -> list[dict[str, str]]:
        payload = json.loads(
            self.gpu(
                "get",
                "pods",
                "-A",
                "--field-selector",
                f"spec.nodeName={self.settings.target_node},status.phase=Running",
                "-o",
                "json",
            ).stdout
        )
        allowed_namespaces = {
            "aws-hyperpod",
            self.settings.namespace,
            "kube-system",
        }
        return [
            {
                "namespace": str(item["metadata"].get("namespace", "")),
                "name": str(item["metadata"].get("name", "")),
            }
            for item in payload.get("items", [])
            if item["metadata"].get("namespace") not in allowed_namespaces
        ]

    def select_worker(self) -> str:
        if self.worker_pod:
            return self.worker_pod
        payload = json.loads(
            self.cpu(
                "-n",
                self.settings.namespace,
                "get",
                "pods",
                "-l",
                "app=gpu-fault-control-worker",
                "--field-selector=status.phase=Running",
                "-o",
                "json",
            ).stdout
        )
        names = sorted(
            str(item["metadata"]["name"]) for item in payload.get("items", [])
        )
        if not names:
            raise CaseError("no Running control worker pod")
        self.worker_pod = names[0]
        return self.worker_pod

    def store_probe(self) -> dict[str, Any]:
        script = r'''
import json
import os
import sys

from gpu_fault.store import PostgresStore
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, drill_id, *test_ids = sys.argv[1:]
store = PostgresStore(
    os.environ["GPU_FAULT_STORE_URL"],
    initialize_schema=False,
    pool_min_size=0,
    pool_max_size=1,
)
try:
    evidence = store.list_raw_evidence(
        cluster_id,
        node_id=node_id,
        kind=EvidenceKind.NVIDIA_KERNEL,
        limit=1000,
    )
    matched_evidence = []
    for record in evidence:
        encoded = json.dumps(record.payload, sort_keys=True, default=str)
        matched = [value for value in test_ids if value in encoded]
        if matched:
            matched_evidence.append(
                {
                    "record_id": record.record_id,
                    "observed_at": record.observed_at.isoformat(),
                    "test_ids": matched,
                }
            )
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT payload
            FROM gpu_fault_objects
            WHERE kind='incident'
              AND payload->>'drill_id'=%s
            ORDER BY key
            """,
            (drill_id,),
        )
        incidents = [
            store._decode("incident", row[0]) for row in cursor.fetchall()
        ]
    incident_rows = []
    for incident in incidents:
        decision = store.get_xid_policy_decision(incident.event_id)
        incident_rows.append(
            {
                "incident_id": incident.incident_id,
                "event_id": incident.event_id,
                "state": incident.state.value,
                "workflow_request_id": incident.workflow_request_id,
                "effective_action": (
                    incident.effective_action.value
                    if incident.effective_action is not None
                    else None
                ),
                "decision_disposition": (
                    decision.disposition.value if decision is not None else None
                ),
                "decision_action": (
                    decision.action.value
                    if decision is not None and decision.action is not None
                    else None
                ),
                "official_action": (
                    decision.official_action if decision is not None else None
                ),
            }
        )
    notifications = [
        item for item in store.list_notifications() if item.drill_id == drill_id
    ]
    notification_rows = []
    for notification in notifications:
        result = store.get_notification_result(notification.notification_id)
        notification_rows.append(
            {
                "notification_id": notification.notification_id,
                "incident_id": notification.incident_id,
                "category": notification.category,
                "result_status": (
                    result.status.value if result is not None else None
                ),
                "provider_message_id_present": bool(
                    result is not None and result.provider_message_id
                ),
            }
        )
    print(
        json.dumps(
            {
                "evidence": matched_evidence,
                "incidents": incident_rows,
                "notifications": notification_rows,
            },
            sort_keys=True,
        )
    )
finally:
    store.close()
'''
        completed = self.cpu(
            "-n",
            self.settings.namespace,
            "exec",
            "-i",
            self.select_worker(),
            "--",
            "python",
            "-",
            self.settings.cluster_id,
            self.settings.target_node,
            self.drill_id,
            *self.test_ids,
            input_text=script,
        )
        return cast(dict[str, Any], json.loads(completed.stdout))

    @staticmethod
    def matching_ids(snapshot: dict[str, Any]) -> list[str]:
        matching = snapshot["outboxes"]["kernel"]["matching"]
        return sorted(
            test_id for item in matching for test_id in item.get("test_ids", [])
        )

    def validate_services(self, snapshot: dict[str, Any]) -> None:
        if self.baseline is None:
            raise CaseError("service baseline is missing")
        for service in SERVICES:
            current = snapshot["services"][service]
            baseline = self.baseline["services"][service]
            if current.get("ActiveState") != "active":
                raise CaseError(f"{service} is not active: {current}")
            if current.get("NRestarts") != baseline.get("NRestarts"):
                raise CaseError(
                    f"{service} restarted: "
                    f"{baseline.get('NRestarts')} -> {current.get('NRestarts')}"
                )

    def record_timeline(
        self,
        phase: str,
        snapshot: dict[str, Any],
        store: dict[str, Any],
    ) -> None:
        entry = {
            "captured_at": utc_now(),
            "phase": phase,
            "matching_outbox_ids": self.matching_ids(snapshot),
            "kernel_outbox_line_count": snapshot["outboxes"]["kernel"]["line_count"],
            "evidence_count": len(store["evidence"]),
            "incident_count": len(store["incidents"]),
            "notification_count": len(store["notifications"]),
            "rules": snapshot.get("rules", []),
        }
        self.timeline.append(entry)
        write_json(self.case_dir / "timeline.json", self.timeline)
        print(
            f"{phase}: outbox={entry['matching_outbox_ids']} "
            f"evidence={entry['evidence_count']}",
            flush=True,
        )

    def validate_preflight(self) -> None:
        if self.active_business_workloads():
            raise CaseError("target node has an active business workload")
        args = [
            "preflight",
            "--endpoint-host",
            self.settings.endpoint_host,
            "--tag-prefix",
            "gpu-fault-net001",
        ]
        for test_id in self.test_ids:
            args.extend(["--test-id", test_id])
        self.baseline = self.host_tool(*args)
        write_json(self.case_dir / "host-baseline.json", self.baseline)
        self.validate_services(self.baseline)
        if not self.baseline["kmsg_exists"] or not self.baseline["kmsg_writable"]:
            raise CaseError("/dev/kmsg is not writable from the approved host probe")
        if self.baseline["existing_tagged_rules"]:
            raise CaseError("stale NET-001 firewall rules already exist")
        if not all(self.baseline["connectivity"].values()):
            raise CaseError("target node cannot reach every current NLB IPv4 address")
        kernel = self.baseline["outboxes"]["kernel"]
        if kernel["exists"] and kernel["line_count"] != 0:
            raise CaseError(f"kernel outbox baseline is not empty: {kernel}")
        if not kernel["parent_exists"] or not kernel["parent_writable"]:
            raise CaseError(f"kernel outbox parent is not writable: {kernel}")
        if kernel["malformed_count"]:
            raise CaseError("kernel outbox contains malformed records")
        self.ips = list(self.baseline["endpoint_ipv4"])
        baseline_store = self.store_probe()
        write_json(self.case_dir / "store-baseline.json", baseline_store)
        if any(baseline_store.values()):
            raise CaseError("NET-001 drill identifiers already exist in the store")

    def arm_and_block(self, tag: str, ttl_seconds: int) -> None:
        args = [
            "arm",
            "--tag",
            tag,
            "--ttl-seconds",
            str(ttl_seconds),
        ]
        for ip in self.ips:
            args.extend(["--ip", ip])
        armed = self.host_tool(*args)
        write_json(self.case_dir / f"{tag}-armed.json", armed)
        args = ["block", "--tag", tag]
        for ip in self.ips:
            args.extend(["--ip", ip])
        blocked = self.host_tool(*args)
        write_json(self.case_dir / f"{tag}-blocked.json", blocked)
        if any(blocked["connectivity"].values()):
            raise CaseError("NLB TCP/443 remains reachable after firewall injection")

    def cleanup_tag(self, tag: str) -> dict[str, Any]:
        if not self.ips:
            return {}
        args = ["cleanup", "--tag", tag]
        for ip in self.ips:
            args.extend(["--ip", ip])
        result = self.host_tool(*args)
        write_json(self.case_dir / f"{tag}-cleanup.json", result)
        if result["rules"]:
            raise CaseError(f"firewall cleanup left tagged rules: {result['rules']}")
        if not all(result["connectivity"].values()):
            raise CaseError("NLB TCP/443 was not restored by cleanup")
        return result

    def snapshot(self, tag: str | None = None) -> dict[str, Any]:
        args = ["snapshot"]
        if tag:
            args.extend(["--tag", tag])
        for test_id in self.test_ids:
            args.extend(["--test-id", test_id])
        return self.host_tool(*args)

    def write_event(self, test_id: str, *, drill_id: str | None = None) -> None:
        if self.baseline is None:
            raise CaseError("host baseline is missing")
        result = self.host_tool(
            "write-kmsg",
            "--test-id",
            test_id,
            "--drill-id",
            drill_id or self.drill_id,
            "--pci-bdf",
            str(self.baseline["gpu_pci_bdf"]),
        )
        write_json(self.case_dir / f"{test_id}-write.json", result)

    def outage(self) -> None:
        self.arm_and_block(self.tag, 720)
        self.blocked = True
        started = time.monotonic()
        for index, test_id in enumerate(self.test_ids):
            target = index * 30
            delay = target - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            self.write_event(test_id)

        for minute in range(1, 11):
            target = minute * 60 + (15 if minute == 1 else 0)
            delay = target - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            if self.active_business_workloads():
                raise CaseError("target node acquired a business workload")
            snapshot = self.snapshot(self.tag)
            self.validate_services(snapshot)
            expected = sorted(self.test_ids)
            if self.matching_ids(snapshot) != expected:
                raise CaseError(
                    f"minute {minute} outbox IDs differ: "
                    f"{self.matching_ids(snapshot)} != {expected}"
                )
            if len(snapshot["rules"]) != len(self.ips):
                raise CaseError(f"minute {minute} firewall rules drifted")
            store = self.store_probe()
            if store["evidence"] or store["incidents"] or store["notifications"]:
                raise CaseError(
                    f"minute {minute} control plane received blocked events"
                )
            self.record_timeline(f"outage-minute-{minute}", snapshot, store)

        delay = 600 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
        self.cleanup_tag(self.tag)
        self.blocked = False
        self.write_event(self.wake_test_id, drill_id=self.wake_drill_id)

    def wait_for_replay(self) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + 300
        last_snapshot: dict[str, Any] = {}
        last_store: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last_snapshot = self.snapshot()
            self.validate_services(last_snapshot)
            last_store = self.store_probe()
            self.record_timeline("recovery", last_snapshot, last_store)
            if (
                not self.matching_ids(last_snapshot)
                and len(last_store["evidence"]) == 3
                and len(last_store["incidents"]) == 3
                and len(last_store["notifications"]) == 3
                and all(
                    value["replayable_count"] == 0
                    for value in last_snapshot["outboxes"].values()
                )
                and all(
                    item["result_status"] == "SKIPPED"
                    for item in last_store["notifications"]
                )
            ):
                return last_snapshot, last_store
            time.sleep(15)
        raise CaseError(
            "collector replay or drill notification suppression did not converge"
        )

    def reconnect_once(self) -> None:
        before = self.store_probe()
        self.arm_and_block(self.reconnect_tag, 60)
        self.reconnect_blocked = True
        time.sleep(5)
        self.cleanup_tag(self.reconnect_tag)
        self.reconnect_blocked = False
        _snapshot, after = self.wait_for_replay()
        write_json(
            self.case_dir / "reconnect-dedup.json",
            {"before": before, "after": after},
        )
        before_ids = sorted(item["record_id"] for item in before["evidence"])
        after_ids = sorted(item["record_id"] for item in after["evidence"])
        if len(after_ids) != 3 or after_ids != before_ids:
            raise CaseError("second reconnect created duplicate evidence")

    def validate_final(
        self,
        snapshot: dict[str, Any],
        store: dict[str, Any],
    ) -> None:
        self.validate_services(snapshot)
        if self.matching_ids(snapshot):
            raise CaseError("test events remain in the kernel outbox")
        evidence_ids = [item["record_id"] for item in store["evidence"]]
        if len(evidence_ids) != 3 or len(set(evidence_ids)) != 3:
            raise CaseError(f"evidence records are not exactly three: {evidence_ids}")
        for incident in store["incidents"]:
            if incident["state"] != "RECOVERED":
                raise CaseError(f"incident is not recovered: {incident}")
            if incident["workflow_request_id"] is not None:
                raise CaseError(f"NET-001 created a workflow: {incident}")
            if incident["decision_disposition"] != "MONITOR_ONLY":
                raise CaseError(f"unexpected decision disposition: {incident}")
            if incident["decision_action"] != "NO_ACTION":
                raise CaseError(f"unexpected decision action: {incident}")
            if incident["official_action"] != "IGNORE":
                raise CaseError(f"unexpected official action: {incident}")
        for notification in store["notifications"]:
            if notification["result_status"] != "SKIPPED":
                raise CaseError(
                    f"drill notification was not suppressed: {notification}"
                )
            if notification["provider_message_id_present"]:
                raise CaseError(
                    f"drill notification reached a provider: {notification}"
                )

        if self.baseline is None:
            raise CaseError("host baseline is missing")
        for name in ("dcgm", "host", "fabric-manager"):
            current_replayable = snapshot["outboxes"][name]["replayable_count"]
            if current_replayable:
                raise CaseError(
                    f"{name} outbox still has {current_replayable} replayable records"
                )

    def cleanup_resources(self) -> dict[str, bool]:
        if self.created:
            self.gpu(
                "-n",
                self.settings.namespace,
                "exec",
                self.pod,
                "--",
                "rm",
                "-f",
                "/host/tmp/gpu-fault-net001-node-probe.py",
                check=False,
            )
        self.gpu(
            "-n",
            self.settings.namespace,
            "delete",
            "pod",
            self.pod,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=120s",
            check=False,
        )
        self.gpu(
            "-n",
            self.settings.namespace,
            "delete",
            "configmap",
            self.configmap,
            "--ignore-not-found=true",
            check=False,
        )
        return {
            "pod": bool(
                self.gpu(
                    "-n",
                    self.settings.namespace,
                    "get",
                    "pod",
                    self.pod,
                    "--ignore-not-found",
                    "-o",
                    "name",
                    check=False,
                ).stdout.strip()
            ),
            "configmap": bool(
                self.gpu(
                    "-n",
                    self.settings.namespace,
                    "get",
                    "configmap",
                    self.configmap,
                    "--ignore-not-found",
                    "-o",
                    "name",
                    check=False,
                ).stdout.strip()
            ),
        }

    def run(self) -> int:
        started_at = utc_now()
        verdict = "FAIL"
        error: str | None = None
        final_snapshot: dict[str, Any] = {}
        final_store: dict[str, Any] = {}
        residuals: dict[str, bool] = {}
        try:
            if not self.settings.predecessor.get("valid", False):
                raise CaseError("formal predecessor evidence is not PASS")
            if datetime.now(timezone.utc).timestamp() + 900 >= (
                self.maintenance_window_end.timestamp()
            ):
                raise CaseError(
                    "maintenance window must have at least 15 minutes remaining"
                )
            self.create_resources()
            self.validate_preflight()
            print("NET-001 preflight: PASS", flush=True)
            self.outage()
            print("NET-001 10-minute outage: PASS", flush=True)
            final_snapshot, final_store = self.wait_for_replay()
            print("NET-001 replay: PASS", flush=True)
            self.reconnect_once()
            final_snapshot = self.snapshot()
            final_store = self.store_probe()
            self.validate_final(final_snapshot, final_store)
            verdict = "PASS"
        except Exception as exc:
            error = str(exc)
            print(f"NET-001 STOP: {error}", file=sys.stderr, flush=True)
        finally:
            try:
                if self.blocked:
                    self.cleanup_tag(self.tag)
                    self.blocked = False
                if self.reconnect_blocked:
                    self.cleanup_tag(self.reconnect_tag)
                    self.reconnect_blocked = False
            except Exception as cleanup_error:
                error = (
                    f"{error}; cleanup failed: {cleanup_error}"
                    if error
                    else f"cleanup failed: {cleanup_error}"
                )
                verdict = "FAIL"
            residuals = self.cleanup_resources()
            if any(residuals.values()):
                error = (
                    f"{error}; probe resources remain after cleanup"
                    if error
                    else "probe resources remain after cleanup"
                )
                verdict = "FAIL"

        result = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": CASE_ID,
            "verdict": verdict,
            "started_at": started_at,
            "ended_at": utc_now(),
            "cluster_id": self.settings.cluster_id,
            "target_node": self.settings.target_node,
            "region": self.settings.region,
            "gpu_context": self.settings.gpu_context,
            "maintenance_window_end": self.maintenance_window_end.isoformat(),
            "predecessor": self.settings.predecessor,
            "drill_id": self.drill_id,
            "test_ids": self.test_ids,
            "wake_test_id": self.wake_test_id,
            "wake_drill_id": self.wake_drill_id,
            "endpoint_ipv4": self.ips,
            "error": error,
            "checks": {
                "collector_services_active_without_restart": bool(final_snapshot)
                and all(
                    final_snapshot["services"][service].get("ActiveState") == "active"
                    and final_snapshot["services"][service].get("NRestarts")
                    == (
                        self.baseline["services"][service].get("NRestarts")
                        if self.baseline is not None
                        else None
                    )
                    for service in SERVICES
                ),
                "outbox_contains_exact_test_ids_during_outage": bool(self.timeline)
                and all(
                    item["matching_outbox_ids"] == sorted(self.test_ids)
                    for item in self.timeline
                    if item["phase"].startswith("outage-minute-")
                ),
                "control_plane_receives_nothing_during_outage": bool(self.timeline)
                and all(
                    item["evidence_count"] == 0
                    and item["incident_count"] == 0
                    and item["notification_count"] == 0
                    for item in self.timeline
                    if item["phase"].startswith("outage-minute-")
                ),
                "three_unique_records_after_recovery": (
                    len(final_store.get("evidence", [])) == 3
                ),
                "no_mutation_workflow": all(
                    item.get("workflow_request_id") is None
                    for item in final_store.get("incidents", [])
                ),
                "drill_notifications_suppressed": all(
                    item.get("result_status") == "SKIPPED"
                    and not item.get("provider_message_id_present")
                    for item in final_store.get("notifications", [])
                ),
            },
            "probe_residuals": residuals,
            "limitations": [
                "Collector disk outboxes are bounded to 1000 records per channel.",
                "An unwritable or corrupt outbox can still lose telemetry.",
                "Kernel outbox replay waits for a subsequent live collector post.",
                "Completion Watcher has no persistent disk outbox.",
            ],
        }
        write_json(self.case_dir / f"{CASE_ID}.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if verdict == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guarded NET-001 Collector outbox replay acceptance."
    )
    add_live_arguments(parser, confirmation=CONFIRMATION)
    parser.add_argument(
        "--cpu-kubeconfig",
        default=(
            os.getenv("GPU_FAULT_CONTROL_KUBECONFIG") or os.getenv("CPU_KUBECONFIG", "")
        ),
    )
    parser.add_argument("--cpu-context", default=os.getenv("CPU_EKS_CONTEXT", ""))
    parser.add_argument(
        "--gpu-kubeconfig",
        default=os.getenv("GPU_KUBECONFIG") or os.getenv("KUBECONFIG", ""),
    )
    parser.add_argument(
        "--gpu-context",
        default=(
            os.getenv("GPU_EKS_CONTEXT") or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
        ),
    )
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--cluster-id", default=os.getenv("GPU_FAULT_CLUSTER_ID", ""))
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", ""),
    )
    parser.add_argument("--node", default=os.getenv("GPU_FAULT_NET_TARGET_NODE", ""))
    parser.add_argument(
        "--endpoint-host",
        default=os.getenv("GPU_FAULT_NET_ENDPOINT_HOST", ""),
    )
    parser.add_argument(
        "--host-probe-image",
        default=os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
    )
    parser.add_argument("--predecessor-evidence", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.umask(0o077)
    predecessor_id, path = predecessor_path(
        args.run_dir,
        CASE_ID,
        args.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    settings = Settings(
        cpu_kubeconfig=Path(args.cpu_kubeconfig).expanduser().resolve(),
        cpu_context=args.cpu_context,
        gpu_kubeconfig=Path(args.gpu_kubeconfig).expanduser().resolve(),
        gpu_context=args.gpu_context,
        namespace=args.namespace,
        cluster_id=args.cluster_id,
        region=args.region,
        target_node=args.node,
        endpoint_host=args.endpoint_host,
        host_probe_image=args.host_probe_image,
        predecessor=predecessor,
    )
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=CASE_ID,
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details={
                "risk": "live-kernel-log-injection",
                "predecessor": predecessor,
                "target_node": settings.target_node,
                "endpoint_host": settings.endpoint_host,
                "mutation": (
                    "arm a 720-second host rollback timer, reject outbound TCP/443 "
                    "to the resolved control-plane addresses, and write three "
                    "monitor-only XID 63 records to /dev/kmsg"
                ),
                "stop_conditions": [
                    "formal predecessor evidence is not PASS",
                    "the node has a business workload or is not healthy",
                    "the rollback timer is not active before network rejection",
                    "any Collector service restarts",
                    "a monitor-only event creates a mutation workflow",
                    "any outbox, firewall rule, timer, Pod or ConfigMap remains",
                ],
                "rollback": {
                    "host_timer_seconds": 720,
                    "runner_finally_removes_tagged_firewall_rules": True,
                    "runner_finally_deletes_probe_resources": True,
                },
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    deadline = authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return Runner(args.run_dir, settings, args.attempt, deadline).run()


if __name__ == "__main__":
    raise SystemExit(main())
