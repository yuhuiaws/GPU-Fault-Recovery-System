from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.probes import notification_drill
from tests.regional._cov95_ha001_harness import DEADLINE, Clock

NODES = ("node-a", "node-b", "node-c")


class Notifier:
    def __init__(self, now=None) -> None:
        self.sent: list[AdvisoryNotification] = []
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sent_at: list[datetime] = []

    def send(self, value: AdvisoryNotification) -> NotificationResult:
        self.sent.append(value)
        self.sent_at.append(self.now())
        return NotificationResult(
            notification_id=value.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=f"unit-message-{len(self.sent)}",
        )


class Workload:
    def __init__(self, site: NotificationSite, settings: Any) -> None:
        self.site, self.settings = site, settings
        self.name = settings.job_id
        self.document = yaml.safe_load(settings.manifest.read_text())
        self.resource = "job" if self.document["kind"] == "Job" else "pytorchjob"
        self.uid = f"uid-{self.name}"
        self.deleted = False
        site.fixtures.append(self)

    def submit(self) -> dict[str, str]:
        self.site.events.append("submit")
        self.site.resources[self.name] = self.uid
        if self.site.submit_failure:
            raise OSError("unit submit ACK lost")
        return {"uid": self.uid}

    def workload(self) -> dict[str, Any]:
        document = copy.deepcopy(self.document)
        if document["kind"] == "PyTorchJob":
            for role, offset in (("Master", "0"), ("Worker", "1")):
                metadata = document["spec"]["pytorchReplicaSpecs"][role]["template"][
                    "metadata"
                ]
                metadata["labels"]["gpu-fault.io/role"] = role.lower()
                metadata["annotations"] = {
                    "gpu-fault.io/rank-offset": offset,
                    "gpu-fault.io/expected-critical-ranks": "3",
                }
                if self.site.bad_metadata and role == "Master":
                    metadata["labels"]["gpu-fault.io/role"] = "worker"
        return document

    def pods(self) -> list[dict[str, Any]]:
        if self.deleted:
            return []
        return [
            {
                "name": f"{self.name}-{i}",
                "uid": f"{self.uid}-{i}",
                "node": node,
                "phase": "Running",
                "ready": True,
            }
            for i, node in enumerate(NODES[: self.settings.expected_pods])
        ]

    def delete(self) -> None:
        self.site.events.append("delete")
        if self.site.resources.get(self.name, self.uid) != self.uid:
            raise RuntimeError("unit replacement workload UID")
        if self.site.delete_failure:
            raise RuntimeError("unit workload delete unavailable")
        self.site.resources.pop(self.name, None)
        self.deleted = True


class NotificationSite:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
        self.patch, self.path = monkeypatch, path
        self.clock = Clock()
        self.store = InMemoryStore()
        self.notifier = Notifier(
            now=lambda: datetime.now(timezone.utc) + timedelta(seconds=self.clock.now)
        )
        self.cluster = SimpleNamespace(cluster_id="unit-cluster", context="unit-gpu")
        self.site_file = path / "unit-site"
        self.site_file.write_text("unit fixture")
        self.cpu_kubeconfig = path / "cpu.kubeconfig"
        self.gpu_kubeconfig = path / "gpu.kubeconfig"
        self.cpu_kubeconfig.write_text("unit CPU connection fixture\n")
        self.gpu_kubeconfig.write_text("unit GPU connection fixture\n")
        self.events: list[str] = []
        self.resources: dict[str, str] = {}
        self.fixtures: list[Workload] = []
        self.ownership_paths: list[Path] = []
        self.replicas = {app: 1 for app in runner.NOTIFICATION_DEPLOYMENTS}
        self.ready_counts = self.replicas.copy()
        self.env = {
            app: {
                "service_role": role,
                "dispatcher_enabled": "true",
                "async_delivery": "true",
            }
            for app, role in zip(
                runner.NOTIFICATION_DEPLOYMENTS,
                ("ingress", "worker", "spool-worker"),
                strict=True,
            )
        }
        self.observed_override: dict[str, Any] = {}
        self.nodes = {
            node: {
                "name": node,
                "uid": f"uid-{node}",
                "ready": "True",
                "unschedulable": False,
                "gpu_allocatable": "8",
                "ownership_annotations": {},
            }
            for node in NODES
        }
        self.node_reads = 0
        self.node_drift_after: int | None = None
        self.foreign: list[dict[str, Any]] = []
        self.query_counts: dict[str, int] = {}
        self.latch_counts: dict[tuple[str, ...], int] = {}
        self.silent = False
        self.duplicate = False
        self.bad_metadata = False
        self.submit_failure = False
        self.delete_failure = False
        self.denial = {"result": "DENIED", "code": "AccessDenied"}
        self.gpu_pods = ["executor-a", "executor-b"]
        self.drill_override: dict[str, Any] = {}
        self.deadline = DEADLINE
        self.focused_code = 0
        monkeypatch.setattr(
            notification_drill,
            "notification_notifier_from_environment",
            lambda: self.notifier,
        )
        clock = self.clock

        class DrillDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + timedelta(seconds=clock.now)

        monkeypatch.setattr(notification_drill, "datetime", DrillDatetime)
        monkeypatch.setattr(
            notification_drill, "time", SimpleNamespace(sleep=clock.sleep)
        )
        monkeypatch.setattr(runner, "drill_source_digest", lambda: "unit-source")
        monkeypatch.setattr(
            ApplicationContext,
            "from_environment",
            classmethod(lambda cls: SimpleNamespace(store=self.store)),
        )
        monkeypatch.setattr(
            runner,
            "time",
            SimpleNamespace(
                monotonic=self.clock.monotonic,
                sleep=self.clock.sleep,
                time=lambda: 1000 + self.clock.now,
            ),
        )

        def workload(regional, settings, *, state_path):
            self.ownership_paths.append(state_path)
            return Workload(self, settings)

        monkeypatch.setattr(runner, "ManagedWorkloadFixture", workload)
        monkeypatch.setattr(runner, "reusable_focused_tests", lambda _: None)
        monkeypatch.setattr(runner, "run_fixture_command", self.run_command)
        now = datetime.now(timezone.utc)
        window = {
            "method": "unit-inbox",
            "reference": "unit-receipt",
            "window_start": (now - timedelta(hours=1)).isoformat(),
            "window_end": (now + timedelta(hours=1)).isoformat(),
        }
        self.receipt, self.dedup = path / "receipt.json", path / "dedup.json"
        self.receipt.write_text(json.dumps({**window, "received": True}))
        self.dedup.write_text(
            json.dumps({**window, "send_count_delta": 0, "duplicate_inbox_count": 0})
        )

    def ready_pods(self, plane: str, app: str, target: Any) -> list[str]:
        if plane == "gpu":
            return self.gpu_pods.copy()
        return [f"{app}-{i}" for i in range(self.ready_counts[app])]

    def cpu(self, *args: str) -> str:
        app = args[2]
        values = self.env[app]
        if args[1] == "configmap":
            return json.dumps(
                {
                    "data": {
                        runner.NOTIFICATION_ENV_NAMES[key]: value
                        for key, value in values.items()
                    }
                }
            )
        return json.dumps(
            {
                "spec": {
                    "replicas": self.replicas[app],
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "envFrom": [
                                        {"configMapRef": {"name": app}},
                                        {"secretRef": {"name": "unused"}},
                                    ],
                                    "env": [
                                        {
                                            "name": runner.NOTIFICATION_ENV_NAMES[
                                                "service_role"
                                            ],
                                            "value": values["service_role"],
                                        },
                                        {
                                            "name": "UNRELATED",
                                            "valueFrom": {
                                                "fieldRef": {
                                                    "fieldPath": "metadata.name"
                                                }
                                            },
                                        },
                                    ],
                                }
                            ]
                        }
                    },
                }
            }
        )

    def run_command(
        self, argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert argv[1:3] == ["-m", "pytest"]
        self.events.append("focused-tests")
        return subprocess.CompletedProcess(
            argv, self.focused_code, "unit focused result", ""
        )

    def pod_json(
        self, plane: str, target: Any, pod: str, script: Any, *args: str, **kwargs: Any
    ) -> dict[str, Any]:
        if script == runner.NOTIFICATION_ENV_PROBE:
            app = pod.rsplit("-", 1)[0]
            return {**self.env[app], **self.observed_override}
        if script == runner.LOW_UTILIZATION_CONFIG_PROBE:
            return {"duration_seconds": 1}
        if script == runner.SES_DENIAL_PROBE:
            self.events.append(f"denial:{pod}")
            return self.denial.copy()
        output = io.StringIO()
        with self.patch.context() as local, redirect_stdout(output):
            local.setattr(sys, "argv", ["unit-cpu", *args])
            if script == runner.DRILL_PROBE:
                notification_drill.main()
            elif script == runner.REQUEUE_PROBE.read_text(encoding="utf-8"):
                exec(script, {"__name__": "__main__"})
            else:
                assert script == runner.LIVE_NOTIFICATION_PROBE
                exec(script, {})
        value = json.loads(output.getvalue().splitlines()[-1])
        if script == runner.DRILL_PROBE and "notify001-dedup-" in value["drill_id"]:
            start = datetime.fromisoformat(value["initial_completed_at"])
            end = datetime.fromisoformat(value["duplicate_window_end"])
            count = sum(start <= instant <= end for instant in self.notifier.sent_at)
            self.dedup.write_text(
                json.dumps(
                    {
                        "method": "unit-provider-observer",
                        "reference": "unit-duplicate-window",
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                        "send_count_delta": count,
                        "duplicate_inbox_count": count,
                    }
                )
            )
        return (
            {**value, **self.drill_override} if script == runner.DRILL_PROBE else value
        )

    def regional(self, target: Any) -> NotificationSite:
        return self

    def target(self, cluster_id: str) -> Any:
        return self.cluster

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "unit-release", "cluster_id": "unit-cluster"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.node_reads += 1
        value = copy.deepcopy(self.nodes[node])
        if (
            self.node_drift_after is not None
            and self.node_reads >= self.node_drift_after
        ):
            value["uid"] = "replacement"
        return value

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if kwargs.get("all_namespaces"):
            return json.dumps({"items": self.foreign})
        return args[2] if args[2] in self.resources else ""

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if script == runner.ATTEMPT_OBSERVATION_PROBE:
            return {"records": [{"unit": "observation"}]}
        if script == runner.LATCH_STATE_PROBE:
            nodes = tuple(args[1:])
            self.latch_counts[nodes] = self.latch_counts.get(nodes, 0) + 1
            return {
                "records": [
                    {
                        "node_id": node,
                        "notified": self.latch_counts[nodes] == 1,
                        "active": False,
                    }
                    for node in nodes
                ]
            }
        assert script == runner.NOTIFICATION_QUERY_PROBE
        needle = args[1]
        if not needle:
            return {"records": []}
        self.query_counts[needle] = self.query_counts.get(needle, 0) + 1
        if self.silent or self.query_counts[needle] == 1:
            return {"records": []}
        records = [
            {
                "matched_nodes": [node],
                "status": "SENT",
                "low_gpu_utilization": True,
                "provider_message_id_present": True,
                "gpu_devices": ["GPU-a", "GPU-b"],
            }
            for node in args[3:]
        ]
        if self.duplicate:
            records.append(copy.deepcopy(records[0]))
        return {"records": records}

    def live_record(self, kind: str) -> AdvisoryNotification:
        value = notification_drill.build_notification(
            kind, f"unit-{kind}", self.cluster.cluster_id
        )
        value = value.model_copy(update={"drill_id": None})
        self.store.save_notification_if_absent(value)
        result = NotificationResult(
            notification_id=value.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=f"unit-{kind}",
        )
        self.store.save_notification_result(result)
        case = (
            "GF-REGIONAL-DESTR-001" if kind == "gpu-reset" else "GF-REGIONAL-DESTR-009"
        )
        directory = self.path / "cases" / case
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{case}.json").write_text(
            json.dumps(
                {
                    "case_id": case,
                    "verdict": "PASS",
                    **self.evidence_identity(),
                    "notifications": [
                        {
                            "notification": value.model_dump(mode="json"),
                            "result": result.model_dump(mode="json"),
                        }
                    ],
                }
            )
        )
        return value

    def run_completion(self) -> dict[str, Any]:
        return runner.run_notify001(
            self,
            self.cluster,
            attempt=1,
            run_dir=self.path,
            receipt_evidence=self.receipt,
            ses_window_evidence=self.dedup,
            maintenance_window_end=self.deadline,
            release_id="unit-release",
        )

    def run_aggregation(self, **kwargs: Any) -> dict[str, Any]:
        return runner.run_notify005(
            self,
            self.cluster,
            nodes=NODES,
            case_dir=self.path,
            attempt=1,
            training_image="registry.invalid/unit@sha256:" + "a" * 64,
            maintenance_window_end=self.deadline,
            **kwargs,
        )
