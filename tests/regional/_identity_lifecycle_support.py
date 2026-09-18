"""In-process workload, regional and host substitutes for identity lifecycle tests."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.e2e.regional import run_workload_acceptance as workload


def recovery_state(cluster: str, node: str, marker: str) -> dict[str, Any]:
    event_id, incident_id, workflow_id = (
        f"{kind}-{cluster}" for kind in ("event", "incident", "workflow")
    )
    operations = ["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"]
    commands = [
        {
            "command_id": f"command-{cluster}-{index}",
            "cluster_id": cluster,
            "workflow_request_id": workflow_id,
            "incident_id": incident_id,
            "step_index": index,
            "step": {"operation": operation},
            "status": "SUCCEEDED",
            "last_lease_owner": f"executor-{cluster}",
        }
        for index, operation in enumerate(operations)
        if index
    ]
    return {
        "event": {
            "event_id": event_id,
            "cluster_id": cluster,
            "node_id": node,
            "job_id": "job-test",
            "attempt_id": "attempt-test",
            "xid": 11,
            "raw_message": "marker=" + marker,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source_boot_id": "boot-test",
            "source_monotonic_us": 12345,
            "evidence_ref": f"kmsg://{node}/boot-test/7",
            "pci_bdf": "0000:59:00.0",
            "gpu_uuid": f"GPU-{cluster}-0",
        },
        "decision": {"event_id": event_id, "official_action": "RESTART_APP"},
        "incident": {
            "incident_id": incident_id,
            "event_id": event_id,
            "cluster_id": cluster,
            "workflow_request_id": workflow_id,
            "job_id": "job-test",
            "attempt_id": "attempt-test",
            "node_ids": [node],
        },
        "workflow": {
            "request_id": workflow_id,
            "incident_id": incident_id,
            "status": "SUCCEEDED",
            "official_steps": [{"operation": operation} for operation in operations],
            "step_executions": [
                {
                    "step_index": command["step_index"],
                    "operation": command["step"]["operation"],
                    "adapter_operation_id": "remote/" + command["command_id"],
                    "status": "SUCCEEDED",
                    "details": {"source_gpu_count": 24, "target_gpu_count": 24},
                }
                for command in commands
            ],
        },
        "commands": commands,
        "restart_budget": {
            "cluster_id": cluster,
            "job_id": "job-test",
            "restart_count": 1,
        },
        "notifications": [
            {
                "notification": {
                    "notification_id": f"notice-{cluster}-{category}",
                    "category": category,
                    "incident_id": incident_id,
                    "cluster_name": f"hp-{cluster}",
                },
                "result": {
                    "status": "SENT",
                    "provider_message_id": f"message-{cluster}-{category}",
                },
            }
            for category in ("FAULT_DETECTED", "ACTION_COMPLETED")
        ],
    }


def workload_types(events: list[str], defect: str) -> tuple[type[Any], type[Any]]:
    class Managed:
        resource = "pytorchjob"

        def __init__(self, regional: Any, *args: Any, **kwargs: Any) -> None:
            self.regional = regional
            regional.workload = self
            self.restart_budget = args[0].restart_budget if args else 1
            self.restart_authorizations: list[dict[str, Any]] = []
            self.source = {
                "pods": [
                    {
                        "name": f"pod-{regional.cluster}-{index}",
                        "uid": f"uid-{regional.cluster}-{index}",
                        "node": f"node-{regional.cluster}-{index}",
                        "attempt_id": "attempt-test",
                        "phase": "Running",
                        "ready": True,
                    }
                    for index in range(3)
                ]
            }

        def submit(self) -> None:
            events.append("submit-" + self.regional.cluster)
            self.regional.submitted = True

        def wait_running(self, **kwargs: Any) -> dict[str, Any]:
            return copy.deepcopy(self.source)

        def authorize_restart(self, state: dict[str, Any]) -> None:
            assert state is self.regional.state
            events.append("authorize-" + self.regional.cluster)
            self.restart_authorizations.append(state)

        def wait_restarted(self, old_uids: set[str], **kwargs: Any) -> dict[str, Any]:
            assert self.restart_authorizations, (
                "workload_types: expected self.restart_authorizations"
            )
            assert self.restart_authorizations[-1] is self.regional.state
            events.append("wait-restarted-" + self.regional.cluster)
            return {
                "pods": [
                    {**pod, "uid": pod["uid"] + "-new", "attempt_id": "attempt-new"}
                    for pod in self.source["pods"]
                ]
            }

        def delete(self) -> None:
            events.append("delete-" + self.regional.cluster)
            if defect == "cleanup":
                raise workload.RegionalFixtureError("cleanup read failed")

        def snapshot(self) -> dict[str, Any]:
            return copy.deepcopy(self.source)

        def pods(self) -> list[dict[str, Any]]:
            return (
                []
                if self.restart_budget == 0 and defect != "running"
                else self.snapshot()["pods"]
            )

    class Prewarm:
        def __init__(self, regional: Any, **kwargs: Any) -> None:
            self.regional = regional

        def create(self, nodes: list[str]) -> None:
            events.append("prewarm-" + self.regional.cluster)

        def cleanup(self) -> dict[str, bool]:
            events.append("prewarm-cleanup-" + self.regional.cluster)
            return {}

    return Managed, Prewarm


def regional_type(
    tmp_path: Path, defect: str, regions: list[Any], events: list[str]
) -> type[Any]:
    class Region:
        def __init__(self, cluster: str) -> None:
            self.cluster = cluster
            self.settings = SimpleNamespace(
                cluster_id=cluster,
                namespace="gpu-system",
                gpu_kubeconfig=tmp_path / ("gpu-" + cluster),
                gpu_context="context-" + cluster,
                cpu_kubeconfig=tmp_path / "cpu",
                region="us-west-2",
            )
            self.submitted = False
            self.observation: dict[str, Any] | None = None
            self.state: dict[str, Any] = {}
            regions.append(self)

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "release-test", "cluster_id": self.cluster}

        def gpu_workloads(self) -> list[Any]:
            return []

        def gpu_nodes(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": f"node-{self.cluster}-{index}",
                    "uid": f"node-uid-{self.cluster}-{index}",
                    "ready": "True",
                    "unschedulable": False,
                    "labels": {},
                    "taints": [],
                }
                for index in range(3)
            ]

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            return {
                "nodes": {"cpu": {"uid": "cpu-uid"}},
                "gpu_fault_jobs": [],
                "eviction_events": [],
            }

        def kubectl(self, *args: str, **kwargs: Any) -> str:
            if args[:3] == ("cpu", "get", "node"):
                return json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "cpu", "uid": "cpu-uid"}, "spec": {}}
                        ]
                    }
                )
            if args[:2] == ("gpu", "logs"):
                return "executor healthy\n"
            raise AssertionError(f"unexpected fake kubectl: {args}")

        def ready_pods(self, *args: Any) -> list[dict[str, str]]:
            return [{"name": "executor-" + self.cluster}]

        def node_metadata(self, node: str) -> dict[str, str]:
            return {"product": "H100"}

        def post_xid_event(self, payload: dict[str, Any]) -> dict[str, bool]:
            events.append("inject-" + self.cluster)
            self.last_payload = payload
            if defect == "ack" and self.cluster == "a":
                raise workload.RegionalFixtureError("injection ACK lost")
            return {"accepted": True}

        def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
            events.append("settle-" + self.cluster)
            if defect == "branch" and self.cluster == "a":
                raise workload.RegionalFixtureError("branch failed")
            self.state = recovery_state(self.cluster, kwargs["node"], kwargs["marker"])
            if getattr(self, "last_payload", None):
                self.state["event"]["evidence_ref"] = self.last_payload["evidence_ref"]
            if self.workload.restart_budget == 0:
                self.state["workflow"]["status"] = "FAILED"
                restart = self.state["workflow"]["step_executions"][-1]
                restart.update(
                    {
                        "status": "FAILED",
                        "adapter_operation_id": None,
                        "details": {
                            "reason": "OTHER_FAILURE"
                            if defect == "wrong-reason"
                            else "RESTART_BUDGET_EXHAUSTED",
                            "restart_budget": 0,
                            "restart_count": 0,
                            "notification_id": "budget-notice-a",
                        },
                    }
                )
                if defect != "unexpected-restart":
                    self.state["commands"].pop()
                self.state["restart_budget"].update({"budget": 0, "restart_count": 0})
                self.state["notifications"] = [
                    self.state["notifications"][0],
                    {
                        "notification": {
                            "notification_id": "budget-notice-a",
                            "incident_id": "incident-a",
                            "cluster_name": "a",
                            "deduplication_key": "a/job-test/restart-budget-exhausted/0",
                        },
                        "result": {
                            "status": "SENT",
                            "provider_message_id": "budget-message-a",
                        },
                    },
                ]
            if defect == "b-failed" and self.cluster == "b":
                self.state["workflow"]["status"] = "FAILED"
            if defect == "binding":
                self.state["commands"][0]["incident_id"] = "foreign"
            return self.state

        def cpu_python(self, script: str, *args: str, **kwargs: Any) -> dict[str, Any]:
            assert script == workload.KMSG_EVIDENCE_PROBE
            event = self.state["event"]
            return {
                "evidence": [
                    {
                        "record_id": "nvidia-kernel/kmsg-boot-test-7",
                        "cluster_id": self.cluster,
                        "node_id": event["node_id"],
                        "payload": {
                            "record_id": "kmsg-boot-test-7",
                            "cluster_id": self.cluster,
                            "node_id": event["node_id"],
                            "source_boot_id": "boot-test",
                            "evidence_ref": args[1],
                        },
                    }
                ]
            }

    return Region


def host_type(tmp_path: Path, events: list[str]) -> type[Any]:
    class Host:
        def __init__(self, settings: Any) -> None:
            assert settings.state_directory == tmp_path / "host-probes"
            self.settings = settings

        def create(self) -> None:
            events.append("host-create")

        def execute(self, command: str, *args: str) -> dict[str, Any]:
            if command == "snapshot":
                return {
                    "boot_id": "boot-test",
                    "gpu_bdf": "0000:59:00.0",
                    "kmsg_writable": True,
                    "kernel_collector": {"ActiveState": "active"},
                }
            events.append("kmsg-inject")
            return {"bytes_written": 100}

        def cleanup(self) -> dict[str, bool]:
            events.append("host-cleanup")
            return {"pod": False, "configmap": False}

    return Host
