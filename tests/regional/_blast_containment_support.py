from __future__ import annotations

import contextlib
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import run_destr001_gpu_reset as producer
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from tests.regional._reset_notification_support import reset_notification_entry


def produce_containment(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    cpu_nodes: dict[str, Any],
    *,
    release: str = "release-1",
    cluster: str = "cluster-a",
) -> dict[str, Any]:
    """Run the real DESTR producer with in-process hardware/transport substitutes."""

    identity = {"release_id": release, "cluster_id": cluster}
    node = {
        "name": "gpu-1",
        "uid": "gpu-uid",
        "gpu_allocatable": "1",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    nodes = BlastCasesOne.node_security_snapshot(cpu_nodes)
    cpu = {
        "nodes": {
            name: {
                "taints": [
                    [item["key"], item["value"], item["effect"]]
                    for item in value["taints"]
                ],
                "unschedulable": value["unschedulable"],
                "labels": value["gpu_fault_labels"],
                "annotations": value["gpu_fault_annotations"],
            }
            for name, value in nodes.items()
        },
        "gpu_fault_jobs": [],
        "eviction_events": [],
    }
    preflight = {
        "errors": [],
        "evidence_identity": identity,
        "node": node,
        "cpu_blast": cpu,
        "focused_tests": {"passed": True},
    }
    operations = producer.EXPECTED_STEPS
    commands = [
        {
            "command_id": f"containment-command-{index}",
            "cluster_id": cluster,
            "workflow_request_id": "containment-workflow",
            "incident_id": "containment-incident",
            "step_index": index,
            "idempotency_key": f"containment-workflow/{index}/{operation}",
            "step": {"operation": operation, "node_ids": ["gpu-1"]},
            "fencing_token": 1,
            "status": "SUCCEEDED",
            "last_lease_owner": cluster + "/executor",
        }
        for index, operation in enumerate(operations)
        if operation not in {"FREEZE_EVIDENCE", "VALIDATE_GPU"}
    ]
    state = {
        "release_id": release,
        "event": {
            "event_id": "containment-event",
            "cluster_id": cluster,
            "node_id": "gpu-1",
            "xid": 46,
            "evidence_ref": "kmsg://gpu-1/boot/10",
        },
        "decision": {"official_action": "RESET_GPU"},
        "incident": {
            "incident_id": "containment-incident",
            "event_id": "containment-event",
            "cluster_id": cluster,
            "workflow_request_id": "containment-workflow",
            "node_ids": ["gpu-1"],
        },
        "workflow": {
            "request_id": "containment-workflow",
            "incident_id": "containment-incident",
            "fencing_token": 1,
            "status": "SUCCEEDED",
            "official_steps": [{"operation": item} for item in operations],
            "completed_operations": operations,
            "step_executions": [
                {
                    "operation": item["step"]["operation"],
                    "step_index": item["step_index"],
                    "status": "SUCCEEDED",
                    "adapter_operation_id": "remote/" + item["command_id"],
                }
                for item in commands
            ],
        },
        "commands": commands,
        "notifications": [
            reset_notification_entry(
                cluster_id=cluster,
                incident_id="containment-incident",
                operation_id="containment-workflow/4/RESET_GPU",
            )
        ],
    }

    class Regional:
        def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
            state["event"]["observed_at"] = datetime.now(timezone.utc).isoformat()
            return state

        def node_snapshot(self, name: str) -> dict[str, Any]:
            assert name == "gpu-1"
            return node

        def provider_events(self, *args: Any) -> list[Any]:
            return []

        def provider_events_provisional(self, end: datetime) -> bool:
            return False

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            return cpu

    class Host:
        host_script = "/unit/probe"

        def create(self) -> None:
            pass

        def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
            if command == "snapshot":
                return {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
                    "compute_clients": [],
                    "quiesce_states": [],
                    "kmsg_writable": True,
                }
            assert command in {
                "start-reset-sampler",
                "stop-reset-sampler",
                "write-xid46",
                "restore-quiesce",
            }
            return {}

        def cleanup(self) -> dict[str, bool]:
            return {"pod": False, "configmap": False}

    def read_preflight(settings: Any, case_dir: Path, **kwargs: Any) -> dict[str, Any]:
        base.write_json(case_dir / "preflight.json", preflight)
        return preflight

    with monkeypatch.context() as local, contextlib.redirect_stdout(io.StringIO()):
        local.setattr(producer, "read_only_preflight", read_preflight)
        local.setattr(producer, "verify_plan_identity", lambda *args: None)
        local.setattr(producer, "RegionalLiveFixture", lambda settings: Regional())
        local.setattr(
            producer, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
        )
        local.setattr(producer, "HostProbeFixture", lambda settings: Host())
        local.setattr(producer, "host_errors", lambda *args, **kwargs: [])
        settings = SimpleNamespace(
            node="gpu-1",
            host_probe_image="unit@sha256:" + "a" * 64,
            regional=SimpleNamespace(
                gpu_kubeconfig=root / "gpu", gpu_context="unit", namespace="unit"
            ),
        )
        assert (
            producer.execute_case(
                settings, root, 1, datetime.now(timezone.utc) + timedelta(minutes=30)
            )
            == 0
        )
    return {
        "valid": True,
        "verdict": "PASS",
        "case_id": producer.CASE_ID,
        "path": str(root / "cases" / producer.CASE_ID / f"{producer.CASE_ID}.json"),
    }
