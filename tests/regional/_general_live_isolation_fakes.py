"""Scaffolding the GF-GENERAL-LIVE watcher-isolation tests share.

A fake completion-watcher Pod inventory, a ``FakeRegional`` with the kubectl
reads/execs these variants make, the probe documents a healthy block/cleanup
returns, and scripted completion records (an attempt entry with its containment
and recovery workflows, the restart ledger) for the single-action graders.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_attempts as att
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
WATCHER_NODE = "ip-10-0-40-9.us-west-2.compute.internal"
POD_IP = "10.0.40.77"
T0 = datetime(2026, 10, 24, tzinfo=timezone.utc)


def watcher_items(*pods: dict[str, Any]) -> dict[str, Any]:
    items = []
    for pod in pods:
        items.append(
            {
                "metadata": {
                    "name": pod.get("name", "gpu-fault-completion-watcher-abc"),
                    "uid": pod.get("uid", "uid-old"),
                    "resourceVersion": pod.get("rv", "100"),
                },
                "spec": {
                    "nodeName": pod.get("node", WATCHER_NODE),
                    "hostNetwork": pod.get("host_network", False),
                    "containers": [{"name": "watcher"}],
                },
                "status": {
                    "phase": pod.get("phase", "Running"),
                    "podIP": pod.get("pod_ip", POD_IP),
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "True" if pod.get("ready", True) else "False",
                        }
                    ],
                    "containerStatuses": [
                        {
                            "name": "watcher",
                            "ready": pod.get("ready", True),
                            "started": True,
                            "state": {"running": {}},
                        }
                    ],
                },
            }
        )
    return {"items": items}


class Regional(FakeRegional):
    """``FakeRegional`` plus the kubectl reads/execs these variants make."""

    def __init__(self, *, kubectl_responder: Any, **options: Any) -> None:
        super().__init__(**options)
        self.kubectl_responder = kubectl_responder
        self.kubectl_calls: list[tuple[str, ...]] = []

    def kubectl(self, plane: str, *arguments: str, **options: Any) -> str:
        self.kubectl_calls.append((plane, *arguments))
        return self.kubectl_responder(plane, arguments, options)


def kubectl_responder(
    inventory: dict[str, Any], url: str = "https://cp.example.internal/"
) -> Any:
    def respond(plane: str, arguments: tuple[str, ...], _options: Any) -> str:
        if arguments[:2] == ("get", "pod"):
            return json.dumps(inventory)
        if arguments[:2] == ("get", "secret"):
            return base64.b64encode(url.encode()).decode()
        return ""

    return respond


def isolation_context(
    tmp_path: Path,
    regional: Any,
    *,
    probes: dict[str, FakeProbe],
    window_minutes: int = 120,
):
    spec = next(s for s in att.VARIANTS if s.variant == "api_unreachable")
    settings = make_settings(spec, node=NODE)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(settings)(
        **{**settings.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes,
        settings=settings,
        window_minutes=window_minutes,
    )


def probe_docs(kind: str) -> dict[str, Any]:
    return {
        "block": {
            "rules": [f"-A X --comment tag-{kind}"],
            "deadman": {"timer": {"ActiveState": "active"}},
            "connectivity": False,
        },
        "status": {"rules": [f"-A X --comment tag-{kind}"]},
        "cleanup": {
            "clean": True,
            "kinds": {
                kind: {
                    "remaining": [],
                    "timer": {"ActiveState": "inactive"},
                    "connectivity": True,
                }
            },
        },
    }


def heartbeat(at: datetime) -> dict[str, Any]:
    return {
        "cluster_id": "gpu-eks-cluster",
        "observed_at": at.isoformat(),
        "watched_pods": 1,
        "watched_attempts": 1,
        "watcher_instance": "w",
    }


def workflow_doc(
    request_id: str,
    ops: list[str],
    executions: list[tuple[str, str]],
    status: str = "SUCCEEDED",
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "official_steps": [{"operation": op, "parameters": {}} for op in ops],
        "step_executions": [
            {"step_index": ops.index(op), "operation": op, "status": st, "details": {}}
            for op, st in executions
        ],
    }


def attempt_entry(
    *,
    stops: int = 1,
    restarts: int = 1,
    recovery: int = 1,
    decision: str | None = "PLAN_CREATED",
) -> dict[str, Any]:
    containment = workflow_doc(
        "wf-c",
        ["FREEZE_EVIDENCE", "STOP_WORKLOADS"],
        [("FREEZE_EVIDENCE", "SUCCEEDED")] + [("STOP_WORKLOADS", "SUCCEEDED")] * stops,
    )
    recoveries = [
        {
            "incident": {"incident_id": f"inc-r{i}", "state": "RECOVERED"},
            "workflow": workflow_doc(
                f"wf-r{i}",
                ["RESTART_WORKLOAD"],
                [("RESTART_WORKLOAD", "SUCCEEDED")] * (restarts if i == 0 else 1),
            ),
        }
        for i in range(recovery)
    ]
    return {
        "event": {"terminal_status": "STOPPED"},
        "decision": {"status": decision} if decision else None,
        "containment_incident": {"incident_id": "inc-c", "state": "RECOVERED"},
        "containment_workflows": [containment],
        "recovery": recoveries,
    }


def attempt_records(
    entry: dict[str, Any], *, count: int = 1, reservations: list[str] | None = None
) -> dict[str, Any]:
    reservations = (
        reservations
        if reservations is not None
        else ["wf-r0/0/RESTART_WORKLOAD"][:count]
    )
    return {
        "attempts": {"job-a001": entry},
        "restart_budget": {
            "restart_count": count,
            "reservation_ids": reservations,
            "budget": 1,
        },
    }
