"""Collector window case observations derived from a fake host lifecycle."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collect018_rejected_event as c018
from scripts.e2e.regional import run_collect019_nvidia_smi_hang as c019
from scripts.e2e.regional import run_collect020_gpu_identity as c020
from tests.regional._cov95_collect_net import Clock


class WindowHost:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.node = "node-a"
        self.clock = Clock()
        self.calls: list[tuple[Any, ...]] = []
        self.waits: list[str] = []
        self.window = False
        self.posted = False
        self.unparsed = False
        self.recovered = False
        self.restored = False
        self.problem = ""
        self.fallback = False
        self.pending = False
        self.wait_name = ""
        self.poll = 0
        self.snapshot_count = 0
        self.marker = ""
        self.errors_seen = 0
        self.failures: dict[str, BaseException] = {}
        self.inventory = [
            {"uuid": f"GPU-{index:08d}", "pci_bus_id": f"0000:{index:02x}:00.0"}
            for index in range(8)
        ]
        self.env = {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_NODE_INSTANCE_TYPE": "ml.p5.48xlarge",
            "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": "2",
        }
        self.regional = SimpleNamespace(
            node_snapshot=lambda node: {
                "ownership_annotations": {"workflow": "incident-a"}
                if module is c020
                and not self.restored
                and self.problem != "already-restored"
                else {},
                "unschedulable": False,
                "taints": [],
            },
            store_snapshot=lambda **kwargs: {
                "profile": {"profile_version": "profile-a"}
            },
        )
        self.settings = SimpleNamespace(
            node=self.node, regional=SimpleNamespace(cluster_id="cluster-a")
        )

    def service(self) -> dict[str, str]:
        return {
            "ActiveState": "active",
            "NRestarts": "1"
            if self.problem == "service" and self.snapshot_count > 1
            else "0",
            "MainPID": "123",
            "InvocationID": "invocation-a",
        }

    def snapshot(self) -> dict[str, Any]:
        self.snapshot_count += 1
        return {
            "boot_id": "boot-a",
            "collector_env": deepcopy(self.env),
            "gpu_inventory": deepcopy(self.inventory),
            "services": {c019.verdicts.UNIT: self.service()},
        }

    def execute(self, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((command, *args))
        if command in self.failures:
            raise self.failures[command]
        if command == "post-rejected-event":
            self.posted = True
            self.marker = args[args.index("--marker") + 1]
            return {
                "record_id": "foreign"
                if self.problem == "record-id"
                else f"acceptance-rejected-{self.marker}"
            }
        if command == "write-kmsg":
            kind = args[args.index("--kind") + 1]
            if kind == "unparsed-xid":
                self.unparsed = True
            else:
                self.recovered = True
            return {"kind": kind}
        if command == "open-window":
            self.window = True
            unit = args[args.index("--unit") + 1]
            shadow = args[args.index("--shadow-nvidia-smi") + 1].split(":")
            return {
                "unit": unit,
                "shadow": shadow,
                "unset": ["GPU_FAULT_EXPECTED_GPU_COUNT"],
                "after": {"ActiveState": "inactive"}
                if self.problem == "window"
                else self.service(),
                "deadman_timer": {"ActiveState": "active"},
            }
        if command == "close-window":
            self.window = False
            return {
                "dropin_removed": self.problem != "close",
                "window_root_removed": True,
                "after": self.service(),
            }
        raise AssertionError(command)

    def collector_statuses(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        if self.module is c018:
            return [
                {
                    "collector": "NVIDIA_KERNEL",
                    "batch_id": "kernel-health-fixture",
                    "observed_at": (now - timedelta(seconds=10)).isoformat(),
                    "last_error_at": now.isoformat() if self.posted else None,
                    "last_success_at": (
                        now + timedelta(seconds=1)
                        if self.recovered
                        else now - timedelta(seconds=1)
                    ).isoformat(),
                    "errors": [c018.verdicts.REJECTED_PREFIX + " field is forbidden"]
                    if self.posted and self.problem != "rejection"
                    else [],
                }
            ]
        self.errors_seen += 1
        errors = []
        if self.window:
            errors = [
                "nvidia-smi timed out"
                if self.errors_seen == 1
                else c019.verdicts.BREAKER_TEXT
            ]
        if self.problem == "status":
            errors = []
        return [
            {
                "collector": "HOST_TELEMETRY",
                "last_error_at": now.isoformat(),
                "last_success_at": (
                    now - timedelta(seconds=1)
                    if self.window
                    else now + timedelta(seconds=1)
                ).isoformat(),
                "errors": errors,
            }
        ]

    def control_plane_metrics(self) -> list[str]:
        if self.module is c018:
            v = c018.verdicts
            count = int(self.posted)
            return [
                "\n".join(
                    [
                        f"{v.FAULT_REJECTIONS_METRIC} {count}",
                        f'{v.COMPLETIONS_METRIC}{{path="{v.KERNEL_PATH}",status_class="4xx"}} {count}',
                        f'{v.ERRORING_METRIC}{{cluster_id="cluster-a",channel="NVIDIA_KERNEL"}} {count}',
                        f'{v.SILENT_METRIC}{{cluster_id="cluster-a",channel="NVIDIA_KERNEL"}} 0',
                        f'{v.UNRESOLVED_METRIC}{{kind="{v.UNPARSED_KIND}"}} {int(self.unparsed)}',
                    ]
                )
            ]
        v = c019.verdicts
        return [
            f'{v.ERRORING_METRIC}{{cluster_id="cluster-a",channel="HOST_TELEMETRY"}} {int(self.window)}\n'
            f'{v.SILENT_METRIC}{{cluster_id="cluster-a",channel="HOST_TELEMETRY"}} {int(self.problem == "silence" and self.window)}'
        ]

    def control_plane_logs(self, since: int) -> str:
        assert since >= 60
        return f"WARNING {c018.verdicts.REJECTION_LOG} request_id=fixture status=422"

    def node_activity(self, since: Any, **kwargs: Any) -> dict[str, Any]:
        if self.problem == "finding" or (
            self.pending
            and self.wait_name in {"finding", "unparsed-finding"}
            and self.poll == 0
        ):
            return {}
        if self.module is c018:
            return {
                "incidents": [{"incident_id": "incident-unparsed_xid_line"}],
                "workflows": [
                    {
                        "incident_id": "incident-unparsed_xid_line",
                        "status": "SUCCEEDED",
                        "official_steps": [{"operation": "FREEZE_EVIDENCE"}],
                    }
                ],
                "notifications": [
                    {
                        "incident_id": "incident-unparsed_xid_line",
                        "category": "OPERATOR_REVIEW",
                    }
                ],
                "evidence": [
                    {"record_id": "raw-a", "payload": {"message": self.marker}}
                ],
            }
        dropped = self.inventory[-1]["uuid"]
        status = (
            "RUNNING"
            if self.pending and self.wait_name == "workflow" and self.poll == 0
            else "SUCCEEDED"
        )
        operations = ["RUN_DCGM_DIAGNOSTIC", "VALIDATE_GPU"]
        if self.problem in {"reboot", "reboot-restore"}:
            operations.append("RESET_GPU")
        return {
            "incidents": [
                {
                    "incident_id": "incident-a",
                    "reasons": ["gpu_inventory_identity_changed", dropped],
                    "state": "RECOVERED"
                    if self.restored or self.problem == "already-restored"
                    else "ACTION_PENDING",
                }
            ],
            "workflows": [
                {
                    "incident_id": "incident-a",
                    "request_id": "workflow-a",
                    "status": status,
                    "official_steps": [{"operation": value} for value in operations],
                }
            ],
            "notifications": [
                {"incident_id": "incident-a", "category": category}
                for category in ("OPERATOR_REVIEW", "DCGM")
            ],
            "evidence": [
                {
                    "record_id": "inventory-a",
                    "payload": {
                        "devices": [
                            {"gpu_uuid": item["uuid"]} for item in self.inventory[:-1]
                        ],
                        "expected_gpu_count": 8,
                    },
                }
            ],
            "markers": [
                {
                    "incident_id": "incident-a",
                    "marker_id": "marker-snapshot-gpu_inventory_identity_changed",
                    "scope": {"gpu_uuids": [dropped]},
                    "severity": "critical",
                    "retired_at": "timestamp",
                    "retired_reason": "restored",
                }
            ],
        }

    def wait_until(self, accept: Any, *, name: str, **kwargs: Any) -> Any:
        self.waits.append(name)
        self.wait_name = name
        value = None
        for self.poll in range(3):
            value = accept()
            if value is not None:
                break
        self.wait_name = ""
        return None if self.fallback else value


@pytest.fixture(
    params=(c018, c019, c020), ids=("collect018", "collect019", "collect020")
)
def window_case(request: Any, monkeypatch: Any) -> WindowHost:
    module = request.param
    host = WindowHost(module)
    monkeypatch.setattr(module, "time", host.clock)

    class Restore:
        def __init__(self, regional: Any, profile: str) -> None:
            assert regional is host.regional

        def create_restore_workflow(self, **kwargs: Any) -> Any:
            host.calls.append(("restore-workflow", kwargs))
            if host.problem == "reboot-restore":
                raise RuntimeError("restore unavailable")
            return {"workflow_request_id": "restore-a"}

        def wait_workflow_id(self, request_id: str) -> Any:
            host.restored = True
            return {"request_id": request_id, "status": "SUCCEEDED"}

    if module is c020:
        monkeypatch.setattr(c020, "WarmSpareLiveFixture", Restore)
    return host
