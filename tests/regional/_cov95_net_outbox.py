"""Stateful in-memory host protocol for the NET-008 runner."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from scripts.e2e.regional import run_net008_outbox_dead_letter as runner
from tests.regional._cov95_collect_net import Clock


class OutboxHost:
    def __init__(self) -> None:
        self.node = "node-a"
        self.clock = Clock()
        self.calls: list[tuple[str, ...]] = []
        self.window = False
        self.blocked = False
        self.stopped = False
        self.seeded = False
        self.requeued = False
        self.markers: list[str] = []
        self.delivered: list[str] = []
        self.failures: dict[str, list[BaseException | None]] = {}
        self.problem = ""
        self.fallback = False
        self.pending = False
        self.poll = 0
        self.wait_name = ""
        self.waits: list[str] = []
        self.depth: Any = 0
        self.inventory = [{"uuid": "GPU-aaaaaaaa", "pci_bus_id": "0000:af:00.0"}]
        self.regional = SimpleNamespace()

    def service(self) -> dict[str, str]:
        return {
            "ActiveState": "inactive" if self.stopped else "active",
            "NRestarts": "0",
            "MainPID": "123",
            "InvocationID": "invocation-a",
        }

    def execute(self, verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((verb, *args))
        key = verb + (
            ":" + args[args.index("--action") + 1] if verb == "outbox" else ""
        )
        pending = self.failures.get(key, [])
        if pending:
            error = pending.pop(0)
            if error is not None:
                raise error
        if verb == "resolve":
            return {"endpoint_ipv4": [] if self.problem == "no-ip" else ["192.0.2.1"]}
        if verb == "open-window":
            self.window = True
            return {
                "unit": runner.verdicts.UNIT,
                "overrides": ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
                "after": {"ActiveState": "inactive"}
                if self.problem == "window"
                else self.service(),
            }
        if verb == "close-window":
            self.window = False
            return {"dropin_removed": True}
        if verb == "write-kmsg":
            marker = args[args.index("--marker") + 1]
            self.markers.append(marker)
            if not self.window and not self.blocked:
                self.delivered = list(self.markers)
                self.requeued = False
            return {"marker": marker}
        if verb in {"stop-unit", "start-unit"}:
            self.stopped = verb == "stop-unit"
            after = self.service()
            if self.problem == verb:
                after["ActiveState"] = "unknown"
            return {"unit": runner.verdicts.UNIT, "after": after}
        if verb == "seed-outbox-record":
            assert self.stopped, "outbox seed must be ordered after confirmed stop"
            self.seeded = True
            return {"record_id": "seeded"}
        if verb == "purge-outbox-record":
            assert self.stopped, "purge must be ordered after confirmed stop"
            self.seeded = False
            return {"removed": 1}
        if verb == "outbox":
            action = args[args.index("--action") + 1]
            if action == "list":
                return {
                    "lines": []
                    if self.problem == "listing"
                    else [
                        "dead\t" + runner.verdicts.RETIRED_CHANNEL_PATH + "\tHTTP 404"
                    ]
                }
            assert action == "requeue-dead"
            self.requeued = True
            return {
                "refused_without_yes": True,
                "output": "requeued 1 dead record",
                "stats": {"replayable": 1},
            }
        if verb == "block":
            self.blocked = True
            return {
                "connectivity": {"192.0.2.1": self.problem == "block"},
                "timer": {"ActiveState": "active"},
            }
        if verb == "unblock":
            self.blocked = False
            return {
                "rules": [] if self.problem != "unblock" else ["remaining"],
                "connectivity": {"192.0.2.1": True},
                "timer": {"ActiveState": "inactive"},
            }
        raise AssertionError(f"unexpected fake host command {verb}")

    def snapshot(self, marker: str | None = None) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        transient = [
            {
                "path": "/events",
                "marker_present": True,
                "marker_count": 1,
                "replayable": True,
                "error": "HTTP 403",
            }
            for item in self.markers
            if "-t" in item
        ]
        if self.window:
            records = transient
        if self.seeded:
            records.append(
                {
                    "path": runner.verdicts.RETIRED_CHANNEL_PATH,
                    "marker_present": True,
                    "replayable": self.requeued,
                    "error": "HTTP 404",
                }
            )
        if self.problem == "transient" and self.window:
            records = [{**item, "replayable": False} for item in records]
        if self.problem == "dead-letter" and self.seeded:
            records = []
        if self.pending:
            if self.wait_name == "transient" and self.poll == 0:
                records = []
            if self.wait_name == "replay" and self.poll == 0:
                records = transient
            if self.wait_name == "replay" and self.poll == 1:
                records = []
        replayable = sum(item.get("replayable") is True for item in records)
        return {
            "gpu_inventory": self.inventory,
            "services": {runner.verdicts.UNIT: self.service()},
            "outboxes": {
                "kernel": {
                    "records": deepcopy(records),
                    "stats": {
                        "depth": len(records) if self.markers else self.depth,
                        "replayable": replayable,
                        "dead": len(records) - replayable,
                    },
                }
            },
            "kmsg_stream": {
                "pid": 123,
                "invocation_id": "invocation-a",
                "kmsg_streams": [{"fd": 3, "pos": len(self.markers)}],
            },
        }

    def node_activity(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if (
            self.problem == "evidence"
            or (self.pending and self.wait_name == "replay" and self.poll == 2)
            or (self.pending and self.wait_name == "blackout-replay" and self.poll == 0)
        ):
            return {"evidence": []}
        return {
            "evidence": [
                {"record_id": f"record-{index}", "payload": {"message": marker}}
                for index, marker in enumerate(self.delivered)
            ]
        }

    def wait_until(self, accept: Any, *, name: str, **kwargs: Any) -> Any:
        self.waits.append(name)
        self.wait_name = name
        value = None
        for self.poll in range(4):
            value = accept()
            if value is not None:
                break
        self.wait_name = ""
        return None if self.fallback else value
