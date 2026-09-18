from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha001_control_plane_failover as common
from scripts.e2e.regional import run_ha002_pdb_topology as ha
from tests.regional._cov95_ha001_harness import CHAIN, DEADLINE, HA001Harness, pod


def apply_patch(document: dict[str, Any], operations: list[dict[str, Any]]) -> dict:
    result = copy.deepcopy(document)
    for operation in operations:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in operation["path"].split("/")[1:]
        ]
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        if operation["op"] == "test":
            if parent.get(key) != operation["value"]:
                raise RuntimeError("unit API rejected stale CAS")
        elif operation["op"] in {"add", "replace"}:
            parent[key] = copy.deepcopy(operation["value"])
        elif operation["op"] == "remove":
            del parent[key]
        else:
            raise AssertionError("unsupported JSON Patch operation")
    return result


class Watchdog:
    def __init__(self) -> None:
        self.pid = 123456
        self.returncode: int | None = None
        self.waits: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: int) -> int:
        self.waits.append(timeout)
        self.returncode = 0
        return 0


class HA002Harness(HA001Harness):
    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, path: Path, *, spool: int = 0
    ) -> None:
        self.nodes = {
            f"node-{i}": {
                "metadata": {
                    "name": f"node-{i}",
                    "uid": f"node-uid-{i}",
                    "resourceVersion": "1",
                },
                "spec": {"unschedulable": False, "taints": []},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
            for i in range(3)
        }
        self.evicted: list[dict[str, Any]] = []
        self.evictions: list[dict[str, Any]] = []
        self.patches: list[list[dict[str, Any]]] = []
        self.signals: list[tuple[int, int]] = []
        self.watchdog = Watchdog()
        self.after_cordon: Any = None
        self.restore_failure: BaseException | None = None
        self.eviction_failure: str | None = None
        self.pdb_reopened = False
        self.pdb_reads: dict[str, int] = {}
        super().__init__(monkeypatch, path)
        self.replicas = {
            common.INGRESS_APP: 3,
            common.WORKER_APP: 6,
            common.SPOOL_APP: spool,
        }
        self.pods = {}
        for app, count in self.replicas.items():
            for i in range(count):
                name = f"{app}-{i}"
                value = pod(name, app)
                value["spec"]["nodeName"] = f"node-{i // 2 if count == 6 else i}"
                self.pods[name] = value
        monkeypatch.setattr(common, "run", self.run)
        monkeypatch.setattr(ha, "time", self.clock)
        monkeypatch.setattr(
            ha,
            "subprocess",
            SimpleNamespace(**{**vars(subprocess), "Popen": self.spawn}),
        )
        monkeypatch.setattr(
            ha, "os", SimpleNamespace(**{**vars(ha.os), "killpg": self.killpg})
        )
        monkeypatch.setattr(ha, "chain_preflight", lambda *a: {**CHAIN, "errors": []})
        monkeypatch.setattr(
            ha,
            "guard_build_plan",
            lambda **kw: {
                "attempt": kw["attempt"],
                "details": kw["details"],
                "preflight_passed": kw["preflight_passed"],
            },
        )
        envelope = ha.build_plan(path, 1, arguments=argparse.Namespace())
        self.plan = envelope["details"]
        directory = path / "cases" / ha.CASE_ID
        directory.mkdir(parents=True)
        (directory / "plan.json").write_text(json.dumps(envelope))
        self.events.clear()

    def spawn(self, command: list[str], **kwargs: Any) -> Watchdog:
        assert command[:2] == ["/bin/bash", "-c"]
        assert kwargs["start_new_session"] is True
        self.events.append("watchdog-start")
        return self.watchdog

    def killpg(self, pid: int, sig: int) -> None:
        assert pid == self.watchdog.pid
        self.signals.append((pid, sig))
        self.events.append("watchdog-stop")

    def cpu(self, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": list(self.pods.values())})
        if args[:2] == ("get", "pod") and "-l" not in args:
            value = self.pods.get(args[2])
            return json.dumps(value) if value else ""
        if args[:2] == ("get", "pdb") and len(args) == 5:
            name = args[2]
            self.pdb_reads[name] = self.pdb_reads.get(name, 0) + 1
            budget = 1 if self.pdb_reopened and self.pdb_reads[name] >= 2 else 0
            return json.dumps({"status": {"disruptionsAllowed": budget}})
        if args[:2] == ("get", "endpointslice"):
            count = len(common.ready_pods(common.INGRESS_APP))
            return json.dumps(
                {"items": [{"endpoints": [{"conditions": {"ready": True}}] * count}]}
            )
        return super().cpu(*args, **kwargs)

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "get" in argv:
            offset = argv.index("get")
            if argv[offset + 1] == "nodes":
                value = {"items": list(self.nodes.values())}
            else:
                value = self.nodes[argv[offset + 2]]
            return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")
        if "patch" in argv:
            name = argv[argv.index("node") + 1]
            operations = json.loads(kwargs["stdin"])
            restoring = any(
                item["path"] == "/spec/unschedulable" and item.get("value") is False
                for item in operations
            )
            if restoring and self.restore_failure is not None:
                raise self.restore_failure
            self.nodes[name] = apply_patch(self.nodes[name], operations)
            self.patches.append(operations)
            self.events.append("uncordon" if restoring else "cordon")
            if restoring:
                for removed in self.evicted:
                    old_name = removed["metadata"]["name"]
                    value = copy.deepcopy(removed)
                    value["metadata"]["name"] = f"{old_name}-replacement"
                    value["metadata"]["uid"] = f"{old_name}-replacement-uid"
                    self.pods[value["metadata"]["name"]] = value
                self.evicted.clear()
            elif self.after_cordon is not None:
                self.after_cordon()
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "create" in argv:
            value = json.loads(kwargs["stdin"])
            name = value["metadata"]["name"]
            options = value["deleteOptions"]
            assert self.pods[name]["metadata"]["uid"] == options["preconditions"]["uid"]
            self.evictions.append(value)
            if self.eviction_failure:
                return subprocess.CompletedProcess(argv, 1, "", self.eviction_failure)
            if options.get("dryRun") == ["All"]:
                return subprocess.CompletedProcess(argv, 1, "", "disruption budget")
            self.evicted.append(self.pods.pop(name))
            self.events.append(f"evict:{name}")
            return subprocess.CompletedProcess(argv, 0, "evicted", "")
        raise AssertionError("unexpected fake Kubernetes command")

    def execute(self) -> tuple[int, dict[str, Any]]:
        code = ha.execute(
            self.path,
            1,
            ha.CONFIRMATION,
            maintenance_window_end=DEADLINE,
            chain=copy.deepcopy(CHAIN),
        )
        report = self.path / "cases" / ha.CASE_ID / f"{ha.CASE_ID}.json"
        return code, json.loads(report.read_text())
