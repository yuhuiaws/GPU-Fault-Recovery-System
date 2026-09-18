from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr022_verdicts as verdicts
from scripts.e2e.regional import run_destr022_spare_reservation_reclaim as case
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings

SPARE = "node-spare"


class ReclaimHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path),
            "fake-hyperpod",
            SPARE,
            tmp_path / "predecessor.json",
            30,
        )
        self.incident_id = case.synthetic_incident(tmp_path, 1)
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.injected = False
        self.reclaimed = False
        self.after_patch_reads = 0
        self.reclaim_after_reads = 4
        self.unexpected_uncordon = False
        self.drop_patch = False
        self.patch_ack_lost = False
        self.node = {
            "name": SPARE,
            "uid": "spare-uid",
            "resource_version": "1",
            "ready": "True",
            "unschedulable": True,
            "taints": [],
            "labels": {
                verdicts.SPARE_LABEL: "true",
                verdicts.HYPERPOD_HEALTH_LABEL: "Schedulable",
            },
            "annotations": {
                verdicts.SPARE_RESERVATION_ANNOTATION: None,
                verdicts.SPARE_RESERVED_AT_ANNOTATION: None,
                verdicts.SPARE_POOL_STATE_ANNOTATION: "AVAILABLE",
            },
        }
        self.nodes = [
            {"name": SPARE, "uid": "spare-uid"},
            {"name": "node-active", "uid": "active-uid"},
        ]
        self.baseline = deepcopy(self.node)
        self.declared = [SPARE]
        self.workloads: list[dict[str, Any]] = []
        self.executor_env = [
            {"spare_failover": "true", "remote_state": "true", "allow_replace": "false"}
        ]
        self.predecessor_valid = True
        self.tests_pass = True
        self.lookup = {"found": False, "workflows": []}
        self.counter_delta = 1
        self.breadcrumb_stale = False
        self.events: list[dict[str, Any]] = []
        self.extra_logs: list[str] = []
        self.executor_value: Any = None
        self.regional = ReclaimRegional(self)
        self.warm = ReclaimWarm(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(
            case, "focused_tests", lambda _p: {"passed": self.tests_pass}
        )
        monkeypatch.setattr(
            case, "predecessor_evidence", lambda *_a: {"valid": self.predecessor_valid}
        )
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        if name in self.failures:
            raise self.failures[name]

    def plan(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "cases" / case.CASE_ID
        path.mkdir(parents=True, exist_ok=True)
        preflight = case.read_only_preflight(self.settings, path, self.incident_id)
        (path / "plan.json").write_text(
            json.dumps({"details": case.plan_details(self.settings, preflight)}),
            encoding="utf-8",
        )
        return preflight

    def execute(self, run_dir: Path, seconds: int = 3600) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / case.CASE_ID / f"{case.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))

    def executor_probe(self) -> dict[str, Any]:
        now = NOW if self.breadcrumb_stale else self.clock.now()
        count = 3 + (self.counter_delta if self.reclaimed else 0)
        return {
            "claim_state": {
                "executor_id": "cluster-a/executor",
                "execution_owners": ["gpu-fault-kubernetes-adapter"],
                "last_successful_claim_at": now.isoformat(),
                "counters": {"claimed_total": 10, verdicts.RECLAIM_COUNTER: count},
            },
            "claim_state_error": None,
            "env": {"GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER": "true"},
            "observed_at": (now + timedelta(seconds=1)).isoformat(),
        }


class ReclaimWarm:
    def __init__(self, harness: ReclaimHarness) -> None:
        self.h = harness
        self.regional = harness.regional

    def node_snapshot(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("node.snapshot", {"node": node, **kwargs})
        if self.h.injected and not self.h.reclaimed:
            self.h.after_patch_reads += 1
            if self.h.after_patch_reads >= self.h.reclaim_after_reads:
                self.h.reclaimed = True
                self.h.node["annotations"] = deepcopy(self.h.baseline["annotations"])
            if self.h.after_patch_reads >= 3 and self.h.unexpected_uncordon:
                self.h.node["unschedulable"] = False
        return deepcopy(self.h.node)

    def spare_nodes(self) -> list[str]:
        return list(self.h.declared)

    def executor_environment(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.executor_env)


class ReclaimRegional:
    def __init__(self, harness: ReclaimHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "patch":
            self.h.call("node.patch", {"plane": plane, "args": args})
            body = json.loads(args[args.index("-p") + 1])
            metadata = body["metadata"]
            assert metadata["uid"] == self.h.node["uid"], body
            assert metadata["resourceVersion"] == self.h.node["resource_version"], body
            assert args[2] == SPARE and plane == "gpu", args
            if not self.h.drop_patch:
                for field in ("labels", "annotations"):
                    for key, value in metadata.get(field, {}).items():
                        self.h.node[field][key] = value
                if "spec" in body:
                    self.h.node["unschedulable"] = body["spec"]["unschedulable"]
                self.h.node["resource_version"] = str(
                    int(self.h.node["resource_version"]) + 1
                )
            self.h.injected = True
            if self.h.patch_ack_lost:
                self.h.patch_ack_lost = False
                raise TimeoutError("fake patch ACK loss")
            return ""
        if args[0] == "exec":
            self.h.call("executor.probe", args)
            return json.dumps(
                self.h.executor_probe()
                if self.h.executor_value is None
                else self.h.executor_value
            )
        if args[0] == "logs":
            self.h.call("executor.logs", args)
            line = (
                "WARNING gpu_fault.cluster_executor reclaimed stale spare reservation: "
                f"node={SPARE} incident={self.h.incident_id} reason=reservation by "
                f"{self.h.incident_id} exceeded 86400s TTL"
            )
            return "\n".join([line, *self.h.extra_logs])
        raise AssertionError(f"unexpected fake request: {args}")

    def ready_pods(self, *args: str) -> list[dict[str, str]]:
        return [{"name": "executor-pod", "uid": "executor-uid"}]

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.h.workloads)

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "release_id": "release-test",
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        }

    def cpu_python(self, script: str, incident_id: str) -> dict[str, Any]:
        self.h.call("incident.lookup", incident_id)
        return deepcopy(self.h.lookup)

    def runtime_identity(self) -> dict[str, Any]:
        return ready_runtime()

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.h.call("gpu.nodes")
        return deepcopy(self.h.nodes)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: Any, **kwargs: Any) -> None:
        self.h.call("runtime.verify")
        assert identity == ready_runtime(), "restore must remain release-bound"
