from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr008_warm_spare_shortage as shortage
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from scripts.e2e.regional.warm_spare_fixture import (
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABELS,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
)

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
FAULT = "node-a"
SPARE = "node-b"
INCIDENT = "incident-owned"


class Clock:
    fromisoformat = staticmethod(datetime.fromisoformat)

    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return NOW.timestamp() + self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def now(self, tz: Any = None) -> datetime:
        return NOW + timedelta(seconds=self.elapsed)


def regional_settings(tmp_path: Path) -> RegionalLiveSettings:
    cpu = tmp_path / "cpu.config"
    gpu = tmp_path / "gpu.config"
    cpu.write_text("fake CPU transport\n", encoding="utf-8")
    gpu.write_text("fake GPU transport\n", encoding="utf-8")
    return RegionalLiveSettings(
        cpu, gpu, "fake-gpu", "test-system", "cluster-a", "us-west-2"
    )


def settings_for(module: ModuleType, tmp_path: Path) -> Any:
    site = tmp_path / "site.yaml"
    site.write_text("fake site input\n", encoding="utf-8")
    manifest = tmp_path / "workload.yaml"
    manifest.write_text("fake workload input\n", encoding="utf-8")
    values: dict[str, Any] = {
        "regional": regional_settings(tmp_path),
        "site_file": site,
        "manifest": manifest,
        "hyperpod_cluster": "fake-hyperpod",
        "fault_node": FAULT,
        "spare_node": SPARE,
        "predecessor_path": tmp_path / "predecessor.json",
    }
    if module is failover:
        values.update(job_id="owned-job", attempt_id="owned-job-a001")
    else:
        values.update(
            host_probe_image="example.test/probe@sha256:" + "a" * 64,
            scenarios=("no-spare", "topology-mismatch", "reserved-by-other"),
        )
    return module.Settings(**values)


def node_snapshot(node: str) -> dict[str, Any]:
    spare = node == SPARE
    return {
        "name": node,
        "uid": f"uid-{node}",
        "resource_version": "1",
        "ready": "True",
        "unschedulable": spare,
        "gpu_allocatable": 8,
        "labels": {
            SPARE_LABEL: "true" if spare else "false",
            HYPERPOD_HEALTH_LABEL: "Schedulable",
            INSTANCE_GROUP_LABEL: "group-a",
            INSTANCE_TYPE_LABELS[0]: "p4d.24xlarge",
        },
        "annotations": {},
        "taints": [],
    }


def profile() -> dict[str, Any]:
    return {
        "profile_version": "profile-v1",
        "warnings": [],
        "capabilities": [
            {
                "capability": name,
                "mode": "OWN",
                "owner": owner,
                "adapter": "regional-cluster-executor",
            }
            for name, owner in [
                ("nodeReplace", "gpu-fault-hyperpod-adapter"),
                ("workloadStop", "gpu-fault-kubernetes-adapter"),
                ("workloadRestart", "gpu-fault-kubernetes-adapter"),
            ]
        ],
    }


def observation() -> dict[str, Any]:
    return {
        "workload_phase": "RUNNING",
        "runtime_profile_version": "profile-v1",
        "workload_ids": ["workload-source"],
        "containers": [{"gpu_uuids": [f"GPU-{index}" for index in range(8)]}],
    }


def successful_failover() -> dict[str, Any]:
    return {
        "incident": {"incident_id": INCIDENT},
        "workflow": {
            "request_id": "workflow-replace",
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": operation,
                    "parameters": (
                        {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
                        if operation == "REPLACE_NODE"
                        else {}
                    ),
                }
                for operation in failover.EXPECTED_OPERATIONS
            ],
            "step_executions": [
                {
                    "operation": "REPLACE_NODE",
                    "status": "SUCCEEDED",
                    "details": {
                        "action": "SPARE_FAILOVER",
                        "activated_spare_nodes": [SPARE],
                        "provider_mutation_submitted": False,
                        "node_rebindings": {FAULT: SPARE},
                        "notification_id": "notification-owned",
                    },
                },
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "SUCCEEDED",
                    "details": {
                        "notification_context": {
                            "source_gpu_count": 8,
                            "target_gpu_count": 8,
                            "restart_count": 1,
                        }
                    },
                },
            ],
        },
        "agents": [{"node_id": FAULT, "lifecycle_state": "REVOKED"}],
        "notifications": [{"notification_id": "notification-owned"}],
        "restart_budget": {"budget": 1, "restart_count": 1},
    }


def failed_shortage(scenario: str, event_id: str) -> dict[str, Any]:
    return {
        "incident": {"incident_id": INCIDENT},
        "workflow": {
            "request_id": "workflow-shortage",
            "status": "FAILED",
            "official_steps": [
                {
                    "operation": "REPLACE_NODE",
                    "parameters": {
                        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
                        "activation_forbidden": True,
                    },
                }
            ],
            "step_executions": [
                {"operation": "STOP_WORKLOADS", "status": "SUCCEEDED"},
                {
                    "operation": "REPLACE_NODE",
                    "status": "FAILED",
                    "error": shortage.EXPECTED_REASON[scenario],
                },
            ],
        },
        "notifications": (
            [{"deduplication_key": f"hyperpod-spare-insufficient/{event_id}"}]
            if scenario in shortage.ALERT_SCENARIOS
            else []
        ),
        "markers": [{"marker_id": f"marker-{event_id}"}],
    }


class WarmHarness:
    """In-memory cluster and resources with observable lifecycle boundaries."""

    def __init__(
        self, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        self.module = module
        self.settings = settings_for(module, tmp_path)
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.nodes = {node: node_snapshot(node) for node in (FAULT, SPARE)}
        self.fault_state = {
            "profile": profile(),
            "release_id": "release-test",
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        }
        self.agents = [
            {"node_id": node, "lifecycle_state": "ACTIVE"} for node in (FAULT, SPARE)
        ]
        self.cluster = {"status": "InService", "node_recovery": "None"}
        self.executor_env = [
            {"spare_failover": "true", "remote_state": "true", "allow_replace": "false"}
        ]
        self.gates = [{"enabled": "true"}]
        self.gpu_workloads: list[dict[str, Any]] = []
        self.declared_spares = [SPARE]
        self.provider = {"sha256": "a" * 64, "count": 2}
        self.cpu_blast = {"nodes": ["cpu-a"], "pods": ["api-a"]}
        self.events: list[dict[str, Any]] = []
        self.test_result = {"passed": True, "returncode": 0}
        self.predecessor_valid = True
        self.injection_status = 200
        self.posted: dict[str, Any] = {}
        self.observations = [observation()]
        self.workflow = successful_failover() if module is failover else None
        self.returned_nodes: dict[str, dict[str, Any]] = {}
        self.target_nodes = [SPARE]
        self.lookup_incident = INCIDENT
        self.incident_state = "QUARANTINED"
        self.restore_status = "SUCCEEDED"
        self.prewarm_residuals = {"pods": False}
        self.regional = FakeRegional(self)
        self.warm = FakeWarm(self)
        self.workload = FakeWorkload(self)
        self.prewarm = FakePrewarm(self)
        monkeypatch.setattr(module, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(module, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(
            module, "ManagedWorkloadFixture", lambda *_a, **_k: self.workload
        )
        monkeypatch.setattr(
            module, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm
        )
        monkeypatch.setattr(module, "render_node_pinned_manifest", self.render)
        monkeypatch.setattr(module, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(module, "focused_tests", self.focused)
        monkeypatch.setattr(
            module, "reusable_focused_tests", lambda _p: self.test_result
        )
        monkeypatch.setattr(module, "record_focused_tests", self.record_focused)
        monkeypatch.setattr(module, "time", self.clock)
        monkeypatch.setattr(module, "datetime", self.clock)
        if module is shortage:
            monkeypatch.setattr(module, "NodeMutationFixture", self.node_mutation)
            monkeypatch.setattr(module, "read_capabilities", self.capabilities)

    def capabilities(self, regional: Any) -> dict[str, Any]:
        self.call("activation.capability")
        return {
            "supported": True,
            "populations": [
                {
                    "plane": plane,
                    "pods": [{"uid": f"{plane}-pod"}],
                    "probes": [{"local_guard_supported": True}],
                }
                for plane in ("cpu", "gpu")
            ],
        }

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        if name in self.advance_at:
            self.clock.sleep(self.advance_at[name])
        if name in self.failures:
            raise self.failures[name]

    def render(self, source: Path, target: Path, *, node: str) -> Path:
        self.call("render", {"source": str(source), "node": node})
        target.write_text("fake pinned workload\n", encoding="utf-8")
        return target

    def predecessor(self, path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        self.call("predecessor", {"case_id": case_id, **identity})
        return {"valid": self.predecessor_valid}

    def focused(self, _path: Path) -> dict[str, Any]:
        self.call("focused")
        return deepcopy(self.test_result)

    def record_focused(self, details: dict[str, Any], result: dict[str, Any]) -> None:
        details["focused_tests"] = deepcopy(result)

    def node_mutation(self, _warm: Any, node: str, **_kwargs: Any) -> Any:
        return FakeNodeMutation(self, node)

    def plan(self, run_dir: Path) -> dict[str, Any]:
        case_dir = run_dir / "cases" / self.module.CASE_ID
        case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        preflight = self.module.read_only_preflight(self.settings, case_dir)
        details = self.module.plan_details(self.settings, preflight)
        (case_dir / "plan.json").write_text(
            json.dumps({"details": details}), encoding="utf-8"
        )
        return preflight

    def execute(self, run_dir: Path, seconds: int = 3600) -> tuple[int, dict[str, Any]]:
        code = self.module.execute_case(
            self.settings, run_dir, 2, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / self.module.CASE_ID / f"{self.module.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class FakeRegional:
    def __init__(self, harness: WarmHarness):
        self.h = harness
        self.settings = harness.settings.regional

    def evidence_identity(self) -> dict[str, str]:
        return {
            "release_id": "release-test",
            "cluster_id": "cluster-a",
            "region": "us-west-2",
        }

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("regional.store", kwargs)
        return deepcopy(self.h.fault_state)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        # ``RegionalLiveFixture`` shape: ownership annotations, not the warm
        # fixture's full annotation map.
        value = deepcopy(self.h.returned_nodes.get(node, self.h.nodes[node]))
        value["ownership_annotations"] = {
            key: item
            for key, item in value.pop("annotations", {}).items()
            if key.startswith("gpu-fault.io/") and "spare" not in key
        }
        return value

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.gpu_workloads)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        self.h.call("cpu.blast")
        return deepcopy(self.h.cpu_blast)

    def provider_events(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        self.h.call("provider.events", {"start": start, "end": end})
        return deepcopy(self.h.events)

    def provider_events_provisional(self, end: datetime) -> bool:
        self.h.call("provider.provisional", end)
        return True


class FakeWarm:
    def __init__(self, harness: WarmHarness):
        self.h = harness
        self.regional = harness.regional

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call(f"node.{node}")
        return deepcopy(self.h.returned_nodes.get(node, self.h.nodes[node]))

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("warm.store", kwargs)
        if kwargs.get("event_id"):
            self.h.call("incident.lookup", kwargs["event_id"])
            return {"incident": {"incident_id": self.h.lookup_incident}}
        if kwargs.get("attempt_id"):
            return {"observations": deepcopy(self.h.observations)}
        return {"agents": deepcopy(self.h.agents)}

    def spare_nodes(self) -> list[str]:
        return list(self.h.declared_spares)

    def cluster_recovery(self) -> dict[str, Any]:
        return deepcopy(self.h.cluster)

    def executor_environment(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.executor_env)

    def synthetic_replacement_gates(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.gates)

    def provider_inventory(self) -> dict[str, Any]:
        self.h.call("provider.inventory")
        return deepcopy(self.h.provider)

    def post_synthetic_replacement(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.h.posted = deepcopy(payload)
        self.h.nodes[FAULT].update(
            unschedulable=True,
            taints=[{"key": QUARANTINE_TAINT, "effect": "NoSchedule"}],
            annotations={"gpu-fault.io/incident-id": INCIDENT},
        )
        if self.h.module is failover:
            self.h.nodes[SPARE].update(
                unschedulable=False,
                annotations={
                    SPARE_RESERVATION_ANNOTATION: INCIDENT,
                    SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
                },
            )
        self.h.call("replacement.post", payload)
        return {"status": self.h.injection_status}

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        if self.h.workflow is not None:
            return deepcopy(self.h.workflow)
        scenario = next(
            name for name in shortage.SCENARIOS if f"-{name}-" in kwargs["event_id"]
        )
        return failed_shortage(scenario, kwargs["event_id"])

    def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
        self.h.call("incident.idle", incident_id)
        return {"incident_id": incident_id, "idle": True}

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        self.h.call("incident.read", incident_id)
        return {
            "incident_id": incident_id,
            "cluster_id": "cluster-a",
            "state": self.h.incident_state,
            "job_id": self.h.posted.get("job_id", "owned-job"),
            "attempt_id": self.h.posted.get("attempt_id", "owned-attempt"),
            "node_ids": [FAULT],
        }

    def close_incident_with_evidence(self, incident_id: str, **kwargs: Any) -> dict:
        self.h.call("incident.close", {"incident_id": incident_id, **kwargs})
        return {"closed": True, "refusal": None, "state": "RECOVERED"}

    def release_spares(self, nodes: list[str], incident_id: str) -> dict[str, Any]:
        self.h.call("spares.release", {"nodes": nodes, "incident_id": incident_id})
        self.h.nodes[SPARE] = node_snapshot(SPARE)
        return {"released": nodes}

    def reactivate_agent(self, node: str) -> dict[str, Any]:
        self.h.call("agent.reactivate", node)
        return {"node": node}

    def wait_agent_active(self, node: str) -> dict[str, Any]:
        self.h.call("agent.active", node)
        return {"lifecycle_state": "ACTIVE"}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": "restore-owned"}

    def wait_workflow_id(self, request_id: str) -> dict[str, Any]:
        self.h.call("restore.wait", request_id)
        if self.h.restore_status == "SUCCEEDED":
            self.h.nodes[FAULT] = node_snapshot(FAULT)
        return {"request_id": request_id, "status": self.h.restore_status}


class FakeWorkload:
    def __init__(self, harness: WarmHarness):
        self.h = harness
        self.restart_state: dict[str, Any] | None = None

    def submit(self) -> dict[str, Any]:
        self.h.call("workload.submit")
        return {"submitted": True}

    def wait_running(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workload.running", kwargs)
        return {"pods": [{"uid": "source-uid", "node": FAULT}]}

    def authorize_restart(self, state: dict[str, Any]) -> None:
        self.h.call("workload.authorize_restart", state)
        self.restart_state = state

    def wait_restarted(self, source_uids: set[str], **kwargs: Any) -> dict[str, Any]:
        assert self.restart_state is not None
        self.h.call("workload.restarted", {"source_uids": source_uids, **kwargs})
        return {
            "pods": [
                {"uid": "target-uid", "node": node} for node in self.h.target_nodes
            ]
        }

    def delete(self) -> None:
        self.h.call("workload.delete")


class FakePrewarm:
    def __init__(self, harness: WarmHarness):
        self.h = harness

    def create(self, nodes: list[str]) -> None:
        self.h.call("prewarm.create", nodes)

    def cleanup(self) -> dict[str, bool]:
        self.h.call("prewarm.cleanup")
        return dict(self.h.prewarm_residuals)


class FakeNodeMutation:
    def __init__(self, harness: WarmHarness, node: str):
        self.h = harness
        self.node = node
        self.baseline = deepcopy(harness.nodes[node])

    def apply(self, patch: Any) -> None:
        self.h.call(
            "shortage.apply", {"labels": patch.labels, "annotations": patch.annotations}
        )
        for field in ("labels", "annotations"):
            for key, value in getattr(patch, field).items():
                if value is None:
                    self.h.nodes[self.node][field].pop(key, None)
                else:
                    self.h.nodes[self.node][field][key] = value

    def restore(self) -> dict[str, Any]:
        self.h.call("shortage.restore")
        self.h.nodes[self.node] = deepcopy(self.baseline)
        return {"restored": True}
