from __future__ import annotations

import copy
import functools
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_ha001_control_plane_failover as ha
from scripts.e2e.regional.regional_commands import RegionalFixtureError

DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)
CHAIN = {
    "identity": {"release_id": "unit-release", "cluster_id": "unit-cluster"},
    "predecessor": {"case_id": "unit-predecessor", "valid": True},
}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.01)


def pod(name: str, app: str, *, uid: str | None = None) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "uid": uid or f"uid-{name}",
            "namespace": ha.NAMESPACE,
            "labels": {"app": app},
        },
        "spec": {"nodeName": "unit-node", "containers": [{"name": "main"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"name": "main", "ready": True}],
        },
    }


def deployment(app: str, replicas: int) -> dict[str, Any]:
    return {
        "metadata": {"name": app, "uid": f"uid-{app}"},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {
                    "terminationGracePeriodSeconds": 90,
                    "containers": [
                        {
                            "name": "main",
                            "ports": [{"containerPort": 8080}],
                            "args": ["uvicorn --timeout-graceful-shutdown 60"],
                            "lifecycle": {
                                "preStop": {
                                    "exec": {"command": ["/bin/sh", "-c", "sleep 20"]}
                                }
                            },
                        }
                    ],
                }
            },
        },
        "status": {"readyReplicas": replicas},
    }


class HA001Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
        self.patch = monkeypatch
        self.path = path
        self.clock = Clock()
        self.store = InMemoryStore()
        self.replicas = {ha.INGRESS_APP: 3, ha.WORKER_APP: 3, ha.SPOOL_APP: 0}
        self.pods = {
            f"{app}-{i}": pod(f"{app}-{i}", app)
            for app, count in self.replicas.items()
            for i in range(count)
        }
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}
        self.events: list[str] = []
        self.deletions: list[tuple[str, str]] = []
        self.create_failure: tuple[str, BaseException] | None = None
        self.delete_failure: BaseException | None = None
        self.after_delete: Any = None
        self.before_seed: Any = None
        self.seed_ack_lost = False
        self.probe_errors: dict[str, int] = {}
        self.final_cycles: int | None = None
        self.ledger: dict[str, Any] = {"physical_count": 0, "operations": []}
        self.seed: dict[str, Any] = {}
        self.cleaned = False
        self.executor_id = "ha001-probe/unit-pod"
        self.patch.setenv("GPU_FAULT_STORE_URL", "postgresql://unit.invalid/ha")
        self.patch.setattr(ha, "cpu", self.cpu)
        self.patch.setattr(ha, "gpu", self.gpu)
        self.patch.setattr(ha, "time", self.clock)
        self.patch.setattr(ha, "log", lambda _: None)
        self.patch.setattr(ha, "database_residuals", self.residuals)
        self.patch.setattr(ha, "cleanup_closure", self.cleanup_closure)
        self.patch.setattr(
            ha,
            "observe_phase",
            functools.partial(
                ha.observe_phase, sleep=self.clock.sleep, clock=self.clock.monotonic
            ),
        )
        self.patch.setattr(
            ApplicationContext,
            "from_environment",
            classmethod(lambda cls: SimpleNamespace(store=self.store)),
        )
        self.plan = self.make_plan()

    def make_plan(self) -> dict[str, Any]:
        topology = ha.deployment_and_pdb_snapshot()
        ingress, workers = ha.ready_pods(ha.INGRESS_APP), ha.ready_pods(ha.WORKER_APP)
        plan = {
            "replicas": self.replicas,
            "failure_window_limits": {
                app: ha.failure_window_limit(topology["deployments"][app])
                for app in (ha.INGRESS_APP, ha.WORKER_APP)
            },
            "baseline": {
                "queue": {"depth": 0},
                "ingress_pods": ingress,
                "worker_pods": workers,
            },
            "targets": [
                {"phase": "ingress-1", "app": ha.INGRESS_APP, **ingress[0]},
                {"phase": "worker-1", "app": ha.WORKER_APP, **workers[0]},
                {"phase": "worker-2", "app": ha.WORKER_APP, **workers[1]},
            ],
        }
        case_dir = self.path / "cases" / ha.CASE_ID
        case_dir.mkdir(parents=True)
        (case_dir / "plan.json").write_text(json.dumps({"attempt": 1, "details": plan}))
        return plan

    def residuals(self) -> dict[str, int]:
        return {"objects": 0 if self.cleaned or not self.seed else 5, "links": 0}

    def cleanup_closure(self, seed: dict[str, Any]) -> dict[str, Any]:
        assert not self.resources, "Store cleanup must follow confirmed probe removal"
        if self.seed:
            assert seed == self.seed, "cleanup must use the exact owned seed"
        self.events.append("cleanup-closure")
        self.cleaned = True
        return {"remaining_objects": 0, "remaining_links": 0}

    def complete_commands(self) -> None:
        self.events.append("complete-commands")
        for _ in range(3):
            claimed = self.store.claim_remote_commands(
                "unit-cluster",
                self.executor_id,
                limit=1,
                lease_seconds=60,
                execution_owners={ha.OWNER},
            )
            assert len(claimed) == 1, "each seeded command must be claimable once"
            command = claimed[0]
            self.ledger["physical_count"] += 1
            self.ledger["operations"].append(command.step.operation.value)
            self.store.complete_remote_command(
                command.cluster_id,
                command.command_id,
                RemoteCommandResult(
                    lease_token=command.lease_token,
                    status=RemoteCommandStatus.SUCCEEDED,
                    details={
                        "simulated": True,
                        "cached": False,
                        "physical_count": self.ledger["physical_count"],
                    },
                ),
            )

    def cpu(self, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "deployment"):
            return json.dumps(
                {
                    "items": [
                        deployment(app, self.replicas[app])
                        for app in args[2:]
                        if app in self.replicas
                    ]
                }
            )
        if args[:2] == ("get", "pdb"):
            return json.dumps({"items": []})
        if args[:2] == ("get", "pod"):
            selector = args[args.index("-l") + 1]
            items = list(self.pods.values())
            if selector.startswith("app="):
                app = selector.removeprefix("app=")
                items = [p for p in items if p["metadata"]["labels"]["app"] == app]
            return json.dumps({"items": items})
        if args[:2] == ("get", "endpointslice"):
            return json.dumps(
                {"items": [{"endpoints": [{"conditions": {"ready": True}}] * 3}]}
            )
        if args[0] == "exec" and args[-1] == "8080":
            app = self.pods[args[2]]["metadata"]["labels"]["app"]
            ingress = app == ha.INGRESS_APP
            return json.dumps(
                {
                    "health": {
                        "service_role": "ingress" if ingress else "worker",
                        "processor_role": "inactive" if ingress else "active-consumer",
                        "processor_mode": "active-active",
                        "processor_epoch": "",
                    },
                    "processor_active_consumer": 0.0 if ingress else 4.0,
                }
            )
        if args[0] == "exec":
            if self.before_seed is not None and ha.OWNER in args:
                self.before_seed()
            output = io.StringIO()
            with self.patch.context() as local, redirect_stdout(output):
                local.setattr(sys, "argv", ["probe", *args[6:]])
                exec(kwargs["stdin"], {})
            value = json.loads(output.getvalue().splitlines()[-1])
            if "command_ids" in value:
                self.seed = value
                self.events.append("seed")
                if self.seed_ack_lost:
                    raise RuntimeError("unit lost seed ACK")
                self.complete_commands()
            return output.getvalue()
        if args[:2] == ("delete", "--raw"):
            name = unquote(args[2].rsplit("/", 1)[-1])
            uid = json.loads(kwargs["stdin"])["preconditions"]["uid"]
            current = self.pods[name]
            assert current["metadata"]["uid"] == uid, "fake API rejected stale UID"
            app = current["metadata"]["labels"]["app"]
            del self.pods[name]
            replacement = f"{app}-replacement-{len(self.deletions)}"
            self.pods[replacement] = pod(replacement, app)
            self.deletions.append((name, uid))
            self.events.append(f"delete:{name}")
            if self.after_delete is not None:
                self.after_delete()
            return ""
        if args[:2] == ("rollout", "status"):
            self.events.append(f"rollout:{args[2]}")
            return ""
        raise AssertionError(f"unexpected CPU transport operation: {args[:2]}")

    def gpu(self, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "deployment"):
            value = deployment("executor", 1)
            container = value["spec"]["template"]["spec"]["containers"][0]
            container.update(
                image="unit-image",
                env=[
                    {"name": name, "value": "unit-artifact-identity"}
                    for name in (
                        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
                        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
                    )
                ],
            )
            return json.dumps(value)
        if args[0] == "get":
            kind = {"pod": "Pod", "configmap": "ConfigMap"}.get(args[1], args[1])
            value = self.resources.get((kind, args[2]))
            if args[-1] == "name":
                return args[2] if value else ""
            return json.dumps(value) if value else ""
        if args[0] == "create":
            value = json.loads(kwargs["stdin"])
            kind, name = value["kind"], value["metadata"]["name"]
            assert (kind, name) not in self.resources, "create must not overwrite"
            value["metadata"]["uid"] = f"owned-{kind}-{name}"
            self.resources[kind, name] = value
            self.events.append(f"create:{kind}")
            if self.create_failure is not None and self.create_failure[0] == kind:
                raise self.create_failure[1]
            return ""
        if args[:2] == ("delete", "--raw"):
            if self.delete_failure is not None:
                raise self.delete_failure
            name = unquote(args[2].rsplit("/", 1)[-1])
            uid = json.loads(kwargs["stdin"])["preconditions"]["uid"]
            kind = {"pods": "Pod", "configmaps": "ConfigMap"}[args[2].split("/")[-2]]
            key = kind, name
            assert self.resources[key]["metadata"]["uid"] == uid, "UID CAS rejected"
            del self.resources[key]
            self.events.append(f"cleanup:{key[0]}")
            return ""
        if args[0] == "wait":
            return ""
        if args[0] == "exec" and "sh" in args:
            return "present"
        if args[0] == "exec" and "touch" in args:
            self.events.append("stop-probe")
            return ""
        if args[0] == "exec" and args[-1] == "/state/ready.json":
            return json.dumps(
                {"cluster_id": "unit-cluster", "executor_id": self.executor_id}
            )
        if args[0] == "exec" and args[-1] == "/state/stats.json":
            cycles = (
                self.final_cycles
                if self.final_cycles is not None
                else int(self.clock.now / ha.PROBE_CYCLE_SECONDS) + 1
            )
            return json.dumps(
                {
                    "current_failure_window_seconds": 0,
                    "max_failure_window_seconds": 0,
                    "error_counts": self.probe_errors,
                    "counters": {"claim_success": cycles, "health_success": cycles},
                    "ledger": self.ledger,
                }
            )
        raise AssertionError(f"unexpected GPU transport operation: {args[:2]}")

    def execute(self) -> tuple[int, dict[str, Any]]:
        code = ha.execute_case(
            self.path,
            1,
            ha.CONFIRMATION,
            maintenance_window_end=DEADLINE,
            chain=copy.deepcopy(CHAIN),
        )
        report = self.path / "cases" / ha.CASE_ID / f"{ha.CASE_ID}.json"
        return code, json.loads(report.read_text())

    def drift_target(self, index: int) -> None:
        name = self.plan["targets"][index]["name"]
        self.pods[name]["metadata"]["uid"] = "foreign-replacement"

    def abort(self) -> None:
        raise RegionalFixtureError("unit acceptance abort")
