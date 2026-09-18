from __future__ import annotations

import copy
import functools
import io
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import ha010_verdicts as verdicts
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RUNTIME_IDENTITY_DEPLOYMENTS,
    RegionalLiveSettings,
)
from tests.regional._cov95_ha001_harness import DEADLINE, Clock, pod
from tests.regional._cov95_ha_reset_harness import RegionalFactory

T0 = datetime(2026, 9, 12, tzinfo=timezone.utc)


class SamplerProcess:
    def __init__(self, harness: HA010Harness, name: str, duration: int) -> None:
        self.harness, self.name, self.duration = harness, name, duration
        self.started = harness.clock.now
        self.stdin: Any = io.StringIO()
        self.returncode: int | None = None
        self.calls: list[str] = []
        self.stuck = False

    def communicate(self, **kwargs: Any) -> tuple[str, str]:
        self.calls.append("communicate")
        assert self.stdin is None, "closed sampler stdin must be detached"
        self.harness.clock.now = max(
            self.harness.clock.now, self.started + self.duration
        )
        self.returncode = self.harness.sampler_returncode
        samples = [
            {
                "t": (T0 + timedelta(seconds=self.started + offset)).timestamp(),
                "livez": self.harness.liveness_status,
                "healthz": 200,
                "registry_ready": True,
                "secret_drift": False,
            }
            for offset in range(0, self.duration, 2)
        ]
        return json.dumps({"samples": samples}), ""

    def poll(self) -> int | None:
        return None if self.stuck else self.returncode

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, *, timeout: int) -> int | None:
        self.calls.append("wait")
        return self.returncode


class HA010Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
        self.clock = Clock()
        self.events: list[str] = []
        self.path = path
        for name in ("cpu", "gpu"):
            (path / name).write_text("apiVersion: v1\n")
        self.settings = RegionalLiveSettings(
            cpu_kubeconfig=path / "cpu",
            gpu_kubeconfig=path / "gpu",
            gpu_context="unit-gpu",
            namespace="unit",
            cluster_id="unit-cluster",
            region="us-east-1",
        )
        self.pods: dict[str, list[dict[str, Any]]] = {}
        self.replicas = dict(zip(verdicts.CPU_DEPLOYMENTS, [3, 2, 1], strict=True))
        for app, count in self.replicas.items():
            self.pods[app] = [self.document(f"{app}-{i}", app) for i in range(count)]
        self.queue = {"depth": 0}
        self.remote_commands: dict[str, Any] = {"open_by_cluster": {}}
        self.busy_at_boundary = False
        self.store_reads = 0
        self.env = {
            verdicts.STALE_SECONDS_VARIABLE: "90",
            verdicts.STARTUP_RETRY_VARIABLE: "120",
        }
        self.health_status = 200
        self.binding_reads = 0
        self.binding_failure_at: int | None = None
        self.failover_requested = False
        self.rds_reads = 0
        self.replace_on_delete = False
        self.replacement_at: float | None = None
        self.never_ready = False
        self.liveness_status = 200
        self.sampler_returncode = 0
        self.processes: list[SamplerProcess] = []
        self.spawn_failure_at: int | None = None
        self.verify_failure = False
        self.cleanup_failure = False
        self.deadline = DEADLINE
        previous = path / "previous.json"
        previous.write_text(
            json.dumps(
                {
                    "case_id": ha.PREDECESSOR_CASE_ID,
                    "verdict": "PASS",
                    **self.evidence_identity(),
                }
            )
        )
        self.case_settings = ha.Settings(self.settings, "unit-rds", previous, 420)
        self.directory = path / "cases" / ha.CASE_ID
        self.directory.mkdir(parents=True)
        monkeypatch.setattr(ha, "RegionalLiveFixture", lambda _: self)
        monkeypatch.setattr(ha003, "RegionalLiveFixture", RegionalFactory(self))
        monkeypatch.setattr(
            ha, "regional_binding", lambda *a: SimpleNamespace(read=self.binding)
        )
        clock_api = SimpleNamespace(
            monotonic=self.clock.monotonic, sleep=self.clock.sleep
        )
        monkeypatch.setattr(ha, "time", clock_api)
        monkeypatch.setattr(ha003, "time", clock_api)
        monkeypatch.setattr(ha, "datetime", SimpleNamespace(now=self.now))
        monkeypatch.setattr(
            ha003,
            "wait_rds_failover",
            functools.partial(ha003.wait_rds_failover, sleep=self.clock.sleep),
        )
        monkeypatch.setattr(
            ha,
            "subprocess",
            SimpleNamespace(**{**vars(subprocess), "Popen": self.spawn}),
        )
        preflight = ha.read_only_preflight(self.case_settings, self.directory)
        assert preflight["errors"] == [], preflight["errors"]
        details = ha.plan_details(self.case_settings, preflight)
        (self.directory / "plan.json").write_text(json.dumps({"details": details}))
        self.events.clear()
        self.binding_reads = self.store_reads = 0

    def now(self, tz: Any = None) -> datetime:
        return T0 + timedelta(seconds=self.clock.now)

    def document(self, name: str, app: str) -> dict[str, Any]:
        value = pod(name, app)
        value["metadata"]["creationTimestamp"] = T0.isoformat()
        value["spec"]["containers"][0]["ports"] = [
            {"name": "http", "containerPort": 8080}
        ]
        value["status"]["conditions"][0]["lastTransitionTime"] = T0.isoformat()
        value["status"]["containerStatuses"][0]["restartCount"] = 0
        return value

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "unit-release", "cluster_id": "unit-cluster"}

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "release_state": {"phase": "complete"},
            "deployments": {
                plane: {
                    name: {
                        "generation": 1,
                        "observed_generation": 1,
                        "desired_replicas": 1,
                        "updated_replicas": 1,
                        "ready_replicas": 1,
                        "available_replicas": 1,
                    }
                    for name in names
                }
                for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items()
            },
        }

    def verify_runtime_identity(
        self, expected: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.events.append("verify-runtime")
        if self.verify_failure:
            raise RegionalFixtureError("unit runtime drift")
        assert expected == self.runtime_identity()
        return self.runtime_identity()

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"unit": "stable"}

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.store_reads += 1
        queue = (
            {"depth": 1}
            if self.busy_at_boundary and self.store_reads >= 2
            else self.queue
        )
        return {
            "release_id": "unit-release",
            "queue": queue,
            "remote_commands": self.remote_commands,
        }

    def binding(self, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        self.binding_reads += 1
        if self.binding_reads == self.binding_failure_at:
            raise RegionalFixtureError("unit Aurora binding drift")
        value = {"identity": {"database": "unit-rds"}}
        if expected is not None:
            assert expected == value
        return value

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        if args[:2] == ("get", "deployment"):
            return str(self.replicas[args[2]])
        if args[:2] == ("get", "pod"):
            app = args[3].removeprefix("app=")
            records = copy.deepcopy(self.pods[app])
            if self.replacement_at is not None:
                for record in records:
                    if record["metadata"]["name"] == "replacement":
                        ready = (
                            self.clock.now >= self.replacement_at
                            and not self.never_ready
                        )
                        record["status"]["containerStatuses"][0]["ready"] = ready
                        record["status"]["conditions"][0]["status"] = (
                            "True" if ready else "False"
                        )
            return json.dumps({"items": records})
        if args[0] == "exec":
            if kwargs["input_text"] == ha.ENV_READ:
                return json.dumps(self.env)
            return json.dumps(
                {
                    "http_status": self.health_status,
                    "payload": {
                        "regional_registry": {
                            "ready": True,
                            "secret_drift": False,
                            "secret_config_sha256": "unit-digest",
                        }
                    },
                }
            )
        if args[:2] == ("delete", "--raw"):
            name = args[2].rsplit("/", 1)[-1]
            records = self.pods[verdicts.ROLLED_DEPLOYMENT]
            before = next(item for item in records if item["metadata"]["name"] == name)
            expected_uid = json.loads(kwargs["input_text"])["preconditions"]["uid"]
            assert before["metadata"]["uid"] == expected_uid
            if self.replace_on_delete:
                raise RegionalFixtureError("unit stale Pod UID")
            records.remove(before)
            value = self.document("replacement", verdicts.ROLLED_DEPLOYMENT)
            self.replacement_at = self.clock.now + 20
            value["status"]["conditions"][0]["lastTransitionTime"] = (
                T0 + timedelta(seconds=self.replacement_at)
            ).isoformat()
            records.append(value)
            self.events.append("delete-pod")
            return ""
        if args[0] == "logs":
            return "regional registry bootstrap attempt 1 retrying\n"
        if args[0] == "rollout":
            self.events.append(f"ready:{args[2]}")
            if self.cleanup_failure:
                raise RegionalFixtureError("unit rollout unavailable")
            return ""
        raise AssertionError("unexpected CPU fake transport call")

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ["aws", "rds"]
        if argv[2] == "failover-db-cluster":
            self.failover_requested = True
            self.events.append("failover")
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        if self.failover_requested:
            self.rds_reads += 1
        cluster = {
            "Status": "failing-over" if self.rds_reads == 1 else "available",
            "DBClusterMembers": [
                {"DBInstanceIdentifier": "old", "IsClusterWriter": self.rds_reads < 2},
                {"DBInstanceIdentifier": "new", "IsClusterWriter": self.rds_reads >= 2},
            ],
        }
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"DBClusters": [cluster]}), ""
        )

    def spawn(self, argv: list[str], **kwargs: Any) -> SamplerProcess:
        if len(self.processes) + 1 == self.spawn_failure_at:
            raise OSError("unit spawn refused")
        name = argv[argv.index("exec") + 2]
        process = SamplerProcess(self, name, int(argv[-2]))
        self.processes.append(process)
        return process

    def execute(self) -> tuple[int, dict[str, Any]]:
        code = ha.execute_case(self.case_settings, self.path, 1, self.deadline)
        report = json.loads((self.directory / f"{ha.CASE_ID}.json").read_text())
        return code, report
