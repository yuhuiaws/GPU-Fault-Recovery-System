from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_data as data
from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as capacity


class AmpWire:
    """Read-only AMP responses; the notification preflight remains real."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        self.error: Exception | None = None
        self.config: dict[str, Any] = {
            "route": {
                "receiver": "notifications",
                "routes": [
                    {
                        "receiver": "drill-sink",
                        "matchers": ['cluster_id=~"perf-cap-.*"'],
                    }
                ],
            },
            "receivers": [
                {"name": "drill-sink"},
                {
                    "name": "notifications",
                    "sns_configs": [
                        {"topic_arn": "arn:aws:sns:us-west-2:123456789012:test"}
                    ],
                },
            ],
        }
        monkeypatch.setattr(registry, "AMP_WORKSPACE_ID", "")
        monkeypatch.setattr(registry, "run", self.run)

    def run(
        self, argv: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is True, "AMP preflight reads must propagate transport failures"
        operation = argv[2] if len(argv) > 2 else ""
        assert operation in {"list-workspaces", "describe-alert-manager-definition"}, (
            "the AMP fake permits only workspace and Alertmanager reads"
        )
        expected = ["aws", "amp", operation]
        if operation == "describe-alert-manager-definition":
            expected.extend(("--workspace-id", "ws-perf-test"))
        expected.extend(("--region", registry.AWS_REGION, "--output", "json"))
        assert argv == expected, "AMP reads must retain the selected workspace/Region"
        self.calls.append(operation)
        if self.error is not None:
            raise self.error
        value = (
            {
                "workspaces": [
                    {
                        "workspaceId": "ws-perf-test",
                        "status": {"statusCode": "ACTIVE"},
                        "tags": {"gpu-fault:site-id": "perf-test"},
                    }
                ]
            }
            if operation == "list-workspaces"
            else {
                "alertManagerDefinition": {
                    "status": {"statusCode": "ACTIVE"},
                    "data": base64.b64encode(json.dumps(self.config).encode()).decode(),
                }
            }
        )
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(value).encode(), stderr=b""
        )


class CapacityWire:
    """In-memory API boundaries; callers, registration and receipts stay real."""

    def __init__(self, artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.artifacts = artifacts
        self.amp = AmpWire(monkeypatch)
        self.objects: dict[str, dict[str, Any]] = {}
        self.entries: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.creations: list[dict[str, Any]] = []
        self.teardowns: list[dict[str, Any]] = []
        self.fail_create = ""
        self.fail_delete = ""
        self.fail_publish = False
        self.fail_data = False
        self.remaining: object = 0
        self.fail_wait = False
        self.identity = {
            "release_id": "test-release",
            "wheel_configmap": "test-wheel",
            "wheel_sha256": "a" * 64,
            "module_digest": "b" * 64,
        }
        for module in (capacity, action, registry):
            monkeypatch.setattr(module, "dataplane", self.dataplane)
            monkeypatch.setattr(module, "control", self.control)
        monkeypatch.setattr(capacity, "DATAPLANE_CONTEXT", "fake-context")
        monkeypatch.setattr(registry, "validate_notification_safety", lambda: None)
        monkeypatch.setattr(
            registry,
            "sync_dataplane_connection_secret",
            lambda: self.events.append("connection-mirror") or {"changed": False},
        )
        monkeypatch.setattr(
            registry, "load_registry", lambda: copy.deepcopy(self.entries)
        )
        monkeypatch.setattr(registry, "write_registry", self.publish)
        # Secret/head/replica alignment and the Secret byte baseline are the
        # registry module's own contract (test_perf_registry_alignment).
        monkeypatch.setattr(
            registry, "verify_registry_alignment", lambda **_kwargs: {"aligned": True}
        )
        monkeypatch.setattr(registry, "capture_secret_baseline", lambda *_args: None)
        monkeypatch.setattr(data, "invoke", self.data)
        for module in (capacity, action):
            monkeypatch.setattr(module, "release_identity", lambda: dict(self.identity))
            monkeypatch.setattr(module, "teardown", self.teardown)
            monkeypatch.setattr(module, "artifact_dir", lambda *_args: self.artifacts)
            monkeypatch.setattr(module, "scrape_metrics", lambda *_args: {})
            monkeypatch.setattr(module, "scrape_cgroup", lambda *_args: {})
            monkeypatch.setattr(module, "postgres_counters", lambda: {})
        monkeypatch.setattr(capacity, "control_pods", lambda: ["api"])
        monkeypatch.setattr(capacity, "release_id", lambda: "test-release")
        monkeypatch.setattr(capacity, "wait_for_job", self.wait_for_job)
        monkeypatch.setattr(capacity, "collect_logs", lambda *_args: [])
        monkeypatch.setattr(capacity, "queue_drain", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(capacity, "processor_priority_latency", lambda: {})
        monkeypatch.setattr(capacity, "aurora_window", lambda *_args: {})
        monkeypatch.setattr(capacity, "TopSampler", lambda *_args: self)
        monkeypatch.setattr(
            action,
            "executor_identity",
            lambda **_kwargs: {
                "executor_protocol_version": 1,
                "executor_artifact_sha256": "c" * 64,
                "executor_compatibility_digest": "d" * 64,
            },
        )
        monkeypatch.setattr(action, "release_agent_identity", lambda: {})
        monkeypatch.setattr(action, "seed", self.seed)
        monkeypatch.setattr(
            action,
            "database_snapshot",
            lambda _run_id: {
                "workflows": {"SUCCEEDED": 1},
                "commands": {"SUCCEEDED": 10},
            },
        )
        monkeypatch.setattr(action, "database_details", lambda _run_id: {})
        monkeypatch.setattr(
            action,
            "collect_executor_logs",
            lambda *_args: [{"expected_commands": 10, "completed_commands": 10}],
        )

    def control(self, *args: str, **_kwargs: Any) -> str:
        if args[:1] == ("exec",):
            # The live remediation budget read: answer with the raw capacity
            # tier so the fake runs keep their full executor concurrency.
            return json.dumps(
                {
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "128",
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": "4",
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": "4",
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": "1",
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": "1",
                }
            )
        assert args[:2] == ("get", "pod"), "unexpected control-plane API boundary"
        return "api"

    def publish(self, entries: list[dict[str, Any]], **_kwargs: Any) -> None:
        self.entries = copy.deepcopy(entries)
        self.events.append("publish")
        if self.fail_publish:
            raise TimeoutError("registration ACK lost")

    def data(
        self,
        _control: Any,
        *,
        run_id: str,
        cluster_ids: list[str],
        cleanup: bool,
        force_nonterminal: bool = False,
    ) -> dict[str, Any]:
        assert run_id == "run-a", "data scope must retain the caller run identity"
        assert cluster_ids == ["perf-cap-000"], (
            "data scope must be the exact intent inventory"
        )
        self.events.append("data-cleanup" if cleanup else "data-inspect")
        if cleanup:
            assert set(self.objects) <= {f"secret/{registry.TOKEN_SECRET}"}, (
                "owned Jobs and supporting resources must stop before data cleanup"
            )
            if self.fail_data:
                raise RuntimeError("data cleanup unavailable")
        return {"run_id": run_id, "cluster_ids": cluster_ids, "total": self.remaining}

    def teardown(self, **kwargs: Any) -> None:
        assert kwargs["artifacts"] == self.artifacts, (
            "teardown must reuse the registration directory"
        )
        assert kwargs["run_id"] == "run-a", "teardown must retain the run identity"
        assert self.events[-1] in {"data-cleanup", "data-inspect"}, (
            "exact data proof must immediately precede shared teardown"
        )
        proof = json.loads((self.artifacts / "registry-token-proof.json").read_text())
        token = self.objects[f"secret/{registry.TOKEN_SECRET}"]["metadata"]
        assert proof["uid"] == token["uid"], (
            "the original token UID must reach shared teardown"
        )
        self.events.append("teardown")
        self.teardowns.append(kwargs)
        self.entries = [
            item for item in self.entries if item.get("synthetic_run_id") != "run-a"
        ]
        del self.objects[f"secret/{registry.TOKEN_SECRET}"]

    def dataplane(self, *args: str, **kwargs: Any) -> str:
        if args[0] == "get":
            return self.read(args)
        assert kwargs.get("check", True) is True, (
            "mutations must never suppress API errors"
        )
        if args[0] == "create":
            return self.create(args, kwargs)
        if args[0] == "patch":
            value = self.objects[f"configmap/{args[2]}"]
            patch = json.loads(kwargs["stdin"])
            assert patch[0]["value"] == value["metadata"]["uid"], (
                "start gate patch needs the receipt UID"
            )
            assert patch[1]["value"] == value["metadata"]["resourceVersion"], (
                "start gate patch needs a fresh resourceVersion"
            )
            value["data"] = patch[2]["value"]
            value["metadata"]["resourceVersion"] = "2"
            self.events.append("gate-release")
            return json.dumps(value)
        assert args[:2] == ("delete", "--raw"), (
            "cleanup must use conditional raw deletion"
        )
        options = json.loads(kwargs["stdin"])
        uid = options["preconditions"]["uid"]
        key = next(
            key
            for key, value in self.objects.items()
            if value["metadata"]["uid"] == uid
        )
        value = self.objects[key]
        assert (
            options["preconditions"]["resourceVersion"]
            == value["metadata"]["resourceVersion"]
        ), "cleanup must pin the current version with the recorded UID"
        assert options["propagationPolicy"] == "Foreground", (
            "producer descendants must stop first"
        )
        del self.objects[key]
        self.events.append(f"delete:{key}")
        if key == self.fail_delete:
            raise TimeoutError("delete ACK lost")
        return ""

    def create(self, args: tuple[str, ...], kwargs: dict[str, Any]) -> str:
        assert self.amp.calls == [
            "list-workspaces",
            "describe-alert-manager-definition",
        ], "AMP drill-sink validation must precede token or load-resource creation"
        value = json.loads(kwargs["stdin"])
        kind, name = value["kind"].lower(), value["metadata"]["name"]
        key = f"{kind}/{name}"
        assert key not in self.objects, "a caller must not replace an existing resource"
        assert value["metadata"]["labels"][registry.RUN_LABEL] == "run-a", (
            "resource creation must carry the acceptance run label"
        )
        if kind == "secret":
            intent = json.loads(
                (self.artifacts / "registry-registration-intent.json").read_text()
            )
            assert intent["run_id"] == "run-a", (
                "registration intent must precede token creation"
            )
        else:
            receipt = json.loads(
                (self.artifacts / "capacity-resources.json").read_text()
            )
            assert receipt["resources"][key]["uid"] is None, (
                "creation intent must precede the API call"
            )
        if kind == "job":
            assert (
                value["spec"]["template"]["metadata"]["labels"][registry.RUN_LABEL]
                == "run-a"
            ), "Job descendants must inherit run ownership"
        value["metadata"].update(uid=f"uid-{len(self.creations)}", resourceVersion="1")
        self.objects[key] = value
        self.creations.append(
            {"kind": value["kind"], "metadata": copy.deepcopy(value["metadata"])}
        )
        self.events.append(f"create:{key}")
        if name == self.fail_create:
            raise TimeoutError("create ACK lost")
        return json.dumps(
            value["metadata"] if "jsonpath={.metadata}" in args else value
        )

    def read(self, args: tuple[str, ...]) -> str:
        if args[1] in {"pod", "pods"}:
            if "name" in args:
                return "pod/load-0"
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "load-0"},
                            "status": {
                                "phase": "Running",
                                "conditions": [{"type": "Ready", "status": "True"}],
                            },
                        }
                    ]
                }
            )
        value = self.objects.get(f"{args[1]}/{args[2]}")
        if value is None:
            return ""
        if "jsonpath={.metadata}" in args:
            return json.dumps(value["metadata"])
        if "jsonpath={.status.succeeded}|{.status.failed}|{.spec.completions}" in args:
            return "1||1"
        return json.dumps(value)

    def seed(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["run_id"] == "run-a", "seed must use the registered run identity"
        self.events.append("seed")
        return {"workflows_created": 1}

    def wait_for_job(self, *_args: Any) -> str:
        if self.fail_wait:
            raise RuntimeError("load observation failed")
        return "Complete"

    def start(self) -> None:
        self.events.append("sampler-start")

    def stop(self) -> None:
        self.events.append("sampler-stop")

    def join(self, **_kwargs: Any) -> None:
        self.events.append("sampler-joined")


def capacity_args(command: str, artifacts: Path, *extra: str) -> list[str]:
    args = [
        command,
        "--suite-id",
        "run-a",
        "--clusters",
        "1",
        "--nodes-per-cluster",
        "1",
        "--artifact-root",
        str(artifacts),
    ]
    if command in {"run", "purge", "teardown"}:
        args.extend(("--run-dir", str(artifacts)))
    return [*args, *extra]


def action_args(artifacts: Path, *extra: str) -> list[str]:
    return [
        "--suite-id",
        "run-a",
        "--clusters",
        "1",
        "--workflows-per-cluster",
        "1",
        "--nodes-per-workflow",
        "1",
        "--artifact-root",
        str(artifacts),
        *extra,
    ]
