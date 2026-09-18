"""Public fake Kubernetes boundary; no command here reaches a real cluster."""

from __future__ import annotations

import base64
import copy
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_resources as resources
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic

RUN_ID = "a" * 32
RUNTIME_IMAGE = "example.invalid/control-plane@sha256:" + "b" * 64
PG_IMAGE = "postgres:16-bookworm@sha256:" + "c" * 64
IMAGE_ID = "containerd://sha256:" + "d" * 64
POD_UID = "owned-pod"
INTENT = "e" * 64


@pytest.fixture(autouse=True)
def blocked_external_transports(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "HA011 unit tests require an explicit fake SQL or Kubernetes transport"
        )

    try:
        import psycopg
    except ImportError:
        pass
    else:
        monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr(resources, "run_fixture_command", forbidden)


def settings_at(tmp_path: Path) -> contracts.Settings:
    kubeconfig = tmp_path / "cpu-kubeconfig"
    kubeconfig.write_text("public-fake-kubeconfig\n", encoding="utf-8")
    predecessor = tmp_path / "predecessor.json"
    write_json_atomic(
        predecessor,
        {
            "case_id": "GF-REGIONAL-HA-010",
            "verdict": "PASS",
            "release_id": "release-fixture",
            "cluster_id": "cluster-fixture",
        },
    )
    return contracts.Settings(
        kubeconfig,
        "cpu-fixture",
        "gpu-fault-system",
        "cluster-fixture",
        RUN_ID,
        PG_IMAGE,
        "GF-REGIONAL-HA-010",
        predecessor,
        "us-west-2",
    )


def passing_probe(*, pod_uid: str = POD_UID, intent: str = INTENT) -> dict[str, Any]:
    role = {
        "same_durable_work": True,
        "work_sha256": "f" * 64,
        "fence_changed": True,
        "early_claim_count": 0,
        "old_exitcode": -9,
        "late_completion_refused": True,
        "replacement_live_before_late": True,
        "replacement_live_after_late": True,
        "backlog_at_crash": 4,
        "completed_count": 4,
        "final_depth": 0,
        "owned_processes_stopped": True,
        "old_cpu_seconds": 0.1,
        "replacement_cpu_seconds": 0.1,
        "owners": [pod_uid + ":101", pod_uid + ":102"],
    }
    return {
        "case_id": contracts.CASE_ID,
        "validation_scope": contracts.BOUNDARY,
        "isolation_id": RUN_ID,
        "pod_uid": pod_uid,
        "arm_intent_sha256": intent,
        "postgres_major": 16,
        "business_worker_targeted": False,
        "cpu_saturation_tested": False,
        "roles": {name: copy.deepcopy(role) for name in contracts.ROLES},
    }


class Clock:
    def __init__(self, step: float = 0.1) -> None:
        self.now = 10.0
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay


class Kubernetes:
    def __init__(self, settings: contracts.Settings) -> None:
        self.settings = settings
        self.objects: dict[tuple[str, str | None, str], dict[str, Any]] = {}
        self.calls: list[tuple[str, list[str]]] = []
        self.created: list[dict[str, Any]] = []
        self.before: Any = lambda _args: None
        self.deleted = False
        self.armed = False
        self.probe: dict[str, Any] | None = None
        self.omit_defaults = True
        self.seed_business()

    def put(self, value: dict[str, Any]) -> None:
        metadata = value["metadata"]
        self.objects[value["kind"], metadata.get("namespace"), metadata["name"]] = value

    def seed_business(self) -> None:
        namespace = self.settings.namespace
        self.put(
            {"kind": "Namespace", "metadata": {"name": namespace, "uid": "business-ns"}}
        )
        self.put(
            {
                "kind": "ConfigMap",
                "metadata": {
                    "name": "gpu-fault-regional-release-state",
                    "namespace": namespace,
                    "uid": "release-cm",
                },
                "data": {"state.json": json.dumps({"release_id": "release-fixture"})},
            }
        )
        self.put(
            {
                "kind": "Deployment",
                "metadata": {
                    "name": contracts.DEPLOYMENT,
                    "namespace": namespace,
                    "uid": "business-deploy",
                    "generation": 1,
                },
                "spec": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [{"name": "api", "image": RUNTIME_IMAGE}]
                        }
                    },
                },
                "status": {
                    "observedGeneration": 1,
                    "updatedReplicas": 1,
                    "readyReplicas": 1,
                },
            }
        )
        self.business_pods = [
            {
                "metadata": {
                    "name": "business-worker",
                    "namespace": namespace,
                    "uid": "business-worker-uid",
                },
                "spec": {
                    "nodeName": "cpu-node",
                    "containers": [{"name": "api", "image": RUNTIME_IMAGE}],
                },
                "status": {
                    "containerStatuses": [
                        {"name": "api", "ready": True, "imageID": IMAGE_ID}
                    ]
                },
            }
        ]
        self.put(
            {
                "kind": "Node",
                "metadata": {
                    "name": "cpu-node",
                    "uid": "cpu-node-uid",
                    "labels": {
                        "kubernetes.io/os": "linux",
                        "kubernetes.io/hostname": "cpu-host",
                        "node.kubernetes.io/instance-type": "m6i.2xlarge",
                        "topology.kubernetes.io/region": "us-west-2",
                    },
                },
                "spec": {"providerID": "aws:///example-zone/i-00000000000000001"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"cpu": "8", "memory": "32Gi"},
                    "allocatable": {"cpu": "7500m", "memory": "30Gi"},
                },
            }
        )
        self.put(
            {
                "kind": "Lease",
                "metadata": {
                    "name": "cpu-node",
                    "namespace": "kube-node-lease",
                    "uid": "cpu-lease-uid",
                    "ownerReferences": [{"uid": "cpu-node-uid"}],
                },
                "spec": {
                    "holderIdentity": "cpu-node",
                    "renewTime": datetime.now(timezone.utc).isoformat(),
                },
            }
        )

    def object(self, kind: str, name: str, *, business: bool = False) -> dict[str, Any]:
        namespace = (
            self.settings.namespace if business else self.settings.isolated_namespace
        )
        if kind in {"Namespace", "Node", "PriorityClass"}:
            namespace = None
        elif kind == "Lease":
            namespace = "kube-node-lease"
        return self.objects[kind, namespace, name]

    def transport(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[:3] == [
            "kubectl",
            "--kubeconfig",
            str(self.settings.cpu_kubeconfig),
        ], "commands require the explicit fake CPU kubeconfig"
        assert command[3:5] == ["--context", self.settings.cpu_context], (
            "the test boundary forbids implicit or GPU contexts"
        )
        args = command[5:]
        namespace = None
        if args[:1] == ["--namespace"]:
            namespace, args = args[1], args[2:]
        self.calls.append((namespace or "", args))
        self.before(args)
        output = self.dispatch(args, namespace, kwargs.get("input_text"))
        return subprocess.CompletedProcess(command, 0, output, "")

    def dispatch(self, args: list[str], namespace: str | None, text: str | None) -> str:
        if args[:2] == ["get", "pods"]:
            return json.dumps({"items": self.business_pods})
        if args[0] == "get":
            value = self.objects.get((args[1], namespace, args[2]))
            if value is None:
                assert "--ignore-not-found=true" in args, (
                    "unexpected required resource read"
                )
                return ""
            return json.dumps(value)
        if args[0] == "create":
            value = json.loads(text or "")
            metadata = value["metadata"]
            target_ns = metadata.get("namespace")
            assert target_ns in (None, self.settings.isolated_namespace), (
                "creation escaped the owned namespace"
            )
            if value["kind"] == "Namespace":
                assert metadata["name"] == self.settings.isolated_namespace, (
                    "business namespace mutation"
                )
            metadata["uid"] = (
                POD_UID if value["kind"] == "Pod" else "owned-" + value["kind"].lower()
            )
            metadata["resourceVersion"] = "1"
            self.created.append(copy.deepcopy(value))
            if value["kind"] == "Secret":
                value["data"] = {
                    key: base64.b64encode(item.encode()).decode()
                    for key, item in value.pop("stringData").items()
                }
            if value["kind"] == "Pod":
                value["spec"]["serviceAccount"] = "default"
                value["status"] = {
                    "containerStatuses": [
                        {
                            "name": name,
                            "restartCount": 0,
                            "imageID": image_id,
                            "state": {"waiting": {"reason": "SchedulingGated"}},
                        }
                        for name, image_id in (
                            ("runtime", IMAGE_ID),
                            ("postgres", "containerd://sha256:" + "c" * 64),
                        )
                    ]
                }
                if self.omit_defaults:
                    for key in (
                        "hostNetwork",
                        "hostPID",
                        "hostIPC",
                        "initContainers",
                        "imagePullSecrets",
                    ):
                        value["spec"].pop(key)
                    for container in value["spec"]["containers"]:
                        container.pop("envFrom")
                        container["securityContext"].pop("privileged")
            if value["kind"] == "NetworkPolicy" and self.omit_defaults:
                value["spec"].pop("ingress")
                value["spec"].pop("egress")
            if value["kind"] == "PriorityClass" and self.omit_defaults:
                value.pop("globalDefault")
            self.put(value)
            return metadata["uid"]
        if args[0] == "patch":
            assert namespace == self.settings.isolated_namespace, (
                "activation escaped the owned namespace"
            )
            pod = self.object("Pod", contracts.POD_NAME)
            operations = json.loads(text or "")
            assert operations[:2] == [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": pod["metadata"]["uid"],
                },
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": pod["metadata"]["resourceVersion"],
                },
            ], "activation requires both UID and resource-version preconditions"
            assert operations[2]["value"] == pod["spec"]["schedulingGates"], (
                "activation must test the expected owned scheduling gate"
            )
            assert operations[3] == {"op": "remove", "path": "/spec/schedulingGates"}, (
                "activation may change only the scheduling gate"
            )
            pod["spec"].pop("schedulingGates")
            pod["spec"]["nodeName"] = "cpu-node"
            pod["metadata"]["resourceVersion"] = "2"
            for status in pod.get("status", {}).get("containerStatuses", []):
                status["state"] = {"running": {"startedAt": "observed"}}
            return pod["metadata"]["uid"]
        if args[0] == "exec":
            assert namespace == self.settings.isolated_namespace, (
                "arm escaped the owned namespace"
            )
            pod_uid, isolation_id, intent = args[-3:]
            assert pod_uid == POD_UID and isolation_id == RUN_ID, (
                "arm did not bind the observed Pod"
            )
            self.armed = True
            pod = self.object("Pod", contracts.POD_NAME)
            pod["status"]["containerStatuses"][0]["state"] = {
                "terminated": {"exitCode": 0}
            }
            self.probe = passing_probe(intent=intent)
            return json.dumps(
                {
                    "armed": True,
                    "pod_uid": pod_uid,
                    "isolation_id": isolation_id,
                    "intent_sha256": intent,
                }
            )
        if args[0] == "logs":
            assert self.armed, (
                "probe evidence must not exist before the owned arm barrier"
            )
            return json.dumps(self.probe)
        if args[0] == "delete":
            priority_path = (
                "/apis/scheduling.k8s.io/v1/priorityclasses/"
                + self.settings.priority_class_name
            )
            if args[:3] == ["delete", "--raw", priority_path]:
                assert namespace is None, "PriorityClass is cluster-scoped"
                current = self.object(
                    "PriorityClass", self.settings.priority_class_name
                )
                options = json.loads(text or "")
                assert options["preconditions"] == {
                    "uid": current["metadata"]["uid"],
                    "resourceVersion": current["metadata"]["resourceVersion"],
                }, "PriorityClass deletion must bind UID and resource version"
                del self.objects[
                    "PriorityClass", None, self.settings.priority_class_name
                ]
                return "{}"
            assert args[:3] == [
                "delete",
                "--raw",
                "/api/v1/namespaces/" + self.settings.isolated_namespace,
            ], "cleanup must address only the new isolated namespace"
            options = json.loads(text or "")
            current = self.object("Namespace", self.settings.isolated_namespace)
            assert options["preconditions"] == {"uid": current["metadata"]["uid"]}, (
                "namespace cleanup must carry the current owned UID precondition"
            )
            self.deleted = True
            self.objects = {
                key: value
                for key, value in self.objects.items()
                if key[1] != self.settings.isolated_namespace
                and key != ("Namespace", None, self.settings.isolated_namespace)
            }
            return "{}"
        raise AssertionError(f"unhandled fake Kubernetes operation: {args[0]}")


def install_kubernetes(monkeypatch: Any, settings: contracts.Settings) -> Kubernetes:
    fake = Kubernetes(settings)
    monkeypatch.setattr(resources, "run_fixture_command", fake.transport)
    return fake


def deadline() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=20)
