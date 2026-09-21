"""Fake CPU API and explicit Kubernetes defaults for resource contract tests."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import destr008_parallel as parallel
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

NAMESPACE = "gpu-fault-system"
NAME = "destr008-example"
CONTROL_UID = "control-uid"
JOB_UID = "job-uid"
POD_UID = "pod-uid"
POD_NAME = NAME + "-abcde"
IMAGE = "registry.example/cpu@sha256:" + "a" * 64
STARTED = "2026-09-13T00:01:00Z"
FINISHED = "2026-09-13T00:02:00Z"


def metadata(name: str, uid: str) -> dict[str, Any]:
    return {"name": name, "namespace": NAMESPACE, "uid": uid, "resourceVersion": "10"}


@dataclass
class Harness:
    regional: RegionalLiveFixture
    source: Path
    objects: dict[tuple[str, str], dict[str, Any]]
    calls: list[tuple[str, ...]] = field(default_factory=list)
    response: str | None = None

    @property
    def deployment(self) -> dict[str, Any]:
        return self.objects["deployment", resources.DEPLOYMENT]

    @property
    def container(self) -> dict[str, Any]:
        return cast(
            dict[str, Any], self.deployment["spec"]["template"]["spec"]["containers"][0]
        )

    def kube(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu", "resource discovery must never address a GPU cluster"
        assert args[0] == "get", (
            "this harness only permits read-only resource discovery"
        )
        assert args[-2:] == ("-o", "json"), "discovery must request structured objects"
        assert kwargs == {"timeout": 30}, "source reads must have a bounded timeout"
        self.calls.append((plane, *args))
        if self.response is not None:
            return self.response
        assert args[1] in {"namespace", "deployment", "configmap"}, (
            "resource discovery must not read any Secret"
        )
        return json.dumps(self.objects[args[1], args[2]])

    def plan(self) -> wire.Plan:
        return wire.Plan(
            schema_version=1,
            run_id="destr008-example",
            cluster_id="cluster-a",
            job_id="job-a",
            attempt_id="attempt-a",
            event_id="event-a",
            release_id="release-a",
            fault_node="fault-a",
            spare_node="spare-a",
            runtime_profile_version="profile-a",
            workload_ids=["training/a"],
            probe_sha256=wire.source_sha256(self.source),
            created_at=1789257600,
            deadline_at=1789258200,
            fence=wire.Fence(
                policy="policy-a",
                policy_uid="policy-uid",
                binding="binding-a",
                binding_uid="binding-uid",
                marker="b" * 64,
                node="spare-a",
                node_uid="node-uid",
            ),
        )

    def manifests(self) -> list[dict[str, Any]]:
        return resources.supporting_manifests(
            self.plan(),
            resources.read_runtime(self.regional),
            name=NAME,
            control_uid=CONTROL_UID,
        )


def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    cpu, gpu = tmp_path / "cpu-config", tmp_path / "gpu-config"
    cpu.write_text("fixture-only\n")
    gpu.write_text("fixture-only\n")
    source = tmp_path / "source"
    source.mkdir()
    for file in wire.SOURCE_FILES:
        (source / file).write_bytes(b"# Inert resource fixture.\r\n")
    monkeypatch.setattr(resources, "CODE_SOURCE", source)
    monkeypatch.setattr(parallel, "workers", 1)  # deterministic fakes
    regional = RegionalLiveFixture(
        RegionalLiveSettings(cpu, gpu, "gpu-a", NAMESPACE, "cluster-a", "us-west-2")
    )
    namespace = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": NAMESPACE, "uid": "namespace-uid", "resourceVersion": "1"},
    }
    config = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata("worker-postgres", "config-uid"),
        "data": {
            resources.MODE_KEYS[0]: "dedicated",
            resources.MODE_KEYS[1]: "legacy",
            "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT": "false",
            "GPU_FAULT_STORE_URL_FILE": resources.STORE_DIRECTORY
            + "/"
            + resources.STORE_KEY,
            "UNRELATED_SETTING": "not-inherited",
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            **metadata(resources.DEPLOYMENT, "deployment-uid"),
            "generation": 7,
        },
        "spec": {
            "replicas": 2,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "control-worker",
                            "image": IMAGE,
                            "envFrom": [{"configMapRef": {"name": "worker-postgres"}}],
                            "env": [
                                {
                                    "name": "GPU_FAULT_STORE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": resources.STORE_SECRET,
                                            "key": resources.STORE_KEY,
                                        }
                                    },
                                },
                                {
                                    "name": "GPU_FAULT_EXECUTION_TOKEN",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "not-projected",
                                            "key": "execution-token",
                                        }
                                    },
                                },
                            ],
                            "volumeMounts": [
                                {
                                    "name": "store",
                                    "mountPath": resources.STORE_DIRECTORY,
                                    "readOnly": True,
                                },
                                {
                                    "name": "ca",
                                    "mountPath": resources.CA_DIRECTORY,
                                    "readOnly": True,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "store",
                            "secret": {"secretName": resources.STORE_SECRET},
                        },
                        {"name": "ca", "configMap": {"name": resources.CA_CONFIGMAP}},
                    ],
                }
            },
        },
        "status": {
            "observedGeneration": 7,
            "replicas": 2,
            "updatedReplicas": 2,
            "readyReplicas": 2,
            "availableReplicas": 2,
            "conditions": [
                {"type": "Available", "status": "True"},
                {"type": "Progressing", "status": "True"},
            ],
        },
    }
    ca = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": metadata(resources.CA_CONFIGMAP, "ca-uid"),
        "data": {resources.CA_KEY: "public CA fixture\n"},
    }
    fixture = Harness(
        regional,
        source,
        {
            ("namespace", NAMESPACE): namespace,
            ("deployment", resources.DEPLOYMENT): deployment,
            ("configmap", "worker-postgres"): config,
            ("configmap", resources.CA_CONFIGMAP): ca,
        },
    )
    monkeypatch.setattr(regional, "kubectl", fixture.kube)
    return fixture


def pod_defaults(spec: dict[str, Any]) -> None:
    """Apply API serialization/defaults independently of the production normalizer."""
    spec["serviceAccount"] = spec["serviceAccountName"]
    for key in ("hostNetwork", "hostPID", "hostIPC", "initContainers"):
        spec.pop(key)
    for container in spec["containers"]:
        container.pop("envFrom")
        container.update(
            terminationMessagePath="/dev/termination-log",
            terminationMessagePolicy="File",
        )
        container["env"][0]["value"] = ""
        container["env"][0]["valueFrom"]["secretKeyRef"]["optional"] = False
        for mount in container["volumeMounts"]:
            mount.setdefault("readOnly", False)
            mount["mountPropagation"] = "None"
    for volume in spec["volumes"]:
        for kind in ("configMap", "secret"):
            if kind in volume:
                volume[kind].setdefault("defaultMode", 420)
                volume[kind]["optional"] = False
        if "projected" in volume:
            volume["projected"]["sources"][1]["configMap"]["optional"] = False
        if "emptyDir" in volume:
            volume["emptyDir"]["medium"] = ""


def controller_labels() -> dict[str, str]:
    return {
        "controller-uid": JOB_UID,
        "batch.kubernetes.io/controller-uid": JOB_UID,
        "job-name": NAME,
        "batch.kubernetes.io/job-name": NAME,
    }


def admitted(expected: dict[str, Any], *, uid: str) -> dict[str, Any]:
    actual = copy.deepcopy(expected)
    actual["metadata"].update(
        uid=uid,
        resourceVersion="20",
        creationTimestamp=STARTED,
        generation=1,
        managedFields=[{"manager": "kube-controller-manager"}],
    )
    kind = expected["kind"]
    if kind == "ServiceAccount":
        actual.update(secrets=[], imagePullSecrets=[])
    elif kind == "RoleBinding":
        actual["subjects"][0]["apiGroup"] = ""
    elif kind == "ConfigMap":
        actual["binaryData"] = {}
    elif kind == "Job":
        spec = actual["spec"]
        spec.update(
            parallelism=1,
            completions=1,
            completionMode="NonIndexed",
            manualSelector=False,
            suspend=False,
            podReplacementPolicy="TerminatingOrFailed",
            managedBy="kubernetes.io/job-controller",
            selector={"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
        )
        spec["template"]["metadata"].update(creationTimestamp=None)
        spec["template"]["metadata"]["labels"].update(controller_labels())
        pod_defaults(spec["template"]["spec"])
        actual["status"] = {"active": 1, "ready": 0}
    return actual


def pod(
    expected_job: dict[str, Any], phase: resources.WatchdogPhase = "gated"
) -> dict[str, Any]:
    template = copy.deepcopy(expected_job["spec"]["template"])
    actual = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            **template["metadata"],
            **metadata(POD_NAME, POD_UID),
            "generateName": NAME + "-",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": NAME,
                    "uid": JOB_UID,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "spec": template["spec"],
        "status": {
            "phase": "Pending",
            "conditions": [
                {"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}
            ],
        },
    }
    actual["metadata"]["labels"].update(controller_labels())
    pod_defaults(actual["spec"])
    # Priority admission stamps both on every admitted Pod (no PriorityClass).
    actual["spec"]["priority"] = 0
    actual["spec"]["preemptionPolicy"] = "PreemptLowerPriority"
    if phase != "gated":
        actual["spec"].pop("schedulingGates")
        actual["spec"]["nodeName"] = "cpu-node"
        running = phase == "running"
        container_id = "containerd://" + "c" * 64
        container: dict[str, Any] = {
            "name": "cancellation-watchdog",
            # kubelet reports the image config id for a digest-pinned pull
            "image": "sha256:" + "e" * 64,
            "imageID": IMAGE,
            "containerID": container_id,
            "restartCount": 0,
            "lastState": {},
            "ready": running,
            "started": running,
            "state": {"running": {"startedAt": STARTED}}
            if running
            else {
                "terminated": {
                    "startedAt": STARTED,
                    "finishedAt": FINISHED,
                    "exitCode": 0,
                    "reason": "Completed",
                    "containerID": container_id,
                }
            },
        }
        actual["status"] = {
            "phase": "Running" if running else "Succeeded",
            "startTime": STARTED,
            "containerStatuses": [container],
            "conditions": [
                {"type": key, "status": value}
                for key, value in (
                    ("PodScheduled", "True"),
                    ("Initialized", "True"),
                    ("Ready", "True" if running else "False"),
                    ("ContainersReady", "True" if running else "False"),
                )
            ],
        }
    return actual


def validate_pod(
    actual: dict[str, Any],
    expected: dict[str, Any],
    phase: resources.WatchdogPhase = "gated",
) -> None:
    resources.validate_pod(
        actual,
        expected,
        job_uid=JOB_UID,
        pod_name=POD_NAME,
        pod_uid=POD_UID,
        phase=phase,
    )


def edit(document: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
