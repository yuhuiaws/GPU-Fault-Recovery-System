from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

from scripts.e2e.regional import notify008_fixture as fixture
from scripts.e2e.regional import notify008_runner as runner
from scripts.e2e.regional import notify008_target as binding
from scripts.e2e.regional.notify008_bundle import source_bundle
from scripts.e2e.regional.notify008_resources import CASE_LABEL, GATE, OWNER_LABEL
from scripts.e2e.regional.probes.notify008_protocol import CASE_ID, Target, digest
from tests.regional._cov95_notify008_support import RUN_ID, report_record

RUNTIME_IMAGE = "example.invalid/cpu@sha256:" + "1" * 64
POSTGRES_IMAGE = "example.invalid/pg16@sha256:" + "2" * 64
MODULE_DIGEST = "3" * 64


def uid(index):
    return str(UUID(int=index))


def target():
    return Target(
        run_id=RUN_ID,
        cluster_id="cluster-local",
        region="us-west-2",
        release_id="release-local",
        node="cpu-node",
        node_uid=uid(1),
        source_pod="source-pod",
        source_pod_uid=uid(2),
        deployment_uid=uid(3),
        deployment_generation=4,
        runtime_image=RUNTIME_IMAGE,
        postgres_image=POSTGRES_IMAGE,
        runtime_version="0.10.0",
        runtime_module_digest=MODULE_DIGEST,
    )


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeCPU:
    def __init__(self):
        self.objects = {}
        self.pod = None
        self.calls = []
        self.armed = False
        self.stopped = False
        self.lose_create = set()
        self.lose_arm = False
        self.lose_delete = False
        self.gc_priorityclass = True
        self.mutate_running = None
        self.mutate_after_run = None
        self.fail_probe = None
        self.probe_exception = None
        self.next_uid = 10
        self.source = self.source_objects()

    def source_objects(self):
        labels = {"app": binding.APP}
        node = {
            "metadata": {
                "name": "cpu-node",
                "uid": uid(1),
                "labels": {
                    "kubernetes.io/os": "linux",
                    "node.kubernetes.io/instance-type": "m7i.large",
                },
            },
            "status": {
                "capacity": {"cpu": "2"},
                "allocatable": {"cpu": "1900m"},
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        }
        pod = {
            "metadata": {
                "name": "source-pod",
                "uid": uid(2),
                "labels": labels,
                "ownerReferences": [
                    {"kind": "ReplicaSet", "name": "source-rs", "uid": uid(4)}
                ],
            },
            "spec": {
                "nodeName": "cpu-node",
                "containers": [{"name": "worker", "image": RUNTIME_IMAGE}],
            },
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {
                        "name": "worker",
                        "imageID": RUNTIME_IMAGE,
                        "ready": True,
                        "restartCount": 0,
                    }
                ],
            },
        }
        deployment = {
            "metadata": {"name": binding.APP, "uid": uid(3), "generation": 4},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {"containers": [{"name": "worker", "image": RUNTIME_IMAGE}]}
                },
            },
            "status": {
                "observedGeneration": 4,
                "updatedReplicas": 1,
                "readyReplicas": 1,
                "availableReplicas": 1,
            },
        }
        replica = {
            "metadata": {
                "name": "source-rs",
                "uid": uid(4),
                "ownerReferences": [{"kind": "Deployment", "uid": uid(3)}],
            }
        }
        release = {
            "metadata": {"name": "gpu-fault-regional-release-state"},
            "data": {
                "state.json": json.dumps(
                    {
                        "release_id": "release-local",
                        "phase": "complete",
                        "transaction_committed": True,
                    }
                )
            },
        }
        return {
            "node": node,
            "pod": pod,
            "deployment": deployment,
            "replicaset": replica,
            "configmap": release,
        }

    def read(self, kind, name, *, namespace=None):
        self.calls.append(("read", kind, name, namespace))
        if namespace == "gpu-fault-system" or kind == "node":
            return deepcopy(self.source.get(kind))
        if kind == "pod":
            return deepcopy(self.pod)
        return deepcopy(self.objects.get(kind))

    def pods(self, namespace, job_uid):
        self.calls.append(("pods", namespace, job_uid))
        return [] if self.pod is None else [deepcopy(self.pod)]

    def call(self, *arguments, namespace=None, body=None, timeout=30):
        self.calls.append(tuple(arguments))
        verb = arguments[0]
        if verb == "get":
            if "--raw" in arguments:
                return json.dumps({"minor": "33"})
            return json.dumps({"items": [self.source["pod"]]})
        if verb == "create":
            value = deepcopy(body)
            kind = value["kind"].lower()
            self.next_uid += 1
            value["metadata"].update(uid=uid(self.next_uid), resourceVersion="1")
            if kind == "priorityclass":
                value.pop("globalDefault")
            if kind == "job":
                for key in ("hostNetwork", "hostPID", "hostIPC"):
                    value["spec"]["template"]["spec"].pop(key, None)
                value["spec"]["selector"] = {
                    "matchLabels": {
                        "batch.kubernetes.io/controller-uid": value["metadata"]["uid"]
                    }
                }
            self.objects[kind] = value
            if kind in self.lose_create:
                self.lose_create.remove(kind)
                raise TimeoutError("fake create acknowledgement lost")
            return json.dumps(value)
        if verb == "patch":
            kind = arguments[1]
            current = self.objects[kind] if kind == "job" else self.pod
            patches = json.loads(arguments[arguments.index("-p") + 1])
            assert patches[0]["value"] == current["metadata"]["uid"], (
                "patch UID precondition differs"
            )
            assert patches[1]["value"] == current["metadata"]["resourceVersion"], (
                "patch resource version differs"
            )
            if kind == "job":
                priority = self.objects["priorityclass"]
                spec = current["spec"]["template"]["spec"]
                assert spec["priorityClassName"] == priority["metadata"]["name"], (
                    "Pod admission requires the run-owned PriorityClass"
                )
                assert spec["preemptionPolicy"] == priority["preemptionPolicy"], (
                    "Pod preemption policy must match the computed class policy"
                )
                assert spec["priority"] == priority["value"] == 0, (
                    "the probe must have ordinary zero priority"
                )
                current["spec"]["suspend"] = False
                self.pod = {
                    "metadata": {
                        "name": "notify008-pod",
                        "namespace": RUN_ID,
                        "uid": uid(30),
                        "resourceVersion": "1",
                        "labels": {OWNER_LABEL: RUN_ID, CASE_LABEL: CASE_ID},
                        "ownerReferences": [
                            {
                                "kind": "Job",
                                "uid": current["metadata"]["uid"],
                                "controller": True,
                            }
                        ],
                    },
                    "spec": deepcopy(current["spec"]["template"]["spec"]),
                    "status": {"phase": "Pending"},
                }
            else:
                assert self.pod["spec"]["schedulingGates"] == [{"name": GATE}], (
                    "Pod must remain gated until inspected"
                )
                self.pod["spec"].pop("schedulingGates")
                self.pod["spec"]["nodeName"] = "cpu-node"
                self.pod["status"] = {
                    "phase": "Running",
                    "containerStatuses": [
                        {
                            "name": name,
                            "ready": True,
                            "restartCount": 0,
                            "imageID": image,
                            "containerID": f"containerd://{name}",
                            "state": {"running": {}},
                        }
                        for name, image in (
                            ("runtime", RUNTIME_IMAGE),
                            ("database", POSTGRES_IMAGE),
                        )
                    ],
                }
                if self.mutate_running is not None:
                    self.mutate_running(self.pod)
            current["metadata"]["resourceVersion"] = "2"
            return json.dumps(current)
        if verb == "exec":
            if namespace == "gpu-fault-system":
                return json.dumps(
                    {
                        "distribution": "gpu-fault-control-plane",
                        "version": "0.10.0",
                        "module_digest": MODULE_DIGEST,
                    }
                )
            command = arguments[
                arguments.index("scripts.e2e.regional.probes.notify008_probe") + 1
            ]
            assert (
                arguments[arguments.index("--expected-pod-uid") + 1]
                == self.pod["metadata"]["uid"]
            ), "probe must bind actual Pod UID"
            if self.fail_probe == command:
                if self.probe_exception is not None:
                    raise self.probe_exception
                raise RuntimeError("fake probe refused")
            if command == "arm":
                self.armed = True
                if self.lose_arm:
                    self.lose_arm = False
                    raise TimeoutError("fake ARM acknowledgement lost")
            if command == "stop":
                self.stopped = True
                for item in self.pod["status"]["containerStatuses"]:
                    item["state"] = {"terminated": {"exitCode": 0}}
                    item["ready"] = False
            if command == "prepare":
                assert self.armed, "schema preparation before ARM is forbidden"
                return json.dumps(
                    {
                        "postgres_major": 16,
                        "database_is_unix_socket": True,
                        "production_credentials_loaded": False,
                    }
                )
            if command == "run":
                assert self.armed, "runtime work before ARM is forbidden"
                result = report_record()
                result.update(pod_uid=self.pod["metadata"]["uid"], verdict="PASS")
                if self.mutate_after_run is not None:
                    self.mutate_after_run(self)
                return json.dumps(result)
            return json.dumps(
                {
                    "pod_uid": self.pod["metadata"]["uid"],
                    "run_id": RUN_ID,
                    "armed": self.armed,
                    "stop_requested": self.stopped,
                }
            )
        if verb == "delete":
            resource = (
                "priorityclass"
                if arguments[2].startswith(
                    "/apis/scheduling.k8s.io/v1/priorityclasses/"
                )
                else "namespace"
            )
            assert (
                body["preconditions"]["uid"]
                == self.objects[resource]["metadata"]["uid"]
            ), "resource cleanup must use a UID precondition"
            if resource == "priorityclass":
                assert (
                    body["preconditions"]["resourceVersion"]
                    == self.objects[resource]["metadata"]["resourceVersion"]
                ), "PriorityClass cleanup must bind the current resource version"
                self.objects.pop(resource)
            else:
                namespace_uid = self.objects["namespace"]["metadata"]["uid"]
                priority = self.objects.get("priorityclass")
                self.objects.clear()
                if priority is not None and (
                    not self.gc_priorityclass
                    or priority["metadata"].get("ownerReferences")
                    != [
                        {
                            "apiVersion": "v1",
                            "kind": "Namespace",
                            "name": RUN_ID,
                            "uid": namespace_uid,
                            "controller": False,
                            "blockOwnerDeletion": False,
                        }
                    ]
                ):
                    self.objects["priorityclass"] = priority
                self.pod = None
            if self.lose_delete:
                raise TimeoutError("fake deletion acknowledgement lost")
            return ""
        raise AssertionError(f"unexpected fake CPU operation {verb}")


def setup_run(tmp_path, monkeypatch):
    config = tmp_path / "cpu-kubeconfig"
    config.write_text("local fake CPU connection")
    settings = binding.Settings(
        config,
        "cpu-context",
        "gpu-fault-system",
        "cluster-local",
        "us-west-2",
        POSTGRES_IMAGE,
    )
    case_dir = tmp_path / "cases" / CASE_ID
    case_dir.mkdir(parents=True)
    value = target()
    (case_dir / "plan.json").write_text(
        json.dumps(
            {
                "details": {
                    "target": asdict(value),
                    "bundle_sha256": digest(source_bundle()),
                }
            }
        )
    )
    previous = tmp_path / "cases/GF-REGIONAL-NOTIFY-007/GF-REGIONAL-NOTIFY-007.json"
    previous.parent.mkdir()
    previous.write_text(
        json.dumps(
            {
                "case_id": "GF-REGIONAL-NOTIFY-007",
                "verdict": "PASS",
                "release_id": value.release_id,
                "cluster_id": value.cluster_id,
                "execution_scope": "formal",
                "formal_sequence_satisfied": True,
            }
        )
    )
    monkeypatch.setattr(
        binding,
        "case_metadata",
        lambda case: SimpleNamespace(risk="live-non-destructive", automation="manual"),
    )
    monkeypatch.setattr(
        binding,
        "predecessor_path",
        lambda root, case, explicit: ("GF-REGIONAL-NOTIFY-007", previous),
    )
    api = FakeCPU()
    monkeypatch.setattr(runner, "CpuAPI", lambda *_args: api)
    monkeypatch.setattr(fixture, "time", Clock())
    return settings, api, case_dir, datetime.now(UTC) + timedelta(hours=1)
