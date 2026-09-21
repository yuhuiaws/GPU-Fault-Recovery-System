"""Controlled CPU API, clock, and child-process state for lifecycle tests."""

from __future__ import annotations

import copy
import json
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_parallel as parallel
from scripts.e2e.regional import destr008_watchdog_control as control_api
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional import seeded_command_fixture as deletion
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from tests.regional._destr008_watchdog_resources import Harness as SourceHarness
from tests.regional._destr008_watchdog_resources import harness as source_harness
from tests.regional._destr008_watchdog_resources import pod_defaults

HOST = "f" * 64


@dataclass
class Clock:
    epoch: int = 1789257601
    elapsed: float = 0

    def now(self) -> float:
        return self.epoch + self.elapsed

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.now(), timezone.utc).isoformat()


def raw_list_request(args: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    """Kind and labelSelector of a ``kubectl get --raw <typed list>`` read.

    The population survey and the Pod discovery read the server's typed list
    (kubectl v1.35 prints a client-side v1/List with no resourceVersion for
    ``get -o json``); the fake serves the same objects, filtered by
    labelSelector exactly like the API server does.
    """

    resource_path, _, query = args[2].partition("?")
    assert resource_path.startswith(
        ("/apis/apps/v1/namespaces/", "/api/v1/namespaces/")
    ) and resource_path.endswith(("/replicasets", "/pods")), args
    selector: dict[str, str] = {}
    for key, value in urllib.parse.parse_qsl(query):
        assert key == "labelSelector", args
        label, _, wanted = value.partition("=")
        selector[label] = wanted
    kind = "replicaset" if resource_path.endswith("/replicasets") else "pod"
    return kind, selector


@dataclass
class CpuApi:
    regional: RegionalLiveFixture
    source: Path
    objects: dict[tuple[str, str], dict[str, Any]]
    directory: Path
    clock: Clock = field(default_factory=Clock)
    calls: list[tuple[str, tuple[str, ...], Any]] = field(default_factory=list)
    counter: int = 100
    auto_arm: bool = True
    auto_quiet: bool = True
    heartbeat_churn: bool = False
    keep_pods: bool = False
    omit_pods: bool = False
    gate_lost_ack: bool = False
    gate_no_apply: bool = False
    failed_create: set[str] = field(default_factory=set)
    lost_create: set[str] = field(default_factory=set)
    malformed_create: set[str] = field(default_factory=set)
    failed_delete: set[str] = field(default_factory=set)
    lost_delete: set[str] = field(default_factory=set)
    create_changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    pod_changes: dict[str, Any] = field(default_factory=dict)
    read_changes: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    list_override: Any = None
    raw_override: str | None = None
    frozen_status: bool = False
    completion_lag: int = 0
    capability_change: dict[str, Any] = field(default_factory=dict)
    full_reads: dict[tuple[str, str], str] = field(default_factory=dict)

    def next(self) -> str:
        self.counter += 1
        return str(self.counter)

    def bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.workers()
        monkeypatch.setattr(self.regional, "kubectl", self.kube)
        monkeypatch.setattr(
            self.regional,
            "evidence_identity",
            lambda: {"cluster_id": "cluster-a", "release_id": "release-a"},
        )
        monkeypatch.setattr(locking, "host_identity", lambda: HOST)
        monkeypatch.setattr(resources, "CODE_SOURCE", self.source)
        monkeypatch.setattr(parallel, "workers", 1)  # deterministic fakes
        monkeypatch.setattr(
            control_api, "source_sha256", lambda: wire.source_sha256(self.source)
        )
        monkeypatch.setattr(
            deletion,
            "time",
            SimpleNamespace(monotonic=self.clock.monotonic, sleep=self.clock.sleep),
        )

    def workers(self) -> None:
        name = resources.DEPLOYMENT + "-rs"
        if ("replicaset", name) in self.objects:
            return
        deployment = self.objects["deployment", resources.DEPLOYMENT]
        namespace = self.regional.settings.namespace
        self.objects["replicaset", name] = {
            "apiVersion": "apps/v1",
            "kind": "ReplicaSet",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "uid": "worker-rs-uid",
                "resourceVersion": "1",
                "ownerReferences": [
                    {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": resources.DEPLOYMENT,
                        "uid": deployment["metadata"]["uid"],
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
            },
        }
        for index in range(deployment["spec"]["replicas"]):
            pod_name = "cpu-worker-" + str(index)
            spec = copy.deepcopy(deployment["spec"]["template"]["spec"])
            image = spec["containers"][0]["image"]
            self.objects["pod", pod_name] = {
                "apiVersion": "v1",
                "kind": "Pod",
                "spec": spec,
                "metadata": {
                    "name": pod_name,
                    "namespace": namespace,
                    "uid": pod_name + "-uid",
                    "resourceVersion": "1",
                    "labels": {"app": resources.DEPLOYMENT},
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "ReplicaSet",
                            "name": name,
                            "uid": "worker-rs-uid",
                            "controller": True,
                            "blockOwnerDeletion": True,
                        }
                    ],
                },
                "status": {
                    "phase": "Running",
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                        {"type": "ContainersReady", "status": "True"},
                    ],
                    "containerStatuses": [
                        {
                            "name": "control-worker",
                            # kubelet's shape for a digest-pinned pull: the
                            # bare image config id, identity in imageID
                            "image": "sha256:" + "e" * 64,
                            "imageID": image,
                            "containerID": "containerd://worker-" + str(index),
                            "ready": True,
                            "restartCount": 0,
                            "state": {"running": {"startedAt": self.clock.timestamp()}},
                        }
                    ],
                },
            }

    def watchdog(
        self, plan: wire.Plan, runtime: resources.CpuRuntime
    ) -> lifecycle.CancellationWatchdog:
        return lifecycle.CancellationWatchdog(
            self.regional,
            plan,
            runtime,
            self.directory,
            clock=self.clock.now,
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep,
        )

    def journal(self) -> dict[str, Any]:
        files = list(self.directory.glob("cancellation-*.json"))
        assert len(files) == 1, "there must be one durable run journal"
        return locking.read_private_document(files[0])

    def acknowledge(self, manifest: dict[str, Any]) -> dict[str, Any]:
        actual = copy.deepcopy(manifest)
        kind, name = actual["kind"].lower(), actual["metadata"]["name"]
        uid = "uid-" + self.next()
        actual["metadata"].update(uid=uid, resourceVersion=self.next(), generation=1)
        if kind == "serviceaccount":
            actual.update(secrets=[], imagePullSecrets=[])
        elif kind == "rolebinding":
            actual["subjects"][0]["apiGroup"] = ""
        elif kind == "configmap":
            actual["binaryData"] = {}
        elif kind == "job":
            spec = actual["spec"]
            spec.update(
                selector={"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
                parallelism=1,
                completions=1,
                completionMode="NonIndexed",
                manualSelector=False,
                suspend=False,
                podReplacementPolicy="TerminatingOrFailed",
                managedBy="kubernetes.io/job-controller",
            )
            spec["template"]["metadata"]["labels"].update(
                {
                    "controller-uid": uid,
                    "batch.kubernetes.io/controller-uid": uid,
                    "job-name": name,
                    "batch.kubernetes.io/job-name": name,
                }
            )
            spec["template"]["metadata"]["creationTimestamp"] = None
            pod_defaults(spec["template"]["spec"])
            actual["status"] = {"active": 1, "ready": 0}
        for path, value in self.create_changes.get(kind, {}).items():
            set_path(actual, path, value)
        self.objects[kind, name] = actual
        if kind == "job" and not self.omit_pods:
            self.make_pod(actual)
        return copy.deepcopy(actual)

    def make_pod(self, job: dict[str, Any]) -> dict[str, Any]:
        metadata = job["metadata"]
        template = copy.deepcopy(job["spec"]["template"])
        name = metadata["name"] + "-abcde"
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "spec": template["spec"],
            "metadata": {
                **template["metadata"],
                "name": name,
                "namespace": metadata["namespace"],
                "uid": "pod-" + self.next(),
                "resourceVersion": self.next(),
                "ownerReferences": [
                    {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "name": metadata["name"],
                        "uid": metadata["uid"],
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
            },
            "status": {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": "SchedulingGated",
                    }
                ],
            },
        }
        for path, value in self.pod_changes.items():
            set_path(pod, path, value)
        self.objects["pod", name] = pod
        return pod

    def run_pod(self, pod: dict[str, Any]) -> None:
        pod["spec"]["nodeName"] = "cpu-node"
        image = pod["spec"]["containers"][0]["image"]
        container_id = "containerd://" + "d" * 64
        pod["status"] = {
            "phase": "Running",
            "startTime": self.clock.timestamp(),
            "conditions": [
                {"type": key, "status": "True"}
                for key in ("Initialized", "PodScheduled", "Ready", "ContainersReady")
            ],
            "containerStatuses": [
                {
                    "name": "cancellation-watchdog",
                    "image": "sha256:" + "e" * 64,
                    "imageID": image,
                    "containerID": container_id,
                    "ready": True,
                    "started": True,
                    "restartCount": 0,
                    "lastState": {},
                    "state": {"running": {"startedAt": self.clock.timestamp()}},
                }
            ],
        }
        pod["metadata"]["resourceVersion"] = self.next()
        job = self.objects["job", pod["metadata"]["ownerReferences"][0]["name"]]
        job["status"] = {"active": 1, "ready": 1}
        job["metadata"]["resourceVersion"] = self.next()

    def finish(self, pod: dict[str, Any]) -> None:
        container = pod["status"]["containerStatuses"][0]
        started = container["state"]["running"]["startedAt"]
        container.update(
            ready=False,
            started=False,
            state={
                "terminated": {
                    "startedAt": started,
                    "finishedAt": self.clock.timestamp(),
                    "exitCode": 0,
                    "reason": "Completed",
                    "containerID": container["containerID"],
                }
            },
        )
        pod["status"]["phase"] = "Succeeded"
        for condition in pod["status"]["conditions"]:
            if condition["type"] in {"Ready", "ContainersReady"}:
                condition["status"] = "False"
        pod["metadata"]["resourceVersion"] = self.next()
        job = self.objects["job", pod["metadata"]["ownerReferences"][0]["name"]]
        job["status"] = {
            "succeeded": 1,
            "conditions": [
                {"type": "SuccessCriteriaMet", "status": "True"},
                {"type": "Complete", "status": "True"},
            ],
        }
        job["metadata"]["resourceVersion"] = self.next()

    def progress(self) -> None:
        if self.frozen_status:
            return
        for (kind, _), pod in list(self.objects.items()):
            if kind != "pod" or pod.get("status", {}).get("phase") != "Running":
                continue
            if (
                "job",
                pod["metadata"]["ownerReferences"][0]["name"],
            ) not in self.objects:
                continue
            command = pod["spec"]["containers"][0]["command"]
            name = command[command.index("--configmap") + 1]
            control_map = self.objects.get(("configmap", name))
            if control_map is None:
                continue
            data = control_map["data"]
            plan = wire.decode(wire.Plan, data["plan.json"])
            control = wire.decode(wire.Control, data["control.json"])
            previous = (
                None
                if data["status.json"] == "null"
                else wire.decode(wire.Receipt, data["status.json"])
            )
            cleanup = "--cleanup-only" in command
            if (
                cleanup
                or control.close_request is not None
                or self.clock.now() >= plan.deadline_at
            ):
                control = wire.revoke(
                    control,
                    now=int(self.clock.now()),
                    reason="FAILURE"
                    if cleanup
                    else "PARENT_CLOSE"
                    if control.close_request is not None
                    else "DEADLINE",
                )
                attempt = None
                if cleanup:
                    seconds = int(command[command.index("--cleanup-seconds") + 1])
                    attempt_id = command[command.index("--cleanup-attempt-id") + 1]
                    attempt = (
                        previous.cleanup
                        if previous is not None
                        and previous.cleanup is not None
                        and previous.cleanup.attempt_id == attempt_id
                        else wire.CleanupAttempt(
                            attempt_id=attempt_id,
                            started_at=int(self.clock.now()),
                            deadline_at=int(self.clock.now()) + seconds,
                        )
                    )
                status = wire.receipt(
                    plan,
                    control,
                    previous,
                    uid=control_map["metadata"]["uid"],
                    now=int(self.clock.now()),
                    state="REVOKED",
                    cleanup=attempt,
                )
                if self.auto_quiet and control.producer.state != "SUBMITTING":
                    self.clock.sleep(wire.QUIET_SECONDS)
                    proof: dict[str, Any] = {}
                    if control.producer.ack is not None:
                        ack = control.producer.ack
                        proof = {
                            "root": wire.Root(
                                incident_id=ack.incident_id,
                                workflow_request_id=ack.workflow_request_id,
                            ),
                            "workflow_ids": [ack.workflow_request_id],
                        }
                    status = wire.receipt(
                        plan,
                        control,
                        status,
                        uid=control_map["metadata"]["uid"],
                        now=int(self.clock.now()),
                        state="QUIESCENT",
                        source_complete=True,
                        commands_active=0,
                        workflows_active=0,
                        pending_creation=False,
                        inventory_sha256="b" * 64,
                        quiet_since=int(self.clock.now()) - wire.QUIET_SECONDS,
                        monitoring=False,
                        **proof,
                    )
                    self.finish(pod)
                data["control.json"] = wire.encode(control)
                data["status.json"] = wire.encode(status)
                control_map["metadata"]["resourceVersion"] = self.next()
            elif self.auto_arm and previous is None:
                data["status.json"] = wire.encode(
                    wire.receipt(
                        plan,
                        control,
                        None,
                        uid=control_map["metadata"]["uid"],
                        now=int(self.clock.now()),
                        state="ARMED",
                    )
                )
                control_map["metadata"]["resourceVersion"] = self.next()
            elif self.heartbeat_churn and previous is not None:
                # The independent daemon rewrites status.json every POLL_SECONDS,
                # advancing the receipt sequence and bumping the ConfigMap
                # resourceVersion. On the live control plane such a heartbeat
                # lands between the parent's read and its CAS patch (~2.4 s round
                # trip), so a parent patch that pins the whole resourceVersion or
                # the status.json key can never converge.
                data["status.json"] = wire.encode(
                    wire.receipt(
                        plan,
                        control,
                        previous,
                        uid=control_map["metadata"]["uid"],
                        now=int(self.clock.now()),
                        state=previous.state,
                    )
                )
                control_map["metadata"]["resourceVersion"] = self.next()

    def kube(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu", (
            "the watchdog lifecycle must never address GPU Kubernetes"
        )
        self.progress()
        verb = args[0]
        stdin = kwargs.get("input_text")
        payload: Any = (
            stdin if verb == "exec" else None if stdin is None else json.loads(stdin)
        )
        self.calls.append((verb, args, payload))
        if verb == "get":
            if self.raw_override is not None:
                return self.raw_override
            kind = args[1]
            raw_list = kind == "--raw"
            selector: dict[str, str] = {}
            if raw_list:
                kind, selector = raw_list_request(args)
                args = ("get", kind, "-o", "json")
            if kind in {"pod", "replicaset"} and args[2:4] == ("-o", "json"):
                if kind == "pod" and self.list_override is not None:
                    return json.dumps(self.list_override)
                pods = [
                    copy.deepcopy(value)
                    for (resource_kind, _), value in self.objects.items()
                    if resource_kind == kind
                    and all(
                        value["metadata"].get("labels", {}).get(label) == wanted
                        for label, wanted in selector.items()
                    )
                ]
                for value in pods:
                    self.full_reads[kind, value["metadata"]["name"]] = value[
                        "metadata"
                    ]["resourceVersion"]
                    # A typed list from the API server carries bare items: only
                    # kubectl stamps apiVersion/kind on each one. The live
                    # survey read the raw list and refused every ReplicaSet as
                    # "owner is unproven" (DESTR-008 attempt 9) while this fake
                    # kept serving kubectl-shaped items.
                    if raw_list:
                        value.pop("apiVersion", None)
                        value.pop("kind", None)
                return json.dumps(
                    {
                        "apiVersion": "v1" if kind == "pod" else "apps/v1",
                        "kind": "PodList" if kind == "pod" else "ReplicaSetList",
                        "metadata": {"resourceVersion": str(self.counter)},
                        "items": pods,
                    }
                )
            name = args[2]
            observed = self.objects.get((kind, name))
            if observed is None:
                assert "--ignore-not-found" in args, (
                    "missing required resources must not be silently ignored"
                )
                return ""
            for path, change in self.read_changes.get((kind, name), {}).items():
                set_path(observed, path, change)
            if args[-1] == "jsonpath={.metadata}":
                return json.dumps(observed["metadata"])
            self.full_reads[kind, name] = observed["metadata"]["resourceVersion"]
            if (
                kind == "job"
                and observed.get("status", {}).get("succeeded") == 1
                and self.completion_lag
            ):
                self.completion_lag -= 1
                view = copy.deepcopy(observed)
                owned = [
                    value
                    for (candidate, _), value in self.objects.items()
                    if candidate == "pod"
                    and value["metadata"]["ownerReferences"][0]["uid"]
                    == observed["metadata"]["uid"]
                ]
                view["status"] = {
                    "uncountedTerminatedPods": {
                        "succeeded": [owned[0]["metadata"]["uid"]]
                    }
                }
                return json.dumps(view)
            return json.dumps(observed)
        if verb == "exec":
            assert args[1] == "-i" and args[3:6] == ("-c", "control-worker", "--"), (
                "feature inspection must target the bound worker container explicitly"
            )
            assert args[6] == "/opt/gpu-fault/control-plane/bin/python", (
                "the feature probe must use the installed CPU interpreter"
            )
            assert args[7:11] == ("-I", "-B", "-c", resources.HISTORY_LOADER), (
                "the feature source must run isolated and hash-checked"
            )
            result = {**resources.history_api_identity(), "probe_sha256": args[-1]}
            for path, change in self.capability_change.items():
                set_path(result, path, change)
            return json.dumps(result)
        assert kwargs.get("timeout") is not None, (
            "every mutation needs a bounded timeout"
        )
        if verb == "create":
            expected = payload
            kind, name = expected["kind"].lower(), expected["metadata"]["name"]
            saved = self.journal()
            record = (
                saved["jobs"][-1]["resource"]
                if kind == "job"
                else saved["support"][kind + "/" + name]
            )
            assert record["uid"] is None, (
                "intent must precede creation and cannot contain a guessed UID"
            )
            assert saved["plan"]["probe_sha256"], (
                "source/plan pins must predate remote mutations"
            )
            if kind in self.failed_create:
                raise TimeoutError("private API diagnostic fixture")
            result = self.acknowledge(expected)
            if kind in self.lost_create:
                raise TimeoutError("creation ACK lost after remote commit")
            if kind in self.malformed_create:
                return "invalid-create-ack"
            return json.dumps(result)
        if verb == "patch":
            kind, name = args[1:3]
            value = self.objects[kind, name]
            if kind == "pod":
                saved = self.journal()
                job = next(
                    item
                    for item in saved["jobs"]
                    if item["pod"] is not None and item["pod"]["name"] == name
                )
                assert job["pod"]["uid"] == value["metadata"]["uid"], (
                    "the Pod UID must be durable before gate release"
                )
                assert job["pod"]["shape_sha256"], (
                    "full admitted shape proof must precede gate release"
                )
                assert job["release_requested"], (
                    "gate CAS intent must already be durable"
                )
                assert any(
                    item["op"] == "test" and item["path"] == "/spec" for item in payload
                ), "gate release must test the full verified Pod spec"
            if kind != "pod" or not self.gate_no_apply:
                apply_patch(value, payload)
                value["metadata"]["resourceVersion"] = self.next()
                if kind == "pod":
                    self.run_pod(value)
            if kind == "pod" and self.gate_lost_ack:
                raise TimeoutError("known Pod gate patch ACK lost")
            return json.dumps(value)
        if verb == "delete":
            path = args[2]
            assert args[1] == "--raw", "deletion must use explicit UID/RV preconditions"
            assert f"/namespaces/{self.regional.settings.namespace}/" in path, (
                "cleanup cannot escape the bound CPU namespace"
            )
            plural, name = path.split("/")[-2:]
            kind = {
                "jobs": "job",
                "pods": "pod",
                "configmaps": "configmap",
                "roles": "role",
                "rolebindings": "rolebinding",
                "serviceaccounts": "serviceaccount",
            }[plural]
            value = self.objects[kind, name]
            assert payload["preconditions"] == {
                "uid": value["metadata"]["uid"],
                "resourceVersion": value["metadata"]["resourceVersion"],
            }, "DELETE must bind the current exact UID and version"
            assert (
                payload["preconditions"]["resourceVersion"]
                == self.full_reads[kind, name]
            ), "DELETE must carry the version from the full-spec read"
            assert payload["propagationPolicy"] == "Foreground", (
                "Job children must be stopped"
            )
            if kind in self.failed_delete:
                raise TimeoutError("delete refused fixture")
            del self.objects[kind, name]
            if kind == "job" and not self.keep_pods:
                for key, candidate in list(self.objects.items()):
                    if key[0] == "pod" and any(
                        owner.get("uid") == value["metadata"]["uid"]
                        for owner in candidate["metadata"].get("ownerReferences", [])
                    ):
                        del self.objects[key]
            if kind in self.lost_delete:
                raise TimeoutError("delete ACK lost after remote commit")
            return '{"kind":"Status","status":"Success"}'
        raise AssertionError(f"unexpected controlled CPU API command: {verb}")

    def dump(self, path: Path) -> None:
        data = {
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.regional.settings).items()
            },
            "source": str(self.source),
            "directory": str(self.directory),
            "clock": asdict(self.clock),
            "counter": self.counter,
            "objects": [
                {"kind": kind, "name": name, "value": value}
                for (kind, name), value in self.objects.items()
            ],
        }
        path.write_text(json.dumps(data))


def apply_patch(document: dict[str, Any], patch: list[dict[str, Any]]) -> None:
    for operation in patch:
        parts = operation["path"].lstrip("/").split("/")
        target: Any = document
        for raw in parts[:-1]:
            target = target[int(raw)] if isinstance(target, list) else target[raw]
        key: Any = int(parts[-1]) if isinstance(target, list) else parts[-1]
        if operation["op"] == "test":
            if wire.encode(target[key]) != wire.encode(operation["value"]):
                raise RuntimeError("JSON Patch precondition failed")
        elif operation["op"] == "replace":
            target[key] = operation["value"]
        elif operation["op"] == "remove":
            del target[key]
        else:
            raise AssertionError("unexpected patch operation")


def build_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    source: SourceHarness = source_harness(tmp_path, monkeypatch)
    runtime = resources.read_runtime(source.regional)
    api = CpuApi(
        source.regional, source.source, source.objects, tmp_path / "controller"
    )
    api.bind(monkeypatch)
    return api, source.plan(), runtime


def load_api(path: Path, monkeypatch: pytest.MonkeyPatch) -> CpuApi:
    value = json.loads(path.read_text())
    settings = value["settings"]
    for key in ("cpu_kubeconfig", "gpu_kubeconfig"):
        settings[key] = Path(settings[key])
    api = CpuApi(
        RegionalLiveFixture(RegionalLiveSettings(**settings)),
        Path(value["source"]),
        {(item["kind"], item["name"]): item["value"] for item in value["objects"]},
        Path(value["directory"]),
        Clock(**value["clock"]),
        counter=value["counter"],
    )
    api.bind(monkeypatch)
    return api


def set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    target: Any = value
    parts = path.split("/")
    for raw in parts[:-1]:
        target = target[int(raw)] if isinstance(target, list) else target[raw]
    final: Any = int(parts[-1]) if isinstance(target, list) else parts[-1]
    target[final] = copy.deepcopy(replacement)
