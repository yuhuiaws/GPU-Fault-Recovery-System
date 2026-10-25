"""GF-GENERAL-LIVE-027's progress job, started and fed through its own objects.

Live 2026-10-24 (plain-EKS lane, release bf1bb0037847): all nine executable 027
variants failed before any fault with "managed workload did not become
healthy". The PyTorchJob's only Pod sat Running and NotReady for the whole
900-s start wait while its trainer heartbeated every 2 s. The progress-reporter
sidecar ran the Node Agent's *host* install path,
``/opt/gpu-fault/current/venv/bin/gpu-fault-collector``, inside the public
PyTorch training image, which ships no GPU fault product: the container never
started, ``restartPolicy: Never`` left it terminated, and a Pod is Ready only
when every container is. The sidecar also carried none of the control-plane
connection ``gpu-fault-collector`` refuses to start without, and the runner's
progress write piped its JSON through ``kubectl exec`` without ``-i``, so the
trainer would only ever have written empty files.

The verdict tests swap ``ManagedProgressJob`` for a fake, so no test ran the
Pod the runner renders. Here the job, its image prewarm, the managed workload
fixture, ``RegionalLiveFixture`` and ``VariantContext`` are the runner's own;
only ``RegionalLiveFixture.run`` -- kubectl -- is a fake (plus a virtual clock
for the fixture's start wait). Its kubelet does what Kubernetes does: a
container starts only if its image ships the program it executes, a
``gpu-fault-collector`` container keeps running only if the product CLI's own
start-up (``gpu_fault.collectors_cli.main``) accepts the rendered arguments and
environment, and the Pod is Ready only when every container runs and none waits
on a readiness probe. ``kubectl exec`` runs the command in the container's
volumes and forwards stdin only with ``-i``; the reporter the kubelet started
then reads what the runner wrote.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from unittest import mock

import pytest

from gpu_fault import collectors_cli
from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.collectors.training_progress import TrainingProgressCollector
from gpu_fault.training_health import TrainingProgressHeartbeat
from scripts import component_wheels
from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional import managed_workload_fixture
from scripts.e2e.regional.general_live_fixture import (
    Evidence,
    GeneralLiveSettings,
    VariantContext,
)
from scripts.e2e.regional.managed_workload_fixture import TRAINING_IMAGE
from scripts.e2e.regional.regional_commands import (
    RegionalCommandFailed,
    RegionalFixtureError,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
    component_python,
)
from tests.admin.test_admin_site import site_file as write_site_file

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
NAMESPACE = "gpu-fault-system"
CLUSTER = "gpu-a"  # the site file's own cluster
CONTEXT = "gpu-a"
EXECUTOR_DEPLOYMENT = "gpu-fault-cluster-executor"
EXECUTOR_IMAGE = (
    "111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-fault-cluster-executor"
    "@sha256:" + "e" * 64
)
#: The GPU cluster's connection Secret, as the product's own Deployments read it
#: (``deploy/dataplane/*.yaml``). Nothing else is mountable in the namespace.
CONNECTION_SECRET = "gpu-fault-regional-connection"
CONNECTION = {
    "control-plane-url": "https://gpu-fault-api.example.internal:8443",
    "cluster-id": CLUSTER,
    "cluster-token": "t" * 64,
    "ca.crt": "test control-plane CA\n",
}
SHELLS = frozenset({"/bin/sh", "/bin/bash"})
SPECS = {spec.variant: spec for spec in progress.VARIANTS}


def image_programs(image: str) -> frozenset[str]:
    """The programs an image ships, from what builds it.

    The training image is the public PyTorch Deep Learning Container: shells
    and its Python stack, no GPU fault product. The GPU-side runtime image is
    ``deploy/image/Dockerfile.component`` built for the executor: one venv
    reachable as the directory ``component_python("gpu")`` names, holding the
    executor wheel's console scripts.
    """

    if image == TRAINING_IMAGE:
        return SHELLS
    if image == EXECUTOR_IMAGE:
        runtime = PurePosixPath(component_python("gpu")).parent
        scripts = component_wheels.COMPONENTS["executor"].scripts
        return SHELLS | {str(runtime / "python")} | {str(runtime / s) for s in scripts}
    return frozenset()


def is_launcher(command: list[str]) -> bool:
    """``sh -c '<script running "$@">' <name> <program> <args...>``."""

    return (
        len(command) > 4
        and command[0] in SHELLS
        and command[1] == "-c"
        and '"$@"' in command[2]
    )


def program_argv(command: list[str]) -> list[str]:
    """The argv of the program a container runs, through a launcher shell."""

    return command[4:] if is_launcher(command) else command


def executed(command: list[str]) -> list[str]:
    """Every program the container's start executes from its image."""

    return [command[0], command[4]] if is_launcher(command) else [command[0]]


def container_command(container: dict[str, Any]) -> list[str]:
    return [*container.get("command", []), *container.get("args", [])]


class VirtualClock:
    """``time`` for the fixture's waits: a sleep advances, never blocks."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class CapturingSink:
    """The reporter's ``EventSink`` for one cycle: what it would have posted."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        return {"accepted": True}


class ProgressCluster:
    """kubectl, a training operator and a kubelet for one GPU node, in memory."""

    def __init__(self, root: Path, *, executor_image: str = EXECUTOR_IMAGE) -> None:
        self.root = root
        self.objects: dict[tuple[str, str], dict[str, Any]] = {
            ("deployment", EXECUTOR_DEPLOYMENT): {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": EXECUTOR_DEPLOYMENT,
                    "namespace": NAMESPACE,
                    "uid": str(uuid.uuid4()),
                    "resourceVersion": "7",
                },
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "executor",
                                    "image": executor_image,
                                    "command": ["/bin/sh", "-c"],
                                    "args": ["exec gpu-fault-cluster-executor"],
                                }
                            ]
                        }
                    }
                },
            }
        }
        #: The product collectors the kubelet started, by (Pod, container).
        self.reporters: dict[tuple[str, str], TrainingProgressCollector] = {}
        self.environments: dict[tuple[str, str], dict[str, str]] = {}
        self.execs: list[dict[str, Any]] = []

    # -- kubectl -------------------------------------------------------------
    def run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: float = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "kubectl" and "-n" in command, command
        assert command[command.index("-n") + 1] == NAMESPACE, command
        arguments = list(command[command.index("-n") + 2 :])
        if "--context" not in command:
            assert arguments[:3] == [
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
            ], f"the only CPU read is the release state: {arguments}"
            state = {"phase": "complete", "transaction_committed": True}
            stdout = json.dumps({"data": {"state.json": json.dumps(state)}})
            return subprocess.CompletedProcess(command, 0, stdout, "")
        assert command[command.index("--context") + 1] == CONTEXT, command
        verb = arguments[0]
        if arguments == ["api-resources", "-o", "name"]:
            stdout = "pods\njobs.batch\npytorchjobs.kubeflow.org\nservices\n"
        elif verb == "get":
            stdout = self.get(arguments[1:])
        elif verb == "create":
            stdout = self.create(arguments[1:], input_text)
        elif verb == "logs":
            stdout = self.logs(arguments[1:])
        elif verb == "exec":
            stdout = self.exec(arguments[1:], input_text)
        else:
            raise AssertionError(f"unexpected kubectl call: {arguments}")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def get(self, arguments: list[str]) -> str:
        assert arguments[-2:] == ["-o", "json"], arguments
        arguments = arguments[:-2]
        ignore = "--ignore-not-found" in arguments
        arguments = [item for item in arguments if item != "--ignore-not-found"]
        if "-l" in arguments:
            kinds, _flag, selector = arguments
            wanted = dict(item.split("=", 1) for item in selector.split(","))
            items = [
                value
                for (kind, _name), value in sorted(self.objects.items())
                if kind in kinds.split(",")
                and all(
                    (value["metadata"].get("labels") or {}).get(key) == label
                    for key, label in wanted.items()
                )
            ]
            return json.dumps({"apiVersion": "v1", "kind": "List", "items": items})
        if arguments == ["node"]:
            images = [{"names": [TRAINING_IMAGE]}]  # the trainer image is cached
            node = {"metadata": {"name": NODE}, "status": {"images": images}}
            return json.dumps({"items": [node]})
        kind, name = arguments
        value = self.objects.get((kind, name))
        if value is None:
            if ignore:
                return ""
            raise RegionalCommandFailed(1, f'{kind} "{name}" not found')
        return json.dumps(value)

    def create(self, arguments: list[str], body: str | None) -> str:
        assert arguments[:2] == ["-f", "-"] and body is not None, arguments
        document = json.loads(body)
        kind, name = document["kind"].lower(), document["metadata"]["name"]
        assert kind == "pytorchjob", f"only the managed PyTorchJob is created: {kind}"
        assert (kind, name) not in self.objects, f"{kind}/{name} exists"
        document["metadata"].update(uid=str(uuid.uuid4()), resourceVersion="1")
        self.objects[(kind, name)] = document
        for role, replica in document["spec"]["pytorchReplicaSpecs"].items():
            for index in range(int(replica.get("replicas", 1))):
                self.schedule(document, role, index, replica["template"])
        return json.dumps(document)

    def logs(self, arguments: list[str]) -> str:
        pod = self.objects[("pod", arguments[0])]
        name = (
            arguments[arguments.index("-c") + 1]
            if "-c" in arguments
            else (pod["metadata"].get("annotations") or {}).get(
                "kubectl.kubernetes.io/default-container"
            )
            or pod["spec"]["containers"][0]["name"]
        )
        container = self.container(pod, name)
        if container["image"] != TRAINING_IMAGE or not self.running(pod, name):
            return ""
        return "".join(
            f"HEARTBEAT rank=0 step={step} node={pod['metadata']['name']} "
            f"world_size=8 all_reduce=36.0 elapsed={2.0 * step:.1f}\n"
            for step in range(1, 6)
        )

    def exec(self, arguments: list[str], input_text: str | None) -> str:
        separator = arguments.index("--")
        options, command = arguments[:separator], arguments[separator + 1 :]
        name = options[options.index("-c") + 1]
        pods = [
            item
            for index, item in enumerate(options)
            if not item.startswith("-") and options[index - 1] != "-c"
        ]
        assert len(pods) == 1, options
        pod = self.objects[("pod", pods[0])]
        if not self.running(pod, name):
            raise RegionalCommandFailed(1, f"container {name} is not running")
        stdin = "-i" in options or "--stdin" in options
        self.execs.append(
            {"container": name, "stdin": stdin, "command": command, "input": input_text}
        )
        local = [self.remap(pod, self.container(pod, name), item) for item in command]
        assert local[0] in SHELLS, f"the runner execs a shell: {command}"
        completed = subprocess.run(
            local,
            # kubectl forwards its stdin only with -i; otherwise the command
            # in the container reads end-of-file at once.
            input=(input_text or "") if stdin else "",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            raise RegionalCommandFailed(completed.returncode, completed.stderr)
        return completed.stdout

    # -- training operator and kubelet ---------------------------------------
    def schedule(
        self, job: dict[str, Any], role: str, index: int, template: dict[str, Any]
    ) -> None:
        metadata = template.get("metadata") or {}
        name = f"{job['metadata']['name']}-{role.lower()}-{index}"
        pod: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": NAMESPACE,
                "uid": str(uuid.uuid4()),
                "resourceVersion": "1",
                "labels": {
                    **(metadata.get("labels") or {}),
                    "training.kubeflow.org/job-name": job["metadata"]["name"],
                    "training.kubeflow.org/replica-type": role.lower(),
                    "training.kubeflow.org/replica-index": str(index),
                },
                "annotations": dict(metadata.get("annotations") or {}),
                "ownerReferences": [
                    {
                        "apiVersion": job["apiVersion"],
                        "kind": job["kind"],
                        "name": job["metadata"]["name"],
                        "uid": job["metadata"]["uid"],
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
            },
            "spec": copy.deepcopy(template["spec"]),
        }
        statuses = [self.start(pod, item) for item in pod["spec"]["containers"]]
        ready = all(item["ready"] for item in statuses)
        pod["status"] = {
            "phase": (
                "Running"
                if any("running" in item["state"] for item in statuses)
                else "Pending"
            ),
            "conditions": [
                {"type": "ContainersReady", "status": str(ready)},
                {"type": "Ready", "status": str(ready)},
            ],
            "containerStatuses": statuses,
        }
        self.objects[("pod", name)] = pod

    def start(self, pod: dict[str, Any], container: dict[str, Any]) -> dict[str, Any]:
        """The container's status once the kubelet tried to start it."""

        name = container["name"]
        volumes = {item["name"]: item for item in pod["spec"].get("volumes", [])}
        for mount in container.get("volumeMounts", []):
            volume = volumes.get(mount["name"])
            problem = None if volume is None else self.mount_problem(volume)
            if volume is None or problem:
                return status(name, waiting="ContainerCreating", message=problem)
        try:
            environment = self.environment(pod, container)
        except KeyError as exc:
            return status(name, waiting="CreateContainerConfigError", message=str(exc))
        command = container_command(container)
        missing = [
            item
            for item in executed(command)
            if item not in image_programs(container["image"])
        ]
        if missing:
            message = f'exec: "{missing[0]}": stat {missing[0]}: no such file'
            return status(name, terminated=(128, "StartError"), message=message)
        argv = program_argv(command)
        if PurePosixPath(argv[0]).name == "gpu-fault-collector":
            try:
                collector = self.collector(argv, environment)
            except (Exception, SystemExit) as exc:
                message = f"{type(exc).__name__}: {exc}"
                return status(name, terminated=(1, "Error"), message=message)
            key = (pod["metadata"]["name"], name)
            self.reporters[key] = collector
            self.environments[key] = environment
        # A probe this kubelet cannot run never passes.
        probed = any(key in container for key in ("readinessProbe", "startupProbe"))
        return status(name, ready=not probed)

    def collector(self, argv: list[str], environment: dict[str, str]) -> Any:
        """``gpu-fault-collector`` started by its own entry point, minus the loop."""

        assert argv[1:2] == ["training-progress"], f"only the reporter runs: {argv}"
        built: list[TrainingProgressCollector] = []

        def capture(collector: TrainingProgressCollector) -> None:
            built.append(collector)

        with (
            mock.patch.object(TrainingProgressCollector, "run", capture),
            mock.patch.object(collectors_cli, "configure_logging", lambda: None),
            mock.patch.object(sys, "argv", list(argv)),
            mock.patch.dict(os.environ, environment, clear=True),
        ):
            collectors_cli.main()
        assert len(built) == 1, f"the entry point built one collector: {built}"
        return built[0]

    def mount_problem(self, volume: dict[str, Any]) -> str | None:
        if "emptyDir" in volume:
            return None
        secret = volume.get("secret")
        if isinstance(secret, dict):
            keys = [item["key"] for item in secret.get("items", [])]
            if secret.get("secretName") != CONNECTION_SECRET:
                return f"secret {secret.get('secretName')} not found"
            if any(key not in CONNECTION for key in keys):
                return f"secret {CONNECTION_SECRET} lacks one of {keys}"
            return None
        return f"volume {volume['name']} is not modelled"

    def environment(
        self, pod: dict[str, Any], container: dict[str, Any]
    ) -> dict[str, str]:
        fields = {
            "spec.nodeName": pod["spec"].get("nodeName"),
            "metadata.uid": pod["metadata"]["uid"],
            "metadata.name": pod["metadata"]["name"],
            "metadata.namespace": NAMESPACE,
        }
        result: dict[str, str] = {}
        for item in container.get("env", []):
            source = item.get("valueFrom")
            if source is None:
                result[item["name"]] = str(item.get("value", ""))
            elif "secretKeyRef" in source:
                reference = source["secretKeyRef"]
                if reference.get("name") != CONNECTION_SECRET or (
                    reference.get("key") not in CONNECTION
                ):
                    raise KeyError(f"{reference.get('name')}/{reference.get('key')}")
                result[item["name"]] = CONNECTION[reference["key"]]
            else:
                result[item["name"]] = str(fields[source["fieldRef"]["fieldPath"]])
        return result

    # -- what a test reads back ----------------------------------------------
    def container(self, pod: dict[str, Any], name: str) -> dict[str, Any]:
        (container,) = [c for c in pod["spec"]["containers"] if c["name"] == name]
        return container

    def status(self, pod: str, name: str) -> dict[str, Any]:
        statuses = self.objects[("pod", pod)]["status"]["containerStatuses"]
        (found,) = [item for item in statuses if item["name"] == name]
        return found

    def running(self, pod: dict[str, Any], name: str) -> bool:
        return "running" in self.status(pod["metadata"]["name"], name)["state"]

    def mounts(
        self, pod: dict[str, Any], container: dict[str, Any]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        volumes = {item["name"]: item for item in pod["spec"].get("volumes", [])}
        return [(mount, volumes[mount["name"]]) for mount in container["volumeMounts"]]

    def remap(self, pod: dict[str, Any], container: dict[str, Any], text: str) -> str:
        """``text`` with the container's emptyDir mount paths made local, once."""

        local = {
            mount["mountPath"]: str(self.volume(pod, mount["name"]))
            for mount, volume in self.mounts(pod, container)
            if "emptyDir" in volume
        }
        if not local:
            return text
        pattern = "|".join(map(re.escape, sorted(local, key=len, reverse=True)))
        return re.sub(
            rf"(?<![\w/])({pattern})(?=/|$|[^\w/])",
            lambda match: local[match.group(1)],
            text,
        )

    def volume(self, pod: dict[str, Any], name: str) -> Path:
        path = self.root / pod["metadata"]["name"] / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def writable(self, pod: str, name: str, path: str) -> bool:
        """Whether ``path`` in the container is on a writable file system."""

        value = self.objects[("pod", pod)]
        container = self.container(value, name)
        covering = [
            mount
            for mount, _volume in self.mounts(value, container)
            if path == mount["mountPath"] or path.startswith(mount["mountPath"] + "/")
        ]
        if covering:
            deepest = max(covering, key=lambda mount: len(mount["mountPath"]))
            return not deepest.get("readOnly", False)
        context = container.get("securityContext") or {}
        return not context.get("readOnlyRootFilesystem", False)

    def mounted_secret(self, pod: str, name: str, path: str) -> str | None:
        """The Secret item a mount serves at ``path`` in the container."""

        value = self.objects[("pod", pod)]
        for mount, volume in self.mounts(value, self.container(value, name)):
            secret = volume.get("secret")
            if not isinstance(secret, dict):
                continue
            for item in secret.get("items", []):
                if f"{mount['mountPath']}/{item['path']}" == path:
                    return CONNECTION[item["key"]]
        return None

    def report(self, pod: str, name: str) -> TrainingProgressHeartbeat:
        """One cycle of the reporter the kubelet started, through its own mounts."""

        collector = self.reporters[(pod, name)]
        value = self.objects[("pod", pod)]
        local = self.remap(
            value, self.container(value, name), str(collector.progress_path)
        )
        sink = CapturingSink()
        collector.sink = sink
        collector.progress_path = Path(local)
        heartbeat = collector.collect_once()
        assert [path for path, _payload in sink.posts] == [TRAINING_PROGRESS_PATH], (
            sink.posts
        )
        return heartbeat

    def local_path(self, pod: str, name: str, path: str) -> Path:
        value = self.objects[("pod", pod)]
        return Path(self.remap(value, self.container(value, name), path))


def status(
    name: str,
    *,
    ready: bool = False,
    waiting: str | None = None,
    terminated: tuple[int, str] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    if waiting is not None:
        state: dict[str, Any] = {"waiting": {"reason": waiting, "message": message}}
    elif terminated is not None:
        code, reason = terminated
        state = {"terminated": {"exitCode": code, "reason": reason, "message": message}}
    else:
        state = {"running": {"startedAt": "2026-10-24T22:50:00Z"}}
    return {
        "name": name,
        "ready": ready and "running" in state,
        "restartCount": 0,
        "started": "running" in state,
        "state": state,
    }


@pytest.fixture
def private_umask() -> Iterator[None]:
    """``run_general_live.main`` runs under 0o077; the journals require it."""

    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


pytestmark = pytest.mark.usefixtures("private_umask")


def progress_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    variant: str = "nan",
    ranks: tuple[int, ...] = (0,),
    executor_image: str = EXECUTOR_IMAGE,
) -> tuple[ProgressCluster, Any]:
    """The runner's own job wrapper over a private context and a fake cluster."""

    cluster = ProgressCluster(tmp_path / "node", executor_image=executor_image)
    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(cluster.run))
    monkeypatch.setattr(managed_workload_fixture, "time", VirtualClock())
    kube = tmp_path / "kube"
    kube.mkdir()
    for plane in ("cpu", "gpu"):
        (kube / plane).write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    site_root = tmp_path / "site"
    site_root.mkdir()
    regional = RegionalLiveSettings(
        cpu_kubeconfig=kube / "cpu",
        gpu_kubeconfig=kube / "gpu",
        gpu_context=CONTEXT,
        namespace=NAMESPACE,
        cluster_id=CLUSTER,
        region="us-west-2",
    )
    spec = SPECS[variant]
    settings = GeneralLiveSettings(
        regional=regional,
        case_id=spec.case_id,
        variant=spec.variant,
        node=NODE,
        peer_node="",
        host_probe_image="public.ecr.aws/gfrs/probe@sha256:" + "1" * 64,
        hyperpod_cluster="",
        site_file=write_site_file(site_root),
        approvals=(),
    )
    case_dir = tmp_path / "run" / "cases" / spec.key
    ctx = VariantContext(
        spec=spec,
        settings=settings,
        regional=RegionalLiveFixture(regional),
        case_dir=case_dir,
        run_id=f"gl027-{spec.variant}-a1-0123abcd",
        attempt=1,
        deadline=datetime.now(timezone.utc) + timedelta(hours=1),
        evidence=Evidence(case_dir),
    )
    name = f"gl27-{spec.variant.replace('_', '-')}-a1-0123abcd"
    return cluster, progress.ManagedProgressJob(ctx, name=name, ranks=ranks)


@pytest.mark.parametrize("ranks", [(0,), (0, 1)], ids=["one-rank", "two-ranks"])
def test_the_progress_job_turns_ready_and_reports_what_the_runner_writes(
    ranks: tuple[int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = "nan" if ranks == (0,) else "straggler_group"
    cluster, job = progress_job(tmp_path, monkeypatch, variant=variant, ranks=ranks)

    snapshot = job.start()  # prewarm, gpu-training-submit render, create, health wait

    (pod,) = snapshot["pods"]
    assert (pod["phase"], pod["ready"], pod["node"]) == ("Running", True, NODE), (
        cluster.objects[("pod", pod["name"])]["status"]
    )
    reporters = [f"progress-reporter-{rank}" for rank in ranks]
    assert sorted(name for _pod, name in cluster.reporters) == reporters, (
        "every reporter is the product's own collector, started by its own CLI"
    )
    states: list[Path] = []
    for rank, reporter in zip(ranks, reporters, strict=True):
        key = (pod["name"], reporter)
        environment = cluster.environments[key]
        # TLS to the private control-plane endpoint trusts the Secret's CA.
        served = cluster.mounted_secret(
            pod["name"], reporter, environment["SSL_CERT_FILE"]
        )
        assert served == CONNECTION["ca.crt"], environment
        # The durable outbox has somewhere to go.
        outbox = cluster.reporters[key].sink.outbox_path
        assert outbox is not None, "the reporter keeps a durable outbox"
        assert cluster.writable(pod["name"], reporter, str(outbox.parent)), outbox
        states.append(cluster.local_path(pod["name"], reporter, str(outbox.parent)))

        job.write_progress(
            rank,
            step=10 + rank,
            numerical_error=True,
            checkpoint_ref="s3://test/unchecked",
            labels={"progress_group": "data-parallel"},
        )
        heartbeat = cluster.report(pod["name"], reporter)

        assert (heartbeat.cluster_id, heartbeat.attempt_id, heartbeat.rank) == (
            CLUSTER,
            job.attempt_id,
            rank,
        )
        # The identity the control plane matches against the trainer's allocation.
        assert (heartbeat.node_id, heartbeat.pod_uid, heartbeat.container_name) == (
            NODE,
            pod["uid"],
            "pytorch",
        )
        assert (
            heartbeat.step,
            heartbeat.numerical_error,
            heartbeat.checkpoint_ref,
            heartbeat.labels,
        ) == (
            10 + rank,
            True,
            "s3://test/unchecked",
            {"progress_group": "data-parallel"},
        )
    writes = [item for item in cluster.execs if item["container"] == "pytorch"]
    assert [item["stdin"] for item in writes] == [True] * len(ranks), writes
    assert len(set(states)) == len(ranks), "each reporter keeps its own outbox"


def test_a_reporter_image_without_the_product_never_turns_the_pod_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live failure, rebuilt: a reporter whose image ships no collector."""

    cluster, job = progress_job(tmp_path, monkeypatch, executor_image=TRAINING_IMAGE)

    with pytest.raises(RegionalFixtureError, match="did not become healthy") as raised:
        job.start()

    assert "'phase': 'Running', 'ready': False" in str(raised.value), raised.value
    (pod,) = [name for kind, name in cluster.objects if kind == "pod"]
    reporter = cluster.status(pod, "progress-reporter-0")["state"]
    assert reporter["terminated"]["reason"] == "StartError", reporter
    assert cluster.status(pod, "pytorch")["ready"] is True, "the trainer runs fine"
    # The failure names the container the Pod waits on, and why.
    assert (
        f"not ready: {{'{pod}/progress-reporter-0': {{'state': 'terminated', "
        "'reason': 'StartError', 'exit_code': 128"
    ) in str(raised.value), raised.value


@pytest.mark.parametrize(
    "image",
    [
        "public.ecr.aws/docker/library/python:3.12-slim",  # the unrendered template
        "111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-fault:latest",
    ],
)
def test_a_reporter_image_that_is_not_digest_pinned_is_refused_before_any_create(
    image: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RegionalFixtureError, match="digest-pinned"):
        progress_job(tmp_path, monkeypatch, executor_image=image)
