from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, cast

yaml = importlib.import_module("yaml")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
)


TRAINING_IMAGE = (
    "public.ecr.aws/deep-learning-containers/"
    "pytorch-training@sha256:"
    "ff2c928a2e7b3b290b7c3e353085a373b3cfc7f6b165e338b9be4df439feae38"
)
# The training fixtures print `HEARTBEAT ... all_reduce=<value>` every step and
# one `SUCCESS rank=<r>/<world>` line at the end; the loss lines
# (`rank=<r> step=<s> loss=<l>`) are what a per-rank loss check reads.
HEARTBEAT_LOG_MARKERS = ("HEARTBEAT", "SUCCESS", "loss=")
ALL_REDUCE_PATTERN = re.compile(r"all_reduce=([-+0-9.eE]+)")
SUCCESS_RANK_PATTERN = re.compile(r"SUCCESS rank=(\d+)/(\d+)")


def expected_all_reduce(world_size: int) -> float:
    """The all-reduce every fixture rank checks: sum of (rank + 1) over ranks."""

    if world_size < 1:
        raise ValueError("world size must be positive")
    return world_size * (world_size + 1) / 2


def heartbeat_healthy(log_text: str, *, world_size: int) -> bool:
    """Whether ``log_text`` proves the collective ran at ``world_size``.

    A substring test on ``all_reduce=`` passed for any Pod that had printed
    one heartbeat, whatever the value -- a job that came up at the wrong world
    size prints the same prefix. The value has to parse and equal the sum the
    fixture asserts, or a SUCCESS line has to name the same world size.
    """

    expected = expected_all_reduce(world_size)
    for match in ALL_REDUCE_PATTERN.finditer(log_text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value == expected:
            return True
    return any(
        int(match.group(2)) == world_size
        for match in SUCCESS_RANK_PATTERN.finditer(log_text)
    )


def render_node_pinned_manifest(
    source: Path,
    destination: Path,
    *,
    node: str,
) -> Path:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("node-pinned workload manifest is not a mapping")
    kind = str(document.get("kind") or "").lower()
    spec = document.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("node-pinned workload manifest has no spec")
    templates: list[dict[str, Any]] = []
    if kind == "job":
        template = spec.get("template")
        if isinstance(template, dict):
            templates.append(template)
    elif kind == "pytorchjob":
        replicas = spec.get("pytorchReplicaSpecs")
        if isinstance(replicas, dict):
            templates.extend(
                replica["template"]
                for replica in replicas.values()
                if isinstance(replica, dict)
                and isinstance(replica.get("template"), dict)
            )
    else:
        raise ValueError("node-pinned fixture supports Job or PyTorchJob")
    if not templates:
        raise ValueError("node-pinned workload has no Pod templates")
    for template in templates:
        pod_spec = template.get("spec")
        if not isinstance(pod_spec, dict):
            raise ValueError("node-pinned workload Pod template has no spec")
        pod_spec["nodeName"] = node
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    destination.chmod(0o600)
    return destination


@dataclass(frozen=True)
class ManagedWorkloadSettings:
    manifest: Path
    site_file: Path
    job_id: str
    attempt_id: str
    restart_budget: int
    expected_pods: int
    expected_gpu_count: int

    def __post_init__(self) -> None:
        if not self.manifest.is_file():
            raise ValueError("managed workload manifest does not exist")
        if not self.site_file.is_file():
            raise ValueError("regional site file does not exist")
        if not self.job_id or not self.attempt_id:
            raise ValueError("managed workload identity is empty")
        if self.restart_budget < 0:
            raise ValueError("restart budget cannot be negative")
        if self.expected_pods < 1 or self.expected_gpu_count < 1:
            raise ValueError("managed workload expectations must be positive")


class ImagePrewarmFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        case_id: str,
        run_id: str,
        image: str = TRAINING_IMAGE,
    ) -> None:
        self.regional = regional
        self.image = image
        suffix = hashlib.sha256(f"{case_id}\0{run_id}\0{image}".encode()).hexdigest()[
            :10
        ]
        self.prefix = f"gpu-fault-image-prewarm-{suffix}"
        self.pods: list[str] = []

    def manifest(self, node: str, index: int) -> dict[str, Any]:
        name = f"{self.prefix}-{index}"
        self.pods.append(name)
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": self.regional.settings.namespace,
                "labels": {
                    "app": "gpu-fault-image-prewarm",
                    "gpu-fault.io/acceptance-run": self.prefix,
                },
            },
            "spec": {
                "nodeName": node,
                "restartPolicy": "Never",
                "activeDeadlineSeconds": 1800,
                "terminationGracePeriodSeconds": 0,
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "prewarm",
                        "image": self.image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-ceu", "echo cached"],
                        "resources": {
                            "requests": {"cpu": "10m", "memory": "32Mi"},
                            "limits": {"cpu": "100m", "memory": "128Mi"},
                        },
                    }
                ],
            },
        }

    def create(self, nodes: list[str]) -> dict[str, list[str]]:
        """Pull the image onto ``nodes`` that do not already hold it.

        A node that reports the digest in ``status.images`` needs no Pod; the
        pull was the whole point, and the prewarm Pod's schedule/pull/exit
        cycle is the slowest step of the case when it runs on every candidate.
        Returns which nodes were skipped and which received a Pod.
        """

        if self.pods:
            raise RegionalFixtureError("image prewarm fixture is already created")
        cached = set(self.cached_nodes())
        skipped = [node for node in nodes if node in cached]
        pending = [node for node in nodes if node not in cached]
        for index, node in enumerate(pending):
            manifest = self.manifest(node, index)
            self.regional.kubectl(
                "gpu",
                "apply",
                "-f",
                "-",
                input_text=json.dumps(manifest),
            )
        for pod in self.pods:
            self.regional.kubectl(
                "gpu",
                "wait",
                "--for=jsonpath={.status.phase}=Succeeded",
                f"pod/{pod}",
                "--timeout=1800s",
                timeout=1830,
            )
        return {"skipped": skipped, "created": pending}

    def cached_nodes(self) -> list[str]:
        value = json.loads(self.regional.kubectl("gpu", "get", "node", "-o", "json"))
        digest = self.image.rsplit("@", 1)[-1]
        return sorted(
            item["metadata"]["name"]
            for item in value.get("items", [])
            if any(
                digest in str(name)
                for image in item.get("status", {}).get("images", [])
                for name in image.get("names", [])
            )
        )

    def cleanup(self) -> dict[str, bool]:
        for pod in self.pods:
            self.regional.kubectl(
                "gpu",
                "delete",
                "pod",
                pod,
                "--ignore-not-found",
                "--wait=true",
                check=False,
                timeout=180,
            )
        return {
            pod: bool(
                self.regional.kubectl(
                    "gpu",
                    "get",
                    "pod",
                    pod,
                    "--ignore-not-found",
                    "-o",
                    "name",
                    check=False,
                ).strip()
            )
            for pod in self.pods
        }


class ManagedWorkloadFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        settings: ManagedWorkloadSettings,
    ) -> None:
        self.regional = regional
        self.settings = settings
        document = yaml.safe_load(settings.manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("managed workload manifest is not a mapping")
        self.kind = str(document.get("kind") or "")
        self.name = str((document.get("metadata") or {}).get("name") or "")
        if not self.kind or not self.name:
            raise ValueError("managed workload manifest has no kind/name")

    @property
    def resource(self) -> str:
        return self.kind.lower()

    def delete(self) -> None:
        # Take the Pods away before the job: a graceful job delete first
        # SIGTERMs torchrun, which exits 1, and the watcher reads that non-zero
        # exit as a training failure -- the passive path then spends a restart
        # (or, with the budget gone, ESCALATES and mails the operator) over a
        # cleanup. A Pod object that simply disappears is read as a user stop
        # (the tombstone path), which is what a fixture tearing down its own
        # workload is.
        self.regional.kubectl(
            "gpu",
            "delete",
            "pod",
            "-l",
            f"gpu-fault.io/job-id={self.settings.job_id}",
            "--ignore-not-found",
            "--grace-period=0",
            "--force",
            "--wait=false",
            check=False,
            timeout=120,
        )
        self.regional.kubectl(
            "gpu",
            "delete",
            self.resource,
            self.name,
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=300,
        )
        # RESTART_WORKLOAD resubmits the workload under a new name
        # (``<name>-r-<hash>``) that keeps this job id as a label. Deleting the
        # source name alone leaves the restarted copy holding its GPUs for the
        # rest of its lifetime, and every later live case then fails preflight
        # with "GPU cluster already has a GPU workload".
        self.regional.kubectl(
            "gpu",
            "delete",
            self.resource,
            "-l",
            f"gpu-fault.io/job-id={self.settings.job_id}",
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=300,
        )

    def submit(self) -> dict[str, Any]:
        self.delete()
        command = [
            sys.executable,
            "-m",
            "gpu_fault.training_submit_cli",
            str(self.settings.manifest),
            "--site",
            str(self.settings.site_file),
            "--job-id",
            self.settings.job_id,
            "--attempt-id",
            self.settings.attempt_id,
            "--restart-budget",
            str(self.settings.restart_budget),
            "--kubeconfig",
            str(self.regional.settings.gpu_kubeconfig),
            "--context",
            self.regional.settings.gpu_context,
            "--namespace",
            self.regional.settings.namespace,
        ]
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
        }
        completed = self.regional.run(
            command,
            cwd=ROOT,
            env=environment,
            timeout=300,
        )
        return {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def annotate_auto_resume(self, value: str | None) -> None:
        annotation = "sagemaker.amazonaws.com/enable-job-auto-resume"
        if value is None:
            expression = f"{annotation}-"
        else:
            expression = f"{annotation}={value}"
        self.regional.kubectl(
            "gpu",
            "annotate",
            self.resource,
            self.name,
            expression,
            "--overwrite",
        )

    def workload(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                self.resource,
                self.name,
                "-o",
                "json",
            )
        )
        if not isinstance(value, dict):
            raise RegionalFixtureError("workload query did not return a JSON object")
        return cast(dict[str, Any], value)

    def pods(self) -> list[dict[str, Any]]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "pod",
                "-l",
                f"gpu-fault.io/job-id={self.settings.job_id}",
                "-o",
                "json",
            )
        )
        result = []
        for item in value.get("items", []):
            statuses = item.get("status", {}).get("containerStatuses", [])
            result.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "node": item["spec"].get("nodeName"),
                    "phase": item.get("status", {}).get("phase"),
                    "ready": bool(statuses)
                    and all(bool(status.get("ready")) for status in statuses),
                    "attempt_id": item["metadata"]
                    .get("labels", {})
                    .get("gpu-fault.io/attempt-id"),
                }
            )
        return sorted(result, key=lambda item: str(item["name"]))

    def heartbeat_logs(self, pods: list[dict[str, Any]]) -> dict[str, str]:
        logs = {}
        for pod in pods:
            completed = self.regional.kubectl(
                "gpu",
                "logs",
                str(pod["name"]),
                "--tail=200",
                check=False,
                timeout=60,
            )
            lines = completed.splitlines()
            # The heartbeat/SUCCESS tail and the loss tail are kept separately
            # so a step that prints a loss line per rank cannot push the last
            # heartbeats out of the window a consumer substring-checks.
            heartbeats = [
                line for line in lines if "HEARTBEAT" in line or "SUCCESS" in line
            ]
            losses = [line for line in lines if "loss=" in line]
            logs[str(pod["name"])] = "\n".join([*heartbeats[-20:], *losses[-20:]])
        return logs

    def pods_healthy(self, pods: list[dict[str, Any]]) -> bool:
        """The Pod-level half of ``wait_running``: count, phase, spread."""

        return bool(
            len(pods) == self.settings.expected_pods
            and all(item["phase"] == "Running" and item["ready"] for item in pods)
            and len({item["node"] for item in pods}) == self.settings.expected_pods
        )

    def logs_healthy(self, pods: list[dict[str, Any]], logs: dict[str, str]) -> bool:
        """The log half: every Pod heartbeats at the expected world size."""

        return all(
            "HEARTBEAT" in logs.get(str(item["name"]), "")
            and heartbeat_healthy(
                logs.get(str(item["name"]), ""),
                world_size=self.settings.expected_gpu_count,
            )
            for item in pods
        )

    def snapshot(self) -> dict[str, Any]:
        workload = self.workload()
        pods = self.pods()
        return {
            "workload": {
                "kind": self.kind,
                "name": self.name,
                "uid": workload["metadata"]["uid"],
                "annotations": workload["metadata"].get("annotations", {}),
                "labels": workload["metadata"].get("labels", {}),
                "suspend": workload.get("spec", {}).get("suspend"),
            },
            "pods": pods,
            "heartbeat_logs": self.heartbeat_logs(pods),
        }

    def wait_running(self, timeout_seconds: int = 900) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.snapshot()
            except Exception:
                time.sleep(5)
                continue
            pods = last["pods"]
            if self.pods_healthy(pods) and self.logs_healthy(
                pods, last["heartbeat_logs"]
            ):
                return last
            time.sleep(10)
        raise RegionalFixtureError(f"managed workload did not become healthy: {last}")

    def wait_restarted(
        self,
        old_uids: set[str],
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 10,
    ) -> dict[str, Any]:
        """Wait for the workload's Pods to be replaced and heartbeat again.

        Replacement is detected from ``pods()`` alone -- one ``kubectl get``
        per poll -- because the old shape ran ``wait_running`` in the loop,
        which pulled every Pod's logs every ten seconds for up to fifteen
        minutes while the old Pods were still terminating. Logs are read only
        once the new Pods are Running, and then only until they heartbeat.
        """

        deadline = time.monotonic() + timeout_seconds
        last_pods: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                last_pods = self.pods()
            except Exception:
                time.sleep(poll_seconds)
                continue
            current_uids = {str(item["uid"]) for item in last_pods}
            if (
                current_uids
                and current_uids.isdisjoint(old_uids)
                and self.pods_healthy(last_pods)
            ):
                break
            time.sleep(poll_seconds)
        else:
            raise RegionalFixtureError(
                f"managed workload Pods were not replaced: {last_pods}"
            )
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.snapshot()
            except Exception:
                time.sleep(poll_seconds)
                continue
            pods = last["pods"]
            if {str(item["uid"]) for item in pods} & old_uids:
                raise RegionalFixtureError(
                    f"an old managed workload Pod reappeared: {pods}"
                )
            if self.pods_healthy(pods) and self.logs_healthy(
                pods, last["heartbeat_logs"]
            ):
                return last
            time.sleep(poll_seconds)
        raise RegionalFixtureError(
            f"replaced managed workload Pods did not heartbeat: {last or last_pods}"
        )

    def wait_pod_uids_unchanged(
        self,
        expected_uids: set[str],
        *,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.snapshot()
            current_uids = {str(item["uid"]) for item in last["pods"]}
            if current_uids != expected_uids:
                raise RegionalFixtureError("managed workload Pod UIDs changed")
            time.sleep(5)
        return last
