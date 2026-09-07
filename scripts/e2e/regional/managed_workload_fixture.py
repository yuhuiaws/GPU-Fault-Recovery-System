from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
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

    def create(self, nodes: list[str]) -> None:
        if self.pods:
            raise RegionalFixtureError("image prewarm fixture is already created")
        for index, node in enumerate(nodes):
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
            lines = [
                line
                for line in completed.splitlines()
                if "HEARTBEAT" in line or "SUCCESS" in line
            ]
            logs[str(pod["name"])] = "\n".join(lines[-20:])
        return logs

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
            logs = last["heartbeat_logs"]
            if (
                len(pods) == self.settings.expected_pods
                and all(item["phase"] == "Running" and item["ready"] for item in pods)
                and len({item["node"] for item in pods}) == self.settings.expected_pods
                and all(
                    "HEARTBEAT" in logs.get(str(item["name"]), "")
                    and "all_reduce=" in logs.get(str(item["name"]), "")
                    for item in pods
                )
            ):
                return last
            time.sleep(10)
        raise RegionalFixtureError(f"managed workload did not become healthy: {last}")

    def wait_restarted(
        self,
        old_uids: set[str],
        *,
        timeout_seconds: int = 900,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.wait_running(timeout_seconds=30)
            except RegionalFixtureError:
                time.sleep(5)
                continue
            current_uids = {str(item["uid"]) for item in last["pods"]}
            if current_uids and current_uids.isdisjoint(old_uids):
                return last
            time.sleep(5)
        raise RegionalFixtureError(f"managed workload Pods were not replaced: {last}")

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
