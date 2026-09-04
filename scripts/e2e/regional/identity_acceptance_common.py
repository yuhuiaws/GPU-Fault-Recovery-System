from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
)

EXECUTOR_APP = "gpu-fault-cluster-executor"
CPU_APPS = ("gpu-fault-api-ha", "gpu-fault-control-worker")
REGISTRY_SECRET = "gpu-fault-regional-clusters"
CONNECTION_SECRET = "gpu-fault-regional-connection"
FORBIDDEN_NAMESPACE = "gf-forbidden-probe"
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class IdentityAcceptanceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    if check and completed.returncode:
        raise IdentityAcceptanceError(
            f"command failed ({completed.returncode}): {' '.join(command[:5])}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    return completed


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: str
    context: str
    region: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    executor_role_arn: str
    control_plane_url: str
    ca_file: Path


class IdentitySite:
    def __init__(self, site_file: Path) -> None:
        self.site_file = site_file.resolve()
        self.site = load_site(self.site_file, repository_root=ROOT)
        self.config = self.site.release_config
        self.namespace = str(self.config["namespace"])
        self.region = str(self.config["aws_region"])
        self.cpu_kubeconfig = Path(str(self.config["cpu_kubeconfig"])).resolve()
        gpu_value = str(
            self.config.get("gpu_kubeconfig")
            or self.site.environment.get("KUBECONFIG")
            or ""
        )
        if not gpu_value:
            raise IdentityAcceptanceError("site does not resolve a GPU kubeconfig")
        self.gpu_kubeconfig = Path(gpu_value).resolve()
        self.targets = {
            str(item["cluster_id"]): ClusterTarget(
                cluster_id=str(item["cluster_id"]),
                context=str(item["context"]),
                region=str(item["region"]),
                hyperpod_cluster_name=str(item["hyperpod_cluster_name"]),
                eks_cluster_arn=str(item["eks_cluster_arn"]),
                executor_role_arn=str(item["executor_irsa_role_arn"]),
                control_plane_url=str(item["control_plane_url"]),
                ca_file=Path(str(item["ca_file"])).resolve(),
            )
            for item in self.config["clusters"]
        }
        if not self.targets:
            raise IdentityAcceptanceError("site contains no GPU clusters")

    def target(self, cluster_id: str) -> ClusterTarget:
        if not cluster_id:
            if len(self.targets) != 1:
                raise IdentityAcceptanceError(
                    "--cluster-id is required for a multi-cluster site"
                )
            return next(iter(self.targets.values()))
        try:
            return self.targets[cluster_id]
        except KeyError as exc:
            raise IdentityAcceptanceError(
                f"cluster is not present in the site: {cluster_id}"
            ) from exc

    def regional(self, target: ClusterTarget) -> RegionalLiveFixture:
        return RegionalLiveFixture(
            RegionalLiveSettings(
                cpu_kubeconfig=self.cpu_kubeconfig,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=target.context,
                namespace=self.namespace,
                cluster_id=target.cluster_id,
                region=self.region,
            )
        )

    def cpu(self, *arguments: str, **kwargs: Any) -> str:
        target = next(iter(self.targets.values()))
        return str(self.regional(target).kubectl("cpu", *arguments, **kwargs))

    def gpu(self, target: ClusterTarget, *arguments: str, **kwargs: Any) -> str:
        return str(self.regional(target).kubectl("gpu", *arguments, **kwargs))

    def ready_pods(self, plane: str, app: str, target: ClusterTarget) -> list[str]:
        value = json.loads(
            self.regional(target).kubectl(
                plane,
                "get",
                "pod",
                "-l",
                f"app={app}",
                "-o",
                "json",
            )
        )
        result = []
        for item in value.get("items", []):
            conditions = item.get("status", {}).get("conditions", [])
            if item.get("status", {}).get("phase") != "Running":
                continue
            if not any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in conditions
            ):
                continue
            result.append(str(item["metadata"]["name"]))
        return sorted(result)

    def any_executor_pod(self, target: ClusterTarget) -> str:
        value = json.loads(
            self.gpu(
                target,
                "get",
                "pod",
                "-l",
                f"app={EXECUTOR_APP}",
                "-o",
                "json",
            )
        )
        names = sorted(
            str(item["metadata"]["name"])
            for item in value.get("items", [])
            if item.get("status", {}).get("phase") == "Running"
        )
        if not names:
            raise IdentityAcceptanceError(
                f"no Running executor Pod for {target.cluster_id}"
            )
        return names[0]

    def pod_json(
        self,
        plane: str,
        target: ClusterTarget,
        pod: str,
        script: str,
        *arguments: str,
        timeout: int = 180,
    ) -> dict[str, Any]:
        output = self.regional(target).kubectl(
            plane,
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-",
            *arguments,
            input_text=script,
            timeout=timeout,
        )
        value = json.loads(output.splitlines()[-1])
        if not isinstance(value, dict):
            raise IdentityAcceptanceError("Pod probe did not return a JSON object")
        return cast(dict[str, Any], value)

    def registry(self) -> list[dict[str, Any]]:
        value = json.loads(
            self.cpu(
                "get",
                "secret",
                REGISTRY_SECRET,
                "-o",
                "json",
            )
        )
        return cast(
            list[dict[str, Any]],
            json.loads(base64.b64decode(value["data"]["clusters.json"])),
        )

    def write_registry(self, entries: list[dict[str, Any]]) -> None:
        patch = {
            "stringData": {"clusters.json": json.dumps(entries, separators=(",", ":"))}
        }
        self.cpu(
            "patch",
            "secret",
            REGISTRY_SECRET,
            "--type=merge",
            "--patch-file=/dev/stdin",
            input_text=json.dumps(patch, separators=(",", ":")),
        )

    def rollout_control(self) -> float:
        started = time.monotonic()
        for deployment in CPU_APPS:
            self.cpu("rollout", "restart", f"deployment/{deployment}")
        for deployment in CPU_APPS:
            self.cpu(
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=600s",
                timeout=660,
            )
        return time.monotonic() - started


CLAIM_PROBE = r"""
import json
import os
import ssl
import urllib.error
import urllib.request

payload = {
    "executor_id": "regional-identity-acceptance",
    "executor_protocol_version": int(
        os.getenv("GPU_FAULT_EXECUTOR_PROTOCOL_VERSION", "2")
    ),
    "executor_artifact_sha256": os.environ[
        "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"
    ],
    "executor_compatibility_digest": os.environ[
        "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"
    ],
    "execution_owners": ["gpu-fault-kubernetes-adapter"],
    "max_commands": 1,
    "lease_seconds": 60,
}
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    + "/v1/regional/executors/claim",
    data=json.dumps(payload, separators=(",", ":")).encode(),
    method="POST",
    headers={
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    },
)
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
try:
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        print(json.dumps({"status": response.status}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"status": exc.code, "detail": exc.read().decode()}))
"""


def claim(site: IdentitySite, target: ClusterTarget) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    return site.pod_json("gpu", target, pod, CLAIM_PROBE)


def secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_cluster_token(site: IdentitySite, target: ClusterTarget) -> str:
    value = json.loads(
        site.gpu(
            target,
            "get",
            "secret",
            CONNECTION_SECRET,
            "-o",
            "json",
        )
    )
    return base64.b64decode(value["data"]["cluster-token"]).decode().strip()


def write_cluster_token(
    site: IdentitySite,
    target: ClusterTarget,
    token: str,
) -> None:
    patch = {
        "stringData": {
            "cluster-token": token,
        },
    }
    site.gpu(
        target,
        "patch",
        "secret",
        CONNECTION_SECRET,
        "--type=merge",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(patch, separators=(",", ":")),
    )


def rollout_executor(site: IdentitySite, target: ClusterTarget) -> float:
    started = time.monotonic()
    site.gpu(target, "rollout", "restart", f"deployment/{EXECUTOR_APP}")
    site.gpu(
        target,
        "rollout",
        "status",
        f"deployment/{EXECUTOR_APP}",
        "--timeout=600s",
        timeout=660,
    )
    return time.monotonic() - started
