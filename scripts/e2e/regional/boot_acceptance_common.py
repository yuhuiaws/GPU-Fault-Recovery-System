from __future__ import annotations

import json
import os
import subprocess
import sys
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


class BootAcceptanceError(RuntimeError):
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
    umask: int | None = None,
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
        preexec_fn=(lambda: os.umask(umask)) if umask is not None else None,
    )
    if check and completed.returncode:
        raise BootAcceptanceError(
            f"command failed ({completed.returncode}): {' '.join(command[:5])}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    return completed


def write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    path.chmod(0o600)


def arn_resource_name(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def parse_probe_json(output: str) -> Any:
    """The JSON a probe printed: the whole output, or its last line.

    Probes written for these fixtures print one compact line, but audit scripts
    reused as probes (``audit_executor_readiness.py``) print an indented
    document whose last line is ``}``. Accept both; a kubectl warning ahead of
    a single-line document still resolves through the last line.
    """

    text = output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        if not lines:
            raise
        return json.loads(lines[-1])


class SiteFixture:
    def __init__(self, site_file: Path, cluster_id: str = "") -> None:
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
            raise BootAcceptanceError("site does not resolve a GPU kubeconfig")
        self.gpu_kubeconfig = Path(gpu_value).resolve()
        candidates = [
            item
            for item in self.config["clusters"]
            if not cluster_id or str(item["cluster_id"]) == cluster_id
        ]
        if len(candidates) != 1:
            raise BootAcceptanceError(
                "the selected BOOT case requires exactly one target cluster"
            )
        self.target = candidates[0]
        self.cluster_id = str(self.target["cluster_id"])
        self.gpu_context = str(self.target["context"])
        self.regional = RegionalLiveFixture(
            RegionalLiveSettings(
                cpu_kubeconfig=self.cpu_kubeconfig,
                gpu_kubeconfig=self.gpu_kubeconfig,
                gpu_context=self.gpu_context,
                namespace=self.namespace,
                cluster_id=self.cluster_id,
                region=self.region,
            )
        )

    def pods(self, plane: str, app: str) -> list[str]:
        value = json.loads(
            self.regional.kubectl(
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

    def pod_json(
        self,
        plane: str,
        pod: str,
        script: str,
        *arguments: str,
        timeout: int = 180,
    ) -> dict[str, Any]:
        output = self.regional.kubectl(
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
        value = parse_probe_json(output)
        if not isinstance(value, dict):
            raise BootAcceptanceError("Pod probe did not return a JSON object")
        return cast(dict[str, Any], value)

    def exec(
        self,
        plane: str,
        pod: str,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        command = ["kubectl", "--kubeconfig"]
        if plane == "cpu":
            command.append(str(self.cpu_kubeconfig))
        elif plane == "gpu":
            command.extend(
                [
                    str(self.gpu_kubeconfig),
                    "--context",
                    self.gpu_context,
                ]
            )
        else:
            raise ValueError(f"unknown plane: {plane}")
        command.extend(
            [
                "-n",
                self.namespace,
                "exec",
                "-i",
                pod,
                "--",
                *arguments,
            ]
        )
        return run(
            command,
            input_text=input_text,
            check=check,
            timeout=timeout,
        )
