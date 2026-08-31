from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


class HostProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class HostProbeSettings:
    kubeconfig: Path
    context: str
    namespace: str
    node: str
    image: str
    case_id: str
    run_id: str
    probe_script: Path
    active_deadline_seconds: int = 1800

    def __post_init__(self) -> None:
        if not self.kubeconfig.is_file():
            raise ValueError("host probe kubeconfig does not exist")
        if not all(
            (
                self.context,
                self.namespace,
                self.node,
                self.image,
                self.case_id,
                self.run_id,
            )
        ):
            raise ValueError("host probe settings contain an empty identity")
        if not self.probe_script.is_file():
            raise ValueError("host probe script does not exist")
        if not 60 <= self.active_deadline_seconds <= 7200:
            raise ValueError("host probe active deadline is outside 60..7200 seconds")


class HostProbeFixture:
    def __init__(self, settings: HostProbeSettings) -> None:
        self.settings = settings
        identity = f"{settings.case_id}\0{settings.run_id}\0{settings.node}".encode()
        suffix = hashlib.sha256(identity).hexdigest()[:10]
        self.configmap = f"gpu-fault-host-probe-{suffix}"
        self.pod = f"gpu-fault-host-probe-{suffix}"
        self.host_script = f"/run/gpu-fault-host-probe-{suffix}.py"

    def _kubectl(
        self,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(self.settings.kubeconfig),
                "--context",
                self.settings.context,
                "-n",
                self.settings.namespace,
                *arguments,
            ],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode:
            raise HostProbeError(
                f"kubectl failed ({completed.returncode}): "
                f"{' '.join(arguments)}: {completed.stderr.strip()}"
            )
        return completed

    def manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        labels = {
            "app": "gpu-fault-acceptance-host-probe",
            "gpu-fault.io/acceptance-case": self.settings.case_id,
            "gpu-fault.io/acceptance-run": self.settings.run_id,
        }
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.configmap,
                "namespace": self.settings.namespace,
                "labels": labels,
            },
            "data": {
                self.settings.probe_script.name: self.settings.probe_script.read_text(
                    encoding="utf-8"
                )
            },
        }
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod,
                "namespace": self.settings.namespace,
                "labels": labels,
            },
            "spec": {
                "restartPolicy": "Never",
                "nodeName": self.settings.node,
                "hostPID": True,
                "hostNetwork": True,
                "activeDeadlineSeconds": self.settings.active_deadline_seconds,
                "terminationGracePeriodSeconds": 0,
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "probe",
                        "image": self.settings.image,
                        "securityContext": {"privileged": True},
                        "command": ["/bin/bash", "-ceu"],
                        "args": ["trap : TERM INT; sleep 7200 & wait"],
                        "volumeMounts": [
                            {"name": "host-root", "mountPath": "/host"},
                            {
                                "name": "probe-script",
                                "mountPath": "/probe",
                                "readOnly": True,
                            },
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "host-root",
                        "hostPath": {"path": "/", "type": "Directory"},
                    },
                    {
                        "name": "probe-script",
                        "configMap": {
                            "name": self.configmap,
                            "defaultMode": 0o555,
                        },
                    },
                ],
            },
        }
        return configmap, pod

    def create(self) -> None:
        configmap, pod = self.manifests()
        self._kubectl("apply", "-f", "-", input_text=json.dumps(configmap))
        self._kubectl("apply", "-f", "-", input_text=json.dumps(pod))
        self._kubectl(
            "wait",
            "--for=condition=Ready",
            f"pod/{self.pod}",
            "--timeout=180s",
            timeout=210,
        )

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        command = (
            f"install -m 0700 /probe/{self.settings.probe_script.name} "
            f"/host{self.host_script}; "
            "exec chroot /host /opt/gpu-fault/venv/bin/python "
            f'{self.host_script} "$@"'
        )
        completed = self._kubectl(
            "exec",
            self.pod,
            "--",
            "/bin/bash",
            "-ceu",
            command,
            "host-probe",
            *arguments,
            check=False,
            timeout=timeout,
        )
        lines = completed.stdout.strip().splitlines()
        payload = json.loads(lines[-1]) if lines else {}
        if completed.returncode or "error" in payload:
            raise HostProbeError(
                f"host probe failed: {payload or completed.stderr.strip()}"
            )
        return payload

    def residuals(self) -> dict[str, bool]:
        result = {}
        for kind, name in (("pod", self.pod), ("configmap", self.configmap)):
            present = self._kubectl(
                "get",
                kind,
                name,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            ).stdout.strip()
            result[f"{kind}/{name}"] = bool(present)
        return result

    def cleanup(self) -> dict[str, bool]:
        self._kubectl(
            "exec",
            self.pod,
            "--",
            "/bin/rm",
            "-f",
            f"/host{self.host_script}",
            check=False,
            timeout=30,
        )
        self._kubectl(
            "delete",
            "pod",
            self.pod,
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=180,
        )
        self._kubectl(
            "delete",
            "configmap",
            self.configmap,
            "--ignore-not-found",
            check=False,
        )
        deadline = time.monotonic() + 30
        residuals = self.residuals()
        while any(residuals.values()) and time.monotonic() < deadline:
            time.sleep(1)
            residuals = self.residuals()
        return residuals
