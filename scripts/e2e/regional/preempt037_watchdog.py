"""Create and remove the run-owned CPU dispatcher restoration watchdog."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from scripts.e2e.regional import preempt037_verdicts as verdicts
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.seeded_command_fixture import delete_owned_resource

PROBE = Path(__file__).with_name("probes") / "preempt037_dispatcher_watchdog.py"


class DispatcherWatchdog:
    def __init__(
        self, regional: Any, *, baseline: dict[str, Any], restore_at: float
    ) -> None:
        self.regional = regional
        self.baseline = baseline
        self.restore_at = restore_at
        self.run_id = f"p037-{uuid4().hex[:16]}"
        self.name = f"gpu-fault-{self.run_id}"
        self.attempted: list[str] = []

    def cpu(self, *args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        return self.regional.kubectl(
            "cpu",
            *args,
            input_text=None if stdin is None else stdin.decode(),
            **kwargs,
        )

    def manifests(self) -> list[dict[str, Any]]:
        namespace = self.regional.settings.namespace
        metadata = {
            "name": self.name,
            "namespace": namespace,
            "labels": {
                "gpu-fault.io/acceptance-run": self.run_id,
                "gpu-fault.io/acceptance-case": verdicts.CASE_ID,
            },
        }
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata,
        }
        role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata,
            "rules": [
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments"],
                    "resourceNames": [verdicts.DEPLOYMENT],
                    "verbs": ["get", "patch"],
                }
            ],
        }
        binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata,
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": self.name,
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": self.name, "namespace": namespace}
            ],
        }
        environment = {
            "PREEMPT037_RESTORE_AT": str(self.restore_at),
            "PREEMPT037_NAMESPACE": namespace,
            "PREEMPT037_DEPLOYMENT_UID": self.baseline["uid"],
            "PREEMPT037_BASELINE": json.dumps(
                {
                    "present": self.baseline["present"],
                    "value": self.baseline["value"],
                    "image": self.baseline["image"],
                }
            ),
        }
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 3600,
                "ttlSecondsAfterFinished": 300,
                "template": {
                    "metadata": {"labels": metadata["labels"]},
                    "spec": {
                        "serviceAccountName": self.name,
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "watchdog",
                                "image": self.baseline["image"],
                                "command": [
                                    "/opt/gpu-fault/control-plane/bin/python",
                                    "-c",
                                    PROBE.read_text(encoding="utf-8"),
                                ],
                                "env": [
                                    {"name": name, "value": value}
                                    for name, value in environment.items()
                                ],
                                "resources": {
                                    "requests": {"cpu": "50m", "memory": "128Mi"},
                                    "limits": {"cpu": "200m", "memory": "256Mi"},
                                },
                            }
                        ],
                    },
                },
            },
        }
        return [service_account, role, binding, job]

    def arm(self) -> None:
        if not self.baseline.get("uid") or "@sha256:" not in str(
            self.baseline.get("image")
        ):
            raise RegionalFixtureError(
                "watchdog needs the bound Deployment UID and immutable image"
            )
        if self.restore_at <= time.time() + 30:
            raise RegionalFixtureError("watchdog restoration deadline is too close")
        for manifest in self.manifests():
            self.attempted.append(manifest["kind"].lower())
            self.cpu(
                "create", "-f", "-", stdin=json.dumps(manifest).encode(), timeout=120
            )
        self.cpu(
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            f"job-name={self.name}",
            "--timeout=180s",
            timeout=210,
        )
        lines = self.cpu("logs", f"job/{self.name}", "--tail=20", timeout=60)
        if not any(self._armed_line(line) for line in lines.splitlines()):
            raise RegionalFixtureError(
                "watchdog did not acknowledge its restoration deadline"
            )
        if self.restore_at <= time.time() + 30:
            raise RegionalFixtureError(
                "watchdog restoration deadline passed during startup"
            )

    def _armed_line(self, line: str) -> bool:
        try:
            value = json.loads(line)
        except ValueError:
            return False
        return (
            isinstance(value, dict)
            and value.get("state") == "ARMED"
            and value.get("restore_at") == self.restore_at
            and value.get("deployment_uid") == self.baseline["uid"]
            and value.get("image") == self.baseline["image"]
        )

    def cleanup(self) -> None:
        errors = []
        for kind in reversed(self.attempted):
            try:
                delete_owned_resource(
                    kind,
                    self.name,
                    self.run_id,
                    client=self.cpu,
                    namespace=self.regional.settings.namespace,
                )
            except Exception as exc:
                errors.append(f"{kind}: {type(exc).__name__}")
        if errors:
            raise RegionalFixtureError(
                "watchdog cleanup unproven: " + "; ".join(errors)
            )
