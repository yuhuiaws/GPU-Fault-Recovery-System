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


# How long cleanup waits for a Pod delete to take effect before forcing it. A
# privileged hostPID Pod whose node is mid-reboot can sit Terminating for the
# whole reboot; waiting forever on it is what used to leave the ConfigMap and
# the residual check unreached.
POD_DELETE_WAIT_SECONDS = 120
POD_FORCE_DELETE_WAIT_SECONDS = 30


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
        try:
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
        except subprocess.TimeoutExpired as exc:
            raise HostProbeError(
                f"kubectl timed out after {timeout}s: {' '.join(arguments)}"
            ) from exc
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
        # The pod name is a digest of (case, run, node), so a rerun reuses it.
        # An operator abort deliberately skips cleanup, and the previous pod
        # then outlives its activeDeadlineSeconds as phase Failed. `apply`
        # onto that object is a no-op and `wait --for=condition=Ready` can
        # only time out (observed 2026-09-06 06:34Z after a SIGTERM at
        # 05:22Z). Delete whatever carries our name first; a Running pod of
        # this name can only be a dead runner's, never this fixture's.
        self._kubectl(
            "delete",
            "pod",
            self.pod,
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=180,
        )
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
            "exec chroot /host /opt/gpu-fault/current/venv/bin/python "
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
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-20:])
        payload: Any = {}
        if lines:
            try:
                payload = json.loads(lines[-1])
            except json.JSONDecodeError as exc:
                # A probe that died before `emit` -- OOM, a missing interpreter,
                # a chroot that failed -- leaves a traceback or nothing on its
                # last line. That is a probe failure with a stderr to read, not
                # a JSONDecodeError for the runner's cleanup path to trip over.
                raise HostProbeError(
                    f"host probe returned no JSON (exit {completed.returncode}); "
                    f"last stdout line: {lines[-1][:200]!r}; stderr: {stderr_tail}"
                ) from exc
        if not isinstance(payload, dict):
            raise HostProbeError(
                f"host probe returned a non-object (exit {completed.returncode}): "
                f"{lines[-1][:200]!r}; stderr: {stderr_tail}"
            )
        if completed.returncode or "error" in payload:
            raise HostProbeError(f"host probe failed: {payload or stderr_tail}")
        return payload

    def _present(self, kind: str, name: str) -> bool:
        # `--ignore-not-found` exits 0 with empty output for an absent object;
        # any non-zero exit is an API failure, and `check=True` refuses to read
        # its empty stdout as "gone".
        return bool(
            self._kubectl(
                "get",
                kind,
                name,
                "--ignore-not-found",
                "-o",
                "name",
                timeout=60,
            ).stdout.strip()
        )

    def residuals(self) -> dict[str, bool]:
        """Whether the Pod and ConfigMap still exist; raises when kubectl fails.

        An earlier version read a failed kubectl's empty stdout as "no
        residual", which is the one answer a residual check must never give
        by accident: it turns an unreachable API server into a clean audit.
        """

        result = {}
        for kind, name in (("pod", self.pod), ("configmap", self.configmap)):
            result[f"{kind}/{name}"] = self._present(kind, name)
        return result

    def _wait_pod_gone(self, seconds: int) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            try:
                if not self._present("pod", self.pod):
                    return True
            except HostProbeError:
                # A transient API failure mid-poll is not "still present";
                # keep polling until the bound, then let the force path and
                # the final residual check speak.
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(2)

    def cleanup(self) -> dict[str, bool]:
        """Remove the host script, the Pod and the ConfigMap, then audit.

        Every step is bounded so the ConfigMap delete and the residual check are
        reached even when the Pod refuses to die: delete without waiting, poll
        for it to vanish, force it, and only then move on. The residual check
        is what the case records; a cleanup that never got there recorded
        nothing.
        """

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
            "--wait=false",
            check=False,
            timeout=60,
        )
        if not self._wait_pod_gone(POD_DELETE_WAIT_SECONDS):
            self._kubectl(
                "delete",
                "pod",
                self.pod,
                "--ignore-not-found",
                "--wait=false",
                "--grace-period=0",
                "--force",
                check=False,
                timeout=60,
            )
            self._wait_pod_gone(POD_FORCE_DELETE_WAIT_SECONDS)
        self._kubectl(
            "delete",
            "configmap",
            self.configmap,
            "--ignore-not-found",
            check=False,
            timeout=60,
        )
        deadline = time.monotonic() + 30
        residuals = self.residuals()
        while any(residuals.values()) and time.monotonic() < deadline:
            time.sleep(1)
            residuals = self.residuals()
        return residuals
