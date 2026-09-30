from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.regional_commands import (
    RegionalCommandTimeout,
    RegionalFixtureError,
    run_fixture_command,
)


class HostProbeError(RuntimeError):
    pass


class HostProbeTransportError(HostProbeError):
    """The owned probe transport is unavailable, not a rejected host action."""


class HostProbeMissingResponseError(HostProbeError):
    """No response arrived; neither execution nor transport outcome is known."""


# How long cleanup waits for a Pod delete to take effect before forcing it. A
# privileged hostPID Pod whose node is mid-reboot can sit Terminating for the
# whole reboot; waiting forever on it is what used to leave the ConfigMap and
# the residual check unreached.
POD_DELETE_WAIT_SECONDS = 120
POD_FORCE_DELETE_WAIT_SECONDS = 30
PROBE_KEY = "probe.py"


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
    state_directory: Path
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
        self._script = settings.probe_script.read_text(encoding="utf-8")
        identity = json.dumps(
            [
                str(settings.kubeconfig.resolve()),
                settings.context,
                settings.namespace,
                settings.case_id,
                settings.run_id,
                settings.node,
                settings.probe_script.name,
            ]
        ).encode()
        suffix = hashlib.sha256(identity).hexdigest()[:10]
        self.configmap = f"gpu-fault-host-probe-{suffix}"
        self.pod = f"gpu-fault-host-probe-{suffix}"
        self.host_script = f"/run/gpu-fault-host-probe-{suffix}.py"
        self.state_path = settings.state_directory / f"{self.pod}.json"
        self._lock_fd: int | None = None
        self._scope = {
            "kubeconfig": str(settings.kubeconfig.resolve()),
            "context": settings.context,
            "namespace": settings.namespace,
            "node": settings.node,
            "image": settings.image,
            "case_id": settings.case_id,
            "run_id": settings.run_id,
            "script_sha256": hashlib.sha256(self._script.encode()).hexdigest(),
        }
        self._record: dict[str, Any] | None = None

    def _acquire(self) -> None:
        if self._lock_fd is not None:
            return
        self.settings.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self.state_path.with_suffix(".lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise HostProbeError("host probe ownership lock is invalid")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd = descriptor
            if self.state_path.is_symlink():
                raise HostProbeError("host probe ownership record is not private")
            if self.state_path.exists():
                state_stat = self.state_path.stat()
                if (
                    not stat.S_ISREG(state_stat.st_mode)
                    or state_stat.st_uid != os.geteuid()
                    or state_stat.st_mode & 0o077
                ):
                    raise HostProbeError("host probe ownership record is not private")
                record = json.loads(self.state_path.read_text())
                record = self._adopt_superseded_record(record)
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != 1
                    or record.get("scope") != self._scope
                    or not isinstance(record.get("owner"), str)
                    or not record["owner"]
                    or not isinstance(record.get("resources"), dict)
                    or set(record["resources"]) != {"pod", "configmap"}
                    or not isinstance(record.get("node_uid"), str)
                    or not record["node_uid"]
                    or type(record.get("closed")) is not bool
                    or type(record.get("script_may_exist")) is not bool
                ):
                    raise HostProbeError(
                        "host probe ownership record does not match this run"
                    )
                for kind, name in (("pod", self.pod), ("configmap", self.configmap)):
                    resource = record["resources"][kind]
                    if (
                        not isinstance(resource, dict)
                        or resource.get("name") != name
                        or type(resource.get("create_started")) is not bool
                        or (
                            resource.get("uid") is not None
                            and (
                                not isinstance(resource["uid"], str)
                                or not resource["uid"]
                            )
                        )
                    ):
                        raise HostProbeError("host probe resource receipt is malformed")
                self._record = record
        except BlockingIOError:
            os.close(descriptor)
            raise HostProbeError("another process owns this host probe") from None
        except BaseException:
            os.close(descriptor)
            self._lock_fd = None
            raise

    def _adopt_superseded_record(self, record: Any) -> Any:
        """A CLOSED record left by an earlier version of the same probe script.

        Records are keyed by (case, run, node); a later attempt sweeping an
        earlier attempt's identity meets that record with the old
        ``script_sha256`` (DESTR-017's fence probe after the 2026-09-18
        redesign refused its own preflight this way). A closed record with no
        script left on the node owns nothing, so it is adopted under the
        current script digest; anything open, or differing in any other scope
        field, is still refused.
        """

        if (
            not isinstance(record, dict)
            or record.get("closed") is not True
            or record.get("script_may_exist") is not False
            or not isinstance(record.get("scope"), dict)
        ):
            return record
        theirs = {k: v for k, v in record["scope"].items() if k != "script_sha256"}
        ours = {k: v for k, v in self._scope.items() if k != "script_sha256"}
        if theirs != ours or record["scope"] == self._scope:
            return record
        adopted = dict(record)
        adopted["superseded_script_sha256"] = record["scope"].get("script_sha256")
        adopted["scope"] = dict(self._scope)
        return adopted

    def _release(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _save(self) -> None:
        if self._record is None:
            raise HostProbeError("host probe has no ownership record")
        write_json_atomic(self.state_path, self._record)

    def _read(self, kind: str, name: str) -> dict[str, Any] | None:
        result = self._kubectl(
            "get", kind, name, "--ignore-not-found", "-o", "json", timeout=60
        )
        if not result.stdout.strip():
            return None
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
            raise HostProbeError("host probe resource read is malformed")
        if value["metadata"].get("name") != name:
            raise HostProbeError("host probe resource name differs")
        return value

    def _node_uid(self) -> str:
        value = self._read("node", self.settings.node)
        uid = value["metadata"].get("uid") if value else None
        if not isinstance(uid, str) or not uid:
            raise HostProbeError("host probe node identity is unavailable")
        return uid

    def _owned(self, kind: str, value: dict[str, Any]) -> str:
        if self._record is None:
            raise HostProbeError("host probe ownership is unproven")
        metadata = value["metadata"]
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        uid = metadata.get("uid")
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(labels, dict)
            or not isinstance(annotations, dict)
            or labels.get("gpu-fault.io/acceptance-case") != self.settings.case_id
            or labels.get("gpu-fault.io/acceptance-run") != self.settings.run_id
            or annotations.get("gpu-fault.io/probe-owner") != self._record["owner"]
            or annotations.get("gpu-fault.io/probe-script-sha256")
            != self._scope["script_sha256"]
        ):
            raise HostProbeError("host probe resource ownership changed")
        expected = self._record["resources"].get(kind, {}).get("uid")
        if expected is not None and expected != uid:
            raise HostProbeError("host probe resource UID changed")
        if kind == "pod" and (
            value.get("spec", {}).get("nodeName") != self.settings.node
            or annotations.get("gpu-fault.io/probe-node-uid")
            != self._record["node_uid"]
        ):
            raise HostProbeError("host probe target node changed")
        if kind == "configmap" and value.get("data") != {PROBE_KEY: self._script}:
            raise HostProbeError("host probe ConfigMap script changed")
        if kind == "pod":
            containers = value.get("spec", {}).get("containers", [])
            probe = [item for item in containers if item.get("name") == "probe"]
            if len(probe) != 1 or probe[0].get("image") != self.settings.image:
                raise HostProbeError("host probe container image changed")
        return uid

    def _delete_owned(self, kind: str, name: str, *, force: bool = False) -> None:
        value = self._read(kind, name)
        if value is None:
            return
        uid = self._owned(kind, value)
        if self._record is None:
            raise HostProbeError("host probe has no ownership record")
        self._record["resources"][kind]["uid"] = uid
        self._save()
        options: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": uid},
            "propagationPolicy": "Foreground",
        }
        if force:
            options["gracePeriodSeconds"] = 0
        plural = {"pod": "pods", "configmap": "configmaps"}[kind]
        path = (
            f"/api/v1/namespaces/{quote(self.settings.namespace, safe='')}/"
            f"{plural}/{quote(name, safe='')}"
        )
        result = self._kubectl(
            "delete",
            "--raw",
            path,
            "-f",
            "-",
            input_text=json.dumps(options),
            check=False,
            timeout=60,
        )
        if result.returncode and self._read(kind, name) is not None:
            raise HostProbeError("owned host probe deletion was not acknowledged")

    def _create_resource(self, kind: str, name: str, manifest: dict[str, Any]) -> None:
        if self._record is None:
            raise HostProbeError("host probe has no ownership record")
        entry = self._record["resources"][kind]
        current = self._read(kind, name)
        if current is not None:
            if not entry.get("create_started"):
                raise HostProbeError(
                    "pre-existing host probe resource was not created by this run"
                )
            entry["uid"] = self._owned(kind, current)
            self._save()
            return
        entry["uid"] = None
        entry["create_started"] = True
        self._save()
        try:
            result = self._kubectl(
                "create", "-f", "-", "-o", "json", input_text=json.dumps(manifest)
            )
        except HostProbeError:
            current = self._read(kind, name)
            if current is not None:
                entry["uid"] = self._owned(kind, current)
                self._save()
            raise
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
            raise HostProbeError("host probe create response is malformed")
        entry["uid"] = self._owned(kind, value)
        self._save()

    def _kubectl(
        self,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return run_fixture_command(
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
                input_text=input_text,
                timeout=timeout,
                check=check,
            )
        except RegionalCommandTimeout as exc:
            raise HostProbeTransportError(
                f"host probe kubectl timed out after {exc.timeout}s"
            ) from None
        except RegionalFixtureError as exc:
            raise HostProbeError(str(exc)) from None

    def manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        labels = {
            "app": "gpu-fault-acceptance-host-probe",
            "gpu-fault.io/acceptance-case": self.settings.case_id,
            "gpu-fault.io/acceptance-run": self.settings.run_id,
        }
        annotations = {
            "gpu-fault.io/probe-owner": (
                self._record["owner"] if self._record else "unallocated"
            ),
            "gpu-fault.io/probe-script-sha256": self._scope["script_sha256"],
        }
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.configmap,
                "namespace": self.settings.namespace,
                "labels": labels,
                "annotations": annotations,
            },
            "data": {PROBE_KEY: self._script},
        }
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod,
                "namespace": self.settings.namespace,
                "labels": labels,
                "annotations": {
                    **annotations,
                    "gpu-fault.io/probe-node-uid": (
                        self._record["node_uid"] if self._record else "unallocated"
                    ),
                },
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
        self._acquire()
        node_uid = self._node_uid()
        if self._record is None or self._record.get("closed"):
            if (
                self._read("pod", self.pod) is not None
                or self._read("configmap", self.configmap) is not None
            ):
                raise HostProbeError(
                    "pre-existing host probe resources have no active ownership"
                )
            self._record = {
                "schema_version": 1,
                "scope": self._scope,
                "owner": uuid4().hex,
                "node_uid": node_uid,
                "resources": {
                    kind: {"name": name, "uid": None, "create_started": False}
                    for kind, name in (("pod", self.pod), ("configmap", self.configmap))
                },
                "script_may_exist": False,
                "closed": False,
            }
            self._save()
        if self._record["node_uid"] != node_uid:
            raise HostProbeError("host probe node UID changed")
        configmap, pod = self.manifests()
        self._create_resource("configmap", self.configmap, configmap)
        current = self._read("pod", self.pod)
        if current is not None and current.get("status", {}).get("phase") in {
            "Failed",
            "Succeeded",
        }:
            self._owned("pod", current)
            self._delete_owned("pod", self.pod)
            if not self._wait_pod_gone(POD_DELETE_WAIT_SECONDS):
                raise HostProbeError("terminal host probe Pod did not disappear")
            self._record["resources"]["pod"]["uid"] = None
            self._save()
        self._create_resource("pod", self.pod, pod)
        self._kubectl(
            "wait",
            "--for=condition=Ready",
            f"pod/{self.pod}",
            "--timeout=180s",
            timeout=210,
        )
        current = self._read("pod", self.pod)
        if current is None:
            raise HostProbeError("host probe Pod disappeared during creation")
        self._owned("pod", current)
        if self._node_uid() != self._record["node_uid"]:
            raise HostProbeError("host probe node UID changed during creation")

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self._acquire()
        pod = self._check_target()
        if pod.get("status", {}).get("phase") in {"Failed", "Succeeded"}:
            raise HostProbeTransportError("owned host probe Pod has terminated")
        if self._record is None:
            raise HostProbeError("host probe has no ownership record")
        self._record["script_may_exist"] = True
        self._save()
        command = (
            "reject() { printf '%s\\n' "
            """'{"error":"host probe setup rejected"}'; exit 1; }; """
            'test ! -L "$2" || reject; if test -e "$2"; then '
            'digest="$(sha256sum -- "$2" 2>/dev/null)" || reject; '
            'test "${digest%% *}" = "$3" || reject; fi; '
            'install -m 0700 "$1" "$2" 2>/dev/null || reject; shift 3; '
            'exec chroot /host /opt/gpu-fault/current/venv/bin/python "$@"'
        )
        completed = self._kubectl(
            "exec",
            self.pod,
            "-c",
            "probe",
            "--",
            "/bin/bash",
            "-ceu",
            command,
            "host-probe",
            f"/probe/{PROBE_KEY}",
            f"/host{self.host_script}",
            self._scope["script_sha256"],
            self.host_script,
            *arguments,
            check=False,
            timeout=timeout,
        )
        # Classify explicit rejection before another API read can time out.
        if completed.returncode and any(
            reason in completed.stderr.casefold()
            for reason in ("forbidden", "unauthorized", "must be logged in")
        ):
            raise HostProbeError("host probe request was rejected; output withheld")
        if completed.stdout == "" and completed.returncode in {0, 1}:
            self._check_target()
            raise HostProbeMissingResponseError(
                f"host probe response is missing (exit {completed.returncode})"
            )
        lines = completed.stdout.strip().splitlines()
        if not lines:
            raise HostProbeError(
                f"host probe returned no JSON (exit {completed.returncode})"
            )
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            raise HostProbeError(
                f"host probe returned invalid JSON (exit {completed.returncode})"
            ) from None
        if not isinstance(payload, dict) or not payload:
            raise HostProbeError(
                f"host probe returned an empty or non-object result (exit {completed.returncode})"
            )
        if completed.returncode or "error" in payload:
            # Every shown token is whitelisted by shape: a 16-hex digest of the
            # message, an ``identifier:line`` site, an identifier class name.
            # The message itself (``error``) never leaves the node.
            code = payload.get("error_code")
            if not (isinstance(code, str) and re.fullmatch(r"[a-f0-9]{16}", code)):
                code = None
            site = payload.get("error_site")
            if not (
                isinstance(site, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:[1-9][0-9]*", site)
            ):
                site = None
            kind = payload.get("error_kind")
            diagnostic = ""
            if kind == "collector_env_guard":
                if code:
                    diagnostic += f"; collector-env guard {code}"
                if site:
                    diagnostic += f" at {site}"
            # A probe's generic failure line carries ``error_kind: probe`` with
            # the exception class, digest and site (collector_node_probe.main);
            # older probes print ``{"error": <ExceptionType>}`` and the bare
            # type name is the one clue left. Anything wordier (a message, a
            # path) stays withheld.
            error = (
                payload.get("error_class") if kind == "probe" else payload.get("error")
            )
            if isinstance(error, str) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", error
            ):
                diagnostic += f"; error {error}"
                if kind == "probe":
                    if code:
                        diagnostic += f" {code}"
                    if site:
                        diagnostic += f" at {site}"
            raise HostProbeError(
                f"host probe failed (exit {completed.returncode}{diagnostic}); output withheld"
            )
        self._check_target()
        return payload

    def _check_target(self) -> dict[str, Any]:
        if self._record is None or self._record["closed"]:
            raise HostProbeError("host probe has no active ownership record")
        if self._node_uid() != self._record["node_uid"]:
            raise HostProbeError("host probe node UID changed")
        pod: dict[str, Any] = {}
        for kind, name in (("pod", self.pod), ("configmap", self.configmap)):
            current = self._read(kind, name)
            if current is None:
                raise HostProbeError("host probe resource disappeared")
            self._owned(kind, current)
            if kind == "pod":
                pod = current
        return pod

    def residuals(self) -> dict[str, bool]:
        temporary_lock = self._lock_fd is None
        self._acquire()
        try:
            result = {}
            for kind, name in (("pod", self.pod), ("configmap", self.configmap)):
                result[f"{kind}/{name}"] = self._read(kind, name) is not None
            if self._record is not None:
                result["host_script"] = self._record["script_may_exist"]
                result["creation_unresolved"] = any(
                    item["create_started"] and item["uid"] is None
                    for item in self._record["resources"].values()
                )
            return result
        finally:
            if temporary_lock:
                self._release()

    def _wait_pod_gone(self, seconds: int) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            current = self._read("pod", self.pod)
            if current is None:
                return True
            self._owned("pod", current)
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(2, max(0, deadline - time.monotonic())))

    def _remove_host_script(self) -> None:
        if self._record is None or not self._record["script_may_exist"]:
            return
        # A reboot can terminate the owned Pod. Re-establish the same scoped
        # cleanup channel before removing it; never switch to a replacement node.
        self.create()
        self._check_target()
        self._kubectl(
            "exec",
            self.pod,
            "-c",
            "probe",
            "--",
            "/bin/bash",
            "-ceu",
            'test ! -L "$1"; if test -e "$1"; then '
            'digest="$(sha256sum -- "$1")"; test "${digest%% *}" = "$2"; '
            'rm -- "$1"; fi; test ! -e "$1"; test ! -L "$1"',
            "host-probe-cleanup",
            f"/host{self.host_script}",
            self._scope["script_sha256"],
            timeout=30,
        )
        self._check_target()
        self._record["script_may_exist"] = False
        self._save()

    def cleanup(self) -> dict[str, bool]:
        """Delete only proved-owned resources and retain a failed cleanup journal."""
        self._acquire()
        try:
            if self._record is None:
                residuals = self.residuals()
                if any(residuals.values()):
                    raise HostProbeError(
                        "cannot clean host probe resources without ownership"
                    )
                return residuals
            for kind, entry in self._record["resources"].items():
                current = self._read(kind, entry["name"])
                if current is not None:
                    if not entry["create_started"]:
                        raise HostProbeError(
                            "host probe resource has no creation receipt"
                        )
                    entry["uid"] = self._owned(kind, current)
                    self._save()
                elif entry["create_started"] and entry["uid"] is None:
                    raise HostProbeError("host probe creation remains unresolved")
            self._remove_host_script()
            self._delete_owned("pod", self.pod)
            if not self._wait_pod_gone(POD_DELETE_WAIT_SECONDS):
                self._delete_owned("pod", self.pod, force=True)
                if not self._wait_pod_gone(POD_FORCE_DELETE_WAIT_SECONDS):
                    raise HostProbeError("owned host probe Pod cleanup is incomplete")
            self._delete_owned("configmap", self.configmap)
            deadline = time.monotonic() + 30
            residuals = self.residuals()
            while any(residuals.values()) and time.monotonic() < deadline:
                time.sleep(1)
                residuals = self.residuals()
            if any(residuals.values()):
                raise HostProbeError("host probe cleanup has residual resources")
            self._record["closed"] = True
            self._save()
            return residuals
        finally:
            self._release()
