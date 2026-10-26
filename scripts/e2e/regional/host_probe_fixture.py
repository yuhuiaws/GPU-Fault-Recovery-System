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


#: The interpreter a host probe runs under: the Agent's venv when the node
#: runs the Agent, else the node's own python3 (AL2023 ships it; the probe
#: scripts are stdlib-only).
HOST_PROBE_AGENT_PYTHON = "/opt/gpu-fault/current/venv/bin/python"
HOST_PROBE_SYSTEM_PYTHON = "/usr/bin/python3"


#: Probe error kinds whose ``message`` is one constant sentence (no path, no
#: credential) and the label the diagnostic shows it under.
CONSTANT_MESSAGE_LABELS = {
    "recovery": "recovery refused",
    "nfs_mount": "nfs mount failed",
}
#: The case-directory record of a failed probe call (``probe-error.json``): the
#: probe's own message and structured detail, redacted and bounded, so an
#: operator reads what the probe refused instead of brute-forcing the 16-hex
#: digest the exception text carries (two lanes did, 2026-10-26). The
#: exception text itself keeps its whitelisted shape.
PROBE_ERROR_RECORD = "probe-error.json"
PROBE_ERROR_MESSAGE_CHARS = 2000
PROBE_ERROR_DETAIL_CHARS = 65536
#: Credential-looking tokens are masked before anything is written; a value
#: that follows a secret-ish key, and any long opaque blob.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer|basic)(\s+)[^\s,;}\]\"']+"),
    re.compile(
        r"(?i)\b(token|secret|password|passwd|authorization|api[_-]?key|"
        r"credential)s?\b([=: ]+)[^\s,;}\]\"']+"
    ),
    re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9+/=_-])"),
)


def redact_probe_text(text: str, limit: int = PROBE_ERROR_MESSAGE_CHARS) -> str:
    """``text`` with credential-shaped tokens masked and its length bounded."""

    value = str(text)
    for pattern in _SECRET_PATTERNS[:2]:
        value = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", value)
    value = _SECRET_PATTERNS[2].sub("<redacted>", value)
    return value[:limit]


def redact_probe_detail(value: Any, depth: int = 0) -> Any:
    """``value`` (JSON) with every string redacted; depth- and width-bounded."""

    if depth > 8:
        return "<nested>"
    if isinstance(value, str):
        return redact_probe_text(value, 1024)
    if isinstance(value, dict):
        return {
            redact_probe_text(str(key), 128): redact_probe_detail(item, depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, list):
        return [redact_probe_detail(item, depth + 1) for item in value[:200]]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_probe_text(repr(value), 256)


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
#: Kubernetes mount-propagation modes for the ``/host`` hostPath. ``None`` is
#: rprivate (the probe's own mounts stay inside its Pod mount namespace),
#: ``HostToContainer`` is rslave (host mounts flow in only), ``Bidirectional``
#: is rshared (the probe's mounts also propagate out to the host root mount
#: namespace -- the privileged node-plugin pattern).
MOUNT_PROPAGATION_MODES = ("None", "HostToContainer", "Bidirectional")


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
    #: Mount propagation for the ``/host`` hostPath. The default ``None``
    #: (rprivate) keeps every mount a probe makes inside its own Pod mount
    #: namespace -- read-only probes want exactly that. A probe that creates a
    #: *new* test mount the installed collector must read (the storage probe's
    #: loop/dm filesystem) asks for ``Bidirectional`` so that mount propagates
    #: to the host root mount namespace, where the systemd-managed collector it
    #: restarts -- and the ``systemd-run`` writer units it starts -- live. See
    #: ``MOUNT_PROPAGATION_MODES``.
    host_root_propagation: str = "None"

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
        if self.host_root_propagation not in MOUNT_PROPAGATION_MODES:
            raise ValueError(
                "host probe host-root propagation is not a Kubernetes mode"
            )


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

    def _record_probe_error(
        self,
        payload: dict[str, Any],
        returncode: int,
        code: str | None,
        site: str | None,
        reason: str | None = None,
    ) -> bool:
        """Keep a failed probe call's message and detail, redacted, in the case dir.

        ``probe-error.json`` sits next to ``host-probes/`` (the state directory
        is ``<case dir>/host-probes`` by convention). The message is bounded and
        credential-masked (``redact_probe_text``); the structured detail -- the
        source-replay probe's verification checks, for one -- is redacted
        string by string and bounded. Written only when the probe said more
        than a class name; a write that fails never hides the failure itself.
        """

        message = payload.get("error_message")
        detail = payload.get("error_detail")
        if not isinstance(message, str) and detail is None:
            return False
        command = payload.get("command")
        record = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe": self.pod,
            "node": self.settings.node,
            "case_id": self.settings.case_id,
            "run_id": self.settings.run_id,
            "command": command if isinstance(command, str) else None,
            "returncode": returncode,
            "error_class": payload.get("error_class")
            if isinstance(payload.get("error_class"), str)
            else None,
            "error_code": code,
            "error_site": site,
            "error_reason": reason,
            "message": redact_probe_text(message) if isinstance(message, str) else None,
            "detail": redact_probe_detail(detail),
            "redacted": True,
        }
        if len(json.dumps(record["detail"], default=str)) > PROBE_ERROR_DETAIL_CHARS:
            record["detail"] = {"truncated": True}
        try:
            write_json_atomic(
                self.settings.state_directory.parent / PROBE_ERROR_RECORD, record
            )
        except OSError:
            return False
        return True

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
        host_root_mount: dict[str, Any] = {"name": "host-root", "mountPath": "/host"}
        if self.settings.host_root_propagation != "None":
            # Make the mounts this privileged probe creates under ``/host``
            # propagate back to the host root mount namespace (the CSI
            # node-plugin pattern). The storage probe's loop/dm test filesystem
            # is otherwise confined to this Pod's own mount namespace, so the
            # systemd-run writer units and the collector restarted by the window
            # -- both of which run in the host root namespace -- would statvfs
            # the parent (root) filesystem at the appended path instead.
            host_root_mount["mountPropagation"] = self.settings.host_root_propagation
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
                            host_root_mount,
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
        # The agent's venv is the interpreter on a node that runs the Agent; a
        # node without it (the executor's CPU node on eks-managed, where the
        # ISO-006 block is also aimed) falls back to the system python3, and a
        # node with neither says so in the probe's own JSON shape instead of
        # dying "exit 127" before any JSON (live, ISO-006 attempt 4). The
        # choice is made inside the chroot, where the paths are the node's.
        interpreters = " ".join((HOST_PROBE_AGENT_PYTHON, HOST_PROBE_SYSTEM_PYTHON))
        missing = (
            '{\\"error\\":\\"host probe interpreter missing\\",'
            '\\"error_kind\\":\\"missing_binary\\",\\"missing_binary\\":\\"python3\\"}'
        )
        command = (
            "reject() { printf '%s\\n' "
            """'{"error":"host probe setup rejected"}'; exit 1; }; """
            'test ! -L "$2" || reject; if test -e "$2"; then '
            'digest="$(sha256sum -- "$2" 2>/dev/null)" || reject; '
            'test "${digest%% *}" = "$3" || reject; fi; '
            'install -m 0700 "$1" "$2" 2>/dev/null || reject; shift 3; '
            "exec chroot /host /bin/sh -c '"
            f'for p in {interpreters}; do test -x "$p" && exec "$p" "$@"; done; '
            f'printf "%s\\n" "{missing}"; exit 1'
            '\' sh "$@"'
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
            missing = payload.get("missing_binary")
            if (
                kind == "missing_binary"
                and isinstance(missing, str)
                and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", missing)
            ):
                # The basename of a binary the probe needs and the node lacks:
                # a fixed token (iptables, systemd-run, python3), never a path.
                diagnostic += f"; missing binary on the node: {missing}"
            if kind == "collector_env_guard":
                if code:
                    diagnostic += f"; collector-env guard {code}"
                if site:
                    diagnostic += f" at {site}"
            message = payload.get("message")
            if (
                kind in CONSTANT_MESSAGE_LABELS
                and isinstance(message, str)
                and re.fullmatch(r"[A-Za-z0-9 ,;:'()./_-]{1,160}", message)
            ):
                # The recovery probe's refusals and the shared-storage probe's
                # mount-failure classes are constant sentences; a message in
                # this shape names the condition and carries no path or
                # credential (the mount helper's stderr itself never leaves
                # the node).
                diagnostic += f"; {CONSTANT_MESSAGE_LABELS[kind]}: {message}"
            if kind == "power_identity":
                # A power restore refused after a reboot (exit 78): the digest
                # and site as above, plus the *kinds* of files the operation
                # left on the node -- fixed identifiers, never paths -- so the
                # operator knows what the cleanup-only restore has to close.
                if code:
                    diagnostic += f"; power identity {code}"
                if site:
                    diagnostic += f" at {site}"
                left = payload.get("left_behind")
                if (
                    isinstance(left, list)
                    and left
                    and all(
                        isinstance(item, str) and re.fullmatch(r"[a-z_]{1,40}", item)
                        for item in left
                    )
                ):
                    diagnostic += "; left behind: " + ",".join(left)
            # A probe's generic failure line carries ``error_kind: probe`` with
            # the exception class, digest and site (collector_node_probe.main);
            # older probes print ``{"error": <ExceptionType>}`` and the bare
            # type name is the one clue left. Anything wordier (a message, a
            # path) stays withheld.
            error = (
                payload.get("error_class")
                if kind in {"probe", "power_identity"}
                else payload.get("error")
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
            # A probe may name *why* it refused with a fixed identifier from its
            # own closed vocabulary (the DCGM controller's ``ProbeError(...,
            # "profile_shape")``): one token, like ``failure_class`` beside a
            # redacted command output, never the message. Live 2026-10-26 three
            # 022 variants read ``error ProbeError <digest> at tool_command:800``
            # and nothing else; the digest had to be brute-forced.
            reason = payload.get("error_reason")
            if not (
                kind == "probe"
                and isinstance(reason, str)
                and re.fullmatch(r"[a-z][a-z0-9_]{2,40}", reason)
            ):
                reason = None
            if reason:
                diagnostic += f"; reason {reason}"
            if self._record_probe_error(
                payload, completed.returncode, code, site, reason
            ):
                diagnostic += f"; detail in {PROBE_ERROR_RECORD}"
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
