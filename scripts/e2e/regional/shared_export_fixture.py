"""A disposable NFSv4 test export for GF-GENERAL-LIVE-005, served from a Pod.

This site mounts no independent NFS/Lustre export a test may fill, hang or
disconnect, so the shared-storage variants bring their own: one Pod per export
(``gfrs-export-<digest>-<suffix>``) running a *userspace* NFSv4 server
(NFS-Ganesha, VFS FSAL) over a memory-backed ``emptyDir`` whose ``sizeLimit`` is
the export's whole capacity -- so "full" is a few hundred MiB on tmpfs, never a
node disk -- plus the ConfigMap holding its ``ganesha.conf``. A client node
mounts ``<pod ip>:/a`` with ``probes/general_live_shared_probe.py``; the host
collector then discovers the ``nfs4`` mount from its own mount table
(``_shared_filesystems``) -- the product has no shared-filesystem setting.

Why userspace: a hung export is produced by ``SIGSTOP`` on the server process
(``pause``) -- the kernel still accepts the client's TCP segments, nothing
answers, every ``statvfs`` blocks until the client's ``soft`` timeout -- and
``SIGCONT`` (``resume``) restores it with the same file handles, so recovery
needs no remount. A kernel ``nfsd`` cannot be paused that way. Ganesha runs as
a child of the container's ``sh`` (not PID 1: a PID namespace's init ignores
SIGSTOP sent from inside it), so ``kubectl exec`` can stop and continue it.

Bounded and gfrs-only: a bare Pod (``restartPolicy: Never``) with
``activeDeadlineSeconds`` as the cluster-side deadman, labelled with the
acceptance case/run, scheduled away from the client nodes (or pinned with
``node``), deleted by UID precondition. The runner's LIFO cleanup unmounts the
clients first (the probe's deadman does the same), then deletes the Pod.

Live prerequisites (the coordinator arranges them): the image (``--export-image``
or ``GPU_FAULT_GENERAL_LIVE_EXPORT_IMAGE``, an immutable digest) must ship
``ganesha.nfsd`` and ``/bin/sh``; the client nodes' AMI needs ``nfs-utils``
(``mount.nfs4``) and the ``nfsv4`` module (loaded under KERNEL_MODULE_LOAD when
missing); TCP/2049 from the node to the Pod IP must be allowed (in the GPU
cluster the VPC CNI gives Pods node ENI addresses, so the node security group's
self-rule usually covers it; the CPU cluster needs VPC reachability plus an
explicit rule); and the kubelet must honour ``sizeLimit`` on memory-backed
``emptyDir`` (``SizeMemoryBackedVolumes``, on by default since 1.22), which
the first client ``statvfs`` verifies.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalFixtureError

APP_LABEL = "gfrs-test-export"
CASE_LABEL = "gpu-fault.io/acceptance-case"
RUN_LABEL = "gpu-fault.io/acceptance-run"
OWNER_LABEL = "gpu-fault.io/acceptance-owner"
CONTAINER = "export"
NFS_PORT = 2049
EXPORT_PATH = "/export/a"
PSEUDO_PATH = "/a"
PID_FILE = "/run/gfrs/ganesha.pid"
DEFAULT_SIZE_MIB = 256
MIN_SIZE_MIB = 64
MAX_SIZE_MIB = 2048
#: ``SIGSTOP``/``SIGCONT`` the ganesha child and read its state back.
PAUSE_SCRIPT = (
    f'pid="$(cat {PID_FILE})" && kill -STOP "$pid" && sleep 0.3 && '
    'grep -E "^State:" "/proc/$pid/status"'
)
RESUME_SCRIPT = (
    f'pid="$(cat {PID_FILE})" && kill -CONT "$pid" && sleep 0.3 && '
    'grep -E "^State:" "/proc/$pid/status"'
)
#: The container's entry: export directory, writable ganesha dirs, ganesha as a
#: *child* of this shell (pausable from an exec), pid recorded, then wait.
ENTRY_SCRIPT = f"""set -eu
mkdir -p {EXPORT_PATH} /var/run/ganesha /var/lib/nfs/ganesha /run/gfrs
chmod 1777 {EXPORT_PATH}
GANESHA="$(command -v ganesha.nfsd || echo /usr/bin/ganesha.nfsd)"
"$GANESHA" -F -L /dev/stderr -N NIV_EVENT -f /etc/ganesha/ganesha.conf &
echo $! > {PID_FILE}
wait
"""


def ganesha_config() -> str:
    """One NFSv4-only, TCP-only, root-squashed read-write VFS export.

    ``Root_Squash`` with a ``1777`` export directory lets the client (root on
    the node) write as ``nobody`` -- so a server-side ``chmod 000`` denies its
    I/O while ``statvfs`` keeps answering (``io_only_failure``). The attribute
    cache expires in 1 s so such a change is seen at once; ``Graceless`` lets a
    resumed/restarted server answer without a grace period.
    """

    return f"""NFS_CORE_PARAM {{
    Protocols = 4;
    NFS_Port = {NFS_PORT};
    Enable_NLM = false;
    Enable_RQUOTA = false;
    mount_path_pseudo = true;
}}
NFSV4 {{
    Graceless = true;
    Lease_Lifetime = 20;
    Grace_Period = 20;
    Allow_Numeric_Owners = true;
    Only_Numeric_Owners = true;
}}
MDCACHE {{
    Attr_Expiration_Time = 1;
}}
EXPORT {{
    Export_Id = 1;
    Path = {EXPORT_PATH};
    Pseudo = {PSEUDO_PATH};
    Access_Type = RW;
    Squash = Root_Squash;
    Anonymous_Uid = 65534;
    Anonymous_Gid = 65534;
    Protocols = 4;
    Transports = TCP;
    SecType = sys;
    Disable_ACL = true;
    FSAL {{
        Name = VFS;
    }}
}}
LOG {{
    Default_Log_Level = EVENT;
}}
"""


@dataclass(frozen=True)
class ExportServerSettings:
    """One test export server: image, identity, placement, capacity."""

    image: str
    case_id: str
    run_id: str
    suffix: str = "a"
    #: Client nodes the Pod must not land on (node anti-affinity) ...
    avoid_nodes: tuple[str, ...] = ()
    #: ... unless pinned to one node explicitly.
    node: str = ""
    size_mib: int = DEFAULT_SIZE_MIB
    active_deadline_seconds: int = 3600
    plane: str = "gpu"

    def __post_init__(self) -> None:
        if "@sha256:" not in self.image:
            raise RegionalFixtureError(
                "export server image must use an immutable digest"
            )
        if not MIN_SIZE_MIB <= self.size_mib <= MAX_SIZE_MIB:
            raise RegionalFixtureError(
                f"export size must be {MIN_SIZE_MIB}..{MAX_SIZE_MIB} MiB"
            )
        if (
            not self.suffix.isalpha()
            or not self.suffix.islower()
            or len(self.suffix) > 8
        ):
            raise RegionalFixtureError("export suffix must be 1-8 lower-case letters")
        if self.node and self.node in self.avoid_nodes:
            raise RegionalFixtureError(
                "the export server cannot be pinned to a client node"
            )
        if self.plane not in {"gpu", "cpu"}:
            raise RegionalFixtureError("export plane must be gpu or cpu")
        if not 300 <= self.active_deadline_seconds <= 4 * 3600:
            raise RegionalFixtureError("export deadline must be 300..14400 s")


class TestExportServer:
    """Create, drive and delete one test export Pod (+ its ConfigMap).

    ``regional`` is a ``RegionalLiveFixture`` (its ``kubectl(plane, ...)`` and
    ``settings.namespace`` are all that is used); ``journal`` records the
    intended and created objects so a cleanup after a crash deletes exactly
    them.
    """

    def __init__(
        self,
        regional: Any,
        settings: ExportServerSettings,
        *,
        journal: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.regional = regional
        self.settings = settings
        self.journal = journal
        self.clock = clock
        self.sleep = sleep
        digest = hashlib.sha256(
            f"{settings.case_id}\0{settings.run_id}".encode()
        ).hexdigest()[:10]
        self.name = f"gfrs-export-{digest}-{settings.suffix}"
        self.owner = hashlib.sha256(
            f"{self.name}\0{settings.image}".encode()
        ).hexdigest()[:16]
        self.created: dict[str, str | None] = {}
        self.ready: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # manifests
    # ------------------------------------------------------------------ #
    @property
    def namespace(self) -> str:
        return str(self.regional.settings.namespace)

    def labels(self) -> dict[str, str]:
        return {
            "app": APP_LABEL,
            CASE_LABEL: self.settings.case_id,
            RUN_LABEL: self.settings.run_id,
            OWNER_LABEL: self.owner,
        }

    def manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        settings = self.settings
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels(),
            },
            "data": {"ganesha.conf": ganesha_config()},
        }
        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": settings.active_deadline_seconds,
            "terminationGracePeriodSeconds": 5,
            "tolerations": [{"operator": "Exists"}],
            "containers": [
                {
                    "name": CONTAINER,
                    "image": settings.image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/sh", "-c", ENTRY_SCRIPT],
                    "ports": [
                        {"name": "nfs", "containerPort": NFS_PORT, "protocol": "TCP"}
                    ],
                    "readinessProbe": {
                        "tcpSocket": {"port": NFS_PORT},
                        "periodSeconds": 2,
                        "failureThreshold": 3,
                    },
                    "securityContext": {
                        "runAsUser": 0,
                        "privileged": False,
                        "allowPrivilegeEscalation": False,
                        # open_by_handle_at (VFS FSAL) and the rlimit raise.
                        "capabilities": {"add": ["DAC_READ_SEARCH", "SYS_RESOURCE"]},
                    },
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {
                            "cpu": "1",
                            # The tmpfs export counts against the container.
                            "memory": f"{settings.size_mib + 512}Mi",
                        },
                    },
                    "volumeMounts": [
                        {"name": "export", "mountPath": "/export"},
                        {
                            "name": "ganesha-config",
                            "mountPath": "/etc/ganesha",
                            "readOnly": True,
                        },
                        {"name": "run", "mountPath": "/run"},
                        {"name": "var-run-ganesha", "mountPath": "/var/run/ganesha"},
                        {
                            "name": "var-lib-ganesha",
                            "mountPath": "/var/lib/nfs/ganesha",
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "export",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": f"{settings.size_mib}Mi",
                    },
                },
                {"name": "ganesha-config", "configMap": {"name": self.name}},
                {"name": "run", "emptyDir": {}},
                {"name": "var-run-ganesha", "emptyDir": {}},
                {"name": "var-lib-ganesha", "emptyDir": {}},
            ],
        }
        if settings.node:
            spec["nodeName"] = settings.node
        elif settings.avoid_nodes:
            spec["affinity"] = {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "kubernetes.io/hostname",
                                        "operator": "NotIn",
                                        "values": list(settings.avoid_nodes),
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels(),
            },
            "spec": spec,
        }
        return configmap, pod

    # ------------------------------------------------------------------ #
    # kubectl
    # ------------------------------------------------------------------ #
    def _kubectl(self, *arguments: str, **options: Any) -> str:
        return str(self.regional.kubectl(self.settings.plane, *arguments, **options))

    def _journal(self, **extra: Any) -> None:
        if self.journal is None:
            return
        write_json_atomic(
            self.journal,
            {
                "name": self.name,
                "namespace": self.namespace,
                "owner": self.owner,
                "created": dict(self.created),
                "ready": self.ready,
                **extra,
            },
        )

    def read(self, kind: str) -> dict[str, Any] | None:
        raw = self._kubectl(
            "get", kind, self.name, "--ignore-not-found", "-o", "json", timeout=60
        )
        if not raw.strip():
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RegionalFixtureError("export server query is not an object")
        metadata = value.get("metadata") or {}
        if (
            metadata.get("name") != self.name
            or metadata.get("namespace") != self.namespace
        ):
            raise RegionalFixtureError("export server query returned another identity")
        return value

    def _owned(self, document: dict[str, Any], kind: str) -> str:
        metadata = document.get("metadata") or {}
        if (metadata.get("labels") or {}).get(OWNER_LABEL) != self.owner:
            raise RegionalFixtureError(
                f"a foreign {kind} {self.name} occupies the test export's name"
            )
        return str(metadata.get("uid") or "")

    def create(self) -> dict[str, Any]:
        """Create the ConfigMap and the Pod; refuse a foreign occupant."""

        configmap, pod = self.manifests()
        for kind, manifest in (("configmap", configmap), ("pod", pod)):
            current = self.read(kind)
            if current is not None:
                self.created[kind] = self._owned(current, kind)
                continue
            self.created[kind] = None
            self._journal(intent=kind)
            created = json.loads(
                self._kubectl(
                    "create", "-f", "-", "-o", "json", input_text=json.dumps(manifest)
                )
            )
            self.created[kind] = self._owned(created, kind)
            self._journal()
        return {"name": self.name, "namespace": self.namespace, **self.created}

    def pod(self) -> dict[str, Any] | None:
        return self.read("pod")

    def wait_ready(
        self, *, timeout_seconds: float, poll_seconds: float = 5.0
    ) -> dict[str, Any]:
        """Running and Ready: the Pod IP (the mount server), node and UID."""

        started = self.clock()
        last: dict[str, Any] | None = None
        while True:
            last = self.pod()
            status = (last or {}).get("status") or {}
            phase = status.get("phase")
            if phase in {"Failed", "Succeeded"}:
                raise RegionalFixtureError(
                    f"test export {self.name} ended {phase}: {self.logs()[-600:]}"
                )
            ready = any(
                item.get("type") == "Ready" and item.get("status") == "True"
                for item in status.get("conditions") or []
            )
            if phase == "Running" and ready and status.get("podIP"):
                self.ready = {
                    "ip": str(status["podIP"]),
                    "node": str((last or {}).get("spec", {}).get("nodeName") or ""),
                    "uid": str((last or {}).get("metadata", {}).get("uid") or ""),
                    "export": PSEUDO_PATH,
                    "ready_at": status.get("startTime"),
                }
                self._journal()
                return dict(self.ready)
            if self.clock() - started >= timeout_seconds:
                raise RegionalFixtureError(
                    f"test export {self.name} is not Ready after {timeout_seconds:.0f} s "
                    f"(phase {phase}): {self.logs()[-600:]}"
                )
            self.sleep(poll_seconds)

    def logs(self, tail: int = 40) -> str:
        try:
            return self._kubectl(
                "logs",
                self.name,
                "-c",
                CONTAINER,
                f"--tail={tail}",
                check=False,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return f"(logs unavailable: {type(exc).__name__})"

    def exec(self, script: str, *, timeout: int = 60) -> str:
        return self._kubectl(
            "exec",
            self.name,
            "-c",
            CONTAINER,
            "--",
            "/bin/sh",
            "-c",
            script,
            timeout=timeout,
        )

    def pause(self) -> dict[str, Any]:
        """``SIGSTOP`` the server: every client call now hangs (stat timeout)."""

        state = self.exec(PAUSE_SCRIPT)
        if "T" not in state.split(":", 1)[-1].split("(")[0]:
            raise RegionalFixtureError(
                f"test export {self.name} did not stop: {state!r}"
            )
        return {"paused_at": _now(), "state": state.strip()}

    def resume(self) -> dict[str, Any]:
        state = self.exec(RESUME_SCRIPT)
        if "T" in state.split(":", 1)[-1].split("(")[0]:
            raise RegionalFixtureError(
                f"test export {self.name} is still stopped: {state!r}"
            )
        return {"resumed_at": _now(), "state": state.strip()}

    def set_export_mode(self, mode: str) -> dict[str, Any]:
        """``chmod`` the export root: ``000`` denies the squashed client's I/O
        while ``statvfs`` keeps answering; ``1777`` restores it."""

        if mode not in {"000", "1777"}:
            raise RegionalFixtureError("export mode must be 000 or 1777")
        listing = self.exec(f"chmod {mode} {EXPORT_PATH} && stat -c %a {EXPORT_PATH}")
        return {"mode": mode, "stat": listing.strip()}

    def delete(self) -> dict[str, Any]:
        """Delete the Pod and the ConfigMap this fixture created (UID-bound)."""

        results: dict[str, Any] = {}
        for kind, api, plural in (
            ("pod", "api/v1", "pods"),
            ("configmap", "api/v1", "configmaps"),
        ):
            current = self.read(kind)
            if current is None:
                results[kind] = "absent"
                continue
            uid = self._owned(current, kind)
            recorded = self.created.get(kind)
            if recorded and recorded != uid:
                raise RegionalFixtureError(
                    f"test export {kind} was replaced; not deleting"
                )
            options: dict[str, Any] = {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": uid},
                "propagationPolicy": "Foreground",
            }
            if kind == "pod":
                options["gracePeriodSeconds"] = 5
            namespace = quote(self.namespace, safe="")
            self._kubectl(
                "delete",
                "--raw",
                f"/{api}/namespaces/{namespace}/{plural}/{quote(self.name, safe='')}",
                "-f",
                "-",
                input_text=json.dumps(options),
                timeout=120,
            )
            deadline = self.clock() + 180
            while self.read(kind) is not None:
                if self.clock() >= deadline:
                    raise RegionalFixtureError(
                        f"test export {kind} deletion not confirmed"
                    )
                self.sleep(2)
            results[kind] = "deleted"
        self.created = {}
        self._journal(deleted=results)
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "plane": self.settings.plane,
            "image": self.settings.image,
            "size_mib": self.settings.size_mib,
            "export": PSEUDO_PATH,
            "owner": self.owner,
            "created": dict(self.created),
            "ready": dict(self.ready),
        }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "APP_LABEL",
    "DEFAULT_SIZE_MIB",
    "ENTRY_SCRIPT",
    "EXPORT_PATH",
    "ExportServerSettings",
    "NFS_PORT",
    "PSEUDO_PATH",
    "TestExportServer",
    "ganesha_config",
]
