"""Who holds a GPU when ``nvidia-smi --gpu-reset`` says "In use by another client".

Live 2026-10-05 (gpu-eks-cluster, XID 46 on ip-10-0-41-164): quiesce had
stopped every configured service and container, the device-holder sweep and
``VERIFY_NO_GPU_CLIENTS`` saw no ``/dev/nvidiaN`` holder, and the reset still
failed three times with ``In use by another client`` -- and nothing in the
step, the journal or the result said who. The preflight scan only looks at
``/dev/nvidia<N>`` of the target GPU; a client can hold the driver through
``/dev/nvidiactl``, ``/dev/nvidia-uvm``, ``/dev/nvidia-caps/*``, a device node
that was recreated under it (``(deleted)``), or a kernel module
(``nvidia_peermem``, ``gdrdrv``) that has no process at all.

This module gathers that evidence at the moment of the busy failure and nothing
else: every process with any NVIDIA device node open (pid, ``comm``, the nodes,
its cgroup paths and -- when the cgroup names one -- the Kubernetes container id
and pod uid), ``nvidia-smi --query-compute-apps``, and the NVIDIA kernel
modules with their reference counts. It never reads a command line or an
environment: those can carry tokens into the journal. Bounded, sanitized,
provider-neutral, failure-path only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Every NVIDIA device node a client can hold the driver through, with the
#: ``(deleted)`` suffix a replaced node leaves on an open descriptor.
NVIDIA_DEVICE_HOLDER_PATTERN = re.compile(
    r"^/dev/nvidia(?:\d+|ctl|-uvm|-uvm-tools|-modeset|-caps/[A-Za-z0-9_.-]+)"
    r"(?P<deleted> \(deleted\))?$"
)
KUBERNETES_CONTAINER_ID_PATTERN = re.compile(r"([0-9a-f]{64})")
KUBERNETES_POD_UID_PATTERN = re.compile(
    r"pod([0-9a-f]{8}(?:[-_][0-9a-f]{4}){3}[-_][0-9a-f]{12})"
)
NVIDIA_KERNEL_MODULE_PREFIXES = ("nvidia", "gdrdrv")
GPU_CLIENT_HOLDER_LIMIT = 16
HOLDER_FIELD_LIMIT = 200
CGROUP_PATH_LIMIT = 4


def _printable(value: Any, limit: int = HOLDER_FIELD_LIMIT) -> str:
    text = "".join(char if char.isprintable() else "?" for char in str(value))
    return text[:limit]


def process_cgroup_paths(process: Path) -> list[str]:
    """The cgroup paths of one ``/proc/<pid>``, the shape the quiesce sweep reads."""

    try:
        rows = (process / "cgroup").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    paths = []
    for row in rows:
        parts = row.split(":", 2)
        if len(parts) != 3:
            continue
        path = parts[2].strip()
        if path and path != "/":
            paths.append(path.rstrip("/"))
    return sorted(set(paths))


def kubernetes_attribution(cgroup_paths: Iterable[str]) -> dict[str, Any]:
    """The Kubernetes container id and pod uid a cgroup path carries, if any.

    ``kubepods`` in the path is the fact itself (the process runs in a
    Kubernetes container on this node); the ids follow kubelet's and
    containerd's slice naming and are reported only when they parse.
    """

    paths = list(cgroup_paths)
    kubernetes = any("kubepods" in path for path in paths)
    result: dict[str, Any] = {"kubernetes": kubernetes}
    for path in paths:
        container = KUBERNETES_CONTAINER_ID_PATTERN.search(path)
        pod = KUBERNETES_POD_UID_PATTERN.search(path)
        if container and "container_id" not in result:
            result["container_id"] = container.group(1)
        if pod and "pod_uid" not in result:
            result["pod_uid"] = pod.group(1).replace("_", "-")
    return result


def all_device_holders(proc_root: Path) -> list[dict[str, Any]]:
    """Every process holding any NVIDIA device node, keyed by pid."""

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    own = str(os.getpid())
    holders: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.name.isdigit() or entry.name == own:
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except OSError:
            continue
        devices: set[str] = set()
        deleted = False
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = NVIDIA_DEVICE_HOLDER_PATTERN.fullmatch(target)
            if match is None:
                continue
            deleted = deleted or bool(match.group("deleted"))
            devices.add(target)
        if not devices:
            continue
        try:
            name = (entry / "comm").read_text().strip()
        except (OSError, UnicodeError):
            name = "unknown"
        cgroups = process_cgroup_paths(entry)
        holders.append(
            {
                "pid": entry.name,
                "process_name": _printable(name, 64),
                "devices": ",".join(sorted(devices)),
                "deleted_device_node": deleted,
                "cgroup_paths": [
                    _printable(path) for path in cgroups[:CGROUP_PATH_LIMIT]
                ],
                **kubernetes_attribution(cgroups),
            }
        )
    return sorted(holders, key=lambda item: int(item["pid"]))


def nvidia_kernel_module_clients(proc_root: Path) -> list[dict[str, Any]]:
    """NVIDIA-related kernel modules with their reference counts.

    A module that holds the device (``nvidia_peermem`` for GPUDirect RDMA over
    EFA, ``gdrdrv``) is a client no process scan can show; the refcount and
    the ``used_by`` list are the only evidence of it.
    """

    try:
        rows = (proc_root / "modules").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    modules = []
    for row in rows:
        fields = row.split()
        if len(fields) < 3 or not fields[0].startswith(NVIDIA_KERNEL_MODULE_PREFIXES):
            continue
        try:
            refcount = int(fields[2])
        except ValueError:
            continue
        used_by = [
            _printable(item, 64)
            for item in fields[3].split(",")
            if len(fields) > 3 and item and item != "-"
        ]
        modules.append(
            {
                "module": _printable(fields[0], 64),
                "refcount": refcount,
                "used_by": used_by,
            }
        )
    return modules


#: The display stack: ``nvidia_drm`` registers a DRM device per GPU and, with
#: modeset on, holds a client on it that ``nvidia-smi --gpu-reset`` reports as
#: "In use by another client" although no process has any device node open
#: (live, gpu-eks-cluster XID 46: ``nvidia_modeset refcount=1 used_by
#: nvidia_drm``, ``nvidia refcount=2``, no holder anywhere). Unload order: the
#: DRM module first, then modeset once nothing uses it. Never ``nvidia_uvm``,
#: ``nvidia_peermem`` or ``gdrdrv``: those are compute / RDMA clients.
IDLE_DISPLAY_MODULES = ("nvidia_drm", "nvidia_modeset")


def idle_display_modules(proc_root: Path) -> list[str]:
    """The display-stack modules that are loaded and used by nothing but each
    other, in unload order: ``nvidia_drm`` when its refcount is 0, then
    ``nvidia_modeset`` when nothing else (``nvidia_drm`` aside) uses it. Empty
    when a process holds the DRM device (an X server, a container with
    ``/dev/dri``) -- that holder is in ``device_holders`` and is not unloaded
    from under it."""

    modules = {item["module"]: item for item in nvidia_kernel_module_clients(proc_root)}
    releasable: list[str] = []
    drm = modules.get("nvidia_drm")
    if drm is not None and drm["refcount"] == 0:
        releasable.append("nvidia_drm")
    modeset = modules.get("nvidia_modeset")
    if modeset is not None:
        users = set(modeset["used_by"]) - ({"nvidia_drm"} if releasable else set())
        if (
            modeset["refcount"]
            - (1 if releasable and "nvidia_drm" in modeset["used_by"] else 0)
            == 0
            and not users
        ):
            releasable.append("nvidia_modeset")
    return releasable


def display_module_state(proc_root: Path) -> list[dict[str, Any]]:
    """The display-stack modules still loaded, with refcount and users -- the
    evidence of what a release left in place."""

    return [
        item
        for item in nvidia_kernel_module_clients(proc_root)
        if item["module"] in IDLE_DISPLAY_MODULES
    ]


def gpu_client_holder_evidence(
    *,
    proc_root: Path,
    gpu_uuids: Sequence[str],
    compute_clients: Callable[[], list[dict[str, str]]],
    device_paths: Callable[[], Mapping[str, str]],
) -> dict[str, Any]:
    """The bounded evidence a busy reset carries; it never raises."""

    evidence: dict[str, Any] = {"target_gpu_uuids": sorted(set(gpu_uuids))}
    try:
        evidence["target_device_paths"] = sorted(
            path for uuid, path in device_paths().items() if uuid in set(gpu_uuids)
        )
    except Exception as exc:
        evidence["target_device_paths_error"] = type(exc).__name__
    try:
        clients = compute_clients()
        evidence["compute_clients"] = [
            {key: _printable(value, 64) for key, value in item.items()}
            for item in clients[:GPU_CLIENT_HOLDER_LIMIT]
        ]
        evidence["compute_client_count"] = len(clients)
    except Exception as exc:
        evidence["compute_clients_error"] = type(exc).__name__
    try:
        holders = all_device_holders(proc_root)
        evidence["device_holders"] = holders[:GPU_CLIENT_HOLDER_LIMIT]
        evidence["device_holder_count"] = len(holders)
    except Exception as exc:
        evidence["device_holders_error"] = type(exc).__name__
    try:
        evidence["kernel_modules"] = nvidia_kernel_module_clients(proc_root)
    except Exception as exc:
        evidence["kernel_modules_error"] = type(exc).__name__
    return evidence


def gpu_client_holder_summary(evidence: Mapping[str, Any]) -> str:
    """One line for the step's error: who held the device, or that nobody visible did."""

    holders = evidence.get("device_holders") or []
    if not holders:
        count = evidence.get("compute_client_count", 0)
        modules = [
            f"{item['module']}(refcount={item['refcount']})"
            for item in evidence.get("kernel_modules") or []
            if item.get("refcount")
        ]
        return (
            "no process holds an NVIDIA device node in /proc; "
            f"compute apps: {count}; kernel modules in use: "
            + (", ".join(modules) if modules else "none")
        )
    parts = []
    for item in holders[:8]:
        text = f"{item['pid']}:{item['process_name']}:{item['devices']}"
        if item.get("container_id"):
            text += f" (container {item['container_id'][:12]})"
        elif item.get("kubernetes"):
            text += " (kubernetes)"
        parts.append(text)
    omitted = len(holders) - len(parts)
    summary = "device holders: " + ", ".join(parts)
    return f"{summary} (+{omitted} more)" if omitted > 0 else summary
