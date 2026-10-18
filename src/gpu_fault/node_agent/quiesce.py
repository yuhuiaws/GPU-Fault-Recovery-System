from __future__ import annotations

from gpu_fault.node_agent.late_ownership import require_physical_ownership

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.node_agent.operations.holders import (
    fabric_manager_process,
    kubernetes_attribution,
    process_cgroup_paths,
)
from gpu_fault.node_agent.quiesce_persistence import (
    TargetPersistence,
    restored_daemon,
    restored_modes,
)
from gpu_fault.node_agent.common import (
    DEFAULT_DEVICE_SWEEP_PROCESSES,
    DEFAULT_QUIESCE_CONTAINERS,
    DEFAULT_QUIESCE_PROCESSES,
    DEFAULT_QUIESCE_SERVICES,
    FABRIC_PROCESSES,
    NVIDIA_GPU_DEVICE_PATTERN,
    PROCESS_NAME_PATTERN,
    SYSTEMD_NAME_PATTERN,
    unit_start_timeout_seconds,
)
from gpu_fault.node_agent.protocol import (
    QUIESCE_FABRIC_MANAGER_KEEP,
    QUIESCE_FABRIC_MANAGER_PARAMETER,
)

LOGGER = logging.getLogger(__name__)


def restore_gpu_services() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore services from a GPU quiesce fail-safe state"
    )
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    state_path = Path(args.state_file)
    result = GpuServiceQuiesceManager.restore_state_file(state_path)
    LOGGER.info("GPU service restore result: %s", result)


BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def _reset_claim_refusal(issued: dict[str, Any], command_id: str | None) -> str:
    """Why a second reset in one quiesce window is refused, naming the claimant.

    Never states that the reset ran: the claim is taken before nvidia-smi is
    spawned, so only the claiming command's own result says whether the GPU was
    reset. An operator who read "already reset" here would reboot or replace a
    node that may never have been touched. When the claimant is this very
    command_id -- a resubmit after an attempt whose outcome is unknown -- the
    text names the earlier attempt instead of quoting the caller its own id.
    """

    owner = issued.get("command_id")
    if owner is not None and owner == command_id:
        attempt = issued.get("attempt")
        claimant = "an earlier attempt of this command" + (
            f" (see attempt {attempt})" if attempt is not None else ""
        )
    else:
        claimant = owner or "an earlier command"
    return (
        "the single GPU reset this quiesce window allows was already claimed by "
        f"{claimant} at {issued.get('issued_at')}; read that command's result to "
        "learn whether the reset ran, and restore the GPU services and quiesce "
        "again before any further reset"
    )


def read_boot_id(path: Path = BOOT_ID_PATH) -> str:
    """The kernel boot id, or "" where it cannot be read (tests, containers)."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _arm_quiesce_failsafe(
    manager: GpuServiceQuiesceManager, timer_unit: str, state_path: Path
) -> None:
    require_physical_ownership()
    manager._run(
        [
            "systemd-run",
            f"--unit={timer_unit}",
            f"--on-active={manager.failsafe_seconds}s",
            f"--on-unit-active={manager.retry_seconds}s",
            "--timer-property=AccuracySec=1s",
            "--property=Type=oneshot",
            manager.restore_command,
            "--state-file",
            str(state_path),
        ],
        check=True,
    )


#: Why a window keeps Fabric Manager running (``kept_services``).
FABRIC_MANAGER_KEPT_REASON = (
    "the window's only GPU-state change is a per-GPU reset: from Hopper on a GPU "
    "resets individually while Fabric Manager runs, and stopping Fabric Manager "
    "would lose the fabric state of every other GPU (NVIDIA Fabric Manager user "
    "guide), so no quiesce, restore or fail-safe stops or starts it"
)


def fabric_manager_kept(parameters: Mapping[str, Any]) -> bool:
    """Whether a QUIESCE command keeps Fabric Manager running.

    Absent means stop it, as before; the documented value ``keep`` keeps it.
    Anything else is refused before the window opens, so a plan this Agent does
    not understand never half-applies.
    """

    value = parameters.get(QUIESCE_FABRIC_MANAGER_PARAMETER)
    if value is None:
        return False
    if value != QUIESCE_FABRIC_MANAGER_KEEP:
        raise RuntimeError(
            f"QUIESCE_GPU_SERVICES parameter {QUIESCE_FABRIC_MANAGER_PARAMETER} "
            f"must be {QUIESCE_FABRIC_MANAGER_KEEP!r} or absent, got {value!r}"
        )
    return True


def _service_inventory(
    manager: GpuServiceQuiesceManager,
    keep_fabric_manager: bool = False,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """``(active, inactive, kept)`` among the configured, unprotected services.

    ``active`` is what the quiesce stops (and the restore starts); ``inactive``
    is what it found nothing to stop -- a unit that is not running or not
    installed on this image. Both travel in the receipt and the state file:
    live (gpu-eks-cluster, XID 46) the reset failed busy and nothing in the
    record said whether ``nvidia-persistenced`` / ``nvidia-fabricmanager`` had
    been stopped or were simply not there to stop. ``kept`` is a running Fabric
    Manager the window keeps (``keep_fabric_manager``): it is in neither list,
    so no restore -- the step, the fail-safe timer, the boot reconcile -- starts
    a second instance that would know only the GPUs this window reset.
    """

    active: list[str] = []
    inactive: list[str] = []
    kept: list[dict[str, str]] = []
    for service in manager.services:
        if service in manager.protected_services:
            continue
        if not manager._service_active(service):
            inactive.append(service)
        elif keep_fabric_manager and service == FABRIC_MANAGER_UNIT:
            kept.append({"service": service, "reason": FABRIC_MANAGER_KEPT_REASON})
        else:
            active.append(service)
    return active, inactive, kept


def _container_filter(selector: str) -> str:
    namespace, name = selector.split("/", 1)
    return (
        'labels."io.kubernetes.pod.namespace"=='
        f'{namespace},labels."io.kubernetes.container.name"=={name}'
    )


def _cgroup_matches(holder_paths: list[str], workload_cgroup_paths: set[str]) -> bool:
    for holder_path in holder_paths:
        normalized_holder = holder_path.rstrip("/")
        for workload_path in workload_cgroup_paths:
            normalized_workload = workload_path.rstrip("/")
            if not normalized_workload or normalized_workload == "/":
                continue
            if normalized_holder == normalized_workload or normalized_holder.startswith(
                normalized_workload + "/"
            ):
                return True
    return False


class GpuServiceQuiesceManager:
    """Quiesces host GPU services with an independent systemd restore timer.

    The timer is a transient unit a reboot takes with it while the
    ``quiesce-*.json`` state file survives; ``reconcile_after_boot`` restores
    such a file on a later boot, so the node neither reports quiesce residue
    forever nor passes ``assert_quiesced`` on a boot that never quiesced.
    """

    def __init__(
        self,
        *,
        state_dir: str,
        services: tuple[str, ...] = DEFAULT_QUIESCE_SERVICES,
        processes: tuple[str, ...] = DEFAULT_QUIESCE_PROCESSES,
        containers: tuple[str, ...] = DEFAULT_QUIESCE_CONTAINERS,
        failsafe_seconds: int = 420,
        retry_seconds: int = 60,
        settle_seconds: float = 2,
        restore_settle_seconds: float = 30,
        container_stop_timeout_seconds: int = 30,
        container_restore_timeout_seconds: int = 180,
        device_sweep_timeout_seconds: int = 20,
        device_sweep_processes: tuple[str, ...] = (DEFAULT_DEVICE_SWEEP_PROCESSES),
        sweep_kubernetes_device_holders: bool = False,
        protected_services: tuple[str, ...] = (),
        proc_root: str = "/proc",
        restore_command: str = (
            "/opt/gpu-fault/current/venv/bin/gpu-fault-restore-gpu-services"
        ),
        runner: Callable[..., subprocess.CompletedProcess] = (subprocess.run),
        sleeper: Callable[[float], None] = time.sleep,
        boot_id_reader: Callable[[], str] = read_boot_id,
    ) -> None:
        if not 30 <= failsafe_seconds <= 3600:
            raise ValueError("quiesce fail-safe seconds must be between 30 and 3600")
        if not 10 <= retry_seconds <= 600:
            raise ValueError("quiesce retry seconds must be between 10 and 600")
        if not 0 <= settle_seconds <= 60:
            raise ValueError("quiesce settle seconds must be between 0 and 60")
        if not 0 <= restore_settle_seconds <= 120:
            raise ValueError("restore settle seconds must be between 0 and 120")
        if not services:
            raise ValueError("at least one quiesce service is required")
        if len(set(services)) != len(services):
            raise ValueError("quiesce services contain duplicates")
        if len(set(processes)) != len(processes):
            raise ValueError("quiesce processes contain duplicates")
        if len(set(containers)) != len(containers):
            raise ValueError("quiesce containers contain duplicates")
        if len(set(device_sweep_processes)) != len(device_sweep_processes):
            raise ValueError("device sweep processes contain duplicates")
        for service in services:
            if not SYSTEMD_NAME_PATTERN.fullmatch(service):
                raise ValueError(f"invalid quiesce service name: {service}")
        for process in processes:
            if not PROCESS_NAME_PATTERN.fullmatch(process):
                raise ValueError(f"invalid quiesce process name: {process}")
        if FABRIC_PROCESSES.intersection((*processes, *device_sweep_processes)):
            raise ValueError("the quiesce never signals Fabric Manager or NVLSM")
        for process in device_sweep_processes:
            if not PROCESS_NAME_PATTERN.fullmatch(process):
                raise ValueError(f"invalid device sweep process name: {process}")
        for container in containers:
            parts = container.split("/")
            if len(parts) != 2 or not all(
                SYSTEMD_NAME_PATTERN.fullmatch(part) for part in parts
            ):
                raise ValueError(f"invalid quiesce container selector: {container}")
        if not 5 <= container_stop_timeout_seconds <= 120:
            raise ValueError("container stop timeout must be between 5 and 120 seconds")
        if not 30 <= container_restore_timeout_seconds <= 900:
            raise ValueError(
                "container restore timeout must be between 30 and 900 seconds"
            )
        if not 0 <= device_sweep_timeout_seconds <= 120:
            raise ValueError("device sweep timeout must be between 0 and 120 seconds")
        if not Path(restore_command).is_absolute():
            raise ValueError("restore command must be an absolute path")
        self.state_dir = Path(state_dir)
        self.services = services
        self.processes = processes
        self.containers = containers
        self.failsafe_seconds = failsafe_seconds
        self.retry_seconds = retry_seconds
        self.settle_seconds = settle_seconds
        self.restore_settle_seconds = restore_settle_seconds
        self.container_stop_timeout_seconds = container_stop_timeout_seconds
        self.container_restore_timeout_seconds = container_restore_timeout_seconds
        self.device_sweep_timeout_seconds = device_sweep_timeout_seconds
        self.device_sweep_processes = frozenset(device_sweep_processes)
        # eks-managed: also stop a Kubernetes container (customer DCGM exporter,
        # GPU feature discovery) still holding the target GPU; kubelet is down
        # and restarts it on restore. Off (HyperPod), VERIFY fails closed on it.
        self.sweep_kubernetes_device_holders = sweep_kubernetes_device_holders
        # eks-managed: kubelet and containerd stay up (no HyperPod health agent
        # carries a NotReady node; the GPU clients are containers the runtime
        # stops). Not in the config digest, so the configured list stays HyperPod's.
        self.protected_services = tuple(protected_services)
        self.proc_root = Path(proc_root)
        self.restore_command = restore_command
        self.runner = runner
        self.sleeper = sleeper
        self.boot_id_reader = boot_id_reader

    @staticmethod
    def _key(incident_id: str) -> str:
        return hashlib.sha256(incident_id.encode()).hexdigest()[:20]

    def _state_path(self, incident_id: str) -> Path:
        return self.state_dir / f"quiesce-{self._key(incident_id)}.json"

    @staticmethod
    def _timer_unit(incident_id: str) -> str:
        return f"gpu-fault-quiesce-{GpuServiceQuiesceManager._key(incident_id)}"

    def _run(
        self,
        command: list[str],
        *,
        check: bool,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        try:
            return self.runner(
                command,
                check=check,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or "").strip()
            if len(output) > 2000:
                output = output[-2000:]
            detail = f": {output}" if output else ""
            raise RuntimeError(
                f"{command[0]} exited with status {exc.returncode}{detail}"
            ) from exc

    def _service_active(self, service: str) -> bool:
        completed = self._run(
            ["systemctl", "is-active", "--quiet", service],
            check=False,
        )
        return completed.returncode == 0

    def _task_states(self) -> dict[str, str]:
        completed = self._run(
            ["ctr", "-n", "k8s.io", "tasks", "list"],
            check=True,
        )
        states = {}
        for line in completed.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3:
                states[fields[0]] = fields[-1].upper()
        return states

    def _running_container_ids(self, selector: str) -> list[str]:
        completed = self._run(
            [
                "ctr",
                "-n",
                "k8s.io",
                "containers",
                "list",
                "--quiet",
                _container_filter(selector),
            ],
            check=True,
        )
        candidates = {
            value.strip() for value in completed.stdout.splitlines() if value.strip()
        }
        states = self._task_states()
        return sorted(
            container_id
            for container_id in candidates
            if states.get(container_id) == "RUNNING"
        )

    def _resolve_container_targets(self) -> list[dict[str, str]]:
        targets = []
        for selector in self.containers:
            running = self._running_container_ids(selector)
            for container_id in running:
                targets.append(
                    {
                        "selector": selector,
                        "container_id": container_id,
                    }
                )
        return targets

    def _wait_task_stopped(self, container_id: str) -> bool:
        deadline = time.monotonic() + self.container_stop_timeout_seconds
        while time.monotonic() < deadline:
            if self._task_states().get(container_id) != "RUNNING":
                return True
            self.sleeper(0.5)
        return False

    def _kill_task(self, container_id: str, signal_name: str) -> None:
        """Signal a container task; ``ctr``'s exit status is advisory: a task
        that exited on its own first (kubelet is stopped, nothing restarts it)
        fails "process already finished"; ``_wait_task_stopped`` decides."""

        completed = self._run(
            [
                "ctr",
                "-n",
                "k8s.io",
                "tasks",
                "kill",
                "--signal",
                signal_name,
                "--all",
                container_id,
            ],
            check=False,
        )
        if completed.returncode:
            LOGGER.info(
                "ctr tasks kill %s for %s exited with status %s: %s",
                signal_name,
                container_id,
                completed.returncode,
                (completed.stderr or "").strip()[-200:],
            )

    def _stop_containers(self, targets: list[dict[str, str]]) -> None:
        for target in targets:
            container_id = target["container_id"]
            self._kill_task(container_id, "SIGTERM")
            if self._wait_task_stopped(container_id):
                continue
            self._kill_task(container_id, "SIGKILL")
            if not self._wait_task_stopped(container_id):
                raise RuntimeError(f"container task {container_id} did not stop")

    def _wait_containers_restored(self, targets: list[dict[str, str]]) -> list[str]:
        if not targets:
            return []
        deadline = time.monotonic() + self.container_restore_timeout_seconds
        pending = {target["selector"] for target in targets}
        while pending and time.monotonic() < deadline:
            pending = {
                selector
                for selector in pending
                if not self._running_container_ids(selector)
            }
            if pending:
                self.sleeper(1)
        return sorted(pending)

    @contextmanager
    def _locked(self, state_path: Path):
        import fcntl

        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = state_path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield

    @staticmethod
    def _read_state(state_path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("invalid quiesce state")
        return value

    @staticmethod
    def _write_state(state_path: Path, state: dict[str, Any]) -> None:
        temporary = state_path.with_suffix(".tmp")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, state_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _device_holders(
        self,
        target_device_paths: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """PIDs holding an open fd on a /dev/nvidiaN device node, read straight
        from ``/proc`` (no ``fuser``/``lsof`` on the image) exactly as
        :meth:`NodeActionExecutor._device_clients` reads them; keyed by PID."""
        holders: dict[str, dict[str, str]] = {}
        try:
            entries = list(self.proc_root.iterdir())
        except OSError:
            return []
        own = str(os.getpid())
        for entry in entries:
            if not entry.name.isdigit() or entry.name == own:
                continue
            try:
                descriptors = list((entry / "fd").iterdir())
            except OSError:
                continue
            devices = set()
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if NVIDIA_GPU_DEVICE_PATTERN.fullmatch(target):
                    devices.add(target)
            if not devices:
                continue
            if target_device_paths is not None:
                devices.intersection_update(target_device_paths)
                if not devices:
                    continue
            try:
                name = (entry / "comm").read_text().strip()
            except (OSError, UnicodeError):
                name = "unknown"
            holders[entry.name] = {
                "pid": entry.name,
                "process_name": name,
                "devices": ",".join(sorted(devices)),
                "cgroup_paths": process_cgroup_paths(entry),
            }
        return [holders[pid] for pid in sorted(holders, key=int)]

    def _sweep_device_holders(
        self,
        *,
        target_device_paths: set[str] | None,
        workload_cgroup_paths: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Terminate GPU clients that survived the service/container stop.

        ``systemctl stop`` only reaches what systemd owns: a hand-started
        ``nvidia-persistenced`` (``MainPID=0``) keeps every ``/dev/nvidiaN``
        open while the stop is a silent no-op, so before this sweep quiesce
        said ``quiesced: True`` and ``VERIFY_NO_GPU_CLIENTS`` failed closed
        forever. The sweep is limited to the target GPU and to an explicit
        process whitelist, the affected workload's cgroup, or -- when
        ``sweep_kubernetes_device_holders`` is set -- a Kubernetes container,
        never Fabric Manager or NVLSM (signalled, the fabric loses its state).
        Others stay alive: VERIFY, the authority on survivors too, fails closed
        rather than kill an unrelated workload. Returns the signalled holders.
        """
        swept: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        signalled: dict[str, dict[str, Any]] = {}
        for holder in self._device_holders(target_device_paths):
            process_allowed = holder["process_name"] in self.device_sweep_processes
            cgroup_allowed = _cgroup_matches(
                holder.get("cgroup_paths", []),
                workload_cgroup_paths,
            )
            attribution = kubernetes_attribution(holder.get("cgroup_paths", []))
            kubernetes_allowed = (
                self.sweep_kubernetes_device_holders and attribution["kubernetes"]
            )
            reason = (
                "fabric_manager_never_swept"
                if fabric_manager_process(holder)
                else None
                if process_allowed or cgroup_allowed or kubernetes_allowed
                else "outside_workload_cgroup_and_process_whitelist"
            )
            if reason is not None:
                skipped.append({**holder, **attribution, "reason": reason})
                continue
            record = dict(holder)
            if kubernetes_allowed and not process_allowed and not cgroup_allowed:
                record.update(attribution)
                record["reason"] = "kubernetes_container_on_quarantined_node"
            record["signal"] = "TERM"
            self._run(
                ["kill", "--signal", "TERM", holder["pid"]],
                check=False,
            )
            signalled[holder["pid"]] = record
            swept.append(record)
        if not signalled:
            return swept, skipped
        # Until each signalled process holds no GPU at all, not only the
        # target: nvidia-persistenced closes the target first and tears the
        # other devices down for seconds more, into VERIFY (live, HyperPod).
        deadline = time.monotonic() + self.device_sweep_timeout_seconds
        while time.monotonic() < deadline:
            remaining = {holder["pid"] for holder in self._device_holders()}
            if not remaining & set(signalled):
                return swept, skipped
            self.sleeper(0.5)
        for pid in sorted(
            {holder["pid"] for holder in self._device_holders()} & set(signalled),
            key=int,
        ):
            signalled[pid]["signal"] = "KILL"
            self._run(["kill", "--signal", "KILL", pid], check=False)
        return swept, skipped

    def _restore_failed_window(
        self,
        state_path: Path,
        existing: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Clear a failed quiesce window (``QUIESCE_FAILED`` / ``RESTORE_FAILED``)
        by restoring it inline so the same command can retry from scratch;
        every retry used to raise "quiesce is incomplete" until the fail-safe
        fired. Returns the state unchanged for a healthy ``QUIESCED`` window,
        None after a restore; a restore that itself fails keeps the window."""

        phase = existing.get("phase")
        if phase == "QUIESCED":
            return existing
        if phase not in {"QUIESCE_FAILED", "RESTORE_FAILED"}:
            # ARMING/QUIESCING/RESTORING means the writer died mid-step, so
            # the node state is unknown: leave it to the fail-safe timer and
            # the boot reconciler rather than stopping services again.
            raise RuntimeError(
                "GPU service quiesce is incomplete; fail-safe restore remains armed"
            )
        try:
            restored = self._restore_locked(state_path)
        except Exception as exc:
            raise RuntimeError(
                f"the previous {phase} GPU service quiesce could not be restored, "
                f"so a fresh quiesce is refused: {exc}"
            ) from exc
        LOGGER.warning(
            "restored a %s quiesce window before re-quiescing %s: %s",
            phase,
            existing.get("incident_id"),
            restored.get("restored_services"),
        )
        return None

    def quiesce(
        self,
        *,
        incident_id: str,
        workflow_request_id: str,
        target_device_paths: set[str] | None = None,
        workload_cgroup_paths: set[str] | None = None,
        target_pci_bus_ids: Sequence[str] | None = None,
        whole_node: bool = False,
        keep_fabric_manager: bool = False,
    ) -> dict[str, Any]:
        """Open one incident's quiesce window: stop, release, sweep, record.

        ``target_pci_bus_ids`` are the targets' bus ids from the Agent's own
        device map (``None`` when it has none) and ``whole_node`` names every
        GPU. Persistence ends on the targets alone, so the node's other GPUs
        stay initialised and no ``nvidia-smi`` of the window re-initialises
        them; the device sweep stays the fallback for a holder still there
        afterwards. ``keep_fabric_manager`` leaves a running Fabric Manager
        alone (``_service_inventory``). A QUIESCED window is answered as it is;
        a failed one is restored first and quiesced again.
        """

        workload_cgroup_paths = {
            path.rstrip("/")
            for path in (workload_cgroup_paths or set())
            if path and path.rstrip("/") != ""
        }
        normalized_target_paths = (
            sorted(target_device_paths) if target_device_paths is not None else []
        )
        state_path = self._state_path(incident_id)
        timer_unit = self._timer_unit(incident_id)
        with self._locked(state_path):
            existing = self._read_state(state_path)
            if existing is not None:
                if (
                    existing.get("incident_id") != incident_id
                    or existing.get("workflow_request_id") != workflow_request_id
                ):
                    raise RuntimeError("quiesce state belongs to another workflow")
                existing = self._restore_failed_window(state_path, existing)
            if existing is not None:
                if existing.get("target_device_paths", []) != (normalized_target_paths):
                    raise RuntimeError("quiesce target GPU scope changed after quiesce")
                return {
                    "quiesced": True,
                    "already_quiesced": True,
                    "active_services": existing.get("active_services", []),
                    "kept_services": existing.get("kept_services", []),
                    "timer_unit": timer_unit + ".timer",
                    "state_path": str(state_path),
                }

            active_services, inactive_services, kept_services = _service_inventory(
                self, keep_fabric_manager
            )
            container_targets = self._resolve_container_targets()
            state = {
                "schema_version": 1,
                "incident_id": incident_id,
                "workflow_request_id": workflow_request_id,
                "phase": "ARMING",
                "active_services": active_services,
                "inactive_services": inactive_services,
                "configured_services": list(self.services),
                "protected_services": list(self.protected_services),
                "container_targets": container_targets,
                "target_device_paths": normalized_target_paths,
                "workload_cgroup_paths": sorted(workload_cgroup_paths),
                "timer_unit": timer_unit,
                "boot_id": self.boot_id_reader(),
                # One quiesce window buys one reset: assert_quiesced claims
                # this slot before nvidia-smi runs. A file written before this
                # key existed has no claim, so it reads as "not yet reset".
                "reset_issued": None,
            }
            self._write_state(state_path, state)
            try:
                _arm_quiesce_failsafe(self, timer_unit, state_path)
            except Exception:
                state_path.unlink(missing_ok=True)
                raise
            state["phase"] = "QUIESCING"
            self._write_state(state_path, state)
            persistence = TargetPersistence(
                run=self._run,
                holders=lambda: self._device_holders(target_device_paths),
                proc_root=self.proc_root,
                kept_services=kept_services,
            )
            try:
                # Each service, last configured first, around the targets'
                # persistence release; a daemon unit that took it keeps running.
                persistence.stop_services(
                    active_services=active_services,
                    pci_bus_ids=target_pci_bus_ids,
                    whole_node=whole_node,
                )
                self._stop_containers(container_targets)
                for process in self.processes:
                    self._run(
                        ["pkill", "--signal", "TERM", "--exact", process],
                        check=False,
                    )
                if self.settle_seconds:
                    self.sleeper(self.settle_seconds)
                (
                    swept_device_holders,
                    unswept_device_holders,
                ) = self._sweep_device_holders(
                    target_device_paths=target_device_paths,
                    workload_cgroup_paths=workload_cgroup_paths,
                )
                window = persistence.finish(swept_device_holders)
            except Exception:
                state["phase"] = "QUIESCE_FAILED"
                self._write_state(state_path, state)
                raise
            state["kept_services"] = window["kept_services"]
            if "persistence_daemon_swept" in window:
                state["persistence_daemon_swept"] = window["persistence_daemon_swept"]
            state["phase"] = "QUIESCED"
            self._write_state(state_path, state)
            return {
                "quiesced": True,
                "active_services": active_services,
                "inactive_services": inactive_services,
                "protected_services": list(self.protected_services),
                "terminated_process_names": list(self.processes),
                "swept_device_holders": swept_device_holders,
                "unswept_device_holders": unswept_device_holders,
                "stopped_containers": container_targets,
                "timer_unit": timer_unit + ".timer",
                "failsafe_seconds": self.failsafe_seconds,
                "state_path": str(state_path),
                **window,
            }

    def assert_quiesced(
        self,
        *,
        incident_id: str,
        command_id: str | None = None,
        for_reset: bool = True,
        attempt: int | None = None,
    ) -> None:
        """Fence a reset on the quiesce state and claim the window for it.

        The claim is recorded under the same flock as the phase check and
        before nvidia-smi is spawned, so a resubmit of a reset whose outcome is
        unknown -- and any other command_id for the same incident -- is refused
        instead of resetting the GPU a second time. A new window needs a real
        restore plus a fresh quiesce, which clears the claim.

        ``for_reset`` defaults to True so that a caller which forgets it fails
        closed. Driver and firmware remediation are fenced on the same state
        but are not resets, so they pass ``for_reset=False`` and leave the
        window's single reset allowance alone.
        """

        state_path = self._state_path(incident_id)
        with self._locked(state_path):
            state = self._read_state(state_path)
            if state is None or state.get("phase") != "QUIESCED":
                raise RuntimeError("GPU reset requires an active service quiesce state")
            if not for_reset:
                return
            issued = state.get("reset_issued")
            if isinstance(issued, dict):
                raise RuntimeError(_reset_claim_refusal(issued, command_id))
            state["reset_issued"] = {
                "command_id": command_id,
                "attempt": attempt,
                "issued_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_state(state_path, state)

    def reconcile_after_boot(self) -> dict[str, Any]:
        """Restore quiesce state files that a reboot orphaned: a file from
        another boot (or, before boot ids were recorded, without its timer) is
        bookkeeping -- start what it names, drop the timer, remove it; a file
        of this boot whose timer is armed is an in-flight window and stays."""

        boot_id = self.boot_id_reader()
        report: dict[str, Any] = {
            "boot_id": boot_id,
            "restored": [],
            "kept": [],
            "failed": [],
        }
        if not self.state_dir.is_dir():
            return report
        for state_path in sorted(self.state_dir.glob("quiesce-*.json")):
            try:
                state = self._read_state(state_path) or {}
            except (OSError, ValueError, RuntimeError) as exc:
                report["failed"].append(
                    {"state_path": str(state_path), "error": str(exc)}
                )
                continue
            recorded = state.get("boot_id")
            timer_unit = str(state.get("timer_unit") or "")
            entry = {
                "state_path": str(state_path),
                "incident_id": state.get("incident_id"),
                "recorded_boot_id": recorded,
                "phase": state.get("phase"),
            }
            if recorded is not None and recorded == boot_id:
                report["kept"].append(entry)
                continue
            if (
                recorded is None
                and timer_unit
                and self._service_active(timer_unit + ".timer")
            ):
                report["kept"].append(entry)
                continue
            try:
                restored = self.restore_state_file(
                    state_path,
                    runner=self.runner,
                    proc_root=str(self.proc_root),
                    container_restore_timeout_seconds=(
                        self.container_restore_timeout_seconds
                    ),
                    # This boot recreated every container, so there is nothing
                    # of ours left to come back: polling ``ctr`` here only
                    # delayed the HTTP server and the heartbeat by up to
                    # container_restore_timeout_seconds *per stale file*.
                    wait_for_containers=False,
                )
            except Exception as exc:  # noqa: BLE001 - one file must not stop the sweep
                report["failed"].append(
                    {**entry, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            skipped = restored.get("container_restore_wait_skipped")
            if skipped:
                # The only place the skip is visible: whoever reads the
                # start-up report has to know these are not workload-ready.
                entry["container_restore_wait_skipped"] = skipped
            report["restored"].append(entry)
        return report

    def restore(self, *, incident_id: str) -> dict[str, Any]:
        """Close the window; record each GPU's persistence mode after the settle."""

        result = self.restore_state_file(
            self._state_path(incident_id),
            runner=self.runner,
            proc_root=str(self.proc_root),
            container_restore_timeout_seconds=(self.container_restore_timeout_seconds),
        )
        if not result.get("already_restored"):
            if self.restore_settle_seconds:
                self.sleeper(self.restore_settle_seconds)
            # By now the restore unit's ``nvidia-smi -pm 1`` has run.
            result["persistence_mode"] = restored_modes(self._run)
        result["restore_settle_seconds"] = self.restore_settle_seconds
        return result

    @classmethod
    def restore_state_file(
        cls,
        state_path: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        proc_root: str = "/proc",
        container_restore_timeout_seconds: int = 180,
        wait_for_containers: bool = True,
    ) -> dict[str, Any]:
        manager = cls(
            state_dir=str(state_path.parent),
            runner=runner,
            proc_root=proc_root,
            container_restore_timeout_seconds=(container_restore_timeout_seconds),
        )
        with manager._locked(state_path):
            return manager._restore_locked(
                state_path,
                wait_for_containers=wait_for_containers,
            )

    def _restore_locked(
        self,
        state_path: Path,
        *,
        wait_for_containers: bool = True,
    ) -> dict[str, Any]:
        """Restore one state file with its per-incident flock already held.

        ``quiesce`` restores a failed window inline, and flock is owned by the
        open file description: re-entering :meth:`_locked` in the same process
        would deadlock against the lock this thread already holds.

        ``wait_for_containers`` is False only for the boot reconcile: the
        containers this file names died with the previous boot and kubelet
        recreates them on its own schedule, so waiting for them buys nothing
        and costs the whole restore timeout per stale file.
        """

        state = self._read_state(state_path)
        if state is None:
            return {
                "restored": True,
                "already_restored": True,
                "state_path": str(state_path),
            }
        services = state.get("active_services")
        container_targets = state.get("container_targets", [])
        timer_unit = state.get("timer_unit")
        if (
            not isinstance(services, list)
            or not all(
                isinstance(item, str) and SYSTEMD_NAME_PATTERN.fullmatch(item)
                for item in services
            )
            or not isinstance(timer_unit, str)
            or not SYSTEMD_NAME_PATTERN.fullmatch(timer_unit)
            or not isinstance(container_targets, list)
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("selector"), str)
                and len(item["selector"].split("/")) == 2
                and all(
                    SYSTEMD_NAME_PATTERN.fullmatch(part)
                    for part in item["selector"].split("/")
                )
                and isinstance(item.get("container_id"), str)
                and re.fullmatch(r"[a-f0-9]{64}", item["container_id"])
                for item in container_targets
            )
        ):
            raise RuntimeError("invalid quiesce restore state")
        state["phase"] = "RESTORING"
        # The window is over: a later quiesce of the same incident starts
        # with a clean claim even if this restore fails and keeps the file.
        state["reset_issued"] = None
        self._write_state(state_path, state)
        try:
            for service in services:
                self._run(
                    ["systemctl", "start", service],
                    check=True,
                    timeout=unit_start_timeout_seconds(service),
                )
            inactive = [
                service for service in services if not self._service_active(service)
            ]
            if inactive:
                raise RuntimeError(
                    "restored services are not active: " + ", ".join(inactive)
                )
        except Exception:
            # Every way a restore can fail records RESTORE_FAILED. A start
            # that raises used to leave the file at RESTORING, which the next
            # quiesce reads as a writer that died mid-step and refuses until
            # the fail-safe timer fires: the dead end again, one failure
            # later, with the GPU services still down.
            state["phase"] = "RESTORE_FAILED"
            self._write_state(state_path, state)
            raise
        pending_containers = (
            self._wait_containers_restored(container_targets)
            if wait_for_containers
            else []
        )
        self._run(
            ["systemctl", "stop", timer_unit + ".timer"],
            check=False,
        )
        self._run(
            ["systemctl", "reset-failed", timer_unit + ".service"],
            check=False,
        )
        state_path.unlink()
        result = {
            "restored": True,
            "restored_services": services,
            "timer_cancelled": timer_unit + ".timer",
            "state_path": str(state_path),
            **restored_daemon(state, self.proc_root),
        }
        if not wait_for_containers and container_targets:
            result["container_restore_wait_skipped"] = sorted(
                {str(target["selector"]) for target in container_targets}
            )
        if pending_containers:
            result.update(
                {
                    "container_restore_warning": True,
                    "pending_container_selectors": (pending_containers),
                }
            )
        return result


def open_quiesce_windows(manager: GpuServiceQuiesceManager) -> list[dict[str, Any]]:
    """Every quiesce window open on this boot: incident, workflow and phase.

    A state file exists from the moment a QUIESCE arms its fail-safe until the
    restore removes it, in every phase in between (a failed quiesce or restore
    included: its services may still be down). A file a previous boot left is
    the boot reconcile's to restore, not a window. One that cannot be read is
    reported with no owner, so foreign GPU work stays out (fail closed).
    """

    if not manager.state_dir.is_dir():
        return []
    boot_id = manager.boot_id_reader()
    windows: list[dict[str, Any]] = []
    for state_path in sorted(manager.state_dir.glob("quiesce-*.json")):
        try:
            state = manager._read_state(state_path)
        except (OSError, ValueError, RuntimeError):
            windows.append({"incident_id": None, "phase": "UNREADABLE"})
            continue
        if state is None:
            continue
        recorded = state.get("boot_id")
        if recorded and boot_id and recorded != boot_id:
            continue
        windows.append(
            {
                "incident_id": state.get("incident_id"),
                "workflow_request_id": state.get("workflow_request_id"),
                "phase": state.get("phase"),
            }
        )
    return windows


def record_reset_scope(
    manager: GpuServiceQuiesceManager, incident_id: str, scope: str
) -> None:
    """Note on the window's reset claim what the claimed reset resets.

    The restore reads it back (:func:`claimed_reset_scope`): after a full GPU +
    NVSwitch reset it starts a Fabric Manager the quiesce found down, which a
    single-GPU reset never asks for. Best effort, because it runs between the
    claim and the reset: a failure here must not cost the window its one reset.
    A claim without the note -- an older Agent's, or a failed write -- reads as
    no scope, and the restore then starts only what the quiesce stopped. A
    quiesce manager that keeps no state file (a stand-in) has nothing to note.
    """

    if not isinstance(manager, GpuServiceQuiesceManager):
        return
    state_path = manager._state_path(incident_id)
    try:
        with manager._locked(state_path):
            state = manager._read_state(state_path)
            issued = (state or {}).get("reset_issued")
            if state is None or not isinstance(issued, dict):
                return
            issued["scope"] = scope
            manager._write_state(state_path, state)
    except (OSError, ValueError, RuntimeError):
        LOGGER.warning(
            "could not note the reset scope on the quiesce claim of %s",
            incident_id,
            exc_info=True,
        )


def _window_state(
    manager: GpuServiceQuiesceManager, incident_id: str
) -> dict[str, Any] | None:
    """The incident's quiesce state, read under its lock; None for a stand-in
    manager, no window, or one that cannot be read."""

    if not isinstance(manager, GpuServiceQuiesceManager):
        return None
    state_path = manager._state_path(incident_id)
    try:
        with manager._locked(state_path):
            return manager._read_state(state_path)
    except (OSError, ValueError, RuntimeError):
        return None


def window_kept_services(
    manager: GpuServiceQuiesceManager, incident_id: str
) -> frozenset[str]:
    """The services the incident's quiesce window keeps running (``kept_services``).

    The reset reads it right after its claim: a Fabric Manager is a bystander
    of the display-stack release only when this window's own record kept it
    (``holders.fabric_manager_bystander``), never one a window meant to stop. A
    stand-in manager or an unreadable window keeps nothing; never raises.
    """

    kept = (_window_state(manager, incident_id) or {}).get("kept_services")
    return frozenset(
        item["service"]
        for item in (kept if isinstance(kept, list) else ())
        if isinstance(item, dict) and isinstance(item.get("service"), str)
    )


def refuse_kept_fabric_manager(
    manager: GpuServiceQuiesceManager, incident_id: str, action: str
) -> None:
    """Refuse ``action`` inside a window that kept Fabric Manager running.

    A window keeps Fabric Manager for its per-GPU reset (``fabric_manager_kept``).
    The full GPU + NVSwitch reset and a driver or firmware install need it
    stopped -- NVIDIA's order for H100 and later is stop Fabric Manager,
    ``nvidia-smi -r``, start Fabric Manager -- and meet such a window only when
    a preempting successor adopts it. Nothing is stopped here: the command fails
    before it touches the node (the window's single reset stays unclaimed), the
    window is restored as planned, and the ladder's reboot re-registers the
    fabric as well. A stand-in manager or an unreadable window says nothing.
    """

    if FABRIC_MANAGER_UNIT in window_kept_services(manager, incident_id):
        raise RuntimeError(
            f"{action} needs {FABRIC_MANAGER_UNIT} stopped, and this quiesce "
            "window kept it running for a per-GPU reset; restore the GPU services "
            "and quiesce again, or let the ladder reboot the node"
        )


def claimed_reset_scope(
    manager: GpuServiceQuiesceManager, incident_id: str
) -> str | None:
    """The scope noted on the window's reset claim, if a reset was claimed.

    An unreadable window reads as no scope: the restore that follows reports
    the problem itself.
    """

    issued = (_window_state(manager, incident_id) or {}).get("reset_issued")
    scope = issued.get("scope") if isinstance(issued, dict) else None
    return scope if isinstance(scope, str) else None
