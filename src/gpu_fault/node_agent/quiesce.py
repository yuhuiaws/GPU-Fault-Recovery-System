from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.common import (
    DEFAULT_DEVICE_SWEEP_PROCESSES,
    DEFAULT_QUIESCE_CONTAINERS,
    DEFAULT_QUIESCE_PROCESSES,
    DEFAULT_QUIESCE_SERVICES,
    NVIDIA_GPU_DEVICE_PATTERN,
    PROCESS_NAME_PATTERN,
    SYSTEMD_NAME_PATTERN,
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


class GpuServiceQuiesceManager:
    """Quiesces host GPU services with an independent systemd restore timer."""

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
        proc_root: str = "/proc",
        restore_command: str = (
            "/opt/gpu-fault/current/venv/bin/gpu-fault-restore-gpu-services"
        ),
        runner: Callable[..., subprocess.CompletedProcess] = (subprocess.run),
        sleeper: Callable[[float], None] = time.sleep,
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
        self.proc_root = Path(proc_root)
        self.restore_command = restore_command
        self.runner = runner
        self.sleeper = sleeper

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

    @staticmethod
    def _container_filter(selector: str) -> str:
        namespace, name = selector.split("/", 1)
        return (
            'labels."io.kubernetes.pod.namespace"=='
            f'{namespace},labels."io.kubernetes.container.name"=={name}'
        )

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
                self._container_filter(selector),
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

    def _stop_containers(self, targets: list[dict[str, str]]) -> None:
        for target in targets:
            container_id = target["container_id"]
            self._run(
                [
                    "ctr",
                    "-n",
                    "k8s.io",
                    "tasks",
                    "kill",
                    "--signal",
                    "SIGTERM",
                    "--all",
                    container_id,
                ],
                check=True,
            )
            if self._wait_task_stopped(container_id):
                continue
            self._run(
                [
                    "ctr",
                    "-n",
                    "k8s.io",
                    "tasks",
                    "kill",
                    "--signal",
                    "SIGKILL",
                    "--all",
                    container_id,
                ],
                check=True,
            )
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

    @staticmethod
    def _process_cgroup_paths(process: Path) -> list[str]:
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

    def _device_holders(
        self,
        target_device_paths: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """PIDs holding an open fd on a /dev/nvidiaN device node.

        Read straight from ``/proc`` rather than via ``fuser``/``lsof``
        so the sweep needs no extra package on the node image, and so it
        sees exactly what :meth:`NodeActionExecutor._device_clients`
        sees.  Control-plane and per-GPU accounting live there; here we
        only need "who is still holding a GPU", so the result is keyed
        by PID.
        """
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
                "cgroup_paths": self._process_cgroup_paths(entry),
            }
        return [holders[pid] for pid in sorted(holders, key=int)]

    @staticmethod
    def _cgroup_matches(
        holder_paths: list[str],
        workload_cgroup_paths: set[str],
    ) -> bool:
        for holder_path in holder_paths:
            normalized_holder = holder_path.rstrip("/")
            for workload_path in workload_cgroup_paths:
                normalized_workload = workload_path.rstrip("/")
                if not normalized_workload or normalized_workload == "/":
                    continue
                if (
                    normalized_holder == normalized_workload
                    or normalized_holder.startswith(normalized_workload + "/")
                ):
                    return True
        return False

    def _sweep_device_holders(
        self,
        *,
        target_device_paths: set[str] | None,
        workload_cgroup_paths: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Terminate GPU clients that survived the service/container stop.

        ``systemctl stop`` only reaches what systemd owns.  A
        ``nvidia-persistenced`` started by hand outside systemd reports
        ``MainPID=0 ActiveState=inactive`` -- the stop is a silent no-op
        -- while the process keeps every ``/dev/nvidiaN`` open, so
        ``nvidia-smi --gpu-reset`` can never run.  Before this sweep the
        only backstop was ``GPU_FAULT_QUIESCE_PROCESSES``, an operator
        maintained name list that is empty by default, so quiesce
        returned ``quiesced: True`` and the very next step,
        ``VERIFY_NO_GPU_CLIENTS``, failed closed forever. The sweep is
        now limited to the target GPU and either an explicit system
        process whitelist or the affected workload's cgroup. Unknown
        holders are left alive so the verification step fails closed
        instead of killing an unrelated workload.

        Returns the holders that were signalled, in the order they were
        found.  Survivors are *not* an error here: quiesce stays
        advisory and ``VERIFY_NO_GPU_CLIENTS`` remains the authority
        that refuses to reset.
        """
        swept: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        signalled: dict[str, dict[str, Any]] = {}
        for holder in self._device_holders(target_device_paths):
            process_allowed = holder["process_name"] in self.device_sweep_processes
            cgroup_allowed = self._cgroup_matches(
                holder.get("cgroup_paths", []),
                workload_cgroup_paths,
            )
            if not process_allowed and not cgroup_allowed:
                skipped.append(
                    {
                        **holder,
                        "reason": ("outside_workload_cgroup_and_process_whitelist"),
                    }
                )
                continue
            record = dict(holder)
            record["signal"] = "TERM"
            self._run(
                ["kill", "--signal", "TERM", holder["pid"]],
                check=False,
            )
            signalled[holder["pid"]] = record
            swept.append(record)
        if not signalled:
            return swept, skipped
        deadline = time.monotonic() + self.device_sweep_timeout_seconds
        while time.monotonic() < deadline:
            remaining = {
                holder["pid"] for holder in self._device_holders(target_device_paths)
            } & set(signalled)
            if not remaining:
                return swept, skipped
            self.sleeper(0.5)
        for pid in sorted(
            {holder["pid"] for holder in self._device_holders(target_device_paths)}
            & set(signalled),
            key=int,
        ):
            signalled[pid]["signal"] = "KILL"
            self._run(["kill", "--signal", "KILL", pid], check=False)
        return swept, skipped

    def quiesce(
        self,
        *,
        incident_id: str,
        workflow_request_id: str,
        target_device_paths: set[str] | None = None,
        workload_cgroup_paths: set[str] | None = None,
    ) -> dict[str, Any]:
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
                if existing.get("phase") != "QUIESCED":
                    raise RuntimeError(
                        "GPU service quiesce is incomplete; "
                        "fail-safe restore remains armed"
                    )
                if existing.get("target_device_paths", []) != (normalized_target_paths):
                    raise RuntimeError("quiesce target GPU scope changed after quiesce")
                return {
                    "quiesced": True,
                    "already_quiesced": True,
                    "active_services": existing.get("active_services", []),
                    "timer_unit": timer_unit + ".timer",
                    "state_path": str(state_path),
                }

            active_services = [
                service for service in self.services if self._service_active(service)
            ]
            container_targets = self._resolve_container_targets()
            state = {
                "schema_version": 1,
                "incident_id": incident_id,
                "workflow_request_id": workflow_request_id,
                "phase": "ARMING",
                "active_services": active_services,
                "configured_services": list(self.services),
                "container_targets": container_targets,
                "target_device_paths": normalized_target_paths,
                "workload_cgroup_paths": sorted(workload_cgroup_paths),
                "timer_unit": timer_unit,
            }
            self._write_state(state_path, state)
            try:
                self._run(
                    [
                        "systemd-run",
                        f"--unit={timer_unit}",
                        f"--on-active={self.failsafe_seconds}s",
                        f"--on-unit-active={self.retry_seconds}s",
                        "--timer-property=AccuracySec=1s",
                        "--property=Type=oneshot",
                        self.restore_command,
                        "--state-file",
                        str(state_path),
                    ],
                    check=True,
                )
            except Exception:
                state_path.unlink(missing_ok=True)
                raise
            state["phase"] = "QUIESCING"
            self._write_state(state_path, state)
            try:
                for service in reversed(active_services):
                    self._run(
                        ["systemctl", "stop", service],
                        check=True,
                        timeout=120,
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
            except Exception:
                state["phase"] = "QUIESCE_FAILED"
                self._write_state(state_path, state)
                raise
            state["phase"] = "QUIESCED"
            self._write_state(state_path, state)
            return {
                "quiesced": True,
                "active_services": active_services,
                "terminated_process_names": list(self.processes),
                "swept_device_holders": swept_device_holders,
                "unswept_device_holders": unswept_device_holders,
                "stopped_containers": container_targets,
                "timer_unit": timer_unit + ".timer",
                "failsafe_seconds": self.failsafe_seconds,
                "state_path": str(state_path),
            }

    def assert_quiesced(self, *, incident_id: str) -> None:
        state_path = self._state_path(incident_id)
        with self._locked(state_path):
            state = self._read_state(state_path)
            if state is None or state.get("phase") != "QUIESCED":
                raise RuntimeError("GPU reset requires an active service quiesce state")

    def restore(self, *, incident_id: str) -> dict[str, Any]:
        result = self.restore_state_file(
            self._state_path(incident_id),
            runner=self.runner,
            container_restore_timeout_seconds=(self.container_restore_timeout_seconds),
        )
        if not result.get("already_restored") and self.restore_settle_seconds:
            self.sleeper(self.restore_settle_seconds)
        result["restore_settle_seconds"] = self.restore_settle_seconds
        return result

    @classmethod
    def restore_state_file(
        cls,
        state_path: Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        container_restore_timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        manager = cls(
            state_dir=str(state_path.parent),
            runner=runner,
            container_restore_timeout_seconds=(container_restore_timeout_seconds),
        )
        with manager._locked(state_path):
            state = manager._read_state(state_path)
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
            manager._write_state(state_path, state)
            for service in services:
                manager._run(
                    ["systemctl", "start", service],
                    check=True,
                    timeout=120,
                )
            inactive = [
                service for service in services if not manager._service_active(service)
            ]
            if inactive:
                state["phase"] = "RESTORE_FAILED"
                manager._write_state(state_path, state)
                raise RuntimeError(
                    "restored services are not active: " + ", ".join(inactive)
                )
            pending_containers = manager._wait_containers_restored(container_targets)
            manager._run(
                ["systemctl", "stop", timer_unit + ".timer"],
                check=False,
            )
            manager._run(
                ["systemctl", "reset-failed", timer_unit + ".service"],
                check=False,
            )
            state_path.unlink()
            result = {
                "restored": True,
                "restored_services": services,
                "timer_cancelled": timer_unit + ".timer",
                "state_path": str(state_path),
            }
            if pending_containers:
                result.update(
                    {
                        "container_restore_warning": True,
                        "pending_container_selectors": (pending_containers),
                    }
                )
            return result
