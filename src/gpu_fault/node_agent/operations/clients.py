from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from xml.etree import ElementTree as ET


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class ClientOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    device_client_samples: Any
    gpu_device_path_finder: Callable[..., Any]

    _run_checked: Callable[..., Any]
    device_client_finder: Callable[..., Any]
    device_client_sample_interval_seconds: Any
    proc_root: Any
    sleep: Callable[..., Any]

    # Owned by ``_device_path_cache_window``.  Class defaults so the mixin
    # needs nothing from the concrete executor's ``__init__``.
    _device_path_cache: dict[str, str] | None = None
    _device_path_cache_depth: int = 0

    def _compute_clients(self) -> list[dict[str, str]]:
        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        clients = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",", 2)]
            if len(values) == 3 and values[0]:
                clients.append(
                    {
                        "gpu_uuid": values[0],
                        "pid": values[1],
                        "process_name": values[2],
                    }
                )
        return clients

    @contextmanager
    def _device_path_cache_window(self) -> Iterator[None]:
        """Resolve the uuid -> device-node map at most once inside the block.

        The map is a property of the driver's probe order and cannot change
        while the node holds a GPU still, yet ``_persistent_device_clients``
        rebuilt it for every sample: with three samples on an 8-GPU node, one
        busy-retrying reset spent up to 96 ``nvidia-smi`` invocations learning
        the same answer -- and each one is a driver round trip that competes
        with the very reset it is waiting for.  The window is reference counted
        so two actions running side by side cannot clear each other's cache,
        and it is always dropped on the way out: nothing is cached across
        commands, where a driver reload really can renumber the devices.
        """

        self._device_path_cache_depth += 1
        try:
            yield
        finally:
            self._device_path_cache_depth -= 1
            if self._device_path_cache_depth <= 0:
                self._device_path_cache_depth = 0
                self._device_path_cache = None

    def _gpu_device_paths(self) -> dict[str, str]:
        cached = self._device_path_cache
        if cached is not None:
            return cached
        paths = self._minor_number_device_paths()
        if paths is None:
            paths = self._index_device_paths()
        if self._device_path_cache_depth:
            self._device_path_cache = paths
        return paths

    def _minor_number_device_paths(self) -> dict[str, str] | None:
        """uuid -> ``/dev/nvidia<minor>`` from the driver's own minor numbers.

        ``--query-gpu=index`` is ordered by PCI bus id; the device node's minor
        number is the order the driver probed the GPUs in, which is why
        nvidia-smi reports "Minor Number" separately at all.  Assuming they
        match made ``VERIFY_NO_GPU_CLIENTS`` inspect a *different* GPU's device
        node on any node where the two disagree: it passed with a live holder
        on the target GPU, and blocked on a holder of a GPU nobody was
        resetting.

        Answers ``None`` -- never a partial map -- when the XML does not give a
        usable minor number for every GPU, so the caller falls back to the
        index map as a whole.  A half-and-half map is the one outcome that
        could hide a holder.
        """

        try:
            completed = self._run_checked(["nvidia-smi", "-q", "-x"], timeout=15)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return None
        try:
            root = ET.fromstring(completed.stdout or "")
        except ET.ParseError:
            return None
        paths: dict[str, str] = {}
        for gpu in root.findall(".//gpu"):
            gpu_uuid = (gpu.findtext("uuid") or "").strip()
            minor = (gpu.findtext("minor_number") or "").strip()
            if not gpu_uuid or not minor.isdigit():
                return None
            paths[gpu_uuid] = f"/dev/nvidia{minor}"
        if not paths or len(set(paths.values())) != len(paths):
            return None
        return paths

    def _index_device_paths(self) -> dict[str, str]:
        """The pre-minor-number mapping, kept for drivers that omit it."""

        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=uuid,index",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        paths = {}
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",", 1)]
            if len(values) == 2 and values[0] and values[1].isdigit():
                paths[values[0]] = f"/dev/nvidia{values[1]}"
        return paths

    def _quiesce_scope(self, command: NodeActionCommand) -> tuple[set[str], set[str]]:
        device_paths = self.gpu_device_path_finder()
        if command.gpu_uuids:
            missing = sorted(set(command.gpu_uuids) - set(device_paths))
            if missing:
                raise RuntimeError(
                    "quiesce cannot resolve target GPU UUIDs: " + ", ".join(missing)
                )
            target_device_paths = {
                device_paths[gpu_uuid] for gpu_uuid in command.gpu_uuids
            }
        else:
            target_device_paths = set(device_paths.values())

        raw_by_node = command.parameters.get("workload_cgroup_paths_by_node")
        raw_paths = (
            raw_by_node.get(command.node_id, [])
            if isinstance(raw_by_node, dict)
            else command.parameters.get("workload_cgroup_paths", [])
        )
        if raw_paths is None:
            raw_paths = []
        if not isinstance(raw_paths, list) or not all(
            isinstance(path, str) for path in raw_paths
        ):
            raise RuntimeError("workload cgroup paths must be a list of strings")
        workload_cgroup_paths = {
            path.rstrip("/") for path in raw_paths if path and path.rstrip("/")
        }
        return target_device_paths, workload_cgroup_paths

    def _gpu_inventory(self) -> list[str]:
        completed = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        inventory = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        if not inventory:
            raise RuntimeError("local GPU inventory is empty")
        return inventory

    @staticmethod
    def _process_name(process: Path) -> str:
        try:
            return (process / "comm").read_text().strip()
        except (OSError, UnicodeError):
            return "unknown"

    def _device_clients(self, gpu_uuids: set[str]) -> list[dict[str, str]]:
        device_paths = self._gpu_device_paths()
        targets = {
            uuid: path
            for uuid, path in device_paths.items()
            if not gpu_uuids or uuid in gpu_uuids
        }
        path_to_uuid = {path: uuid for uuid, path in targets.items()}
        clients: dict[tuple[str, str], dict[str, str]] = {}
        try:
            processes = list(self.proc_root.iterdir())
        except OSError:
            processes = []
        for process in processes:
            if not process.name.isdigit():
                continue
            try:
                descriptors = list((process / "fd").iterdir())
            except OSError:
                continue
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                gpu_uuid = path_to_uuid.get(target)
                if gpu_uuid is None:
                    continue
                key = (gpu_uuid, process.name)
                clients[key] = {
                    "gpu_uuid": gpu_uuid,
                    "pid": process.name,
                    "process_name": self._process_name(process),
                    "device": target,
                }
        return sorted(
            clients.values(),
            key=lambda item: (
                item["gpu_uuid"],
                int(item["pid"]),
            ),
        )

    def _persistent_device_clients(
        self, target: set[str]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split device holders into persistent and transient ones.

        A holder counts as persistent only when the *same* pid still
        holds the *same* device across every sample.  Without this,
        short lived GPU processes that respawn on a timer deadlock the
        remediation forever: quiesce sweeps the holder, the next
        ``VERIFY_NO_GPU_CLIENTS`` attempt observes a brand new pid, and
        the step keeps failing until the quiesce maintenance window
        expires -- so the GPU is never reset and the node stays
        quarantined.  Measured on a HyperPod node, 96 of 98 attempts
        failed this way, blocked by ``nvidia-persistenced`` (which these
        AMIs ship with no systemd unit, so ``systemctl stop`` cannot
        reach it) and by per-second ``nvidia-smi`` invocations.

        Sampling stays fail-closed: a holder that survives every sample
        still blocks the reset, and a single sample (the default when
        the deadline leaves no room) behaves exactly as before.
        """
        samples: list[dict[tuple[str, str, str], dict[str, str]]] = []
        for index in range(self.device_client_samples):
            if index:
                self.sleep(self.device_client_sample_interval_seconds)
            samples.append(
                {
                    (
                        item["gpu_uuid"],
                        item["pid"],
                        item.get("device", ""),
                    ): item
                    for item in self.device_client_finder(target)
                }
            )
        if not samples:
            return [], []
        shared = set(samples[0])
        for sample in samples[1:]:
            shared &= set(sample)
        persistent = [samples[-1][key] for key in sorted(shared)]
        transient = [
            item
            for sample in samples
            for key, item in sorted(sample.items())
            if key not in shared
        ]
        return persistent, transient

    def _verify_no_clients(
        self,
        gpu_uuids: list[str],
        *,
        include_device_clients: bool = True,
    ) -> dict[str, Any]:
        target = set(gpu_uuids)
        compute_clients = [
            item
            for item in self._compute_clients()
            if not target or item["gpu_uuid"] in target
        ]
        if compute_clients:
            raise RuntimeError(
                "GPU compute clients are still active: "
                + ", ".join(
                    f"{item['gpu_uuid']}:{item['pid']}" for item in compute_clients
                )
            )
        transient_device_clients: list[dict[str, str]] = []
        if include_device_clients:
            with self._device_path_cache_window():
                (
                    device_clients,
                    transient_device_clients,
                ) = self._persistent_device_clients(target)
            if device_clients:
                raise RuntimeError(
                    "GPU device clients are still active: "
                    + ", ".join(
                        f"{item['gpu_uuid']}:{item['pid']}:{item['process_name']}"
                        for item in device_clients
                    )
                )
        return {
            "verified_no_gpu_clients": True,
            "gpu_uuids": gpu_uuids,
            "device_clients_checked": include_device_clients,
            "transient_device_clients": transient_device_clients,
        }
