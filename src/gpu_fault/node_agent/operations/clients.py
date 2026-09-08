from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

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

    def _gpu_device_paths(self) -> dict[str, str]:
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
            device_clients, transient_device_clients = self._persistent_device_clients(
                target
            )
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
