from __future__ import annotations

import subprocess
from typing import Any, Callable

from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class ResetOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _gpu_inventory: Callable[..., Any]
    _run_checked: Callable[..., Any]
    _verify_no_clients: Callable[..., Any]
    device_client_sample_interval_seconds: Any
    fabric_manager_restart_enabled: Any
    fabric_reset_enabled: Any
    reset_enabled: Any
    single_gpu_reset_supported: Any
    sleep: Callable[..., Any]

    _RESET_BUSY_ATTEMPTS = 3

    def _run_reset_with_busy_retry(
        self,
        command: list[str],
        *,
        gpu_uuids: list[str],
        timeout: int,
    ) -> int:
        for attempt in range(1, self._RESET_BUSY_ATTEMPTS + 1):
            try:
                self._run_checked(command, timeout=timeout)
                return attempt
            except subprocess.TimeoutExpired as exc:
                # subprocess.run SIGKILLs nvidia-smi, but the reset it asked
                # the driver for keeps going: the outcome is unknown. A
                # TimeoutExpired would be classified retryable and the control
                # plane would resubmit, so convert it to a RuntimeError that
                # nothing retries. Recovery is a human decision (re-quiesce,
                # or reboot the node).
                raise RuntimeError(
                    f"gpu reset outcome unknown after {timeout}s; "
                    "refusing to retry automatically"
                ) from exc
            except RuntimeError as exc:
                if (
                    "In use by another client" not in str(exc)
                    or attempt == self._RESET_BUSY_ATTEMPTS
                ):
                    raise
                self.sleep(self.device_client_sample_interval_seconds)
                self._verify_no_clients(gpu_uuids)
        raise AssertionError("unreachable GPU reset retry state")

    def _reset_gpu_preflight(self, gpu_uuids: list[str]) -> None:
        """Everything a single-GPU reset can refuse on, before it resets.

        The executor runs this before it claims the quiesce window's single
        reset allowance. A config refusal, or a compute-app probe whose
        ``TimeoutExpired`` is retryable, must not spend that allowance: the
        resubmit the control plane sends would otherwise be refused for a
        reset that never ran. ``_reset_gpu`` runs it again immediately before
        nvidia-smi, so the check the reset relies on remains its own.
        """

        if not self.reset_enabled:
            raise RuntimeError("GPU reset is disabled by node configuration")
        if not self.single_gpu_reset_supported:
            raise RuntimeError(
                "single-GPU reset is not supported by this node "
                "topology; use RESET_ALL_GPUS_NVSWITCHES or reboot"
            )
        if not gpu_uuids:
            raise RuntimeError("GPU reset requires explicit GPU UUID targets")
        self._verify_no_clients(gpu_uuids)

    def _reset_gpu(self, gpu_uuids: list[str]) -> dict[str, Any]:
        self._reset_gpu_preflight(gpu_uuids)
        reset_attempts = 0
        for gpu_uuid in gpu_uuids:
            reset_attempts += self._run_reset_with_busy_retry(
                [
                    "nvidia-smi",
                    "--gpu-reset",
                    "-i",
                    gpu_uuid,
                ],
                gpu_uuids=[gpu_uuid],
                timeout=120,
            )
        return {
            "reset_gpu_uuids": gpu_uuids,
            "verified_no_gpu_clients": True,
            "reset_attempts": reset_attempts,
        }

    def _reset_all_preflight(self, gpu_uuids: list[str]) -> list[str]:
        """Verify a full fabric reset is allowed, and return the inventory.

        Same reason as :meth:`_reset_gpu_preflight`: the inventory query and
        the client probe can time out retryably, and neither has touched a
        GPU, so they run before the quiesce window's reset is claimed.
        """

        if not self.fabric_reset_enabled:
            raise RuntimeError(
                "full GPU/NVSwitch reset is disabled by node configuration"
            )
        requested = sorted(set(gpu_uuids))
        if not requested or len(requested) != len(gpu_uuids):
            raise RuntimeError(
                "full GPU/NVSwitch reset requires unique explicit GPU UUIDs"
            )
        local_inventory: list[str] = self._gpu_inventory()
        if requested != local_inventory:
            raise RuntimeError(
                "requested GPU inventory does not match local inventory: "
                f"requested={requested}; local={local_inventory}"
            )
        self._verify_no_clients(local_inventory)
        return local_inventory

    def _reset_all_gpus_nvswitches(self, gpu_uuids: list[str]) -> dict[str, Any]:
        local_inventory = self._reset_all_preflight(gpu_uuids)
        reset_attempts = self._run_reset_with_busy_retry(
            ["nvidia-smi", "--gpu-reset"],
            gpu_uuids=local_inventory,
            timeout=180,
        )
        observed_after = self._gpu_inventory()
        if observed_after != local_inventory:
            raise RuntimeError(
                "GPU inventory changed after full fabric reset: "
                f"before={local_inventory}; after={observed_after}"
            )
        return {
            "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
            "reset_gpu_uuids": local_inventory,
            "inventory_verified_before": True,
            "inventory_verified_after": True,
            "verified_no_gpu_clients": True,
            "command": ["nvidia-smi", "--gpu-reset"],
            "reset_attempts": reset_attempts,
        }

    def _restart_fabric_manager(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.fabric_manager_restart_enabled:
            raise RuntimeError(
                "Fabric Manager restart is disabled by node configuration"
            )
        self._verify_no_clients(
            command.gpu_uuids,
            include_device_clients=False,
        )
        service = "nvidia-fabricmanager"
        before = self._run_checked(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                service,
            ],
            timeout=30,
        ).stdout.strip()
        self._run_checked(
            ["systemctl", "restart", service],
            timeout=120,
        )
        self._run_checked(
            ["systemctl", "is-active", "--quiet", service],
            timeout=30,
        )
        after = self._run_checked(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                service,
            ],
            timeout=30,
        ).stdout.strip()
        return {
            "service": service,
            "active": True,
            "previous_main_pid": before or None,
            "current_main_pid": after or None,
        }
