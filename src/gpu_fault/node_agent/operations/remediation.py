from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)

if TYPE_CHECKING:
    from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager


class RemediationOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    driver_remediation_command: tuple[str, ...]

    _run_checked: Callable[..., Any]
    _verify_no_clients: Callable[..., Any]
    driver_remediation_enabled: bool
    driver_remediation_sha256: str | None
    efa_driver_bind_path: Path
    efa_driver_module: str
    efa_driver_remediation_enabled: bool
    efa_pci_devices_root: Path
    firmware_update_command: tuple[str, ...]
    firmware_update_enabled: bool
    firmware_update_sha256: str | None
    firmware_verify_command: tuple[str, ...]
    firmware_verify_sha256: str | None
    quiesce_manager: GpuServiceQuiesceManager | None
    service_quiesce_enabled: bool
    target_driver_branch: int | None
    target_firmware_version: str | None

    @staticmethod
    def _validate_remediation_config(
        name: str,
        command: tuple[str, ...],
        expected_sha256: str | None,
        target: str,
    ) -> None:
        if not command or not Path(command[0]).is_absolute():
            raise ValueError(f"{name} remediation command must be absolute")
        if not target:
            raise ValueError(f"{name} remediation target is required")
        if not expected_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError(f"{name} remediation SHA-256 is required")
        executable = Path(command[0])
        if not executable.is_file():
            raise ValueError(f"{name} remediation executable does not exist")
        actual = hashlib.sha256(executable.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
            raise ValueError(f"{name} remediation executable SHA-256 mismatch")

    @staticmethod
    def _render_remediation_command(command: tuple[str, ...], target: str) -> list[str]:
        return [item.replace("{target}", target) for item in command]

    def _assert_remediation_quiesced(self, incident_id: str) -> None:
        if not self.service_quiesce_enabled or self.quiesce_manager is None:
            raise RuntimeError("driver/firmware remediation requires service quiesce")
        # for_reset=False: remediation runs *under* the quiesce but is not a
        # GPU reset, so it must not consume the window's single reset slot
        # (driver then firmware in one window is legitimate).
        self.quiesce_manager.assert_quiesced(
            incident_id=incident_id,
            for_reset=False,
        )

    def _remediate_driver(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.driver_remediation_enabled:
            raise RuntimeError("driver remediation is disabled")
        target = command.parameters.get("target_driver_branch")
        if target != self.target_driver_branch:
            raise RuntimeError("driver remediation target does not match node config")
        self._assert_remediation_quiesced(command.incident_id)
        self._verify_no_clients(command.gpu_uuids)
        rendered = self._render_remediation_command(
            self.driver_remediation_command, str(target)
        )
        self._run_checked(rendered, timeout=1800)
        versions = self._run_checked(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        ).stdout.splitlines()
        branches = {
            int(value.strip().split(".", 1)[0])
            for value in versions
            if value.strip().split(".", 1)[0].isdigit()
        }
        if branches != {target}:
            raise RuntimeError(
                "driver branch verification failed: "
                + ",".join(str(item) for item in sorted(branches))
            )
        return {
            "target_driver_branch": target,
            "verified_driver_branches": sorted(branches),
            "command_sha256": self.driver_remediation_sha256,
        }

    @staticmethod
    def _is_efa_pci_device(device: Path) -> bool:
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            device_id = (device / "device").read_text().strip().lower()
        except OSError:
            return False
        return vendor == "0x1d0f" and device_id.startswith("0xefa")

    @staticmethod
    def _bound_driver(device: Path) -> str | None:
        try:
            return (device / "driver").resolve(strict=True).name
        except OSError:
            return None

    def _remediate_efa_driver(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.efa_driver_remediation_enabled:
            raise RuntimeError("EFA driver remediation is disabled")
        expected = int(command.parameters.get("expected_count", 1))
        devices = sorted(
            (
                device
                for device in self.efa_pci_devices_root.iterdir()
                if device.is_dir() and self._is_efa_pci_device(device)
            ),
            key=lambda item: item.name,
        )
        if len(devices) < expected:
            raise RuntimeError(
                "EFA PCI inventory is incomplete; driver remediation "
                f"cannot recover missing devices: {len(devices)}/{expected}"
            )
        conflicting = {
            device.name: driver
            for device in devices
            if (driver := self._bound_driver(device))
            not in {None, self.efa_driver_module}
        }
        if conflicting:
            raise RuntimeError(
                "EFA PCI functions are bound to unexpected drivers: "
                + ", ".join(
                    f"{bdf}={driver}" for bdf, driver in sorted(conflicting.items())
                )
            )
        before = {device.name: self._bound_driver(device) for device in devices}
        if all(driver == self.efa_driver_module for driver in before.values()):
            return {
                "expected_count": expected,
                "pci_discovered_count": len(devices),
                "driver_bound_count": len(devices),
                "already_bound": True,
            }
        self._run_checked(["modprobe", self.efa_driver_module], timeout=30)
        rebound = []
        bind_errors: dict[str, str] = {}
        for device in devices:
            if self._bound_driver(device) == self.efa_driver_module:
                continue
            try:
                self.efa_driver_bind_path.write_text(
                    device.name + "\n", encoding="ascii"
                )
                rebound.append(device.name)
            except OSError as exc:
                bind_errors[device.name] = f"{type(exc).__name__}: {exc}"
        bound = [
            device.name
            for device in devices
            if self._bound_driver(device) == self.efa_driver_module
        ]
        if bind_errors or len(bound) < expected:
            error_details = (
                "; bind_errors="
                + ",".join(
                    f"{bdf}={error}" for bdf, error in sorted(bind_errors.items())
                )
                if bind_errors
                else ""
            )
            raise RuntimeError(
                "EFA driver bind verification failed: "
                f"{len(bound)}/{expected}{error_details}"
            )
        return {
            "expected_count": expected,
            "pci_discovered_count": len(devices),
            "driver_bound_count": len(bound),
            "rebound_pci_bdfs": rebound,
            "already_bound": False,
        }

    def _update_firmware(self, command: NodeActionCommand) -> dict[str, Any]:
        if not self.firmware_update_enabled:
            raise RuntimeError("firmware update is disabled")
        target = command.parameters.get("target_firmware_version")
        if target != self.target_firmware_version:
            raise RuntimeError("firmware target does not match node config")
        self._assert_remediation_quiesced(command.incident_id)
        self._verify_no_clients(command.gpu_uuids)
        rendered = self._render_remediation_command(
            self.firmware_update_command, str(target)
        )
        self._run_checked(rendered, timeout=1800)
        verify = self._run_checked(
            self._render_remediation_command(self.firmware_verify_command, str(target)),
            timeout=120,
        ).stdout.strip()
        if verify != target:
            raise RuntimeError(
                "firmware version verification failed: " + (verify or "empty output")
            )
        return {
            "target_firmware_version": target,
            "verified_firmware_version": verify,
            "command_sha256": self.firmware_update_sha256,
            "verification_command_sha256": (self.firmware_verify_sha256),
        }
