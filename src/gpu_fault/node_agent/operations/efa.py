from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class EfaOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    infiniband_root: Any
    runner: Callable[..., Any]

    def _capture_efa_rdma_state(
        self,
        work_dir: Path,
        command: NodeActionCommand,
        manifest: dict[str, Any],
    ) -> None:
        if command.parameters.get("diagnostic_reason") != "EFA_TRAFFIC_HUNG_SUSPECTED":
            return
        snapshot: dict[str, Any] = {
            "root": str(self.infiniband_root),
            "devices": {},
        }
        interfaces: set[str] = set()
        try:
            if not self.infiniband_root.exists():
                raise FileNotFoundError(self.infiniband_root)
            for device in sorted(
                self.infiniband_root.iterdir(),
                key=lambda item: item.name,
            ):
                if not device.is_dir():
                    continue
                network_interfaces = sorted(
                    item.name for item in (device / "device" / "net").glob("*")
                )
                interfaces.update(network_interfaces)
                device_data: dict[str, Any] = {
                    "network_interfaces": network_interfaces,
                    "ports": {},
                }
                for port in sorted(
                    (device / "ports").glob("*"),
                    key=lambda item: item.name,
                ):
                    port_data: dict[str, Any] = {}
                    for name in ("state", "phys_state"):
                        path = port / name
                        if path.exists():
                            try:
                                port_data[name] = path.read_text(
                                    encoding="ascii",
                                    errors="replace",
                                ).strip()
                            except OSError as exc:
                                port_data[name] = {
                                    "error": (f"{type(exc).__name__}: {exc}")
                                }
                    for group in (
                        "counters",
                        "hw_counters",
                    ):
                        values: dict[str, Any] = {}
                        for path in sorted(
                            (port / group).glob("*"),
                            key=lambda item: item.name,
                        ):
                            if not path.is_file():
                                continue
                            try:
                                values[path.name] = path.read_text(
                                    encoding="ascii",
                                    errors="replace",
                                ).strip()
                            except OSError as exc:
                                values[path.name] = {
                                    "error": (f"{type(exc).__name__}: {exc}")
                                }
                        port_data[group] = values
                    device_data["ports"][port.name] = port_data
                snapshot["devices"][device.name] = device_data
            returncode = 0
            error = None
        except OSError as exc:
            returncode = None
            error = f"{type(exc).__name__}: {exc}"
            snapshot["error"] = error
        snapshot_path = work_dir / "infiniband-counters.json"
        snapshot_path.write_text(
            json.dumps(snapshot, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        capture: dict[str, Any] = {
            "file": snapshot_path.name,
            "source": str(self.infiniband_root),
            "returncode": returncode,
        }
        if error:
            capture["error"] = error
        manifest["captures"].append(capture)

        for interface in sorted(interfaces):
            safe_interface = re.sub(r"[^A-Za-z0-9_.-]", "_", interface)
            filename = f"ethtool-{safe_interface}-statistics.txt"
            argv = ["ethtool", "-S", interface]
            try:
                completed = self.runner(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                output = completed.stdout or ""
                if completed.stderr:
                    output += "\n[stderr]\n" + completed.stderr
                (work_dir / filename).write_text(
                    output,
                    encoding="utf-8",
                    errors="replace",
                )
                manifest["captures"].append(
                    {
                        "file": filename,
                        "command": argv,
                        "returncode": completed.returncode,
                    }
                )
            except (
                OSError,
                subprocess.TimeoutExpired,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
                (work_dir / filename).write_text(error, encoding="utf-8")
                manifest["captures"].append(
                    {
                        "file": filename,
                        "command": argv,
                        "returncode": None,
                        "error": error,
                    }
                )

    def _efa_counter_snapshot(self) -> dict[str, int]:
        result = {}
        try:
            paths = self.infiniband_root.glob("*/ports/*/hw_counters/*")
            for path in paths:
                name = path.name.lower()
                if not any(
                    token in name
                    for token in (
                        "rx",
                        "tx",
                        "read",
                        "write",
                        "send",
                        "recv",
                        "rcv",
                        "xmit",
                    )
                ):
                    continue
                try:
                    result[str(path)] = int(path.read_text().strip())
                except (OSError, ValueError):
                    continue
        except OSError:
            return {}
        return dict(list(sorted(result.items()))[:128])

    @staticmethod
    def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, Any]:
        values = {
            key: max(0, after[key] - before.get(key, after[key])) for key in after
        }
        return {
            "counter_deltas": values,
            "total_delta": sum(values.values()),
        }
