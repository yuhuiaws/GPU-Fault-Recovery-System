"""Explicit local platform checks; not part of default pytest discovery.

Run through make collector-platform-check. No GPU, cloud, production service,
host reboot or network access is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe


def command(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    runtime = f"/run/user/{os.getuid()}"
    return subprocess.run(
        arguments,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "XDG_RUNTIME_DIR": runtime,
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
            "LC_ALL": "C",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=check,
    )


@pytest.mark.parametrize("restart_during_read", [False, True])
@pytest.mark.allows_cluster_binaries("systemctl")
def test_real_user_systemd_process_configuration_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart_during_read: bool
) -> None:
    assert os.geteuid() != 0, "run the isolated service check as a non-root user"
    assert (
        command(["systemctl", "--user", "is-system-running"]).stdout.strip()
        == "running"
    )
    unit = f"gpu-fault-inventory-check-{uuid4().hex}.service"
    description = f"isolated inventory check {unit}"
    config = tmp_path / "collector.env"
    values = {
        "GPU_FAULT_EXPECTED_GPU_COUNT": "8",
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": "16",
        "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": "2",
    }
    config.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    config.chmod(0o600)

    def read_unit() -> dict[str, str]:
        result = command(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=LoadState,Description,Transient",
            ],
            check=False,
        )
        return dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )

    assert read_unit().get("LoadState") == "not-found", "test unit must not preexist"
    attempted = False
    try:
        attempted = True
        command(
            [
                "systemd-run",
                "--user",
                "--quiet",
                "--collect",
                f"--unit={unit}",
                f"--description={description}",
                "--property=Type=simple",
                "--property=RuntimeMaxSec=60",
                "--property=NoNewPrivileges=yes",
                f"--property=EnvironmentFile={config}",
                "/usr/bin/sleep",
                "50",
            ]
        )
        monkeypatch.setattr(probe, "HOST_COLLECTOR_UNIT", unit)
        monkeypatch.setattr(probe, "COLLECTOR_ENV", config)
        monkeypatch.setattr(probe, "PROC_ROOT", Path("/proc"))

        def read_command(
            arguments: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            assert arguments[:3] == ["systemctl", "show", unit], (
                "the probe may only read its isolated service"
            )
            return command(["systemctl", "--user", *arguments[1:]])

        monkeypatch.setattr(probe, "run", read_command)
        deadline = time.monotonic() + 10
        while True:
            state = probe.collector_unit_state(unit)
            if state["ActiveState"] == "active" and int(state["MainPID"]) > 1:
                break
            assert time.monotonic() < deadline, "isolated service did not become active"
            time.sleep(0.1)
        reads = 0

        def snapshots() -> dict[str, Any]:
            nonlocal reads
            reads += 1
            if restart_during_read and reads == 2:
                command(["systemctl", "--user", "restart", unit])
            return {unit: probe.collector_unit_state(unit)}

        monkeypatch.setattr(probe, "service_snapshot", snapshots)
        if restart_during_read:
            with pytest.raises(probe.ProbeError, match="changed during read"):
                probe.inventory_configuration()
        else:
            result = probe.inventory_configuration()
            assert result["running_env"] == result["file_env"] == values
            assert result["pid"] == int(state["MainPID"])
    finally:
        if attempted:
            current = read_unit()
            if current.get("LoadState") != "not-found":
                assert current.get("Description") == description
                assert current.get("Transient") == "yes"
                command(["systemctl", "--user", "stop", unit])
            assert read_unit().get("LoadState") == "not-found"


@pytest.mark.allows_cluster_binaries("docker")
def test_real_read_only_container_mount_namespace(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    source = tmp_path / "collector_node_probe.py"
    shutil.copyfile(Path(probe.__file__), source)
    source.chmod(0o644)
    (tmp_path / "runtime").mkdir()
    image = command(
        ["docker", "image", "inspect", "python:3.12.14-slim", "--format={{.Id}}"]
    ).stdout.strip()
    assert image.startswith("sha256:") and len(image) == 71
    name = f"gpu-fault-mount-check-{uuid4().hex}"
    program = (
        "import importlib.util,json,os,pathlib\n"
        "spec=importlib.util.spec_from_file_location('probe','/probe/collector_node_probe.py')\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "target=pathlib.Path('/probe/runtime')\n"
        "device=target.stat().st_dev\n"
        "result=module.collector_runtime_mount(target,device)\n"
        "print(json.dumps({'target':result['target'],"
        "'device_bound':result['maj:min']==f'{os.major(device)}:{os.minor(device)}'}))\n"
    )
    try:
        result = command(
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--name",
                name,
                "--label",
                f"gpu-fault-platform-check={name}",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=32",
                "--memory=128m",
                "--cpus=1",
                "--mount",
                f"type=bind,src={tmp_path},dst=/probe,readonly",
                image,
                "python",
                "-I",
                "-S",
                "-B",
                "-c",
                program,
            ]
        )
        assert json.loads(result.stdout) == {"target": "/probe", "device_bound": True}
    finally:
        remaining = command(
            [
                "docker",
                "container",
                "inspect",
                name,
                "--format={{json .Config.Labels}}",
            ],
            check=False,
        )
        if remaining.returncode == 0:
            assert json.loads(remaining.stdout).get("gpu-fault-platform-check") == name
            command(["docker", "rm", "--force", name])
