"""Node probe parsing and syscall boundaries against a fully temporary host."""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.collectors.gpu import discovery
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional import _cov95_collect_node as node_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional._cov95_collect_node import process

node_host = node_support.node_host


@pytest.mark.parametrize("command", ["snapshot", "fm-cursor", "efa-inventory"])
def test_default_read_entrypoints_use_local_facts(
    node_host: Any, monkeypatch: Any, command: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["collector-probe", command])
    assert probe.main() == 0
    result = node_host.emitted[-1]
    assert result["captured_at"], "snapshots must carry a capture timestamp"
    if command == "snapshot":
        assert result["boot_id"] == "boot-a"
        assert result["collector_env"] == {
            "GPU_FAULT_EXPECTED_GPU_COUNT": "2",
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
        }
        assert result["gpu_inventory"][0]["pci_bdf"] == "0000:af:00.0"
        assert result["gpu_power"][0]["power_limit_w"] == 300
    if command == "fm-cursor":
        assert result["fabric_manager_cursor"] is None
    if command == "efa-inventory":
        assert result["efa_inventory"]["discovered_count"] == 0


@pytest.mark.parametrize(
    "output", ["Enabled\n\n", "Disabled\n", "", "Enabled\nDisabled\n", "N/A"]
)
def test_persistence_mode_requires_a_uniform_known_state(
    node_host: Any, output: str
) -> None:
    node_host.persistence = output
    assert probe.persistence_mode() is (
        True if output == "Enabled\n\n" else False if output == "Disabled\n" else None
    )


def test_service_and_inventory_parsers_skip_unknown_and_malformed_rows(
    node_host: Any,
) -> None:
    node_host.unit_status["kubelet.service"] = (1, "loaded", "active")
    node_host.unit_status["nvidia-persistenced.service"] = (0, "not-found", "inactive")
    services = probe.service_snapshot()
    assert "kubelet.service" not in services, "failed command is not service proof"
    assert "nvidia-persistenced.service" not in services, (
        "missing service cannot be observed"
    )
    node_host.gpu_stdout += "1, GPU-bbbbbbbb, unknown, NVIDIA H100\n"
    assert probe.gpu_inventory()[1]["pci_bdf"] == "unknown"
    node_host.power_stdout = "bad-row\n0, GPU-aaaaaaaa, N/A, 300, 100, 300, 90\n"
    power = probe.gpu_power_state()
    assert power[0]["power_draw_w"] is None
    assert power[0]["utilization_percent"] == 90


@pytest.mark.parametrize("kind", ["missing", "directory", "large", "file"])
def test_file_snapshot_hashes_only_bounded_regular_files(
    node_host: Any, tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "snapshot"
    if kind == "directory":
        path.mkdir()
    elif kind == "large":
        path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    elif kind == "file":
        path.write_text("record")
    value = probe.file_snapshot(path)
    if kind == "missing":
        assert value is None
    else:
        assert value["inode"] == path.stat().st_ino
        assert bool(value["sha256"]) is (kind == "file")


@pytest.mark.parametrize(
    "content",
    [
        '{"files":{"log":{"offset":12}},"journal_cursor":"cursor"}',
        "[]",
        '{"files":[]}',
        "{bad",
    ],
)
def test_fm_cursor_handles_missing_malformed_and_nonobject_state(
    node_host: Any, content: str
) -> None:
    target = probe.FM_STATE_CANDIDATES[1]
    target.write_text(content)
    observed = probe.fabric_manager_cursor()
    assert observed["path"] == str(target)
    if content == "{bad":
        assert observed["error"] == "state file is not JSON"
    elif '"offset"' in content:
        assert observed["files"] == {"log": {"offset": 12}}
        assert observed["journal_cursor"] == "cursor"
    else:
        assert observed["files"] == {}


def test_efa_inventory_reads_only_bound_device_and_active_ports(node_host: Any) -> None:
    host = node_host
    root = host.paths("/sys/class/infiniband")
    root.mkdir(parents=True)
    device = root / "efa0"
    device.mkdir()
    pci = host.paths("/sys/bus/pci/devices/0000:af:00.0")
    pci.mkdir(parents=True)
    (device / "device").symlink_to(pci)
    (pci / "driver").symlink_to(host.paths("/sys/bus/pci/drivers/efa"))
    (pci / "uevent").write_text("DRIVER=efa\n")
    for name, state, physical in [
        ("1", "4: ACTIVE", None),
        ("2", "3: ARMED", "5: LinkUp"),
        ("3", "4: ACTIVE", "3: Polling"),
        ("4", None, None),
    ]:
        port = device / "ports" / name
        port.mkdir(parents=True)
        if state is not None:
            (port / "state").write_text(state)
        if physical is not None:
            (port / "phys_state").write_text(physical)
    other = root / "other0"
    other.mkdir()
    (other / "device").mkdir()
    (other / "device/uevent").write_text("DRIVER=other\n")
    observed = probe.efa_inventory()
    assert observed["discovered_count"] == 1
    assert observed["active_count"] == 1
    assert observed["devices"][0]["active_ports"] == ["1"]
    assert observed["devices"][0]["pci_bdf"] == "0000:af:00.0"


def test_efa_inventory_tolerates_vanished_sysfs_links(
    node_host: Any, monkeypatch: Any
) -> None:
    root = node_host.paths("/sys/class/infiniband")
    (root / "efa0/device").mkdir(parents=True)
    (root / "efa0/device/uevent").write_text("DRIVER=efa\n")
    original = Path.resolve

    def resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path.is_relative_to(root):
            raise OSError("device disappeared")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    observed = probe.efa_inventory()
    assert observed["devices"] == [
        {
            "name": "efa0",
            "pci_bdf": None,
            "driver": "uevent",
            "active_ports": [],
            "active": False,
        }
    ]


@pytest.mark.parametrize("pid", ["0", "123"])
def test_kernel_fd_inspection_skips_disappeared_and_non_kmsg_fds(
    node_host: Any, monkeypatch: Any, pid: str
) -> None:
    root = node_host.paths("/proc/123/fd")
    root.mkdir(parents=True)
    (root / "3").symlink_to("/dev/kmsg")
    (root / "4").symlink_to("/dev/null")
    (root / "5").touch()
    monkeypatch.setattr(
        probe,
        "service_snapshot",
        lambda: {"gpu-fault-kernel-collector.service": {"MainPID": pid}},
    )
    assert probe.kernel_collector_fd() == {
        "pid": int(pid),
        "kmsg_fds": ["3"] if pid == "123" else [],
    }


def test_workload_pid_selection_excludes_foreign_and_sandbox_processes(
    node_host: Any,
) -> None:
    root = node_host.paths("/proc")
    for pid in (1, 555, 9):
        process(root, pid, cgroup="kubepods/poduid-a")
    process(root, 10, cgroup="kubepods/poduid_a", comm="python")
    process(root, 11, cgroup="kubepods/poduid-a", comm=None)
    process(root, 12, cgroup="kubepods/poduid-a", comm="pause")
    process(root, 13, cgroup="kubepods/other", comm="python")
    (root / "14").mkdir()
    assert probe.workload_processes("uid-a", proc=root, excluded_pids=[9]) == [
        {"pid": 10, "comm": "python"},
        {"pid": 11, "comm": ""},
    ]
    with pytest.raises(probe.ProbeError, match="unsafe pod UID"):
        probe.workload_processes("../unsafe", proc=root)


def test_workload_kill_reports_exit_race_without_host_signals(node_host: Any) -> None:
    root = node_host.paths("/proc")
    process(root, 10, cgroup="poduid-a")
    process(root, 11, cgroup="poduid-a")
    calls: list[Any] = []

    def kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        if pid == 11:
            raise ProcessLookupError("exited")

    probe.kill_workload(SimpleNamespace(pod_uid="uid-a"), proc=root, kill=kill)
    assert calls == [(10, signal.SIGKILL), (11, signal.SIGKILL)]
    assert node_host.emitted[-1]["killed"] == [{"pid": 10, "comm": "python"}]
    assert node_host.emitted[-1]["already_exited"] == [{"pid": 11, "comm": "python"}]
    with pytest.raises(probe.ProbeError, match="no process found"):
        probe.kill_workload(SimpleNamespace(pod_uid="unknown"), proc=root, kill=kill)


@pytest.mark.parametrize("cuda", ["12.4", None])
def test_gpu_identity_requires_known_cuda(
    node_host: Any, monkeypatch: Any, cuda: Any
) -> None:
    monkeypatch.setattr(discovery, "discover_gpu_product", lambda: "H100")
    monkeypatch.setattr(
        discovery, "discover_gpu_software_versions", lambda: ("550", cuda)
    )
    if cuda is None:
        with pytest.raises(probe.ProbeError, match="CUDA version"):
            probe.gpu_identity(SimpleNamespace())
    else:
        probe.gpu_identity(SimpleNamespace())
        assert node_host.emitted[-1] == {
            "product": "H100",
            "driver_branch": "550",
            "cuda_version": cuda,
        }


@pytest.mark.parametrize(
    "mode",
    ["valid", "inactive", "pid", "invocation", "changed-service", "changed-process"],
)
def test_firmware_premise_binds_service_and_process_start_identity(
    node_host: Any, monkeypatch: Any, mode: str
) -> None:
    root = process(node_host.paths("/proc"), 123)
    stat = root / "stat"
    stat.write_text("123 (node-agent) " + " ".join(["0"] * 20))
    (root / "environ").write_bytes(
        b"ignored\0GPU_FAULT_NODE_ALLOW_FIRMWARE_UPDATE=false\0"
        b"GPU_FAULT_TARGET_FIRMWARE_VERSION=fixture\0"
        b"GPU_FAULT_FIRMWARE_UPDATE_COMMAND=fixture\0"
        b"GPU_FAULT_FIRMWARE_VERIFY_COMMAND=fixture\0"
    )
    identity = {
        "ActiveState": "active",
        "MainPID": "123",
        "InvocationID": "invocation-a",
    }
    if mode == "inactive":
        identity["ActiveState"] = "inactive"
    if mode == "pid":
        identity["MainPID"] = "not-a-pid"
    if mode == "invocation":
        identity["InvocationID"] = ""
    calls: list[int] = []

    def service() -> dict[str, Any]:
        calls.append(1)
        value = dict(identity)
        if len(calls) == 2:
            if mode == "changed-service":
                value["InvocationID"] = "another"
            if mode == "changed-process":
                stat.write_text("123 (node-agent) " + " ".join(["1"] * 20))
        return {"gpu-fault-node-agent.service": value}

    monkeypatch.setattr(probe, "service_snapshot", service)
    if mode != "valid":
        with pytest.raises(probe.ProbeError):
            probe.firmware_premise(SimpleNamespace())
    else:
        probe.firmware_premise(SimpleNamespace())
        assert node_host.emitted[-1] == {
            "pid": 123,
            "invocation_id": "invocation-a",
            "start_ticks": "0",
            "allow_disabled": True,
            "target_present": True,
            "update_command_present": True,
            "verify_command_present": True,
        }


def test_reset_audit_reads_only_allowlisted_ledger_fields_from_local_sqlite(
    node_host: Any,
) -> None:
    with pytest.raises(probe.ProbeError, match="ledger is missing"):
        probe.reset_audit(SimpleNamespace())
    with sqlite3.connect(probe.NODE_LEDGER) as connection:
        connection.execute("""
            CREATE TABLE results (
                command_id TEXT, attempt INTEGER, state TEXT, operation TEXT,
                started_at TEXT, completed_at TEXT, incident_id TEXT,
                workflow_request_id TEXT, fencing_token INTEGER, gpu_uuids TEXT,
                parameters_digest TEXT, signature_digest TEXT, payload TEXT
            )
        """)
        for index in range(2):
            connection.execute(
                "INSERT INTO results VALUES (?,1,'COMPLETE','RESET_GPU','a','b','i','w',1,?,?,?,?)",
                (
                    f"command-{index}",
                    '["GPU-aaaaaaaa"]' if index else None,
                    "parameters",
                    "present" if index else None,
                    json.dumps(
                        {
                            "status": "SUCCEEDED",
                            "details": {
                                "reset_gpu_uuids": ["GPU-aaaaaaaa"],
                                "unrelated": "not-exported",
                            },
                        }
                    ),
                ),
            )
    probe.reset_audit(SimpleNamespace())
    rows = node_host.emitted[-1]["ledger"]
    assert len(rows) == 2
    assert rows[0]["gpu_uuids"] is None
    assert rows[1]["gpu_uuids"] == ["GPU-aaaaaaaa"]
    assert rows[0]["signature_digest_present"] is False
    assert rows[1]["signature_digest_present"] is True
    assert "signature_digest" not in rows[1], (
        "signature material cannot leave the probe"
    )
    assert "unrelated" not in rows[1]["result"]["details"], (
        "export only audited result fields"
    )


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_persistence_change_and_restart_commands_are_checked(
    node_host: Any, monkeypatch: Any, enabled: str
) -> None:
    probe.set_persistence_mode(SimpleNamespace(enabled=enabled))
    assert node_host.calls[-1][0] == [
        "nvidia-smi",
        "-pm",
        "1" if enabled == "true" else "0",
    ]
    service = probe.HOST_COLLECTOR_UNIT
    probe.restart_service(SimpleNamespace(service=service))
    assert node_host.emitted[-1]["after"]["ActiveState"] == "active"
    node_host.unit_status[service] = (0, "loaded", "inactive")
    with pytest.raises(probe.ProbeError, match="not active"):
        probe.restart_service(SimpleNamespace(service=service))
    with pytest.raises(probe.ProbeError, match="not allowlisted"):
        probe.restart_service(SimpleNamespace(service="foreign.service"))
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 8, "", " fixture-error "),
    )
    with pytest.raises(probe.ProbeError, match=r"failed \(8\).*fixture-error"):
        probe.run(["fake"])


@pytest.mark.parametrize("include_switch", [False, True])
def test_sxid_appends_only_to_private_log_and_rejects_unknown_codes(
    node_host: Any, include_switch: bool
) -> None:
    args = probe.parser().parse_args(
        [
            "append-sxid",
            "--sxid",
            "10003",
            "--marker",
            "marker-a",
            "--pci-bdf",
            "0000:af:00",
        ]
    )
    args.include_switch = include_switch
    probe.append_sxid(args)
    text = probe.FM_LOG.read_text()
    assert ("nvidia-nvswitch0:" in text) is include_switch
    assert "SXid (PCI:0000:af:00.0): 10003" in text
    assert node_host.emitted[-1]["size"] == len(text.encode())
    args.sxid = 0
    with pytest.raises(probe.ProbeError, match="not allowlisted"):
        probe.append_sxid(args)


@pytest.mark.parametrize("restore", ["59", "901", "60"])
def test_main_rejects_out_of_window_action_before_execution(
    node_host: Any, monkeypatch: Any, restore: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "throttle-gpu", "--run-id", "test", "--restore-seconds", restore],
    )
    assert probe.main() == 1
    if restore != "60":
        assert node_host.calls == [], (
            "invalid TTL must stop before host reads or changes"
        )
    else:
        assert "must end before" in node_host.emitted[-1]["error"]
