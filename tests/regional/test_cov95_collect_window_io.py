"""Window probe command, filesystem and snapshot behavior on a fake host."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_window_probe as probe
from tests.regional import _cov95_collect_window as window_support
from tests.regional._cov95_collect_net import (
    forbidden,
    no_external_effects,  # noqa: F401
)
from tests.regional._cov95_collect_window import write_records

window_host = window_support.window_host


@pytest.mark.parametrize("cli_present", [True, False])
def test_snapshot_reports_only_safe_env_and_local_file_identity(
    window_host: Any, monkeypatch: Any, cli_present: bool
) -> None:
    host = window_host
    if not cli_present:
        probe.COLLECTOR_CLI.unlink()
    else:
        probe.ACCEPTANCE_ROOT.mkdir(parents=True)
        (probe.ACCEPTANCE_ROOT / "window-a").mkdir()
        (probe.ACCEPTANCE_ROOT / "not-a-window").touch()
    monkeypatch.setattr(sys, "argv", ["window-probe", "snapshot"])
    assert probe.main() == 0
    result = host.emitted[-1]
    assert result["boot_id"] == "boot-a"
    assert result["collector_env"]["GPU_FAULT_CLUSTER_ID"] == "cluster-a"
    assert result["collector_env"]["GPU_FAULT_CONTROL_PLANE_TOKEN_present"] is True
    assert "GPU_FAULT_CONTROL_PLANE_TOKEN" not in result["collector_env"], (
        "snapshot may expose credential presence but never its value"
    )
    assert result["gpu_inventory"] == [
        {
            "index": "0",
            "uuid": "GPU-aaaaaaaa",
            "pci_bus_id": "00000000:AF:00.0",
            "name": "NVIDIA H100",
        }
    ]
    assert result["outboxes"]["kernel"]["stats"] == (
        {"pending": 1} if cli_present else None
    )
    assert result["windows"] == (["window-a"] if cli_present else [])


@pytest.mark.parametrize("pid", ["0", "123"])
def test_kmsg_stream_handles_missing_links_and_unreadable_positions(
    window_host: Any, pid: str
) -> None:
    host = window_host
    host.timer_pid = pid
    root = host.paths("/proc/123")
    (root / "fd").mkdir(parents=True)
    (root / "fdinfo").mkdir()
    for name, target in {
        "1": "/dev/kmsg",
        "2": "/dev/null",
        "3": "/dev/kmsg",
        "4": "/dev/kmsg",
    }.items():
        (root / "fd" / name).symlink_to(target)
    (root / "fd" / "5").touch()
    (root / "fdinfo/1").write_text("flags: 0\npos: 17\n")
    (root / "fdinfo/3").write_text("pos: corrupt\n")
    result = probe.kmsg_stream_identity()
    assert result["pid"] == int(pid)
    assert result["kmsg_streams"] == (
        [{"fd": 1, "pos": 17}, {"fd": 3, "pos": None}, {"fd": 4, "pos": None}]
        if pid != "0"
        else []
    )


@pytest.mark.parametrize("output", ["\n", "[]", "malformed"])
def test_outbox_stats_empty_nonobject_and_invalid_json(
    window_host: Any, output: str
) -> None:
    window_host.stdout = output
    if output == "\n":
        assert probe.outbox_stats("kernel") == {}
    else:
        with pytest.raises((probe.ProbeError, json.JSONDecodeError)):
            probe.outbox_stats("kernel")


@pytest.mark.parametrize("missing", ["env", "cli", "collector"])
def test_missing_config_and_unknown_collector_fail_closed(
    window_host: Any, missing: str
) -> None:
    if missing == "env":
        probe.COLLECTOR_ENV.unlink()
    elif missing == "cli":
        probe.COLLECTOR_CLI.unlink()
    with pytest.raises(probe.ProbeError):
        if missing == "env":
            probe.parse_env()
        else:
            probe.outbox_stats("kernel" if missing == "cli" else "not-owned")


@pytest.mark.parametrize("action", ["stats", "list", "requeue-dead", "unknown"])
@pytest.mark.parametrize("failure", [False, True])
def test_outbox_cli_protocol_and_nonzero_exit(
    window_host: Any, action: str, failure: bool
) -> None:
    host = window_host
    host.cli_status = int(failure)
    record = {"path": "/p", "payload": ["marker-a", "marker-a"], "error": "x" * 300}
    write_records("kernel", [record, [], "not-json"])
    arguments = SimpleNamespace(collector="kernel", action=action, marker="marker-a")
    if failure or action == "unknown":
        with pytest.raises(probe.ProbeError):
            probe.outbox_command(arguments)
    else:
        probe.outbox_command(arguments)
        result = host.emitted[-1]
        assert result["collector"] == "kernel"
        if action == "list":
            assert result["records"][0]["marker_count"] == 2
            assert len(result["records"][0]["error"]) == 200
            assert result["records"][1:] == [{"malformed": True}, {"malformed": True}]
            assert "payload" not in result["records"][0], (
                "only record metadata may leave"
            )
        if action == "requeue-dead":
            assert result["refused_without_yes"] is True
            assert host.calls[-2][0][-1] == "--yes"
    assert probe.outbox_records("kernel", None)[0]["marker_count"] == 0
    assert probe.outbox_records("host", None) == []


@pytest.mark.parametrize("collector", probe.OUTBOX_COLLECTORS)
def test_seed_and_purge_use_inactive_owner_and_preserve_foreign_records(
    window_host: Any, collector: str
) -> None:
    host = window_host
    unit = {
        "kernel": probe.ALLOWED_UNITS[0],
        "host": probe.ALLOWED_UNITS[1],
        "dcgm": probe.ALLOWED_UNITS[2],
        "fabric-manager": probe.ALLOWED_UNITS[3],
    }[collector]
    arguments = SimpleNamespace(collector=collector, marker="marker-a")
    with pytest.raises(probe.ProbeError, match="active"):
        probe.seed_outbox_record(arguments)
    with pytest.raises(probe.ProbeError, match="active"):
        probe.purge_outbox_record(arguments)
    host.states[unit] = "inactive"
    probe.purge_outbox_record(arguments)
    assert host.emitted[-1]["removed"] == 0
    foreign = [
        "not-json",
        [],
        {"path": probe.RETIRED_CHANNEL_PATH, "payload": []},
        {"path": "/other", "payload": {"record_id": "acceptance-dead-letter-marker-a"}},
        {"path": probe.RETIRED_CHANNEL_PATH, "payload": {"record_id": "foreign"}},
    ]
    path = write_records(collector, foreign)
    before = path.read_bytes()
    probe.seed_outbox_record(arguments)
    added = json.loads(path.read_text().splitlines()[-1])
    assert added["payload"]["record_id"] == "acceptance-dead-letter-marker-a"
    assert added["payload"]["node_id"] == "node-a"
    assert added["replayable"] is True
    probe.purge_outbox_record(arguments)
    assert host.emitted[-1]["removed"] == 1
    assert host.emitted[-1]["remaining"] == len(foreign)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "operation", [probe.seed_outbox_record, probe.purge_outbox_record]
)
def test_outbox_mutations_refuse_unknown_owner(
    window_host: Any, operation: Any
) -> None:
    with pytest.raises(probe.ProbeError, match="not allow-listed"):
        operation(SimpleNamespace(collector="other", marker="marker-a"))


@pytest.mark.parametrize("output", ["\n", '{"sink_outcome":"accepted"}\n'])
def test_rejected_event_uses_node_environment_and_private_interpreter(
    window_host: Any, output: str
) -> None:
    host = window_host
    host.stdout = output
    arguments = SimpleNamespace(marker="marker-a", node_id="node-a")
    probe.post_rejected_event(arguments)
    command, options = host.calls[-1]
    assert command[0] == str(probe.VENV_PYTHON)
    assert command[-2:] == ["marker-a", "node-a"]
    assert options["timeout"] == 240
    assert options["env"]["GPU_FAULT_CLUSTER_ID"] == "cluster-a"
    assert host.emitted[-1]["synthetic_api_injection"] is True
    probe.VENV_PYTHON.unlink()
    with pytest.raises(probe.ProbeError, match="no collector venv"):
        probe.post_rejected_event(arguments)


@pytest.mark.parametrize("kind", ["xid63", "unparsed-xid"])
@pytest.mark.parametrize("fail_write", [False, True])
def test_kmsg_is_labelled_and_descriptor_closes_on_error(
    window_host: Any, monkeypatch: Any, kind: str, fail_write: bool
) -> None:
    host = window_host
    if fail_write:

        def write(fd: int, data: bytes) -> int:
            raise OSError("fixture write failure")

        monkeypatch.setattr(probe.os, "write", write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "window-probe",
            "write-kmsg",
            "--kind",
            kind,
            "--marker",
            "marker-a",
            "--pci-bdf",
            "00000000:AF:00.0",
        ],
    )
    assert probe.main() == int(fail_write)
    assert host.closed == [42]
    if fail_write:
        assert "OSError" in host.emitted[-1]["error"]
    else:
        assert b"user-space injection marker=marker-a" in host.writes[-1][1]
        assert host.emitted[-1]["pci_bdf"] == "0000:af:00"


def test_run_reports_nonzero_process_exit(window_host: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 8, "", " fixture failure "),
    )
    with pytest.raises(probe.ProbeError, match=r"failed \(8\).*fixture failure"):
        probe.run(["fake", "value with spaces"])


def test_stop_arms_recovery_first_and_start_verifies_active(
    window_host: Any, monkeypatch: Any
) -> None:
    host = window_host
    unit = probe.ALLOWED_UNITS[0]
    arguments = SimpleNamespace(unit=unit, run_id="stop-a", restore_seconds=300)
    probe.stop_unit(arguments)
    commands = [call[0] for call in host.calls]
    assert next(
        i for i, c in enumerate(commands) if c[0] == "systemd-run"
    ) < commands.index(["systemctl", "stop", unit])
    assert host.emitted[-1]["after"]["ActiveState"] == "inactive"
    probe.start_unit(arguments)
    assert host.emitted[-1]["after"]["ActiveState"] == "active"
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "inactive"})
    with pytest.raises(probe.ProbeError, match="did not start"):
        probe.start_unit(arguments)


def test_network_block_and_unblock_restore_all_owned_rules(
    window_host: Any, monkeypatch: Any
) -> None:
    host = window_host
    monkeypatch.setattr(
        sys, "argv", ["probe", "resolve", "--endpoint-host", "control.invalid"]
    )
    assert probe.main() == 0
    assert host.emitted[-1]["endpoint_ipv4"] == ["192.0.2.1"]
    args = SimpleNamespace(tag="test-net", ip=["192.0.2.1"], ttl_seconds=60)
    probe.block(args)
    assert host.rules["192.0.2.1"] == 1
    host.rules["192.0.2.1"] = 2
    probe.unblock(args)
    assert host.emitted[-1]["rules"] == []
    assert not host.paths("/run/test-net-cleanup.sh").exists(), (
        "remove owned cleanup script"
    )
    monkeypatch.setattr(probe.socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(probe.ProbeError, match="no IPv4"):
        probe.resolve_ipv4("control.invalid")

    def connect(*a: Any, **k: Any) -> Any:
        raise OSError("unreachable")

    monkeypatch.setattr(probe.socket, "create_connection", connect)
    assert probe.connectivity(["192.0.2.1"]) == {"192.0.2.1": False}
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    with pytest.raises(probe.ProbeError, match="not installed"):
        probe.firewall()


@pytest.mark.parametrize(
    "failure", ["no-ip", "existing-rule", "inactive-timer", "unsafe-tag"]
)
def test_block_admission_rejects_unsafe_or_unrecoverable_state(
    window_host: Any, monkeypatch: Any, failure: str
) -> None:
    host = window_host
    args = SimpleNamespace(tag="test-net", ip=["192.0.2.1"], ttl_seconds=60)
    if failure == "no-ip":
        args.ip = []
    elif failure == "existing-rule":
        host.rules["192.0.2.1"] = 1
    elif failure == "inactive-timer":
        monkeypatch.setattr(
            probe, "unit_state", lambda unit: {"ActiveState": "inactive"}
        )
    else:
        args.tag = "../not-owned"
    with pytest.raises(probe.ProbeError):
        probe.block(args)
    assert not any("-I" in call[0] for call in host.calls), (
        "admission failure cannot block traffic"
    )


def test_block_preserves_rule_that_appeared_after_initial_scan(
    window_host: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(probe, "rule_present", lambda *a: True)
    probe.block(SimpleNamespace(tag="test-net", ip=["192.0.2.1"], ttl_seconds=60))
    assert not any("-I" in call[0] for call in window_host.calls), (
        "a rule observed before insertion must not be duplicated"
    )
    monkeypatch.setattr(probe.os, "open", forbidden)
    with pytest.raises(probe.ProbeError, match="unsafe marker"):
        probe.write_kmsg(
            SimpleNamespace(marker="bad marker", pci_bdf="0000:af:00", kind="xid63")
        )
