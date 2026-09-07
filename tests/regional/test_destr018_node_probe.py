"""Unit tests for the GF-REGIONAL-DESTR-018 on-node GPU device holder.

Everything the probe can do on a live node is behind three refusals: a device
allow-list, a systemctl verb/unit allow-list that does *not* include the Node
Agent unit, and an arm-trigger allow-list that excludes
``VERIFY_NO_GPU_CLIENTS``. The last one is not cosmetic: arming after a
successful VERIFY moves the workflow on to ``RESET_GPU``, whose client check has
no WAITING branch, so the step fails non-retryably and the workflow escalates to
a node reboot the case never authorized.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.ledger import NodeActionLedger
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
)
from scripts.e2e.regional.probes import destr018_node_probe as probe

QUIESCE = "QUIESCE_GPU_SERVICES"
VERIFY = "VERIFY_NO_GPU_CLIENTS"
RUN_ID = "destr018-run-a1"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


def _result(
    command_id: str,
    operation: WorkflowOperation,
    status: NodeActionStatus,
    completed_at: datetime,
) -> NodeActionResult:
    return NodeActionResult(
        command_id=command_id,
        operation=operation,
        status=status,
        error=None if status is NodeActionStatus.SUCCEEDED else "clients still active",
        retryable=status is not NodeActionStatus.SUCCEEDED,
        attempt=1,
        completed_at=completed_at,
    )


def _ledger(tmp_path: Path) -> tuple[Path, NodeActionLedger]:
    path = tmp_path / "node-actions.db"
    return path, NodeActionLedger(str(path))


def _command(
    command_id: str, operation: WorkflowOperation, issued_at: datetime
) -> NodeActionCommand:
    return NodeActionCommand(
        command_id=command_id,
        workflow_request_id="wf-new",
        incident_id="inc-1",
        fencing_token=1,
        operation=operation,
        node_id="node-b",
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
    )


def _record(
    ledger: NodeActionLedger,
    command_id: str,
    operation: WorkflowOperation,
    status: NodeActionStatus,
    completed_at: datetime,
) -> None:
    """Mark in progress and then save, the way the Agent executor does.

    ``NodeActionLedger.save`` never writes ``started_at``; only
    ``mark_in_progress`` does, and ``node_agent/executor.py`` calls it on every
    execution it owns. A verdict that reads ``started_at`` therefore has to be
    tested against rows written in that order, not against ``save`` alone.
    """

    ledger.mark_in_progress(_command(command_id, operation, completed_at), 1)
    ledger.save(_result(command_id, operation, status, completed_at))


def _fake_proc(root: Path, holders: dict[str, tuple[str, str]]) -> Path:
    """A ``/proc`` tree: ``{pid: (comm, device the pid holds)}``."""

    for pid, (comm, device) in holders.items():
        process = root / pid
        (process / "fd").mkdir(parents=True)
        (process / "comm").write_text(comm + "\n", encoding="utf-8")
        (process / "fd" / "3").symlink_to(device)
    (root / "self").mkdir()
    (root / "uptime").write_text("1 1\n", encoding="utf-8")
    return root


def test_ledger_rows_read_the_real_schema(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _record(
        ledger,
        "wf-old/3/QUIESCE_GPU_SERVICES/commit",
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.SUCCEEDED,
        T0 - timedelta(hours=1),
    )
    _record(
        ledger,
        f"wf-new/4/{VERIFY}/node-a/attempt-1",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.FAILED,
        T0 + timedelta(seconds=20),
    )

    rows = probe.ledger_rows(path)

    assert [row["operation"] for row in rows] == [QUIESCE, VERIFY]
    assert rows[0]["state"] == "SUCCEEDED"
    assert rows[1]["state"] == "FAILED"
    assert rows[1]["attempt"] == 1
    assert rows[1]["completed_at"].startswith("2026-09-06T10:00:20"), rows[1]
    assert rows[1]["started_at"] is not None, (
        "the straddling-row verdict needs started_at, so the probe must read it"
    )


def test_ledger_rows_are_empty_without_a_ledger(tmp_path: Path) -> None:
    assert probe.ledger_rows(tmp_path / "missing.db") == []


def test_match_ignores_baselines_other_operations_and_failures(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/3/QUIESCE_GPU_SERVICES/commit"
    ledger.save(
        _result(
            baseline_id,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(minutes=5),
        )
    )
    ledger.save(
        _result(
            "wf-new/2/MARK/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=10),
        )
    )
    ledger.save(
        _result(
            "wf-new/3/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.FAILED,
            T0 + timedelta(seconds=20),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids={baseline_id},
        armed_at=T0.isoformat(),
    )

    assert matched is None, matched


def test_match_finds_the_first_new_succeeded_row_of_the_operation(
    tmp_path: Path,
) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-new/3/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=25),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids=set(),
        armed_at=T0.isoformat(),
    )

    assert matched is not None, "the new QUIESCE row must match"
    assert matched["command_id"] == "wf-new/3/QUIESCE_GPU_SERVICES/commit"


def test_match_refuses_a_row_completed_before_arming(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-x/3/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(seconds=1),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids=set(),
        armed_at=T0.isoformat(),
    )

    assert matched is None, matched


def test_in_progress_rows_never_match(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    command = NodeActionCommand(
        command_id="wf-new/3/QUIESCE_GPU_SERVICES/commit",
        workflow_request_id="wf-new",
        incident_id="inc-1",
        fencing_token=1,
        operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
        node_id="node-b",
        issued_at=T0,
        expires_at=T0 + timedelta(minutes=1),
    )
    ledger.mark_in_progress(command, 1)

    rows = probe.ledger_rows(path)

    assert rows[0]["state"] == "IN_PROGRESS"
    assert (
        probe.match_ledger_row(
            rows,
            operation=QUIESCE,
            baseline_command_ids=set(),
            armed_at=(T0 - timedelta(minutes=1)).isoformat(),
        )
        is None
    ), "an in-progress row is not a succeeded one"


@pytest.mark.parametrize("device", ["/dev/nvidia0", "/dev/nvidia7", "/dev/nvidia15"])
def test_safe_device_accepts_gpu_device_nodes(device: str) -> None:
    assert probe.safe_device(device) == device


@pytest.mark.parametrize(
    "device",
    [
        "/dev/sda",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia16",
        "/dev/nvidia0;rm -rf /",
        "nvidia0",
        "/dev/nvidia0/../sda",
        "",
    ],
)
def test_safe_device_refuses_everything_else(device: str) -> None:
    with pytest.raises(probe.ProbeError):
        probe.safe_device(device)


def test_unit_allowlist_covers_only_the_two_transient_units() -> None:
    assert probe.unit_name(probe.holder_unit(RUN_ID), RUN_ID) == probe.holder_unit(
        RUN_ID
    )
    assert (
        probe.unit_name(probe.arm_unit(RUN_ID) + ".service", RUN_ID)
        == probe.arm_unit(RUN_ID) + ".service"
    )
    for unit in (
        "gpu-fault-node-agent.service",
        "kubelet.service",
        "nvidia-fabricmanager.service",
        "sshd",
        probe.holder_unit("destr018-run-a2"),
    ):
        with pytest.raises(probe.ProbeError):
            probe.unit_name(unit, RUN_ID)


def test_checked_command_allows_only_the_listed_verbs_and_units() -> None:
    allowed = [
        ["systemctl", "stop", probe.holder_unit(RUN_ID) + ".service"],
        ["systemctl", "reset-failed", probe.arm_unit(RUN_ID) + ".service"],
        ["systemctl", "is-active", probe.holder_unit(RUN_ID) + ".service"],
        [
            "systemctl",
            "show",
            probe.holder_unit(RUN_ID) + ".service",
            "--property=ActiveState",
        ],
    ]
    for command in allowed:
        assert probe.checked_command(command, RUN_ID) == command
    refused = [
        ["rm", "-rf", "/"],
        ["systemctl", "stop", "gpu-fault-node-agent.service"],
        ["systemctl", "disable", probe.holder_unit(RUN_ID) + ".service"],
        ["systemctl", "mask", probe.holder_unit(RUN_ID) + ".service"],
        ["systemctl", "restart", "kubelet.service"],
        ["systemctl", "daemon-reload"],
        ["systemd-run", "--unit", "gpu-fault-destr018-holder-x", "/bin/true"],
        ["bash", "-c", "systemctl stop gpu-fault-node-agent.service"],
        ["systemctl", "stop", probe.holder_unit(RUN_ID) + ".service", "--now"],
    ]
    for command in refused:
        with pytest.raises(probe.ProbeError):
            probe.checked_command(command, RUN_ID)


def test_arm_trigger_allowlist_excludes_the_verify_step() -> None:
    assert probe.arm_mode("") == "immediate"
    assert probe.arm_mode(None) == "immediate"
    assert probe.arm_mode(QUIESCE) == "ledger"
    assert probe.ARM_LEDGER_OPERATIONS == (QUIESCE,)
    for trigger in (VERIFY, "RESET_GPU", "RESTORE_GPU_SERVICES", "quiesce"):
        with pytest.raises(probe.ProbeError):
            probe.arm_mode(trigger)


def test_holder_and_arm_unit_names_are_digests_of_the_run_id() -> None:
    first = probe.holder_unit(RUN_ID)

    assert first.startswith("gpu-fault-destr018-holder-"), first
    assert first != probe.holder_unit("destr018-run-a2")
    assert probe.arm_unit(RUN_ID).startswith("gpu-fault-destr018-arm-"), probe.arm_unit(
        RUN_ID
    )
    assert probe.arm_unit(RUN_ID) != first
    with pytest.raises(probe.ProbeError):
        probe.holder_unit("bad run id with spaces")


def test_max_hold_bounds_are_enforced() -> None:
    assert probe.checked_max_hold(900) == 900
    assert probe.checked_max_hold(probe.MIN_HOLD_SECONDS) == probe.MIN_HOLD_SECONDS
    for value in (0, 59, 3601):
        with pytest.raises(probe.ProbeError):
            probe.checked_max_hold(value)


def test_state_file_roundtrip_is_scoped_to_the_run(tmp_path: Path) -> None:
    path = probe.state_path(RUN_ID, state_dir=tmp_path)

    assert path.parent == tmp_path
    assert path.name == f"destr018-{RUN_ID}.json"
    probe.write_state(path, {"armed_at": "2026-09-06T10:00:00+00:00"})
    probe.update_state(path, {"hold_started_at": "2026-09-06T10:00:01+00:00"})
    state = probe.read_state(path)
    assert state == {
        "armed_at": "2026-09-06T10:00:00+00:00",
        "hold_started_at": "2026-09-06T10:00:01+00:00",
    }
    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert probe.read_state(tmp_path / "absent.json") == {}
    with pytest.raises(probe.ProbeError):
        probe.state_path("bad id!", state_dir=tmp_path)


def test_device_clients_reports_only_holders_of_the_named_device(
    tmp_path: Path,
) -> None:
    root = _fake_proc(
        tmp_path,
        {
            "1207": ("destr018-ab12", "/dev/nvidia3"),
            "990": ("python3", "/dev/nvidia3"),
            "1500": ("nvidia-persiste", "/dev/nvidia0"),
            "2000": ("sshd", "/var/log/x"),
        },
    )

    holders = probe.device_clients("/dev/nvidia3", proc_root=root)

    assert [item["pid"] for item in holders] == ["990", "1207"]
    assert {item["process_name"] for item in holders} == {"python3", "destr018-ab12"}
    assert all(item["device"] == "/dev/nvidia3" for item in holders), holders
    assert probe.device_clients("/dev/nvidia1", proc_root=root) == []
    assert probe.device_clients("/dev/nvidia0", proc_root=tmp_path / "gone") == []
    with pytest.raises(probe.ProbeError):
        probe.device_clients("/dev/sda", proc_root=root)


def test_parser_accepts_every_documented_subcommand() -> None:
    parser = probe.parser()

    arm = parser.parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia3",
            "--drill-id",
            RUN_ID,
            "--max-hold-seconds",
            "900",
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/destr018_node_probe.py",
        ]
    )
    assert arm.command == "arm-holder"
    assert arm.max_hold_seconds == 900
    assert arm.after_ledger_op == "", (
        "the default must arm immediately, before the injection"
    )
    ledger_armed = parser.parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia3",
            "--drill-id",
            RUN_ID,
            "--after-ledger-op",
            QUIESCE,
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/destr018_node_probe.py",
        ]
    )
    assert ledger_armed.after_ledger_op == QUIESCE
    for command in (
        ["disarm-holder", "--run-id", RUN_ID],
        ["holder-status", "--run-id", RUN_ID],
        ["watch-ledger", "--run-id", RUN_ID],
        ["snapshot"],
        ["snapshot", "--run-id", RUN_ID, "--device", "/dev/nvidia3"],
    ):
        assert parser.parse_args(command).command == command[0]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "arm-holder",
                "--device",
                "/dev/nvidia3",
                "--drill-id",
                RUN_ID,
                "--after-ledger-op",
                VERIFY,
                "--run-id",
                RUN_ID,
                "--probe-script",
                "/run/destr018_node_probe.py",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_the_probe_script_identity_is_the_resolved_host_path(tmp_path: Path) -> None:
    """The holder unit runs the path on the host. A basename match would let a
    runner hand over the Pod-side ``/host/run/...`` path, which the unit could
    not find (DESTR-017 once shipped exactly that)."""

    assert probe.checked_probe_script(probe.__file__) == probe.__file__
    same_name = tmp_path / "host" / Path(probe.__file__).name
    same_name.parent.mkdir()
    same_name.write_text("# not the probe\n", encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="identity mismatch"):
        probe.checked_probe_script(str(same_name))
    with pytest.raises(probe.ProbeError, match="identity mismatch"):
        probe.checked_probe_script("/host" + probe.__file__)
