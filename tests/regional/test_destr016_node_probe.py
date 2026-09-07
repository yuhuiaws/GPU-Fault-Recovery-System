"""Unit tests for the DESTR-016 on-node GPU client holder probe.

The probe's pure helpers are tested against a real Node Agent ledger file, so
the SQL it runs is checked against the shipped schema rather than a hand-made
table, and against the allow-lists that keep it from touching anything but its
own transient units.
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
from scripts.e2e.regional.probes import destr016_node_probe as probe

QUIESCE = "QUIESCE_GPU_SERVICES"
VERIFY = "VERIFY_NO_GPU_CLIENTS"
RUN_ID = "destr016-run-a1"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


def _result(
    command_id: str,
    operation: WorkflowOperation,
    status: NodeActionStatus,
    completed_at: datetime,
    *,
    attempt: int = 1,
) -> NodeActionResult:
    return NodeActionResult(
        command_id=command_id,
        operation=operation,
        status=status,
        error=None
        if status is NodeActionStatus.SUCCEEDED
        else "clients are still active",
        retryable=status is not NodeActionStatus.SUCCEEDED,
        attempt=attempt,
        completed_at=completed_at,
    )


def _ledger(tmp_path: Path) -> tuple[Path, NodeActionLedger]:
    path = tmp_path / "node-actions.db"
    return path, NodeActionLedger(str(path))


def _save(
    ledger: NodeActionLedger,
    command_id: str,
    operation: WorkflowOperation,
    status: NodeActionStatus,
    offset_seconds: int,
    *,
    attempt: int = 1,
) -> None:
    ledger.save(
        _result(
            command_id,
            operation,
            status,
            T0 + timedelta(seconds=offset_seconds),
            attempt=attempt,
        )
    )


# --------------------------------------------------------------------------- #
# Ledger reads
# --------------------------------------------------------------------------- #
def test_ledger_rows_read_the_real_schema(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/2/QUIESCE/commit",
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.SUCCEEDED,
        10,
    )
    _save(
        ledger,
        "wf/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.FAILED,
        40,
        attempt=2,
    )

    rows = probe.ledger_rows(path)

    assert [row["operation"] for row in rows] == [QUIESCE, VERIFY]
    assert rows[0]["state"] == "SUCCEEDED"
    assert rows[1]["state"] == "FAILED"
    assert rows[1]["attempt"] == 2
    assert rows[1]["completed_at"].startswith("2026-09-06T10:00:40"), rows[1]


def test_ledger_rows_cover_the_host_and_fabric_validations(tmp_path: Path) -> None:
    """The successor validates host and fabric too; both must be visible."""

    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/4/HOST/commit",
        WorkflowOperation.VALIDATE_HOST,
        NodeActionStatus.SUCCEEDED,
        50,
    )
    _save(
        ledger,
        "wf/5/FABRIC/commit",
        WorkflowOperation.VALIDATE_FABRIC,
        NodeActionStatus.SUCCEEDED,
        60,
    )

    assert [row["operation"] for row in probe.ledger_rows(path)] == [
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
    ]
    assert set(probe.LEDGER_OPERATIONS) >= {"VALIDATE_HOST", "VALIDATE_FABRIC"}


def test_ledger_rows_are_empty_without_a_ledger(tmp_path: Path) -> None:
    assert probe.ledger_rows(tmp_path / "missing.db") == []


# --------------------------------------------------------------------------- #
# Arming on the quiesce row
# --------------------------------------------------------------------------- #
def test_arming_matches_this_drills_quiesce_row(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf-new/2/QUIESCE/commit",
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.SUCCEEDED,
        25,
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids=set(),
        armed_at=T0.isoformat(),
    )

    assert matched is not None, "the new quiesce row must match"
    assert matched["command_id"] == "wf-new/2/QUIESCE/commit"


def test_arming_ignores_baseline_rows_other_operations_and_failures(
    tmp_path: Path,
) -> None:
    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/2/QUIESCE/commit"
    _save(
        ledger,
        baseline_id,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.SUCCEEDED,
        -300,
    )
    _save(
        ledger,
        "wf-new/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.SUCCEEDED,
        10,
    )
    _save(
        ledger,
        "wf-new/2/QUIESCE/commit",
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.FAILED,
        20,
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids={baseline_id},
        armed_at=T0.isoformat(),
    )

    assert matched is None, matched


def test_arming_refuses_a_row_completed_before_arming(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf-x/2/QUIESCE/commit",
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        NodeActionStatus.SUCCEEDED,
        -1,
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
        command_id="wf-new/2/QUIESCE/commit",
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


# --------------------------------------------------------------------------- #
# The arming race
# --------------------------------------------------------------------------- #
def test_no_verify_success_means_the_race_was_not_lost(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.FAILED,
        30,
    )

    assert (
        probe.arm_race_lost(
            probe.ledger_rows(path),
            baseline_command_ids=set(),
            armed_at=T0.isoformat(),
            hold_started_at=(T0 + timedelta(seconds=20)).isoformat(),
        )
        is False
    ), "a failed verification is the boundary the case wants"


def test_a_verification_that_succeeded_before_the_holder_loses_the_race(
    tmp_path: Path,
) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.SUCCEEDED,
        10,
    )

    assert (
        probe.arm_race_lost(
            probe.ledger_rows(path),
            baseline_command_ids=set(),
            armed_at=T0.isoformat(),
            hold_started_at=(T0 + timedelta(seconds=20)).isoformat(),
        )
        is True
    ), "the reset committed before the holder existed"


def test_a_verification_that_succeeded_after_the_holder_started_is_not_the_race(
    tmp_path: Path,
) -> None:
    """A later success is a retry outcome, not a lost arming race."""

    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.SUCCEEDED,
        300,
    )

    assert (
        probe.arm_race_lost(
            probe.ledger_rows(path),
            baseline_command_ids=set(),
            armed_at=T0.isoformat(),
            hold_started_at=(T0 + timedelta(seconds=20)).isoformat(),
        )
        is False
    )


def test_a_verify_success_without_any_holder_loses_the_race(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.SUCCEEDED,
        10,
    )

    assert (
        probe.arm_race_lost(
            probe.ledger_rows(path),
            baseline_command_ids=set(),
            armed_at=T0.isoformat(),
            hold_started_at="",
        )
        is True
    ), "no holder ever started, so the verification passed unopposed"


def test_a_baseline_verify_success_is_not_this_drills_race(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    _save(
        ledger,
        "wf-old/3/VERIFY/commit",
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        NodeActionStatus.SUCCEEDED,
        10,
    )

    assert (
        probe.arm_race_lost(
            probe.ledger_rows(path),
            baseline_command_ids={"wf-old/3/VERIFY/commit"},
            armed_at=T0.isoformat(),
            hold_started_at="",
        )
        is False
    ), "an earlier drill's verification says nothing about this one"


# --------------------------------------------------------------------------- #
# Allow-lists
# --------------------------------------------------------------------------- #
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


def test_unit_allowlist_refuses_units_the_probe_does_not_own() -> None:
    assert probe.unit_name(probe.AGENT_UNIT, RUN_ID) == probe.AGENT_UNIT
    holder = probe.holder_unit(RUN_ID)
    assert probe.unit_name(holder, RUN_ID) == holder
    for unit in ("kubelet.service", "nvidia-fabricmanager.service", "sshd"):
        with pytest.raises(probe.ProbeError):
            probe.unit_name(unit, RUN_ID)
    with pytest.raises(probe.ProbeError):
        probe.unit_name(probe.holder_unit("destr016-other-run"), RUN_ID)


def test_the_node_agent_unit_may_only_be_read() -> None:
    """This case needs the Agent alive to take the reboot and re-register."""

    for verb in probe.READ_ONLY_SYSTEMCTL_VERBS:
        command = ["systemctl", verb, probe.AGENT_UNIT]
        assert probe.checked_command(command, RUN_ID) == command
    for verb in ("stop", "start", "disable", "restart", "reset-failed", "mask"):
        with pytest.raises(probe.ProbeError):
            probe.checked_command(["systemctl", verb, probe.AGENT_UNIT], RUN_ID)


def test_checked_command_allows_only_the_listed_verbs_and_units() -> None:
    allowed = [
        ["systemctl", "stop", probe.holder_unit(RUN_ID) + ".service"],
        ["systemctl", "reset-failed", probe.arm_unit(RUN_ID) + ".service"],
        ["systemctl", "show", probe.holder_unit(RUN_ID), "--property=ActiveState"],
        ["systemctl", "is-active", probe.arm_unit(RUN_ID)],
    ]
    for command in allowed:
        assert probe.checked_command(command, RUN_ID) == command
    refused = [
        ["rm", "-rf", "/"],
        ["systemctl", "stop", "kubelet.service"],
        ["systemctl", "mask", probe.holder_unit(RUN_ID)],
        ["systemctl", "daemon-reload"],
        ["systemd-run", "--unit", "gpu-fault-quiesce-x", "/bin/true"],
        ["bash", "-c", "systemctl stop gpu-fault-node-agent.service"],
        ["systemctl", "show", probe.holder_unit(RUN_ID), "--now"],
    ]
    for command in refused:
        with pytest.raises(probe.ProbeError):
            probe.checked_command(command, RUN_ID)


def test_holder_and_arm_unit_names_are_digests_of_the_run_id() -> None:
    first = probe.holder_unit(RUN_ID)
    second = probe.holder_unit("destr016-run-a2")
    assert first.startswith("gpu-fault-destr016-holder-"), first
    assert first != second
    assert probe.arm_unit(RUN_ID).startswith("gpu-fault-destr016-arm-"), probe.arm_unit(
        RUN_ID
    )
    assert probe.holder_unit(RUN_ID) != probe.arm_unit(RUN_ID)
    with pytest.raises(probe.ProbeError):
        probe.holder_unit("bad run id with spaces")


def test_max_hold_bounds_are_enforced() -> None:
    assert probe.checked_max_hold(900) == 900
    assert probe.checked_max_hold(probe.MIN_HOLD_SECONDS) == probe.MIN_HOLD_SECONDS
    for value in (0, 59, 3601):
        with pytest.raises(probe.ProbeError):
            probe.checked_max_hold(value)


# --------------------------------------------------------------------------- #
# Per-run state
# --------------------------------------------------------------------------- #
def test_state_file_roundtrip_is_scoped_to_the_run(tmp_path: Path) -> None:
    path = probe.state_path(RUN_ID, state_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name == f"destr016-{RUN_ID}.json"
    probe.write_state(path, {"armed_at": T0.isoformat()})
    probe.update_state(path, {"arm_race_lost": False})
    state = probe.read_state(path)
    assert state == {"armed_at": T0.isoformat(), "arm_race_lost": False}
    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(probe.ProbeError):
        probe.state_path("bad id!", state_dir=tmp_path)


def test_reading_a_state_file_that_does_not_exist_is_empty(tmp_path: Path) -> None:
    assert probe.read_state(tmp_path / "absent.json") == {}


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_parser_accepts_every_documented_subcommand() -> None:
    parser = probe.parser()
    arm = parser.parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia3",
            "--drill-id",
            RUN_ID,
            "--after-ledger-op",
            QUIESCE,
            "--max-hold-seconds",
            "1800",
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/destr016_node_probe.py",
        ]
    )
    assert arm.command == "arm-holder"
    assert arm.max_hold_seconds == 1800
    assert arm.after_ledger_op == QUIESCE
    for command in (
        ["disarm-holder", "--run-id", RUN_ID],
        ["holder-status", "--run-id", RUN_ID],
        ["watch-ledger", "--run-id", RUN_ID],
        ["snapshot"],
        ["snapshot", "--run-id", RUN_ID],
    ):
        assert parser.parse_args(command).command == command[0]


def test_the_probe_has_no_subcommand_that_touches_the_agent_or_the_gpu() -> None:
    """No stop/disable/reset subcommand exists here at all."""

    help_text = probe.parser().format_help()
    for forbidden in ("disable-agent", "restore-agent", "reset", "reboot", "write-xid"):
        assert forbidden not in help_text, forbidden


def test_parser_refuses_an_arm_operation_outside_the_allowlist() -> None:
    parser = probe.parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "arm-holder",
                "--device",
                "/dev/nvidia0",
                "--drill-id",
                RUN_ID,
                "--after-ledger-op",
                "RESTORE_GPU_SERVICES",
                "--run-id",
                RUN_ID,
                "--probe-script",
                "/run/destr016_node_probe.py",
            ]
        )


def test_the_arm_operations_are_the_two_steps_around_the_boundary() -> None:
    assert probe.ARM_LEDGER_OPERATIONS == (QUIESCE, VERIFY)
    assert probe.RACE_OPERATION == VERIFY


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
