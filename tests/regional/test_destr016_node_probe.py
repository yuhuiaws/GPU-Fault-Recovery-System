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
from typing import Any

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
        ["pre-authorize", "--run-id", RUN_ID, "--authorization", "{}"],
        ["fire-injection", "--run-id", RUN_ID, "--phase", "absorb"],
        ["snapshot"],
        ["snapshot", "--run-id", RUN_ID],
    ):
        assert parser.parse_args(command).command == command[0]
    # No exec-time authorization remains: quiesce stops kubelet, so nothing
    # could deliver it after the barrier.
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["authorize-injection", "--run-id", RUN_ID, "--phase", "absorb"]
        )


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


def test_arm_holder_schedules_the_absorb_and_escalation_writes_after_the_holder() -> (
    None
):
    """QUIESCE stops kubelet and the exec channel; the two writes that must land
    inside the WAITING window are systemd timers armed with the holder."""
    arguments = probe.parser().parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia0",
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
            "--inject-script",
            "/run/gpu-fault-host-probe-ab894dd753.py",
            "--pci-bdf",
            "0000:59:00",
            "--absorb-marker",
            "m-absorb",
            "--absorb-drill-id",
            f"{RUN_ID}-s",
            "--absorb-after-seconds",
            "90",
            "--escalate-marker",
            "m-escalate",
            "--escalate-drill-id",
            f"{RUN_ID}-e",
            "--escalate-after-seconds",
            "240",
            "--maintenance-window-end",
            "2099-01-01T00:00:00+00:00",
        ]
    )

    plan = probe.injection_plan(arguments)

    assert [
        (item["phase"], item["subcommand"], item["after_seconds"]) for item in plan
    ] == [("absorb", "write-xid46", 90), ("escalate", "write-xid79", 240)], plan
    command = probe.injection_command(RUN_ID, plan[1])
    assert command[:2] == ["systemd-run", "--unit"], command
    assert command[2] == probe.injection_unit(RUN_ID, "escalate"), command
    assert "--on-active=1s" in command, command
    assert command[-5:] == [
        "fire-injection",
        "--run-id",
        RUN_ID,
        "--phase",
        "escalate",
    ], command
    assert "write-xid79" not in command, (
        "the timer must recheck authorization before writing"
    )
    assert "--property=RuntimeMaxSec=700" in probe.injection_command(
        RUN_ID, plan[1], runtime_max_seconds=700
    ), "the fire unit lives as long as the pre-authorization it polls for"
    # The scheduled units are the probe's own: the allow-listed systemctl verbs
    # may stop them, so disarm-holder can clear a timer that never fired.
    timer = probe.injection_unit(RUN_ID, "absorb") + ".timer"
    assert probe.checked_command(["systemctl", "stop", timer], RUN_ID)[-1] == timer


def test_arm_holder_without_an_inject_script_schedules_nothing() -> None:
    arguments = probe.parser().parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia0",
            "--drill-id",
            RUN_ID,
            "--after-ledger-op",
            QUIESCE,
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/destr016_node_probe.py",
        ]
    )
    assert probe.injection_plan(arguments) == [], "no inject script means no timers"


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--inject-script", "/tmp/evil.py", "installed host probe script"),
        ("--pci-bdf", "0000:59:00;rm", "unsafe PCI BDF"),
        ("--absorb-after-seconds", "3", "below"),
        ("--escalate-after-seconds", "1800", "bounded lifetime"),
    ],
)
def test_arm_holder_refuses_unsafe_or_out_of_window_injections(
    option: str, value: str, message: str
) -> None:
    base = {
        "--inject-script": "/run/gpu-fault-host-probe-ab894dd753.py",
        "--pci-bdf": "0000:59:00",
        "--absorb-marker": "m-a",
        "--absorb-drill-id": f"{RUN_ID}-s",
        "--absorb-after-seconds": "90",
        "--escalate-marker": "m-e",
        "--escalate-drill-id": f"{RUN_ID}-e",
        "--escalate-after-seconds": "240",
    }
    base[option] = value
    argv = [
        "arm-holder",
        "--device",
        "/dev/nvidia0",
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
    for key, item in base.items():
        argv += [key, item]
    with pytest.raises(probe.ProbeError, match=message):
        probe.injection_plan(probe.parser().parse_args(argv))


# --------------------------------------------------------------------------- #
# The host-side barrier condition, against a real Node Agent ledger
# --------------------------------------------------------------------------- #
NODE = "node-b"
NEW_WORKFLOW = "workflow-destr016-new"
BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"


def _proof(**overrides: Any) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "kind": probe.CONDITIONAL_KIND,
        "conditional": True,
        "run_id": RUN_ID,
        "node_id": NODE,
        "boot_id": BOOT_A,
        "device": "/dev/nvidia0",
        "drill_id": f"{RUN_ID}-r",
        "marker": f"{RUN_ID}-r",
        "maintenance_window_end": (T0 + timedelta(hours=1)).isoformat(),
        "maintenance_window_seconds": 420,
        "authorized_at": T0.isoformat(),
        "expires_at": (T0 + timedelta(seconds=600)).isoformat(),
        "not_before_seconds": {"absorb": 90, "escalate": 240},
        "ledger": dict(probe.LEDGER_SHAPE),
    }
    proof.update(overrides)
    return proof


def _parked_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "boot_id": BOOT_A,
        "hold_started_at": (T0 + timedelta(seconds=20)).isoformat(),
        "pre_authorization": _proof(),
        "pre_authorized_at": T0.isoformat(),
        "ledger_baseline_workflow_ids": ["wf-old"],
        "injections_fired": {},
    }
    state.update(overrides)
    return state


def _command(
    command_id: str, operation: WorkflowOperation, workflow: str, node: str = NODE
) -> NodeActionCommand:
    return NodeActionCommand(
        command_id=command_id,
        workflow_request_id=workflow,
        incident_id="inc-1",
        fencing_token=7,
        operation=operation,
        node_id=node,
        issued_at=T0,
        expires_at=T0 + timedelta(minutes=5),
    )


def _record(
    ledger: NodeActionLedger,
    command: NodeActionCommand,
    status: NodeActionStatus | None,
    offset_seconds: int,
    *,
    error: str | None = None,
    attempt: int = 1,
) -> None:
    ledger.mark_in_progress(command, attempt, agent_generation=4)
    if status is None:
        return
    ledger.save(
        NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=status,
            error=error,
            retryable=False,
            attempt=attempt,
            completed_at=T0 + timedelta(seconds=offset_seconds),
        )
    )


def _barrier_ledger(
    tmp_path: Path,
    *,
    verify: NodeActionStatus | None | bool = NodeActionStatus.FAILED,
    verify_error: str | None = "GPU compute clients are still active: GPU-a:4242",
    verify_attempts: int = 2,
    extra: tuple[WorkflowOperation, ...] = (),
    node: str = NODE,
) -> list[dict[str, Any]]:
    """A real ledger: an old drill's rows, then this run's barrier rows."""

    path, ledger = _ledger(tmp_path)
    _record(
        ledger,
        _command(
            f"wf-old/3/{VERIFY}/{NODE}/agent-3",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            "wf-old",
        ),
        NodeActionStatus.SUCCEEDED,
        -600,
    )
    _record(
        ledger,
        _command(
            f"{NEW_WORKFLOW}/2/{QUIESCE}/{node}/agent-4",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NEW_WORKFLOW,
            node,
        ),
        NodeActionStatus.SUCCEEDED,
        20,
    )
    if verify is not False:
        verify_command = _command(
            f"{NEW_WORKFLOW}/3/{VERIFY}/{node}/agent-4",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NEW_WORKFLOW,
            node,
        )
        for attempt in range(1, verify_attempts + 1):
            _record(
                ledger,
                verify_command,
                None if verify is None else verify,
                30 + 10 * attempt,
                error=verify_error,
                attempt=attempt,
            )
    for index, operation in enumerate(extra, start=4):
        _record(
            ledger,
            _command(
                f"{NEW_WORKFLOW}/{index}/{operation.value}/{node}/agent-4",
                operation,
                NEW_WORKFLOW,
                node,
            ),
            NodeActionStatus.SUCCEEDED,
            60,
        )
    return probe.ledger_rows(path)


def test_ledger_rows_expose_workflow_incident_generation_and_refusal(
    tmp_path: Path,
) -> None:
    rows = _barrier_ledger(tmp_path)
    verify = [row for row in rows if row["operation"] == VERIFY][-1]
    assert verify["workflow_request_id"] == NEW_WORKFLOW, verify
    assert verify["incident_id"] == "inc-1" and verify["agent_generation"] == 4, verify
    assert verify["attempt"] == 2, verify
    assert "clients are still active" in verify["error"], verify


def test_barrier_condition_holds_on_a_real_ledger_parked_at_the_verification(
    tmp_path: Path,
) -> None:
    verdict = probe.barrier_condition(
        _parked_state(),
        _barrier_ledger(tmp_path),
        phase="absorb",
        now=T0 + timedelta(seconds=120),
        boot_id=BOOT_A,
        holder_active=True,
    )
    assert verdict["holds"] is True and verdict["final"] is False, verdict
    condition = verdict["condition"]
    assert condition["workflow_request_id"] == NEW_WORKFLOW, condition
    assert condition["incident_id"] == "inc-1" and condition["boot_id"] == BOOT_A, (
        condition
    )
    assert condition["agent_generation"] == 4 and condition["verify_attempt"] == 2, (
        condition
    )
    assert condition["quiesce_command_id"].endswith(f"/{NODE}/agent-4"), condition
    assert (
        condition["pinned_window_expires_at"]
        == (T0 + timedelta(seconds=20 + 420)).isoformat()
    ), condition


@pytest.mark.parametrize(
    ("change", "final", "reason"),
    [
        ("no-verify", False, "has not executed"),
        ("verify-in-progress", False, "has not executed"),
        ("verify-succeeded", True, "did not hold"),
        ("reset-row", True, "advanced beyond the barrier"),
        ("fabric-reset-row", True, "advanced beyond the barrier"),
        ("restore-row", True, "advanced beyond the barrier"),
        ("boot", True, "boot id changed"),
        ("expired", True, "expired before the barrier condition held"),
        ("window", True, "pinned maintenance window has expired"),
        ("not-due", False, "not due"),
        ("holder-dead", True, "no longer active"),
        ("no-holder", False, "has not started"),
        ("baseline-only", False, "no workflow of this run"),
        ("other-node", True, "does not name this node"),
        ("wrong-refusal", False, "did not refuse on clients"),
        ("no-preauth", True, "no conditional pre-authorization"),
        ("disarmed", True, "disarmed"),
        ("race", True, "before the holder started"),
        ("shape", True, "ledger shape"),
    ],
)
def test_barrier_condition_waits_or_refuses_for_every_defect(
    change: str, final: bool, reason: str, tmp_path: Path
) -> None:
    state = _parked_state()
    now = T0 + timedelta(seconds=120)
    boot = BOOT_A
    holder = True
    if change == "no-verify":
        rows = _barrier_ledger(tmp_path, verify=False)
    elif change == "verify-in-progress":
        rows = _barrier_ledger(tmp_path, verify=None, verify_attempts=1)
    elif change == "verify-succeeded":
        rows = _barrier_ledger(
            tmp_path, verify=NodeActionStatus.SUCCEEDED, verify_error=None
        )
    elif change == "reset-row":
        rows = _barrier_ledger(tmp_path, extra=(WorkflowOperation.RESET_GPU,))
    elif change == "fabric-reset-row":
        rows = _barrier_ledger(
            tmp_path, extra=(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,)
        )
    elif change == "restore-row":
        rows = _barrier_ledger(
            tmp_path, extra=(WorkflowOperation.RESTORE_GPU_SERVICES,)
        )
    elif change == "other-node":
        rows = _barrier_ledger(tmp_path, node="node-c")
    elif change == "wrong-refusal":
        rows = _barrier_ledger(tmp_path, verify_error="node agent lease expired")
    else:
        rows = _barrier_ledger(tmp_path)
    if change == "boot":
        boot = BOOT_B
    elif change == "expired":
        now = T0 + timedelta(seconds=601)
    elif change == "window":
        now = T0 + timedelta(seconds=20 + 420)
    elif change == "not-due":
        now = T0 + timedelta(seconds=100)
    elif change == "holder-dead":
        holder = False
    elif change == "no-holder":
        state = _parked_state(hold_started_at=None)
    elif change == "baseline-only":
        state = _parked_state(ledger_baseline_workflow_ids=["wf-old", NEW_WORKFLOW])
    elif change == "no-preauth":
        state = _parked_state(pre_authorization=None)
    elif change == "disarmed":
        state = _parked_state(disarmed_at=T0.isoformat())
    elif change == "race":
        state = _parked_state(arm_race_lost=True)
    elif change == "shape":
        state = _parked_state(pre_authorization=_proof(ledger={"quiesce": QUIESCE}))
    verdict = probe.barrier_condition(
        state, rows, phase="absorb", now=now, boot_id=boot, holder_active=holder
    )
    assert verdict["holds"] is False, verdict
    assert verdict["final"] is final and reason in verdict["reason"], verdict


def test_the_escalation_waits_for_the_absorbed_fault_to_fire_first(
    tmp_path: Path,
) -> None:
    rows = _barrier_ledger(tmp_path)
    now = T0 + timedelta(seconds=300)
    waiting = probe.barrier_condition(
        _parked_state(),
        rows,
        phase="escalate",
        now=now,
        boot_id=BOOT_A,
        holder_active=True,
    )
    assert waiting["holds"] is False and waiting["final"] is False, waiting
    assert "absorb has not fired yet" in waiting["reason"], waiting
    fired = _parked_state(
        injections_fired={
            "absorb": {"fired_at": (T0 + timedelta(seconds=115)).isoformat()}
        }
    )
    ready = probe.barrier_condition(
        fired, rows, phase="escalate", now=now, boot_id=BOOT_A, holder_active=True
    )
    assert ready["holds"] is True, ready
    early = probe.barrier_condition(
        fired,
        rows,
        phase="escalate",
        now=T0 + timedelta(seconds=200),
        boot_id=BOOT_A,
        holder_active=True,
    )
    assert early["holds"] is False and "not due" in early["reason"], early


def test_new_workflow_ids_ignore_the_baseline_and_rows_before_the_authorization(
    tmp_path: Path,
) -> None:
    rows = _barrier_ledger(tmp_path)
    assert probe.new_workflow_ids(
        rows, baseline={"wf-old"}, not_before=T0.isoformat()
    ) == {NEW_WORKFLOW}
    assert probe.new_workflow_ids(rows, baseline=set(), not_before=T0.isoformat()) == {
        NEW_WORKFLOW
    }, "the old drill's row completed before the authorization"
    assert (
        probe.new_workflow_ids(
            rows, baseline=set(), not_before=(T0 + timedelta(minutes=5)).isoformat()
        )
        == set()
    )
