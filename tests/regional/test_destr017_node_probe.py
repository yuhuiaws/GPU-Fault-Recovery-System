"""Unit tests for the GF-REGIONAL-DESTR-017 on-node probe.

The probe does two irreversible things on a real node -- it opens a GPU device
and it reboots the machine -- so what is tested here is the guard rails: the
ledger row it waits for, the units it is allowed to touch, the bounds on both
timers, and the durable boot-id bookkeeping that proves exactly one boot
happened. The ledger tests write real ``NodeActionLedger`` rows rather than a
hand-made table, so a schema change breaks them.
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
from scripts.e2e.regional.probes import destr017_node_probe as probe

QUIESCE = "QUIESCE_GPU_SERVICES"
VERIFY = "VERIFY_NO_GPU_CLIENTS"
RUN_ID = "destr017-abc123-a1"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"


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


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def test_ledger_rows_read_the_real_schema(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-old/2/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(hours=1),
        )
    )
    ledger.save(
        _result(
            "wf-new/3/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.FAILED,
            T0 + timedelta(seconds=40),
        )
    )

    rows = probe.ledger_rows(path)

    assert [row["operation"] for row in rows] == [QUIESCE, VERIFY]
    assert rows[0]["state"] == "SUCCEEDED"
    assert rows[1]["state"] == "FAILED"
    assert rows[1]["attempt"] == 1
    assert rows[1]["completed_at"].startswith("2026-09-06T10:00:40"), rows[1]


def test_ledger_rows_are_empty_without_a_ledger(tmp_path: Path) -> None:
    assert probe.ledger_rows(tmp_path / "missing.db") == []


def test_the_holder_arms_on_this_drills_quiesce_row(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/2/QUIESCE_GPU_SERVICES/commit"
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
            "wf-new/2/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=25),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=QUIESCE,
        baseline_command_ids={baseline_id},
        armed_at=T0.isoformat(),
    )

    assert matched is not None, "the new quiesce row must arm the holder"
    assert matched["command_id"] == "wf-new/2/QUIESCE_GPU_SERVICES/commit"


def test_an_earlier_drills_rows_never_arm_the_holder(tmp_path: Path) -> None:
    """The Node Agent ledger replays by command id for days, so a row a previous
    case left behind must not open a GPU holder on an unrelated workflow."""

    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/2/QUIESCE_GPU_SERVICES/commit"
    ledger.save(
        _result(
            baseline_id,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=30),
        )
    )

    assert (
        probe.match_ledger_row(
            probe.ledger_rows(path),
            operation=QUIESCE,
            baseline_command_ids={baseline_id},
            armed_at=T0.isoformat(),
        )
        is None
    ), "a baseline command id is not this drill's row"


def test_a_row_completed_before_arming_never_matches(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-x/2/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(seconds=1),
        )
    )

    assert (
        probe.match_ledger_row(
            probe.ledger_rows(path),
            operation=QUIESCE,
            baseline_command_ids=set(),
            armed_at=T0.isoformat(),
        )
        is None
    ), "a row older than the arm timestamp belongs to an earlier drill"


def test_a_failed_or_in_progress_row_is_not_a_success(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.mark_in_progress(
        NodeActionCommand(
            command_id="wf-new/2/QUIESCE_GPU_SERVICES/commit",
            workflow_request_id="wf-new",
            incident_id="inc-1",
            fencing_token=1,
            operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
            node_id="node-b",
            issued_at=T0,
            expires_at=T0 + timedelta(minutes=1),
        ),
        1,
    )
    ledger.save(
        _result(
            "wf-new/3/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.FAILED,
            T0 + timedelta(seconds=20),
        )
    )

    rows = probe.ledger_rows(path)

    assert rows[0]["state"] == "IN_PROGRESS"
    for operation in (QUIESCE, VERIFY):
        assert (
            probe.match_ledger_row(
                rows,
                operation=operation,
                baseline_command_ids=set(),
                armed_at=(T0 - timedelta(minutes=1)).isoformat(),
            )
            is None
        ), operation


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
        "/dev/nvidia0;rm -rf /",
        "nvidia0",
        "/dev/nvidia0/../sda",
        "/dev/nvidia16",
        "",
    ],
)
def test_safe_device_refuses_everything_else(device: str) -> None:
    with pytest.raises(probe.ProbeError):
        probe.safe_device(device)


def test_the_probe_owns_only_the_three_units_named_after_its_run() -> None:
    for unit in (probe.holder_unit, probe.arm_unit, probe.reboot_unit):
        name = unit(RUN_ID)
        assert probe.unit_name(name, RUN_ID) == name
        assert probe.unit_name(f"{name}.service", RUN_ID) == f"{name}.service"
        assert probe.unit_name(f"{name}.timer", RUN_ID) == f"{name}.timer"
    for unit in ("kubelet.service", "nvidia-fabricmanager.service", "sshd"):
        with pytest.raises(probe.ProbeError):
            probe.unit_name(unit, RUN_ID)
    with pytest.raises(probe.ProbeError):
        probe.unit_name(probe.holder_unit("destr017-other-a1"), RUN_ID)


def test_the_node_agent_unit_may_never_be_mutated_by_this_probe() -> None:
    """DESTR-017 reads the Agent's state and lets the reboot restart it; a probe
    that could stop or disable it would be injecting a different fault."""

    with pytest.raises(probe.ProbeError):
        probe.unit_name(probe.AGENT_UNIT, RUN_ID)
    for verb in ("stop", "reset-failed", "show", "disable"):
        with pytest.raises(probe.ProbeError):
            probe.checked_command(["systemctl", verb, probe.AGENT_UNIT], RUN_ID)


def test_checked_command_allows_only_the_listed_verbs_and_units() -> None:
    holder = probe.holder_unit(RUN_ID) + ".service"
    timer = probe.reboot_unit(RUN_ID) + ".timer"
    for command in (
        ["systemctl", "stop", holder],
        ["systemctl", "reset-failed", holder],
        ["systemctl", "stop", timer],
        ["systemctl", "show", timer, "--property=ActiveState"],
    ):
        assert probe.checked_command(command, RUN_ID) == command
    for command in (
        ["rm", "-rf", "/"],
        ["systemctl", "reboot"],
        ["systemctl", "start", holder],
        ["systemctl", "mask", holder],
        ["systemctl", "daemon-reload"],
        ["systemctl", "stop", "kubelet.service"],
        ["systemd-run", "--unit", "gpu-fault-quiesce-x", "/bin/true"],
        ["bash", "-c", f"systemctl stop {holder}"],
        ["systemctl", "show", timer, "--all"],
    ):
        with pytest.raises(probe.ProbeError):
            probe.checked_command(command, RUN_ID)


def test_the_only_reboot_form_is_an_ordinary_systemctl_reboot() -> None:
    """A forced reset would destroy the Node Agent ledger the case reads back
    across the boot, and is not a fault a site operator would plausibly cause."""

    assert probe.reboot_command() == ["/bin/systemctl", "reboot"]
    for token in ("-f", "--force", "sysrq", "reboot -f"):
        assert token not in probe.reboot_command()


def test_unit_names_are_digests_of_the_run_id() -> None:
    first = probe.holder_unit(RUN_ID)
    assert first.startswith("gpu-fault-destr017-holder-"), first
    assert first != probe.holder_unit("destr017-abc123-a2")
    assert probe.arm_unit(RUN_ID).startswith("gpu-fault-destr017-arm-"), RUN_ID
    assert probe.reboot_unit(RUN_ID).startswith("gpu-fault-destr017-reboot-"), RUN_ID
    assert (
        len(
            {
                probe.holder_unit(RUN_ID),
                probe.arm_unit(RUN_ID),
                probe.reboot_unit(RUN_ID),
            }
        )
        == 3
    )
    with pytest.raises(probe.ProbeError):
        probe.holder_unit("bad run id with spaces")


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
def test_max_hold_bounds_are_enforced() -> None:
    assert probe.checked_max_hold(900) == 900
    for value in (0, 59, 3601):
        with pytest.raises(probe.ProbeError):
            probe.checked_max_hold(value)


def test_the_reboot_delay_is_bounded_at_both_ends() -> None:
    """Too soon and the arming exec never returns before the transport dies; too
    late and the reboot lands after the pinned maintenance window, which would
    prove the window fence instead of the generation fence."""

    assert probe.checked_reboot_delay(30) == 30
    assert probe.checked_reboot_delay(600) == 600
    for value in (0, 1, 29, 601, 86400):
        with pytest.raises(probe.ProbeError):
            probe.checked_reboot_delay(value)


def test_the_reboot_is_armed_on_node_and_records_its_boot_id_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """quiesce stops kubelet, so the reboot cannot be exec'd from the runner
    after the fence is WAITING. ``_place_reboot_timer`` is the shared path the
    on-node watcher takes: it must write the pre-reboot boot id durably *before*
    the systemd timer that will run ``systemctl reboot`` exists."""

    calls: list[list[str]] = []
    monkeypatch.setattr(
        probe, "run", lambda command, **kwargs: calls.append(list(command))
    )
    monkeypatch.setattr(probe, "boot_id", lambda: BOOT_A)
    monkeypatch.setattr(probe, "_reboot_unit_state", lambda run_id: {})
    path = tmp_path / "state.json"

    record = probe._place_reboot_timer(RUN_ID, 45, path)

    assert record["reboot_delay_seconds"] == 45
    assert record["boot_id_before_reboot"] == BOOT_A
    state = json.loads(path.read_text())
    assert state["boot_id_before_reboot"] == BOOT_A, "durable marker before timer"
    assert state["reboot_delay_seconds"] == 45
    timer = probe.reboot_unit(RUN_ID)
    armed = [c for c in calls if "systemd-run" in c and f"--on-active=45s" in c]
    assert len(armed) == 1, calls
    assert f"--unit={timer}" in armed[0]
    assert armed[0][-2:] == ["/bin/systemctl", "reboot"], "only an ordinary reboot"


# --------------------------------------------------------------------------- #
# Durable state
# --------------------------------------------------------------------------- #
def test_state_file_roundtrip_is_scoped_to_the_run(tmp_path: Path) -> None:
    path = probe.state_path(RUN_ID, state_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name == f"destr017-{RUN_ID}.json"
    probe.write_state(path, {"reboot_armed_at": "2026-09-06T10:00:00+00:00"})
    probe.update_state(path, {"boot_id_before_reboot": BOOT_A})
    state = probe.read_state(path)
    assert state == {
        "reboot_armed_at": "2026-09-06T10:00:00+00:00",
        "boot_id_before_reboot": BOOT_A,
    }
    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(probe.ProbeError):
        probe.state_path("bad id!", state_dir=tmp_path)


def test_the_state_file_lives_outside_the_node_agents_own_directory() -> None:
    """Acceptance artifacts never share a directory with the Agent's state, so a
    forgotten marker file can never be read as product state."""

    assert probe.STATE_DIR != probe.LEDGER.parent
    assert probe.STATE_DIR.name == "gpu-fault-acceptance"


def test_read_state_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert probe.read_state(tmp_path / "absent.json") == {}


# --------------------------------------------------------------------------- #
# Boot observations
# --------------------------------------------------------------------------- #
def test_the_boot_history_records_one_entry_per_boot() -> None:
    state = probe.record_boot_observation({}, BOOT_A, observed_at="t0")
    assert state["observed_boot_ids"] == [BOOT_A]
    assert state["boot_changes"] == 0
    state = probe.record_boot_observation(state, BOOT_A, observed_at="t1")
    assert state["observed_boot_ids"] == [BOOT_A]
    assert state["boot_changes"] == 0
    assert state["boot_observed_at"] == "t1"
    state = probe.record_boot_observation(state, BOOT_B, observed_at="t2")
    assert state["observed_boot_ids"] == [BOOT_A, BOOT_B]
    assert state["boot_changes"] == 1


def test_a_second_reboot_shows_up_as_a_third_boot_entry() -> None:
    third = "33333333-3333-4333-8333-333333333333"
    state: dict[str, object] = {}
    for boot in (BOOT_A, BOOT_B, third):
        state = probe.record_boot_observation(state, boot, observed_at="t")
    assert state["observed_boot_ids"] == [BOOT_A, BOOT_B, third]
    assert state["boot_changes"] == 2


def test_the_boot_history_never_loses_the_rest_of_the_state() -> None:
    state = probe.record_boot_observation(
        {"reboot_delay_seconds": 30, "boot_id_before_reboot": BOOT_A},
        BOOT_B,
        observed_at="t",
    )
    assert state["reboot_delay_seconds"] == 30
    assert state["boot_id_before_reboot"] == BOOT_A


# --------------------------------------------------------------------------- #
# CLI
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
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/probe.py",
        ]
    )
    assert arm.command == "arm-holder"
    assert arm.after_ledger_op == QUIESCE, "the holder arms on quiesce by default"
    assert arm.reboot_delay_seconds is None, "no on-node reboot unless asked"
    armed_reboot = parser.parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia3",
            "--drill-id",
            RUN_ID,
            "--run-id",
            RUN_ID,
            "--probe-script",
            "/run/probe.py",
            "--reboot-delay-seconds",
            "30",
        ]
    )
    assert armed_reboot.reboot_delay_seconds == 30
    reboot = parser.parse_args(
        ["arm-reboot", "--run-id", RUN_ID, "--delay-seconds", "45"]
    )
    assert reboot.delay_seconds == 45
    default_reboot = parser.parse_args(["arm-reboot", "--run-id", RUN_ID])
    assert default_reboot.delay_seconds == probe.MIN_REBOOT_DELAY_SECONDS
    for command in (
        ["disarm-holder", "--run-id", RUN_ID],
        ["holder-status", "--run-id", RUN_ID],
        ["reboot-status", "--run-id", RUN_ID],
        ["cancel-reboot", "--run-id", RUN_ID],
        ["watch-ledger", "--run-id", RUN_ID],
        ["snapshot"],
        ["snapshot", "--run-id", RUN_ID],
    ):
        parsed = parser.parse_args(command)
        assert parsed.command == command[0]


def test_the_parser_refuses_a_ledger_operation_the_drill_may_not_arm_on() -> None:
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
                "RESET_GPU",
                "--run-id",
                RUN_ID,
                "--probe-script",
                "/run/p.py",
            ]
        )


def test_the_parser_has_no_subcommand_that_stops_a_service() -> None:
    """The probe's only service verbs are the internal cleanup of its own
    transient units; nothing an operator can call stops anything on the node."""

    parser = probe.parser()
    for command in (["stop-agent", "--run-id", RUN_ID], ["reboot", "--run-id", RUN_ID]):
        with pytest.raises(SystemExit):
            parser.parse_args(command)
