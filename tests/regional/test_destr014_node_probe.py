from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.ledger import NodeActionLedger
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from scripts.e2e.regional.probes import destr014_node_probe as probe

VERIFY = "VERIFY_NO_GPU_CLIENTS"
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
        error=None if status is NodeActionStatus.SUCCEEDED else "clients still active",
        retryable=status is not NodeActionStatus.SUCCEEDED,
        attempt=attempt,
        completed_at=completed_at,
    )


def _ledger(tmp_path: Path) -> tuple[Path, NodeActionLedger]:
    path = tmp_path / "node-actions.db"
    ledger = NodeActionLedger(str(path))
    return path, ledger


def test_ledger_rows_read_the_real_schema(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-old/4/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(hours=1),
        )
    )
    ledger.save(
        _result(
            "wf-new/5/RESET_GPU/commit",
            WorkflowOperation.RESET_GPU,
            NodeActionStatus.FAILED,
            T0 + timedelta(seconds=40),
        )
    )

    rows = probe.ledger_rows(path)

    assert [row["operation"] for row in rows] == [VERIFY, "RESET_GPU"]
    assert rows[0]["state"] == "SUCCEEDED"
    assert rows[1]["state"] == "FAILED"
    assert rows[1]["attempt"] == 1
    assert rows[1]["completed_at"].startswith("2026-09-06T10:00:40"), rows[1]


def test_ledger_rows_are_empty_without_a_ledger(tmp_path: Path) -> None:
    assert probe.ledger_rows(tmp_path / "missing.db") == []


def test_match_ignores_baseline_rows_other_operations_and_failures(
    tmp_path: Path,
) -> None:
    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/4/VERIFY_NO_GPU_CLIENTS/commit"
    ledger.save(
        _result(
            baseline_id,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(minutes=5),
        )
    )
    ledger.save(
        _result(
            "wf-new/3/QUIESCE_GPU_SERVICES/commit",
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=10),
        )
    )
    ledger.save(
        _result(
            "wf-new/4/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.FAILED,
            T0 + timedelta(seconds=20),
        )
    )

    rows = probe.ledger_rows(path)
    matched = probe.match_ledger_row(
        rows,
        operation=VERIFY,
        baseline_command_ids={baseline_id},
        armed_at=T0.isoformat(),
    )

    assert matched is None, matched


def test_match_finds_the_first_new_succeeded_row_of_the_operation(
    tmp_path: Path,
) -> None:
    path, ledger = _ledger(tmp_path)
    baseline_id = "wf-old/4/VERIFY_NO_GPU_CLIENTS/commit"
    ledger.save(
        _result(
            baseline_id,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(minutes=5),
        )
    )
    ledger.save(
        _result(
            "wf-new/4/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 + timedelta(seconds=25),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=VERIFY,
        baseline_command_ids={baseline_id},
        armed_at=T0.isoformat(),
    )

    assert matched is not None, "the new VERIFY row must match"
    assert matched["command_id"] == "wf-new/4/VERIFY_NO_GPU_CLIENTS/commit"


def test_match_refuses_a_row_completed_before_arming(tmp_path: Path) -> None:
    path, ledger = _ledger(tmp_path)
    ledger.save(
        _result(
            "wf-x/4/VERIFY_NO_GPU_CLIENTS/commit",
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.SUCCEEDED,
            T0 - timedelta(seconds=1),
        )
    )

    matched = probe.match_ledger_row(
        probe.ledger_rows(path),
        operation=VERIFY,
        baseline_command_ids=set(),
        armed_at=T0.isoformat(),
    )

    assert matched is None, matched


def test_in_progress_rows_never_match(tmp_path: Path) -> None:
    from gpu_fault.node_agent.protocol import NodeActionCommand

    path, ledger = _ledger(tmp_path)
    command = NodeActionCommand(
        command_id="wf-new/4/VERIFY_NO_GPU_CLIENTS/commit",
        workflow_request_id="wf-new",
        incident_id="inc-1",
        fencing_token=1,
        operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
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
            operation=VERIFY,
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
    run_id = "destr014-abc-a1"
    assert probe.unit_name(probe.AGENT_UNIT, run_id) == probe.AGENT_UNIT
    assert probe.unit_name(probe.holder_unit(run_id), run_id) == probe.holder_unit(
        run_id
    )
    for unit in ("kubelet.service", "nvidia-fabricmanager.service", "sshd"):
        with pytest.raises(probe.ProbeError):
            probe.unit_name(unit, run_id)
    with pytest.raises(probe.ProbeError):
        probe.unit_name(probe.holder_unit("some-other-run"), run_id)


def test_checked_command_allows_only_the_listed_verbs_and_units() -> None:
    run_id = "destr014-abc-a1"
    allowed = [
        ["systemctl", "disable", probe.AGENT_UNIT],
        ["systemctl", "enable", probe.AGENT_UNIT],
        ["systemctl", "is-enabled", probe.AGENT_UNIT],
        ["systemctl", "stop", probe.holder_unit(run_id) + ".service"],
        ["systemctl", "reset-failed", probe.arm_unit(run_id) + ".service"],
        ["systemctl", "show", probe.AGENT_UNIT, "--property=ActiveState"],
    ]
    for command in allowed:
        assert probe.checked_command(command, run_id) == command
    refused = [
        ["rm", "-rf", "/"],
        ["systemctl", "disable", "kubelet.service"],
        ["systemctl", "mask", probe.AGENT_UNIT],
        ["systemctl", "stop", "nvidia-fabricmanager.service"],
        ["systemctl", "daemon-reload"],
        ["systemd-run", "--unit", "gpu-fault-quiesce-x", "/bin/true"],
        ["bash", "-c", "systemctl disable gpu-fault-node-agent.service"],
    ]
    for command in refused:
        with pytest.raises(probe.ProbeError):
            probe.checked_command(command, run_id)


def test_holder_and_arm_unit_names_are_digests_of_the_run_id() -> None:
    first = probe.holder_unit("destr014-run-a1")
    second = probe.holder_unit("destr014-run-a2")
    assert first.startswith("gpu-fault-destr014-holder-"), first
    assert first != second
    assert probe.arm_unit("destr014-run-a1").startswith("gpu-fault-destr014-arm-"), (
        probe.arm_unit("destr014-run-a1")
    )
    with pytest.raises(probe.ProbeError):
        probe.holder_unit("bad run id with spaces")


def test_agent_baseline_record_and_restore_actions() -> None:
    enabled_active = probe.agent_baseline_record(
        enabled_state="enabled", active_state="active"
    )
    assert enabled_active["agent_enabled_baseline"] == "enabled"
    assert enabled_active["agent_active_baseline"] == "active"
    assert probe.restore_actions(enabled_active) == ["enable", "start"]

    disabled_inactive = probe.agent_baseline_record(
        enabled_state="disabled", active_state="inactive"
    )
    assert probe.restore_actions(disabled_inactive) == []

    enabled_inactive = probe.agent_baseline_record(
        enabled_state="enabled", active_state="inactive"
    )
    assert probe.restore_actions(enabled_inactive) == ["enable"]

    with pytest.raises(probe.ProbeError):
        probe.restore_actions({})


def test_state_file_roundtrip_is_scoped_to_the_run(tmp_path: Path) -> None:
    run_id = "destr014-run-a1"
    path = probe.state_path(run_id, state_dir=tmp_path)
    assert path.parent == tmp_path
    assert path.name == f"destr014-{run_id}.json"
    probe.write_state(path, {"armed_at": "2026-09-06T10:00:00+00:00"})
    probe.update_state(path, {"matched_row": {"command_id": "x"}})
    state = probe.read_state(path)
    assert state == {
        "armed_at": "2026-09-06T10:00:00+00:00",
        "matched_row": {"command_id": "x"},
    }
    assert json.loads(path.read_text(encoding="utf-8")) == state
    with pytest.raises(probe.ProbeError):
        probe.state_path("bad id!", state_dir=tmp_path)


def test_parser_accepts_every_documented_subcommand() -> None:
    parser = probe.parser()
    arm = parser.parse_args(
        [
            "arm-holder",
            "--device",
            "/dev/nvidia3",
            "--drill-id",
            "destr014-run-a1",
            "--after-ledger-op",
            "VERIFY_NO_GPU_CLIENTS",
            "--max-hold-seconds",
            "900",
            "--run-id",
            "destr014-run-a1",
            "--probe-script",
            "/run/probe.py",
        ]
    )
    assert arm.command == "arm-holder"
    assert arm.max_hold_seconds == 900
    for command in (
        ["disarm-holder", "--run-id", "r"],
        ["holder-status", "--run-id", "r"],
        ["disable-agent-restart", "--run-id", "r"],
        ["restore-agent", "--run-id", "r"],
        ["snapshot"],
        ["snapshot", "--run-id", "r"],
        ["watch-ledger", "--run-id", "r"],
    ):
        parsed = parser.parse_args(command)
        assert parsed.command == command[0]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "arm-holder",
                "--device",
                "/dev/nvidia0",
                "--drill-id",
                "d",
                "--after-ledger-op",
                "RESTORE_GPU_SERVICES",
                "--run-id",
                "r",
                "--probe-script",
                "/run/p.py",
            ]
        )


def test_max_hold_bounds_are_enforced() -> None:
    assert probe.checked_max_hold(900) == 900
    for value in (0, 59, 3601):
        with pytest.raises(probe.ProbeError):
            probe.checked_max_hold(value)
