"""Unit tests for the GF-REGIONAL-DESTR-019 on-node probe.

The probe's one mutation is a Node Agent restart, so what is pinned here is
the guard rail around it (the systemctl verb/unit allow-list, the fail-safe
bounds), plus the three read paths the case's verdicts depend on: the ARCH-C3
journald line parser, the ARCH-C4 ledger audit reader against rows a real
``NodeActionLedger`` wrote, and the migration drill that rebuilds the pre-C4
schema and lets the shipped ledger migrate it in place.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.ledger import LEDGER_SCHEMA_VERSION, NodeActionLedger
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
)
from scripts.e2e.regional import destr019_verdicts as verdicts
from scripts.e2e.regional.probes import destr019_node_probe as probe

RUN_ID = "destr019-run-a1"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
COMMAND_ID = "workflow-x/1/RESTART_FABRIC_MANAGER/node-a/agent-4"


def _command(command_id: str, issued_at: datetime) -> NodeActionCommand:
    return NodeActionCommand(
        command_id=command_id,
        workflow_request_id="workflow-x",
        incident_id="inc-x",
        fencing_token=2,
        operation=WorkflowOperation.RESTART_FABRIC_MANAGER,
        node_id="node-a",
        agent_generation=4,
        gpu_uuids=["GPU-1", "GPU-2"],
        parameters={"a": 1},
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
    )


# --------------------------------------------------------------------------- #
# systemctl allow-list
# --------------------------------------------------------------------------- #
def test_the_agent_unit_may_be_restarted_and_started_but_never_stopped() -> None:
    for verb in ("restart", "start", "show", "is-active"):
        command = ["systemctl", verb, probe.AGENT_UNIT]
        assert probe.checked_command(command, RUN_ID) == command, verb
    with pytest.raises(probe.ProbeError, match="may not stop the Node Agent"):
        probe.checked_command(["systemctl", "stop", probe.AGENT_UNIT], RUN_ID)


def test_only_the_runs_own_fail_safe_units_may_be_touched() -> None:
    unit = probe.restore_unit(RUN_ID)
    for suffix in (".timer", ".service"):
        stop = ["systemctl", "stop", unit + suffix]
        assert probe.checked_command(stop, RUN_ID) == stop, suffix
    with pytest.raises(probe.ProbeError, match="allow-list"):
        probe.checked_command(
            ["systemctl", "stop", probe.restore_unit("another-run") + ".timer"], RUN_ID
        )
    with pytest.raises(probe.ProbeError, match="allow-list"):
        probe.checked_command(["systemctl", "stop", "kubelet.service"], RUN_ID)
    with pytest.raises(probe.ProbeError, match="systemd-run"):
        probe.checked_command(["systemctl", "start", unit + ".service"], RUN_ID)
    with pytest.raises(probe.ProbeError, match="only systemctl"):
        probe.checked_command(["rm", "-rf", "/"], RUN_ID)
    with pytest.raises(probe.ProbeError, match="not permitted"):
        probe.checked_command(["systemctl", "show", probe.AGENT_UNIT, "--all"], RUN_ID)


def test_fail_safe_units_are_digests_of_the_run_id() -> None:
    first = probe.restore_unit(RUN_ID)
    assert first == probe.restore_unit(RUN_ID), "unit name must be deterministic"
    assert first != probe.restore_unit("destr019-run-a2"), "runs must not share units"
    assert first.startswith("gpu-fault-destr019-restore-"), first
    assert RUN_ID not in first, "the raw run id must not leak into the unit name"


def test_restore_seconds_are_bounded() -> None:
    assert probe.checked_restore_seconds(180) == 180, "180s is inside the bounds"
    for value in (59, 601):
        with pytest.raises(probe.ProbeError, match="restore seconds"):
            probe.checked_restore_seconds(value)


def test_state_file_is_scoped_to_the_run_and_lives_outside_the_agent_dir(
    tmp_path: Path,
) -> None:
    path = probe.state_path(RUN_ID, state_dir=tmp_path)
    probe.write_state(path, {"run_id": RUN_ID})
    assert probe.read_state(path) == {"run_id": RUN_ID}, "state must round-trip"
    assert path.stat().st_mode & 0o777 == 0o600, oct(path.stat().st_mode)
    assert not str(probe.STATE_DIR).startswith("/var/lib/gpu-fault/"), probe.STATE_DIR
    with pytest.raises(probe.ProbeError, match="unsafe"):
        probe.state_path("../escape", state_dir=tmp_path)


def test_agent_env_keys_exclude_the_shared_secret_and_token() -> None:
    lowered = " ".join(probe.AGENT_ENV_KEYS).lower()
    assert "secret" not in lowered, probe.AGENT_ENV_KEYS
    assert "token" not in lowered, probe.AGENT_ENV_KEYS


def test_health_url_prefers_the_advertised_url_then_tls_then_plaintext() -> None:
    assert (
        probe.health_url({"GPU_FAULT_NODE_ADVERTISE_URL": "https://n:9099/"})
        == "https://n:9099/healthz"
    ), "advertised URL wins"
    assert (
        probe.health_url(
            {
                "GPU_FAULT_NODE_AGENT_HOST": "10.0.0.5",
                "GPU_FAULT_NODE_AGENT_PORT": "9099",
                "GPU_FAULT_NODE_AGENT_TLS_CERT": "/etc/gpu-fault/agent.crt",
            }
        )
        == "https://10.0.0.5:9099/healthz"
    ), "a TLS certificate means https"
    assert probe.health_url({}) == "http://127.0.0.1:9099/healthz", "defaults"


# --------------------------------------------------------------------------- #
# journald parser (ARCH-C3)
# --------------------------------------------------------------------------- #
def test_parse_log_line_extracts_the_phase_and_key_value_fields() -> None:
    line = (
        "node action completed command_id=" + COMMAND_ID + " incident_id=inc-x "
        "workflow_request_id=workflow-x operation=RESTART_FABRIC_MANAGER "
        "node_id=node-a fencing_token=2 agent_generation=4 gpu_count=0 attempt=1 "
        "status=SUCCEEDED duration_ms=812"
    )
    parsed = probe.parse_log_line(line)
    assert parsed is not None, "a completed line must parse"
    assert parsed["phase"] == "completed", parsed
    assert parsed["fields"]["command_id"] == COMMAND_ID, parsed
    assert parsed["fields"]["duration_ms"] == "812", parsed
    assert probe.parse_log_line("node agent serving HTTPS on 10.0.0.5:9099") is None, (
        "an unrelated Agent line must not parse as a command phase"
    )
    rejected = probe.parse_log_line(
        "node action rejected command_id=c incident_id=i workflow_request_id=w "
        "operation=RESET_GPU node_id=n fencing_token=1 agent_generation=None "
        "gpu_count=1 attempt=0 reason=operation RESET_GPU is not allowed"
    )
    assert rejected is not None and rejected["phase"] == "rejected", rejected


# --------------------------------------------------------------------------- #
# ledger audit reader (ARCH-C4) against the real ledger
# --------------------------------------------------------------------------- #
def test_ledger_audit_reads_the_audit_columns_of_a_real_row(tmp_path: Path) -> None:
    path = tmp_path / "node-actions.db"
    ledger = NodeActionLedger(str(path))
    ledger.mark_in_progress(_command(COMMAND_ID, T0), 1, signature="sig")
    ledger.save(
        NodeActionResult(
            command_id=COMMAND_ID,
            operation=WorkflowOperation.RESTART_FABRIC_MANAGER,
            status=NodeActionStatus.SUCCEEDED,
            attempt=1,
            completed_at=T0 + timedelta(seconds=3),
        )
    )
    ledger.close()

    audit = probe.ledger_audit_rows(path, command_id=COMMAND_ID)

    assert audit["present"] is True, audit
    assert audit["user_version"] == LEDGER_SCHEMA_VERSION, audit
    assert audit["primary_key"] == list(verdicts.LEDGER_PRIMARY_KEY), audit
    assert set(verdicts.AUDIT_COLUMNS) <= set(audit["columns"]), audit["columns"]
    assert len(audit["rows"]) == 1, audit["rows"]
    row = audit["rows"][0]
    assert row["attempt"] == 1 and row["state"] == "SUCCEEDED", row
    assert row["incident_id"] == "inc-x" and row["fencing_token"] == 2, row
    assert row["gpu_uuid_count"] == 2 and row["gpu_uuids_present"] is True, row
    assert "gpu_uuids" not in row, "the UUID list itself must not be exported"
    assert verdicts.HEX_DIGEST.fullmatch(row["parameters_digest"]), row
    assert verdicts.HEX_DIGEST.fullmatch(row["signature_digest"]), row
    assert row["started_at"] is not None and row["completed_at"] is not None, row
    assert audit["interrupted_count"] == 0, audit


def test_ledger_audit_is_read_only_and_tolerates_a_missing_ledger(
    tmp_path: Path,
) -> None:
    missing = probe.ledger_audit_rows(tmp_path / "absent.db")
    assert missing["present"] is False and missing["rows"] == [], missing
    path = tmp_path / "node-actions.db"
    NodeActionLedger(str(path)).close()
    before = path.stat().st_mtime_ns
    probe.ledger_audit_rows(path, since="2026-01-01T00:00:00+00:00")
    assert path.stat().st_mtime_ns == before, "the audit read must not write"


# --------------------------------------------------------------------------- #
# migration drill (ARCH-C4) -- the pre-C4 schema, migrated by the shipped ledger
# --------------------------------------------------------------------------- #
def test_the_legacy_ledger_is_built_exactly_as_the_old_agent_left_it(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "legacy.db"
    command_ids = probe.build_legacy_ledger(scratch, now=T0)
    connection = sqlite3.connect(f"file:{scratch}?mode=ro", uri=True)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        primary_key = probe.ledger_primary_key(connection)
        columns = probe.ledger_columns(connection)
        rows = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    finally:
        connection.close()
    assert version == 0, "a pre-C4 ledger carries user_version 0"
    assert primary_key == ["command_id"], primary_key
    assert "incident_id" not in columns and "parameters_digest" not in columns, columns
    assert rows == len(command_ids) == probe.LEGACY_ROW_COUNT, rows


def test_the_migration_drill_report_satisfies_the_case_verdict(tmp_path: Path) -> None:
    """The red/green test of ARCH-C4 through the probe: the pre-C4 ledger
    opens under the shipped ``NodeActionLedger``, keeps every row, gains the
    audit columns and the per-attempt key, and accepts an appended attempt."""

    report = probe.migration_drill_report(tmp_path / "drill" / "legacy.db", now=T0)

    assert verdicts.migration_errors(report) == [], report
    assert report["after_open"]["user_version"] == LEDGER_SCHEMA_VERSION, report
    assert report["scratch_removed"] is True, "the drill must clean up after itself"
    assert not list((tmp_path / "drill").iterdir()), "no scratch files may remain"


def test_migration_verdict_rejects_a_ledger_that_did_not_migrate(
    tmp_path: Path,
) -> None:
    report = probe.migration_drill_report(tmp_path / "legacy.db", now=T0)
    stale = dict(report)
    stale["after_open"] = dict(report["before"])
    errors = verdicts.migration_errors(stale)
    assert any("user_version" in item for item in errors), errors
    assert any("primary key" in item for item in errors), errors

    leftover = dict(report)
    leftover["legacy_table_present"] = True
    assert any(
        "results_legacy" in item for item in verdicts.migration_errors(leftover)
    ), "a leftover legacy table must fail the drill"

    lost_row = dict(report)
    lost_row["after_open"] = {**report["after_open"], "row_count": 0}
    assert any(
        "row count changed" in item for item in verdicts.migration_errors(lost_row)
    ), "a migration that dropped rows must fail the drill"


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def test_parser_accepts_every_documented_subcommand() -> None:
    parser = probe.parser()
    for argv in (
        ["restart-agent", "--run-id", RUN_ID, "--restore-seconds", "120"],
        ["disarm-restore", "--run-id", RUN_ID],
        ["ensure-agent-active", "--run-id", RUN_ID],
        ["agent-health"],
        ["journal", "--since-epoch", "1.5", "--command-id", COMMAND_ID],
        ["ledger-audit", "--command-id", COMMAND_ID],
        ["migration-drill", "--run-id", RUN_ID],
        ["snapshot", "--run-id", RUN_ID],
    ):
        arguments = parser.parse_args(argv)
        assert callable(arguments.handler), argv
    with pytest.raises(SystemExit):
        parser.parse_args(["stop-agent", "--run-id", RUN_ID])
