"""DESTR-013's final invariant: what it asserts beyond "no replace in the window".

The 2026-09-07 review found the audit collecting evidence it never judged
(``DeleteCluster``/``UpdateCluster``), a focused-test result that was a
constant, no check that the synthetic replacement route DESTR-003/008 open had
been closed again, and a CloudTrail window taken at face value -- an empty
query in the wrong region reads exactly like a clean one.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as destr013

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def _replica(pod: str, **overrides: str | None) -> dict[str, str | None]:
    value: dict[str, str | None] = {
        "pod": pod,
        "allow_replace": "false",
        "allow_reboot": "true",
        "allow_automatic": None,
        "legacy_mutation": None,
    }
    value.update(overrides)
    return value


def test_every_executor_replica_must_independently_enable_reboot() -> None:
    # DESTR-011's unique check, absorbed here now that it is superseded: a
    # fleet that disabled every HyperPod verb satisfies the replace invariant
    # while making the reboot path DESTR-002 depends on impossible.
    assert destr013.environment_errors([_replica("a"), _replica("b")]) == []
    errors = destr013.environment_errors(
        [_replica("a"), _replica("b", allow_reboot=None)]
    )
    assert errors == ["an executor replica does not independently enable reboot"]
    errors = destr013.environment_errors([_replica("a", allow_replace="true")])
    assert "an executor replica violates the mutation invariants" in errors
    assert destr013.environment_errors([]) == [
        "no running executor replicas were inspected"
    ]


def test_the_synthetic_replacement_route_must_be_closed_on_every_api_replica() -> None:
    closed = [{"pod": "api-a", "synthetic_route": None}]
    assert destr013.synthetic_route_errors(closed) == []
    left_open = [
        {"pod": "api-a", "synthetic_route": None},
        {"pod": "api-b", "synthetic_route": "true"},
        {"pod": "api-c", "synthetic_route": "false"},
    ]
    errors = destr013.synthetic_route_errors(left_open)
    # Even a literal "false" is a variable the shipped manifest does not have;
    # the window helper restores absence, so presence means an unclosed window.
    assert errors == [
        f"{destr013.SYNTHETIC_ROUTE_ENV} is still set on API replicas: api-b, api-c"
    ]
    assert destr013.synthetic_route_errors([]) == [
        "no running API replicas were inspected for the synthetic route"
    ]


def test_delete_and_update_cluster_are_judged_not_just_collected() -> None:
    clean: dict[str, list[dict[str, str]]] = {name: [] for name in destr013.EVENT_NAMES}
    assert destr013.cloudtrail_invariant_errors(clean) == []
    flipped = {**clean, "UpdateCluster": [{"event_name": "UpdateCluster"}]}
    assert destr013.cloudtrail_invariant_errors(flipped) == [
        "CloudTrail contains UpdateCluster"
    ]
    deleted = {**clean, "DeleteCluster": [{"event_name": "DeleteCluster"}]}
    assert destr013.cloudtrail_invariant_errors(deleted) == [
        "CloudTrail contains DeleteCluster"
    ]
    replaced = {**clean, "ReplaceClusterNodes": [{"event_name": "ReplaceClusterNodes"}]}
    assert destr013.cloudtrail_invariant_errors(replaced) == [
        "CloudTrail contains provider replacement"
    ]
    # The positive control is not a violation.
    rebooted = {**clean, "BatchRebootClusterNodes": [{"event_name": "x"}]}
    assert destr013.cloudtrail_invariant_errors(rebooted) == []


def test_a_window_ending_inside_cloudtrail_lag_is_refused_unless_accepted() -> None:
    recent_end = NOW - timedelta(minutes=5)
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=recent_end,
        now=NOW,
        accept_recent_window=False,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert len(errors) == 1 and "--accept-recent-window" in errors[0], errors
    assert provisional is False

    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=recent_end,
        now=NOW,
        accept_recent_window=True,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert errors == []
    assert provisional is True

    settled_end = NOW - timedelta(minutes=20)
    errors, provisional = destr013.window_errors(
        started_at=NOW - timedelta(hours=3),
        ended_at=settled_end,
        now=NOW,
        accept_recent_window=False,
        reboot_events=2,
        expect_no_reboots=False,
    )
    assert (errors, provisional) == ([], False)


def test_an_empty_window_needs_a_positive_control_or_an_explicit_waiver() -> None:
    common: dict[str, Any] = {
        "started_at": NOW - timedelta(hours=3),
        "ended_at": NOW - timedelta(minutes=20),
        "now": NOW,
        "accept_recent_window": False,
    }
    errors, _ = destr013.window_errors(
        **common, reboot_events=0, expect_no_reboots=False
    )
    assert len(errors) == 1 and "wrong region or hours" in errors[0], errors
    errors, _ = destr013.window_errors(
        **common, reboot_events=0, expect_no_reboots=True
    )
    assert errors == []
    # The waiver is itself checked: reboots in a window the operator swore was
    # reboot-free mean the operator described a different run.
    errors, _ = destr013.window_errors(
        **common, reboot_events=1, expect_no_reboots=True
    )
    assert len(errors) == 1 and "--expect-no-reboots was given" in errors[0], errors


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_timestamps_read_top_level_keys_and_timeline_entries_only(
    tmp_path: Path,
) -> None:
    case = tmp_path / "cases" / "GF-REGIONAL-DESTR-003"
    _write(
        case / "GF-REGIONAL-DESTR-003.json",
        {
            "case_id": "GF-REGIONAL-DESTR-003",
            "started_at": "2026-09-07T09:00:00+00:00",
            "maintenance_window_end": "2026-09-08T00:00:00+00:00",
            "store": {"agents": [{"created_at": "2026-01-01T00:00:00+00:00"}]},
        },
    )
    _write(
        case / "timeline.json",
        {"entries": [{"observed_at": "2026-09-07T09:05:00Z"}, {"observed_at": None}]},
    )
    _write(
        case / "step-timeline.json", {"transitions": [{"observed_at": "not-a-time"}]}
    )
    # Other case families and non-object documents are not the run's window.
    _write(
        tmp_path / "cases" / "GF-REGIONAL-HA-004" / "x.json",
        {"started_at": "2020-01-01T00:00:00Z"},
    )
    _write(case / "list.json", ["2026-09-07T09:00:00+00:00"])

    found = destr013.run_timestamps(tmp_path)

    assert sorted(item["at"] for item in found) == [
        "2026-09-07T09:00:00+00:00",
        "2026-09-07T09:05:00+00:00",
    ], found
    assert {item["key"] for item in found} == {"started_at", "entries[].observed_at"}


def test_the_window_must_cover_the_runs_recorded_timestamps() -> None:
    timestamps = [
        {"path": "a", "key": "started_at", "at": "2026-09-07T09:00:00+00:00"},
        {"path": "b", "key": "observed_at", "at": "2026-09-07T10:30:00+00:00"},
    ]
    start = datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc)
    assert (
        destr013.coverage_errors(
            started_at=start,
            ended_at=datetime(2026, 9, 7, 11, 0, tzinfo=timezone.utc),
            timestamps=timestamps,
        )
        == []
    )
    errors = destr013.coverage_errors(
        started_at=start,
        ended_at=datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
        timestamps=timestamps,
    )
    assert len(errors) == 1 and "1 timestamp(s) outside" in errors[0], errors
    assert destr013.coverage_errors(
        started_at=start, ended_at=start, timestamps=[]
    ) == ["--run-dir carries no destructive-case timestamps to check against"]


def test_focused_tests_report_a_failing_run_instead_of_a_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="1 failed\n", stderr="")

    monkeypatch.setattr(destr013.subprocess, "run", fake_run)

    result = destr013.focused_tests(tmp_path)

    assert result["passed"] is False
    assert result["returncode"] == 1
    assert (tmp_path / "focused-tests.log").read_text(encoding="utf-8") == "1 failed\n"


def test_parser_accepts_the_window_edge_flags_and_the_control_plane(
    tmp_path: Path,
) -> None:
    arguments = destr013.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--window-start",
            "2026-08-30T00:00:00Z",
            "--window-end",
            "2026-08-31T00:00:00Z",
            "--accept-recent-window",
            "--expect-no-reboots",
            "--cpu-kubeconfig",
            "/tmp/cpu",
        ]
    )
    assert arguments.accept_recent_window is True
    assert arguments.expect_no_reboots is True
    assert arguments.cpu_kubeconfig == "/tmp/cpu"
    defaults = destr013.parser().parse_args(
        ["--run-dir", str(tmp_path), "--window-start", "x", "--window-end", "y"]
    )
    assert defaults.accept_recent_window is False
    assert defaults.expect_no_reboots is False


def test_destr011_is_recorded_as_superseded() -> None:
    assert destr013.SUPERSEDED_CASE_IDS == ("GF-REGIONAL-DESTR-011",)
    assert destr013.POSITIVE_CONTROL_EVENT in destr013.EVENT_NAMES
    assert "UpdateCluster" in destr013.FORBIDDEN_EVENT_NAMES
